# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------
"""DexSim W1 bimanual cloth grasp, lift, then move the whole robot.

This extends ``demo_cloth_dexforce_bimanual_grasp_mjvbd.py``.  The base demo's
scene, cloth, W1 setup, MJVBD solver, and IK trajectory generation are reused.
After the cloth is grasped and lifted, the W1 keeps the same joint posture and
the whole robot moves backward then rotates 90 degrees, so the cloth stays
held by the robot instead of being left behind by arm-only IK.

Key mechanism
-------------
The base demo loads W1 with ``fix_root_link=True`` (a FIXED base joint).  In
that mode the root's world pose is NOT represented in ``joint_q`` (a FIXED
joint has 0 coords), so ``newton.eval_fk`` always rewrites ``body_q[root]``
from the build-time joint anchors -- any ``set_world_pose`` write to
``body_q[root]`` is clobbered every substep and the robot never actually moves.

To move the whole robot we instead load W1 with ``fix_root_link=False`` so the
root becomes a FREE joint whose 7 coords (xyz + xyzw quaternion) live in
``joint_q[0:7]``.  ``eval_fk`` respects those coords and propagates the root
pose to every child link and to cloth contact.  The 40-DOF arm/hand trajectory
from the base demo is unchanged (``get_dof()`` is 40 in both base modes); we
only bake a root trajectory into the trajectory-driver cache columns
``[0:7]`` so the driver's per-substep interpolation produces the root motion
and FK carries it through the kinematic tree.

The grasp/lift IK script is sped up by ``--lift-time-scale`` (default 2.0)
using the same segment-duration trick as
``demo_cloth_dexforce_bimanual_fold_tshirt_mjvbd``.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

import demo_cloth_dexforce_bimanual_grasp_mjvbd as grasp_demo
from dexsim.engine.newton_physics import get_newton_manager
from dexsim.types import ArticulationFlag
from dexsim.utility import Color as NvtxColor
from dexsim.utility import scope as nvtx_scope


BACKWARD_DISTANCE = 0.35
MOVE_BACK_TIME = 1.5
ROTATE_DEGREES = 90.0
ROTATE_TIME = 3.0
FINAL_HOLD_TIME = 1.0

# Original IK script duration (home + approach + grasp + close + lift + hold).
LIFT_SCRIPT_DURATION = 0.8 + 2.0 + 2.0 + 2.0 + 3.0 + 2.0
DEFAULT_LIFT_TIME_SCALE = 4.0
TOTAL_DURATION = (
    LIFT_SCRIPT_DURATION / DEFAULT_LIFT_TIME_SCALE
    + MOVE_BACK_TIME + ROTATE_TIME + FINAL_HOLD_TIME
)
DEFAULT_STEPS = int(math.ceil(TOTAL_DURATION / grasp_demo.DEFAULT_DT)) + 60


def parse_args() -> argparse.Namespace:
    previous_default_steps = grasp_demo.DEFAULT_STEPS
    grasp_demo.DEFAULT_STEPS = DEFAULT_STEPS
    # Pre-extract --lift-time-scale; grasp_demo.parse_args' parser does not know
    # this flag and would error on it, so strip it from argv first.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--lift-time-scale",
        type=float,
        default=DEFAULT_LIFT_TIME_SCALE,
        help="Speed-up factor for the grasp/lift IK script (>=1). Larger is faster.",
    )
    lift_args, remaining = pre.parse_known_args()
    saved_argv = sys.argv
    sys.argv = [sys.argv[0]] + remaining
    try:
        parser_args = grasp_demo.parse_args()
    finally:
        sys.argv = saved_argv
        grasp_demo.DEFAULT_STEPS = previous_default_steps
    parser_args.lift_time_scale = float(lift_args.lift_time_scale)
    return parser_args


def _root_coords_at_time(initial_pose: np.ndarray, t: float) -> np.ndarray:
    """Root FREE-joint coords [x, y, z, qx, qy, qz, qw] at base-motion time t.

    The robot keeps its lifted posture; only the root translates backward then
    rotates in place.  During the lift phase (t < 0) the root stays at its
    initial pose so the arm-only IK motion is undisturbed.
    """
    t = max(float(t), 0.0)
    move_u = min(t / max(MOVE_BACK_TIME, 1.0e-6), 1.0)
    rotate_u = min(max(t - MOVE_BACK_TIME, 0.0) / max(ROTATE_TIME, 1.0e-6), 1.0)

    x = float(initial_pose[0, 3]) - BACKWARD_DISTANCE * move_u
    y = float(initial_pose[1, 3])
    z = float(initial_pose[2, 3])
    yaw = math.radians(ROTATE_DEGREES) * rotate_u
    quat_xyzw = Rotation.from_euler("z", yaw).as_quat().astype(np.float32)
    return np.array([x, y, z, quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]],
                    dtype=np.float32)


def _build_root_trajectory(initial_pose: np.ndarray, frame_count: int,
                           fps: int, lift_duration: float) -> np.ndarray:
    """Per-frame root coords [frame_count, 7] aligned with the IK trajectory."""
    inv_fps = 1.0 / float(fps)
    traj = np.zeros((frame_count, 7), dtype=np.float32)
    for f in range(frame_count):
        sim_time = f * inv_fps
        base_t = sim_time - lift_duration
        traj[f] = _root_coords_at_time(initial_pose, base_t)
    return traj


def _build_w1_trajectory_with_lift_scale(args: argparse.Namespace, w1,
                                         lift_time_scale: float):
    """Build the arm/hand IK trajectory with the grasp/lift script sped up.

    Mirrors the time-scale trick from
    ``demo_cloth_dexforce_bimanual_fold_tshirt_mjvbd``: temporarily replace
    ``grasp_demo._build_motion_segments`` with a wrapper that divides each
    segment's duration by ``lift_time_scale``, then restore it. The trajectory
    fps stays the simulation fps, so the same number of frames covers the
    shortened script -- the robot simply moves faster.
    """
    original_builder = grasp_demo._build_motion_segments

    def _scaled_motion_segments(left_home, right_home):
        segments = original_builder(left_home, right_home)
        return tuple(
            (duration / lift_time_scale,) + tuple(rest)
            for (duration, *rest) in segments
        )

    grasp_demo._build_motion_segments = _scaled_motion_segments
    try:
        return grasp_demo._build_w1_joint_target_trajectory(args, w1)
    finally:
        grasp_demo._build_motion_segments = original_builder


def _load_w1_floating(env, *, enable_collision: bool):
    """Load W1 with a floating base so the root pose lives in joint_q[0:7].

    Mirrors ``grasp_demo._load_w1`` but uses ``fix_root_link=False`` and clears
    the FIX_BASE flag, turning the root into a Newton FREE joint whose 7
    coords are driven by the root trajectory baked into the driver cache.
    """
    if not grasp_demo.W1_URDF.exists():
        raise FileNotFoundError(f"Dexforce W1 URDF not found: {grasp_demo.W1_URDF}")

    w1 = env.load_urdf(str(grasp_demo.W1_URDF), fix_root_link=False)
    w1.set_articulation_flag(ArticulationFlag.FIX_BASE, False)
    w1.set_articulation_flag(ArticulationFlag.DISABLE_SELF_COLLISION, True)
    w1.enable_collision(enable_collision)

    for link_name in w1.get_link_names():
        attr = grasp_demo._copy_physical_attr(w1.get_physical_attr(link_name))
        attr.dynamic_friction = grasp_demo.RIGID_CONTACT_MU
        attr.static_friction = grasp_demo.RIGID_CONTACT_MU
        attr.contact_stiffness = grasp_demo.RIGID_CONTACT_KE
        attr.contact_damping = grasp_demo.RIGID_CONTACT_KD
        if any(keyword in link_name.lower() for keyword in grasp_demo.HAND_CONTACT_KEYWORDS):
            attr.dynamic_friction = grasp_demo.HAND_CONTACT_MU
            attr.static_friction = grasp_demo.HAND_CONTACT_MU
            attr.contact_stiffness = grasp_demo.HAND_CONTACT_KE
            attr.contact_damping = grasp_demo.HAND_CONTACT_KD
        w1.set_physical_attr(attr, link_name=link_name)

    return w1


def _bake_root_trajectory_into_driver(mgr, root_trajectory: np.ndarray) -> bool:
    """Write the per-frame root coords into the trajectory driver cache [0:7].

    The MJVBD kinematic-substep driver builds a full ``joint_q`` cache of shape
    ``[frames, joint_coord_count]``; the FREE root's 7 coords occupy columns
    ``[0:7]``.  Baking the root motion there makes the driver's per-substep
    interpolation produce the root pose, and ``eval_fk`` propagates it through
    the kinematic tree so cloth contact sees the whole robot moving.
    """
    for drv in mgr._joint_trajectory_drivers:
        jt = getattr(drv, "joint_targets", None)
        if jt is None:
            continue
        jt = np.asarray(jt, dtype=np.float32).copy()
        frames = min(jt.shape[0], root_trajectory.shape[0])
        jt[:frames, 0:7] = root_trajectory[:frames]
        # If the cache is longer than the root trajectory, hold the last root pose.
        if jt.shape[0] > frames:
            jt[frames:, 0:7] = root_trajectory[-1]
        drv.joint_targets = np.ascontiguousarray(jt, dtype=np.float32)
        # The CUDA-graph driver uploads joint_targets into a device array on its
        # first call; resetting the cached device array forces a fresh upload so
        # the baked root columns take effect.
        if hasattr(drv, "cached_joint_targets_wp"):
            drv.cached_joint_targets_wp = None
        return True
    return False


def run_demo(args: argparse.Namespace) -> None:
    world = None
    env = None
    w1 = None
    mgr = None
    cloth = None
    try:
        with nvtx_scope("W1WholeBodyMove::setup_world", color=NvtxColor.NEWTON_SIM):
            world, env = grasp_demo._setup_world(args)
        if not args.headless:
            with nvtx_scope("W1WholeBodyMove::setup_render", color=NvtxColor.RENDER):
                grasp_demo._setup_render(world, env)

        with nvtx_scope("W1WholeBodyMove::create_scene", color=NvtxColor.NEWTON_CLOTH):
            grasp_demo._create_ground(env)
            grasp_demo._create_table(env)
            w1 = _load_w1_floating(env, enable_collision=not args.disable_w1_collision)
            mgr = get_newton_manager(world)
            if mgr is not None and not args.disable_w1_collision:
                mgr.add_on_model_ready_callback(lambda: grasp_demo._configure_w1_particle_only_collision(w1))
            cloth = grasp_demo._create_cloth(env)
            grasp_demo._configure_w1_drive(w1)

        # Build the arm/hand IK trajectory spanning the whole demo so the cache
        # has enough frames to also hold the root motion after the lift. The
        # grasp/lift script is sped up by args.lift_time_scale.
        lift_duration = LIFT_SCRIPT_DURATION / max(float(args.lift_time_scale), 1.0e-6)
        with nvtx_scope("W1WholeBodyMove::build_trajectory", color=NvtxColor.ARTICULATION):
            ik_args = argparse.Namespace(**vars(args))
            ik_args.steps = max(int(args.steps), 2)
            trajectory, trajectory_fps, trajectory_joint_names = _build_w1_trajectory_with_lift_scale(
                ik_args,
                w1,
                float(args.lift_time_scale),
            )
        if trajectory.shape[1] != int(w1.get_dof()):
            raise ValueError(
                "W1 trajectory DOF does not match DexSim articulation DOF: "
                f"trajectory={trajectory.shape[1]}, articulation={int(w1.get_dof())}"
            )

        zero_qvel = np.zeros(int(w1.get_dof()), dtype=np.float32)
        active_joint_names = list(w1.get_actived_joint_names(False))
        initial_api_qpos = grasp_demo._trajectory_row_to_joint_order(
            trajectory[0],
            trajectory_joint_names,
            active_joint_names,
        )
        w1.set_current_qpos(initial_api_qpos)
        w1.set_current_qvel(zero_qvel)
        w1.set_target_qpos(initial_api_qpos)
        w1.set_target_qvel(zero_qvel)

        initial_base_pose = np.asarray(w1.get_world_pose(), dtype=np.float32)
        root_trajectory = _build_root_trajectory(
            initial_base_pose,
            frame_count=int(trajectory.shape[0]),
            fps=trajectory_fps,
            lift_duration=lift_duration,
        )

        if not args.static_w1:
            w1.set_joint_target_trajectory(
                trajectory,
                fps=trajectory_fps,
                mode="kinematic_substep",
                hold_last=True,
            )
        if mgr is not None:
            # Bake the root motion after the trajectory driver is installed in
            # start_simulation(); the on-start callback runs after the driver is
            # registered, so mgr._joint_trajectory_drivers is populated.
            mgr.add_on_start_callback(
                lambda: _bake_root_trajectory_into_driver(mgr, root_trajectory)
            )

        print(
            "trajectory: grasp-lift-then-whole-robot-back-rotate | "
            "driver: kinematic_substep_lift_plus_floating_root | "
            f"sim_device: {args.device} | ik_device: {args.ik_device} | "
            f"w1_collision: {not args.disable_w1_collision} | "
            f"base: floating | lift_time_scale: {args.lift_time_scale:.1f} | "
            f"lift_frames: {len(trajectory)} | fps: {trajectory_fps} | "
            f"dof: {int(w1.get_dof())} | cloth particles: {cloth.particle_count}",
            flush=True,
        )

        if args.debug_warp_verify:
            wp.config.verify_cuda = True
            print("[debug] Warp CUDA verification enabled for simulation loop", flush=True)

        start_time = time.perf_counter()
        for step in range(args.steps):
            sim_time = step * float(args.dt)

            should_debug_cloth = (
                args.debug_cloth_state
                and (step % max(1, args.debug_cloth_interval) == 0 or step == args.steps - 1)
            )
            if args.debug_cuda_sync:
                grasp_demo._debug_sync_cuda(args.device, step=step, phase="pre-update")
            if args.debug_cloth_state:
                grasp_demo._debug_cloth_state(
                    cloth,
                    step=step,
                    phase="pre-update",
                    force=should_debug_cloth,
                )

            with nvtx_scope("W1WholeBodyMove::world_update", color=NvtxColor.NEWTON_SIM):
                world.update(args.dt)

            if args.debug_cuda_sync:
                grasp_demo._debug_sync_cuda(args.device, step=step, phase="post-update")
            if args.debug_cloth_state:
                grasp_demo._debug_cloth_state(
                    cloth,
                    step=step,
                    phase="post-update",
                    force=should_debug_cloth,
                )

            if step % 120 == 0 or step == args.steps - 1:
                elapsed = time.perf_counter() - start_time
                fps = (step + 1) / elapsed if elapsed > 0.0 else 0.0
                if sim_time < lift_duration:
                    phase = "lift"
                elif sim_time < lift_duration + MOVE_BACK_TIME:
                    phase = "move_back"
                elif sim_time < lift_duration + MOVE_BACK_TIME + ROTATE_TIME:
                    phase = "rotate"
                else:
                    phase = "final_hold"
                print(f"step={step + 1}/{args.steps}, phase={phase}, fps={fps:.1f}", flush=True)
            if args.real_time:
                time.sleep(args.dt)
    finally:
        cloth = None
        mgr = None
        w1 = None
        env = None
        if world is not None:
            world.quit()


def main() -> None:
    run_demo(parse_args())


if __name__ == "__main__":
    main()
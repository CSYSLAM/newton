# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
###########################################################################
# Dexforce W1 bimanual tablecloth placing, ported to the coupled MuJoCo + VBD
# solver stack.
#
# Robot/articulation: SolverMuJoCo (arms + hands driven by position PD; the
#   floating base is driven kinematically by the scripted root motion).
# Garment particles:  SolverVBD
# Robot-garment interaction: SolverCoupledProxy in one-way mode (zero feedback
#   relaxation) so the arm tracks IK targets undisturbed by cloth contact while
#   the tablecloth follows the hand proxies through the VBD contact constraint.
#
# The original demo (example_cloth_dexforce_bimanual_place_tablecloth.py in the
# legacy branch) used a custom ``newton.mjvbd.SolverMJVBD`` with a fully
# kinematic robot (``step_rigid_bodies=False``).  This version keeps the same
# scripted bimanual + root-motion behaviour but runs the robot through MuJoCo
# position PD and couples it to VBD through SolverCoupledProxy, matching the
# approach in example_cloth_folding_coupled.py.
#
# Run from the Newton repository root:
#   uv run --extra examples -m newton.examples cloth_dexforce_place_tablecloth_coupled
###########################################################################

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.solvers import SolverMuJoCo, SolverVBD

# ---------------------------------------------------------------------------
# Scene constants (metres / seconds; matched to the legacy demo).
# ---------------------------------------------------------------------------
TABLE_POS = wp.vec3(0.60, 0.0, 1.15)
TABLE_HALF_EXTENTS = (0.32, 0.78, 0.02)
TABLE_TOP_Z = float(TABLE_POS[2] + TABLE_HALF_EXTENTS[2])
TABLE_COLOR = (0.35, 0.42, 0.48)

TABLECLOTH_START_Y = -0.58
CLOTH_DIM_X = 18
CLOTH_DIM_Y = 18
CLOTH_CELL_X = 0.020
CLOTH_CELL_Y = 0.020
CLOTH_HALF_X = 0.5 * CLOTH_DIM_X * CLOTH_CELL_X
CLOTH_HALF_Y = 0.5 * CLOTH_DIM_Y * CLOTH_CELL_Y
CLOTH_CENTER = wp.vec3(float(TABLE_POS[0]) - 0.12, TABLECLOTH_START_Y, TABLE_TOP_Z + 0.018)
CLOTH_POS = wp.vec3(
    float(CLOTH_CENTER[0]) - 0.5 * CLOTH_DIM_X * CLOTH_CELL_X,
    float(CLOTH_CENTER[1]) - 0.5 * CLOTH_DIM_Y * CLOTH_CELL_Y,
    float(CLOTH_CENTER[2]),
)
CLOTH_COLOR = (0.78, 0.12, 0.10)
CLOTH_COLLISION_RADIUS = 0.010
SOFT_CONTACT_MARGIN = 0.020
SELF_CONTACT_RADIUS = 0.010
SELF_CONTACT_MARGIN = 0.012

ROBOT_CONTACT_KE = 5.0e4
ROBOT_CONTACT_KD = 1.0e-4
ROBOT_CONTACT_MU = 1.5
HAND_CONTACT_KE = 3.0e6
HAND_CONTACT_KD = 1.0e-4
HAND_CONTACT_MU = 2.2
HAND_RELEASE_CONTACT_MU = 0.08
HAND_CONTACT_KEYWORDS = ("hand", "thumb", "index", "middle", "ring", "pinky")
TABLE_CONTACT_MU_NORMAL = 1.2
TABLE_CONTACT_MU_LOW = 0.12

# Root-motion + bimanual script timing.
RETREAT_DISTANCE = 0.48
RETREAT_TIME = 1.4
TURN_OUT_DEGREES = 90.0
TURN_OUT_TIME = 1.4
CENTER_SHIFT_Y = -TABLECLOTH_START_Y
CENTER_MOVE_TIME = 2.2
TURN_IN_DEGREES = -90.0
TURN_IN_TIME = 1.4
APPROACH_DISTANCE = 0.40
PRE_LAYDOWN_TIME = 1.2
LAYDOWN_TIME = 1.8
APPROACH_TIME = PRE_LAYDOWN_TIME + LAYDOWN_TIME
RELEASE_TIME = 0.8
POST_RELEASE_LIFT_TIME = 1.0
POST_RELEASE_HOLD_TIME = 0.8
FINAL_HOLD_TIME = 0.8
HOME_HOLD_TIME = 0.5
APPROACH_TIME_HANDS = 1.4
GRASP_TIME = 1.4
CLOSE_TIME = 1.8
LIFT_TIME = 2.0
HANG_SETTLE_TIME = 0.6
ROOT_MOTION_START_TIME = HOME_HOLD_TIME + APPROACH_TIME_HANDS + GRASP_TIME + CLOSE_TIME + LIFT_TIME + HANG_SETTLE_TIME
ROOT_MOTION_DURATION = RETREAT_TIME + TURN_OUT_TIME + CENTER_MOVE_TIME + TURN_IN_TIME + APPROACH_TIME
ROOT_PRE_APPROACH_TIME = RETREAT_TIME + TURN_OUT_TIME + CENTER_MOVE_TIME + TURN_IN_TIME
SCRIPT_DURATION = (
    ROOT_MOTION_START_TIME + ROOT_MOTION_DURATION + RELEASE_TIME + POST_RELEASE_HOLD_TIME + POST_RELEASE_LIFT_TIME
)
DEFAULT_NUM_FRAMES = int(math.ceil((SCRIPT_DURATION + FINAL_HOLD_TIME) * 60.0))

CAMERA_POS = wp.vec3(1.10, -2.35, 1.70)
CAMERA_PITCH = -16.0
CAMERA_YAW = 105.0

TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)

RIGHT_TCP_ROT = wp.quat(0.0245, 0.6878, 0.7139, -0.1294)
RIGHT_APPROACH_TF = wp.transform(
    wp.vec3(float(CLOTH_CENTER[0]) + CLOTH_HALF_X - 0.06, TABLECLOTH_START_Y - CLOTH_HALF_Y, 1.30),
    RIGHT_TCP_ROT,
)
RIGHT_GRASP_TF = wp.transform(
    wp.vec3(
        float(CLOTH_CENTER[0]) + CLOTH_HALF_X - 0.012, TABLECLOTH_START_Y - CLOTH_HALF_Y + 0.012, TABLE_TOP_Z + 0.055
    ),
    RIGHT_TCP_ROT,
)
GRASP_HEIGHT_OFFSET = 0.015
LIFT_HEIGHT = 0.24
LAYDOWN_HEIGHT = 0.015
LAYDOWN_FORWARD = 0.14


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------
@wp.kernel
def lock_joint_q_kernel(
    ik_joint_q: wp.array2d[wp.float32],
    locked_q_indices: wp.array[wp.int32],
    locked_q_values: wp.array[wp.float32],
):
    i = wp.tid()
    ik_joint_q[0, locked_q_indices[i]] = locked_q_values[i]


@wp.kernel
def scatter_arm_targets_kernel(
    ik_joint_q: wp.array2d[wp.float32],
    arm_coord_indices: wp.array[wp.int32],
    arm_target_indices: wp.array[wp.int32],
    joint_target_q: wp.array[wp.float32],
):
    i = wp.tid()
    joint_target_q[arm_target_indices[i]] = ik_joint_q[0, arm_coord_indices[i]]


@wp.kernel
def set_indexed_target_kernel(
    q_indices: wp.array[wp.int32],
    open_values: wp.array[wp.float32],
    grasp_values: wp.array[wp.float32],
    alpha_buf: wp.array[wp.float32],
    joint_target_q: wp.array[wp.float32],
):
    i = wp.tid()
    alpha = alpha_buf[0]
    joint_target_q[q_indices[i]] = open_values[i] * (1.0 - alpha) + grasp_values[i] * alpha


@wp.func
def _smoothstep_wp(u: float) -> float:
    x = wp.clamp(u, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


@wp.kernel
def set_free_root_motion_kernel(
    root_q_start: int,
    root_q0: wp.array[wp.float32],
    base_time: float,
    retreat_distance: float,
    retreat_time: float,
    turn_out_radians: float,
    turn_out_time: float,
    center_shift_y: float,
    center_move_time: float,
    turn_in_radians: float,
    turn_in_time: float,
    approach_distance: float,
    approach_time: float,
    joint_q_out: wp.array[wp.float32],
):
    t0 = retreat_time
    t1 = t0 + turn_out_time
    t2 = t1 + center_move_time
    t3 = t2 + turn_in_time

    retreat_u = _smoothstep_wp(base_time / wp.max(retreat_time, 1.0e-6))
    turn_out_u = _smoothstep_wp((base_time - t0) / wp.max(turn_out_time, 1.0e-6))
    center_u = _smoothstep_wp((base_time - t1) / wp.max(center_move_time, 1.0e-6))
    turn_in_u = _smoothstep_wp((base_time - t2) / wp.max(turn_in_time, 1.0e-6))
    approach_u = _smoothstep_wp((base_time - t3) / wp.max(approach_time, 1.0e-6))

    yaw = turn_out_radians * turn_out_u + turn_in_radians * turn_in_u
    q_yaw = wp.quat(0.0, 0.0, wp.sin(0.5 * yaw), wp.cos(0.5 * yaw))
    q0 = wp.quat(root_q0[3], root_q0[4], root_q0[5], root_q0[6])
    q = q_yaw * q0

    joint_q_out[root_q_start + 0] = root_q0[0] - retreat_distance * retreat_u + approach_distance * approach_u
    joint_q_out[root_q_start + 1] = root_q0[1] + center_shift_y * center_u
    joint_q_out[root_q_start + 2] = root_q0[2]
    joint_q_out[root_q_start + 3] = q[0]
    joint_q_out[root_q_start + 4] = q[1]
    joint_q_out[root_q_start + 5] = q[2]
    joint_q_out[root_q_start + 6] = q[3]


@wp.kernel
def zero_free_root_velocity_kernel(
    root_qd_start: int,
    joint_qd: wp.array[wp.float32],
):
    i = wp.tid()
    joint_qd[root_qd_start + i] = 0.0


@wp.kernel
def copy_root_q_kernel(
    root_q_start: int,
    root_target: wp.array[wp.float32],
    joint_q: wp.array[wp.float32],
):
    i = wp.tid()
    joint_q[root_q_start + i] = root_target[i]


@wp.kernel
def set_shape_material_mu_kernel(
    shape_indices: wp.array[wp.int32],
    mu_value: float,
    shape_mu: wp.array[wp.float32],
):
    i = wp.tid()
    shape_mu[shape_indices[i]] = mu_value


def _smoothstep(u: float) -> float:
    x = float(np.clip(u, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _resolve_robot_urdf(cli_override: str | None = None) -> Path:
    """Locate the DexforceW1V021 URDF under the repo ``assets/`` folder.

    The robot assets live in the repository ``assets/`` directory and may also
    be overridden via env var or CLI. Resolution order:
      1. ``--robot-urdf`` CLI arg (a .urdf file or its parent directory)
      2. ``DEXFORCE_W1_URDF`` env var (same)
      3. ``<repo>/assets/DexforceW1V021/DexforceW1V021.urdf``
    """
    import os

    candidate = cli_override or os.environ.get("DEXFORCE_W1_URDF")
    if candidate:
        p = Path(candidate)
        if p.is_dir():
            p = p / "DexforceW1V021.urdf"
        return p
    return Path(__file__).resolve().parents[4] / "assets" / "DexforceW1V021" / "DexforceW1V021.urdf"


class Example:
    LEFT_ARM_JOINTS = (
        "LEFT_J1",
        "LEFT_J2",
        "LEFT_J3",
        "LEFT_J4",
        "LEFT_J5",
        "LEFT_J6",
        "LEFT_J7",
    )
    RIGHT_ARM_JOINTS = (
        "RIGHT_J1",
        "RIGHT_J2",
        "RIGHT_J3",
        "RIGHT_J4",
        "RIGHT_J5",
        "RIGHT_J6",
        "RIGHT_J7",
    )
    LEFT_HAND_JOINTS = (
        "LEFT_HAND_THUMB2",
        "LEFT_HAND_THUMB1",
        "LEFT_HAND_INDEX",
        "LEFT_INDEX_PIP",
        "LEFT_HAND_MIDDLE",
        "LEFT_MIDDLE_PIP",
        "LEFT_HAND_RING",
        "LEFT_RING_PIP",
        "LEFT_HAND_PINKY",
        "LEFT_PINKY_PIP",
    )
    RIGHT_HAND_JOINTS = (
        "RIGHT_HAND_THUMB2",
        "RIGHT_HAND_THUMB1",
        "RIGHT_HAND_INDEX",
        "RIGHT_INDEX_PIP",
        "RIGHT_HAND_MIDDLE",
        "RIGHT_MIDDLE_PIP",
        "RIGHT_HAND_RING",
        "RIGHT_RING_PIP",
        "RIGHT_HAND_PINKY",
        "RIGHT_PINKY_PIP",
    )

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = int(args.substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.print_interval = float(args.print_interval)
        self.last_print_time = -1.0

        self.particle_radius = CLOTH_COLLISION_RADIUS
        self.self_contact_radius = SELF_CONTACT_RADIUS
        self.self_contact_margin = SELF_CONTACT_MARGIN

        self._build_scene()
        self.device = self.model.device

        self.control = self.model.control()
        self._build_solver(args)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="explicit",
            soft_contact_margin=SOFT_CONTACT_MARGIN,
            contact_matching="latest",
        )
        self.contacts = self.collision_pipeline.contacts()
        self.solver.prepare_contacts(self.contacts)

        self._configure_robot_contacts()

        self.root_joint_index = self._root_free_joint_index()
        self.root_q_start = int(self.model.joint_q_start.numpy()[self.root_joint_index])
        self.root_qd_start = int(self.model.joint_qd_start.numpy()[self.root_joint_index])
        root_q0_np = self.model.joint_q.numpy()[self.root_q_start : self.root_q_start + 7].copy()
        self.root_q0_np = root_q0_np
        self.root_q0 = wp.array(root_q0_np, dtype=wp.float32, device=self.device)
        self.root_motion_start_time = ROOT_MOTION_START_TIME

        # Scale the root-motion phase durations by 1/motion_speed so the
        # retreat/turn/walk/turn/approach can be slowed down (the original
        # timing is too fast for the 3 g tablecloth - tunnelling + grasp
        # ejection).  The hand-motion segments (approach/grasp/close/lift)
        # are NOT scaled; only the walking/turning root motion is.
        ms = float(self.args.motion_speed)
        self.rt_retreat = RETREAT_TIME / ms
        self.rt_turn_out = TURN_OUT_TIME / ms
        self.rt_center = CENTER_MOVE_TIME / ms
        self.rt_turn_in = TURN_IN_TIME / ms
        self.rt_pre_laydown = PRE_LAYDOWN_TIME / ms
        self.rt_laydown = LAYDOWN_TIME / ms
        self.rt_approach = self.rt_pre_laydown + self.rt_laydown
        self.rt_pre_approach = self.rt_retreat + self.rt_turn_out + self.rt_center + self.rt_turn_in
        self.rt_root_duration = self.rt_pre_approach + self.rt_approach

        self.left_ee_index = self._body_index("left_j7")
        self.right_ee_index = self._body_index("right_j7")
        self.left_ee_offset = TCP_OFFSET
        self.right_ee_offset = TCP_OFFSET

        self._setup_ik()

        self.left_home_tf = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        self.right_home_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        self.right_approach_tf = RIGHT_APPROACH_TF
        self.left_approach_tf = self._mirror_about_tablecloth_center_y(RIGHT_APPROACH_TF)
        self.right_grasp_tf = self._offset_transform(RIGHT_GRASP_TF, wp.vec3(0.0, 0.0, GRASP_HEIGHT_OFFSET))
        self.left_grasp_tf = self._mirror_about_tablecloth_center_y(self.right_grasp_tf)
        self.left_lift_tf = self._offset_transform(self.left_grasp_tf, wp.vec3(0.0, 0.0, LIFT_HEIGHT))
        self.right_lift_tf = self._offset_transform(self.right_grasp_tf, wp.vec3(0.0, 0.0, LIFT_HEIGHT))
        self.left_pre_laydown_tf = self._offset_transform(self.left_lift_tf, wp.vec3(0.7 * LAYDOWN_FORWARD, 0.0, 0.0))
        self.right_pre_laydown_tf = self._offset_transform(self.right_lift_tf, wp.vec3(0.7 * LAYDOWN_FORWARD, 0.0, 0.0))
        self.left_laydown_tf = self._offset_transform(
            self.left_lift_tf, wp.vec3(LAYDOWN_FORWARD, 0.0, -LIFT_HEIGHT + LAYDOWN_HEIGHT)
        )
        self.right_laydown_tf = self._offset_transform(
            self.right_lift_tf, wp.vec3(LAYDOWN_FORWARD, 0.0, -LIFT_HEIGHT + LAYDOWN_HEIGHT)
        )
        self.left_post_release_tf = self._offset_transform(self.left_laydown_tf, wp.vec3(0.0, 0.0, 0.16))
        self.right_post_release_tf = self._offset_transform(self.right_laydown_tf, wp.vec3(0.0, 0.0, 0.16))
        self.left_tf = self.left_home_tf
        self.right_tf = self.right_home_tf

        self.motion_segments = self._build_motion_segments()
        self.locked_q_indices, self.locked_q_values = self._build_locked_joint_arrays()
        self.hand_q_indices, self.hand_open, self.hand_grasp = self._build_hand_targets()
        self.arm_coord_indices, self.arm_target_indices = self._arm_q_indices()
        self.hand_shape_indices = self._hand_shape_indices()
        self.hand_shape_indices_wp = wp.array(self.hand_shape_indices, dtype=wp.int32, device=self.device)
        table_shape_idx_range = np.arange(self.table_shape_start, self.table_shape_end, dtype=np.int32)
        self.table_shape_indices_wp = wp.array(table_shape_idx_range, dtype=wp.int32, device=self.device)

        # Kinematic root-motion target (7 free-joint coords).  Updated outside the
        # CUDA graph each frame; the captured simulate() copies it into state_0.
        self.root_target_q = wp.zeros(7, dtype=wp.float32, device=self.device)
        # grasp close fraction; updated outside the graph, read inside it.
        self.grasp_alpha_buf = wp.zeros(1, dtype=wp.float32, device=self.device)

        # Initial PD targets come from the model (home pose set in _configure_robot).
        # The free base target is unused (ke=0; the base is driven kinematically).

        self.initial_cloth_height = float(np.max(self.state_0.particle_q.numpy()[:, 2]))
        self.max_cloth_height = self.initial_cloth_height

        newton.examples.configure_coupled_view(self, args)

        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(CAMERA_POS, CAMERA_PITCH, CAMERA_YAW)
        self._report_pose(force=True)

        self.use_graph = bool(args.graph_capture) and self.device.is_cuda
        self.graph = None
        self.capture()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    def _build_scene(self) -> None:
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.default_joint_cfg.armature = 0.02
        builder.default_shape_cfg.ke = 5.0e5
        builder.default_shape_cfg.kd = 1.0e-6
        builder.default_shape_cfg.mu = 2.0

        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)

        self.robot_body_start = builder.body_count
        self.robot_joint_start = builder.joint_count
        self.robot_shape_start = builder.shape_count
        urdf_path = _resolve_robot_urdf(getattr(self.args, "robot_urdf", None))
        builder.add_urdf(
            str(urdf_path),
            xform=self._robot_xform(),
            floating=True,
            enable_self_collisions=bool(self.args.enable_self_collisions),
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.robot_body_end = builder.body_count
        self.robot_joint_end = builder.joint_count
        self.robot_shape_end = builder.shape_count
        self.franka_bodies = list(range(self.robot_body_start, self.robot_body_end))
        self.franka_joints = list(range(self.robot_joint_start, self.robot_joint_end))
        self.franka_shapes = list(range(self.robot_shape_start, self.robot_shape_end))
        self._configure_robot(builder)

        # Body-level gravity compensation so the free base does not fall; the
        # base pose is overwritten each step by the scripted root motion.
        body_gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        if body_gravcomp.values is None:
            body_gravcomp.values = {}
        for body in self.franka_bodies:
            body_gravcomp.values[body] = 1.0

        # Hand links are the proxy bodies coupled into VBD for cloth contact.
        self.gripper_bodies = [
            body
            for body in self.franka_bodies
            if any(kw in builder.body_label[body].lower() for kw in HAND_CONTACT_KEYWORDS)
        ]
        if not self.gripper_bodies:
            raise RuntimeError("No dexforce hand bodies found for proxy coupling")

        self.table_shape_start = builder.shape_count
        table_cfg = newton.ModelBuilder.ShapeConfig()
        table_cfg.ke = 5.0e5
        table_cfg.kd = 1.0e-6
        table_cfg.mu = TABLE_CONTACT_MU_NORMAL
        self.table_shape = builder.add_shape_box(
            body=-1,
            xform=wp.transform(TABLE_POS, wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=TABLE_COLOR,
            label="place_tablecloth_table",
        )
        self.ground_shape = builder.add_ground_plane(cfg=table_cfg, label="ground")
        self.table_shape_end = builder.shape_count

        self.cloth_particle_start = builder.particle_count
        self._add_cloth(builder)
        self.cloth_particle_end = builder.particle_count
        if self.cloth_particle_end <= self.cloth_particle_start:
            raise RuntimeError("The tablecloth mesh produced no particles")

        builder.color(include_bending=True)
        # Configure shape flags BEFORE finalize so the precomputed
        # shape_contact_pairs reflect the desired collision graph.  Robot
        # shapes collide with particles only (no rigid-rigid); non-hand robot
        # shapes are dropped from particle collision too so the broad phase
        # does not enumerate 86 mesh shapes against the cloth.
        self._configure_shape_flags(builder)
        self.model = builder.finalize()
        self.device = self.model.device

        # Flat rest bending angles (the grid is already flat, but keep this for
        # consistency with the cloth-folding example).
        if self.model.edge_rest_angle is not None:
            self.model.edge_rest_angle.zero_()

        self.model.soft_contact_ke = 1.0e6
        self.model.soft_contact_kd = 1.0e-6
        self.model.soft_contact_mu = 2.0

    def _configure_shape_flags(self, builder: newton.ModelBuilder) -> None:
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape_idx in range(builder.shape_count):
            body_idx = int(builder.shape_body[shape_idx])
            is_robot = shape_idx < self.robot_shape_end
            if is_robot and body_idx >= 0:
                link_name = builder.body_label[body_idx].lower()
                is_hand = any(kw in link_name for kw in HAND_CONTACT_KEYWORDS)
                # Robot shapes never do rigid-rigid; only hand shapes collide
                # with the cloth particles.
                builder.shape_flags[shape_idx] &= ~collide_shapes
                if not is_hand:
                    builder.shape_flags[shape_idx] &= ~collide_particles
                else:
                    builder.shape_flags[shape_idx] |= collide_particles

    def _robot_xform(self) -> wp.transform:
        return wp.transform(wp.vec3(0.0, TABLECLOTH_START_Y, 0.0), wp.quat_identity())

    def _configure_robot(self, builder: newton.ModelBuilder) -> None:
        qd_start = builder.joint_qd_start
        q_start = builder.joint_q_start
        joint_type = builder.joint_type
        joint_parent = builder.joint_parent
        for joint_idx in range(builder.joint_count):
            dof_begin = int(qd_start[joint_idx])
            dof_end = int(qd_start[joint_idx + 1]) if joint_idx + 1 < builder.joint_count else builder.joint_dof_count
            q_begin = int(q_start[joint_idx])
            q_end = int(q_start[joint_idx + 1]) if joint_idx + 1 < builder.joint_count else builder.joint_coord_count
            is_free_root = (
                int(joint_type[joint_idx]) == int(newton.JointType.FREE) and int(joint_parent[joint_idx]) == -1
            )
            for local_dof, dof_idx in enumerate(range(dof_begin, dof_end)):
                q_idx = q_begin + local_dof
                if q_idx < q_end:
                    builder.joint_target_pos[dof_idx] = builder.joint_q[q_idx]
                # Free base: no PD (driven kinematically). Arms: position PD.
                builder.joint_target_ke[dof_idx] = 0.0 if is_free_root else 650.0
                builder.joint_target_kd[dof_idx] = 0.0 if is_free_root else 65.0
                builder.joint_effort_limit[dof_idx] = 0.0 if is_free_root else 180.0
                builder.joint_armature[dof_idx] = 0.0 if is_free_root else 0.02

        for joint_name in (*self.LEFT_HAND_JOINTS, *self.RIGHT_HAND_JOINTS):
            joint_idx = self._builder_joint_index(builder, joint_name)
            dof_idx = int(qd_start[joint_idx])
            builder.joint_target_ke[dof_idx] = 950.0
            builder.joint_target_kd[dof_idx] = 75.0
            builder.joint_effort_limit[dof_idx] = 45.0
            builder.joint_armature[dof_idx] = 0.005

    def _add_cloth(self, builder: newton.ModelBuilder) -> None:
        builder.add_cloth_grid(
            pos=CLOTH_POS,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=CLOTH_DIM_X,
            dim_y=CLOTH_DIM_Y,
            cell_x=CLOTH_CELL_X,
            cell_y=CLOTH_CELL_Y,
            mass=0.003,
            tri_ke=7.0e2,
            tri_ka=7.0e2,
            tri_kd=5.0e-5,
            edge_ke=0.35,
            edge_kd=0.12,
            particle_radius=self.particle_radius,
            label="bimanual_place_tablecloth",
        )

    def _configure_robot_contacts(self) -> None:
        shape_ke = self.model.shape_material_ke.numpy().copy()
        shape_kd = self.model.shape_material_kd.numpy().copy()
        shape_mu = self.model.shape_material_mu.numpy().copy()
        shape_body = self.model.shape_body.numpy()
        for shape_idx in range(self.robot_shape_end):
            body_idx = int(shape_body[shape_idx])
            if body_idx < 0:
                continue
            link_name = self.model.body_label[body_idx].lower()
            shape_ke[shape_idx] = ROBOT_CONTACT_KE
            shape_kd[shape_idx] = ROBOT_CONTACT_KD
            shape_mu[shape_idx] = ROBOT_CONTACT_MU
            if any(kw in link_name for kw in HAND_CONTACT_KEYWORDS):
                shape_ke[shape_idx] = HAND_CONTACT_KE
                shape_kd[shape_idx] = HAND_CONTACT_KD
                shape_mu[shape_idx] = HAND_CONTACT_MU
        self.model.shape_material_ke = wp.array(shape_ke, dtype=self.model.shape_material_ke.dtype, device=self.device)
        self.model.shape_material_kd = wp.array(shape_kd, dtype=self.model.shape_material_kd.dtype, device=self.device)
        self.model.shape_material_mu = wp.array(shape_mu, dtype=self.model.shape_material_mu.dtype, device=self.device)

    def _hand_shape_indices(self) -> np.ndarray:
        shape_body = self.model.shape_body.numpy()
        indices = []
        for shape_idx in range(self.robot_shape_end):
            body_idx = int(shape_body[shape_idx])
            if body_idx < 0:
                continue
            if any(kw in self.model.body_label[body_idx].lower() for kw in HAND_CONTACT_KEYWORDS):
                indices.append(shape_idx)
        return np.array(indices, dtype=np.int32)

    def _set_hand_contact_mu(self, hand_mu: float) -> None:
        if self.hand_shape_indices.size == 0:
            return
        wp.launch(
            set_shape_material_mu_kernel,
            dim=self.hand_shape_indices.shape[0],
            inputs=[self.hand_shape_indices_wp, float(hand_mu), self.model.shape_material_mu],
            device=self.device,
        )

    def _set_table_contact_mu(self, table_mu: float) -> None:
        wp.launch(
            set_shape_material_mu_kernel,
            dim=self.table_shape_indices_wp.shape[0],
            inputs=[self.table_shape_indices_wp, float(table_mu), self.model.shape_material_mu],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Coupled solver
    # ------------------------------------------------------------------
    def _build_solver(self, args) -> None:
        cloth_particles = list(range(self.cloth_particle_start, self.cloth_particle_end))
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mjc",
                    solver=lambda view: SolverMuJoCo(
                        model=view,
                        solver="newton",
                        integrator="implicitfast",
                        cone="elliptic",
                        iterations=int(args.mujoco_iterations),
                        ls_iterations=int(args.mujoco_ls_iterations),
                        use_mujoco_contacts=False,
                        njmax=512,
                        nconmax=256,
                    ),
                    bodies=self.franka_bodies,
                    joints=self.franka_joints,
                ),
                SolverCoupledProxy.Entry(
                    name="vbd",
                    solver=lambda view: SolverVBD(
                        model=view,
                        iterations=int(args.vbd_iterations),
                        rigid_contact_history=False,
                        particle_enable_self_contact=bool(args.cloth_self_contact),
                        particle_self_contact_radius=self.self_contact_radius,
                        particle_self_contact_margin=self.self_contact_margin,
                        particle_topological_contact_filter_threshold=1,
                        particle_rest_shape_contact_exclusion_radius=0.03,
                        particle_vertex_contact_buffer_size=96,
                        particle_edge_contact_buffer_size=128,
                        rigid_body_particle_contact_buffer_size=1024,
                        particle_collision_detection_interval=-1,
                    ),
                    bodies=[],
                    particles=cloth_particles,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mjc",
                        destination="vbd",
                        bodies=self.gripper_bodies,
                        mass_scale=float(args.mass_scale),
                        mode=args.coupling_mode,
                        # One-way coupling: zero feedback relaxation keeps the
                        # coupling force at zero so the MuJoCo arm tracks IK
                        # targets undisturbed by cloth contact, while a large
                        # mass_scale makes the VBD hand proxies near-kinematic
                        # so the cloth follows them through the contact constraint.
                        proxy_relaxation=float(args.proxy_relaxation),
                        proxy_relaxation_mode=args.proxy_relaxation_mode,
                        collision_pipeline=lambda model: newton.CollisionPipeline(
                            model,
                            broad_phase="explicit",
                            soft_contact_margin=SOFT_CONTACT_MARGIN,
                            contact_matching="latest",
                        ),
                        collide_interval=1,
                    )
                ],
                iterations=int(args.proxy_iterations),
            ),
        )

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------
    def _setup_ik(self) -> None:
        self.ik_model = self._build_ik_model()
        ikb = self._ik_builder
        self.world_count = 1
        self.n_coords = self.ik_model.joint_coord_count
        self.ik_joint_q = wp.clone(self.model.joint_q.reshape((1, -1))[:, : self.n_coords])
        self.control_joint_target_q = self.control.joint_target_q.reshape((1, -1))

        left_tcp = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        right_tcp = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        self.left_pos_obj = ik.IKObjectivePosition(
            link_index=self._ik_body_index(ikb, "left_j7"),
            link_offset=self.left_ee_offset,
            target_positions=wp.array([wp.transform_get_translation(left_tcp)], dtype=wp.vec3),
        )
        self.left_rot_obj = ik.IKObjectiveRotation(
            link_index=self._ik_body_index(ikb, "left_j7"),
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([self._quat_to_vec4(wp.transform_get_rotation(left_tcp))], dtype=wp.vec4),
        )
        self.right_pos_obj = ik.IKObjectivePosition(
            link_index=self._ik_body_index(ikb, "right_j7"),
            link_offset=self.right_ee_offset,
            target_positions=wp.array([wp.transform_get_translation(right_tcp)], dtype=wp.vec3),
        )
        self.right_rot_obj = ik.IKObjectiveRotation(
            link_index=self._ik_body_index(ikb, "right_j7"),
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([self._quat_to_vec4(wp.transform_get_rotation(right_tcp))], dtype=wp.vec4),
        )
        lower, upper = self._joint_limits_with_locked_dofs()
        self.joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=wp.array(lower, dtype=wp.float32, device=self.device),
            joint_limit_upper=wp.array(upper, dtype=wp.float32, device=self.device),
            weight=25.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[
                self.left_pos_obj,
                self.left_rot_obj,
                self.right_pos_obj,
                self.right_rot_obj,
                self.joint_limits_obj,
            ],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = int(self.args.ik_iterations)

    def _build_ik_model(self):
        ik_builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        urdf_path = _resolve_robot_urdf(getattr(self.args, "robot_urdf", None))
        ik_builder.add_urdf(
            str(urdf_path),
            xform=self._robot_xform(),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self._ik_builder = ik_builder
        self._ik_body_labels = list(ik_builder.body_label)
        return ik_builder.finalize(device=self.device)

    # ------------------------------------------------------------------
    # Motion script
    # ------------------------------------------------------------------
    def _build_motion_segments(self):
        # Segments 1-6 (home/approach/grasp/close/lift/hang) are hand motion
        # and keep their original timing.  Segments 7-9 (root pre-approach hold,
        # pre-laydown, laydown) must use the motion-speed-scaled root-motion
        # durations so the hands stay closed and only lay down once the robot
        # has actually reached the table; otherwise the extended root motion
        # desynchronises from the hand script and the cloth is released early.
        return (
            (HOME_HOLD_TIME, self.left_home_tf, self.left_home_tf, self.right_home_tf, self.right_home_tf, 0.0, 0.0),
            (
                APPROACH_TIME_HANDS,
                self.left_home_tf,
                self.left_approach_tf,
                self.right_home_tf,
                self.right_approach_tf,
                0.0,
                0.0,
            ),
            (
                GRASP_TIME,
                self.left_approach_tf,
                self.left_grasp_tf,
                self.right_approach_tf,
                self.right_grasp_tf,
                0.0,
                0.0,
            ),
            (CLOSE_TIME, self.left_grasp_tf, self.left_grasp_tf, self.right_grasp_tf, self.right_grasp_tf, 0.0, 1.00),
            (LIFT_TIME, self.left_grasp_tf, self.left_lift_tf, self.right_grasp_tf, self.right_lift_tf, 0.99, 1.0),
            (HANG_SETTLE_TIME, self.left_lift_tf, self.left_lift_tf, self.right_lift_tf, self.right_lift_tf, 1.0, 1.0),
            (
                self.rt_pre_approach,
                self.left_lift_tf,
                self.left_lift_tf,
                self.right_lift_tf,
                self.right_lift_tf,
                1.0,
                1.0,
            ),
            (
                self.rt_pre_laydown,
                self.left_lift_tf,
                self.left_pre_laydown_tf,
                self.right_lift_tf,
                self.right_pre_laydown_tf,
                1.0,
                1.0,
            ),
            (
                self.rt_laydown,
                self.left_pre_laydown_tf,
                self.left_laydown_tf,
                self.right_pre_laydown_tf,
                self.right_laydown_tf,
                1.0,
                1.0,
            ),
            (
                RELEASE_TIME,
                self.left_laydown_tf,
                self.left_laydown_tf,
                self.right_laydown_tf,
                self.right_laydown_tf,
                1.0,
                0.0,
            ),
            (
                POST_RELEASE_HOLD_TIME,
                self.left_laydown_tf,
                self.left_laydown_tf,
                self.right_laydown_tf,
                self.right_laydown_tf,
                0.0,
                0.0,
            ),
            (
                POST_RELEASE_LIFT_TIME,
                self.left_laydown_tf,
                self.left_post_release_tf,
                self.right_laydown_tf,
                self.right_post_release_tf,
                0.0,
                0.0,
            ),
        )

    def _build_hand_targets(self):
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        target_start = self.model.joint_target_q_start.numpy()
        q_indices, open_values, grasp_values = [], [], []
        targets = {
            "HAND_THUMB2": 0.84,
            "HAND_THUMB1": 0.46,
            "HAND_INDEX": 0.70,
            "INDEX_PIP": 0.90,
        }
        for side in ("LEFT", "RIGHT"):
            for suffix, target in targets.items():
                joint_idx = self._joint_index(f"{side}_{suffix}")
                q_idx = int(q_start[joint_idx])
                q_indices.append(int(target_start[joint_idx]))
                open_values.append(float(q_home[q_idx]))
                grasp_values.append(target)
        return (
            wp.array(q_indices, dtype=wp.int32, device=self.device),
            wp.array(open_values, dtype=wp.float32, device=self.device),
            wp.array(grasp_values, dtype=wp.float32, device=self.device),
        )

    def _arm_q_indices(self):
        q_start = self.model.joint_q_start.numpy()
        target_start = self.model.joint_target_q_start.numpy()
        coord_indices, target_indices = [], []
        for name in (*self.LEFT_ARM_JOINTS, *self.RIGHT_ARM_JOINTS):
            joint_idx = self._joint_index(name)
            coord_indices.append(int(q_start[joint_idx]))
            target_indices.append(int(target_start[joint_idx]))
        return (
            wp.array(coord_indices, dtype=wp.int32, device=self.device),
            wp.array(target_indices, dtype=wp.int32, device=self.device),
        )

    def _build_locked_joint_arrays(self):
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        controlled = self._controlled_joint_labels()
        locked_q_indices, locked_q_values = [], []
        for joint_idx, label in enumerate(self.model.joint_label):
            if label in controlled:
                continue
            for q_idx in range(int(q_start[joint_idx]), int(q_start[joint_idx + 1])):
                locked_q_indices.append(q_idx)
                locked_q_values.append(float(q_home[q_idx]))
        return (
            wp.array(locked_q_indices, dtype=wp.int32, device=self.device),
            wp.array(locked_q_values, dtype=wp.float32, device=self.device),
        )

    def _joint_limits_with_locked_dofs(self):
        lower = self.model.joint_limit_lower.numpy().copy()
        upper = self.model.joint_limit_upper.numpy().copy()
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        controlled = self._controlled_joint_labels()
        for joint_idx, label in enumerate(self.model.joint_label):
            if label in controlled:
                continue
            q_begin = int(q_start[joint_idx])
            q_end = int(q_start[joint_idx + 1])
            dof_begin = int(qd_start[joint_idx])
            dof_end = int(qd_start[joint_idx + 1])
            for q_idx, dof_idx in zip(range(q_begin, q_end), range(dof_begin, dof_end), strict=False):
                lower[dof_idx] = q_home[q_idx] - 1.0e-4
                upper[dof_idx] = q_home[q_idx] + 1.0e-4
        return lower, upper

    def _controlled_joint_labels(self) -> set[str]:
        return {f"DexforceW1V021/{name}" for name in (*self.LEFT_ARM_JOINTS, *self.RIGHT_ARM_JOINTS)}

    # ------------------------------------------------------------------
    # Per-frame control
    # ------------------------------------------------------------------
    def _base_motion_time(self, query_time: float) -> float:
        return max(float(query_time) - self.root_motion_start_time, 0.0)

    def _motion_phase(self, query_time: float | None = None) -> str:
        t = self.sim_time if query_time is None else query_time
        base_time = t - self.root_motion_start_time
        if base_time < 0.0:
            return "grasp_lift_hang"
        if base_time < self.rt_retreat:
            return "retreat"
        if base_time < self.rt_retreat + self.rt_turn_out:
            return "turn_out"
        if base_time < self.rt_retreat + self.rt_turn_out + self.rt_center:
            return "move_to_center"
        if base_time < self.rt_pre_approach:
            return "turn_to_table"
        if base_time < self.rt_pre_approach + self.rt_pre_laydown:
            return "pre_laydown"
        if base_time < self.rt_root_duration:
            return "lay_down"
        if base_time < self.rt_root_duration + RELEASE_TIME:
            return "release_hold"
        if base_time < self.rt_root_duration + RELEASE_TIME + POST_RELEASE_HOLD_TIME:
            return "post_release_hold"
        return "post_release_lift"

    def _sample_script(self, query_time: float):
        remaining = query_time
        for duration, left_start, left_end, right_start, right_end, grasp_start, grasp_end in self.motion_segments:
            if remaining <= duration:
                alpha = float(np.clip(remaining / duration, 0.0, 1.0))
                left_tf = self._interpolate_transform(left_start, left_end, alpha)
                right_tf = self._interpolate_transform(right_start, right_end, alpha)
                grasp_alpha = grasp_start * (1.0 - alpha) + grasp_end * alpha
                return left_tf, right_tf, grasp_alpha
            remaining -= duration
        _, _, left_end, _, right_end, _, grasp_end = self.motion_segments[-1]
        return left_end, right_end, grasp_end

    def _apply_root_motion(self, joint_q_out: wp.array, query_time: float) -> None:
        wp.launch(
            set_free_root_motion_kernel,
            dim=1,
            inputs=[
                self.root_q_start,
                self.root_q0,
                self._base_motion_time(query_time),
                RETREAT_DISTANCE,
                RETREAT_TIME,
                math.radians(TURN_OUT_DEGREES),
                TURN_OUT_TIME,
                CENTER_SHIFT_Y,
                CENTER_MOVE_TIME,
                math.radians(TURN_IN_DEGREES),
                TURN_IN_TIME,
                APPROACH_DISTANCE,
                APPROACH_TIME,
            ],
            outputs=[joint_q_out],
            device=self.device,
        )

    def _prepare_frame_targets(self, query_time: float) -> float:
        self.left_tf, self.right_tf, grasp_alpha = self._sample_script(query_time)
        self.left_pos_obj.set_target_position(0, wp.transform_get_translation(self.left_tf))
        self.left_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.left_tf)))
        self.right_pos_obj.set_target_position(0, wp.transform_get_translation(self.right_tf))
        self.right_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.right_tf)))
        # Kinematic root pose for this frame (written to a buffer the captured
        # graph reads back each substep).
        root_tf = self._root_motion_transform(query_time)
        root_pos = wp.transform_get_translation(root_tf)
        root_rot = wp.transform_get_rotation(root_tf)
        self.root_target_q.assign(
            np.array(
                [
                    float(root_pos[0]),
                    float(root_pos[1]),
                    float(root_pos[2]),
                    float(root_rot[0]),
                    float(root_rot[1]),
                    float(root_rot[2]),
                    float(root_rot[3]),
                ],
                dtype=np.float32,
            )
        )
        self.grasp_alpha_buf.assign(np.array([grasp_alpha], dtype=np.float32))
        # Contact friction schedule (in-place writes, safe under graph capture).
        # Time the friction off the same query_time the hand script was sampled
        # at, and key the hand friction off the grasp alpha directly so the
        # release friction kicks in as soon as the fingers start opening.
        #
        # Table friction: LOW for the entire lifted-transport window (retreat /
        # turn / walk / turn / pre-laydown / lay-down).  While the cloth is off
        # the table its hanging hem can swipe the table edge; with normal
        # friction it catches and stalls the robot.  LOW lets it slide off.
        # NORMAL before the grasp (cloth resting) and after the release (cloth
        # placed) so the tablecloth stays put.
        phase = self._motion_phase(query_time)
        if phase in (
            "retreat",
            "turn_out",
            "move_to_center",
            "turn_to_table",
            "pre_laydown",
            "lay_down",
        ):
            self._set_table_contact_mu(TABLE_CONTACT_MU_LOW)
        else:
            self._set_table_contact_mu(TABLE_CONTACT_MU_NORMAL)
        # Hand friction follows the grasp script: high while the fingers are
        # closed, drop to release friction as soon as they start opening.
        if grasp_alpha >= 0.5:
            self._set_hand_contact_mu(HAND_CONTACT_MU)
        else:
            self._set_hand_contact_mu(HAND_RELEASE_CONTACT_MU)
        return grasp_alpha

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    def capture(self) -> None:
        self.graph = None
        if self.use_graph:
            with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
                self.simulate()
            if capture.graph is None:
                raise RuntimeError(f"Graph capture failed on device {self.device}")
            self.graph = capture.graph

    def simulate(self) -> None:
        # IK result drives MuJoCo's joint-space PD targets.
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            lock_joint_q_kernel,
            dim=self.locked_q_indices.shape[0],
            inputs=[self.ik_joint_q, self.locked_q_indices, self.locked_q_values],
            device=self.device,
        )
        wp.launch(
            scatter_arm_targets_kernel,
            dim=self.arm_coord_indices.shape[0],
            inputs=[self.ik_joint_q, self.arm_coord_indices, self.arm_target_indices, self.control.joint_target_q],
            device=self.device,
        )
        wp.launch(
            set_indexed_target_kernel,
            dim=self.hand_q_indices.shape[0],
            inputs=[
                self.hand_q_indices,
                self.hand_open,
                self.hand_grasp,
                self.grasp_alpha_buf,
                self.control.joint_target_q,
            ],
            device=self.device,
        )

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            # Drive the free base kinematically: overwrite the root q in the
            # input state so MuJoCo simulates the arms about the scripted base,
            # then zero the root velocity so the free joint does not drift.
            wp.launch(
                copy_root_q_kernel,
                dim=7,
                inputs=[self.root_q_start, self.root_target_q],
                outputs=[self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                zero_free_root_velocity_kernel,
                dim=6,
                inputs=[self.root_qd_start],
                outputs=[self.state_0.joint_qd],
                device=self.device,
            )
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
            self.model.collide(self.state_0, self.contacts, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self._prepare_frame_targets(self.sim_time + self.frame_dt)
        if self.graph is not None:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame_index += 1
        self._track_cloth_height()
        self._report_pose()

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        if hasattr(self.viewer, "log_coupled_view"):
            newton.examples.log_coupled_view(self, self.contacts)
        else:
            self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/bimanual_place_tablecloth",
            self.state_0.particle_q,
            self.model.tri_indices.flatten(),
            hidden=not getattr(self.viewer, "show_triangles", True),
            backface_culling=False,
            color=CLOTH_COLOR,
        )
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        self._track_cloth_height()

    def test_final(self) -> None:
        if self.use_graph:
            assert self.graph is not None, "Graph capture was requested but no graph was captured"
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        assert np.all(np.isfinite(particle_q)), "Cloth particle positions are not finite"
        assert np.all(np.isfinite(body_q)), "Robot body transforms are not finite"
        bbox = float(np.linalg.norm(particle_q.max(axis=0) - particle_q.min(axis=0)))
        assert bbox < 5.0, f"Cloth particle bounding box exploded: {bbox:.3f} m"
        print(
            f"[cloth_dexforce_place_tablecloth_coupled] "
            f"task_duration={self.sim_time:.2f}s, "
            f"max_cloth_height={self.max_cloth_height:.3f} m"
        )

    def _track_cloth_height(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        self.max_cloth_height = max(self.max_cloth_height, float(np.max(particle_q[:, 2])))

    # ------------------------------------------------------------------
    # Helpers (transforms, indices, reporting)
    # ------------------------------------------------------------------
    def _current_tcp_transform(self, body_index: int, offset: wp.vec3) -> wp.transform:
        body_q_np = self.state_0.body_q.numpy()
        body_tf = wp.transform(*body_q_np[body_index])
        body_pos = wp.transform_get_translation(body_tf)
        body_rot = wp.transform_get_rotation(body_tf)
        tcp_pos = body_pos + wp.quat_rotate(body_rot, offset)
        return wp.transform(tcp_pos, body_rot)

    def _mirror_about_tablecloth_center_y(self, tf: wp.transform) -> wp.transform:
        pos = wp.transform_get_translation(tf)
        quat = wp.transform_get_rotation(tf)
        mirrored_y = 2.0 * TABLECLOTH_START_Y - float(pos[1])
        return wp.transform(
            wp.vec3(float(pos[0]), mirrored_y, float(pos[2])),
            wp.quat(-float(quat[0]), float(quat[1]), -float(quat[2]), float(quat[3])),
        )

    def _offset_transform(self, tf: wp.transform, offset: wp.vec3) -> wp.transform:
        pos = wp.transform_get_translation(tf)
        return wp.transform(
            wp.vec3(
                float(pos[0]) + float(offset[0]), float(pos[1]) + float(offset[1]), float(pos[2]) + float(offset[2])
            ),
            wp.transform_get_rotation(tf),
        )

    def _interpolate_transform(self, tf_a: wp.transform, tf_b: wp.transform, alpha: float) -> wp.transform:
        pos_a = self._vec3_to_np(wp.transform_get_translation(tf_a))
        pos_b = self._vec3_to_np(wp.transform_get_translation(tf_b))
        quat_a = self._quat_to_np(wp.transform_get_rotation(tf_a))
        quat_b = self._quat_to_np(wp.transform_get_rotation(tf_b))
        pos = pos_a * (1.0 - alpha) + pos_b * alpha
        quat = self._slerp_quat_xyzw(quat_a, quat_b, alpha)
        return wp.transform(wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), wp.quat(*quat.tolist()))

    def _slerp_quat_xyzw(self, quat_a: np.ndarray, quat_b: np.ndarray, alpha: float) -> np.ndarray:
        qa = self._normalize_quat(quat_a)
        qb = self._normalize_quat(quat_b)
        dot = float(np.dot(qa, qb))
        if dot < 0.0:
            qb = -qb
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            return self._normalize_quat(qa * (1.0 - alpha) + qb * alpha)
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * alpha
        return qa * (np.sin(theta_0 - theta) / sin_theta_0) + qb * (np.sin(theta) / sin_theta_0)

    def _normalize_quat(self, quat: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(quat))
        if norm == 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return quat / norm

    def _report_pose(self, force: bool = False) -> None:
        if not force and self.print_interval > 0.0 and self.sim_time - self.last_print_time < self.print_interval:
            return
        left_actual = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        right_actual = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        left_target = self._target_to_current_root_world(self.left_tf)
        right_target = self._target_to_current_root_world(self.right_tf)
        lp = self._vec3_to_np(wp.transform_get_translation(left_actual))
        lt = self._vec3_to_np(wp.transform_get_translation(left_target))
        rp = self._vec3_to_np(wp.transform_get_translation(right_actual))
        rt = self._vec3_to_np(wp.transform_get_translation(right_target))
        print(
            f"[{self.sim_time:7.3f}s] L pos_err={float(np.linalg.norm(lp - lt)):.4f} m | "
            f"R pos_err={float(np.linalg.norm(rp - rt)):.4f} m | "
            f"cloth_max_z={self.max_cloth_height:.3f}"
        )
        self.last_print_time = self.sim_time

    def _target_to_current_root_world(self, target_tf: wp.transform) -> wp.transform:
        root_initial_pos = wp.vec3(float(self.root_q0_np[0]), float(self.root_q0_np[1]), float(self.root_q0_np[2]))
        root_initial_rot = wp.quat(
            float(self.root_q0_np[3]), float(self.root_q0_np[4]), float(self.root_q0_np[5]), float(self.root_q0_np[6])
        )
        root_current = self._root_motion_transform(self.sim_time)
        root_current_pos = wp.transform_get_translation(root_current)
        root_current_rot = wp.transform_get_rotation(root_current)
        target_pos = wp.transform_get_translation(target_tf)
        target_rot = wp.transform_get_rotation(target_tf)
        local_pos = wp.quat_rotate_inv(root_initial_rot, target_pos - root_initial_pos)
        local_rot = wp.quat_inverse(root_initial_rot) * target_rot
        world_pos = root_current_pos + wp.quat_rotate(root_current_rot, local_pos)
        world_rot = root_current_rot * local_rot
        return wp.transform(world_pos, world_rot)

    def _root_motion_transform(self, query_time: float) -> wp.transform:
        base_time = self._base_motion_time(query_time)
        t0 = self.rt_retreat
        t1 = t0 + self.rt_turn_out
        t2 = t1 + self.rt_center
        t3 = t2 + self.rt_turn_in
        retreat_u = _smoothstep(base_time / max(self.rt_retreat, 1.0e-6))
        turn_out_u = _smoothstep((base_time - t0) / max(self.rt_turn_out, 1.0e-6))
        center_u = _smoothstep((base_time - t1) / max(self.rt_center, 1.0e-6))
        turn_in_u = _smoothstep((base_time - t2) / max(self.rt_turn_in, 1.0e-6))
        approach_u = _smoothstep((base_time - t3) / max(self.rt_approach, 1.0e-6))
        yaw = math.radians(TURN_OUT_DEGREES) * turn_out_u + math.radians(TURN_IN_DEGREES) * turn_in_u
        q_yaw = wp.quat(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))
        q0 = wp.quat(
            float(self.root_q0_np[3]), float(self.root_q0_np[4]), float(self.root_q0_np[5]), float(self.root_q0_np[6])
        )
        q = q_yaw * q0
        p = wp.vec3(
            float(self.root_q0_np[0]) - RETREAT_DISTANCE * retreat_u + APPROACH_DISTANCE * approach_u,
            float(self.root_q0_np[1]) + CENTER_SHIFT_Y * center_u,
            float(self.root_q0_np[2]),
        )
        return wp.transform(p, q)

    def _body_index(self, body_name: str) -> int:
        suffix = f"/{body_name}"
        return next(i for i, label in enumerate(self.model.body_label) if label.endswith(suffix))

    def _ik_body_index(self, ik_builder, body_name: str) -> int:
        suffix = f"/{body_name}"
        return next(i for i, label in enumerate(ik_builder.body_label) if label.endswith(suffix))

    def _builder_joint_index(self, builder: newton.ModelBuilder, joint_name: str) -> int:
        suffix = f"/{joint_name}"
        return next(i for i, label in enumerate(builder.joint_label) if label.endswith(suffix))

    def _joint_index(self, joint_name: str) -> int:
        suffix = f"/{joint_name}"
        return next(i for i, label in enumerate(self.model.joint_label) if label.endswith(suffix))

    def _root_free_joint_index(self) -> int:
        joint_type = self.model.joint_type.numpy()
        joint_parent = self.model.joint_parent.numpy()
        for i, (jtype, parent) in enumerate(zip(joint_type, joint_parent, strict=False)):
            if int(parent) == -1 and int(jtype) == int(newton.JointType.FREE):
                return i
        raise ValueError("Expected the Dexforce W1 root to be imported as a FREE joint")

    def _quat_to_vec4(self, quat: wp.quat) -> wp.vec4:
        return wp.vec4(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    def _quat_to_np(self, quat: wp.quat) -> np.ndarray:
        return np.array([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])], dtype=np.float64)

    def _vec3_to_np(self, vec: wp.vec3) -> np.ndarray:
        return np.array([float(vec[0]), float(vec[1]), float(vec[2])], dtype=np.float64)

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------
    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        # Default to half speed: the retreat/turn/walk/turn root motion is too
        # fast for the light tablecloth (tunnelling + grasp ejection).  The
        # default num_frames is sized for this default speed.
        default_motion_speed = 0.5
        scaled_root_duration = (
            RETREAT_TIME + TURN_OUT_TIME + CENTER_MOVE_TIME + TURN_IN_TIME + APPROACH_TIME
        ) / default_motion_speed
        scaled_script_duration = (
            ROOT_MOTION_START_TIME
            + scaled_root_duration
            + RELEASE_TIME
            + POST_RELEASE_HOLD_TIME
            + POST_RELEASE_LIFT_TIME
        )
        parser.set_defaults(num_frames=int(math.ceil((scaled_script_duration + FINAL_HOLD_TIME) * 60.0)))
        parser.add_argument(
            "--motion-speed",
            type=float,
            default=default_motion_speed,
            help="Root-motion (retreat/turn/walk/turn/approach) speed multiplier.  <1 slows the "
            "robot down to avoid cloth tunnelling and grasp ejection; raise num_frames to match.",
        )
        parser.add_argument("--substeps", type=int, default=12, help="Coupled substeps per rendered frame.")
        parser.add_argument("--ik-iterations", type=int, default=24, help="Newton GPU IK iterations per frame.")
        parser.add_argument(
            "--print-interval", type=float, default=3.0, help="Seconds between TCP reports; 0.0 prints every frame."
        )
        parser.add_argument("--enable-self-collisions", action="store_true", help="Enable URDF self-collisions.")
        parser.add_argument(
            "--robot-urdf",
            default=None,
            help="Path to DexforceW1V021.urdf or its parent dir (git-ignored assets). "
            "Overrides DEXFORCE_W1_URDF env var; defaults to the local copy next to this file.",
        )
        parser.add_argument("--proxy-iterations", type=int, default=2, help="Proxy coupling passes per substep.")
        parser.add_argument(
            "--mass-scale",
            type=float,
            default=10000.0,
            help="Proxy effective mass/inertia scale.  1e4 makes the VBD hand proxies near-kinematic so the tablecloth cannot be pulled from the grasp by the turn-in centrifugal swing.",
        )
        parser.add_argument(
            "--coupling-mode",
            choices=["lagged", "staggered"],
            default="staggered",
            help="SolverCoupledProxy transfer mode.",
        )
        parser.add_argument(
            "--proxy-relaxation",
            type=float,
            default=0.0,
            help="Feedback relaxation; 0.0 = one-way (no force back to MuJoCo), 1.0 = full two-way.",
        )
        parser.add_argument(
            "--proxy-relaxation-mode",
            choices=["fixed", "aitken"],
            default="fixed",
            help="Feedback relaxation update rule.",
        )
        parser.add_argument("--mujoco-iterations", type=int, default=12)
        parser.add_argument("--mujoco-ls-iterations", type=int, default=25)
        parser.add_argument("--vbd-iterations", type=int, default=24)
        parser.add_argument(
            "--cloth-self-contact",
            action="store_true",
            default=False,
            help="Enable cloth self-contact.  Off by default: the current VBD "
            "self-contact (log-barrier, keyed off soft_contact_ke=1e6) is too "
            "aggressive for this 3 g tablecloth and ejects it from the grasp "
            "during the retreat.  Enable only after retuning the contact stiffness.",
        )
        parser.add_argument(
            "--no-graph-capture",
            action="store_false",
            dest="graph_capture",
            default=True,
            help="Disable CUDA graph capture.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

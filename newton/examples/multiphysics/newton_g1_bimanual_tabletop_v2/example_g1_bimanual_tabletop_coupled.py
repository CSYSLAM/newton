# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""Unitree G1 bimanual tabletop grasp with Newton 1.4 Coupled Proxy solver.

The example intentionally combines four pieces of Newton's current public API:

1. The official 43-DoF Unitree G1 asset and MJWarp locomotion ONNX policy.
2. SolverMuJoCo for the complete floating-base humanoid and its three-finger hands.
3. Newton GPU IK for simultaneous left/right arm task-space targets.
4. SolverVBD plus SolverCoupledProxy for two deformable tabletop objects.

The robot walks to the workbench under the locomotion policy, stops, reaches with
both arms, closes both hands, and lifts one deformable object in each hand.

Run from a Newton checkout or a Python environment with ``newton[examples]``::

    python example_g1_bimanual_tabletop_coupled.py

This corrected file targets Newton v1.4.0 and fixes contact/trajectory issues in the first draft.  ``SolverCoupledProxy`` is currently under the
experimental coupled-solver namespace, so pinning Newton is recommended.
"""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import warp as wp
import yaml
from newton.solvers.experimental.coupled import SolverCoupledProxy
from warp_nn.runtime import OnnxRuntime

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton import JointTargetMode
from newton.solvers import SolverMuJoCo, SolverVBD

# -----------------------------------------------------------------------------
# Scene and task constants
# -----------------------------------------------------------------------------
PHYSICS_DT = 1.0 / 400.0
CONTROL_DECIMATION = 8
CONTROL_DT = PHYSICS_DT * CONTROL_DECIMATION

TABLE_TOP_Z = 0.72
TABLE_CENTER = np.array([0.0, 0.34, TABLE_TOP_Z - 0.025], dtype=np.float32)
TABLE_HALF_EXTENTS = np.array([0.58, 0.42, 0.025], dtype=np.float32)

# Robot starts facing +world-Y.  In the G1 body frame +X is forward and +Y is left.
ROBOT_START_POS = np.array([0.0, -1.12, 0.76], dtype=np.float32)
ROBOT_START_QUAT = np.array([0.0, 0.0, 0.70710678, 0.70710678], dtype=np.float32)
ROBOT_STOP_Y = -0.50
ROBOT_HARD_LIMIT_Y = -0.40

LEFT_OBJECT_CENTER = np.array([-0.17, 0.08, TABLE_TOP_Z + 0.065], dtype=np.float32)
RIGHT_OBJECT_CENTER = np.array([0.17, 0.08, TABLE_TOP_Z + 0.070], dtype=np.float32)

ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

LEFT_HAND_CLOSED = {
    "left_hand_index_0_joint": -0.95,
    "left_hand_index_1_joint": -1.18,
    "left_hand_middle_0_joint": -0.95,
    "left_hand_middle_1_joint": -1.18,
    "left_hand_thumb_0_joint": 0.42,
    "left_hand_thumb_1_joint": 0.68,
    "left_hand_thumb_2_joint": 1.05,
}
RIGHT_HAND_CLOSED = {
    "right_hand_index_0_joint": 0.95,
    "right_hand_index_1_joint": 1.18,
    "right_hand_middle_0_joint": 0.95,
    "right_hand_middle_1_joint": 1.18,
    "right_hand_thumb_0_joint": -0.42,
    "right_hand_thumb_1_joint": -0.68,
    "right_hand_thumb_2_joint": -1.05,
}
HAND_JOINT_NAMES = tuple(LEFT_HAND_CLOSED) + tuple(RIGHT_HAND_CLOSED)

# Point on each wrist-yaw link located approximately in the palm/finger workspace.
LEFT_TCP_OFFSET = wp.vec3(0.125, 0.0, 0.0)
RIGHT_TCP_OFFSET = wp.vec3(0.125, 0.0, 0.0)
LEFT_TCP_OFFSET_NP = np.array([0.125, 0.0, 0.0], dtype=np.float32)
RIGHT_TCP_OFFSET_NP = np.array([0.125, 0.0, 0.0], dtype=np.float32)


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def _require_newton_140() -> None:
    try:
        installed = version("newton")
    except PackageNotFoundError:
        return
    numbers = tuple(int(part) for part in installed.split("+")[0].split(".")[:3])
    if numbers < (1, 4, 0):
        raise RuntimeError(f"This example requires newton>=1.4.0; found {installed}")


def _tail(label: str) -> str:
    return str(label).replace("\\", "/").split("/")[-1]


def _find_label_index(labels: list[str], suffix: str) -> int:
    matches = [i for i, label in enumerate(labels) if _tail(label) == suffix or str(label).endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one label ending in {suffix!r}; got {matches}")
    return matches[0]


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector by unit quaternion, xyzw convention."""
    xyz = q[:3]
    w = q[3]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def _quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector by inverse unit quaternion, xyzw convention."""
    xyz = q[:3]
    w = q[3]
    # q^-1 * v * q, expanded without allocating quaternion objects.
    t = 2.0 * np.cross(-xyz, v)
    return v + w * t + np.cross(-xyz, t)


def _smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _interp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    s = _smoothstep01(alpha)
    return (1.0 - s) * a + s * b


# -----------------------------------------------------------------------------
# Locomotion-policy kernels (same observation/action layout as Newton's
# example_robot_policy.py).
# -----------------------------------------------------------------------------
@wp.kernel
def _compute_policy_observation(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_pos_initial: wp.array[float],
    gravity_w: wp.vec3,
    command: wp.vec3,
    prev_act: wp.array2d[float],
    num_dofs: int,
    obs: wp.array2d[float],
):
    q = wp.quat(joint_q[3], joint_q[4], joint_q[5], joint_q[6])
    lin_w = wp.vec3(joint_qd[0], joint_qd[1], joint_qd[2])
    ang_w = wp.vec3(joint_qd[3], joint_qd[4], joint_qd[5])

    vel_b = wp.quat_rotate_inv(q, lin_w)
    avel_b = wp.quat_rotate_inv(q, ang_w)
    grav_b = wp.quat_rotate_inv(q, gravity_w)

    obs[0, 0] = vel_b[0]
    obs[0, 1] = vel_b[1]
    obs[0, 2] = vel_b[2]
    obs[0, 3] = avel_b[0]
    obs[0, 4] = avel_b[1]
    obs[0, 5] = avel_b[2]
    obs[0, 6] = grav_b[0]
    obs[0, 7] = grav_b[1]
    obs[0, 8] = grav_b[2]
    obs[0, 9] = command[0]
    obs[0, 10] = command[1]
    obs[0, 11] = command[2]

    for k in range(num_dofs):
        obs[0, 12 + k] = joint_q[7 + k] - joint_pos_initial[k]
        obs[0, 12 + num_dofs + k] = joint_qd[6 + k]
        obs[0, 12 + 2 * num_dofs + k] = prev_act[0, k]


@wp.kernel
def _build_policy_targets(
    act: wp.array2d[float],
    joint_pos_initial: wp.array[float],
    action_scale: float,
    num_prefix_zeros: int,
    out: wp.array[float],
):
    i = wp.tid()
    if i < num_prefix_zeros:
        out[i] = 0.0
    else:
        j = i - num_prefix_zeros
        out[i] = joint_pos_initial[j] + action_scale * act[0, j]


@wp.kernel
def _gather_named_joint_coordinates(
    source_q: wp.array[float],
    source_indices: wp.array[int],
    destination_indices: wp.array[int],
    ik_q: wp.array2d[float],
    posture_target: wp.array2d[float],
):
    i = wp.tid()
    value = source_q[source_indices[i]]
    dst = destination_indices[i]
    ik_q[0, dst] = value
    posture_target[0, dst] = value


@wp.kernel
def _scatter_arm_targets(
    ik_q: wp.array2d[float],
    ik_coord_indices: wp.array[int],
    control_target_indices: wp.array[int],
    joint_target_q: wp.array[float],
):
    i = wp.tid()
    joint_target_q[control_target_indices[i]] = ik_q[0, ik_coord_indices[i]]


@wp.kernel
def _set_hand_targets(
    joint_target_q: wp.array[float],
    target_indices: wp.array[int],
    closed_targets: wp.array[float],
    close_alpha: float,
):
    i = wp.tid()
    joint_target_q[target_indices[i]] = close_alpha * closed_targets[i]


# -----------------------------------------------------------------------------
# Joint-posture IK objective.  It freezes every non-arm coordinate to the
# current dynamic pose while leaving the 14 arm joints free for bimanual IK.
# -----------------------------------------------------------------------------
@wp.kernel
def _posture_residuals(
    joint_q: wp.array2d[float],
    target_q: wp.array2d[float],
    selected_coords: wp.array[int],
    start_idx: int,
    weight: float,
    problem_idx_map: wp.array[int],
    residuals: wp.array2d[float],
):
    row, selected = wp.tid()
    base = problem_idx_map[row]
    coord = selected_coords[selected]
    residuals[row, start_idx + selected] = weight * (joint_q[row, coord] - target_q[base, coord])


@wp.kernel
def _posture_jacobian(
    selected_dofs: wp.array[int],
    start_idx: int,
    weight: float,
    jacobian: wp.array3d[float],
):
    row, selected = wp.tid()
    dof = selected_dofs[selected]
    jacobian[row, start_idx + selected, dof] = weight


class JointPostureObjective(ik.IKObjective):
    """Analytic diagonal objective for selected scalar joint coordinates."""

    def __init__(
        self,
        target_q: wp.array2d[float],
        selected_coords: wp.array[int],
        selected_dofs: wp.array[int],
        weight: float,
    ) -> None:
        super().__init__()
        self.target_q = target_q
        self.selected_coords = selected_coords
        self.selected_dofs = selected_dofs
        self.weight = float(weight)
        self.count = int(selected_coords.shape[0])

    def residual_dim(self) -> int:
        return self.count

    def supports_analytic(self) -> bool:
        return True

    def compute_residuals(self, body_q, joint_q, model, residuals, start_idx, problem_idx):
        del body_q, model
        wp.launch(
            _posture_residuals,
            dim=(joint_q.shape[0], self.count),
            inputs=[joint_q, self.target_q, self.selected_coords, start_idx, self.weight, problem_idx],
            outputs=[residuals],
            device=self.device,
        )

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx):
        del body_q, model, joint_S_s
        wp.launch(
            _posture_jacobian,
            dim=(joint_q.shape[0], self.count),
            inputs=[self.selected_dofs, start_idx, self.weight],
            outputs=[jacobian],
            device=self.device,
        )


# -----------------------------------------------------------------------------
# Main example
# -----------------------------------------------------------------------------
class Example:
    def __init__(self, viewer, args):
        _require_newton_140()
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()
        self.sim_time = 0.0
        self.frame_dt = CONTROL_DT
        self.sim_dt = PHYSICS_DT
        self.sim_substeps = CONTROL_DECIMATION
        self.phase = "approach"
        self.phase_start_time = 0.0
        self.settle_start_time: float | None = None
        self.manip_start_time: float | None = None
        self.manip_start_left_world: np.ndarray | None = None
        self.manip_start_right_world: np.ndarray | None = None
        self.stop_y = float(args.robot_stop_y)

        self.asset_dir = Path(newton.utils.download_asset("unitree_g1"))
        self.usd_path = self.asset_dir / "usd" / "g1_isaac.usd"
        self.policy_path = self.asset_dir / "rl_policies" / "mjw_g1_29DOF.onnx"
        self.yaml_path = self.asset_dir / "rl_policies" / "g1_29dof.yaml"

        with self.yaml_path.open(encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        if int(self.config["num_dofs"]) != 43:
            raise RuntimeError("The current G1-with-hands configuration is expected to contain 43 DoFs")

        self._build_scene()
        self._build_coupled_solver()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.collision_pipeline = newton.CollisionPipeline(self.model, broad_phase="explicit")
        self.contacts = self.collision_pipeline.contacts()
        self.solver.prepare_contacts(self.contacts)

        self._build_policy()
        self._build_bimanual_ik()
        newton.examples.configure_coupled_view(self, args)

        self.viewer.set_model(self.model)
        self.viewer.vsync = True
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = True
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(2.15, -2.05, 1.55), pitch=-17.0, yaw=137.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "look_at"):
            self.viewer.camera.look_at(wp.vec3(0.0, -0.05, 0.72))

        print("[INFO] Unitree G1 model:", self.usd_path)
        print("[INFO] Locomotion policy:", self.policy_path)
        print(f"[INFO] G1 rigid bodies={len(self.g1_bodies)}, joints={len(self.g1_joints)}")
        print(f"[INFO] VBD particles={self.soft_particle_end - self.soft_particle_start}")
        print(f"[INFO] Hand proxy bodies={len(self.hand_proxy_bodies)}, proxy joints={len(self.hand_proxy_joints)}")

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------
    def _build_scene(self) -> None:
        builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Z)
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e2,
            limit_kd=1.0,
        )
        builder.default_shape_cfg.ke = 8.0e4
        builder.default_shape_cfg.kd = 8.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.9
        builder.default_particle_radius = 0.012
        builder.rigid_gap = 0.004

        body_start = builder.body_count
        joint_start = builder.joint_count
        shape_start = builder.shape_count
        builder.add_usd(
            newton.examples.get_asset(str(self.usd_path)),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.8), wp.quat_identity()),
            floating=True,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
            hide_collision_shapes=True,
            skip_mesh_approximation=False,
        )
        builder.approximate_meshes("convex_hull")
        self.g1_bodies = list(range(body_start, builder.body_count))
        self.g1_joints = list(range(joint_start, builder.joint_count))
        self.g1_shapes = list(range(shape_start, builder.shape_count))

        # Match Newton's official G1 policy example: free base q + 43 scalar joints.
        if builder.joint_coord_count < 7 + int(self.config["num_dofs"]):
            raise RuntimeError("Imported G1 articulation does not expose the expected free base plus 43 joints")
        builder.joint_q[:3] = ROBOT_START_POS.tolist()
        builder.joint_q[3:7] = ROBOT_START_QUAT.tolist()
        builder.joint_q[7 : 7 + 43] = self.config["mjw_joint_pos"]
        builder.joint_target_q[:6] = [0.0] * 6

        for i in range(43):
            dof = i + 6
            builder.joint_target_ke[dof] = float(self.config["mjw_joint_stiffness"][i])
            builder.joint_target_kd[dof] = float(self.config["mjw_joint_damping"][i])
            builder.joint_armature[dof] = float(self.config["mjw_joint_armature"][i])
            builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)

        self._add_realistic_workbench(builder)

        self.soft_particle_start = builder.particle_count
        self._add_soft_sponge(builder)
        self.left_particle_end = builder.particle_count
        self._add_soft_carton(builder)
        self.soft_particle_end = builder.particle_count

        self.hand_proxy_bodies = [
            body
            for body in self.g1_bodies
            if any(
                token in _tail(builder.body_label[body])
                for token in (
                    "wrist_yaw_link",
                    "hand_thumb",
                    "hand_index",
                    "hand_middle",
                )
            )
        ]
        if len(self.hand_proxy_bodies) < 14:
            raise RuntimeError(
                "Could not find the full pair of wrist/three-finger hand collision chains; "
                f"found {len(self.hand_proxy_bodies)} bodies"
            )

        # Keep the internal revolute joints of each three-finger hand enabled in
        # the VBD proxy view.  Treating every finger link as an independent free
        # proxy body lets links drift apart under contact and is a major source of
        # apparent hand/table tunnelling.
        proxy_body_set = set(self.hand_proxy_bodies)
        self.hand_proxy_joints = []
        for joint in self.g1_joints:
            parent = int(builder.joint_parent[joint])
            child = int(builder.joint_child[joint])
            if child in proxy_body_set and (parent < 0 or parent in proxy_body_set):
                self.hand_proxy_joints.append(joint)

        builder.color()
        self.model = builder.finalize(device=self.device)
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.model.soft_contact_ke = float(self.args.soft_contact_ke)
        self.model.soft_contact_kd = 5.0e-4
        self.model.soft_contact_kf = 1.0e3
        self.model.soft_contact_mu = 1.4

    def _add_realistic_workbench(self, builder: newton.ModelBuilder) -> None:
        collision_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=1.2e5,
            kd=1.0e3,
            kf=1.0e3,
            mu=0.95,
            margin=0.004,
            gap=0.001,
        )
        visual_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        self.ground_shape = builder.add_ground_plane(cfg=collision_cfg)

        # Hardwood worktop.
        self.table_top_shape = builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(*TABLE_CENTER), wp.quat_identity()),
            hx=float(TABLE_HALF_EXTENTS[0]),
            hy=float(TABLE_HALF_EXTENTS[1]),
            hz=float(TABLE_HALF_EXTENTS[2]),
            cfg=collision_cfg,
            color=wp.vec3(0.32, 0.16, 0.075),
            label="workbench_top",
        )

        # Four steel legs and lower shelf; all are genuine static collision geometry.
        leg_x = float(TABLE_HALF_EXTENTS[0] - 0.055)
        leg_y = float(TABLE_CENTER[1] + TABLE_HALF_EXTENTS[1] - 0.06)
        leg_z = 0.5 * (TABLE_TOP_Z - 0.05)
        for x in (-leg_x, leg_x):
            for y in (float(TABLE_CENTER[1] - TABLE_HALF_EXTENTS[1] + 0.06), leg_y):
                builder.add_shape_box(
                    body=-1,
                    xform=wp.transform(wp.vec3(x, y, leg_z), wp.quat_identity()),
                    hx=0.035,
                    hy=0.035,
                    hz=leg_z,
                    cfg=collision_cfg,
                    color=wp.vec3(0.12, 0.13, 0.15),
                    label="workbench_leg",
                )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, TABLE_CENTER[1] + 0.03, 0.25), wp.quat_identity()),
            hx=0.48,
            hy=0.31,
            hz=0.018,
            cfg=collision_cfg,
            color=wp.vec3(0.16, 0.17, 0.19),
            label="lower_shelf",
        )

        # Visual backsplash, rubber task mat, tool rail and two bins make the scene
        # read like a workshop rather than an isolated box-on-plane benchmark.
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.755, 1.06), wp.quat_identity()),
            hx=0.58,
            hy=0.018,
            hz=0.34,
            cfg=visual_cfg,
            color=wp.vec3(0.30, 0.32, 0.35),
            label="workbench_backsplash_visual",
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.07, TABLE_TOP_Z + 0.006), wp.quat_identity()),
            hx=0.34,
            hy=0.20,
            hz=0.006,
            cfg=visual_cfg,
            color=wp.vec3(0.055, 0.065, 0.075),
            label="rubber_task_mat",
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(-0.46, 0.53, TABLE_TOP_Z + 0.08), wp.quat_identity()),
            hx=0.08,
            hy=0.12,
            hz=0.08,
            cfg=visual_cfg,
            color=wp.vec3(0.14, 0.34, 0.50),
            label="left_storage_bin_visual",
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.46, 0.53, TABLE_TOP_Z + 0.08), wp.quat_identity()),
            hx=0.08,
            hy=0.12,
            hz=0.08,
            cfg=visual_cfg,
            color=wp.vec3(0.48, 0.20, 0.12),
            label="right_storage_bin_visual",
        )

    @staticmethod
    def _add_soft_sponge(builder: newton.ModelBuilder) -> None:
        # 8 x 7 x 10 cm compliant cleaning sponge.
        dims = np.array([0.08, 0.07, 0.10], dtype=np.float32)
        corner = LEFT_OBJECT_CENTER - 0.5 * dims
        builder.add_soft_grid(
            pos=wp.vec3(*corner),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            dim_z=3,
            cell_x=float(dims[0] / 2),
            cell_y=float(dims[1] / 2),
            cell_z=float(dims[2] / 3),
            density=125.0,
            k_mu=3.0e3,
            k_lambda=7.0e3,
            k_damp=0.18,
            tri_ke=1.0e2,
            tri_ka=1.0e2,
            tri_kd=0.02,
            particle_radius=0.011,
            label="left_soft_sponge",
        )

    @staticmethod
    def _add_soft_carton(builder: newton.ModelBuilder) -> None:
        # 8 x 8 x 11 cm foam-filled retail carton: stiffer than the sponge but
        # still deformable, giving the two hands visibly different responses.
        dims = np.array([0.08, 0.08, 0.11], dtype=np.float32)
        corner = RIGHT_OBJECT_CENTER - 0.5 * dims
        builder.add_soft_grid(
            pos=wp.vec3(*corner),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=2,
            dim_y=2,
            dim_z=3,
            cell_x=float(dims[0] / 2),
            cell_y=float(dims[1] / 2),
            cell_z=float(dims[2] / 3),
            density=260.0,
            k_mu=1.1e4,
            k_lambda=2.8e4,
            k_damp=0.28,
            tri_ke=2.0e2,
            tri_ka=2.0e2,
            tri_kd=0.03,
            particle_radius=0.011,
            label="right_soft_carton",
        )

    # ------------------------------------------------------------------
    # Coupled solver
    # ------------------------------------------------------------------
    def _build_coupled_solver(self) -> None:
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
                        iterations=int(self.args.mujoco_iterations),
                        ls_iterations=int(self.args.mujoco_ls_iterations),
                        use_mujoco_contacts=False,
                        nconmax=4096,
                        njmax=16384,
                    ),
                    bodies=self.g1_bodies,
                    joints=self.g1_joints,
                ),
                SolverCoupledProxy.Entry(
                    name="vbd",
                    solver=lambda view: SolverVBD(
                        model=view,
                        iterations=int(self.args.vbd_iterations),
                        particle_enable_self_contact=False,
                        particle_enable_tile_solve=False,
                        rigid_contact_hard=False,
                        rigid_contact_history=False,
                        rigid_joint_linear_ke=2.0e7,
                        rigid_joint_angular_ke=2.0e6,
                        rigid_body_particle_contact_buffer_size=16384,
                        rigid_avbd_beta=0.5,
                        rigid_contact_k_start=0.2,
                    ),
                    particles=list(range(self.soft_particle_start, self.soft_particle_end)),
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mjc",
                        destination="vbd",
                        bodies=self.hand_proxy_bodies,
                        joints=self.hand_proxy_joints,
                        mass_scale=float(self.args.proxy_mass_scale),
                        mode=self.args.coupling_mode,
                        proxy_relaxation=float(self.args.proxy_relaxation),
                        proxy_relaxation_mode=self.args.proxy_relaxation_mode,
                        proxy_relaxation_min=0.1,
                        proxy_relaxation_max=1.0,
                        collision_pipeline=lambda model: newton.examples.create_collision_pipeline(
                            model,
                            broad_phase="explicit",
                        ),
                        collide_interval=1,
                    )
                ],
                iterations=int(self.args.proxy_iterations),
            ),
        )

    # ------------------------------------------------------------------
    # Locomotion policy
    # ------------------------------------------------------------------
    def _build_policy(self) -> None:
        self.policy = OnnxRuntime(str(self.policy_path), device=self.device)
        self.policy_input_name = self.policy.input_names[0]
        self.policy_output_name = self.policy.output_names[0]
        self.num_dofs = 43
        self.joint_pos_initial = wp.array(
            np.asarray(self.config["mjw_joint_pos"], dtype=np.float32),
            dtype=wp.float32,
            device=self.device,
        )
        self.policy_obs = wp.zeros((1, 12 + 3 * self.num_dofs), dtype=wp.float32, device=self.device)
        self.prev_action = wp.zeros((1, self.num_dofs), dtype=wp.float32, device=self.device)
        self.gravity_w = wp.vec3(0.0, 0.0, -1.0)
        self.command = wp.vec3(0.0, 0.0, 0.0)

    def _run_policy(self) -> None:
        wp.launch(
            _compute_policy_observation,
            dim=1,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.joint_pos_initial,
                self.gravity_w,
                self.command,
                self.prev_action,
                self.num_dofs,
                self.policy_obs,
            ],
            device=self.device,
        )
        outputs = self.policy({self.policy_input_name: self.policy_obs})
        action = outputs[self.policy_output_name]
        wp.launch(
            _build_policy_targets,
            dim=6 + self.num_dofs,
            inputs=[
                action,
                self.joint_pos_initial,
                float(self.config["action_scale"]),
                6,
                self.control.joint_target_q,
            ],
            device=self.device,
        )
        wp.copy(self.prev_action, action)

    # ------------------------------------------------------------------
    # Bimanual IK model and mapping
    # ------------------------------------------------------------------
    def _build_bimanual_ik(self) -> None:
        ik_builder = newton.ModelBuilder(gravity=-9.81, up_axis=newton.Axis.Z)
        ik_builder.add_usd(
            newton.examples.get_asset(str(self.usd_path)),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
            hide_collision_shapes=True,
            load_visual_shapes=False,
            skip_mesh_approximation=True,
        )
        self.ik_model = ik_builder.finalize(device=self.device)
        self.ik_ncoords = self.ik_model.joint_coord_count
        self.ik_joint_q = wp.zeros((1, self.ik_ncoords), dtype=wp.float32, device=self.device)
        self.ik_posture_target = wp.zeros((1, self.ik_ncoords), dtype=wp.float32, device=self.device)

        dyn_q_start = self.model.joint_q_start.numpy()
        dyn_target_start = self.model.joint_target_q_start.numpy()
        ik_q_start = self.ik_model.joint_q_start.numpy()
        ik_qd_start = self.ik_model.joint_qd_start.numpy()

        source_q_indices: list[int] = []
        destination_q_indices: list[int] = []
        arm_ik_coords: list[int] = []
        arm_control_targets: list[int] = []
        selected_posture_coords: list[int] = []
        selected_posture_dofs: list[int] = []

        for name in self.config["mjw_joint_names"]:
            dyn_joint = _find_label_index(self.model.joint_label, name)
            ik_joint = _find_label_index(self.ik_model.joint_label, name)
            dyn_coord = int(dyn_q_start[dyn_joint])
            ik_coord = int(ik_q_start[ik_joint])
            source_q_indices.append(dyn_coord)
            destination_q_indices.append(ik_coord)
            if name in ARM_JOINT_NAMES:
                arm_ik_coords.append(ik_coord)
                arm_control_targets.append(int(dyn_target_start[dyn_joint]))
            else:
                selected_posture_coords.append(ik_coord)
                selected_posture_dofs.append(int(ik_qd_start[ik_joint]))

        self.dynamic_source_q_indices = wp.array(source_q_indices, dtype=wp.int32, device=self.device)
        self.ik_destination_q_indices = wp.array(destination_q_indices, dtype=wp.int32, device=self.device)
        self.arm_ik_coords = wp.array(arm_ik_coords, dtype=wp.int32, device=self.device)
        self.arm_control_targets = wp.array(arm_control_targets, dtype=wp.int32, device=self.device)

        hand_target_indices: list[int] = []
        hand_closed_values: list[float] = []
        closed = {**LEFT_HAND_CLOSED, **RIGHT_HAND_CLOSED}
        for name in HAND_JOINT_NAMES:
            dyn_joint = _find_label_index(self.model.joint_label, name)
            hand_target_indices.append(int(dyn_target_start[dyn_joint]))
            hand_closed_values.append(float(closed[name]))
        self.hand_target_indices = wp.array(hand_target_indices, dtype=wp.int32, device=self.device)
        self.hand_closed_values = wp.array(hand_closed_values, dtype=wp.float32, device=self.device)

        selected_coords_wp = wp.array(selected_posture_coords, dtype=wp.int32, device=self.device)
        selected_dofs_wp = wp.array(selected_posture_dofs, dtype=wp.int32, device=self.device)
        posture_obj = JointPostureObjective(
            target_q=self.ik_posture_target,
            selected_coords=selected_coords_wp,
            selected_dofs=selected_dofs_wp,
            weight=float(self.args.posture_weight),
        )

        left_wrist = _find_label_index(self.ik_model.body_label, "left_wrist_yaw_link")
        right_wrist = _find_label_index(self.ik_model.body_label, "right_wrist_yaw_link")
        self.left_wrist_body = _find_label_index(self.model.body_label, "left_wrist_yaw_link")
        self.right_wrist_body = _find_label_index(self.model.body_label, "right_wrist_yaw_link")
        self.left_target_position = wp.array([wp.vec3(0.35, 0.18, 0.05)], dtype=wp.vec3, device=self.device)
        self.right_target_position = wp.array([wp.vec3(0.35, -0.18, 0.05)], dtype=wp.vec3, device=self.device)

        # Identity in pelvis coordinates keeps both palms pointing forward.  Rotation
        # has lower weight than position so the solver can use wrist redundancy.
        self.left_target_rotation = wp.array([wp.vec4(0.0, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=self.device)
        self.right_target_rotation = wp.array([wp.vec4(0.0, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=self.device)

        self.left_pos_obj = ik.IKObjectivePosition(
            link_index=left_wrist,
            link_offset=LEFT_TCP_OFFSET,
            target_positions=self.left_target_position,
            weight=1.0,
        )
        self.right_pos_obj = ik.IKObjectivePosition(
            link_index=right_wrist,
            link_offset=RIGHT_TCP_OFFSET,
            target_positions=self.right_target_position,
            weight=1.0,
        )
        self.left_rot_obj = ik.IKObjectiveRotation(
            link_index=left_wrist,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self.left_target_rotation,
            weight=0.18,
        )
        self.right_rot_obj = ik.IKObjectiveRotation(
            link_index=right_wrist,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self.right_target_rotation,
            weight=0.18,
        )
        joint_limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.ik_model.joint_limit_lower,
            joint_limit_upper=self.ik_model.joint_limit_upper,
            weight=12.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=1,
            objectives=[
                self.left_pos_obj,
                self.right_pos_obj,
                self.left_rot_obj,
                self.right_rot_obj,
                posture_obj,
                joint_limit_obj,
            ],
            lambda_initial=0.06,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _seed_ik_from_dynamic_state(self) -> None:
        wp.launch(
            _gather_named_joint_coordinates,
            dim=len(self.config["mjw_joint_names"]),
            inputs=[
                self.state_0.joint_q,
                self.dynamic_source_q_indices,
                self.ik_destination_q_indices,
                self.ik_joint_q,
                self.ik_posture_target,
            ],
            device=self.device,
        )

    def _world_to_pelvis_position(self, world_position: np.ndarray, root_q: np.ndarray) -> np.ndarray:
        root_p = root_q[:3]
        root_rot = root_q[3:7]
        return _quat_rotate_inv(root_rot, world_position - root_p).astype(np.float32)

    def _body_point_world(self, body_index: int, local_point: np.ndarray) -> np.ndarray:
        body_q = self.state_0.body_q.numpy()[body_index].astype(np.float32)
        return body_q[:3] + _quat_rotate(body_q[3:7], local_point)

    def _manipulation_targets(self, task_t: float) -> tuple[np.ndarray, np.ndarray, float]:
        # Reach in collision-safe stages.  The original version jumped directly
        # to a pre-grasp target and then drove the TCP into the object centre.
        # With strong position actuators that can overpower a finite contact
        # solve and push the forearm/palm through the tabletop.
        left_hover = LEFT_OBJECT_CENTER + np.array([0.0, -0.10, 0.22], dtype=np.float32)
        right_hover = RIGHT_OBJECT_CENTER + np.array([0.0, -0.10, 0.22], dtype=np.float32)
        left_front = LEFT_OBJECT_CENTER + np.array([0.0, -0.085, 0.075], dtype=np.float32)
        right_front = RIGHT_OBJECT_CENTER + np.array([0.0, -0.085, 0.075], dtype=np.float32)
        left_grasp = LEFT_OBJECT_CENTER + np.array([0.0, -0.020, 0.035], dtype=np.float32)
        right_grasp = RIGHT_OBJECT_CENTER + np.array([0.0, -0.020, 0.035], dtype=np.float32)
        left_lift = LEFT_OBJECT_CENTER + np.array([-0.055, -0.065, 0.25], dtype=np.float32)
        right_lift = RIGHT_OBJECT_CENTER + np.array([0.055, -0.065, 0.25], dtype=np.float32)

        left_start = self.manip_start_left_world if self.manip_start_left_world is not None else left_hover
        right_start = self.manip_start_right_world if self.manip_start_right_world is not None else right_hover

        if task_t < 1.6:
            alpha = task_t / 1.6
            return _interp(left_start, left_hover, alpha), _interp(right_start, right_hover, alpha), 0.0
        if task_t < 2.8:
            alpha = (task_t - 1.6) / 1.2
            return _interp(left_hover, left_front, alpha), _interp(right_hover, right_front, alpha), 0.0
        if task_t < 3.8:
            alpha = (task_t - 2.8) / 1.0
            return _interp(left_front, left_grasp, alpha), _interp(right_front, right_grasp, alpha), 0.0
        if task_t < 4.9:
            alpha = (task_t - 3.8) / 1.1
            return left_grasp, right_grasp, _smoothstep01(alpha)
        if task_t < 6.7:
            alpha = (task_t - 4.9) / 1.8
            return _interp(left_grasp, left_lift, alpha), _interp(right_grasp, right_lift, alpha), 1.0
        return left_lift, right_lift, 1.0

    def _apply_bimanual_ik(self, task_t: float, root_q: np.ndarray) -> None:
        left_world, right_world, close_alpha = self._manipulation_targets(task_t)
        left_local = self._world_to_pelvis_position(left_world, root_q)
        right_local = self._world_to_pelvis_position(right_world, root_q)

        self.left_pos_obj.set_target_position(0, wp.vec3(*left_local))
        self.right_pos_obj.set_target_position(0, wp.vec3(*right_local))
        self._seed_ik_from_dynamic_state()
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=int(self.args.ik_iterations))
        wp.launch(
            _scatter_arm_targets,
            dim=len(ARM_JOINT_NAMES),
            inputs=[self.ik_joint_q, self.arm_ik_coords, self.arm_control_targets, self.control.joint_target_q],
            device=self.device,
        )
        wp.launch(
            _set_hand_targets,
            dim=len(HAND_JOINT_NAMES),
            inputs=[self.control.joint_target_q, self.hand_target_indices, self.hand_closed_values, close_alpha],
            device=self.device,
        )

    def _keep_hands_open(self) -> None:
        wp.launch(
            _set_hand_targets,
            dim=len(HAND_JOINT_NAMES),
            inputs=[self.control.joint_target_q, self.hand_target_indices, self.hand_closed_values, 0.0],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # State machine and simulation
    # ------------------------------------------------------------------
    def _update_task_controller(self) -> None:
        root_q = self.state_0.joint_q.numpy()[:7].astype(np.float32)
        root_y = float(root_q[1])

        if self.phase == "approach":
            distance = self.stop_y - root_y
            if distance > 0.018:
                # Taper the command near the desk instead of switching from full
                # speed to zero in one policy tick.  This removes the large
                # locomotion overshoot seen in the first version.
                speed = float(np.clip(1.6 * distance, 0.06, float(self.args.walk_speed)))
                self.command = wp.vec3(speed, 0.0, 0.0)
            else:
                self.phase = "settle"
                self.settle_start_time = self.sim_time
                self.command = wp.vec3(0.0, 0.0, 0.0)
                print(f"[STATE] reached workbench at t={self.sim_time:.2f}s, pelvis_y={root_y:.3f}")
        elif self.phase == "settle":
            # Small balance correction if the policy coasted past the stop pose.
            error = self.stop_y - root_y
            correction = float(np.clip(1.2 * error, -0.10, 0.10)) if abs(error) > 0.025 else 0.0
            self.command = wp.vec3(correction, 0.0, 0.0)
            if self.settle_start_time is not None and self.sim_time - self.settle_start_time >= float(
                self.args.settle_time
            ):
                self.phase = "manipulate"
                self.manip_start_time = self.sim_time
                self.manip_start_left_world = self._body_point_world(self.left_wrist_body, LEFT_TCP_OFFSET_NP)
                self.manip_start_right_world = self._body_point_world(self.right_wrist_body, RIGHT_TCP_OFFSET_NP)
                print(f"[STATE] starting bimanual reach at t={self.sim_time:.2f}s")
        else:
            # Hard safety envelope: if whole-body balance drifts toward the desk,
            # command a short retreat and suspend arm motion until clearance is
            # recovered.
            if root_y > ROBOT_HARD_LIMIT_Y:
                self.command = wp.vec3(-0.10, 0.0, 0.0)
            else:
                self.command = wp.vec3(0.0, 0.0, 0.0)

        self._run_policy()
        safe_to_reach = root_y <= ROBOT_HARD_LIMIT_Y
        if self.phase == "manipulate" and self.manip_start_time is not None and safe_to_reach:
            self._apply_bimanual_ik(self.sim_time - self.manip_start_time, root_q)
        else:
            self._keep_hands_open()

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.model.collide(self.state_0, self.contacts, collision_pipeline=self.collision_pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self._update_task_controller()
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    def test_final(self) -> None:
        body_q = self.state_0.body_q.numpy()
        particle_q = self.state_0.particle_q.numpy()[self.soft_particle_start : self.soft_particle_end]
        assert np.all(np.isfinite(body_q)), "G1 body state contains NaN or inf"
        assert np.all(np.isfinite(particle_q)), "VBD object particles contain NaN or inf"
        assert float(np.min(body_q[:, 2])) > -0.15, "Robot fell through the floor"

        if self.manip_start_time is not None and self.sim_time - self.manip_start_time > 5.0:
            left_particles = self.state_0.particle_q.numpy()[self.soft_particle_start : self.left_particle_end]
            right_particles = self.state_0.particle_q.numpy()[self.left_particle_end : self.soft_particle_end]
            left_center_z = float(np.mean(left_particles[:, 2]))
            right_center_z = float(np.mean(right_particles[:, 2]))
            print(f"[RESULT] left object center z={left_center_z:.3f} m")
            print(f"[RESULT] right object center z={right_center_z:.3f} m")

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=750)
        newton.examples.add_coupled_view_args(parser)
        parser.add_argument("--walk-speed", type=float, default=0.22, help="G1 forward velocity command [m/s].")
        parser.add_argument(
            "--robot-stop-y", type=float, default=ROBOT_STOP_Y, help="Pelvis Y stop position before reaching [m]."
        )
        parser.add_argument("--settle-time", type=float, default=1.6, help="Balance settling time before reaching [s].")
        parser.add_argument("--ik-iterations", type=int, default=28, help="GPU IK iterations per control step.")
        parser.add_argument("--posture-weight", type=float, default=4.0, help="Weight freezing non-arm IK joints.")
        parser.add_argument("--mujoco-iterations", type=int, default=40)
        parser.add_argument("--mujoco-ls-iterations", type=int, default=15)
        parser.add_argument("--vbd-iterations", type=int, default=40)
        parser.add_argument("--proxy-iterations", type=int, default=1)
        parser.add_argument("--proxy-mass-scale", type=float, default=1.0)
        parser.add_argument("--proxy-relaxation", type=float, default=1.0)
        parser.add_argument(
            "--proxy-relaxation-mode",
            choices=["fixed", "aitken"],
            default="fixed",
        )
        parser.add_argument(
            "--coupling-mode",
            choices=["lagged", "staggered"],
            default="lagged",
        )
        parser.add_argument("--soft-contact-ke", type=float, default=8.0e4)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

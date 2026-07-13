# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Dexforce Grasp Cube
#
# Loads DexforceW1V021 in the same table scene as the bimanual IK grasp
# example, drives the right TCP through recorded IK poses, closes the right
# hand around the cube, then moves back through the recorded poses.
#
# Command: python -m newton.examples cloth_dexforce_grasp_cube
#
###########################################################################

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverMuJoCo


@wp.kernel
def lock_joint_q_kernel(
    joint_q: wp.array2d[wp.float32],
    locked_q_indices: wp.array[wp.int32],
    locked_q_values: wp.array[wp.float32],
):
    i = wp.tid()
    joint_q[0, locked_q_indices[i]] = locked_q_values[i]


@wp.kernel
def copy_ik_to_joint_targets_kernel(
    ik_joint_q: wp.array2d[wp.float32],
    joint_target_pos: wp.array[wp.float32],
):
    i = wp.tid()
    joint_target_pos[i] = ik_joint_q[0, i]


@wp.kernel
def set_indexed_joint_targets_kernel(
    target_indices: wp.array[wp.int32],
    open_values: wp.array[wp.float32],
    grasp_values: wp.array[wp.float32],
    alpha: float,
    joint_target_pos: wp.array[wp.float32],
):
    i = wp.tid()
    joint_target_pos[target_indices[i]] = open_values[i] * (1.0 - alpha) + grasp_values[i] * alpha


TABLE_POS = wp.vec3(0.60, 0.0, 1.15)
TABLE_HALF_EXTENTS = (0.32, 0.78, 0.025)
TABLE_TOP_Z = float(TABLE_POS[2]) + TABLE_HALF_EXTENTS[2]
CUBE_HALF = 0.025
CUBE_Z = TABLE_TOP_Z + CUBE_HALF
LEFT_CUBE_POS = wp.vec3(0.52, 0.53, CUBE_Z)
RIGHT_CUBE_POS = wp.vec3(0.52, -0.53, CUBE_Z)
RIGHT_CUBE_ROT = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.35)

TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)

LEFT_HOLD_TF = wp.transform(
    wp.vec3(0.0, 0.9050, 1.36),
    wp.quat(-0.5000, 0.5000, -0.5000, 0.5000),
)
RIGHT_APPROACH_TF = wp.transform(
    wp.vec3(0.3041, -0.7382, 1.36),
    wp.quat(0.0950, 0.7010, 0.7006, 0.0940),
)
RIGHT_CLEAR_TF = wp.transform(
    wp.vec3(0.4096, -0.3579, 1.31),
    wp.quat(-0.0908, 0.6817, 0.7061, -0.1684),
)
RIGHT_GRASP_TF = wp.transform(
    wp.vec3(0.5238, -0.5389, 1.2516),
    wp.quat(0.1921, 0.7073, 0.6790, -0.0415),
)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.max_cube_height = float(RIGHT_CUBE_POS[2])

        builder = newton.ModelBuilder(gravity=-9.81)
        SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg.armature = 0.02
        builder.default_joint_cfg.target_ke = 650.0
        builder.default_joint_cfg.target_kd = 65.0
        builder.default_shape_cfg.ke = 2.0e4
        builder.default_shape_cfg.kd = 8.0e2
        builder.default_shape_cfg.mu = 1.6
        builder.default_shape_cfg.margin = 0.002
        builder.default_shape_cfg.gap = 0.001

        urdf_path = Path(__file__).with_name("DexforceW1V021") / "DexforceW1V021.urdf"
        builder.add_urdf(
            urdf_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=args.enable_self_collisions,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        self._disable_robot_gravity(builder)
        self._configure_robot(builder)
        self._add_table_scene(builder)

        self.model = builder.finalize(requires_grad=False)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.left_ee_index = self._body_index("left_j7")
        self.right_ee_index = self._body_index("right_j7")
        self.left_ee_offset = TCP_OFFSET
        self.right_ee_offset = TCP_OFFSET
        self.left_tf = LEFT_HOLD_TF
        self.right_home_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        self.right_tf = self.right_home_tf

        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.locked_q_indices, self.locked_q_values = self._build_locked_joint_arrays()
        self.right_hand_indices, self.right_hand_open, self.right_hand_grasp = self._build_right_hand_targets()
        self.setup_ik()
        wp.launch(
            copy_ik_to_joint_targets_kernel,
            dim=self.model.joint_dof_count,
            inputs=[self.ik_joint_q],
            outputs=[self.control.joint_target_pos],
            device=self.model.device,
        )

        self.motion_segments = self._build_motion_segments()

        self.solver = SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            njmax=4096,
            nconmax=4096,
            iterations=30,
            ls_iterations=30,
            use_mujoco_contacts=False,
            impratio=25.0,
            cone="elliptic",
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(1.05, -1.35, 1.55), -18.0, 42.0)

    def _configure_robot(self, builder: newton.ModelBuilder) -> None:
        for i in range(builder.joint_dof_count):
            builder.joint_target_pos[i] = builder.joint_q[i]
            builder.joint_target_ke[i] = 650.0
            builder.joint_target_kd[i] = 65.0
            builder.joint_effort_limit[i] = 180.0
            builder.joint_armature[i] = 0.02

        for joint_name in self.RIGHT_HAND_JOINTS:
            joint_idx = self._builder_joint_index(builder, joint_name)
            dof_idx = builder.joint_qd_start[joint_idx]
            builder.joint_target_ke[dof_idx] = 950.0
            builder.joint_target_kd[dof_idx] = 75.0
            builder.joint_effort_limit[dof_idx] = 45.0
            builder.joint_armature[dof_idx] = 0.005

    def _disable_robot_gravity(self, builder: newton.ModelBuilder) -> None:
        gravcomp_body = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp_body.values is None:
            gravcomp_body.values = {}
        for body_idx in range(len(builder.body_label)):
            gravcomp_body.values[body_idx] = 1.0

        gravcomp_joint = builder.custom_attributes["mujoco:jnt_actgravcomp"]
        if gravcomp_joint.values is None:
            gravcomp_joint.values = {}
        for dof_idx in range(builder.joint_dof_count):
            gravcomp_joint.values[dof_idx] = True

    def _add_table_scene(self, builder: newton.ModelBuilder) -> None:
        table_cfg = newton.ModelBuilder.ShapeConfig()
        table_cfg.ke = 2.0e4
        table_cfg.kd = 8.0e2
        table_cfg.mu = 1.2

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(TABLE_POS, wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(LEFT_CUBE_POS, wp.quat_identity()),
            hx=CUBE_HALF,
            hy=CUBE_HALF,
            hz=CUBE_HALF,
            color=(0.95, 0.38, 0.16),
        )

        cube_cfg = newton.ModelBuilder.ShapeConfig()
        cube_cfg.density = 160.0
        cube_cfg.ke = 2.0e4
        cube_cfg.kd = 8.0e2
        cube_cfg.mu = 2.2
        cube_cfg.margin = 0.001

        self.cube_body = builder.add_body(
            xform=wp.transform(RIGHT_CUBE_POS, RIGHT_CUBE_ROT),
            label="grasp_cube",
        )
        builder.add_shape_box(
            body=self.cube_body,
            xform=wp.transform_identity(),
            hx=CUBE_HALF,
            hy=CUBE_HALF,
            hz=CUBE_HALF,
            cfg=cube_cfg,
            color=(0.16, 0.55, 0.78),
        )
        builder.add_ground_plane()

    def setup_ik(self) -> None:
        left_tcp = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        right_tcp = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)

        self.left_pos_obj = ik.IKObjectivePosition(
            link_index=self.left_ee_index,
            link_offset=self.left_ee_offset,
            target_positions=wp.array([wp.transform_get_translation(left_tcp)], dtype=wp.vec3),
        )
        self.left_rot_obj = ik.IKObjectiveRotation(
            link_index=self.left_ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([self._quat_to_vec4(wp.transform_get_rotation(left_tcp))], dtype=wp.vec4),
        )
        self.right_pos_obj = ik.IKObjectivePosition(
            link_index=self.right_ee_index,
            link_offset=self.right_ee_offset,
            target_positions=wp.array([wp.transform_get_translation(right_tcp)], dtype=wp.vec3),
        )
        self.right_rot_obj = ik.IKObjectiveRotation(
            link_index=self.right_ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([self._quat_to_vec4(wp.transform_get_rotation(right_tcp))], dtype=wp.vec4),
        )

        lower, upper = self._joint_limits_with_locked_dofs()
        self.joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=wp.array(lower, dtype=wp.float32, device=self.model.device),
            joint_limit_upper=wp.array(upper, dtype=wp.float32, device=self.model.device),
            weight=25.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.model,
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
        self.ik_iters = 24

    def _build_motion_segments(self) -> tuple[tuple[float, wp.transform, wp.transform, float, float], ...]:
        return (
            (0.8, self.right_home_tf, self.right_home_tf, 0.0, 0.0),
            (2.0, self.right_home_tf, RIGHT_APPROACH_TF, 0.0, 0.0),
            (2.0, RIGHT_APPROACH_TF, RIGHT_CLEAR_TF, 0.0, 0.0),
            (2.0, RIGHT_CLEAR_TF, RIGHT_GRASP_TF, 0.0, 0.0),
            (1.2, RIGHT_GRASP_TF, RIGHT_GRASP_TF, 0.0, 1.0),
            (2.0, RIGHT_GRASP_TF, RIGHT_CLEAR_TF, 1.0, 1.0),
            (2.0, RIGHT_CLEAR_TF, RIGHT_APPROACH_TF, 1.0, 1.0),
            (2.0, RIGHT_APPROACH_TF, self.right_home_tf, 1.0, 1.0),
            (1.2, self.right_home_tf, self.right_home_tf, 1.0, 1.0),
        )

    def _build_right_hand_targets(self) -> tuple[wp.array, wp.array, wp.array]:
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        target_indices = []
        open_values = []
        grasp_values = []
        grasp_targets = {
            "RIGHT_HAND_THUMB2": 1.05,
            "RIGHT_HAND_THUMB1": 0.58,
            "RIGHT_HAND_INDEX": 0.88,
            "RIGHT_INDEX_PIP": 1.12,
            "RIGHT_HAND_MIDDLE": 0.95,
            "RIGHT_MIDDLE_PIP": 1.18,
            "RIGHT_HAND_RING": 0.92,
            "RIGHT_RING_PIP": 1.12,
            "RIGHT_HAND_PINKY": 0.84,
            "RIGHT_PINKY_PIP": 1.02,
        }

        for joint_name, target in grasp_targets.items():
            joint_idx = self._joint_index(joint_name)
            target_indices.append(int(qd_start[joint_idx]))
            open_values.append(float(q_home[int(q_start[joint_idx])]))
            grasp_values.append(target)

        return (
            wp.array(target_indices, dtype=wp.int32, device=self.model.device),
            wp.array(open_values, dtype=wp.float32, device=self.model.device),
            wp.array(grasp_values, dtype=wp.float32, device=self.model.device),
        )

    def _joint_limits_with_locked_dofs(self) -> tuple[np.ndarray, np.ndarray]:
        lower = self.model.joint_limit_lower.numpy().copy()
        upper = self.model.joint_limit_upper.numpy().copy()
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        controlled = self._controlled_joint_labels()

        for joint_idx, label in enumerate(self.model.joint_label):
            if label in controlled:
                continue
            q_idx = int(q_start[joint_idx])
            dof_idx = int(qd_start[joint_idx])
            lower[dof_idx] = q_home[q_idx] - 1.0e-4
            upper[dof_idx] = q_home[q_idx] + 1.0e-4

        return lower, upper

    def _build_locked_joint_arrays(self) -> tuple[wp.array, wp.array]:
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        controlled = self._controlled_joint_labels()
        locked_q_indices = []
        locked_q_values = []

        for joint_idx, label in enumerate(self.model.joint_label):
            if label in controlled:
                continue
            q_idx = int(q_start[joint_idx])
            locked_q_indices.append(q_idx)
            locked_q_values.append(float(q_home[q_idx]))

        return (
            wp.array(locked_q_indices, dtype=wp.int32, device=self.model.device),
            wp.array(locked_q_values, dtype=wp.float32, device=self.model.device),
        )

    def _sample_script(self) -> tuple[wp.transform, float]:
        remaining = self.sim_time
        for duration, start_tf, end_tf, start_grasp, end_grasp in self.motion_segments:
            if remaining <= duration:
                alpha = float(np.clip(remaining / duration, 0.0, 1.0))
                return self._interpolate_transform(start_tf, end_tf, alpha), start_grasp * (1.0 - alpha) + end_grasp * alpha
            remaining -= duration

        _, _, end_tf, _, end_grasp = self.motion_segments[-1]
        return end_tf, end_grasp

    def _update_control_targets(self) -> None:
        self.right_tf, grasp_alpha = self._sample_script()

        self.left_pos_obj.set_target_position(0, wp.transform_get_translation(self.left_tf))
        self.left_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.left_tf)))
        self.right_pos_obj.set_target_position(0, wp.transform_get_translation(self.right_tf))
        self.right_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.right_tf)))
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            lock_joint_q_kernel,
            dim=self.locked_q_indices.shape[0],
            inputs=[self.ik_joint_q, self.locked_q_indices, self.locked_q_values],
            device=self.model.device,
        )
        wp.launch(
            copy_ik_to_joint_targets_kernel,
            dim=self.model.joint_dof_count,
            inputs=[self.ik_joint_q],
            outputs=[self.control.joint_target_pos],
            device=self.model.device,
        )
        wp.launch(
            set_indexed_joint_targets_kernel,
            dim=self.right_hand_indices.shape[0],
            inputs=[self.right_hand_indices, self.right_hand_open, self.right_hand_grasp, grasp_alpha],
            outputs=[self.control.joint_target_pos],
            device=self.model.device,
        )

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self._update_control_targets()
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        if hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo(
                "dexforce_left_tcp_target",
                self.left_tf,
                snap_to=self._current_tcp_transform(self.left_ee_index, self.left_ee_offset),
            )
            self.viewer.log_gizmo(
                "dexforce_right_tcp_target",
                self.right_tf,
                snap_to=self._current_tcp_transform(self.right_ee_index, self.right_ee_offset),
            )
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        cube_z = float(self.state_0.body_q.numpy()[self.cube_body, 2])
        self.max_cube_height = max(self.max_cube_height, cube_z)

    def test_final(self) -> None:
        cube_pos = self.state_0.body_q.numpy()[self.cube_body, :3]
        if not np.all(np.isfinite(cube_pos)):
            raise ValueError(f"Cube position is not finite: {cube_pos}")
        if not (-0.1 < cube_pos[0] < 0.8 and -1.0 < cube_pos[1] < -0.2 and 1.0 < cube_pos[2] < 1.6):
            raise ValueError(f"Cube ended outside the expected grasp volume: {cube_pos}")

    def _current_tcp_transform(self, body_index: int, offset: wp.vec3) -> wp.transform:
        body_q_np = self.state_0.body_q.numpy()
        body_tf = wp.transform(*body_q_np[body_index])
        body_pos = wp.transform_get_translation(body_tf)
        body_rot = wp.transform_get_rotation(body_tf)
        tcp_pos = body_pos + wp.quat_rotate(body_rot, offset)
        return wp.transform(tcp_pos, body_rot)

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
        scale_a = np.sin(theta_0 - theta) / sin_theta_0
        scale_b = np.sin(theta) / sin_theta_0
        return qa * scale_a + qb * scale_b

    def _normalize_quat(self, quat: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(quat))
        if norm == 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return quat / norm

    def _body_index(self, body_name: str) -> int:
        suffix = f"/{body_name}"
        return next(i for i, label in enumerate(self.model.body_label) if label.endswith(suffix))

    def _builder_joint_index(self, builder: newton.ModelBuilder, joint_name: str) -> int:
        suffix = f"/{joint_name}"
        return next(i for i, label in enumerate(builder.joint_label) if label.endswith(suffix))

    def _joint_index(self, joint_name: str) -> int:
        suffix = f"/{joint_name}"
        return next(i for i, label in enumerate(self.model.joint_label) if label.endswith(suffix))

    def _controlled_joint_labels(self) -> set[str]:
        return {f"DexforceW1V021/{name}" for name in (*self.LEFT_ARM_JOINTS, *self.RIGHT_ARM_JOINTS)}

    def _quat_to_vec4(self, quat: wp.quat) -> wp.vec4:
        return wp.vec4(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    def _quat_to_np(self, quat: wp.quat) -> np.ndarray:
        return np.array([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])], dtype=np.float64)

    def _vec3_to_np(self, vec: wp.vec3) -> np.ndarray:
        return np.array([float(vec[0]), float(vec[1]), float(vec[2])], dtype=np.float64)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=900)
        parser.add_argument(
            "--enable-self-collisions",
            action="store_true",
            help="Enable imported URDF self-collisions. This is slower but useful for debugging.",
        )
        return parser

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


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

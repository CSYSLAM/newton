# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Dexforce Bimanual IK
#
# Loads DexforceW1V021, places a table in front of the robot, and gives each
# wrist TCP its own IK gizmo. The console reports target and solved TCP
# positions for both hands so bimanual reach targets are easy to tune.
#
# Command: python -m newton.examples cloth_dexforce_bimanual_ik_grasp
#
###########################################################################

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik


@wp.kernel
def lock_joint_q_kernel(
    joint_q: wp.array2d[wp.float32],
    locked_q_indices: wp.array[wp.int32],
    locked_q_values: wp.array[wp.float32],
):
    i = wp.tid()
    joint_q[0, locked_q_indices[i]] = locked_q_values[i]


TABLE_POS = wp.vec3(0.60, 0.0, 1.15)
TABLE_HALF_EXTENTS = (0.32, 0.78, 0.025)
TABLE_TOP_Z = float(TABLE_POS[2]) + TABLE_HALF_EXTENTS[2]
CUBE_HALF = 0.035
CUBE_Z = TABLE_TOP_Z + CUBE_HALF
LEFT_CUBE_POS = wp.vec3(0.52, 0.53, CUBE_Z)
RIGHT_CUBE_POS = wp.vec3(0.52, -0.53, CUBE_Z)

TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.print_interval = float(args.print_interval)
        self.last_print_time = -1.0

        builder = newton.ModelBuilder()
        builder.default_joint_cfg.armature = 0.02

        urdf_path = Path("E:/csy_work/CG/assets/DexforceW1V021") / "DexforceW1V021.urdf"
        builder.add_urdf(
            urdf_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=args.enable_self_collisions,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        self._add_table_scene(builder)

        self.model = builder.finalize(requires_grad=False)
        self.state = self.model.state()
        self.joint_q = self.model.joint_q.reshape((1, self.model.joint_coord_count))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        self.left_ee_index = self._body_index("left_j7")
        self.right_ee_index = self._body_index("right_j7")
        self.left_ee_offset = TCP_OFFSET
        self.right_ee_offset = TCP_OFFSET

        self.locked_q_indices, self.locked_q_values = self._build_locked_joint_arrays()
        self.setup_ik()

        self.left_tf = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        self.right_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(1.05, -1.35, 1.55), -18.0, 42.0)
        self._report_pose(force=True)
        self.capture()

    def _add_table_scene(self, builder: newton.ModelBuilder) -> None:
        table_cfg = newton.ModelBuilder.ShapeConfig()
        table_cfg.mu = 1.2

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(TABLE_POS, wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 1.70, 0.48),
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(LEFT_CUBE_POS, wp.quat_identity()),
            hx=CUBE_HALF,
            hy=CUBE_HALF,
            hz=CUBE_HALF,
            color=(0.95, 0.38, 0.16),
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                RIGHT_CUBE_POS,
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.35),
            ),
            hx=CUBE_HALF,
            hy=CUBE_HALF,
            hz=CUBE_HALF,
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
        self.ik_iters = 22

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

    def capture(self) -> None:
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self) -> None:
        self.ik_solver.step(self.joint_q, self.joint_q, iterations=self.ik_iters)
        wp.launch(
            lock_joint_q_kernel,
            dim=self.locked_q_indices.shape[0],
            inputs=[self.joint_q, self.locked_q_indices, self.locked_q_values],
            device=self.model.device,
        )
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

    def _push_targets_from_gizmos(self) -> None:
        left_pos = wp.transform_get_translation(self.left_tf)
        left_rot = wp.transform_get_rotation(self.left_tf)
        right_pos = wp.transform_get_translation(self.right_tf)
        right_rot = wp.transform_get_rotation(self.right_tf)

        self.left_pos_obj.set_target_position(0, left_pos)
        self.left_rot_obj.set_target_rotation(0, self._quat_to_vec4(left_rot))
        self.right_pos_obj.set_target_position(0, right_pos)
        self.right_rot_obj.set_target_rotation(0, self._quat_to_vec4(right_rot))

    def step(self) -> None:
        self._push_targets_from_gizmos()
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._report_pose()

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
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def _current_tcp_transform(self, body_index: int, offset: wp.vec3) -> wp.transform:
        body_q_np = self.state.body_q.numpy()
        body_tf = wp.transform(*body_q_np[body_index])
        body_pos = wp.transform_get_translation(body_tf)
        body_rot = wp.transform_get_rotation(body_tf)
        tcp_pos = body_pos + wp.quat_rotate(body_rot, offset)
        return wp.transform(tcp_pos, body_rot)

    def _report_pose(self, force: bool = False) -> None:
        if not force and self.print_interval > 0.0 and self.sim_time - self.last_print_time < self.print_interval:
            return

        left_actual_tf = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        right_actual_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)

        left_target_pos = self._vec3_to_np(wp.transform_get_translation(self.left_tf))
        left_actual_pos = self._vec3_to_np(wp.transform_get_translation(left_actual_tf))
        right_target_pos = self._vec3_to_np(wp.transform_get_translation(self.right_tf))
        right_actual_pos = self._vec3_to_np(wp.transform_get_translation(right_actual_tf))

        left_target_rot = self._quat_to_np(wp.transform_get_rotation(self.left_tf))
        left_actual_rot = self._quat_to_np(wp.transform_get_rotation(left_actual_tf))
        right_target_rot = self._quat_to_np(wp.transform_get_rotation(self.right_tf))
        right_actual_rot = self._quat_to_np(wp.transform_get_rotation(right_actual_tf))

        left_pos_err = float(np.linalg.norm(left_target_pos - left_actual_pos))
        right_pos_err = float(np.linalg.norm(right_target_pos - right_actual_pos))
        left_rot_err = self._quat_angle_error_deg(left_target_rot, left_actual_rot)
        right_rot_err = self._quat_angle_error_deg(right_target_rot, right_actual_rot)

        print(
            f"[{self.sim_time:7.3f}s] "
            f"L target={self._format_pose(left_target_pos, left_target_rot)} "
            f"actual={self._format_pose(left_actual_pos, left_actual_rot)} "
            f"pos_err={left_pos_err:.5f} m rot_err={left_rot_err:.3f} deg | "
            f"R target={self._format_pose(right_target_pos, right_target_rot)} "
            f"actual={self._format_pose(right_actual_pos, right_actual_rot)} "
            f"pos_err={right_pos_err:.5f} m rot_err={right_rot_err:.3f} deg"
        )
        self.last_print_time = self.sim_time

    def test_final(self) -> None:
        for label, body_index, offset in (
            ("left", self.left_ee_index, self.left_ee_offset),
            ("right", self.right_ee_index, self.right_ee_offset),
        ):
            pos = self._vec3_to_np(wp.transform_get_translation(self._current_tcp_transform(body_index, offset)))
            if not np.all(np.isfinite(pos)):
                raise ValueError(f"Dexforce {label} TCP position is not finite: {pos}")

    def _body_index(self, body_name: str) -> int:
        suffix = f"/{body_name}"
        return next(i for i, label in enumerate(self.model.body_label) if label.endswith(suffix))

    def _controlled_joint_labels(self) -> set[str]:
        return {f"DexforceW1V021/{name}" for name in (*self.LEFT_ARM_JOINTS, *self.RIGHT_ARM_JOINTS)}

    def _quat_to_vec4(self, quat: wp.quat) -> wp.vec4:
        return wp.vec4(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

    def _quat_to_np(self, quat: wp.quat) -> np.ndarray:
        return np.array([float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])], dtype=np.float64)

    def _quat_angle_error_deg(self, quat_a: np.ndarray, quat_b: np.ndarray) -> float:
        dot = abs(float(np.dot(quat_a, quat_b)))
        return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))

    def _vec3_to_np(self, vec: wp.vec3) -> np.ndarray:
        return np.array([float(vec[0]), float(vec[1]), float(vec[2])], dtype=np.float64)

    def _format_xyz(self, xyz: np.ndarray) -> str:
        return f"[{xyz[0]: .4f}, {xyz[1]: .4f}, {xyz[2]: .4f}]"

    def _format_xyzw(self, xyzw: np.ndarray) -> str:
        return f"[{xyzw[0]: .4f}, {xyzw[1]: .4f}, {xyzw[2]: .4f}, {xyzw[3]: .4f}]"

    def _format_pose(self, xyz: np.ndarray, xyzw: np.ndarray) -> str:
        return f"pos={self._format_xyz(xyz)} quat_xyzw={self._format_xyzw(xyzw)}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=300)
        parser.add_argument(
            "--print-interval",
            type=float,
            default=3.0,
            help="Seconds between TCP position reports. Use 0.0 to print every frame.",
        )
        parser.add_argument(
            "--enable-self-collisions",
            action="store_true",
            help="Enable imported URDF self-collisions while building the Dexforce model.",
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


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

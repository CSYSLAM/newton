# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Dexforce Bimanual IK Cloth
#
# Loads DexforceW1V021 in its URDF pose, places a large cloth patch on the
# table, and exposes an IK gizmo for each wrist TCP so grasp trajectories can
# be debugged interactively.
#
# Command: python -m newton.examples cloth_dexforce_bimanual_ik_cloth
#
###########################################################################

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.mjvbd


@wp.kernel
def lock_joint_q_kernel(
    joint_q: wp.array2d[wp.float32],
    locked_q_indices: wp.array[wp.int32],
    locked_q_values: wp.array[wp.float32],
):
    i = wp.tid()
    joint_q[0, locked_q_indices[i]] = locked_q_values[i]


@wp.kernel
def copy_ik_to_joint_q_kernel(
    ik_joint_q: wp.array2d[wp.float32],
    joint_q_out: wp.array[wp.float32],
):
    i = wp.tid()
    joint_q_out[i] = ik_joint_q[0, i]


@wp.kernel
def interpolate_joint_positions_kernel(
    joint_q_start: wp.array[wp.float32],
    joint_q_end: wp.array[wp.float32],
    alpha: float,
    joint_q_out: wp.array[wp.float32],
):
    i = wp.tid()
    joint_q_out[i] = joint_q_start[i] * (1.0 - alpha) + joint_q_end[i] * alpha


@wp.kernel
def update_joint_velocity_kernel(
    joint_q_prev: wp.array[wp.float32],
    joint_q_next: wp.array[wp.float32],
    inv_dt: float,
    joint_qd: wp.array[wp.float32],
):
    i = wp.tid()
    joint_qd[i] = (joint_q_next[i] - joint_q_prev[i]) * inv_dt


TABLE_POS = wp.vec3(0.60, 0.0, 1.15)
TABLE_HALF_EXTENTS = (0.32, 0.78, 0.025)
TABLE_TOP_Z = float(TABLE_POS[2]) + TABLE_HALF_EXTENTS[2]
TABLE_COLOR = (0.35, 0.42, 0.48)

CLOTH_DIM_X = 24
CLOTH_DIM_Y = 36
CLOTH_CELL_X = 0.022
CLOTH_CELL_Y = 0.025
CLOTH_CENTER = wp.vec3(float(TABLE_POS[0]), 0.0, TABLE_TOP_Z + 0.004)
CLOTH_POS = wp.vec3(
    float(CLOTH_CENTER[0]) - 0.5 * CLOTH_DIM_X * CLOTH_CELL_X,
    float(CLOTH_CENTER[1]) - 0.5 * CLOTH_DIM_Y * CLOTH_CELL_Y,
    float(CLOTH_CENTER[2]),
)
CLOTH_COLOR = (0.78, 0.12, 0.10)

TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.print_interval = float(args.print_interval)
        self.last_print_time = -1.0

        self.particle_radius = 0.006
        self.soft_contact_margin = 0.012
        self.particle_self_contact_radius = 0.006
        self.particle_self_contact_margin = 0.006
        self.self_contact_bvh_rebuild_interval_frames = 30

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.default_joint_cfg.armature = 0.02
        builder.default_shape_cfg.ke = 5.0e5
        builder.default_shape_cfg.kd = 1.0e-6
        builder.default_shape_cfg.mu = 2.0

        urdf_path = Path(__file__).with_name("DexforceW1V021") / "DexforceW1V021.urdf"
        builder.add_urdf(
            urdf_path,
            xform=self._robot_xform(),
            floating=False,
            enable_self_collisions=args.enable_self_collisions,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        self.robot_shape_end = builder.shape_count
        self._configure_robot(builder)
        self._add_table_scene(builder)
        self.cloth_start = builder.particle_count
        self._add_cloth(builder)
        builder.color(include_bending=True)

        self.model = builder.finalize(requires_grad=False)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self._configure_particle_contacts()

        self.left_ee_index = self._body_index("left_j7")
        self.right_ee_index = self._body_index("right_j7")
        self.left_ee_offset = TCP_OFFSET
        self.right_ee_offset = TCP_OFFSET

        self.left_tf = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        self.right_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)

        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.frame_joint_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_joint_q_end = wp.zeros_like(self.model.joint_q)
        self.locked_q_indices, self.locked_q_values = self._build_locked_joint_arrays()
        self.setup_ik()

        self.model.soft_contact_ke = 5.0e5
        self.model.soft_contact_kd = 1.0e-6
        self.model.soft_contact_mu = 2.0

        self.solver = newton.mjvbd.SolverMJVBD(
            self.model,
            rigid_contact_max=0,
            soft_contact_margin=self.soft_contact_margin,
            iterations=10,
            particle_self_contact_radius=self.particle_self_contact_radius,
            particle_self_contact_margin=self.particle_self_contact_margin,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.03,
            particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=20,
            particle_collision_detection_interval=-1,
        )

        self.initial_cloth_height = float(np.max(self.state_0.particle_q.numpy()[:, 2]))
        self.max_cloth_height = self.initial_cloth_height

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(1.05, -1.35, 1.55), -18.0, 42.0)
        self._report_pose(force=True)

    def _robot_xform(self) -> wp.transform:
        return wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity())

    def _configure_robot(self, builder: newton.ModelBuilder) -> None:
        for i in range(builder.joint_dof_count):
            builder.joint_target_pos[i] = builder.joint_q[i]
            builder.joint_target_ke[i] = 650.0
            builder.joint_target_kd[i] = 65.0
            builder.joint_effort_limit[i] = 180.0
            builder.joint_armature[i] = 0.02

    def _add_table_scene(self, builder: newton.ModelBuilder) -> None:
        table_cfg = newton.ModelBuilder.ShapeConfig()
        table_cfg.ke = 5.0e5
        table_cfg.kd = 1.0e-6
        table_cfg.mu = 1.2

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(TABLE_POS, wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=TABLE_COLOR,
        )
        builder.add_ground_plane()

    def _add_cloth(self, builder: newton.ModelBuilder) -> None:
        builder.add_cloth_grid(
            pos=CLOTH_POS,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=CLOTH_DIM_X,
            dim_y=CLOTH_DIM_Y,
            cell_x=CLOTH_CELL_X,
            cell_y=CLOTH_CELL_Y,
            mass=0.002,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e-5,
            edge_ke=1.0,
            edge_kd=0.05,
            particle_radius=self.particle_radius,
            label="debug_cloth",
        )

    def _configure_particle_contacts(self) -> None:
        flags = self.model.shape_flags.numpy()
        flags |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
        flags[: self.robot_shape_end] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

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

    def _refresh_self_contact_bvh(self) -> None:
        if self.frame_index > 0 and self.frame_index % self.self_contact_bvh_rebuild_interval_frames == 0:
            self.solver.rebuild_bvh(self.state_0)

    def _push_targets_from_gizmos(self) -> None:
        self.left_pos_obj.set_target_position(0, wp.transform_get_translation(self.left_tf))
        self.left_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.left_tf)))
        self.right_pos_obj.set_target_position(0, wp.transform_get_translation(self.right_tf))
        self.right_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.right_tf)))

    def _prepare_frame_targets(self) -> None:
        wp.copy(self.frame_joint_q_start, self.state_0.joint_q)
        self._push_targets_from_gizmos()
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            lock_joint_q_kernel,
            dim=self.locked_q_indices.shape[0],
            inputs=[self.ik_joint_q, self.locked_q_indices, self.locked_q_values],
            device=self.model.device,
        )
        wp.launch(
            copy_ik_to_joint_q_kernel,
            dim=self.model.joint_coord_count,
            inputs=[self.ik_joint_q],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )

    def simulate(self) -> None:
        self._refresh_self_contact_bvh()
        self._prepare_frame_targets()

        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            wp.copy(self.state_1.particle_q, self.state_0.particle_q)
            wp.copy(self.state_1.particle_qd, self.state_0.particle_qd)
            wp.copy(self.state_1.body_q, self.state_0.body_q)
            wp.copy(self.state_1.body_qd, self.state_0.body_qd)

            substep_alpha = float((substep + 1) / self.sim_substeps)
            wp.launch(
                interpolate_joint_positions_kernel,
                dim=self.model.joint_coord_count,
                inputs=[self.frame_joint_q_start, self.frame_joint_q_end, substep_alpha],
                outputs=[self.state_1.joint_q],
                device=self.model.device,
            )
            wp.launch(
                update_joint_velocity_kernel,
                dim=self.model.joint_dof_count,
                inputs=[self.state_0.joint_q, self.state_1.joint_q, 1.0 / self.sim_dt],
                outputs=[self.state_1.joint_qd],
                device=self.model.device,
            )
            newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
            self.solver.step(self.state_0, self.state_1, self.control, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt
        self.frame_index += 1
        self._track_cloth_height()

    def step(self) -> None:
        self.simulate()
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
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/debug_cloth",
            self.state_0.particle_q,
            self.model.tri_indices.flatten(),
            hidden=not self.viewer.show_triangles,
            backface_culling=False,
            color=CLOTH_COLOR,
        )
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        self._track_cloth_height()

    def test_final(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        if not np.all(np.isfinite(particle_q)):
            raise ValueError("Cloth particle positions are not finite")

    def _track_cloth_height(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        self.max_cloth_height = max(self.max_cloth_height, float(np.max(particle_q[:, 2])))

    def _current_tcp_transform(self, body_index: int, offset: wp.vec3) -> wp.transform:
        body_q_np = self.state_0.body_q.numpy()
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
        parser.set_defaults(num_frames=900)
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

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Dexforce Burger Stack
#
# Uses Dexforce W1 to kinematically move a rigid top bun over a soft
# BurgerMeat body. The buns are particle colliders; the meat is simulated by
# MJVBD. This intentionally avoids rigid-soft two-way coupling.
#
# Command: python -m newton.examples cloth_dexforce_burger_stack
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
def set_indexed_joint_q_kernel(
    q_indices: wp.array[wp.int32],
    open_values: wp.array[wp.float32],
    grasp_values: wp.array[wp.float32],
    alpha: float,
    joint_q_out: wp.array[wp.float32],
):
    i = wp.tid()
    joint_q_out[q_indices[i]] = open_values[i] * (1.0 - alpha) + grasp_values[i] * alpha


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


@wp.kernel
def set_shape_transform_kernel(
    shape_transform: wp.array[wp.transform],
    shape_index: int,
    xform: wp.transform,
):
    shape_transform[shape_index] = xform


TABLE_POS = wp.vec3(0.60, 0.0, 1.15)
TABLE_HALF_EXTENTS = (0.32, 0.78, 0.025)
TABLE_TOP_Z = float(TABLE_POS[2]) + TABLE_HALF_EXTENTS[2]
TABLE_COLOR = (0.35, 0.42, 0.48)

BURGER_ASSET = Path("E:/csy_work/CG/assets/BurgerMeat.glb")
BURGER_CENTER_XY = wp.vec3(float(TABLE_POS[0]), 0.0, 0.0)
BURGER_ROT = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2.0)
BURGER_SCALE = 1.0
BURGER_COLLISION_RADIUS = 0.004
BURGER_COLOR = (0.38, 0.11, 0.06)

BUN_RADIUS = 0.105
BUN_HALF_HEIGHT = 0.010
BUN_DENSITY = 60.0
BUN_COLOR = (0.90, 0.62, 0.30)
BOTTOM_BUN_CENTER = wp.vec3(float(TABLE_POS[0]), 0.0, TABLE_TOP_Z + BUN_HALF_HEIGHT)
TOP_BUN_START_CENTER = wp.vec3(0.52, -0.53, TABLE_TOP_Z + BUN_HALF_HEIGHT)
MEAT_CENTER_Z = TABLE_TOP_Z + 2.0 * BUN_HALF_HEIGHT + 0.027
TOP_BUN_FINAL_CENTER = wp.vec3(float(TABLE_POS[0]), 0.0, TABLE_TOP_Z + 2.0 * BUN_HALF_HEIGHT + 0.038)
TOP_BUN_CLEARANCE = 0.18

SOFT_CONTACT_MARGIN = 0.022
SELF_CONTACT_RADIUS = 0.006
SELF_CONTACT_MARGIN = 0.006
SOLVER_ITERATIONS = 14

TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)

LEFT_HOLD_TF = wp.transform(
    wp.vec3(0.0, 0.9050, 1.36),
    wp.quat(-0.5000, 0.5000, -0.5000, 0.5000),
)
RIGHT_APPROACH_TF = wp.transform(
    wp.vec3(0.3041, -0.7382, 1.36),
    wp.quat(0.0950, 0.7010, 0.7006, 0.0940),
)
RIGHT_GRASP_TF = wp.transform(
    wp.vec3(0.5238, -0.5389, 1.2516),
    wp.quat(0.1921, 0.7073, 0.6790, -0.0415),
)


def make_star_tet_mesh(mesh: newton.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Converts the BurgerMeat surface mesh to a simple center-fan tet mesh."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)

    bbox_center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    vertices = vertices - bbox_center

    center_idx = vertices.shape[0]
    vertices = np.vstack([vertices, np.zeros((1, 3), dtype=np.float32)])

    tet_indices: list[int] = []
    center = vertices[center_idx]
    for a, b, c in faces:
        pa = vertices[a]
        pb = vertices[b]
        pc = vertices[c]
        signed_volume = np.linalg.det(np.column_stack((pb - pa, pc - pa, center - pa))) / 6.0

        if abs(signed_volume) < 1.0e-12:
            continue
        if signed_volume > 0.0:
            tet_indices.extend((int(a), int(b), int(c), center_idx))
        else:
            tet_indices.extend((int(a), int(c), int(b), center_idx))

    return vertices, np.asarray(tet_indices, dtype=np.int32)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.print_interval = float(args.print_interval)
        self.last_print_time = -1.0
        self.self_contact_bvh_rebuild_interval_frames = 30

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.default_joint_cfg.armature = 0.02
        builder.default_shape_cfg.ke = 5.0e5
        builder.default_shape_cfg.kd = 1.0e-6
        builder.default_shape_cfg.mu = 2.0

        urdf_path = Path("E:/csy_work/CG/assets/DexforceW1V021") / "DexforceW1V021.urdf"
        builder.add_urdf(
            urdf_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=args.enable_self_collisions,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        self.robot_shape_end = builder.shape_count
        self._configure_robot(builder)
        self._add_table_and_buns(builder)
        self.burger_particle_start = builder.particle_count
        self._add_burger_meat(builder)
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

        self.left_tf = LEFT_HOLD_TF
        self.right_home_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        self.right_tf = self.right_home_tf

        self.top_bun_start_tf = wp.transform(TOP_BUN_START_CENTER, wp.quat_identity())
        self.top_bun_end_tf = self.top_bun_start_tf
        self.top_bun_tf = self.top_bun_start_tf

        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.frame_joint_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_joint_q_end = wp.zeros_like(self.model.joint_q)
        self.locked_q_indices, self.locked_q_values = self._build_locked_joint_arrays()
        self.right_hand_q_indices, self.right_hand_open, self.right_hand_grasp = self._build_right_hand_targets()
        self.setup_ik()
        self.motion_segments = self._build_motion_segments()

        self.model.soft_contact_ke = 2.0e6
        self.model.soft_contact_kd = 1.0e-6
        self.model.soft_contact_mu = 2.0
        self.model.shape_material_ke.fill_(2.0e6)
        self.model.shape_material_kd.fill_(1.0e-6)
        self.model.shape_material_mu.fill_(2.0)

        self.solver = newton.mjvbd.SolverMJVBD(
            self.model,
            rigid_contact_max=0,
            soft_contact_margin=SOFT_CONTACT_MARGIN,
            iterations=SOLVER_ITERATIONS,
            particle_self_contact_radius=SELF_CONTACT_RADIUS,
            particle_self_contact_margin=SELF_CONTACT_MARGIN,
            particle_topological_contact_filter_threshold=3,
            particle_rest_shape_contact_exclusion_radius=0.025,
            particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=32,
            particle_edge_contact_buffer_size=48,
            particle_collision_detection_interval=2,
        )

        print(
            f"[newton] Burger stack: particles={self.model.particle_count}, "
            f"tets={self.model.tet_count}, tris={self.model.tri_count}"
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(1.05, -1.35, 1.55), -18.0, 42.0)
        self._report_pose(force=True)

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

    def _add_table_and_buns(self, builder: newton.ModelBuilder) -> None:
        table_cfg = newton.ModelBuilder.ShapeConfig()
        table_cfg.ke = 5.0e5
        table_cfg.kd = 1.0e-6
        table_cfg.mu = 1.2

        bun_cfg = newton.ModelBuilder.ShapeConfig()
        bun_cfg.ke = 1.0e6
        bun_cfg.kd = 1.0e-6
        bun_cfg.mu = 2.0
        bun_cfg.density = BUN_DENSITY

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(TABLE_POS, wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=TABLE_COLOR,
            label="burger_table",
        )
        builder.add_shape_cylinder(
            body=-1,
            xform=wp.transform(BOTTOM_BUN_CENTER, wp.quat_identity()),
            radius=BUN_RADIUS,
            half_height=BUN_HALF_HEIGHT,
            cfg=bun_cfg,
            color=BUN_COLOR,
            label="bottom_bun",
        )
        self.top_bun_shape = builder.add_shape_cylinder(
            body=-1,
            xform=wp.transform(TOP_BUN_START_CENTER, wp.quat_identity()),
            radius=BUN_RADIUS,
            half_height=BUN_HALF_HEIGHT,
            cfg=bun_cfg,
            color=BUN_COLOR,
            label="top_bun_kinematic",
        )
        builder.add_ground_plane()

    def _add_burger_meat(self, builder: newton.ModelBuilder) -> None:
        burger_mesh = newton.Mesh.create_from_file(str(BURGER_ASSET), compute_inertia=False)
        vertices, tet_indices = make_star_tet_mesh(burger_mesh)

        burger_pos = wp.vec3(
            float(BURGER_CENTER_XY[0]),
            float(BURGER_CENTER_XY[1]),
            MEAT_CENTER_Z,
        )
        builder.add_soft_mesh(
            pos=burger_pos,
            rot=BURGER_ROT,
            scale=BURGER_SCALE,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices.tolist(),
            indices=tet_indices.tolist(),
            density=180.0,
            k_mu=2.0e4,
            k_lambda=2.0e4,
            k_damp=1.2e-3,
            particle_radius=BURGER_COLLISION_RADIUS,
            label="burger_meat",
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

    def _build_motion_segments(
        self,
    ) -> tuple[tuple[float, wp.transform, wp.transform, wp.transform, wp.transform, float, float], ...]:
        grasp_pos = self._vec3_to_np(wp.transform_get_translation(RIGHT_GRASP_TF))
        bun_grasp_offset = self._vec3_to_np(TOP_BUN_START_CENTER) - grasp_pos
        final_center = self._vec3_to_np(TOP_BUN_FINAL_CENTER)
        place_pos = final_center - bun_grasp_offset

        right_lift_tf = self._offset_transform(RIGHT_GRASP_TF, wp.vec3(0.0, 0.0, TOP_BUN_CLEARANCE))
        right_place_high_tf = wp.transform(
            wp.vec3(float(place_pos[0]), float(place_pos[1]), float(place_pos[2] + TOP_BUN_CLEARANCE)),
            wp.transform_get_rotation(RIGHT_GRASP_TF),
        )
        right_place_tf = wp.transform(
            wp.vec3(float(place_pos[0]), float(place_pos[1]), float(place_pos[2])),
            wp.transform_get_rotation(RIGHT_GRASP_TF),
        )
        right_retreat_tf = self._offset_transform(right_place_tf, wp.vec3(-0.10, -0.24, 0.14))

        top_start = self.top_bun_start_tf
        top_lift = wp.transform(
            self._offset_vec3(TOP_BUN_START_CENTER, wp.vec3(0.0, 0.0, TOP_BUN_CLEARANCE)),
            wp.quat_identity(),
        )
        top_place_high = wp.transform(
            self._offset_vec3(TOP_BUN_FINAL_CENTER, wp.vec3(0.0, 0.0, TOP_BUN_CLEARANCE)),
            wp.quat_identity(),
        )
        top_final = wp.transform(TOP_BUN_FINAL_CENTER, wp.quat_identity())

        return (
            (0.25, self.right_home_tf, self.right_home_tf, top_start, top_start, 0.0, 0.0),
            (0.70, self.right_home_tf, RIGHT_APPROACH_TF, top_start, top_start, 0.0, 0.0),
            (0.55, RIGHT_APPROACH_TF, RIGHT_GRASP_TF, top_start, top_start, 0.0, 0.0),
            (0.35, RIGHT_GRASP_TF, RIGHT_GRASP_TF, top_start, top_start, 0.0, 1.0),
            (0.70, RIGHT_GRASP_TF, right_lift_tf, top_start, top_lift, 1.0, 1.0),
            (0.80, right_lift_tf, right_place_high_tf, top_lift, top_place_high, 1.0, 1.0),
            (0.65, right_place_high_tf, right_place_tf, top_place_high, top_final, 1.0, 1.0),
            (0.35, right_place_tf, right_place_tf, top_final, top_final, 1.0, 0.0),
            (0.55, right_place_tf, right_retreat_tf, top_final, top_final, 0.0, 0.0),
        )

    def _build_right_hand_targets(self) -> tuple[wp.array, wp.array, wp.array]:
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        q_indices = []
        open_values = []
        grasp_values = []
        targets = {
            "RIGHT_HAND_THUMB2": 0.84,
            "RIGHT_HAND_THUMB1": 0.46,
            "RIGHT_HAND_INDEX": 0.70,
            "RIGHT_INDEX_PIP": 0.90,
            "RIGHT_HAND_MIDDLE": 0.66,
            "RIGHT_MIDDLE_PIP": 0.82,
        }

        for joint_name, target in targets.items():
            joint_idx = self._joint_index(joint_name)
            q_idx = int(q_start[joint_idx])
            q_indices.append(q_idx)
            open_values.append(float(q_home[q_idx]))
            grasp_values.append(target)

        return (
            wp.array(q_indices, dtype=wp.int32, device=self.model.device),
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

    def _refresh_self_contact_bvh(self) -> None:
        if self.frame_index > 0 and self.frame_index % self.self_contact_bvh_rebuild_interval_frames == 0:
            self.solver.rebuild_bvh(self.state_0)

    def _sample_script(self, query_time: float) -> tuple[wp.transform, wp.transform, float]:
        remaining = query_time
        for duration, right_start, right_end, bun_start, bun_end, grasp_start, grasp_end in self.motion_segments:
            if remaining <= duration:
                alpha = float(np.clip(remaining / duration, 0.0, 1.0))
                right_tf = self._interpolate_transform(right_start, right_end, alpha)
                bun_tf = self._interpolate_transform(bun_start, bun_end, alpha)
                grasp_alpha = grasp_start * (1.0 - alpha) + grasp_end * alpha
                return right_tf, bun_tf, grasp_alpha
            remaining -= duration

        _, _, right_end, _, bun_end, _, grasp_end = self.motion_segments[-1]
        return right_end, bun_end, grasp_end

    def _prepare_frame_targets(self) -> None:
        wp.copy(self.frame_joint_q_start, self.state_0.joint_q)
        self.right_tf, self.top_bun_end_tf, grasp_alpha = self._sample_script(self.sim_time + self.frame_dt)
        self.top_bun_start_tf = self.top_bun_tf

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
            copy_ik_to_joint_q_kernel,
            dim=self.model.joint_coord_count,
            inputs=[self.ik_joint_q],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )
        wp.launch(
            set_indexed_joint_q_kernel,
            dim=self.right_hand_q_indices.shape[0],
            inputs=[self.right_hand_q_indices, self.right_hand_open, self.right_hand_grasp, grasp_alpha],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )

    def _set_top_bun_transform(self, xform: wp.transform) -> None:
        wp.launch(
            set_shape_transform_kernel,
            dim=1,
            inputs=[self.model.shape_transform, self.top_bun_shape, xform],
            device=self.model.device,
        )
        self.top_bun_tf = xform

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
            top_bun_tf = self._interpolate_transform(self.top_bun_start_tf, self.top_bun_end_tf, substep_alpha)
            self._set_top_bun_transform(top_bun_tf)

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

    def step(self) -> None:
        self.simulate()
        self._report_pose()

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        if hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo(
                "dexforce_right_tcp_target",
                self.right_tf,
                snap_to=self._current_tcp_transform(self.right_ee_index, self.right_ee_offset),
            )
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/burger_meat",
            self.state_0.particle_q,
            self.model.tri_indices.flatten(),
            hidden=not getattr(self.viewer, "show_triangles", True),
            backface_culling=False,
            color=BURGER_COLOR,
        )
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        if not np.all(np.isfinite(particle_q)):
            raise ValueError("BurgerMeat particle positions are not finite")

    def test_final(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        if not np.all(np.isfinite(particle_q)):
            raise ValueError("BurgerMeat particle positions are not finite")
        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)
        if bbox_size > 1.0:
            raise ValueError(f"BurgerMeat bounding box exploded: size={bbox_size:.3f}")
        if min_pos[2] < TABLE_TOP_Z - 0.05:
            raise ValueError(f"BurgerMeat fell through the table: z_min={min_pos[2]:.4f}")

    def _current_tcp_transform(self, body_index: int, offset: wp.vec3) -> wp.transform:
        body_q_np = self.state_0.body_q.numpy()
        body_tf = wp.transform(*body_q_np[body_index])
        body_pos = wp.transform_get_translation(body_tf)
        body_rot = wp.transform_get_rotation(body_tf)
        tcp_pos = body_pos + wp.quat_rotate(body_rot, offset)
        return wp.transform(tcp_pos, body_rot)

    def _offset_transform(self, tf: wp.transform, offset: wp.vec3) -> wp.transform:
        pos = wp.transform_get_translation(tf)
        return wp.transform(self._offset_vec3(pos, offset), wp.transform_get_rotation(tf))

    def _offset_vec3(self, vec: wp.vec3, offset: wp.vec3) -> wp.vec3:
        return wp.vec3(
            float(vec[0]) + float(offset[0]),
            float(vec[1]) + float(offset[1]),
            float(vec[2]) + float(offset[2]),
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
        scale_a = np.sin(theta_0 - theta) / sin_theta_0
        scale_b = np.sin(theta) / sin_theta_0
        return qa * scale_a + qb * scale_b

    def _normalize_quat(self, quat: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(quat))
        if norm == 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return quat / norm

    def _report_pose(self, force: bool = False) -> None:
        if not force and self.print_interval > 0.0 and self.sim_time - self.last_print_time < self.print_interval:
            return

        right_actual_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        right_target_pos = self._vec3_to_np(wp.transform_get_translation(self.right_tf))
        right_actual_pos = self._vec3_to_np(wp.transform_get_translation(right_actual_tf))
        right_target_rot = self._quat_to_np(wp.transform_get_rotation(self.right_tf))
        right_actual_rot = self._quat_to_np(wp.transform_get_rotation(right_actual_tf))
        bun_pos = self._vec3_to_np(wp.transform_get_translation(self.top_bun_tf))

        right_pos_err = float(np.linalg.norm(right_target_pos - right_actual_pos))
        right_rot_err = self._quat_angle_error_deg(right_target_rot, right_actual_rot)

        print(
            f"[{self.sim_time:7.3f}s] "
            f"R target={self._format_pose(right_target_pos, right_target_rot)} "
            f"actual={self._format_pose(right_actual_pos, right_actual_rot)} "
            f"pos_err={right_pos_err:.5f} m rot_err={right_rot_err:.3f} deg | "
            f"top_bun_pos={self._format_xyz(bun_pos)}"
        )
        self.last_print_time = self.sim_time

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
        parser.set_defaults(num_frames=360)
        parser.add_argument(
            "--print-interval",
            type=float,
            default=3.0,
            help="Seconds between TCP and top-bun pose reports. Use 0.0 to print every frame.",
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

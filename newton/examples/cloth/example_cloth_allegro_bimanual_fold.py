# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Allegro Bimanual Fold
#
# Two Allegro dexterous hands pick the midpoints of the left and right cloth
# edges, lift the cloth from a table, and bring the grasp points together to
# fold it in the air. The cloth is lifted through particle-shape contact with
# the hands; no cloth particles are kinematically pinned to the hand path.
#
# Command: python -m newton.examples cloth_allegro_bimanual_fold
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.mjvbd
import newton.utils
from newton import ModelBuilder, eval_fk


@wp.kernel
def set_hand_roots_kernel(
    left_root_joint: int,
    right_root_joint: int,
    left_xform: wp.transform,
    right_xform: wp.transform,
    joint_x_p: wp.array[wp.transform],
):
    if wp.tid() == 0:
        joint_x_p[left_root_joint] = left_xform
        joint_x_p[right_root_joint] = right_xform


@wp.kernel
def set_hand_finger_q_kernel(
    hand_q_indices: wp.array[wp.int32],
    hand_qd_indices: wp.array[wp.int32],
    hand_q_open: wp.array[wp.float32],
    hand_q_closed: wp.array[wp.float32],
    close_alpha: float,
    inv_dt: float,
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
):
    i = wp.tid()
    q_idx = hand_q_indices[i]
    q_target = hand_q_open[i] * (1.0 - close_alpha) + hand_q_closed[i] * close_alpha
    if inv_dt > 0.0:
        joint_qd[hand_qd_indices[i]] = (q_target - joint_q[q_idx]) * inv_dt
    else:
        joint_qd[hand_qd_indices[i]] = 0.0
    joint_q[q_idx] = q_target


@wp.kernel
def update_hand_body_velocity_kernel(
    left_body_start: int,
    left_body_end: int,
    right_body_start: int,
    right_body_end: int,
    inv_dt: float,
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    body_idx = wp.tid()
    in_left_hand = body_idx >= left_body_start and body_idx < left_body_end
    in_right_hand = body_idx >= right_body_start and body_idx < right_body_end
    if not (in_left_hand or in_right_hand):
        return

    q = body_q[body_idx]
    q_prev = body_q_prev[body_idx]
    velocity = wp.vec3(0.0, 0.0, 0.0)
    if inv_dt > 0.0:
        velocity = (wp.transform_get_translation(q) - wp.transform_get_translation(q_prev)) * inv_dt
    body_qd[body_idx] = wp.spatial_vector(velocity, wp.vec3(0.0, 0.0, 0.0))
    body_q_prev[body_idx] = q


TABLE_POS = wp.vec3(0.28, 0.0, 0.625)
TABLE_HALF_EXTENTS = (0.42, 0.42, 0.035)
TABLE_TOP_Z = float(TABLE_POS[2]) + TABLE_HALF_EXTENTS[2]

CLOTH_DIM_X = 18
CLOTH_DIM_Y = 16
CLOTH_CELL_X = 0.025
CLOTH_CELL_Y = 0.025
CLOTH_CENTER = wp.vec3(0.28, 0.0, TABLE_TOP_Z + 0.006)
CLOTH_POS = wp.vec3(
    float(CLOTH_CENTER[0]) - 0.5 * CLOTH_DIM_X * CLOTH_CELL_X,
    float(CLOTH_CENTER[1]) - 0.5 * CLOTH_DIM_Y * CLOTH_CELL_Y,
    float(CLOTH_CENTER[2]),
)
CLOTH_COLOR = (0.9, 0.2, 0.16)

HAND_ROT = wp.normalize(wp.quat(0.21643, 0.706218, -0.648166, 0.185191))
LEFT_HAND_SIDE_OFFSET = wp.vec3(0.0, -0.03, 0.02)
RIGHT_HAND_SIDE_OFFSET = wp.vec3(0.0, 0.03, 0.02)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0

        self.particle_radius = 0.007
        self.soft_contact_margin = 0.014
        self.particle_self_contact_radius = 0.006
        self.particle_self_contact_margin = 0.006
        self.self_contact_bvh_rebuild_interval_frames = 30

        builder = ModelBuilder(gravity=-9.81)
        self.hand_shape_start = builder.shape_count
        self._add_hands(builder)
        self.hand_shape_end = builder.shape_count
        self._add_table(builder)
        self.cloth_start = builder.particle_count
        self._add_cloth(builder)
        builder.color(include_bending=True)

        self.model = builder.finalize(requires_grad=False)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self._configure_hand_collisions()

        self.left_root_joint = self._joint_index("allegro_hand_left/fixed_base")
        self.right_root_joint = self._joint_index("allegro_hand_right/fixed_base")
        self.hand_q_indices, self.hand_qd_indices, self.hand_q_open, self.hand_q_closed = (
            self._build_hand_joint_arrays()
        )
        self.hand_body_q_prev = wp.clone(self.state_0.body_q, device=self.model.device)

        self.model.soft_contact_ke = 5.0e5
        self.model.soft_contact_kd = 1.0e-6
        self.model.soft_contact_mu = 1.25
        self.model.shape_material_ke.fill_(5.0e5)
        self.model.shape_material_kd.fill_(1.0e-6)
        self.model.shape_material_mu.fill_(2.0)

        self.solver = newton.mjvbd.SolverMJVBD(
            self.model,
            rigid_contact_max=0,
            soft_contact_margin=self.soft_contact_margin,
            iterations=10,
            particle_self_contact_radius=self.particle_self_contact_radius,
            particle_self_contact_margin=self.particle_self_contact_margin,
            particle_enable_self_contact=True,
            particle_vertex_contact_buffer_size=16,
            particle_edge_contact_buffer_size=24,
            particle_collision_detection_interval=-1,
        )

        self.left_grasp_start, self.right_grasp_start = self._initial_grasp_targets()
        self.initial_cloth_height = float(np.max(self.state_0.particle_q.numpy()[:, 2]))
        self.max_cloth_height = self.initial_cloth_height
        self.expected_lift_height = TABLE_TOP_Z + 0.10
        self._update_kinematic_hands(0.0, self.state_0, inv_dt=0.0)
        self.state_1.assign(self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.72, -0.95, 0.98), -28.0, 42.0)
        self.capture()

    def _add_hands(self, builder: ModelBuilder) -> None:
        asset_path = newton.utils.download_asset("wonik_allegro")
        self.left_body_start = builder.body_count
        builder.add_urdf(
            asset_path / "urdf" / "allegro_hand_description_left.urdf",
            xform=self._hand_xform(wp.vec3(0.24, 0.30, TABLE_TOP_Z + 0.10), left=True),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            force_show_colliders=False,
        )
        self.left_body_end = builder.body_count
        self.right_body_start = builder.body_count
        builder.add_urdf(
            asset_path / "urdf" / "allegro_hand_description_right.urdf",
            xform=self._hand_xform(wp.vec3(0.24, -0.30, TABLE_TOP_Z + 0.10), left=False),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            force_show_colliders=False,
        )
        self.right_body_end = builder.body_count
        self._add_fingertip_grip_pads(builder)
        self._add_palm_grasp_pads(builder)

        for joint_idx, label in enumerate(builder.joint_label):
            if not (label.startswith("allegro_hand_left/") or label.startswith("allegro_hand_right/")):
                continue
            if builder.joint_type[joint_idx] != newton.JointType.REVOLUTE:
                continue
            q_idx = builder.joint_q_start[joint_idx]
            dof_idx = builder.joint_qd_start[joint_idx]
            builder.joint_q[q_idx] = 0.15
            builder.joint_target_pos[dof_idx] = 0.15
            builder.joint_target_ke[dof_idx] = 150.0
            builder.joint_target_kd[dof_idx] = 8.0
            builder.joint_armature[dof_idx] = 1.0e-2

    def _add_fingertip_grip_pads(self, builder: ModelBuilder) -> None:
        pad_cfg = ModelBuilder.ShapeConfig()
        pad_cfg.has_shape_collision = False
        pad_cfg.has_particle_collision = True
        pad_cfg.mu = 3.0
        pad_cfg.ke = 5.0e5
        pad_cfg.kd = 1.0e-6

        tip_names = ("link_11.0_tip", "link_7.0_tip", "link_3.0_tip", "link_15.0_tip")
        for hand_name in ("allegro_hand_left", "allegro_hand_right"):
            for tip_name in tip_names:
                body = self._builder_body_index(builder, f"{hand_name}/{tip_name}")
                builder.add_shape_sphere(
                    body=body,
                    xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                    radius=0.018,
                    cfg=pad_cfg,
                    color=(0.02, 0.025, 0.025),
                    label=f"{hand_name}/{tip_name}_grip_pad",
                )

    def _add_palm_grasp_pads(self, builder: ModelBuilder) -> None:
        pad_cfg = ModelBuilder.ShapeConfig()
        pad_cfg.has_shape_collision = False
        pad_cfg.has_particle_collision = True
        pad_cfg.mu = 3.0
        pad_cfg.ke = 5.0e5
        pad_cfg.kd = 1.0e-6

        for hand_name, left, side_offset in (
            ("allegro_hand_left", True, LEFT_HAND_SIDE_OFFSET),
            ("allegro_hand_right", False, RIGHT_HAND_SIDE_OFFSET),
        ):
            palm_body = self._builder_body_index(builder, f"{hand_name}/palm_link")
            grasp_target = self._nominal_grasp_target(left)
            hand_xform = self._hand_xform(grasp_target + side_offset, left)
            lower_pad = wp.transform(
                wp.vec3(float(grasp_target[0]), float(grasp_target[1]), TABLE_TOP_Z + 0.001),
                wp.quat_identity(),
            )
            upper_pad = wp.transform(
                wp.vec3(float(grasp_target[0]), float(grasp_target[1]), TABLE_TOP_Z + 0.032),
                wp.quat_identity(),
            )

            for name, world_xform, hz in (("lower", lower_pad, 0.005), ("upper", upper_pad, 0.006)):
                local_xform = wp.transform_multiply(wp.transform_inverse(hand_xform), world_xform)
                builder.add_shape_box(
                    body=palm_body,
                    xform=local_xform,
                    hx=0.055,
                    hy=0.012,
                    hz=hz,
                    cfg=pad_cfg,
                    color=(0.015, 0.018, 0.018),
                    label=f"{hand_name}/palm_{name}_grasp_pad",
                )

    def _add_table(self, builder: ModelBuilder) -> None:
        table_cfg = ModelBuilder.ShapeConfig()
        table_cfg.mu = 0.8
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(TABLE_POS, wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.36, 0.42, 0.48),
        )

    def _add_cloth(self, builder: ModelBuilder) -> None:
        builder.add_cloth_grid(
            pos=CLOTH_POS,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=CLOTH_DIM_X,
            dim_y=CLOTH_DIM_Y,
            cell_x=CLOTH_CELL_X,
            cell_y=CLOTH_CELL_Y,
            mass=0.003,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e-5,
            edge_ke=1.0,
            edge_kd=0.05,
            particle_radius=self.particle_radius,
            label="fold_cloth",
        )

    def _configure_hand_collisions(self) -> None:
        flags = self.model.shape_flags.numpy()
        hand_slice = slice(self.hand_shape_start, self.hand_shape_end)
        flags[hand_slice] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
        flags[hand_slice] &= ~int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

    def _build_hand_joint_arrays(self) -> tuple[wp.array, wp.array, wp.array, wp.array]:
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        lower = self.model.joint_limit_lower.numpy()
        upper = self.model.joint_limit_upper.numpy()
        indices: list[int] = []
        qd_indices: list[int] = []
        open_values: list[float] = []
        closed_values: list[float] = []

        for joint_idx, label in enumerate(self.model.joint_label):
            if not (label.startswith("allegro_hand_left/") or label.startswith("allegro_hand_right/")):
                continue
            if self.model.joint_type.numpy()[joint_idx] != int(newton.JointType.REVOLUTE):
                continue
            q_idx = int(q_start[joint_idx])
            indices.append(q_idx)
            qd_indices.append(int(qd_start[joint_idx]))
            open_values.append(float(np.clip(0.12, lower[q_idx], upper[q_idx])))
            closed_values.append(float(np.clip(0.75, lower[q_idx], upper[q_idx])))

        return (
            wp.array(indices, dtype=wp.int32, device=self.model.device),
            wp.array(qd_indices, dtype=wp.int32, device=self.model.device),
            wp.array(open_values, dtype=wp.float32, device=self.model.device),
            wp.array(closed_values, dtype=wp.float32, device=self.model.device),
        )

    def _initial_grasp_targets(self) -> tuple[wp.vec3, wp.vec3]:
        return self._nominal_grasp_target(left=True), self._nominal_grasp_target(left=False)

    def _nominal_grasp_target(self, left: bool) -> wp.vec3:
        x = float(CLOTH_POS[0]) + float(CLOTH_DIM_X // 2) * CLOTH_CELL_X
        y = float(CLOTH_POS[1]) + (CLOTH_DIM_Y * CLOTH_CELL_Y if left else 0.0)
        return wp.vec3(x, y, TABLE_TOP_Z + 0.004)

    def _trajectory(self, t: float) -> tuple[wp.vec3, wp.vec3, float]:
        left_start = self.left_grasp_start + wp.vec3(0.0, 0.035, 0.055)
        right_start = self.right_grasp_start + wp.vec3(0.0, -0.035, 0.055)
        left_grasp = self.left_grasp_start
        right_grasp = self.right_grasp_start
        lift_z = TABLE_TOP_Z + 0.30
        fold_z = TABLE_TOP_Z + 0.34
        left_lift = wp.vec3(float(left_grasp[0]), float(left_grasp[1]), lift_z)
        right_lift = wp.vec3(float(right_grasp[0]), float(right_grasp[1]), lift_z)
        left_fold = wp.vec3(float(CLOTH_CENTER[0]), 0.10, fold_z)
        right_fold = wp.vec3(float(CLOTH_CENTER[0]), -0.10, fold_z + 0.015)

        if t < 0.8:
            alpha = self._smoothstep(t / 0.8)
            return self._lerp_vec3(left_start, left_grasp, alpha), self._lerp_vec3(right_start, right_grasp, alpha), 0.0
        if t < 1.8:
            alpha = self._smoothstep((t - 0.8) / 1.0)
            return left_grasp, right_grasp, alpha
        if t < 4.4:
            alpha = self._smoothstep((t - 1.8) / 2.6)
            return self._lerp_vec3(left_grasp, left_lift, alpha), self._lerp_vec3(right_grasp, right_lift, alpha), 1.0
        if t < 7.0:
            alpha = self._smoothstep((t - 4.4) / 2.6)
            return self._lerp_vec3(left_lift, left_fold, alpha), self._lerp_vec3(right_lift, right_fold, alpha), 1.0
        return left_fold, right_fold, 1.0

    def _update_kinematic_hands(self, t: float, state: newton.State, inv_dt: float) -> None:
        left_target, right_target, close_alpha = self._trajectory(t)

        left_hand = self._hand_xform(left_target + LEFT_HAND_SIDE_OFFSET, left=True)
        right_hand = self._hand_xform(right_target + RIGHT_HAND_SIDE_OFFSET, left=False)
        wp.launch(
            set_hand_roots_kernel,
            dim=1,
            inputs=[self.left_root_joint, self.right_root_joint, left_hand, right_hand],
            outputs=[self.model.joint_X_p],
            device=self.model.device,
        )
        wp.launch(
            set_hand_finger_q_kernel,
            dim=self.hand_q_indices.shape[0],
            inputs=[
                self.hand_q_indices,
                self.hand_qd_indices,
                self.hand_q_open,
                self.hand_q_closed,
                close_alpha,
                inv_dt,
            ],
            outputs=[self.model.joint_q, self.model.joint_qd],
            device=self.model.device,
        )
        eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
        wp.launch(
            update_hand_body_velocity_kernel,
            dim=self.model.body_count,
            inputs=[
                self.left_body_start,
                self.left_body_end,
                self.right_body_start,
                self.right_body_end,
                inv_dt,
                state.body_q,
            ],
            outputs=[self.hand_body_q_prev, state.body_qd],
            device=self.model.device,
        )

    def _refresh_self_contact_bvh(self) -> None:
        if self.frame_index > 0 and self.frame_index % self.self_contact_bvh_rebuild_interval_frames == 0:
            self.solver.rebuild_bvh(self.state_0)

    def capture(self) -> None:
        # The hand targets are generated from Python time each substep. Capturing
        # this example would freeze the trajectory at the capture-time pose.
        self.graph = None

    def simulate(self) -> None:
        self._refresh_self_contact_bvh()
        for substep in range(self.sim_substeps):
            substep_time = self.sim_time + (substep + 1) * self.sim_dt
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self._update_kinematic_hands(substep_time, self.state_0, 1.0 / self.sim_dt)

            wp.copy(self.state_1.particle_q, self.state_0.particle_q)
            wp.copy(self.state_1.particle_qd, self.state_0.particle_qd)
            wp.copy(self.state_1.body_q, self.state_0.body_q)
            wp.copy(self.state_1.body_qd, self.state_0.body_qd)

            self.solver.step(self.state_0, self.state_1, self.control, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.sim_time += self.frame_dt
        self.frame_index += 1
        self._track_cloth_height()

    def step(self) -> None:
        if self.graph:
            wp.capture_launch(self.graph)
            self.sim_time += self.frame_dt
            self.frame_index += 1
        else:
            self.simulate()

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/fold_cloth",
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
        if self.max_cloth_height < self.expected_lift_height:
            raise ValueError(
                f"Expected hand contacts to lift the cloth above {self.expected_lift_height:.3f} m, "
                f"max cloth height was {self.max_cloth_height:.3f} m"
            )
        particle_q = self.state_0.particle_q.numpy()
        if not np.all(np.isfinite(particle_q)):
            raise ValueError("Cloth particle positions contain non-finite values")

    def _track_cloth_height(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        self.max_cloth_height = max(self.max_cloth_height, float(np.max(particle_q[:, 2])))

    def _cloth_grid_index(self, x: int, y: int) -> int:
        return self.cloth_start + y * (CLOTH_DIM_X + 1) + x

    def _cloth_particle_initial_pos(self, particle_idx: int) -> wp.vec3:
        return wp.vec3(*self.model.particle_q.numpy()[particle_idx])

    def _builder_body_index(self, builder: ModelBuilder, label: str) -> int:
        return next(i for i, body_label in enumerate(builder.body_label) if body_label == label)

    def _joint_index(self, label: str) -> int:
        return next(i for i, joint_label in enumerate(self.model.joint_label) if joint_label == label)

    def _hand_xform(self, pos: wp.vec3, left: bool) -> wp.transform:
        yaw = -np.pi * 0.5 if left else np.pi * 0.5
        side_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(yaw))
        return wp.transform(pos, side_rot * HAND_ROT)

    def _lerp_vec3(self, a: wp.vec3, b: wp.vec3, alpha: float) -> wp.vec3:
        return a * (1.0 - alpha) + b * alpha

    def _smoothstep(self, x: float) -> float:
        t = float(np.clip(x, 0.0, 1.0))
        return t * t * (3.0 - 2.0 * t)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=360)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
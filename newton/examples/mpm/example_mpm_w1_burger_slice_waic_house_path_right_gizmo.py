# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# WAIC house W1 burger slice scripted carry path + right-hand gizmo
#
# Runs the full-house V4 burger-slicing scene through the scripted knife work,
# pan grasp, and pan lift. The carried W1/pan/meat group then follows a fixed
# right-turn / forward-move path. At the end only the right-hand TCP gizmo is
# exposed so the final hand pose can be tuned interactively.
#
# Command:
#   python -m newton.examples mpm_w1_burger_slice_waic_house_path_right_gizmo
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples

from newton.examples.mpm import example_mpm_w1_burger_slice_waic_house as house
from newton.examples.mpm import example_mpm_w1_burger_slice_waic_kitchen_V4 as v4


RIGHT_TURN_DEGREES = -90.0
DESTINATION_TABLE_TOP_Z = 1.1662
DESTINATION_TABLE_BOX_HALF_EXTENTS = (0.42, 0.42, 0.025)
DESTINATION_PAN_CENTER = wp.vec3(
    3.470792,
    1.881144,
    DESTINATION_TABLE_TOP_Z + v4.PAN_DISK_HALF_HEIGHT,
)
FINAL_RIGHT_TARGET_TF = wp.transform(
    wp.vec3(3.382711, 1.783910, 1.337491),
    wp.quat(0.544761, -0.008398, 0.837510, 0.041724),
)
ROOT_TARGETS = (
    wp.vec3(4.8, -0.85, 0.29),
    wp.vec3(3.0, -0.85, 0.29),
    wp.vec3(2.8, 2.22, 0.29),
)


class Example(house.Example):
    """Script the W1 carry route, then open a right-hand TCP gizmo."""

    def __init__(self, viewer, args):
        self.turn_time = float(args.path_turn_time)
        self.move_time = float(args.path_move_time)
        self.right_gizmo_print_interval = float(args.pose_print_interval)
        self.last_pose_print_time = -1.0e9
        self.right_gizmo_mode = False
        self.path_start_root_tf: wp.transform | None = None
        self.gizmo_carry_tf = wp.transform_identity()
        self.pan_placed_on_destination_table = False
        super().__init__(viewer, args)

        self.path_duration = 4.0 * self.turn_time + 3.0 * self.move_time
        self.right_gizmo_tf = self.pan_handle_lift_tf
        self.left_hold_tf_for_gizmo = self.left_tf

    def _add_simplified_pan(self, builder: newton.ModelBuilder) -> None:
        # Use V4's hidden simplified pan collision. The full-house base class
        # makes this debug collision visible; this path demo should render only
        # the visual pan mesh.
        v4.Example._add_simplified_pan(self, builder)

    def _add_rear_table(self, builder: newton.ModelBuilder) -> None:
        super()._add_rear_table(builder)

        table_cfg = newton.ModelBuilder.ShapeConfig()
        table_cfg.ke = 8.0e5
        table_cfg.kd = 1.0e-6
        table_cfg.mu = 1.2
        table_cfg.density = 0.0
        table_cfg.has_shape_collision = False
        table_cfg.has_particle_collision = True
        table_cfg.is_visible = False

        hx, hy, hz = DESTINATION_TABLE_BOX_HALF_EXTENTS
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(
                wp.vec3(
                    float(DESTINATION_PAN_CENTER[0]),
                    float(DESTINATION_PAN_CENTER[1]),
                    DESTINATION_TABLE_TOP_Z - hz,
                ),
                wp.quat_identity(),
            ),
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=table_cfg,
            color=(0.20, 0.24, 0.28),
            label="waic_destination_hidden_table_box",
        )

    def _global_carry_transform(self, carry_alpha: float) -> wp.transform:
        # V4 calls this during __init__ before bodies exist. Provide a harmless
        # fallback final transform; the actual scripted path is computed by
        # _rigid_carry_transform once the body state is available.
        yaw = math.radians(4.0 * RIGHT_TURN_DEGREES) * float(carry_alpha)
        yaw_rot = wp.quat(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))
        return wp.transform(wp.vec3(0.0, 0.0, 0.0), yaw_rot)

    def _sample_post_place_script(self, t: float):
        return None

    def _rigid_carry_transform(
        self, query_time: float
    ) -> tuple[wp.transform, wp.transform, wp.transform, float, float]:
        path_t = max(float(query_time) - self.rigid_carry_start_time, 0.0)
        if path_t <= 0.0:
            return (
                wp.transform_identity(),
                wp.transform_identity(),
                wp.transform_identity(),
                0.0,
                0.0,
            )

        if self.path_start_root_tf is None:
            self.path_start_root_tf = self._current_robot_root_transform()

        path_alpha = float(np.clip(path_t / max(self.path_duration, 1.0e-6), 0.0, 1.0))
        target_root_tf = self._sample_root_path(path_t)
        carry_tf = self._compose_transform(target_root_tf, self._transform_inverse(self.path_start_root_tf))
        return carry_tf, carry_tf, carry_tf, path_alpha, 0.0

    def _sample_root_path(self, path_t: float) -> wp.transform:
        start_tf = self.path_start_root_tf or self._current_robot_root_transform()
        start_pos = wp.transform_get_translation(start_tf)
        start_rot = wp.transform_get_rotation(start_tf)

        waypoints = (start_pos, *ROOT_TARGETS)
        segment = 0
        local_t = float(path_t)
        yaw_turns = 0

        while segment < 7:
            duration = self.turn_time if segment % 2 == 0 else self.move_time
            if local_t <= duration:
                break
            local_t -= duration
            if segment % 2 == 0:
                yaw_turns += 1
            segment += 1

        segment = min(segment, 6)
        duration = self.turn_time if segment % 2 == 0 else self.move_time
        u = self._smoothstep(local_t / max(duration, 1.0e-6))

        if segment % 2 == 0:
            yaw = math.radians(RIGHT_TURN_DEGREES) * (yaw_turns + u)
            pos_index = min(yaw_turns, len(waypoints) - 1)
            pos = waypoints[pos_index]
        else:
            yaw = math.radians(RIGHT_TURN_DEGREES) * yaw_turns
            move_index = min(yaw_turns, len(waypoints) - 1)
            pos = self._lerp_vec3(waypoints[move_index - 1], waypoints[move_index], u)

        yaw_rot = wp.quat(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))
        return wp.transform(pos, self._quat_multiply(yaw_rot, start_rot))

    def step(self) -> None:
        gizmo_start_time = self.rigid_carry_start_time + self.path_duration
        if not self.right_gizmo_mode and self.sim_time < gizmo_start_time:
            self.simulate()
            return

        if not self.right_gizmo_mode:
            self._enter_right_gizmo_mode()

        self._solve_right_gizmo_ik()
        self.sim_time += self.frame_dt
        self.frame_index += 1
        self._report_right_gizmo_pose()

    def _enter_right_gizmo_mode(self) -> None:
        self.right_gizmo_mode = True
        root_tf = self._current_robot_root_transform()
        path_start_tf = self.path_start_root_tf or root_tf
        self.gizmo_carry_tf = self._compose_transform(root_tf, self._transform_inverse(path_start_tf))
        self._place_pan_and_meat_on_destination_table()
        self.right_gizmo_tf = FINAL_RIGHT_TARGET_TF
        self.left_hold_tf_for_gizmo = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        self.right_tf = self.right_gizmo_tf
        self.left_tf = self.left_hold_tf_for_gizmo
        print(
            "[newton] Scripted carry path complete. Pan placed on destination hidden table box; "
            "drag dexforce_right_pan_tcp_target to tune the right hand."
        )

    def _place_pan_and_meat_on_destination_table(self) -> None:
        if self.pan_placed_on_destination_table:
            return

        current_pan_tf = self._current_pan_transform()
        current_pan_pos = wp.transform_get_translation(current_pan_tf)
        current_pan_rot = wp.transform_get_rotation(current_pan_tf)
        final_pan_tf = wp.transform(DESTINATION_PAN_CENTER, current_pan_rot)
        pan_delta = DESTINATION_PAN_CENTER - current_pan_pos

        wp.launch(
            v4.set_body_transform_kernel,
            dim=1,
            inputs=[self.state_0.body_q, self.pan_body, final_pan_tf],
            device=self.model.device,
        )
        wp.launch(
            v4.set_body_velocity_kernel,
            dim=1,
            inputs=[self.state_0.body_qd, self.pan_body, wp.vec3(0.0, 0.0, 0.0)],
            device=self.model.device,
        )
        wp.launch(
            v4.apply_particle_delta_transform_kernel,
            dim=self.model.particle_count,
            inputs=[
                self.state_0.particle_q,
                self.state_0.particle_qd,
                wp.transform(pan_delta, wp.quat_identity()),
                1.0 / self.frame_dt,
            ],
            device=self.model.device,
        )

        self.pan_placed_on_destination_table = True

    def _solve_right_gizmo_ik(self) -> None:
        inv_carry_tf = self._transform_inverse(self.gizmo_carry_tf)
        self.right_tf = self._compose_transform(inv_carry_tf, self.right_gizmo_tf)
        self.left_tf = self._compose_transform(inv_carry_tf, self.left_hold_tf_for_gizmo)
        wp.copy(self.frame_joint_q_start, self.state_0.joint_q)

        self.right_pos_obj.set_target_position(0, wp.transform_get_translation(self.right_tf))
        self.right_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.right_tf)))
        self.left_pos_obj.set_target_position(0, wp.transform_get_translation(self.left_tf))
        self.left_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.left_tf)))

        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            v4.lock_joint_q_kernel,
            dim=self.locked_q_indices.shape[0],
            inputs=[self.ik_joint_q, self.locked_q_indices, self.locked_q_values],
            device=self.model.device,
        )
        wp.launch(
            v4.copy_ik_to_joint_q_kernel,
            dim=self.model.joint_coord_count,
            inputs=[self.ik_joint_q],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )
        wp.launch(
            v4.set_indexed_joint_q_kernel,
            dim=self.right_hand_q_indices.shape[0],
            inputs=[self.right_hand_q_indices, self.right_hand_open, self.right_hand_grasp, 1.0],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )

        wp.copy(self.state_0.joint_q, self.frame_joint_q_end)
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._reapply_gizmo_carry_pose()

    def _reapply_gizmo_carry_pose(self) -> None:
        wp.launch(
            v4.apply_body_transform_kernel,
            dim=self.rigid_robot_body_indices.shape[0],
            inputs=[self.state_0.body_q, self.rigid_robot_body_indices, self.gizmo_carry_tf],
            device=self.model.device,
        )
        wp.launch(
            v4.set_transformed_body_poses_kernel,
            dim=self.rigid_pan_body_indices.shape[0],
            inputs=[
                self.rigid_carry_body_q_start,
                self.rigid_pan_body_indices,
                self.gizmo_carry_tf,
                1.0 / self.frame_dt,
            ],
            outputs=[self.state_0.body_q, self.state_0.body_qd],
            device=self.model.device,
        )
        if self.pan_placed_on_destination_table:
            current_pan_rot = wp.transform_get_rotation(self._current_pan_transform())
            wp.launch(
                v4.set_body_transform_kernel,
                dim=1,
                inputs=[
                    self.state_0.body_q,
                    self.pan_body,
                    wp.transform(DESTINATION_PAN_CENTER, current_pan_rot),
                ],
                device=self.model.device,
            )
        if self.rigid_carry_shape_indices.shape[0] > 0:
            wp.launch(
                v4.set_transformed_shape_transforms_kernel,
                dim=self.rigid_carry_shape_indices.shape[0],
                inputs=[
                    self.rigid_carry_shape_transform_start,
                    self.rigid_carry_shape_indices,
                    self.gizmo_carry_tf,
                ],
                outputs=[self.model.shape_transform],
                device=self.model.device,
            )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        if self.right_gizmo_mode and hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo(
                "dexforce_right_pan_tcp_target",
                self.right_gizmo_tf,
                snap_to=self._current_tcp_transform(self.right_ee_index, self.right_ee_offset),
            )
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "/meat",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.end_frame()
        wp.synchronize()

    def _report_right_gizmo_pose(self, force: bool = False) -> None:
        if not self.right_gizmo_mode:
            return
        if (
            not force
            and self.right_gizmo_print_interval > 0.0
            and self.sim_time - self.last_pose_print_time < self.right_gizmo_print_interval
        ):
            return
        self.last_pose_print_time = self.sim_time

        root_tf = self._current_robot_root_transform()
        root_pos = self._vec3_to_np(wp.transform_get_translation(root_tf))
        root_rot = self._quat_to_np(wp.transform_get_rotation(root_tf))
        right_pos = self._vec3_to_np(wp.transform_get_translation(self.right_gizmo_tf))
        right_rot = self._quat_to_np(wp.transform_get_rotation(self.right_gizmo_tf))
        actual_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        actual_pos = self._vec3_to_np(wp.transform_get_translation(actual_tf))
        actual_rot = self._quat_to_np(wp.transform_get_rotation(actual_tf))
        pos_err = float(np.linalg.norm(right_pos - actual_pos))
        rot_err = self._quat_angle_error_deg(right_rot, actual_rot)
        pan_pos = self._vec3_to_np(wp.transform_get_translation(self._current_pan_transform()))

        print(
            "[waic path right gizmo] "
            f"root_pos=({root_pos[0]:.6f}, {root_pos[1]:.6f}, {root_pos[2]:.6f}) "
            f"root_quat_xyzw=({root_rot[0]:.6f}, {root_rot[1]:.6f}, {root_rot[2]:.6f}, {root_rot[3]:.6f}) "
            f"pan_pos=({pan_pos[0]:.6f}, {pan_pos[1]:.6f}, {pan_pos[2]:.6f}) "
            f"right_target_pos=({right_pos[0]:.6f}, {right_pos[1]:.6f}, {right_pos[2]:.6f}) "
            f"right_target_quat_xyzw=({right_rot[0]:.6f}, {right_rot[1]:.6f}, {right_rot[2]:.6f}, {right_rot[3]:.6f}) "
            f"right_actual_pos=({actual_pos[0]:.6f}, {actual_pos[1]:.6f}, {actual_pos[2]:.6f}) "
            f"right_actual_quat_xyzw=({actual_rot[0]:.6f}, {actual_rot[1]:.6f}, {actual_rot[2]:.6f}, {actual_rot[3]:.6f}) "
            f"right_err=({pos_err:.4f} m, {rot_err:.2f} deg)"
        )

    def _current_robot_root_transform(self) -> wp.transform:
        body_q = self.state_0.body_q.numpy()
        return wp.transform(*body_q[int(self.rigid_robot_body_indices.numpy()[0])])

    def _current_pan_transform(self) -> wp.transform:
        body_q = self.state_0.body_q.numpy()
        return wp.transform(*body_q[self.pan_body])

    @staticmethod
    def _lerp_vec3(a: wp.vec3, b: wp.vec3, alpha: float) -> wp.vec3:
        return wp.vec3(
            float(a[0]) * (1.0 - alpha) + float(b[0]) * alpha,
            float(a[1]) * (1.0 - alpha) + float(b[1]) * alpha,
            float(a[2]) * (1.0 - alpha) + float(b[2]) * alpha,
        )

    @staticmethod
    def _quat_angle_error_deg(qa: np.ndarray, qb: np.ndarray) -> float:
        qa = qa / max(float(np.linalg.norm(qa)), 1.0e-9)
        qb = qb / max(float(np.linalg.norm(qb)), 1.0e-9)
        dot = abs(float(np.dot(qa, qb)))
        dot = float(np.clip(dot, -1.0, 1.0))
        return float(np.degrees(2.0 * np.arccos(dot)))

    @staticmethod
    def create_parser():
        parser = house.Example.create_parser()
        parser.set_defaults(num_frames=100000, paused=False)
        parser.add_argument(
            "--path-turn-time",
            type=float,
            default=2.0,
            help="Seconds for each 90-degree right turn in the scripted carry path.",
        )
        parser.add_argument(
            "--path-move-time",
            type=float,
            default=3.0,
            help="Seconds for each forward move between root target points.",
        )
        parser.add_argument(
            "--pose-print-interval",
            type=float,
            default=0.5,
            help="Seconds between console prints after the right-hand gizmo opens.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

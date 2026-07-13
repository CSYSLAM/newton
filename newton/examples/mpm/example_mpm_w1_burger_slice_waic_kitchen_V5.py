# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MPM W1 Burger Slice in the WAIC kitchen V5
#
# Based on the full-house scripted carry path. After reaching the destination,
# this version does not open an IK gizmo. Instead, it keeps the carried pan/hand
# relation, moves the pan a short distance forward onto the lowered table,
# levels it onto the hidden table box, and clears the hand from the pan's right
# side.
#
# Command:
#   python -m newton.examples mpm_w1_burger_slice_waic_kitchen_V5
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

from newton.examples.mpm import example_mpm_w1_burger_slice_waic_house_path_right_gizmo as path
from newton.examples.mpm import example_mpm_w1_burger_slice_waic_kitchen_V4 as v4


TABLE_TOP_Z = 0.9000
TABLE_NEAR_EDGE_X = 3.45926
TABLE_CENTER_Y = 1.881144
TABLE_BOX_HALF_EXTENTS = (0.58, 0.50, 0.025)
TABLE_BOX_CENTER = wp.vec3(
    TABLE_NEAR_EDGE_X + TABLE_BOX_HALF_EXTENTS[0],
    TABLE_CENTER_Y,
    TABLE_TOP_Z - TABLE_BOX_HALF_EXTENTS[2],
)

PAN_TABLE_EDGE_CLEARANCE = 0.060
PAN_APPROACH_CLEARANCE_Z = 0.030
V5_MEAT_PAN_GAP = 0.003
PAN_PLACE_CENTER = wp.vec3(
    TABLE_NEAR_EDGE_X + v4.PAN_RADIUS + PAN_TABLE_EDGE_CLEARANCE,
    TABLE_CENTER_Y,
    TABLE_TOP_Z + v4.PAN_DISK_HALF_HEIGHT,
)


class Example(path.Example):
    """Scripted V5: carry, place pan on the lowered table, clear hand."""

    def __init__(self, viewer, args):
        self.post_forward_time = float(args.post_forward_time)
        self.post_lower_time = float(args.post_lower_time)
        self.post_release_time = float(args.post_release_time)
        self.post_clear_time = float(args.post_clear_time)
        self.pose_print_interval = float(args.pose_print_interval)
        self.last_pose_print_time = -1.0e9
        self.post_mode = False
        self.post_start_time = 0.0
        self.post_start_right_tf = wp.transform_identity()
        self.post_start_pan_tf = wp.transform_identity()
        self.post_right_local_pos = wp.vec3(0.0, 0.0, 0.0)
        self.post_right_local_rot = wp.quat_identity()
        self.post_approach_tf = wp.transform_identity()
        self.post_place_tf = wp.transform_identity()
        self.post_place_right_tf = wp.transform_identity()
        self.post_clear_tf = wp.transform_identity()
        self.post_prev_pan_tf = wp.transform_identity()
        self.post_current_hand_alpha = 1.0
        self.post_current_elapsed = 0.0
        super().__init__(viewer, args)

    def _add_rear_table(self, builder: newton.ModelBuilder) -> None:
        v4.Example._add_rear_table(self, builder)

        table_cfg = newton.ModelBuilder.ShapeConfig()
        table_cfg.ke = 8.0e5
        table_cfg.kd = 1.0e-6
        table_cfg.mu = 1.2
        table_cfg.density = 0.0
        table_cfg.has_shape_collision = False
        table_cfg.has_particle_collision = True
        table_cfg.is_visible = False

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(TABLE_BOX_CENTER, wp.quat_identity()),
            hx=TABLE_BOX_HALF_EXTENTS[0],
            hy=TABLE_BOX_HALF_EXTENTS[1],
            hz=TABLE_BOX_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.20, 0.24, 0.28),
            label="waic_v5_destination_hidden_table_box",
        )

    def _emit_meat(self, builder: newton.ModelBuilder, args) -> None:
        self.meat_lo = np.array(
            [
                float(self.pan_center[0]) - 0.5 * self.meat_length,
                float(self.pan_center[1]) - 0.5 * self.meat_width,
                self.pan_top_z + float(args.v5_meat_pan_gap),
            ]
        )
        self.meat_hi = np.array(
            [
                float(self.pan_center[0]) + 0.5 * self.meat_length,
                float(self.pan_center[1]) + 0.5 * self.meat_width,
                self.pan_top_z + float(args.v5_meat_pan_gap) + self.meat_height,
            ]
        )
        super()._emit_meat(builder, args)

    def step(self) -> None:
        path_end_time = self.rigid_carry_start_time + self.path_duration
        if not self.post_mode and self.sim_time < path_end_time:
            self.simulate()
            return

        if not self.post_mode:
            self._enter_post_mode()

        self._step_post_place_script()
        self.sim_time += self.frame_dt
        self.frame_index += 1
        self._report_post_pose()

    def _enter_post_mode(self) -> None:
        self.post_mode = True
        self.post_start_time = self.sim_time

        root_tf = self._current_robot_root_transform()
        path_start_tf = self.path_start_root_tf or root_tf
        self.gizmo_carry_tf = self._compose_transform(root_tf, self._transform_inverse(path_start_tf))

        self.post_start_right_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        self.post_start_pan_tf = self._current_pan_transform()
        self.post_right_local_pos, self.post_right_local_rot = self._relative_transform(
            self.post_start_pan_tf,
            self.post_start_right_tf,
        )
        start_pan_pos = wp.transform_get_translation(self.post_start_pan_tf)
        approach_z = max(float(start_pan_pos[2]), float(PAN_PLACE_CENTER[2]) + PAN_APPROACH_CLEARANCE_Z)
        self.post_approach_tf = wp.transform(
            wp.vec3(float(PAN_PLACE_CENTER[0]), float(PAN_PLACE_CENTER[1]), approach_z),
            self.scene_rotation,
        )
        self.post_place_tf = wp.transform(PAN_PLACE_CENTER, self.scene_rotation)
        self.post_place_right_tf = self._right_tf_from_pan(self.post_place_tf)
        self.post_clear_tf = wp.transform(
            wp.transform_get_translation(self.post_place_right_tf) + wp.vec3(0.0, -0.34, 0.10),
            wp.transform_get_rotation(self.post_place_right_tf),
        )
        self.post_prev_pan_tf = self.post_start_pan_tf
        self.left_hold_tf_for_gizmo = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        self.post_current_hand_alpha = 1.0

        print(
            "[newton] V5 post-place script started: move pan forward to lowered table, "
            "level it down, clear right hand."
        )

    def _step_post_place_script(self) -> None:
        elapsed = max(self.sim_time + self.frame_dt - self.post_start_time, 0.0)
        self.post_current_elapsed = elapsed
        right_tf, pan_tf, hand_alpha = self._sample_v5_post_place_script(elapsed)
        self.post_current_hand_alpha = hand_alpha
        self._solve_scripted_right_ik(right_tf, hand_alpha)
        self._set_pan_and_meat_pose(pan_tf)

    def _sample_post_place_script(self, t: float):
        return None

    def _sample_v5_post_place_script(self, elapsed: float) -> tuple[wp.transform, wp.transform, float]:
        t0 = self.post_forward_time
        t1 = t0 + self.post_lower_time
        t2 = t1 + self.post_release_time
        t3 = t2 + self.post_clear_time

        if elapsed <= t0:
            u = self._smoothstep(elapsed / max(self.post_forward_time, 1.0e-6))
            pan_tf = self._interpolate_transform(self.post_start_pan_tf, self.post_approach_tf, u)
            return self._right_tf_from_pan(pan_tf), pan_tf, 1.0
        if elapsed <= t1:
            u = self._smoothstep((elapsed - t0) / max(self.post_lower_time, 1.0e-6))
            pan_tf = self._interpolate_transform(self.post_approach_tf, self.post_place_tf, u)
            return self._right_tf_from_pan(pan_tf), pan_tf, 1.0
        if elapsed <= t2:
            u = self._smoothstep((elapsed - t1) / max(self.post_release_time, 1.0e-6))
            pan_tf = self.post_place_tf
            return self._right_tf_from_pan(pan_tf), pan_tf, 1.0 - u
        if elapsed <= t3:
            u = self._smoothstep((elapsed - t2) / max(self.post_clear_time, 1.0e-6))
            return (
                self._interpolate_transform(self.post_place_right_tf, self.post_clear_tf, u),
                self.post_place_tf,
                0.0,
            )

        return self.post_clear_tf, self.post_place_tf, 0.0

    def _right_tf_from_pan(self, pan_tf: wp.transform) -> wp.transform:
        return self._compose_transform(
            pan_tf,
            wp.transform(self.post_right_local_pos, self.post_right_local_rot),
        )

    def _pan_tf_from_right(self, right_tf: wp.transform) -> wp.transform:
        return self._compose_transform(
            right_tf,
            self._transform_inverse(wp.transform(self.post_right_local_pos, self.post_right_local_rot)),
        )

    def _solve_scripted_right_ik(self, world_right_tf: wp.transform, hand_alpha: float) -> None:
        inv_carry_tf = self._transform_inverse(self.gizmo_carry_tf)
        self.right_tf = self._compose_transform(inv_carry_tf, world_right_tf)
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
            inputs=[
                self.right_hand_q_indices,
                self.right_hand_open,
                self.right_hand_grasp,
                float(hand_alpha),
            ],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )

        wp.copy(self.state_0.joint_q, self.frame_joint_q_end)
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._reapply_carried_robot_pose()

    def _reapply_carried_robot_pose(self) -> None:
        wp.launch(
            v4.apply_body_transform_kernel,
            dim=self.rigid_robot_body_indices.shape[0],
            inputs=[self.state_0.body_q, self.rigid_robot_body_indices, self.gizmo_carry_tf],
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

    def _set_pan_and_meat_pose(self, pan_tf: wp.transform) -> None:
        pan_pos = wp.transform_get_translation(pan_tf)
        delta_tf = self._compose_transform(pan_tf, self._transform_inverse(self.post_prev_pan_tf))
        wp.launch(
            v4.set_body_transform_kernel,
            dim=1,
            inputs=[self.state_0.body_q, self.pan_body, pan_tf],
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
                delta_tf,
                1.0 / self.frame_dt,
            ],
            device=self.model.device,
        )
        self.post_prev_pan_tf = pan_tf

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
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

    def _report_post_pose(self) -> None:
        if self.pose_print_interval > 0.0 and self.sim_time - self.last_pose_print_time < self.pose_print_interval:
            return
        self.last_pose_print_time = self.sim_time

        root_tf = self._current_robot_root_transform()
        root_pos = self._vec3_to_np(wp.transform_get_translation(root_tf))
        pan_pos = self._vec3_to_np(wp.transform_get_translation(self._current_pan_transform()))
        right_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        right_pos = self._vec3_to_np(wp.transform_get_translation(right_tf))
        right_rot = self._quat_to_np(wp.transform_get_rotation(right_tf))
        print(
            "[waic kitchen V5] "
            f"root_pos=({root_pos[0]:.6f}, {root_pos[1]:.6f}, {root_pos[2]:.6f}) "
            f"pan_pos=({pan_pos[0]:.6f}, {pan_pos[1]:.6f}, {pan_pos[2]:.6f}) "
            f"right_actual_pos=({right_pos[0]:.6f}, {right_pos[1]:.6f}, {right_pos[2]:.6f}) "
            f"right_actual_quat_xyzw=({right_rot[0]:.6f}, {right_rot[1]:.6f}, {right_rot[2]:.6f}, {right_rot[3]:.6f}) "
            f"hand_alpha={self.post_current_hand_alpha:.3f}"
        )

    @staticmethod
    def create_parser():
        parser = path.Example.create_parser()
        parser.add_argument("--post-forward-time", type=float, default=3.0)
        parser.add_argument("--post-lower-time", type=float, default=2.0)
        parser.add_argument("--post-release-time", type=float, default=0.7)
        parser.add_argument("--post-clear-time", type=float, default=1.5)
        parser.add_argument(
            "--v5-meat-pan-gap",
            type=float,
            default=V5_MEAT_PAN_GAP,
            help="Initial vertical gap between the pan collision top and the MPM meat particles in V5.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

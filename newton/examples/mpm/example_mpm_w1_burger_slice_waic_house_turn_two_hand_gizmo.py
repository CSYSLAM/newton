# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# WAIC house W1 burger slice right turn + two-hand gizmo
#
# Runs the full-house V4 burger-slicing scene through the scripted knife work,
# pan grasp, and pan lift. The carried W1/pan/meat group then turns right by
# 90 degrees. After that the body is forced upright and both TCP IK gizmos are
# exposed so a two-hand pan-carry pose can be tuned interactively.
#
# Command:
#   python -m newton.examples mpm_w1_burger_slice_waic_house_turn_two_hand_gizmo
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
RIGHT_TURN_TIME = 3.0


class Example(house.Example):
    """Turn right once, straighten the body, then expose left/right TCP gizmos."""

    BODY_JOINTS = ("BUTTOCK", "WAIST")

    def __init__(self, viewer, args):
        self.right_turn_time = float(args.right_turn_time)
        self.right_turn_degrees = float(args.right_turn_degrees)
        self.straight_buttock_q = float(args.straight_buttock_q)
        self.straight_waist_q = float(args.straight_waist_q)
        self.pose_print_interval = float(args.pose_print_interval)
        self.edit_hand_alpha = float(args.edit_hand_alpha)
        self.last_pose_print_time = -1.0e9
        self.edit_mode = False
        self.turn_start_root_tf: wp.transform | None = None
        self.gizmo_carry_tf = wp.transform_identity()
        super().__init__(viewer, args)

        self.edit_start_time = self.rigid_carry_start_time + self.right_turn_time
        self.left_gizmo_tf = self.left_tf
        self.right_gizmo_tf = self.pan_handle_lift_tf
        self.left_hold_local_tf = self.left_tf

    def _global_carry_transform(self, carry_alpha: float) -> wp.transform:
        # V4 calls this during __init__. Return a harmless right-turn transform;
        # the actual pivoted transform is built in _rigid_carry_transform after
        # the current body state exists.
        yaw = math.radians(self.right_turn_degrees) * float(carry_alpha)
        yaw_rot = wp.quat(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))
        return wp.transform(wp.vec3(0.0, 0.0, 0.0), yaw_rot)

    def _sample_post_place_script(self, t: float):
        return None

    def _rigid_carry_transform(
        self, query_time: float
    ) -> tuple[wp.transform, wp.transform, wp.transform, float, float]:
        turn_t = max(float(query_time) - self.rigid_carry_start_time, 0.0)
        turn_alpha = self._smoothstep(turn_t / max(self.right_turn_time, 1.0e-6))
        if turn_alpha <= 0.0:
            return (
                wp.transform_identity(),
                wp.transform_identity(),
                wp.transform_identity(),
                0.0,
                0.0,
            )

        if self.turn_start_root_tf is None:
            self.turn_start_root_tf = self._current_robot_root_transform()

        target_root_tf = self._sample_turn_root_tf(turn_alpha)
        carry_tf = self._compose_transform(target_root_tf, self._transform_inverse(self.turn_start_root_tf))
        return carry_tf, carry_tf, carry_tf, min(turn_alpha, 1.0), 0.0

    def _sample_turn_root_tf(self, turn_alpha: float) -> wp.transform:
        start_tf = self.turn_start_root_tf or self._current_robot_root_transform()
        start_pos = wp.transform_get_translation(start_tf)
        start_rot = wp.transform_get_rotation(start_tf)
        yaw = math.radians(self.right_turn_degrees) * float(np.clip(turn_alpha, 0.0, 1.0))
        yaw_rot = wp.quat(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))
        return wp.transform(start_pos, self._quat_multiply(yaw_rot, start_rot))

    def step(self) -> None:
        if not self.edit_mode and self.sim_time < self.edit_start_time:
            self.simulate()
            return

        if not self.edit_mode:
            self._enter_edit_mode()

        self._solve_two_hand_gizmo_ik()
        self.sim_time += self.frame_dt
        self.frame_index += 1
        self._report_gizmo_pose()

    def _enter_edit_mode(self) -> None:
        self.edit_mode = True
        root_tf = self._current_robot_root_transform()
        turn_start_tf = self.turn_start_root_tf or root_tf
        self.gizmo_carry_tf = self._compose_transform(root_tf, self._transform_inverse(turn_start_tf))

        self._straighten_body_at_current_carry_pose()
        self.left_gizmo_tf = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        self.right_gizmo_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        self.left_tf = self.left_gizmo_tf
        self.right_tf = self.right_gizmo_tf

        print(
            "[newton] Entered two-hand pan pose gizmo mode after right turn. "
            "Body is held upright; drag dexforce_left_pan_tcp_target and dexforce_right_pan_tcp_target."
        )
        self._report_gizmo_pose(force=True)

    def _straighten_body_at_current_carry_pose(self) -> None:
        q = self.state_0.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        for joint_name, value in (
            ("BUTTOCK", self.straight_buttock_q),
            ("WAIST", self.straight_waist_q),
        ):
            try:
                joint_idx = self._joint_index(joint_name)
            except StopIteration:
                continue
            q[int(q_start[joint_idx])] = float(value)

        self.state_0.joint_q = wp.array(q, dtype=wp.float32, device=self.model.device)
        self.state_0.joint_qd.zero_()
        self.ik_joint_q = wp.array(self.state_0.joint_q, shape=(1, self.model.joint_coord_count))
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._reapply_gizmo_carry_pose()

    def _solve_two_hand_gizmo_ik(self) -> None:
        inv_carry_tf = self._transform_inverse(self.gizmo_carry_tf)
        self.left_tf = self._compose_transform(inv_carry_tf, self.left_gizmo_tf)
        self.right_tf = self._compose_transform(inv_carry_tf, self.right_gizmo_tf)

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

        q = self.frame_joint_q_end.numpy()
        q_start = self.model.joint_q_start.numpy()
        q[int(q_start[self._joint_index("BUTTOCK")])] = self.straight_buttock_q
        q[int(q_start[self._joint_index("WAIST")])] = self.straight_waist_q
        self.frame_joint_q_end = wp.array(q, dtype=wp.float32, device=self.model.device)

        wp.launch(
            v4.set_indexed_joint_q_kernel,
            dim=self.right_hand_q_indices.shape[0],
            inputs=[self.right_hand_q_indices, self.right_hand_open, self.right_hand_grasp, self.edit_hand_alpha],
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
        if self.edit_mode and hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo(
                "dexforce_left_pan_tcp_target",
                self.left_gizmo_tf,
                snap_to=self._current_tcp_transform(self.left_ee_index, self.left_ee_offset),
            )
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

    def render_ui(self, imgui):
        super().render_ui(imgui)
        _changed, self.edit_hand_alpha = imgui.slider_float("Right pan hand close", self.edit_hand_alpha, 0.0, 1.0)

    def _report_gizmo_pose(self, force: bool = False) -> None:
        if not self.edit_mode:
            return
        if (
            not force
            and self.pose_print_interval > 0.0
            and self.sim_time - self.last_pose_print_time < self.pose_print_interval
        ):
            return
        self.last_pose_print_time = self.sim_time

        root_tf = self._current_robot_root_transform()
        root_pos = self._vec3_to_np(wp.transform_get_translation(root_tf))
        root_rot = self._quat_to_np(wp.transform_get_rotation(root_tf))
        left_pos, left_rot, left_actual_pos, left_actual_rot, left_pos_err, left_rot_err = self._target_actual_report(
            self.left_gizmo_tf,
            self.left_ee_index,
            self.left_ee_offset,
        )
        right_pos, right_rot, right_actual_pos, right_actual_rot, right_pos_err, right_rot_err = self._target_actual_report(
            self.right_gizmo_tf,
            self.right_ee_index,
            self.right_ee_offset,
        )
        print(
            "[two-hand pan pose] "
            f"root_pos=({root_pos[0]:.6f}, {root_pos[1]:.6f}, {root_pos[2]:.6f}) "
            f"root_quat_xyzw=({root_rot[0]:.6f}, {root_rot[1]:.6f}, {root_rot[2]:.6f}, {root_rot[3]:.6f}) "
            f"left_target_pos=({left_pos[0]:.6f}, {left_pos[1]:.6f}, {left_pos[2]:.6f}) "
            f"left_target_quat_xyzw=({left_rot[0]:.6f}, {left_rot[1]:.6f}, {left_rot[2]:.6f}, {left_rot[3]:.6f}) "
            f"left_actual_pos=({left_actual_pos[0]:.6f}, {left_actual_pos[1]:.6f}, {left_actual_pos[2]:.6f}) "
            f"left_actual_quat_xyzw=({left_actual_rot[0]:.6f}, {left_actual_rot[1]:.6f}, {left_actual_rot[2]:.6f}, {left_actual_rot[3]:.6f}) "
            f"left_err=({left_pos_err:.4f} m, {left_rot_err:.2f} deg) "
            f"right_target_pos=({right_pos[0]:.6f}, {right_pos[1]:.6f}, {right_pos[2]:.6f}) "
            f"right_target_quat_xyzw=({right_rot[0]:.6f}, {right_rot[1]:.6f}, {right_rot[2]:.6f}, {right_rot[3]:.6f}) "
            f"right_actual_pos=({right_actual_pos[0]:.6f}, {right_actual_pos[1]:.6f}, {right_actual_pos[2]:.6f}) "
            f"right_actual_quat_xyzw=({right_actual_rot[0]:.6f}, {right_actual_rot[1]:.6f}, {right_actual_rot[2]:.6f}, {right_actual_rot[3]:.6f}) "
            f"right_err=({right_pos_err:.4f} m, {right_rot_err:.2f} deg) "
            f"right_hand_alpha={self.edit_hand_alpha:.3f}"
        )

    def _target_actual_report(self, target_tf: wp.transform, body_index: int, offset: wp.vec3):
        target_pos = self._vec3_to_np(wp.transform_get_translation(target_tf))
        target_rot = self._quat_to_np(wp.transform_get_rotation(target_tf))
        actual_tf = self._current_tcp_transform(body_index, offset)
        actual_pos = self._vec3_to_np(wp.transform_get_translation(actual_tf))
        actual_rot = self._quat_to_np(wp.transform_get_rotation(actual_tf))
        return (
            target_pos,
            target_rot,
            actual_pos,
            actual_rot,
            float(np.linalg.norm(target_pos - actual_pos)),
            self._quat_angle_error_deg(target_rot, actual_rot),
        )

    def _current_robot_root_transform(self) -> wp.transform:
        body_q = self.state_0.body_q.numpy()
        return wp.transform(*body_q[int(self.rigid_robot_body_indices.numpy()[0])])

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
        parser.add_argument("--right-turn-time", type=float, default=RIGHT_TURN_TIME)
        parser.add_argument("--right-turn-degrees", type=float, default=RIGHT_TURN_DEGREES)
        parser.add_argument("--straight-buttock-q", type=float, default=0.0)
        parser.add_argument("--straight-waist-q", type=float, default=0.0)
        parser.add_argument("--edit-hand-alpha", type=float, default=1.0)
        parser.add_argument("--pose-print-interval", type=float, default=0.5)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

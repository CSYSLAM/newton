# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# WAIC kitchen W1 pan-grasp IK gizmo
#
# Runs the kitchen burger-slicing setup only until the knife is returned and
# both hands have lifted near the frying pan. From that point on, MPM motion
# is frozen and both wrist TCPs are controlled by viewer gizmos so a two-hand
# pan-carry pose can be tuned interactively.
#
# Command:
#   python -m newton.examples mpm_w1_burger_slice_waic_kitchen_ik_gizmo
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples

from newton.examples.mpm import example_mpm_w1_burger_slice_waic_kitchen as base


class Example(base.Example):
    """Interactive left/right TCP gizmos for tuning a two-hand pan carry pose."""

    def __init__(self, viewer, args):
        self.edit_start_time = float(args.edit_start_time)
        self.edit_hand_alpha = float(args.edit_hand_alpha)
        self.pose_print_interval = float(args.pose_print_interval)
        self.last_pose_print_time = -1.0e9
        self.edit_mode = False
        super().__init__(viewer, args)

        # Prepare the edit-mode gizmos, but do not push them into IK yet. The
        # scripted slicing phase should stay identical to the base demo: left
        # hand parked, right hand cuts and puts the knife down. The gizmos take
        # over only after _enter_edit_mode().
        self.right_gizmo_tf = self.pan_handle_approach_tf
        self.left_gizmo_tf = self._default_left_pan_gizmo_tf()
        self.pan_grasp_joint_names = list(self.RIGHT_HAND_JOINTS)
        self.pan_grasp_values_host = self.right_hand_pan_grasp.numpy().astype(np.float32)
        self.pan_grasp_upper_host = self._right_hand_upper_limits()

    def step(self) -> None:
        if not self.edit_mode and self.sim_time < self.edit_start_time:
            self.simulate()
            return

        if not self.edit_mode:
            self._enter_edit_mode()

        self._solve_gizmo_ik()
        self.sim_time += self.frame_dt
        self.frame_index += 1
        self._report_gizmo_pose()

    def _enter_edit_mode(self) -> None:
        self.edit_mode = True
        self.right_gizmo_tf = self.pan_handle_approach_tf
        self.left_gizmo_tf = self._default_left_pan_gizmo_tf()
        self.right_tf = self.right_gizmo_tf
        self.left_tf = self.left_gizmo_tf
        self.grip_mode = 1.0
        self.meat_carry_alpha = 0.0
        print(
            "[newton] Entered two-hand pan IK gizmo mode. "
            "Drag dexforce_left_pan_tcp_target and dexforce_right_pan_tcp_target to tune the carry pose."
        )

    def _solve_gizmo_ik(self) -> None:
        self.right_tf = self.right_gizmo_tf
        self.left_tf = self.left_gizmo_tf
        wp.copy(self.frame_joint_q_start, self.state_0.joint_q)

        self.right_pos_obj.set_target_position(0, wp.transform_get_translation(self.right_tf))
        self.right_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.right_tf)))
        self.left_pos_obj.set_target_position(0, wp.transform_get_translation(self.left_tf))
        self.left_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.left_tf)))

        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            base.lock_joint_q_kernel,
            dim=self.locked_q_indices.shape[0],
            inputs=[self.ik_joint_q, self.locked_q_indices, self.locked_q_values],
            device=self.model.device,
        )
        wp.launch(
            base.copy_ik_to_joint_q_kernel,
            dim=self.model.joint_coord_count,
            inputs=[self.ik_joint_q],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )
        wp.launch(
            base.set_indexed_joint_q_kernel,
            dim=self.right_hand_q_indices.shape[0],
            inputs=[
                self.right_hand_q_indices,
                self.right_hand_open,
                self.right_hand_knife_grasp,
                self.right_hand_pan_grasp,
                self.edit_hand_alpha,
                1.0,
            ],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )

        wp.copy(self.state_0.joint_q, self.frame_joint_q_end)
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Keep the knife and pan proxies where they are for alignment reference.
        self._update_knife_transform(0.0)
        self._update_pan_transform(self.pan_initial_tf)

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

    def render_ui(self, imgui):
        super().render_ui(imgui)
        _changed, self.edit_hand_alpha = imgui.slider_float("Right hand close", self.edit_hand_alpha, 0.0, 1.0)
        any_changed = False
        for i, joint_name in enumerate(self.pan_grasp_joint_names):
            label = joint_name.replace("RIGHT_", "").replace("_", " ")
            changed, value = imgui.slider_float(
                label,
                float(self.pan_grasp_values_host[i]),
                0.0,
                float(self.pan_grasp_upper_host[i]),
                "%.4f",
            )
            if changed:
                self.pan_grasp_values_host[i] = value
                any_changed = True
        if any_changed:
            self.right_hand_pan_grasp = wp.array(
                self.pan_grasp_values_host,
                dtype=wp.float32,
                device=self.model.device,
            )

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

        left_target_pos = self._vec3_to_np(wp.transform_get_translation(self.left_gizmo_tf))
        left_target_rot = self._quat_to_np(wp.transform_get_rotation(self.left_gizmo_tf))
        right_target_pos = self._vec3_to_np(wp.transform_get_translation(self.right_gizmo_tf))
        right_target_rot = self._quat_to_np(wp.transform_get_rotation(self.right_gizmo_tf))

        left_actual_tf = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        right_actual_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        left_actual_pos = self._vec3_to_np(wp.transform_get_translation(left_actual_tf))
        left_actual_rot = self._quat_to_np(wp.transform_get_rotation(left_actual_tf))
        right_actual_pos = self._vec3_to_np(wp.transform_get_translation(right_actual_tf))
        right_actual_rot = self._quat_to_np(wp.transform_get_rotation(right_actual_tf))

        left_pos_err = float(np.linalg.norm(left_target_pos - left_actual_pos))
        right_pos_err = float(np.linalg.norm(right_target_pos - right_actual_pos))
        left_rot_err = self._quat_angle_error_deg(left_target_rot, left_actual_rot)
        right_rot_err = self._quat_angle_error_deg(right_target_rot, right_actual_rot)

        print(
            "[two-hand pan IK] "
            f"left_target_pos=({left_target_pos[0]:.6f}, {left_target_pos[1]:.6f}, {left_target_pos[2]:.6f}) "
            f"left_target_quat_xyzw=({left_target_rot[0]:.6f}, {left_target_rot[1]:.6f}, {left_target_rot[2]:.6f}, {left_target_rot[3]:.6f}) "
            f"left_actual_pos=({left_actual_pos[0]:.6f}, {left_actual_pos[1]:.6f}, {left_actual_pos[2]:.6f}) "
            f"left_actual_quat_xyzw=({left_actual_rot[0]:.6f}, {left_actual_rot[1]:.6f}, {left_actual_rot[2]:.6f}, {left_actual_rot[3]:.6f}) "
            f"left_err=({left_pos_err:.4f} m, {left_rot_err:.2f} deg) "
            f"right_target_pos=({right_target_pos[0]:.6f}, {right_target_pos[1]:.6f}, {right_target_pos[2]:.6f}) "
            f"right_target_quat_xyzw=({right_target_rot[0]:.6f}, {right_target_rot[1]:.6f}, {right_target_rot[2]:.6f}, {right_target_rot[3]:.6f}) "
            f"right_actual_pos=({right_actual_pos[0]:.6f}, {right_actual_pos[1]:.6f}, {right_actual_pos[2]:.6f}) "
            f"right_actual_quat_xyzw=({right_actual_rot[0]:.6f}, {right_actual_rot[1]:.6f}, {right_actual_rot[2]:.6f}, {right_actual_rot[3]:.6f}) "
            f"right_err=({right_pos_err:.4f} m, {right_rot_err:.2f} deg) "
            f"right_hand_alpha={self.edit_hand_alpha:.3f} "
            f"right_pan_grasp_q={self._format_pan_grasp_values()}"
        )

    def _default_left_pan_gizmo_tf(self) -> wp.transform:
        pos = wp.vec3(
            float(self.pan_center[0]) + 0.08,
            float(self.pan_center[1]) + 0.08,
            self.pan_top_z + 0.17,
        )
        return wp.transform(pos, wp.quat(0.512743, -0.312141, -0.393223, 0.696447))

    def _right_hand_upper_limits(self) -> np.ndarray:
        upper = self.model.joint_limit_upper.numpy()
        values = []
        for joint_name in self.RIGHT_HAND_JOINTS:
            values.append(float(upper[self._joint_index(joint_name)]))
        return np.asarray(values, dtype=np.float32)

    def _format_pan_grasp_values(self) -> str:
        pairs = []
        for joint_name, value in zip(self.pan_grasp_joint_names, self.pan_grasp_values_host):
            pairs.append(f"{joint_name}={float(value):.5f}")
        return "{" + ", ".join(pairs) + "}"

    @staticmethod
    def _quat_angle_error_deg(qa: np.ndarray, qb: np.ndarray) -> float:
        qa = qa / max(float(np.linalg.norm(qa)), 1.0e-9)
        qb = qb / max(float(np.linalg.norm(qb)), 1.0e-9)
        dot = abs(float(np.dot(qa, qb)))
        dot = float(np.clip(dot, -1.0, 1.0))
        return float(np.degrees(2.0 * np.arccos(dot)))

    @staticmethod
    def create_parser():
        parser = base.Example.create_parser()
        parser.set_defaults(num_frames=100000, paused=False)
        parser.add_argument(
            "--edit-start-time",
            type=float,
            default=11.9,
            help="Script time at which to freeze MPM and enable the pan-grasp IK gizmo.",
        )
        parser.add_argument(
            "--edit-hand-alpha",
            type=float,
            default=0.0,
            help="Initial right-hand closure shown while tuning the TCPs. Keep 0.0 for open hands.",
        )
        parser.add_argument(
            "--pose-print-interval",
            type=float,
            default=0.5,
            help="Seconds between console prints of the target/actual TCP pose in gizmo mode.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

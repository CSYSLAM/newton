# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# WAIC house W1 burger slice carry/root gizmo tuner
#
# Runs the full-house V4 burger-slicing scene through the scripted knife work,
# pan grasp, pan lift, and an initial right turn. Then MPM is frozen and the
# viewer exposes one root-position gizmo so the carried robot/pan/meat group
# can be translated interactively along a fixed orientation.
#
# Command:
#   python -m newton.examples mpm_w1_burger_slice_waic_house_root_gizmo
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
    """Interactive root-position gizmo after the pan is lifted and W1 turns right."""

    def __init__(self, viewer, args):
        self.edit_start_time_arg = float(args.edit_start_time)
        self.pose_print_interval = float(args.pose_print_interval)
        self.root_position_only = bool(args.root_position_only)
        self.right_turn_time = float(args.right_turn_time)
        self.right_turn_degrees = float(args.right_turn_degrees)
        self.last_pose_print_time = -1.0e9
        self.edit_mode = False
        super().__init__(viewer, args)

        if self.edit_start_time_arg < 0.0:
            self.edit_start_time = self.rigid_carry_start_time + self.right_turn_time
        else:
            self.edit_start_time = self.edit_start_time_arg

        self.root_gizmo_tf = wp.transform(self.waic_robot_base_pos, self.scene_rotation)
        self.root_edit_start_tf = self.root_gizmo_tf
        self.root_edit_last_tf = wp.transform_identity()
        self.edit_reference_captured = False

    def _global_carry_transform(self, carry_alpha: float) -> wp.transform:
        turn_alpha = float(np.clip(carry_alpha, 0.0, 1.0))
        yaw = math.radians(self.right_turn_degrees) * turn_alpha
        yaw_rot = wp.quat(0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))
        pivot = self.waic_robot_base_pos
        pos = pivot - self._quat_rotate_vec3(yaw_rot, pivot)
        return wp.transform(pos, yaw_rot)

    def _rigid_carry_transform(
        self, query_time: float
    ) -> tuple[wp.transform, wp.transform, wp.transform, float, float]:
        carry_t = max(float(query_time) - self.rigid_carry_start_time, 0.0)
        carry_alpha = self._smoothstep(carry_t / max(self.right_turn_time, 1.0e-6))
        if carry_alpha <= 0.0:
            return (
                wp.transform_identity(),
                wp.transform_identity(),
                wp.transform_identity(),
                0.0,
                0.0,
            )

        carry_tf = self._global_carry_transform(carry_alpha)
        return carry_tf, carry_tf, carry_tf, carry_alpha, 0.0

    def step(self) -> None:
        if not self.edit_mode and self.sim_time < self.edit_start_time:
            self.simulate()
            return

        if not self.edit_mode:
            self._enter_edit_mode()

        self._apply_root_gizmo_delta()
        self.sim_time += self.frame_dt
        self.frame_index += 1
        self._report_root_pose()

    def _enter_edit_mode(self) -> None:
        self.edit_mode = True
        self._capture_edit_reference()
        self.root_gizmo_tf = self._current_root_transform()
        self.root_edit_start_tf = self.root_gizmo_tf
        self.root_edit_last_tf = wp.transform_identity()
        print(
            "[newton] Entered WAIC root gizmo mode. "
            "Drag dexforce_root_position to translate W1, the pan, and the meat as one rigid group."
        )
        self._report_root_pose(force=True)

    def _capture_edit_reference(self) -> None:
        if self.edit_reference_captured:
            return
        wp.copy(self.rigid_carry_body_q_start, self.state_0.body_q)
        wp.copy(self.rigid_carry_shape_transform_start, self.model.shape_transform)
        self.rigid_carry_prev_object_tf = wp.transform_identity()
        self.rigid_carry_prev_alpha = 0.0
        self.rigid_place_prev_alpha = 0.0
        self.rigid_carry_initialized = True
        self.edit_reference_captured = True

    def _apply_root_gizmo_delta(self) -> None:
        delta_pos = wp.transform_get_translation(self.root_gizmo_tf) - wp.transform_get_translation(
            self.root_edit_start_tf
        )
        delta_tf = wp.transform(delta_pos, wp.quat_identity())

        if self._transform_translation_norm(delta_tf) <= 1.0e-8:
            return

        self._apply_edit_delta_to_carried_scene(delta_tf)
        self.root_edit_last_tf = delta_tf
        self.root_edit_start_tf = wp.transform(
            wp.transform_get_translation(self.root_gizmo_tf),
            wp.transform_get_rotation(self.root_edit_start_tf),
        )

    def _apply_edit_delta_to_carried_scene(self, delta_tf: wp.transform) -> None:
        wp.launch(
            v4.apply_body_transform_kernel,
            dim=self.rigid_robot_body_indices.shape[0],
            inputs=[self.state_0.body_q, self.rigid_robot_body_indices, delta_tf],
            device=self.model.device,
        )
        wp.launch(
            v4.apply_body_transform_kernel,
            dim=self.rigid_pan_body_indices.shape[0],
            inputs=[self.state_0.body_q, self.rigid_pan_body_indices, delta_tf],
            device=self.model.device,
        )
        if self.rigid_carry_shape_indices.shape[0] > 0:
            wp.launch(
                v4.set_transformed_shape_transforms_kernel,
                dim=self.rigid_carry_shape_indices.shape[0],
                inputs=[self.model.shape_transform, self.rigid_carry_shape_indices, delta_tf],
                outputs=[self.model.shape_transform],
                device=self.model.device,
            )
        wp.launch(
            v4.apply_particle_delta_transform_kernel,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.state_0.particle_qd, delta_tf, 1.0 / self.frame_dt],
            device=self.model.device,
        )

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        if self.edit_mode and hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo(
                "dexforce_root_position",
                self.root_gizmo_tf,
                rotate=(),
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

    def _current_pan_transform(self) -> wp.transform:
        body_q = self.state_0.body_q.numpy()
        return wp.transform(*body_q[self.pan_body])

    def _current_root_transform(self) -> wp.transform:
        body_q = self.state_0.body_q.numpy()
        return wp.transform(*body_q[int(self.rigid_robot_body_indices.numpy()[0])])

    def _report_root_pose(self, force: bool = False) -> None:
        if not self.edit_mode:
            return
        if (
            not force
            and self.pose_print_interval > 0.0
            and self.sim_time - self.last_pose_print_time < self.pose_print_interval
        ):
            return
        self.last_pose_print_time = self.sim_time

        root_pos = self._vec3_to_np(wp.transform_get_translation(self.root_gizmo_tf))
        root_rot = self._quat_to_np(wp.transform_get_rotation(self._current_root_transform()))
        pan_pos = self._vec3_to_np(wp.transform_get_translation(self._current_pan_transform()))
        delta = self._vec3_to_np(wp.transform_get_translation(self.root_edit_last_tf))

        print(
            "[waic root position] "
            f"root_pos=({root_pos[0]:.6f}, {root_pos[1]:.6f}, {root_pos[2]:.6f}) "
            f"root_quat_xyzw=({root_rot[0]:.6f}, {root_rot[1]:.6f}, {root_rot[2]:.6f}, {root_rot[3]:.6f}) "
            f"pan_pos=({pan_pos[0]:.6f}, {pan_pos[1]:.6f}, {pan_pos[2]:.6f}) "
            f"delta=({delta[0]:.6f}, {delta[1]:.6f}, {delta[2]:.6f})"
        )

    @staticmethod
    def _offset_transform_translation(tf: wp.transform, offset: wp.vec3) -> wp.transform:
        return wp.transform(wp.transform_get_translation(tf) + offset, wp.transform_get_rotation(tf))

    @staticmethod
    def _transform_translation_norm(tf: wp.transform) -> float:
        pos = wp.transform_get_translation(tf)
        return math.sqrt(float(pos[0]) ** 2 + float(pos[1]) ** 2 + float(pos[2]) ** 2)

    @staticmethod
    def create_parser():
        parser = house.Example.create_parser()
        parser.set_defaults(num_frames=100000, paused=False)
        parser.add_argument(
            "--edit-start-time",
            type=float,
            default=-1.0,
            help="Time to enter gizmo mode. Negative means after pan lift plus the scripted right turn.",
        )
        parser.add_argument(
            "--right-turn-degrees",
            type=float,
            default=RIGHT_TURN_DEGREES,
            help="Initial scripted W1/pan/meat turn after pan lift. Negative is right turn in this scene.",
        )
        parser.add_argument(
            "--right-turn-time",
            type=float,
            default=RIGHT_TURN_TIME,
            help="Duration of the initial scripted right turn before gizmo mode.",
        )
        parser.add_argument(
            "--pose-print-interval",
            type=float,
            default=0.0,
            help="Seconds between console prints of root/pan position. 0 prints every frame.",
        )
        parser.add_argument(
            "--root-position-only",
            action="store_true",
            default=True,
            help="Only use root gizmo translation; keep the right-turn orientation fixed.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

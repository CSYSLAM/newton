# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MPM W1 Burger Slice WAIC Kitchen V6.4 Pan Grasp IK
#
# Runs the V6.4 WAIC kitchen burger-slicing sequence until the right hand has
# closed on the pan handle and paused before lifting. The script then freezes
# the scene and exposes a right TCP gizmo for tuning the pan-handle grasp,
# lift and carry posture. The pan (and the meat inside it) can optionally
# follow the hand so the full pot-holding pose is visible while tuning.
#
# Command: python -m newton.examples mpm_w1_burger_slice_waic_kitchen_v6_4_pan_grasp_ik
#
###########################################################################

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

# The v6.4 module filename contains a dot, so it cannot be imported with a
# regular ``import`` statement. Load it via importlib.spec_from_file_location
# (same pattern as example_mpm_w1_burger_slice_waic_house.py for v5.1).
_V64_PATH = Path(__file__).with_name("example_mpm_w1_burger_slice_waic_kitchen_v6.4.py")
_V64_SPEC = importlib.util.spec_from_file_location(
    "newton.examples.mpm.example_mpm_w1_burger_slice_waic_kitchen_v6_4",
    _V64_PATH,
)
if _V64_SPEC is None or _V64_SPEC.loader is None:
    raise ImportError(f"Failed to load kitchen v6.4 example from {_V64_PATH}")
kitchen_v64 = importlib.util.module_from_spec(_V64_SPEC)
sys.modules[_V64_SPEC.name] = kitchen_v64
_V64_SPEC.loader.exec_module(kitchen_v64)

# Reuse the kernels and constants defined in the v6.4 module.
compute_deformation_colors = kitchen_v64.compute_deformation_colors
copy_ik_to_joint_q_kernel = kitchen_v64.copy_ik_to_joint_q_kernel
interpolate_joint_positions_kernel = kitchen_v64.interpolate_joint_positions_kernel
lock_joint_q_kernel = kitchen_v64.lock_joint_q_kernel
set_indexed_joint_q_kernel = kitchen_v64.set_indexed_joint_q_kernel
update_joint_velocity_kernel = kitchen_v64.update_joint_velocity_kernel
set_body_transform_kernel = kitchen_v64.set_body_transform_kernel
set_body_velocity_kernel = kitchen_v64.set_body_velocity_kernel
apply_particle_delta_transform_kernel = kitchen_v64.apply_particle_delta_transform_kernel

CUT_COLOR = kitchen_v64.CUT_COLOR
MEAT_COLOR = kitchen_v64.MEAT_COLOR
PAN_HAND_GRASP_ALPHA = kitchen_v64.PAN_HAND_GRASP_ALPHA


class Example(kitchen_v64.Example):
    """Right-hand IK tuner for the V6.4 WAIC kitchen pan-handle grasp."""

    def __init__(self, viewer, args):
        self.print_interval = float(args.print_interval)
        self.last_print_time = -1.0
        self.ik_control_enabled = False
        self.pan_follows_hand = bool(args.pan_follows_hand)
        self._ik_pan_prev_tf = wp.transform_identity()

        super().__init__(viewer, args)

        self.script_duration = sum(segment[0] for segment in self.motion_segments)
        print(
            "[newton] V6.4 script will stop after the pan handle is grasped "
            f"at t={self.script_duration:.3f}s, then the right TCP gizmo takes over."
        )

    def _build_motion_segments(self):
        # Stop after the pan handle grasp closes and the pre-lift hold finishes.
        # The lift / carry / place phases are intentionally omitted so the user
        # can manually tune the lift and pot-holding posture via the gizmo.
        return super()._build_motion_segments()[:15]

    # ------------------------------------------------------------------ IK
    def _enter_ik_control(self) -> None:
        if self.ik_control_enabled:
            return

        self.ik_control_enabled = True
        self.left_tf = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        self.right_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        self.ik_joint_q = wp.array(
            self.state_0.joint_q.numpy(),
            dtype=wp.float32,
            shape=(1, self.model.joint_coord_count),
            device=self.model.device,
        )
        self._ik_pan_prev_tf = self._current_pan_transform()
        self._push_targets_from_gizmo()
        self._report_pose(force=True)
        print(
            "[newton] Right-hand IK gizmo enabled. MPM simulation is frozen after pan grasp. "
            "Drag the gizmo to tune the lift / carry posture."
        )

    def _push_targets_from_gizmo(self) -> None:
        self.left_pos_obj.set_target_position(0, wp.transform_get_translation(self.left_tf))
        self.left_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.left_tf)))
        self.right_pos_obj.set_target_position(0, wp.transform_get_translation(self.right_tf))
        self.right_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.right_tf)))

    def _prepare_frame_targets(self) -> None:
        wp.copy(self.frame_joint_q_start, self.state_0.joint_q)

        next_time = self.sim_time + self.frame_dt
        if not self.ik_control_enabled and next_time <= self.script_duration + 1.0e-7:
            self.right_tf, grasp_alpha, self.knife_alpha, self.pan_alpha = self._sample_script(next_time)
            self.right_pos_obj.set_target_position(0, wp.transform_get_translation(self.right_tf))
            self.right_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.right_tf)))
            self.left_pos_obj.set_target_position(0, wp.transform_get_translation(self.left_tf))
            self.left_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(self.left_tf)))
        else:
            self._enter_ik_control()
            grasp_alpha = PAN_HAND_GRASP_ALPHA
            self.knife_alpha = 0.0
            self.pan_alpha = 1.0 if self.pan_follows_hand else 0.0
            self._push_targets_from_gizmo()

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

    def _update_pan_and_meat_from_ik(self) -> None:
        """Move the pan body and the meat particles to follow the right-hand gizmo target.

        The pan transform is recovered from the right-hand TCP target via the
        cached handle-to-pan local transform. The delta from the previous pan
        pose is applied to the meat particles so they ride along with the pan.
        """
        pan_tf = self._pan_transform_from_handle(self.right_tf)
        delta_tf = self._compose_transform(pan_tf, self._transform_inverse(self._ik_pan_prev_tf))

        wp.launch(
            set_body_transform_kernel,
            dim=1,
            inputs=[self.state_0.body_q, self.pan_body, pan_tf],
            device=self.model.device,
        )
        wp.launch(
            set_body_velocity_kernel,
            dim=1,
            inputs=[self.state_0.body_qd, self.pan_body, wp.vec3(0.0, 0.0, 0.0)],
            device=self.model.device,
        )
        wp.launch(
            apply_particle_delta_transform_kernel,
            dim=self.model.particle_count,
            inputs=[self.state_0.particle_q, self.state_0.particle_qd, delta_tf, 1.0 / self.frame_dt],
            device=self.model.device,
        )
        self._ik_pan_prev_tf = pan_tf

    # ------------------------------------------------------------------ step
    def simulate(self) -> None:
        self._prepare_frame_targets()
        knife_alpha = self.knife_alpha
        pan_alpha = self.pan_alpha

        for substep in range(self.sim_substeps):
            wp.copy(self.substep_joint_q_prev, self.state_0.joint_q)
            substep_alpha = float((substep + 1) / self.sim_substeps)
            wp.launch(
                interpolate_joint_positions_kernel,
                dim=self.model.joint_coord_count,
                inputs=[self.frame_joint_q_start, self.frame_joint_q_end, substep_alpha],
                outputs=[self.state_0.joint_q],
                device=self.model.device,
            )
            wp.launch(
                update_joint_velocity_kernel,
                dim=self.model.joint_dof_count,
                inputs=[self.substep_joint_q_prev, self.state_0.joint_q, 1.0 / self.sim_dt],
                outputs=[self.state_0.joint_qd],
                device=self.model.device,
            )
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
            self._update_knife_transform(knife_alpha)

            if self.ik_control_enabled and self.pan_follows_hand:
                self._update_pan_and_meat_from_ik()
            else:
                self._update_pan_transform(pan_alpha)

            if not self.ik_control_enabled:
                carry_active = self._apply_rigid_carry_transform(self.sim_time + (substep + 1) * self.sim_dt)
                if not carry_active:
                    self.solver.step(self.state_0, self.state_0, None, None, self.sim_dt)
                    self.solver.project_outside(self.state_0, self.state_0, self.sim_dt)

        self.sim_time += self.frame_dt
        self.frame_index += 1

    def step(self) -> None:
        self.simulate()
        if self.ik_control_enabled:
            self._report_pose()

    # ----------------------------------------------------------------- render
    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        if self.ik_control_enabled and hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo(
                "dexforce_right_tcp_target",
                self.right_tf,
                snap_to=self._current_tcp_transform(self.right_ee_index, self.right_ee_offset),
            )
        self.viewer.log_state(self.state_0)
        if self.show_deformation:
            wp.launch(
                compute_deformation_colors,
                dim=self.model.particle_count,
                inputs=[
                    self.state_0.mpm.particle_Jp,
                    self.particle_colors,
                    MEAT_COLOR,
                    CUT_COLOR,
                    12.0,
                ],
                device=self.model.device,
            )
        else:
            self.particle_colors.fill_(MEAT_COLOR)
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
        _changed, self.pan_follows_hand = imgui.checkbox("Pan follows hand", self.pan_follows_hand)

    # ----------------------------------------------------------- diagnostics
    def _report_pose(self, force: bool = False) -> None:
        if not force and self.print_interval > 0.0 and self.sim_time - self.last_print_time < self.print_interval:
            return

        actual_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        target_pos = self._vec3_to_np(wp.transform_get_translation(self.right_tf))
        actual_pos = self._vec3_to_np(wp.transform_get_translation(actual_tf))
        target_rot = self._quat_to_np(wp.transform_get_rotation(self.right_tf))
        actual_rot = self._quat_to_np(wp.transform_get_rotation(actual_tf))
        pos_err = float(np.linalg.norm(target_pos - actual_pos))
        rot_err = self._quat_angle_error_deg(target_rot, actual_rot)

        pan_tf = self._current_pan_transform()
        pan_pos = self._vec3_to_np(wp.transform_get_translation(pan_tf))
        pan_rot = self._quat_to_np(wp.transform_get_rotation(pan_tf))

        print(
            f"[{self.sim_time:7.3f}s] "
            f"R target={self._format_pose(target_pos, target_rot)} "
            f"actual={self._format_pose(actual_pos, actual_rot)} "
            f"pos_err={pos_err:.5f} m rot_err={rot_err:.3f} deg | "
            f"pan pos={self._format_xyz(pan_pos)} quat_xyzw={self._format_xyzw(pan_rot)}"
        )
        self.last_print_time = self.sim_time

    def _quat_angle_error_deg(self, quat_a: np.ndarray, quat_b: np.ndarray) -> float:
        dot = abs(float(np.dot(quat_a, quat_b)))
        return float(np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0))))

    def _format_xyz(self, xyz: np.ndarray) -> str:
        return f"[{xyz[0]: .4f}, {xyz[1]: .4f}, {xyz[2]: .4f}]"

    def _format_xyzw(self, xyzw: np.ndarray) -> str:
        return f"[{xyzw[0]: .4f}, {xyzw[1]: .4f}, {xyzw[2]: .4f}, {xyzw[3]: .4f}]"

    def _format_pose(self, xyz: np.ndarray, xyzw: np.ndarray) -> str:
        return f"pos={self._format_xyz(xyz)} quat_xyzw={self._format_xyzw(xyzw)}"

    # ------------------------------------------------------------------ parser
    @staticmethod
    def create_parser():
        parser = kitchen_v64.Example.create_parser()
        parser.set_defaults(num_frames=0)
        parser.add_argument(
            "--print-interval",
            type=float,
            default=1.0,
            help="Seconds between right TCP pose reports after the IK gizmo is enabled. Use 0.0 to print every frame.",
        )
        parser.add_argument(
            "--pan-follows-hand",
            type=lambda v: str(v).lower() in ("1", "true", "yes"),
            default=True,
            help="Move the pan and meat to follow the right-hand gizmo (default: true).",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

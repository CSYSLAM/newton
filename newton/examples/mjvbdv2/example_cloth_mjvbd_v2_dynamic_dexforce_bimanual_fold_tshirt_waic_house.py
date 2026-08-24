# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Dynamic Dexforce W1 folding a T-shirt with MJVBDV2.

This example reuses the scene, two-pass IK trajectory, and cloth parameters
from the kinematic MJVBDV2 example. It solves the current Cartesian TCP targets
with realtime IK before every displayed frame, then uses consecutive IK
solutions as MuJoCo position/velocity-drive targets. Robot joint positions are
never written directly during simulation. MJVBDV2 advances the dynamic robot
joints in MuJoCo and the shirt in VBD, with robot links acting as one-way
collision proxies.

Run, from the repository root::

    uv run --extra examples -m newton.examples \
        cloth_mjvbd_v2_dynamic_dexforce_bimanual_fold_tshirt_waic_house
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mjvbdv2 import example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house as reference
from newton.solvers import SolverMJVBDV2


@wp.kernel
def _accumulate_soft_contact_count(contact_count: wp.array[int], maximum: wp.array[int]):
    if wp.tid() == 0:
        wp.atomic_max(maximum, 0, contact_count[0])


class Example(reference.Example):
    """Dynamic-joint counterpart of the kinematic MJVBDV2 folding example."""

    def __init__(self, viewer, args):
        # capture() is overridden below so the base class does not capture the
        # temporary kinematic solver constructed before the dynamic backend.
        self._dynamic_solver_ready = False
        super().__init__(viewer, args)

        # Dynamic PD interpolation must start at the previous IK target, not at
        # the lagging simulated joint state. Both are identical initially.
        wp.copy(self.frame_q_start, self.model.joint_q)
        wp.copy(self.frame_q_end, self.model.joint_q)

        self._align_dynamic_mimic_offsets()
        self._configure_dynamic_contact_response()
        self._configure_dynamic_joint_drives()
        self.control = self.model.control()
        wp.copy(self.control.joint_target_q, self.model.joint_q)

        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=self.robot_articulations,
            joint_mode="dynamic",
            contact_mode="soft",
            collision_options={"soft_contact_margin": reference.SOFT_MARGIN},
            vbd_options={
                "iterations": args.vbd_iterations,
                "particle_enable_self_contact": args.self_contact,
                "particle_self_contact_radius": reference.SELF_RADIUS,
                "particle_self_contact_margin": reference.SELF_MARGIN,
                "particle_topological_contact_filter_threshold": 1,
                "particle_rest_shape_contact_exclusion_radius": 0.03,
                "particle_vertex_contact_buffer_size": 16,
                "particle_edge_contact_buffer_size": 20,
                "rigid_body_particle_contact_buffer_size": 256,
                "particle_collision_detection_interval": args.self_contact_interval,
            },
            mujoco_options={"use_mujoco_cpu": self.model.device.is_cpu},
        )
        if self.solver.features.backend != "coupled":
            raise RuntimeError(f"Dynamic folding requires the coupled backend, got {self.solver.features.backend}")

        self.contacts = self.solver.contacts
        self.initial_dynamic_joint_q = self.state_0.joint_q.numpy().copy()
        self.initial_shirt_q = self.state_0.particle_q.numpy().copy()
        self.script_sim_duration = sum(segment[0] for segment in self.segments) / self.args.trajectory_time_scale
        self.max_soft_contact_count = wp.zeros(1, dtype=wp.int32, device=self.device)
        self.track_contact_count = bool(args.test)
        self._dynamic_solver_ready = True
        self.use_graph = bool(args.graph_capture) and self.device.is_cuda
        self.capture()

    def _align_dynamic_mimic_offsets(self):
        """Match the fully open PIP pose authored by the kinematic trajectory."""
        if self.model.constraint_mimic_count == 0:
            return
        offsets = self.model.constraint_mimic_coef0.numpy()
        followers = self.model.constraint_mimic_joint0.numpy()
        for constraint, follower in enumerate(followers):
            joint_name = self.model.joint_label[int(follower)].rsplit("/", maxsplit=1)[-1]
            if joint_name.endswith("_PIP"):
                offsets[constraint] = 0.0
        self.model.constraint_mimic_coef0.assign(offsets)

    def _configure_dynamic_contact_response(self):
        """Match the kinematic example's contact law for the dynamic proxies."""
        self.model.soft_contact_ke = self.args.contact_ke
        self.model.soft_contact_kd = self.args.contact_kd
        stiffness = self.model.shape_material_ke.numpy()
        stiffness[: self.robot_shape_end] = self.args.contact_ke
        self.model.shape_material_ke.assign(stiffness)

        # The reference scene specifies legacy proportional damping. Convert it
        # to the current VBD stiffness-scaled units after applying any override.
        damping = self.model.shape_material_kd.numpy()
        legacy_mixed_kd = 0.5 * (self.args.contact_kd + reference.LEGACY_SHAPE_CONTACT_KD)
        mixed_kd = legacy_mixed_kd * 0.5 * (self.model.soft_contact_ke + stiffness)
        damping[:] = 2.0 * mixed_kd - self.model.soft_contact_kd
        self.model.shape_material_kd.assign(damping)

    def _configure_dynamic_joint_drives(self):
        """Configure stable position drives before MuJoCo model conversion."""
        mode = self.model.joint_target_mode.numpy().copy()
        stiffness = self.model.joint_target_ke.numpy().copy()
        damping = self.model.joint_target_kd.numpy().copy()
        effort_limit = self.model.joint_effort_limit.numpy().copy()
        qd_start = self.model.joint_qd_start.numpy()
        q_start = self.model.joint_q_start.numpy()

        arm_names = {*self.LEFT_ARM, *self.RIGHT_ARM}
        target_coord_indices = []
        for joint, label in enumerate(self.model.joint_label):
            begin = int(qd_start[joint])
            end = int(qd_start[joint + 1])
            if begin == end:
                continue

            short_name = label.rsplit("/", maxsplit=1)[-1]
            if short_name in arm_names:
                kp, kd = self.args.arm_kp, self.args.arm_kd
                effort_limit[begin:end] = np.maximum(effort_limit[begin:end], self.args.arm_effort_limit)
                target_coord_indices.extend(range(int(q_start[joint]), int(q_start[joint + 1])))
            elif short_name.endswith("_PIP"):
                mode[begin:end] = int(newton.JointTargetMode.NONE)
                stiffness[begin:end] = 0.0
                damping[begin:end] = 0.0
                continue
            elif "_HAND_" in short_name:
                kp, kd = self.args.hand_kp, self.args.hand_kd
                effort_limit[begin:end] = np.maximum(effort_limit[begin:end], self.args.hand_effort_limit)
                target_coord_indices.extend(range(int(q_start[joint]), int(q_start[joint + 1])))
            else:
                kp, kd = self.args.hold_kp, self.args.hold_kd
                effort_limit[begin:end] = np.maximum(effort_limit[begin:end], self.args.hold_effort_limit)
                target_coord_indices.extend(range(int(q_start[joint]), int(q_start[joint + 1])))

            mode[begin:end] = int(newton.JointTargetMode.POSITION_VELOCITY)
            stiffness[begin:end] = kp
            damping[begin:end] = kd

        self.model.joint_target_mode.assign(mode)
        self.model.joint_target_ke.assign(stiffness)
        self.model.joint_target_kd.assign(damping)
        self.model.joint_effort_limit.assign(effort_limit)
        self.dynamic_target_coord_indices = np.asarray(target_coord_indices, dtype=np.int32)

    def _prepare_runtime_ik_frame_start(self):
        """Advance the persistent dynamic PD-target interval by one frame."""
        wp.copy(self.frame_q_start, self.frame_q_end)

    def _simulate_substeps(self):
        for step in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (step + 1) / self.sim_substeps
            wp.launch(
                reference._interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.control.joint_target_q],
                device=self.device,
            )
            wp.launch(
                reference._joint_velocity,
                self.model.joint_dof_count,
                [
                    self.frame_q_start,
                    self.frame_q_end,
                    1.0 / self.frame_dt,
                    self.control.joint_target_qd,
                ],
                device=self.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            if self.track_contact_count:
                wp.launch(
                    _accumulate_soft_contact_count,
                    1,
                    [self.contacts.soft_contact_count, self.max_soft_contact_count],
                    device=self.device,
                )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _capture_dynamic_graph(self, materials):
        """Capture realtime IK and one complete dynamic MuJoCo/VBD frame."""
        self.model.soft_contact_mu = materials[0]
        state_0_backup = self.model.state()
        state_1_backup = self.model.state()
        state_0_backup.assign(self.state_0)
        state_1_backup.assign(self.state_1)
        ik_q_backup = wp.clone(self.ik_q)
        frame_q_start_backup = wp.clone(self.frame_q_start)
        frame_q_end_backup = wp.clone(self.frame_q_end)

        with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
            wp.launch(
                reference._set_shape_friction,
                self.model.shape_count,
                [self.model.shape_material_mu, self.robot_shape_end, materials[1], materials[2]],
                device=self.device,
            )
            self._solve_runtime_ik_frame()
            self._simulate_substeps()

        self.state_0.assign(state_0_backup)
        self.state_1.assign(state_1_backup)
        self.ik_q.assign(ik_q_backup)
        self.frame_q_start.assign(frame_q_start_backup)
        self.frame_q_end.assign(frame_q_end_backup)
        return capture.graph

    def capture(self):
        """Capture complete dynamic MuJoCo/VBD frames for every material variant."""
        self.graph = None
        self.graphs = {}
        if not self._dynamic_solver_ready or not self.use_graph:
            return

        for materials in self.material_variants:
            self.graphs[materials] = self._capture_dynamic_graph(materials)
        self.graph = self.graphs[self.material_variants[0]]

    def test_post_step(self):
        particle_q = self.state_0.particle_q.numpy()
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        target_q = self.frame_q_end.numpy()
        indices = self.dynamic_target_coord_indices
        tracking_error = float(np.max(np.abs(joint_q[indices] - target_q[indices])))
        max_joint_speed = float(np.max(np.abs(joint_qd)))
        if not np.all(np.isfinite(joint_q)) or not np.all(np.isfinite(joint_qd)):
            raise ValueError(
                f"Dynamic robot became non-finite at frame {self.frame_index}, "
                f"script time {self.sim_time * self.args.trajectory_time_scale:.3f} s, "
                f"tracking error {tracking_error:.3f} rad, max speed {max_joint_speed:.3f} rad/s"
            )
        if not np.all(np.isfinite(particle_q)):
            current_contacts = int(self.contacts.soft_contact_count.numpy()[0])
            raise ValueError(
                f"T-shirt became non-finite at frame {self.frame_index}, "
                f"script time {self.sim_time * self.args.trajectory_time_scale:.3f} s, "
                f"tracking error {tracking_error:.3f} rad, max speed {max_joint_speed:.3f} rad/s, "
                f"soft contacts {current_contacts}, max soft contacts {int(self.max_soft_contact_count.numpy()[0])}"
            )

    def test_final(self):
        super().test_final()
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        if not np.all(np.isfinite(joint_q)) or not np.all(np.isfinite(joint_qd)):
            raise ValueError("Dynamic robot joint state is not finite")
        if self.solver.features.backend != "coupled" or self.solver.mujoco_solver is None:
            raise ValueError("Dynamic folding did not use the MuJoCo/VBD coupled backend")

        if self.sim_time < 2.0:
            return
        indices = self.dynamic_target_coord_indices
        motion = float(np.max(np.abs(joint_q[indices] - self.initial_dynamic_joint_q[indices])))
        if motion < 0.05:
            raise ValueError(f"Dynamic robot did not follow the folding trajectory: motion={motion}")

        target_q = self.frame_q_end.numpy()
        tracking_error = float(np.max(np.abs(joint_q[indices] - target_q[indices])))
        if tracking_error > self.args.max_tracking_error:
            raise ValueError(
                f"Dynamic robot tracking error {tracking_error:.3f} exceeds "
                f"--max-tracking-error={self.args.max_tracking_error:.3f}"
            )
        if self.sim_time >= 3.0 and int(self.max_soft_contact_count.numpy()[0]) == 0:
            raise ValueError("Dynamic robot never generated a contact with the T-shirt")
        if self.sim_time >= self.script_sim_duration:
            initial_span = np.ptp(self.initial_shirt_q[:, :2], axis=0)
            final_span = np.ptp(self.state_0.particle_q.numpy()[:, :2], axis=0)
            initial_area = float(np.prod(initial_span))
            final_area = float(np.prod(final_span))
            if final_area >= 0.85 * initial_area:
                raise ValueError(
                    "T-shirt did not become compact after the dynamic fold: "
                    f"initial XY area={initial_area:.4f}, final XY area={final_area:.4f}"
                )

    @staticmethod
    def create_parser():
        parser = reference.Example.create_parser()
        parser.set_defaults(
            graph_capture=True,
            trajectory_time_scale=4.0,
        )
        parser.add_argument(
            "--vbd-iterations",
            type=int,
            default=reference.VBD_ITERATIONS,
            help="VBD iterations per substep; matches the kinematic example by default.",
        )
        parser.add_argument("--arm-kp", type=float, default=12000.0, help="Arm position-drive stiffness.")
        parser.add_argument("--arm-kd", type=float, default=350.0, help="Arm velocity-drive gain.")
        parser.add_argument(
            "--arm-effort-limit",
            type=float,
            default=1000.0,
            help="Minimum MuJoCo arm actuator force limit [N·m].",
        )
        parser.add_argument("--hand-kp", type=float, default=300.0, help="Finger position-drive stiffness.")
        parser.add_argument("--hand-kd", type=float, default=15.0, help="Finger position-drive damping.")
        parser.add_argument(
            "--hand-effort-limit",
            type=float,
            default=40.0,
            help="Minimum MuJoCo finger actuator force limit [N·m].",
        )
        parser.add_argument(
            "--hold-kp",
            type=float,
            default=50000.0,
            help="Stiffness for non-folding support joints.",
        )
        parser.add_argument(
            "--hold-kd",
            type=float,
            default=1000.0,
            help="Velocity gain for non-folding support joints.",
        )
        parser.add_argument(
            "--hold-effort-limit",
            type=float,
            default=1000.0,
            help="Minimum MuJoCo support-joint force limit [N·m].",
        )
        parser.add_argument(
            "--self-contact",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable cloth self-contact; enabled in the kinematic reference.",
        )
        parser.add_argument(
            "--self-contact-interval",
            type=int,
            default=-1,
            help="Self-contact detection interval; -1 matches the kinematic reference.",
        )
        parser.add_argument(
            "--contact-ke",
            type=float,
            default=3.0e5,
            help="Robot/cloth contact stiffness; matches the kinematic reference.",
        )
        parser.add_argument(
            "--contact-kd",
            type=float,
            default=reference.LEGACY_SOFT_CONTACT_KD,
            help="Legacy proportional contact damping; matches the kinematic reference.",
        )
        parser.add_argument(
            "--max-tracking-error",
            type=float,
            default=0.75,
            help="Maximum accepted final driven-joint error [rad].",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

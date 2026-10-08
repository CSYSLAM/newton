# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Insert a rigid plug with a dynamic Dexforce W1 and two-way MJVBDV2 coupling.

This is the ``mjvbd_v2_plug_socket`` task with the robot integrated by MuJoCo
instead of prescribed. Realtime IK writes position targets for gravity-
compensated joint drives. The thumb holds the authored pinch angle while the
index closes past the plug, so the effort-limited index stops on it through
VBD contact feedback, and the plug's resistance in the socket acts back on the
arm. MuJoCo also resolves the fingers against the table and socket.

A physical grasp does not leave the plug where the kinematic demo's tuned
carry and wrist corrections assume, so after the lift the controller reads the
plug's pose in the hand once, like an in-hand pose estimate, and aims the hand
so that the plug itself reaches the socket axis. The plug is never moved
directly; it follows only gravity and contact.

Run, from the repository root::

    uv run --extra examples -m newton.examples mjvbd_v2_plug_socket_two_way
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mjvbdv2 import example_mjvbd_v2_plug_socket as kinematic
from newton.examples.mjvbdv2.example_mjvbd_v2_plug_socket import (
    _interpolate_indexed_q,
    _update_indexed_joint_velocity,
)
from newton.solvers import SolverMJVBDV2


class Example(kinematic.Example):
    """Run the plug/socket insertion on a dynamic, two-way coupled W1."""

    floating_robot = False
    expected_backend = "coupled"
    # The thumb holds the authored pinch angle as a stiff anvil. The index
    # closes past the plug, so its effort limit, not its target, sets the
    # pinch force once contact stops it.
    GRASP_HAND_JOINTS: ClassVar[dict[str, float]] = {
        "RIGHT_HAND_INDEX": 0.75,
        "RIGHT_INDEX_PIP": 0.75,
        "RIGHT_HAND_THUMB1": 0.35,
        "RIGHT_HAND_THUMB2": 1.4,
    }
    # Pinch through the plug's center of mass. Two fingertip contacts resist
    # almost no torque about the pinch axis, so the kinematic grip 1.5 cm
    # forward of it lets the plug swing around the fingertips when lifted.
    plug_to_grip = wp.vec3(0.002, 0.01, 0.0)
    RIGHT_THUMB = ("RIGHT_HAND_THUMB1", "RIGHT_HAND_THUMB2")
    RIGHT_INDEX = ("RIGHT_HAND_INDEX",)

    def _configure_robot_bodies(self, builder: newton.ModelBuilder) -> None:
        """Cancel link weight passively so the drives only correct tracking error."""
        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp.values is None:
            gravcomp.values = {}
        for body in range(self.robot_body_end):
            gravcomp.values[body] = 1.0

    def _robot_root_q_start(self) -> int:
        return -1

    def _write_root_pose(self, destination: wp.array[float]) -> None:
        del destination

    def _configure_drives(self) -> None:
        """Track IK with the right arm, pinch with the hand, and hold everything else."""
        args = self.args
        mode = self.model.joint_target_mode.numpy().copy()
        ke = self.model.joint_target_ke.numpy().copy()
        kd = self.model.joint_target_kd.numpy().copy()
        effort = self.model.joint_effort_limit.numpy().copy()
        armature = self.model.joint_armature.numpy().copy()
        damping = self.model.joint_damping.numpy().copy()
        qd_start = self.model.joint_qd_start.numpy()
        articulation_start = int(self.model.articulation_start.numpy()[self.robot_articulations[0]])
        articulation_end = int(self.model.articulation_end.numpy()[self.robot_articulations[0]])
        for joint in range(articulation_start, articulation_end):
            begin, end = int(qd_start[joint]), int(qd_start[joint + 1])
            if begin == end:
                continue
            name = self.model.joint_label[joint].rsplit("/", 1)[-1]
            if name.endswith("_PIP"):
                # Driven through the URDF mimic constraint on the proximal joint.
                mode[begin:end] = int(newton.JointTargetMode.NONE)
                ke[begin:end] = kd[begin:end] = 0.0
                continue
            if name in self.RIGHT_ARM:
                gains = (args.arm_kp, args.arm_kd, max(float(effort[begin]), args.arm_effort))
            elif name in self.RIGHT_THUMB or name in self.RIGHT_INDEX:
                limit = args.index_effort if name in self.RIGHT_INDEX else args.thumb_effort
                gains = (args.hand_kp, args.hand_kd, limit)
                armature[begin:end] = np.maximum(armature[begin:end], args.hand_armature)
                damping[begin:end] = np.maximum(damping[begin:end], args.hand_damping)
            else:
                gains = (args.hold_kp, args.hold_kd, max(float(effort[begin]), args.hold_effort))
            mode[begin:end] = int(newton.JointTargetMode.POSITION_VELOCITY)
            ke[begin:end], kd[begin:end], effort[begin:end] = gains
        self.model.joint_target_mode.assign(mode)
        self.model.joint_target_ke.assign(ke)
        self.model.joint_target_kd.assign(kd)
        self.model.joint_effort_limit.assign(effort)
        self.model.joint_armature.assign(armature)
        self.model.joint_damping.assign(damping)

    def _create_solver(self) -> SolverMJVBDV2:
        """Create the dynamic two-way MuJoCo/VBD solver."""
        self._configure_drives()
        wp.copy(self.control.joint_target_q, self.model.joint_q)
        self.control.joint_target_qd.zero_()
        return SolverMJVBDV2(
            self.model,
            mujoco_articulations=self.robot_articulations,
            joint_mode="dynamic",
            contact_mode="full",
            coupling="two_way",
            coupling_options={"mass_scale": self.args.proxy_mass_scale},
            vbd_options=self._vbd_options(),
            collision_options=self._collision_options(),
        )

    def test_final(self) -> None:
        """Also require a contact-limited pinch that physically lifted the plug."""
        super().test_final()
        if not self.solver.features.two_way_coupling_enabled:
            raise ValueError("The plug/socket variant must run with two-way coupling")
        if self.frame_index < int(8.0 * kinematic.FPS):
            return
        lift = self._peak_plug_height - float(self.initial_plug_position[2])
        if lift < 0.10:
            raise ValueError(f"The pinch did not lift the plug off the table: lift={lift:.4f} m")
        index_target = self.GRASP_HAND_JOINTS["RIGHT_HAND_INDEX"]
        if self._peak_carry_index > index_target - 0.15:
            raise ValueError(
                f"The index closed toward its {index_target} rad target instead of stopping on the plug: "
                f"{self._peak_carry_index:.3f} rad"
            )
        plug = self.state_0.body_q.numpy()[self.plug_body]
        print(
            f"[PlugTwoWay] PASS: lifted {lift * 100:.1f} cm, index stopped at {self._peak_carry_index:.3f} of "
            f"{index_target} rad, plug retained at {np.round(plug[:3], 4).tolist()} after release.",
            flush=True,
        )

    def _hand_base_pose(self, body_q: np.ndarray) -> wp.transform:
        """Pose of the IK hand frame, which collapses into the wrist link in the scene."""
        if not hasattr(self, "_wrist_to_hand"):
            ik_state = self.ik_model.state()
            newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, ik_state)
            ik_body_q = ik_state.body_q.numpy()
            hand = wp.transform(*ik_body_q[self._body_index(self.ik_model.body_label, "right_hand_base")])
            wrist = wp.transform(*ik_body_q[self._wrist_link(self.ik_model)])
            self._wrist_to_hand = wp.transform_inverse(wrist) * hand
            self._scene_wrist = self._wrist_link(self.model)
        return wp.transform(*body_q[self._scene_wrist]) * self._wrist_to_hand

    def _hand_goal(self, plug_position: wp.vec3) -> tuple[wp.vec3, wp.quat]:
        """IK target that puts the measured in-hand plug upright at ``plug_position``."""
        hand = wp.transform(plug_position, wp.quat_identity()) * wp.transform_inverse(self._plug_in_hand)
        return wp.transform_point(hand, self.hand_target_offset), wp.transform_get_rotation(hand)

    def _sample_insertion(self, time_seconds: float) -> tuple[wp.vec3, wp.quat] | None:
        """Plan align, insert, hold, and retract from the measured plug-in-hand pose."""
        k = kinematic
        align_start = (
            k.SETTLE_SECONDS
            + k.APPROACH_SECONDS
            + k.DESCEND_SECONDS
            + k.GRASP_SECONDS
            + k.POST_GRASP_HOLD_SECONDS
            + k.RAISE_SECONDS
        )
        if time_seconds < align_start:
            return None
        if self._plug_in_hand is None:
            body_q = self.state_0.body_q.numpy()
            self._plug_in_hand = wp.transform_inverse(self._hand_base_pose(body_q)) * wp.transform(
                *body_q[self.plug_body]
            )
            self._align_from = (self.last_target, self.last_rotation)
        forward = self._hand_goal(k.PLUG_FORWARD_POSITION)
        inserted = self._hand_goal(k.PLUG_INSERTED_POSITION)
        elapsed = time_seconds - align_start
        if elapsed < k.ALIGN_SECONDS:
            alpha = self._smoothstep(elapsed / k.ALIGN_SECONDS)
            return (
                self._lerp_vec3(self._align_from[0], forward[0], alpha),
                wp.quat_slerp(self._align_from[1], forward[1], alpha),
            )
        elapsed -= k.ALIGN_SECONDS
        if elapsed < k.INSERT_SECONDS:
            alpha = self._smoothstep(elapsed / k.INSERT_SECONDS)
            return self._lerp_vec3(forward[0], inserted[0], alpha), inserted[1]
        elapsed -= k.INSERT_SECONDS + k.INSERT_SETTLE_SECONDS + k.RELEASE_SECONDS
        if elapsed < 0.0:
            return inserted
        alpha = self._smoothstep(min(elapsed / k.RETRACT_SECONDS, 1.0))
        return (
            self._lerp_vec3(inserted[0], k.HAND_STANDBY_POSITION, alpha),
            wp.quat_slerp(inserted[1], k.HAND_TARGET_ROTATION, alpha),
        )

    def _prepare_frame(self) -> None:
        """Solve IK and advance the per-frame joint-target interval."""
        target, grasp_alpha, self.phase = self._sample_controller(self.sim_time)
        rotation = self._sample_hand_rotation(self.sim_time)
        planned = self._sample_insertion(self.sim_time)
        if planned is not None:
            target, rotation = planned
        self.last_target, self.last_rotation = target, rotation
        self._set_ik_target(target, rotation)
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=self.ik_iterations)
        self._restore_locked_ik_q()
        self.target_position = np.asarray(target, dtype=np.float32)

        wp.copy(self.frame_q_start, self.frame_q_end)
        self._copy_ik_to_scene(self.frame_q_end)
        self._write_hand_pose(grasp_alpha, self.frame_q_end)

    def _simulate_substeps(self) -> None:
        """Interpolate drive targets; MuJoCo integrates the robot."""
        for substep in range(self.sim_substeps):
            alpha = (substep + 1) / self.sim_substeps
            wp.launch(
                _interpolate_indexed_q,
                self.robot_q_indices.shape[0],
                [self.robot_q_indices, self.frame_q_start, self.frame_q_end, alpha, self.control.joint_target_q],
                device=self.device,
            )
            wp.launch(
                _update_indexed_joint_velocity,
                self.robot_joint_indices.shape[0],
                [
                    self.robot_joint_indices,
                    self.frame_q_start,
                    self.frame_q_end,
                    self.model.joint_type,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    1.0 / self.frame_dt,
                    self.control.joint_target_qd,
                ],
                device=self.device,
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    _peak_plug_height = 0.0
    _peak_carry_index = 0.0

    def step(self) -> None:
        super().step()
        if self.phase in ("raise", "align", "insert"):
            q = self.state_0.joint_q.numpy()
            self._peak_carry_index = max(self._peak_carry_index, float(q[self.hand_q_indices.numpy()[0]]))
            self._peak_plug_height = max(self._peak_plug_height, float(self.state_0.body_q.numpy()[self.plug_body, 2]))
        if self.args.verbose_trace and self.frame_index % 30 == 0:
            q = self.state_0.joint_q.numpy()
            hand = q[self.hand_q_indices.numpy()]
            body_q = self.state_0.body_q.numpy()
            plug = body_q[self.plug_body]
            reached = self.reached_target_point(body_q)
            print(
                f"[PlugTwoWay] t={self.sim_time:5.2f} phase={self.phase:15s} plug={np.round(plug[:3], 4).tolist()} "
                f"plug_qw={plug[6]:+.4f} hand_q={np.round(hand, 3).tolist()} "
                f"track_err_mm={np.linalg.norm(reached - self.target_position) * 1e3:.1f}",
                flush=True,
            )

    _plug_in_hand = None

    def _wrist_link(self, model: newton.Model) -> int:
        """Return the RIGHT_J7 child, which carries the hand once fixed joints collapse."""
        joint = next(index for index, label in enumerate(model.joint_label) if label.endswith("/RIGHT_J7"))
        return int(model.joint_child.numpy()[joint])

    def reached_target_point(self, body_q: np.ndarray) -> np.ndarray:
        """World position of the IK target point on the simulated hand."""
        return np.asarray(wp.transform_point(self._hand_base_pose(body_q), self.hand_target_offset))

    @staticmethod
    def create_parser():
        parser = kinematic.Example.create_parser()
        parser.add_argument("--arm-kp", type=float, default=12000.0, help="Right-arm drive stiffness [N*m/rad].")
        parser.add_argument("--arm-kd", type=float, default=350.0, help="Right-arm drive damping [N*m*s/rad].")
        parser.add_argument("--arm-effort", type=float, default=0.0, help="Minimum right-arm effort limit [N*m].")
        parser.add_argument("--hand-kp", type=float, default=300.0, help="Thumb/index drive stiffness [N*m/rad].")
        parser.add_argument("--hand-kd", type=float, default=15.0, help="Thumb/index drive damping [N*m*s/rad].")
        parser.add_argument("--index-effort", type=float, default=3.0, help="Index pinch effort limit [N*m].")
        parser.add_argument("--thumb-effort", type=float, default=10.0, help="Thumb anvil effort limit [N*m].")
        parser.add_argument("--hand-armature", type=float, default=0.005, help="Thumb/index armature [kg*m^2].")
        parser.add_argument("--hand-damping", type=float, default=0.5, help="Passive thumb/index damping [N*m*s/rad].")
        parser.add_argument("--hold-kp", type=float, default=50000.0, help="Support-joint drive stiffness.")
        parser.add_argument("--hold-kd", type=float, default=1000.0, help="Support-joint drive damping.")
        parser.add_argument("--hold-effort", type=float, default=1000.0, help="Minimum support-joint effort [N*m].")
        parser.add_argument("--proxy-mass-scale", type=float, default=4.0, help="Two-way VBD proxy mass scale.")
        parser.add_argument("--verbose-trace", action="store_true", help="Print plug and hand state twice a second.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

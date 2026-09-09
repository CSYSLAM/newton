# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Share experimental right-hand optical teleoperation between W1 examples."""

from pathlib import Path

import numpy as np
import warp as wp

from ._webxr_teleop import XRFrame
from ._webxr_w1_hand import HAND_SUFFIXES, W1HandRetargeter


@wp.kernel
def _limit_finger_targets(
    current: wp.array[float],
    indices: wp.array[int],
    values: wp.array[float],
    maximum_step: float,
    destination: wp.array[float],
):
    index = wp.tid()
    coordinate = indices[index]
    destination[coordinate] = current[coordinate] + wp.clamp(
        values[index] - current[coordinate], -maximum_step, maximum_step
    )


class W1SingleHandTeleop:
    """Mix optical input into examples with a right wrist and a relative pose retargeter."""

    def _init_optical_hand(self, urdf_path: Path) -> None:
        starts = self.model.joint_q_start.numpy()
        joints = {label.rsplit("/", 1)[-1]: index for index, label in enumerate(self.model.joint_label)}
        indices = []
        for suffix in HAND_SUFFIXES:
            name = f"RIGHT_{suffix}"
            if name not in joints:
                raise ValueError(f"W1 joint is missing: {name}")
            joint = joints[name]
            if starts[joint + 1] - starts[joint] != 1:
                raise ValueError(f"W1 hand joint must have one coordinate: {name}")
            indices.append(int(starts[joint]))
        self._optical_indices_host = np.asarray(indices, dtype=np.int32)
        self._optical_indices = wp.array(indices, dtype=wp.int32, device=self.device)
        self._optical_command = wp.zeros(len(indices), dtype=wp.float32, device=self.device)
        self._optical_mapper = W1HandRetargeter(Path(urdf_path), "right")
        self._optical_target = None
        self._optical_token = None
        self._optical_blocked = None
        self._optical_sequence = None
        self._optical_last_pose = None
        self._optical_status = "paused"
        self._input_mode = "controllers"
        self._recording_request = None

    def _hold_optical_hand(self) -> None:
        """Invalidate the wrist baseline while retaining the current finger coordinates."""
        self.retargeter.reset()
        self._optical_blocked = self._optical_token
        self._optical_last_pose = None
        self._optical_status = "paused"
        if self._input_mode == "hands":
            self._optical_target = self.state_0.joint_q.numpy()[self._optical_indices_host].copy()

    def _prepare_teleop_input(self, frame: XRFrame | None, lower: np.ndarray, upper: np.ndarray) -> bool:
        """Choose the motion source while preserving physical controller shortcut edges."""
        if frame is not None and frame.visibility_state != "visible":
            frame = None
        if frame is not None:
            request = (frame.stream_id, frame.recording_request)
            if request != self._recording_request:
                if frame.recording_request:
                    self.trajectory_recorder.toggle()
                self._recording_request = request
            if frame.input_mode != self._input_mode:
                self._hold_optical_hand()
                self._input_mode = frame.input_mode
                if self._input_mode == "controllers":
                    self._optical_target = None
            if frame.hands.get("right") is not None and not self._has_seen_controller:
                self._has_seen_controller = True
                if self.args.record_on_connect:
                    self.trajectory_recorder.start()
        controller = None if frame is None else frame.controllers.get("right")
        if controller is not None:
            if self._process_controller_buttons(frame.stream_id, frame.sequence, controller):
                self.reset_physics(source="quest-controller")
                return True
        else:
            self._record_button_pressed = False
            self._reset_button_pressed = False

        if self._input_mode == "hands":
            self._prepare_optical_hand(frame, lower, upper)
        elif controller is None:
            self.retargeter.reset()
            if self.teleoperation_active:
                self.phase = "waiting_for_quest" if not self._has_seen_controller else "quest_input_stale"
            else:
                self.phase = "teleoperation_standby"
        else:
            target = self.retargeter.update(
                controller.pose,
                clutch=controller.clutch,
                robot_position=self._teleop_position,
                robot_orientation=self._teleop_orientation,
                source_to_robot_rotation=np.eye(3) if frame.controller_space == "newton-world" else None,
            )
            if target is not None:
                self._teleop_position = np.clip(target.position, lower, upper)
                self._teleop_orientation = target.orientation
            self._teleop_grasp = controller.trigger_value
            self.phase = "quest_clutched" if controller.clutch else "quest_idle"
        return False

    def _prepare_optical_hand(self, frame: XRFrame | None, lower: np.ndarray, upper: np.ndarray) -> None:
        """Re-anchor fresh activations and hold through missing or rejected skeletons."""
        sample = None if frame is None else frame.hands.get("right")
        token = None if sample is None else (frame.stream_id, sample.activation)
        if sample is None or not sample.enabled or token == self._optical_blocked:
            if self._optical_status == "tracking" or self._optical_target is None:
                self._optical_target = self.state_0.joint_q.numpy()[self._optical_indices_host].copy()
            self.retargeter.reset()
            self._optical_blocked = self._optical_token
            self._optical_last_pose = None
            self._optical_status = "tracking-lost" if sample is None else "paused"
            self.phase = f"optical_right:{self._optical_status}"
            return
        if token != self._optical_token:
            self.retargeter.reset()
            self._optical_mapper.reset(self.state_0.joint_q.numpy()[self._optical_indices_host])
            self._optical_last_pose = None
            self._optical_token = token
        sequence = (frame.stream_id, frame.sequence)
        if sequence == self._optical_sequence:
            return
        self._optical_sequence = sequence
        try:
            previous = self._optical_last_pose
            if previous is not None:
                distance = np.linalg.norm(sample.pose.position - previous.position)
                angle = 2 * np.arccos(np.clip(abs(np.dot(sample.pose.orientation, previous.orientation)), 0, 1))
                if distance > 0.15 or angle > np.radians(60):
                    raise ValueError("Optical wrist pose jumped")
            desired = self._optical_mapper.solve(sample.joints)
        except (ValueError, np.linalg.LinAlgError):
            self._hold_optical_hand()
            self._optical_status = "invalid-tracking"
            self.phase = "optical_right:invalid-tracking"
            return
        self._optical_target = desired
        self._optical_last_pose = sample.pose
        self._optical_status = "tracking"
        # Keep the bag's existing release/material signal tied to finger closure.
        self._teleop_grasp = float(np.clip(np.mean(desired[[2, 4, 6, 8]]) / 1.309, 0, 1))
        target = self.retargeter.update(
            sample.pose,
            clutch=True,
            robot_position=self._teleop_position,
            robot_orientation=self._teleop_orientation,
            source_to_robot_rotation=np.eye(3) if frame.controller_space == "newton-world" else None,
        )
        if target is not None:
            self._teleop_position = np.clip(target.position, lower, upper)
            self._teleop_orientation = target.orientation
        self.phase = "optical_right:tracking"

    def _write_optical_fingers(self, destination: wp.array[float]) -> None:
        """Limit optical finger motion to 180 degrees per second in the plug example."""
        step = float(np.radians(180) * self.frame_dt)
        self._optical_command.assign(self._optical_target)
        wp.launch(
            _limit_finger_targets,
            len(self._optical_indices_host),
            [self.state_0.joint_q, self._optical_indices, self._optical_command, step, destination],
            device=self.device,
        )

    def _optical_scene_state(self) -> dict:
        """Advertise right-hand control and acknowledge the browser's wrist baseline."""
        return {
            "handTrackingEnabled": True,
            "handTrackingSides": ["right"],
            "inputMode": self._input_mode,
            "handTrackingState": {"right": self._optical_status},
            "handTrackingActivation": {"right": None if self._optical_token is None else self._optical_token[1]},
        }

    def _optical_record(self, frame: XRFrame | None) -> dict:
        """Record raw skeletons alongside the independently fitted finger targets."""
        return {
            "inputMode": self._input_mode,
            "handTrackingState": {"right": self._optical_status},
            "handJointTargets": {"right": None if self._optical_target is None else self._optical_target.tolist()},
            "xrHands": {}
            if frame is None
            else {
                side: {
                    "pose": [*sample.pose.position.tolist(), *sample.pose.orientation.tolist()],
                    "joints": sample.joints.tolist(),
                    "enabled": sample.enabled,
                    "activation": sample.activation,
                }
                for side, sample in frame.hands.items()
            },
        }

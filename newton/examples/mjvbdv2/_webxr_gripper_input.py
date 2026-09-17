# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Retarget one parallel gripper from controller or optical WebXR input."""

import numpy as np

from ._webxr_parallel_gripper import ParallelGripperRetargeter
from ._webxr_teleop import Pose, RelativePoseRetargeter, XRFrame


class GripperInput:
    """Hold measured targets on tracking loss and re-anchor each new activation."""

    def __init__(
        self,
        side: str,
        mapper: ParallelGripperRetargeter,
        pose: Pose,
        jaws: np.ndarray,
        *,
        translation_scale: float = 1.0,
        max_translation: float = 0.8,
    ):
        self.side, self.mapper = side, mapper
        self.retargeter = RelativePoseRetargeter(translation_scale=translation_scale, max_translation=max_translation)
        self.position, self.orientation, self.jaws = pose.position.copy(), pose.orientation.copy(), jaws.copy()
        self.mode = "controllers"
        self.status = "paused"
        self.token = self.blocked = self.sequence = self.previous_pose = None
        self.controller_stream = None

    def hold(self, pose: Pose, jaws: np.ndarray) -> None:
        """Stop target motion at the current robot pose without opening the jaws."""
        self.position, self.orientation = pose.position.copy(), pose.orientation.copy()
        self.jaws = jaws.copy()
        self.retargeter.reset()
        self.blocked = self.token
        self.previous_pose = None
        self.status = "paused"

    def update(self, frame: XRFrame | None, actual: Pose, jaws: np.ndarray) -> None:
        """Consume a fresh frame; each hand owns its own clutch and recovery token."""
        if frame is None or frame.visibility_state != "visible":
            self.hold(actual, jaws)
            self.status = "tracking-lost"
            return
        if self.mode != frame.input_mode:
            self.hold(actual, jaws)
            self.mode = frame.input_mode
            self.sequence = None
        basis = np.eye(3) if frame.controller_space == "newton-world" else None
        if self.mode == "controllers":
            if self.controller_stream != frame.stream_id:
                self.hold(actual, jaws)
                self.controller_stream = frame.stream_id
            sample = frame.controllers.get(self.side)
            if sample is None:
                self.hold(actual, jaws)
                return
            if not sample.clutch:
                self.hold(actual, jaws)
            self.jaws = self.mapper.coordinates(sample.trigger_value)
            target = self.retargeter.update(
                sample.pose,
                clutch=sample.clutch,
                robot_position=self.position,
                robot_orientation=self.orientation,
                source_to_robot_rotation=basis,
            )
            self.status = "tracking" if sample.clutch else "paused"
        else:
            sample = frame.hands.get(self.side)
            token = None if sample is None else (frame.stream_id, sample.activation)
            if sample is None or not sample.enabled or token == self.blocked:
                self.hold(actual, jaws)
                self.status = "tracking-lost" if sample is None else "paused"
                return
            if token != self.token:
                self.hold(actual, jaws)
                self.token = token
                self.mapper.reset(jaws)
            sequence = (frame.stream_id, frame.sequence)
            if sequence == self.sequence:
                return
            self.sequence = sequence
            try:
                if self.previous_pose is not None:
                    distance = np.linalg.norm(sample.pose.position - self.previous_pose.position)
                    dot = abs(np.dot(sample.pose.orientation, self.previous_pose.orientation))
                    if distance > 0.15 or dot < np.cos(np.radians(30)):
                        raise ValueError("Optical wrist pose jumped")
                desired = self.mapper.solve(sample.joints)
            except (ValueError, np.linalg.LinAlgError):
                self.hold(actual, jaws)
                self.status = "invalid-tracking"
                return
            self.jaws = desired
            self.previous_pose = sample.pose
            target = self.retargeter.update(
                sample.pose,
                clutch=True,
                robot_position=self.position,
                robot_orientation=self.orientation,
                source_to_robot_rotation=basis,
            )
            self.status = "tracking"
        if target is not None:
            self.position, self.orientation = target.position, target.orientation

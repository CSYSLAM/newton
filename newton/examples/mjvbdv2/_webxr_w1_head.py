# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared W1 head tracking for the Quest WebXR teleoperation examples."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp

from ._webxr_teleop import Pose

OBSERVER_VIEW_MODE = "observer"
FIRST_PERSON_VIEW_MODE = "robot-first-person"
HEAD_YAW_LIMITS = (-0.5 * np.pi, 0.5 * np.pi)
HEAD_PITCH_LIMITS = (-0.25 * np.pi, np.deg2rad(25.0))
HEAD_MAX_SPEED_RADIANS_S = np.deg2rad(50.0)
EYES_POSITION_IN_NECK2 = np.array((0.091, -0.051, 0.0), dtype=np.float32)
ROBOT_FORWARD = np.array((1.0, 0.0, 0.0), dtype=np.float32)
ROBOT_UP = np.array((0.0, 0.0, 1.0), dtype=np.float32)


@wp.kernel
def _write_neck_pose(
    indices: wp.array[int],
    yaw: float,
    pitch: float,
    joint_q: wp.array[float],
):
    index = wp.tid()
    if index == 0:
        joint_q[indices[index]] = yaw
    else:
        joint_q[indices[index]] = pitch


def head_pose_to_neck_targets(head_pose: Pose | None) -> tuple[float, float]:
    """Map a relative WebXR head pose to W1 yaw and pitch offsets."""
    if head_pose is None:
        return 0.0, 0.0
    x, y, z, w = (float(value) for value in head_pose.orientation)
    forward_x = -2.0 * (x * z + y * w)
    forward_y = 2.0 * (x * w - y * z)
    forward_z = 2.0 * (x * x + y * y) - 1.0
    yaw = np.arctan2(-forward_x, -forward_z)
    pitch = np.arctan2(forward_y, np.hypot(forward_x, forward_z))
    return (
        float(np.clip(yaw, *HEAD_YAW_LIMITS)),
        float(np.clip(pitch, *HEAD_PITCH_LIMITS)),
    )


def serialize_head_pose(head_pose: Pose | None) -> list[float] | None:
    """Return a compact position/quaternion trajectory value."""
    if head_pose is None:
        return None
    return [*head_pose.position.tolist(), *head_pose.orientation.tolist()]


class W1HeadController:
    """Rate-limit Quest head motion into W1 NECK1/NECK2 targets."""

    def __init__(self, model, device, base_rotation):
        neck_joints = [self._label_index(model.joint_label, name) for name in ("NECK1", "NECK2")]
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        joint_q = model.joint_q.numpy()
        joint_limit_lower = model.joint_limit_lower.numpy()
        joint_limit_upper = model.joint_limit_upper.numpy()
        for joint in neck_joints:
            if int(joint_q_start[joint + 1] - joint_q_start[joint]) != 1:
                raise ValueError(f"W1 neck joint must have one coordinate: {model.joint_label[joint]}")

        self._device = device
        self.coordinate_indices_host = np.asarray(
            [int(joint_q_start[joint]) for joint in neck_joints],
            dtype=np.int32,
        )
        self._coordinate_indices = wp.array(self.coordinate_indices_host, dtype=wp.int32, device=device)
        self.neutral = np.asarray(joint_q[self.coordinate_indices_host], dtype=np.float32)
        self._lower = np.asarray(
            [joint_limit_lower[int(joint_qd_start[joint])] for joint in neck_joints],
            dtype=np.float32,
        )
        self._upper = np.asarray(
            [joint_limit_upper[int(joint_qd_start[joint])] for joint in neck_joints],
            dtype=np.float32,
        )
        self.targets = self.neutral.copy()
        self._desired_targets = self.neutral.copy()
        self.hidden_body_ids = tuple(self._label_index(model.body_label, name) for name in ("neck1", "neck2"))
        self._eye_body = self.hidden_body_ids[-1]

        base_orientation = self._quat_array(base_rotation)
        self._robot_forward = self._rotate_vector(base_orientation, ROBOT_FORWARD)
        self._robot_up = self._rotate_vector(base_orientation, ROBOT_UP)

    @staticmethod
    def _label_index(labels: Sequence[str], name: str) -> int:
        matches = [index for index, label in enumerate(labels) if label == name or label.endswith("/" + name)]
        if len(matches) != 1:
            raise ValueError(f"Expected one W1 {name} label, found {len(matches)}")
        return matches[0]

    @staticmethod
    def _quat_array(value) -> np.ndarray:
        return np.asarray([float(value[index]) for index in range(4)], dtype=np.float32)

    @staticmethod
    def _rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
        xyz = np.asarray(quaternion[:3], dtype=np.float32)
        value = np.asarray(vector, dtype=np.float32)
        twice_cross = 2.0 * np.cross(xyz, value)
        return value + float(quaternion[3]) * twice_cross + np.cross(xyz, twice_cross)

    def set_desired_pose(self, view_mode: str, head_pose: Pose | None) -> None:
        """Track the headset in first person, or return toward neutral."""
        if view_mode == FIRST_PERSON_VIEW_MODE:
            offsets = np.asarray(head_pose_to_neck_targets(head_pose), dtype=np.float32)
            self._desired_targets = np.clip(self.neutral + offsets, self._lower, self._upper)
        else:
            self._desired_targets = self.neutral.copy()

    def write_targets(self, joint_q: wp.array, frame_dt: float) -> None:
        """Advance bounded neck targets and write them after arm IK assembly."""
        maximum_step = HEAD_MAX_SPEED_RADIANS_S * float(frame_dt)
        self.targets += np.clip(
            self._desired_targets - self.targets,
            -maximum_step,
            maximum_step,
        )
        wp.launch(
            _write_neck_pose,
            self._coordinate_indices.shape[0],
            [
                self._coordinate_indices,
                float(self.targets[0]),
                float(self.targets[1]),
                joint_q,
            ],
            device=self._device,
        )

    def reset(self) -> None:
        """Restore both current and desired neck targets to the authored pose."""
        self.targets = self.neutral.copy()
        self._desired_targets = self.neutral.copy()

    def camera_state(self, body_q: np.ndarray) -> dict[str, list[float]]:
        """Return an eye anchor whose orientation is supplied by WebXR."""
        neck_pose = body_q[self._eye_body]
        eye_position = neck_pose[:3] + self._rotate_vector(neck_pose[3:7], EYES_POSITION_IN_NECK2)
        return {
            "position": eye_position.tolist(),
            "front": self._robot_forward.tolist(),
            "up": self._robot_up.tolist(),
        }

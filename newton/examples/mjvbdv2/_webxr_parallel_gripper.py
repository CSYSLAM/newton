# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Map optical thumb/index separation to the MR 1269 W1 parallel gripper."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

DEFAULT_GRIPPER_URDF = (
    Path(__file__).resolve().parents[3]
    / "assets/w1-pikka-gripper/DexforceW1V021_pikka_gripper_simple_visual_collision.urdf"
)


class ParallelGripperRetargeter:
    """Convert a 15-100 mm human pinch span into the gripper's bounded opening."""

    def __init__(self, urdf_path: Path, *, side: str = "right", closed_span: float = 0.015, open_span: float = 0.10):
        if side not in ("left", "right"):
            raise ValueError("Gripper side must be left or right")
        if not np.isfinite([closed_span, open_span]).all() or not 0 <= closed_span < open_span:
            raise ValueError("Pinch spans must be finite and satisfy 0 <= closed < open")
        self.joint_names = tuple(f"{side.upper()}_FINGER{index}_JOINT" for index in (1, 2))
        joints = {joint.attrib["name"]: joint for joint in ET.parse(urdf_path).iter("joint")}
        selected = [joints[name] for name in self.joint_names]
        if any(joint.attrib["type"] != "prismatic" for joint in selected):
            raise ValueError("Parallel gripper must use two prismatic finger joints")
        self.lower = np.asarray([float(joint.find("limit").attrib["lower"]) for joint in selected])
        self.upper = np.asarray([float(joint.find("limit").attrib["upper"]) for joint in selected])
        if not np.isfinite([self.lower, self.upper]).all() or np.any(self.upper <= self.lower):
            raise ValueError("Gripper limits must be finite with positive travel")
        axes = [np.fromstring(joint.find("axis").attrib["xyz"], sep=" ") for joint in selected]
        if not np.allclose(axes[0], -axes[1]) or not np.isclose(np.linalg.norm(axes[0]), 1):
            raise ValueError("Expected opposite unit finger axes")
        mimic = selected[1].find("mimic")
        if (
            mimic is None
            or mimic.attrib["joint"] != self.joint_names[0]
            or float(mimic.get("multiplier", "1")) != 1
            or float(mimic.get("offset", "0")) != 0
            or not np.allclose(self.lower, self.lower[0])
            or not np.allclose(self.upper, self.upper[0])
        ):
            raise ValueError("Expected a symmetric one-to-one gripper mimic")
        self.closed_span = closed_span
        self.open_span = open_span
        self.q = self.upper.astype(np.float32)

    def reset(self, current_q: np.ndarray) -> None:
        """Seed the mapper with the held jaw coordinates [m]."""
        self.q = np.clip(current_q, self.lower, self.upper).astype(np.float32)

    def coordinates(self, closure: float) -> np.ndarray:
        """Map normalized closure (zero open, one closed) to both jaw coordinates [m]."""
        if not np.isfinite(closure):
            raise ValueError("Gripper closure must be finite")
        return (self.upper - np.clip(closure, 0, 1) * (self.upper - self.lower)).astype(np.float32)

    def closure(self, coordinates: np.ndarray) -> float:
        """Return normalized closure for a pair of jaw coordinates [m]."""
        return float(np.clip(1 - np.mean((coordinates - self.lower) / (self.upper - self.lower)), 0, 1))

    def solve(self, joints: np.ndarray) -> np.ndarray:
        """Follow thumb/index spacing while rejecting missing or degenerate skeletons."""
        joints = np.asarray(joints, dtype=np.float64)
        if joints.shape != (25, 3) or not np.isfinite(joints).all():
            raise ValueError("Expected 25 finite hand joint positions")
        palm_length = np.linalg.norm(joints[11] - joints[0])
        if not 0.03 <= palm_length <= 0.18 or np.any(np.linalg.norm(joints - joints[0], axis=1) > 0.35):
            raise ValueError("Invalid hand skeleton dimensions")
        span = float(np.linalg.norm(joints[4] - joints[9]))
        closure = 1 - (span - self.closed_span) / (self.open_span - self.closed_span)
        self.q = self.coordinates(closure)
        return self.q.copy()

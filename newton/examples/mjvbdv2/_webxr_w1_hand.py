# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental optical hand retargeting for the example's Dexforce W1 URDF.

Solve six independent coordinates using palm-local fingertip vectors, pinch
distances and a flexion prior. Expand the four mimic coordinates only after
solving. This module does not send commands to physical robot hardware.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HAND_SUFFIXES = (
    "HAND_THUMB2",
    "HAND_THUMB1",
    "HAND_INDEX",
    "INDEX_PIP",
    "HAND_MIDDLE",
    "MIDDLE_PIP",
    "HAND_RING",
    "RING_PIP",
    "HAND_PINKY",
    "PINKY_PIP",
)
_INDEPENDENT = np.array([0, 1, 2, 4, 6, 8])
_CHAINS = ((1, 2, 3, 4), (6, 7, 8, 9), (11, 12, 13, 14), (16, 17, 18, 19), (21, 22, 23, 24))


def _unit(vector):
    length = np.linalg.norm(vector)
    if length < 1.0e-6:
        raise ValueError("Degenerate hand skeleton")
    return vector / length


def _rotation(axis, angle):
    x, y, z = _unit(np.asarray(axis, dtype=float))
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def _origin(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        roll, pitch, yaw = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        transform[:3, :3] = _rotation([0, 0, 1], yaw) @ _rotation([0, 1, 0], pitch) @ _rotation([1, 0, 0], roll)
    return transform


class W1HandRetargeter:
    """Fit a W1 hand with bounded, warm-started damped least squares."""

    def __init__(self, urdf_path: Path, side: str):
        if side not in ("left", "right"):
            raise ValueError("Hand side must be left or right")
        root = ET.parse(urdf_path).getroot()
        by_name = {joint.get("name"): joint for joint in root.findall("joint")}
        joints = [by_name[f"{side.upper()}_{suffix}"] for suffix in HAND_SUFFIXES]
        self.origins = np.array([_origin(joint.find("origin")) for joint in joints])
        self.axes = np.array([np.fromstring(joint.find("axis").get("xyz"), sep=" ") for joint in joints])
        self.lower = np.array([float(joint.find("limit").get("lower")) for joint in joints])
        self.upper = np.array([float(joint.find("limit").get("upper")) for joint in joints])
        self.multipliers = np.array([float(joints[i].find("mimic").get("multiplier")) for i in (3, 5, 7, 9)])
        self.offsets = np.array([float(joints[i].find("mimic").get("offset", "0")) for i in (3, 5, 7, 9)])
        self.active_lower = self.lower[_INDEPENDENT].copy()
        self.active_upper = self.upper[_INDEPENDENT].copy()
        self.active_lower[2:] = np.maximum(self.active_lower[2:], (self.lower[3::2] - self.offsets) / self.multipliers)
        self.active_upper[2:] = np.minimum(self.active_upper[2:], (self.upper[3::2] - self.offsets) / self.multipliers)

        # Estimate fingertip centers from each distal collision mesh's far end.
        links = {link.get("name"): link for link in root.findall("link")}
        tips = []
        for i in (1, 3, 5, 7, 9):
            link = links[joints[i].find("child").get("link")]
            collision = link.find("collision")
            mesh = collision.find("geometry/mesh")
            path = urdf_path.parent / mesh.get("filename")
            vertices = np.array(
                [
                    np.fromstring(line[2:], sep=" ")[:3]
                    for line in path.read_text().splitlines()
                    if line.startswith("v ")
                ]
            )
            vertices *= np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
            transform = _origin(collision.find("origin"))
            vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
            distances = np.linalg.norm(vertices, axis=1)
            tips.append(vertices[distances >= np.quantile(distances, 0.95)].mean(axis=0))
        self.tips = np.array(tips)
        self.bases = self.origins[::2, :3, 3]
        radial = _unit(self.bases[1] - self.bases[4])
        forward = _unit(self.bases[2] - radial * np.dot(self.bases[2], radial))
        self.basis = np.column_stack((radial, forward, np.cross(radial, forward)))
        self.lengths = np.linalg.norm(self.origins[1::2, :3, 3], axis=1) + np.linalg.norm(self.tips, axis=1)
        self.palm_width = np.linalg.norm(self.bases[1] - self.bases[4])
        self.q = self.active_lower.copy()

    def expand(self, q):
        """Expand six independent angles [rad] to the ten named coordinates."""
        result = np.empty(10)
        result[_INDEPENDENT] = q
        result[3::2] = self.multipliers * q[2:] + self.offsets
        return result

    def fingertips(self, q):
        """Return five fingertip centers [m] in the hand base-offset frame."""
        full = self.expand(q)
        result = []
        for finger in range(5):
            transform = np.eye(4)
            for index in (2 * finger, 2 * finger + 1):
                rotation = np.eye(4)
                rotation[:3, :3] = _rotation(self.axes[index], full[index])
                transform = transform @ self.origins[index] @ rotation
            result.append(transform[:3, :3] @ self.tips[finger] + transform[:3, 3])
        return np.array(result)

    def reset(self, current_q):
        """Warm-start from the current ten simulation hand coordinates."""
        self.q = np.clip(np.asarray(current_q)[_INDEPENDENT], self.active_lower, self.active_upper)

    def solve(self, joints):
        """Fit one valid 25-joint WebXR skeleton, independent of world pose."""
        points = np.asarray(joints, dtype=float)
        if points.shape != (25, 3) or not np.all(np.isfinite(points)):
            raise ValueError("Expected 25 finite hand joint positions")
        width = np.linalg.norm(points[6] - points[21])
        if not 0.025 <= width <= 0.14:
            raise ValueError("Implausible tracked palm width")
        radial = _unit(points[6] - points[21])
        forward = points[11] - points[0]
        forward = _unit(forward - radial * np.dot(forward, radial))
        basis = np.column_stack((radial, forward, np.cross(radial, forward)))
        local = (points - points[0]) @ basis @ self.basis.T
        targets = []
        prior = self.q.copy()
        for finger, chain in enumerate(_CHAINS):
            segments = np.diff(local[list(chain)], axis=0)
            lengths = np.linalg.norm(segments, axis=1)
            if np.any(lengths < 0.003) or np.any(lengths > 0.09):
                raise ValueError("Implausible tracked finger bone length")
            targets.append(
                self.bases[finger] + (local[chain[-1]] - local[chain[0]]) * self.lengths[finger] / lengths.sum()
            )
            if finger:
                metacarpal = local[chain[0]] - local[chain[0] - 1]
                directions = np.array(
                    [_unit(metacarpal), *[segment / length for segment, length in zip(segments, lengths, strict=True)]]
                )
                curl = np.arccos(np.clip(np.sum(directions[:-1] * directions[1:], axis=1), -1, 1)).sum()
                prior[finger + 1] = (curl - self.offsets[finger - 1]) / (1 + self.multipliers[finger - 1])
        targets = np.array(targets)
        prior = np.clip(prior, self.active_lower, self.active_upper)
        pinch = np.linalg.norm(points[[9, 14, 19, 24]] - points[4], axis=1)
        pinch_targets = np.maximum(0.008, pinch * self.palm_width / width)
        pinch_weights = 2.0 * np.clip((0.065 - pinch) / 0.045, 0, 1)
        previous = self.q.copy()

        def residual(q):
            tips = self.fingertips(q)
            return np.concatenate(
                (
                    ((tips - targets) / 0.05).reshape(-1),
                    pinch_weights * (np.linalg.norm(tips[1:] - tips[0], axis=1) - pinch_targets) / 0.05,
                    0.3 * (q - prior),
                    0.08 * (q - previous),
                )
            )

        q = self.q.copy()
        for _ in range(4):
            error = residual(q)
            jacobian = np.column_stack([(residual(q + np.eye(6)[i] * 1.0e-4) - error) / 1.0e-4 for i in range(6)])
            step = np.linalg.solve(jacobian.T @ jacobian + 0.02 * np.eye(6), -jacobian.T @ error)
            step = np.clip(step, -0.25, 0.25)
            improved = False
            for scale in (1.0, 0.5, 0.25):
                candidate = np.clip(q + scale * step, self.active_lower, self.active_upper)
                candidate_error = residual(candidate)
                if candidate_error @ candidate_error < error @ error:
                    q = candidate
                    improved = True
                    break
            if not improved or np.linalg.norm(step) < 1.0e-4:
                break
        self.q = q
        return self.expand(q).astype(np.float32)

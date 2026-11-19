# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""CPU pose planning for the Piper arm; motor targets still use physical joints in the simulation."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

DEFAULT_ASSET = "robots/piper/piper_with_texture.xml"
DEFAULT_BASE = (0.0, -0.52, -0.0004)
DEFAULT_GRIP_POINT = (0.0, 0.0, 0.105)


def top_down_rotation(yaw=0.0):
    """Return a world rotation with finger extension along -Z and local finger width at ``yaw``."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array(((c, s, 0.0), (s, -c, 0.0), (0.0, 0.0, -1.0)))


def loading_drop_rotation(downward_tilt=np.pi / 6, yaw=0.0):
    """Point the fingers forward and down; the default 30-degree tilt reaches the bag opening.

    ``downward_tilt=pi/2`` points vertically down. Set ``yaw`` to the target's azimuth from the base
    while lifting an apple to keep the wrist on one continuous branch.
    """
    horizontal = np.array(((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    return Rotation.from_euler("z", yaw).as_matrix() @ Rotation.from_euler("x", -downward_tilt).as_matrix() @ horizontal


def _vector(element, name, default):
    return np.fromstring(element.get(name, default), sep=" ", dtype=np.float64)


def _transform(position, rotation):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = position
    return result


@dataclass(frozen=True)
class _Body:
    name: str
    parent: str | None
    rest: np.ndarray
    joint: int | None
    axis: np.ndarray
    anchor: np.ndarray
    slide: bool


class PiperKinematics:
    """Read the MJCF joint frames and solve bounded six-axis poses without creating a simulator.

    Rotations are world-from-wrist 3x3 matrices or quaternions in XYZW order.
    ``opening`` is each finger's travel, so 0.035 gives approximately 70 mm between the pads.
    IK checks the endpoint pose and joint limits; collision-free motion needs separate scene validation.
    """

    def __init__(self, asset_path=None, base_position=DEFAULT_BASE, base_yaw=np.pi / 2):
        if asset_path is None:
            # Keep asset downloads and simulator imports out of lightweight pose-module imports.
            from coupled_high_accuracy import get_asset

            asset_path = get_asset(DEFAULT_ASSET)
        root = ET.parse(asset_path).getroot()
        compiler = root.find("compiler")
        if compiler is not None and compiler.get("angle", "degree") != "radian":
            raise ValueError("PiperKinematics requires MJCF joint angles in radians")
        self.base_transform = _transform(
            np.asarray(base_position, dtype=np.float64), Rotation.from_euler("z", base_yaw).as_matrix()
        )
        self._base_yaw = float(base_yaw)
        self.limits = np.zeros((8, 2), dtype=np.float64)
        self._bodies = []

        def visit(element, parent):
            quaternion = _vector(element, "quat", "1 0 0 0")
            rotation = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
            rest = _transform(_vector(element, "pos", "0 0 0"), rotation)
            joint = element.find("joint")
            index, axis, anchor, slide = None, np.zeros(3), np.zeros(3), False
            if joint is not None:
                name = joint.get("name", "")
                if name not in {f"joint{i}" for i in range(1, 9)}:
                    raise ValueError(f"Unexpected Piper joint {name!r}")
                index = int(name[5:]) - 1
                axis = _vector(joint, "axis", "0 0 1")
                axis /= np.linalg.norm(axis)
                anchor = _vector(joint, "pos", "0 0 0")
                slide = joint.get("type", "hinge") == "slide"
                self.limits[index] = _vector(joint, "range", "0 0")
            self._bodies.append(_Body(element.get("name"), parent, rest, index, axis, anchor, slide))
            for child in element.findall("body"):
                visit(child, element.get("name"))

        base = root.find("worldbody/body[@name='base_link']")
        if base is None:
            raise ValueError("Piper MJCF is missing base_link")
        visit(base, None)
        by_name = {body.name: body for body in self._bodies}
        self._wrist_chain = []
        name = "link6"
        while name is not None:
            body = by_name[name]
            self._wrist_chain.append(body)
            name = body.parent
        self._wrist_chain.reverse()
        if {body.joint for body in self._bodies if body.joint is not None} != set(range(8)):
            raise ValueError("Piper MJCF must define joint1 through joint8")

    @staticmethod
    def _motion(body, q):
        if body.joint is None:
            return body.rest
        amount = q[body.joint]
        if body.slide:
            motion = _transform(amount * body.axis, np.eye(3))
        else:
            rotation = Rotation.from_rotvec(amount * body.axis).as_matrix()
            motion = _transform(body.anchor - rotation @ body.anchor, rotation)
        return body.rest @ motion

    def body_transforms(self, q):
        """Return world-from-body transforms, including both moving fingers, for eight joint coordinates."""
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (8,) or not np.isfinite(q).all():
            raise ValueError("Body FK requires eight finite joint coordinates")
        transforms = {}
        for body in self._bodies:
            parent = self.base_transform if body.parent is None else transforms[body.parent]
            transforms[body.name] = parent @ self._motion(body, q)
        return transforms

    def fk(self, q, grip_point_local=DEFAULT_GRIP_POINT):
        """Return a 4x4 world pose at the requested wrist-local grip point for six or eight coordinates."""
        q = np.asarray(q, dtype=np.float64)
        if q.shape not in ((6,), (8,)) or not np.isfinite(q).all():
            raise ValueError("Arm FK requires six or eight finite joint coordinates")
        result = self.base_transform.copy()
        for body in self._wrist_chain:
            result = result @ self._motion(body, q)
        result[:3, 3] += result[:3, :3] @ np.asarray(grip_point_local, dtype=np.float64)
        return result

    def solve_position(
        self, grip_position, seed, active_joints, grip_point_local=DEFAULT_GRIP_POINT, position_tolerance=2.0e-4
    ):
        """Solve translation with only selected arm joints, preserving every other motor coordinate exactly."""
        target = np.asarray(grip_position, dtype=np.float64)
        result = np.asarray(seed, dtype=np.float64).copy()
        active = np.asarray(active_joints)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError("Grip position must contain three finite coordinates")
        if result.shape != (8,) or not np.isfinite(result).all():
            raise ValueError("Position IK requires eight finite seed coordinates")
        if (
            active.ndim != 1
            or not len(active)
            or not np.issubdtype(active.dtype, np.integer)
            or len(np.unique(active)) != len(active)
            or np.any((active < 0) | (active >= 6))
        ):
            raise ValueError("Active joints must be distinct zero-based arm joint indices")
        if np.any(result < self.limits[:, 0]) or np.any(result > self.limits[:, 1]):
            raise ValueError("Position IK seed exceeds the joint limits")

        def residual(coordinates):
            pose = result.copy()
            pose[active] = coordinates
            return self.fk(pose, grip_point_local)[:3, 3] - target

        fit = least_squares(
            residual,
            result[active],
            bounds=self.limits[active].T,
            max_nfev=250,
            ftol=1.0e-11,
            xtol=1.0e-11,
            gtol=1.0e-11,
        )
        error = np.linalg.norm(residual(fit.x))
        if error > position_tolerance:
            raise ValueError(f"Piper position is unreachable with fixed joints: error {error:.6f} m")
        result[active] = fit.x
        return result

    def solve(
        self,
        grip_position,
        grip_rotation,
        seed=None,
        opening=0.035,
        grip_point_local=DEFAULT_GRIP_POINT,
        position_tolerance=2.0e-4,
        rotation_tolerance=2.0e-3,
    ):
        """Return eight bounded motor coordinates or raise if the target pose is unreachable.

        Pass the preceding solution as ``seed`` to preserve the elbow branch along a waypoint sequence.
        The orientation residual has a length scale of 0.15 m/rad to balance translation and rotation.
        """
        target_position = np.asarray(grip_position, dtype=np.float64)
        target_rotation = np.asarray(grip_rotation, dtype=np.float64)
        if target_position.shape != (3,) or not np.isfinite(target_position).all():
            raise ValueError("Grip position must contain three finite coordinates")
        if target_rotation.shape == (4,):
            target_rotation = Rotation.from_quat(target_rotation).as_matrix()
        if target_rotation.shape != (3, 3) or not np.isfinite(target_rotation).all():
            raise ValueError("Grip rotation must be a finite 3x3 matrix or XYZW quaternion")
        if not np.allclose(target_rotation.T @ target_rotation, np.eye(3), atol=1.0e-6):
            raise ValueError("Grip rotation must be orthonormal")
        if np.linalg.det(target_rotation) < 0.0:
            raise ValueError("Grip rotation must have positive determinant")
        fingers = np.asarray((opening, -opening), dtype=np.float64)
        if (
            not np.isfinite(fingers).all()
            or np.any(fingers < self.limits[6:, 0])
            or np.any(fingers > self.limits[6:, 1])
        ):
            raise ValueError("Requested opening exceeds the native finger joint limits")
        if seed is not None:
            seed = np.asarray(seed, dtype=np.float64)
            if seed.shape not in ((6,), (8,)) or not np.isfinite(seed).all():
                raise ValueError("IK seed must contain six or eight finite joint coordinates")
        lower, upper = self.limits[:6].T

        def residual(q):
            pose = self.fk(q, grip_point_local)
            rotation_error = Rotation.from_matrix(target_rotation.T @ pose[:3, :3]).as_rotvec()
            return np.r_[pose[:3, 3] - target_position, 0.15 * rotation_error]

        azimuth = (
            np.arctan2(target_position[1] - self.base_transform[1, 3], target_position[0] - self.base_transform[0, 3])
            - self._base_yaw
        )
        starts = [] if seed is None else [seed[:6]]
        starts.extend(
            (
                np.array((azimuth, 1.5, elbow, 0.0, wrist, roll))
                for elbow, wrist, roll in ((-1.5, 0.5, 0.0), (-1.8, 0.8, 1.57), (-2.4, 1.0, -1.57), (-0.8, -0.5, 0.0))
            )
        )
        best, best_error = None, np.inf
        for start in starts:
            fit = least_squares(
                residual,
                np.clip(start, lower + 1.0e-9, upper - 1.0e-9),
                bounds=(lower, upper),
                max_nfev=250,
                ftol=1.0e-11,
                xtol=1.0e-11,
                gtol=1.0e-11,
            )
            error = residual(fit.x)
            if np.linalg.norm(error) < best_error:
                best, best_error = fit.x, np.linalg.norm(error)
            if (
                np.linalg.norm(error[:3]) <= position_tolerance
                and np.linalg.norm(error[3:]) <= 0.15 * rotation_tolerance
            ):
                return np.r_[fit.x, fingers]
        error = residual(best)
        raise ValueError(
            f"Piper pose is unreachable: position error {np.linalg.norm(error[:3]):.6f} m, "
            f"rotation error {np.linalg.norm(error[3:]) / 0.15:.6f} rad"
        )

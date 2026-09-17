# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint-space timing for the conveyor example's kinematic robot."""

import math
import xml.etree.ElementTree as ET

import numpy as np


def interpolate_waypoints(points, fraction):
    """Interpolate a short Cartesian path with zero speed at its waypoints."""
    progress = np.clip(fraction, 0.0, 1.0) * (len(points) - 1)
    segment = min(int(progress), len(points) - 2)
    t = progress - segment
    blend = t * t * (3.0 - 2.0 * t)
    return (1.0 - blend) * points[segment] + blend * points[segment + 1]


def read_motion_limits(model, path, coordinate_count, *, finger_speed):
    """Read scalar robot joint limits directly from the source URDF."""
    definitions = {joint.get("name"): joint for joint in ET.parse(path).getroot().findall("joint")}
    lower = np.full(coordinate_count, -np.inf)
    upper = np.full(coordinate_count, np.inf)
    speed = np.full(coordinate_count, np.inf)
    starts = model.joint_q_start.numpy()
    for j, name in enumerate(model.joint_label):
        begin, end = starts[j : j + 2]
        if begin >= coordinate_count or begin == end:
            continue
        joint = definitions[name.rsplit("/", 1)[-1]]
        limit = joint.find("limit")
        if end - begin != 1 or limit is None:
            raise ValueError(f"Expected a scalar URDF joint with motion limits: {name}")
        lower[begin] = float(limit.get("lower", "-inf"))
        upper[begin] = float(limit.get("upper", "inf"))
        speed[begin] = float(limit.get("velocity", "nan"))
        if speed[begin] == 0.0 and ("HAND_" in name or name.endswith("_PIP")):
            speed[begin] = finger_speed
    if not np.all(np.isfinite(speed) & (speed > 0)):
        raise ValueError("Every robot coordinate requires a positive finite URDF velocity limit")
    return lower, upper, speed


class JointMotionRetimer:
    """Execute each requested joint segment over enough physical frames."""

    def __init__(self, speed, dt):
        self.speed = np.asarray(speed)
        if not math.isfinite(dt) or dt <= 0 or not np.all(np.isfinite(self.speed) & (self.speed > 0)):
            raise ValueError("Motion timing requires positive finite speeds and timestep")
        self.dt = dt
        self.remaining = 0

    def begin(self, start, goal):
        """Preserve the requested endpoint and stretch time to satisfy all joint speeds."""
        self.start = np.asarray(start).copy()
        self.goal = np.asarray(goal).copy()
        if self.start.shape != self.speed.shape or self.goal.shape != self.speed.shape:
            raise ValueError("Joint positions and speed limits must have matching shapes")
        if not np.isfinite(self.start).all() or not np.isfinite(self.goal).all():
            raise ValueError("Joint positions must be finite")
        self.frames = max(1, math.ceil(float(np.max(np.abs(self.goal - self.start) / (0.9 * self.speed * self.dt)))))
        self.remaining = self.frames

    def advance(self):
        """Advance one physical frame along the time-scaled segment."""
        if self.remaining <= 0:
            raise RuntimeError("Begin a motion segment before advancing")
        self.remaining -= 1
        fraction = (self.frames - self.remaining) / self.frames
        return (1.0 - fraction) * self.start + fraction * self.goal

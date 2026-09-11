# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check scene-local W1 gaze tracking independently of parcel simulation."""

import math
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.mjvbdv2.example_mjvbd_v2_conveyor_sorting import Example


class TestConveyorGaze(unittest.TestCase):
    def make_tracker(self, parent=None):
        """Build a head controller with the W1 neck limits and a movable target."""
        example = Example.__new__(Example)
        example.frame_dt = 1.0 / 60.0
        example.active = 0
        example.parcels = [object() for _ in range(4)]
        example.neck_parent = 0
        example.neck_origin = wp.transform_identity()
        example.neck_indices = np.array([1, 3])
        example.neck_lower = np.radians([-90.0, -45.0])
        example.neck_upper = np.radians([90.0, 25.0])
        example.neck_angles = np.zeros(2)
        example.neck_velocity = np.zeros(2)
        example.state_0 = SimpleNamespace(
            body_q=wp.array([parent or wp.transform_identity()], dtype=wp.transform, device="cpu")
        )
        return example

    def test_follow_in_neck_frame(self):
        """Track a target relative to the rotated torso and change only neck coordinates."""
        parent = wp.transform(wp.vec3(0.3, 0.1, 1.0), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.6))
        example = self.make_tracker(parent)
        target = wp.transform_point(parent, wp.vec3(0.7, -0.3, -0.2))
        example._center = lambda parcel: np.asarray(target)
        end = np.full(6, 0.123)
        for _ in range(300):
            example._track_parcel(end)
        expected = [math.atan2(-0.3, 0.7), math.atan2(-0.281, math.hypot(0.7, -0.3))]
        np.testing.assert_allclose(end[example.neck_indices], expected, atol=1.0e-4)
        np.testing.assert_array_equal(end[[0, 2, 4, 5]], np.full(4, 0.123))

    def test_switch_and_complete(self):
        """Respect mechanical limits and motion bounds when switching parcels and finishing."""
        example = self.make_tracker()
        end = np.zeros(6)
        targets = [(0.4, 1.0, 1.0), (0.4, -1.0, -1.0), (-1.0, -0.1, -2.0), (1.0, 0.0, 0.1)]
        example._center = lambda parcel: np.array(targets[example.parcels.index(parcel)])
        previous = example.neck_angles.copy()
        previous_velocity = example.neck_velocity.copy()
        for active in (0, 1, 2, 3, 4):
            example.active = active
            for _ in range(240):
                example._track_parcel(end)
                self.assertTrue(np.isfinite(end).all())
                self.assertTrue(np.all(example.neck_angles >= example.neck_lower))
                self.assertTrue(np.all(example.neck_angles <= example.neck_upper))
                self.assertLessEqual(float(np.max(np.abs(example.neck_angles - previous))), 0.70 / 60 + 1e-8)
                self.assertLessEqual(float(np.max(np.abs(example.neck_velocity))), 0.70 + 1e-8)
                self.assertLessEqual(float(np.max(np.abs(example.neck_velocity - previous_velocity))), 2 / 60 + 1e-8)
                previous = example.neck_angles.copy()
                previous_velocity = example.neck_velocity.copy()
        self.assertLess(abs(example.neck_angles[0]), 1e-4)


if __name__ == "__main__":
    unittest.main()

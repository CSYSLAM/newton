# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check timing and source-URDF limits without running the sorting simulation."""

import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.mjvbdv2.example_mjvbd_v2_conveyor_sorting import Example, _move_belt, _move_rollers
from newton.examples.mjvbdv2.support.conveyor_motion import JointMotionRetimer, read_motion_limits


class TestConveyorMotion(unittest.TestCase):
    def test_feed_waits_for_hand_reorientation(self):
        """An early parcel must not interrupt the empty hand's orientation change."""
        example = Example.__new__(Example)
        example.completed = False
        example.active = 0
        example.parcels = [SimpleNamespace(kind="rigid")]
        example.phase = "feed"
        example.phase_time = 0.5
        example.frame_dt = 1 / 60
        example.grasp_rotations = [wp.quat_identity()]
        example.feed_rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 1.0)
        example.target = np.zeros(3)
        example._center = lambda parcel: np.array([0.6, -0.2, 0.8])
        self.assertEqual(example._controller(), (0.0, 0.0))
        self.assertEqual(example.phase, "feed")
        example.phase_time = 2.5
        self.assertEqual(example._controller(), (0.0, 0.0))
        self.assertEqual(example.phase, "settle")

    def test_belt_end_geometry(self):
        """Keep moving flat sections inside the tangent span and rotate the end rollers."""
        device = "cpu"
        bodies = wp.array(np.arange(25), dtype=int, device=device)
        scales = wp.zeros(25, dtype=wp.vec3, device=device)
        transforms = wp.zeros(25, dtype=wp.transform, device=device)
        poses = wp.zeros(25, dtype=wp.transform, device=device)
        velocities = wp.zeros(25, dtype=wp.spatial_vector, device=device)
        for offset in (0.0, 0.03, 0.079, 0.15, 1.99):
            motion = wp.array([offset, 0.12], dtype=float, device=device)
            wp.launch(
                _move_belt, 25, [bodies, bodies, scales, transforms, motion, 1 / 480, poses, velocities], device=device
            )
            centers = poses.numpy()[:, 1] + transforms.numpy()[:, 1]
            half = scales.numpy()[:, 1]
            intervals = sorted(zip(centers - half, centers + half, strict=True))
            self.assertAlmostEqual(float(intervals[0][0]), -0.24, places=5)
            self.assertAlmostEqual(float(intervals[-1][1]), 1.76, places=5)
            for (_, end), (start, _) in itertools.pairwise(intervals):
                self.assertLessEqual(start, end + 1e-6)
            wp.launch(_move_rollers, 2, [bodies, motion, 1 / 480, poses, velocities], device=device)
            np.testing.assert_allclose(poses.numpy()[:2, :3], [[0.6, -0.24, 0.708], [0.6, 1.76, 0.708]], atol=1e-6)
            np.testing.assert_allclose(velocities.numpy()[:2, 3], 0.12 / 0.0521, atol=1e-6)

    def test_retime_without_losing_endpoint(self):
        """Stretch large joint changes and reach each endpoint without exceeding speeds."""
        speed = np.array([0.8, 0.5, 1.2])
        timer = JointMotionRetimer(speed, 1 / 60)
        previous = np.array([0.3, -0.7, 1.0])
        for goal in (np.array([-0.4, 0.1, 0.8]), previous.copy(), np.array([0.0, 0.0, 0.0])):
            timer.begin(previous, goal)
            while timer.remaining:
                current = timer.advance()
                self.assertTrue(np.all(np.abs(current - previous) <= speed / 60 + 1e-12))
                previous = current
            np.testing.assert_allclose(previous, goal, atol=1e-12)
        with self.assertRaises(RuntimeError):
            timer.advance()

    def test_invalid_timing(self):
        """Reject missing, zero, or nonfinite timing constraints."""
        for speed, dt in (([0], 0.1), ([np.nan], 0.1), ([np.inf], 0.1), ([1], 0)):
            with self.assertRaises(ValueError):
                JointMotionRetimer(speed, dt)

    def test_urdf_finger_override(self):
        """Use URDF arm speeds and an explicit fallback only for unspecified fingers."""
        model = SimpleNamespace(
            joint_label=["robot/RIGHT_J1", "robot/RIGHT_HAND_INDEX"],
            joint_q_start=SimpleNamespace(numpy=lambda: np.array([0, 1, 2])),
        )
        xml = '<robot><joint name="RIGHT_J1"><limit lower="-1" upper="1" velocity="0.8"/></joint><joint name="RIGHT_HAND_INDEX"><limit lower="0" upper="1.3" velocity="0"/></joint></robot>'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.urdf"
            path.write_text(xml)
            lower, upper, speed = read_motion_limits(model, path, 2, finger_speed=1.5)
            np.testing.assert_allclose(lower, [-1, 0])
            np.testing.assert_allclose(upper, [1, 1.3])
            np.testing.assert_allclose(speed, [0.8, 1.5])
            path.write_text(xml.replace('velocity="0.8"', 'velocity="0"'))
            with self.assertRaises(ValueError):
                read_motion_limits(model, path, 2, finger_speed=1.5)


if __name__ == "__main__":
    unittest.main()

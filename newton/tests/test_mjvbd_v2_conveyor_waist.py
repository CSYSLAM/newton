# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the scene-local W1 torso motion profile."""

import math
import unittest

from newton.examples.mjvbdv2.example_mjvbd_v2_conveyor_sorting import Example


class TestConveyorWaist(unittest.TestCase):
    def test_crouch_and_return(self):
        """Lower the torso smoothly and return without changing the net torso pitch."""
        example = Example.__new__(Example)
        example.frame_dt = 1.0 / 60.0
        example.stance_angle = example.stance_velocity = 0.0
        for active, target in ((2, 10.0), (3, 0.0)):
            example.active = active
            for _ in range(360):
                previous = example.stance_angle
                velocity = example.stance_velocity
                example._lower_body()
                self.assertLessEqual(abs(example.stance_angle - previous), 0.2 * example.frame_dt + 1e-8)
                self.assertLessEqual(abs(example.stance_velocity - velocity), 0.4 * example.frame_dt + 1e-8)
                self.assertLessEqual(abs(example.stance_velocity), 0.2 + 1e-8)
                self.assertAlmostEqual(sum(example.stance_angle * factor for factor in (1, -2, 1)), 0.0)
            self.assertAlmostEqual(example.stance_angle, math.radians(target), places=5)

    def test_turn_and_return(self):
        """Bound torso speed and acceleration through placement and return."""
        example = Example.__new__(Example)
        example.frame_dt = 1.0 / 60.0
        example.waist_angle = example.waist_velocity = 0.0
        for active, expected in ((0, 0.0), (1, 0.0), (2, -30.0), (3, -15.0)):
            example.active = active
            for phase, target in (("transfer", expected), ("release", expected), ("retreat", 0.0)):
                example.phase = phase
                for _ in range(360):
                    angle, velocity = example.waist_angle, example.waist_velocity
                    example._turn_waist()
                    self.assertLessEqual(abs(example.waist_velocity), 0.45 + 1.0e-8)
                    self.assertLessEqual(abs(example.waist_velocity - velocity), 0.8 * example.frame_dt + 1.0e-8)
                    self.assertLessEqual(abs(example.waist_angle - angle), 0.45 * example.frame_dt + 1.0e-8)
                    self.assertLessEqual(abs(example.waist_angle), math.radians(30.0) + 1.0e-8)
                self.assertAlmostEqual(example.waist_angle, math.radians(target), places=5)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the popcorn slip failure report without running the full scene."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.mjvbdv2.example_mjvbd_v2_popcorn import Example


class TestPopcornDiagnostics(unittest.TestCase):
    def test_cup_slip_reports_context_without_changing_state(self):
        """Report the original slip threshold and preserve physical arrays."""
        example = Example.__new__(Example)
        example.sim_time = 25.75
        example.cup_grasp = (wp.transform_identity(), np.zeros(3))
        example.wrists = [0, 1]
        example.grasp_force_filtered = np.arange(1.0, 6.0)
        example.grasp_joint_offset = np.zeros(5)
        example.args = SimpleNamespace(substeps=8, vbd_iterations=8, popcorn_count=160)
        state = {
            "body_q": np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32),
            "particle_q": np.array([[0, 0, -0.0314]], dtype=np.float32),
        }
        before = {name: value.copy() for name, value in state.items()}
        example._read_state = state.__getitem__
        with self.assertRaises(AssertionError) as failure:
            example.step()
        message = str(failure.exception)
        self.assertIn("25.750s: 31.4 mm", message)
        self.assertIn("wrist_local_delta_mm=", message)
        self.assertIn("filtered_digit_force_N=[1.0, 2.0, 3.0, 4.0, 5.0]", message)
        self.assertIn("digit_offset_deg=", message)
        self.assertIn("substeps=8, iterations=8, grains=160", message)
        for name, value in state.items():
            np.testing.assert_array_equal(value, before[name])


if __name__ == "__main__":
    unittest.main()

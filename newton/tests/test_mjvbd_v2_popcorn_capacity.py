# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check contact overflow reporting without constructing the robot scene."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.mjvbdv2.example_mjvbd_v2_popcorn import (
    CUP_GRIP_MAX_OFFSET,
    Example,
    _grasp_offset_update,
    record_contact_peaks,
)


class TestPopcornCapacity(unittest.TestCase):
    def test_peak_retains_transient_overflow(self):
        """Retain a substep overflow even after subsequent counts decrease."""
        peaks = wp.zeros(2, dtype=int, device="cpu")
        for rigid, soft in ((40961, 100), (1, 8193), (0, 0)):
            wp.launch(
                record_contact_peaks,
                1,
                [wp.array([rigid], dtype=int, device="cpu"), wp.array([soft], dtype=int, device="cpu"), peaks],
                device="cpu",
            )
        np.testing.assert_array_equal(peaks.numpy(), [40961, 8193])
        example = Example.__new__(Example)
        example.contact_peaks = peaks
        example.solver = SimpleNamespace(contacts=SimpleNamespace(rigid_contact_max=40960, soft_contact_max=8192))
        with self.assertRaisesRegex(AssertionError, "contact capacity exceeded"):
            example._measure()

    def test_default_population(self):
        """Select 640 grains while preserving explicit population overrides."""
        parser = Example.create_parser()
        self.assertEqual(parser.parse_args([]).popcorn_count, 640)
        self.assertEqual(parser.parse_args(["--popcorn-count", "320"]).popcorn_count, 320)

    def test_pinky_travel_preserves_force_feedback(self):
        """Permit needed closure without changing the force target or release."""
        offset = CUP_GRIP_MAX_OFFSET.copy()
        limits = offset.copy()
        limits[4] = np.radians(26)
        forces = np.array([4, 1, 1, 1, 0.4])
        original = _grasp_offset_update(offset, forces, True)
        extended = _grasp_offset_update(offset, forces, True, max_offset=limits)
        self.assertEqual(original[4], offset[4])
        self.assertGreater(extended[4], offset[4])
        self.assertLessEqual(extended[4] - offset[4], 0.06 / 60)
        released = _grasp_offset_update(offset, np.full(5, 100.0), True, max_offset=limits)
        self.assertTrue(np.all(released < offset))
        np.testing.assert_array_equal(limits[:4], CUP_GRIP_MAX_OFFSET[:4])


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for grasp IK tracking and untruncated updates."""

import math
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
import newton.viewer
from newton._src.solvers.mjvbd_v2.vbd.particle_vbd_kernels import (
    apply_truncation_ts,
    apply_untruncated_displacements,
)
from newton.examples.mjvbdv2.example_mjvbd_v2_inflatable_bag_grasp import Example


class TestInflatableGrasp(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Full pneumatic grasp acceptance requires CUDA")
    @unittest.skipUnless(
        (Path(newton.__file__).parent.parent / "assets/DexforceW1V021/DexforceW1V021.urdf").is_file(),
        "Full grasp acceptance requires the local Dexforce W1 asset",
    )
    def test_complete_grasp_ik_tracking(self):
        """Verify IK tracking throughout the original grasp and release motion."""
        viewer = newton.viewer.ViewerNull()
        try:
            example = Example(viewer, newton.examples.default_args(Example.create_parser()))
            for _ in range(math.ceil(example.script_duration / example.frame_dt) + 2):
                example.step()
                example.test_post_step()
            # Full physical acceptance remains in the scene test_final().
            # This regression checks tracking throughout the original motion.
            self.assertIsNotNone(example.ik_graph)
        finally:
            viewer.close()

    def test_untruncated_update_matches_reference_with_aliasing(self):
        """Match reference updates with aliased positions and optional output."""
        rng = np.random.default_rng(123)
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            pos = wp.array(rng.normal(size=(257, 3)).astype(np.float32), dtype=wp.vec3, device=device)
            displacement = wp.array(rng.normal(size=(257, 3)).astype(np.float32), dtype=wp.vec3, device=device)
            displacement_before = displacement.numpy().copy()
            factors = wp.full(257, 1.0, dtype=float, device=device)
            expected = wp.clone(pos)
            excluded = wp.full(257, 7, dtype=int, device=device)
            status = wp.array([3], dtype=int, device=device)
            wp.launch(
                apply_truncation_ts,
                dim=257,
                inputs=[expected, displacement, factors, wp.inf],
                outputs=[displacement, expected, excluded, status],
                device=device,
            )
            factors.fill_(0.25)
            wp.launch(
                apply_untruncated_displacements,
                dim=257,
                inputs=[pos, displacement],
                outputs=[factors, pos],
                device=device,
            )
            np.testing.assert_array_equal(pos.numpy(), expected.numpy())
            np.testing.assert_array_equal(displacement.numpy(), displacement_before)
            np.testing.assert_array_equal(factors.numpy(), 1.0)
            np.testing.assert_array_equal(excluded.numpy(), 7)
            np.testing.assert_array_equal(status.numpy(), [3])
            factors.zero_()
            wp.launch(
                apply_untruncated_displacements,
                dim=257,
                inputs=[pos, displacement],
                outputs=[factors, None],
                device=device,
            )
            np.testing.assert_array_equal(pos.numpy(), expected.numpy())
            np.testing.assert_array_equal(factors.numpy(), 1.0)


if __name__ == "__main__":
    unittest.main()

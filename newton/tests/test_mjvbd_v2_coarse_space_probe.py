# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate the offline coarse-space experiment independently of a demo."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2.contact_projection import ContactProjectionData
from scripts.diagnose_mjvbd_v2_coarse_space import _matrix_product, local_residual_step_bound, ritz_step


class TestCoarseSpaceProbe(unittest.TestCase):
    def test_local_residual_bound(self):
        """Select the largest unit-bounded step that preserves the frozen residual norm."""
        local = np.array([[1.0, 0.0, 0.0]])
        image = 4.0 * local
        alpha = local_residual_step_bound(local, image)
        self.assertEqual(alpha, 0.5)
        self.assertLessEqual(np.linalg.norm(local - alpha * image), np.linalg.norm(local))
        self.assertGreater(np.linalg.norm(local - (alpha + 0.01) * image), np.linalg.norm(local))
        self.assertEqual(local_residual_step_bound(local, -image), 0.0)
        self.assertEqual(local_residual_step_bound(local, np.zeros_like(image)), 1.0)
        self.assertEqual(local_residual_step_bound(local, np.full_like(image, np.nan)), 0.0)

    def test_ritz_matches_restricted_quadratic(self):
        """Match dense minimization and preserve invariance to basis scaling."""
        rng = np.random.default_rng(29)
        factor = rng.normal(size=(18, 18))
        matrix = factor.T @ factor + np.eye(18)
        force = rng.normal(size=(6, 3))
        directions = rng.normal(size=(3, 6, 3))
        products = np.array([(matrix @ d.ravel()).reshape(6, 3) for d in directions])
        basis = directions.reshape(3, -1).T
        reference = basis @ np.linalg.solve(basis.T @ matrix @ basis, basis.T @ force.ravel())
        result, _, _ = ritz_step(directions, products, force)
        np.testing.assert_allclose(result.ravel(), reference, atol=1e-12)
        scales = np.array([1e-4, -3, 1e3])[:, None, None]
        result, _, _ = ritz_step(directions * scales, products * scales, force)
        np.testing.assert_allclose(result.ravel(), reference, atol=1e-12)
        for direction in directions:
            self.assertAlmostEqual(float(direction.ravel() @ (matrix @ result.ravel() - force.ravel())), 0.0)

    def test_reject_dependent_or_indefinite_space(self):
        """Reject a singular or nonpositive reduced operator instead of amplifying it."""
        direction = np.ones((2, 3))
        with self.assertRaises(ValueError):
            ritz_step([direction, direction], [direction, direction], direction)
        with self.assertRaises(ValueError):
            ritz_step([direction], [-direction], direction)

    def test_matrix_product_matches_dense(self):
        """Match the diagnostic block sparse multiply on CPU and CUDA."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                blocks = np.array([2 * np.eye(3), -np.eye(3), -np.eye(3), 3 * np.eye(3)], dtype=np.float32)
                vector = np.arange(6, dtype=np.float32).reshape(2, 3)
                result = wp.empty(2, dtype=wp.vec3, device=device)
                wp.launch(
                    _matrix_product,
                    dim=2,
                    inputs=[
                        wp.array([0, 2, 4], dtype=wp.int32, device=device),
                        wp.array([0, 1, 0, 1], dtype=wp.int32, device=device),
                        wp.array(blocks, dtype=wp.mat33, device=device),
                        ContactProjectionData(),
                        wp.array(vector, dtype=wp.vec3, device=device),
                    ],
                    outputs=[result],
                    device=device,
                )
                expected = np.array([2 * vector[0] - vector[1], -vector[0] + 3 * vector[1]])
                np.testing.assert_allclose(result.numpy(), expected)


if __name__ == "__main__":
    unittest.main()

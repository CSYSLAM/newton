# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the projected operator and direct solve against double precision."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp
from rigid_ritz_probe import assess_solution, project_operator, solve_dense, symmetrize
from two_level_probe import TwoLevelPCG

from newton._src.solvers.mjvbd_v2.contact_projection import ContactProjection, ContactProjectionData, project_records


class TestRigidRitz(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Tile Cholesky requires CUDA")
    def test_projection_and_solve(self):
        """Match a double-precision Galerkin projection including cross contacts."""
        self._check_projection(False)
        self._check_projection(True)

    def _check_projection(self, with_contact):
        device = "cuda:0"
        rng = np.random.default_rng(48)
        count, dofs = 12, 12
        positions = rng.normal(size=(count, 3)) * 0.03
        groups = np.repeat(np.arange(2), 6).astype(np.int32)
        centers = np.array([positions[groups == group].mean(axis=0) for group in range(2)])
        raw = rng.normal(size=(3 * count, 3 * count))
        hessian = raw.T @ raw + 20 * np.eye(3 * count)
        force = rng.normal(size=(count, 3))
        basis = np.zeros((3 * count, dofs))
        for particle, group in enumerate(groups):
            rows = slice(3 * particle, 3 * particle + 3)
            basis[rows, 6 * group : 6 * group + 3] = np.eye(3)
            basis[rows, 6 * group + 3 : 6 * group + 6] = np.cross(
                np.eye(3), (positions[particle] - centers[group]) / 0.05
            ).T
        expected_hessian = hessian.copy()
        contact_data = ContactProjectionData()
        if with_contact:
            contact = ContactProjection(count, 4, device)
            contact.reset()
            weights = np.zeros(count)
            weights[[0, 1, 6]] = [0.2, 0.3, 0.5]
            hessian += np.kron(np.diag(weights * weights), 2 * np.eye(3))
            expected_hessian += np.kron(np.outer(weights, weights), 2 * np.eye(3))
            wp.launch(
                project_records,
                dim=1,
                inputs=[
                    wp.array([[0, 1, 6, -1]], dtype=wp.vec4i, device=device),
                    wp.array([[0.2, 0.3, 0.5, 0]], dtype=wp.vec4, device=device),
                    wp.array([2 * np.eye(3)], dtype=wp.mat33, device=device),
                    wp.array(np.arange(count), dtype=int, device=device),
                    contact.data,
                ],
                device=device,
            )
            contact_data = contact.data
        expected_matrix, expected_rhs = basis.T @ expected_hessian @ basis, basis.T @ force.reshape(-1)
        blocks = hessian.reshape(count, 3, count, 3).transpose(0, 2, 1, 3).reshape(-1, 3, 3)
        matrix, symmetric = wp.zeros((32, 32), device=device), wp.empty((32, 32), device=device)
        rhs, solution = wp.zeros(32, device=device), wp.zeros(32, device=device)
        status, counters, metrics = (
            wp.zeros(1, dtype=int, device=device),
            wp.zeros(2, dtype=int, device=device),
            wp.zeros(8, device=device),
        )
        wp.launch(
            project_operator,
            dim=128 * dofs,
            block_dim=64,
            inputs=[
                wp.array([0, 6, 12], dtype=int, device=device),
                wp.array(np.arange(count), dtype=int, device=device),
                wp.array(groups, dtype=int, device=device),
                wp.array(centers, dtype=wp.vec3, device=device),
                wp.array(positions, dtype=wp.vec3, device=device),
                wp.array(np.arange(count + 1) * count, dtype=int, device=device),
                wp.array(np.tile(np.arange(count), count), dtype=int, device=device),
                wp.array(blocks, dtype=wp.mat33, device=device),
                wp.array(force, dtype=wp.vec3, device=device),
                contact_data,
                dofs,
            ],
            outputs=[matrix, rhs],
            device=device,
        )
        np.testing.assert_allclose(matrix.numpy()[:dofs, :dofs], expected_matrix, rtol=2e-5, atol=2e-4)
        np.testing.assert_allclose(rhs.numpy()[:dofs], expected_rhs, rtol=2e-5, atol=2e-5)
        wp.launch(symmetrize, dim=(32, 32), inputs=[matrix, dofs], outputs=[symmetric], device=device)
        wp.launch(
            solve_dense,
            dim=64,
            block_dim=64,
            inputs=[symmetric, rhs],
            outputs=[solution, status, counters, metrics],
            device=device,
        )
        wp.launch(
            assess_solution,
            dim=32,
            block_dim=32,
            inputs=[symmetric, rhs, solution],
            outputs=[status, metrics],
            device=device,
        )
        self.assertEqual(int(status.numpy()[0]), 0)
        np.testing.assert_allclose(
            solution.numpy()[:dofs], np.linalg.solve(expected_matrix, expected_rhs), rtol=2e-5, atol=2e-6
        )
        # Reuse the independent fine SPD matrix, not a solver-generated oracle.
        fake_ritz = SimpleNamespace(
            model=SimpleNamespace(device=device),
            project=lambda launch, inputs: None,
            q=wp.array(positions, dtype=wp.vec3, device=device),
            centers=wp.array(centers, dtype=wp.vec3, device=device),
            groups=wp.array(groups, dtype=int, device=device),
            symmetric=symmetric,
            rhs=rhs,
        )
        pcg = TwoLevelPCG(fake_ritz)
        inputs = [
            count,
            wp.array(np.arange(count + 1) * count, dtype=int, device=device),
            wp.array(np.tile(np.arange(count), count), dtype=int, device=device),
            wp.array(np.arange(count) * (count + 1), dtype=int, device=device),
            wp.array(blocks, dtype=wp.mat33, device=device),
            wp.array(force, dtype=wp.vec3, device=device),
            24,
            True,
            1e-4,
            contact_data,
        ]
        outputs = [wp.zeros(count, dtype=wp.vec3, device=device) for _ in range(5)]
        outputs += [
            wp.zeros(count, dtype=wp.mat33, device=device),
            wp.zeros(1, dtype=int, device=device),
            wp.zeros(32, device=device),
            wp.zeros(2, dtype=int, device=device),
        ]
        pcg.solve(inputs, outputs)
        with wp.ScopedCapture(device=device) as capture:
            pcg.solve(inputs, outputs)
        for _ in range(2):
            wp.capture_launch(capture.graph)
            self.assertEqual(int(outputs[6].numpy()[0]), 0)
            actual = outputs[0].numpy().reshape(-1)
            np.testing.assert_allclose(
                actual, np.linalg.solve(expected_hessian, force.reshape(-1)), rtol=1e-4, atol=2e-6
            )
            relative_residual = np.linalg.norm(expected_hessian @ actual - force.reshape(-1)) / np.linalg.norm(force)
            self.assertLess(relative_residual, 1e-5)


if __name__ == "__main__":
    unittest.main()

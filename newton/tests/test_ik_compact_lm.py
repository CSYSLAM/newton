# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare masked LM elimination with the original full solve and a dense oracle."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.sim.ik.compact_lm import CompactLMSolve
from newton._src.sim.ik.ik_lm_optimizer import IKOptimizerLM


class TestCompactLM(unittest.TestCase):
    def test_eliminate_only_zero_columns(self):
        """Retain active updates, predicted reduction and exactly zero locked updates."""
        rng = np.random.default_rng(23)
        mask = np.zeros(40, dtype=bool)
        mask[rng.choice(40, 14, replace=False)] = True
        device = "cuda:0"
        owner = SimpleNamespace(
            n_batch=2,
            n_dofs=40,
            n_residuals=64,
            device=wp.get_device(device),
            TILE_THREADS=32,
            joint_dof_mask=wp.array(mask, dtype=bool, device=device),
        )
        j = rng.normal(size=(2, 64, 40)).astype(np.float32)
        j[:, :, ~mask] = 0
        r = rng.normal(size=(2, 64, 1)).astype(np.float32)
        lam = np.array([0.001, 10.0], dtype=np.float32)
        jac, residuals, lambdas = [wp.array(x, device=device) for x in (j, r, lam)]
        full = IKOptimizerLM._build_specialized((40, 64, "cuda"))
        compact = CompactLMSolve(owner)
        results = []
        for solve in (lambda *args: full._solve_tiled(owner, *args), compact):
            delta = wp.zeros((2, 40), device=device)
            predicted = wp.zeros(2, device=device)
            solve(jac, residuals, lambdas, delta, predicted)
            with wp.ScopedCapture(device=device) as capture:
                solve(jac, residuals, lambdas, delta, predicted)
            for _ in range(2):
                wp.capture_launch(capture.graph)
            results.append((delta.numpy(), predicted.numpy()))
        np.testing.assert_allclose(results[1][0], results[0][0], rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(results[1][1], results[0][1], rtol=2e-5, atol=2e-6)
        np.testing.assert_array_equal(results[1][0][:, ~mask], 0)
        for batch in range(2):
            matrix = j[batch].astype(np.float64)
            rhs = matrix.T @ r[batch, :, 0]
            expected = np.linalg.solve(matrix.T @ matrix + lam[batch] * np.eye(40), -rhs)
            np.testing.assert_allclose(results[1][0][batch], expected, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()

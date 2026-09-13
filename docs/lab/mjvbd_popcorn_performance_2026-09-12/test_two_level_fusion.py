# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare fused PCG updates against the two-launch reference across a full CTA."""

import unittest

import numpy as np
import warp as wp
from two_level_probe import precondition_and_direction, update_and_restrict, update_precondition_direction


class TestFusion(unittest.TestCase):
    def test_full_size_update(self):
        """Preserve every PCG work array for multiple rows per thread and three clusters."""
        rng = np.random.default_rng(41)
        n = 1002
        device = "cuda:0"
        q = wp.array(rng.normal(size=(n, 3)), dtype=wp.vec3, device=device)
        centers = wp.zeros(3, dtype=wp.vec3, device=device)
        groups = wp.array(np.concatenate([np.arange(841) // 400, np.full(161, -1)]), dtype=int, device=device)
        matrix = wp.array(np.eye(32, dtype=np.float32), device=device)
        inverse = wp.array(np.tile(np.eye(3), (n, 1, 1)), dtype=wp.mat33, device=device)
        for iteration in (-1, 0, 3):
            base = [wp.array(rng.normal(size=(n, 3)), dtype=wp.vec3, device=device) for _ in range(5)]
            # Positive p^T A p.
            wp.copy(base[4], base[3])
            outputs = []
            for fused in (False, True):
                solution, residual, preconditioned, direction, product = [wp.clone(x) for x in base]
                status = wp.zeros(1, dtype=int, device=device)
                metrics = wp.zeros(20, device=device)
                rz = wp.ones(1, device=device)
                coarse_rhs = wp.zeros(32, device=device)
                if fused:
                    wp.launch(
                        update_precondition_direction,
                        dim=256,
                        block_dim=256,
                        inputs=[
                            n,
                            iteration,
                            q,
                            centers,
                            groups,
                            matrix,
                            solution,
                            residual,
                            preconditioned,
                            direction,
                            product,
                            inverse,
                            status,
                            metrics,
                            rz,
                        ],
                        device=device,
                    )
                else:
                    wp.launch(
                        update_and_restrict,
                        dim=256,
                        block_dim=256,
                        inputs=[
                            n,
                            iteration < 0,
                            q,
                            centers,
                            groups,
                            solution,
                            residual,
                            direction,
                            product,
                            status,
                            rz,
                            coarse_rhs,
                        ],
                        device=device,
                    )
                    wp.launch(
                        precondition_and_direction,
                        dim=256,
                        block_dim=256,
                        inputs=[
                            n,
                            iteration,
                            q,
                            centers,
                            groups,
                            matrix,
                            coarse_rhs,
                            residual,
                            preconditioned,
                            direction,
                            inverse,
                            status,
                            metrics,
                            rz,
                        ],
                        device=device,
                    )
                outputs.append([x.numpy() for x in (solution, residual, preconditioned, direction, rz, status)])
            for expected, actual in zip(*outputs, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

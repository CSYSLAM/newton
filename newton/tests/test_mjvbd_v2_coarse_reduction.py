# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check block-parallel Galerkin restriction against a double precision sum."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2 import particle_multilevel as ml


class TestCoarseReduction(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Tile reductions require CUDA")
    def test_ragged_clusters_and_graph_replay(self):
        """Cover empty, partial, and multi-block-sized clusters with permuted slots."""
        rng = np.random.default_rng(942)
        sizes = np.array([0, 1, 63, 64, 65, 400, 1025])
        offsets = np.concatenate(([0], np.cumsum(sizes))).astype(np.int32)
        count = int(offsets[-1])
        particles = rng.permutation(count).astype(np.int32)
        factors = rng.normal(size=(count, 3, 3)).astype(np.float32)
        hessian = factors @ factors.transpose(0, 2, 1) + np.eye(3, dtype=np.float32)
        corrections = rng.normal(size=(count, 3)).astype(np.float32)
        slots = rng.permutation(len(sizes)).astype(np.int32)
        device = "cuda:0"
        delta = wp.array(corrections, dtype=wp.vec3, device=device)
        args = [
            wp.array(offsets, dtype=int, device=device),
            wp.array(particles, dtype=int, device=device),
            delta,
            wp.array(hessian, dtype=wp.mat33, device=device),
            wp.array(slots, dtype=int, device=device),
        ]
        rhs = wp.zeros(len(sizes), dtype=wp.vec3, device=device)
        blocks = wp.zeros(len(sizes), dtype=wp.mat33, device=device)

        def launch():
            wp.launch(
                ml._restrict_energy_galerkin_tiled,
                dim=len(sizes) * 64,
                block_dim=64,
                inputs=args,
                outputs=[rhs, blocks],
                device=device,
            )

        launch()
        with wp.ScopedCapture(device=device) as capture:
            launch()
        for scale in (1.0, -0.5):
            delta.assign(corrections * scale)
            wp.capture_launch(capture.graph)
            expected_rhs = np.zeros((len(sizes), 3))
            expected_blocks = np.zeros((len(sizes), 3, 3))
            for cluster in range(len(sizes)):
                ids = particles[offsets[cluster] : offsets[cluster + 1]]
                expected_rhs[cluster] = np.einsum(
                    "nij,nj->ni", hessian[ids].astype(float), corrections[ids].astype(float) * scale
                ).sum(axis=0)
                expected_blocks[slots[cluster]] = hessian[ids].astype(float).sum(axis=0)
            np.testing.assert_allclose(rhs.numpy(), expected_rhs, rtol=2e-5, atol=1e-4)
            np.testing.assert_allclose(blocks.numpy(), expected_blocks, rtol=2e-5, atol=1e-4)


if __name__ == "__main__":
    unittest.main()

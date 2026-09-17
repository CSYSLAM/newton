# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify reused bending gradients preserve every cross block."""

import unittest

import numpy as np
import warp as wp
from bending_assembly_probe import assemble_bending

from newton._src.solvers.mjvbd_v2.particle_multilevel import _assemble_bending_energy_galerkin


class TestBendingAssembly(unittest.TestCase):
    def test_shared_gradients(self):
        """Match random deformed edges, invalid boundaries and projected shared slots."""
        rng = np.random.default_rng(829)
        n = 512
        device = "cuda:0"
        positions = wp.array(rng.normal(size=(4 * n, 3)) * 0.02, dtype=wp.vec3, device=device)
        indices = np.arange(4 * n, dtype=np.int32).reshape(n, 4)
        indices[::19, 0] = -1
        indices[::23, 1] = -1
        edges = wp.array(indices, device=device)
        lengths = wp.array(rng.uniform(0.01, 0.03, n), dtype=float, device=device)
        material = rng.uniform(0.01, 2.0, (n, 2)).astype(np.float32)
        material[::29, 0] = 0.0
        materials = wp.array(material, device=device)
        for collapsed in (False, True):
            slots = np.arange(n * 16, dtype=np.int32).reshape(n, 4, 4)
            if collapsed:
                slots = np.minimum(slots, slots.transpose(0, 2, 1))
            slots[::17, 0, 1] = -1
            slot_array = wp.array(slots.reshape(-1), device=device)
            results = []
            for kernel, dim in ((_assemble_bending_energy_galerkin, 6 * n), (assemble_bending, n)):
                blocks = wp.zeros(16 * n, dtype=wp.mat33, device=device)
                wp.launch(
                    kernel,
                    dim=dim,
                    inputs=[1 / 480, positions, edges, lengths, materials, slot_array, blocks],
                    device=device,
                )
                results.append(blocks.numpy())
            np.testing.assert_allclose(results[1], results[0], rtol=2e-6, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the established independent reaction oracle against the dense prototype."""

import unittest
from unittest import mock

import warp as wp
from dense_single_probe import accumulate_body_particle_contact_dense_single

from newton._src.solvers.mjvbd_v2.vbd import rigid_vbd_kernels as rk
from newton.tests.test_mjvbd_v2_contact_optimizations import TestMJVBDV2ContactOptimizations


class TestDenseSingle(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Tile reduction requires CUDA")
    def test_active_prefix_matches_independent_reaction(self):
        original = wp.launch
        pending = None

        def launch(kernel, *args, **kwargs):
            nonlocal pending
            if kernel is rk.accumulate_body_particle_contact_dense_partials:
                pending = kwargs
                return None
            if kernel is rk.accumulate_body_particle_contact_dense_reduction:
                inputs = pending["inputs"]
                result = original(
                    accumulate_body_particle_contact_dense_single,
                    dim=inputs[1].size * 64,
                    block_dim=64,
                    inputs=inputs,
                    outputs=kwargs["outputs"],
                    device=kwargs["device"],
                )
                pending = None
                return result
            return original(kernel, *args, **kwargs)

        oracle = TestMJVBDV2ContactOptimizations()
        with mock.patch.object(wp, "launch", launch):
            for count in (128, 129, 257, 1024):
                with self.subTest(count=count):
                    oracle._check_dense_body_particle_reduction(count)


if __name__ == "__main__":
    unittest.main()

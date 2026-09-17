# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prevent small tet color groups from disabling tiled cloth solves."""

import unittest

import numpy as np
import warp as wp

from newton.tests.test_mjvbd_v2_particle_multilevel import SolverVBDComplete, SolverVBDSoft, _build_tets


class TestMJVBDV2SmallTileGroups(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Tiled solves require CUDA")
    def test_group_size_boundaries_with_surface_cache(self):
        """Preserve mixed-model dynamics with small groups, damping and fixed vertices."""
        for count in (1, 15, 16, 17):
            model = _build_tets("cuda:0", include_surface=True, tet_count=count)
            bending = model.edge_bending_properties.numpy()
            bending[:, 0], bending[:, 1] = 0.1, 0.002
            model.edge_bending_properties.assign(bending)
            flags = model.particle_flags.numpy()
            flags[0] = 0
            model.particle_flags.assign(flags)
            mass, inverse_mass = model.particle_mass.numpy(), model.particle_inv_mass.numpy()
            mass[1], inverse_mass[1] = 0.0, 0.0
            model.particle_mass.assign(mass)
            model.particle_inv_mass.assign(inverse_mass)
            initial = model.particle_q.numpy()
            initial += np.random.default_rng(617).normal(0, 0.0001, initial.shape).astype(np.float32)
            force = np.zeros_like(initial)
            force[-1] = (0.001, 0.002, -0.001)
            for solver_type in (SolverVBDComplete, SolverVBDSoft):
                for cached in (False, True):
                    with self.subTest(count=count, solver=solver_type.__module__, surface_cache=cached):
                        outputs = []
                        for enabled in (False, True):
                            solver = solver_type(
                                model,
                                iterations=6,
                                particle_enable_tile_solve=enabled,
                                particle_enable_surface_cache=cached,
                            )
                            self.assertEqual(solver.use_particle_tile_solve, enabled)
                            state_in, state_out = model.state(), model.state()
                            state_in.particle_q.assign(initial)
                            control = model.control()
                            for _ in range(12):
                                state_in.particle_f.assign(force)
                                solver.step(state_in, state_out, control, None, 1.0 / 600.0)
                                state_in, state_out = state_out, state_in
                            outputs.append(state_in.particle_q.numpy())
                            self.assertTrue(np.isfinite(outputs[-1]).all())
                        np.testing.assert_allclose(*outputs, rtol=1.0e-5, atol=2.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "Tiled solves require CUDA")
    def test_small_groups_match_scalar_and_replay(self):
        """Solve singleton tet groups alongside cloth, including CUDA graph replay."""
        model = _build_tets("cuda:0", include_surface=True, tet_count=1)
        initial = model.particle_q.numpy()
        initial[3, 2] += 0.003
        initial[-1, 2] += 0.005
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__):
                outputs = []
                for enabled in (False, True):
                    solver = solver_type(model, iterations=6, particle_enable_tile_solve=enabled)
                    self.assertEqual(solver.use_particle_tile_solve, enabled)
                    self.assertEqual(min(g.size for g in solver.volumetric_particle_color_groups if g.size), 1)
                    self.assertGreater(max(g.size for g in solver.surface_particle_color_groups), 16)
                    state_in, state_out = model.state(), model.state()
                    state_in.particle_q.assign(initial)
                    control = model.control()
                    solver.step(state_in, state_out, control, None, 1.0 / 600.0)
                    outputs.append(state_out.particle_q.numpy())
                    if enabled:
                        graph_in, graph_out = model.state(), model.state()
                        graph_in.particle_q.assign(initial)
                        with wp.ScopedCapture(device=model.device) as capture:
                            solver.step(graph_in, graph_out, control, None, 1.0 / 600.0)
                        wp.capture_launch(capture.graph)
                        np.testing.assert_array_equal(graph_out.particle_q.numpy(), outputs[-1])
                np.testing.assert_allclose(*outputs, rtol=1.0e-5, atol=1.0e-7)


if __name__ == "__main__":
    unittest.main()

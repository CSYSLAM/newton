# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate the opt-in MJVBDV2 color-batched Jacobi surface solve."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2.vbd_soft.solver_vbd import SolverVBD
from newton.tests.test_mjvbd_v2_surface_relaxation import _cloth


class TestMJVBDV2BatchedJacobi(unittest.TestCase):
    def test_rejects_invalid_or_unsupported_options(self):
        """Keep the CUDA-only approximation explicit and validate its controls."""
        cpu_model = _cloth("cpu")
        with self.assertRaisesRegex(ValueError, "particle_enable_batched_jacobi"):
            SolverVBD(cpu_model, particle_enable_surface_cache=True, particle_enable_batched_jacobi=True)
        with self.assertRaisesRegex(ValueError, "particle_jacobi_relaxation"):
            SolverVBD(cpu_model, particle_jacobi_relaxation=0.0)
        with self.assertRaisesRegex(TypeError, "particle_jacobi_batch_count"):
            SolverVBD(cpu_model, particle_jacobi_batch_count=1.0)

    @unittest.skipUnless(wp.is_cuda_available(), "Batched Jacobi requires CUDA")
    def test_batches_cover_every_particle(self):
        """Build balanced schedules without repeatedly merging a color pair."""
        model = _cloth("cuda:0")
        solver = SolverVBD(
            model,
            iterations=8,
            particle_enable_surface_cache=True,
            particle_enable_batched_jacobi=True,
            particle_jacobi_batch_count=2,
        )
        self.assertEqual(len(solver._particle_jacobi_color_schedules), 8)
        original_colors = model.particle_colors.numpy()
        color_count = len(model.particle_color_groups)
        first_particle_by_color = [int(np.flatnonzero(original_colors == color)[0]) for color in range(color_count)]
        color_assignments = []
        for colors, groups in zip(
            solver._particle_jacobi_color_schedules,
            solver._particle_jacobi_group_schedules,
            strict=True,
        ):
            batched_colors = colors.numpy()
            self.assertEqual(set(np.unique(batched_colors)), {0, 1})
            grouped = np.concatenate([group.numpy() for group in groups])
            np.testing.assert_array_equal(np.sort(grouped), np.arange(model.particle_count))
            color_assignments.append(batched_colors[first_particle_by_color])
        color_assignments = np.asarray(color_assignments)
        co_batch_count = np.sum(
            color_assignments[:, :, None] == color_assignments[:, None, :],
            axis=0,
        )
        self.assertLessEqual(int(np.max(co_batch_count[np.triu_indices(color_count, 1)])), 4)

        with self.assertRaisesRegex(ValueError, "particle_jacobi_batch_count"):
            SolverVBD(
                model,
                particle_enable_surface_cache=True,
                particle_enable_batched_jacobi=True,
                particle_jacobi_batch_count=color_count + 1,
            )

    @unittest.skipUnless(wp.is_cuda_available(), "Batched Jacobi requires CUDA")
    def test_full_color_count_matches_cached_gauss_seidel(self):
        """Recover the cached colored solve when no original colors are merged."""
        model = _cloth("cuda:0")
        force = np.zeros_like(model.particle_q.numpy())
        force[-1, 0] = 0.1
        options = {
            "iterations": 4,
            "particle_enable_self_contact": False,
            "particle_enable_surface_cache": True,
        }
        ordinary = SolverVBD(model, **options)
        batched = SolverVBD(
            model,
            **options,
            particle_enable_batched_jacobi=True,
            particle_jacobi_batch_count=len(model.particle_color_groups),
            particle_jacobi_relaxation=1.0,
        )
        results = []
        for solver in (ordinary, batched):
            state_in, state_out = model.state(), model.state()
            state_in.particle_f.assign(force)
            solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
            results.append(state_out.particle_q.numpy())
        np.testing.assert_allclose(*results, rtol=2.0e-6, atol=2.0e-7)

    @unittest.skipUnless(wp.is_cuda_available(), "Batched Jacobi requires CUDA")
    def test_chebyshev_schedule_replays_in_cuda_graph(self):
        """Match eager execution when the accelerated two-batch schedule is captured."""
        model = _cloth("cuda:0")
        force = np.zeros_like(model.particle_q.numpy())
        force[-1, 0] = 0.1
        options = {
            "iterations": 8,
            "particle_enable_self_contact": False,
            "particle_enable_surface_cache": True,
            "particle_enable_batched_jacobi": True,
            "particle_jacobi_batch_count": 2,
            "particle_jacobi_relaxation": 1.0,
            "particle_chebyshev_spectral_radius": 0.8,
        }
        eager_solver = SolverVBD(model, **options)
        eager_in, eager_out = model.state(), model.state()
        eager_in.particle_f.assign(force)
        eager_solver.step(eager_in, eager_out, model.control(), None, 1.0 / 60.0)

        graph_solver = SolverVBD(model, **options)
        graph_in, graph_out = model.state(), model.state()
        graph_in.particle_f.assign(force)
        with wp.ScopedCapture(device=model.device) as capture:
            graph_solver.step(graph_in, graph_out, model.control(), None, 1.0 / 60.0)
        wp.capture_launch(capture.graph)
        wp.synchronize_device(model.device)

        np.testing.assert_array_equal(graph_out.particle_q.numpy(), eager_out.particle_q.numpy())
        self.assertTrue(np.isfinite(graph_out.particle_q.numpy()).all())

    @unittest.skipUnless(wp.is_cuda_available(), "Batched Jacobi requires CUDA")
    def test_rejected_multilevel_correction_uses_gauss_seidel_fallback(self):
        """Switch rejected schedules to the conservative solver for remaining sweeps."""
        model = _cloth("cuda:0")
        force = np.zeros_like(model.particle_q.numpy())
        force[-1, 0] = 1.0
        solver = SolverVBD(
            model,
            iterations=2,
            particle_enable_self_contact=False,
            particle_enable_surface_cache=True,
            particle_enable_batched_jacobi=True,
            particle_jacobi_batch_count=2,
            particle_enable_multilevel_correction=True,
            particle_multilevel_checkpoints=(1,),
            particle_multilevel_max_radius_fraction=1.0e-6,
            particle_multilevel_max_clamp_fraction=0.0,
            particle_multilevel_fallback_iterations=4,
        )
        state_in, state_out = model.state(), model.state()
        state_in.particle_f.assign(force)
        with mock.patch.object(
            solver,
            "_solve_particle_jacobi_iteration",
            wraps=solver._solve_particle_jacobi_iteration,
        ) as batched_sweep:
            solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)

        self.assertNotEqual(int(solver.particle_multilevel.runtime_status.numpy()[0]), 0)
        self.assertEqual(batched_sweep.call_count, 2)
        self.assertTrue(solver.particle_enable_batched_jacobi)
        self.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())

        graph_solver = SolverVBD(
            model,
            iterations=2,
            particle_enable_self_contact=False,
            particle_enable_surface_cache=True,
            particle_enable_batched_jacobi=True,
            particle_jacobi_batch_count=2,
            particle_enable_multilevel_correction=True,
            particle_multilevel_checkpoints=(1,),
            particle_multilevel_max_radius_fraction=1.0e-6,
            particle_multilevel_max_clamp_fraction=0.0,
            particle_multilevel_fallback_iterations=4,
        )
        graph_in, graph_out = model.state(), model.state()
        graph_in.particle_f.assign(force)
        with mock.patch.object(
            graph_solver,
            "_solve_particle_jacobi_iteration",
            wraps=graph_solver._solve_particle_jacobi_iteration,
        ) as captured_batched_sweep:
            with wp.ScopedCapture(device=model.device) as capture:
                graph_solver.step(graph_in, graph_out, model.control(), None, 1.0 / 60.0)
        wp.capture_launch(capture.graph)
        wp.synchronize_device(model.device)

        self.assertNotEqual(int(graph_solver.particle_multilevel.runtime_status.numpy()[0]), 0)
        self.assertEqual(captured_batched_sweep.call_count, 2)
        self.assertTrue(graph_solver.particle_enable_batched_jacobi)
        self.assertTrue(np.isfinite(graph_out.particle_q.numpy()).all())


if __name__ == "__main__":
    unittest.main()

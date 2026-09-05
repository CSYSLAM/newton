# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the private MJVBDV2 particle Chebyshev accelerator."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD as SolverVBDComplete
from newton._src.solvers.mjvbd_v2.vbd_soft.solver_vbd import SolverVBD as SolverVBDSoft


def _build_cloth(device, *, requires_grad=False):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 0.5),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=4,
        dim_y=2,
        cell_x=0.02,
        cell_y=0.02,
        mass=0.001,
        tri_ke=1.0e4,
        tri_ka=1.0e4,
        tri_kd=0.0,
        edge_ke=0.1,
        fix_left=True,
    )
    builder.color(include_bending=True)
    return builder.finalize(device=device, requires_grad=requires_grad)


class TestMJVBDV2ParticleChebyshev(unittest.TestCase):
    def test_invalid_options_are_rejected(self):
        model = _build_cloth("cpu")
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            for spectral_radius in (-0.1, 0.0, 1.0, 1.1):
                with self.subTest(solver=solver_type.__module__, spectral_radius=spectral_radius):
                    with self.assertRaisesRegex(ValueError, "particle_chebyshev_spectral_radius"):
                        solver_type(model, particle_chebyshev_spectral_radius=spectral_radius)
            with self.subTest(solver=solver_type.__module__, max_radius_fraction=0.0):
                with self.assertRaisesRegex(ValueError, "particle_chebyshev_max_radius_fraction"):
                    solver_type(model, particle_chebyshev_max_radius_fraction=0.0)
            for option in (
                "particle_chebyshev_warmup_iterations",
                "particle_chebyshev_polish_iterations",
                "particle_chebyshev_contact_rings",
            ):
                with self.subTest(solver=solver_type.__module__, option=option):
                    with self.assertRaisesRegex(ValueError, option):
                        solver_type(model, particle_chebyshev_spectral_radius=0.9, **{option: -1})
            with self.subTest(solver=solver_type.__module__, mode="empty_accelerated_window"):
                with self.assertRaisesRegex(ValueError, "leave at least one accelerated iteration"):
                    solver_type(
                        model,
                        iterations=4,
                        particle_chebyshev_spectral_radius=0.9,
                        particle_chebyshev_warmup_iterations=2,
                        particle_chebyshev_polish_iterations=2,
                    )
            with self.subTest(solver=solver_type.__module__, mode="cleanup_without_fallback"):
                with self.assertRaisesRegex(ValueError, "cleanup requires fallback iterations"):
                    solver_type(
                        model,
                        iterations=4,
                        particle_chebyshev_spectral_radius=0.9,
                        particle_chebyshev_cleanup_max_radius_fraction=0.03,
                    )
            with self.subTest(solver=solver_type.__module__, mode="cleanup_without_multilevel"):
                with self.assertRaisesRegex(ValueError, "requires particle_enable_multilevel_correction"):
                    solver_type(
                        model,
                        iterations=4,
                        particle_chebyshev_spectral_radius=0.9,
                        particle_chebyshev_cleanup_max_radius_fraction=0.03,
                        particle_multilevel_fallback_iterations=5,
                    )

    def test_accelerator_is_opt_in_and_weights_follow_vbd_recurrence(self):
        model = _build_cloth("cpu")
        differentiable_model = _build_cloth("cpu", requires_grad=True)
        rho = 0.9
        expected = [1.0, 2.0 / (2.0 - rho * rho)]
        for _ in range(2):
            expected.append(4.0 / (4.0 - rho * rho * expected[-1]))

        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__, mode="default"):
                solver = solver_type(model, iterations=4)
                self.assertFalse(solver.particle_chebyshev_enabled)
                self.assertEqual(solver.particle_chebyshev_weights, ())
                self.assertIsNone(solver.particle_chebyshev_older)

            with self.subTest(solver=solver_type.__module__, mode="enabled"):
                solver = solver_type(model, iterations=4, particle_chebyshev_spectral_radius=rho)
                self.assertTrue(solver.particle_chebyshev_enabled)
                np.testing.assert_allclose(solver.particle_chebyshev_weights, expected)
                self.assertIsNotNone(solver.particle_chebyshev_older)
                self.assertIsNotNone(solver.particle_chebyshev_collided)

            with self.subTest(solver=solver_type.__module__, mode="differentiable"):
                solver = solver_type(
                    differentiable_model,
                    iterations=4,
                    particle_chebyshev_spectral_radius=rho,
                )
                self.assertFalse(solver.particle_chebyshev_enabled)
                self.assertIsNone(solver.particle_chebyshev_older)

    def test_guarded_schedule_builds_compact_topology_and_accelerated_window(self):
        model = _build_cloth("cpu")
        rho = 0.9
        expected_weights = (1.0, 2.0 / (2.0 - rho * rho))
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__):
                solver = solver_type(
                    model,
                    iterations=4,
                    particle_chebyshev_spectral_radius=rho,
                    particle_chebyshev_warmup_iterations=1,
                    particle_chebyshev_polish_iterations=1,
                    particle_chebyshev_contact_rings=2,
                )

                self.assertTrue(solver.particle_chebyshev_guarded)
                np.testing.assert_allclose(solver.particle_chebyshev_weights, expected_weights)
                self.assertEqual(len(solver.particle_chebyshev_guard_masks), 2)
                offsets = solver.particle_chebyshev_neighbor_offsets.numpy()
                neighbors = solver.particle_chebyshev_neighbors.numpy()
                self.assertEqual(offsets.shape, (model.particle_count + 1,))
                self.assertEqual(offsets[-1], neighbors.size)
                self.assertGreater(neighbors.size, 0)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA Graph execution requires CUDA")
    def test_cuda_graph_executes_in_both_private_vbd_solvers(self):
        device = wp.get_device("cuda:0")
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__):
                model = _build_cloth(device)
                solver = solver_type(model, iterations=3, particle_chebyshev_spectral_radius=0.9)
                state_0 = model.state()
                state_1 = model.state()
                control = model.control()
                solver.step(state_0, state_1, control, None, 1.0 / 60.0)
                with wp.ScopedCapture(device=device) as capture:
                    solver.step(state_1, state_0, control, None, 1.0 / 60.0)
                wp.capture_launch(capture.graph)
                wp.synchronize_device(device)

                self.assertTrue(np.isfinite(state_0.particle_q.numpy()).all())

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA Graph execution requires CUDA")
    def test_guarded_cleanup_graph_executes_in_both_private_vbd_solvers(self):
        device = wp.get_device("cuda:0")
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__):
                model = _build_cloth(device)
                solver = solver_type(
                    model,
                    iterations=4,
                    particle_chebyshev_spectral_radius=0.9,
                    particle_chebyshev_warmup_iterations=1,
                    particle_chebyshev_polish_iterations=1,
                    particle_chebyshev_contact_rings=1,
                    particle_chebyshev_cleanup_max_radius_fraction=0.03,
                    particle_enable_multilevel_correction=True,
                    particle_multilevel_fallback_iterations=5,
                )
                state_0 = model.state()
                state_1 = model.state()
                control = model.control()
                solver.step(state_0, state_1, control, None, 1.0 / 60.0)
                with wp.ScopedCapture(device=device) as capture:
                    solver.step(state_1, state_0, control, None, 1.0 / 60.0)
                wp.capture_launch(capture.graph)
                wp.synchronize_device(device)

                self.assertTrue(np.isfinite(state_0.particle_q.numpy()).all())
                self.assertIsNotNone(solver.particle_chebyshev_cleanup_status)
                self.assertGreaterEqual(float(solver.particle_chebyshev_cleanup_metrics.numpy()[0]), 0.0)


if __name__ == "__main__":
    unittest.main()

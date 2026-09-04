# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the private MJVBDV2 particle multilevel correction."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjvbd_v2.particle_multilevel import _build_clusters
from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD as SolverVBDComplete
from newton._src.solvers.mjvbd_v2.vbd_soft.solver_vbd import SolverVBD as SolverVBDSoft


def _build_cloth(device, *, dim_x=16, dim_y=4, fix_left=True, requires_grad=False):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_cloth_grid(
        pos=wp.vec3(),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=dim_x,
        dim_y=dim_y,
        cell_x=0.02,
        cell_y=0.02,
        mass=0.001,
        tri_ke=1.0e5,
        tri_ka=1.0e5,
        tri_kd=0.0,
        edge_ke=0.0,
        fix_left=fix_left,
    )
    builder.color()
    return builder.finalize(device=device, requires_grad=requires_grad)


class TestMJVBDV2ParticleMultilevel(unittest.TestCase):
    def test_invalid_automatic_mode_is_rejected(self):
        model = _build_cloth("cpu")
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__):
                with self.assertRaisesRegex(ValueError, "particle_enable_multilevel_correction"):
                    solver_type(model, iterations=2, particle_enable_multilevel_correction="sometimes")

    def test_cpu_uses_original_vbd_path(self):
        model = _build_cloth("cpu")
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__):
                solver = solver_type(model, iterations=2, particle_enable_multilevel_correction=True)
                self.assertIsNone(solver.particle_multilevel)

    def test_clusters_keep_fixed_particles_as_anchors(self):
        model = _build_cloth("cpu", dim_x=8, dim_y=2)
        (
            fine_to_coarse,
            cluster_offsets,
            cluster_particles,
            _coarse_neighbor_offsets,
            _coarse_neighbors,
            _coarse_neighbor_multiplicity,
            _coarse_incident_edges,
            coarse_anchor_edges,
        ) = _build_clusters(model, 4)

        mass = model.particle_mass.numpy()
        fixed = mass == 0.0
        movable = ~fixed
        self.assertTrue(np.all(fine_to_coarse[fixed] == -1))
        self.assertTrue(np.all(fine_to_coarse[movable] >= 0))
        self.assertEqual(cluster_offsets[-1], cluster_particles.size)
        self.assertGreater(int(coarse_anchor_edges.sum()), 0)

    def test_tetrahedral_particles_stay_on_original_vbd_path(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for position in (
            (0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.0, 0.1, 0.0),
            (0.0, 0.0, 0.1),
            (1.0, 0.0, 0.0),
            (1.1, 0.0, 0.0),
            (1.0, 0.1, 0.0),
        ):
            builder.add_particle(wp.vec3(*position), wp.vec3(), 0.01, radius=0.01)
        builder.add_tetrahedron(0, 1, 2, 3)
        builder.add_triangle(4, 5, 6)
        builder.color()
        model = builder.finalize(device="cpu")

        fine_to_coarse = _build_clusters(model, 2)[0]
        self.assertTrue(np.all(fine_to_coarse[:4] == -1))
        self.assertTrue(np.all(fine_to_coarse[4:] >= 0))

    @unittest.skipUnless(wp.is_cuda_available(), "Particle multilevel correction requires CUDA")
    def test_automatic_mode_uses_conservative_topology_and_scale_gate(self):
        device = wp.get_device("cuda:0")
        small_model = _build_cloth(device)
        eligible_model = _build_cloth(device, dim_x=64, dim_y=15)

        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__, scale="small"):
                solver = solver_type(
                    small_model,
                    iterations=2,
                    particle_enable_multilevel_correction="auto",
                )
                self.assertIsNone(solver.particle_multilevel)
                self.assertEqual(solver.particle_multilevel_auto_rejection_reason, "too_few_active_particles")

            with self.subTest(solver=solver_type.__module__, scale="eligible"):
                solver = solver_type(
                    eligible_model,
                    iterations=2,
                    particle_enable_multilevel_correction="auto",
                )
                self.assertIsNotNone(solver.particle_multilevel)
                self.assertIsNone(solver.particle_multilevel_auto_rejection_reason)

            with self.subTest(solver=solver_type.__module__, scale="self_contact"):
                solver = solver_type(
                    eligible_model,
                    iterations=2,
                    particle_enable_multilevel_correction="auto",
                    particle_enable_self_contact=True,
                )
                self.assertIsNone(solver.particle_multilevel)
                self.assertEqual(
                    solver.particle_multilevel_auto_rejection_reason,
                    "self_contact_requires_explicit_enable",
                )

    @unittest.skipUnless(wp.is_cuda_available(), "Particle multilevel correction requires CUDA")
    def test_cuda_graph_executes_in_both_private_vbd_solvers(self):
        device = wp.get_device("cuda:0")
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__):
                model = _build_cloth(device)
                solver = solver_type(model, iterations=2, particle_enable_multilevel_correction=True)
                state_0 = model.state()
                state_1 = model.state()
                control = model.control()
                solver.step(state_0, state_1, control, None, 1.0 / 60.0)
                with wp.ScopedCapture(device=device) as capture:
                    solver.step(state_1, state_0, control, None, 1.0 / 60.0)
                wp.capture_launch(capture.graph)
                wp.synchronize_device(device)

                self.assertIsNotNone(solver.particle_multilevel)
                self.assertTrue(np.isfinite(state_0.particle_q.numpy()).all())

    @unittest.skipUnless(wp.is_cuda_available(), "Particle multilevel correction requires CUDA")
    def test_deterministic_mode_uses_original_vbd_path(self):
        device = wp.get_device("cuda:0")
        model = _build_cloth(device)
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__):
                solver = solver_type(
                    model,
                    iterations=2,
                    particle_enable_multilevel_correction=True,
                    deterministic=wp.DeterministicMode.RUN_TO_RUN,
                )
                self.assertIsNone(solver.particle_multilevel)

        grad_model = _build_cloth(device, requires_grad=True)
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__, requires_grad=True):
                solver = solver_type(grad_model, iterations=2, particle_enable_multilevel_correction=True)
                self.assertIsNone(solver.particle_multilevel)

    @unittest.skipUnless(wp.is_cuda_available(), "Particle multilevel correction requires CUDA")
    def test_coarse_correction_improves_long_range_propagation(self):
        device = wp.get_device("cuda:0")
        dim_x = 32
        dim_y = 4
        model = _build_cloth(device, dim_x=dim_x, dim_y=dim_y)
        rest = model.particle_q.numpy().copy()
        forces = np.zeros_like(rest)
        right_edge = np.arange(dim_y + 1) * (dim_x + 1) + dim_x
        forces[right_edge, 0] = 1.0

        results = {}
        for name, iterations, multilevel in (("reference", 100, False), ("plain", 8, False), ("coarse", 8, True)):
            state_in = model.state()
            state_out = model.state()
            state_in.particle_f.assign(forces)
            solver = SolverVBDSoft(
                model,
                iterations=iterations,
                particle_enable_multilevel_correction=multilevel,
            )
            solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
            results[name] = state_out.particle_q.numpy().copy()

        plain_error = np.linalg.norm(results["plain"] - results["reference"])
        coarse_error = np.linalg.norm(results["coarse"] - results["reference"])
        middle = np.arange(dim_y + 1) * (dim_x + 1) + dim_x // 2
        plain_middle_motion = np.mean(np.abs(results["plain"][middle, 0] - rest[middle, 0]))
        coarse_middle_motion = np.mean(np.abs(results["coarse"][middle, 0] - rest[middle, 0]))

        self.assertLess(coarse_error, plain_error)
        self.assertGreater(coarse_middle_motion, plain_middle_motion + 1.0e-5)

    @unittest.skipUnless(wp.is_cuda_available(), "Particle multilevel correction requires CUDA")
    def test_excessive_clamping_rejects_correction_transactionally(self):
        device = wp.get_device("cuda:0")
        dim_x = 32
        dim_y = 4
        model = _build_cloth(device, dim_x=dim_x, dim_y=dim_y)
        forces = np.zeros_like(model.particle_q.numpy())
        right_edge = np.arange(dim_y + 1) * (dim_x + 1) + dim_x
        forces[right_edge, 0] = 1.0

        def simulate(
            solver_type,
            *,
            iterations=8,
            multilevel,
            max_clamp_fraction=1.0,
            fallback_iterations=None,
        ):
            state_in = model.state()
            state_out = model.state()
            state_in.particle_f.assign(forces)
            solver = solver_type(
                model,
                iterations=iterations,
                particle_enable_multilevel_correction=multilevel,
                particle_multilevel_max_radius_fraction=1.0e-6,
                particle_multilevel_max_clamp_fraction=max_clamp_fraction,
                particle_multilevel_fallback_iterations=fallback_iterations,
            )
            solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
            return state_out.particle_q.numpy().copy(), solver

        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__):
                plain_positions, _plain_solver = simulate(solver_type, multilevel=False)
                rejected_positions, rejected_solver = simulate(
                    solver_type,
                    multilevel=True,
                    max_clamp_fraction=0.0,
                )

                correction = rejected_solver.particle_multilevel
                self.assertIsNotNone(correction)
                self.assertGreater(float(correction.runtime_metrics.numpy()[2]), 0.0)
                self.assertNotEqual(int(correction.runtime_status.numpy()[0]), 0)
                np.testing.assert_array_equal(rejected_positions, plain_positions)

                reference_positions, _reference_solver = simulate(solver_type, iterations=12, multilevel=False)
                fallback_positions, _fallback_solver = simulate(
                    solver_type,
                    multilevel=True,
                    max_clamp_fraction=0.0,
                    fallback_iterations=12,
                )
                np.testing.assert_array_equal(fallback_positions, reference_positions)

                graph_state_in = model.state()
                graph_state_out = model.state()
                graph_state_in.particle_f.assign(forces)
                graph_solver = solver_type(
                    model,
                    iterations=8,
                    particle_enable_multilevel_correction=True,
                    particle_multilevel_max_radius_fraction=1.0e-6,
                    particle_multilevel_max_clamp_fraction=0.0,
                    particle_multilevel_fallback_iterations=12,
                )
                with wp.ScopedCapture(device=device) as capture:
                    graph_solver.step(graph_state_in, graph_state_out, model.control(), None, 1.0 / 60.0)
                wp.capture_launch(capture.graph)
                wp.synchronize_device(device)
                np.testing.assert_array_equal(graph_state_out.particle_q.numpy(), reference_positions)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the private MJVBDV2 particle multilevel correction."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjvbd_v2.particle_multilevel import ParticleMultilevelCorrection, _build_clusters
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


def _build_two_tets(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for x_offset in (0.0, 0.3):
        first_particle = builder.particle_count
        for position in (
            (x_offset, 0.0, 0.0),
            (x_offset + 0.1, 0.0, 0.0),
            (x_offset, 0.1, 0.0),
            (x_offset, 0.0, 0.1),
        ):
            builder.add_particle(wp.vec3(*position), wp.vec3(), 0.01, radius=0.01)
        builder.add_tetrahedron(
            first_particle,
            first_particle + 1,
            first_particle + 2,
            first_particle + 3,
        )
    builder.color()
    return builder.finalize(device=device)


class TestMJVBDV2ParticleMultilevel(unittest.TestCase):
    def test_invalid_automatic_mode_is_rejected(self):
        """Reject unknown automatic-mode strings."""

        model = _build_cloth("cpu")
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__):
                with self.assertRaisesRegex(ValueError, "particle_enable_multilevel_correction"):
                    solver_type(model, iterations=2, particle_enable_multilevel_correction="sometimes")

    @unittest.skipUnless(wp.is_cuda_available(), "Particle multilevel correction requires CUDA")
    def test_multilevel_correction_is_opt_in(self):
        """Keep the existing CUDA solver path unchanged unless explicitly enabled."""
        model = _build_cloth(wp.get_device("cuda:0"))
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__):
                solver = solver_type(model, iterations=2)
                self.assertIsNone(solver.particle_multilevel)

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

    def test_mixed_surface_and_tet_particles_use_separate_clusters(self):
        """Assign tet particles without mixing their rigid basis into surface clusters."""
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
        # Deliberately connect both domains: basis type, rather than graph
        # connectivity alone, must form a cluster boundary.
        builder.add_spring(3, 4, ke=1.0, kd=0.0, control=0.0)
        builder.color()
        model = builder.finalize(device="cpu")

        fine_to_coarse, offsets, particles, *_ = _build_clusters(model, 16)
        self.assertTrue(np.all(fine_to_coarse[:4] >= 0))
        self.assertTrue(np.all(fine_to_coarse[4:] >= 0))
        tet_particle = np.zeros(model.particle_count, dtype=bool)
        tet_particle[:4] = True
        for cluster in range(offsets.size - 1):
            members = particles[offsets[cluster] : offsets[cluster + 1]]
            self.assertTrue(np.all(tet_particle[members] == tet_particle[members[0]]))

    def test_fixed_tets_do_not_select_rigid_coarse_path(self):
        """Do not penalize a surface solve for inactive tetrahedral topology."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for position in ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0), (0.0, 0.0, 0.1)):
            builder.add_particle(wp.vec3(*position), wp.vec3(), 0.0, radius=0.01)
        builder.add_tetrahedron(0, 1, 2, 3)
        builder.add_cloth_grid(
            pos=wp.vec3(1.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=4,
            dim_y=2,
            cell_x=0.02,
            cell_y=0.02,
            mass=0.001,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=0.0,
            edge_ke=0.0,
        )
        builder.color()
        model = builder.finalize(device="cpu")
        correction = ParticleMultilevelCorrection(
            model,
            cluster_size=4,
            coarse_iterations=2,
            coupling=0.5,
            relaxation=0.1,
            max_radius_fraction=0.05,
            minimum_residual_reduction=None,
            max_clamp_fraction=1.0,
        )

        self.assertFalse(correction.use_rigid_basis)

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
    def test_rigid_cluster_basis_executes_for_tetrahedra(self):
        """Execute the six-DOF cluster solve for a tetrahedral model."""
        device = wp.get_device("cuda:0")
        model = _build_two_tets(device)
        solver = SolverVBDComplete(
            model,
            iterations=2,
            particle_enable_multilevel_correction=True,
            particle_multilevel_cluster_size=4,
        )
        state_in = model.state()
        state_out = model.state()
        solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)

        self.assertIsNotNone(solver.particle_multilevel)
        self.assertTrue(solver.particle_multilevel.use_rigid_basis)
        self.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())
        self.assertTrue(np.isfinite(solver.particle_multilevel.coarse_residual_ratio.numpy()).all())

    @unittest.skipUnless(wp.is_cuda_available(), "Particle multilevel correction requires CUDA")
    def test_rigid_cluster_basis_responds_to_cluster_rotation(self):
        """Respond to a representable infinitesimal rotation on each tet cluster."""
        device = wp.get_device("cuda:0")
        model = _build_two_tets(device)
        correction = ParticleMultilevelCorrection(
            model,
            cluster_size=4,
            coarse_iterations=8,
            coupling=0.5,
            relaxation=0.1,
            max_radius_fraction=10.0,
            minimum_residual_reduction=None,
            max_clamp_fraction=1.0,
        )
        positions = model.particle_q.numpy()
        masses = model.particle_mass.numpy()
        local_updates = np.zeros_like(positions)
        for cluster in range(2):
            particles = np.arange(cluster * 4, cluster * 4 + 4)
            centroid = np.average(positions[particles], axis=0, weights=masses[particles])
            omega = np.asarray((0.12, -0.08, 0.05), dtype=np.float32)
            local_updates[particles] = np.cross(omega, positions[particles] - centroid)
        dt = 1.0 / 60.0
        hessians = masses[:, None, None] / (dt * dt) * np.eye(3, dtype=np.float32)[None, :, :]
        correction.local_correction.assign(local_updates)
        correction.local_hessians.assign(hessians)
        particle_displacements = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)

        correction.restrict_and_prolong(model, model.particle_q, particle_displacements, dt)
        result = particle_displacements.numpy()

        self.assertGreater(float(np.linalg.norm(result)), 1.0e-4)
        self.assertLess(float(np.linalg.norm(result - 0.1 * local_updates)), 1.1e-3)
        self.assertLess(float(correction.coarse_residual_ratio.numpy()[0]), 1.0e-3)

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

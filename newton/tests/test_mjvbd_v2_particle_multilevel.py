# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the private MJVBDV2 particle multilevel correction."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjvbd_v2.particle_multilevel import (
    ParticleMultilevelCorrection,
    _bending_hessian_block,
    _build_clusters,
    _build_energy_galerkin_structure,
    _membrane_hessian_block,
    _solve_energy_galerkin_pcg_persistent,
)
from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD as SolverVBDComplete
from newton._src.solvers.mjvbd_v2.vbd_soft.particle_vbd_kernels import (
    evaluate_dihedral_angle_based_bending_force_hessian,
    evaluate_neo_hookean_membrane_force_hessian,
)
from newton._src.solvers.mjvbd_v2.vbd_soft.solver_vbd import SolverVBD as SolverVBDSoft


@wp.kernel
def _evaluate_membrane_blocks(
    dt: float,
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_materials: wp.array2d[float],
    tri_areas: wp.array[float],
    blocks: wp.array[wp.mat33],
):
    index = wp.tid()
    face = index // 9
    pair = index % 9
    blocks[index] = _membrane_hessian_block(
        face,
        pair // 3,
        pair % 3,
        pos,
        tri_indices,
        tri_poses[face],
        tri_areas[face],
        tri_materials[face, 0],
        tri_materials[face, 1],
        tri_materials[face, 2],
        dt,
    )


@wp.kernel
def _compare_membrane_diagonal_blocks(
    dt: float,
    pos: wp.array[wp.vec3],
    pos_anchor: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_materials: wp.array2d[float],
    tri_areas: wp.array[float],
    expected: wp.array[wp.mat33],
    actual: wp.array[wp.mat33],
):
    index = wp.tid()
    face = index // 3
    vertex = index - face * 3
    _force, hessian = evaluate_neo_hookean_membrane_force_hessian(
        face,
        vertex,
        pos,
        pos_anchor,
        tri_indices,
        tri_poses[face],
        tri_areas[face],
        tri_materials[face, 0],
        tri_materials[face, 1],
        tri_materials[face, 2],
        dt,
    )
    expected[index] = hessian
    actual[index] = _membrane_hessian_block(
        face,
        vertex,
        vertex,
        pos,
        tri_indices,
        tri_poses[face],
        tri_areas[face],
        tri_materials[face, 0],
        tri_materials[face, 1],
        tri_materials[face, 2],
        dt,
    )


@wp.kernel
def _compare_bending_diagonal_blocks(
    dt: float,
    pos: wp.array[wp.vec3],
    pos_anchor: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_angle: wp.array[float],
    edge_rest_length: wp.array[float],
    edge_bending_properties: wp.array2d[float],
    expected: wp.array[wp.mat33],
    actual: wp.array[wp.mat33],
):
    index = wp.tid()
    edge = index // 4
    vertex = index - edge * 4
    _force, hessian = evaluate_dihedral_angle_based_bending_force_hessian(
        edge,
        vertex,
        pos,
        pos_anchor,
        edge_indices,
        edge_rest_angle,
        edge_rest_length,
        edge_bending_properties[edge, 0],
        edge_bending_properties[edge, 1],
        dt,
    )
    expected[index] = hessian
    actual[index] = _bending_hessian_block(
        edge,
        vertex,
        vertex,
        pos,
        edge_indices,
        edge_rest_length,
        edge_bending_properties[edge, 0],
        edge_bending_properties[edge, 1],
        dt,
    )


def _build_cloth(
    device,
    *,
    dim_x=16,
    dim_y=4,
    fix_left=True,
    requires_grad=False,
    tri_kd=0.0,
    edge_ke=0.0,
    edge_kd=0.0,
):
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
        tri_kd=tri_kd,
        edge_ke=edge_ke,
        edge_kd=edge_kd,
        fix_left=fix_left,
    )
    builder.color()
    return builder.finalize(device=device, requires_grad=requires_grad)


def _build_tets(device, *, include_surface=False, tet_count=2):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for x_offset in np.arange(tet_count) * 0.3:
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
    if include_surface:
        builder.add_cloth_grid(
            pos=wp.vec3(1.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=16,
            dim_y=4,
            cell_x=0.02,
            cell_y=0.02,
            mass=0.001,
            tri_ke=1.0e5,
            tri_ka=1.0e5,
            tri_kd=0.0,
            edge_ke=0.0,
            fix_left=True,
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

    def test_invalid_coarse_operator_is_rejected(self):
        """Reject unknown surface operator names independently of device eligibility."""
        model = _build_cloth("cpu")
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__):
                with self.assertRaisesRegex(ValueError, "particle_multilevel_operator"):
                    solver_type(model, iterations=2, particle_multilevel_operator="dense")

    def test_surface_operator_does_not_allocate_tet_translation_matrix(self):
        """Keep mixed/tet corrections on the six-DOF path for either surface option."""
        model = _build_tets("cpu", include_surface=True)
        for operator in ("graph", "galerkin"):
            correction = ParticleMultilevelCorrection(
                model,
                operator=operator,
                cluster_size=4,
                coarse_iterations=4,
                coupling=0.5,
                relaxation=0.1,
                max_radius_fraction=0.05,
                minimum_residual_reduction=None,
                max_clamp_fraction=1.0,
            )
            self.assertTrue(correction.use_rigid_basis)
            self.assertEqual(correction.coarse_matrix_blocks.size, 0)
            self.assertGreater(correction.coarse_tet_blocks.size, 0)

    @unittest.skipUnless(wp.is_cuda_available(), "Mixed cache integration requires CUDA")
    def test_mixed_tet_surface_cache_and_operator_graph_replay(self):
        """Preserve the mixed six-DOF result and cache replay in both private solvers."""
        # Both surface and tet color groups must meet the existing tile-size gate.
        model = _build_tets("cuda:0", include_surface=True, tet_count=32)
        force = np.zeros_like(model.particle_q.numpy())
        force[0, 0] = 0.05
        force[-1, 0] = 0.02
        control = model.control()
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            positions = []
            for operator in ("graph", "galerkin"):
                solver = solver_type(
                    model,
                    iterations=4,
                    particle_enable_multilevel_correction=True,
                    particle_multilevel_operator=operator,
                    particle_multilevel_cluster_size=4,
                    particle_enable_surface_cache=True,
                )
                self.assertTrue(solver.particle_multilevel.use_rigid_basis)
                self.assertIsNotNone(solver.surface_anchor_angles)
                state_in, state_out = model.state(), model.state()
                state_in.particle_f.assign(force)
                solver.step(state_in, state_out, control, None, 1.0 / 600.0)
                eager = state_out.particle_q.numpy()
                self.assertTrue(np.isfinite(eager).all())
                positions.append(eager)
                graph_in, graph_out = model.state(), model.state()
                graph_in.particle_f.assign(force)
                with wp.ScopedCapture(device=model.device) as capture:
                    solver.step(graph_in, graph_out, control, None, 1.0 / 600.0)
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(graph_out.particle_q.numpy(), eager, atol=1.0e-7, rtol=1.0e-6)
            np.testing.assert_array_equal(*positions)

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

    def test_energy_galerkin_structure_is_symmetric(self):
        """Build a symmetric block pattern with one diagonal per aggregate."""
        model = _build_cloth("cpu", dim_x=8, dim_y=2)
        fine_to_coarse = _build_clusters(model, 4)[0]
        offsets, columns, diagonal_slots, triangle_slots, edge_slots, spring_slots = _build_energy_galerkin_structure(
            model, fine_to_coarse
        )
        entries = {
            (row, int(columns[slot]))
            for row in range(offsets.size - 1)
            for slot in range(int(offsets[row]), int(offsets[row + 1]))
        }
        self.assertTrue(all((column, row) in entries for row, column in entries))
        self.assertTrue(all(int(columns[diagonal_slots[row]]) == row for row in range(diagonal_slots.size)))
        self.assertEqual(triangle_slots.size, model.tri_count * 9)
        self.assertEqual(edge_slots.size, model.edge_count * 16)
        self.assertEqual(spring_slots.size, model.spring_count * 4)

    def test_galerkin_element_diagonals_match_vbd(self):
        """Use the same projected diagonal blocks as the fine VBD solve."""
        model = _build_cloth("cpu", dim_x=4, dim_y=2, tri_kd=0.2, edge_ke=10.0, edge_kd=0.1)
        membrane_expected = wp.empty(model.tri_count * 3, dtype=wp.mat33, device=model.device)
        membrane_actual = wp.empty_like(membrane_expected)
        wp.launch(
            _compare_membrane_diagonal_blocks,
            dim=membrane_expected.shape[0],
            inputs=[
                1.0 / 60.0,
                model.particle_q,
                model.particle_q,
                model.tri_indices,
                model.tri_poses,
                model.tri_materials,
                model.tri_areas,
            ],
            outputs=[membrane_expected, membrane_actual],
            device=model.device,
        )
        np.testing.assert_allclose(membrane_actual.numpy(), membrane_expected.numpy(), rtol=2.0e-5, atol=1.0e-3)

        bending_expected = wp.empty(model.edge_count * 4, dtype=wp.mat33, device=model.device)
        bending_actual = wp.empty_like(bending_expected)
        wp.launch(
            _compare_bending_diagonal_blocks,
            dim=bending_expected.shape[0],
            inputs=[
                1.0 / 60.0,
                model.particle_q,
                model.particle_q,
                model.edge_indices,
                model.edge_rest_angle,
                model.edge_rest_length,
                model.edge_bending_properties,
            ],
            outputs=[bending_expected, bending_actual],
            device=model.device,
        )
        np.testing.assert_allclose(bending_actual.numpy(), bending_expected.numpy(), rtol=2.0e-5, atol=1.0e-3)

    def test_stretched_membrane_coarse_blocks_are_positive_semidefinite(self):
        """Keep the assembled membrane positive semidefinite under tension."""
        model = _build_cloth("cpu", dim_x=1, dim_y=1, fix_left=False)
        rest = model.particle_q.numpy()
        positions = wp.empty_like(model.particle_q)
        blocks = wp.empty(model.tri_count * 9, dtype=wp.mat33, device=model.device)
        for stretch in (1.0, 1.5, 2.0, 4.0):
            with self.subTest(stretch=stretch):
                positions.assign(rest * stretch)
                wp.launch(
                    _evaluate_membrane_blocks,
                    dim=model.tri_count * 9,
                    inputs=[
                        1.0 / 60.0,
                        positions,
                        model.tri_indices,
                        model.tri_poses,
                        model.tri_materials,
                        model.tri_areas,
                    ],
                    outputs=[blocks],
                    device=model.device,
                )
                for element in blocks.numpy().reshape((-1, 3, 3, 3, 3)):
                    matrix = element.transpose((0, 2, 1, 3)).reshape((9, 9)).astype(np.float64)
                    scale = float(np.max(np.abs(matrix)))
                    np.testing.assert_allclose(matrix, matrix.T, atol=scale * 1.0e-6)
                    self.assertGreaterEqual(float(np.linalg.eigvalsh(matrix)[0]), -scale * 1.0e-6)
                    translations = np.tile(np.eye(3), (3, 1))
                    np.testing.assert_allclose(matrix @ translations, 0.0, atol=scale * 1.0e-6)

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
    def test_galerkin_rejects_nonpositive_pcg_curvature(self):
        """Reject an indefinite coarse system even without residual validation."""
        device = wp.get_device("cuda:0")
        identity = np.eye(3, dtype=np.float32)
        matrix = wp.array(np.array([identity, 2.0 * identity, 2.0 * identity, identity]), dtype=wp.mat33, device=device)
        offsets = wp.array([0, 2, 4], dtype=wp.int32, device=device)
        columns = wp.array([0, 1, 0, 1], dtype=wp.int32, device=device)
        diagonal = wp.array([0, 3], dtype=wp.int32, device=device)
        rhs = wp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        vectors = [wp.zeros(2, dtype=wp.vec3, device=device) for _ in range(5)]
        inverse = wp.zeros(2, dtype=wp.mat33, device=device)
        status = wp.zeros(1, dtype=wp.int32, device=device)
        metrics = wp.zeros(9, dtype=float, device=device)
        counters = wp.zeros(2, dtype=wp.int32, device=device)
        wp.launch(
            _solve_energy_galerkin_pcg_persistent,
            dim=256,
            block_dim=256,
            inputs=[2, offsets, columns, diagonal, matrix, rhs, 4, False, 0.0],
            outputs=[*vectors, inverse, status, metrics, counters],
            device=device,
        )
        self.assertNotEqual(int(status.numpy()[0]), 0)
        np.testing.assert_array_equal(vectors[0].numpy(), np.zeros((2, 3), dtype=np.float32))

    @unittest.skipUnless(wp.is_cuda_available(), "Particle multilevel correction requires CUDA")
    def test_galerkin_pcg_matches_dense_positive_definite_solve(self):
        """Keep valid block systems accepted after adding curvature rejection."""
        device = wp.get_device("cuda:0")
        rng = np.random.default_rng(42)
        factor = rng.standard_normal((6, 6))
        dense = (factor.T @ factor + np.eye(6)).astype(np.float32)
        rhs_numpy = rng.standard_normal((2, 3)).astype(np.float32)
        blocks = np.array([dense[:3, :3], dense[:3, 3:], dense[3:, :3], dense[3:, 3:]])
        matrix = wp.array(blocks, dtype=wp.mat33, device=device)
        offsets = wp.array([0, 2, 4], dtype=wp.int32, device=device)
        columns = wp.array([0, 1, 0, 1], dtype=wp.int32, device=device)
        diagonal = wp.array([0, 3], dtype=wp.int32, device=device)
        rhs = wp.array(rhs_numpy, dtype=wp.vec3, device=device)
        vectors = [wp.zeros(2, dtype=wp.vec3, device=device) for _ in range(5)]
        inverse = wp.zeros(2, dtype=wp.mat33, device=device)
        status = wp.zeros(1, dtype=wp.int32, device=device)
        metrics = wp.zeros(17, dtype=float, device=device)
        counters = wp.zeros(2, dtype=wp.int32, device=device)
        wp.launch(
            _solve_energy_galerkin_pcg_persistent,
            dim=256,
            block_dim=256,
            inputs=[2, offsets, columns, diagonal, matrix, rhs, 12, True, 1.0e-4],
            outputs=[*vectors, inverse, status, metrics, counters],
            device=device,
        )
        self.assertEqual(int(status.numpy()[0]), 0)
        expected = np.linalg.solve(dense.astype(np.float64), rhs_numpy.ravel()).reshape((2, 3))
        np.testing.assert_allclose(vectors[0].numpy(), expected, rtol=1.0e-5, atol=1.0e-6)

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
            for operator in ("graph", "galerkin"):
                with self.subTest(solver=solver_type.__module__, operator=operator):
                    model = _build_cloth(device)
                    solver = solver_type(
                        model,
                        iterations=2,
                        particle_enable_multilevel_correction=True,
                        particle_multilevel_operator=operator,
                    )
                    state_0 = model.state()
                    state_1 = model.state()
                    control = model.control()
                    solver.step(state_0, state_1, control, None, 1.0 / 60.0)
                    with wp.ScopedCapture(device=device) as capture:
                        solver.step(state_1, state_0, control, None, 1.0 / 60.0)
                    wp.capture_launch(capture.graph)
                    wp.synchronize_device(device)

                    correction = solver.particle_multilevel
                    self.assertIsNotNone(correction)
                    self.assertEqual(correction.operator, operator)
                    self.assertTrue(np.isfinite(state_0.particle_q.numpy()).all())
                    if operator == "galerkin":
                        offsets = correction.coarse_matrix_offsets.numpy()
                        columns = correction.coarse_matrix_columns.numpy()
                        blocks = correction.coarse_matrix_blocks.numpy()
                        lookup = {
                            (row, int(columns[slot])): slot
                            for row in range(offsets.size - 1)
                            for slot in range(int(offsets[row]), int(offsets[row + 1]))
                        }
                        matrix_scale = float(np.max(np.abs(blocks)))
                        asymmetry = max(
                            float(np.max(np.abs(blocks[slot] - blocks[lookup[column, row]].T)))
                            for (row, column), slot in lookup.items()
                        )
                        self.assertLessEqual(asymmetry, 1.0e-5 * matrix_scale)

    @unittest.skipUnless(wp.is_cuda_available(), "Particle multilevel correction requires CUDA")
    def test_rigid_cluster_basis_executes_for_tetrahedra(self):
        """Execute the six-DOF cluster solve for a tetrahedral model."""
        device = wp.get_device("cuda:0")
        model = _build_tets(device)
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
        model = _build_tets(device)
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
        variants = (
            ("reference", 100, False, "graph", 8, 0.1),
            ("plain", 8, False, "graph", 8, 0.1),
            ("coarse", 8, True, "graph", 8, 0.1),
            ("galerkin", 8, True, "galerkin", 4, 0.6),
        )
        for name, iterations, multilevel, operator, cluster_size, relaxation in variants:
            state_in = model.state()
            state_out = model.state()
            state_in.particle_f.assign(forces)
            solver = SolverVBDSoft(
                model,
                iterations=iterations,
                particle_enable_multilevel_correction=multilevel,
                particle_multilevel_operator=operator,
                particle_multilevel_cluster_size=cluster_size,
                particle_multilevel_relaxation=relaxation,
            )
            solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
            results[name] = state_out.particle_q.numpy().copy()

        plain_error = np.linalg.norm(results["plain"] - results["reference"])
        coarse_error = np.linalg.norm(results["coarse"] - results["reference"])
        galerkin_error = np.linalg.norm(results["galerkin"] - results["reference"])
        middle = np.arange(dim_y + 1) * (dim_x + 1) + dim_x // 2
        plain_middle_motion = np.mean(np.abs(results["plain"][middle, 0] - rest[middle, 0]))
        coarse_middle_motion = np.mean(np.abs(results["coarse"][middle, 0] - rest[middle, 0]))
        galerkin_middle_motion = np.mean(np.abs(results["galerkin"][middle, 0] - rest[middle, 0]))

        self.assertLess(coarse_error, plain_error)
        self.assertLess(galerkin_error, plain_error)
        self.assertGreater(coarse_middle_motion, plain_middle_motion + 1.0e-5)
        self.assertGreater(galerkin_middle_motion, plain_middle_motion + 1.0e-5)

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
                with mock.patch.object(wp, "capture_if", side_effect=AssertionError("unexpected capture_if")):
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

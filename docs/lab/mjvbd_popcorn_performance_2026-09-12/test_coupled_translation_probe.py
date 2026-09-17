# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check equal/opposite reactions and the augmented contact Hessian."""

import unittest
from unittest import mock

import numpy as np
import warp as wp
from coupled_translation_probe import CoupledTranslationPCG, assemble_soft
from rigid_fusion_probe import RigidFusionAdapter, SoftInputs
from rigid_ritz_probe import RigidRitz

import newton
from newton._src.solvers.mjvbd_v2 import particle_multilevel as ml
from newton._src.solvers.mjvbd_v2.contact_projection import ContactProjection
from newton._src.solvers.mjvbd_v2.full_contact_pipeline import MJVBDV2CollisionPipeline
from newton._src.solvers.mjvbd_v2.vbd import rigid_vbd_kernels as rk
from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD


class TestCoupledTranslation(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Diagnostic targets CUDA")
    def test_actual_augmented_step_against_dense_system(self):
        """Solve the captured particle/body operator against an independent dense solve."""
        builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
        builder.add_ground_plane()
        for x in (-0.035, 0.035):
            body = builder.add_body(xform=wp.transform(wp.vec3(x, 0, 0.039), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.04, cfg=builder.ShapeConfig(density=10))
        builder.add_cloth_grid(
            pos=wp.vec3(-0.06, -0.02, 0.076),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=6,
            dim_y=2,
            cell_x=0.02,
            cell_y=0.02,
            mass=0.001,
            tri_ke=1e4,
            tri_ka=1e4,
            tri_kd=0.01,
            edge_ke=0.001,
            particle_radius=0.003,
        )
        builder.color()
        model = builder.finalize(device="cuda:0")
        original_clusters = ml._build_clusters
        with mock.patch.object(ml, "_build_clusters", lambda model, size: original_clusters(model, 1)):
            solver = SolverVBD(
                model,
                iterations=2,
                rigid_contact_hard=False,
                particle_enable_multilevel_correction=True,
                particle_multilevel_operator="galerkin",
                particle_multilevel_checkpoints=(2,),
                particle_multilevel_coarse_iterations=64,
                particle_multilevel_fallback_iterations=2,
                rigid_body_particle_contact_buffer_size=1024,
            )
        correction = solver.particle_multilevel
        ritz = RigidRitz(model, correction, 400)
        adapter = RigidFusionAdapter(wp.launch, model)
        coupled = CoupledTranslationPCG(solver, correction, ritz, adapter)
        correction.coarse_use_split_pcg = True
        correction._split_coarse_pcg = coupled
        original_restrict = correction.restrict_and_prolong

        def restrict(model, q, displacements, dt):
            ritz.q = q
            original_restrict(model, q, displacements, dt)

        correction.restrict_and_prolong = restrict
        original_launch = wp.launch

        def launch(*args, **kwargs):
            kernel = kwargs.get("kernel", args[0] if args else None)
            if kernel in (
                rk.accumulate_body_particle_contacts_per_body,
                rk.accumulate_body_body_contacts_per_body,
                rk.accumulate_body_particle_contact_dense_partials,
                rk.accumulate_body_particle_contact_dense_reduction,
                rk.accumulate_body_particle_contact_dense_single,
                rk.solve_rigid_body,
            ):
                return adapter(*args, **kwargs)
            result = original_launch(*args, **kwargs)
            if kernel is ml._commit_prolonged_corrections:
                coupled.commit()
            return result

        pipeline = MJVBDV2CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
        state, output = model.state(), model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        control = model.control()
        with mock.patch.object(wp, "launch", launch):
            solver.step(state, output, control, contacts, 1 / 960)
            with wp.ScopedCapture(device=model.device) as capture:
                solver.step(state, output, control, contacts, 1 / 960)
        for _ in range(2):
            wp.capture_launch(capture.graph)
            self.assertEqual(int(correction.runtime_status.numpy()[0]), 0)
            count = coupled.count
            matrix = np.zeros((3 * count, 3 * count))
            offsets, columns, blocks = coupled.offsets.numpy(), coupled.columns.numpy(), coupled.blocks.numpy()
            for row in range(count):
                for slot in range(offsets[row], offsets[row + 1]):
                    col = columns[slot]
                    matrix[3 * row : 3 * row + 3, 3 * col : 3 * col + 3] += blocks[slot]
            data = coupled.contacts.data
            keys, pairs, cross = data.edge_keys.numpy(), data.edge_clusters.numpy(), data.edge_blocks.numpy()
            self.assertGreater(int(data.edge_count.numpy()[0]), 0)
            for slot in np.flatnonzero(keys >= 0):
                row, col = pairs[slot]
                matrix[3 * row : 3 * row + 3, 3 * col : 3 * col + 3] += cross[slot]
                matrix[3 * col : 3 * col + 3, 3 * row : 3 * row + 3] += cross[slot].T
            rhs = coupled.rhs.numpy().reshape(-1)
            scale = float(coupled.scale.numpy()[0])
            self.assertGreater(scale, 0)
            actual = coupled.work[0].numpy().reshape(-1)
            expected = np.linalg.solve(matrix, rhs) * scale
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-7)
            relative = np.linalg.norm(matrix @ actual - scale * rhs) / max(np.linalg.norm(scale * rhs), 1e-20)
            self.assertLess(relative, 1e-3)

    @unittest.skipUnless(wp.is_cuda_available(), "Diagnostic targets CUDA")
    def test_point_edge_face_augmented_operator(self):
        """Match k*w*w^T including the negative body weight for all contact kinds."""
        device = "cuda:0"
        for weights in ([1.0, 0, 0], [0.25, 0.75, 0], [0.2, 0.3, 0.5]):
            with self.subTest(weights=weights):
                data = SoftInputs()
                data.dt = 1 / 960
                data.particle_q = wp.zeros(3, dtype=wp.vec3, device=device)
                data.particle_q_prev = wp.zeros(3, dtype=wp.vec3, device=device)
                data.particle_radius = wp.full(3, 0.001, device=device)
                data.body_q = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
                data.body_q_prev = wp.clone(data.body_q)
                data.body_com = wp.zeros(1, dtype=wp.vec3, device=device)
                data.body_qd = wp.zeros(1, dtype=wp.spatial_vector, device=device)
                data.shape_body = wp.array([0], dtype=int, device=device)
                data.body_particle_contact_penalty_k = wp.full(1, 1000.0, device=device)
                data.body_particle_contact_material_kd = wp.zeros(1, device=device)
                data.body_particle_contact_material_mu = wp.zeros(1, device=device)
                data.body_particle_contact_count = wp.array([1], dtype=int, device=device)
                corners = [i if value else -1 for i, value in enumerate(weights)]
                data.soft_contact_indices = wp.array([corners], dtype=wp.vec3i, device=device)
                data.soft_contact_barycentric = wp.array([weights], dtype=wp.vec3, device=device)
                data.body_particle_contact_shape = wp.array([0], dtype=int, device=device)
                data.body_particle_contact_body_pos = wp.zeros(1, dtype=wp.vec3, device=device)
                data.body_particle_contact_body_vel = wp.zeros(1, dtype=wp.vec3, device=device)
                data.body_particle_contact_normal = wp.array([[0, 0, 1]], dtype=wp.vec3, device=device)
                data.shape_margin = wp.zeros(1, device=device)
                data.friction_epsilon = 1e-4
                hessian = np.diag([0.0, 0.0, 1000.0])
                diagonal = np.array([2 * np.eye(3) + weight * weight * hessian for weight in [*weights, 0]])
                blocks = wp.array(diagonal, dtype=wp.mat33, device=device)
                force = np.zeros((4, 3))
                force[:3, 2] = weights
                rhs = wp.array(force, dtype=wp.vec3, device=device)
                projection = ContactProjection(4, 1, device)
                identity = wp.array(np.arange(4), dtype=int, device=device)
                body_rows = wp.array([3], dtype=int, device=device)
                projection.reset()
                wp.launch(
                    assemble_soft,
                    dim=4096,
                    inputs=[data, body_rows, identity, identity, blocks, rhs, projection.data],
                    device=device,
                )
                self.assertEqual(int(projection.data.overflow.numpy()[0]), 0)
                actual = np.zeros((12, 12))
                for row, block in enumerate(blocks.numpy()):
                    actual[3 * row : 3 * row + 3, 3 * row : 3 * row + 3] = block
                keys = projection.data.edge_keys.numpy()
                pairs = projection.data.edge_clusters.numpy()
                cross = projection.data.edge_blocks.numpy()
                for slot in np.flatnonzero(keys >= 0):
                    row, col = pairs[slot]
                    actual[3 * row : 3 * row + 3, 3 * col : 3 * col + 3] = cross[slot]
                    actual[3 * col : 3 * col + 3, 3 * row : 3 * row + 3] = cross[slot].T
                all_weights = np.array([*weights, -1.0])
                expected = 2 * np.eye(12) + np.kron(np.outer(all_weights, all_weights), hessian)
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-4)
                force[3, 2] = -1.0
                np.testing.assert_allclose(rhs.numpy(), force, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(rhs.numpy().sum(axis=0), 0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

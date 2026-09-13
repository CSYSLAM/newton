# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate opted-in CUDA scheduling against unchanged solver paths."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjvbd_v2.coupled_free_body.two_level import _build_kernels
from newton._src.solvers.mjvbd_v2.full_contact_pipeline import MJVBDV2CollisionPipeline
from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestSelfContactCertificate(unittest.TestCase):
    def test_motion_and_contact(self):
        """Preserve reference detections under small motion, jumps, and new overlaps."""
        builder = newton.ModelBuilder()
        points = np.array(
            [[0, 0, 0], [0.02, 0, 0], [0, 0.02, 0], [0, 0, 0.03], [0.02, 0, 0.03], [0, 0.02, 0.03]], dtype=np.float32
        )
        builder.add_cloth_mesh(
            pos=wp.vec3(),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            vertices=points,
            indices=[0, 1, 2, 3, 4, 5],
            density=1.0,
            scale=1.0,
        )
        builder.color()
        model = builder.finalize(device="cuda:0")
        original_detection = SolverVBD._collision_detection_penetration_free
        solvers = [
            SolverVBD(
                model,
                enable_cuda_fast_path=fast,
                iterations=2,
                particle_enable_self_contact=True,
                particle_self_contact_radius=0.001,
                particle_self_contact_margin=0.003,
                particle_rest_shape_contact_exclusion_radius=0.015,
                particle_enable_truncation_cache=True,
            )
            for fast in (False, True)
        ]
        states = [model.state(), model.state()]
        detect = [
            lambda: original_detection(solvers[0], states[0]),
            lambda: solvers[1]._collision_detection_penetration_free(states[1]),
        ]
        for function in detect:
            function()
        graphs = []
        for function in detect:
            with wp.ScopedCapture(device=model.device) as capture:
                function()
            graphs.append(capture.graph)
        poses = []
        for translation, gap in (
            (0.0001, 0.03),
            (0.0002, 0.03),
            (0.02, 0.03),
            (0.0201, 0.002),
            (0.0202, 0.0005),
            (0.04, 0.03),
            (0.0401, 0.03),
            (0.0402, 0.0034),
            (0.0402, 0.0029),
        ):
            pose = points.copy()
            pose[:, 0] += translation
            pose[3:, 2] = gap
            poses.append((pose, gap < 0.003))
        rng = np.random.default_rng(638)
        for _ in range(48):
            pose = points.copy()
            rotation, _unused = np.linalg.qr(rng.normal(size=(3, 3)))
            pose[3:] = (points[3:] - points[3:].mean(axis=0)) @ rotation
            pose[3:] += rng.uniform(-0.015, 0.015, (1, 3))
            pose += rng.uniform(-2.0, 2.0, (1, 3))
            poses.append((pose, False))
            poses.append((pose + rng.uniform(-0.0001, 0.0001, pose.shape).astype(np.float32), False))
        for pose, expect_contact in poses:
            for state, graph in zip(states, graphs, strict=True):
                state.particle_q.assign(pose)
                wp.capture_launch(graph)
            a, b = [solver.trimesh_collision_detector for solver in solvers]
            for name in (
                "vertex_colliding_triangles_count",
                "vertex_colliding_triangles_min_dist",
                "triangle_colliding_vertices_min_dist",
                "edge_colliding_edges_count",
                "edge_colliding_edges_min_dist",
                "vertex_colliding_triangles",
                "edge_colliding_edges",
            ):
                np.testing.assert_array_equal(getattr(a, name).numpy(), getattr(b, name).numpy())
            for solver in solvers:
                np.testing.assert_array_equal(solver.pos_prev_collision_detection.numpy(), pose)
                np.testing.assert_array_equal(solver.particle_displacements.numpy(), 0.0)
                np.testing.assert_array_equal(solver.truncation_ts.numpy(), 1.0)
            if expect_contact:
                self.assertGreater(int(b.vertex_colliding_triangles_count.numpy().sum()), 0)
        statistics = solvers[1]._self_contact_certificate.statistics.numpy()
        print("rebuild / empty certificates / reuses:", statistics.tolist(), flush=True)
        self.assertGreater(int(statistics[1]), 0)
        self.assertGreater(int(statistics[2]), 0)
        certificate = solvers[1]._self_contact_certificate
        certificate.valid.fill_(1)
        solvers[1].notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        np.testing.assert_array_equal(certificate.valid.numpy(), 0)
        certificate.valid.fill_(1)
        solvers[1].rebuild_bvh(states[1])
        np.testing.assert_array_equal(certificate.valid.numpy(), 0)


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestBatchedSurface(unittest.TestCase):
    def test_contact_branches(self):
        """Preserve both empty self-contact acceleration and active-contact fallback."""
        original_step = SolverVBD.step
        for gap in (0.05, 0.001):
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            for height in (0.002, 0.002 + gap):
                builder.add_cloth_grid(
                    pos=wp.vec3(-0.04, -0.04, height),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(),
                    dim_x=4,
                    dim_y=4,
                    cell_x=0.02,
                    cell_y=0.02,
                    mass=0.001,
                    tri_ke=1e4,
                    tri_ka=1e4,
                    tri_kd=0.01,
                    edge_ke=0.01,
                    particle_radius=0.001,
                )
            builder.color()
            model = builder.finalize(device="cuda:0")
            pipeline = MJVBDV2CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
            contacts = pipeline.contacts()
            control = model.control()
            states = [(model.state(), model.state()) for fast in (False, True)]
            pipeline.collide(states[0][0], contacts)
            solvers = [
                SolverVBD(
                    model,
                    enable_cuda_fast_path=fast,
                    iterations=8,
                    particle_enable_self_contact=True,
                    particle_enable_truncation_cache=True,
                    particle_self_contact_radius=0.001,
                    particle_self_contact_margin=0.0025,
                    particle_rest_shape_contact_exclusion_radius=0.0,
                )
                for fast in (False, True)
            ]
            graphs = []
            for index, solver in enumerate(solvers):
                if index == 1:
                    self.assertTrue(solver._cuda_surface.eligible)
                function = original_step if index == 0 else SolverVBD.step
                function(solver, *states[index], control, contacts, 1 / 480)
                with wp.ScopedCapture(device=model.device) as capture:
                    function(solver, *states[index], control, contacts, 1 / 480)
                graphs.append(capture.graph)
            for _ in range(8):
                for graph in graphs:
                    wp.capture_launch(graph)
                for name in ("particle_q", "particle_qd"):
                    np.testing.assert_allclose(
                        getattr(states[0][1], name).numpy(), getattr(states[1][1], name).numpy(), atol=2e-6, rtol=2e-5
                    )

            print("gap / fallback:", gap, solvers[1]._cuda_surface.fallback.numpy().tolist(), flush=True)


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestInverseFactor(unittest.TestCase):
    def test_conditioned_spd(self):
        """Preserve positive preconditioning and solve accuracy across basis sizes."""
        rng = np.random.default_rng(859)
        device = "cuda:0"
        for width in (6, 18, 30):
            original_factor, original_update = _build_kernels(width)
            factor_kernel, update = _build_kernels(width, True)
            count = width // 6
            q = wp.zeros(count, dtype=wp.vec3, device=device)
            centers = wp.zeros_like(q)
            groups = wp.array(np.arange(count), dtype=int, device=device)
            inverse = wp.zeros(count, dtype=wp.mat33, device=device)
            for condition in (1.0, 1e2, 1e4, 1e6):
                orthogonal, _ = np.linalg.qr(rng.normal(size=(width, width)))
                matrix = (orthogonal @ np.diag(np.geomspace(1, condition, width)) @ orthogonal.T).astype(np.float32)
                matrix = (matrix + matrix.T) * np.float32(0.5)
                device_matrix = wp.array(matrix, device=device)
                residual = rng.normal(size=(count, 3)).astype(np.float32)
                rhs = np.zeros(width)
                for row in range(count):
                    rhs[6 * row : 6 * row + 3] = residual[row]
                reference = np.linalg.solve(matrix.astype(float), rhs)
                expected = np.stack([reference[6 * row : 6 * row + 3] for row in range(count)])
                errors = []
                for factor, solve in ((original_factor, original_update), (factor_kernel, update)):
                    output_factor = wp.empty_like(device_matrix)
                    wp.launch(factor, 256, [device_matrix, output_factor], block_dim=256, device=device)
                    x, z, p, product = [wp.zeros_like(q) for _ in range(4)]
                    r = wp.array(residual, dtype=wp.vec3, device=device)
                    status = wp.zeros(1, dtype=int, device=device)
                    metrics = wp.zeros(20, device=device)
                    rz = wp.ones(1, device=device)
                    wp.launch(
                        solve,
                        256,
                        [
                            count,
                            -1,
                            q,
                            centers,
                            groups,
                            output_factor,
                            x,
                            r,
                            z,
                            p,
                            product,
                            inverse,
                            status,
                            metrics,
                            rz,
                        ],
                        block_dim=256,
                        device=device,
                    )
                    actual = z.numpy()
                    self.assertEqual(int(status.numpy()[0]), 0)
                    self.assertGreater(float(np.sum(actual * residual)), 0.0)
                    error = np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30)
                    errors.append(error)
                self.assertLess(errors[1], max(2e-5, errors[0] * 1.1 + 2e-6))
                self.assertLess(errors[1], 0.03)
                print("SPD", width, condition, "relative errors", errors, flush=True)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test MJVBDV2-private rigid-soft contact optimizations."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.kernels import create_soft_contacts
from newton._src.solvers.mjvbd_v2 import collision_pipeline, soft_contact_pipeline
from newton._src.solvers.mjvbd_v2.full_contact_pipeline import MJVBDV2CollisionPipeline
from newton._src.solvers.mjvbd_v2.vbd import particle_vbd_kernels as complete_particle_kernels
from newton._src.solvers.mjvbd_v2.vbd import rigid_vbd_kernels as complete_rigid_kernels
from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD as SolverVBDComplete
from newton._src.solvers.mjvbd_v2.vbd_soft import particle_vbd_kernels as soft_particle_kernels
from newton._src.solvers.mjvbd_v2.vbd_soft import rigid_vbd_kernels as soft_rigid_kernels
from newton._src.solvers.mjvbd_v2.vbd_soft.solver_vbd import SolverVBD as SolverVBDSoft

_PARTICLE_KERNEL_VARIANTS = (
    ("complete", complete_particle_kernels, True),
    ("soft", soft_particle_kernels, False),
)
_RIGID_KERNEL_VARIANTS = (
    ("complete", complete_rigid_kernels),
    ("soft", soft_rigid_kernels),
)


def _launch_reference_soft_contacts(model, pairs, particle_state, rigid_state, contacts, margin):
    contacts.clear()
    wp.launch(
        create_soft_contacts,
        dim=pairs.shape[0],
        inputs=[
            pairs,
            particle_state.particle_q,
            model.particle_radius,
            model.particle_flags,
            model.particle_world,
            rigid_state.body_q,
            model.shape_transform,
            model.shape_body,
            model.shape_type,
            model.shape_scale,
            model.shape_source_ptr,
            model._shape_mesh_properties,
            model.shape_world,
            margin,
            model.shape_margin,
            contacts.soft_contact_max,
            model.shape_flags,
            model.shape_heightfield_index,
            model.heightfield_data,
            model.heightfield_elevations,
        ],
        outputs=[
            contacts.soft_contact_count,
            contacts.soft_contact_particle,
            contacts.soft_contact_indices,
            contacts.soft_contact_barycentric,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
            contacts.soft_contact_tids,
        ],
        device=model.device,
    )


def _sorted_soft_contact_records(contacts):
    count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
    particles = contacts.soft_contact_particle.numpy()[:count]
    shapes = contacts.soft_contact_shape.numpy()[:count]
    order = np.lexsort((shapes, particles))
    return (
        particles[order],
        shapes[order],
        contacts.soft_contact_indices.numpy()[:count][order],
        contacts.soft_contact_barycentric.numpy()[:count][order],
        contacts.soft_contact_body_pos.numpy()[:count][order],
        contacts.soft_contact_body_vel.numpy()[:count][order],
        contacts.soft_contact_normal.numpy()[:count][order],
    )


def _assert_soft_contact_records_equal(test_case, expected, actual):
    for expected_array, actual_array in zip(expected[:3], actual[:3], strict=True):
        test_case.assertTrue(np.array_equal(actual_array, expected_array))
    for expected_array, actual_array in zip(expected[3:], actual[3:], strict=True):
        np.testing.assert_allclose(actual_array, expected_array, rtol=1.0e-6, atol=1.0e-6)


def _make_point_contact_data(device, capacity=7):
    particle_count = 4
    particle_q = wp.array(
        [[0.02 * particle, 0.0, 0.04] for particle in range(particle_count)],
        dtype=wp.vec3,
        device=device,
    )
    particle_q_prev = wp.array(
        [[0.02 * particle, 0.0, 0.05] for particle in range(particle_count)],
        dtype=wp.vec3,
        device=device,
    )
    particle_colors = wp.array([0, 1, 0, 1], dtype=int, device=device)
    color_groups = (
        wp.array([0, 2], dtype=wp.int32, device=device),
        wp.array([1, 3], dtype=wp.int32, device=device),
    )
    base_forces = np.arange(particle_count * 3, dtype=np.float32).reshape(particle_count, 3) * 0.01
    base_hessians = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], particle_count, axis=0) * 0.02
    return {
        "capacity": capacity,
        "particle_count": particle_count,
        "particle_q": particle_q,
        "particle_q_prev": particle_q_prev,
        "particle_colors": particle_colors,
        "particle_radius": wp.full(particle_count, 0.1, dtype=float, device=device),
        "color_groups": color_groups,
        "contact_indices": wp.array(
            [[contact % particle_count, -1, -1] for contact in range(capacity)],
            dtype=wp.vec3i,
            device=device,
        ),
        "contact_penalty_k": wp.array([100.0 + contact for contact in range(capacity)], dtype=float, device=device),
        "contact_material_ke": wp.full(capacity, 200.0, dtype=float, device=device),
        "contact_material_kd": wp.full(capacity, 3.0, dtype=float, device=device),
        "contact_material_mu": wp.zeros(capacity, dtype=float, device=device),
        "shape_body": wp.array([-1], dtype=int, device=device),
        "body_q": wp.zeros(0, dtype=wp.transform, device=device),
        "body_q_prev": wp.zeros(0, dtype=wp.transform, device=device),
        "body_qd": wp.zeros(0, dtype=wp.spatial_vector, device=device),
        "body_com": wp.zeros(0, dtype=wp.vec3, device=device),
        "contact_shape": wp.zeros(capacity, dtype=int, device=device),
        "contact_body_pos": wp.zeros(capacity, dtype=wp.vec3, device=device),
        "contact_body_vel": wp.zeros(capacity, dtype=wp.vec3, device=device),
        "contact_normal": wp.array([[0.0, 0.0, 1.0]] * capacity, dtype=wp.vec3, device=device),
        "shape_margin": wp.zeros(1, dtype=float, device=device),
        "contact_barycentric": wp.array([[1.0, 0.0, 0.0]] * capacity, dtype=wp.vec3, device=device),
        "base_forces": base_forces,
        "base_hessians": base_hessians,
    }


def _contact_material_inputs(data):
    return [
        data["contact_penalty_k"],
        data["contact_material_kd"],
        data["contact_material_mu"],
        data["shape_body"],
        data["body_q"],
        data["body_q_prev"],
        data["body_qd"],
        data["body_com"],
        data["contact_shape"],
        data["contact_body_pos"],
        data["contact_body_vel"],
        data["contact_normal"],
        data["shape_margin"],
        data["contact_barycentric"],
    ]


def _make_body_particle_reaction_data(device, contact_count=192, capacity=256):
    particle_count = 8
    body_count = 2
    particle_q = np.zeros((particle_count, 3), dtype=np.float32)
    particle_q[:, 0] = np.linspace(-0.14, 0.14, particle_count)
    particle_q[:, 2] = 0.04
    particle_q_prev = particle_q.copy()
    particle_q_prev[:, 2] = 0.045
    body_positions = np.zeros((capacity, 3), dtype=np.float32)
    body_positions[:, 0] = np.linspace(-0.12, 0.12, capacity)
    return {
        "dt": 1.0 / 120.0,
        "contact_count": wp.array([contact_count], dtype=int, device=device),
        "capacity": capacity,
        "body_count": body_count,
        "color_group": wp.array([0, 1], dtype=wp.int32, device=device),
        "particle_q": wp.array(particle_q, dtype=wp.vec3, device=device),
        "particle_q_prev": wp.array(particle_q_prev, dtype=wp.vec3, device=device),
        "particle_radius": wp.full(particle_count, 0.1, dtype=float, device=device),
        "body_q": wp.array([wp.transform_identity()] * body_count, dtype=wp.transform, device=device),
        "body_q_prev": wp.array([wp.transform_identity()] * body_count, dtype=wp.transform, device=device),
        "body_qd": wp.zeros(body_count, dtype=wp.spatial_vector, device=device),
        "body_com": wp.zeros(body_count, dtype=wp.vec3, device=device),
        "body_inv_mass": wp.array([1.0, 0.0], dtype=float, device=device),
        "shape_body": wp.array([0], dtype=int, device=device),
        "contact_penalty_k": wp.full(capacity, 200.0, dtype=float, device=device),
        "contact_material_ke": wp.full(capacity, 200.0, dtype=float, device=device),
        "contact_material_kd": wp.full(capacity, 2.0, dtype=float, device=device),
        "contact_material_mu": wp.full(capacity, 0.2, dtype=float, device=device),
        "contact_indices": wp.array(
            [[contact % particle_count, -1, -1] for contact in range(capacity)],
            dtype=wp.vec3i,
            device=device,
        ),
        "contact_shape": wp.zeros(capacity, dtype=int, device=device),
        "contact_body_pos": wp.array(body_positions, dtype=wp.vec3, device=device),
        "contact_body_vel": wp.zeros(capacity, dtype=wp.vec3, device=device),
        "contact_normal": wp.array([[0.0, 0.0, 1.0]] * capacity, dtype=wp.vec3, device=device),
        "contact_barycentric": wp.array([[1.0, 0.0, 0.0]] * capacity, dtype=wp.vec3, device=device),
        "shape_margin": wp.zeros(1, dtype=float, device=device),
        "body_contact_counts": wp.array([contact_count, 0], dtype=wp.int32, device=device),
        "body_contact_indices": wp.array(
            np.concatenate([np.arange(capacity, dtype=np.int32), np.zeros(capacity, dtype=np.int32)]),
            dtype=wp.int32,
            device=device,
        ),
    }


def _body_particle_accumulation_outputs(device, body_count):
    return (
        wp.zeros(body_count, dtype=wp.vec3, device=device),
        wp.zeros(body_count, dtype=wp.vec3, device=device),
        wp.zeros(body_count, dtype=wp.mat33, device=device),
        wp.zeros(body_count, dtype=wp.mat33, device=device),
        wp.zeros(body_count, dtype=wp.mat33, device=device),
    )


def _launch_legacy_contacts(kernels, uses_color_masks, data, contact_count, forces, hessians, device):
    for color, _color_group in enumerate(data["color_groups"]):
        color_mask_inputs = []
        if uses_color_masks:
            color_mask_inputs = [wp.zeros(data["capacity"], dtype=wp.uint32, device=device), False]
        wp.launch(
            kernels.accumulate_particle_body_contact_force_and_hessian,
            dim=data["capacity"],
            inputs=[
                0.01,
                color,
                data["particle_q_prev"],
                data["particle_q"],
                data["particle_colors"],
                *color_mask_inputs,
                1.0,
                data["particle_radius"],
                data["contact_indices"],
                contact_count,
                data["capacity"],
                data["contact_penalty_k"],
                data["contact_material_ke"],
                *_contact_material_inputs(data)[1:],
            ],
            outputs=[forces, hessians],
            device=device,
        )


def _launch_gather_contacts(kernels, data, contact_count, forces, hessians, device):
    worker_count = 3
    contact_head = wp.full(data["particle_count"], -1, dtype=int, device=device)
    contact_next = wp.empty(data["capacity"], dtype=int, device=device)
    wp.launch(
        kernels.build_particle_body_contact_adjacency_active,
        dim=worker_count,
        inputs=[
            data["contact_indices"],
            contact_count,
            data["capacity"],
            worker_count,
            contact_head,
            contact_next,
        ],
        device=device,
    )
    for color_group in data["color_groups"]:
        wp.launch(
            kernels.gather_particle_body_contact_force_and_hessian,
            dim=color_group.size,
            inputs=[
                0.01,
                color_group,
                data["particle_q_prev"],
                data["particle_q"],
                1.0,
                data["particle_radius"],
                data["contact_indices"],
                contact_head,
                contact_next,
                *_contact_material_inputs(data)[:-1],
            ],
            outputs=[forces, hessians],
            device=device,
        )


def _launch_active_soft_contacts(data, contact_count, forces, hessians, device):
    worker_count = 3
    for color, _color_group in enumerate(data["color_groups"]):
        wp.launch(
            soft_particle_kernels.accumulate_particle_body_contact_force_and_hessian_active,
            dim=worker_count,
            inputs=[
                0.01,
                color,
                data["particle_q_prev"],
                data["particle_q"],
                data["particle_colors"],
                1.0,
                data["particle_radius"],
                data["contact_indices"],
                contact_count,
                data["capacity"],
                worker_count,
                data["contact_penalty_k"],
                data["contact_material_ke"],
                *_contact_material_inputs(data)[1:],
            ],
            outputs=[forces, hessians],
            device=device,
        )


def _make_dual_data(device, capacity=7):
    particle_count = 4
    return {
        "capacity": capacity,
        "indices": wp.array(
            [[contact % particle_count, -1, -1] for contact in range(capacity)],
            dtype=wp.vec3i,
            device=device,
        ),
        "shape": wp.zeros(capacity, dtype=int, device=device),
        "body_pos": wp.array(
            [[0.0, 0.0, 0.01 * (contact + 1)] for contact in range(capacity)],
            dtype=wp.vec3,
            device=device,
        ),
        "normal": wp.array([[0.0, 0.0, 1.0]] * capacity, dtype=wp.vec3, device=device),
        "barycentric": wp.array([[1.0, 0.0, 0.0]] * capacity, dtype=wp.vec3, device=device),
        "particle_q": wp.zeros(particle_count, dtype=wp.vec3, device=device),
        "particle_radius": wp.full(particle_count, 0.1, dtype=float, device=device),
        "shape_body": wp.array([-1], dtype=int, device=device),
        "shape_margin": wp.zeros(1, dtype=float, device=device),
        "body_q": wp.zeros(0, dtype=wp.transform, device=device),
        "material_ke": wp.full(capacity, 100.0, dtype=float, device=device),
    }


class TestMJVBDV2ContactOptimizations(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Dense rigid-side reduction requires CUDA")
    def test_dense_body_particle_reduction_matches_legacy(self):
        """Match legacy rigid reactions with dense contact chunk reduction."""
        device = wp.get_device("cuda:0")
        data = _make_body_particle_reaction_data(device, contact_count=1024, capacity=1024)
        legacy = _body_particle_accumulation_outputs(device, data["body_count"])
        dense = _body_particle_accumulation_outputs(device, data["body_count"])
        common_inputs = [
            data["dt"],
            data["color_group"],
            data["particle_q"],
            data["particle_q_prev"],
            data["particle_radius"],
            data["body_q_prev"],
            data["body_q"],
            data["body_qd"],
            data["body_com"],
            data["body_inv_mass"],
            data["shape_body"],
            1.0,
            data["contact_penalty_k"],
            data["contact_material_ke"],
            data["contact_material_kd"],
            data["contact_material_mu"],
            data["contact_count"],
            data["contact_indices"],
            data["contact_shape"],
            data["contact_body_pos"],
            data["contact_body_vel"],
            data["contact_normal"],
            data["contact_barycentric"],
            data["shape_margin"],
            data["capacity"],
            data["body_contact_counts"],
            data["body_contact_indices"],
        ]
        wp.launch(
            complete_rigid_kernels.accumulate_body_particle_contacts_per_body,
            dim=data["color_group"].size * 4,
            inputs=[*common_inputs, 0],
            outputs=list(legacy),
            device=device,
        )

        block_dim = 64
        chunks_per_body = data["capacity"] // block_dim
        partial_count = data["body_count"] * chunks_per_body
        partials = (
            wp.zeros(partial_count, dtype=wp.vec3, device=device),
            wp.zeros(partial_count, dtype=wp.vec3, device=device),
            wp.zeros(partial_count, dtype=wp.mat33, device=device),
            wp.zeros(partial_count, dtype=wp.mat33, device=device),
            wp.zeros(partial_count, dtype=wp.mat33, device=device),
        )
        wp.launch(
            complete_rigid_kernels.accumulate_body_particle_contact_dense_partials,
            dim=data["color_group"].size * chunks_per_body * block_dim,
            block_dim=block_dim,
            inputs=[
                data["dt"],
                data["color_group"],
                chunks_per_body,
                128,
                data["particle_q"],
                data["particle_q_prev"],
                data["particle_radius"],
                data["body_q_prev"],
                data["body_q"],
                data["body_qd"],
                data["body_com"],
                data["shape_body"],
                1.0,
                data["contact_penalty_k"],
                data["contact_material_kd"],
                data["contact_material_mu"],
                data["contact_count"],
                data["contact_indices"],
                data["contact_shape"],
                data["contact_body_pos"],
                data["contact_body_vel"],
                data["contact_normal"],
                data["contact_barycentric"],
                data["shape_margin"],
                data["capacity"],
                data["body_contact_counts"],
                data["body_contact_indices"],
            ],
            outputs=list(partials),
            device=device,
        )
        wp.launch(
            complete_rigid_kernels.accumulate_body_particle_contact_dense_reduction,
            dim=data["color_group"].size * block_dim,
            block_dim=block_dim,
            inputs=[
                data["color_group"],
                chunks_per_body,
                128,
                data["capacity"],
                data["body_contact_counts"],
                *partials,
            ],
            outputs=list(dense),
            device=device,
        )

        for legacy_array, dense_array in zip(legacy, dense, strict=True):
            np.testing.assert_allclose(dense_array.numpy(), legacy_array.numpy(), rtol=2.0e-5, atol=5.0e-4)

    @unittest.skipUnless(wp.is_cuda_available(), "Private full-surface pruning requires CUDA")
    def test_private_full_surface_pipeline_matches_shared_contacts(self):
        """Preserve full-surface contacts while pruning gap-only shape pairs."""
        device = wp.get_device("cuda:0")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_shape_box(
            body=-1,
            hx=0.5,
            hy=0.5,
            hz=0.5,
            cfg=newton.ModelBuilder.ShapeConfig(margin=0.1),
        )
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.12), wp.quat_identity()),
            hx=0.5,
            hy=0.5,
            hz=0.5,
        )
        builder.add_cloth_grid(
            pos=wp.vec3(-1.0, -1.0, 0.56),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=1,
            dim_y=1,
            cell_x=2.0,
            cell_y=2.0,
            mass=0.1,
            particle_radius=0.01,
        )
        builder.color()
        model = builder.finalize(device=device)
        options = {
            "broad_phase": "nxn",
            "soft_contact_margin": 0.0,
            "enable_rigid_soft_full_surface_contact": True,
        }
        shared_pipeline = newton.CollisionPipeline(model, **options)
        private_pipeline = MJVBDV2CollisionPipeline(model, **options)
        differentiable_pipeline = MJVBDV2CollisionPipeline(model, **options, requires_grad=True)
        self.assertFalse(differentiable_pipeline._use_soft_surface_compaction)
        shared_contacts = shared_pipeline.contacts()
        private_contacts = private_pipeline.contacts()
        state = model.state()

        shared_pipeline.collide(state, shared_contacts)
        private_pipeline.collide(state, private_contacts)

        edge_active = private_pipeline._soft_edge_pair_active.numpy()
        face_active = private_pipeline._soft_face_pair_active.numpy()
        edge_pairs = private_pipeline.soft_edge_rigid_pairs.numpy()
        face_pairs = private_pipeline.soft_face_rigid_pairs.numpy()
        self.assertGreater(int(np.count_nonzero(edge_active)), 0)
        self.assertGreater(int(np.count_nonzero(edge_active == 0)), 0)
        self.assertGreater(int(np.count_nonzero(face_active)), 0)
        self.assertGreater(int(np.count_nonzero(face_active == 0)), 0)
        np.testing.assert_array_equal(edge_active[edge_pairs[:, 1] == 1], 0)
        np.testing.assert_array_equal(face_active[face_pairs[:, 1] == 1], 0)
        self.assertTrue(private_pipeline._use_soft_surface_compaction)
        compact_counts = private_pipeline._soft_surface_compact_counts.numpy()
        edge_compact = private_pipeline._soft_edge_compact_pair_indices.numpy()[: compact_counts[0]]
        face_compact = private_pipeline._soft_face_compact_pair_indices.numpy()[: compact_counts[1]]
        np.testing.assert_array_equal(np.sort(edge_compact), np.flatnonzero(edge_active))
        np.testing.assert_array_equal(np.sort(face_compact), np.flatnonzero(face_active))

        def sorted_records(contacts):
            count = int(contacts.soft_contact_count.numpy()[0])
            indices = contacts.soft_contact_indices.numpy()[:count]
            shapes = contacts.soft_contact_shape.numpy()[:count]
            order = np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0], shapes))
            return (
                shapes[order],
                indices[order],
                contacts.soft_contact_particle.numpy()[:count][order],
                contacts.soft_contact_barycentric.numpy()[:count][order],
                contacts.soft_contact_body_pos.numpy()[:count][order],
                contacts.soft_contact_normal.numpy()[:count][order],
            )

        def assert_record_tuples_equal(expected_records, actual_records):
            for expected, actual in zip(expected_records[:3], actual_records[:3], strict=True):
                np.testing.assert_array_equal(actual, expected)
            for expected, actual in zip(expected_records[3:], actual_records[3:], strict=True):
                np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)

        def assert_records_equal():
            assert_record_tuples_equal(sorted_records(shared_contacts), sorted_records(private_contacts))

        assert_records_equal()
        compact_records = sorted_records(private_contacts)
        private_pipeline._use_soft_surface_compaction = False
        private_pipeline.collide(state, private_contacts)
        assert_record_tuples_equal(compact_records, sorted_records(private_contacts))
        self.assertEqual(int(np.count_nonzero(private_pipeline._soft_face_cache_state.numpy() == 2)), 0)
        private_pipeline._use_soft_surface_compaction = True

        particle_q = state.particle_q.numpy()
        particle_q[:, 2] += 1.0e-4
        state.particle_q.assign(particle_q)
        shared_pipeline.collide(state, shared_contacts)
        private_pipeline.collide(state, private_contacts)
        assert_records_equal()

        model.shape_margin.zero_()
        shared_pipeline.collide(state, shared_contacts, soft_contact_margin=0.1)
        private_pipeline.collide(state, private_contacts, soft_contact_margin=0.1)
        assert_records_equal()

        model.shape_margin.assign(np.array([0.1, 0.0], dtype=np.float32))
        private_pipeline.collide(state, private_contacts)

        expected_count = int(private_contacts.soft_contact_count.numpy()[0])
        with wp.ScopedCapture(device=device) as capture:
            private_pipeline.collide(state, private_contacts)
        for _ in range(3):
            wp.capture_launch(capture.graph)
            self.assertEqual(int(private_contacts.soft_contact_count.numpy()[0]), expected_count)

        shape_flags = model.shape_flags.numpy()
        shape_flags[0] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        model.shape_flags.assign(shape_flags)
        wp.capture_launch(capture.graph)
        self.assertEqual(int(private_contacts.soft_contact_count.numpy()[0]), 0)
        np.testing.assert_array_equal(private_pipeline._soft_surface_compact_counts.numpy(), [0, 0])

        shape_flags[0] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
        model.shape_flags.assign(shape_flags)
        wp.capture_launch(capture.graph)
        self.assertEqual(int(private_contacts.soft_contact_count.numpy()[0]), expected_count)
        self.assertTrue(private_contacts._enable_rigid_soft_full_surface_contact)

    @unittest.skipUnless(wp.is_cuda_available(), "Temporal face cache requires CUDA texture SDFs")
    def test_private_full_surface_face_cache_preserves_contact_keys(self):
        """Reuse consecutive mesh-SDF face contacts without changing emitted keys."""
        device = wp.get_device("cuda:0")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        shape_cfg = newton.ModelBuilder.ShapeConfig(margin=0.02)
        shape_cfg.configure_sdf(force_sdf=True)
        shape = builder.add_shape_mesh(
            body=-1,
            mesh=newton.Mesh.create_box(0.5, 0.5, 0.5),
            cfg=shape_cfg,
        )
        builder.shape_sdf_max_resolution[shape] = 16
        builder.add_cloth_grid(
            pos=wp.vec3(-0.4, -0.4, 0.45),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=4,
            dim_y=4,
            cell_x=0.2,
            cell_y=0.2,
            mass=0.1,
            particle_radius=0.01,
        )
        builder.color()
        model = builder.finalize(device=device)
        options = {
            "broad_phase": "nxn",
            "soft_contact_margin": 0.1,
            "enable_rigid_soft_full_surface_contact": True,
        }
        shared_pipeline = newton.CollisionPipeline(model, **options)
        private_pipeline = MJVBDV2CollisionPipeline(model, **options)
        shared_contacts = shared_pipeline.contacts()
        private_contacts = private_pipeline.contacts()
        state = model.state()

        def contact_keys(contacts):
            count = min(int(contacts.soft_contact_count.numpy()[0]), contacts.soft_contact_max)
            particles = contacts.soft_contact_particle.numpy()[:count]
            shapes = contacts.soft_contact_shape.numpy()[:count]
            indices = contacts.soft_contact_indices.numpy()[:count]
            return sorted(
                (int(shape_id), int(particle), *(int(index) for index in corner_ids))
                for particle, shape_id, corner_ids in zip(particles, shapes, indices, strict=True)
            )

        private_pipeline.collide(state, private_contacts)
        private_pipeline.collide(state, private_contacts)
        self.assertGreater(int(np.count_nonzero(private_pipeline._soft_face_cache_state.numpy() > 1)), 0)
        shared_pipeline.collide(state, shared_contacts)
        self.assertEqual(contact_keys(private_contacts), contact_keys(shared_contacts))

        particle_q = state.particle_q.numpy()
        particle_q[:, 2] += 1.0e-4
        state.particle_q.assign(particle_q)
        private_pipeline.collide(state, private_contacts)
        shared_pipeline.collide(state, shared_contacts)
        self.assertEqual(contact_keys(private_contacts), contact_keys(shared_contacts))

        particle_q[:, 2] += 2.0
        state.particle_q.assign(particle_q)
        private_pipeline.collide(state, private_contacts)
        self.assertEqual(int(np.count_nonzero(private_pipeline._soft_face_cache_state.numpy())), 0)

    def test_solver_contact_preallocation_excludes_cross_world_pairs(self):
        """Preallocate contact state from world-compatible pairs instead of a Cartesian product."""
        world = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        world.add_shape_sphere(body=-1, radius=0.1)
        world.add_particle(wp.vec3(0.0, 0.0, 0.2), wp.vec3(), 0.01, radius=0.01)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_world(world)
        builder.add_world(world)
        builder.color()
        model = builder.finalize(device="cpu")

        expected_capacity = collision_pipeline._count_world_compatible_particle_shape_pairs(model)
        self.assertEqual(expected_capacity, 2)
        self.assertEqual(collision_pipeline._build_particle_shape_pairs(model).shape[0], expected_capacity)
        self.assertEqual(soft_contact_pipeline._build_particle_shape_pairs(model).shape[0], expected_capacity)
        self.assertEqual(model.particle_count * model.shape_count, 4)

        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__):
                solver = solver_type(model, iterations=1, particle_enable_tile_solve=False)
                self.assertEqual(solver.body_particle_contact_penalty_k.shape[0], expected_capacity)

    def test_particle_contact_gather_matches_legacy_scatter(self):
        """Match legacy point-contact forces and Hessians for active-prefix boundaries."""
        devices = [wp.get_device("cpu")]
        if wp.is_cuda_available():
            devices.append(wp.get_device("cuda:0"))
        for device in devices:
            for variant, kernels, uses_color_masks in _PARTICLE_KERNEL_VARIANTS:
                data = _make_point_contact_data(device)
                for raw_count in (0, 1, 4, data["capacity"], data["capacity"] + 2):
                    with self.subTest(device=device, variant=variant, raw_count=raw_count):
                        contact_count = wp.array([raw_count], dtype=int, device=device)
                        legacy_forces = wp.array(data["base_forces"], dtype=wp.vec3, device=device)
                        legacy_hessians = wp.array(data["base_hessians"], dtype=wp.mat33, device=device)
                        gather_forces = wp.array(data["base_forces"], dtype=wp.vec3, device=device)
                        gather_hessians = wp.array(data["base_hessians"], dtype=wp.mat33, device=device)

                        _launch_legacy_contacts(
                            kernels,
                            uses_color_masks,
                            data,
                            contact_count,
                            legacy_forces,
                            legacy_hessians,
                            device,
                        )
                        _launch_gather_contacts(
                            kernels,
                            data,
                            contact_count,
                            gather_forces,
                            gather_hessians,
                            device,
                        )

                        np.testing.assert_allclose(gather_forces.numpy(), legacy_forces.numpy(), rtol=1.0e-6)
                        np.testing.assert_allclose(gather_hessians.numpy(), legacy_hessians.numpy(), rtol=1.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "Persistent contact traversal requires CUDA")
    def test_persistent_particle_contact_force_matches_legacy_scatter(self):
        """Match legacy scatter over empty, partial, full, and overflow-clamped prefixes."""
        device = wp.get_device("cuda:0")
        data = _make_point_contact_data(device)
        for raw_count in (0, 1, 4, data["capacity"], data["capacity"] + 2):
            with self.subTest(raw_count=raw_count):
                contact_count = wp.array([raw_count], dtype=int, device=device)
                legacy_forces = wp.array(data["base_forces"], dtype=wp.vec3, device=device)
                legacy_hessians = wp.array(data["base_hessians"], dtype=wp.mat33, device=device)
                active_forces = wp.array(data["base_forces"], dtype=wp.vec3, device=device)
                active_hessians = wp.array(data["base_hessians"], dtype=wp.mat33, device=device)

                _launch_legacy_contacts(
                    soft_particle_kernels,
                    False,
                    data,
                    contact_count,
                    legacy_forces,
                    legacy_hessians,
                    device,
                )
                _launch_active_soft_contacts(
                    data,
                    contact_count,
                    active_forces,
                    active_hessians,
                    device,
                )

                np.testing.assert_allclose(active_forces.numpy(), legacy_forces.numpy(), rtol=1.0e-6)
                np.testing.assert_allclose(active_hessians.numpy(), legacy_hessians.numpy(), rtol=1.0e-6)

    def test_body_particle_dual_updates_active_prefix_once(self):
        """Update every clamped active-prefix contact once and preserve the inactive tail."""
        devices = [wp.get_device("cpu")]
        if wp.is_cuda_available():
            devices.append(wp.get_device("cuda:0"))
        beta = 2.5
        initial_penalty = 1.0
        worker_count = 3
        for device in devices:
            for variant, kernels in _RIGID_KERNEL_VARIANTS:
                data = _make_dual_data(device)
                for raw_count in (0, 1, worker_count + 1, data["capacity"], data["capacity"] + 2):
                    with self.subTest(device=device, variant=variant, raw_count=raw_count):
                        contact_count = wp.array([raw_count], dtype=int, device=device)
                        penalty_k = wp.full(data["capacity"], initial_penalty, dtype=float, device=device)
                        wp.launch(
                            kernels.update_duals_body_particle_contacts,
                            dim=worker_count,
                            inputs=[
                                contact_count,
                                data["capacity"],
                                worker_count,
                                data["indices"],
                                data["shape"],
                                data["body_pos"],
                                data["normal"],
                                data["barycentric"],
                                data["particle_q"],
                                data["particle_radius"],
                                data["shape_body"],
                                data["shape_margin"],
                                data["body_q"],
                                data["material_ke"],
                                beta,
                                penalty_k,
                            ],
                            device=device,
                        )

                        expected = np.full(data["capacity"], initial_penalty, dtype=np.float32)
                        active_count = min(raw_count, data["capacity"])
                        expected[:active_count] += beta * (
                            0.1 + 0.01 * np.arange(1, active_count + 1, dtype=np.float32)
                        )
                        np.testing.assert_allclose(penalty_k.numpy(), expected, rtol=1.0e-6, atol=1.0e-6)

    def test_point_contact_aabb_rejection_matches_reference(self):
        """Preserve point contacts while rejecting spatially remote shape pairs."""
        devices = [wp.get_device("cpu")]
        if wp.is_cuda_available():
            devices.append(wp.get_device("cuda:0"))

        for device in devices:
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            moving_body = builder.add_body(
                xform=wp.transform(wp.vec3(3.0, 0.0, 0.0), wp.quat_identity()),
                is_kinematic=True,
            )
            shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, margin=0.02)
            builder.add_shape_box(body=-1, hx=0.2, hy=0.2, hz=0.2, cfg=shape_cfg)
            moving_shape = builder.add_shape_sphere(body=moving_body, radius=0.15, cfg=shape_cfg)
            mesh = newton.Mesh.create_box(0.2, 0.2, 0.2, compute_inertia=False)
            offset_mesh = newton.Mesh(
                mesh.vertices + np.array((1.0, 0.0, 0.0), dtype=np.float32),
                mesh.indices,
                compute_inertia=False,
            )
            builder.add_shape_mesh(
                body=-1,
                xform=wp.transform(wp.vec3(-1.0, 0.6, 0.0), wp.quat_identity()),
                mesh=offset_mesh,
                cfg=shape_cfg,
            )
            remote_shape = builder.add_shape_mesh(
                body=-1,
                xform=wp.transform(wp.vec3(100.0, 0.0, 0.0), wp.quat_identity()),
                mesh=mesh,
                cfg=shape_cfg,
            )
            for position in ((0.22, 0.0, 0.0), (0.6, 0.0, 0.0), (0.19, 0.6, 0.0)):
                builder.add_particle(wp.vec3(*position), wp.vec3(), 0.01, radius=0.05)
            builder.color()
            model = builder.finalize(device=device)
            state = model.state()
            margin = 0.01

            pipeline_factories = (
                (
                    collision_pipeline.MJVBDV2SoftContactPipeline,
                    lambda pipeline, contacts, state=state: pipeline.collide(state, contacts),
                    lambda pipeline: pipeline.contacts(),
                ),
                (
                    soft_contact_pipeline.MJVBDSoftContactPipeline,
                    lambda pipeline, contacts, state=state: pipeline.generate(state, state, contacts),
                    lambda pipeline: pipeline.make_contacts(),
                ),
            )
            for pipeline_type, generate, make_contacts in pipeline_factories:
                with self.subTest(device=device, pipeline=pipeline_type.__name__):
                    body_q = state.body_q.numpy()
                    body_q[moving_body, :3] = (3.0, 0.0, 0.0)
                    state.body_q.assign(body_q)
                    pipeline = pipeline_type(model, margin=margin)
                    actual_contacts = make_contacts(pipeline)
                    reference_contacts = make_contacts(pipeline)

                    generate(pipeline, actual_contacts)
                    _launch_reference_soft_contacts(
                        model,
                        pipeline.pairs,
                        state,
                        state,
                        reference_contacts,
                        margin,
                    )
                    _assert_soft_contact_records_equal(
                        self,
                        _sorted_soft_contact_records(reference_contacts),
                        _sorted_soft_contact_records(actual_contacts),
                    )
                    initial_count = int(reference_contacts.soft_contact_count.numpy()[0])
                    self.assertGreater(initial_count, 0)
                    self.assertGreater(pipeline.shape_aabb_lower.numpy()[remote_shape, 0], 99.0)
                    self.assertGreater(pipeline.shape_aabb_lower.numpy()[moving_shape, 0], 2.0)

                    body_q = state.body_q.numpy()
                    body_q[moving_body, :3] = (0.6, 0.0, 0.0)
                    state.body_q.assign(body_q)
                    generate(pipeline, actual_contacts)
                    _launch_reference_soft_contacts(
                        model,
                        pipeline.pairs,
                        state,
                        state,
                        reference_contacts,
                        margin,
                    )
                    _assert_soft_contact_records_equal(
                        self,
                        _sorted_soft_contact_records(reference_contacts),
                        _sorted_soft_contact_records(actual_contacts),
                    )
                    moved_count = int(reference_contacts.soft_contact_count.numpy()[0])
                    self.assertGreater(moved_count, initial_count)

                    flags = model.shape_flags.numpy()
                    flags[moving_shape] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
                    model.shape_flags.assign(flags)
                    generate(pipeline, actual_contacts)
                    _launch_reference_soft_contacts(
                        model,
                        pipeline.pairs,
                        state,
                        state,
                        reference_contacts,
                        margin,
                    )
                    _assert_soft_contact_records_equal(
                        self,
                        _sorted_soft_contact_records(reference_contacts),
                        _sorted_soft_contact_records(actual_contacts),
                    )
                    self.assertLess(int(reference_contacts.soft_contact_count.numpy()[0]), moved_count)

                    flags[moving_shape] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
                    model.shape_flags.assign(flags)

                    if device.is_cuda:
                        with wp.ScopedCapture(device=device) as capture:
                            generate(pipeline, actual_contacts)
                        shape_margin = model.shape_margin.numpy()
                        shape_margin += 0.01
                        model.shape_margin.assign(shape_margin)
                        wp.capture_launch(capture.graph)
                        _launch_reference_soft_contacts(
                            model,
                            pipeline.pairs,
                            state,
                            state,
                            reference_contacts,
                            margin,
                        )
                        _assert_soft_contact_records_equal(
                            self,
                            _sorted_soft_contact_records(reference_contacts),
                            _sorted_soft_contact_records(actual_contacts),
                        )

    @unittest.skipUnless(wp.is_cuda_available(), "Shape-major contact candidates require CUDA")
    def test_cuda_sparse_contact_candidates_are_shape_major(self):
        """Group both private sparse candidate arrays by rigid shape on CUDA."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
        for x in (-0.3, 0.0, 0.3):
            builder.add_shape_sphere(
                body=-1,
                xform=wp.transform(wp.vec3(x, 0.0, 0.0), wp.quat_identity()),
                radius=0.1,
                cfg=shape_cfg,
            )
        for x in (-0.2, -0.05, 0.1, 0.25):
            builder.add_particle(wp.vec3(x, 0.0, 0.0), wp.vec3(), 0.01, radius=0.01)
        builder.color()
        model = builder.finalize(device="cuda:0")

        pair_builders = (
            collision_pipeline._build_particle_shape_pairs,
            soft_contact_pipeline._build_particle_shape_pairs,
        )
        expected = {(particle, shape) for particle in range(4) for shape in range(3)}
        for build_pairs in pair_builders:
            pairs = build_pairs(model).numpy()
            with self.subTest(module=build_pairs.__module__):
                self.assertEqual({tuple(int(value) for value in pair) for pair in pairs}, expected)
                self.assertTrue(np.all(pairs[:-1, 1] <= pairs[1:, 1]))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA batch threshold requires CUDA")
    def test_small_color_groups_keep_legacy_contact_path(self):
        """Keep single-scene color groups on the existing contact and dual paths."""
        device = wp.get_device("cuda:0")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_shape_sphere(body=-1, radius=0.1)
        for particle in range(8):
            builder.add_particle(wp.vec3(0.02 * particle, 0.0, 0.2), wp.vec3(), 0.01, radius=0.01)
        builder.color()
        model = builder.finalize(device=device)
        solver = SolverVBDSoft(model, iterations=1)

        self.assertFalse(solver._particle_contact_gather_supported)
        self.assertFalse(solver._use_active_soft_contact_prefix)
        self.assertEqual(solver._active_soft_contact_worker_dim(123), 123)

    @unittest.skipUnless(wp.is_cuda_available(), "Persistent contact traversal requires CUDA")
    def test_persistent_particle_contact_force_dispatch_gates(self):
        """Restrict persistent traversal to large nondeterministic, non-gradient CUDA Graphs."""
        device = wp.get_device("cuda:0")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_shape_sphere(body=-1, radius=0.1)
        for particle in range(8):
            builder.add_particle(wp.vec3(0.02 * particle, 0.0, 0.2), wp.vec3(), 0.01, radius=0.01)
        builder.color()
        model = builder.finalize(device=device)
        solver = SolverVBDSoft(model, iterations=1)

        self.assertTrue(solver._persistent_particle_contact_force_supported)
        self.assertFalse(solver._should_use_persistent_particle_contact_force(289620))

        deterministic_solver = SolverVBDSoft(
            model,
            iterations=1,
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        self.assertFalse(deterministic_solver._persistent_particle_contact_force_supported)

        grad_builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        grad_builder.add_shape_sphere(body=-1, radius=0.1)
        grad_builder.add_particle(wp.vec3(0.0, 0.0, 0.2), wp.vec3(), 0.01, radius=0.01)
        grad_builder.color()
        grad_model = grad_builder.finalize(device=device, requires_grad=True)
        grad_solver = SolverVBDSoft(grad_model, iterations=1)
        self.assertFalse(grad_solver._persistent_particle_contact_force_supported)

        with wp.ScopedCapture(device=device) as capture:
            self.assertFalse(solver._should_use_persistent_particle_contact_force(32767))
            self.assertTrue(solver._should_use_persistent_particle_contact_force(32768))
            solver.particle_forces.zero_()
        wp.capture_launch(capture.graph)

    @unittest.skipUnless(wp.is_cuda_available(), "Particle-contact gather requires CUDA")
    def test_cuda_solvers_execute_forced_point_contact_gather(self):
        """Execute the gather wiring in both private VBD solvers on point contacts."""
        device = wp.get_device("cuda:0")
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_shape_sphere(body=-1, radius=0.1)
        for position in ((-0.02, 0.0, 0.105), (0.02, 0.0, 0.105), (0.0, 0.03, 0.105)):
            builder.add_particle(wp.vec3(*position), wp.vec3(), 0.01, radius=0.01)
        builder.add_triangle(0, 1, 2, tri_ke=1.0e3, tri_ka=1.0e3)
        builder.color()
        model = builder.finalize(device=device)
        pipeline = collision_pipeline.MJVBDV2SoftContactPipeline(model)

        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            with self.subTest(solver=solver_type.__module__):
                contacts = pipeline.contacts()
                state_in = model.state()
                state_out = model.state()
                control = model.control()
                pipeline.collide(state_in, contacts)
                solver = solver_type(model, iterations=1, particle_enable_tile_solve=False)
                solver._particle_contact_gather_supported = True
                solver._use_active_soft_contact_prefix = True
                solver._particle_contact_head = wp.full(model.particle_count, -1, dtype=wp.int32, device=device)
                solver._particle_contact_next = wp.empty(0, dtype=wp.int32, device=device)
                solver._particle_contact_adjacency_initialized = False

                solver.step(state_in, state_out, control, contacts, 1.0 / 60.0)
                with wp.ScopedCapture(device=device) as capture:
                    pipeline.collide(state_out, contacts)
                    solver.step(state_out, state_in, control, contacts, 1.0 / 60.0)
                wp.capture_launch(capture.graph)

                self.assertEqual(solver._particle_contact_next.shape[0], contacts.soft_contact_max)
                self.assertTrue(np.isfinite(state_in.particle_q.numpy()).all())


if __name__ == "__main__":
    unittest.main()

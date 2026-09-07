# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check coarse contact coupling independently of trajectories."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2 import contact_projection as cp
from newton._src.solvers.mjvbd_v2 import particle_multilevel as ml
from newton._src.solvers.mjvbd_v2.vbd_soft import multilevel_contacts as extract
from newton._src.solvers.mjvbd_v2.vbd_soft import particle_vbd_kernels as k


@wp.kernel
def _multiply(
    diagonal: wp.array[wp.mat33],
    vector: wp.array[wp.vec3],
    contacts: cp.ContactProjectionData,
    result: wp.array[wp.vec3],
):
    row = wp.tid()
    result[row] = diagonal[row] * vector[row] + cp.off_diagonal_product(row, vector, contacts)


class TestMJVBDV2ContactProjection(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Persistent coarse PCG requires CUDA")
    def test_invalid_contact_operator_is_rejected_before_solving(self):
        """Reject incomplete contacts before attempting an indefinite coarse solve."""
        device = "cuda:0"
        op = cp.ContactProjection(2, 2, device)
        op.reset()
        op.data.overflow.fill_(16)
        slots = wp.array([0, 1], dtype=wp.int32, device=device)
        vectors = [wp.ones(2, dtype=wp.vec3, device=device) for _ in range(5)]
        status = wp.zeros(1, dtype=wp.int32, device=device)
        metrics = wp.ones(13, dtype=float, device=device)
        wp.launch(
            ml._solve_energy_galerkin_pcg_persistent,
            dim=256,
            block_dim=256,
            inputs=[
                2,
                wp.array([0, 1, 2], dtype=wp.int32, device=device),
                slots,
                slots,
                wp.array([-np.eye(3)] * 2, dtype=wp.mat33, device=device),
                wp.ones(2, dtype=wp.vec3, device=device),
                8,
                True,
                1e-4,
                op.data,
            ],
            outputs=[
                *vectors,
                wp.zeros(2, dtype=wp.mat33, device=device),
                status,
                metrics,
                wp.ones(2, dtype=wp.int32, device=device),
            ],
            device=device,
        )
        self.assertEqual(status.numpy()[0], 32)
        np.testing.assert_array_equal(vectors[0].numpy(), np.zeros((2, 3)))
        np.testing.assert_array_equal(metrics.numpy(), np.zeros(13))

    def test_duplicate_contacts_reduce_to_one_edge_and_replay(self):
        """Combine repeated cluster pairs without losing atomic updates on replay."""
        for device in ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",):
            count = 4096
            op = cp.ContactProjection(2, count, device)
            block = np.array([[2, 1, 0], [1, 3, 0], [0, 0, 4]], dtype=np.float32)
            inputs = [
                wp.array([[0, 1, -1, -1]] * count, dtype=wp.vec4i, device=device),
                wp.array([[1.0, -1.0, 0.0, 0.0]] * count, dtype=wp.vec4, device=device),
                wp.array(np.tile(block, (count, 1, 1)), dtype=wp.mat33, device=device),
                wp.array([0, 1], dtype=wp.int32, device=device),
                op.data,
            ]
            vector = wp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
            diagonal = wp.array(np.tile(count * block, (2, 1, 1)), dtype=wp.mat33, device=device)
            result = wp.zeros(2, dtype=wp.vec3, device=device)

            def run(op=op, inputs=inputs, vector=vector, diagonal=diagonal, result=result, device=device, count=count):
                op.reset()
                wp.launch(cp.project_records, dim=count, inputs=inputs, device=device)
                wp.launch(_multiply, dim=2, inputs=[diagonal, vector, op.data], outputs=[result], device=device)

            run()
            if device == "cuda:0":
                with wp.ScopedCapture(device=device) as capture:
                    run()
                for _ in range(3):
                    wp.capture_launch(capture.graph)
            self.assertEqual(op.data.count.numpy()[0], count)
            self.assertEqual(op.data.edge_count.numpy()[0], 1)
            self.assertEqual(op.data.overflow.numpy()[0], 0)
            np.testing.assert_allclose(result.numpy(), count * np.array([block[:, 0], -block[:, 0]]), atol=1e-5)

    def test_external_edge_face_projection_keeps_contact_stiffness(self):
        """Retain external contact stiffness when soft corners share one cluster."""

        for device in ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",):

            def array(values, dtype=wp.int32, device=device):
                return wp.array(values, dtype=dtype, device=device)

            for face in (False, True):
                with self.subTest(device=device, face=face):
                    pos = array([[0.0, 0.0, 0.008], [1.0, 0.0, 0.008], [0.0, 1.0, 0.008]], wp.vec3)
                    anchor = array(pos.numpy() + np.array([0.001, 0.0, 0.001]), wp.vec3)
                    radius = array([0.01] * 3, float)
                    indices = array([[0, 1, 2 if face else -1]], wp.vec3i)
                    bary = array([[0.25, 0.25, 0.5] if face else [0.25, 0.75, 0.0]], wp.vec3)
                    count = array([1])
                    penalty, damping, friction = array([1000.0], float), array([0.1], float), array([0.4], float)
                    shape_body = array([-1])
                    body_q = wp.empty(0, dtype=wp.transform, device=device)
                    body_qd = wp.empty(0, dtype=wp.spatial_vector, device=device)
                    body_com = wp.empty(0, dtype=wp.vec3, device=device)
                    suffix = [
                        shape_body,
                        body_q,
                        body_q,
                        body_qd,
                        body_com,
                        array([0]),
                        array([[0.0, 0.0, 0.0]], wp.vec3),
                        array([[0.0, 0.0, 0.0]], wp.vec3),
                        array([[0.0, 0.0, 1.0]], wp.vec3),
                        array([0.0], float),
                        bary,
                    ]
                    force = wp.zeros(3, dtype=wp.vec3, device=device)
                    hessian = wp.zeros(3, dtype=wp.mat33, device=device)
                    wp.launch(
                        k.accumulate_particle_body_contact_force_and_hessian,
                        dim=1,
                        inputs=[
                            0.01,
                            -1,
                            anchor,
                            pos,
                            array([0, 0, 0]),
                            0.01,
                            radius,
                            indices,
                            count,
                            1,
                            penalty,
                            penalty,
                            damping,
                            friction,
                            *suffix,
                        ],
                        outputs=[force, hessian],
                        device=device,
                    )
                    op = cp.ContactProjection(1, 1, device)
                    op.reset()
                    wp.launch(
                        extract.project_body_contacts,
                        dim=1,
                        inputs=[
                            0.01,
                            anchor,
                            pos,
                            radius,
                            indices,
                            count,
                            1,
                            penalty,
                            damping,
                            friction,
                            0.01,
                            *suffix,
                            array([0, 0, 0]),
                            op.data,
                        ],
                        device=device,
                    )
                    assembled = hessian.numpy()
                    expected = assembled[0] / (0.25 * 0.25)
                    self.assertGreater(np.linalg.norm(expected), 0.0)
                    np.testing.assert_allclose(
                        assembled.sum(axis=0) + op.data.diagonal_correction.numpy()[0], expected, rtol=2e-6, atol=2e-3
                    )
                    self.assertEqual(op.data.count.numpy()[0], 1)

    def test_self_contact_extraction_matches_fine_assembly(self):
        """Extract EE/VT blocks with the same normal, damping and friction as VBD."""

        for device in ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",):

            def array(values, dtype=wp.int32, device=device):
                return wp.array(values, dtype=dtype, device=device)

            for edge in (False, True):
                with self.subTest(device=device, edge=edge):
                    q = np.array([[-1, -1, 0], [1, -1, 0], [0, 2, 0], [0, 0, 0.008]], dtype=np.float32)
                    if edge:
                        q = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0.008], [0, 1, 0.008]], dtype=np.float32)
                    prev = q.copy()
                    prev[0] += [0.001, 0.001, 0.001]
                    pos, anchor = array(q, wp.vec3), array(prev, wp.vec3)
                    info = k.TriMeshCollisionInfo()
                    info.vertex_colliding_triangles_count = array([0, 0, 0, int(not edge)])
                    info.vertex_colliding_triangles_buffer_sizes = array([0, 0, 0, 1])
                    info.vertex_colliding_triangles_offsets = array([0, 0, 0, 0, 1])
                    info.vertex_colliding_triangles = array([3, 0])
                    info.edge_colliding_edges_count = array([int(edge), int(edge)])
                    info.edge_colliding_edges_buffer_sizes = array([1, 1])
                    info.edge_colliding_edges_offsets = array([0, 1, 2])
                    info.edge_colliding_edges = array([0, 1, 1, 0])
                    triangles = array([[0, 1, 2]])
                    edges = array([[0, 0, 0, 1], [0, 0, 2, 3]])
                    collision = array([info], k.TriMeshCollisionInfo)
                    material = wp.empty(0, dtype=wp.vec3, device=device)
                    material_ids = array([])
                    active = array([1])
                    suffix = [
                        triangles,
                        edges,
                        collision,
                        0.01,
                        1000.0,
                        0.1,
                        0.4,
                        material,
                        material_ids,
                        False,
                        0.01,
                        1e-6,
                        active,
                    ]
                    forces = wp.zeros(4, dtype=wp.vec3, device=device)
                    hessians = wp.zeros(4, dtype=wp.mat33, device=device)
                    wp.launch(
                        k.accumulate_self_contact_force_and_hessian,
                        dim=16,
                        inputs=[0.01, -1, anchor, pos, array([0, 0, 0, 0]), *suffix],
                        outputs=[forces, hessians],
                        device=device,
                    )
                    op = cp.ContactProjection(1, 4, device)
                    op.reset()
                    wp.launch(
                        extract.project_self_contacts,
                        dim=4,
                        inputs=[0.01, anchor, pos, *suffix, array([0, 0, 0, 0]), op.data],
                        device=device,
                    )
                    fine_sum = hessians.numpy().sum(axis=0)
                    self.assertGreater(np.linalg.norm(fine_sum), 0.0)
                    np.testing.assert_allclose(op.data.diagonal_correction.numpy()[0], -fine_sum, rtol=2e-5, atol=2e-3)
                    self.assertEqual(op.data.count.numpy()[0], 1)
                    self.assertEqual(op.data.overflow.numpy()[0], 0)
                    if edge:
                        info.edge_colliding_edges_count.assign(np.array([1, 0], dtype=np.int32))
                        op.reset()
                        wp.launch(
                            extract.project_self_contacts,
                            dim=4,
                            inputs=[0.01, anchor, pos, *suffix, array([0, 0, 0, 0]), op.data],
                            device=device,
                        )
                        self.assertNotEqual(op.data.overflow.numpy()[0], 0)

    def test_projected_operator_matches_dense_and_handles_pins(self):
        """Match P-transpose H P for VT, EE and external edge/face stencils."""
        for device in ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",):
            with self.subTest(device=device):
                mapping = np.array([0, 0, 1, 2, -1], dtype=np.int32)
                vertices = np.array([[0, 1, 2, 3], [0, 2, 3, 4], [1, 2, 4, -1], [0, 2, -1, -1]], dtype=np.int32)
                weights = np.array(
                    [[-0.2, -0.3, -0.5, 1], [0.4, 0.6, -0.75, -0.25], [0.2, 0.3, 0.5, 0], [0.4, 0.6, 0, 0]],
                    dtype=np.float32,
                )
                hessians = np.array([np.diag([2, 3, 100]) * i for i in range(1, 5)], dtype=np.float32)
                legacy = np.tile(np.eye(3, dtype=np.float32), (3, 1, 1))
                expected = np.eye(9)
                for ids, b, h in zip(vertices, weights, hessians, strict=True):
                    projected = np.zeros(3)
                    for particle, weight in zip(ids, b, strict=True):
                        if particle >= 0 and mapping[particle] >= 0:
                            row = mapping[particle]
                            projected[row] += weight
                            legacy[row] += weight * weight * h
                    expected += np.kron(np.outer(projected, projected), h)
                op = cp.ContactProjection(3, 4, device)
                op.reset()
                wp.launch(
                    cp.project_records,
                    dim=4,
                    inputs=[
                        wp.array(vertices, dtype=wp.vec4i, device=device),
                        wp.array(weights, dtype=wp.vec4, device=device),
                        wp.array(hessians, dtype=wp.mat33, device=device),
                        wp.array(mapping, dtype=wp.int32, device=device),
                        op.data,
                    ],
                    device=device,
                )
                blocks = wp.array(legacy, dtype=wp.mat33, device=device)
                slots = wp.array([0, 1, 2], dtype=wp.int32, device=device)
                op.correct_diagonal(blocks, slots)
                for column in range(9):
                    vector = np.eye(9, dtype=np.float32)[:, column].reshape(3, 3)
                    product = wp.zeros(3, dtype=wp.vec3, device=device)
                    wp.launch(
                        _multiply,
                        dim=3,
                        inputs=[blocks, wp.array(vector, dtype=wp.vec3, device=device), op.data],
                        outputs=[product],
                        device=device,
                    )
                    np.testing.assert_allclose(product.numpy().ravel(), expected[:, column], atol=3e-5, rtol=2e-6)
                self.assertEqual(op.data.overflow.numpy()[0], 0)
                if device == "cuda:0":
                    rhs = np.arange(9, dtype=np.float32).reshape(3, 3) * 0.1
                    vectors = [wp.zeros(3, dtype=wp.vec3, device=device) for _ in range(5)]
                    status = wp.zeros(1, dtype=wp.int32, device=device)
                    wp.launch(
                        ml._solve_energy_galerkin_pcg_persistent,
                        dim=256,
                        block_dim=256,
                        inputs=[
                            3,
                            wp.array([0, 1, 2, 3], dtype=wp.int32, device=device),
                            slots,
                            slots,
                            blocks,
                            wp.array(rhs, dtype=wp.vec3, device=device),
                            24,
                            True,
                            0.001,
                            op.data,
                        ],
                        outputs=[
                            *vectors,
                            wp.zeros(3, dtype=wp.mat33, device=device),
                            status,
                            wp.zeros(29, dtype=float, device=device),
                            wp.zeros(2, dtype=wp.int32, device=device),
                        ],
                        device=device,
                    )
                    self.assertEqual(status.numpy()[0], 0)
                    np.testing.assert_allclose(
                        vectors[0].numpy().ravel(), np.linalg.solve(expected, rhs.ravel()), atol=2e-6, rtol=2e-5
                    )

    def test_overflow_rejects_and_reset_clears_contacts(self):
        """Reject incomplete operators without following uninitialized entries."""
        for device in ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",):
            op = cp.ContactProjection(1, 1, device)
            op.reset()
            wp.launch(
                cp.project_records,
                dim=2,
                inputs=[
                    wp.array([[0, -1, -1, -1]] * 2, dtype=wp.vec4i, device=device),
                    wp.array([[1, 0, 0, 0]] * 2, dtype=wp.vec4, device=device),
                    wp.array([np.eye(3)] * 2, dtype=wp.mat33, device=device),
                    wp.zeros(1, dtype=wp.int32, device=device),
                    op.data,
                ],
                device=device,
            )
            status = wp.zeros(1, dtype=wp.int32, device=device)
            wp.launch(cp.reject_overflow, dim=1, inputs=[op.data], outputs=[status], device=device)
            self.assertNotEqual(status.numpy()[0], 0)
            op.reset()
            np.testing.assert_array_equal(op.data.row_heads.numpy(), [-1])
            np.testing.assert_array_equal(op.data.diagonal_correction.numpy(), np.zeros((1, 3, 3)))
            self.assertEqual(op.data.count.numpy()[0], 0)

    def test_same_cluster_translation_has_zero_self_contact_stiffness(self):
        """Cancel self-contact diagonal blocks under a cluster translation."""
        for device in ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",):
            with self.subTest(device=device):
                weights = np.array([-0.25, -0.25, -0.5, 1.0], dtype=np.float32)
                hessian = np.diag([2.0, 3.0, 100.0]).astype(np.float32)
                blocks = wp.zeros(1, dtype=wp.mat33, device=device)
                wp.launch(
                    ml._restrict_energy_galerkin,
                    dim=1,
                    inputs=[
                        wp.array([0, 4], dtype=wp.int32, device=device),
                        wp.array([0, 1, 2, 3], dtype=wp.int32, device=device),
                        wp.zeros(4, dtype=wp.vec3, device=device),
                        wp.array(weights[:, None, None] ** 2 * hessian, dtype=wp.mat33, device=device),
                        wp.array([0], dtype=wp.int32, device=device),
                    ],
                    outputs=[wp.zeros(1, dtype=wp.vec3, device=device), blocks],
                    device=device,
                )
                operator = cp.ContactProjection(1, 1, device)
                operator.reset()
                wp.launch(
                    cp.project_records,
                    dim=1,
                    inputs=[
                        wp.array([[0, 1, 2, 3]], dtype=wp.vec4i, device=device),
                        wp.array([weights], dtype=wp.vec4, device=device),
                        wp.array([hessian], dtype=wp.mat33, device=device),
                        wp.zeros(4, dtype=wp.int32, device=device),
                        operator.data,
                    ],
                    device=device,
                )
                operator.correct_diagonal(blocks, wp.array([0], dtype=wp.int32, device=device))
                np.testing.assert_allclose(blocks.numpy(), 0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

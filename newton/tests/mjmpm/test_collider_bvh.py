# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check rigid collider queries across BVH construction strategies."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjmpm.mpm.implicit_mpm_model import _create_body_collider_mesh
from newton._src.solvers.mjmpm.mpm.rasterized_collisions import MJMPMCollider, _query_collider_sdf


@wp.kernel
def _query(
    collider: MJMPMCollider,
    body_q: wp.array[wp.transform],
    local_points: wp.array[wp.vec3],
    hit: wp.array[int],
    distance: wp.array[float],
    gradient: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    closest: wp.array[wp.vec3],
    material: wp.array[int],
):
    i = wp.tid()
    point = wp.transform_point(body_q[0], local_points[i])
    found, d, g, v, cp, mat = _query_collider_sdf(point, collider, body_q, 0)
    hit[i] = int(found)
    distance[i] = d
    gradient[i] = g
    velocity[i] = v
    closest[i] = cp
    material[i] = mat


class TestColliderBVH(unittest.TestCase):
    def test_multiscale_rigid_queries_match_reference(self):
        """Retain signed distances, normals and per-face materials under rigid motion and graph replay."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                body = builder.add_body()
                builder.add_shape_box(body, hx=100.0, hy=100.0, hz=0.01)
                builder.add_shape_sphere(
                    body, xform=wp.transform(wp.vec3(0.15, 0.0, 0.12), wp.quat_identity()), radius=0.04
                )
                model = builder.finalize(device=device)
                mesh, materials = _create_body_collider_mesh(model, np.array([0, 1]), np.array([0, 1]))
                reference = wp.Mesh(
                    mesh.points,
                    mesh.indices,
                    mesh.velocities,
                    bvh_constructor="lbvh" if model.device.is_cuda else "sah",
                )
                rng = np.random.default_rng(21)
                directions = rng.normal(size=(48, 3))
                directions /= np.linalg.norm(directions, axis=1)[:, None]
                sphere_points = np.concatenate(
                    [np.array([0.15, 0.0, 0.12]) + scale * 0.04 * directions for scale in (0.3, 0.99, 1.0, 1.01, 1.8)]
                )
                points = np.concatenate(
                    [sphere_points, [[0, 0, 0], [0, 0, 0.01], [0, 0, 0.0105], [0.19, 0, 0.12], [500, 500, 500]]]
                )
                local_points = wp.array(points, dtype=wp.vec3)
                body_q = wp.array([wp.transform_identity()], dtype=wp.transform)
                commands, results, colliders = [], [], []
                for geometry in (reference, mesh):
                    collider = MJMPMCollider()
                    collider.collider_mesh = wp.array([geometry.id], dtype=wp.uint64)
                    collider.collider_body_index = wp.array([0], dtype=int)
                    collider.collider_max_thickness = wp.array([0.0004], dtype=float)
                    collider.collider_face_offset = wp.array([0], dtype=int)
                    collider.face_material_index = wp.array(materials, dtype=int)
                    collider.material_thickness = wp.array([0.0, 0.0004], dtype=float)
                    collider.query_max_dist = 0.3
                    colliders.append(collider)
                    output = [
                        wp.zeros(len(points), dtype=dtype) for dtype in (int, float, wp.vec3, wp.vec3, wp.vec3, int)
                    ]
                    results.append(output)
                    commands.append(
                        wp.launch(
                            _query,
                            dim=len(points),
                            inputs=[collider, body_q, local_points],
                            outputs=output,
                            record_cmd=True,
                        )
                    )
                for pose in (
                    wp.transform_identity(),
                    wp.transform(wp.vec3(0.2, -0.3, 0.7), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.6)),
                ):
                    body_q.assign(wp.array([pose], dtype=wp.transform))
                    for command in commands:
                        command.launch()
                    if model.device.is_cuda:
                        with wp.ScopedCapture() as capture:
                            for command in commands:
                                command.launch()
                        wp.capture_launch(capture.graph)
                    expected, actual = ([array.numpy() for array in result] for result in results)
                    np.testing.assert_array_equal(actual[0], expected[0])
                    active = expected[0].astype(bool)
                    self.assertTrue(active.any())
                    self.assertFalse(active[-1])
                    np.testing.assert_array_equal(actual[-1][active], expected[-1][active])
                    for field, (got, want) in enumerate(zip(actual[1:-1], expected[1:-1], strict=True)):
                        # Normalizing a short offset amplifies closest-point
                        # roundoff on the 200 m box. Keep distance/position
                        # tolerances separate from the dimensionless normal.
                        atol = 5.0e-4 if field == 1 else 2.0e-5
                        np.testing.assert_allclose(got[active], want[active], rtol=2.0e-5, atol=atol)


if __name__ == "__main__":
    unittest.main()

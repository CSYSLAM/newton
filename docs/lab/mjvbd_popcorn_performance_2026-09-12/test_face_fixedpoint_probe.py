# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the exact fixed-point exit preserves original search results."""

import unittest

import numpy as np
import warp as wp
from face_fixedpoint_probe import optimize_face_fixedpoint

import newton
from newton import GeoType
from newton._src.geometry.sdf_texture import TextureSDFData
from newton._src.geometry.soft_contacts_sdf import optimize_face_sdf
from newton.examples.mjvbdv2.example_mjvbd_v2_popcorn import popcorn_mesh


@wp.kernel(enable_backward=False)
def compare_search(
    points: wp.array[wp.vec3],
    table: wp.array[TextureSDFData],
    sdf_index: int,
    scale: wp.vec3,
    reference: wp.array2d[float],
    candidate: wp.array2d[float],
):
    i = wp.tid()
    geo = int(GeoType.BOX) if i % 2 == 0 else int(GeoType.SPHERE)
    if sdf_index >= 0:
        geo = int(GeoType.CONVEX_MESH)
    a, b, c = points[3 * i], points[3 * i + 1], points[3 * i + 2]
    rb, rx, rp, rg = optimize_face_sdf(geo, scale, a, b, c, sdf_index, table, 24, 16)
    cb, cx, cp, cg = optimize_face_fixedpoint(geo, scale, a, b, c, sdf_index, table, 24, 16)
    for axis in range(3):
        reference[i, axis] = rb[axis]
        candidate[i, axis] = cb[axis]
        reference[i, 3 + axis] = rx[axis]
        candidate[i, 3 + axis] = cx[axis]
        reference[i, 7 + axis] = rg[axis]
        candidate[i, 7 + axis] = cg[axis]
    reference[i, 6] = rp
    candidate[i, 6] = cp


class TestFixedPoint(unittest.TestCase):
    def test_popcorn_texture(self):
        """Match the actual popcorn convex SDF, including mirrored nonuniform scale."""
        builder = newton.ModelBuilder()
        config = newton.ModelBuilder.ShapeConfig()
        config.configure_sdf(force_sdf=True)
        builder.add_shape_convex_hull(body=-1, mesh=popcorn_mesh(), cfg=config)
        model = builder.finalize(device="cuda:0")
        index = int(model._shape_sdf_index.numpy()[0])
        self.assertGreaterEqual(index, 0)
        points = wp.array(np.random.default_rng(72).normal(0, 0.008, (768, 3)), dtype=wp.vec3, device=model.device)
        reference = wp.empty((256, 10), dtype=float, device=model.device)
        candidate = wp.empty_like(reference)
        for scale in (wp.vec3(1.0), wp.vec3(-2.0, 0.75, 1.5)):
            wp.launch(
                compare_search,
                dim=256,
                inputs=[points, model._texture_sdf_data, index, scale, reference, candidate],
                device=model.device,
            )
            np.testing.assert_array_equal(reference.numpy(), candidate.numpy())

    def test_search(self):
        """Match all barycentrics, positions, distances and normals exactly."""
        device = "cuda:0"
        points = wp.array(np.random.default_rng(91).normal(0, 0.2, (768, 3)), dtype=wp.vec3, device=device)
        table = wp.empty(0, dtype=TextureSDFData, device=device)
        reference = wp.empty((256, 10), dtype=float, device=device)
        candidate = wp.empty_like(reference)
        wp.launch(
            compare_search, dim=256, inputs=[points, table, -1, wp.vec3(0.1), reference, candidate], device=device
        )
        np.testing.assert_array_equal(reference.numpy(), candidate.numpy())
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(
                compare_search, dim=256, inputs=[points, table, -1, wp.vec3(0.1), reference, candidate], device=device
            )
        for _replay in range(2):
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(reference.numpy(), candidate.numpy())


if __name__ == "__main__":
    unittest.main()

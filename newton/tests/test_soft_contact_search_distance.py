# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that value-only contact searches preserve the SDF lower bound."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.sdf_texture import TextureSDFData
from newton._src.geometry.soft_contacts_sdf import eval_shape_sdf, eval_shape_sdf_lower_bound


@wp.kernel
def sample_bounds(
    points: wp.array[wp.vec3],
    scale: wp.vec3,
    sdf: int,
    table: wp.array[TextureSDFData],
    result: wp.array[wp.vec2],
):
    i = wp.tid()
    original, _distance, _gradient = eval_shape_sdf(int(newton.GeoType.MESH), scale, points[i], sdf, table)
    optimized = eval_shape_sdf_lower_bound(int(newton.GeoType.MESH), scale, points[i], sdf, table)
    result[i] = wp.vec2(original, optimized)


@unittest.skipUnless(wp.is_cuda_available(), "Texture SDF queries require CUDA")
class TestSoftContactSearchDistance(unittest.TestCase):
    def test_value_only_preserves_scaled_search_distance(self):
        """Compare near-surface and far-field queries with mirrored/nonuniform scales."""
        builder = newton.ModelBuilder()
        config = newton.ModelBuilder.ShapeConfig()
        config.configure_sdf(force_sdf=True)
        builder.add_shape_mesh(-1, mesh=newton.Mesh.create_box(0.03, 0.04, 0.05), cfg=config)
        model = builder.finalize(device="cuda:0")
        sdf = int(model._shape_sdf_index.numpy()[0])
        self.assertGreaterEqual(sdf, 0)
        rng = np.random.default_rng(741)
        local = rng.uniform(-0.15, 0.15, (8192, 3)).astype(np.float32)
        output = wp.empty(len(local), dtype=wp.vec2, device=model.device)
        for scale in ((1.0, 1.0, 1.0), (2.0, 0.5, 1.25), (-1.0, 2.0, 0.3)):
            with self.subTest(scale=scale):
                points = wp.array(local * np.asarray(scale), dtype=wp.vec3, device=model.device)
                wp.launch(sample_bounds, len(local), [points, wp.vec3(*scale), sdf, model._texture_sdf_data, output])
                values = output.numpy()
                np.testing.assert_array_equal(values[:, 1], values[:, 0])


if __name__ == "__main__":
    unittest.main()

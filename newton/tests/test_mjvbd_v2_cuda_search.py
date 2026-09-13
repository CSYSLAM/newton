# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Require the speculative golden search to reproduce the original exactly."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry import soft_contacts_sdf as soft
from newton._src.solvers.mjvbd_v2 import fast_kernels as probe
from newton._src.solvers.mjvbd_v2.full_contact_pipeline import MJVBDV2CollisionPipeline
from newton.examples.mjvbdv2.example_mjvbd_v2_popcorn import popcorn_mesh

LANES = wp.constant(8)


@wp.kernel
def evaluate(
    points: wp.array[wp.vec3],
    index: int,
    scale: wp.vec3,
    table: wp.array[probe.TextureSDFData],
    iterations: int,
    output: wp.array2d[float],
):
    tid = wp.tid()
    i = tid // LANES
    a, b = points[2 * i], points[2 * i + 1]
    cell_cache = probe.CellCache()
    u, x, phi, grad, cell_cache = probe.optimize_edge_sdf(
        int(newton.GeoType.CONVEX_MESH), scale, a, b, index, table, iterations, cell_cache
    )
    if tid % LANES == 0:
        ref_u, ref_x, ref_phi, ref_grad = soft.optimize_edge_sdf(
            int(newton.GeoType.CONVEX_MESH), scale, a, b, index, table, iterations
        )
        output[i, 0] = u
        output[i, 8] = ref_u
        output[i, 1] = phi
        output[i, 9] = ref_phi
        for j in range(3):
            output[i, 2 + j] = x[j]
            output[i, 10 + j] = ref_x[j]
            output[i, 5 + j] = grad[j]
            output[i, 13 + j] = ref_grad[j]


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestSpeculativeSearch(unittest.TestCase):
    def test_pipeline_contacts_and_cache(self):
        """Keep warm and cold face contacts equal between independent pipelines."""
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig()
        cfg.configure_sdf(force_sdf=True)
        builder.add_shape_convex_hull(-1, mesh=newton.Mesh.create_box(0.03, 0.04, 0.05), cfg=cfg)
        builder.add_cloth_grid(
            pos=wp.vec3(-0.04, -0.04, 0.051),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=4,
            dim_y=4,
            cell_x=0.02,
            cell_y=0.02,
            mass=0.001,
            particle_radius=0.003,
        )
        model = builder.finalize(device="cuda:0")
        pipelines = [
            MJVBDV2CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True, enable_cuda_fast_path=fast)
            for fast in (False, True)
        ]
        self.assertIsNone(pipelines[0]._fast_face_search)
        self.assertIsNotNone(pipelines[1]._fast_face_search)
        contacts = [pipeline.contacts() for pipeline in pipelines]
        states = [model.state(), model.state()]
        original = model.particle_q.numpy()
        observed = 0
        for displacement in (0.0, 0.0001, -0.0001, 0.015, -0.003, 0.0):
            for pipeline, state, contact in zip(pipelines, states, contacts, strict=True):
                q = original.copy()
                q[:, 2] += displacement
                state.particle_q.assign(q)
                pipeline.collide(state, contact)
            rows = []
            for contact in contacts:
                count = int(contact.soft_contact_count.numpy()[0])
                observed += count
                ids = contact.soft_contact_indices.numpy()[:count]
                shapes = contact.soft_contact_shape.numpy()[:count]
                order = np.lexsort((ids[:, 2], ids[:, 1], ids[:, 0], shapes))
                rows.append(
                    {
                        name: getattr(contact, name).numpy()[:count][order]
                        for name in (
                            "soft_contact_indices",
                            "soft_contact_barycentric",
                            "soft_contact_shape",
                            "soft_contact_body_pos",
                            "soft_contact_normal",
                        )
                    }
                )
            for name in rows[0]:
                if name == "soft_contact_normal":
                    np.testing.assert_allclose(rows[0][name], rows[1][name], atol=2e-7, rtol=0)
                else:
                    np.testing.assert_array_equal(rows[0][name], rows[1][name])
            for name in ("_soft_face_cache_state", "_soft_face_cached_barycentric"):
                np.testing.assert_array_equal(getattr(pipelines[0], name).numpy(), getattr(pipelines[1], name).numpy())
        self.assertGreater(observed, 0)

    def test_exact(self):
        """Preserve samples, branch choices, witnesses, distances, and normals."""
        builder = newton.ModelBuilder()
        cfg = builder.ShapeConfig()
        cfg.configure_sdf(force_sdf=True)
        builder.add_shape_convex_hull(-1, mesh=popcorn_mesh(), cfg=cfg)
        builder.add_shape_convex_hull(-1, mesh=newton.Mesh.create_box(0.03, 0.04, 0.05), cfg=cfg)
        model = builder.finalize(device="cuda:0")
        rng = np.random.default_rng(172)
        count = 8192
        points = rng.normal(0.0, 0.025, (count, 2, 3)).astype(np.float32)
        points[: count // 2, 1] = points[: count // 2, 0] + rng.normal(0.0, 1e-5, (count // 2, 3))
        points[:100, 1] = points[:100, 0]
        data = wp.array(points.reshape(-1, 3), dtype=wp.vec3)
        output = wp.empty((count, 16))
        for index in model._shape_sdf_index.numpy():
            for scale in (wp.vec3(1.0), wp.vec3(-2.0, 0.5, 1.5)):
                for iterations in (0, 1, 2, 3, 4, 7, 16, 24):
                    wp.launch(
                        evaluate, count * 8, [data, int(index), scale, model._texture_sdf_data, iterations, output]
                    )
                    result = output.numpy()
                    print(
                        "index/scale/iterations/max difference:",
                        index,
                        scale,
                        iterations,
                        float(np.max(np.abs(result[:, :8] - result[:, 8:]))),
                        flush=True,
                    )
                    np.testing.assert_array_equal(result[:, :8], result[:, 8:])


if __name__ == "__main__":
    unittest.main()

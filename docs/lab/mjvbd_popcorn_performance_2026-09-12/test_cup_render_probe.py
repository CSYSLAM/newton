# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check per-material normals without changing mesh geometry or smoothing groups."""

import unittest

import numpy as np
import warp as wp
from cup_render_probe import pack_cup_vertices

from newton._src.viewer.gl.opengl import RenderVertex


class TestCupNormals(unittest.TestCase):
    def test_material_domains(self):
        """Preserve independent material normals, moved vertices and isolated vertices."""
        rng = np.random.default_rng(301)
        n = 40
        points = rng.normal(size=(n, 3)).astype(np.float32)
        groups = [rng.integers(0, n - 1, size=(60, 3)) for _ in range(4)]
        offsets, incident = [0], []
        for faces in groups:
            rows = [[] for _ in range(n)]
            for tri in faces:
                for vertex in tri:
                    rows[int(vertex)].append(tri)
            for row in rows:
                incident.extend(row)
                offsets.append(len(incident))
        q = wp.array(points, dtype=wp.vec3, device="cpu")
        starts = wp.array(offsets, dtype=int, device="cpu")
        faces = wp.array(np.asarray(incident), dtype=wp.vec3i, device="cpu")
        packed = wp.empty(4 * n, dtype=RenderVertex, device="cpu")
        for _ in range(2):
            wp.launch(pack_cup_vertices, dim=packed.size, inputs=[n, q, starts, faces, packed], device="cpu")
            result = packed.numpy()
            expected = []
            for triangles in groups:
                normals = np.zeros_like(points)
                for a, b, c in triangles:
                    normal = np.cross(points[b] - points[a], points[c] - points[a])
                    for vertex in (a, b, c):
                        normals[vertex] += normal
                lengths = np.linalg.norm(normals, axis=1)
                normals[lengths > 0] /= lengths[lengths > 0, None]
                expected.append(normals)
            np.testing.assert_allclose(result["normal"], np.concatenate(expected), rtol=2e-5, atol=2e-6)
            np.testing.assert_array_equal(result["pos"], np.tile(points, (4, 1)))
            np.testing.assert_array_equal(result["uv"], 0)
            points += rng.normal(size=points.shape).astype(np.float32) * 0.1
            q.assign(points)


if __name__ == "__main__":
    unittest.main()

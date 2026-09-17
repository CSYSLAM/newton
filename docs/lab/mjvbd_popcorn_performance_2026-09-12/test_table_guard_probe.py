# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Retain every table-overlapping triangle through GPU candidate filtering."""

import unittest

import numpy as np
import warp as wp
from table_guard_probe import mesh_candidates, triangle_candidates

from newton.examples.mjvbdv2 import example_mjvbd_v2_popcorn as scene


class TestTableGuard(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "GPU guard requires CUDA")
    def test_candidates_preserve_exact_sat(self):
        """Match CPU table SAT after rotated broad-phase filtering and graph replay."""
        rng = np.random.default_rng(12)
        mesh_count, triangles_per_mesh = 8, 30
        vertices = rng.normal(size=(mesh_count, triangles_per_mesh * 3, 3)) * 0.03
        centers = (vertices.min(axis=1) + vertices.max(axis=1)) / 2
        extents = (vertices.max(axis=1) - vertices.min(axis=1)) / 2
        poses = np.zeros((mesh_count, 7), dtype=np.float32)
        poses[:, :3] = [scene.TABLE_FRONT + 0.03, 0, scene.TABLE + 0.02]
        poses[:, 2] += np.linspace(-0.10, 0.25, mesh_count)
        rotations = rng.normal(size=(mesh_count, 4))
        poses[:, 3:] = rotations / np.linalg.norm(rotations, axis=1)[:, None]
        device = "cuda:0"
        q = wp.array(poses, dtype=wp.transform, device=device)
        bodies = wp.array(np.arange(mesh_count), dtype=int, device=device)
        owners = np.repeat(np.arange(mesh_count), triangles_per_mesh)
        indices = np.arange(vertices.size // 3).reshape(-1, 3)
        lower = wp.vec3d(scene.TABLE_FRONT - 0.001, -0.761, scene.TABLE - 0.051)
        upper = wp.vec3d(1.501 + scene.WORKSPACE_X, 0.761, scene.TABLE + 0.001)
        active = wp.zeros(mesh_count, dtype=int, device=device)
        count = wp.zeros(1, dtype=int, device=device)
        candidates = wp.empty(len(indices), dtype=int, device=device)
        mesh_inputs = [
            q,
            bodies,
            wp.array(centers, dtype=wp.vec3d, device=device),
            wp.array(extents, dtype=wp.vec3d, device=device),
            lower,
            upper,
            active,
        ]
        tri_inputs = [
            q,
            bodies,
            wp.array(owners, dtype=int, device=device),
            wp.array(vertices.reshape(-1, 3), dtype=wp.vec3d, device=device),
            wp.array(indices, dtype=wp.vec3i, device=device),
            active,
            lower,
            upper,
            count,
            candidates,
        ]

        def run():
            count.zero_()
            wp.launch(mesh_candidates, dim=mesh_count, inputs=mesh_inputs, device=device)
            wp.launch(triangle_candidates, dim=len(indices), inputs=tri_inputs, device=device)

        run()
        with wp.ScopedCapture(device=device) as capture:
            run()
        for _ in range(2):
            wp.capture_launch(capture.graph)
            selected = set(candidates.numpy()[: int(count.numpy()[0])].tolist())
            hit_count = 0
            for mesh in range(mesh_count):
                rotation = np.asarray(wp.quat_to_matrix(wp.quat(*poses[mesh, 3:]))).reshape(3, 3)
                world = vertices[mesh] @ rotation.T + poses[mesh, :3]
                for triangle in range(triangles_per_mesh):
                    tri = world[3 * triangle : 3 * triangle + 3]
                    if scene._mesh_intersects_table(tri, np.array([[0, 1, 2]])):
                        hit_count += 1
                        self.assertIn(mesh * triangles_per_mesh + triangle, selected)
            self.assertGreater(hit_count, 0)
            self.assertLess(len(selected), len(indices))


if __name__ == "__main__":
    unittest.main()

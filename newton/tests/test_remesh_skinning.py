# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for surface remesher and volume skinning."""

import unittest
import numpy as np


def _native_available():
    try:
        from newton._src.softbody import _voxel_tet  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_native_available(), "Native _voxel_tet module not built")
class TestRemeshSurface(unittest.TestCase):
    """Tests for the surface remesher."""

    def _make_cube(self):
        """Create a unit cube surface mesh."""
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=np.int32,
        )
        return vertices, faces

    def test_remesh_cube_basic(self):
        """Remesh a cube and verify output is valid."""
        from newton._src.softbody._voxelize_native import remesh_surface_native

        verts, faces = self._make_cube()
        result = remesh_surface_native(verts, faces, resolution=10)

        out_verts = result["vertices"]
        out_faces = result["tri_ids"]
        stats = result["debug_stats"]

        # Should produce some output
        self.assertGreater(len(out_verts), 0)
        self.assertGreater(len(out_faces), 0)

        # Vertices should be (N, 3) float32
        self.assertEqual(out_verts.ndim, 2)
        self.assertEqual(out_verts.shape[1], 3)
        self.assertEqual(out_verts.dtype, np.float32)

        # Faces should be (M, 3) uint32
        self.assertEqual(out_faces.ndim, 2)
        self.assertEqual(out_faces.shape[1], 3)

        # No NaN in output
        self.assertFalse(np.any(np.isnan(out_verts)))

        # All face indices should be valid
        max_idx = out_faces.max()
        self.assertLess(max_idx, len(out_verts))

        # Stats should be populated
        self.assertEqual(stats["input_vertex_count"], 8)
        self.assertEqual(stats["input_triangle_count"], 12)
        self.assertGreater(stats["output_vertex_count"], 0)

    def test_remesh_preserves_bounds(self):
        """Remeshed surface should be roughly within the input bounds."""
        from newton._src.softbody._voxelize_native import remesh_surface_native

        verts, faces = self._make_cube()
        result = remesh_surface_native(verts, faces, resolution=10)

        out_verts = result["vertices"]

        # Output should be near the input bounds (with some tolerance for
        # surface distance offset)
        tol = 0.5  # generous tolerance
        self.assertTrue(np.all(out_verts >= -tol))
        self.assertTrue(np.all(out_verts <= 1.0 + tol))

    def test_remesh_higher_resolution_more_vertices(self):
        """Higher resolution should produce more vertices."""
        from newton._src.softbody._voxelize_native import remesh_surface_native

        verts, faces = self._make_cube()

        result_low = remesh_surface_native(verts, faces, resolution=5)
        result_high = remesh_surface_native(verts, faces, resolution=20)

        self.assertGreater(
            result_high["debug_stats"]["output_vertex_count"],
            result_low["debug_stats"]["output_vertex_count"],
        )

    def test_remesh_sphere(self):
        """Remesh a sphere and verify basic properties."""
        from newton._src.softbody._voxelize_native import remesh_surface_native

        # Create a UV sphere
        n_lat, n_lon = 10, 20
        vertices = []
        for i in range(n_lat + 1):
            theta = np.pi * i / n_lat
            for j in range(n_lon):
                phi = 2 * np.pi * j / n_lon
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                vertices.append([x, y, z])

        vertices = np.array(vertices, dtype=np.float32)

        faces = []
        for i in range(n_lat):
            for j in range(n_lon):
                v0 = i * n_lon + j
                v1 = i * n_lon + (j + 1) % n_lon
                v2 = (i + 1) * n_lon + (j + 1) % n_lon
                v3 = (i + 1) * n_lon + j
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        faces = np.array(faces, dtype=np.int32)

        result = remesh_surface_native(vertices, faces, resolution=15)

        out_verts = result["vertices"]
        out_faces = result["tri_ids"]

        self.assertGreater(len(out_verts), 0)
        self.assertGreater(len(out_faces), 0)
        self.assertFalse(np.any(np.isnan(out_verts)))


@unittest.skipUnless(_native_available(), "Native _voxel_tet module not built")
class TestVolumeSkinning(unittest.TestCase):
    """Tests for the volume skinning / embedding."""

    def _make_tet_cube(self):
        """Create a unit cube divided into 5 tetrahedra."""
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        )
        # 5 tets that fill the cube
        tets = np.array(
            [
                [0, 1, 3, 4],
                [1, 2, 3, 6],
                [1, 4, 5, 6],
                [3, 4, 6, 7],
                [1, 3, 4, 6],
            ],
            dtype=np.int32,
        )
        return vertices, tets

    def test_embedding_basic(self):
        """Embed center point of cube and verify barycentric weights."""
        from newton._src.softbody._voxelize_native import compute_volume_embedding_native

        tet_verts, tet_indices = self._make_tet_cube()

        # Test point at the center of the cube
        render_verts = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)

        result = compute_volume_embedding_native(tet_verts, tet_indices, render_verts)

        tet_idx = result["tet_idx"]
        bary = result["bary_weights"]

        # Should have one embedding
        self.assertEqual(len(tet_idx), 1)
        self.assertEqual(bary.shape, (1, 4))

        # Weights should sum to ~1.0
        weight_sum = np.sum(bary[0])
        self.assertAlmostEqual(weight_sum, 1.0, places=3)

        # No negative weights for point inside the tet
        self.assertTrue(np.all(bary[0] >= -0.01))

    def test_embedding_corner(self):
        """Embed a corner vertex of the cube."""
        from newton._src.softbody._voxelize_native import compute_volume_embedding_native

        tet_verts, tet_indices = self._make_tet_cube()

        # Test point at origin (vertex 0)
        render_verts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        result = compute_volume_embedding_native(tet_verts, tet_indices, render_verts)

        bary = result["bary_weights"]
        weight_sum = np.sum(bary[0])
        self.assertAlmostEqual(weight_sum, 1.0, places=2)

    def test_embedding_multiple_points(self):
        """Embed multiple points at once."""
        from newton._src.softbody._voxelize_native import compute_volume_embedding_native

        tet_verts, tet_indices = self._make_tet_cube()

        # Multiple interior points
        render_verts = np.array(
            [
                [0.5, 0.5, 0.5],
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.75],
            ],
            dtype=np.float32,
        )

        result = compute_volume_embedding_native(tet_verts, tet_indices, render_verts)

        tet_idx = result["tet_idx"]
        bary = result["bary_weights"]

        self.assertEqual(len(tet_idx), 3)
        self.assertEqual(bary.shape, (3, 4))

        # All weights should sum to ~1.0
        for i in range(3):
            weight_sum = np.sum(bary[i])
            self.assertAlmostEqual(weight_sum, 1.0, places=2)

    def test_deform_identity(self):
        """Deformation with original positions should recover the original."""
        from newton._src.softbody._voxelize_native import (
            compute_volume_embedding_native,
            deform_with_embedding_native,
        )

        tet_verts, tet_indices = self._make_tet_cube()

        # Embed the tet vertices themselves
        render_verts = tet_verts.copy()
        result = compute_volume_embedding_native(tet_verts, tet_indices, render_verts)

        # Deform with original positions
        deformed = deform_with_embedding_native(
            tet_verts, tet_indices, result["tet_idx"], result["bary_weights"]
        )

        # Should approximately recover original positions
        np.testing.assert_allclose(deformed, render_verts, atol=1e-3)

    def test_deform_translation(self):
        """Translation of tet vertices should translate render vertices."""
        from newton._src.softbody._voxelize_native import (
            compute_volume_embedding_native,
            deform_with_embedding_native,
        )

        tet_verts, tet_indices = self._make_tet_cube()

        # Embed a point at center
        render_verts = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
        result = compute_volume_embedding_native(tet_verts, tet_indices, render_verts)

        # Translate all tet vertices by (1, 0, 0)
        translated = tet_verts + np.array([1, 0, 0], dtype=np.float32)

        deformed = deform_with_embedding_native(
            translated, tet_indices, result["tet_idx"], result["bary_weights"]
        )

        # Render vertex should also be translated
        np.testing.assert_allclose(deformed[0], [1.5, 0.5, 0.5], atol=1e-3)


@unittest.skipUnless(_native_available(), "Native _voxel_tet module not built")
class TestRemeshAndSkinningIntegration(unittest.TestCase):
    """Integration test: remesh -> tetrahedralize -> skin -> verify deformation."""

    def test_full_pipeline(self):
        """Full pipeline: remesh surface -> tetrahedralize -> compute embedding -> deform."""
        from newton._src.softbody._voxelize_native import (
            remesh_surface_native,
            compute_volume_embedding_native,
            deform_with_embedding_native,
            voxelize_soft_body_native,
        )

        # Create a cube
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ],
            dtype=np.int32,
        )

        # Step 1: Remesh surface
        remesh_result = remesh_surface_native(vertices, faces, resolution=8)
        remeshed_verts = remesh_result["vertices"]
        remeshed_faces = remesh_result["tri_ids"]
        self.assertGreater(len(remeshed_verts), 0)

        # Step 2: Tetrahedralize the remeshed surface
        tet_result = voxelize_soft_body_native(remeshed_verts, remeshed_faces, resolution=8)
        tet_verts = tet_result["tet_vertices"]
        tet_indices = tet_result["tet_indices"].reshape(-1, 4)
        self.assertGreater(len(tet_verts), 0)

        # Step 3: Compute embedding for remeshed vertices
        embed_result = compute_volume_embedding_native(
            tet_verts, tet_indices, remeshed_verts
        )
        self.assertEqual(len(embed_result["tet_idx"]), len(remeshed_verts))

        # Step 4: Verify deformation with identity
        deformed = deform_with_embedding_native(
            tet_verts, tet_indices, embed_result["tet_idx"], embed_result["bary_weights"]
        )
        # Should approximately recover the remeshed vertices
        np.testing.assert_allclose(deformed, remeshed_verts, atol=0.1)


if __name__ == "__main__":
    unittest.main()

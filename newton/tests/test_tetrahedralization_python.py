# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Python tetrahedralization backend."""

import unittest
from unittest import mock

import numpy as np

from newton._src.geometry import tetgen
from newton._src.geometry.tetgen_python.predicates import orient3d, insphere
from newton._src.geometry.tetgen_python.api import tetrahedralize_surface_mesh_python


class TestPredicates(unittest.TestCase):
    """Test geometric predicates."""

    def test_orient3d_sign(self):
        """Test orient3d sign convention: (a-d) dot ((b-d) cross (c-d))."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        d = np.array([0.0, 0.0, 1.0])
        # orient3d(a,b,c,d) = (a-d).dot((b-d).cross(c-d)) = -1
        vol = orient3d(a, b, c, d)
        self.assertEqual(vol, -1.0)

        # Swap b,c to flip sign
        vol2 = orient3d(a, c, b, d)
        self.assertEqual(vol2, 1.0)

    def test_orient3d_coplanar(self):
        """Test orient3d with coplanar points gives zero."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        c = np.array([0.0, 1.0, 0.0])
        d = np.array([0.5, 0.5, 0.0])
        vol = orient3d(a, b, c, d)
        self.assertAlmostEqual(vol, 0.0, places=10)

    def test_insphere_inside(self):
        """Test insphere: point inside circumsphere returns > 0."""
        # Use a tet with positive orient3d so sign convention is standard
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])
        d = np.array([0.0, 0.0, 1.0])
        # orient3d(a,b,c,d) should be positive
        vol = orient3d(a, b, c, d)
        self.assertGreater(vol, 0.0)

        # Center of circumsphere should be inside
        center = np.array([0.25, 0.25, 0.25])
        result = insphere(a, b, c, d, center)
        self.assertGreater(result, 0.0)

    def test_insphere_outside(self):
        """Test insphere: point far outside circumsphere returns < 0."""
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.array([1.0, 0.0, 0.0])
        d = np.array([0.0, 0.0, 1.0])
        e = np.array([10.0, 10.0, 10.0])
        result = insphere(a, b, c, d, e)
        self.assertLess(result, 0.0)


class TestTetrahedralizationPython(unittest.TestCase):
    """Test the Python tetrahedralization backend."""

    def test_cube_convex(self):
        """Test tetrahedralization of a cube (convex shortcut)."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float32)

        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5],
        ], dtype=np.int32)

        tet_verts, tet_indices = tetrahedralize_surface_mesh_python(
            vertices, faces, verbose=False
        )

        self.assertGreater(len(tet_verts), 0)
        self.assertGreater(len(tet_indices), 0)
        self.assertEqual(len(tet_indices) % 4, 0)
        self.assertLess(tet_indices.max(), len(tet_verts))
        self.assertGreaterEqual(tet_indices.min(), 0)

        # Check all tets have consistent orientation (non-zero volume)
        tets = tet_indices.reshape(-1, 4)
        for t in tets:
            vol = orient3d(tet_verts[t[0]], tet_verts[t[1]],
                          tet_verts[t[2]], tet_verts[t[3]])
            self.assertNotAlmostEqual(vol, 0.0, places=10)

    def test_output_format(self):
        """Test that output format matches TetMesh expectations."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float32)

        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5],
        ], dtype=np.int32)

        tet_verts, tet_indices = tetrahedralize_surface_mesh_python(
            vertices, faces, verbose=False
        )

        self.assertEqual(tet_verts.dtype, np.float32)
        self.assertEqual(tet_indices.dtype, np.int32)
        self.assertEqual(tet_verts.ndim, 2)
        self.assertEqual(tet_verts.shape[1], 3)
        self.assertEqual(tet_indices.ndim, 1)

    def test_backend_python_integration(self):
        """Test backend='python' through the tetgen module."""
        from newton._src.geometry.tetgen import tetrahedralize_surface_mesh

        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ], dtype=np.float32)

        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5],
        ], dtype=np.int32)

        tet_verts, tet_indices = tetrahedralize_surface_mesh(
            vertices, faces, backend="python", verbose=False
        )

        self.assertGreater(len(tet_verts), 0)
        self.assertGreater(len(tet_indices), 0)

    def test_tetrahedralize_obj_forwards_voxel_parameters(self):
        """Test that tetrahedralize_obj forwards voxel backend parameters."""
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        tet_vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        tet_indices = np.array([0, 1, 2, 3], dtype=np.int32)

        with (
            mock.patch("newton._src.geometry.tetgen.os.path.exists", return_value=True),
            mock.patch("newton._src.geometry.tetgen.load_mesh", return_value=(vertices, faces)),
            mock.patch(
                "newton._src.geometry.tetgen.tetrahedralize_surface_mesh",
                return_value=(tet_vertices, tet_indices),
            ) as tetrahedralize_mock,
        ):
            tetgen.tetrahedralize_obj(
                "cow.obj",
                backend="voxel",
                resolution=16,
                num_relaxation_iters=7,
                rel_min_tet_volume=0.125,
                surface_dist_ratio=0.35,
            )

        tetrahedralize_mock.assert_called_once()
        kwargs = tetrahedralize_mock.call_args.kwargs
        self.assertEqual(kwargs["backend"], "voxel")
        self.assertEqual(kwargs["resolution"], 16)
        self.assertEqual(kwargs["num_relaxation_iters"], 7)
        self.assertEqual(kwargs["rel_min_tet_volume"], 0.125)
        self.assertEqual(kwargs["surface_dist_ratio"], 0.35)


if __name__ == "__main__":
    unittest.main()

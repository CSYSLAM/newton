# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np


def _voxel_available():
    try:
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_voxel_available(), "Native voxel module not available")
class TestVoxelTetrahedralization(unittest.TestCase):

    @staticmethod
    def _cube_mesh():
        verts = np.array([
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
        return verts, faces

    def test_cube_basic(self):
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native
        verts, faces = self._cube_mesh()
        result = voxelize_soft_body_native(verts, faces, resolution=8)
        tv = result["tet_vertices"]
        ti = result["tet_indices"]
        self.assertEqual(tv.ndim, 2)
        self.assertEqual(tv.shape[1], 3)
        self.assertGreater(len(tv), 0)
        self.assertGreater(len(ti), 0)
        self.assertEqual(len(ti) % 4, 0)

    def test_indices_valid(self):
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native
        verts, faces = self._cube_mesh()
        result = voxelize_soft_body_native(verts, faces, resolution=8)
        tv = result["tet_vertices"]
        ti = result["tet_indices"]
        self.assertTrue(np.all(ti >= 0))
        self.assertTrue(np.all(ti < len(tv)))

    def test_no_nan(self):
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native
        verts, faces = self._cube_mesh()
        result = voxelize_soft_body_native(verts, faces, resolution=8)
        tv = result["tet_vertices"]
        self.assertFalse(np.any(np.isnan(tv)))

    def test_positive_abs_volumes(self):
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native
        verts, faces = self._cube_mesh()
        result = voxelize_soft_body_native(verts, faces, resolution=8)
        tv = result["tet_vertices"]
        ti = result["tet_indices"].reshape(-1, 4)
        for row in ti:
            a, b, c, d = tv[row[0]], tv[row[1]], tv[row[2]], tv[row[3]]
            vol = abs(np.dot(a - d, np.cross(b - d, c - d))) / 6.0
            self.assertGreater(vol, 0.0)

    def test_debug_stats(self):
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native
        verts, faces = self._cube_mesh()
        result = voxelize_soft_body_native(verts, faces, resolution=8)
        stats = result["debug_stats"]
        self.assertIn("num_tets", stats)
        self.assertIn("num_unique_verts", stats)
        self.assertIn("grid_spacing", stats)
        self.assertGreater(stats["num_tets"], 0)
        self.assertGreater(stats["num_unique_verts"], 0)
        self.assertGreater(stats["grid_spacing"], 0.0)

    def test_surface_dist_ratio(self):
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native
        verts, faces = self._cube_mesh()
        result_low = voxelize_soft_body_native(verts, faces, resolution=8, surface_dist_ratio=0.1)
        result_high = voxelize_soft_body_native(verts, faces, resolution=8, surface_dist_ratio=0.5)
        self.assertGreater(len(result_low["tet_vertices"]), 0)
        self.assertGreater(len(result_high["tet_vertices"]), 0)

    def test_summarize(self):
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native, summarize_voxelize_result
        verts, faces = self._cube_mesh()
        result = voxelize_soft_body_native(verts, faces, resolution=8)
        summary = summarize_voxelize_result(result)
        self.assertIn("vertex_count", summary)
        self.assertIn("tet_count", summary)
        self.assertIn("total_signed_volume", summary)
        self.assertEqual(summary["vertex_count"], len(result["tet_vertices"]))
        self.assertEqual(summary["tet_count"], len(result["tet_indices"]) // 4)

    def test_input_validation_bad_shape(self):
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native
        with self.assertRaises(ValueError):
            voxelize_soft_body_native(
                np.array([[0, 0]], dtype=np.float32),
                np.array([[0, 0, 0]], dtype=np.int32),
                resolution=8,
            )

    def test_input_validation_bad_resolution(self):
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native
        verts, faces = self._cube_mesh()
        with self.assertRaises(ValueError):
            voxelize_soft_body_native(verts, faces, resolution=-1)

    def test_tetgen_voxel_backend(self):
        from newton._src.geometry.tetgen import tetrahedralize_surface_mesh
        verts, faces = self._cube_mesh()
        tv, ti = tetrahedralize_surface_mesh(verts, faces, backend="voxel", resolution=8)
        self.assertEqual(tv.dtype, np.float32)
        self.assertEqual(ti.dtype, np.int32)
        self.assertGreater(len(tv), 0)
        self.assertGreater(len(ti), 0)

    def test_tetgen_voxel_with_surface_dist_ratio(self):
        from newton._src.geometry.tetgen import tetrahedralize_surface_mesh
        verts, faces = self._cube_mesh()
        tv, ti = tetrahedralize_surface_mesh(
            verts, faces, backend="voxel", resolution=8,
            num_relaxation_iters=5, rel_min_tet_volume=0.05,
            surface_dist_ratio=0.3,
        )
        self.assertGreater(len(tv), 0)
        self.assertGreater(len(ti), 0)

    def test_sphere_mesh(self):
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native
        # Generate a simple sphere mesh
        verts_list = []
        faces_list = []
        n_lat, n_lon = 8, 12
        for i in range(n_lat + 1):
            theta = np.pi * i / n_lat
            for j in range(n_lon):
                phi = 2 * np.pi * j / n_lon
                x = np.sin(theta) * np.cos(phi)
                y = np.sin(theta) * np.sin(phi)
                z = np.cos(theta)
                verts_list.append([x, y, z])
        verts = np.array(verts_list, dtype=np.float32)
        for i in range(n_lat):
            for j in range(n_lon):
                j1 = (j + 1) % n_lon
                v0 = i * n_lon + j
                v1 = i * n_lon + j1
                v2 = (i + 1) * n_lon + j
                v3 = (i + 1) * n_lon + j1
                faces_list.append([v0, v1, v2])
                faces_list.append([v1, v3, v2])
        faces = np.array(faces_list, dtype=np.int32)
        result = voxelize_soft_body_native(verts, faces, resolution=8)
        tv = result["tet_vertices"]
        ti = result["tet_indices"]
        self.assertGreater(len(tv), 0)
        self.assertGreater(len(ti), 0)
        self.assertFalse(np.any(np.isnan(tv)))

    def test_high_resolution(self):
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native
        verts, faces = self._cube_mesh()
        result = voxelize_soft_body_native(verts, faces, resolution=16)
        self.assertGreater(len(result["tet_vertices"]), 0)

    def test_zero_relaxation_iters(self):
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native
        verts, faces = self._cube_mesh()
        result = voxelize_soft_body_native(verts, faces, resolution=8, num_relaxation_iters=0)
        self.assertGreater(len(result["tet_vertices"]), 0)
        self.assertFalse(np.any(np.isnan(result["tet_vertices"])))

    def test_input_validation_bad_surface_dist_ratio(self):
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native
        verts, faces = self._cube_mesh()
        with self.assertRaises(ValueError):
            voxelize_soft_body_native(verts, faces, resolution=8, surface_dist_ratio=-0.1)

    def test_input_validation_bad_rel_min_tet_volume(self):
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native
        verts, faces = self._cube_mesh()
        with self.assertRaises(ValueError):
            voxelize_soft_body_native(verts, faces, resolution=8, rel_min_tet_volume=-0.1)

    def test_input_validation_bad_num_relaxation_iters(self):
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native
        verts, faces = self._cube_mesh()
        with self.assertRaises(ValueError):
            voxelize_soft_body_native(verts, faces, resolution=8, num_relaxation_iters=-1)

    def test_debug_stats_grid_dims(self):
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native
        verts, faces = self._cube_mesh()
        result = voxelize_soft_body_native(verts, faces, resolution=8)
        stats = result["debug_stats"]
        self.assertGreater(stats["num_voxel_grid_x"], 0)
        self.assertGreater(stats["num_voxel_grid_y"], 0)
        self.assertGreater(stats["num_voxel_grid_z"], 0)


if __name__ == "__main__":
    unittest.main()

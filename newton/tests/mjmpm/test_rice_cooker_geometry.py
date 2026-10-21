# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check the hollow cup shared by MJMPM contact and Blender visual export."""

import json
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from newton.examples.mjmpm.support import rice_cooker_geometry
from newton.examples.mjmpm.support.rice_cooker_geometry import CUP_WALL, HANDLE_BOXES, cup_mesh


class TestRiceCookerGeometry(unittest.TestCase):
    def test_round_cup_is_closed_with_open_cavity_and_handle(self):
        """Keep the cup manifold and outward oriented while leaving its mouth and grip opening empty."""
        vertices, triangles = cup_mesh()
        a, b, c = (vertices[triangles[:, i]] for i in range(3))
        areas = np.linalg.norm(np.cross(b - a, c - a), axis=1)
        self.assertGreater(float(areas.min()), 1e-9)
        volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
        self.assertGreater(float(volume), 0)
        edges = Counter()
        for triangle in triangles:
            for i in range(3):
                edges[(int(triangle[i]), int(triangle[(i + 1) % 3]))] += 1
        for (i, j), count in edges.items():
            self.assertEqual(count, 1)
            self.assertEqual(edges[(j, i)], 1)

        # A vertical ray through the cavity must encounter only the two floor
        # surfaces; a hidden cap at the mouth would trap material during a pour.
        hits = []
        for triangle in vertices[triangles]:
            matrix = (triangle[1:, :2] - triangle[0, :2]).T
            if abs(np.linalg.det(matrix)) < 1e-12:
                continue
            uv = np.linalg.solve(matrix, np.array((0.001, 0.002)) - triangle[0, :2])
            if min(uv) >= 0 and uv.sum() <= 1:
                hits.append(triangle[0, 2] + uv @ (triangle[1:, 2] - triangle[0, 2]))
        np.testing.assert_allclose(sorted(hits), [0.0, CUP_WALL], atol=1e-7)
        opening = np.array((-0.05, 0.0, 0.055))
        for center, half in HANDLE_BOXES:
            self.assertFalse(np.all(np.abs(opening - np.asarray(center)) <= half))

        # The display wall is thin, while its inner contact boundary and floor
        # agree with the outward-thickened numerical shell.
        visual, _ = rice_cooker_geometry.cup_visual_mesh(segments=96)
        physical, _ = cup_mesh(segments=96)
        np.testing.assert_array_equal(visual[1 + 3 * 96 :], physical[1 + 3 * 96 :])
        thickness = np.linalg.norm(visual[1 + 2 * 96 : 1 + 3 * 96, :2] - visual[1 + 3 * 96 : 1 + 4 * 96, :2], axis=1)
        np.testing.assert_allclose(thickness, 0.002, atol=1e-7)

        # Detect stale Blender exports independently of the numerical thickness.
        path = Path(rice_cooker_geometry.__file__).parents[1] / "assets/rice_cooker/realistic_visuals.npz"
        with np.load(path, allow_pickle=False) as archive:
            part = next(p for p in json.loads(str(archive["manifest"])) if p["name"] == "Newton pitcher shell")
            displayed = archive[part["key"] + "_points"]
        np.testing.assert_array_equal(np.unique(displayed, axis=0), np.unique(visual, axis=0))

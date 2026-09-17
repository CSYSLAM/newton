# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate the authored belt loop and its shared motion clock."""

import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.mjvbdv2.support.conveyor_belt import ConveyorBeltVisual

ASSETS = Path(__file__).resolve().parents[2] / "assets/conveyor_station"


class TestConveyorBelt(unittest.TestCase):
    def test_closed_laminate(self):
        """Require a watertight belt with outward winding and continuous UV seam."""
        with np.load(ASSETS / "belt.npz", allow_pickle=False) as asset:
            i = next(i for i in range(len(asset["materials"])) if len(asset[f"indices_{i}"]))
            vertices = asset[f"vertices_{i}"]
            normals = asset[f"normals_{i}"]
            uv = asset[f"uvs_{i}"]
            triangles = asset[f"indices_{i}"].reshape(-1, 3)
            self.assertTrue(np.isfinite(vertices).all())
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
            self.assertAlmostEqual(float(vertices[:, 2].max()), 0.7601, places=6)
            self.assertAlmostEqual(float(vertices[:, 2].min()), 0.6559, places=6)
            cross = np.cross(
                vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
            )
            self.assertTrue(np.all(np.sum(cross * normals[triangles].mean(axis=1), axis=1) > 0))
            _, weld = np.unique(np.round(vertices, 6), axis=0, return_inverse=True)
            edges = Counter(
                tuple(sorted((int(a), int(b))))
                for tri in weld[triangles]
                for a, b in zip(tri, np.roll(tri, -1), strict=True)
            )
            self.assertTrue(all(count == 2 for count in edges.values()))
            start = np.flatnonzero(np.isclose(uv[:, 1], 0))
            end = np.flatnonzero(np.isclose(uv[:, 1], 1))
            self.assertEqual(len(start), len(end))
            np.testing.assert_allclose(vertices[start], vertices[end], atol=1e-6)
            # Blender's encoded custom normals have finite angular precision.
            np.testing.assert_allclose(normals[start], normals[end], atol=2e-5)

    def test_motion_and_stops(self):
        """Move the top toward pickup, reverse the return, and freeze both on a stop."""
        viewer = SimpleNamespace(log_mesh=lambda *args, **kwargs: None, log_instances=lambda *args, **kwargs: None)
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            with self.subTest(device=device):
                visual = ConveyorBeltVisual(ASSETS, device)
                visual.render(viewer, 0.0)
                uv0 = visual.belt_uv.numpy().copy()
                points = visual.parts[0][2].numpy().copy()
                dt, speed = 0.001, 0.12
                visual.render(viewer, speed * dt)
                uv1 = visual.belt_uv.numpy().copy()
                rotation = wp.quat(*visual.roller_transforms.numpy()[0, 3:])
                top = np.asarray(wp.quat_rotate(rotation, wp.vec3(0, 0, visual.radius)))
                bottom = np.asarray(wp.quat_rotate(rotation, wp.vec3(0, 0, -visual.radius)))
                self.assertAlmostEqual(float(top[1]) / dt, -speed, places=5)
                self.assertAlmostEqual(float(bottom[1]) / dt, speed, places=5)
                np.testing.assert_allclose(uv1[:, 1] - uv0[:, 1], -speed * dt / visual.path_length, atol=1e-7)
                np.testing.assert_array_equal(points, visual.parts[0][2].numpy())
                transforms = visual.roller_transforms.numpy().copy()
                visual.render(viewer, speed * dt)
                np.testing.assert_array_equal(uv1, visual.belt_uv.numpy())
                np.testing.assert_array_equal(transforms, visual.roller_transforms.numpy())
                visual.render(viewer, visual.path_length)
                np.testing.assert_allclose(uv0, visual.belt_uv.numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()

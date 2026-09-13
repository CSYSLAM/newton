# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the Blender export against the contact-driven demo's physical frames."""

import json
import unittest

import numpy as np
import warp as wp

from newton.examples.mjvbdv2.example_mjvbd_v2_popcorn import (
    HANDLE,
    POPCORN_PROPS,
    _popcorn_spawn_position,
    _scoop_floor_height_offset,
    cup_mesh,
    popcorn_mesh,
    scoop_bowl_panels,
)


class TestPopcornAssets(unittest.TestCase):
    def test_continuous_sheet_matches_contact_envelope(self):
        """Smooth the displayed pan without changing its contact profile."""
        data = json.loads(POPCORN_PROPS.read_text(encoding="utf-8"))
        sheets = [m for m in data["meshes"] if m["name"].startswith("Continuous pressed scoop sheet")]
        self.assertEqual(len(sheets), 1)
        vertices = np.asarray(sheets[0]["vertices"])
        reference = np.concatenate([p.vertices for p in scoop_bowl_panels()])
        distance = np.linalg.norm(vertices[:, None] - reference[None], axis=-1).min(axis=1)
        self.assertLess(float(distance.max()), 1e-7)
        internal = [m for m in data["meshes"] if m["name"].startswith("Formed aluminum pan")]
        self.assertEqual(sum(not m.get("display", True) for m in internal), 12)

    def test_extra_grains_clear_initial_scoop(self):
        """Keep the enlarged pile outside the initial pan including grain size."""
        pan = np.concatenate([panel.vertices for panel in scoop_bowl_panels()]) + HANDLE
        grain = np.asarray(popcorn_mesh().vertices)
        radius = np.linalg.norm(grain[:, :2], axis=1).max()
        extent = np.array((radius, radius, np.abs(grain[:, 2]).max()))
        for index in range(192):
            center = _popcorn_spawn_position(index)
            overlap = np.all(center + extent > pan.min(axis=0)) and np.all(center - extent < pan.max(axis=0))
            self.assertFalse(overlap, f"grain {index} initially intersects the held pan")

    def test_observed_floor_clearance(self):
        """Use the actual pan attitude to preserve clearance after grasp slip."""
        vertices = np.concatenate([panel.vertices for panel in scoop_bowl_panels()])
        for angle in (-0.12, 0.0, 0.09):
            rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), angle)
            matrix = np.asarray(wp.quat_to_matrix(rotation)).reshape(3, 3)
            offset = _scoop_floor_height_offset(vertices, 0.09, observed_rotation=rotation)
            self.assertAlmostEqual(float((vertices @ matrix[2]).min()) + offset, float(vertices[:, 2].min()), places=7)

    def test_export_topology(self):
        """Preserve every physical cup vertex and face in the Blender export."""
        data = json.loads(POPCORN_PROPS.read_text(encoding="utf-8"))
        self.assertEqual(data["units"], "m")
        self.assertEqual(data["version"], 1)
        cups = [mesh for mesh in data["meshes"] if mesh["group"] == "cup"]
        self.assertEqual(len(cups), 1)
        vertices, faces = cup_mesh()
        np.testing.assert_allclose(cups[0]["vertices"], vertices, atol=1e-7, rtol=0)
        np.testing.assert_array_equal(cups[0]["faces"], faces)
        self.assertEqual(len(cups[0]["material_indices"]), len(faces))
        self.assertEqual(set(cups[0]["material_indices"]), {0, 1, 2})

    def test_pan_contact_correspondence(self):
        """Keep the displayed sheet-metal pan coincident with its contact panels."""
        data = json.loads(POPCORN_PROPS.read_text(encoding="utf-8"))
        panels = sorted(
            (m for m in data["meshes"] if m["name"].startswith("Formed aluminum pan")),
            key=lambda m: m["name"],
        )
        original = scoop_bowl_panels()
        self.assertEqual(len(panels), len(original))
        for exported, physical in zip(panels, original, strict=True):
            np.testing.assert_allclose(exported["vertices"], physical.vertices, atol=1e-7, rtol=0)
            np.testing.assert_array_equal(exported["faces"], physical.indices.reshape(-1, 3))

    def test_valid_geometry(self):
        """Reject invalid mesh indices, nonfinite vertices and duplicate names."""
        data = json.loads(POPCORN_PROPS.read_text(encoding="utf-8"))
        names = [mesh["name"] for mesh in data["meshes"]]
        self.assertEqual(len(names), len(set(names)))
        for mesh in data["meshes"]:
            vertices = np.asarray(mesh["vertices"])
            faces = np.asarray(mesh["faces"])
            self.assertTrue(np.isfinite(vertices).all(), mesh["name"])
            self.assertEqual(vertices.shape[1], 3)
            self.assertEqual(faces.shape[1], 3)
            self.assertGreaterEqual(faces.min(), 0)
            self.assertLess(faces.max(), len(vertices))
            self.assertIn(mesh["group"], ("machine", "scoop", "cup"))
            if mesh["group"] != "cup":
                display = np.asarray(mesh["render_vertices"])
                normals = np.asarray(mesh["render_normals"])
                np.testing.assert_allclose(display, vertices[faces.reshape(-1)], atol=1e-7)
                self.assertEqual(normals.shape, display.shape)
                np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

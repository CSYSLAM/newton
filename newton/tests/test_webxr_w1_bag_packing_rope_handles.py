# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check independent rope-handle geometry, attachment, layout, and Quest export."""

import json
import struct
import unittest

import numpy as np
import warp as wp

from newton.examples.mjvbdv2 import example_mjvbd_v2_w1_bag_packing as packing
from newton.examples.mjvbdv2.example_mjvbd_v2_webxr_w1_bag_packing import Example as Original
from newton.examples.mjvbdv2.example_mjvbd_v2_webxr_w1_bag_packing_rope_handles import Example
from newton.examples.mjvbdv2.support.w1_bag_recording import scene_signature
from newton.viewer import ViewerNull


class TestRopeHandlePacking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Build real scene models on CPU without IK, SDFs, physics solvers, or servers."""
        cls.scenes = []
        with wp.ScopedDevice("cpu"):
            for example in (Original, Example):
                scene = example.__new__(example)
                packing.Example.__init__(scene, ViewerNull(), example.create_parser().parse_args([]), render_only=True)
                cls.scenes.append(scene)

    def test_round_handles_point_away_from_mouth_and_join_paper(self):
        """Attach two rounded tubes without holes, flipped faces, or initial paper crossings."""
        original, rope = self.scenes
        np.testing.assert_array_equal(rope.rest[: rope.paper_count], original.rest[: original.paper_count])
        np.testing.assert_array_equal(rope.faces[: rope.paper_faces], original.faces[: original.paper_faces])
        directed = np.concatenate([rope.faces[:, [0, 1]], rope.faces[:, [1, 2]], rope.faces[:, [2, 0]]])
        edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
        self.assertTrue(np.all(counts <= 2))
        np.testing.assert_array_equal(np.unique(edges[counts == 1]), np.sort(rope.rim))
        direction = np.where(directed[:, 0] < directed[:, 1], 1, -1)
        np.testing.assert_array_equal(np.bincount(inverse, weights=direction)[counts == 2], 0)
        triangles = rope.rest[rope.faces]
        areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
        self.assertGreater(float(areas.min()), 1e-8)
        with np.load(rope.assets / "bag.npz") as data:
            for key in ("handle_0", "handle_1"):
                ids = data[key]
                self.assertTrue(np.all(ids[:4] < rope.paper_count))
                self.assertTrue(np.all(ids[-4:] < rope.paper_count))
                rings = rope.rest[data[key + "_rings"]]
                centers = rings.mean(axis=1)
                root_height = rope.rest[ids[:4], 2].mean()
                self.assertLess(float(centers[:, 2].max()), root_height)
                self.assertGreater(float(root_height - centers[:, 2].min()), 0.10)
                # Circular sections remain close to 9 mm diameter along both tubes.
                radii = np.linalg.norm(rings - centers[:, None], axis=2)
                self.assertGreater(float(radii.min()), 0.004)
                self.assertLess(float(radii.max()), 0.005)
                self.assertGreater(float(np.abs(rings[:, :, 0]).min()), packing.DEPTH / 2)
        self.assertEqual(
            packing.count_handle_crossings(rope.rest, rope.faces[: rope.paper_faces], rope.free_handle_edges), 0
        )
        particles = rope.state_0.particle_q.numpy()[: rope.bag_particle_count]
        self.assertGreater(float(particles[:, 2].min()), packing.TABLE_Z)

    def test_preserve_existing_scene_parameters_and_layout(self):
        """Shift and stiffen only the rope bag while preserving other objects and contacts."""
        original, rope = self.scenes
        a, b = vars(original.args).copy(), vars(rope.args).copy()
        for key in ("bag_variant", "webxr_port"):
            a.pop(key)
            b.pop(key)
        self.assertEqual(a, b)
        self.assertEqual(rope.args.bag_variant, "rope-handles")
        self.assertEqual(rope.args.webxr_port, 8775)
        self.assertEqual(Original.handle_color, (0.42, 0.32, 0.16))
        np.testing.assert_allclose(
            original.state_0.particle_q.numpy()[: original.paper_count] + np.array((0.0, 0.08, 0.0)),
            rope.state_0.particle_q.numpy()[: rope.paper_count],
            atol=1e-6,
        )
        for name in (
            "body_mass",
            "body_inertia",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_mu",
            "shape_transform",
            "shape_scale",
            "shape_margin",
            "shape_gap",
            "tet_materials",
        ):
            np.testing.assert_array_equal(getattr(original.model, name).numpy(), getattr(rope.model, name).numpy())
        paper_materials = original.model.tri_materials.numpy()[: original.paper_faces].copy()
        paper_materials[:, :2] *= 1.5
        np.testing.assert_array_equal(paper_materials, rope.model.tri_materials.numpy()[: rope.paper_faces])
        for name in ("particle_mass", "particle_q", "particle_radius"):
            np.testing.assert_array_equal(
                getattr(original.model, name).numpy()[original.soft_cube_start :],
                getattr(rope.model, name).numpy()[rope.soft_cube_start :],
            )
        for name in ("step", "reset_physics", "_build_simulation", "_update_grasp_limit"):
            self.assertIs(getattr(Original, name), getattr(Example, name))
        self.assertNotEqual(scene_signature(original), scene_signature(rope))

    def test_export_dark_rope_as_deformable_mesh(self):
        """Export complete round geometry with a separate dark rope material."""
        rope = self.scenes[1]
        rope._static_boxes, rope._bag_meshes = [], []
        rope._soft_cube_mesh = None
        payload = rope._build_webxr_geometry()
        size = struct.unpack_from("<I", payload, 4)[0]
        header = json.loads(payload[8 : 8 + size])
        bags = [shape for shape in header["shapes"] if shape["role"] == "bag"]
        self.assertEqual(len(bags), 2)
        self.assertEqual(bags[1]["color"], list(Example.handle_color))
        self.assertEqual(header["meshes"][bags[1]["mesh"]]["indexCount"], 3 * (len(rope.faces) - rope.paper_faces))


if __name__ == "__main__":
    unittest.main()

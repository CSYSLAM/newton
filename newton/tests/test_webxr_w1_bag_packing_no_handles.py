# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the separate bag asset, shared physics, and Quest geometry on CPU."""

import json
import struct
import unittest

import numpy as np
import warp as wp

from newton.examples.mjvbdv2 import example_mjvbd_v2_w1_bag_packing as packing
from newton.examples.mjvbdv2.example_mjvbd_v2_webxr_w1_bag_packing import Example as Handled
from newton.examples.mjvbdv2.example_mjvbd_v2_webxr_w1_bag_packing_no_handles import Example as HandleFree
from newton.examples.mjvbdv2.support.w1_bag_recording import scene_signature
from newton.viewer import ViewerNull


class TestHandleFreePacking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Build real scene models without starting CUDA, solvers, IK, or servers."""
        cls.scenes = []
        with wp.ScopedDevice("cpu"):
            for example in (Handled, HandleFree):
                scene = example.__new__(example)
                args = example.create_parser().parse_args([])
                packing.Example.__init__(scene, ViewerNull(), args, render_only=True)
                cls.scenes.append(scene)

    def test_closed_walls_and_original_paper_vertices(self):
        """Remove ribbons and close their four attachment holes, leaving only the mouth open."""
        original, plain = self.scenes
        self.assertEqual(plain.bag_particle_count, original.paper_count)
        self.assertEqual(plain.paper_faces, original.paper_faces + 8)
        self.assertEqual(len(plain.free_handle_edges), 0)
        np.testing.assert_array_equal(plain.rest, original.rest[: original.paper_count])
        np.testing.assert_array_equal(plain.faces[: original.paper_faces], original.faces[: original.paper_faces])
        np.testing.assert_array_equal(np.unique(plain.faces), np.arange(plain.bag_particle_count))
        edges = np.sort(
            np.concatenate([plain.faces[:, [0, 1]], plain.faces[:, [1, 2]], plain.faces[:, [2, 0]]]), axis=1
        )
        edges, counts = np.unique(edges, axis=0, return_counts=True)
        self.assertTrue(np.all(counts <= 2))
        np.testing.assert_array_equal(np.unique(edges[counts == 1]), np.sort(plain.rim))
        patches = plain.rest[plain.faces[original.paper_faces :]]
        normals = np.cross(patches[:, 1] - patches[:, 0], patches[:, 2] - patches[:, 0])
        self.assertTrue(np.all(normals[:, 0] * patches[:, :, 0].mean(axis=1) > 0))
        for name in ("snacks.npz", "snacks.json", "kraft.png", "dimensions.json"):
            self.assertEqual((original.assets / name).read_bytes(), (plain.assets / name).read_bytes())

    def test_same_layout_materials_and_controls(self):
        """Keep robot, paper-body placement, snacks, cube, and contact parameters unchanged."""
        original, plain = self.scenes
        a, b = vars(original.args).copy(), vars(plain.args).copy()
        for key in ("bag_variant", "webxr_port"):
            a.pop(key)
            b.pop(key)
        self.assertEqual(a, b)
        self.assertEqual(original.args.bag_variant, "handles")
        self.assertEqual(plain.args.bag_variant, "no-handles")
        self.assertEqual((original.args.webxr_port, plain.args.webxr_port), (8773, 8774))
        for name in ("step", "_build_simulation", "_update_grasp_limit", "reset_physics"):
            self.assertIs(getattr(Handled, name), getattr(HandleFree, name))
        np.testing.assert_allclose(
            original.state_0.particle_q.numpy()[: original.paper_count],
            plain.state_0.particle_q.numpy()[: plain.paper_count],
            atol=1e-6,
        )
        for name in ("body_q", "joint_q"):
            np.testing.assert_array_equal(getattr(original.state_0, name).numpy(), getattr(plain.state_0, name).numpy())
        for name in (
            "body_mass",
            "body_inertia",
            "shape_type",
            "shape_transform",
            "shape_scale",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_mu",
            "shape_margin",
            "shape_gap",
            "tet_materials",
        ):
            np.testing.assert_array_equal(getattr(original.model, name).numpy(), getattr(plain.model, name).numpy())
        for name in ("soft_contact_ke", "soft_contact_kd", "soft_contact_mu"):
            self.assertEqual(getattr(original.model, name), getattr(plain.model, name))
        np.testing.assert_array_equal(
            original.model.tri_materials.numpy()[: original.paper_faces],
            plain.model.tri_materials.numpy()[: original.paper_faces],
        )
        for name in ("particle_q", "particle_mass", "particle_radius"):
            np.testing.assert_array_equal(
                getattr(original.model, name).numpy()[original.soft_cube_start :],
                getattr(plain.model, name).numpy()[plain.soft_cube_start :],
            )
        self.assertNotEqual(scene_signature(original), scene_signature(plain))

    def test_quest_exports_only_paper_shell_and_preserves_cube(self):
        """Transmit no handle mesh and retain independent deformable cube geometry."""
        for scene, count in zip(self.scenes, (2, 1), strict=True):
            scene._static_boxes, scene._bag_meshes = [], []
            scene._soft_cube_mesh = None
            payload = scene._build_webxr_geometry()
            header_size = struct.unpack_from("<I", payload, 4)[0]
            header = json.loads(payload[8 : 8 + header_size])
            self.assertEqual(len(scene._bag_meshes), count)
            self.assertEqual(sum(shape["role"] == "bag" for shape in header["shapes"]), count)
            self.assertEqual(sum(shape["role"] == "soft-cube" for shape in header["shapes"]), 1)
            self.assertEqual(sum(shape["role"] == "snack" for shape in header["shapes"]), 2)


if __name__ == "__main__":
    unittest.main()

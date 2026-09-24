# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check independent rope-handle geometry, attachment, layout, and Quest export."""

import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.examples.mjvbdv2 import example_mjvbd_v2_w1_bag_packing as packing
from newton.examples.mjvbdv2._webxr_teleop import JsonlTrajectoryRecorder
from newton.examples.mjvbdv2.example_mjvbd_v2_webxr_w1_bag_packing import Example as Original
from newton.examples.mjvbdv2.example_mjvbd_v2_webxr_w1_bag_packing_rope_handles import Example
from newton.examples.mjvbdv2.support.w1_bag_recording import (
    SCENE_OPTIONS,
    TeleopRecordingReader,
    load_replay_scene,
    scene_signature,
)
from newton.viewer import ViewerNull


@unittest.skipUnless(
    (Path.home() / "下载/scale_aligned_usd_minimal_20260918/toy/toy.usd").is_file(),
    "Requires the scale-aligned grocery USD bundle",
)
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

    def test_table_height_and_vertical_robot_adjustment(self):
        """Match the measured Blender tabletop while preserving upper-body XY and orientation."""
        original, rope = self.scenes
        delta = rope.table_z - packing.TABLE_Z
        self.assertAlmostEqual(rope.table_z, 0.9297508001327515)
        table = next(i for i, label in enumerate(rope.model.shape_label) if label == "Video worktable")
        top = rope.model.shape_transform.numpy()[table, 2] + rope.model.shape_scale.numpy()[table, 2]
        self.assertAlmostEqual(float(top), rope.table_z, places=6)
        for example in self.scenes:
            newton.eval_fk(example.model, example.model.joint_q, example.model.joint_qd, example.state_0)
        old, new = original.state_0.body_q.numpy(), rope.state_0.body_q.numpy()
        # The same arm coordinates must produce a pure vertical translation.
        for body in rope.ee:
            np.testing.assert_allclose(new[body, :3] - old[body, :3], (0, 0, delta), atol=2e-6)
            np.testing.assert_allclose(new[body, 3:], old[body, 3:], atol=2e-6)
        np.testing.assert_allclose(new[0], old[0], atol=1e-7)
        np.testing.assert_allclose(
            original.state_0.particle_q.numpy()[: original.paper_count] + np.array((0, 0.08, delta)),
            rope.state_0.particle_q.numpy()[: rope.paper_count],
            atol=1e-6,
        )
        materials = original.model.tri_materials.numpy()[: original.paper_faces].copy()
        materials[:, :2] *= 1.5
        np.testing.assert_array_equal(materials, rope.model.tri_materials.numpy()[: rope.paper_faces])

    def test_grocery_geometry_mass_and_recording_alignment(self):
        """Scale visuals, collisions and mass together while preserving USD alignment."""
        rope = self.scenes[1]
        self.assertEqual(rope.kinds, ("toy", "soda", "biscuit"))
        self.assertEqual(rope.soft_cube_start, rope.soft_cube_end)
        np.testing.assert_allclose(
            rope.model.body_mass.numpy()[rope.objects], np.array((0.045, 0.345, 0.045)) * 0.75**3, rtol=1e-5
        )
        np.testing.assert_allclose(
            rope.half * 2,
            np.array(
                (
                    (0.07541053, 0.07541053, 0.11936057),
                    (0.08468434, 0.08468434, 0.13388267),
                    (0.11388972, 0.09274235, 0.15),
                )
            )
            * 0.75,
            atol=1e-7,
        )
        np.testing.assert_allclose(rope.pick[:, 2] - rope.half[:, 2], rope.table_z + 0.002)
        self.assertFalse(np.any(rope.model.body_flags.numpy()[rope.objects] & int(newton.BodyFlags.KINEMATIC)))
        for i, asset in enumerate(rope.groceries):
            vertices = np.concatenate([part["mesh"].vertices for part in asset["parts"]])
            np.testing.assert_allclose(np.ptp(vertices, axis=0), 2 * rope.half[i], atol=1e-7)
            self.assertGreater(2 * rope._gripper_open_limit - 0.001402, min(2 * rope.half[i, :2]))
            self.assertLess(rope.snack_openings[asset["name"]], rope._gripper_open_limit)
            np.testing.assert_allclose(
                rope.pick[i, :2] - rope.half[i, :2] >= (packing.TABLE_CENTER - packing.TABLE_HALF)[:2], True
            )
        metadata = rope._recording_extras()
        self.assertEqual(len(metadata["groceryAssets"]), 3)
        for asset in metadata["groceryAssets"]:
            self.assertEqual(rope.model.body_label[asset["body"]], asset["label"])
            self.assertEqual(len(asset["sha256"]), 64)
            self.assertEqual(np.asarray(asset["sourceToBody"]).shape, (4, 4))
            self.assertEqual(asset["scale"], 0.75)
            np.testing.assert_allclose(np.asarray(asset["sourceToBody"])[:3, :3], np.eye(3) * 0.75)
        self.assertNotEqual(scene_signature(self.scenes[0]), scene_signature(rope))

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

    def test_replay_selects_grocery_geometry_without_teleoperation(self):
        """Restore a grocery recording through the shared entry without physics or servers."""
        rope = self.scenes[1]
        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cpu"):
            path = Path(directory) / "groceries.jsonl"
            recorder = JsonlTrajectoryRecorder(
                path,
                {
                    "scene": "w1-bag-packing",
                    "recordingKind": "full-state",
                    "frameDtSeconds": 1 / 60,
                    "sceneSignature": scene_signature(rope),
                    "sceneOptions": {name: getattr(rope.args, name) for name in SCENE_OPTIONS},
                    **rope._recording_extras(),
                },
            )
            recorder.start()
            recorder.append(
                {
                    "simulationTimeSeconds": 0,
                    **{
                        key: getattr(rope.state_0, field).numpy().tolist()
                        for field, key in TeleopRecordingReader._fields.items()
                    },
                }
            )
            recorder.close()
            args = packing.Example.create_parser().parse_args(["--replay", str(path)])
            # Recording choices must win even after the live demo's defaults change.
            with (
                patch.object(Example, "_grocery_names", ("toy", "soda", "glue")),
                patch.object(Example, "_grocery_scale", 1.0),
            ):
                rendered, playback = load_replay_scene(packing.Example, ViewerNull(), args)
            self.assertEqual(rendered.kinds, rope.kinds)
            np.testing.assert_allclose(rendered.half, rope.half)
            self.assertEqual(rendered.table_z, rope.table_z)
            self.assertFalse(hasattr(rendered, "solver"))
            self.assertFalse(hasattr(rendered, "webxr_server"))
            playback.recording.restore(rendered, 0)
            rendered.render()
            np.testing.assert_array_equal(rendered.state_0.body_q.numpy(), rope.state_0.body_q.numpy())


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify the independent pneumatic snack scene and its full-state recording."""

import copy
import json
import tempfile
import unittest
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from newton.examples.mjvbdv2 import example_mjvbd_v2_w1_bag_packing as packing
from newton.examples.mjvbdv2._webxr_parallel_gripper import ParallelGripperRetargeter
from newton.examples.mjvbdv2._webxr_teleop import JsonlTrajectoryRecorder
from newton.examples.mjvbdv2.example_mjvbd_v2_webxr_w1_bag_packing_rope_handles import Example as Groceries
from newton.examples.mjvbdv2.example_mjvbd_v2_webxr_w1_bag_packing_rope_shrimp import Example
from newton.examples.mjvbdv2.support.w1_bag_recording import (
    SCENE_OPTIONS,
    TeleopRecordingReader,
    load_replay_scene,
    scene_signature,
)
from newton.viewer import ViewerNull


@unittest.skipUnless(
    (Path.home() / "下载/oishi_softbag_usd_v3_20260922/oishi_shrimp_softbag/physics_mesh.npz").is_file()
    and (Path.home() / "下载/scale_aligned_usd_minimal_20260918/water_bottle/water_bottle.usd").is_file(),
    "Requires the local shrimp pouch and grocery USD bundles",
)
class TestRopeShrimpPacking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Build actual geometry without IK, physics solvers or network resources."""
        with wp.ScopedDevice("cpu"):
            cls.scene = Example.create_render_scene(ViewerNull(), Example.create_parser().parse_args([]))

    def test_preserve_three_grocery_defaults(self):
        """Keep separate assets, particle counts and network/recording identities."""
        old, new = Groceries.create_parser().parse_args([]), Example.create_parser().parse_args([])
        self.assertEqual(Groceries._grocery_names, ("toy", "soda", "biscuit"))
        self.assertEqual(old.snacks, 3)
        self.assertEqual(old.webxr_port, 8775)
        self.assertEqual((new.snacks, new.webxr_port), (1, 8776))
        self.assertNotEqual(Example.recording_prefix, Groceries.recording_prefix)
        self.assertEqual(self.scene.table_z, Groceries._table_height)
        self.assertEqual(self.scene.bag_particle_count, 1565)
        self.assertEqual(self.scene.kinds, ("water_bottle",))
        self.assertEqual(Groceries._paper_panel_bending_scale, 1.0)
        self.assertEqual(Groceries._paper_grasp_opening, packing.SUPPORT_OPENING)

    def test_free_pouch_mass_and_closed_pressure_subset(self):
        """Keep all film points free and exclude airless seals from the cavity."""
        s = self.scene
        mesh = s.shrimp.mesh
        self.assertEqual(s.soft_cube_end - s.soft_cube_start, 980)
        mass = s.model.particle_mass.numpy()[s.soft_cube_start : s.soft_cube_end]
        self.assertTrue(np.all(mass > 0))
        self.assertAlmostEqual(float(mass.sum()), 0.010, places=6)
        source = json.loads((s.args.shrimp_assets / "physics.json").read_text())
        self.assertEqual(source["mass_kg"], 0.025)
        self.assertEqual(s._recording_extras()["softBagAsset"]["massKg"], 0.010)
        self.assertEqual(s._recording_extras()["softBagAsset"]["materialModel"], "elastic")
        faces = mesh["triangles"][mesh["cavity_triangle_indices"]]
        self.assertEqual(len(faces), 1728)
        self.assertEqual(len(mesh["triangles"]), 1944)
        edges = np.sort(np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, 2)

    def test_both_grippers_close_on_thin_paper(self):
        """Select paper clearance instead of retaining the rigid snack opening."""
        s = copy.copy(self.scene)
        s.inputs = {}
        for side, body in zip(("left", "right"), s.ee, strict=True):
            mapper = ParallelGripperRetargeter(packing.ASSET, side=side)
            mapper.lower[:] = packing.SNACK_OPENINGS["can"]
            s.inputs[side] = SimpleNamespace(mapper=mapper, jaws=mapper.coordinates(1.0))
            poses = s.state_0.body_q.numpy()
            # Put the actual TCP on a paper rim vertex, away from both snacks.
            tcp = s.state_0.particle_q.numpy()[s.rim[0]]
            poses[body, :3] = tcp - np.asarray(wp.quat_rotate(wp.quat(*poses[body, 3:]), packing.TCP))
            s._update_grasp_limit(side, poses)
            self.assertEqual(getattr(s, f"_{side}_grasp_target"), "bag")
            np.testing.assert_allclose(s.inputs[side].jaws, 0.0015)

    def test_stiffen_panels_without_changing_folds_or_initial_geometry(self):
        """Keep authored bag positions, crease hinges and ropes while bracing panels."""
        s = self.scene
        with wp.ScopedDevice("cpu"):
            old = Groceries.create_render_scene(ViewerNull(), Groceries.create_parser().parse_args([]))
        np.testing.assert_array_equal(
            s.state_0.particle_q.numpy()[: s.bag_particle_count],
            old.state_0.particle_q.numpy()[: old.bag_particle_count],
        )
        count = len(old.model.edge_indices)
        np.testing.assert_array_equal(s.model.edge_indices.numpy()[:count], old.model.edge_indices.numpy())
        before = old.model.edge_bending_properties.numpy()
        after = s.model.edge_bending_properties.numpy()[:count]
        panels = np.isclose(before[:, 0], 60 * old._paper_stiffness_scale)
        self.assertTrue(panels.any())
        np.testing.assert_allclose(after[panels], before[panels] * (4, 2))
        np.testing.assert_array_equal(after[~panels], before[~panels])

    def test_only_jaw_closing_is_rate_limited(self):
        """Follow both arms immediately while cushioning closure of either gripper."""
        s = copy.copy(self.scene)
        s.arm_indices = np.array([s.coords[f"{side}_J{i}"] for side in ("LEFT", "RIGHT") for i in range(1, 8)])
        s.finger_indices = {
            side: np.array([s.coords[f"{side.upper()}_FINGER{i}_JOINT"] for i in (1, 2)]) for side in ("left", "right")
        }
        previous = s.state_0.joint_q.numpy()
        for indices in s.finger_indices.values():
            previous[indices] = 0.04
        solved = previous + 0.2
        s.inputs = {side: SimpleNamespace(jaws=np.full(2, 0.0015)) for side in ("left", "right")}
        result = s._limit_joint_targets(previous, solved)
        np.testing.assert_array_equal(result[s.arm_indices], solved[s.arm_indices])
        for indices in s.finger_indices.values():
            np.testing.assert_allclose(result[indices], 0.04 - s.args.gripper_speed * s.frame_dt)
        untouched = sorted(
            set(range(len(previous))) - set(s.arm_indices) - set(np.concatenate(list(s.finger_indices.values())))
        )
        np.testing.assert_array_equal(result[untouched], previous[untouched])
        # Opening either hand is unrestricted, and a small close must not overshoot.
        s.inputs["right"].jaws[:] = 0.05
        s.inputs["left"].jaws[:] = 0.05
        result = s._limit_joint_targets(previous, solved)
        for side in ("left", "right"):
            np.testing.assert_allclose(result[s.finger_indices[side]], 0.05)
        for side in ("left", "right"):
            s.inputs[side].jaws[:] = 0.0399
            np.testing.assert_allclose(s._limit_joint_targets(previous, solved)[s.finger_indices[side]], 0.0399)

    def test_realtime_solver_budget_keeps_contact_rules(self):
        """Reduce solve work explicitly while retaining paper and pouch contact settings."""
        s = copy.copy(self.scene)
        reference = s._vbd_options()
        s.args = Example.create_parser().parse_args(
            ["--substeps", "4", "--iterations", "12", "--coarse-iterations", "8", "--coarse-passes", "2"]
        )
        fast = s._vbd_options()
        changed = {"iterations", "particle_multilevel_coarse_iterations", "particle_multilevel_checkpoints"}
        for key in reference.keys() - changed:
            self.assertEqual(fast[key], reference[key], key)
        self.assertEqual(s.args.substeps * fast["iterations"], 48)
        self.assertEqual(fast["particle_multilevel_checkpoints"], (6, 10))
        self.assertEqual(fast["particle_multilevel_coarse_iterations"], 8)

    def test_json_encoder_preserves_precision_and_fallback(self):
        """Keep JSON values exact with the optional encoder and standard fallback."""
        payload = {
            "title": "虾片包",
            "particleQ": self.scene.state_0.particle_q.numpy().tolist(),
            "numbers": [0.0, -0.0, 1e-30, 1e30, 1.0000000000000002],
            "tracking": True,
            "target": None,
        }
        self.assertEqual(json.loads(Example._json_dumps(payload)), payload)
        with patch("newton.examples.mjvbdv2.example_mjvbd_v2_webxr_w1_bag_packing_rope_shrimp._encode_json", None):
            self.assertEqual(json.loads(Example._json_dumps(payload)), payload)

    def test_visible_bottle_includes_transparent_source_shell(self):
        """The untextured view must show a full bottle instead of only its label."""
        asset = self.scene.groceries[0]
        visible = [p["mesh"].vertices for p in asset["parts"] if not p["collision"] and p["opacity"] >= 0.5]
        extent = np.ptp(np.concatenate(visible), axis=0)
        np.testing.assert_allclose(extent, 2 * asset["half"], atol=0.001)
        self.assertAlmostEqual(asset["source"]["scale"], 0.75)

    @unittest.skipUnless(wp.is_cuda_available(), "Pneumatic color coupling requires CUDA")
    def test_paper_startup_matches_three_grocery_scene(self):
        """The pouch must not make the original folded paper repel or open itself."""
        samples = []
        with wp.ScopedDevice("cuda:0"):
            for scene_type in (Groceries, Example):
                s = scene_type(ViewerNull(), scene_type.create_parser().parse_args(["--no-webxr-server"]))
                try:
                    frames = {0: s.state_0.particle_q.numpy()[: s.bag_particle_count].copy()}
                    for frame in range(1, 181):
                        s.step()
                        if frame in (30, 90, 180):
                            frames[frame] = s.state_0.particle_q.numpy()[: s.bag_particle_count].copy()
                    samples.append(frames)
                finally:
                    s.close()
        np.testing.assert_array_equal(samples[0][0], samples[1][0])
        for frame in (30, 90, 180):
            # Compare the paper panels, not freely swinging rope tips.
            paper = slice(0, self.scene.paper_count)
            displacement = np.linalg.norm(samples[0][frame][paper] - samples[1][frame][paper], axis=1)
            self.assertLess(float(np.sqrt(np.mean(displacement**2))), 0.004, f"Paper changed at frame {frame}")
            self.assertLess(float(displacement.max()), 0.012, f"Paper fold opened at frame {frame}")

    @unittest.skipUnless(wp.is_cuda_available(), "Pneumatic color coupling requires CUDA")
    def test_default_solver_preserves_pouch_during_pick_and_release(self):
        """Reduced solve work must retain pressure, grip and table contact."""
        with wp.ScopedDevice("cuda:0"):
            s = Example(ViewerNull(), Example.create_parser().parse_args(["--no-webxr-server"]))
            try:
                rest_angles = s.model.edge_rest_angle.numpy().copy()
                bending_properties = s.model.edge_bending_properties.numpy().copy()
                # Feed a reproducible hand target through the actual IK and
                # fingers; the pouch has no attachment or prescribed motion.
                s._prepare_frame = lambda: None
                right = s.inputs["right"]
                retargeter = right.retargeter
                right.retargeter = SimpleNamespace(active=True)
                home, rotation = right.position.copy(), wp.quat(*right.orientation)
                target_rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), np.pi / 2) * wp.quat_from_axis_angle(
                    wp.vec3(0, 0, 1), np.pi / 2
                )
                z = s.table_z + 0.012
                closed_gap = s._deformable_grasp_opening
                waypoints = [
                    (0, home, 0.035),
                    (30, home, 0.035),
                    (90, (0.28, -0.19, z), 0.035),
                    (150, (0.346, -0.19, z), 0.035),
                    (190, (0.346, -0.19, z), closed_gap),
                    (230, (0.346, -0.19, z + 0.12), closed_gap),
                    (350, (0.346, -0.19, z + 0.12), closed_gap),
                    (410, (0.48, -0.19, z + 0.12), closed_gap),
                    (445, (0.48, -0.19, z + 0.12), 0.035),
                    (475, (0.48, -0.19, z + 0.26), 0.035),
                    (600, (0.48, -0.19, z + 0.26), 0.035),
                ]
                mesh = s.shrimp.mesh
                faces = mesh["triangles"][mesh["cavity_triangle_indices"]]
                edges = np.concatenate(
                    [mesh["triangles"][:, [0, 1]], mesh["triangles"][:, [1, 2]], mesh["triangles"][:, [2, 0]]]
                )
                rest_lengths = np.linalg.norm(mesh["points"][edges[:, 0]] - mesh["points"][edges[:, 1]], axis=1)
                released_volumes = []
                for frame in range(601):
                    for (a, p, gap), (b, end, end_gap) in pairwise(waypoints):
                        if a <= frame <= b:
                            t = (frame - a) / (b - a)
                            t = t * t * (3 - 2 * t)
                            right.position = (1 - t) * np.asarray(p) + t * np.asarray(end)
                            right.jaws[:] = (1 - t) * gap + t * end_gap
                            break
                    right.orientation = np.asarray(wp.quat_slerp(rotation, target_rotation, min(1.0, frame / 90)))
                    s.step()
                    s.test_post_step()
                    q = s.state_0.particle_q.numpy()[s.soft_cube_start : s.soft_cube_end]
                    tri = (q - q.mean(axis=0))[faces]
                    volume = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
                    # Pinching and landing compress the cavity; allow variation
                    # between nondeterministic CUDA solves, but reject collapse.
                    # The shared paper contact law keeps the initial cavity
                    # within 15% of authored volume before pinching.
                    volume_floor = 0.85 if frame < 150 else 0.60
                    self.assertGreater(volume / s.shrimp.config["rest_volume_m3"], volume_floor, f"frame={frame}")
                    if frame >= 540:
                        released_volumes.append(volume / s.shrimp.config["rest_volume_m3"])
                    stretch = np.linalg.norm(q[edges[:, 0]] - q[edges[:, 1]], axis=1) / rest_lengths
                    self.assertLess(float(stretch.max()), 1.6)
                    if 240 <= frame <= 410:
                        self.assertGreater(float(q[:, 2].min()), s.table_z + 0.025, "Pouch slipped during transport")
                self.assertGreater(float(q[:, 2].min()), s.table_z - 0.006)
                self.assertLess(float(q[:, 2].mean()), s.table_z + 0.03, "Released pouch failed to settle")
                # Average the last second: release excites small volume
                # oscillations can persist after the pinch.
                self.assertGreater(float(np.mean(released_volumes)), 0.80, "Released pouch stayed deflated")
                np.testing.assert_array_equal(s.model.edge_rest_angle.numpy(), rest_angles)
                np.testing.assert_array_equal(s.model.edge_bending_properties.numpy(), bending_properties)
                right.retargeter = retargeter
                s.reset_physics(source="test")
                np.testing.assert_array_equal(s.state_0.particle_q.numpy(), s._initial_state.particle_q.numpy())
                np.testing.assert_array_equal(s.model.edge_rest_angle.numpy(), rest_angles)
                np.testing.assert_array_equal(s.model.edge_bending_properties.numpy(), bending_properties)
            finally:
                s.close()

    def test_authored_render_binding_follows_world_translation(self):
        """Carry all detailed pouch vertices with its recorded particle positions."""
        s = self.scene
        particles = s.state_0.particle_q.numpy()
        source = s.shrimp.bind_render(particles)
        self.assertEqual(source.shape, (23572, 3))
        offset = np.array((0.1, -0.2, 0.3), np.float32)
        moved = s.shrimp.bind_render(particles + offset)
        np.testing.assert_allclose(moved, source + offset, atol=5e-7)
        np.testing.assert_allclose(np.linalg.norm(s.shrimp.bind_normals(particles), axis=1), 1, atol=2e-6)

    def test_record_and_restore_all_pouch_particles(self):
        """Replay a deformed pouch through the existing entry with no live simulation."""
        s = self.scene
        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cpu"):
            path = Path(directory) / "shrimp.jsonl"
            recorder = JsonlTrajectoryRecorder(
                path,
                {
                    "scene": "w1-bag-packing",
                    "recordingKind": "full-state",
                    "frameDtSeconds": 1 / 60,
                    "sceneSignature": scene_signature(s),
                    "sceneOptions": {name: getattr(s.args, name) for name in SCENE_OPTIONS},
                    **s._recording_extras(),
                },
                json_dumps=s._json_dumps,
            )
            particle_q = s.state_0.particle_q.numpy()
            particle_q[s.soft_cube_start :, 2] += np.linspace(0, 0.01, 980)
            sample = {
                key: getattr(s.state_0, field).numpy().tolist() for field, key in TeleopRecordingReader._fields.items()
            }
            sample.update(
                simulationTimeSeconds=1.0,
                bagParticleQ=particle_q[: s.bag_particle_count].tolist(),
                bagParticleQd=np.zeros((s.bag_particle_count, 3)).tolist(),
                softCubeParticleQ=particle_q[s.soft_cube_start :].tolist(),
                softCubeParticleQd=np.zeros((980, 3)).tolist(),
            )
            recorder.start()
            recorder.append(sample)
            recorder.close()
            args = packing.Example.create_parser().parse_args(["--replay", str(path)])
            rendered, playback = load_replay_scene(packing.Example, ViewerNull(), args)
            self.assertFalse(hasattr(rendered, "solver"))
            self.assertFalse(hasattr(rendered, "webxr_server"))
            playback.recording.restore(rendered, 0)
            np.testing.assert_array_equal(rendered.state_0.particle_q.numpy(), particle_q)
            self.assertEqual(playback.recording.metadata["soft_bag_asset"]["kind"], "oishi-shrimp-v3")
            self.assertAlmostEqual(float(rendered.model.particle_mass.numpy()[s.soft_cube_start :].sum()), 0.010)
            lines = path.read_text().splitlines()
            header = json.loads(lines[0])
            del header["softBagAsset"]["massKg"]
            path.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
            legacy, _ = load_replay_scene(packing.Example, ViewerNull(), args)
            self.assertAlmostEqual(float(legacy.model.particle_mass.numpy()[s.soft_cube_start :].sum()), 0.025)


if __name__ == "__main__":
    unittest.main()

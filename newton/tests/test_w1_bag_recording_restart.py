# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check W1 retakes with real file writes and input handling, without a GPU solver."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import warp as wp

from newton.examples.mjvbdv2._webxr_teleop import JsonlTrajectoryRecorder
from newton.examples.mjvbdv2.example_mjvbd_v2_webxr_w1_bag_packing import Example
from newton.examples.mjvbdv2.example_mjvbd_v2_webxr_w1_bag_packing_no_handles import Example as HandleFree
from newton.tests.test_webxr_w1_bag_packing import frame


class TestPackingRetakes(unittest.TestCase):
    def make_scene(self, scene_type, path, mode):
        """Keep the real recorder and control path while stubbing physics and IK."""
        scene = scene_type.__new__(scene_type)
        self.recorder = scene.trajectory_recorder = JsonlTrajectoryRecorder(
            path, {"sceneOptions": {"bag_variant": scene_type.create_parser().parse_args([]).bag_variant}}
        )
        self.addCleanup(self.recorder.close)
        scene.args = SimpleNamespace(xr_stale_seconds=1, record_on_connect=False)
        scene.teleoperation_active = True
        scene.xr_state = Mock()
        scene._buttons = scene._record_request = None
        scene._seen_input = False
        scene.state_0, scene.state_1, scene._initial_state, scene.solver = Mock(), Mock(), Mock(), Mock()
        scene.state_0.body_q.numpy.return_value = np.zeros((2, 7), dtype=np.float32)
        scene.state_0.joint_q.numpy.return_value = np.zeros(4, dtype=np.float32)
        scene.ik_q = wp.zeros((1, 4), device="cpu")
        scene._initial_ik = wp.zeros((1, 4), device="cpu")
        scene.frame_start, scene.frame_end = wp.zeros(4, device="cpu"), wp.zeros(4, device="cpu")
        scene.ee = [0, 1]
        scene.finger_indices = {"left": [0, 1], "right": [2, 3]}
        scene.inputs = {
            side: SimpleNamespace(
                mapper=SimpleNamespace(lower=np.zeros(2)), update=Mock(), position=np.zeros(3), status="tracking"
            )
            for side in ("left", "right")
        }
        scene._hold_inputs, scene._publish_scene_state = Mock(), Mock()
        scene._update_head_gaze, scene._update_grasp_limit, scene._tcp_pose = Mock(), Mock(), Mock()
        scene.head_control = Mock()
        scene.episode_index = scene.episode_frame = scene.frame = 0
        self.scene, self.mode, self.request = scene, mode, 0
        self.value = frame(mode=mode)
        scene.view_mode = self.value.view_mode
        self.send()
        return scene

    def send(self, pressed=False):
        """Deliver controller A edges or optical-panel record requests to the actual scene."""
        if self.mode == "controllers":
            right = replace(
                self.value.controllers["left"], handedness="right", button_pressed=(False, False, False, False, pressed)
            )
            value = replace(self.value, controllers={"right": right})
        else:
            if pressed:
                self.request += 1
            value = replace(self.value, controllers={}, recording_request=self.request)
        self.scene.xr_state.snapshot.return_value = value
        self.scene._prepare_frame()

    def toggle(self):
        """Press and release a recording control once."""
        self.send(True)
        self.send(False)

    def test_paused_reset_replaces_all_frames_on_next_record(self):
        """Retain failures until restart, then truncate all old frames and reset the sample count."""
        for scene_type in (Example, HandleFree):
            for mode in ("controllers", "hands"):
                with self.subTest(scene=scene_type.__module__, mode=mode), tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "take.jsonl"
                    scene = self.make_scene(scene_type, path, mode)
                    self.toggle()
                    self.recorder.append({"frame": 1, "payload": "failed" * 1000})
                    self.toggle()
                    scene.reset_physics(source="test")
                    scene.reset_physics(source="test")
                    self.assertFalse(self.recorder.recording)
                    self.assertIn("failed", path.read_text())
                    self.toggle()
                    self.assertEqual(self.recorder.sample_count, 0)
                    self.recorder.append({"frame": 4, "episode": scene.episode_index})
                    self.toggle()
                    records = [json.loads(line) for line in path.read_text().splitlines()]
                    self.assertEqual(len(records), 2)
                    self.assertEqual(records[0]["sceneOptions"], self.recorder.metadata["sceneOptions"])
                    self.assertEqual(records[1], {"type": "frame", "frame": 4, "episode": 2})
                    self.assertEqual(self.recorder.sample_count, 1)
                    # The discard request is consumed: ordinary resume now appends.
                    self.toggle()
                    self.recorder.append({"frame": 5})
                    self.toggle()
                    self.assertEqual(len(path.read_text().splitlines()), 3)
                    self.recorder.close()

    def test_active_reset_and_plain_pause_keep_existing_frames(self):
        """Preserve append behavior without a paused reset, including other recorder users."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "take.jsonl"
            scene = self.make_scene(Example, path, "controllers")
            self.toggle()
            self.recorder.append({"frame": 1})
            scene.reset_physics(source="test")
            self.assertTrue(self.recorder.recording)
            self.toggle()
            self.toggle()
            self.recorder.append({"frame": 2})
            self.toggle()
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([item["frame"] for item in records if item["type"] == "frame"], [1, 2])
            self.assertEqual(self.recorder.sample_count, 2)


if __name__ == "__main__":
    unittest.main()

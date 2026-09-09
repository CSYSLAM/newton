# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise optical hand protocol, mechanical mapping and loss/re-arm behavior."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from newton.examples.mjvbdv2 import (
    example_cloth_mjvbd_v2_dexforce_webxr_bimanual_fold_tshirt_waic_house_final00 as tshirt,
)
from newton.examples.mjvbdv2 import example_mjvbd_v2_dexforce_webxr_plug_socket as plug
from newton.examples.mjvbdv2 import (
    example_vbd_mjvbd_v2_dexforce_webxr_plastic_inflatable_bag_pick_release_final00 as bag,
)
from newton.examples.mjvbdv2._webxr_teleop import ProtocolError, RelativePoseRetargeter, XRFrame
from newton.examples.mjvbdv2._webxr_w1_hand import W1HandRetargeter

_ROOT = Path(__file__).resolve().parents[2]
_URDF = _ROOT / "assets/DexforceW1V021/DexforceW1V021.urdf"


def skeleton(*, curl=0.0, side="right"):
    """Construct an anatomical-sized hand with independently adjustable fingers."""
    result = np.zeros((25, 3))
    result[1:5] = [[0.022, 0.015, 0], [0.041, 0.033, 0], [0.061, 0.048, 0], [0.080, 0.063, 0]]
    curls = np.broadcast_to(curl, (4,))
    for finger, (base, x) in enumerate(zip((5, 10, 15, 20), (0.03, 0.01, -0.01, -0.03), strict=True)):
        result[base] = [x * 0.5, 0.025, 0]
        result[base + 1] = [x, 0.075, 0]
        for joint, length in enumerate((0.035, 0.025, 0.020)):
            angle = curls[finger] * (joint + 1)
            result[base + joint + 2] = result[base + joint + 1] + [0, length * np.cos(angle), length * np.sin(angle)]
    if side == "left":
        result[:, 0] *= -1
    return result


def payload(*, activation=1, enabled=True, sequence=0, position=(0, 0, 0)):
    return {
        "type": "xr-frame",
        "version": 1,
        "streamId": "hands-test",
        "sequence": sequence,
        "timeMs": sequence * 14,
        "referenceSpace": "local-floor",
        "controllerSpace": "newton-world",
        "visibilityState": "visible",
        "inputMode": "hands",
        "hands": {
            "right": {
                "pose": {"position": list(position), "orientation": [0, 0, 0, 1]},
                "joints": skeleton().tolist(),
                "enabled": enabled,
                "activation": activation,
            }
        },
    }


class TestWebXRHandProtocol(unittest.TestCase):
    def test_accept_optical_hand_without_controller(self):
        """Decode hands without synthesizing any gamepad input."""
        frame = XRFrame.from_json(json.dumps(payload()))
        self.assertEqual(frame.controllers, {})
        self.assertEqual(frame.hands["right"].joints.shape, (25, 3))
        self.assertTrue(frame.hands["right"].enabled)

    def test_preserve_controller_only_frame(self):
        """Keep legacy frames on the controller path."""
        value = payload()
        del value["hands"], value["inputMode"]
        frame = XRFrame.from_mapping(value)
        self.assertEqual(frame.input_mode, "controllers")
        self.assertEqual(frame.hands, {})

    def test_reject_bad_skeleton_and_activation(self):
        """Reject malformed or non-finite optical data before simulation."""
        for key, value in (
            ("joints", [[0, 0, 0]] * 24),
            ("joints", [[0, float("nan"), 0]] * 25),
            ("enabled", "true"),
            ("activation", True),
            ("activation", -1),
        ):
            with self.subTest(key=key, value=str(value)[:40]):
                message = payload()
                message["hands"]["right"][key] = value
                with self.assertRaises(ProtocolError):
                    XRFrame.from_mapping(message)


class TestW1HandRetargeting(unittest.TestCase):
    def test_respect_limits_and_mimic_on_both_hands(self):
        """Keep all fitted coordinates inside the URDF mechanical constraints."""
        for side in ("left", "right"):
            mapper = W1HandRetargeter(_URDF, side)
            for curl in (0.0, 0.4, 0.8, 1.2):
                q = mapper.solve(skeleton(curl=curl, side=side))
                self.assertTrue(np.all(q >= mapper.lower - 1e-6))
                self.assertTrue(np.all(q <= mapper.upper + 1e-6))
                np.testing.assert_allclose(q[3::2], mapper.multipliers * q[2::2] + mapper.offsets, atol=1e-6)

    def test_drive_fingers_independently(self):
        """Bend the index while keeping the other fingers substantially open."""
        for side in ("left", "right"):
            mapper = W1HandRetargeter(_URDF, side)
            for _ in range(3):
                opened = mapper.solve(skeleton(side=side))
            for _ in range(3):
                bent = mapper.solve(skeleton(curl=[0.8, 0, 0, 0], side=side))
            self.assertGreater(bent[2] - opened[2], 0.35)
            np.testing.assert_allclose(bent[[4, 6, 8]], opened[[4, 6, 8]], atol=0.12)

    def test_ignore_global_pose(self):
        """Retarget the same shape identically after rigid world motion."""
        points = skeleton(curl=0.6)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        first = W1HandRetargeter(_URDF, "right").solve(points)
        second = W1HandRetargeter(_URDF, "right").solve(points @ rotation.T + [2, -3, 1])
        np.testing.assert_allclose(first, second, atol=1e-5)

    def test_close_thumb_index_pinch(self):
        """Bring the robot fingertips together for an anatomical pinch input."""
        for side in ("left", "right"):
            mapper = W1HandRetargeter(_URDF, side)
            points = skeleton(curl=[0.65, 0, 0, 0], side=side)
            points[4] = points[9] + [0.005, 0, 0]
            points[2] = points[1] + (points[4] - points[1]) / 3
            points[3] = points[1] + (points[4] - points[1]) * 2 / 3
            for _ in range(6):
                mapper.solve(points)
            tips = mapper.fingertips(mapper.q)
            self.assertLess(np.linalg.norm(tips[0] - tips[1]), 0.015)

    def test_reject_degenerate_tracking(self):
        """Refuse a collapsed skeleton instead of producing arbitrary targets."""
        with self.assertRaises(ValueError):
            W1HandRetargeter(_URDF, "right").solve(np.zeros((25, 3)))


class TestOpticalHandEngagement(unittest.TestCase):
    def setUp(self):
        """Build the optical control path without a physics device or server."""
        example = tshirt.Example.__new__(tshirt.Example)
        self.example = example
        self.current = np.zeros(20, dtype=np.float32)
        example.state_0 = SimpleNamespace(joint_q=SimpleNamespace(numpy=self.current.copy))
        example.retargeters = {side: RelativePoseRetargeter() for side in tshirt.HANDS}
        example._optical_retargeters = {side: W1HandRetargeter(_URDF, side) for side in tshirt.HANDS}
        for field in ("targets", "tokens", "blocked", "sequences", "last_poses"):
            setattr(example, f"_optical_{field}", dict.fromkeys(tshirt.HANDS))
        example._optical_status = dict.fromkeys(tshirt.HANDS, "paused")
        example._hand_indices_by_side = {
            "left": SimpleNamespace(numpy=lambda: np.arange(10)),
            "right": SimpleNamespace(numpy=lambda: np.arange(10, 20)),
        }
        example._teleop_positions = {side: np.array([0.5, 0.5, 1.0]) for side in tshirt.HANDS}
        example._teleop_orientations = {side: np.array([0.0, 0, 0, 1]) for side in tshirt.HANDS}
        example._teleop_grasps = dict.fromkeys(tshirt.HANDS, 0.0)
        example._target_position_min = np.array([-2.0, -2, -2])
        example._target_position_max = np.array([2.0, 2, 2])
        example._input_mode = "hands"

    def test_loss_holds_current_pose_and_requires_new_activation(self):
        """Hold on loss and prevent the old activation from re-engaging."""
        example = self.example
        example._prepare_optical_hands(XRFrame.from_mapping(payload()))
        self.current[10:] = 0.25
        example._prepare_optical_hands(None)
        np.testing.assert_allclose(example._optical_targets["right"], 0.25)
        before = example._teleop_positions["right"].copy()
        example._prepare_optical_hands(XRFrame.from_mapping(payload(sequence=1, position=(0.4, 0, 0))))
        np.testing.assert_allclose(example._teleop_positions["right"], before)
        self.assertFalse(example.retargeters["right"].active)
        example._prepare_optical_hands(XRFrame.from_mapping(payload(activation=2, sequence=2, position=(0.4, 0, 0))))
        np.testing.assert_allclose(example._teleop_positions["right"], before)
        example._prepare_optical_hands(XRFrame.from_mapping(payload(activation=2, sequence=3, position=(0.42, 0, 0))))
        np.testing.assert_allclose(example._teleop_positions["right"], before + np.array([0.02, 0, 0]), atol=1e-6)

    def test_pause_and_jump_do_not_open_fingers(self):
        """Pause on deliberate disengagement and implausible wrist jumps."""
        example = self.example
        example._prepare_optical_hands(XRFrame.from_mapping(payload()))
        self.current[10:] = 0.3
        example._prepare_optical_hands(XRFrame.from_mapping(payload(sequence=1, position=(0.5, 0, 0))))
        self.assertEqual(example._optical_status["right"], "invalid-tracking")
        np.testing.assert_allclose(example._optical_targets["right"], 0.3)
        example._prepare_optical_hands(XRFrame.from_mapping(payload(sequence=2, enabled=False)))
        np.testing.assert_allclose(example._optical_targets["right"], 0.3)

    def test_duplicate_frame_does_not_resolve_fingers(self):
        """Reuse the latest mapping when simulation outpaces the XR stream."""
        frame = XRFrame.from_mapping(payload())
        self.example._prepare_optical_hands(frame)
        before = self.example._optical_targets["right"].copy()
        self.example._prepare_optical_hands(frame)
        np.testing.assert_array_equal(before, self.example._optical_targets["right"])

    def test_clutch_resume_reanchors_after_hand_reposition(self):
        """Resume from the held robot pose after a large real-hand reposition."""
        example = self.example
        example._prepare_optical_hands(XRFrame.from_mapping(payload()))
        example._prepare_optical_hands(XRFrame.from_mapping(payload(sequence=1, position=(0.1, 0, 0))))
        held = example._teleop_positions["right"].copy()
        example._prepare_optical_hands(XRFrame.from_mapping(payload(sequence=2, enabled=False, position=(-0.5, 0, 0))))
        example._prepare_optical_hands(XRFrame.from_mapping(payload(activation=2, sequence=3, position=(-0.5, 0, 0))))
        np.testing.assert_allclose(example._teleop_positions["right"], held)
        example._prepare_optical_hands(XRFrame.from_mapping(payload(activation=2, sequence=4, position=(-0.48, 0, 0))))
        np.testing.assert_allclose(example._teleop_positions["right"], held + np.array([0.02, 0, 0]), atol=1e-6)


class TestSingleHandOpticalTeleop(unittest.TestCase):
    def setUp(self):
        """Build both scene input paths without constructing their physics or servers."""
        self.examples = []
        for scene in (plug, bag):
            example = scene.Example.__new__(scene.Example)
            current = np.zeros(10, dtype=np.float32)
            example.state_0 = SimpleNamespace(joint_q=SimpleNamespace(numpy=current.copy))
            example.retargeter = RelativePoseRetargeter()
            urdf_path = plug.plug_socket.DEFAULT_ROBOT_URDF if scene is plug else _URDF
            example._optical_mapper = W1HandRetargeter(urdf_path, "right")
            example._optical_indices_host = np.arange(10)
            for field in ("target", "token", "blocked", "sequence", "last_pose"):
                setattr(example, f"_optical_{field}", None)
            example._optical_status = "paused"
            example._input_mode = "controllers"
            example._recording_request = None
            example._teleop_position = np.array([0.5, 0.5, 1.0])
            example._teleop_orientation = np.array([0.0, 0, 0, 1])
            example._teleop_grasp = 0.0
            example.teleoperation_active = True
            example._last_input_stream = None
            example._last_input_sequence = None
            example._record_button_pressed = False
            example._reset_button_pressed = False
            example._has_seen_controller = False
            example.args = SimpleNamespace(record_on_connect=False)
            example.trajectory_recorder = Mock()
            self.examples.append((scene.__name__, example, current))
        self.lower = np.full(3, -2.0)
        self.upper = np.full(3, 2.0)

    def _prepare(self, example, value):
        return example._prepare_teleop_input(
            None if value is None else XRFrame.from_mapping(value), self.lower, self.upper
        )

    def test_follow_hold_and_reanchor_in_both_scenes(self):
        """Follow right-hand input and resume from the held target after a tracking gap."""
        for name, example, current in self.examples:
            with self.subTest(scene=name):
                self._prepare(example, payload())
                self.assertEqual(example._optical_status, "tracking")
                self.assertEqual(example._optical_target.shape, (10,))
                self._prepare(example, payload(sequence=1, position=(0.02, 0, 0)))
                held = example._teleop_position.copy()
                np.testing.assert_allclose(held, [0.52, 0.5, 1.0])
                current[:] = 0.25
                self._prepare(example, None)
                np.testing.assert_allclose(example._optical_target, current)
                self._prepare(example, payload(sequence=2, position=(-0.5, 0, 0)))
                np.testing.assert_allclose(example._teleop_position, held)
                self._prepare(example, payload(activation=2, sequence=3, position=(-0.5, 0, 0)))
                np.testing.assert_allclose(example._teleop_position, held)
                self._prepare(example, payload(activation=2, sequence=4, position=(-0.48, 0, 0)))
                np.testing.assert_allclose(example._teleop_position, held + np.array([0.02, 0, 0]))
                state = example._optical_scene_state()
                self.assertEqual(state["handTrackingSides"], ["right"])
                self.assertEqual(state["handTrackingActivation"]["right"], 2)

    def test_clutch_invalid_tracking_and_hidden_session_hold(self):
        """Hold fingers and wrist through gaze clutching, invalid bones and lost visibility."""
        for name, example, current in self.examples:
            with self.subTest(scene=name):
                self._prepare(example, payload())
                held = example._teleop_position.copy()
                current[:] = 0.3
                self._prepare(example, payload(sequence=1, enabled=False, position=(-0.5, 0, 0)))
                np.testing.assert_allclose(example._optical_target, current)
                self._prepare(example, payload(activation=2, sequence=2, position=(-0.5, 0, 0)))
                np.testing.assert_allclose(example._teleop_position, held)
                invalid = payload(activation=2, sequence=3, position=(-0.5, 0, 0))
                invalid["hands"]["right"]["joints"] = np.zeros((25, 3)).tolist()
                self._prepare(example, invalid)
                self.assertEqual(example._optical_status, "invalid-tracking")
                np.testing.assert_allclose(example._optical_target, current)
                self._prepare(example, payload(activation=3, sequence=4))
                self.assertEqual(example._optical_status, "tracking")
                hidden = payload(activation=3, sequence=5, position=(0.05, 0, 0))
                hidden["visibilityState"] = "hidden"
                self._prepare(example, hidden)
                np.testing.assert_allclose(example._teleop_position, held)
                np.testing.assert_allclose(example._optical_target, current)

    def test_controller_shortcuts_do_not_override_optical_motion(self):
        """Retain record/reset edges and controller mode while separating optical finger targets."""
        for name, example, _ in self.examples:
            with self.subTest(scene=name):
                example._optical_mapper.solve = Mock(
                    return_value=np.array([0.2, 0.1, 0.3, 0.2, 0.4, 0.3, 0.5, 0.4, 0.6, 0.5])
                )
                value = payload()
                value["controllers"] = {
                    "right": {
                        "pose": {"position": [1, 1, 1], "orientation": [0, 0, 0, 1]},
                        "clutch": True,
                        "triggerValue": 1.0,
                        "buttons": [{"pressed": False}] * 5,
                        "axes": [],
                    }
                }
                self._prepare(example, value)
                fingers = example._optical_target.copy()
                value["sequence"] = 1
                value["controllers"]["right"]["buttons"][4] = {"pressed": True}
                self._prepare(example, value)
                example.trajectory_recorder.toggle.assert_called_once()
                np.testing.assert_allclose(example._teleop_position, [0.5, 0.5, 1.0])
                np.testing.assert_allclose(example._optical_target, fingers)
                self.assertLess(example._teleop_grasp, 1.0)
                # Re-reading one XR packet must not re-trigger a shortcut.
                self._prepare(example, value)
                example.trajectory_recorder.toggle.assert_called_once()
                value["sequence"] = 2
                value["inputMode"] = "controllers"
                self._prepare(example, value)
                self.assertIsNone(example._optical_target)
                self.assertEqual(example._teleop_grasp, 1.0)
                example.reset_physics = Mock()
                value["sequence"] = 3
                value["inputMode"] = "hands"
                value["controllers"]["right"]["buttons"][3] = {"pressed": True}
                self.assertTrue(self._prepare(example, value))
                example.reset_physics.assert_called_once_with(source="quest-controller")

    def test_left_hand_does_not_drive_right_robot_and_panel_records_once(self):
        """Ignore left-only tracking and deduplicate the optical panel recording request."""
        for name, example, _ in self.examples:
            with self.subTest(scene=name):
                value = payload()
                value["hands"]["left"] = value["hands"].pop("right")
                value["recordingRequest"] = 1
                self._prepare(example, value)
                self._prepare(example, value)
                self.assertEqual(example._optical_status, "tracking-lost")
                np.testing.assert_allclose(example._teleop_position, [0.5, 0.5, 1.0])
                example.trajectory_recorder.toggle.assert_called_once()
                record = example._optical_record(XRFrame.from_mapping(value))
                self.assertIn("left", record["xrHands"])
                self.assertEqual(len(record["handJointTargets"]["right"]), 10)


if __name__ == "__main__":
    unittest.main()

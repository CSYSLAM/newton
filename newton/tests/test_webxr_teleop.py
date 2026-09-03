# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the example-local WebXR teleoperation transport and mapping."""

import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

from newton.examples.mjvbdv2 import (
    example_cloth_mjvbd_v2_dexforce_webxr_bimanual_fold_tshirt_waic_house_final00 as webxr_tshirt_example,
)
from newton.examples.mjvbdv2 import example_mjvbd_v2_dexforce_webxr_plug_socket as webxr_example
from newton.examples.mjvbdv2 import example_mjvbd_v2_dexforce_webxr_push_chair as webxr_chair_example
from newton.examples.mjvbdv2 import example_mjvbd_v2_webxr_bimanual_nut_bolt as webxr_nut_bolt_example
from newton.examples.mjvbdv2 import (
    example_vbd_mjvbd_v2_dexforce_webxr_plastic_inflatable_bag_pick_release_final00 as webxr_bag_example,
)
from newton.examples.mjvbdv2 import (
    example_vbd_mjvbd_v2_dexforce_webxr_soft_then_rigid_cube_into_bag_final00 as webxr_soft_rigid_bag_example,
)
from newton.examples.mjvbdv2._webxr_teleop import (
    JsonlTrajectoryRecorder,
    LatestXRFrame,
    Pose,
    ProtocolError,
    RelativePoseRetargeter,
    WebXRServer,
    XRFrame,
    pack_scene_geometry,
)
from newton.examples.mjvbdv2._webxr_w1_head import (
    FIRST_PERSON_VIEW_MODE,
    W1HeadController,
    head_pose_to_neck_targets,
)


def _frame_payload(*, sequence=0, position=(0.0, 1.2, -0.5), orientation=(0.0, 0.0, 0.0, 1.0)):
    return {
        "type": "xr-frame",
        "version": 1,
        "streamId": "quest-test",
        "sequence": sequence,
        "timeMs": 1234.5,
        "referenceSpace": "local-floor",
        "visibilityState": "visible",
        "controllers": {
            "right": {
                "pose": {"position": list(position), "orientation": list(orientation)},
                "clutch": True,
                "buttons": [{"pressed": True, "value": 1.5}],
                "axes": [0.1, -0.2],
                "triggerValue": 1.5,
            }
        },
    }


class TestWebXRProtocol(unittest.TestCase):
    def test_frame_normalizes_pose_and_clamps_trigger(self):
        payload = _frame_payload(orientation=(0.0, 0.0, 0.0, 2.0))
        frame = XRFrame.from_json(json.dumps(payload), received_monotonic=10.0)

        controller = frame.controllers["right"]
        np.testing.assert_allclose(controller.pose.orientation, [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(controller.trigger_value, 1.0)
        np.testing.assert_allclose(controller.thumbstick, (0.1, -0.2))
        self.assertEqual(frame.received_monotonic, 10.0)
        self.assertEqual(frame.view_mode, "observer")
        self.assertIsNone(frame.head_pose)

    def test_frame_rejects_non_finite_pose(self):
        payload = _frame_payload(position=(0.0, float("nan"), 0.0))

        with self.assertRaisesRegex(ProtocolError, "finite"):
            XRFrame.from_mapping(payload)

    def test_frame_accepts_newton_world_controller_space(self):
        payload = _frame_payload()
        payload["controllerSpace"] = "newton-world"

        frame = XRFrame.from_mapping(payload)

        self.assertEqual(frame.controller_space, "newton-world")

        state = LatestXRFrame()
        state.update(frame)
        self.assertEqual(state.status().controller_space, "newton-world")

    def test_frame_accepts_optional_head_pose_and_first_person_mode(self):
        payload = _frame_payload()
        payload["viewMode"] = "robot-first-person"
        payload["headPose"] = {
            "position": [0.01, 0.02, -0.03],
            "orientation": [0.0, 0.0, 0.0, 2.0],
        }

        frame = XRFrame.from_mapping(payload)

        self.assertEqual(frame.view_mode, "robot-first-person")
        np.testing.assert_allclose(frame.head_pose.position, [0.01, 0.02, -0.03])
        np.testing.assert_allclose(frame.head_pose.orientation, [0.0, 0.0, 0.0, 1.0])

    def test_frame_rejects_unknown_view_mode(self):
        payload = _frame_payload()
        payload["viewMode"] = "flying-camera"

        with self.assertRaisesRegex(ProtocolError, "view mode"):
            XRFrame.from_mapping(payload)

    def test_frame_rejects_unknown_controller_space(self):
        payload = _frame_payload()
        payload["controllerSpace"] = "camera-magic"

        with self.assertRaisesRegex(ProtocolError, "controller space"):
            XRFrame.from_mapping(payload)


class TestLatestXRFrame(unittest.TestCase):
    def test_keeps_latest_sequence_and_expires_stale_frame(self):
        state = LatestXRFrame()
        newer = XRFrame.from_mapping(_frame_payload(sequence=2), received_monotonic=10.0)
        older = XRFrame.from_mapping(_frame_payload(sequence=1), received_monotonic=11.0)

        self.assertTrue(state.update(newer))
        self.assertFalse(state.update(older))
        self.assertIs(state.snapshot(max_age_seconds=0.2, now=10.1), newer)
        self.assertIsNone(state.snapshot(max_age_seconds=0.2, now=10.3))

    def test_scene_reset_requests_coalesce_and_are_consumed_once(self):
        state = LatestXRFrame()

        self.assertIsNone(state.consume_reset())
        self.assertEqual(state.request_reset(), 1)
        self.assertEqual(state.request_reset(), 2)
        self.assertEqual(state.consume_reset(), 2)
        self.assertIsNone(state.consume_reset())

    def test_shutdown_requests_coalesce_and_are_consumed_once(self):
        state = LatestXRFrame()

        self.assertIsNone(state.consume_shutdown())
        self.assertEqual(state.request_shutdown(), 1)
        self.assertEqual(state.request_shutdown(), 2)
        self.assertEqual(state.consume_shutdown(), 2)
        self.assertIsNone(state.consume_shutdown())

    def test_teleoperation_mode_requests_are_latest_value(self):
        state = LatestXRFrame()

        self.assertTrue(state.status().teleoperation_active)
        self.assertTrue(state.status().simulation_active)
        self.assertEqual(state.request_teleoperation_mode(False), 1)
        self.assertEqual(state.request_teleoperation_mode(True), 2)
        self.assertTrue(state.status().teleoperation_mode_pending)
        self.assertTrue(state.status().teleoperation_active)
        self.assertEqual(state.consume_teleoperation_mode(), (2, True, True))
        self.assertFalse(state.status().teleoperation_mode_pending)
        self.assertIsNone(state.consume_teleoperation_mode())

        self.assertEqual(state.request_teleoperation_mode(False), 3)
        self.assertTrue(state.status().teleoperation_active)
        self.assertEqual(state.consume_teleoperation_mode(), (3, False, False))
        self.assertFalse(state.status().teleoperation_active)
        self.assertFalse(state.status().simulation_active)

    def test_standby_disarms_teleoperation_without_stopping_simulation(self):
        state = LatestXRFrame()

        self.assertEqual(state.request_standby(), 1)
        self.assertEqual(state.consume_teleoperation_mode(), (1, False, True))

        status = state.status()
        self.assertFalse(status.teleoperation_active)
        self.assertTrue(status.simulation_active)
        self.assertEqual(status.operation_mode, "standby")

    def test_standby_does_not_restart_a_parked_simulation(self):
        state = LatestXRFrame()
        state.request_teleoperation_mode(False)
        state.consume_teleoperation_mode()

        self.assertIsNone(state.request_standby())
        self.assertIsNone(state.consume_teleoperation_mode())
        self.assertFalse(state.status().simulation_active)


class TestWebXRServerReadiness(unittest.TestCase):
    def test_deferred_health_waits_for_simulation_warmup(self):
        server = WebXRServer(require_simulation_ready=True)

        status, payload = server.health_snapshot()
        self.assertEqual(status, 503)
        self.assertFalse(payload["ok"])
        self.assertFalse(payload["simulationReady"])

        server.mark_simulation_ready()
        status, payload = server.health_snapshot()
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["simulationReady"])


class TestWebXRExampleConfiguration(unittest.TestCase):
    def test_teleop_parsers_disable_automatic_recording(self):
        examples = (
            webxr_example,
            webxr_chair_example,
            webxr_bag_example,
            webxr_soft_rigid_bag_example,
            webxr_tshirt_example,
            webxr_nut_bolt_example,
        )

        for example in examples:
            with self.subTest(example=example.__name__):
                parser = example.Example.create_parser()
                self.assertFalse(parser.parse_args([]).record_on_connect)
                self.assertTrue(parser.parse_args(["--record-on-connect"]).record_on_connect)

    def test_parser_accepts_persistent_sdf_cache_directory(self):
        cache_dir = Path("/tmp/newton-webxr-test-sdf-cache")

        args = webxr_example.Example.create_parser().parse_args(["--sdf-cache-dir", str(cache_dir)])

        self.assertEqual(args.sdf_cache_dir, cache_dir)

    def test_chair_parser_uses_separate_port_and_disables_graph_capture(self):
        args = webxr_chair_example.Example.create_parser().parse_args([])

        self.assertEqual(args.webxr_port, 8766)
        self.assertFalse(args.graph_capture)

    def test_plug_parser_disables_graph_capture(self):
        args = webxr_example.Example.create_parser().parse_args([])

        self.assertFalse(args.graph_capture)

    def test_bag_parser_uses_separate_port_and_disables_graph_capture(self):
        args = webxr_bag_example.Example.create_parser().parse_args([])

        self.assertEqual(args.webxr_port, 8767)
        self.assertFalse(args.graph_capture)
        self.assertTrue(args.plastic)

    def test_soft_rigid_bag_parser_uses_separate_port_and_disables_graph_capture(self):
        args = webxr_soft_rigid_bag_example.Example.create_parser().parse_args([])

        self.assertEqual(args.webxr_port, 8768)
        self.assertFalse(args.graph_capture)
        self.assertEqual(webxr_soft_rigid_bag_example.WEBXR_CAMERA_DOLLY_METERS, 1.8)

    def test_remaining_scenes_publish_robot_first_person_head_tracking(self):
        """Keep every Quest scene on the shared W1 eye-camera protocol."""
        examples = (
            webxr_example,
            webxr_chair_example,
            webxr_bag_example,
            webxr_soft_rigid_bag_example,
        )

        for example in examples:
            with self.subTest(example=example.__name__):
                source = Path(example.__file__).read_text(encoding="utf-8")
                self.assertIn("W1HeadController", source)
                self.assertIn('"firstPersonCamera"', source)
                self.assertIn('"firstPersonHiddenBodies"', source)
                self.assertIn('"firstPersonEnabled": True', source)
                self.assertIn('"viewMode"', source)
                self.assertIn('"headPose"', source)
                self.assertIn('"neckJointTargets"', source)

    def test_tshirt_parser_uses_separate_port_and_safe_defaults(self):
        args = webxr_tshirt_example.Example.create_parser().parse_args([])

        self.assertEqual(args.webxr_port, 8769)
        self.assertFalse(args.graph_capture)
        self.assertFalse(args.record_on_connect)
        self.assertEqual(webxr_tshirt_example.WEBXR_CAMERA_DOLLY_METERS, 1.8)
        self.assertEqual(webxr_tshirt_example.WEBXR_CAMERA_HEIGHT_METERS, 0.35)
        self.assertEqual(webxr_tshirt_example.WEBXR_CAMERA_PITCH_OFFSET_DEGREES, -8.0)

    def test_tshirt_finger_targets_use_contact_aware_step_limits(self):
        """Limit T-shirt finger motion further after hand contact."""
        self.assertEqual(
            webxr_tshirt_example.MAX_FINGER_SPEED_DEG_S,
            2.0 * webxr_bag_example.bag_recorder.MAX_FINGER_SPEED_DEG_S,
        )
        self.assertEqual(
            webxr_tshirt_example.MAX_FINGER_CONTACT_SPEED_DEG_S,
            2.0 * webxr_bag_example.bag_recorder.MAX_FINGER_CONTACT_SPEED_DEG_S,
        )
        current_q = wp.array([0.0, 0.2, 0.0, 0.8], dtype=wp.float32, device="cpu")
        finger_indices = wp.array([1, 3], dtype=wp.int32, device="cpu")
        desired_q = wp.array([1.0, 0.0], dtype=wp.float32, device="cpu")
        soft_contact_count = wp.array([0], dtype=wp.int32, device="cpu")
        soft_contact_shape = wp.array([1], dtype=wp.int32, device="cpu")
        hand_shape_mask = wp.array([0, 1], dtype=wp.int32, device="cpu")
        target_q = wp.clone(current_q)

        wp.launch(
            webxr_tshirt_example._limit_hand_target_step,
            finger_indices.shape[0],
            [
                current_q,
                finger_indices,
                desired_q,
                soft_contact_count,
                soft_contact_shape,
                hand_shape_mask,
                0.1,
                0.03,
                target_q,
            ],
            device="cpu",
        )
        np.testing.assert_allclose(target_q.numpy()[[1, 3]], [0.3, 0.7], atol=1.0e-6)

        soft_contact_count.fill_(1)
        target_q.assign(current_q)
        wp.launch(
            webxr_tshirt_example._limit_hand_target_step,
            finger_indices.shape[0],
            [
                current_q,
                finger_indices,
                desired_q,
                soft_contact_count,
                soft_contact_shape,
                hand_shape_mask,
                0.1,
                0.03,
                target_q,
            ],
            device="cpu",
        )
        np.testing.assert_allclose(target_q.numpy()[[1, 3]], [0.23, 0.77], atol=1.0e-6)

    def test_nut_bolt_parser_uses_dedicated_port_and_safe_defaults(self):
        """Configure nut/bolt teleoperation without graph capture or automatic recording."""
        args = webxr_nut_bolt_example.Example.create_parser().parse_args([])

        self.assertEqual(args.webxr_port, 8770)
        self.assertFalse(args.graph_capture)
        self.assertFalse(args.record_on_connect)
        self.assertIsNotNone(args.sdf_cache_dir)

    def test_nut_bolt_fingers_slow_on_rigid_contact(self):
        """Reduce both closing and opening steps when either rigid-contact shape is a hand."""
        current_q = wp.array([0.0, 0.2, 0.0, 0.8], dtype=wp.float32, device="cpu")
        finger_indices = wp.array([1, 3], dtype=wp.int32, device="cpu")
        desired_q = wp.array([1.0, 0.0], dtype=wp.float32, device="cpu")
        contact_count = wp.array([1], dtype=wp.int32, device="cpu")
        contact_shape0 = wp.array([2], dtype=wp.int32, device="cpu")
        contact_shape1 = wp.array([1], dtype=wp.int32, device="cpu")
        hand_shape_mask = wp.array([0, 1, 0], dtype=wp.int32, device="cpu")
        target_q = wp.clone(current_q)

        wp.launch(
            webxr_nut_bolt_example._limit_hand_target_step,
            finger_indices.shape[0],
            [
                current_q,
                finger_indices,
                desired_q,
                contact_count,
                contact_shape0,
                contact_shape1,
                hand_shape_mask,
                0.1,
                0.03,
                target_q,
            ],
            device="cpu",
        )

        np.testing.assert_allclose(target_q.numpy()[[1, 3]], [0.23, 0.77], atol=1.0e-6)

    def test_nut_bolt_adds_a_physical_table_below_the_prethreaded_pair(self):
        """Support the initial threaded pair on one visible, physical table."""

        class FakeBuilder:
            def __init__(self):
                self.box_call = None
                self.filter_pairs = []

            def add_shape_box(self, *args, **kwargs):
                self.box_call = (args, kwargs)
                return 41

            def add_shape_collision_filter_pair(self, shape0, shape1):
                self.filter_pairs.append((shape0, shape1))

        example = object.__new__(webxr_nut_bolt_example.Example)
        example.nut_thread_shape = 23
        builder = FakeBuilder()

        example._add_scene_support(builder)

        self.assertEqual(example.table_shape, 41)
        self.assertEqual(builder.filter_pairs, [(41, 23)])
        self.assertEqual(builder.box_call[0], (-1,))
        self.assertEqual(builder.box_call[1]["label"], "webxr_nut_bolt_table")
        table = example._static_boxes[0]
        self.assertEqual(table["role"], "table")
        self.assertAlmostEqual(
            table["position"][2] + 0.5 * table["scale"][2],
            webxr_nut_bolt_example.TABLE_TOP_Z,
        )
        self.assertLess(
            webxr_nut_bolt_example.TABLE_TOP_Z,
            float(webxr_nut_bolt_example.nut_bolt.ASSEMBLY_ORIGIN[2]),
        )

    def test_shared_head_pose_maps_webxr_yaw_and_pitch_to_neck(self):
        yaw = np.deg2rad(30.0)
        yaw_pose = Pose(
            position=np.zeros(3, dtype=np.float32),
            orientation=np.array([0.0, np.sin(0.5 * yaw), 0.0, np.cos(0.5 * yaw)], dtype=np.float32),
        )
        pitch = np.deg2rad(-20.0)
        pitch_pose = Pose(
            position=np.zeros(3, dtype=np.float32),
            orientation=np.array([np.sin(0.5 * pitch), 0.0, 0.0, np.cos(0.5 * pitch)], dtype=np.float32),
        )

        yaw_target, neutral_pitch = head_pose_to_neck_targets(yaw_pose)
        neutral_yaw, pitch_target = head_pose_to_neck_targets(pitch_pose)

        self.assertAlmostEqual(yaw_target, yaw, places=6)
        self.assertAlmostEqual(neutral_pitch, 0.0, places=6)
        self.assertAlmostEqual(neutral_yaw, 0.0, places=6)
        self.assertAlmostEqual(pitch_target, pitch, places=6)

        extreme_yaw = np.deg2rad(120.0)
        extreme_pose = Pose(
            position=np.zeros(3, dtype=np.float32),
            orientation=np.array(
                [0.0, np.sin(0.5 * extreme_yaw), 0.0, np.cos(0.5 * extreme_yaw)],
                dtype=np.float32,
            ),
        )
        clamped_yaw, _ = head_pose_to_neck_targets(extreme_pose)
        self.assertAlmostEqual(clamped_yaw, 0.5 * np.pi, places=6)

    def test_shared_head_controller_rate_limits_neck_and_anchors_both_eyes(self):
        class FakeModel:
            joint_label = ("W1/NECK1", "W1/NECK2")
            body_label = ("W1/neck1", "W1/neck2")
            joint_q_start = wp.array([0, 1, 2], dtype=wp.int32, device="cpu")
            joint_qd_start = wp.array([0, 1, 2], dtype=wp.int32, device="cpu")
            joint_q = wp.array([0.0, 0.0], dtype=wp.float32, device="cpu")
            joint_limit_lower = wp.array([-2.0, -2.0], dtype=wp.float32, device="cpu")
            joint_limit_upper = wp.array([2.0, 2.0], dtype=wp.float32, device="cpu")

        yaw = np.deg2rad(45.0)
        pose = Pose(
            position=np.zeros(3, dtype=np.float32),
            orientation=np.array([0.0, np.sin(0.5 * yaw), 0.0, np.cos(0.5 * yaw)], dtype=np.float32),
        )
        controller = W1HeadController(FakeModel(), "cpu", wp.quat_identity())
        target_q = wp.zeros(2, dtype=wp.float32, device="cpu")

        controller.set_desired_pose(FIRST_PERSON_VIEW_MODE, pose)
        controller.write_targets(target_q, 0.1)

        self.assertAlmostEqual(float(target_q.numpy()[0]), np.deg2rad(5.0), places=6)
        self.assertAlmostEqual(float(target_q.numpy()[1]), 0.0, places=6)
        body_q = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        camera = controller.camera_state(body_q)
        np.testing.assert_allclose(camera["position"], [1.091, 1.949, 3.0], atol=1.0e-6)
        self.assertEqual(controller.hidden_body_ids, (0, 1))

    def test_tshirt_one_click_scripts_use_safe_handoff_service(self):
        """Keep the T-shirt wrappers on the shared safe-handoff protocol."""
        repo_root = Path(__file__).resolve().parents[2]
        start_script = repo_root / "scripts" / "start_quest_webxr_tshirt_teleop.sh"
        stop_script = repo_root / "scripts" / "stop_quest_webxr_tshirt_teleop.sh"

        self.assertTrue(start_script.is_file())
        self.assertTrue(stop_script.is_file())
        start_source = start_script.read_text(encoding="utf-8")
        stop_source = stop_script.read_text(encoding="utf-8")
        self.assertIn('NEWTON_WEBXR_PORT="${NEWTON_WEBXR_PORT:-8769}"', start_source)
        self.assertIn('NEWTON_WEBXR_UNIT="newton-quest-webxr-tshirt.service"', start_source)
        self.assertIn('exec "${script_dir}/start_quest_webxr_teleop.sh"', start_source)
        self.assertIn('exec "${script_dir}/stop_quest_webxr_teleop.sh"', stop_source)
        for peer_script_name in (
            "start_quest_webxr_teleop.sh",
            "start_quest_webxr_chair_teleop.sh",
            "start_quest_webxr_bag_teleop.sh",
            "start_quest_webxr_soft_rigid_bag_teleop.sh",
            "start_quest_webxr_nut_bolt_teleop.sh",
        ):
            peer_source = (repo_root / "scripts" / peer_script_name).read_text(encoding="utf-8")
            self.assertIn("newton-quest-webxr-tshirt.service:8769", peer_source)
        for peer in (
            "newton-quest-webxr.service:8765",
            "newton-quest-webxr-chair.service:8766",
            "newton-quest-webxr-bag.service:8767",
            "newton-quest-webxr-soft-rigid-bag.service:8768",
            "newton-quest-webxr-nut-bolt.service:8770",
        ):
            self.assertIn(peer, start_source)

    def test_nut_bolt_one_click_scripts_use_safe_handoff_service(self):
        """Wire nut/bolt start, stop, and reload wrappers into every peer scene."""
        repo_root = Path(__file__).resolve().parents[2]
        start_script = repo_root / "scripts" / "start_quest_webxr_nut_bolt_teleop.sh"
        stop_script = repo_root / "scripts" / "stop_quest_webxr_nut_bolt_teleop.sh"
        reload_script = repo_root / "scripts" / "reload_quest_webxr_nut_bolt_teleop.sh"

        self.assertTrue(start_script.is_file())
        self.assertTrue(stop_script.is_file())
        self.assertTrue(reload_script.is_file())
        start_source = start_script.read_text(encoding="utf-8")
        stop_source = stop_script.read_text(encoding="utf-8")
        reload_source = reload_script.read_text(encoding="utf-8")
        self.assertIn('NEWTON_WEBXR_PORT="${NEWTON_WEBXR_PORT:-8770}"', start_source)
        self.assertIn('NEWTON_WEBXR_UNIT="newton-quest-webxr-nut-bolt.service"', start_source)
        self.assertIn("NEWTON_WEBXR_SDF_CACHE=1", start_source)
        self.assertIn("example_mjvbd_v2_bimanual_nut_bolt.py", start_source)
        self.assertIn('exec "${script_dir}/start_quest_webxr_teleop.sh"', start_source)
        self.assertIn('exec "${script_dir}/stop_quest_webxr_teleop.sh"', stop_source)
        self.assertIn('exec "${script_dir}/reload_quest_webxr_teleop.sh"', reload_source)
        for peer_script_name in (
            "start_quest_webxr_teleop.sh",
            "start_quest_webxr_chair_teleop.sh",
            "start_quest_webxr_bag_teleop.sh",
            "start_quest_webxr_soft_rigid_bag_teleop.sh",
            "start_quest_webxr_tshirt_teleop.sh",
        ):
            peer_source = (repo_root / "scripts" / peer_script_name).read_text(encoding="utf-8")
            self.assertIn("newton-quest-webxr-nut-bolt.service:8770", peer_source)


class TestWebXRClientRendering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset_dir = Path(__file__).resolve().parents[1] / "examples" / "assets" / "webxr_teleop"
        cls.source = (asset_dir / "app.js").read_text(encoding="utf-8")
        cls.html_source = (asset_dir / "index.html").read_text(encoding="utf-8")
        cls.style_source = (asset_dir / "style.css").read_text(encoding="utf-8")

    def test_mixed_cube_scene_supports_left_thumbstick_view_rotation(self):
        self.assertIn("function updateViewRotation", self.source)
        self.assertIn('source.handedness === "left"', self.source)
        self.assertIn("leftThumbstickRotate", self.source)

    def test_bag_meshes_are_drawn_without_backface_culling(self):
        self.assertIn('shape.role === "bag"', self.source)
        self.assertIn("context.disable(context.CULL_FACE)", self.source)

    def test_control_panel_can_be_hidden_and_restored(self):
        self.assertIn('id="teleop-panel"', self.html_source)
        self.assertIn('id="toggle-panel"', self.html_source)
        self.assertIn('aria-controls="teleop-panel"', self.html_source)
        self.assertIn("function setPanelHidden", self.source)
        self.assertIn("panel.hidden = hidden", self.source)
        self.assertIn(".panel-toggle", self.style_source)
        self.assertIn(".panel[hidden]", self.style_source)

    def test_tshirt_geometry_has_a_readable_role_label(self):
        self.assertIn('shirt: "T 恤"', self.source)

    def test_scene_camera_supports_independent_height_and_pitch_offsets(self):
        self.assertIn("cameraHeightMeters", self.source)
        self.assertIn("cameraPitchOffsetDegrees", self.source)
        self.assertIn('const TSHIRT_SCENE_KIND = "bimanual-fold-tshirt"', self.source)
        self.assertIn("legacyTshirtScene", self.source)

    def test_tshirt_scene_can_toggle_robot_first_person_head_tracking(self):
        self.assertIn('id="toggle-view-mode"', self.html_source)
        self.assertIn('let viewMode = "observer"', self.source)
        self.assertIn('"robot-first-person"', self.source)
        self.assertIn("headPose", self.source)
        self.assertIn("firstPersonCamera", self.source)
        self.assertIn("firstPersonHiddenBodies", self.source)
        self.assertIn("第一人称需重载 Newton 进程", self.source)


class TestQuestBrowserLaunch(unittest.TestCase):
    def test_start_refreshes_the_reused_tab_and_reports_stale_python_sources(self):
        repo_root = Path(__file__).resolve().parents[2]
        start_source = (repo_root / "scripts" / "start_quest_webxr_teleop.sh").read_text(encoding="utf-8")
        stop_source = (repo_root / "scripts" / "stop_quest_webxr_teleop.sh").read_text(encoding="utf-8")

        self.assertIn("?launch=${browser_launch_id}", start_source)
        self.assertIn("NEWTON_WEBXR_RELOAD_SOURCES", start_source)
        self.assertIn('-nt "/proc/${demo_pid}"', start_source)
        self.assertIn("当前进程未加载这些更新", start_source)
        self.assertIn("再次启动只会恢复同一 PID", stop_source)

    def test_start_reuses_browser_tab_created_by_the_same_application_id(self):
        repo_root = Path(__file__).resolve().parents[2]
        start_script = repo_root / "scripts" / "start_quest_webxr_teleop.sh"
        example_name = "mjvbd_v2_dexforce_webxr_plug_socket"

        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            bin_dir = temp_root / "bin"
            bin_dir.mkdir()
            actions_path = temp_root / "actions.log"

            commands = {
                "adb": (
                    'printf \'adb %s\\n\' "$*" >> "${TEST_ACTIONS}"\n'
                    "if [[ \"${1:-}\" == 'get-state' ]]; then\n"
                    "  printf 'device\\n'\n"
                    "fi\n"
                ),
                "curl": (
                    'printf \'curl %s\\n\' "$*" >> "${TEST_ACTIONS}"\n'
                    "if [[ \"$*\" == *'/healthz'* ]]; then\n"
                    "  printf '{\"teleoperationActive\": true}\\n'\n"
                    "fi\n"
                ),
                "systemctl": (
                    'printf \'systemctl %s\\n\' "$*" >> "${TEST_ACTIONS}"\n'
                    "if [[ \"$*\" == *'ActiveState'* ]]; then\n"
                    "  printf 'active\\n'\n"
                    "elif [[ \"$*\" == *'MainPID'* ]]; then\n"
                    "  printf '%s\\n' \"${TEST_DEMO_PID}\"\n"
                    "fi\n"
                ),
                "systemd-run": 'printf \'systemd-run %s\\n\' "$*" >> "${TEST_ACTIONS}"\n',
                "uv": "exit 0\n",
            }
            for name, body in commands.items():
                command = bin_dir / name
                command.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{body}", encoding="utf-8")
                command.chmod(0o755)

            demo = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    f"exec -a 'uv run python -u -m newton.examples {example_name}' /usr/bin/sleep 30",
                ]
            )
            try:
                reload_source = temp_root / "updated-example.py"
                reload_source.write_text("# updated after the managed process started\n", encoding="utf-8")
                os.utime(reload_source, (2_000_000_000, 2_000_000_000))
                environment = os.environ.copy()
                environment.update(
                    {
                        "PATH": f"{bin_dir}{os.pathsep}{environment['PATH']}",
                        "TEST_ACTIONS": str(actions_path),
                        "TEST_DEMO_PID": str(demo.pid),
                        "XDG_CACHE_HOME": str(temp_root / "cache"),
                        "XDG_RUNTIME_DIR": str(temp_root / "run"),
                        "XDG_STATE_HOME": str(temp_root / "state"),
                        "NEWTON_WEBXR_DEVICE": "cpu",
                        "NEWTON_WEBXR_PEERS": " ",
                        "NEWTON_WEBXR_PORT": "18765",
                        "NEWTON_WEBXR_UNIT": "newton-webxr-start-test.service",
                        "NEWTON_WEBXR_EXAMPLE": example_name,
                        "NEWTON_WEBXR_RELOAD_SOURCES": str(reload_source),
                    }
                )
                result = subprocess.run(
                    ["bash", str(start_script)],
                    cwd=repo_root,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            finally:
                demo.terminate()
                demo.wait(timeout=5)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("当前进程未加载这些更新", result.stderr)
            actions = actions_path.read_text(encoding="utf-8")
            self.assertEqual(actions.count("adb shell am start"), 1)
            self.assertRegex(actions, r"http://127\.0\.0\.1:18765/\?launch=[0-9]+")
            self.assertIn("--es com.android.browser.application_id org.newton.webxr.teleop", actions)
            self.assertIn("--ez create_new_tab false", actions)


class TestSafeStopScript(unittest.TestCase):
    def test_default_stop_enters_standby_without_terminating_or_removing_adb(self):
        repo_root = Path(__file__).resolve().parents[2]
        stop_script = repo_root / "scripts" / "stop_quest_webxr_teleop.sh"
        example_name = "mjvbd_v2_dexforce_webxr_plug_socket"

        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            bin_dir = temp_root / "bin"
            bin_dir.mkdir()
            actions_path = temp_root / "actions.log"

            commands = {
                "adb": 'printf \'adb %s\\n\' "$*" >> "${TEST_ACTIONS}"\n',
                "curl": (
                    'printf \'curl %s\\n\' "$*" >> "${TEST_ACTIONS}"\n'
                    "if [[ \"$*\" == *'/healthz'* ]]; then\n"
                    '  printf \'{"teleoperationActive": false, "simulationActive": true}\\n\'\n'
                    "fi\n"
                ),
                "sleep": "exit 0\n",
                "systemctl": (
                    'printf \'systemctl %s\\n\' "$*" >> "${TEST_ACTIONS}"\n'
                    "if [[ \"$*\" == *'ActiveState'* ]]; then\n"
                    "  printf 'active\\n'\n"
                    "elif [[ \"$*\" == *'MainPID'* ]]; then\n"
                    "  printf '%s\\n' \"${TEST_DEMO_PID}\"\n"
                    "fi\n"
                ),
            }
            for name, body in commands.items():
                command = bin_dir / name
                command.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{body}", encoding="utf-8")
                command.chmod(0o755)

            demo = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    f"exec -a 'uv run python -u -m newton.examples {example_name}' /usr/bin/sleep 30",
                ]
            )
            try:
                environment = os.environ.copy()
                environment.update(
                    {
                        "PATH": f"{bin_dir}{os.pathsep}{environment['PATH']}",
                        "TEST_ACTIONS": str(actions_path),
                        "TEST_DEMO_PID": str(demo.pid),
                        "XDG_RUNTIME_DIR": str(temp_root / "run"),
                        "XDG_STATE_HOME": str(temp_root / "state"),
                        "NEWTON_WEBXR_UNIT": "newton-webxr-stop-test.service",
                        "NEWTON_WEBXR_PORT": "18765",
                        "NEWTON_WEBXR_EXAMPLE": example_name,
                    }
                )
                result = subprocess.run(
                    ["bash", str(stop_script)],
                    cwd=repo_root,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            finally:
                demo.terminate()
                demo.wait(timeout=5)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("安全待机", result.stdout)
            actions = actions_path.read_text(encoding="utf-8")
            self.assertIn("/control/exit-immersive", actions)
            self.assertIn("/control/standby", actions)
            self.assertLess(actions.index("/control/standby"), actions.index("/control/exit-immersive"))
            self.assertIn("/healthz", actions)
            self.assertNotIn("/control/park", actions)
            self.assertNotIn("/control/shutdown", actions)
            self.assertNotIn("systemctl --user kill", actions)
            self.assertNotIn("adb reverse --remove", actions)


class TestStagedReloadScript(unittest.TestCase):
    def test_every_scene_has_a_dedicated_reload_wrapper(self):
        repo_root = Path(__file__).resolve().parents[2]
        scenes = (
            ("plug_socket", "start_quest_webxr_teleop.sh"),
            ("chair", "start_quest_webxr_chair_teleop.sh"),
            ("bag", "start_quest_webxr_bag_teleop.sh"),
            ("soft_rigid_bag", "start_quest_webxr_soft_rigid_bag_teleop.sh"),
        )

        for scene, start_name in scenes:
            with self.subTest(scene=scene):
                reload_script = repo_root / "scripts" / f"reload_quest_webxr_{scene}_teleop.sh"
                self.assertTrue(reload_script.is_file())
                reload_source = reload_script.read_text(encoding="utf-8")
                self.assertIn('exec "${script_dir}/reload_quest_webxr_teleop.sh"', reload_source)

                start_source = (repo_root / "scripts" / start_name).read_text(encoding="utf-8")
                self.assertIn(f"./scripts/reload_quest_webxr_{scene}_teleop.sh", start_source)
                self.assertIn("_webxr_w1_head.py", start_source)

    def test_reload_drains_old_process_before_starting_updated_scene(self):
        repo_root = Path(__file__).resolve().parents[2]
        reload_script = repo_root / "scripts" / "reload_quest_webxr_teleop.sh"
        guard_script = repo_root / "scripts" / "quest_webxr_cuda_guard.py"
        tshirt_reload_script = repo_root / "scripts" / "reload_quest_webxr_tshirt_teleop.sh"

        self.assertTrue(reload_script.is_file())
        self.assertTrue(guard_script.is_file())
        self.assertTrue(tshirt_reload_script.is_file())
        reload_source = reload_script.read_text(encoding="utf-8")
        self.assertIn("quest_webxr_cuda_guard.py", reload_source)
        self.assertIn("/control/standby", reload_source)
        self.assertIn("/control/exit-immersive", reload_source)
        self.assertIn("/control/park", reload_source)
        self.assertIn("/control/shutdown", reload_source)
        self.assertNotIn("systemctl --user kill", reload_source)
        self.assertNotIn("kill -KILL", reload_source)

        example_name = "mjvbd_v2_dexforce_webxr_plug_socket"
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            bin_dir = temp_root / "bin"
            bin_dir.mkdir()
            actions_path = temp_root / "actions.log"
            standby_path = temp_root / "standby"
            parked_path = temp_root / "parked"
            stopped_path = temp_root / "stopped"
            start_script = temp_root / "start-updated-scene.sh"
            start_script.write_text(
                '#!/usr/bin/env bash\nset -euo pipefail\nprintf \'start %s\\n\' "$*" >> "${TEST_ACTIONS}"\n',
                encoding="utf-8",
            )
            start_script.chmod(0o755)

            commands = {
                "curl": (
                    'printf \'curl %s\\n\' "$*" >> "${TEST_ACTIONS}"\n'
                    "if [[ \"$*\" == *'/control/standby'* ]]; then\n"
                    '  touch "${TEST_STANDBY}"\n'
                    "elif [[ \"$*\" == *'/control/park'* ]]; then\n"
                    '  touch "${TEST_PARKED}"\n'
                    "elif [[ \"$*\" == *'/control/shutdown'* ]]; then\n"
                    '  touch "${TEST_STOPPED}"\n'
                    '  kill "${TEST_DEMO_PID}"\n'
                    "elif [[ \"$*\" == *'/healthz'* ]]; then\n"
                    '  if [[ -e "${TEST_PARKED}" ]]; then\n'
                    '    printf \'{"teleoperationActive": false, "simulationActive": false}\\n\'\n'
                    '  elif [[ -e "${TEST_STANDBY}" ]]; then\n'
                    '    printf \'{"teleoperationActive": false, "simulationActive": true}\\n\'\n'
                    "  else\n"
                    '    printf \'{"teleoperationActive": true, "simulationActive": true}\\n\'\n'
                    "  fi\n"
                    "fi\n"
                ),
                "nvidia-smi": 'printf \'nvidia-smi %s\\n\' "$*" >> "${TEST_ACTIONS}"\n',
                "sleep": 'printf \'sleep %s\\n\' "$*" >> "${TEST_ACTIONS}"\n',
                "systemctl": (
                    'printf \'systemctl %s\\n\' "$*" >> "${TEST_ACTIONS}"\n'
                    "if [[ \"$*\" == *'newton-quest-webxr-cuda-guard.service'* ]]; then\n"
                    "  if [[ \"$*\" == *'ActiveState'* ]]; then printf 'active\\n'; fi\n"
                    "elif [[ \"$*\" == *'ActiveState'* ]]; then\n"
                    "  if [[ -e \"${TEST_STOPPED}\" ]]; then printf 'inactive\\n'; else printf 'active\\n'; fi\n"
                    "elif [[ \"$*\" == *'MainPID'* ]]; then\n"
                    "  printf '%s\\n' \"${TEST_DEMO_PID}\"\n"
                    "fi\n"
                ),
                "systemd-run": 'printf \'systemd-run %s\\n\' "$*" >> "${TEST_ACTIONS}"\n',
                "uv": "exit 0\n",
            }
            for name, body in commands.items():
                command = bin_dir / name
                command.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{body}", encoding="utf-8")
                command.chmod(0o755)

            runtime_name = "newton-webxr-reload-test"
            state_name = "newton-webxr-reload-test"
            runtime_root = temp_root / "run" / f"{runtime_name}-{os.getuid()}"
            state_root = temp_root / "state" / state_name
            runtime_root.mkdir(parents=True)
            state_root.mkdir(parents=True)
            (runtime_root / "demo.pid").write_text("stale\n", encoding="utf-8")
            (state_root / "active-run").write_text("stale\n", encoding="utf-8")

            guard = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    f"exec -a 'python {guard_script}' /usr/bin/sleep 30",
                ]
            )
            guard_runtime_root = temp_root / "run" / f"newton-webxr-cuda-guard-{os.getuid()}"
            guard_runtime_root.mkdir(parents=True)
            (guard_runtime_root / "ready").write_text(
                f"pid={guard.pid}\ndevice=cuda:0\n",
                encoding="utf-8",
            )
            demo = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    f"exec -a 'uv run python -u -m newton.examples {example_name}' /usr/bin/sleep 30",
                ]
            )
            try:
                environment = os.environ.copy()
                environment.update(
                    {
                        "PATH": f"{bin_dir}{os.pathsep}{environment['PATH']}",
                        "TEST_ACTIONS": str(actions_path),
                        "TEST_DEMO_PID": str(demo.pid),
                        "TEST_STANDBY": str(standby_path),
                        "TEST_PARKED": str(parked_path),
                        "TEST_STOPPED": str(stopped_path),
                        "XDG_CACHE_HOME": str(temp_root / "cache"),
                        "XDG_RUNTIME_DIR": str(temp_root / "run"),
                        "XDG_STATE_HOME": str(temp_root / "state"),
                        "NEWTON_WEBXR_DEVICE": "cuda:0",
                        "NEWTON_WEBXR_UNIT": "newton-webxr-reload-test.service",
                        "NEWTON_WEBXR_PORT": "18765",
                        "NEWTON_WEBXR_EXAMPLE": example_name,
                        "NEWTON_WEBXR_RUNTIME_NAME": runtime_name,
                        "NEWTON_WEBXR_STATE_NAME": state_name,
                        "NEWTON_WEBXR_START_SCRIPT": str(start_script),
                        "NEWTON_WEBXR_PHASE_DELAY_SECONDS": "0",
                        "NEWTON_WEBXR_PARK_DELAY_SECONDS": "0",
                        "NEWTON_WEBXR_RESTART_DELAY_SECONDS": "0",
                    }
                )
                result = subprocess.run(
                    ["bash", str(reload_script), "--record-on-connect"],
                    cwd=repo_root,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            finally:
                if demo.poll() is None:
                    demo.terminate()
                demo.wait(timeout=5)
                guard.terminate()
                guard.wait(timeout=5)

            self.assertEqual(result.returncode, 0, result.stderr)
            actions = actions_path.read_text(encoding="utf-8")
            standby = actions.index("/control/standby")
            exit_immersive = actions.index("/control/exit-immersive")
            park = actions.index("/control/park")
            shutdown = actions.index("/control/shutdown")
            start = actions.index("start --record-on-connect")
            guard_ready = actions.index("nvidia-smi")
            self.assertLess(standby, exit_immersive)
            self.assertLess(exit_immersive, park)
            self.assertLess(park, shutdown)
            self.assertLess(shutdown, start)
            self.assertLess(guard_ready, standby)
            self.assertNotIn("systemd-run", actions)
            self.assertFalse((runtime_root / "demo.pid").exists())
            self.assertFalse((state_root / "active-run").exists())


class TestSafeSceneHandoff(unittest.TestCase):
    def test_start_parks_standby_peer_only_after_target_is_ready(self):
        repo_root = Path(__file__).resolve().parents[2]
        start_script = repo_root / "scripts" / "start_quest_webxr_teleop.sh"
        example_name = "mjvbd_v2_dexforce_webxr_plug_socket"

        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            bin_dir = temp_root / "bin"
            bin_dir.mkdir()
            actions_path = temp_root / "actions.log"
            peer_standby_path = temp_root / "peer-standby"
            peer_parked_path = temp_root / "peer-parked"

            commands = {
                "adb": (
                    'printf \'adb %s\\n\' "$*" >> "${TEST_ACTIONS}"\n'
                    "if [[ \"${1:-}\" == 'get-state' ]]; then\n"
                    "  printf 'device\\n'\n"
                    "fi\n"
                ),
                "curl": (
                    'printf \'curl %s\\n\' "$*" >> "${TEST_ACTIONS}"\n'
                    "if [[ \"$*\" == *'127.0.0.1:28766/control/standby'* ]]; then\n"
                    '  touch "${TEST_PEER_STANDBY}"\n'
                    "elif [[ \"$*\" == *'127.0.0.1:28766/control/park'* ]]; then\n"
                    '  touch "${TEST_PEER_PARKED}"\n'
                    "elif [[ \"$*\" == *'127.0.0.1:28766/healthz'* ]]; then\n"
                    '  if [[ -e "${TEST_PEER_PARKED}" ]]; then\n'
                    '    printf \'{"teleoperationActive": false, "simulationActive": false}\\n\'\n'
                    '  elif [[ -e "${TEST_PEER_STANDBY}" ]]; then\n'
                    '    printf \'{"teleoperationActive": false, "simulationActive": true}\\n\'\n'
                    "  else\n"
                    '    printf \'{"teleoperationActive": true, "simulationActive": true}\\n\'\n'
                    "  fi\n"
                    "elif [[ \"$*\" == *'/healthz'* ]]; then\n"
                    '  printf \'{"teleoperationActive": true, "simulationActive": true}\\n\'\n'
                    "fi\n"
                ),
                "systemctl": (
                    'printf \'systemctl %s\\n\' "$*" >> "${TEST_ACTIONS}"\n'
                    "if [[ \"$*\" == *'ActiveState'* ]]; then\n"
                    "  printf 'active\\n'\n"
                    "elif [[ \"$*\" == *'MainPID'* ]]; then\n"
                    "  printf '%s\\n' \"${TEST_DEMO_PID}\"\n"
                    "fi\n"
                ),
                "systemd-run": 'printf \'systemd-run %s\\n\' "$*" >> "${TEST_ACTIONS}"\n',
                "uv": "exit 0\n",
            }
            for name, body in commands.items():
                command = bin_dir / name
                command.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{body}", encoding="utf-8")
                command.chmod(0o755)

            demo = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    f"exec -a 'uv run python -u -m newton.examples {example_name}' /usr/bin/sleep 30",
                ]
            )
            try:
                environment = os.environ.copy()
                environment.update(
                    {
                        "PATH": f"{bin_dir}{os.pathsep}{environment['PATH']}",
                        "TEST_ACTIONS": str(actions_path),
                        "TEST_DEMO_PID": str(demo.pid),
                        "TEST_PEER_STANDBY": str(peer_standby_path),
                        "TEST_PEER_PARKED": str(peer_parked_path),
                        "XDG_CACHE_HOME": str(temp_root / "cache"),
                        "XDG_RUNTIME_DIR": str(temp_root / "run"),
                        "XDG_STATE_HOME": str(temp_root / "state"),
                        "NEWTON_WEBXR_DEVICE": "cpu",
                        "NEWTON_WEBXR_PEERS": "peer.service:28766",
                        "NEWTON_WEBXR_PORT": "18765",
                        "NEWTON_WEBXR_UNIT": "newton-webxr-start-test.service",
                        "NEWTON_WEBXR_EXAMPLE": example_name,
                    }
                )
                result = subprocess.run(
                    ["bash", str(start_script)],
                    cwd=repo_root,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            finally:
                demo.terminate()
                demo.wait(timeout=5)

            self.assertEqual(result.returncode, 0, result.stderr)
            actions = actions_path.read_text(encoding="utf-8")
            peer_standby = actions.index("127.0.0.1:28766/control/standby")
            target_ready = actions.index("127.0.0.1:18765/healthz")
            peer_park = actions.index("127.0.0.1:28766/control/park")
            self.assertLess(peer_standby, target_ready)
            self.assertLess(target_ready, peer_park)
            self.assertTrue(peer_standby_path.exists())
            self.assertTrue(peer_parked_path.exists())


class TestRelativePoseRetargeter(unittest.TestCase):
    def test_clutch_maps_relative_translation_without_initial_jump(self):
        retargeter = RelativePoseRetargeter(translation_scale=2.0)
        robot_position = np.array([0.2, -0.1, 1.1], dtype=np.float32)
        robot_orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        anchor = Pose(
            position=np.array([0.0, 1.2, -0.5], dtype=np.float32),
            orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

        target = retargeter.update(
            anchor,
            clutch=True,
            robot_position=robot_position,
            robot_orientation=robot_orientation,
        )
        np.testing.assert_allclose(target.position, robot_position)
        np.testing.assert_allclose(target.orientation, robot_orientation)

        moved = Pose(
            position=anchor.position + np.array([0.1, 0.0, -0.2], dtype=np.float32),
            orientation=anchor.orientation.copy(),
        )
        target = retargeter.update(
            moved,
            clutch=True,
            robot_position=robot_position,
            robot_orientation=robot_orientation,
        )
        # WebXR +X right maps to robot -Y; WebXR -Z forward maps to robot +X.
        np.testing.assert_allclose(target.position, [0.6, -0.3, 1.1], atol=1.0e-6)

    def test_release_clears_anchor_and_next_clutch_recaptures(self):
        retargeter = RelativePoseRetargeter()
        controller = Pose(
            position=np.zeros(3, dtype=np.float32),
            orientation=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )
        robot_position = np.array([0.1, 0.2, 1.0], dtype=np.float32)
        robot_orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        retargeter.update(
            controller,
            clutch=True,
            robot_position=robot_position,
            robot_orientation=robot_orientation,
        )
        self.assertIsNone(
            retargeter.update(
                controller,
                clutch=False,
                robot_position=robot_position,
                robot_orientation=robot_orientation,
            )
        )
        new_robot_position = np.array([0.4, 0.3, 0.9], dtype=np.float32)
        target = retargeter.update(
            Pose(controller.position + 10.0, controller.orientation),
            clutch=True,
            robot_position=new_robot_position,
            robot_orientation=robot_orientation,
        )
        np.testing.assert_allclose(target.position, new_robot_position)

    def test_controller_orientation_uses_robot_coordinate_basis(self):
        retargeter = RelativePoseRetargeter()
        identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        anchor = Pose(position=np.zeros(3, dtype=np.float32), orientation=identity)
        retargeter.update(
            anchor,
            clutch=True,
            robot_position=np.zeros(3, dtype=np.float32),
            robot_orientation=identity,
        )

        half_angle = np.sqrt(0.5)
        rotated = Pose(
            position=anchor.position,
            orientation=np.array([0.0, half_angle, 0.0, half_angle], dtype=np.float32),
        )
        target = retargeter.update(
            rotated,
            clutch=True,
            robot_position=np.zeros(3, dtype=np.float32),
            robot_orientation=identity,
        )
        # WebXR +Y is robot +Z, so the same relative turn is about robot Z.
        np.testing.assert_allclose(target.orientation, [0.0, 0.0, half_angle, half_angle], atol=1.0e-6)

    def test_newton_world_pose_uses_identity_coordinate_basis(self):
        retargeter = RelativePoseRetargeter()
        identity_quaternion = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        anchor = Pose(position=np.zeros(3, dtype=np.float32), orientation=identity_quaternion)
        robot_position = np.array([0.2, -0.1, 1.1], dtype=np.float32)
        identity_basis = np.eye(3, dtype=np.float32)
        retargeter.update(
            anchor,
            clutch=True,
            robot_position=robot_position,
            robot_orientation=identity_quaternion,
            source_to_robot_rotation=identity_basis,
        )

        target = retargeter.update(
            Pose(position=np.array([0.1, 0.0, -0.2], dtype=np.float32), orientation=identity_quaternion),
            clutch=True,
            robot_position=robot_position,
            robot_orientation=identity_quaternion,
            source_to_robot_rotation=identity_basis,
        )

        np.testing.assert_allclose(target.position, [0.3, -0.1, 0.9], atol=1.0e-6)


class TestJsonlTrajectoryRecorder(unittest.TestCase):
    def test_writes_metadata_and_only_active_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.jsonl"
            recorder = JsonlTrajectoryRecorder(path, {"frameDtSeconds": 1.0 / 60.0}, flush_every=1)

            self.assertFalse(recorder.append({"frame": 0}))
            recorder.start()
            self.assertTrue(recorder.append({"frame": 1}))
            recorder.pause()
            self.assertFalse(recorder.append({"frame": 2}))
            recorder.close()

            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[0]["format"], "newton_webxr_trajectory_v1")
            self.assertEqual(records[1], {"type": "frame", "frame": 1})

    def test_writes_reset_event_even_while_frame_recording_is_paused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.jsonl"
            recorder = JsonlTrajectoryRecorder(path, {})
            recorder.start()
            recorder.pause()

            self.assertTrue(recorder.append_event({"event": "scene-reset", "episode": 1}))
            recorder.close()

            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(records[1], {"type": "event", "event": "scene-reset", "episode": 1})


class TestSceneGeometry(unittest.TestCase):
    def test_soft_rigid_bag_box_mesh_has_unit_extents_and_sharp_normals(self):
        vertices, normals, indices = webxr_soft_rigid_bag_example.Example._box_mesh()

        np.testing.assert_array_equal(vertices.min(axis=0), [-1.0, -1.0, -1.0])
        np.testing.assert_array_equal(vertices.max(axis=0), [1.0, 1.0, 1.0])
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
        self.assertEqual(vertices.shape, normals.shape)
        self.assertEqual(indices.shape, (36,))
        self.assertGreaterEqual(int(indices.min()), 0)
        self.assertLess(int(indices.max()), len(vertices))

    def test_packs_aligned_mesh_buffers_and_shape_instances(self):
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        normals = np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32)
        indices = np.array([0, 1, 2], dtype=np.uint32)
        shapes = [
            {
                "body": 4,
                "mesh": 0,
                "position": [0.1, 0.2, 0.3],
                "orientation": [0.0, 0.0, 0.0, 1.0],
                "scale": [1.0, 1.0, 1.0],
                "color": [0.2, 0.4, 0.8],
            }
        ]

        payload = pack_scene_geometry([(vertices, normals, indices)], shapes)

        self.assertEqual(payload[:4], b"NXR1")
        header_size = struct.unpack_from("<I", payload, 4)[0]
        header = json.loads(payload[8 : 8 + header_size])
        data_offset = (8 + header_size + 3) & ~3
        mesh = header["meshes"][0]
        self.assertEqual(header["shapes"], shapes)
        self.assertEqual(mesh["vertexCount"], 3)
        self.assertEqual(mesh["indexCount"], 3)
        self.assertEqual((data_offset + mesh["vertexByteOffset"]) % 4, 0)
        self.assertEqual((data_offset + mesh["indexByteOffset"]) % 4, 0)

        interleaved = np.frombuffer(
            payload,
            dtype="<f4",
            count=mesh["vertexCount"] * 6,
            offset=data_offset + mesh["vertexByteOffset"],
        ).reshape(-1, 6)
        packed_indices = np.frombuffer(
            payload,
            dtype="<u4",
            count=mesh["indexCount"],
            offset=data_offset + mesh["indexByteOffset"],
        )
        np.testing.assert_array_equal(interleaved[:, :3], vertices)
        np.testing.assert_array_equal(interleaved[:, 3:], normals)
        np.testing.assert_array_equal(packed_indices, indices)


if __name__ == "__main__":
    unittest.main(verbosity=2)

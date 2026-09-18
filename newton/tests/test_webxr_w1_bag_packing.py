# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise independent W1 V030 controller and optical gripper input."""

import json
import struct
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.examples.mjvbdv2 import example_mjvbd_v2_w1_bag_packing as packing
from newton.examples.mjvbdv2._webxr_gripper_input import GripperInput
from newton.examples.mjvbdv2._webxr_parallel_gripper import ParallelGripperRetargeter
from newton.examples.mjvbdv2._webxr_teleop import ControllerState, HandState, Pose, XRFrame
from newton.examples.mjvbdv2._webxr_w1_head import HEAD_MAX_SPEED_RADIANS_S, W1HeadController
from newton.examples.mjvbdv2.example_mjvbd_v2_w1_pick_place import ASSET
from newton.examples.mjvbdv2.example_mjvbd_v2_webxr_w1_bag_packing import Example
from newton.tests.test_webxr_parallel_gripper import skeleton
from newton.viewer import ViewerNull


def frame(sequence=0, *, mode="controllers", position=(0, 0, 0), activation=1, enabled=True, span=0.10):
    """Build a real protocol frame with independent left and right inputs."""
    pose = Pose(np.array(position, dtype=np.float32), np.array((0, 0, 0, 1), dtype=np.float32))
    controller = ControllerState("left", pose, True, False, (), (), (), (0, 0), 1.0)
    return XRFrame(
        "test",
        sequence,
        sequence * 16,
        0,
        "local-floor",
        "newton-world",
        "visible",
        {"left": controller},
        input_mode=mode,
        hands={"left": HandState(pose, skeleton(span), enabled, activation)},
    )


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA for the full packing solver")
class TestPackingPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device_scope = wp.ScopedDevice("cuda:0")
        cls.device_scope.__enter__()
        cls.addClassCleanup(cls.device_scope.__exit__, None, None, None)
        with patch.object(packing, "SolverMJVBDV2", wraps=packing.SolverMJVBDV2) as factory:
            cls.automatic = packing.Example(ViewerNull(), packing.Example.create_parser().parse_args([]))
            cls.teleop = Example(ViewerNull(), Example.create_parser().parse_args(["--no-webxr-server"]))
            cls.addClassCleanup(cls.teleop.close)
            cls.options = [call.kwargs for call in factory.call_args_list]
            # Compare motion separately with matching initial geometry.
            with (
                patch.object(packing.Example, "_initial_bag_yaw", np.pi / 2),
                patch.object(packing.Example, "_initial_bag_offset", Example._initial_bag_offset),
                patch.object(packing.Example, "_initial_gripper_openings", Example._initial_gripper_openings),
                patch.object(packing.Example, "_initial_head_pitch", Example._initial_head_pitch),
            ):
                cls.aligned = packing.Example(ViewerNull(), packing.Example.create_parser().parse_args([]))

    def test_initial_bag_mouth_faces_snacks(self):
        """Rotate the laid bag toward the snacks while keeping it supported by the table."""
        b = self.teleop
        points = b._initial_state.particle_q.numpy()[: b.bag_particle_count]
        position, rotation, error = packing.fit_bag(b.rest[: b.paper_count], points[: b.paper_count])
        np.testing.assert_allclose(rotation[:, 2], (0, -1, 0), atol=1e-6)
        mouth = position + rotation @ np.array((0, 0, packing.HEIGHT))
        self.assertGreater(float((b.pick.mean(axis=0) - mouth) @ rotation[:, 2]), 0.05)
        self.assertLess(error, 1e-5)
        self.assertAlmostEqual(
            float((points[:, 1].min() + points[:, 1].max()) / 2), Example._initial_bag_offset[1], places=5
        )
        np.testing.assert_allclose(
            b._initial_state.joint_q.numpy()[b.finger_indices["right"]], packing.SNACK_OPENINGS["can"]
        )
        self.assertAlmostEqual(float(b._initial_state.joint_q.numpy()[b.neck[1]]), np.pi / 4, places=6)
        self.assertTrue(np.all(points[:, :2].min(axis=0) >= (packing.TABLE_CENTER - packing.TABLE_HALF)[:2]))
        self.assertTrue(np.all(points[:, :2].max(axis=0) <= (packing.TABLE_CENTER + packing.TABLE_HALF)[:2]))

    def test_physical_model_and_contact_options_match(self):
        """Keep material arrays, gripper contacts and solver budgets identical to automatic packing."""
        a, b = self.automatic, self.teleop
        self.assertEqual(self.options[0], self.options[1])
        self.assertEqual((a.frame_dt, a.sim_dt, a.args.substeps), (b.frame_dt, b.sim_dt, b.args.substeps))
        for name in (
            "particle_mass",
            "particle_radius",
            "tet_materials",
            "tet_indices",
            "tet_poses",
            "tri_materials",
            "edge_bending_properties",
            "shape_material_ke",
            "shape_material_kd",
            "shape_material_mu",
            "shape_margin",
            "shape_gap",
            "shape_flags",
            "shape_type",
            "shape_scale",
            "shape_transform",
            "body_mass",
            "body_inertia",
            "body_flags",
        ):
            with self.subTest(field=name):
                np.testing.assert_allclose(
                    getattr(a.model, name).numpy(), getattr(b.model, name).numpy(), rtol=2e-5, atol=1e-10
                )
        for name in ("soft_contact_ke", "soft_contact_kd", "soft_contact_mu"):
            self.assertEqual(getattr(a.model, name), getattr(b.model, name))
        self.assertEqual(a.model.shape_collision_filter_pairs, b.model.shape_collision_filter_pairs)
        for field in ("particle_color_groups", "body_color_groups"):
            for group_a, group_b in zip(getattr(a.model, field), getattr(b.model, field), strict=True):
                np.testing.assert_array_equal(group_a.numpy(), group_b.numpy())
        for index in self.options[0]["collision_options"]["rigid_soft_full_surface_shape_indices"]:
            source_a, source_b = a.model.shape_source[index], b.model.shape_source[index]
            if isinstance(source_a, newton.Mesh):
                np.testing.assert_array_equal(source_a.vertices, source_b.vertices)
                np.testing.assert_array_equal(source_a.indices, source_b.indices)
                self.assertIsNotNone(source_a.sdf)
                self.assertIsNotNone(source_b.sdf)
                self.assertEqual(source_a.sdf.shape_margin, source_b.sdf.shape_margin)
                for field in ("sparse_voxel_size", "coarse_voxel_size", "center", "half_extents"):
                    np.testing.assert_array_equal(
                        np.asarray(getattr(source_a.sdf.data, field)), np.asarray(getattr(source_b.sdf.data, field))
                    )

    def test_gripper_closure_preserves_paper_clearance(self):
        """Use separate paper and snack grasp openings for both input modes."""
        for hand, control in self.teleop.inputs.items():
            minimum = packing.SUPPORT_OPENING if hand == "left" else packing.SNACK_OPENINGS["can"]
            np.testing.assert_allclose(control.mapper.coordinates(1), minimum)
            np.testing.assert_allclose(control.mapper.solve(skeleton(0.015)), minimum)
            np.testing.assert_allclose(control.mapper.coordinates(0), packing.OPEN)

    def test_soft_cube_settles_and_resets_without_inverted_tetrahedra(self):
        """Keep the free bread cube supported and preserve its volume under gravity."""
        b = self.teleop
        b.reset_physics(source="test")
        self.addCleanup(b.reset_physics, source="test")
        initial = b.state_0.particle_q.numpy()
        indices = b.model.tet_indices.numpy()

        def volumes(points):
            tetrahedra = points[indices].astype(np.float64)
            return np.linalg.det(tetrahedra[:, 1:] - tetrahedra[:, :1]) / 6

        rest = volumes(initial)
        self.assertEqual(len(indices), 320)
        self.assertTrue(np.all(rest > 0))
        masses = b.model.particle_mass.numpy()
        self.assertGreater(masses[: b.bag_particle_count].sum(), 0.15)
        np.testing.assert_allclose(masses[b.soft_cube_start :].sum(), 0.12005, rtol=1e-4)
        self.assertTrue(np.all(b.model.body_mass.numpy()[b.objects] > 0.2))
        for _ in range(120):
            b.step()
        points = b.state_0.particle_q.numpy()
        cube = points[b.soft_cube_start : b.soft_cube_end]
        self.assertTrue(np.isfinite(points).all())
        self.assertGreater((volumes(points) / rest).min(), 0.8)
        self.assertGreater(cube[:, 2].min(), packing.TABLE_Z - 0.002)
        self.assertLess(cube[:, 2].min(), packing.TABLE_Z + 0.003)
        b.reset_physics(source="test")
        np.testing.assert_array_equal(b.state_0.particle_q.numpy(), initial)

    def test_same_commands_produce_matching_contact_motion(self):
        """Use the same physical stepping path and reproduce contact motion after reset."""
        a, b = self.aligned, self.teleop
        self.assertFalse(b.args.no_cuda_graph)
        for _ in range(3):
            a.step()
            b.frame_start.assign(a.frame_start)
            b.frame_end.assign(a.frame_end)
            b._advance_physics()
            actual, expected = b.state_0.particle_q.numpy(), a.state_0.particle_q.numpy()
            np.testing.assert_allclose(actual[: b.bag_particle_count], expected[: b.bag_particle_count], atol=1e-4)
            np.testing.assert_allclose(actual[b.soft_cube_start :], expected[b.soft_cube_start :], atol=1e-4)
            np.testing.assert_allclose(b.state_0.body_q.numpy(), a.state_0.body_q.numpy(), atol=1e-4)
        self.assertIsNotNone(b.graph)
        b.reset_physics(source="test")
        b.step()
        b.test_post_step()

    def test_teleop_graph_tracks_both_input_modes_after_reset(self):
        """Refresh graph targets from both hands and retain independent grippers after reset."""
        b = self.teleop
        for mode in ("controllers", "hands"):
            b.reset_physics(source="test")
            previous = b.ik_q.numpy()[0].copy()
            for sequence in range(40):
                value = frame(sequence, mode=mode, position=(sequence * 0.0005, 0, 0), span=0.015)
                right = replace(value.controllers["left"], handedness="right", trigger_value=0)
                b.xr_state.update(
                    replace(
                        value,
                        stream_id=mode,
                        received_monotonic=time.monotonic(),
                        controllers={"left": value.controllers["left"], "right": right},
                        hands={
                            "left": value.hands["left"],
                            "right": replace(value.hands["left"], joints=skeleton(0.10)),
                        },
                    )
                )
                b.step()
                b.test_post_step()
            self.assertIsNotNone(b.ik_graph)
            current = b.ik_q.numpy()[0]
            self.assertGreaterEqual(float(current[b.neck[1]]), np.deg2rad(20) - 1e-6)
            self.assertLessEqual(float(current[b.neck[1]]), np.pi / 4 + 1e-6)
            self.assertGreater(np.linalg.norm(current[b.arm_indices] - previous[b.arm_indices]), 1e-4)
            np.testing.assert_allclose(current[b.finger_indices["left"]], packing.SUPPORT_OPENING, atol=1e-6)
            np.testing.assert_allclose(current[b.finger_indices["right"]], packing.OPEN, atol=1e-6)
            for sequence in range(40, 70):
                value = replace(
                    value,
                    sequence=sequence,
                    stream_id=mode,
                    received_monotonic=time.monotonic(),
                    controllers={"left": value.controllers["left"], "right": replace(right, trigger_value=1)},
                    hands={"left": value.hands["left"], "right": replace(value.hands["left"], joints=skeleton(0.015))},
                )
                b.xr_state.update(value)
                b.step()
                b.test_post_step()
                self.assertGreaterEqual(float(b.ik_q.numpy()[0, b.finger_indices["right"]].min()), 0.030 - 1e-6)
            np.testing.assert_allclose(b.ik_q.numpy()[0, b.finger_indices["right"]], 0.030, atol=1e-6)


class TestPackingHeadGaze(unittest.TestCase):
    def setUp(self):
        self.example = Example.__new__(Example)
        self.example.head = 1
        self.example.ee = (2, 3)
        model = SimpleNamespace(
            joint_label=["NECK1", "NECK2"],
            joint_q_start=wp.array([0, 1, 2], dtype=int, device="cpu"),
            joint_qd_start=wp.array([0, 1, 2], dtype=int, device="cpu"),
            joint_q=wp.array([0.0, 0.0], device="cpu"),
            joint_limit_lower=wp.array([-np.pi / 2, -0.698], device="cpu"),
            joint_limit_upper=wp.array([np.pi / 2, np.pi / 4], device="cpu"),
            body_label=["head_yaw_j1_link", "head_pitch_j2_link"],
        )
        self.example.head_control = W1HeadController(
            model, "cpu", wp.quat_identity(), body_names=tuple(model.body_label)
        )
        self.joints = wp.zeros(2, device="cpu")
        self.bodies = np.array(
            [
                (0, 0, 1.55, 0, 0, 0, 1),
                (0, 0, 1.6, 0, 0, 0, 1),
                (0.5, 0.3, 1.1, 0, 0, 0, 1),
                (0.5, -0.3, 1.1, 0, 0, 0, 1),
            ]
        )

    def settle(self, pose):
        for _ in range(120):
            self.example._update_head_gaze(self.bodies, pose)
            previous = self.example.head_control.targets.copy()
            self.example.head_control.write_targets(self.joints, 1 / 60)
            self.assertLessEqual(
                np.max(np.abs(self.example.head_control.targets - previous)), HEAD_MAX_SPEED_RADIANS_S / 60 + 1e-7
            )
        return self.joints.numpy().copy()

    def test_pitch_watches_both_hands_while_yaw_tracks_headset(self):
        """Ignore headset pitch in either view while preserving left/right tracking."""
        for view in ("observer", "robot-first-person"):
            self.example.view_mode = view
            for yaw in (-0.4, 0.4):
                for pitch in (-0.5, 0.5):
                    orientation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), yaw) * wp.quat_from_axis_angle(
                        wp.vec3(1, 0, 0), pitch
                    )
                    actual = self.settle(Pose(np.zeros(3), np.asarray(orientation)))
                    hands = np.array([Example._tcp_pose(self.bodies[i]).position for i in self.example.ee])
                    direction = hands.mean(axis=0) - self.bodies[self.example.head, :3]
                    expected = np.arctan2(-direction[2], np.linalg.norm(direction[:2]))
                    np.testing.assert_allclose(actual, (yaw, expected), atol=1e-6)

    def test_hand_height_pitch_limits_and_tracking_loss(self):
        """Stay looking down when hands rise and hold yaw when tracking disappears."""
        self.bodies[2:, 2] += 0.2
        orientation = np.asarray(wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.4))
        original = self.settle(Pose(np.zeros(3), orientation))
        self.bodies[2:, 2] -= 1
        lowered = self.settle(None)
        self.assertGreater(lowered[1], original[1])
        np.testing.assert_allclose(lowered, (0.4, np.pi / 4), atol=1e-6)
        self.bodies[2:, 2] += 3
        raised = self.settle(None)
        np.testing.assert_allclose(raised, (0.4, np.deg2rad(20)), atol=1e-6)


class TestRightGraspLimit(unittest.TestCase):
    def setUp(self):
        self.example = Example.__new__(Example)
        self.example.ee = (-1, 0)
        self.example.objects = [1, 2]
        self.example.kinds = ("can", "carton")
        self.example._right_grasp_target = None
        self.example.half = np.array(((0.0325, 0.0325, 0.059), (0.024, 0.030, 0.059)))
        self.example.state_0 = SimpleNamespace(particle_q=wp.array([(10, 0, 0)], dtype=wp.vec3, device="cpu"))
        self.example.bag_particle_count = self.example.soft_cube_start = self.example.soft_cube_end = 1
        pose = Pose(np.array((0, 0, 0.125)), np.array((0, 0, 0, 1)))
        mapper = ParallelGripperRetargeter(ASSET, side="right")
        self.control = GripperInput("right", mapper, pose, np.array((0.045, 0.045)))
        self.example.inputs = {"right": self.control}
        self.bodies = np.array(((0, 0, 0, 0, 0, 0, 1), (0.01, 0, 0.125, 0, 0, 0, 1), (0.2, 0, 0.125, 0, 0, 0, 1)))

    def test_closing_uses_snack_limit_and_latches_until_release(self):
        """Keep full closure at the selected snack's scripted grasp opening until release."""
        self.control.jaws = self.control.mapper.coordinates(1)
        self.example._update_right_grasp_limit(self.bodies)
        np.testing.assert_allclose(self.control.jaws, 0.030)
        self.bodies[[1, 2], :3] = self.bodies[[2, 1], :3]
        self.example.state_0.particle_q.assign(np.array(((0.002, 0, 0.125),), dtype=np.float32))
        self.example._update_right_grasp_limit(self.bodies)
        np.testing.assert_allclose(self.control.jaws, 0.030)
        self.control.jaws = self.control.mapper.coordinates(0)
        self.example._update_right_grasp_limit(self.bodies)
        self.control.jaws = self.control.mapper.solve(skeleton(0.015))
        self.example._update_right_grasp_limit(self.bodies)
        np.testing.assert_allclose(self.control.jaws, 0.028)

    def test_no_nearby_snack_uses_conservative_can_limit(self):
        """Prevent full closure to the paper-grasp range when no snack is in reach."""
        self.bodies[1:, 0] += 1
        self.control.jaws = self.control.mapper.coordinates(1)
        self.example._update_right_grasp_limit(self.bodies)
        np.testing.assert_allclose(self.control.jaws, 0.030)
        self.assertIsNone(self.example._right_grasp_target)

    def test_soft_cube_grasp_retains_clearance_beside_bag(self):
        """Treat cube particles separately and keep a cube grasp latched beside the bag."""
        self.bodies[1:, 0] += 1
        self.example.soft_cube_end = 3
        self.example.state_0.particle_q = wp.array(
            [(10, 0, 0), (-0.035, -0.035, 0.09), (0.035, 0.035, 0.16)], dtype=wp.vec3, device="cpu"
        )
        self.control.jaws = self.control.mapper.coordinates(1)
        self.example._update_right_grasp_limit(self.bodies)
        self.assertEqual(self.example._right_grasp_target, "soft-cube")
        np.testing.assert_allclose(self.control.jaws, packing.SOFT_CUBE_OPENING)
        self.example.state_0.particle_q.assign([(0.001, 0, 0.125), (1, 1, 1), (2, 2, 2)])
        self.example._update_right_grasp_limit(self.bodies)
        np.testing.assert_allclose(self.control.jaws, packing.SOFT_CUBE_OPENING)
        self.control.jaws = self.control.mapper.coordinates(0)
        self.example._update_right_grasp_limit(self.bodies)
        self.control.jaws = self.control.mapper.coordinates(1)
        self.example._update_right_grasp_limit(self.bodies)
        np.testing.assert_allclose(self.control.jaws, packing.SUPPORT_OPENING)

    def test_either_hand_limits_cube_compression_for_both_inputs(self):
        """Keep at least 95 percent of the cube width between the collision meshes."""
        self.example.ee = (0, 0)
        self.bodies[1:, 0] += 1
        self.example.soft_cube_end = 3
        self.example.state_0.particle_q = wp.array(
            [(10, 0, 0), (-0.035, -0.035, 0.09), (0.035, 0.035, 0.16)], dtype=wp.vec3, device="cpu"
        )
        for hand in ("left", "right"):
            mapper = ParallelGripperRetargeter(ASSET, side=hand)
            pose = Pose(self.control.position, self.control.orientation)
            self.example.inputs[hand] = GripperInput(hand, mapper, pose, mapper.coordinates(0))
            control = self.example.inputs[hand]
            for optical in (False, True):
                control.jaws = mapper.solve(skeleton(0.015)) if optical else mapper.coordinates(1)
                self.example._update_grasp_limit(hand, self.bodies)
                # Each mesh protrudes approximately 0.701 mm inward past its slider origin.
                self.assertGreaterEqual(float(control.jaws.sum()) - 0.001403, 0.95 * packing.SOFT_CUBE_SIZE)
                self.assertLess(float(control.jaws.sum()), packing.SOFT_CUBE_SIZE)

    def test_bag_grasp_uses_left_hand_limit_until_release(self):
        """Allow a nearby bag grasp to close fully and keep its limit until release."""
        self.bodies[1:, 0] += 1
        self.example.state_0.particle_q.assign(np.array(((0.01, 0, 0.125),), dtype=np.float32))
        self.control.jaws = self.control.mapper.coordinates(1)
        self.example._update_right_grasp_limit(self.bodies)
        np.testing.assert_allclose(self.control.jaws, packing.SUPPORT_OPENING)
        self.bodies[1, 0] = 0.01
        self.example._update_right_grasp_limit(self.bodies)
        np.testing.assert_allclose(self.control.jaws, packing.SUPPORT_OPENING)
        self.control.jaws = self.control.mapper.coordinates(0)
        self.example._update_right_grasp_limit(self.bodies)
        self.control.jaws = self.control.mapper.solve(skeleton(0.015))
        self.example._update_right_grasp_limit(self.bodies)
        np.testing.assert_allclose(self.control.jaws, 0.030)

    def test_snack_in_gripper_takes_priority_over_adjacent_bag(self):
        """Keep snack protection when its volume contains the TCP beside a bag wall."""
        self.example.state_0.particle_q.assign(np.array(((0.002, 0, 0.125),), dtype=np.float32))
        self.control.jaws = self.control.mapper.coordinates(1)
        self.example._update_right_grasp_limit(self.bodies)
        np.testing.assert_allclose(self.control.jaws, 0.030)


class TestPackingGeometry(unittest.TestCase):
    def test_visible_snack_primitives_are_exported(self):
        """Export both plain snacks at their physical size, omitting hidden packaging."""
        builder = newton.ModelBuilder()
        can, carton = builder.add_body(), builder.add_body()
        transform = wp.transform((0.1, 0.2, 0.3), wp.quat_identity())
        builder.add_shape_cylinder(can, radius=0.0325, half_height=0.059, xform=transform)
        builder.add_shape_box(carton, hx=0.024, hy=0.030, hz=0.059)
        builder.add_shape_mesh(
            can,
            mesh=newton.Mesh.create_box(0.1),
            cfg=newton.ModelBuilder.ShapeConfig(is_visible=False),
        )
        builder.add_shape_box(-1, hx=1.0, hy=0.5, hz=0.05)
        for point in ((0, 0, 0), (1, 0, 0), (0, 1, 0)):
            builder.add_particle(point, (0, 0, 0), 1.0)
        example = Example.__new__(Example)
        example.model = builder.finalize(device="cpu")
        example.state_0 = example.model.state()
        example.objects = (can, carton)
        example.faces = np.array(((0, 1, 2),), dtype=np.int32)
        example.paper_faces = 1
        example.bag_particle_count = example.soft_cube_start = example.soft_cube_end = 3
        example.soft_cube_faces = np.empty((0, 3), dtype=np.int32)
        example._soft_cube_mesh = None
        example._static_boxes, example._bag_meshes = [], []

        payload = example._build_webxr_geometry()
        header_size = struct.unpack_from("<I", payload, 4)[0]
        header = json.loads(payload[8 : 8 + header_size])
        snacks = [shape for shape in header["shapes"] if shape["role"] == "snack"]
        self.assertEqual([shape["body"] for shape in snacks], [can, carton])
        self.assertEqual(len(example._static_boxes), 1)
        self.assertEqual(len(example._bag_meshes), 1)
        np.testing.assert_allclose(snacks[0]["position"], (0.1, 0.2, 0.3))
        np.testing.assert_allclose(snacks[0]["orientation"], (0, 0, 0, 1))
        data_offset = (8 + header_size + 3) & ~3
        for shape, half in zip(snacks, ((0.0325, 0.0325, 0.059), (0.024, 0.030, 0.059)), strict=True):
            mesh = header["meshes"][shape["mesh"]]
            vertices = (
                np.frombuffer(
                    payload,
                    dtype="<f4",
                    count=mesh["vertexCount"] * 6,
                    offset=data_offset + mesh["vertexByteOffset"],
                ).reshape(-1, 6)[:, :3]
                * shape["scale"]
            )
            np.testing.assert_allclose(vertices.min(axis=0), -np.array(half), atol=1e-6)
            np.testing.assert_allclose(vertices.max(axis=0), half, atol=1e-6)


class TestPackingInput(unittest.TestCase):
    def setUp(self):
        """Seed both V030 grippers at their measured TCPs and jaw openings."""
        self.pose = Pose(np.array((0.3, 0.25, 1.2)), np.array((0, 0, 0, 1)))
        self.jaws = np.array((0.03, 0.03))
        self.inputs = {
            side: GripperInput(side, ParallelGripperRetargeter(ASSET, side=side), self.pose, self.jaws)
            for side in ("left", "right")
        }

    def update(self, value):
        """Consume one sample with measured robot state independent of requested motion."""
        for control in self.inputs.values():
            control.update(value, self.pose, self.jaws)

    def test_controller_clutch_rotation_and_independent_jaws(self):
        """Move only the clutched hand and map its trigger to symmetric closed jaws."""
        self.update(frame())
        value = frame(1, position=(0.1, 0, 0))
        rotated = replace(value.controllers["left"].pose, orientation=np.array((0, 0, 0.2, np.sqrt(0.96))))
        value = replace(value, controllers={"left": replace(value.controllers["left"], pose=rotated)})
        self.update(value)
        np.testing.assert_allclose(self.inputs["left"].position, self.pose.position + np.array((0.1, 0, 0)))
        np.testing.assert_allclose(self.inputs["left"].orientation, rotated.orientation, atol=1e-6)
        np.testing.assert_allclose(self.inputs["left"].jaws, (0, 0))
        np.testing.assert_allclose(self.inputs["right"].jaws, self.jaws)
        np.testing.assert_allclose(self.inputs["right"].position, self.pose.position)

    def test_optical_pinch_and_loss_require_new_activation(self):
        """Hold measured poses and jaw openings through loss until a fresh wrist baseline arrives."""
        self.update(frame(mode="hands", span=0.015))
        np.testing.assert_allclose(self.inputs["left"].jaws, (0, 0), atol=1e-7)
        self.update(frame(1, mode="hands", position=(0.08, 0, 0), span=0.10))
        np.testing.assert_allclose(self.inputs["left"].jaws, (0.05, 0.05))
        self.update(None)
        np.testing.assert_allclose(self.inputs["left"].jaws, self.jaws)
        self.update(frame(2, mode="hands", position=(1, 0, 0)))
        self.assertEqual(self.inputs["left"].status, "paused")
        np.testing.assert_allclose(self.inputs["left"].position, self.pose.position)
        self.update(frame(3, mode="hands", position=(1, 0, 0), activation=2))
        self.assertEqual(self.inputs["left"].status, "tracking")
        np.testing.assert_allclose(self.inputs["left"].position, self.pose.position)

    def test_visibility_mode_switch_and_invalid_skeleton_hold(self):
        """Disarm hidden sessions and reject skeleton jumps before they move the robot."""
        self.update(frame(mode="hands"))
        self.update(frame(1, mode="hands", position=(0.5, 0, 0)))
        self.assertEqual(self.inputs["left"].status, "invalid-tracking")
        np.testing.assert_allclose(self.inputs["left"].position, self.pose.position)
        self.update(replace(frame(2), visibility_state="hidden"))
        np.testing.assert_allclose(self.inputs["left"].jaws, self.jaws)
        self.update(frame(3, position=(0.8, 0, 0)))
        np.testing.assert_allclose(self.inputs["left"].position, self.pose.position)
        value = frame(4, mode="hands", activation=2)
        bad = replace(value.hands["left"], joints=np.zeros((25, 3)))
        self.update(replace(value, hands={"left": bad}))
        self.assertEqual(self.inputs["left"].status, "invalid-tracking")

    def test_duplicate_frame_and_released_clutch_do_not_advance(self):
        """Avoid reapplying duplicate input and hold immediately when the grip is released."""
        value = frame(mode="hands")
        self.update(value)
        self.update(replace(value, hands={"left": replace(value.hands["left"], joints=skeleton(0.015))}))
        np.testing.assert_allclose(self.inputs["left"].jaws, (0.05, 0.05))
        value = frame(1)
        self.update(replace(value, controllers={"left": replace(value.controllers["left"], clutch=False)}))
        self.assertFalse(self.inputs["left"].retargeter.active)
        np.testing.assert_allclose(self.inputs["left"].position, self.pose.position)

    def test_new_controller_stream_reanchors(self):
        """Re-anchor a reconnected controller instead of applying its old world offset."""
        self.update(frame())
        self.update(frame(1, position=(0.1, 0, 0)))
        self.update(replace(frame(2, position=(2, 0, 0)), stream_id="reconnected"))
        np.testing.assert_allclose(self.inputs["left"].position, self.pose.position)


if __name__ == "__main__":
    unittest.main()

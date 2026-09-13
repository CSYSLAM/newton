# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the experimental paper-shell material independently of robot poses."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

import newton
from newton.examples.mjvbdv2.example_mjvbd_v2_popcorn import (
    BOWL_FRONT,
    CARRY_PITCH,
    CUP,
    CUP_GRIP_FORCE,
    CUP_GRIP_MAX_OFFSET,
    CUP_HEIGHT,
    CUP_RADIUS_SCALE,
    GRAIN_CONTACT_KE,
    HANDLE,
    LEFT_THUMB_APPROACH,
    MACHINE,
    MACHINE_PLAN,
    POUR_PITCH,
    RIGID_CONTACT_BETA,
    SCOOP_ENTRY_PITCH,
    SCOOP_FLOOR_Z,
    STATION_FRAME,
    STATION_ROTATION,
    TCP,
    TOOL_FEEDBACK_MAX_ANGLE,
    TRAY_CONTACT_KE,
    Example,
    _all_grasp_digits_in_contact,
    _bounds_overlap_table,
    _copy_elbow_reference,
    _cup_rim_indices,
    _grasp_finger_commands,
    _grasp_force_targets,
    _grasp_offset_update,
    _inside_cup,
    _inside_scoop,
    _lift_speed_factor,
    _mesh_intersects_table,
    _right_hand_rotation,
    _scoop_floor_height_offset,
    _WristTargetError,
    cup_mesh,
    measure_grasp_pressure,
    popcorn_mesh,
    scoop_bowl_panels,
)
from newton.examples.mjvbdv2.support.paper_shell_material import plastic_return, update_paper_hinges
from newton.solvers import SolverMJVBDV2


@wp.kernel
def sample_return(values: wp.array[wp.vec2], out: wp.array[wp.vec2]):
    i = wp.tid()
    out[i] = plastic_return(values[i][0], values[i][1], 0.3, 0.2)


class TestPaperShell(unittest.TestCase):
    def test_unreachable_recovery_holds_actual_robot_not_failed_iterate(self):
        """Keep physical state untouched while a bounded path pause unwinds IK."""
        example = Example.__new__(Example)
        example.sim_time = 10.0
        example.grasp_wait = 0.0
        example.robot_coords = 2
        example.ik_q = wp.array([[9.0, 9.0]], dtype=float, device="cpu")
        example.state_0 = SimpleNamespace(joint_q=wp.array([0.1, 0.2, 3.0], dtype=float, device="cpu"))
        before = example.state_0.joint_q.numpy().copy()
        example._set_wrist_targets = Mock(return_value=([], 1.0))
        error = _WristTargetError("unreachable", side=1)
        example._solve_wrist_ik = Mock(side_effect=error)
        previous = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), math.radians(10))
        example._recover_tool_ik(previous, error)
        np.testing.assert_array_equal(example.ik_q.numpy()[0], before[:2])
        np.testing.assert_array_equal(example.state_0.joint_q.numpy(), before)
        self.assertTrue(example._tool_ik_holding)
        self.assertEqual(example.sim_time, 10.0)
        self.assertAlmostEqual(example.tool_wait, 1 / 60)

    def test_grasp_keeps_support_while_other_digits_recover(self):
        """Avoid opening the supporting digit when neighboring contact weakens."""
        forces = np.array((7.5, 5.0, 2.8, 0.15, 0.1))
        offset = np.zeros(5)
        update = _grasp_offset_update(offset, forces, True)
        self.assertEqual(update[1], 0.0)
        self.assertTrue(np.all(update[3:] > 0.0))
        forces[1] = 9.0
        self.assertLess(_grasp_offset_update(offset, forces, True)[1], 0.0)

    def test_table_broad_phase_matches_full_mesh_screen(self):
        """Preserve the old exact decision across translated, rotated meshes."""
        rng = np.random.default_rng(514)
        example = Example.__new__(Example)
        points = rng.uniform(-0.06, 0.06, (30, 3))
        faces = np.arange(30).reshape(-1, 3)
        example.table_guard_meshes = [(0, points, faces)]
        example.table_guard_bodies = np.array([0])
        example.table_guard_centers = ((points.min(axis=0) + points.max(axis=0)) * 0.5)[None]
        example.table_guard_extents = ((points.max(axis=0) - points.min(axis=0)) * 0.5)[None]
        example.robot_coords = 1
        example.sim_time = 0.0
        example.table_guard_q = Mock()
        example.ik_model = SimpleNamespace(joint_qd=None, body_label=["test_arm"])
        for _ in range(100):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            quaternion = wp.quat_from_axis_angle(wp.vec3(*axis), float(rng.uniform(-math.pi, math.pi)))
            rotation = np.asarray(wp.quat_to_matrix(quaternion)).reshape(3, 3)
            position = rng.uniform((0.50, -0.85, 0.75), (1.65, 0.85, 0.96))
            pose = np.concatenate((position, np.asarray(quaternion)))[None]
            example.table_guard_state = SimpleNamespace(body_q=SimpleNamespace(numpy=lambda pose=pose: pose))
            expected = _mesh_intersects_table(points @ rotation.T + position, faces)
            with patch.object(newton, "eval_fk"):
                if expected:
                    with self.assertRaisesRegex(RuntimeError, "Robot/table clearance rejected"):
                        example._check_robot_table(np.zeros(1))
                else:
                    example._check_robot_table(np.zeros(1))

    def test_tool_recovery_retimes_only_robot_commands(self):
        """Unwind feedback gradually and preserve the physical simulation clock."""
        example = Example.__new__(Example)
        example.sim_time = 10.0
        example.grasp_wait = 0.0
        example.ik_q = wp.zeros((1, 2), dtype=float, device="cpu")
        example._set_wrist_targets = Mock(return_value=([], 1.0))
        example._solve_wrist_ik = Mock()
        previous = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), math.radians(10))
        example._recover_tool_ik(previous, RuntimeError("unreachable"))
        self.assertEqual(example.sim_time, 10.0)
        self.assertAlmostEqual(example.trajectory_time, 10.0 - 1 / 60)
        angle = 2 * math.acos(float(example.tool_orientation_correction[3]))
        self.assertAlmostEqual(angle, math.radians(9.75), delta=5e-6)
        example._set_wrist_targets.assert_called_once_with(update_feedback=False)
        example.tool_wait = 2.0
        with self.assertRaisesRegex(RuntimeError, "unreachable"):
            example._recover_tool_ik(previous, RuntimeError("unreachable"))

    def test_outlet_feedback_limits_robot_motion_and_does_not_edit_props(self):
        """Rate-limit tool-tip feedback and avoid reintegrating IK trial targets."""
        example = Example.__new__(Example)
        example.cup_grasp = example.cup_pickup_frame = example.tool_grasp = None
        example.sim_time = 17.0
        example.rotations = [wp.quat_identity(), wp.quat_identity()]
        example.offsets = [np.zeros(3), np.zeros(3)]
        example.position_goals, example.rotation_goals = [Mock(), Mock()], [Mock(), Mock()]
        example._targets = lambda: (CUP, HANDLE, 1.0, 0.5)
        example.scoop_body = 0
        points, faces = cup_mesh()
        example.rim_indices = _cup_rim_indices(points, faces)
        pose = wp.transform(wp.vec3(*HANDLE), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.6))
        example.state_0 = SimpleNamespace(
            body_q=wp.array([pose], dtype=wp.transform, device="cpu"),
            particle_q=wp.array(points + CUP, dtype=wp.vec3, device="cpu"),
        )
        before = example.state_0.body_q.numpy().copy()
        previous = np.zeros(3)
        for _ in range(20):
            example._set_wrist_targets()
            self.assertLessEqual(np.linalg.norm(example.outlet_correction - previous), 0.03 / 60 + 1e-10)
            previous = example.outlet_correction.copy()
        example._set_wrist_targets(update_feedback=False)
        np.testing.assert_array_equal(example.outlet_correction, previous)
        np.testing.assert_array_equal(example.state_0.body_q.numpy(), before)

    def test_loaded_cup_pressure_does_not_open_supporting_fingers(self):
        """Hold a loaded grasp while retaining finite overload release."""
        offset = np.radians((1, 1, 1, 1, 1))
        loaded = np.array((6.0, 3.5, 3.0, 2.5, 2.0))
        np.testing.assert_array_equal(_grasp_offset_update(offset, loaded, True), offset)
        self.assertTrue(np.all(_grasp_offset_update(offset, np.full(5, 20.0), True) < offset))

    def test_table_triangle_screen_rejects_bounds_false_positive(self):
        """Distinguish a slanted wrist above the edge from actual mesh overlap."""
        clear = np.array(((0.55, 0, 0.83), (0.585, 0, 0.87), (0.55, 0.03, 0.87)))
        face = np.array(((0, 1, 2),))
        self.assertTrue(_bounds_overlap_table(clear.min(axis=0), clear.max(axis=0)))
        self.assertFalse(_mesh_intersects_table(clear, face))
        crossing = np.array(((0.50, 0, 0.82), (0.7, 0.9, 0.82), (0.7, -0.9, 0.82)))
        self.assertTrue(_mesh_intersects_table(crossing, face))
        self.assertTrue(_mesh_intersects_table(crossing, face[:, ::-1]))

    def test_table_guard_includes_front_edge_and_solid_thickness(self):
        """Reject tabletop overlap while allowing a wrist outside the front edge."""
        self.assertTrue(_bounds_overlap_table(np.array((0.60, 0.2, 0.82)), np.array((0.65, 0.3, 0.88))))
        self.assertFalse(_bounds_overlap_table(np.array((0.40, 0.2, 0.82)), np.array((0.55, 0.3, 0.88))))
        self.assertFalse(_bounds_overlap_table(np.array((0.60, 0.2, 0.85)), np.array((0.65, 0.3, 0.95))))

    def test_scoop_grip_is_overhand(self):
        """Keep the palm direction downward instead of returning to an underhand grip."""
        palm_direction = wp.quat_rotate(_right_hand_rotation(), wp.vec3(0, 0, 1))
        self.assertLess(float(palm_direction[2]), -0.5)

    def test_entry_pitch_preserves_physical_blade_clearance(self):
        """Tilt the scoop without lowering any real panel below its original floor."""
        vertices = np.concatenate([np.asarray(panel.vertices) for panel in scoop_bowl_panels()])
        for pitch in (0.0, SCOOP_ENTRY_PITCH / 2, SCOOP_ENTRY_PITCH):
            offset = _scoop_floor_height_offset(vertices, pitch)
            rotated_z = -math.sin(pitch) * vertices[:, 0] + math.cos(pitch) * vertices[:, 2] + offset
            self.assertAlmostEqual(float(rotated_z.min()), float(vertices[:, 2].min()), places=6)

    def test_ik_backtracks_only_unexecuted_tool_feedback(self):
        """Reduce unreachable feedback without touching physical state or reintegrating it."""
        example = Example.__new__(Example)
        previous = np.array([[0.1, 0.2]], dtype=np.float32)
        example.ik_q = wp.array(previous, dtype=float, device="cpu")
        example.tool_orientation_correction = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.2)
        example._set_wrist_targets = Mock(return_value=(["feasible"], 1.0))
        calls = []

        def solve(targets):
            calls.append(targets)
            np.testing.assert_array_equal(example.ik_q.numpy(), previous)
            if len(calls) == 1:
                example.ik_q.assign(previous + 1)
                raise RuntimeError("unreachable feedback")

        example._solve_wrist_ik = solve
        example._solve_wrist_ik_with_feedback(["unreachable"])
        self.assertEqual(calls, [["unreachable"], ["feasible"]])
        example._set_wrist_targets.assert_called_once_with(update_feedback=False)
        np.testing.assert_allclose(
            np.asarray(example.tool_orientation_correction),
            np.asarray(wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.1)),
            atol=1e-6,
        )

    def test_ik_still_rejects_unreachable_nominal_path(self):
        """Do not silence a failure after all attitude feedback has been removed."""
        example = Example.__new__(Example)
        previous = np.zeros((1, 2), dtype=np.float32)
        example.ik_q = wp.array(previous, dtype=float, device="cpu")
        example.tool_orientation_correction = wp.quat_identity()
        example._set_wrist_targets = Mock(return_value=(["nominal"], 1.0))
        example._solve_wrist_ik = Mock(side_effect=RuntimeError("unreachable nominal path"))
        with self.assertRaisesRegex(RuntimeError, "unreachable nominal path"):
            example._solve_wrist_ik_with_feedback(["first"])
        self.assertEqual(example._solve_wrist_ik.call_count, 4)
        np.testing.assert_array_equal(example.ik_q.numpy(), previous)

    def test_ik_backtracking_does_not_unwind_previous_grasp_correction(self):
        """Avoid a large wrist jump when only the newest feedback increment fails."""
        example = Example.__new__(Example)
        example.ik_q = wp.zeros((1, 2), dtype=float, device="cpu")
        old = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.2)
        example.tool_orientation_correction = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.22)
        example._set_wrist_targets = Mock(return_value=(["reduced increment"], 1.0))
        example._solve_wrist_ik = Mock(side_effect=[RuntimeError("unreachable"), None])
        example._solve_wrist_ik_with_feedback(["full increment"], previous_correction=old)
        np.testing.assert_allclose(
            np.asarray(example.tool_orientation_correction),
            np.asarray(wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.21)),
            atol=1e-6,
        )

    def test_grasp_coverage_requires_each_physical_digit(self):
        """Reject a two-finger pinch or a missing digit in the hold metric."""
        self.assertTrue(_all_grasp_digits_in_contact(CUP_GRIP_FORCE))
        for digit in range(5):
            forces = CUP_GRIP_FORCE.copy()
            forces[digit] = 0.0
            self.assertFalse(_all_grasp_digits_in_contact(forces))
        self.assertFalse(_all_grasp_digits_in_contact(np.full(5, np.nan)))

    def test_grasp_uses_distal_travel_after_knuckle_limit(self):
        """Keep all commanded joints within the original robot limits."""
        mcp, pip = _grasp_finger_commands(math.radians(71), math.radians(3), math.radians(8))
        np.testing.assert_allclose(np.degrees([mcp, pip]), [75, 11])
        mcp, pip = _grasp_finger_commands(math.radians(60), math.radians(20), math.radians(3))
        np.testing.assert_allclose(np.degrees([mcp, pip]), [63, 20])
        mcp, pip = _grasp_finger_commands(math.radians(75), math.radians(119), math.radians(8))
        np.testing.assert_allclose(np.degrees([mcp, pip]), [75, 120])

    def test_grasp_holds_pose_within_load_band(self):
        """Do not reflexively release a grip under moderate added contact load."""
        offset = np.zeros(5)
        np.testing.assert_array_equal(_grasp_offset_update(offset, 1.5 * CUP_GRIP_FORCE, True), offset)
        self.assertTrue(np.all(_grasp_offset_update(offset, 5 * CUP_GRIP_FORCE, True) < 0))

    def test_grasp_feedback_preserves_rate_and_pressure_bounds(self):
        """Allow distal-finger travel without increasing preload or finger speed."""
        offset = np.radians(np.full(5, 12.0))
        forces = np.zeros(5)
        result = _grasp_offset_update(offset, forces, True)
        self.assertGreater(result[0], offset[0])
        np.testing.assert_array_equal(result[1:3], offset[1:3])
        self.assertTrue(np.all(result[3:] > offset[3:]))
        self.assertLessEqual(np.max(np.abs(result - offset)), 0.06 / 60 + 1e-12)
        np.testing.assert_array_equal(_grasp_offset_update(offset, CUP_GRIP_FORCE, True), offset)
        upper = CUP_GRIP_MAX_OFFSET
        np.testing.assert_array_equal(_grasp_offset_update(upper, forces, True), upper)
        lower = np.radians(np.full(5, -8.0))
        np.testing.assert_array_equal(_grasp_offset_update(lower, np.full(5, 10.0), True), lower)
        np.testing.assert_array_equal(offset, np.radians(np.full(5, 12.0)))

    def test_elbow_reference_is_read_only(self):
        """Copy the previous elbow positions without moving any body."""
        values = np.array([[0.2, 0.3, 1, 0, 0, 0, 1], [0.2, -0.3, 1.1, 0, 0, 0, 1]], dtype=np.float32)
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            poses = wp.array(values, dtype=wp.transform, device=device)
            left, right = (wp.zeros(1, dtype=wp.vec3, device=device) for _ in range(2))
            wp.launch(_copy_elbow_reference, 1, [poses, 0, 1, left, right], device=device)
            np.testing.assert_array_equal(left.numpy()[0], values[0, :3])
            np.testing.assert_array_equal(right.numpy()[0], values[1, :3])
            np.testing.assert_array_equal(poses.numpy(), values)

    def test_scoop_payload_uses_curved_inner_surface(self):
        """Include bottom-row centers and reject centers below curved side panels."""
        points = np.array(
            [[0.20, 0, 0.006], [0.20, 0.045, 0.010], [0.20, 0.045, 0.030], [0.30, 0, 0.01], [0.20, 0, 0.05]]
        )
        points[:, 2] += SCOOP_FLOOR_Z
        np.testing.assert_array_equal(_inside_scoop(points), [True, False, True, False, False])

    def test_grasp_gate_requires_sustained_opposition(self):
        """Delay lift for missing contacts without pausing the physical clock."""
        example = Example.__new__(Example)
        example.sim_time = 3.5
        example.grasp_wait = example.grasp_ready_time = 0.0
        example.grasp_ready = False
        example.grasp_force_filtered = np.array((2.0, 0.5, 0.0, 0.5, 0.0))
        for _ in range(30):
            example._wait_for_grasp()
            example.sim_time += 1 / 60
        self.assertFalse(example.grasp_ready)
        self.assertAlmostEqual(example.trajectory_time, 3.5)
        example.grasp_force_filtered[2] = 0.5
        example.grasp_force_filtered[4] = 0.5
        for _ in range(10):
            example._wait_for_grasp()
            example.sim_time += 1 / 60
        self.assertFalse(example.grasp_ready)
        example.grasp_force_filtered[0] = 0.0
        example._wait_for_grasp()
        example.sim_time += 1 / 60
        self.assertEqual(example.grasp_ready_time, 0.0)
        example.grasp_force_filtered[0] = 2.0
        for _ in range(20):
            example._wait_for_grasp()
            example.sim_time += 1 / 60
        self.assertTrue(example.grasp_ready)
        self.assertGreater(example.trajectory_time, 3.5)

    def test_grasp_preload_waits_for_all_four_fingers(self):
        """Avoid full thumb preload against an unsupported cup during closure."""
        forces = np.array((0.2, 0.5, 0.5, 0.0, 0.5))
        before = forces.copy()
        target = _grasp_force_targets(forces, False)
        self.assertAlmostEqual(target[0], 0.2)
        np.testing.assert_array_equal(target[1:], CUP_GRIP_FORCE[1:])
        np.testing.assert_array_equal(forces, before)
        forces[1:] = CUP_GRIP_FORCE[1:]
        np.testing.assert_array_equal(_grasp_force_targets(forces, False), CUP_GRIP_FORCE)
        np.testing.assert_array_equal(_grasp_force_targets(np.zeros(5), True), CUP_GRIP_FORCE)

    def test_lift_speed_requires_each_digit(self):
        """Pace robot targets without modifying contact forces or physical state."""
        self.assertEqual(_lift_speed_factor(CUP_GRIP_FORCE), 1.0)
        self.assertEqual(_lift_speed_factor(np.zeros(5)), 0.0)
        for digit in range(5):
            forces = CUP_GRIP_FORCE.copy()
            forces[digit] = 0.0
            before = forces.copy()
            self.assertEqual(_lift_speed_factor(forces), 0.0)
            np.testing.assert_array_equal(forces, before)
        self.assertAlmostEqual(_lift_speed_factor(0.325 * CUP_GRIP_FORCE), 0.5)
        self.assertEqual(_lift_speed_factor(np.full(5, np.nan)), 0.0)

    def test_lift_pacing_only_delays_trajectory_clock(self):
        """Keep physics time advancing when contact pressure slows the robot lift."""
        example = Example.__new__(Example)
        example.grasp_ready = True
        example.sim_time, example.grasp_wait, example.lift_wait = 4.0, 0.0, 0.0
        example.grasp_force_filtered = np.zeros(5)
        example._wait_for_grasp()
        self.assertEqual(example.sim_time, 4.0)
        example.sim_time += 1 / 60
        self.assertAlmostEqual(example.trajectory_time, 4.0)
        example.lift_wait = 8.0
        with self.assertRaisesRegex(AssertionError, "not maintained"):
            example._wait_for_grasp()

    def test_grasp_gate_rejects_timeout(self):
        """Fail a missing grasp instead of lifting on a scripted schedule."""
        example = Example.__new__(Example)
        example.sim_time = 11.5
        example.grasp_wait = 8.0
        example.grasp_ready_time = 0.0
        example.grasp_ready = False
        example.grasp_force_filtered = np.zeros(5)
        with self.assertRaisesRegex(AssertionError, "opposing contact forces"):
            example._wait_for_grasp()

    def test_scoop_lowers_outside_the_grain_pile(self):
        """Keep the full front lip before the first grain row while lowering."""
        example = Example.__new__(Example)
        example.sim_time = 8.5
        _, position, _, _ = example._targets()
        local = np.asarray(wp.transform_point(wp.transform_inverse(STATION_FRAME), wp.vec3(*position)))
        self.assertLess(local[0] + BOWL_FRONT, MACHINE_PLAN[0] - 0.095)
        self.assertGreater(local[2] + SCOOP_FLOOR_Z, MACHINE_PLAN[2] + 0.029)
        bottom = min(float(np.asarray(panel.vertices)[:, 2].min()) for panel in scoop_bowl_panels())
        clearance = local[2] + bottom - (MACHINE_PLAN[2] + 0.029)
        self.assertGreater(clearance, 0.001)
        self.assertLess(clearance, 0.002)

    def test_loaded_bowl_clears_glass_before_turn(self):
        """Retract all metal vertices before the station-relative yaw starts."""
        example = Example.__new__(Example)
        example.sim_time = 13.5
        _, position, _, _ = example._targets()
        local = np.asarray(wp.transform_point(wp.transform_inverse(STATION_FRAME), wp.vec3(*position)))
        rotation = np.asarray(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(0, 1, 0), CARRY_PITCH))).reshape(3, 3)
        vertices = np.concatenate([np.asarray(panel.vertices) for panel in scoop_bowl_panels()])
        front = (vertices @ rotation.T + local)[:, 0].max()
        self.assertLess(front, MACHINE_PLAN[0] - 0.151)

    def test_grasp_sensor_reads_barycentric_normal_force(self):
        """Measure active hand contacts without changing particles or bodies."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):

                def array(values, dtype, device=device):
                    return wp.array(values, dtype=dtype, device=device)

                positions = array([[0, 0, 0.009], [0.01, 0, 0.007], [0, 0.01, 0.012]], wp.vec3)
                bodies = array([wp.transform_identity()], wp.transform)
                before = positions.numpy()
                measured = wp.zeros(5, dtype=float, device=device)
                wp.launch(
                    measure_grasp_pressure,
                    4,
                    [
                        array([3], int),
                        array([0, 1, 2, 0], int),
                        array([[0, -1, -1], [0, 1, 2], [0, -1, -1], [0, -1, -1]], wp.vec3i),
                        array([[1, 0, 0], [0.25, 0.5, 0.25], [1, 0, 0], [1, 0, 0]], wp.vec3),
                        array([[0, 0, 0]] * 4, wp.vec3),
                        array([[0, 0, 1]] * 4, wp.vec3),
                        array([1000, 2000, 1000, 1000], float),
                        array([0, 0, 0], int),
                        array([0.001, 0.001, 0.001], float),
                        array([0, 1, -1], int),
                        bodies,
                        positions,
                        array([0.01, 0.01, 0.01], float),
                        measured,
                    ],
                    device=device,
                )
                np.testing.assert_allclose(measured.numpy(), [2, 4.5, 0, 0, 0], atol=1e-5)
                np.testing.assert_array_equal(positions.numpy(), before)
                np.testing.assert_array_equal(bodies.numpy()[0], np.asarray(wp.transform_identity()))

    def test_left_grasp_preopposes_thumb_before_approach(self):
        """Avoid sweeping the thumb through the cup during finger closure."""
        for name in ("HAND_INDEX", "INDEX_PIP"):
            self.assertEqual(Example._finger_angle(0, name, 1.0, 0.0), 0.0)
            self.assertEqual(Example._finger_angle(0, name, 1.0, 0.5), 0.5)
            self.assertEqual(Example._finger_angle(0, name, 1.0, 1.0), 1.0)
            self.assertEqual(Example._finger_angle(1, name, 1.0, 0.0), 1.0)
            self.assertEqual(
                Example._finger_angle(1, name, 1.0, 1.0),
                1.0 + (math.radians(1.2) if name.endswith("PIP") else 0.0),
            )
        # A small final thumb squeeze opposes finger pressure after approach;
        # it is not the old 90-degree sweep through the cup wall.
        thumb = math.radians(1.2)
        self.assertAlmostEqual(Example._finger_angle(0, "HAND_THUMB1", thumb, 0), math.radians(LEFT_THUMB_APPROACH))
        self.assertAlmostEqual(Example._finger_angle(0, "HAND_THUMB1", thumb, 0.88), math.radians(LEFT_THUMB_APPROACH))
        self.assertAlmostEqual(Example._finger_angle(0, "HAND_THUMB1", thumb, 1), thumb)
        opposed = math.radians(79.0547)
        self.assertAlmostEqual(Example._finger_angle(0, "HAND_THUMB2", opposed, 0), opposed)
        self.assertAlmostEqual(Example._finger_angle(0, "HAND_THUMB2", opposed, 0.88), opposed)
        self.assertAlmostEqual(Example._finger_angle(0, "HAND_THUMB2", opposed, 1), opposed)

    def test_right_thumb_preload_is_gradual(self):
        angle = math.radians(20.5)
        for closure in (0.0, 0.5, 1.0):
            self.assertAlmostEqual(
                Example._finger_angle(1, "HAND_THUMB1", angle, closure),
                angle + math.radians(2.0) * closure,
            )
            self.assertEqual(Example._finger_angle(1, "HAND_THUMB2", angle, closure), angle)
        self.assertLess(Example._finger_angle(1, "HAND_THUMB1", angle, 1.0), math.radians(50))

    def test_scoop_panels_are_thin_closed_collision_geometry(self):
        """Keep the curved metal panels closed and above their table support."""
        panels = scoop_bowl_panels()
        self.assertEqual(len(panels), 13)
        for panel in panels:
            vertices = np.asarray(panel.vertices)
            faces = np.asarray(panel.indices).reshape(-1, 3)
            triangles = vertices[faces]
            volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
            self.assertGreater(volume, 0.0)
            # The full rear closure is larger than an individual curved strip.
            self.assertLess(volume, 6e-6)
            self.assertGreaterEqual(float(vertices[:, 2].min()), SCOOP_FLOOR_Z - 0.00121)

    def test_initial_scoop_clears_machine_riser(self):
        """Keep the complete pan outside the warmer before the grasp begins."""
        local = np.concatenate([np.asarray(panel.vertices) for panel in scoop_bowl_panels()])
        rotation = np.asarray(wp.quat_to_matrix(STATION_ROTATION)).reshape(3, 3)
        points = local @ rotation.T + HANDLE
        lower, upper = MACHINE - np.array((0.18, 0.22, 0.06)), MACHINE + np.array((0.18, 0.22, 0.0))
        separated = (points.max(axis=0) < lower - 0.001) | (points.min(axis=0) > upper + 0.001)
        self.assertTrue(np.any(separated))

    def test_withdrawal_pan_stays_in_front_of_machine(self):
        local = np.concatenate([np.asarray(panel.vertices) for panel in scoop_bowl_panels()])
        rim = np.array((0.70, 0.0, 1.067))
        for time in np.linspace(21.0, 29.0, 81):
            center, pitch, yaw = Example._withdraw_tool_pose(rim, time)
            rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw) * wp.quat_from_axis_angle(wp.vec3(0, 1, 0), pitch)
            matrix = np.asarray(wp.quat_to_matrix(rotation)).reshape(3, 3)
            points = local @ matrix.T + center
            self.assertLess(float(points[:, 0].max()), float(MACHINE[0] - 0.15 - 0.01))

    def test_pour_withdrawal_is_continuous(self):
        """Clear the cup continuously without flipping the scoop or cup target."""
        example = Example.__new__(Example)
        example.cup_grasp = example.cup_pickup_frame = example.tool_grasp = None
        example.rotations = [wp.quat_identity(), wp.quat_identity()]
        example.offsets = [np.zeros(3), np.zeros(3)]
        example.position_goals, example.rotation_goals = [Mock(), Mock()], [Mock(), Mock()]
        samples = []
        for time in (21.0, 27.0, 28.0, 29.0):
            example.sim_time = time
            self.assertAlmostEqual(float(example._targets()[3]), 1.0 if time == 21.0 else 0.0)
            samples.append(example._set_wrist_targets()[0])
        for current in samples[1:]:
            np.testing.assert_allclose(current[0][0], samples[0][0][0])
            np.testing.assert_allclose(np.asarray(current[1][1]), np.asarray(samples[1][1][1]))
        for endpoint in (21.0, 22.0, 27.0):
            example.sim_time = endpoint - 1e-5
            before = example._set_wrist_targets()[0]
            example.sim_time = endpoint + 1e-5
            after = example._set_wrist_targets()[0]
            for a, b in zip(before, after, strict=True):
                np.testing.assert_allclose(a[0], b[0], atol=1e-7)
                np.testing.assert_allclose(np.asarray(a[1]), np.asarray(b[1]), atol=1e-7)
        np.testing.assert_allclose(samples[-1][1][0], samples[-2][1][0])

    def test_grip_slip_uses_wrist_frame(self):
        """Distinguish tool roll about the grasp from actual axial shaft slip."""
        center = wp.vec3(0.08, 0.02, -0.15)
        reference = wp.transform(center, wp.quat_identity())
        rolled = wp.transform(center, wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.5))
        inverse_reference, inverse_rolled = wp.transform_inverse(reference), wp.transform_inverse(rolled)
        # The old frame-dependent metric falsely classified this pure roll as loss.
        self.assertGreater(np.linalg.norm(np.asarray(inverse_reference)[:3] - np.asarray(inverse_rolled)[:3]), 0.04)
        np.testing.assert_allclose(Example._grip_center_in_wrist(inverse_rolled), np.asarray(center), atol=1e-7)
        shifted = wp.transform(center + wp.vec3(0, 0.05, 0), wp.transform_get_rotation(rolled))
        self.assertAlmostEqual(
            float(np.linalg.norm(Example._grip_center_in_wrist(wp.transform_inverse(shifted)) - np.asarray(center))),
            0.05,
            places=6,
        )

    def test_commanded_pour_has_no_shaft_roll(self):
        """The actual wrist command pitches forward, not just the aiming helper."""
        example = Example.__new__(Example)
        example.cup_grasp = example.cup_pickup_frame = example.tool_grasp = None
        example.rotations = [wp.quat_identity(), STATION_ROTATION]
        example.offsets = [np.zeros(3), np.zeros(3)]
        example.position_goals, example.rotation_goals = [Mock(), Mock()], [Mock(), Mock()]
        previous_forward_z = 1.0
        for time in np.linspace(16.5, 21.0, 40):
            example.sim_time = float(time)
            commands, _ = example._set_wrist_targets()
            rotation = commands[1][1]
            lateral = wp.quat_rotate(rotation, wp.vec3(0, 1, 0))
            forward = wp.quat_rotate(rotation, wp.vec3(1, 0, 0))
            self.assertAlmostEqual(float(lateral[2]), 0.0, places=6)
            self.assertLessEqual(float(forward[2]), previous_forward_z + 1e-6)
            previous_forward_z = float(forward[2])
        self.assertAlmostEqual(previous_forward_z, -math.sin(POUR_PITCH), places=6)

    def test_pour_edge_stays_over_deformed_rim(self):
        """Pitch through the front opening without any axial shaft roll."""
        rim = np.array((0.78, 0.20, 0.99))
        bowl = np.concatenate([np.asarray(panel.vertices) for panel in scoop_bowl_panels()])
        for yaw in (0.0, np.pi / 6, np.pi / 2):
            for pitch in np.linspace(CARRY_PITCH, POUR_PITCH, 15):
                center = Example._pour_tool_position(rim, float(pitch), yaw)
                rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw) * wp.quat_from_axis_angle(
                    wp.vec3(0, 1, 0), float(pitch)
                )
                edge = center + np.asarray(wp.quat_rotate(rotation, wp.vec3(BOWL_FRONT, 0, SCOOP_FLOOR_Z)))
                fraction = (pitch - CARRY_PITCH) / (POUR_PITCH - CARRY_PITCH)
                height = (1 - fraction) * 0.077 + fraction * 0.025
                np.testing.assert_allclose(
                    edge - rim, [-0.02 * fraction * math.cos(yaw), -0.02 * fraction * math.sin(yaw), height], atol=1e-7
                )
                world_bowl = bowl @ np.asarray(wp.quat_to_matrix(rotation)).reshape(3, 3).T + center
                self.assertGreater(float(world_bowl[:, 2].min() - rim[2]), 0.02)
                # Positive pitch points the open lip downward, not backward;
                # the cross-bowl direction stays horizontal throughout.
                lateral = np.asarray(wp.quat_rotate(rotation, wp.vec3(0, 1, 0)))
                self.assertAlmostEqual(float(lateral[2]), 0.0, places=6)
                if pitch > 0:
                    forward = np.asarray(wp.quat_rotate(rotation, wp.vec3(1, 0, 0)))
                    self.assertLess(float(forward[2]), 0.0)

    def test_pour_aim_compensates_observed_tool_angle(self):
        """Aim the real outlet despite an orientation error without moving the prop."""
        rim = np.array((0.70, 0.0, 1.07))
        observed = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.3)
        center = Example._pour_tool_position(rim, POUR_PITCH, 0.0, observed_rotation=observed)
        outlet = center + np.asarray(wp.quat_rotate(observed, wp.vec3(BOWL_FRONT, 0, SCOOP_FLOOR_Z)))
        np.testing.assert_allclose(outlet - rim, (-0.02, 0, 0.025), atol=1e-7)
        np.testing.assert_array_equal(rim, (0.70, 0.0, 1.07))

    def test_popcorn_convex_collision_geometry(self):
        """Keep the rendered kernel closed and its rigid mass/inertia positive."""
        mesh = popcorn_mesh()
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.indices).reshape(-1, 3)
        self.assertTrue(np.isfinite(vertices).all())
        self.assertGreater(len(faces), 12)
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, 2)
        triangles = vertices[faces]
        volume = np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
        self.assertGreater(volume, 1e-7)
        builder = newton.ModelBuilder()
        body = builder.add_body()
        shape = builder.add_shape_convex_hull(body, mesh=mesh, cfg=newton.ModelBuilder.ShapeConfig(density=90))
        self.assertEqual(builder.shape_type[shape], newton.GeoType.CONVEX_MESH)
        self.assertGreater(builder.body_mass[body], 1e-5)
        self.assertTrue(np.all(np.linalg.eigvalsh(np.asarray(builder.body_inertia[body]).reshape(3, 3)) > 0))

    def test_small_grains_settle_on_tray(self):
        """Keep sub-gram rigid grains above their support without numerical bounce."""
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.002
        SolverMJVBDV2.register_custom_attributes(builder)
        fixed = builder.add_body(is_kinematic=True, label="prescribed_tool")
        tray = newton.ModelBuilder.ShapeConfig(ke=TRAY_CONTACT_KE, kd=10, mu=0.7, margin=0.0005, gap=0.002)
        builder.add_shape_box(
            fixed, xform=wp.transform(wp.vec3(1, 0, 0.1), wp.quat_identity()), hx=0.01, hy=0.01, hz=0.01, cfg=tray
        )
        builder.add_shape_box(
            -1, xform=wp.transform(wp.vec3(0, 0, -0.01), wp.quat_identity()), hx=0.3, hy=0.3, hz=0.01, cfg=tray
        )
        grain = newton.ModelBuilder.ShapeConfig(
            ke=GRAIN_CONTACT_KE, kd=1, mu=0.45, density=90, margin=0.0003, gap=0.002
        )
        ids = []
        for x in (-0.03, 0.03):
            body = builder.add_body(xform=wp.transform(wp.vec3(x, 0, 0.02), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.01, cfg=grain)
            ids.append(body)
        builder.color()
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                model = builder.finalize(device=device)
                solver = SolverMJVBDV2(
                    model,
                    mujoco_articulations=(0,),
                    joint_mode="kinematic",
                    contact_mode="full",
                    vbd_options={
                        "iterations": 16,
                        "rigid_contact_history": False,
                        "rigid_contact_hard": False,
                        "rigid_avbd_contact_beta": RIGID_CONTACT_BETA,
                    },
                    collision_options={
                        "broad_phase": "sap",
                        "rigid_contact_max": 128,
                        "include_static_kinematic_pairs": False,
                    },
                )
                state, output = model.state(), model.state()
                control = model.control()
                for _ in range(240):
                    state.clear_forces()
                    solver.step(state, output, control, None, 1 / 480)
                    state, output = output, state
                positions, velocities = state.body_q.numpy()[ids], state.body_qd.numpy()[ids]
                self.assertTrue(np.isfinite(positions).all())
                self.assertTrue(np.all(positions[:, 2] > 0.0105))
                self.assertTrue(np.all(positions[:, 2] < 0.0110))
                self.assertLess(float(np.linalg.norm(velocities[:, :3], axis=1).max()), 0.01)

    def test_dropped_tool_cannot_calibrate_transport(self):
        """Reject a dropped tool instead of treating a stale maximum lift as grasp."""
        example = Example.__new__(Example)
        example.scoop_body, example.wrists = 0, [1, 2]
        example.sim_time = 5.0
        example._targets = lambda: (CUP, HANDLE + np.array((0, 0, 0.2)), 1.0, 0.0)
        poses = np.tile(np.array([*HANDLE, 0, 0, 0, 1], dtype=np.float32), (3, 1))
        state_q = wp.array(poses, dtype=wp.transform, device="cpu")
        example.state_0 = SimpleNamespace(body_q=state_q)
        with self.assertRaisesRegex(AssertionError, "remain lifted"):
            example._calibrate_tool_grasp()
        poses[0, 2] += 0.2
        state_q.assign(poses)
        example._calibrate_tool_grasp()
        self.assertEqual(example.tool_grasp[3], 5.0)
        np.testing.assert_allclose(state_q.numpy(), poses)

    def test_tool_calibration_is_continuous(self):
        """Start from the observed wrist pose and smoothly level the tool target."""
        example = Example.__new__(Example)
        example.cup_grasp = None
        example.cup_pickup_frame = None
        example.rotations = [wp.quat_identity(), wp.quat_identity()]
        example.offsets = [np.zeros(3), np.zeros(3)]
        example.position_goals = [Mock(), Mock()]
        example.rotation_goals = [Mock(), Mock()]
        example._targets = lambda: (CUP, HANDLE, 1.0, 0.0)
        relative = wp.transform(wp.vec3(0.02, -0.04, 0.08), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.4))
        observed = wp.transform(
            wp.vec3(*(HANDLE + np.array((0.01, 0, 0.03)))),
            wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.2),
        )
        example.tool_grasp = (relative, observed, HANDLE, 5.0)
        for time, expected_tool in ((5.0, observed), (6.5, wp.transform(wp.vec3(*HANDLE), STATION_ROTATION))):
            example.sim_time = time
            targets, _ = example._set_wrist_targets()
            expected_wrist = wp.transform_multiply(expected_tool, relative)
            np.testing.assert_allclose(targets[1][0], np.asarray(wp.transform_point(expected_wrist, TCP)), atol=1e-6)
            np.testing.assert_allclose(
                np.asarray(targets[1][1]), np.asarray(wp.transform_get_rotation(expected_wrist)), atol=1e-6
            )

    def test_tool_attitude_feedback_is_bounded_and_read_only(self):
        """Correct only the robot target and keep the observed dynamic tool untouched."""
        example = Example.__new__(Example)
        example.cup_grasp = example.cup_pickup_frame = None
        example.sim_time = 6.5
        example.rotations = [wp.quat_identity(), wp.quat_identity()]
        example.offsets = [np.zeros(3), np.zeros(3)]
        example.position_goals, example.rotation_goals = [Mock(), Mock()], [Mock(), Mock()]
        example._targets = lambda: (CUP, HANDLE, 1.0, 0.0)
        example.scoop_body = 0
        observed = wp.transform(wp.vec3(*HANDLE), STATION_ROTATION * wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.6))
        poses = wp.array([observed], dtype=wp.transform, device="cpu")
        example.state_0 = SimpleNamespace(body_q=poses)
        example.tool_grasp = (wp.transform_identity(), wp.transform(wp.vec3(*HANDLE), STATION_ROTATION), HANDLE, 0.0)
        example.tool_orientation_correction = wp.quat_identity()
        before = poses.numpy()
        for _ in range(60):
            example._set_wrist_targets()
            angle = 2 * math.acos(min(abs(float(example.tool_orientation_correction[3])), 1.0))
            # acos magnifies float32 quaternion rounding at small angles.
            self.assertLessEqual(angle, TOOL_FEEDBACK_MAX_ANGLE + 5e-6)
        self.assertGreater(angle, TOOL_FEEDBACK_MAX_ANGLE - math.radians(1))
        np.testing.assert_array_equal(poses.numpy(), before)

    def test_cup_transport_preserves_the_established_grasp(self):
        """Translate the wrist without rolling the cup or changing physical state."""
        example = Example.__new__(Example)
        example.scoop_vertices = np.concatenate([np.asarray(panel.vertices) for panel in scoop_bowl_panels()])
        example.cup_pickup_frame = None
        points, faces = cup_mesh()
        example.rim_indices = _cup_rim_indices(points, faces)
        example.base_indices = np.flatnonzero(np.isclose(points[:, 2], -CUP_HEIGHT / 2))
        example.rest_cup = points + CUP
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.6)
        matrix = np.asarray(wp.quat_to_matrix(rotation)).reshape(3, 3)
        center = CUP + np.array((0.0, 0.0, 0.16))
        q = points @ matrix.T + center
        wrist = wp.transform(wp.vec3(*(center + np.array((0.01, 0.02, 0.05)))), rotation)
        example.state_0 = SimpleNamespace(
            particle_q=wp.array(q, dtype=wp.vec3, device="cpu"),
            body_q=wp.array([wrist], dtype=wp.transform, device="cpu"),
        )
        example.wrists = [0, 0]
        example._targets = lambda: (center, HANDLE, 1.0, 0.0)
        example.sim_time = 5.0
        example.rotations = [rotation, wp.quat_identity()]
        example.offsets = [np.zeros(3), np.zeros(3)]
        example.position_goals, example.rotation_goals = [Mock(), Mock()], [Mock(), Mock()]
        example.tool_grasp = None
        example._calibrate_cup_grasp()
        targets, _ = example._set_wrist_targets()
        np.testing.assert_allclose(targets[0][0], np.asarray(wp.transform_point(wrist, TCP)), atol=1e-6)
        example.sim_time = 8.0
        shift = np.array((0.02, -0.04, 0.01))
        example._targets = lambda: (center + shift, HANDLE, 1.0, 0.0)
        targets, _ = example._set_wrist_targets()
        np.testing.assert_allclose(targets[0][0], np.asarray(wp.transform_point(wrist, TCP)) + shift, atol=1e-6)
        np.testing.assert_allclose(np.asarray(targets[0][1]), np.asarray(rotation), atol=1e-6)
        np.testing.assert_allclose(example.state_0.particle_q.numpy(), q, atol=1e-7)
        np.testing.assert_array_equal(example.state_0.body_q.numpy()[0], np.asarray(wrist))

    def test_pickup_feedback_freezes_before_lift(self):
        """Observe translation during approach without prescribing cup state."""
        example = Example.__new__(Example)
        example.cup_pickup_frame = example.cup_grasp = example.tool_grasp = None
        example.rotations = [wp.quat_identity(), wp.quat_identity()]
        example.offsets = [np.zeros(3), np.zeros(3)]
        example.position_goals, example.rotation_goals = [Mock(), Mock()], [Mock(), Mock()]
        example._targets = lambda: (CUP, HANDLE, 0.0, 0.0)
        example.rest_cup = cup_mesh()[0] + CUP
        shift = np.array((0.01, -0.02, 0.0), dtype=np.float32)
        actual = example.rest_cup + shift
        example.state_0 = SimpleNamespace(particle_q=wp.array(actual, dtype=wp.vec3, device="cpu"))
        example.sim_time = 1.0
        targets, _ = example._set_wrist_targets()
        np.testing.assert_allclose(targets[0][0], CUP + shift, atol=1e-6)
        np.testing.assert_array_equal(example.state_0.particle_q.numpy(), actual)
        example.state_0.particle_q.assign(actual + shift)
        example.sim_time = 4.0
        targets, _ = example._set_wrist_targets()
        np.testing.assert_allclose(targets[0][0], CUP + shift, atol=1e-6)
        np.testing.assert_array_equal(example.state_0.particle_q.numpy(), actual + shift)

    def test_deformed_cup_containment(self):
        """Reject grains above/beside a cup and follow its deformation and pose."""
        vertices, faces = cup_mesh()
        rim = _cup_rim_indices(vertices, faces)
        points = np.array([[0, 0, 0], [0.032, 0, 0.03], [0, 0, 0.07], [0, 0, -0.07], [0.06, 0, 0]])
        points[:, :2] *= CUP_RADIUS_SCALE
        points[:, 2] *= CUP_HEIGHT / 0.13
        expected = [True, True, False, False, False]
        np.testing.assert_array_equal(_inside_cup(points, vertices, faces, rim), expected)
        transform = np.array([[0, -1, 0], [0.45, 0, 0], [0, 0, 1]])
        offset = np.array([0.3, 0.8, 1.1])
        np.testing.assert_array_equal(
            _inside_cup(points @ transform + offset, vertices @ transform + offset, faces, rim), expected
        )

    def test_unreachable_collider_target_is_rejected(self):
        """Reject IK position, rotation and nonfinite errors before simulation."""
        example = Example.__new__(Example)
        example.sim_time = 0.0
        example.max_ik_position_error = 0.0
        example.wrists = [0, 1]
        example.ik_model = SimpleNamespace(joint_qd=None)
        example.ik_q = [None]
        poses = np.array([[0, 0, 0, 0, 0, 0, 1]] * 2, dtype=np.float32)
        example.ik_state = SimpleNamespace(body_q=wp.array(poses, dtype=wp.transform, device="cpu"))
        target = np.asarray(TCP).copy()
        with patch("newton.eval_fk"):
            example._check_ik([(target, wp.quat_identity())] * 2)
            with self.assertRaisesRegex(RuntimeError, "unreachable"):
                example._check_ik([(target + np.array([0.01, 0, 0]), wp.quat_identity())] * 2)
            with self.assertRaisesRegex(RuntimeError, "unreachable"):
                example._check_ik([(target, wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.2))] * 2)
            with self.assertRaisesRegex(RuntimeError, "unreachable"):
                example._check_ik([(np.full(3, np.nan), wp.quat_identity())] * 2)

    def test_rest_shell_has_no_plastic_flow(self):
        """Keep the constructed shell stress-free, including the bottom seam."""
        points, faces = cup_mesh()
        builder = newton.ModelBuilder()
        builder.add_cloth_mesh(
            pos=wp.vec3(),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=points.tolist(),
            indices=faces.reshape(-1).tolist(),
            density=0.24,
        )
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            model = builder.finalize(device=device)
            reference = wp.clone(model.edge_rest_angle)
            plastic, accumulated = wp.zeros_like(reference), wp.zeros_like(reference)
            wp.launch(
                update_paper_hinges,
                model.edge_count,
                [
                    model.particle_q,
                    model.edge_indices,
                    reference,
                    plastic,
                    accumulated,
                    model.edge_rest_angle,
                    0.3,
                    0.2,
                ],
                device=device,
            )
            np.testing.assert_allclose(plastic.numpy(), 0, atol=1e-7)
            np.testing.assert_allclose(model.edge_rest_angle.numpy(), reference.numpy(), atol=1e-6)

    def test_open_cup_mesh(self):
        """Keep exactly one open rim and a consistently oriented closed bottom."""
        q, faces = cup_mesh()
        self.assertTrue(np.isfinite(q).all())
        edge_counts = {}
        for face in faces:
            self.assertEqual(len(set(face)), 3)
            self.assertGreater(np.linalg.norm(np.cross(q[face[1]] - q[face[0]], q[face[2]] - q[face[0]])), 1e-8)
            for a, b in zip(face, np.roll(face, -1), strict=True):
                key = tuple(sorted((int(a), int(b))))
                edge_counts.setdefault(key, []).append((int(a), int(b)))
        boundary = [edge for edge, pairs in edge_counts.items() if len(pairs) == 1]
        self.assertEqual(len(boundary), 40)
        self.assertTrue(
            all(np.allclose(q[list(edge), 2], CUP_HEIGHT / 2 - 0.0014 * math.sqrt(3) / 2) for edge in boundary)
        )
        self.assertGreater(float(q[:, 2].max()), CUP_HEIGHT / 2 + 0.001)
        self.assertTrue(
            all(np.all(np.linalg.norm(q[list(edge), :2], axis=1) < 0.042 * CUP_RADIUS_SCALE) for edge in boundary)
        )
        self.assertAlmostEqual(float(np.linalg.norm(q[:, :2], axis=1).max()), 0.042 * CUP_RADIUS_SCALE, places=6)
        for pairs in edge_counts.values():
            self.assertLessEqual(len(pairs), 2)
            if len(pairs) == 2:
                self.assertEqual(pairs[0], pairs[1][::-1])
        rim = _cup_rim_indices(q, faces)
        self.assertEqual(set(rim), {vertex for edge in boundary for vertex in edge})
        capped_faces = np.concatenate(
            (faces, np.array([(int(a), len(q), int(b)) for a, b in zip(rim, np.roll(rim, -1), strict=True)]))
        )
        capped_edges = np.concatenate((capped_faces[:, [0, 1]], capped_faces[:, [1, 2]], capped_faces[:, [2, 0]]))
        _, counts = np.unique(np.sort(capped_edges, axis=1), axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, 2)

    def test_return_map_yield_and_dissipation(self):
        """Preserve elastic unloading and dissipate work on either plastic branch."""
        values = np.array([[0.1, 0], [0.6, 0], [-0.6, 0], [0.2, 0.4], [1.2, 0.4]], dtype=np.float32)
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            output = wp.empty(len(values), dtype=wp.vec2, device=device)
            wp.launch(
                sample_return, len(values), [wp.array(values, dtype=wp.vec2, device=device), output], device=device
            )
            result = output.numpy()
            self.assertEqual(result[0, 0], 0)
            self.assertEqual(result[3, 0], 0)
            np.testing.assert_allclose(result[1:3, 0], [0.25, -0.25], atol=1e-7)
            remaining = values[:, 0] - result[:, 0]
            self.assertTrue(np.all(np.abs(remaining) <= 0.3 + 0.2 * result[:, 1] + 1e-7))
            before = 0.5 * values[:, 0] ** 2 + 0.1 * values[:, 1] ** 2
            after = 0.5 * remaining**2 + 0.1 * result[:, 1] ** 2
            self.assertTrue(np.all(after <= before + 1e-7))


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the popcorn slip failure report without running the full scene."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.mjvbdv2.example_mjvbd_v2_popcorn import (
    MACHINE,
    MACHINE_PLAN,
    Example,
    _cup_rim_indices,
    _grasp_offset_update,
    _inside_cup,
    _popcorn_spawn_position,
    cup_mesh,
    popcorn_mesh,
    station_point,
)


class TestPopcornDiagnostics(unittest.TestCase):
    def test_grasp_pressure_feedback_remains_bounded(self):
        """Close unloaded fingers gently and retain bounded overload release."""
        offset = np.full(5, 0.1)
        forces = np.array([4, 0, 1, 1, 1])
        holding = _grasp_offset_update(offset, forces, True)
        self.assertGreater(holding[1], offset[1])
        self.assertAlmostEqual(holding[1] - offset[1], 0.05 / 60)
        for ready in (False, True):
            for load in (0.0, 100.0):
                updated = _grasp_offset_update(offset, np.full(5, load), ready)
                self.assertTrue(np.all(np.abs(updated - offset) <= 0.06 / 60 + 1e-12))

    def test_overload_release_preserves_underloaded_closure(self):
        """Keep closing an underloaded digit while another digit unloads."""
        offsets = np.radians([7, 4, 5, 7, 10])
        updated = _grasp_offset_update(offsets, np.array([5, 0, 45, 14, 1]), True)
        self.assertGreater(updated[1], offsets[1])
        self.assertLess(updated[2], offsets[2])
        np.testing.assert_array_equal(updated[[0, 4]], offsets[[0, 4]])

    def test_double_population_layout(self):
        """Keep all 320 authored hulls clear of the rear baffle and side walls."""
        centers = np.array([_popcorn_spawn_position(i) for i in range(320)])
        self.assertEqual(len(np.unique(centers, axis=0)), 320)
        self.assertTrue(np.isfinite(centers).all())
        distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        self.assertGreater(float(distances.min()), 0.0199)
        for i in range(160):
            if i < 128:
                column, row, layer = i % 8, (i // 8) % 8, i // 64
            else:
                extra = i - 128
                column, row, layer = 3 + extra % 5, (extra // 5) % 8, 2 + extra // 40
            reference = station_point(
                (
                    MACHINE_PLAN[0] - 0.07 + column * 0.020,
                    MACHINE_PLAN[1] - 0.0805 + row * 0.023,
                    MACHINE_PLAN[2] + 0.045 + layer * 0.022,
                )
            )
            np.testing.assert_array_equal(centers[i], reference)
        vertices = np.asarray(popcorn_mesh().vertices)
        rng = np.random.default_rng(37)
        for center in centers - MACHINE:
            angle = rng.uniform(-np.pi, np.pi)
            rng.uniform(0.80, 0.94)
            rng.uniform(0.43, 0.65)
            rotation = np.asarray(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(0, 0, 1), angle))).reshape(3, 3)
            hull = vertices @ rotation.T + center
            # Include the actual grain and tray collision margins.
            self.assertGreater(float(hull[:, 0].min()) - 0.0008, -0.15)
            self.assertLess(float(hull[:, 0].max()) + 0.0008, 0.0975)
            self.assertLess(float(np.abs(hull[:, 1]).max()) + 0.0008, 0.163)
            self.assertGreater(float(hull[:, 2].min()) - 0.0008, 0.029)

    def test_cup_bounds_preserve_ray_parity(self):
        """Reject outside bounds without changing the deformed-shell ray test."""
        rng = np.random.default_rng(519)
        vertices, faces = cup_mesh()
        rim = _cup_rim_indices(vertices, faces)
        for deformation in (0.0, 0.003):
            shell = (vertices + rng.normal(size=vertices.shape) * deformation).astype(np.float32)
            points = rng.uniform(-0.15, 0.15, (700, 3)).astype(np.float32)
            points = np.concatenate((points, shell, np.zeros((1, 3), dtype=np.float32)))
            # Unreferenced vertices expand only the bounds. Faces and the cap
            # stay identical, forcing the reference to ray-test every point.
            reference_shell = np.concatenate((shell, np.full((1, 3), -1.0), np.full((1, 3), 1.0))).astype(np.float32)
            reference = _inside_cup(points, reference_shell, faces, rim)
            actual = _inside_cup(points, shell, faces, rim)
            np.testing.assert_array_equal(actual, reference)

    def test_cup_slip_reports_context_without_changing_state(self):
        """Report the original slip threshold and preserve physical arrays."""
        example = Example.__new__(Example)
        example.sim_time = 25.75
        example.cup_grasp = (wp.transform_identity(), np.zeros(3))
        example.wrists = [0, 1]
        example.grasp_force_filtered = np.arange(1.0, 6.0)
        example.grasp_joint_offset = np.zeros(5)
        example.args = SimpleNamespace(substeps=8, vbd_iterations=8, popcorn_count=160)
        state = {
            "body_q": np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32),
            "particle_q": np.array([[0, 0, -0.0314]], dtype=np.float32),
        }
        before = {name: value.copy() for name, value in state.items()}
        example._read_state = state.__getitem__
        with self.assertRaises(AssertionError) as failure:
            example.step()
        message = str(failure.exception)
        self.assertIn("25.750s: 31.4 mm", message)
        self.assertIn("wrist_local_delta_mm=", message)
        self.assertIn("filtered_digit_force_N=[1.0, 2.0, 3.0, 4.0, 5.0]", message)
        self.assertIn("digit_offset_deg=", message)
        self.assertIn("substeps=8, iterations=8, grains=160", message)
        for name, value in state.items():
            np.testing.assert_array_equal(value, before[name])


if __name__ == "__main__":
    unittest.main()

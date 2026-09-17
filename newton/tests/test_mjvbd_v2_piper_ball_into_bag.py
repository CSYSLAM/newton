# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check the geometric support used by the steel-ball bag acceptance test."""

import unittest
from types import SimpleNamespace

import numpy as np

from newton.examples.mjvbdv2.example_mjvbd_v2_piper_ball_into_bag import (
    Example,
    _inclined_support_distance,
    _sphere_surface_penetration,
    _support_height,
    _surface_winding,
)


class TestBagSupport(unittest.TestCase):
    def test_sphere_crosses_face_without_inside_vertices(self):
        vertices = np.array(((-1, -1, 0), (1, -1, 0), (0, 1, 0)), dtype=float)
        faces = np.array(((0, 1, 2),))
        self.assertAlmostEqual(_sphere_surface_penetration(vertices, faces, np.array((0, 0, 0.01)), 0.03), 0.02)
        self.assertEqual(_sphere_surface_penetration(vertices, faces, np.array((0, 0, 0.04)), 0.03), 0.0)

    def test_pickup_tracking_does_not_move_drop_target(self):
        example = Example.__new__(Example)
        example.initial_tcp = np.zeros(3)
        example.bag_center = np.array((0.02, 0.3, 0.7))
        example.ball_radius = 0.028
        example.args = SimpleNamespace(grip_half_width=0.025)
        for pick in (np.array((0, 0, 0.583)), np.array((0.06, -0.07, 0.583))):
            example.pick = pick
            np.testing.assert_allclose(example._plan(2.0)[0], pick)
            np.testing.assert_allclose(example._plan(4.4)[0][:2], example.bag_center[:2])

    def test_enclosure_rejects_exterior_support(self):
        """A center near an exterior face is not enclosed by that surface."""
        vertices = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=float)
        faces = np.array(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
        self.assertAlmostEqual(abs(_surface_winding(vertices, faces, np.array((0.1, 0.1, 0.1)))), 1.0)
        self.assertAlmostEqual(abs(_surface_winding(vertices, faces, np.array((0.1, 0.1, -0.03)))), 0.0)

    def test_sphere_on_inclined_face(self):
        """Normal sphere clearance differs from vertical clearance on a slope."""
        vertices = np.array(((-1, -1, -1), (1, -1, 1), (0, 1, 0)), dtype=float)
        faces = np.array(((0, 1, 2),))
        point = np.array((-0.02, 0.0, 0.02))
        self.assertAlmostEqual(_inclined_support_distance(vertices, faces, point), np.sqrt(2) * 0.02)
        self.assertAlmostEqual(point[2] - _support_height(vertices, faces, point), 0.04)
        self.assertEqual(_inclined_support_distance(vertices, faces, np.array((5, 0, 6))), float("inf"))

    def test_vertical_wall_is_not_inclined_support(self):
        vertices = np.array(((0, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=float)
        faces = np.array(((0, 1, 2),))
        self.assertEqual(_inclined_support_distance(vertices, faces, np.array((0.03, 0.2, 0.2))), float("inf"))

    def test_support_uses_surface_below_ball(self):
        """Choose the nearest lower surface and reject a ball outside the cloth."""
        base = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=float)
        vertices = np.concatenate([base + np.array((0, 0, z)) for z in (0.55, 0.60, 0.80)])
        faces = np.arange(9).reshape(3, 3)
        self.assertAlmostEqual(_support_height(vertices, faces, np.array((0.2, 0.2, 0.63))), 0.60)
        self.assertEqual(_support_height(vertices, faces, np.array((1.2, 1.2, 0.63))), float("-inf"))
        self.assertEqual(_support_height(vertices, faces, np.array((0.2, 0.2, 0.50))), float("-inf"))

    def test_vertical_triangles_do_not_supply_support(self):
        """Ignore a vertical side wall when checking support underneath the ball."""
        vertices = np.array(((0, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=float)
        self.assertEqual(_support_height(vertices, np.array(((0, 1, 2),)), np.array((0, 0.2, 0.5))), float("-inf"))


if __name__ == "__main__":
    unittest.main()

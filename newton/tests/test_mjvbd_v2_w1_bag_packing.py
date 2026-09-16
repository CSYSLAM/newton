# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check the W1 reference bag's physical topology and moving-frame validation."""

import unittest

import numpy as np

from newton.examples.mjvbdv2.example_mjvbd_v2_w1_bag_packing import (
    ASSETS,
    TABLE_Z,
    bag_frame,
    count_handle_crossings,
    fit_bag,
    lip_deviation,
    snack_extent,
    triangles_overlap_box,
)


class TestW1PaperBag(unittest.TestCase):
    def test_table_intersection_without_inside_vertices(self):
        """Reject a wrist face crossing the tabletop even if all vertices are outside."""
        lower, upper = np.array((-1, -1, -0.1)), np.array((1, 1, 0.0))
        vertices = np.array(((-2.0, -2, -0.05), (2, -2, -0.05), (0, 3, -0.05)))
        triangles = np.array(((0, 1, 2),))
        self.assertTrue(triangles_overlap_box(vertices, triangles, lower, upper))
        vertices[:, 2] = 0.01
        self.assertFalse(triangles_overlap_box(vertices, triangles, lower, upper))
        vertices[:, 2] = -0.05
        vertices[:, 0] += 4
        self.assertFalse(triangles_overlap_box(vertices, triangles, lower, upper))
        diagonal = np.array(((0.9, 1.3, -0.05), (1.3, 0.9, -0.05), (1.3, 1.3, -0.05)))
        self.assertFalse(triangles_overlap_box(diagonal, triangles, lower, upper))

    def test_lip_bowing_ignores_rigid_motion(self):
        """Detect a dented mouth without counting rigid bag tipping as deformation."""
        points = np.column_stack((np.zeros(5), np.linspace(-0.16, 0.16, 5), np.full(5, 0.25)))
        chains = [np.arange(5)]
        position, rotation = bag_frame(0.4)
        moved = points @ rotation.T + position
        self.assertLess(lip_deviation(moved, chains), 1e-6)
        points[2, 0] += 0.025
        self.assertGreater(lip_deviation(points @ rotation.T + position, chains), 0.024)

    def test_handle_wall_crossing_detection(self):
        """Reject a handle edge through paper while allowing nearby separated edges."""
        positions = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0.2, 0.2, -0.1), (0.2, 0.2, 0.1)))
        triangles, edges = np.array(((0, 1, 2),)), np.array(((3, 4),))
        self.assertEqual(count_handle_crossings(positions, triangles, edges), 1)
        positions[3, 2] = 0.02
        self.assertEqual(count_handle_crossings(positions, triangles, edges), 0)

    def test_rest_handles_do_not_intersect_paper(self):
        """Keep folded handles outside the shell before contact detection starts."""
        with np.load(ASSETS / "bag.npz") as data:
            vertices, faces = data["vertices"], data["faces"]
            paper_count, paper_faces = int(data["paper_count"]), int(data["paper_faces"])
        handle = faces[paper_faces:]
        edges = np.concatenate([handle[:, [0, 1]], handle[:, [1, 2]], handle[:, [2, 0]]])
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        edges = edges[np.all(edges >= paper_count, axis=1)]
        self.assertEqual(count_handle_crossings(vertices, faces[:paper_faces], edges), 0)

    def test_can_extent_is_invariant_to_spin(self):
        """Do not reject a contained spinning can using fictitious box corners."""
        half = np.array((0.0325, 0.0325, 0.059))
        angle = np.pi / 4
        rotation = np.array(((np.cos(angle), -np.sin(angle), 0), (np.sin(angle), np.cos(angle), 0), (0, 0, 1)))
        np.testing.assert_allclose(snack_extent("can", rotation, half), half)
        self.assertGreater(snack_extent("carton", rotation, half)[0], 0.045)

    def test_paper_and_handles_form_one_manifold_surface(self):
        """Keep connected handle sleeves and leave only the bag mouth open."""
        with np.load(ASSETS / "bag.npz") as data:
            vertices, faces = data["vertices"], data["faces"]
            paper_count = int(data["paper_count"])
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        self.assertTrue(np.all(counts <= 2))
        np.testing.assert_allclose(vertices[unique[counts == 1], 2], 0.25, atol=1e-7)
        areas = (
            np.linalg.norm(
                np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]),
                axis=1,
            )
            / 2
        )
        self.assertGreater(float(areas.min()), 1e-8)
        neighbors = [set() for _ in vertices]
        for a, b in unique:
            neighbors[a].add(b)
            neighbors[b].add(a)
        reached, pending = {0}, [0]
        while pending:
            for neighbor in neighbors[pending.pop()]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        self.assertEqual(len(reached), len(vertices))
        np.testing.assert_allclose(np.ptp(vertices[:paper_count], axis=0), (0.13, 0.32, 0.25), atol=0.0031)

    def test_laid_bag_and_handles_start_above_table(self):
        """Avoid the initial handle-table overlap that launches a stiff shell."""
        with np.load(ASSETS / "bag.npz") as data:
            rest = data["vertices"]
        position, rotation = bag_frame(0)
        world = rest @ rotation.T + position
        self.assertGreater(float(world[:, 2].min()), TABLE_Z + 0.0012)
        self.assertLess(abs(float(rotation[2, 2])), 1e-6)

    def test_frame_fit_distinguishes_standing_from_deformation(self):
        """Measure tipping independently from the paper body's shape error."""
        with np.load(ASSETS / "bag.npz") as data:
            rest = data["vertices"][: int(data["paper_count"])]
        for progress in (0.0, 0.4, 1.0):
            position, rotation = bag_frame(progress)
            world = rest @ rotation.T + position
            actual_position, actual_rotation, error = fit_bag(rest, world)
            np.testing.assert_allclose(actual_position, position, atol=1e-6)
            np.testing.assert_allclose(actual_rotation, rotation, atol=1e-6)
            self.assertLess(error, 1e-6)
        world[rest[:, 2] > 0.12, 0] += 0.025
        self.assertGreater(fit_bag(rest, world)[2], 0.004)


if __name__ == "__main__":
    unittest.main()

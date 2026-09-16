# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Validate the Blender bag's topology, scale, and deforming visual binding."""

import json
import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.mjvbdv2.support.paper_bag_asset import (
    ASSET_DIR,
    add_paper_bag,
    deform_render_mesh,
    make_paper_bag,
    measure_paper_shape,
    parcel_is_contained,
)


class TestPaperBagAsset(unittest.TestCase):
    def test_dimensions_and_connected_handles(self):
        """Preserve the agreed dimensions and connected load-bearing handles."""
        bag = make_paper_bag()
        end = int(bag.handles[0].min())
        np.testing.assert_allclose(np.ptp(bag.vertices[:end], axis=0), (0.09, 0.20, 0.26), atol=1e-6)
        self.assertAlmostEqual(float(bag.vertices[:, 2].max()), 0.36, delta=0.0021)
        edges = np.concatenate([bag.faces[:, [0, 1]], bag.faces[:, [1, 2]], bag.faces[:, [2, 0]]])
        neighbors = [set() for _ in bag.vertices]
        for a, b in edges:
            neighbors[a].add(b)
            neighbors[b].add(a)
        reached, queue = {0}, [0]
        while queue:
            for v in neighbors[queue.pop()]:
                if v not in reached:
                    reached.add(v)
                    queue.append(v)
        self.assertEqual(len(reached), len(bag.vertices), "Cord loads must have a mesh path to the bag bottom")
        areas = np.linalg.norm(
            np.cross(
                bag.vertices[bag.faces[:, 1]] - bag.vertices[bag.faces[:, 0]],
                bag.vertices[bag.faces[:, 2]] - bag.vertices[bag.faces[:, 0]],
            ),
            axis=1,
        )
        self.assertTrue(np.all(areas > 1e-9))
        unique_edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        self.assertTrue(np.all(counts <= 2))
        # Only the bag mouth is open. Cord ends must share a finite patch with
        # the sheet, rather than pivoting freely around one attachment vertex.
        self.assertTrue(np.all(unique_edges[counts == 1] < end))
        mixed = bag.faces[np.any(bag.faces < end, axis=1) & np.any(bag.faces >= end, axis=1)]
        for handle in bag.handles:
            for ring in (handle[:6], handle[-6:]):
                patch = mixed[np.any(np.isin(mixed, ring), axis=1)]
                self.assertGreaterEqual(len(np.unique(patch[patch < end])), 3)

    def test_reinforcement_preserves_dynamic_mass(self):
        """Give bonded plies their mass and keep the entire paper box dynamic."""
        bag = make_paper_bag()
        builder = newton.ModelBuilder()
        add_paper_bag(builder, bag, (0, 0, 0))
        triangles = bag.vertices[bag.faces]
        areas = (
            np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
        )
        self.assertAlmostEqual(sum(builder.particle_mass), float(np.sum(areas * bag.plies) * 0.18), places=7)
        self.assertTrue(np.all(np.asarray(builder.particle_mass) > 0))
        np.testing.assert_allclose(builder.particle_q, bag.vertices)
        end = int(bag.handles[0].min())
        edges = np.asarray(builder.edge_indices)
        stiffness = np.asarray(builder.edge_bending_properties)[:, 0]
        self.assertTrue(np.all(stiffness[np.max(edges, axis=1) >= end] < stiffness[np.max(edges, axis=1) < end].min()))

    def test_shape_metric_distinguishes_deformation_from_motion(self):
        """Detect a bent wall while accepting rigid transport of the box."""
        bag = make_paper_bag()
        rotated = bag.vertices[:, (1, 0, 2)] * np.array((-1, 1, 1)) + np.array((0.2, 0.3, 0.4))
        self.assertLess(measure_paper_shape(bag, rotated)["shape_max"], 1e-7)
        deformed = rotated.copy()
        deformed[bag.vertices[:, 2] > 0.2, 0] += 0.015
        self.assertGreater(measure_paper_shape(bag, deformed)["shape_rms"], 0.002)

    def test_containment_follows_bag_and_checks_parcel_extent(self):
        """Reject parcels below a lifted bag, across its wall, or above its rim."""
        bag = make_paper_bag()
        rotation = np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1)), dtype=np.float64)
        shift = np.array((0.2, 0.3, 0.7))
        positions = bag.vertices @ rotation.T + shift
        center = np.array(((0.0, 0.0, 0.03),)) @ rotation.T + shift
        self.assertTrue(parcel_is_contained(bag, positions, center, radius=0.023))
        self.assertFalse(parcel_is_contained(bag, positions + np.array((0, 0, 0.14)), center, radius=0.023))
        for local in (((0.03, 0, 0.1),), ((0, 0, 0.24),)):
            point = np.array(local) @ rotation.T + shift
            self.assertFalse(parcel_is_contained(bag, positions, point, radius=0.023))
        # A soft parcel can straddle the mouth while its centroid is inside.
        points = np.array(((0, 0, 0.18), (0, 0, 0.27))) @ rotation.T + shift
        self.assertFalse(parcel_is_contained(bag, positions, points))

    def test_visual_binding_follows_rigid_motion(self):
        """Make Blender render details follow the simulated shell transform."""
        bag = make_paper_bag()
        binding = np.load(ASSET_DIR / "render_binding.npz")
        rest = np.load(ASSET_DIR / "render_mesh.npz")["vertices"]
        rotation = np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1)), dtype=np.float32)
        shift = np.array((0.1, 0.2, 0.4), dtype=np.float32)
        for device in ["cpu", "cuda:0"] if wp.is_cuda_available() else ["cpu"]:
            with self.subTest(device=device):
                q = wp.array(bag.vertices @ rotation.T + shift, dtype=wp.vec3, device=device)
                indices = wp.array(binding["indices"], dtype=wp.vec3i, device=device)
                weights = wp.array(binding["weights"], dtype=wp.vec3, device=device)
                offset = wp.array(binding["offset"], dtype=wp.vec3, device=device)
                out = wp.empty(len(rest), dtype=wp.vec3, device=device)
                wp.launch(deform_render_mesh, len(rest), [q, indices, weights, offset, out], device=device)
                np.testing.assert_allclose(out.numpy(), rest @ rotation.T + shift, atol=2e-6)

    def test_paper_details_sample_local_deformation(self):
        """Do not bridge a bent wall with a corner-bound, wall-sized quad."""
        bag = make_paper_bag()
        mesh = np.load(ASSET_DIR / "render_mesh.npz")
        binding = np.load(ASSET_DIR / "render_binding.npz")
        parts = json.loads((ASSET_DIR / "render_parts.json").read_text())

        def bend(points):
            return 0.008 * (1 - (points[:, 1] / 0.1) ** 2) * np.clip(points[:, 2] / 0.2, 0, 1)

        # An 8 mm inward bow leaves the wall corners still, reproducing the
        # gripper-induced deformation that separated the old straight hem.
        positions = bag.vertices.copy()
        positions[:, 0] += bend(positions)
        out = wp.empty(len(mesh["vertices"]), dtype=wp.vec3, device="cpu")
        wp.launch(
            deform_render_mesh,
            len(out),
            [
                wp.array(positions, dtype=wp.vec3, device="cpu"),
                wp.array(binding["indices"], dtype=wp.vec3i, device="cpu"),
                wp.array(binding["weights"], dtype=wp.vec3, device="cpu"),
                wp.array(binding["offset"], dtype=wp.vec3, device="cpu"),
                out,
            ],
            device="cpu",
        )
        deformed = out.numpy()
        for part in parts:
            if part["material"] != "paper":
                continue
            faces = mesh["faces"][part["start"] : part["end"]]
            rest_centers = mesh["vertices"][faces].mean(axis=1)
            displacement = deformed[faces].mean(axis=1)[:, 0] - rest_centers[:, 0]
            error = np.abs(displacement - bend(rest_centers))
            self.assertLess(float(error.max()), 0.0003, part["name"])


if __name__ == "__main__":
    unittest.main()

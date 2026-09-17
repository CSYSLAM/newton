# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check private contact scheduling ownership and bounded-memory fallback."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjvbd_v2.rigid_contact_groups import BalancedRigidContactGroups, _adjacency, _assign


class TestRigidContactGroups(unittest.TestCase):
    def test_neighbor_topology_invalidation(self):
        """Invalidate on neighbor changes, not row ordering or contact weights."""
        device = "cpu"

        def integers(values):
            return wp.array(values, dtype=int, device=device)

        bodies = integers([0, 1, 2])
        counts = integers([2, 0, 0])
        rows = integers([0, 1, -1, -1, -1, -1])
        points = wp.zeros(2, dtype=wp.vec3, device=device)
        margins = wp.zeros(2, dtype=float, device=device)
        inverse_mass = wp.ones(3, dtype=float, device=device)
        packed = wp.zeros(48, dtype=wp.uint32, device=device)
        overflow = wp.zeros(1, dtype=int, device=device)
        dirty = wp.zeros(1, dtype=int, device=device)
        args = [
            bodies,
            bodies,
            counts,
            rows,
            2,
            bodies,
            integers([0, 0]),
            integers([1, 2]),
            points,
            points,
            points,
            margins,
            margins,
            wp.array([wp.transform_identity()] * 3, dtype=wp.transform, device=device),
            inverse_mass,
            packed,
            overflow,
            dirty,
        ]
        wp.launch(_adjacency, 3, args, device=device)
        self.assertEqual(int(dirty.numpy()[0]), 1)
        dirty.zero_()
        rows.assign([1, 0, -1, -1, -1, -1])
        margins.fill_(0.001)
        wp.launch(_adjacency, 3, args, device=device)
        self.assertEqual(int(dirty.numpy()[0]), 0)
        counts.assign([1, 0, 0])
        wp.launch(_adjacency, 3, args, device=device)
        self.assertEqual(int(dirty.numpy()[0]), 1)
        prior = packed.numpy().copy()
        dirty.zero_()
        inverse_mass.zero_()
        counts.assign([2, 0, 0])
        wp.launch(_adjacency, 3, args, device=device)
        self.assertEqual(int(dirty.numpy()[0]), 0)
        np.testing.assert_array_equal(packed.numpy(), prior)

    def test_membership_ownership_and_overflow(self):
        """Preserve all bodies and shared model colors, including on overflow."""
        devices = [device for device in wp.get_cuda_devices() if device.arch >= 80]
        if not devices:
            self.skipTest("Requires CUDA SM80+")
        for device in devices:
            with self.subTest(device=str(device)):
                builder = newton.ModelBuilder()
                for _ in range(9):
                    body = builder.add_body()
                    builder.add_shape_sphere(body, radius=0.01)
                builder.color()
                model = builder.finalize(device=device)
                original_groups = list(model.body_color_groups)
                original_colors = model.body_colors.numpy().copy()
                grouping = BalancedRigidContactGroups(SimpleNamespace(model=model, device=device))
                args = [
                    grouping.edges,
                    grouping.offsets,
                    grouping.members,
                    grouping.colors,
                    grouping.bodies,
                    grouping.bodies.size,
                    grouping.color_base,
                    grouping.overflow,
                    grouping.dirty,
                    grouping.initialized,
                ]
                wp.launch(_assign, 32, args, block_dim=32, device=device)
                np.testing.assert_array_equal(np.sort(grouping.members.numpy()), np.arange(9))
                np.testing.assert_array_equal(model.body_colors.numpy(), original_colors)
                self.assertEqual(model.body_color_groups, original_groups)
                for color, group in enumerate(grouping.groups):
                    self.assertTrue(np.all(grouping.colors.numpy()[group.numpy()] == color))
                prior = grouping.members.numpy().copy()
                grouping.dirty.zero_()
                grouping.initialized.fill_(2)
                wp.launch(_assign, 32, args, block_dim=32, device=device)
                self.assertEqual(int(grouping.initialized.numpy()[0]), 2)
                np.testing.assert_array_equal(grouping.members.numpy(), prior)
                grouping.dirty.fill_(1)
                wp.launch(_assign, 32, args, block_dim=32, device=device)
                self.assertEqual(int(grouping.initialized.numpy()[0]), 1)
                grouping.overflow.fill_(1)
                wp.launch(_assign, 32, args, block_dim=32, device=device)
                np.testing.assert_array_equal(grouping.members.numpy(), prior)


if __name__ == "__main__":
    unittest.main()

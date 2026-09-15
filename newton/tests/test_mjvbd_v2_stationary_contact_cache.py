# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check fixed-pose contact reuse and mandatory invalidation paths."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjvbd_v2.full_contact_pipeline import MJVBDV2CollisionPipeline
from newton._src.solvers.mjvbd_v2.stationary_contact_cache import _filter_pairs


class TestStationaryContactCache(unittest.TestCase):
    def test_candidate_overflow_is_not_hidden(self):
        """Preserve candidate overflow instead of accepting a truncated cached list."""
        device = "cpu"
        pairs = wp.array([[0, 1]], dtype=wp.vec2i, device=device)
        count = wp.array([2], dtype=int, device=device)
        stable = wp.ones(2, dtype=int, device=device)
        previous = wp.ones((2, 2), dtype=int, device=device)
        current = wp.zeros_like(previous)
        reuse = wp.zeros_like(previous)
        active = wp.empty_like(pairs)
        active_count = wp.zeros(1, dtype=int, device=device)
        old_count = wp.zeros_like(active_count)
        reused = wp.zeros_like(active_count)
        wp.launch(
            _filter_pairs,
            1,
            [pairs, count, stable, previous, current, reuse, active, active_count, old_count, 64, reused],
            device=device,
        )
        np.testing.assert_array_equal(active.numpy(), pairs.numpy())
        self.assertEqual(int(active_count.numpy()[0]), 2)
        self.assertFalse(reuse.numpy().any())

    def test_reuse_move_and_reset(self):
        """Reuse unchanged geometry and recompute when a body moves or history resets."""
        for device in wp.get_devices():
            for matching in (False, True):
                with self.subTest(device=str(device), matching=matching):
                    self._check_reuse(device, matching=matching)

    def _check_reuse(self, device, *, matching):
        """Exercise the same history transitions on CPU and captured CUDA execution."""
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.09), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.1)
        builder.add_ground_plane()
        model = builder.finalize(device=device)
        pipeline = MJVBDV2CollisionPipeline(
            model,
            stationary_rigid_contact_cache=True,
            broad_phase="nxn",
            rigid_contact_max=64,
            **({"contact_matching": "latest"} if matching else {}),
        )
        contacts = pipeline.contacts()
        state = model.state()
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0)
        points = contacts.rigid_contact_point0.numpy()[:count].copy()
        normals = contacts.rigid_contact_normal.numpy()[:count].copy()
        pipeline.collide(state, contacts)
        cache = pipeline._stationary_contact_cache
        graph = None
        if device.is_cuda:
            with wp.ScopedCapture(device=device) as capture:
                pipeline.collide(state, contacts)
            graph = capture.graph
            wp.capture_launch(graph)
        self.assertGreater(int(cache._buffers[-1].numpy()[0]), 0)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), count)
        np.testing.assert_array_equal(contacts.rigid_contact_point0.numpy()[:count], points)
        np.testing.assert_array_equal(contacts.rigid_contact_normal.numpy()[:count], normals)
        poses = state.body_q.numpy()
        poses[body, 2] = 1.0
        state.body_q.assign(poses)
        if graph is None:
            pipeline.collide(state, contacts)
        else:
            wp.capture_launch(graph)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        pipeline.reset_contact_matching()
        self.assertFalse(cache._buffers[4].numpy().any())
        poses[body, 2] = 0.09
        state.body_q.assign(poses)
        pipeline.collide(state, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), count)


if __name__ == "__main__":
    unittest.main()

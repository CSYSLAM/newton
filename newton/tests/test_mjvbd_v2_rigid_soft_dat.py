# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression checks for the PR #4180 rigid-soft DAT port."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2.rigid_soft_dat import _apply_particles
from newton._src.solvers.mjvbd_v2.rigid_soft_dat_kernels import (
    apply_body_truncation_ts,
    apply_rigid_soft_truncation,
    planar_truncation_t,
)
from newton._src.solvers.mjvbd_v2.vbd.particle_vbd_kernels import accelerate_particle_iteration_chebyshev_guarded


@wp.kernel
def _plane_probe(out: wp.array[float]):
    n = wp.vec3(0.0, 0.0, 1.0)
    out[0] = planar_truncation_t(wp.vec3(0.0, 0.0, 0.01), wp.vec3(0.0, 0.0, -0.02), n, wp.vec3(), 0.85)
    out[1] = planar_truncation_t(wp.vec3(0.0, 0.0, -0.01), wp.vec3(0.0, 0.0, -0.02), n, wp.vec3(), 0.85)
    out[2] = planar_truncation_t(wp.vec3(0.0, 0.0, -0.01), wp.vec3(0.0, 0.0, 0.001), n, wp.vec3(), 0.85)


class TestRigidSoftDAT(unittest.TestCase):
    def test_dat_clipping_excludes_further_chebyshev_extrapolation(self):
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                reference = wp.zeros(2, dtype=wp.vec3)
                q = wp.zeros(2, dtype=wp.vec3)
                excluded = wp.zeros(2, dtype=int)
                wp.launch(
                    _apply_particles,
                    2,
                    [
                        reference,
                        wp.array(((0.002, 0, 0), (0.0001, 0, 0)), dtype=wp.vec3),
                        wp.array((0.5, 1.0), dtype=float),
                        0.001,
                        q,
                        excluded,
                    ],
                )
                np.testing.assert_array_equal(excluded.numpy(), (1, 0))
                correction = wp.zeros(2, dtype=wp.vec3)
                wp.launch(
                    accelerate_particle_iteration_chebyshev_guarded,
                    2,
                    [q, reference, wp.full(2, 0.001), wp.ones(2, dtype=int), excluded, 1.5, 1.0, correction],
                )
                np.testing.assert_array_equal(correction.numpy()[0], (0.0, 0.0, 0.0))
                self.assertGreater(float(correction.numpy()[1, 0]), 0.0)

    def test_wrong_side_recovers_but_cannot_deepen(self):
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                out = wp.zeros(3)
                wp.launch(_plane_probe, 1, [out])
                np.testing.assert_allclose(out.numpy(), (0.425, 0.0, 1.0), atol=1.0e-6)

    def test_joint_plane_truncates_both_sides_beyond_worker_grid(self):
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                particle_t, body_t = wp.ones(1), wp.ones(1)
                count = 4097
                indices = np.full((count, 3), -1, dtype=np.int32)
                indices[-1] = (0, -1, -1)
                wp.launch(
                    apply_rigid_soft_truncation,
                    4096,
                    [
                        wp.array([count], dtype=int),
                        wp.array(indices, dtype=wp.vec3i),
                        wp.zeros(count, dtype=int),
                        wp.zeros(count, dtype=wp.vec3),
                        wp.array(np.tile((0, 0, 1), (count, 1)), dtype=wp.vec3),
                        wp.array(np.tile((1, 0, 0), (count, 1)), dtype=wp.vec3),
                        wp.array([0], dtype=int),
                        wp.array([[0, 0, 0.01]], dtype=wp.vec3),
                        wp.array([[0, 0, -0.02]], dtype=wp.vec3),
                        wp.array([wp.transform_identity()], dtype=wp.transform),
                        wp.array([wp.transform(wp.vec3(0, 0, 0.02), wp.quat_identity())], dtype=wp.transform),
                        wp.array([[0, 0, 0]], dtype=wp.vec3),
                        0.85,
                        False,
                        particle_t,
                        body_t,
                    ],
                )
                pt, bt = float(particle_t.numpy()[0]), float(body_t.numpy()[0])
                self.assertLess(pt, 1.0)
                self.assertLess(bt, 1.0)
                self.assertGreater(0.01 - 0.02 * pt, 0.02 * bt)

    def test_kinematic_pose_is_preserved(self):
        for device in wp.get_devices():
            with self.subTest(device=str(device)), wp.ScopedDevice(device):
                target = wp.transform(wp.vec3(0, 0, 1), wp.quat_identity())
                q = wp.array([target, target], dtype=wp.transform)
                wp.launch(
                    apply_body_truncation_ts,
                    2,
                    [
                        wp.array([wp.transform_identity()] * 2, dtype=wp.transform),
                        wp.array([2, 1], dtype=int),
                        wp.zeros(2, dtype=wp.vec3),
                        wp.zeros(2),
                        wp.ones(2),
                        wp.full(2, 0.01),
                        q,
                    ],
                )
                self.assertAlmostEqual(float(q.numpy()[0, 2]), 1.0)
                self.assertAlmostEqual(float(q.numpy()[1, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exercise self-contact at exactly coincident vertex/triangle positions."""

import unittest

import numpy as np
import warp as wp

from newton.tests.test_mjvbd_v2_contact_invariants import _make_self_contact_probe, full_particle, soft_particle


class TestCoincidentContact(unittest.TestCase):
    def test_coincident_vertex_stays_on_previous_side(self):
        """Return finite balanced forces on the previous side of the triangle."""
        for device in ["cpu", "cuda:0"] if wp.is_cuda_available() else ["cpu"]:
            for module in (full_particle, soft_particle):
                for side in (-1.0, 1.0):
                    with self.subTest(module=module.__name__, side=side, device=device):
                        points = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0.25, 0.25, 0)), dtype=np.float32)
                        anchor = points.copy()
                        anchor[3, 2] = side * 0.001
                        forces = wp.zeros((1, 8), dtype=wp.vec3, device=device)
                        wp.launch(
                            _make_self_contact_probe(module),
                            1,
                            [
                                wp.array(points, dtype=wp.vec3, device=device),
                                wp.array(anchor, dtype=wp.vec3, device=device),
                                wp.array([[0, 1, 2]], dtype=int, device=device),
                                wp.zeros((2, 4), dtype=int, device=device),
                                False,
                                0.0,
                                0.0,
                                forces,
                            ],
                            device=device,
                        )
                        f = forces.numpy()[0]
                        self.assertTrue(np.isfinite(f).all())
                        self.assertGreater(float(f[3, 2] * side), 0.0)
                        np.testing.assert_allclose(f[:4], f[4:], atol=1e-5)
                        np.testing.assert_allclose(f[:4].sum(axis=0), 0.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

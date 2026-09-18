# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep near-contact microsteps from crossing a soft contact plane."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2.vbd import particle_vbd_kernels as full
from newton._src.solvers.mjvbd_v2.vbd_soft import particle_vbd_kernels as soft


def _make_probe(module):
    @wp.kernel
    def probe(q: wp.array[wp.vec3], delta: wp.array[wp.vec3], edges: bool, out: wp.array[wp.vec3]):
        if edges:
            dummy, n, d = module.create_edge_edge_division_plane_closest_pt(
                q[0], delta[0], q[1], delta[1], q[2], delta[2], q[3], delta[3]
            )
        else:
            dummy, n, d = module.create_vertex_triangle_division_plane_closest_pt(
                q[0], delta[0], q[1], delta[1], q[2], delta[2], q[3], delta[3]
            )
        for i in range(4):
            t = float(1.0)
            if not dummy[i]:
                t = module.planar_truncation_t(q[i], delta[i], n, d, 1.0e-5, 0.9)
            out[i] = q[i] + t * delta[i]

    return probe


_PROBES = (_make_probe(full), _make_probe(soft))


class TestSelfContactMicrosteps(unittest.TestCase):
    def test_recorded_bag_patch_keeps_cube_vertex_on_original_side(self):
        """Preserve separation for the contact patch from take 091802 frames 3120-3121."""
        # Cube vertex 21 and bag triangle 860. These four positions are enough
        # to exercise the crossing without the robot or the 450 MB recording.
        before = np.array(
            [
                [0.4760831892, 0.0521108098, 0.9371489286],
                [0.4688785970, 0.0404297784, 0.9406855106],
                [0.4737035334, 0.0523568504, 0.9349195361],
                [0.4883403182, 0.0491005182, 0.9411386251],
            ],
            dtype=np.float32,
        )
        after = np.array(
            [
                [0.4769454598, 0.0520524345, 0.9391649961],
                [0.4696426094, 0.0410900936, 0.9442090392],
                [0.4744216204, 0.0528717488, 0.9379404187],
                [0.4889764488, 0.0497375019, 0.9441828728],
            ],
            dtype=np.float32,
        )

        def side(points):
            normal = np.cross(points[2] - points[1], points[3] - points[1])
            return float(np.dot(points[0] - points[1], normal))

        self.assertLess(side(before) * side(after), 0)
        for device in wp.get_devices():
            for kernel in _PROBES:
                # Split the observed motion into microsteps; add a little travel
                # to keep the untruncated crossing larger than float32 roundoff.
                delta = wp.array((after - before) * (1.25 / 512), dtype=wp.vec3, device=device)
                q = wp.array(before, dtype=wp.vec3, device=device)
                for _ in range(512):
                    wp.launch(kernel, 1, [q, delta, False, q], device=device)
                self.assertGreater(side(before) * side(q.numpy()), 0)

    def test_small_approaching_steps_do_not_cross(self):
        """Stop both primitive types even when normal motion is below the old epsilon."""
        for device in wp.get_devices():
            for kernel in _PROBES:
                for edges in (False, True):
                    for gap in (1e-7, 2e-6, 2e-4):
                        for move_surface in (False, True):
                            with self.subTest(
                                device=str(device), kernel=kernel.key, edges=edges, gap=gap, surface=move_surface
                            ):
                                if edges:
                                    q = np.array(
                                        [[-0.02, 0, gap], [0.02, 0, gap], [0, -0.02, 0], [0, 0.02, 0]], dtype=np.float32
                                    )
                                    side = 2
                                else:
                                    q = np.array(
                                        [[0, 0, gap], [-0.02, -0.02, 0], [0.02, -0.02, 0], [0, 0.02, 0]],
                                        dtype=np.float32,
                                    )
                                    side = 1
                                delta = np.zeros_like(q)
                                if move_surface:
                                    delta[side:, 2] = 2 * gap
                                else:
                                    delta[:side, 2] = -2 * gap
                                out = wp.empty(4, dtype=wp.vec3, device=device)
                                wp.launch(
                                    kernel,
                                    1,
                                    [
                                        wp.array(q, dtype=wp.vec3, device=device),
                                        wp.array(delta, dtype=wp.vec3, device=device),
                                        edges,
                                        out,
                                    ],
                                    device=device,
                                )
                                p = out.numpy()
                                self.assertGreater(float(p[:side, 2].min() - p[side:, 2].max()), 0)

    def test_separating_and_tangent_steps_remain_free(self):
        """Avoid introducing adhesion when a vertex separates or slides along a surface."""
        for device in wp.get_devices():
            for kernel in _PROBES:
                for displacement in ((0, 0, 2e-7), (1e-5, 0, 0)):
                    q = np.array([[0, 0, 1e-7], [-0.02, -0.02, 0], [0.02, -0.02, 0], [0, 0.02, 0]], dtype=np.float32)
                    delta = np.zeros_like(q)
                    delta[0] = displacement
                    out = wp.empty(4, dtype=wp.vec3, device=device)
                    wp.launch(
                        kernel,
                        1,
                        [
                            wp.array(q, dtype=wp.vec3, device=device),
                            wp.array(delta, dtype=wp.vec3, device=device),
                            False,
                            out,
                        ],
                        device=device,
                    )
                    np.testing.assert_array_equal(out.numpy(), q + delta)


if __name__ == "__main__":
    unittest.main()

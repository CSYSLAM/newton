# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the geometric hinge evaluation against the original chain rule."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2.particle_surface_cache import _angle, _evaluate_bending
from newton._src.solvers.mjvbd_v2.vbd.particle_vbd_kernels import (
    evaluate_dihedral_angle_based_bending_force_hessian as original,
)


@wp.kernel(enable_backward=False)
def compare_hinges(
    q: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    edges: wp.array2d[int],
    rest: wp.array[float],
    lengths: wp.array[float],
    damping: float,
    forces: wp.array[wp.vec3],
    hessians: wp.array[wp.mat33],
    fast_forces: wp.array[wp.vec3],
    fast_hessians: wp.array[wp.mat33],
):
    tid = wp.tid()
    edge, corner = tid // 4, tid % 4
    f, h = original(edge, corner, q, previous, edges, rest, lengths, 1.0, damping, 1.0 / 480.0)
    ff, hh = _evaluate_bending(
        edge, corner, q, edges, rest, lengths, 1.0, damping, 1.0 / 480.0, _angle(previous, edges, edge)
    )
    forces[tid] = f
    hessians[tid] = h
    fast_forces[tid] = ff
    fast_hessians[tid] = hh


class TestGeometricBending(unittest.TestCase):
    def test_energy_gradient(self):
        """Match central differences of the unchanged signed-dihedral energy."""
        rng = np.random.default_rng(1721)
        count = 48
        positions = rng.normal(size=(count * 4, 3)).astype(np.float32) * 0.05
        q = wp.array(positions, dtype=wp.vec3, device="cpu")
        edges = wp.array(np.arange(count * 4).reshape(-1, 4), dtype=int, device="cpu")
        rest = wp.zeros(count, device="cpu")
        lengths = wp.ones(count, device="cpu")
        outputs = [wp.empty(count * 4, dtype=dtype, device="cpu") for dtype in (wp.vec3, wp.mat33) * 2]
        wp.launch(compare_hinges, count * 4, [q, q, edges, rest, lengths, 0.0], outputs, device="cpu")
        force = outputs[2].numpy().reshape(count, 4, 3)

        def energy(points):
            x0, x1, x2, x3 = points
            n1 = np.cross(x2 - x0, x3 - x0)
            n2 = np.cross(x3 - x1, x2 - x1)
            n1 /= np.linalg.norm(n1)
            n2 /= np.linalg.norm(n2)
            edge = (x3 - x2) / np.linalg.norm(x3 - x2)
            theta = np.arctan2(np.dot(np.cross(n1, n2), edge), np.dot(n1, n2))
            return 0.5 * theta * theta

        reference = np.empty_like(force, dtype=float)
        for index, points in enumerate(positions.reshape(count, 4, 3).astype(float)):
            for corner in range(4):
                for axis in range(3):
                    plus, minus = points.copy(), points.copy()
                    plus[corner, axis] += 1e-7
                    minus[corner, axis] -= 1e-7
                    reference[index, corner, axis] = -(energy(plus) - energy(minus)) / 2e-7
        np.testing.assert_allclose(force, reference, rtol=2e-4, atol=2e-4)

    def test_chain_rule_equivalence(self):
        """Preserve forces and Hessians for random hinges with and without damping."""
        rng = np.random.default_rng(1345)
        count = 4096
        positions = rng.normal(size=(count * 4, 3)).astype(np.float32) * 0.05
        # Include collapsed hinges and boundary edges in the unchanged guards.
        positions[:4] = 0
        edge_ids = np.arange(count * 4, dtype=np.int32).reshape(-1, 4)
        edge_ids[1, 0] = -1
        rest_values = rng.uniform(-3.0, 3.0, count).astype(np.float32)
        length_values = rng.uniform(0.001, 0.1, count).astype(np.float32)
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            q = wp.array(positions, dtype=wp.vec3, device=device)
            previous = wp.array(positions + rng.normal(size=positions.shape) * 0.001, dtype=wp.vec3, device=device)
            edges = wp.array(edge_ids, dtype=int, device=device)
            rest = wp.array(rest_values, device=device)
            lengths = wp.array(length_values, device=device)
            outputs = [wp.empty(count * 4, dtype=dtype, device=device) for dtype in (wp.vec3, wp.mat33) * 2]
            for damping in (0.0, 0.002):
                wp.launch(
                    compare_hinges, count * 4, [q, previous, edges, rest, lengths, damping], outputs, device=device
                )
                arrays = [output.numpy() for output in outputs]
                for slow, fast in zip(arrays[:2], arrays[2:], strict=True):
                    axes = tuple(range(1, slow.ndim))
                    error = np.sqrt(np.sum((slow - fast) ** 2, axis=axes))
                    scale = np.sqrt(np.sum(slow**2, axis=axes))
                    self.assertLess(float(np.max(error / np.maximum(scale, 1e-4))), 2e-4)
                    np.testing.assert_array_equal(fast[:8], slow[:8])


if __name__ == "__main__":
    unittest.main()

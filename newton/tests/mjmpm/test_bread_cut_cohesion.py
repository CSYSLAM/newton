# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Cut connections without eroding particles or losing fragment cohesion."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.mjmpm.support.bread_cut_cohesion import CuttableCohesion, count_detached_samples


class TestBreadCutCohesion(unittest.TestCase):
    def test_connection_product_matches_dense_operator_after_cuts(self):
        rest = (
            np.stack(np.meshgrid(np.arange(3), np.arange(2), np.arange(2), indexing="ij"), axis=-1).reshape(-1, 3)
            * 0.005
        )
        rng = np.random.default_rng(53)
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()[:1]]:
            with self.subTest(device=device.alias), wp.ScopedDevice(device):
                cohesion = CuttableCohesion(
                    rest, mass=3.5e-5, spacing=0.005, device=device, tool_pose=wp.transform_identity()
                )
                pairs = cohesion.pairs.numpy()
                directions = rng.normal(size=(len(pairs), 3)).astype(np.float32)
                directions /= np.linalg.norm(directions, axis=1)[:, None]
                coefficients = rng.uniform(0.1, 4, size=len(pairs)).astype(np.float32)
                coefficients[::3] = 0  # Severed edges remain in the adjacency.
                cohesion.directions.assign(directions)
                cohesion.coefficients.assign(coefficients)
                matrix = np.eye(3 * len(rest))
                for (i, j), n, c in zip(pairs, directions, coefficients, strict=True):
                    block = float(c) * np.outer(n.astype(float), n.astype(float))
                    for a, b, sign in ((i, i, 1), (j, j, 1), (i, j, -1), (j, i, -1)):
                        matrix[3 * a : 3 * a + 3, 3 * b : 3 * b + 3] += sign * block
                host_x = rng.normal(size=(len(rest), 3)).astype(np.float32)
                host_y = rng.normal(size=(len(rest), 3)).astype(np.float32)
                x = wp.array(host_x, dtype=wp.vec3)
                y = wp.array(host_y, dtype=wp.vec3)
                output = wp.empty_like(x)
                alpha, beta = -0.7, 0.3
                expected = alpha * (matrix @ host_x.ravel()).reshape(-1, 3) + beta * host_y
                cohesion._matvec(x, y, output, alpha, beta)
                np.testing.assert_allclose(output.numpy(), expected, rtol=3e-6, atol=3e-6)
                # CG aliases y and z. Capture and replay must preserve this.
                y.assign(host_y)
                if device.is_cuda:
                    with wp.ScopedCapture() as capture:
                        cohesion._matvec(x, y, y, alpha, beta)
                    wp.capture_launch(capture.graph)
                else:
                    cohesion._matvec(x, y, y, alpha, beta)
                np.testing.assert_allclose(y.numpy(), expected, rtol=3e-6, atol=3e-6)

    def test_local_compression_retains_plastic_shape_without_position_projection(self):
        rest = np.array([[0, 0, 0], [0.005, 0, 0]], dtype=float)
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()[:1]]:
            with self.subTest(device=device.alias), wp.ScopedDevice(device):
                zone = wp.ones(2, dtype=float)
                cohesion = CuttableCohesion(
                    rest,
                    mass=3.5e-5,
                    spacing=0.005,
                    device=device,
                    tool_pose=wp.transform_identity(),
                    plastic_zone=zone,
                    plastic_yield=0.015,
                    plastic_limit=0.30,
                )
                compressed = np.array([[0, 0, 0], [0.0045, 0, 0]], dtype=np.float32)
                state = SimpleNamespace(
                    particle_q=wp.array(compressed, dtype=wp.vec3), particle_qd=wp.zeros(2, dtype=wp.vec3)
                )
                cohesion.apply(state, 1 / 360)
                learned = float(cohesion.lengths.numpy()[0])
                self.assertLess(learned, 0.005)
                self.assertGreater(learned, 0.0045)
                np.testing.assert_array_equal(state.particle_q.numpy(), compressed)
                # At its shortened stress-free length, it must not recover
                # to the old shape; only the elastic part has rebounded.
                state.particle_q.assign(wp.array([[0, 0, 0], [learned, 0, 0]], dtype=wp.vec3))
                state.particle_qd.zero_()
                cohesion.apply(state, 1 / 360)
                np.testing.assert_allclose(state.particle_qd.numpy(), 0, atol=1.0e-6)
                self.assertAlmostEqual(float(cohesion.lengths.numpy()[0]), learned)
                # Plastic compaction is bounded and is disabled away from
                # the physically marked cutting zone.
                state.particle_q.assign(wp.array([[0, 0, 0], [0.001, 0, 0]], dtype=wp.vec3))
                zone.zero_()
                cohesion.apply(state, 1 / 360)
                self.assertAlmostEqual(float(cohesion.lengths.numpy()[0]), learned)
                zone.fill_(1.0)
                cohesion.apply(state, 1 / 360)
                self.assertAlmostEqual(float(cohesion.lengths.numpy()[0]), 0.0035, places=7)

    def test_implicit_pair_matches_exact_velocity_and_preserves_momentum(self):
        rest = np.array([[0, 0, 0], [0.005, 0, 0]], dtype=float)
        mass, stiffness, dt = 3.5e-5, 20.0, 1 / 360
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()[:1]]:
            with self.subTest(device=device.alias), wp.ScopedDevice(device):
                cohesion = CuttableCohesion(
                    rest,
                    mass=mass,
                    spacing=0.005,
                    device=device,
                    tool_pose=wp.transform_identity(),
                    stiffness=stiffness,
                )
                state = SimpleNamespace(
                    particle_q=wp.array([[0, 0, 0], [0.006, 0, 0]], dtype=wp.vec3),
                    particle_qd=wp.array([[0.2, 0.1, -0.2], [0.4, 0.1, -0.2]], dtype=wp.vec3),
                )
                coefficient = (dt * dt * stiffness + dt * cohesion.damping) / mass
                relative = (0.2 - 2 * dt * stiffness * 0.001 / mass) / (1 + 2 * coefficient)
                expected = np.array([[0.3 - relative / 2, 0.1, -0.2], [0.3 + relative / 2, 0.1, -0.2]])
                cohesion.apply(state, dt)
                np.testing.assert_allclose(state.particle_qd.numpy(), expected, atol=2.0e-6)
                np.testing.assert_allclose(state.particle_qd.numpy().sum(axis=0), [0.6, 0.2, -0.4], atol=2.0e-6)
                if device.is_cuda:
                    with wp.ScopedCapture() as capture:
                        cohesion.apply(state, dt)
                    state.particle_qd.assign(wp.array([[0.2, 0.1, -0.2], [0.4, 0.1, -0.2]], dtype=wp.vec3))
                    wp.capture_launch(capture.graph)
                    np.testing.assert_allclose(state.particle_qd.numpy(), expected, atol=2.0e-6)
                # A severed connection exerts no impulse, regardless of stretch.
                cohesion.active.zero_()
                previous = state.particle_qd.numpy().copy()
                if device.is_cuda:
                    wp.capture_launch(capture.graph)
                else:
                    cohesion.apply(state, dt)
                np.testing.assert_allclose(state.particle_qd.numpy(), previous, atol=2.0e-6)

    def test_edge_cut_produces_two_pieces_and_flat_push_does_not_make_crumbs(self):
        spacing = 0.005
        rest = np.stack(
            np.meshgrid(
                np.arange(4) * spacing + 0.1,
                np.arange(4) * spacing - 0.0075,
                np.arange(4) * spacing + 0.001,
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        for device in [wp.get_device("cpu"), *wp.get_cuda_devices()[:1]]:
            with self.subTest(device=device.alias), wp.ScopedDevice(device):
                initial = wp.transform(wp.vec3(0.0, 0.0, 0.20), wp.quat_identity())
                cohesion = CuttableCohesion(rest, mass=3.5e-5, spacing=spacing, device=device, tool_pose=initial)
                state = SimpleNamespace(
                    particle_q=wp.array(rest, dtype=wp.vec3), body_q=wp.array([initial], dtype=wp.transform)
                )
                cohesion.cut(state, 0, wp.transform_identity())
                self.assertEqual(len(cohesion.components()[1]), 1)
                state.body_q.assign(
                    wp.array([wp.transform(wp.vec3(0, 0, 0.055), wp.quat_identity())], dtype=wp.transform)
                )
                cohesion.cut(state, 0, wp.transform_identity())
                labels, sizes = cohesion.components()
                np.testing.assert_array_equal(np.sort(sizes), [32, 32])
                # Deliberately push within the body, not outside its bounds.
                # The old cut-volume approach also severed this second plane.
                after_cut = cohesion.active.numpy().copy()
                state.body_q.assign(
                    wp.array([wp.transform(wp.vec3(0, -0.005, 0.055), wp.quat_identity())], dtype=wp.transform)
                )
                if device.is_cuda:
                    with wp.ScopedCapture() as capture:
                        cohesion.cut(state, 0, wp.transform_identity())
                    wp.capture_launch(capture.graph)
                else:
                    cohesion.cut(state, 0, wp.transform_identity())
                np.testing.assert_array_equal(cohesion.active.numpy(), after_cut)
                self.assertEqual(count_detached_samples(rest, labels, spacing), 0)
                crumb = rest.copy()
                crumb[0] += (0, 0, 0.1)
                self.assertEqual(count_detached_samples(crumb, labels, spacing), 1)


if __name__ == "__main__":
    unittest.main()

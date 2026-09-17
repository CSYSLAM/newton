# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check pneumatic derivatives and the bounded color pressure solve."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2.vbd import pneumatic_kernels as kernels
from newton.solvers import SolverMJVBDV2
from newton.tests.test_mjvbd_v2 import _build_pneumatic_shell_builder


def _devices():
    return ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])


class TestPneumaticCoupling(unittest.TestCase):
    def test_pressure_hessian_matches_finite_difference(self):
        """Differentiate pressure forces for two cavities sharing a vertex."""
        x = np.array(
            [[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [-0.2, 0, 0], [0, -0.2, 0], [0, 0, -0.2]], dtype=float
        )
        first = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
        second = np.array([[0, 4, 5], [0, 6, 4], [0, 5, 6], [4, 6, 5]], dtype=np.int32)
        # Interleave face rows to exercise cavity grouping independently of face order.
        triangles = np.stack((first, second), axis=1).reshape(-1, 3)
        cavities = np.tile([0, 1], 4).astype(np.int32)
        stiffness = np.array([1200.0, 700.0])
        pressure = np.array([3.0, 5.0])

        def volume(q):
            a, b, c = (q[triangles[:, j]] for j in range(3))
            values = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
            return np.bincount(cavities, weights=values)

        initial_volume = volume(x)

        def force(q):
            p = pressure - stiffness * (volume(q) - initial_volume)
            result = np.zeros_like(q)
            for face, triangle in enumerate(triangles):
                for j in range(3):
                    result[triangle[j]] += (
                        p[cavities[face]] * np.cross(q[triangle[(j + 1) % 3]], q[triangle[(j + 2) % 3]]) / 6
                    )
            return result

        expected = np.zeros((len(x), 3, 3))
        for particle in range(len(x)):
            for axis in range(3):
                plus, minus = x.copy(), x.copy()
                plus[particle, axis] += 1e-6
                minus[particle, axis] -= 1e-6
                expected[particle, :, axis] = -(force(plus)[particle] - force(minus)[particle]) / 2e-6
        faces, offsets = [], [0]
        for particle in range(len(x)):
            adjacent = np.flatnonzero(np.any(triangles == particle, axis=1))
            faces.extend(adjacent[np.argsort(cavities[adjacent], kind="stable")])
            offsets.append(len(faces))
        for device in _devices():

            def array(values, dtype, device=device):
                return wp.array(values, dtype=dtype, device=device)

            forces = wp.zeros(len(x), dtype=wp.vec3, device=device)
            hessians = wp.zeros(len(x), dtype=wp.mat33, device=device)
            wp.launch(
                kernels.accumulate_pressure_force_and_hessian,
                dim=len(x),
                inputs=[
                    array(np.arange(len(x)), int),
                    array(x, wp.vec3),
                    array(triangles, int),
                    array(cavities, int),
                    array(np.arange(8), int),
                    array(np.ones(8), float),
                    array(offsets, int),
                    array(faces, int),
                    array(pressure, float),
                    array(stiffness, float),
                ],
                outputs=[forces, hessians],
                device=device,
            )
            np.testing.assert_allclose(forces.numpy(), force(x), rtol=2e-6, atol=1e-8)
            np.testing.assert_allclose(hessians.numpy(), expected, rtol=2e-5, atol=1e-8)

    @unittest.skipUnless(wp.is_cuda_available(), "Coupled pressure requires CUDA")
    def test_cached_and_uncached_coupling_agree(self):
        """Preserve pressure updates with cached elasticity and incremental volumes."""
        builder, handle, _ = _build_pneumatic_shell_builder()
        # Reach the solver's existing minimum color size for tile elasticity.
        for copy in range(1, 16):
            offset = len(builder.particle_q)
            for point in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
                builder.add_particle(wp.vec3(point[0] + 3 * copy, point[1], point[2]), wp.vec3(), 1.0)
            for triangle in ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)):
                builder.add_triangle(*(i + offset for i in triangle), tri_ke=2e3, tri_ka=2e3, tri_kd=5.0)
        builder.color()
        model = builder.finalize(device="cuda:0")
        model.pneumatic.mode.fill_(2)
        model.pneumatic.target_volume.fill_(handle.rest_volume * 1.05)
        model.pneumatic.volume_stiffness.fill_(1.0e4)
        model.pneumatic.bulk_damping.fill_(10.0)
        control = model.control()
        results = []
        for cached in (False, True):
            solver = SolverMJVBDV2(
                model,
                vbd_options={
                    "iterations": 4,
                    "pneumatic_enable_color_coupling": True,
                    "pneumatic_enable_incremental_volume": cached,
                    "particle_enable_surface_cache": cached,
                    "particle_enable_self_contact": False,
                },
            )
            self.assertFalse(solver.vbd_solver._pneumatic_single_cavity_force_fusion_enabled)
            state_in, state_out = model.state(), model.state()
            for _ in range(10):
                solver.step(state_in, state_out, control, None, 0.001)
                state_in, state_out = state_out, state_in
            results.append((state_in.particle_q.numpy(), state_in.pneumatic.volume.numpy()))
        for actual, expected in zip(results[1], results[0], strict=True):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "Tile reduction requires CUDA")
    def test_color_solve_matches_bounded_pressure_equilibrium(self):
        """Match dense equilibrium across pressure bounds and multiple block strides."""
        rng = np.random.default_rng(731)
        for count in (7, 513):
            gradients = rng.normal(size=(count, 3)) * 0.03
            factors = rng.normal(size=(count, 3, 3))
            hessians = np.einsum("nji,njk->nik", factors, factors) + np.eye(3) * 2
            hessians[-1] = 0  # A fixed/singular row must not enter the reduction.
            gradients[-1] = 0
            direction = rng.normal(size=(count, 3)) * 0.05
            direction[-1] = 0
            previous = rng.normal(size=(count, 3)) * 0.02
            for target in (-10.0, 0.9, 10.0):
                pressure, volume, old_volume = 20.0, 0.8, 0.81
                stiffness, damping, dt = 120.0, 2.0, 0.01
                ambient, maximum = 100.0, 150.0
                other_force = np.einsum("nij,nj->ni", hessians, direction) - pressure * gradients
                active_hessians = hessians[:-1]
                g = gradients[:-1].reshape(-1)
                # A dense solve independently checks the rank-one interior branch.
                if count == 7:
                    a = np.zeros((3 * (count - 1), 3 * (count - 1)))
                    for i, h in enumerate(active_hessians):
                        a[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] = h
                    raw = stiffness * (target - volume) - damping * (volume - old_volume) / dt
                    d = np.linalg.solve(
                        a + (stiffness + damping / dt) * np.outer(g, g), other_force[:-1].reshape(-1) + raw * g
                    )
                    p = raw - (stiffness + damping / dt) * np.dot(g, d)
                    if not -ambient <= p <= maximum - ambient:
                        p = np.clip(p, -ambient, maximum - ambient)
                        d = np.linalg.solve(a, other_force[:-1].reshape(-1) + p * g)
                    expected = d.reshape(-1, 3)
                device = "cuda:0"

                def array(values, dtype, device=device):
                    return wp.array(values, dtype=dtype, device=device)

                displacement = array(previous + direction, wp.vec3)
                wp.launch(
                    kernels.correct_coupled_color,
                    dim=256,
                    block_dim=256,
                    inputs=[
                        array(np.arange(count), int),
                        array(gradients, wp.vec3),
                        array(hessians, wp.mat33),
                        array(previous, wp.vec3),
                        array([volume], float),
                        array([old_volume], float),
                        array([pressure], float),
                        dt,
                        array([target], float),
                        array([stiffness], float),
                        array([damping], float),
                        array([ambient], float),
                        array([maximum], float),
                        array([1.0], float),
                        array([1.0], float),
                    ],
                    outputs=[displacement],
                    device=device,
                )
                actual = displacement.numpy().astype(float) - previous
                if count == 7:
                    np.testing.assert_allclose(actual[:-1], expected, rtol=2e-4, atol=2e-6)
                new_volume = volume + np.sum(gradients * actual)
                p = np.clip(
                    stiffness * (target - new_volume) - damping * (new_volume - old_volume) / dt,
                    -ambient,
                    maximum - ambient,
                )
                residual = np.einsum("nij,nj->ni", hessians, actual) - other_force - p * gradients
                np.testing.assert_allclose(residual, 0.0, atol=5e-5)
                np.testing.assert_allclose(actual[-1], 0.0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()

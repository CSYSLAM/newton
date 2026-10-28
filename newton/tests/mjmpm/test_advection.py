# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Particle transport accuracy, independently of the pressure/contact solve."""

import unittest
from itertools import product

import numpy as np
import warp as wp
import warp.fem as fem

import newton
from newton._src.solvers.mjmpm.mpm.implicit_mpm_solver_kernels import advect_particles
from newton._src.solvers.mjmpm.mpm.solver_implicit_mpm import SolverImplicitMPM


@fem.integrand
def _extensional_flow(s: fem.Sample, domain: fem.Domain):
    x = domain(s)
    rate = float(fem.environment_index(domain, s) + 1)
    return rate * wp.vec3(x[0], -x[1], 0.0)


class TestParticleAdvection(unittest.TestCase):
    def test_midpoint_reduces_volume_error_and_preserves_worlds(self):
        """A divergence-free extensional flow preserves a material cube's volume.

        Two overlapping worlds have different flow rates, exposing an accidental
        cross-world midpoint lookup. The same fixture checks inactive particles,
        unchanged velocity/gradient transfer, CFL limits and graph replay.
        """
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                self._check_transport(device)

    def _check_transport(self, device):
        with wp.ScopedDevice(device):
            grid = fem.Grid3D(res=wp.vec3i(2), bounds_lo=wp.vec3(-2.0), bounds_hi=wp.vec3(2.0), env_count=2)
            domain = fem.Cells(grid)
            field = fem.make_polynomial_space(grid, degree=1, dtype=wp.vec3).make_field()
            fem.interpolate(_extensional_flow, dest=field)
            cube = np.array(list(product((-0.2, 0.2), repeat=3)), dtype=np.float32)
            initial = np.concatenate((cube, cube, [[0.1, 0.1, 0.1]])).astype(np.float32)
            points = wp.array(initial, dtype=wp.vec3)
            volumes = wp.ones(len(initial), dtype=float)
            flags = wp.array([int(newton.ParticleFlags.ACTIVE)] * 16 + [0], dtype=int)
            pic = fem.PicQuadrature(
                domain, points, measures=volumes, env_indices=wp.array([0] * 8 + [1] * 9, dtype=int)
            )
            output = wp.empty_like(points)
            velocity = wp.zeros_like(points)
            gradient = wp.zeros(len(initial), dtype=wp.mat33)

            def advect(order, max_velocity=100.0):
                output.assign(points)
                velocity.zero_()
                gradient.zero_()
                fem.interpolate(
                    advect_particles,
                    at=pic,
                    fields={"grid_vel": field},
                    values={
                        "dt": 0.2,
                        "max_vel": max_velocity,
                        "advection_order": order,
                        "particle_flags": flags,
                        "particle_volume": volumes,
                        "pos": output,
                        "vel": velocity,
                        "vel_grad": gradient,
                    },
                )

            volume_errors = []
            for order in (1, 2):
                advect(order)
                positions = output.numpy()
                errors = []
                for world in range(2):
                    sl = slice(world * 8, (world + 1) * 8)
                    rate = world + 1
                    expected_velocity = cube * np.array([rate, -rate, 0.0])
                    expected_position = cube + 0.2 * expected_velocity
                    if order == 2:
                        expected_position += 0.02 * expected_velocity * np.array([rate, -rate, 0.0])
                    np.testing.assert_allclose(positions[sl], expected_position, atol=2.0e-7)
                    np.testing.assert_allclose(velocity.numpy()[sl], expected_velocity, atol=2.0e-7)
                    np.testing.assert_allclose(
                        gradient.numpy()[sl], np.tile(np.diag([rate, -rate, 0.0]), (8, 1, 1)), atol=2.0e-7
                    )
                    ratio = np.prod(np.ptp(positions[sl], axis=0)) / np.prod(np.ptp(cube, axis=0))
                    errors.append(abs(float(ratio) - 1.0))
                volume_errors.append(np.array(errors))
                np.testing.assert_array_equal(positions[-1], initial[-1])
                np.testing.assert_array_equal(velocity.numpy()[-1], 0.0)
                np.testing.assert_array_equal(gradient.numpy()[-1], 0.0)
            self.assertTrue(np.all(volume_errors[1] < 0.05 * volume_errors[0]))

            if wp.get_device(device).is_cuda:
                expected = output.numpy()
                with wp.ScopedCapture() as capture:
                    advect(2)
                wp.capture_launch(capture.graph)
                np.testing.assert_allclose(output.numpy(), expected, atol=2.0e-7)
            advect(2, max_velocity=0.1)
            self.assertLessEqual(float(np.linalg.norm(output.numpy() - initial, axis=1).max()), 0.020001)
            self.assertLessEqual(float(np.linalg.norm(velocity.numpy(), axis=1).max()), 0.100001)

    def test_default_and_invalid_order(self):
        self.assertEqual(SolverImplicitMPM.Config().advection_order, 1)
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)
        builder.add_particle((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), 0.1, 0.01)
        model = builder.finalize(device="cpu")
        for order in (0, 3, 1.5):
            with self.subTest(order=order), self.assertRaisesRegex(ValueError, "advection_order"):
                SolverImplicitMPM(model, SolverImplicitMPM.Config(advection_order=order))


if __name__ == "__main__":
    unittest.main()

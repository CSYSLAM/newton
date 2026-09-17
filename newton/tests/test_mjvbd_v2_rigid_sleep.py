# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise required wake conditions independently of the pile scene."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjvbd_v2 import rigid_sleep as sleep
from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD


class TestRigidSleep(unittest.TestCase):
    def test_count_unique_neighbors(self):
        """Count neighboring bodies once despite repeated manifold contacts."""
        device = "cpu"

        def integers(values):
            return wp.array(values, dtype=int, device=device)

        # Four manifold rows per pair; a third pair is an invalid self-pair.
        shape0 = integers([0] * 8 + [1])
        shape1 = integers([1] * 4 + [2] * 4 + [1])
        parent = integers([7, 7, 7])
        adjacent = wp.full((3, 3), 7, dtype=int, device=device)
        support = wp.full(3, 7, dtype=int, device=device)
        veto = wp.full(3, 7, dtype=int, device=device)
        supported = wp.full(3, 7, dtype=int, device=device)
        wp.launch(sleep._initialize, (3, 3), [parent, veto, supported, support, adjacent], device=device)
        np.testing.assert_array_equal(veto.numpy(), [0, 0, 0])
        np.testing.assert_array_equal(supported.numpy(), [0, 0, 0])
        points = wp.zeros(9, dtype=wp.vec3, device=device)
        margins = wp.zeros(9, dtype=float, device=device)
        wp.launch(
            sleep._graph,
            9,
            [
                integers([9]),
                shape0,
                shape1,
                integers([0, 1, 2]),
                points,
                points,
                points,
                margins,
                margins,
                wp.array([wp.transform_identity()] * 3, dtype=wp.transform, device=device),
                wp.zeros(3, dtype=wp.spatial_vector, device=device),
                wp.ones(3, dtype=float, device=device),
                parent,
                adjacent,
                support,
                wp.array([[0, 0, -9.81]], dtype=wp.vec3, device=device),
                integers([0, 0, 0]),
            ],
            device=device,
        )
        np.testing.assert_array_equal(adjacent.numpy(), [[2, 1, 1], [1, 1, 0], [1, 0, 1]])

    def test_reset_preserves_physical_state(self):
        """Restore effective mass on reset without altering physical mass or pose."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_sphere(body, radius=0.1)
        builder.color()
        model = builder.finalize(device="cpu")
        solver = SolverVBD(model, rigid_enable_sleep=True)
        state = model.state()
        runtime = solver._rigid_sleep
        expected_mass = model.body_mass.numpy().copy()
        expected_inverse = runtime.base.numpy().copy()
        expected_pose = state.body_q.numpy().copy()
        runtime.asleep.fill_(1)
        runtime.timer.zero_()
        solver.body_inv_mass_effective.zero_()
        solver.reset(state, flags=0)
        np.testing.assert_array_equal(runtime.asleep.numpy(), [0])
        np.testing.assert_array_equal(solver.body_inv_mass_effective.numpy(), expected_inverse)
        np.testing.assert_array_equal(model.body_mass.numpy(), expected_mass)
        np.testing.assert_array_equal(state.body_q.numpy(), expected_pose)
        self.assertIsNone(SolverVBD(model)._rigid_sleep)

    def test_wake_conditions(self):
        """Wake on moving support, explicit force/velocity, or lost support."""
        device = "cpu"
        for mode in (
            "rest",
            "moving_support",
            "force",
            "velocity",
            "lost_support",
            "soft_contact",
            "overflow",
            "nonfinite_velocity",
        ):
            with self.subTest(mode=mode):

                def integer(x):
                    return wp.array(x, dtype=int, device=device)

                parent = integer([0, 1])
                support = wp.zeros(2, dtype=int, device=device)
                supported = wp.zeros_like(support)
                veto = wp.zeros_like(support)
                adjacent = wp.zeros((2, 2), dtype=int, device=device)
                q = wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=device)
                vel = np.zeros((2, 6), dtype=np.float32)
                if mode == "moving_support":
                    vel[1, 0] = 0.1
                if mode == "velocity":
                    vel[0, 0] = 0.1
                if mode == "nonfinite_velocity":
                    vel[0, 0] = np.nan
                qd = wp.array(vel, dtype=wp.spatial_vector, device=device)
                f = np.zeros_like(vel)
                if mode == "force":
                    f[0, 2] = 0.01
                force = wp.array(f, dtype=wp.spatial_vector, device=device)
                base = wp.array([1, 0], dtype=float, device=device)
                effective = wp.zeros(2, device=device)
                asleep = integer([1, 0])
                timer = wp.zeros(2, device=device)
                accum = wp.zeros(2, dtype=wp.spatial_vector, device=device)
                inertia = wp.array([np.eye(3)] * 2, dtype=wp.mat33, device=device)
                point = wp.zeros(1, dtype=wp.vec3, device=device)
                margin = wp.zeros(1, device=device)
                normal = wp.array([[0, 0, -1]], dtype=wp.vec3, device=device)
                count = integer([int(mode != "lost_support")])
                if mode == "overflow":
                    count.fill_(2)
                wp.launch(
                    sleep._graph,
                    1,
                    [
                        count,
                        integer([0]),
                        integer([1]),
                        integer([0, 1]),
                        point,
                        point,
                        normal,
                        margin,
                        margin,
                        q,
                        qd,
                        base,
                        parent,
                        adjacent,
                        support,
                        wp.array([[0, 0, -9.81]], dtype=wp.vec3, device=device),
                        integer([0, 0]),
                    ],
                    device=device,
                )
                if mode == "soft_contact":
                    wp.launch(
                        sleep._soft_wake, 1, [integer([1]), integer([0]), integer([0, 1]), support], device=device
                    )
                wp.launch(
                    sleep._evaluate,
                    2,
                    [
                        parent,
                        adjacent,
                        support,
                        supported,
                        veto,
                        q,
                        qd,
                        force,
                        inertia,
                        base,
                        asleep,
                        timer,
                        accum,
                        1 / 360,
                        1,
                    ],
                    device=device,
                )
                wp.launch(
                    sleep._apply, 2, [parent, supported, veto, base, asleep, effective, qd, qd, timer], device=device
                )
                self.assertEqual(asleep.numpy()[0], int(mode == "rest"))
                self.assertEqual(effective.numpy()[0], float(mode != "rest"))


if __name__ == "__main__":
    unittest.main()

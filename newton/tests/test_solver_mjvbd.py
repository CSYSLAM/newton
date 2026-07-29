# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the private-snapshot MJVBD composite solver."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMJVBD


class TestSolverMJVBD(unittest.TestCase):
    def _make_model(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        builder.add_shape_sphere(body=body, radius=0.1)
        builder.add_particle(pos=(0.15, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.1)
        builder.color()
        return builder.finalize(device="cpu"), body

    def test_external_mode_contacts_new_rigid_pose(self):
        """MJVBD must query state_out bodies, rather than stale state_in bodies."""
        model, body = self._make_model()
        solver = SolverMJVBD(model, rigid_mode="external", vbd_options={"iterations": 1})
        state_in = model.state()
        state_out = model.state()
        wp.copy(state_out.body_q, state_in.body_q)
        wp.copy(state_out.body_qd, state_in.body_qd)

        new_body_q = state_out.body_q.numpy()
        new_body_q[body, 0] = 0.0
        state_out.body_q.assign(new_body_q)

        solver.step(state_in, state_out, None, None, 1.0 / 60.0)

        self.assertEqual(int(solver.contacts.soft_contact_count.numpy()[0]), 1)
        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))
        self.assertGreater(float(state_out.particle_q.numpy()[0, 0]), 0.15)

    def test_external_mode_rejects_internal_rigid_vbd(self):
        """The composite must retain its one-way rigid-to-soft contract."""
        model, _ = self._make_model()
        with self.assertRaisesRegex(ValueError, "one-way coupling"):
            SolverMJVBD(model, vbd_options={"integrate_with_external_rigid_solver": False})

    def test_mujoco_mode_advances_rigid_state(self):
        """The private MuJoCo snapshot advances rigid bodies before VBD runs."""
        model, body = self._make_model()
        model.set_gravity((0.0, 0.0, -9.81))
        solver = SolverMJVBD(model, rigid_mode="mujoco")
        state_in = model.state()
        state_out = model.state()

        solver.step(state_in, state_out, model.control(), None, 1.0 / 120.0)

        self.assertTrue(np.all(np.isfinite(state_out.body_q.numpy())))
        self.assertLess(float(state_out.body_q.numpy()[body, 2]), 0.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)

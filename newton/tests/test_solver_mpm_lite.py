# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the Newton MPM Lite integration."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMPMLite
from newton.tests.unittest_utils import add_function_test


def test_mpm_lite_step_preserves_input_state(test, device):
    """Advance output state without mutating the input state."""
    builder = newton.ModelBuilder()
    SolverMPMLite.register_custom_attributes(builder)
    builder.add_particle(pos=(0.08, 0.08, 0.08), vel=(0.0, 0.0, 0.0), mass=1.0)
    model = builder.finalize(device=device)
    model.set_gravity((0.0, 0.0, -9.81))
    solver = SolverMPMLite(
        model,
        SolverMPMLite.Config(
            grid_size=(16, 16, 16),
            voxel_size=0.02,
            solver_type="lite_explicit",
            density=1000.0,
        ),
    )
    solver.paint_boundary(np.array([[0, 0, 0]], dtype=np.int32), np.ones(1, dtype=np.int32))
    state_in = model.state()
    state_out = model.state()
    input_position = state_in.particle_q.numpy().copy()

    solver.step(state_in, state_out, control=None, contacts=None, dt=1.0e-4)

    np.testing.assert_allclose(state_in.particle_q.numpy(), input_position)
    test.assertLess(state_out.particle_q.numpy()[0, 2], input_position[0, 2])
    test.assertTrue(np.isfinite(state_out.mpm_lite.particle_F.numpy()).all())


class TestSolverMPMLite(unittest.TestCase):
    pass


add_function_test(
    TestSolverMPMLite,
    "test_mpm_lite_step_preserves_input_state",
    test_mpm_lite_step_preserves_input_state,
    devices=[wp.get_device("cpu")],
)

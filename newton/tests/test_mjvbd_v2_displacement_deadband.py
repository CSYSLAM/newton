# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check substep displacement clamping and subsequent force response."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjvbd_v2.vbd_soft.solver_vbd import SolverVBD
from newton.tests.unittest_utils import add_function_test, get_test_devices


def particles(device, count=3):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for _ in range(count):
        builder.add_particle(pos=wp.vec3(), vel=wp.vec3(), mass=1.0)
    builder.color()
    return builder.finalize(device=device)


def test_deadband_step(test, device):
    """Clamp short Euclidean steps, preserve larger steps and respond to renewed forcing."""
    model = particles(device)
    solver = newton.solvers.SolverMJVBDV2(
        model, vbd_options={"iterations": 1, "particle_displacement_threshold": 0.001}
    )
    state, out = model.state(), model.state()
    velocities = np.array([[0.05, 0, 0], [0.08, 0.08, 0], [0.2, 0, 0]], dtype=np.float32)
    state.particle_qd.assign(velocities)
    solver.step(state, out, model.control(), None, 0.01)
    np.testing.assert_array_equal(out.particle_q.numpy()[0], [0, 0, 0])
    np.testing.assert_array_equal(out.particle_qd.numpy()[0], [0, 0, 0])
    np.testing.assert_allclose(out.particle_q.numpy()[1:], 0.01 * velocities[1:], atol=1e-8)
    np.testing.assert_allclose(out.particle_qd.numpy()[1:], velocities[1:], atol=1e-6)
    # A clamped particle remains active and participates in the next solve.
    out.particle_f.assign(np.array([[20, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32))
    solver.step(out, state, model.control(), None, 0.01)
    test.assertGreater(float(state.particle_q.numpy()[0, 0]), 0.001)
    test.assertGreater(float(state.particle_qd.numpy()[0, 0]), 0.1)


def test_deadband_disabled(test, device):
    """Preserve slow translation when the threshold is zero."""
    model = particles(device)
    solver = SolverVBD(model, iterations=1, particle_displacement_threshold=0.0)
    state, out = model.state(), model.state()
    state.particle_qd.fill_(wp.vec3(0.0001, 0.0, 0.0))
    solver.step(state, out, model.control(), None, 0.01)
    np.testing.assert_allclose(out.particle_q.numpy()[:, 0], 1e-6, atol=1e-10)
    np.testing.assert_allclose(out.particle_qd.numpy()[:, 0], 0.0001, atol=1e-8)


def test_deadband_prescribed(test, device):
    """Leave prescribed, proxy and zero-mass positions untouched by clamping."""
    model = particles(device, count=5)
    model.particle_flags.assign(
        np.array([0, int(newton.ParticleFlags.ACTIVE | newton.ParticleFlags.PROXY), 1, 1, 1], dtype=np.int32)
    )
    model.particle_inv_mass.assign(np.array([1, 1, 0, 1, 1], dtype=np.float32))
    solver = SolverVBD(model, particle_displacement_threshold=0.001)
    state = model.state()
    q = np.full((5, 3), 0.0001, dtype=np.float32)
    q[3] = [0.001, 0, 0]
    state.particle_q.assign(q)
    solver.particle_q_prev.zero_()
    solver.pos_prev_collision_detection.fill_(wp.vec3(2, 3, 4))
    solver._apply_particle_displacement_deadband(state)
    # Equality is outside the deadband; only the last free particle is clamped.
    np.testing.assert_array_equal(state.particle_q.numpy()[:4], q[:4])
    np.testing.assert_array_equal(state.particle_q.numpy()[4], [0, 0, 0])
    np.testing.assert_array_equal(solver.particle_displacements.numpy()[4], [-2, -3, -4])


def test_deadband_graph(test, device):
    """Replay clamped and moving states through the same captured step."""
    if not device.is_cuda:
        return
    model = particles(device)
    solver = newton.solvers.SolverMJVBDV2(
        model, vbd_options={"iterations": 1, "particle_displacement_threshold": 0.001}
    )
    state, out = model.state(), model.state()
    control = model.control()
    solver.step(state, out, control, None, 0.01)
    with wp.ScopedCapture(device=device) as capture:
        solver.step(state, out, control, None, 0.01)
    for speed in (0.05, 0.2, 0.05):
        state.particle_q.zero_()
        state.particle_qd.fill_(wp.vec3(speed, 0, 0))
        wp.capture_launch(capture.graph)
        expected = 0.0 if speed < 0.1 else speed * 0.01
        np.testing.assert_allclose(out.particle_q.numpy()[:, 0], expected, atol=1e-8)
        np.testing.assert_allclose(out.particle_qd.numpy()[:, 0], expected / 0.01, atol=1e-6)


class TestMJVBDV2DisplacementDeadband(unittest.TestCase):
    def test_invalid_threshold(self):
        """Reject negative, nonfinite and differentiable hard-clamp configurations."""
        model = particles("cpu")
        for value in (-1.0, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                SolverVBD(model, particle_displacement_threshold=value)
        model.requires_grad = True
        with self.assertRaisesRegex(ValueError, "differentiable"):
            SolverVBD(model, particle_displacement_threshold=1e-6)


for fn in (test_deadband_step, test_deadband_disabled, test_deadband_prescribed, test_deadband_graph):
    add_function_test(TestMJVBDV2DisplacementDeadband, fn.__name__, fn, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()

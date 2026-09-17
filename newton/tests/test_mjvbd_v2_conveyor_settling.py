# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the conveyor's cloth-only settling filter and renewed force response."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.examples.mjvbdv2.example_mjvbd_v2_conveyor_sorting import _discard_cloth_microsteps
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_local_settling(test, device):
    """Filter Euclidean microsteps only in the tray and preserve other parcels."""
    previous = np.array([[0, 0, 0.70], [0, 0, 0.70], [0.3, 0, 0.70], [0, 0, 0.9]], dtype=np.float32)
    positions = np.vstack(([1, 2, 3], previous)).astype(np.float32)
    positions[1:, 0] += 0.8e-5
    positions[2, 1] += 0.8e-5  # Each component is small, but its Euclidean step is not.
    q = wp.array(positions, dtype=wp.vec3, device=device)
    qd = wp.full(5, wp.vec3(0.1), dtype=wp.vec3, device=device)
    wp.launch(
        _discard_cloth_microsteps,
        4,
        [1, wp.vec3(0, 0, 0.76), 1e-5, 1 / 480, wp.array(previous, dtype=wp.vec3, device=device), q, qd],
        device=device,
    )
    np.testing.assert_array_equal(q.numpy()[1], previous[0])
    np.testing.assert_array_equal(qd.numpy()[1], [0, 0, 0])
    np.testing.assert_array_equal(q.numpy()[[0, 2, 3, 4]], positions[[0, 2, 3, 4]])
    np.testing.assert_allclose(qd.numpy()[[0, 2, 3, 4]], 0.1)


def test_settling_force_response(test, device):
    """A filtered particle remains dynamic and moves when pushed, also in a CUDA graph."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_particle(pos=wp.vec3(0, 0, 0.70), vel=wp.vec3(), mass=0.000033)
    builder.color()
    model = builder.finalize(device=device)
    # Use the same full VBD implementation as the mixed-material station.
    from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD  # noqa: PLC0415

    solver = SolverVBD(model, iterations=1)
    state, out = model.state(), model.state()
    previous = wp.clone(state.particle_q)
    frame_dt = 1 / 60
    dt = frame_dt / 8
    control = model.control()

    def step():
        nonlocal state, out
        wp.copy(previous, state.particle_q)
        for _ in range(8):
            solver.step(state, out, control, None, dt)
            state, out = out, state
        wp.launch(
            _discard_cloth_microsteps,
            1,
            [0, wp.vec3(0, 0, 0.76), 5e-4, frame_dt, previous, state.particle_q, state.particle_qd],
            device=device,
        )

    with wp.ScopedDevice(device):
        step()
        if device.is_cuda:
            with wp.ScopedCapture() as capture:
                step()
        for force in (0.0, 0.01):
            state.particle_q.assign(model.particle_q)
            state.particle_qd.fill_(wp.vec3(0.001, 0, 0))
            state.particle_f.fill_(wp.vec3(force, 0, 0))
            out.particle_f.fill_(wp.vec3(force, 0, 0))
            if device.is_cuda:
                wp.capture_launch(capture.graph)
            else:
                step()
            if force == 0:
                np.testing.assert_array_equal(state.particle_q.numpy(), model.particle_q.numpy())
                np.testing.assert_array_equal(state.particle_qd.numpy(), [[0, 0, 0]])
            else:
                test.assertGreater(float(state.particle_q.numpy()[0, 0]), 0.001)
                test.assertGreater(float(state.particle_qd.numpy()[0, 0]), 0.4)
    np.testing.assert_array_equal(model.particle_flags.numpy(), [int(newton.ParticleFlags.ACTIVE)])
    test.assertGreater(float(model.particle_inv_mass.numpy()[0]), 0)
    # The scene's gravity cap must not suspend unsupported cloth at finer dt.
    model.gravity.fill_(wp.vec3(0, 0, -9.81))
    for substeps in (8, 16, 32):
        dt = 1 / (60 * substeps)
        threshold = min(5e-4, 0.25 * 9.81 * frame_dt**2)
        state.particle_q.assign(model.particle_q)
        state.particle_qd.zero_()
        state.particle_f.zero_()
        out.particle_f.zero_()
        wp.copy(previous, state.particle_q)
        for _ in range(substeps):
            solver.step(state, out, control, None, dt)
            state, out = out, state
        wp.launch(
            _discard_cloth_microsteps,
            1,
            [0, wp.vec3(0, 0, 0.76), threshold, frame_dt, previous, state.particle_q, state.particle_qd],
            device=device,
        )
        fall = float(previous.numpy()[0, 2] - state.particle_q.numpy()[0, 2])
        test.assertGreater(fall, 0.45 * 9.81 * frame_dt**2)


class TestMJVBDV2ConveyorSettling(unittest.TestCase):
    pass


for fn in (test_local_settling, test_settling_force_response):
    add_function_test(TestMJVBDV2ConveyorSettling, fn.__name__, fn, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main()

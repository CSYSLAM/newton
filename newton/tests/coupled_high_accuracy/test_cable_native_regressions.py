"""Cable state remains optional in the low-level native solver interface."""

import numpy as np
import pytest

from newton.solvers import coupled_high_accuracy as s3d

wp = pytest.importorskip("warp")

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Native solver requires CUDA")
FRAMEWORKS = [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON]


def _span(array):
    return s3d.Span(ptr=array.ptr, size=len(array))


def _state(positions, velocities):
    state = s3d.State()
    state.particle_q = _span(positions)
    state.particle_qd = _span(velocities)
    return state


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("captured", [False, True])
def test_empty_native_model_accepts_optional_cable_state(framework, captured):
    solver = s3d.Solver()
    solver.initialize(s3d.Model())
    solver.config.nonlinear_solver = framework
    solver.config.affine_body_block_preconditioner = True
    solver.config.cable_linear_beta = 100.0
    solver.config.cable_angular_beta = 100.0

    def advance():
        solver.reset_cable_penalty_state()
        solver.copy_cable_penalties(s3d.Span())
        solver.step(s3d.State(), s3d.State(), 0.01, wp.get_stream("cuda:0"))

    if captured:
        # No cable state exists, so an empty scene must not launch a zero-block kernel.
        with wp.ScopedCapture(device="cuda:0") as capture:
            advance()
        wp.capture_launch(capture.graph)
    else:
        advance()


@pytest.mark.parametrize("framework", FRAMEWORKS)
@pytest.mark.parametrize("captured", [False, True])
def test_particle_only_native_model_ignores_optional_cable_paths(framework, captured):
    initial = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    velocity = np.asarray([[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]], dtype=np.float32)
    positions_in = wp.array(initial, dtype=wp.vec3, device="cuda:0")
    velocities_in = wp.array(velocity, dtype=wp.vec3, device="cuda:0")
    positions_out = wp.zeros(2, dtype=wp.vec3, device="cuda:0")
    velocities_out = wp.zeros(2, dtype=wp.vec3, device="cuda:0")
    masses = wp.array([1.0, 2.0], dtype=wp.float32, device="cuda:0")
    flags = wp.array([1, 0], dtype=wp.int32, device="cuda:0")
    gravity = wp.zeros(1, dtype=wp.vec3, device="cuda:0")
    model = s3d.Model()
    model.particle_count = 2
    model.particle_masses = _span(masses)
    model.particle_flags = _span(flags)
    model.gravity = _span(gravity)
    solver = s3d.Solver()
    solver.initialize(model)
    solver.config.nonlinear_solver = framework
    solver.config.affine_body_block_preconditioner = True
    solver.config.cable_linear_beta = 100.0
    solver.config.cable_angular_beta = 100.0
    state_in = _state(positions_in, velocities_in)
    state_out = _state(positions_out, velocities_out)

    def advance():
        solver.reset_cable_penalty_state()
        solver.copy_cable_penalties(s3d.Span())
        solver.step(state_in, state_out, 0.01, wp.get_stream("cuda:0"))

    if captured:
        with wp.ScopedCapture(device="cuda:0") as capture:
            advance()
        wp.capture_launch(capture.graph)
    else:
        advance()

    expected = initial.copy()
    expected[0] += 0.01 * velocity[0]
    np.testing.assert_allclose(positions_out.numpy(), expected, rtol=0.0, atol=1.0e-7)
    np.testing.assert_allclose(velocities_out.numpy(), [velocity[0], [0.0, 0.0, 0.0]], atol=1.0e-5)


def test_native_cable_count_requires_complete_metadata():
    model = s3d.Model()
    model.cable_count = 1
    with pytest.raises(ValueError, match="cable metadata is smaller than cable_count"):
        s3d.Solver().initialize(model)


def test_native_cable_buffers_reject_null_addresses_before_solver_entry():
    # Span is immutable, so invalid cable buffers are rejected before either
    # reset or copy can receive them. Empty buffers are exercised above.
    with pytest.raises(ValueError, match="requires a nonzero device address"):
        s3d.Span(ptr=0, size=1)

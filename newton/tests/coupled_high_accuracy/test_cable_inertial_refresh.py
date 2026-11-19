"""Live ADMM affine mass metrics and CUDA graph replay regressions."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton._src.solvers.coupled_high_accuracy.solver_coupled_high_accuracy import NewtonSolver
from newton.solvers import coupled_high_accuracy as s3d

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Inertial refresh requires CUDA")


def _model(*, with_particle=False, prescribed=False):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    if with_particle:
        builder.add_particle(pos=wp.vec3(1.0, 0.0, 0.0), vel=wp.vec3(0.0), mass=3.0)
    body = builder.add_link(
        mass=2.0,
        inertia=wp.mat33(0.4, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.6),
        lock_inertia=True,
    )
    if prescribed:
        builder.body_mass[body] = 0.0
        builder.body_inv_mass[body] = 0.0
        builder.body_inertia[body] = wp.mat33(0.0)
        builder.body_inv_inertia[body] = wp.mat33(0.0)
    return builder.finalize(device="cuda:0"), body


def _set_inertia(model, mass, inertia):
    model.body_mass.assign(np.asarray([mass], dtype=np.float32))
    model.body_inv_mass.assign(np.asarray([1.0 / mass], dtype=np.float32))
    model.body_inertia.assign(np.asarray([inertia], dtype=np.float32))
    model.body_inv_inertia.assign(np.asarray([np.linalg.inv(inertia)], dtype=np.float32))


def _proxy_mass(solver):
    particles = solver._affine_body_indices.numpy()[0]
    result = np.diag(solver._particle_mass_diagonal.numpy()[particles])
    local = {int(particle): axis for axis, particle in enumerate(particles)}
    for pair, value in zip(
        solver._particle_mass_non_diagonal_indices.numpy(), solver._particle_mass_non_diagonal_values.numpy()
    ):
        if int(pair[0]) in local and int(pair[1]) in local:
            first, second = local[int(pair[0])], local[int(pair[1])]
            result[first, second] += value
            result[second, first] += value
    return result


def test_affine_inertial_refresh_keeps_buffer_addresses_and_physical_particle_mass():
    model, _ = _model(with_particle=True)
    solver = NewtonSolver(model)
    assert solver.coupling_supports_inertial_property_refresh()
    names = [
        "_particle_mass_diagonal",
        "_particle_mass_non_diagonal_indices",
        "_particle_mass_non_diagonal_values",
        "_affine_body_masses",
        "_affine_body_inertia",
    ]
    pointers = [getattr(solver, name).ptr for name in names]
    # Initially diagonal inertia still reserves all three mutable column pairs.
    assert len(solver._affine_mass_pair_slots) == 1
    assert np.count_nonzero(solver._particle_mass_non_diagonal_values.numpy()) == 0
    inertia = np.asarray([[0.7, 0.04, -0.03], [0.04, 0.8, 0.02], [-0.03, 0.02, 0.9]], dtype=np.float32)
    _set_inertia(model, 4.0, inertia)
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    assert pointers == [getattr(solver, name).ptr for name in names]
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[0, 0] = 4.0
    expected[1:, 1:] = 0.5 * np.trace(inertia) * np.eye(3) - inertia
    np.testing.assert_allclose(_proxy_mass(solver), expected, atol=1.0e-7)
    assert solver._particle_mass_diagonal.numpy()[0] == 3.0


@pytest.mark.parametrize("framework", [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON])
def test_affine_inertial_refresh_reads_new_angular_metric_during_graph_replay(framework):
    model, body = _model()
    solver = NewtonSolver(model)
    solver.config.nonlinear_solver = framework
    solver.config.nonlinear_iterations = 5
    solver.config.linear_iterations = 30
    solver.config.inertia_mode = s3d.InertiaMode.NONE
    state_in, state_out = model.state(), model.state()
    initial_q = state_in.particle_q.numpy()
    initial_body_q = state_in.body_q.numpy()
    stream = wp.get_stream(model.device)

    def reset_and_force():
        state_in.particle_q.assign(initial_q)
        state_in.particle_qd.zero_()
        state_in.body_q.assign(initial_body_q)
        state_in.body_qd.zero_()
        state_in.clear_forces()
        state_in.body_f.assign(np.asarray([[0.0, 0.0, 0.0, 0.8, -0.4, 0.6]], dtype=np.float32))

    reset_and_force()
    # Warm compilation outside capture; reset clears NONE's additive force scatter.
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    solver.step(state_in, state_out, model.control(), None, 1.0e-3)
    with wp.ScopedCapture(stream=stream) as capture:
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        solver.step(state_in, state_out, model.control(), None, 1.0e-3)
    first_inertia = np.asarray([[0.4, 0.02, -0.03], [0.02, 0.5, 0.04], [-0.03, 0.04, 0.6]], dtype=np.float32)
    _set_inertia(model, 2.0, first_inertia)
    reset_and_force()
    wp.capture_launch(capture.graph, stream=stream)
    first = state_out.body_qd.numpy()[body, 3:]
    second_inertia = 2.0 * first_inertia
    _set_inertia(model, 4.0, second_inertia)
    reset_and_force()
    wp.capture_launch(capture.graph, stream=stream)
    second = state_out.body_qd.numpy()[body, 3:]
    expected = np.linalg.solve(first_inertia, [0.8, -0.4, 0.6]) * 1.0e-3
    # This checks the native cached off-diagonal inertia as well as the predictor:
    # NONE scatters the torque directly, so a stale native ELL matrix fails here.
    np.testing.assert_allclose(first, expected, rtol=0.03, atol=1.0e-6)
    np.testing.assert_allclose(second, 0.5 * first, rtol=0.03, atol=1.0e-6)
    reference_model, reference_body = _model()
    _set_inertia(reference_model, 4.0, second_inertia)
    reference = NewtonSolver(reference_model)
    reference.config.nonlinear_solver = framework
    reference.config.nonlinear_iterations = 5
    reference.config.linear_iterations = 30
    reference.config.inertia_mode = s3d.InertiaMode.NONE
    reference_in, reference_out = reference_model.state(), reference_model.state()
    reference_in.body_f.assign(np.asarray([[0.0, 0.0, 0.0, 0.8, -0.4, 0.6]], dtype=np.float32))
    reference.step(reference_in, reference_out, reference_model.control(), None, 1.0e-3)
    np.testing.assert_allclose(second, reference_out.body_qd.numpy()[reference_body, 3:], rtol=1.0e-3, atol=1.0e-7)


def test_affine_inertial_refresh_preserves_prescribed_mobility_under_proximal_mass():
    model, body = _model(prescribed=True)
    solver = NewtonSolver(model)
    initial_flags = solver._particle_flags.numpy().copy()
    _set_inertia(model, 2.0, np.eye(3, dtype=np.float32))
    # A proximal metric does not grant a prescribed body physical mobility.
    model.body_inv_mass.zero_()
    model.body_inv_inertia.zero_()
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    np.testing.assert_array_equal(solver._particle_flags.numpy(), initial_flags)
    assert solver._fixed_affine_body_ids == [0]
    state_in, state_out = model.state(), model.state()
    state_in.body_f.assign(np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    solver.step(state_in, state_out, model.control(), None, 1.0e-3)
    np.testing.assert_array_equal(state_out.body_q.numpy()[body], state_in.body_q.numpy()[body])


def test_affine_inertial_refresh_preserves_translation_only_proxy_mask():
    model, body = _model()
    solver = NewtonSolver(model)
    model.body_flags.assign(model.body_flags.numpy() | int(newton.BodyFlags.PROXY))
    model.body_inertia.zero_()
    model.body_inv_inertia.zero_()
    solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES | newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    flags = solver._particle_flags.numpy().copy()
    _set_inertia(model, 3.0, np.eye(3, dtype=np.float32))
    model.body_inv_inertia.zero_()
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    np.testing.assert_array_equal(solver._particle_flags.numpy(), flags)
    np.testing.assert_array_equal(solver._affine_body_inertia.numpy()[0], np.eye(3))
    state_in, state_out = model.state(), model.state()
    state_in.body_f.assign(np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    solver.step(state_in, state_out, model.control(), None, 1.0e-3)
    assert state_out.body_q.numpy()[body, 0] > 0.0
    np.testing.assert_allclose(state_out.body_q.numpy()[body, 3:], [0.0, 0.0, 0.0, 1.0], atol=1.0e-7)


@pytest.mark.parametrize("framework", [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON])
def test_eager_affine_inertial_refresh_rebuilds_spring_weights_after_com_edit(framework):
    builder = s3d.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_link(
        mass=2.0,
        inertia=wp.mat33(0.4, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.6),
        lock_inertia=True,
    )
    builder.add_body_spring(body, -1, point0=(0.3, 0.0, 0.0), point1=(1.3, 0.0, 0.0), rest_length=1.0)
    model = builder.finalize(device="cuda:0")
    solver = NewtonSolver(model)
    solver.config.nonlinear_solver = framework
    solver.config.nonlinear_iterations = 5
    solver.config.linear_iterations = 30
    model.body_com.assign(np.asarray([[0.2, 0.0, 0.0]], dtype=np.float32))
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    state_in, state_out = model.state(), model.state()
    # Move the proxy COM to the newly authored COM without moving the body
    # origin or spring endpoint. Cached old weights would stretch this spring.
    solver.sync_affine_body_state(state_in)
    initial = state_in.body_q.numpy().copy()
    for _ in range(10):
        state_in.clear_forces()
        solver.step(state_in, state_out, model.control(), None, 1.0 / 120.0)
        state_in, state_out = state_out, state_in
    np.testing.assert_allclose(state_in.body_q.numpy()[body], initial[body], atol=2.0e-5)


@pytest.mark.parametrize("framework", [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON])
@pytest.mark.parametrize("fixed_mass", [0.0, 2.0])
def test_eager_affine_inertial_refresh_tracks_mobility_changes(framework, fixed_mass):
    model, body = _model()
    solver = NewtonSolver(model)
    solver.config.nonlinear_solver = framework
    solver.config.nonlinear_iterations = 3
    solver.config.linear_iterations = 20
    model.body_mass.assign(np.asarray([fixed_mass], dtype=np.float32))
    model.body_inv_mass.zero_()
    model.body_inv_inertia.zero_()
    if fixed_mass == 0.0:
        model.body_inertia.zero_()
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    assert solver._fixed_affine_body_ids == [0]
    state_in, state_out = model.state(), model.state()
    state_in.body_f.assign(np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    solver.step(state_in, state_out, model.control(), None, 0.01)
    np.testing.assert_array_equal(state_out.body_q.numpy()[body], state_in.body_q.numpy()[body])

    inertia = np.diag([0.4, 0.5, 0.6]).astype(np.float32)
    _set_inertia(model, 2.0, inertia)
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    assert solver._fixed_affine_body_ids == []
    state_in, state_out = model.state(), model.state()
    state_in.body_f.assign(np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    solver.step(state_in, state_out, model.control(), None, 0.01)
    assert state_out.body_q.numpy()[body, 0] > 1.0e-5
    np.testing.assert_allclose(state_out.body_qd.numpy()[body, :3], [0.005, 0.0, 0.0], rtol=0.01, atol=1.0e-6)

"""The optional affine block inverse must preserve the solved equations and mobility."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton._src.solvers.coupled_high_accuracy.solver_coupled_high_accuracy import NewtonSolver
from newton.solvers import coupled_high_accuracy as s3d

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Affine block preconditioning requires CUDA")
FRAMEWORKS = [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON]
STATE_ARRAYS = ("particle_q", "particle_qd", "body_q", "body_qd")


def _configure(solver, framework, *, enabled, nonlinear_iterations=3, linear_iterations=120):
    solver.config.nonlinear_solver = framework
    solver.config.affine_body_block_preconditioner = enabled
    solver.config.affine_body_rigidity_model = s3d.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
    solver.config.constraint_alpha = 1.0
    solver.config.inertia_mode = s3d.InertiaMode.NONE
    solver.config.nonlinear_iterations = nonlinear_iterations
    solver.config.linear_iterations = linear_iterations


def _snapshot(state):
    return {name: getattr(state, name).numpy().copy() for name in STATE_ARRAYS}


def _assert_same_state(actual, expected):
    for name in STATE_ARRAYS:
        assert np.isfinite(actual[name]).all()
        # Positions are stored in float; recovered velocities divide their error by dt.
        atol = 5.0e-4 if name.endswith("qd") else 5.0e-6
        np.testing.assert_allclose(actual[name], expected[name], rtol=1.0e-3, atol=atol, err_msg=name)


def _cable_model():
    builder = s3d.ModelBuilder(gravity=wp.vec3(0.0))
    particle = builder.add_particle(pos=wp.vec3(1.0, 0.0, 0.0), vel=wp.vec3(0.0), mass=3.0)
    points = [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.15), wp.vec3(0.03, 0.0, 0.3)]
    quaternions = newton.Rod(points).quaternions
    bodies, _ = builder.add_rod(
        rod=newton.Rod(points, quaternions=quaternions, radius=0.03),
        stretch_stiffness=2000.0,
        shear_stiffness=1000.0,
        bend_stiffness=0.5,
        twist_stiffness=0.25,
    )
    root = bodies[0]
    builder.body_mass[root] = 0.0
    builder.body_inv_mass[root] = 0.0
    builder.body_inertia[root] = wp.mat33(0.0)
    builder.body_inv_inertia[root] = wp.mat33(0.0)
    builder.color()
    return builder.finalize(device="cuda:0"), bodies, particle


def _run_cable(framework, *, enabled, linear_iterations):
    model, bodies, particle = _cable_model()
    solver = NewtonSolver(model)
    assert not solver.config.affine_body_block_preconditioner
    _configure(solver, framework, enabled=enabled, linear_iterations=linear_iterations)
    state_in, state_out = model.state(), model.state()
    initial = _snapshot(state_in)
    body_forces = np.zeros_like(state_in.body_f.numpy())
    body_forces[bodies[0]] = [50.0, -20.0, 10.0, 1.0, 2.0, 3.0]
    body_forces[bodies[-1]] = [2.0, -1.0, 1.0, 0.08, -0.05, 0.03]
    particle_forces = np.zeros_like(state_in.particle_f.numpy())
    particle_forces[particle] = [0.6, -0.3, 0.9]
    state_in.clear_forces()
    state_in.body_f.assign(body_forces)
    state_in.particle_f.assign(particle_forces)
    dt = 0.01
    solver.step(state_in, state_out, model.control(), None, dt)
    actual = _snapshot(state_out)

    root_affine = int(np.flatnonzero(solver._affine_body_ids.numpy() == bodies[0])[0])
    root_particles = solver._affine_body_indices.numpy()[root_affine]
    flags = solver._particle_flags.numpy()[root_particles]
    assert not np.any(flags & int(newton.ParticleFlags.ACTIVE))
    np.testing.assert_array_equal(actual["body_q"][bodies[0]], initial["body_q"][bodies[0]])
    np.testing.assert_array_equal(actual["particle_q"][root_particles], initial["particle_q"][root_particles])
    np.testing.assert_array_equal(actual["particle_qd"][root_particles], 0.0)
    assert np.linalg.norm(actual["body_q"][bodies[-1], :3] - initial["body_q"][bodies[-1], :3]) > 1.0e-5

    # This row lies outside every affine block and must keep the original Jacobi application.
    expected_velocity = dt * particle_forces[particle] / 3.0
    np.testing.assert_allclose(actual["particle_qd"][particle], expected_velocity, rtol=0.01, atol=5.0e-6)
    np.testing.assert_allclose(
        actual["particle_q"][particle],
        initial["particle_q"][particle] + dt * expected_velocity,
        atol=2.0e-7,
    )
    return actual


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_affine_block_preconditioner_matches_converged_cable_and_free_particle(framework):
    reference = _run_cable(framework, enabled=False, linear_iterations=120)
    longer_reference = _run_cable(framework, enabled=False, linear_iterations=180)
    actual = _run_cable(framework, enabled=True, linear_iterations=120)
    # Verify that the reference is converged before comparing two different PCG recurrences.
    _assert_same_state(reference, longer_reference)
    _assert_same_state(actual, longer_reference)


def _run_offcenter_contact(framework, *, enabled, contact_enabled, linear_iterations):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.09), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.008, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.014),
        lock_inertia=True,
    )
    # An off-center normal and sliding friction exercise translation/rotation
    # cross terms that a centered, frictionless sphere cannot detect.
    builder.add_shape_sphere(
        body=body,
        xform=wp.transform(wp.vec3(0.12, 0.04, 0.0), wp.quat_identity()),
        radius=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, ke=2.0e4, kd=0.0, mu=0.6),
    )
    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device="cuda:0")
    pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
    solver = NewtonSolver(model)
    _configure(solver, framework, enabled=enabled, linear_iterations=linear_iterations)
    solver.config.affine_body_contact_block_preconditioner = contact_enabled
    solver.config.enable_collision_hessian_non_diagonal = True
    solver.config.rigid_contact_hard = True
    solver.config.rigid_contact_history = False
    solver.config.rigid_contact_alpha = 0.95
    solver.config.rigid_contact_gamma = 0.999
    state_in, state_out = model.state(), model.state()
    initial_velocity = np.asarray([[0.2, -0.1, -0.03, 0.1, 0.2, -0.1]], dtype=np.float32)
    state_in.body_qd.assign(initial_velocity)
    solver.sync_affine_body_state(state_in)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    assert int(contacts.rigid_contact_count.numpy()[0]) == 1
    state_in.clear_forces()
    solver.step(state_in, state_out, model.control(), contacts, 0.01)
    actual = _snapshot(state_out)
    assert np.isfinite(actual["body_qd"]).all()
    assert actual["body_qd"][body, 2] > initial_velocity[body, 2] + 1.0e-3
    assert np.linalg.norm(actual["body_qd"][body, 3:] - initial_velocity[body, 3:]) > 1.0e-3
    return actual


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_affine_contact_block_toggle_preserves_converged_hard_contact_solution(framework):
    reference = _run_offcenter_contact(framework, enabled=False, contact_enabled=False, linear_iterations=180)
    longer_reference = _run_offcenter_contact(framework, enabled=False, contact_enabled=False, linear_iterations=240)
    # First establish convergence independently of both dense-block variants.
    _assert_same_state(reference, longer_reference)
    for contact_enabled in (False, True):
        actual = _run_offcenter_contact(
            framework,
            enabled=True,
            contact_enabled=contact_enabled,
            linear_iterations=180,
        )
        _assert_same_state(actual, longer_reference)


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_affine_contact_block_toggle_is_ignored_without_body_blocks(framework):
    without_contact_block = _run_offcenter_contact(
        framework,
        enabled=False,
        contact_enabled=False,
        linear_iterations=8,
    )
    with_contact_block = _run_offcenter_contact(
        framework,
        enabled=False,
        contact_enabled=True,
        linear_iterations=8,
    )
    # The disabled body-block path must select the same Jacobi recurrence,
    # including at low budgets where a preconditioner change is observable.
    for name in STATE_ARRAYS:
        np.testing.assert_array_equal(with_contact_block[name], without_contact_block[name], err_msg=name)


def _free_body_model(*, with_particle=False):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    particle = None
    if with_particle:
        particle = builder.add_particle(pos=wp.vec3(1.0, 0.0, 0.0), vel=wp.vec3(0.0), mass=3.0)
    body = builder.add_link(
        mass=2.0,
        inertia=wp.mat33(0.4, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.6),
        lock_inertia=True,
    )
    return builder.finalize(device="cuda:0"), body, particle


def _set_inertia(model, mass, inertia):
    model.body_mass.assign(np.asarray([mass], dtype=np.float32))
    model.body_inv_mass.assign(np.asarray([1.0 / mass], dtype=np.float32))
    model.body_inertia.assign(np.asarray([inertia], dtype=np.float32))
    model.body_inv_inertia.assign(np.asarray([np.linalg.inv(inertia)], dtype=np.float32))


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_affine_block_preconditioner_preserves_translation_only_proxy(framework):
    def simulate(enabled):
        model, body, _ = _free_body_model()
        solver = NewtonSolver(model)
        _configure(solver, framework, enabled=enabled, nonlinear_iterations=1, linear_iterations=30)
        model.body_flags.assign(model.body_flags.numpy() | int(newton.BodyFlags.PROXY))
        model.body_inertia.zero_()
        model.body_inv_inertia.zero_()
        solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES | newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        indices = solver._affine_body_indices.numpy()[0]
        flags = solver._particle_flags.numpy()[indices]
        assert flags[0] & int(newton.ParticleFlags.ACTIVE)
        assert not np.any(flags[1:] & int(newton.ParticleFlags.ACTIVE))
        state_in, state_out = model.state(), model.state()
        initial_columns = state_in.particle_q.numpy()[indices[1:]].copy()
        state_in.clear_forces()
        state_in.body_f.assign(np.asarray([[2.0, -1.0, 0.5, 0.8, -0.4, 0.6]], dtype=np.float32))
        solver.step(state_in, state_out, model.control(), None, 0.01)
        actual = _snapshot(state_out)
        np.testing.assert_allclose(actual["body_qd"][body, :3], [0.01, -0.005, 0.0025], atol=1.0e-6)
        np.testing.assert_allclose(actual["body_q"][body, 3:], [0.0, 0.0, 0.0, 1.0], atol=1.0e-7)
        np.testing.assert_array_equal(actual["particle_q"][indices[1:]], initial_columns)
        np.testing.assert_array_equal(actual["particle_qd"][indices[1:]], 0.0)
        return actual

    _assert_same_state(simulate(True), simulate(False))


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_affine_block_preconditioner_refreshes_live_inertia_in_cuda_graph(framework):
    model, body, particle = _free_body_model(with_particle=True)
    solver = NewtonSolver(model)
    # A single isolated body block is the complete operator. One PCG iteration
    # therefore checks that the dense inverse is actually used and refreshed.
    _configure(solver, framework, enabled=True, nonlinear_iterations=1, linear_iterations=1)
    state_in, state_out = model.state(), model.state()
    initial = _snapshot(state_in)
    control = model.control()
    stream = wp.get_stream(model.device)
    dt = 1.0e-3

    def reset_and_force(wrench, particle_force):
        for name in STATE_ARRAYS:
            getattr(state_in, name).assign(initial[name])
        state_in.clear_forces()
        state_in.body_f.assign(np.asarray([wrench], dtype=np.float32))
        forces = np.zeros_like(initial["particle_q"])
        forces[particle] = particle_force
        state_in.particle_f.assign(forces)

    reset_and_force([1.0, -0.5, 0.25, 0.8, -0.4, 0.6], [0.6, -0.3, 0.9])
    solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    solver.step(state_in, state_out, control, None, dt)
    reset_and_force([1.0, -0.5, 0.25, 0.8, -0.4, 0.6], [0.6, -0.3, 0.9])
    with wp.ScopedCapture(stream=stream) as capture:
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        solver.step(state_in, state_out, control, None, dt)

    cases = [
        (
            2.0,
            [[0.4, 0.02, -0.03], [0.02, 0.5, 0.04], [-0.03, 0.04, 0.6]],
            [1.0, -0.5, 0.25, 0.8, -0.4, 0.6],
            [0.6, -0.3, 0.9],
        ),
        (
            3.5,
            [[0.9, -0.08, 0.07], [-0.08, 0.7, -0.04], [0.07, -0.04, 1.1]],
            [-0.5, 0.7, 0.2, -0.6, 0.7, 0.3],
            [-0.3, 0.6, -0.9],
        ),
    ]
    for mass, inertia_values, wrench, particle_force in cases:
        inertia = np.asarray(inertia_values, dtype=np.float32)
        _set_inertia(model, mass, inertia)
        reset_and_force(wrench, particle_force)
        wp.capture_launch(capture.graph, stream=stream)
        actual = _snapshot(state_out)
        expected_angular = dt * np.linalg.solve(inertia, np.asarray(wrench[3:]))
        np.testing.assert_allclose(actual["body_qd"][body, 3:], expected_angular, rtol=0.03, atol=1.0e-6)
        np.testing.assert_allclose(actual["body_qd"][body, :3], dt * np.asarray(wrench[:3]) / mass, atol=1.0e-6)
        np.testing.assert_allclose(
            actual["particle_qd"][particle],
            dt * np.asarray(particle_force) / 3.0,
            rtol=0.1,
            atol=5.0e-5,
        )

    reference_model, _, reference_particle = _free_body_model(with_particle=True)
    _set_inertia(reference_model, mass, inertia)
    reference_solver = NewtonSolver(reference_model)
    _configure(reference_solver, framework, enabled=False, nonlinear_iterations=1, linear_iterations=180)
    reference_in, reference_out = reference_model.state(), reference_model.state()
    reference_in.clear_forces()
    reference_in.body_f.assign(np.asarray([wrench], dtype=np.float32))
    reference_forces = np.zeros_like(reference_in.particle_f.numpy())
    reference_forces[reference_particle] = particle_force
    reference_in.particle_f.assign(reference_forces)
    reference_solver.step(reference_in, reference_out, reference_model.control(), None, dt)
    expected = _snapshot(reference_out)
    for name in STATE_ARRAYS:
        atol = 3.0e-5 if name.endswith("qd") else 2.0e-6
        np.testing.assert_allclose(actual[name], expected[name], rtol=1.0e-3, atol=atol, err_msg=name)

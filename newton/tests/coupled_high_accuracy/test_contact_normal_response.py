"""Normal contact response must not inherit the tangential friction stiffness."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton._src.solvers.coupled_high_accuracy import ModelBuilder, NewtonSolver, NonlinearSolver

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Contact response tests require CUDA")


def _normal_contact_response(
    shape: str, friction_gain: float, method: NonlinearSolver
) -> tuple[np.ndarray, np.ndarray]:
    mass = 0.08
    radius = 0.02
    inertia = (0.4 if shape == "sphere" else 2.0 / 3.0) * mass * radius**2
    builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
    cfg = builder.default_shape_cfg
    cfg.density = 0.0
    cfg.ke = 2.0e3
    cfg.kd = 1.0
    cfg.kf = friction_gain
    cfg.mu = 0.3
    cfg.mu_rolling = 0.0
    cfg.mu_torsional = 0.0
    builder.add_ground_plane()
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, radius - 0.001), wp.quat_identity()),
        mass=mass,
        inertia=wp.mat33(inertia, 0.0, 0.0, 0.0, inertia, 0.0, 0.0, 0.0, inertia),
        lock_inertia=True,
    )
    builder.body_qd[body] = wp.spatial_vector(0.0, 0.0, -0.2, 0.0, 0.0, 0.0)
    if shape == "sphere":
        builder.add_shape_sphere(body, radius=radius)
    else:
        builder.add_shape_box(body, hx=radius, hy=radius, hz=radius)
    model = builder.finalize(device="cuda:0")
    solver = NewtonSolver(model)
    solver.config.nonlinear_solver = method
    # Use the demo's finite iteration budget: extra iterations must not hide a directionally incorrect tangent.
    solver.config.nonlinear_iterations = 5
    solver.config.linear_iterations = 50
    solver.config.enable_collision_hessian_non_diagonal = True
    state, following = model.state(), model.state()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    heights, velocities = [], []
    for step in range(4):
        state.clear_forces()
        pipeline.collide(state, contacts)
        if step == 0:
            assert int(contacts.rigid_contact_count.numpy()[0]) > 0
        solver.step(state, following, None, contacts, 1.0 / 480.0)
        state, following = following, state
        heights.append(float(state.body_q.numpy()[body, 2]))
        velocities.append(float(state.body_qd.numpy()[body, 2]))
    return np.asarray(heights), np.asarray(velocities)


@pytest.mark.parametrize("method", (NonlinearSolver.PD_NEWTON, NonlinearSolver.NEWTON))
@pytest.mark.parametrize("shape", ("sphere", "box"))
def test_normal_contact_response_is_independent_of_tangential_friction_gain(
    shape: str,
    method: NonlinearSolver,
) -> None:
    no_friction_heights, no_friction_velocities = _normal_contact_response(shape, 0.0, method)
    friction_heights, friction_velocities = _normal_contact_response(shape, 1.0e3, method)
    assert np.isfinite(friction_heights).all()
    assert np.isfinite(friction_velocities).all()
    # Centered normal motion has zero slip, so changing kf cannot alter the physical normal force.
    assert no_friction_velocities[-1] > -0.15
    np.testing.assert_allclose(friction_velocities, no_friction_velocities, atol=1.0e-3)
    np.testing.assert_allclose(friction_heights, no_friction_heights, atol=1.0e-5)


@pytest.mark.parametrize(
    "method,full_hessian",
    ((NonlinearSolver.NEWTON, False), (NonlinearSolver.NEWTON, True), (NonlinearSolver.PD_NEWTON, True)),
)
@pytest.mark.parametrize("friction_gain", (0.0, 1.0e3))
def test_soft_particle_normal_contact_matches_implicit_solution(
    method: NonlinearSolver,
    full_hessian: bool,
    friction_gain: float,
) -> None:
    mass, radius, height, velocity, dt = 0.1, 0.05, 0.04, -0.2, 1.0 / 120.0
    stiffness, damping = 100.0, 1.0
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
    builder.add_ground_plane()
    builder.add_particle(pos=wp.vec3(0.0, 0.0, height), vel=wp.vec3(0.0, 0.0, velocity), mass=mass, radius=radius)
    model = builder.finalize(device="cuda:0")
    model.soft_contact_ke = stiffness
    model.soft_contact_kd = damping
    model.soft_contact_kf = friction_gain
    model.soft_contact_mu = 1.0
    solver = NewtonSolver(model)
    solver.config.nonlinear_solver = method
    solver.config.nonlinear_iterations = 1
    solver.config.linear_iterations = 50
    solver.config.enable_collision_hessian_non_diagonal = full_hessian
    state, following = model.state(), model.state()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    assert int(contacts.soft_contact_count.numpy()[0]) == 1
    state.clear_forces()
    solver.step(state, following, None, contacts, dt)
    # With zero slip and a fixed normal, this active implicit contact is a scalar linear equation.
    expected_height = mass / dt**2 * (height + dt * velocity) + stiffness * radius + damping / dt * height
    expected_height /= mass / dt**2 + stiffness + damping / dt
    np.testing.assert_allclose(following.particle_q.numpy()[0, 2], expected_height, atol=2.0e-7, rtol=0.0)
    np.testing.assert_allclose(
        following.particle_qd.numpy()[0, 2], (expected_height - height) / dt, atol=3.0e-5, rtol=0.0
    )

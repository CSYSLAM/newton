"""Contact-local spin resistance must dissipate relative rotation and conserve pair momentum."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton._src.solvers.coupled_high_accuracy import ModelBuilder, NewtonSolver, NonlinearSolver

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Torsional friction tests require CUDA")
METHODS = (NonlinearSolver.PD_NEWTON, NonlinearSolver.NEWTON)
RADIUS = 0.25
MASS = 1.0
INERTIA = 0.4 * MASS * RADIUS**2


def _builder(torsional_friction: float, *, gravity: float = -9.81) -> ModelBuilder:
    builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0, 0.0, gravity))
    cfg = builder.default_shape_cfg
    cfg.density = 0.0
    cfg.ke = 1.0e5
    cfg.kd = 100.0
    cfg.kf = 0.0
    cfg.mu = 0.0
    cfg.mu_rolling = 0.0
    cfg.mu_torsional = torsional_friction
    return builder


def _sphere(builder: ModelBuilder, position: tuple[float, float, float], omega: tuple[float, float, float]) -> int:
    body = builder.add_body(
        xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
        mass=MASS,
        inertia=wp.mat33(INERTIA, 0.0, 0.0, 0.0, INERTIA, 0.0, 0.0, 0.0, INERTIA),
        lock_inertia=True,
    )
    builder.body_qd[body] = wp.spatial_vector(0.0, 0.0, 0.0, *omega)
    builder.add_shape_sphere(body, radius=RADIUS)
    return body


def _simulate(
    builder: ModelBuilder,
    method: NonlinearSolver,
    *,
    steps: int,
    dt: float = 1.0 / 240.0,
    configure_model: Callable[[Any], None] | None = None,
) -> tuple[Any, int, NewtonSolver]:
    model = builder.finalize(device="cuda:0")
    if configure_model is not None:
        configure_model(model)
    solver = NewtonSolver(model)
    solver.config.nonlinear_solver = method
    solver.config.nonlinear_iterations = 20
    solver.config.linear_iterations = 30
    solver.config.enable_collision_hessian_non_diagonal = True
    solver.config.rolling_friction_velocity_epsilon = 0.1
    state, following = model.state(), model.state()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    for _ in range(steps):
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, following, None, contacts, dt)
        state, following = following, state
    wp.synchronize_device(model.device)
    return state, int(contacts.rigid_contact_count.numpy()[0]), solver


@pytest.mark.parametrize("method", METHODS)
def test_torsional_friction_stops_loaded_normal_spin(method: NonlinearSolver) -> None:
    def spin(coefficient: float) -> np.ndarray:
        builder = _builder(coefficient)
        builder.add_ground_plane()
        body = _sphere(builder, (0.0, 0.0, RADIUS), (0.0, 0.0, 2.0))
        state, count, _ = _simulate(builder, method, steps=90)
        assert count > 0
        return state.body_qd.numpy()[body]

    free = spin(0.0)
    resisted = spin(0.1)
    np.testing.assert_allclose(free[3:], (0.0, 0.0, 2.0), atol=2.0e-3)
    assert np.isfinite(resisted).all()
    assert np.linalg.norm(resisted[3:]) < 0.05
    assert np.linalg.norm(resisted[:2]) < 1.0e-3


@pytest.mark.parametrize("method", METHODS)
def test_torsional_friction_preserves_airborne_rotation(method: NonlinearSolver) -> None:
    builder = _builder(0.1, gravity=0.0)
    builder.add_ground_plane()
    body = _sphere(builder, (0.0, 0.0, 2.0), (0.0, 0.0, 2.0))
    state, count, _ = _simulate(builder, method, steps=60)
    assert count == 0
    np.testing.assert_allclose(state.body_qd.numpy()[body], (0.0, 0.0, 0.0, 0.0, 0.0, 2.0), atol=2.0e-3)


@pytest.mark.parametrize("method", METHODS)
def test_torsional_friction_transfers_equal_opposite_torques(method: NonlinearSolver) -> None:
    builder = _builder(0.1, gravity=0.0)
    builder.default_shape_cfg.kd = 0.0
    left = _sphere(builder, (-0.2495, 0.0, 0.0), (2.0, 0.0, 0.0))
    right = _sphere(builder, (0.2495, 0.0, 0.0), (0.0, 0.0, 0.0))
    state, count, _ = _simulate(builder, method, steps=1, dt=0.002)
    velocities = state.body_qd.numpy()[[left, right]]
    positions = state.body_q.numpy()[[left, right], :3]
    assert count > 0
    assert np.isfinite(velocities).all()
    assert 0.0 < velocities[0, 3] < 1.999
    assert velocities[1, 3] > 0.001
    np.testing.assert_allclose(MASS * velocities[:, :3].sum(axis=0), 0.0, atol=2.0e-4)
    angular_momentum = (INERTIA * velocities[:, 3:] + np.cross(positions, MASS * velocities[:, :3])).sum(axis=0)
    np.testing.assert_allclose(angular_momentum, (2.0 * INERTIA, 0.0, 0.0), atol=2.0e-4)


@pytest.mark.parametrize("same_material", (True, False), ids=("shared-material", "different-material"))
def test_replicated_torsional_friction_uses_each_world_material(same_material: bool) -> None:
    scene = _builder(0.1)
    _sphere(scene, (0.0, 0.0, RADIUS - 0.001), (0.0, 0.0, 2.0))
    builder = _builder(0.0)
    builder.replicate(scene, 2)
    builder.add_ground_plane()
    # Opposite spins expose a body offset error even when the worlds share identical material storage.
    builder.body_qd[1] = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, -2.0)
    second_coefficient = 0.1 if same_material else 0.0

    def configure_model(model: Any) -> None:
        coefficients = model.shape_material_mu_torsional.numpy()
        coefficients[1] = second_coefficient
        model.shape_material_mu_torsional.assign(coefficients)

    state, count, solver = _simulate(
        builder,
        NonlinearSolver.NEWTON,
        steps=4,
        dt=0.002,
        configure_model=configure_model,
    )
    assert count >= 2
    assert solver.synreal_model.world_count == (2 if same_material else 1)
    assert solver.synreal_model.shape_count == (2 if same_material else 3)
    velocities = state.body_qd.numpy()
    for world, coefficient in enumerate((0.1, second_coefficient)):
        isolated = _builder(coefficient)
        _sphere(isolated, (0.0, 0.0, RADIUS - 0.001), (0.0, 0.0, 2.0 if world == 0 else -2.0))
        isolated.default_shape_cfg.mu_torsional = 0.0
        isolated.add_ground_plane()
        reference, _, _ = _simulate(isolated, NonlinearSolver.NEWTON, steps=4, dt=0.002)
        expected = reference.body_qd.numpy()[0].reshape(2, 3)
        # Float32 reductions can introduce small off-axis errors when PCG uses a different world layout.
        error = np.linalg.norm(velocities[world].reshape(2, 3) - expected, axis=1)
        assert np.all(error <= 2.0e-4 + 5.0e-4 * np.linalg.norm(expected, axis=1)), error
    if not same_material:
        assert abs(float(velocities[0, 5])) < abs(float(velocities[1, 5])) - 0.001

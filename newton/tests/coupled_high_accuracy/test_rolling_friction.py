"""Physical regressions for contact-local rolling resistance."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton._src.solvers.coupled_high_accuracy import ModelBuilder, NewtonSolver, NonlinearSolver

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Rolling friction tests require CUDA")

METHODS = (NonlinearSolver.PD_NEWTON, NonlinearSolver.NEWTON)
RADIUS = 0.25
MASS = 1.0
INERTIA = 0.4 * MASS * RADIUS**2


def _builder(rolling_friction: float, *, gravity: float = -9.81, friction: float = 1.0) -> ModelBuilder:
    builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=gravity)
    cfg = builder.default_shape_cfg
    cfg.density = 0.0
    cfg.ke = 1.0e5
    cfg.kd = 100.0
    cfg.kf = 1.0e4
    cfg.mu = friction
    cfg.mu_rolling = rolling_friction
    cfg.mu_torsional = 0.0
    return builder


def _sphere(
    builder: ModelBuilder,
    position: tuple[float, float, float],
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> int:
    body = builder.add_body(
        xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
        mass=MASS,
        inertia=wp.mat33(INERTIA, 0.0, 0.0, 0.0, INERTIA, 0.0, 0.0, 0.0, INERTIA),
        lock_inertia=True,
    )
    builder.body_qd[body] = wp.spatial_vector(*linear_velocity, *angular_velocity)
    builder.add_shape_sphere(body, radius=RADIUS)
    return body


def _simulate(
    builder: ModelBuilder,
    method: NonlinearSolver,
    *,
    steps: int,
    dt: float = 1.0 / 240.0,
    configure_model: Callable[[Any], None] | None = None,
) -> tuple[Any, Any, int, NewtonSolver]:
    model = builder.finalize(device="cuda:0")
    if configure_model is not None:
        configure_model(model)
    state_in, state_out = model.state(), model.state()
    solver = NewtonSolver(model)
    solver.config.nonlinear_solver = method
    solver.config.nonlinear_iterations = 20
    solver.config.linear_iterations = 30
    solver.config.enable_collision_hessian_non_diagonal = True
    solver.config.rolling_friction_velocity_epsilon = 0.1
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    for _ in range(steps):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in
    wp.synchronize_device(model.device)
    return model, state_in, int(contacts.rigid_contact_count.numpy()[0]), solver


@pytest.mark.parametrize("method", METHODS)
def test_rolling_resistance_stops_translation_and_rotation(method: NonlinearSolver) -> None:
    def roll(coefficient: float) -> np.ndarray:
        builder = _builder(coefficient)
        builder.add_ground_plane()
        # v = R * omega makes the initial contact-point velocity zero.
        body = _sphere(builder, (0.0, 0.0, RADIUS), (0.5, 0.0, 0.0), (0.0, 2.0, 0.0))
        _, state, count, _ = _simulate(builder, method, steps=180)
        assert count > 0
        return state.body_qd.numpy()[body]

    freely_rolling = roll(0.0)
    resisted = roll(0.25)
    assert np.isfinite(resisted).all()
    assert freely_rolling[0] > 0.4
    assert freely_rolling[4] > 1.6
    # Tangential contact forces must slow the center of mass along with the rotation.
    assert abs(float(resisted[0])) < 0.1 * freely_rolling[0]
    assert abs(float(resisted[4])) < 0.1 * freely_rolling[4]


@pytest.mark.parametrize("method", METHODS)
def test_rolling_resistance_does_not_damp_airborne_rotation(method: NonlinearSolver) -> None:
    builder = _builder(0.25, gravity=0.0)
    builder.add_ground_plane()
    body = _sphere(builder, (0.0, 0.0, 2.0), (0.5, 0.0, 0.0), (0.0, 2.0, 0.0))
    _, state, count, _ = _simulate(builder, method, steps=60)
    assert count == 0
    np.testing.assert_allclose(state.body_qd.numpy()[body], (0.5, 0.0, 0.0, 0.0, 2.0, 0.0), atol=2.0e-3)


@pytest.mark.parametrize("method", METHODS)
def test_rolling_resistance_preserves_spin_about_contact_normal(method: NonlinearSolver) -> None:
    builder = _builder(0.25)
    builder.add_ground_plane()
    body = _sphere(builder, (0.0, 0.0, RADIUS), angular_velocity=(0.0, 0.0, 2.0))
    _, state, count, _ = _simulate(builder, method, steps=60)
    assert count > 0
    # Torsional friction is a separate material property; pure normal spin is not rolling.
    np.testing.assert_allclose(state.body_qd.numpy()[body, 3:], (0.0, 0.0, 2.0), atol=2.0e-3)


@pytest.mark.parametrize("method", METHODS)
def test_rolling_resistance_transfers_equal_opposite_torques_between_bodies(method: NonlinearSolver) -> None:
    builder = _builder(0.1, gravity=0.0, friction=0.0)
    builder.default_shape_cfg.kd = 0.0
    left = _sphere(builder, (-0.2495, 0.0, 0.0), angular_velocity=(0.0, 0.0, 2.0))
    right = _sphere(builder, (0.2495, 0.0, 0.0))
    _, state, count, _ = _simulate(builder, method, steps=1, dt=0.002)
    assert count > 0
    velocities = state.body_qd.numpy()[[left, right]]
    positions = state.body_q.numpy()[[left, right], :3]
    assert np.isfinite(velocities).all()
    assert 0.0 < velocities[left, 5] < 1.999
    assert velocities[right, 5] > 0.001
    np.testing.assert_allclose(MASS * velocities[:, :3].sum(axis=0), 0.0, atol=2.0e-4)
    angular_momentum = (INERTIA * velocities[:, 3:] + np.cross(positions, MASS * velocities[:, :3])).sum(axis=0)
    np.testing.assert_allclose(angular_momentum, (0.0, 0.0, 2.0 * INERTIA), atol=2.0e-4)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("same_material", (True, False), ids=("shared-material", "different-material"))
def test_replicated_rolling_resistance_uses_each_world_material(
    method: NonlinearSolver,
    same_material: bool,
) -> None:
    scene = _builder(0.25)
    _sphere(scene, (0.0, 0.0, RADIUS - 0.001), (0.5, 0.0, 0.0), (0.0, 2.0, 0.0))
    builder = _builder(0.0)
    builder.replicate(scene, 2)
    builder.add_ground_plane()
    # Overlapping worlds roll in opposite directions, so an incorrect body offset cannot hide behind symmetry.
    builder.body_qd[1] = wp.spatial_vector(-0.5, 0.0, 0.0, 0.0, -2.0, 0.0)
    second_coefficient = 0.25 if same_material else 0.0

    def configure_model(model: Any) -> None:
        values = model.shape_material_mu_rolling.numpy()
        values[1] = second_coefficient
        model.shape_material_mu_rolling.assign(values)

    _, state, count, solver = _simulate(builder, method, steps=4, dt=0.002, configure_model=configure_model)
    assert count >= 2
    assert solver.synreal_model.world_count == (2 if same_material else 1)
    assert solver.synreal_model.shape_count == (2 if same_material else 3)
    velocities = state.body_qd.numpy()
    for world, coefficient in enumerate((0.25, second_coefficient)):
        isolated = _builder(coefficient)
        direction = 1.0 if world == 0 else -1.0
        _sphere(
            isolated,
            (0.0, 0.0, RADIUS - 0.001),
            (direction * 0.5, 0.0, 0.0),
            (0.0, direction * 2.0, 0.0),
        )
        isolated.default_shape_cfg.mu_rolling = 0.0
        isolated.add_ground_plane()
        _, reference, _, _ = _simulate(isolated, method, steps=4, dt=0.002)
        expected = reference.body_qd.numpy()[0].reshape(2, 3)
        # Float32 PCG reductions differ across layouts, including small errors along nominally zero axes.
        error = np.linalg.norm(velocities[world].reshape(2, 3) - expected, axis=1)
        assert np.all(error <= 2.0e-4 + 5.0e-4 * np.linalg.norm(expected, axis=1)), error
    if not same_material:
        assert abs(float(velocities[0, 4])) < abs(float(velocities[1, 4])) - 0.001

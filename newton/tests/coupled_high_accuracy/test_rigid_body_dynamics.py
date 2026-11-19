import numpy as np
import pytest

wp = pytest.importorskip("warp")

from newton._src.math.spatial import quat_velocity as upstream_quat_velocity
from newton._src.solvers.coupled_high_accuracy.rigid_body_dynamics import (
    VBD_SEMI_IMPLICIT,
    integrate_rigid_body,
    integrate_vbd_semi_implicit,
    recover_rigid_body_end_velocity,
)
from newton._src.solvers.coupled_high_accuracy.solver_coupled_high_accuracy import (
    _predict_affine_body_inertia,
)
from newton._src.solvers.solver import integrate_rigid_body as upstream_integrate_rigid_body

INTEGRATORS = (
    pytest.param(0, id="explicit-euler"),
    pytest.param(1, id="implicit-euler"),
    pytest.param(2, id="lie-group-midpoint"),
    pytest.param(3, id="midpoint-cayley"),
)


@wp.kernel
def _integrate_and_recover(
    inertia: wp.array[wp.mat33],
    angular_momentum: wp.array[wp.vec3],
    dt: float,
    integrator: int,
    next_angular_momentum: wp.array[wp.vec3],
    relative_rotation: wp.array[wp.mat33],
    recovered_end_velocity: wp.array[wp.vec3],
):
    next_momentum, rotation = integrate_rigid_body(
        inertia[0],
        angular_momentum[0],
        dt,
        integrator,
    )
    next_angular_momentum[0] = next_momentum
    relative_rotation[0] = rotation
    recovered_end_velocity[0] = recover_rigid_body_end_velocity(
        inertia[0],
        rotation,
        dt,
        integrator,
    )


def _run_step(
    inertia: np.ndarray,
    angular_velocity: np.ndarray,
    dt: float,
    integrator: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial_momentum = inertia @ angular_velocity
    inertia_wp = wp.array((wp.mat33(inertia),), dtype=wp.mat33, device=device)
    momentum_wp = wp.array((wp.vec3(*initial_momentum),), dtype=wp.vec3, device=device)
    next_momentum_wp = wp.empty(1, dtype=wp.vec3, device=device)
    rotation_wp = wp.empty(1, dtype=wp.mat33, device=device)
    recovered_velocity_wp = wp.empty(1, dtype=wp.vec3, device=device)

    wp.launch(
        _integrate_and_recover,
        dim=1,
        inputs=[inertia_wp, momentum_wp, dt, integrator],
        outputs=[next_momentum_wp, rotation_wp, recovered_velocity_wp],
        device=device,
    )
    wp.synchronize_device(device)
    return (
        next_momentum_wp.numpy()[0],
        rotation_wp.numpy()[0],
        recovered_velocity_wp.numpy()[0],
    )


@pytest.mark.parametrize("integrator", INTEGRATORS)
@pytest.mark.parametrize(
    "device",
    (
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA unavailable"),
        ),
    ),
)
def test_inertia_mode_equations_and_endpoint_recovery(
    integrator: int,
    device: str,
) -> None:
    inertia = np.array(
        ((1.7, 0.3, -0.2), (0.3, 2.4, 0.4), (-0.2, 0.4, 3.1)),
        dtype=np.float32,
    )
    initial_velocity = np.array((0.7, 1.3, -0.4), dtype=np.float32)
    initial_momentum = inertia @ initial_velocity
    dt = 0.02

    next_momentum, rotation, recovered_velocity = _run_step(
        inertia,
        initial_velocity,
        dt,
        integrator,
        device,
    )
    next_velocity = np.linalg.solve(inertia, next_momentum)

    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-6)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=2.0e-6)
    assert np.allclose(recovered_velocity, next_velocity, atol=2.0e-5)

    if integrator == 0:
        expected = initial_momentum + dt * np.cross(initial_momentum, initial_velocity)
        assert np.allclose(next_momentum, expected, atol=2.0e-6)
    elif integrator == 1:
        residual = next_momentum - initial_momentum - dt * np.cross(next_momentum, next_velocity)
        assert np.linalg.norm(residual) < 2.0e-6
    elif integrator == 2:
        assert np.allclose(
            rotation @ next_momentum,
            initial_momentum,
            atol=2.0e-6,
        )
    else:
        initial_energy = 0.5 * initial_velocity @ initial_momentum
        next_energy = 0.5 * next_velocity @ next_momentum
        assert np.allclose(
            rotation @ next_momentum,
            initial_momentum,
            atol=2.0e-6,
        )
        assert np.isclose(next_energy, initial_energy, rtol=2.0e-6)


@pytest.mark.parametrize("integrator", INTEGRATORS)
def test_principal_axis_rotation_is_exact(integrator: int) -> None:
    inertia = np.diag((1.0, 2.0, 3.0)).astype(np.float32)
    initial_velocity = np.array((0.0, 0.0, 4.0), dtype=np.float32)
    dt = 0.05

    next_momentum, rotation, recovered_velocity = _run_step(
        inertia,
        initial_velocity,
        dt,
        integrator,
        "cpu",
    )
    expected_rotation = np.array(
        (
            (np.cos(0.2), -np.sin(0.2), 0.0),
            (np.sin(0.2), np.cos(0.2), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )
    if integrator == 3:
        cayley_angle = 2.0 * np.arctan(0.5 * dt * initial_velocity[2])
        expected_rotation = np.array(
            (
                (np.cos(cayley_angle), -np.sin(cayley_angle), 0.0),
                (np.sin(cayley_angle), np.cos(cayley_angle), 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        )

    assert np.allclose(next_momentum, inertia @ initial_velocity, atol=2.0e-6)
    assert np.allclose(rotation, expected_rotation, atol=2.0e-6)
    assert np.allclose(recovered_velocity, initial_velocity, atol=2.0e-6)


@pytest.mark.parametrize("integrator", INTEGRATORS[:3])
def test_exponential_methods_use_step_angle_for_small_velocity(integrator: int) -> None:
    inertia = np.diag((1.0, 2.0, 3.0)).astype(np.float32)
    initial_velocity = np.array((0.0, 0.0, 1.0e-9), dtype=np.float32)
    dt = 1.0e9

    _, rotation, recovered_velocity = _run_step(
        inertia,
        initial_velocity,
        dt,
        integrator,
        "cpu",
    )
    expected_rotation = np.array(
        (
            (np.cos(1.0), -np.sin(1.0), 0.0),
            (np.sin(1.0), np.cos(1.0), 0.0),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float32,
    )

    assert np.allclose(rotation, expected_rotation, atol=2.0e-6)
    assert np.allclose(recovered_velocity, initial_velocity, rtol=2.0e-6, atol=1.0e-12)


@wp.kernel
def _upstream_rigid_prediction(
    pose: wp.array[wp.transform],
    velocity: wp.array[wp.spatial_vector],
    wrench: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3],
    mass: wp.array[float],
    inertia: wp.array[wp.mat33],
    dt: float,
    predicted_pose: wp.array[wp.transform],
    predicted_velocity: wp.array[wp.spatial_vector],
):
    pose_value, velocity_value = upstream_integrate_rigid_body(
        pose[0],
        velocity[0],
        wrench[0],
        com[0],
        inertia[0],
        1.0 / mass[0],
        wp.inverse(inertia[0]),
        wp.vec3(0.0),
        0.0,
        dt,
    )
    predicted_pose[0] = pose_value
    predicted_velocity[0] = velocity_value


@pytest.mark.parametrize(
    "angular_velocity, torque",
    (
        ((0.7, 1.3, -0.4), (2.4, -1.6, 0.9)),
        ((0.7, 1.3, -0.4), (0.0, 0.0, 0.0)),
        ((0.0, 0.0, 0.0), (2.4, -1.6, 0.9)),
    ),
    ids=("spin-and-torque", "free-spin", "torque-from-rest"),
)
def test_vbd_predictor_matches_upstream_on_cpu(angular_velocity, torque):
    inertia = np.asarray(((1.7, 0.3, -0.2), (0.3, 2.4, 0.4), (-0.2, 0.4, 3.1)), dtype=np.float32)
    axis = np.asarray((0.2, -0.5, 0.7), dtype=np.float32)
    axis /= np.linalg.norm(axis)
    rotation = wp.quat_from_axis_angle(wp.vec3(*axis), 1.2)
    body_pose = wp.array([wp.transform(wp.vec3(0.4, -0.2, 0.7), rotation)], dtype=wp.transform, device="cpu")
    body_velocity = wp.array(
        [wp.spatial_vector(0.3, -0.4, 0.5, *angular_velocity)],
        dtype=wp.spatial_vector,
        device="cpu",
    )
    body_wrench = wp.array(
        [wp.spatial_vector(0.2, 0.7, -0.6, *torque)],
        dtype=wp.spatial_vector,
        device="cpu",
    )
    com = wp.array([wp.vec3(0.07, -0.03, 0.04)], dtype=wp.vec3, device="cpu")
    mass = wp.array([2.3], dtype=float, device="cpu")
    inertia_wp = wp.array([wp.mat33(inertia)], dtype=wp.mat33, device="cpu")
    ids = wp.array([0], dtype=wp.int32, device="cpu")
    fixed = wp.zeros(1, dtype=wp.int32, device="cpu")
    predicted = wp.empty(1, dtype=wp.transform, device="cpu")
    predicted_velocity = wp.empty(1, dtype=wp.spatial_vector, device="cpu")
    reference = wp.empty_like(predicted)
    reference_velocity = wp.empty_like(predicted_velocity)
    dt = 0.07
    wp.launch(
        _predict_affine_body_inertia,
        dim=1,
        inputs=[body_pose, body_velocity, body_wrench, ids, com, mass, inertia_wp, fixed, VBD_SEMI_IMPLICIT, dt],
        outputs=[predicted, predicted_velocity],
        device="cpu",
    )
    wp.launch(
        _upstream_rigid_prediction,
        dim=1,
        inputs=[body_pose, body_velocity, body_wrench, com, mass, inertia_wp, dt],
        outputs=[reference, reference_velocity],
        device="cpu",
    )
    actual_pose, reference_pose = predicted.numpy()[0], reference.numpy()[0]
    assert np.allclose(actual_pose[:3], reference_pose[:3], atol=2.0e-6)
    assert abs(np.dot(actual_pose[3:], reference_pose[3:])) > 1.0 - 2.0e-6
    assert np.allclose(predicted_velocity.numpy(), reference_velocity.numpy(), atol=2.0e-6)


@wp.kernel
def _forced_vbd_step(
    inertia: wp.array[wp.mat33],
    momentum: wp.array[wp.vec3],
    torque: wp.array[wp.vec3],
    dt: float,
    next_momentum: wp.array[wp.vec3],
):
    momentum_value, _rotation = integrate_vbd_semi_implicit(inertia[0], momentum[0], torque[0], dt)
    next_momentum[0] = momentum_value


def test_vbd_gyro_uses_original_spin_before_torque_on_cpu():
    inertia = np.asarray(((1.7, 0.3, -0.2), (0.3, 2.4, 0.4), (-0.2, 0.4, 3.1)), dtype=np.float32)
    spin = np.asarray((0.7, 1.3, -0.4), dtype=np.float32)
    torque = np.asarray((2.4, -1.6, 0.9), dtype=np.float32)
    momentum = inertia @ spin
    dt = 0.07
    expected = momentum + dt * (torque - np.cross(spin, momentum))
    forced_momentum = momentum + dt * torque
    wrong_order = forced_momentum + dt * np.cross(forced_momentum, np.linalg.solve(inertia, forced_momentum))
    output = wp.empty(1, dtype=wp.vec3, device="cpu")
    wp.launch(
        _forced_vbd_step,
        dim=1,
        inputs=[
            wp.array([wp.mat33(inertia)], dtype=wp.mat33, device="cpu"),
            wp.array([wp.vec3(*momentum)], dtype=wp.vec3, device="cpu"),
            wp.array([wp.vec3(*torque)], dtype=wp.vec3, device="cpu"),
            dt,
        ],
        outputs=[output],
        device="cpu",
    )
    assert np.linalg.norm(expected - wrong_order) > 1.0e-3
    assert np.allclose(output.numpy()[0], expected, atol=2.0e-6)


@wp.kernel
def _compare_vbd_endpoint_closure(
    previous: wp.array[wp.quat],
    current: wp.array[wp.quat],
    dt: float,
    actual: wp.array[wp.vec3],
    reference: wp.array[wp.vec3],
):
    relative = wp.quat_inverse(previous[0]) * current[0]
    velocity = recover_rigid_body_end_velocity(
        wp.identity(3, dtype=float),
        wp.quat_to_matrix(relative),
        dt,
        VBD_SEMI_IMPLICIT,
    )
    actual[0] = wp.quat_rotate(current[0], velocity)
    reference[0] = upstream_quat_velocity(current[0], previous[0], dt)


@pytest.mark.parametrize(
    "angle, sign",
    ((1.0e-4, 1.0), (0.8, 1.0), (0.8, -1.0), (3.05, -1.0)),
)
def test_vbd_log_closure_matches_upstream_on_cpu(angle, sign):
    previous = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.2, -0.5, 0.7)), 1.2)
    increment = wp.quat_from_axis_angle(wp.normalize(wp.vec3(-0.3, 0.4, 0.8)), angle)
    current = previous * increment
    current = wp.quat(*(sign * np.asarray(current)))
    actual = wp.empty(1, dtype=wp.vec3, device="cpu")
    reference = wp.empty_like(actual)
    wp.launch(
        _compare_vbd_endpoint_closure,
        dim=1,
        inputs=[
            wp.array([previous], dtype=wp.quat, device="cpu"),
            wp.array([current], dtype=wp.quat, device="cpu"),
            0.02,
        ],
        outputs=[actual, reference],
        device="cpu",
    )
    assert np.allclose(actual.numpy(), reference.numpy(), rtol=2.0e-5, atol=2.0e-5)


def test_vbd_principal_axis_log_recovery_preserves_upstream_attenuation_on_cpu():
    inertia = np.diag((1.0, 2.0, 3.0)).astype(np.float32)
    spin = np.asarray((0.0, 0.0, 4.0), dtype=np.float32)
    dt = 0.05
    momentum, rotation, recovered = _run_step(inertia, spin, dt, VBD_SEMI_IMPLICIT, "cpu")
    angle = 2.0 * np.arctan(0.5 * dt * spin[2])
    expected = np.asarray((0.0, 0.0, angle / dt))
    assert np.allclose(momentum, inertia @ spin, atol=2.0e-6)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-6)
    assert np.allclose(recovered, expected, atol=2.0e-6)
    assert recovered[2] < spin[2]

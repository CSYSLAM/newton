"""Internal bond viscosity dissipates strain while respecting fracture and rigid motion."""

import numpy as np
import pytest
from test_granular import scene, set_positions

wp = pytest.importorskip("warp")
from newton.solvers import coupled_high_accuracy as s3d

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Native granular materials require CUDA")


@pytest.mark.parametrize("method", [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON])
@pytest.mark.parametrize("contact_created", [False, True])
def test_bond_damping_matches_implicit_kelvin_voigt_response_and_conserves_momentum(method, contact_created):
    points = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
    masses = np.array([0.5, 1.5], dtype=np.float32)
    velocities = np.array([[-0.3, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    stiffness, damping, dt = 200.0, 8.0, 0.01
    material = {"cohesion": stiffness, "bond_radius": 0.21}
    if contact_created:
        material = {"adhesion": stiffness, "bond_formation_gap": 0.17}
    _, solver, a, b = scene(
        points,
        masses=masses,
        radius=0.02,
        bond_damping=damping,
        max_compaction=0.0,
        break_strain=1.0,
        method=method,
        iterations=2,
        **material,
    )
    if contact_created:
        # Formation is committed only after an accepted step, with the accepted distance as rest length.
        solver.step(a, b, None, None, dt)
        set_positions(a, points)
    a.particle_qd.assign(velocities)
    solver.step(a, b, None, None, dt)
    inverse_masses = 1.0 / masses.astype(float)
    relative_speed = float(velocities[1, 0] - velocities[0, 0])
    extension = dt * relative_speed / (1.0 + (stiffness * dt**2 + damping * dt) * inverse_masses.sum())
    pair_force = (stiffness + damping / dt) * extension
    expected_velocity = velocities.astype(float)
    expected_velocity[:, 0] += dt * pair_force * np.array([1.0, -1.0]) * inverse_masses
    np.testing.assert_allclose(b.particle_q.numpy(), points + dt * expected_velocity, atol=2.0e-7, rtol=0.0)
    np.testing.assert_allclose(b.particle_qd.numpy(), expected_velocity, atol=2.0e-5, rtol=0.0)
    np.testing.assert_allclose(np.sum(masses[:, None] * b.particle_qd.numpy(), axis=0), 0.0, atol=2.0e-6)


def test_internal_damping_reduces_oscillation_energy():
    energies = []
    for damping in (0.0, 4.0):
        _, solver, a, b = scene(
            [[0, 0, 0], [0.2, 0, 0]],
            radius=0.02,
            cohesion=40.0,
            bond_radius=0.21,
            bond_damping=damping,
            max_compaction=0.0,
            break_strain=1.0,
            iterations=2,
        )
        set_positions(a, [[0, 0, 0], [0.22, 0, 0]])
        for _ in range(100):
            solver.step(a, b, None, None, 0.002)
            a, b = b, a
        positions, velocities = a.particle_q.numpy(), a.particle_qd.numpy()
        extension = np.linalg.norm(positions[1] - positions[0]) - 0.2
        energies.append(0.5 * np.sum(velocities**2) + 0.5 * 40.0 * extension**2)
    assert energies[1] < 0.4 * energies[0]


@pytest.mark.parametrize("damping", [0.0, 80.0])
def test_bond_damping_preserves_uniform_translation(damping):
    points = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
    velocity = np.tile([0.3, -0.2, 0.1], (2, 1)).astype(np.float32)
    _, solver, a, b = scene(
        points,
        velocity=velocity,
        radius=0.02,
        cohesion=200.0,
        bond_radius=0.21,
        bond_damping=damping,
        iterations=2,
    )
    solver.step(a, b, None, None, 0.01)
    np.testing.assert_allclose(b.particle_q.numpy(), points + 0.01 * velocity, atol=2.0e-7, rtol=0.0)
    np.testing.assert_allclose(b.particle_qd.numpy(), velocity, atol=2.0e-6, rtol=0.0)


def test_bond_damping_uses_length_change_and_preserves_finite_rigid_rotation():
    points = np.array([[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float32)
    angle, dt = np.pi / 6.0, 0.01
    rotated = np.outer(points[:, 0], [np.cos(angle), np.sin(angle), 0.0]).astype(np.float32)
    velocity = (rotated - points) / dt
    _, solver, a, b = scene(
        points,
        velocity=velocity,
        radius=0.02,
        cohesion=200.0,
        bond_radius=0.21,
        bond_damping=80.0,
        max_compaction=0.0,
        break_strain=1.0,
        iterations=10,
    )
    # Prescribe an inertial endpoint that rotates the pair without changing its length.
    solver.step(a, b, None, None, dt)
    np.testing.assert_allclose(b.particle_q.numpy(), rotated, atol=2.0e-7, rtol=0.0)
    np.testing.assert_allclose(b.particle_qd.numpy(), velocity, atol=2.0e-5, rtol=0.0)


@pytest.mark.parametrize("contact_created", [False, True])
def test_fracture_removes_bond_damping_permanently(contact_created):
    points = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
    material = {"cohesion": 200.0, "bond_radius": 0.21}
    if contact_created:
        material = {"adhesion": 200.0, "bond_formation_gap": 0.17, "bond_formation_speed": 0.01}
    _, solver, a, b = scene(points, radius=0.02, bond_damping=100.0, **material)
    if contact_created:
        solver.step(a, b, None, None, 0.0001)
    set_positions(a, [[0, 0, 0], [0.24, 0, 0]])
    solver.step(a, b, None, None, 0.0001)
    set_positions(a, points)
    velocity = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
    a.particle_qd.assign(velocity)
    solver.step(a, b, None, None, 0.01)
    np.testing.assert_allclose(b.particle_q.numpy(), points + 0.01 * velocity, atol=2.0e-7, rtol=0.0)
    np.testing.assert_allclose(b.particle_qd.numpy(), velocity, atol=2.0e-6, rtol=0.0)


def test_bond_damping_does_not_drag_unbonded_grains():
    points = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32)
    velocity = np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32)
    _, solver, a, b = scene(points, radius=0.02, velocity=velocity, bond_damping=100.0)
    solver.step(a, b, None, None, 0.01)
    np.testing.assert_allclose(b.particle_q.numpy(), points + 0.01 * velocity, atol=2.0e-7, rtol=0.0)


@pytest.mark.parametrize("damping", [-1.0, float("nan"), float("inf")])
def test_bond_damping_rejects_invalid_values_in_python_and_native_model(damping):
    model, _, _, _ = scene([[0, 0, 0]])
    with pytest.raises(ValueError, match="bond_damping"):
        s3d.configure_granular(model, [0], bond_damping=damping)
    # Bypass the authoring helper so native validation is exercised independently.
    model.synreal_granular_bond_damping = damping
    with pytest.raises(ValueError, match="granular parameters"):
        s3d.NewtonSolver(model)

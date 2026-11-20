"""Contact-created granular bonds retain physical history inside native Newton/PCG."""

import numpy as np
import pytest
from test_granular import scene, set_positions

wp = pytest.importorskip("warp")
from newton.solvers import coupled_high_accuracy as cha

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Native granular materials require CUDA")


@pytest.mark.parametrize("adhesion", [0.0, 200.0])
@pytest.mark.parametrize("method", [cha.NonlinearSolver.NEWTON, cha.NonlinearSolver.PD_NEWTON])
def test_contact_creates_optional_adhesion_with_accepted_rest_length_and_conserves_momentum(adhesion, method):
    masses = np.array([0.5, 1.5], dtype=np.float32)
    _, solver, a, b = scene(
        [[0, 0, 0], [0.09, 0, 0]],
        masses=masses,
        adhesion=adhesion,
        break_strain=0.5,
        method=method,
    )
    # Contact pushes the grains apart before formation, so the accepted rest length differs from the initial one.
    dt = 0.01
    solver.step(a, b, None, None, dt)
    formed = b.particle_q.numpy()
    rest = float(formed[1, 0] - formed[0, 0])
    assert 0.09 < rest < 0.1
    stretched = np.array([[0, 0, 0], [0.11, 0, 0]], dtype=np.float32)
    set_positions(a, stretched)
    solver.step(a, b, None, None, dt)
    inverse_inertia = dt**2 / masses.astype(float)
    displacement = adhesion * (0.11 - rest) * np.array([1.0, -1.0]) * inverse_inertia
    displacement /= 1.0 + adhesion * inverse_inertia.sum()
    expected = stretched.astype(float)
    expected[:, 0] += displacement
    np.testing.assert_allclose(b.particle_q.numpy(), expected, atol=2.0e-7, rtol=0.0)
    np.testing.assert_allclose(np.sum(masses[:, None] * b.particle_qd.numpy(), axis=0), 0.0, atol=1.0e-6)


def test_adhesion_does_not_attract_particles_that_have_never_touched():
    _, solver, a, b = scene([[0, 0, 0], [0.101, 0, 0]], adhesion=200.0, break_strain=0.5)
    solver.step(a, b, None, None, 0.01)
    stretched = np.array([[0, 0, 0], [0.11, 0, 0]], dtype=np.float32)
    set_positions(a, stretched)
    solver.step(a, b, None, None, 0.01)
    np.testing.assert_array_equal(b.particle_q.numpy(), stretched)


@pytest.mark.parametrize("formation_speed, bonds_form", [(0.5, False), (10.0, True)])
def test_bond_formation_speed_rejects_fast_relative_motion(formation_speed, bonds_form):
    _, solver, a, b = scene(
        [[0, 0, 0], [0.1, 0, 0]],
        velocity=[[0, 1, 0], [0, -1, 0]],
        adhesion=200.0,
        bond_formation_gap=0.001,
        bond_formation_speed=formation_speed,
        break_strain=0.5,
    )
    solver.step(a, b, None, None, 0.0001)
    stretched = np.array([[0, 0, 0], [0.11, 0, 0]], dtype=np.float32)
    set_positions(a, stretched)
    solver.step(a, b, None, None, 0.01)
    result = b.particle_q.numpy()
    if bonds_form:
        assert result[1, 0] - result[0, 0] < 0.1099
    else:
        np.testing.assert_array_equal(result, stretched)


def test_adhesion_does_not_duplicate_authored_bonds():
    results = []
    for adhesion in (0.0, 400.0):
        _, solver, a, b = scene(
            [[0, 0, 0], [0.1, 0, 0]],
            cohesion=100.0,
            bond_radius=0.11,
            adhesion=adhesion,
            bond_formation_gap=0.001,
            break_strain=0.5,
        )
        solver.step(a, b, None, None, 0.001)
        set_positions(a, [[0, 0, 0], [0.11, 0, 0]])
        solver.step(a, b, None, None, 0.01)
        results.append(b.particle_q.numpy())
    assert results[0][1, 0] - results[0][0, 0] < 0.1099
    np.testing.assert_allclose(results[1], results[0], atol=2.0e-7, rtol=0.0)


def test_adhesion_does_not_heal_broken_authored_bonds():
    _, solver, a, b = scene(
        [[0, 0, 0], [0.1, 0, 0]],
        cohesion=100.0,
        bond_radius=0.11,
        adhesion=400.0,
        bond_formation_gap=0.001,
        break_strain=0.2,
    )
    for distance in (0.15, 0.1, 0.11):
        set_positions(a, [[0, 0, 0], [distance, 0, 0]])
        solver.step(a, b, None, None, 0.01)
    np.testing.assert_allclose(b.particle_q.numpy(), [[0, 0, 0], [0.11, 0, 0]], atol=1.0e-7, rtol=0.0)


def test_broken_dynamic_bond_cannot_reform_until_the_pair_separates():
    _, solver, a, b = scene([[0, 0, 0], [0.07, 0, 0]], adhesion=200.0, break_strain=0.2)
    solver.step(a, b, None, None, 0.00001)
    # This breaks the short bond while the contact cores still overlap, which must not immediately create a new bond.
    set_positions(a, [[0, 0, 0], [0.099, 0, 0]])
    solver.step(a, b, None, None, 0.00001)
    unbonded = np.array([[0, 0, 0], [0.105, 0, 0]], dtype=np.float32)
    set_positions(a, unbonded)
    solver.step(a, b, None, None, 0.01)
    np.testing.assert_array_equal(b.particle_q.numpy(), unbonded)
    # Once the grains separate, a later slow collision may create a new contact bond.
    set_positions(a, [[0, 0, 0], [0.099, 0, 0]])
    solver.step(a, b, None, None, 0.00001)
    set_positions(a, unbonded)
    solver.step(a, b, None, None, 0.01)
    assert b.particle_q.numpy()[1, 0] - b.particle_q.numpy()[0, 0] < 0.1049


def test_explicit_initial_bonds_select_only_authored_pairs():
    model, solver, a, b = scene(
        [[0, 0, 0], [0.1, 0, 0], [0.3, 0, 0], [0.4, 0, 0]],
        cohesion=200.0,
        bond_indices=[[0, 1]],
        break_strain=0.5,
    )
    np.testing.assert_array_equal(model.synreal_granular_bond_indices.numpy(), [[0, 1]])
    np.testing.assert_allclose(model.synreal_granular_bond_rest_lengths.numpy(), [0.1], atol=1.0e-7, rtol=0.0)
    stretched = np.array([[0, 0, 0], [0.11, 0, 0], [0.3, 0, 0], [0.41, 0, 0]], dtype=np.float32)
    set_positions(a, stretched)
    solver.step(a, b, None, None, 0.01)
    assert b.particle_q.numpy()[1, 0] - b.particle_q.numpy()[0, 0] < 0.1099
    np.testing.assert_array_equal(b.particle_q.numpy()[2:], stretched[2:])


def test_adhesion_never_bonds_overlapping_particles_from_different_worlds():
    points = np.array([[0, 0, 0], [0.1, 0, 0]], dtype=np.float32)
    for worlds in ([0, 0], [0, 1]):
        _, solver, a, b = scene(points, worlds=worlds, adhesion=200.0, bond_formation_gap=0.001, break_strain=0.5)
        solver.step(a, b, None, None, 0.001)
        stretched = np.array([[0, 0, 0], [0.11, 0, 0]], dtype=np.float32)
        set_positions(a, stretched)
        solver.step(a, b, None, None, 0.01)
        if worlds[0] == worlds[1]:
            assert b.particle_q.numpy()[1, 0] - b.particle_q.numpy()[0, 0] < 0.1099
        else:
            np.testing.assert_array_equal(b.particle_q.numpy(), stretched)


def test_dynamic_bond_capacity_limits_owner_coordination():
    outer = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float32)
    outer *= 0.1 / np.sqrt(3.0)
    points = np.vstack((np.zeros((1, 3), dtype=np.float32), outer))
    _, solver, a, b = scene(
        points,
        adhesion=200.0,
        bond_formation_gap=0.0001,
        max_dynamic_bonds=2,
        break_strain=0.5,
    )
    # All four eligible pairs share owner zero; outer grains are too far apart to contact one another.
    solver.step(a, b, None, None, 0.0001)
    stretched = points * 1.1
    set_positions(a, stretched)
    solver.step(a, b, None, None, 0.01)
    moved = np.linalg.norm(b.particle_q.numpy()[1:] - stretched[1:], axis=1)
    assert np.count_nonzero(moved > 1.0e-5) == 2


def test_dynamic_bond_capacity_does_not_truncate_dense_repulsive_contacts():
    grid = np.stack(np.meshgrid(np.arange(5), np.arange(4), np.arange(2), indexing="ij"), axis=-1)
    points = (grid.reshape(-1, 3) * 0.015).astype(np.float32)
    results = []
    for adhesion in (0.0, 200.0):
        _, solver, a, b = scene(points, radius=0.1, adhesion=adhesion, max_dynamic_bonds=1)
        solver.step(a, b, None, None, 0.0001)
        results.append(b.particle_q.numpy())
    # Each grain has 39 repulsive neighbors, exceeding both the bond cap and the contact cache of 32.
    assert np.linalg.norm(results[0] - points) > 1.0e-5
    np.testing.assert_allclose(results[1], results[0], atol=2.0e-7, rtol=0.0)


def test_captured_steps_create_and_use_the_same_dynamic_bonds_as_uncaptured_steps():
    with wp.ScopedDevice("cuda:0"):
        options = {"adhesion": 200.0, "bond_formation_gap": 0.001, "break_strain": 0.5}
        _, direct, da, db = scene([[0, 0, 0], [0.15, 0, 0]], **options)
        _, captured, ca, cb = scene([[0, 0, 0], [0.15, 0, 0]], **options)
        with wp.ScopedCapture() as capture:
            captured.step(ca, cb, None, None, 0.001)
        for distance in (0.15, 0.1, 0.11, 0.09, 0.105):
            points = [[0, 0, 0], [distance, 0, 0]]
            set_positions(da, points)
            set_positions(ca, points)
            direct.step(da, db, None, None, 0.001)
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(cb.particle_q.numpy(), db.particle_q.numpy(), atol=1.0e-6, rtol=0.0)
            np.testing.assert_allclose(cb.particle_qd.numpy(), db.particle_qd.numpy(), atol=1.0e-5, rtol=0.0)
            if distance == 0.11:
                assert db.particle_q.numpy()[1, 0] - db.particle_q.numpy()[0, 0] < distance - 1.0e-6


@pytest.mark.parametrize(
    "invalid",
    [
        {"adhesion": -1.0},
        {"adhesion": float("nan")},
        {"adhesion": float("inf")},
        {"bond_formation_gap": -0.01},
        {"bond_formation_gap": float("inf")},
        {"bond_formation_speed": -0.1},
        {"bond_formation_speed": float("nan")},
        {"max_dynamic_bonds": 0},
        {"max_dynamic_bonds": -1},
        {"max_dynamic_bonds": 1.5},
        {"max_dynamic_bonds": 65},
    ],
)
def test_adhesion_rejects_invalid_material_parameters(invalid):
    with pytest.raises(ValueError):
        scene([[0, 0, 0]], **invalid)


@pytest.mark.parametrize(
    "bonds",
    [
        [0, 1],
        [[0, 1, 2]],
        [[0.0, 1.0]],
        [[0, 0]],
        [[0, 99]],
        [[-1, 0]],
        [[0, 1], [1, 0]],
        [[0, 3]],
    ],
)
def test_explicit_initial_bonds_reject_invalid_pairs(bonds):
    with pytest.raises(ValueError):
        scene(
            [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]],
            indices=[0, 1, 2],
            bond_indices=bonds,
        )


def test_explicit_initial_bonds_reject_cross_world_pairs():
    with pytest.raises(ValueError):
        scene([[0, 0, 0], [0.1, 0, 0]], worlds=[0, 1], bond_indices=[[0, 1]])


def test_explicit_initial_bonds_cannot_also_request_an_automatic_radius():
    with pytest.raises(ValueError):
        scene([[0, 0, 0], [0.1, 0, 0]], bond_indices=[[0, 1]], bond_radius=0.11)

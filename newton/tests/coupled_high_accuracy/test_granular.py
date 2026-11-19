"""Granular contacts and irreversible bond history participate in the native solve."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")
from newton.solvers import coupled_high_accuracy as s3d

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Native granular materials require CUDA")


def scene(
    points,
    *,
    masses=None,
    velocity=None,
    indices=None,
    radius=0.05,
    worlds=None,
    method=s3d.NonlinearSolver.NEWTON,
    iterations=1,
    **material,
):
    points = np.asarray(points, dtype=np.float32)
    masses = np.ones(len(points), dtype=np.float32) if masses is None else np.asarray(masses, dtype=np.float32)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_particles(
        pos=points,
        vel=np.zeros_like(points) if velocity is None else np.asarray(velocity, dtype=np.float32),
        mass=masses,
        radius=np.full(len(points), radius),
    )
    model = builder.finalize(device="cuda:0")
    if worlds is not None:
        model.particle_world.assign(np.asarray(worlds, dtype=np.int32))
    options = {"cohesion": 0.0, "friction": 0.0, **material}
    s3d.configure_granular(model, np.arange(len(points), dtype=np.int32) if indices is None else indices, **options)
    solver = s3d.NewtonSolver(model)
    solver.config.nonlinear_solver = method
    solver.config.nonlinear_iterations = iterations
    solver.config.linear_iterations = 80
    solver.config.enable_self_collision = False
    solver.config.enable_collision_hessian_non_diagonal = True
    return model, solver, model.state(), model.state()


def set_positions(state, points):
    state.particle_q.assign(np.asarray(points, dtype=np.float32))
    state.particle_qd.zero_()


@pytest.mark.parametrize("method", [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON])
@pytest.mark.parametrize("self_collision", [False, True])
def test_contact_step_matches_implicit_pair_spring_without_duplicate_collision(method, self_collision):
    points = np.array([[-0.035, 0.0, 0.0], [0.035, 0.0, 0.0]], dtype=np.float32)
    masses = np.array([0.5, 1.5], dtype=np.float32)
    stiffness, dt, overlap = 2000.0, 0.01, 0.03
    _, solver, a, b = scene(points, masses=masses, contact_stiffness=stiffness, method=method)
    solver.config.enable_self_collision = self_collision
    solver.step(a, b, None, None, dt)
    weights = np.array([-1.0, 1.0])
    inverse_inertia = dt**2 / masses.astype(float)
    displacement = stiffness * overlap * weights * inverse_inertia / (1 + stiffness * inverse_inertia.sum())
    expected = points.astype(float)
    expected[:, 0] += displacement
    np.testing.assert_allclose(b.particle_q.numpy(), expected, atol=2.0e-6, rtol=0.0)
    np.testing.assert_allclose(np.sum(masses[:, None] * b.particle_qd.numpy(), axis=0), 0.0, atol=1.0e-5)


def test_friction_reduces_slip_and_preserves_linear_momentum():
    points = [[-0.035, 0, 0], [0.035, 0, 0]]
    masses = np.array([0.5, 1.5], dtype=np.float32)
    velocity = np.array([[0, 0.3, 0], [0, -0.1, 0]], dtype=np.float32)
    results = []
    for friction in (0.0, 0.8):
        _, solver, a, b = scene(points, masses=masses, velocity=velocity, friction=friction, friction_damping=100.0)
        solver.step(a, b, None, None, 0.002)
        results.append(b.particle_qd.numpy())
    assert abs(results[1][0, 1] - results[1][1, 1]) < abs(results[0][0, 1] - results[0][1, 1])
    assert np.sum(masses * results[1][:, 1] ** 2) < np.sum(masses * results[0][:, 1] ** 2)
    np.testing.assert_allclose(
        np.sum(masses[:, None] * results[1], axis=0),
        np.sum(masses[:, None] * velocity, axis=0),
        atol=1.0e-5,
    )


@pytest.mark.parametrize("method", [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON])
def test_dense_contacts_beyond_neighbor_cache_match_full_pair_matrix(method):
    grid = np.stack(np.meshgrid(np.arange(5), np.arange(4), np.arange(2), indexing="ij"), axis=-1)
    points = (grid.reshape(-1, 3) * 0.015).astype(np.float32)
    masses = np.linspace(0.5, 1.5, len(points), dtype=np.float32)
    dt, stiffness = 0.0001, 2000.0
    force = np.zeros_like(points, dtype=float)
    matrix = np.diag(np.repeat(masses.astype(float) / dt**2, 3))
    # Every grain has 39 contacts, independently checking the uncapped traversal and all off-diagonal blocks.
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            relative = points[i].astype(float) - points[j]
            distance = np.linalg.norm(relative)
            normal = relative / distance
            pair_force = stiffness * (0.2 - distance) * normal
            force[i] += pair_force
            force[j] -= pair_force
            tangent = stiffness * np.outer(normal, normal)
            first, second = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
            matrix[first, first] += tangent
            matrix[second, second] += tangent
            matrix[first, second] -= tangent
            matrix[second, first] -= tangent
    expected = points + np.linalg.solve(matrix, force.reshape(-1)).reshape(-1, 3)
    _, solver, a, b = scene(points, masses=masses, radius=0.1, contact_stiffness=stiffness, method=method)
    solver.step(a, b, None, None, dt)
    np.testing.assert_allclose(b.particle_q.numpy(), expected, atol=2.0e-7, rtol=0.0)
    np.testing.assert_allclose(np.sum(masses[:, None] * b.particle_qd.numpy(), axis=0), 0.0, atol=3.0e-4)


def test_coincident_grains_separate_with_finite_opposite_velocities():
    _, solver, a, b = scene([[0, 0, 0], [0, 0, 0]])
    solver.step(a, b, None, None, 0.01)
    result = b.particle_q.numpy()
    assert np.isfinite(result).all()
    assert np.linalg.norm(result[0] - result[1]) > 0.01
    np.testing.assert_allclose(result.sum(axis=0), 0.0, atol=1.0e-7, rtol=0.0)


def test_different_radii_interact_across_neighboring_hash_cells():
    points = np.array([[0.19, 0, 0], [0.3, 0, 0]], dtype=np.float32)
    _, solver, a, b = scene(points, radius=[0.015, 0.12])
    dt, stiffness = 0.01, 2000.0
    correction = stiffness * (0.135 - 0.11) / (1 / dt**2 + 2 * stiffness)
    expected = points.copy()
    expected[:, 0] += np.array([-correction, correction])
    solver.step(a, b, None, None, dt)
    np.testing.assert_allclose(b.particle_q.numpy(), expected, atol=2.0e-7, rtol=0.0)


@pytest.mark.parametrize("cohesion", [0.0, 200.0])
def test_cohesion_holds_initial_neighbors_but_zero_disables_bonds(cohesion):
    _, solver, a, b = scene([[0, 0, 0], [0.2, 0, 0]], cohesion=cohesion, bond_radius=0.21)
    stretched = np.array([[0, 0, 0], [0.21, 0, 0]], dtype=np.float32)
    set_positions(a, stretched)
    solver.step(a, b, None, None, 0.01)
    if cohesion:
        assert b.particle_q.numpy()[1, 0] - b.particle_q.numpy()[0, 0] < 0.2099
    else:
        np.testing.assert_array_equal(b.particle_q.numpy(), stretched)


def test_broken_cohesive_bond_does_not_heal_when_particles_return():
    _, solver, a, b = scene([[0, 0, 0], [0.2, 0, 0]], cohesion=200.0, bond_radius=0.21)
    set_positions(a, [[0, 0, 0], [0.24, 0, 0]])
    solver.step(a, b, None, None, 0.0001)
    returned = np.array([[0, 0, 0], [0.21, 0, 0]], dtype=np.float32)
    set_positions(a, returned)
    solver.step(a, b, None, None, 0.01)
    np.testing.assert_allclose(b.particle_q.numpy(), returned, atol=1.0e-7, rtol=0.0)


def test_compaction_persists_after_unloading_and_does_not_depend_on_newton_iteration_count():
    results = []
    for iterations in (1, 5):
        _, solver, a, b = scene(
            [[0, 0, 0], [0.2, 0, 0]],
            cohesion=200.0,
            bond_radius=0.21,
            break_strain=0.5,
            iterations=iterations,
        )
        # The short loading step keeps the imposed compression nearly fixed while committing plastic history.
        set_positions(a, [[0, 0, 0], [0.17, 0, 0]])
        solver.step(a, b, None, None, 0.0001)
        set_positions(a, [[0, 0, 0], [0.2, 0, 0]])
        solver.step(a, b, None, None, 0.01)
        result = b.particle_q.numpy()
        assert result[1, 0] - result[0, 0] < 0.1995
        results.append(result)
    np.testing.assert_allclose(results[0], results[1], atol=2.0e-6, rtol=0.0)


@pytest.mark.parametrize("other_material", ["ordinary", "fluid"])
def test_granular_selection_keeps_contact_with_other_materials(other_material):
    model, _, a, b = scene([[0, 0, 0], [0.03, 0, 0]], indices=[0])
    if other_material == "fluid":
        s3d.configure_fluid(model, [1], smoothing_radius=0.1, bulk_modulus=0.0)
    solver = s3d.NewtonSolver(model)
    solver.config.nonlinear_solver = s3d.NonlinearSolver.NEWTON
    solver.config.nonlinear_iterations = 1
    solver.config.linear_iterations = 80
    solver.config.enable_self_collision = True
    solver.config.enable_collision_hessian_non_diagonal = True
    solver.step(a, b, None, None, 0.01)
    result = b.particle_q.numpy()
    assert result[1, 0] - result[0, 0] > 0.04
    np.testing.assert_allclose(np.sum(b.particle_qd.numpy(), axis=0), 0.0, atol=1.0e-5)


def test_overlapping_worlds_do_not_share_granular_contacts_or_bonds():
    points = np.array([[-0.035, 0, 0], [0.035, 0, 0]], dtype=np.float32)
    model, solver, a, b = scene(points, cohesion=200.0, bond_radius=0.08)
    solver.step(a, b, None, None, 0.01)
    expected = b.particle_q.numpy()
    model, solver, a, b = scene(np.tile(points, (2, 1)), worlds=[0, 0, 1, 1], cohesion=200.0, bond_radius=0.08)
    solver.step(a, b, None, None, 0.01)
    np.testing.assert_allclose(b.particle_q.numpy(), np.tile(expected, (2, 1)), atol=2.0e-6, rtol=0.0)


def test_replicated_native_worlds_get_independent_granular_response():
    from test_self_collision_coupling import ContactScene

    results = []
    for world_count in (1, 2):
        contact = ContactScene("particle_pair", True, s3d.NonlinearSolver.NEWTON, world_count=world_count)
        contact.model.granular_particle_count = 2
        contact.model.granular_particle_indices = contact.model.collision_particle_indices
        contact.model.granular_cohesion = 0.0
        contact.model.granular_friction = 0.0
        contact.solver.initialize(contact.model)
        contact.step()
        results.append(contact.position_out.numpy())
    np.testing.assert_allclose(results[1], np.tile(results[0], (2, 1)), atol=2.0e-6, rtol=0.0)


def test_world_reordering_remaps_grains_and_bonds_around_appended_rigid_proxies():
    results = []
    for world_count in (1, 2):
        # Use Newton's builder so the adapter must append and reorder proxies after material authoring.
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for _ in range(world_count):
            builder.begin_world()
            for x in (0.0, 0.2):
                builder.add_particle(pos=(x, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.03)
            builder.add_body(
                xform=wp.transform(wp.vec3(3.0, 0.0, 0.0), wp.quat_identity()),
                mass=2.0,
                inertia=wp.mat33(5.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 7.0),
                lock_inertia=True,
            )
            builder.end_world()
        model = builder.finalize(device="cuda:0")
        s3d.configure_granular(model, np.arange(model.particle_count), bond_radius=0.21)
        np.testing.assert_array_equal(
            model.synreal_granular_bond_indices.numpy(), np.arange(2 * world_count).reshape(-1, 2)
        )
        solver = s3d.NewtonSolver(model)
        expected_pairs = np.arange(world_count)[:, None] * 6 + np.array([0, 1])
        grains = model.synreal_granular_particle_indices.numpy()
        bonds = model.synreal_granular_bond_indices.numpy()
        np.testing.assert_array_equal(grains, expected_pairs.reshape(-1))
        np.testing.assert_array_equal(bonds, expected_pairs)
        worlds = model.particle_world.numpy()
        np.testing.assert_array_equal(worlds[bonds[:, 0]], worlds[bonds[:, 1]])
        np.testing.assert_array_equal(worlds[bonds[:, 0]], np.arange(world_count))
        assert not np.intersect1d(grains, solver._affine_body_indices.numpy()).size
        assert solver.synreal_model.granular_particle_indices.ptr == model.synreal_granular_particle_indices.ptr
        assert solver.synreal_model.granular_bond_indices.ptr == model.synreal_granular_bond_indices.ptr
        solver.config.nonlinear_solver = s3d.NonlinearSolver.NEWTON
        solver.config.nonlinear_iterations = 1
        solver.config.linear_iterations = 80
        solver.config.enable_self_collision = False
        a, b = model.state(), model.state()
        positions = a.particle_q.numpy()
        positions[bonds[:, 1], 0] += 0.01
        a.particle_q.assign(positions)
        solver.step(a, b, None, None, 0.01)
        results.append(b.particle_q.numpy()[grains])
        distances = results[-1][1::2, 0] - results[-1][::2, 0]
        assert np.all(distances < 0.2099)
    np.testing.assert_allclose(results[1], np.tile(results[0], (2, 1)), atol=2.0e-6, rtol=0.0)


def test_granular_particle_transfers_contact_force_to_cloth_face():
    from test_self_collision_coupling import ContactScene

    contact = ContactScene("particle_vertex_face", True, s3d.NonlinearSolver.NEWTON)
    contact.model.granular_particle_count = 1
    contact.model.granular_particle_indices = contact.model.collision_particle_indices
    contact.model.granular_cohesion = 0.0
    contact.solver.initialize(contact.model)
    contact.step()
    result = contact.position_out.numpy()
    assert result[3, 2] > contact.initial[3, 2]
    assert np.all(result[:3, 2] < contact.initial[:3, 2])
    np.testing.assert_allclose(
        np.sum(contact.case.masses[:, None] * (result - contact.initial), axis=0), 0, atol=1.0e-6
    )


@pytest.mark.parametrize("indices", [[0, 0], [-1], [99], [], [0.5]])
def test_material_selection_rejects_invalid_indices(indices):
    builder = newton.ModelBuilder()
    builder.add_particle((0, 0, 0), (0, 0, 0), 1.0)
    model = builder.finalize(device="cuda:0")
    with pytest.raises(ValueError, match="indices"):
        s3d.configure_granular(model, indices)


@pytest.mark.parametrize(
    "invalid",
    [
        {"contact_stiffness": -1.0},
        {"friction": -1.0},
        {"friction_damping": float("nan")},
        {"cohesion": float("inf")},
        {"yield_strain": -0.1},
        {"yield_strain": 1.0},
        {"max_compaction": 1.0},
        {"break_strain": -0.1},
        {"bond_radius": 0.0},
    ],
)
def test_material_rejects_invalid_parameters(invalid):
    builder = newton.ModelBuilder()
    builder.add_particle((0, 0, 0), (0, 0, 0), 1.0)
    model = builder.finalize(device="cuda:0")
    with pytest.raises(ValueError):
        s3d.configure_granular(model, [0], **invalid)


def test_granular_cannot_share_elastic_particles():
    builder = newton.ModelBuilder()
    builder.add_particle((0, 0, 0), (0, 0, 0), 1.0)
    builder.add_particle((0.1, 0, 0), (0, 0, 0), 1.0)
    builder.add_spring(0, 1, ke=100.0, kd=0.0, control=0.0)
    model = builder.finalize(device="cuda:0")
    s3d.configure_granular(model, [0], cohesion=0.0)
    with pytest.raises(ValueError, match="elastic"):
        s3d.NewtonSolver(model)


def test_fluid_and_granular_selections_cannot_overlap():
    model, _, _, _ = scene([[0, 0, 0]], cohesion=0.0)
    with pytest.raises(ValueError, match="overlap|fluid|granular"):
        s3d.configure_fluid(model, [0], smoothing_radius=0.1)
        s3d.NewtonSolver(model)


def test_captured_steps_preserve_the_same_bond_history_as_uncaptured_steps():
    with wp.ScopedDevice("cuda:0"):
        _, direct, da, db = scene([[0, 0, 0], [0.2, 0, 0]], cohesion=200.0, bond_radius=0.21)
        _, captured, ca, cb = scene([[0, 0, 0], [0.2, 0, 0]], cohesion=200.0, bond_radius=0.21)
        with wp.ScopedCapture() as capture:
            captured.step(ca, cb, None, None, 0.001)
        for distance in (0.19, 0.24, 0.21):
            points = [[0, 0, 0], [distance, 0, 0]]
            set_positions(da, points)
            set_positions(ca, points)
            direct.step(da, db, None, None, 0.001)
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(cb.particle_q.numpy(), db.particle_q.numpy(), atol=1.0e-6, rtol=0.0)
            np.testing.assert_allclose(cb.particle_qd.numpy(), db.particle_qd.numpy(), atol=1.0e-5, rtol=0.0)

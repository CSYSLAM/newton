"""Check fluid Newton steps against an independently assembled quadratic model."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")
from newton.solvers import coupled_high_accuracy as cha

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Native fluid requires CUDA")


def make_scene(count, method):
    rng = np.random.default_rng(17)
    points = rng.uniform(-0.025, 0.025, (count, 3)).astype(np.float32)
    velocities = rng.uniform(-0.1, 0.1, (count, 3)).astype(np.float32)
    masses = np.linspace(1.1, 1.4, count).astype(np.float32) / count
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_particles(pos=points, vel=velocities, mass=masses, radius=np.full(count, 0.001))
    model = builder.finalize(device="cuda:0")
    cha.configure_fluid(
        model,
        np.arange(count, dtype=np.int32),
        smoothing_radius=0.1,
        bulk_modulus=2000.0,
        viscosity=0.3,
        boundary_positions=[[0.045, 0.0, 0.0]],
        boundary_volumes=[0.00015],
    )
    solver = cha.NewtonSolver(model)
    solver.config.nonlinear_solver = method
    solver.config.enable_self_collision = False
    solver.config.linear_iterations = 100
    solver.config.fluid_max_step_ratio = 0.0
    return model, solver


def frozen_solution(model, previous, velocity, dt, iterations, reference=None):
    """Assemble density-row outer products and the pairwise viscous Laplacian on the CPU."""
    mass = model.particle_mass.numpy().astype(float)
    count = len(previous)
    prediction = (previous + dt * velocity).astype(np.float32).astype(float)
    reference = prediction if reference is None else reference
    positions = np.concatenate((reference, model.synreal_fluid_boundary_positions.numpy()))
    volumes = np.concatenate((mass / 1000.0, model.synreal_fluid_boundary_volumes.numpy()))
    h = 0.1
    matrix = np.diag(np.repeat(mass / dt**2, 3))
    force = mass[:, None] / dt**2 * (prediction - reference)
    for i in range(count):
        r = positions[i] - positions
        s = np.maximum(1.0 - np.sum(r * r, axis=1) / h**2, 0.0)
        weights = 315.0 / (64.0 * np.pi) * s**3 / h**3
        gradients = -945.0 / (32.0 * np.pi) * s[:, None] ** 2 * r / h**5
        density = volumes @ weights
        row = -volumes[:count, None] * gradients[:count]
        row[i] = np.sum(volumes[:, None] * gradients, axis=0)
        if density > 1.0:
            coefficient = model.synreal_fluid_bulk_modulus * volumes[i]
            force -= coefficient * (density - 1.0) * row
            matrix += coefficient * np.outer(row.reshape(-1), row.reshape(-1))
        for j in range(i + 1, count):
            coefficient = (
                -2.0
                * model.synreal_fluid_viscosity
                * volumes[i]
                * volumes[j]
                * np.dot(gradients[j], r[j])
                / (np.dot(r[j], r[j]) + 0.01 * h**2)
                / dt
            )
            pair_force = -coefficient * (r[j] - (previous[i] - previous[j]))
            force[i] += pair_force
            force[j] -= pair_force
            a, b = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
            tangent = coefficient * np.eye(3)
            matrix[a, a] += tangent
            matrix[b, b] += tangent
            matrix[a, b] -= tangent
            matrix[b, a] -= tangent
    # With fixed A and relaxation=0.5, n Newton steps remove 1 - 0.5**n of the error.
    return reference + (1.0 - 0.5**iterations) * np.linalg.solve(matrix, force.reshape(-1)).reshape(-1, 3)


def newton_solution(model, previous, velocity, dt, iterations):
    reference = None
    # The native solver refreshes its fluid linearization at every Newton iteration.
    for _ in range(iterations):
        reference = frozen_solution(model, previous, velocity, dt, 1, reference)
    return reference


@pytest.mark.parametrize("method", [cha.NonlinearSolver.NEWTON, cha.NonlinearSolver.PD_NEWTON])
@pytest.mark.parametrize("count", [8, 70])
@pytest.mark.parametrize("iterations", [1, 2, 3, 5, 9])
def test_fluid_matches_reference_and_resets_each_step(method, count, iterations):
    # Seventy particles exercise the uncapped fallback beyond the 64-neighbor cache.
    model, solver = make_scene(count, method)
    a, b = model.state(), model.state()
    solver.config.nonlinear_iterations = iterations
    for dt in (0.002, 0.001):
        previous, velocity = a.particle_q.numpy(), a.particle_qd.numpy()
        expected = newton_solution(model, previous, velocity, dt, iterations)
        solver.step(a, b, None, None, dt)
        np.testing.assert_allclose(b.particle_q.numpy(), expected, atol=2.0e-6, rtol=0.0)
        a, b = b, a


def test_fluid_runtime_iteration_changes_match_reference():
    model, solver = make_scene(8, cha.NonlinearSolver.NEWTON)
    results = []
    for iterations in (1, 5, 1):
        solver.config.nonlinear_iterations = iterations
        a, b = model.state(), model.state()
        expected = newton_solution(model, a.particle_q.numpy(), a.particle_qd.numpy(), 0.002, iterations)
        solver.step(a, b, None, None, 0.002)
        results.append(b.particle_q.numpy())
        np.testing.assert_allclose(results[-1], expected, atol=2.0e-6, rtol=0.0)
    np.testing.assert_allclose(results[0], results[2], atol=1.0e-7, rtol=0.0)


@pytest.mark.parametrize("iterations", [2, 5])
def test_fluid_graph_replay_refreshes_linearization(iterations):
    model, solver = make_scene(8, cha.NonlinearSolver.NEWTON)
    solver.config.nonlinear_iterations = iterations
    a, b = model.state(), model.state()
    dt = 0.002
    solver.step(a, b, None, None, dt)
    with wp.ScopedCapture() as capture:
        solver.step(a, b, None, None, dt)
    for offset in (0.0, 0.007):
        positions = model.particle_q.numpy() + np.array([offset, 0.0, 0.0], dtype=np.float32)
        velocity = model.particle_qd.numpy()
        a.particle_q.assign(positions)
        a.particle_qd.assign(velocity)
        expected = newton_solution(model, positions, velocity, dt, iterations)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(b.particle_q.numpy(), expected, atol=2.0e-6, rtol=0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fluid_relaxation", 0.0),
        ("fluid_relaxation", -1.0),
        ("fluid_relaxation", 1.1),
        ("fluid_relaxation", float("nan")),
        ("fluid_relaxation", float("inf")),
        ("fluid_max_step_ratio", -1.0),
        ("fluid_max_step_ratio", float("nan")),
        ("fluid_max_step_ratio", float("inf")),
    ],
)
def test_fluid_rejects_invalid_step_configuration(field, value):
    model, solver = make_scene(8, cha.NonlinearSolver.NEWTON)
    setattr(solver.config, field, value)
    with pytest.raises(ValueError, match="fluid relaxation"):
        solver.step(model.state(), model.state(), None, None, 0.002)

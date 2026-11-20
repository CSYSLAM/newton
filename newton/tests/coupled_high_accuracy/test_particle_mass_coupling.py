"""Non-diagonal particle masses must couple the inertia of particle-only models (no affine bodies)."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton.solvers import coupled_high_accuracy as cha

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="NewtonSolver requires CUDA")


def run(coupling, nonlinear_solver):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
    builder.add_particle(pos=(0.0, 0.0, 0.0), vel=(1.0, 0.0, 0.0), mass=1.0)
    builder.add_particle(pos=(1.0, 0.0, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0)
    builder.add_spring(0, 1, ke=1000.0, kd=0.0, control=0.0)
    model = builder.finalize(device="cuda:0")
    model.spring_rest_length.assign(np.array([0.5], dtype=np.float32))
    model.synreal_particle_mass_diagonal = wp.array([1.0, 1.0], dtype=wp.float32, device=model.device)
    model.synreal_particle_mass_non_diagonal_indices = wp.array([(0, 1)], dtype=wp.vec2i, device=model.device)
    model.synreal_particle_mass_non_diagonal_values = wp.array([coupling], dtype=wp.float32, device=model.device)
    solver = cha.NewtonSolver(model)
    solver.config.nonlinear_solver = nonlinear_solver
    solver.config.nonlinear_iterations, solver.config.linear_iterations = 20, 40
    state, other = model.state(), model.state()
    for _ in range(4):
        state.clear_forces()
        solver.step(state, other, None, None, 0.01)
        state, other = other, state
    wp.synchronize_device(model.device)
    return state.particle_q.numpy(), state.particle_qd.numpy()


@pytest.mark.parametrize("nonlinear_solver", (cha.NonlinearSolver.NEWTON, cha.NonlinearSolver.PD_NEWTON))
def test_particle_only_mass_coupling_conserves_coupled_momentum(nonlinear_solver):
    coupling = 0.5
    positions, velocities = run(coupling, nonlinear_solver)
    uncoupled, _ = run(0.0, nonlinear_solver)
    assert np.isfinite(positions).all() and np.isfinite(velocities).all()
    # The internal spring cannot change M v; M = [[1, c], [c, 1]] per axis.
    momentum = (1.0 + coupling) * (velocities[0] + velocities[1])
    np.testing.assert_allclose(momentum, [(1.0 + coupling) * 1.0, 0.0, 0.0], atol=1.0e-4)
    assert np.abs(positions - uncoupled).max() > 1.0e-4

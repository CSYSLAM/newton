"""A stiff membrane must resolve ground contact within the Newton iteration budget."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton._src.solvers.coupled_high_accuracy import NewtonSolver, NonlinearSolver
from newton.solvers import style3d


@pytest.mark.skipif(not wp.is_cuda_available(), reason="NewtonSolver requires CUDA")
@pytest.mark.parametrize("angle", (0.0, 0.7))
def test_cloth_ground_step_satisfies_implicit_force_balance(angle: float) -> None:
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    builder.add_ground_plane()
    style3d.add_cloth_grid(
        builder,
        pos=wp.vec3(0.0, 0.0, 0.04),
        rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle),
        dim_x=5,
        dim_y=4,
        cell_x=0.1,
        cell_y=0.1,
        vel=wp.vec3(0.0),
        mass=0.1,
        fix_left=True,
        particle_radius=0.05,
        tri_aniso_ke=wp.vec3(1.0e4, 4.0e3, 1.0e3),
        edge_aniso_ke=wp.vec3(0.0),
        edge_kd=0.0,
    )
    model = builder.finalize(device="cuda:0")
    model.soft_contact_ke = 100.0
    model.soft_contact_kd = 0.0
    model.soft_contact_mu = 0.0
    state, following = model.state(), model.state()
    initial = state.particle_q.numpy().copy()
    masses = model.particle_mass.numpy().astype(np.float64)
    free = masses > 0
    # A nonuniform normal velocity excites the membrane's out-of-plane mode at contact.
    velocity = np.zeros_like(initial)
    velocity[free, 2] = -0.2 * (1.0 + np.sin(np.arange(np.count_nonzero(free))))
    state.particle_qd.assign(velocity)
    solver = NewtonSolver(model)
    solver.config.nonlinear_solver = NonlinearSolver.NEWTON
    solver.config.nonlinear_iterations = 10
    solver.config.linear_iterations = 50
    contacts = model.contacts()
    model.collide(state, contacts)
    assert int(contacts.soft_contact_count.numpy()[0]) > 0
    dt = 1.0 / 120.0
    state.clear_forces()
    solver.step(state, following, None, contacts, dt)
    final = following.particle_q.numpy().astype(np.float64)

    indices = model.tri_indices.numpy()
    inverse_dm = model.tri_poses.numpy().astype(np.float64)
    areas = model.tri_areas.numpy().astype(np.float64)
    stiffness = model.style3d.tri_aniso_ke.numpy().astype(np.float64)
    radii = model.particle_radius.numpy().astype(np.float64)
    target = initial.astype(np.float64) + dt * velocity + np.array((0.0, 0.0, -9.81 * dt**2))

    def incremental_energy(points: np.ndarray) -> float:
        edges = points[indices[:, 1:]] - points[indices[:, :1]]
        deformation = edges.transpose(0, 2, 1) @ inverse_dm
        lengths = np.linalg.norm(deformation, axis=1)
        shear = np.sum(deformation[:, :, 0] * deformation[:, :, 1], axis=1)
        shear /= np.sqrt((lengths[:, 0] ** 2 + 1.0e-12) * (lengths[:, 1] ** 2 + 1.0e-12))
        strain = np.column_stack((lengths - 1.0, shear))
        membrane = 0.5 * np.sum(areas[:, None] * stiffness * strain**2)
        inertia = 0.5 * np.sum(masses[:, None] * (points - target) ** 2) / dt**2
        contact = 0.5 * model.soft_contact_ke * np.sum(np.maximum(radii - points[:, 2], 0.0) ** 2)
        return float(membrane + inertia + contact)

    # Differentiate the scalar objective independently of the native force and Hessian assembly.
    residual = []
    epsilon = 1.0e-6
    for particle in np.flatnonzero(free):
        for axis in range(3):
            offset = np.zeros_like(final)
            offset[particle, axis] = epsilon
            residual.append((incremental_energy(final + offset) - incremental_energy(final - offset)) / (2 * epsilon))
    assert np.isfinite(final).all()
    np.testing.assert_array_equal(final[~free], initial[~free])
    assert np.linalg.norm(residual) < 0.01

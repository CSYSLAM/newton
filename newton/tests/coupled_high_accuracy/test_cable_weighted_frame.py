"""Cable torques must use the same mass metric as the exported rigid rotation."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton.solvers import coupled_high_accuracy as cha

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Cable simulation requires CUDA")


def _skew(vector):
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


@pytest.mark.parametrize("framework", [cha.NonlinearSolver.NEWTON, cha.NonlinearSolver.PD_NEWTON])
@pytest.mark.parametrize("rotated_inertia", [False, True])
def test_anisotropic_cable_torque_accelerates_rotation_without_affine_shear(framework, rotated_inertia):
    moment = np.diag([0.001, 0.01, 0.1])
    if rotated_inertia:
        axis = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
        cross = _skew(axis)
        basis = np.eye(3) + np.sin(0.7) * cross + (1.0 - np.cos(0.7)) * cross @ cross
        moment = basis @ moment @ basis.T
    inertia = (np.trace(moment) * np.eye(3) - moment).astype(np.float32)
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(inertia), lock_inertia=True)
    builder.add_joint_rod(
        parent=-1,
        child=body,
        stretch_stiffness=0.0,
        shear_stiffness=0.0,
        bend_stiffness=100.0,
        twist_stiffness=100.0,
    )
    model = builder.finalize(device="cuda:0")
    solver = cha.NewtonSolver(model)
    solver.config.nonlinear_solver = framework
    solver.config.nonlinear_iterations = 1
    solver.config.linear_iterations = 100
    solver.config.constraint_alpha = 0.0
    solver.config.inertia_mode = cha.InertiaMode.NONE
    rest = np.array([0.2, -0.1, 0.15], dtype=np.float32)
    solver._joint_cable_rest_kb.assign(np.array([[rest[0], rest[1], 0.0]], dtype=np.float32))
    solver._joint_cable_rest_twist.assign(np.array([rest[2]], dtype=np.float32))
    state_in, state_out = model.state(), model.state()
    state_in.clear_forces()
    dt = 0.01
    solver.step(state_in, state_out, model.control(), None, dt)

    indices = solver._affine_body_indices.numpy()[0]
    affine = state_out.particle_q.numpy()[indices[1:]].T.astype(np.float64)
    # Disable affine rigidity to expose the cable pullback itself. With the
    # authored mass metric, a pure torque has a purely skew first primal step;
    # an unweighted polar pullback spuriously excites symmetric affine strain.
    angle = np.linalg.solve(inertia.astype(np.float64) + dt**2 * 100.0 * np.eye(3), dt**2 * 100.0 * rest)
    np.testing.assert_allclose(affine, np.eye(3) + _skew(angle), atol=2.0e-6, rtol=0.0)
    np.testing.assert_allclose(state_out.particle_q.numpy()[indices[0]], 0.0, atol=1.0e-7, rtol=0.0)

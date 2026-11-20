"""Contact stabilization and warm starts used by the cable examples."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton.solvers import coupled_high_accuracy as cha

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Rigid contact simulation requires CUDA")


def _overlapping_sphere(framework, *, history=False, upstream=False):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
    builder.default_shape_cfg.mu = 0.0
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.09), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1),
        lock_inertia=True,
    )
    builder.add_shape_sphere(
        body=body,
        radius=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, ke=1.0e4, kd=0.0, mu=0.0),
    )
    builder.add_ground_plane()
    builder.color()
    model = builder.finalize(device="cuda:0")
    pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
    if upstream:
        solver = newton.solvers.SolverVBD(model, iterations=1, rigid_contact_history=history)
    else:
        solver = cha.NewtonSolver(model)
        solver.config.nonlinear_solver = framework
        solver.config.nonlinear_iterations = 1
        solver.config.rigid_contact_hard = True
        solver.config.rigid_contact_history = history
    return model, solver, pipeline, body


@pytest.mark.parametrize("framework", [cha.NonlinearSolver.NEWTON, cha.NonlinearSolver.PD_NEWTON])
def test_hard_contact_initial_violation_matches_analytic_alpha(framework):
    # The engineRTS original also compared against Newton 1.7's SolverVBD hard contacts; Newton main replaced
    # that VBD contact model, so only the analytic first-solve value is checked here.
    heights = []
    for upstream in (False,):
        model, solver, pipeline, body = _overlapping_sphere(framework, upstream=upstream)
        state_in, state_out = model.state(), model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state_in, contacts)
        state_in.clear_forces()
        solver.step(state_in, state_out, model.control(), contacts, 1.0e-3)
        heights.append(float(state_out.body_q.numpy()[body, 2]))
    # alpha=.95 removes 95% of the initial overlap from the first primal solve.
    expected = np.float32(0.09) + 0.05 * 0.01 * 1.0e4 / (1.0e6 + 1.0e4)
    np.testing.assert_allclose(heights, expected, atol=2.0e-8, rtol=0.0)


@pytest.mark.parametrize("framework", [cha.NonlinearSolver.NEWTON, cha.NonlinearSolver.PD_NEWTON])
def test_matched_contact_history_accumulates_normal_support(framework):
    heights = []
    for history in (False, True):
        model, solver, pipeline, body = _overlapping_sphere(framework, history=history)
        state_in, state_out = model.state(), model.state()
        contacts = pipeline.contacts()
        for _ in range(4):
            pipeline.collide(state_in, contacts)
            state_in.clear_forces()
            solver.step(state_in, state_out, model.control(), contacts, 1.0e-3)
            state_in, state_out = state_out, state_in
        heights.append(float(state_in.body_q.numpy()[body, 2]))
    assert np.isfinite(heights).all()
    assert heights[1] > heights[0] + 1.0e-5

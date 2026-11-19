"""Hard cable contacts preserve AVBD stabilization and numeric contact history."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton.solvers import coupled_high_accuracy as s3d

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Hard contact simulation requires CUDA")


def _sphere_contact_scene(framework, history):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
    builder.default_shape_cfg.mu = 0.0
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.09), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
        lock_inertia=True,
    )
    builder.add_shape_sphere(body=body, radius=0.1)
    builder.add_ground_plane()
    model = builder.finalize(device="cuda:0")
    solver = s3d.NewtonSolver(model)
    # Coupled views can create a local collision pipeline after solver setup.
    # Its capacity must drive allocation even when the model estimate was zero.
    capacity = max(32, int(getattr(model, "rigid_contact_max", 0) or 0) + 7)
    pipeline = newton.CollisionPipeline(model, contact_matching="sticky", rigid_contact_max=capacity)
    assert solver._rigid_contact_capacity < pipeline.rigid_contact_max
    solver.config.nonlinear_solver = framework
    solver.config.nonlinear_iterations = 1
    solver.config.linear_iterations = 20
    solver.config.rigid_contact_hard = True
    solver.config.rigid_contact_history = history
    solver.config.rigid_contact_alpha = 0.95
    solver.config.rigid_contact_gamma = 0.999
    return model, pipeline, solver, body


@pytest.mark.parametrize("framework", [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON])
@pytest.mark.parametrize("history", [False, True])
def test_hard_contact_matches_scalar_avbd_recurrence(framework, history):
    model, pipeline, solver, body = _sphere_contact_scene(framework, history)
    state_in, state_out = model.state(), model.state()
    contacts = pipeline.contacts()
    control = model.control()
    dt = 1.0e-3
    stiffness = 1.0e4
    alpha, gamma = 0.95, 0.999
    position = float(state_in.body_q.numpy()[body, 2])
    velocity, multiplier = 0.0, 0.0
    mass = float(model.body_mass.numpy()[body])
    observed, expected = [], []

    for step in range(4):
        pipeline.collide(state_in, contacts)
        assert int(contacts.rigid_contact_count.numpy()[0]) == 1
        if step:
            assert int(contacts.rigid_contact_match_index.numpy()[0]) >= 0
        state_in.clear_forces()
        solver.step(state_in, state_out, control, contacts, dt)
        observed.append(float(state_out.body_q.numpy()[body, 2]))

        # With a centered sphere and zero friction, AVBD reduces to one scalar
        # primal solve followed by one unilateral multiplier update.
        initial_overlap = 0.1 - position
        prediction = position + dt * velocity
        decayed_multiplier = alpha * gamma * multiplier if history else 0.0
        stabilized_overlap = 0.1 - prediction - alpha * initial_overlap
        normal_load = max(stiffness * stabilized_overlap + decayed_multiplier, 0.0)
        updated_position = prediction + normal_load / (mass / dt**2 + stiffness)
        multiplier = max(decayed_multiplier + stiffness * (0.1 - updated_position - alpha * initial_overlap), 0.0)
        velocity = (updated_position - position) / dt
        position = updated_position
        expected.append(position)
        state_in, state_out = state_out, state_in

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=2.0e-7)


@pytest.mark.parametrize("framework", [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON])
def test_hard_contact_history_survives_cuda_graph_replay(framework):
    def simulate(captured):
        model, pipeline, solver, body = _sphere_contact_scene(framework, True)
        states = [model.state(), model.state()]
        contacts = pipeline.contacts()
        control = model.control()
        solver.prepare_contacts(contacts)

        def advance_pair():
            for source, target in ((states[0], states[1]), (states[1], states[0])):
                pipeline.collide(source, contacts)
                source.clear_forces()
                solver.step(source, target, control, contacts, 1.0e-3)

        if captured:
            with wp.ScopedCapture(device=model.device) as capture:
                # An already prepared buffer must remain a no-op in capture.
                solver.prepare_contacts(contacts)
                advance_pair()
            for _ in range(3):
                wp.capture_launch(capture.graph)
        else:
            for _ in range(3):
                advance_pair()
        return states[0].body_q.numpy()[body]

    np.testing.assert_allclose(simulate(True), simulate(False), rtol=0.0, atol=2.0e-7)


@pytest.mark.parametrize("framework", [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON])
def test_separating_hard_contact_releases_saved_normal_support(framework):
    model, pipeline, solver, body = _sphere_contact_scene(framework, True)
    state_in, state_out = model.state(), model.state()
    contacts = pipeline.contacts()
    control = model.control()
    dt = 1.0e-3
    pipeline.collide(state_in, contacts)
    assert int(contacts.rigid_contact_count.numpy()[0]) == 1
    state_in.clear_forces()
    solver.step(state_in, state_out, control, contacts, dt)
    state_in, state_out = state_out, state_in
    assert state_in.body_q.numpy()[body, 2] > 0.09

    velocity = np.zeros_like(state_in.body_qd.numpy())
    velocity[body, 2] = 20.0
    state_in.body_qd.assign(velocity)
    solver.sync_affine_body_state(state_in)
    for _ in range(2):
        initial_height = float(state_in.body_q.numpy()[body, 2])
        # Reuse the contact record so release is tested inside the solver,
        # without letting narrow phase hide a stale attractive multiplier.
        solver.set_rigid_history_update(False)
        state_in.clear_forces()
        solver.step(state_in, state_out, control, contacts, dt)
        actual = state_out.body_q.numpy()[body, 2]
        np.testing.assert_allclose(actual, initial_height + dt * 20.0, rtol=0.0, atol=3.0e-7)
        np.testing.assert_allclose(state_out.body_qd.numpy()[body, 2], 20.0, rtol=0.0, atol=4.0e-4)
        state_in, state_out = state_out, state_in

    poses = state_in.body_q.numpy()
    poses[body, 2] = 0.1
    state_in.body_q.assign(poses)
    state_in.body_qd.zero_()
    solver.sync_affine_body_state(state_in)
    solver.set_rigid_history_update(False)
    state_in.clear_forces()
    solver.step(state_in, state_out, control, contacts, dt)
    # Returning exactly to the surface exposes any multiplier that survived
    # separation, even when it was too small to overcome the outgoing gap.
    np.testing.assert_allclose(state_out.body_q.numpy()[body, 2], 0.1, rtol=0.0, atol=2.0e-7)

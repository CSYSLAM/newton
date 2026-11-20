"""Rest-model edits and selected-world cable hysteresis resets."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton._src.solvers.coupled_high_accuracy.cable_material import (
    ROD_MATERIAL_DOFS,
    register_cable_attributes,
)
from newton._src.solvers.coupled_high_accuracy.solver_coupled_high_accuracy import NewtonSolver
from newton.solvers import coupled_high_accuracy as cha

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Cable lifecycle tests require CUDA")


def _model(*, worlds=1, global_cable=False):
    builder = cha.ModelBuilder(gravity=wp.vec3(0.0))
    register_cable_attributes(builder)
    cables = []

    def add_cable(offset):
        points = [wp.vec3(offset, 0.0, 0.1 * index) for index in range(3)]
        frames = newton.Rod(points).quaternions
        bodies, joints = builder.add_rod(
            rod=newton.Rod(points, quaternions=frames, radius=0.01),
            stretch_stiffness=2000.0,
            shear_stiffness=1000.0,
            bend_stiffness=0.5,
            twist_stiffness=0.25,
        )
        root = bodies[0]
        builder.body_mass[root] = 0.0
        builder.body_inv_mass[root] = 0.0
        builder.body_inertia[root] = wp.mat33(0.0)
        builder.body_inv_inertia[root] = wp.mat33(0.0)
        cables.append((bodies, joints))

    for world in range(worlds):
        builder.begin_world()
        add_cable(float(world))
        builder.end_world()
    if global_cable:
        add_cable(float(worlds))
    builder.color()
    model = builder.finalize(device="cuda:0")
    return model, cables


@pytest.mark.parametrize("framework", [cha.NonlinearSolver.NEWTON, cha.NonlinearSolver.PD_NEWTON])
@pytest.mark.parametrize("edited_rest", ["joint_frame", "body_pose"])
def test_notified_cable_rest_edit_remains_force_free(framework, edited_rest):
    model, cables = _model()
    bodies, joints = cables[0]
    solver = NewtonSolver(model)
    solver.config.nonlinear_solver = framework
    solver.config.nonlinear_iterations = 5
    solver.config.linear_iterations = 30
    previous_rest = solver._cable_rest.numpy()
    rotation = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.3))
    if edited_rest == "joint_frame":
        frames = model.joint_X_c.numpy()
        frames[joints[-1], 3:] = rotation
        model.joint_X_c.assign(frames)
        flag = newton.ModelFlags.JOINT_PROPERTIES
    else:
        poses = model.body_q.numpy()
        poses[bodies[-1], 3:] = rotation
        model.body_q.assign(poses)
        flag = newton.ModelFlags.BODY_PROPERTIES
    solver.notify_model_changed(flag)
    assert np.max(np.abs(solver._cable_rest.numpy() - previous_rest)) > 0.1
    state_in, state_out = model.state(), model.state()
    solver.sync_affine_body_state(state_in)
    initial = state_in.body_q.numpy()
    for _ in range(3):
        state_in.clear_forces()
        solver.step(state_in, state_out, model.control(), None, 1.0e-3)
        state_in, state_out = state_out, state_in
    np.testing.assert_allclose(state_in.body_q.numpy(), initial, atol=2.0e-5)


def test_material_notification_preserves_cable_rest_invariants():
    model, cables = _model()
    solver = NewtonSolver(model)
    rest = solver._cable_rest.numpy().copy()
    # A material-only notification must not silently author a new rest pose.
    poses = model.body_q.numpy()
    poses[cables[0][0][-1], 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.3))
    model.body_q.assign(poses)
    stiffness = model.joint_target_ke.numpy()
    model.joint_target_ke.assign(2.0 * stiffness)
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    np.testing.assert_array_equal(solver._cable_rest.numpy(), rest)
    start = int(model.joint_qd_start.numpy()[cables[0][1][0]])
    np.testing.assert_array_equal(
        solver._cable_stiffnesses.numpy()[0], 2.0 * stiffness[start + np.asarray(ROD_MATERIAL_DOFS)]
    )


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("selected_slot", [0, 2])
def test_cable_reset_selects_world_and_rebaselines_next_input_pose(captured, selected_slot):
    model, cables = _model(worlds=2, global_cable=True)
    model.vbd.dahl_eps_max.fill_(0.1)
    model.vbd.dahl_tau.fill_(0.01)
    solver = NewtonSolver(model)
    state = model.state()
    selected = model.joint_world.numpy() == (selected_slot if selected_slot < 2 else -1)
    mask_values = np.zeros(3, dtype=bool)
    mask_values[selected_slot] = True
    mask = wp.array(mask_values, dtype=wp.bool, device=model.device)
    history_names = (
        "joint_sigma_prev",
        "joint_kappa_prev",
        "_joint_dkappa_prev",
        "_joint_sigma_start",
        "_joint_C_fric",
    )
    sentinel = np.tile(np.asarray([0.2, -0.3, 0.4], dtype=np.float32), (int(model.joint_count), 1))

    def seed_history():
        solver._cable_rebaseline_mask.zero_()
        for name in history_names:
            getattr(solver, name).assign(sentinel)

    seed_history()
    before_body, before_particles = state.body_q.numpy(), state.particle_q.numpy()
    solver.reset(state, mask, flags=0)
    np.testing.assert_array_equal(state.body_q.numpy(), before_body)
    np.testing.assert_array_equal(state.particle_q.numpy(), before_particles)
    np.testing.assert_array_equal(solver._cable_rebaseline_mask.numpy(), mask_values)
    for name in history_names:
        history = getattr(solver, name).numpy()
        np.testing.assert_array_equal(history[selected], 0.0)
        np.testing.assert_array_equal(history[~selected], sentinel[~selected])

    selected_body = cables[selected_slot][0][-1]
    poses = state.body_q.numpy()
    poses[selected_body, 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.3))
    state.body_q.assign(poses)
    if captured:
        # Warm the same kernels before capture; replay must consume a live pose.
        solver._prepare_cable_dahl(state)
        seed_history()
        with wp.ScopedCapture(device=model.device) as capture:
            solver.reset(state, mask, flags=0)
            solver._prepare_cable_dahl(state)
        wp.capture_launch(capture.graph)
        poses[selected_body, 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.4))
        state.body_q.assign(poses)
        seed_history()
        wp.capture_launch(capture.graph)
    else:
        solver._prepare_cable_dahl(state)
    assert np.max(np.abs(solver.joint_kappa_prev.numpy()[selected])) > (0.35 if captured else 0.25)
    np.testing.assert_array_equal(solver.joint_sigma_prev.numpy()[selected], 0.0)
    np.testing.assert_array_equal(solver._joint_dkappa_prev.numpy()[selected], 0.0)
    np.testing.assert_array_equal(solver._joint_sigma_start.numpy()[selected], 0.0)


@pytest.mark.parametrize("captured", [False, True])
def test_reset_restores_only_selected_world_state_and_affine_proxies(captured):
    model, _ = _model(worlds=2, global_cable=True)
    solver = NewtonSolver(model)
    state = model.state()
    mask = wp.array([True, False, False], dtype=wp.bool, device=model.device)
    body_selected = model.body_world.numpy() == 0
    particle_selected = model.particle_world.numpy() == 0
    fields = ("body_q", "body_qd", "particle_q", "particle_qd")
    authored = {}
    for name in fields:
        authored[name] = getattr(state, name).numpy()
        authored[name][:, :3] += 0.25
        getattr(state, name).assign(authored[name])
    if captured:
        solver.reset(state, mask)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.reset(state, mask)
        for name in fields:
            getattr(state, name).assign(authored[name])
        wp.capture_launch(capture.graph)
    else:
        solver.reset(state, mask)
    for name in fields:
        selected = body_selected if name.startswith("body_") else particle_selected
        result = getattr(state, name).numpy()
        np.testing.assert_allclose(result[selected], getattr(model, name).numpy()[selected], atol=1.0e-7)
        np.testing.assert_array_equal(result[~selected], authored[name][~selected])


def test_velocity_only_reset_accepts_missing_body_pose():
    model, _ = _model()
    solver = NewtonSolver(model)
    state = model.state()
    state.body_q = None
    velocities = state.body_qd.numpy()
    velocities[:] = 0.25
    state.body_qd.assign(velocities)
    proxy_velocities = state.particle_qd.numpy()
    solver.reset(state, flags=newton.StateFlags.BODY_QD)
    assert state.body_q is None
    np.testing.assert_array_equal(state.body_qd.numpy(), model.body_qd.numpy())
    # Proxy velocity axes require a pose; incomplete public state cannot supply it.
    np.testing.assert_array_equal(state.particle_qd.numpy(), proxy_velocities)

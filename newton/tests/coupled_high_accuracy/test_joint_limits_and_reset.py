"""Regressions for unilateral angular stops and independent world resets."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from test_joint_mujoco_semantics import _advance, _make_scene

from newton.solvers import coupled_high_accuracy as s3d

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="NewtonSolver requires CUDA")


@pytest.mark.parametrize("sign", [-1, 1], ids=["lower", "upper"])
@pytest.mark.parametrize("floating_parent", [False, True])
@pytest.mark.parametrize("framework", [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON])
def test_angular_stop_reacts_without_motor_and_preserves_momentum(sign, floating_parent, framework):
    scene = _make_scene(
        "revolute",
        "synreal",
        initial_position=sign * 0.22,
        floating_parent=floating_parent,
        lower_limit=-0.2,
        upper_limit=0.2,
        limit_ke=100.0,
        limit_kd=0.0,
        effort_limit=0.0,
    )
    scene[1].config.nonlinear_solver = framework
    velocity, body_velocity = _advance(scene, "revolute")
    inertia = 0.02 if floating_parent else 0.04
    dt = 0.00125
    expected = -sign * 100.0 * 0.02 * dt / (inertia + 100.0 * dt**2)
    np.testing.assert_allclose(velocity, expected, rtol=0.03, atol=2.0e-4)
    if floating_parent:
        np.testing.assert_allclose(body_velocity.sum(), 0.0, atol=2.0e-5)


@pytest.mark.parametrize("q, velocity", [(0.0, 0.3), (0.22, -1.0), (-0.22, 1.0)])
@pytest.mark.parametrize("framework", [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON])
def test_angular_stop_is_inactive_inside_range_and_does_not_attract(q, velocity, framework):
    options = dict(initial_position=q, initial_velocity=velocity)
    bounded = _make_scene(
        "revolute", "synreal", lower_limit=-0.2, upper_limit=0.2, limit_ke=100.0, limit_kd=10.0, **options
    )
    free = _make_scene("revolute", "synreal", **options)
    bounded[1].config.nonlinear_solver = framework
    free[1].config.nonlinear_solver = framework
    actual, _ = _advance(bounded, "revolute")
    expected, _ = _advance(free, "revolute")
    np.testing.assert_allclose(actual, expected, rtol=0.002, atol=1.0e-5)


@pytest.mark.parametrize(
    "options, message",
    [
        ({"joint_limit_lower": 0.2, "joint_limit_upper": -0.2}, "Joint limits"),
        ({"joint_limit_lower": np.nan}, "Joint limits"),
        ({"joint_limit_upper": np.nan}, "Joint limits"),
        ({"joint_limit_ke": -1.0}, "joint_limit_ke"),
        ({"joint_limit_kd": -1.0}, "joint_limit_kd"),
        ({"joint_limit_ke": np.inf}, "joint_limit_ke"),
        ({"joint_limit_kd": np.nan}, "joint_limit_kd"),
    ],
)
def test_invalid_angular_stop_parameters_are_rejected(options, message):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), lock_inertia=True)
    joint = builder.add_joint_revolute(parent=-1, child=body)
    builder.add_articulation([joint])
    model = builder.finalize(device="cuda:0")
    # The builder normalizes non-finite bounds; exercise native model edits directly.
    for name, value in options.items():
        getattr(model, name).fill_(value)
    with pytest.raises(ValueError, match=message):
        s3d.NewtonSolver(model)


def _replicated_scene(upper_limits, *, initial_q=0.0, initial_qd=0.0):
    source = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
    body = source.add_link(
        mass=2.0, inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.04), lock_inertia=True
    )
    joint = source.add_joint_revolute(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        limit_lower=-100.0,
        limit_upper=100.0,
        limit_ke=1000.0,
        limit_kd=1.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    source.add_articulation([joint])
    source.joint_q[0] = initial_q
    source.joint_qd[0] = initial_qd
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
    builder.replicate(source, len(upper_limits))
    builder.joint_limit_upper[:] = list(upper_limits)
    model = builder.finalize(device="cuda:0")
    solver = s3d.NewtonSolver(model, joint_damping=0.0, joint_drive_armature=0.0)
    solver.config.affine_body_rigidity_model = s3d.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
    solver.config.joint_constraint_mode = s3d.JointConstraintMode.AUGMENTED_LAGRANGIAN
    solver.config.nonlinear_solver = s3d.NonlinearSolver.NEWTON
    solver.config.nonlinear_iterations = 10
    solver.config.linear_iterations = 60
    states = [model.state(), model.state()]
    for state in states:
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        solver.sync_affine_body_state(state)
    return model, solver, states, model.control()


@pytest.mark.parametrize("limits", [(0.2, 0.2), (0.15, 0.3)], ids=["shared", "distinct"])
def test_angular_stops_bound_external_torque_in_replicated_graph(limits):
    model, solver, states, control = _replicated_scene(limits)
    assert solver.synreal_model.world_count == (2 if limits[0] == limits[1] else 1)
    control.joint_f.fill_(0.5)
    solver.step(states[0], states[1], control, None, 0.005)
    for state in states:
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        solver.sync_affine_body_state(state)
    with wp.ScopedCapture(device=model.device) as capture:
        solver.step(states[0], states[1], control, None, 0.005)
        solver.step(states[1], states[0], control, None, 0.005)
    positions = []
    for _ in range(100):
        wp.capture_launch(capture.graph)
        pose = states[0].body_q.numpy()
        positions.append(2.0 * np.arctan2(pose[:, 5], pose[:, 6]))
    positions = np.asarray(positions)
    assert np.all(np.max(positions, axis=0) < np.asarray(limits) + 0.03)
    np.testing.assert_allclose(positions[-1], np.asarray(limits) + 0.0005, atol=0.003)
    assert np.isfinite(states[0].body_qd.numpy()).all()


def test_partial_reset_preserves_other_world_state_and_continuous_angle():
    reset_scene = _replicated_scene([100.0, 100.0], initial_q=3.0, initial_qd=3.0)
    reference_scene = _replicated_scene([100.0, 100.0], initial_q=3.0, initial_qd=3.0)
    for model, solver, states, control in (reset_scene, reference_scene):
        solver.enable_joint_drive_feedback()
        for _ in range(60):
            solver.step(states[0], states[1], control, None, 0.005)
            states.reverse()
        # The surviving hinge crossed pi. Losing its native angle history now
        # would create a spurious 2*pi position error on the following step.
        model.joint_target_ke.fill_(1.0)
        model.joint_target_kd.fill_(0.05)
        control.joint_target_q.fill_(3.9)
    model, solver, states, control = reset_scene
    body_mask = wp.array([1, 0], dtype=wp.int32, device=model.device)
    selected = wp.array([0], dtype=wp.int32, device=model.device)
    for state in states:
        before = {
            name: getattr(state, name).numpy().copy() for name in ("body_q", "body_qd", "particle_q", "particle_qd")
        }
        q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
        q[0], qd[0] = 0.0, 0.0
        state.joint_q.assign(q)
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state, indices=selected)
        # The public masked sync must also remain valid under graph replay.
        with wp.ScopedCapture(device=model.device) as capture:
            solver.sync_affine_body_state(state, body_mask=body_mask)
        wp.capture_launch(capture.graph)
        for name, expected in before.items():
            start = 1 if name.startswith("body") else 4
            np.testing.assert_array_equal(getattr(state, name).numpy()[start:], expected[start:])
    for _, current_solver, current_states, current_control in (reset_scene, reference_scene):
        current_solver.step(current_states[0], current_states[1], current_control, None, 0.005)
    for name in ("body_q", "body_qd", "particle_q", "particle_qd"):
        start = 1 if name.startswith("body") else 4
        actual = getattr(states[1], name).numpy()[start:]
        expected = getattr(reference_scene[2][1], name).numpy()[start:]
        np.testing.assert_allclose(actual, expected, atol=3.0e-5, rtol=1.0e-4)
    np.testing.assert_allclose(
        solver.joint_drive_effort.numpy()[1:],
        reference_scene[1].joint_drive_effort.numpy()[1:],
        atol=1.0e-4,
        rtol=1.0e-4,
    )

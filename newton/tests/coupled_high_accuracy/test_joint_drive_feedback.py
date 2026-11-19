"""Focused tests for the optional motor-effort telemetry output."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton._src.solvers.coupled_high_accuracy.solver_coupled_high_accuracy import NewtonSolver


@wp.kernel
def _copy_feedback_sample(
    feedback: wp.array[wp.float32],
    sample_index: int,
    samples: wp.array[wp.float32],
):
    samples[sample_index] = feedback[0]


def _make_revolute_solver(
    *,
    nonlinear_iterations: int = 1,
    target_q: float = 0.0,
    target_ke: float = 20.0,
    target_kd: float = 0.0,
    effort_limit: float = 100.0,
) -> tuple[object, NewtonSolver, object, object, object]:
    """Build an isolated one-DOF drive scene for telemetry tests."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
    inertia = wp.mat33(0.02, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.04)
    parent = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
    child = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
    joint = builder.add_joint_revolute(
        parent=parent,
        child=child,
        axis=(0.0, 0.0, 1.0),
        target_q=target_q,
        target_ke=target_ke,
        target_kd=target_kd,
        damping=0.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    builder.add_articulation([joint])
    builder.joint_effort_limit[0] = effort_limit
    model = builder.finalize(device="cuda:0")
    solver = NewtonSolver(model, joint_damping=0.0, joint_drive_armature=0.0)
    solver.config.constraint_alpha = 0.0
    solver.config.nonlinear_iterations = nonlinear_iterations
    solver.config.linear_iterations = 20
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_out)
    solver.sync_affine_body_state(state_in)
    solver.sync_affine_body_state(state_out)
    control = model.control()
    wp.copy(control.joint_target_q, model.joint_target_q)
    wp.copy(control.joint_target_qd, model.joint_target_qd)
    state_in.clear_forces()
    return model, solver, state_in, state_out, control


@pytest.mark.skipif(not wp.is_cuda_available(), reason="NewtonSolver requires CUDA")
@pytest.mark.parametrize(
    ("joint_kind", "expected_effort"),
    [("revolute", 2.0), ("revolute", -2.0), ("prismatic", 10.0), ("prismatic", -10.0)],
)
def test_joint_drive_feedback_preserves_external_generalized_force(joint_kind: str, expected_effort: float) -> None:
    """Both drive types preserve external force even when it exceeds the actuator limit."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
    inertia = wp.mat33(0.03, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.03)
    body = builder.add_link(mass=2.0, inertia=inertia, lock_inertia=True)
    if joint_kind == "revolute":
        parent = builder.add_link(mass=2.0, inertia=inertia, lock_inertia=True)
        joint = builder.add_joint_revolute(
            parent=parent,
            child=body,
            axis=(0.0, 0.0, 1.0),
            damping=0.0,
        )
    else:
        joint = builder.add_joint_prismatic(
            parent=-1,
            child=body,
            axis=(1.0, 0.0, 0.0),
            target_q=np.copysign(0.1, expected_effort),
            target_ke=0.0,
            target_kd=0.0,
            damping=0.0,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
    builder.add_articulation([joint])
    # MuJoCo's actuator limit excludes externally applied generalized force.
    builder.joint_effort_limit[0] = 0.5
    model = builder.finalize(device="cuda:0")
    solver = NewtonSolver(model, joint_damping=0.0, joint_drive_armature=0.0)
    solver.config.constraint_alpha = 0.0
    solver.config.nonlinear_iterations = 1
    solver.config.linear_iterations = 20

    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_out)
    solver.sync_affine_body_state(state_in)
    solver.sync_affine_body_state(state_out)
    control = model.control()
    control.joint_f.fill_(expected_effort)
    wp.copy(control.joint_target_q, model.joint_target_q)
    wp.copy(control.joint_target_qd, model.joint_target_qd)
    state_in.clear_forces()

    assert solver.joint_drive_effort is None
    effort = solver.enable_joint_drive_feedback()
    solver.step(state_in, state_out, control, None, 1.0e-4)
    wp.synchronize_device(model.device)

    assert effort.ptr == solver.joint_drive_effort.ptr
    assert np.allclose(effort.numpy(), [expected_effort], atol=1.0e-6)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="NewtonSolver requires CUDA")
def test_joint_drive_feedback_does_not_change_trajectory() -> None:
    """Telemetry preserves the trajectory of this deterministic drive scene."""
    model_a, solver_a, state_a, out_a, control_a = _make_revolute_solver(nonlinear_iterations=4)
    _, solver_b, state_b, out_b, control_b = _make_revolute_solver(nonlinear_iterations=4)
    control_a.joint_f.fill_(0.75)
    control_b.joint_f.fill_(0.75)
    solver_b.enable_joint_drive_feedback()
    for step in range(4):
        if step == 2:
            solver_b.disable_joint_drive_feedback()
            assert solver_b.joint_drive_effort is None
        solver_a.step(state_a, out_a, control_a, None, 1.0e-4)
        solver_b.step(state_b, out_b, control_b, None, 1.0e-4)
        wp.synchronize_device(model_a.device)
        for field in ("particle_q", "particle_qd", "body_q", "body_qd"):
            assert np.array_equal(getattr(out_a, field).numpy(), getattr(out_b, field).numpy())
        state_a, out_a = out_a, state_a
        state_b, out_b = out_b, state_b


@pytest.mark.skipif(not wp.is_cuda_available(), reason="NewtonSolver requires CUDA")
def test_joint_drive_feedback_tracks_commands_during_cuda_graph_replay() -> None:
    """A graph retains both substeps, then reads updated command arrays on replay."""
    model, solver, state_in, state_out, control = _make_revolute_solver(target_ke=0.0, effort_limit=0.5)
    feedback = solver.enable_joint_drive_feedback()
    samples = wp.empty(2, dtype=wp.float32, device=model.device)
    commands = [wp.array([value], dtype=wp.float32, device=model.device) for value in (0.75, -1.0)]
    stream = wp.get_stream(model.device)
    with wp.ScopedCapture(stream=stream) as capture:
        for substep in range(2):
            wp.copy(control.joint_f, commands[substep], stream=stream)
            solver.step(state_in, state_out, control, None, 1.0e-4)
            wp.launch(
                _copy_feedback_sample, dim=1, inputs=[feedback, substep, samples], device=model.device, stream=stream
            )
            state_in, state_out = state_out, state_in
    graph = capture.graph
    wp.capture_launch(graph, stream=stream)
    assert np.allclose(samples.numpy(), [0.75, -1.0], atol=1.0e-6)
    for array, value in zip(commands, (-0.25, 0.125), strict=True):
        array.fill_(value)
    wp.capture_launch(graph, stream=stream)
    assert np.allclose(samples.numpy(), [-0.25, 0.125], atol=1.0e-6)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="NewtonSolver requires CUDA")
@pytest.mark.parametrize("iterations", (1, 4))
def test_joint_drive_feedback_samples_pre_step_pose_and_velocity(iterations: int) -> None:
    """Automatic feedback matches MuJoCo force timing; direct queries evaluate their supplied state."""
    model, solver, state_in, state_out, control = _make_revolute_solver(
        nonlinear_iterations=iterations, target_q=0.8, target_kd=0.2
    )
    control.joint_f.fill_(0.3)
    control.joint_target_qd.fill_(0.1)
    effort = solver.enable_joint_drive_feedback()
    solver.step(state_in, state_out, control, None, 0.02)
    expected_pre_step = 0.3 + 20.0 * 0.8
    assert np.isclose(effort.numpy()[0], expected_pre_step, atol=3.0e-5)
    poses = state_out.body_q.numpy()
    angles = 2.0 * np.arctan2(poses[:, 5], poses[:, 6])
    position = angles[1] - angles[0]
    velocity = state_out.body_qd.numpy()[1, 5] - state_out.body_qd.numpy()[0, 5]
    # POSITION includes damping against zero speed; its target_qd is ignored.
    expected_current = 0.3 + 20.0 * (0.8 - position) - 0.2 * velocity
    assert abs(expected_current - expected_pre_step) > 1.0e-3
    assert not hasattr(state_out, "joint_drive_effort")

    fields = ("particle_q", "particle_qd", "body_q", "body_qd")
    snapshots = {name: getattr(state_out, name).numpy().copy() for name in fields}
    state_keys = set(vars(state_out))
    direct = solver.compute_joint_drive_effort(state_out, control)
    assert np.isclose(direct.numpy()[0], expected_current, atol=3.0e-5)
    assert np.isclose(effort.numpy()[0], expected_pre_step, atol=3.0e-5)
    assert set(vars(state_out)) == state_keys
    for name in fields:
        assert np.array_equal(getattr(state_out, name).numpy(), snapshots[name])
    with pytest.raises(ValueError, match="alias"):
        solver.compute_joint_drive_effort(state_out, control, out=control.joint_f)
    for name in ("joint_q", "joint_qd"):
        array = getattr(state_out, name, None)
        if array is not None:
            before = array.numpy().copy()
            with pytest.raises(ValueError, match="alias"):
                solver.compute_joint_drive_effort(state_out, control, out=array)
            assert np.array_equal(array.numpy(), before)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="NewtonSolver requires CUDA")
def test_joint_drive_feedback_clears_previous_samples_even_without_nonlinear_iterations() -> None:
    model, solver, state_in, state_out, control = _make_revolute_solver(target_ke=0.0)
    feedback = solver.enable_joint_drive_feedback()
    # Pre-step actuator evaluation is independent of the number of internal force evaluations.
    for command, iterations, expected in ((1.0, 1, 1.0), (0.0, 1, 0.0), (1.0, 1, 1.0), (1.0, 0, 1.0)):
        control.joint_f.fill_(command)
        solver.config.nonlinear_iterations = iterations
        solver.step(state_in, state_out, control, None, 1.0e-4)
        assert np.allclose(feedback.numpy(), expected, atol=1.0e-6)
        state_in, state_out = state_out, state_in


@pytest.mark.skipif(not wp.is_cuda_available(), reason="NewtonSolver requires CUDA")
def test_joint_drive_feedback_uses_twist_when_angle_crosses_pi() -> None:
    model, solver, state, _, control = _make_revolute_solver(target_q=-np.pi + 0.02, target_ke=0.0, target_kd=2.0)
    current_poses = state.body_q.numpy()
    previous_poses = current_poses.copy()
    previous_angle = np.pi - 0.03
    current_angle = -np.pi + 0.02
    previous_poses[1, 3:] = [0.0, 0.0, np.sin(previous_angle / 2.0), np.cos(previous_angle / 2.0)]
    current_poses[1, 3:] = [0.0, 0.0, np.sin(current_angle / 2.0), np.cos(current_angle / 2.0)]
    state.body_q.assign(current_poses)
    previous_state = model.state()
    previous_state.body_q.assign(previous_poses)
    state.body_qd.zero_()
    velocity = state.body_qd.numpy()
    velocity[1, 5] = 0.7
    state.body_qd.assign(velocity)
    previous_state.body_qd.zero_()
    control.joint_target_qd.zero_()
    # The supplied twist is authoritative even when the prior pose would give
    # a different secant speed across the quaternion branch cut.
    result = solver.compute_joint_drive_effort(state, control, previous_state=previous_state, dt=0.1).numpy()[0]
    assert np.isclose(result, -1.4, atol=1.0e-5)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="NewtonSolver requires CUDA")
def test_joint_drive_feedback_preserves_angle_branch_across_steps() -> None:
    model, solver, state0, state1, control = _make_revolute_solver(target_q=3.2, target_ke=1.0, target_kd=0.0)
    state2 = model.state()
    poses = state0.body_q.numpy()
    for state, angle in ((state0, 3.13), (state1, -3.13), (state2, -3.08)):
        sample = poses.copy()
        sample[1, 3:] = [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]
        state.body_q.assign(sample)
        state.body_qd.zero_()

    first = solver.compute_joint_drive_effort(
        state1, control, previous_state=state0, dt=0.1, _advance_angle_state=True
    ).numpy()[0]
    second = solver.compute_joint_drive_effort(
        state2, control, previous_state=state1, dt=0.1, _advance_angle_state=True
    ).numpy()[0]

    first_angle = -3.13 + 2.0 * np.pi
    second_angle = -3.08 + 2.0 * np.pi
    assert np.isclose(first, 3.2 - first_angle, atol=1.0e-5)
    assert np.isclose(second, 3.2 - second_angle, atol=1.0e-5)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="NewtonSolver requires CUDA")
def test_joint_drive_feedback_keeps_angle_branch_when_reenabled() -> None:
    """Disabling telemetry must not make a later sample lose accumulated turns."""
    model, solver, state0, state1, control = _make_revolute_solver(target_q=3.2, target_ke=1.0, target_kd=0.0)
    state2 = model.state()
    poses = state0.body_q.numpy()
    for state, angle in ((state0, 3.13), (state1, -3.13), (state2, -3.08)):
        sample = poses.copy()
        sample[1, 3:] = [0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)]
        state.body_q.assign(sample)
        state.body_qd.zero_()

    first = solver.compute_joint_drive_effort(
        state1, control, previous_state=state0, dt=0.1, _advance_angle_state=True
    ).numpy()[0]
    solver.disable_joint_drive_feedback()
    solver.enable_joint_drive_feedback()
    second = solver.compute_joint_drive_effort(
        state2, control, previous_state=state1, dt=0.1, _advance_angle_state=True
    ).numpy()[0]

    assert np.isclose(first, 3.2 - (-3.13 + 2.0 * np.pi), atol=1.0e-5)
    assert np.isclose(second, 3.2 - (-3.08 + 2.0 * np.pi), atol=1.0e-5)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="NewtonSolver requires CUDA")
def test_joint_drive_effort_uses_instantaneous_moving_frame_velocity() -> None:
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    inertia = wp.mat33(np.eye(3, dtype=np.float32))
    parent = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
    child = builder.add_link(mass=1.0, com=(0.0, 1.0, 0.0), inertia=inertia, lock_inertia=True)
    joint = builder.add_joint_prismatic(
        parent=parent,
        child=child,
        axis=(1.0, 0.0, 0.0),
        target_q=0.5,
        target_ke=20.0,
        target_kd=2.0,
        actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
    )
    builder.add_articulation([joint])
    model = builder.finalize(device="cuda:0")
    solver = NewtonSolver(model)
    state, previous_state, control = model.state(), model.state(), model.control()
    poses = state.body_q.numpy()
    poses[child, :3] = [0.2, 0.1, 0.0]
    state.body_q.assign(poses)
    previous_poses = poses.copy()
    previous_poses[child, :3] = [-0.1, 0.1, 0.0]
    previous_state.body_q.assign(previous_poses)
    # COM offsets and the moving parent axis both contribute to the
    # instantaneous coordinate speed; previous poses do not override it.
    state.body_qd.assign(np.array([[1, 0, 0, 0, 0, 1], [3, 0, 0, 0, 0, 2]], dtype=np.float32))
    previous_state.body_qd.assign(np.array([[-7, 0, 0, 0, 0, -4], [11, 0, 0, 0, 0, 8]], dtype=np.float32))
    control.joint_target_q.fill_(0.5)
    control.joint_target_qd.fill_(0.2)
    control.joint_f.fill_(0.7)
    # Child origin speed is 5, parent origin speed is 1, and the rotating
    # parent axis adds 0.1, giving qdot = 4.1 m/s.
    expected = 0.7 + 20.0 * (0.5 - 0.2) + 2.0 * (0.2 - 4.1)
    result = solver.compute_joint_drive_effort(state, control, previous_state=previous_state, dt=0.1).numpy()[0]
    assert np.isclose(result, expected, atol=2.0e-5)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="NewtonSolver requires CUDA")
def test_joint_drive_effort_respects_replicated_world_control_indices() -> None:
    scene = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3, dtype=np.float32)), lock_inertia=True)
    joint = scene.add_joint_revolute(parent=-1, child=body, axis=(0.0, 0.0, 1.0))
    scene.add_articulation([joint])
    scene.joint_effort_limit[0] = 0.5
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.replicate(scene, 3)
    model = builder.finalize(device="cuda:0")
    solver = NewtonSolver(model)
    assert solver.synreal_model.world_count == 3
    state, control = model.state(), model.control()
    control.joint_f.assign(np.array([0.3, -0.4, 2.0], dtype=np.float32))
    assert np.allclose(solver.compute_joint_drive_effort(state, control).numpy(), [0.3, -0.4, 2.0], atol=1.0e-6)

"""Physical regressions for scalar joint inertia and MuJoCo actuator semantics."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")
pytest.importorskip("mujoco")

from newton.solvers import coupled_high_accuracy as s3d

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="NewtonSolver requires CUDA")


def _make_scene(
    kind,
    backend,
    *,
    armature=0.0,
    floating_parent=False,
    common_velocity=0.0,
    initial_position=0.0,
    initial_velocity=0.0,
    mode=None,
    target=0.0,
    target_velocity=0.0,
    kp=0.0,
    kd=0.0,
    damping=0.0,
    effort_limit=1.0e6,
    torque=0.0,
    guide_half_length=10.0,
    lower_limit=None,
    upper_limit=None,
    limit_ke=None,
    limit_kd=None,
):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    inertia = wp.mat33(0.02, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.04)
    parent = -1
    joints = []
    if floating_parent:
        parent = builder.add_link(mass=2.0, inertia=inertia, lock_inertia=True)
        joints.append(builder.add_joint_free(child=parent))
        builder.joint_qd[5 if kind == "revolute" else 0] = common_velocity
    child = builder.add_link(mass=2.0, inertia=inertia, lock_inertia=True)
    add_joint = getattr(builder, f"add_joint_{kind}")
    joint = add_joint(
        parent=parent,
        child=child,
        axis=(0.0, 0.0, 1.0) if kind == "revolute" else (1.0, 0.0, 0.0),
        target_q=target,
        target_ke=kp,
        target_kd=kd,
        damping=damping,
        armature=armature,
        limit_lower=lower_limit,
        limit_upper=upper_limit,
        limit_ke=limit_ke,
        limit_kd=limit_kd,
        actuator_mode=newton.JointTargetMode.NONE if mode is None else mode,
    )
    joints.append(joint)
    builder.add_articulation(joints)
    builder.joint_q[-1] = initial_position
    builder.joint_qd[-1] = initial_velocity
    builder.joint_effort_limit[-1] = effort_limit
    model = builder.finalize(device="cuda:0")
    np.testing.assert_allclose(model.joint_armature.numpy()[-1], armature)
    if backend == "synreal":
        solver = s3d.NewtonSolver(
            model, joint_damping=0.0, joint_drive_armature=0.0, joint_line_half_length=guide_half_length
        )
        solver.config.affine_body_rigidity_model = s3d.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
        solver.config.joint_constraint_mode = s3d.JointConstraintMode.AUGMENTED_LAGRANGIAN
        solver.config.nonlinear_solver = s3d.NonlinearSolver.NEWTON
        solver.config.nonlinear_iterations = 10
        solver.config.linear_iterations = 60
        solver.config.joint_constraint_primal_iterations = 5
    else:
        solver = newton.solvers.SolverMuJoCo(
            model, integrator="implicitfast", solver="newton", use_mujoco_cpu=True, nconmax=8, njmax=16
        )
    state_in, state_out = model.state(), model.state()
    for state in (state_in, state_out):
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        if backend == "synreal":
            solver.sync_affine_body_state(state)
    control = model.control()
    control.joint_target_q.fill_(target)
    control.joint_target_qd.fill_(target_velocity)
    force = np.zeros(model.joint_dof_count, dtype=np.float32)
    force[-1] = torque
    control.joint_f.assign(force)
    return model, solver, state_in, state_out, control, parent, child


def _advance(scene, kind, dt=0.00125, body_wrench=0.0):
    model, solver, state_in, state_out, control, parent, child = scene
    state_in.clear_forces()
    if body_wrench:
        force = state_in.body_f.numpy()
        force[child, 5 if kind == "revolute" else 0] = body_wrench
        state_in.body_f.assign(force)
    solver.step(state_in, state_out, control, None, dt)
    wp.synchronize_device(model.device)
    # The common joint axis stays fixed in these COM-centered scenes. Reading
    # body twists avoids quaternion branch choices in inverse kinematics.
    component = 5 if kind == "revolute" else 0
    velocity = state_out.body_qd.numpy()[:, component]
    relative_velocity = velocity[child] - (velocity[parent] if parent >= 0 else 0.0)
    return float(relative_velocity), velocity


@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
@pytest.mark.parametrize("floating_parent", [False, True], ids=["world-parent", "dynamic-parent"])
def test_joint_armature_matches_mujoco_relative_inertia(kind, floating_parent):
    armature = 0.1 if kind == "revolute" else 2.0
    options = dict(armature=armature, floating_parent=floating_parent, torque=0.5)
    synreal = _make_scene(kind, "synreal", **options)
    mujoco = _make_scene(kind, "mujoco", **options)
    actual, body_velocity = _advance(synreal, kind)
    expected, _ = _advance(mujoco, kind)
    body_inertia = 0.04 if kind == "revolute" else 2.0
    reduced_inertia = body_inertia / 2.0 if floating_parent else body_inertia
    analytic_velocity = 0.5 * 0.00125 / (reduced_inertia + armature)
    np.testing.assert_allclose(expected, analytic_velocity, rtol=1.0e-5, atol=1.0e-8)
    np.testing.assert_allclose(actual, expected, rtol=0.02, atol=2.0e-5)
    if floating_parent:
        # Internal motor torque and relative-coordinate inertia must not create
        # net linear/angular momentum of an otherwise isolated equal-mass pair.
        np.testing.assert_allclose(body_velocity.sum(), 0.0, atol=2.0e-5)
    np.testing.assert_allclose(synreal[0].body_inertia.numpy()[:, 2, 2], 0.04)


@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
def test_joint_armature_does_not_slow_common_body_motion(kind):
    options = dict(floating_parent=True, common_velocity=0.7)
    unmodified = _make_scene(kind, "synreal", **options)
    armature = _make_scene(kind, "synreal", armature=2.0, **options)
    _, expected = _advance(unmodified, kind)
    relative, actual = _advance(armature, kind)
    np.testing.assert_allclose(expected, 0.7, rtol=0.003, atol=2.0e-5)
    np.testing.assert_allclose(actual, expected, rtol=0.003, atol=2.0e-5)
    assert abs(relative) < 2.0e-5


@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
def test_joint_armature_also_resists_external_body_wrench(kind):
    # Body wrenches enter the rigid predictor before the constraint solve.
    # Armature must resist their acceleration as well as joint_f acceleration.
    scene = _make_scene(kind, "synreal", armature=0.1)
    scene[1].config.inertia_mode = s3d.InertiaMode.MIDPOINT_CAYLEY
    actual, _ = _advance(scene, kind, body_wrench=0.5)
    body_inertia = 0.04 if kind == "revolute" else 2.0
    expected = 0.5 * 0.00125 / (body_inertia + 0.1)
    np.testing.assert_allclose(actual, expected, rtol=0.02, atol=2.0e-5)


@pytest.mark.parametrize("armatures", [(0.1, 0.1, 0.1), (0.0, 0.1, 0.3)], ids=["shared", "distinct"])
def test_joint_armature_handles_replicated_worlds_during_cuda_graph_replay(armatures):
    scene = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
    body = scene.add_link(mass=2.0, inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.04), lock_inertia=True)
    joint = scene.add_joint_revolute(parent=-1, child=body, axis=(0.0, 0.0, 1.0), armature=0.0)
    scene.add_articulation([joint])
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
    builder.replicate(scene, 3)
    armatures = np.asarray(armatures, dtype=np.float32)
    builder.joint_armature[:] = armatures.tolist()
    model = builder.finalize(device="cuda:0")
    solver = s3d.NewtonSolver(model, joint_damping=0.0, joint_drive_armature=0.0)
    assert solver.synreal_model.world_count == (3 if len(np.unique(armatures)) == 1 else 1)
    solver.config.affine_body_rigidity_model = s3d.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
    solver.config.joint_constraint_mode = s3d.JointConstraintMode.AUGMENTED_LAGRANGIAN
    solver.config.nonlinear_iterations = 10
    solver.config.linear_iterations = 60
    solver.config.joint_constraint_primal_iterations = 5
    state, output = model.state(), model.state()
    for current in (state, output):
        newton.eval_fk(model, model.joint_q, model.joint_qd, current)
        solver.sync_affine_body_state(current)
    initial = model.state()
    initial.assign(state)
    control = model.control()
    control.joint_f.fill_(0.5)
    dt = 0.00125
    solver.step(state, output, control, None, dt)
    for current in (state, output):
        current.assign(initial)
        solver.sync_affine_body_state(current)
    with wp.ScopedCapture(device=model.device) as capture:
        for _ in range(2):
            solver.step(state, output, control, None, dt)
            state, output = output, state
    for _ in range(3):
        wp.capture_launch(capture.graph)
    actual = state.body_qd.numpy()[:, 5]
    expected = 6 * 0.5 * dt / (0.04 + armatures)
    np.testing.assert_allclose(actual, expected, rtol=0.025, atol=3.0e-5)


@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
@pytest.mark.parametrize(
    ("mode", "kp", "target_velocity", "limit", "expected_actuator"),
    [
        (newton.JointTargetMode.NONE, 3.0, 0.3, 0.5, 0.0),
        (newton.JointTargetMode.EFFORT, 3.0, 0.3, 0.5, 0.0),
        (newton.JointTargetMode.POSITION, 3.0, 50.0, 10.0, -0.8),
        (newton.JointTargetMode.POSITION, 2000.0, 50.0, 1.0e6, 398.6),
        (newton.JointTargetMode.VELOCITY, 30.0, 0.3, 10.0, -0.8),
        (newton.JointTargetMode.POSITION_VELOCITY, 3.0, 0.3, 0.5, -0.2),
        (newton.JointTargetMode.POSITION, 3.0, 50.0, 0.5, -0.5),
        (newton.JointTargetMode.POSITION_VELOCITY, 30.0, 0.3, 0.5, 0.5),
    ],
    ids=[
        "none",
        "effort",
        "position",
        "stiff-position",
        "velocity",
        "combined-cancellation",
        "position-clipped",
        "combined-clipped",
    ],
)
def test_joint_drive_modes_and_aggregate_limit_match_mujoco(kind, mode, kp, target_velocity, limit, expected_actuator):
    # A nonzero external force distinguishes clipping actuator force from
    # clipping the sum of actuator and externally applied generalized force.
    options = dict(
        mode=mode,
        target=0.2,
        target_velocity=target_velocity,
        initial_velocity=0.7,
        kp=kp,
        kd=2.0,
        effort_limit=limit,
        torque=0.8,
    )
    synreal = _make_scene(kind, "synreal", **options)
    mujoco = _make_scene(kind, "mujoco", **options)
    feedback = synreal[1].enable_joint_drive_feedback()
    actual, _ = _advance(synreal, kind)
    expected, _ = _advance(mujoco, kind)
    np.testing.assert_allclose(mujoco[1].mj_data.qfrc_actuator[-1], expected_actuator, rtol=1.0e-5, atol=1.0e-6)
    np.testing.assert_allclose(feedback.numpy()[-1], 0.8 + expected_actuator, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(actual - 0.7, expected - 0.7, rtol=0.025, atol=5.0e-5)


@pytest.mark.parametrize("kind", ["revolute", "prismatic"])
def test_passive_joint_damping_with_armature_matches_mujoco(kind):
    options = dict(armature=0.1, initial_velocity=0.7, damping=3.0)
    actual, _ = _advance(_make_scene(kind, "synreal", **options), kind)
    expected, _ = _advance(_make_scene(kind, "mujoco", **options), kind)
    np.testing.assert_allclose(actual - 0.7, expected - 0.7, rtol=0.025, atol=5.0e-5)


@pytest.mark.parametrize("force", [-1.0, 1.0], ids=["inward", "outward"])
def test_prismatic_finite_guide_preserves_endpoint_constraint(force):
    scene = _make_scene("prismatic", "synreal", initial_position=0.1, guide_half_length=0.1, torque=force)
    velocity, _ = _advance(scene, "prismatic", dt=0.01)
    position = scene[3].body_q.numpy()[scene[-1], 0]
    if force < 0:
        # An interior tangent must release axial motion as soon as the slider
        # is pushed away from the endpoint.
        np.testing.assert_allclose(velocity, -0.005, rtol=0.02, atol=2.0e-5)
        np.testing.assert_allclose(position, 0.09995, atol=1.0e-6)
    else:
        assert position <= 0.100001
        assert abs(velocity) < 1.0e-4

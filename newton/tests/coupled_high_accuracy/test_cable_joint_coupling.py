"""Exercise a driven sliding carriage, ball anchors and rod constraints in one native solve."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton._src.solvers.coupled_high_accuracy.cable_material import register_cable_attributes
from newton.solvers import coupled_high_accuracy as cha

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Coupled cables require CUDA")


@pytest.mark.parametrize("capture", (False, True))
@pytest.mark.parametrize("body_block", (False, True))
def test_driven_slider_and_ball_anchored_cable(capture, body_block):
    builder = cha.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    register_cable_attributes(builder)
    carriage = builder.add_link(mass=0.2, inertia=wp.mat33(np.eye(3) * 0.001), lock_inertia=True)
    slider = builder.add_joint_prismatic(
        parent=-1,
        child=carriage,
        axis=(1.0, 0.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.25, 0.0, 0.5), wp.quat_identity()),
        target_q=0.01,
        target_ke=1000.0,
        target_kd=2.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    points = [wp.vec3(0.25, 0.0, 0.5 + 0.05 * i) for i in range(3)]
    frames = newton.Rod(points).quaternions
    bodies, rods = builder.add_rod(
        rod=newton.Rod(points, quaternions=frames, radius=0.005),
        stretch_stiffness=2000.0,
        shear_stiffness=1000.0,
        bend_stiffness=0.5,
        twist_stiffness=0.25,
        wrap_in_articulation=False,
    )
    anchor = builder.add_joint_ball(
        parent=carriage,
        child=bodies[0],
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.025), wp.quat_identity()),
    )
    builder.add_joint_ball(
        parent=-1,
        child=bodies[-1],
        parent_xform=wp.transform(points[-1], wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.025), wp.quat_identity()),
    )
    builder.add_articulation([slider, *rods, anchor])
    model = builder.finalize(device="cuda:0")
    solver = cha.NewtonSolver(model)
    solver.config.affine_body_block_preconditioner = body_block
    solver.config.nonlinear_iterations = 4
    solver.config.linear_iterations = 12
    state, other, control = model.state(), model.state(), model.control()
    wp.copy(control.joint_target_q, model.joint_target_q)
    wp.copy(control.joint_target_qd, model.joint_target_qd)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    solver.sync_affine_body_state(state)
    initial = state.body_q.numpy().copy()
    pipeline = newton.CollisionPipeline(model, soft_contact_max=0)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)

    def two_substeps():
        for current, following in ((state, other), (other, state)):
            current.clear_forces()
            pipeline.collide(current, contacts)
            solver.step(current, following, control, contacts, 0.002)

    if capture:
        with wp.ScopedCapture(device="cuda:0") as recorded:
            two_substeps()
        for _ in range(3):
            wp.capture_launch(recorded.graph)
    else:
        for _ in range(3):
            two_substeps()
    wp.synchronize_device(model.device)
    for field in ("body_q", "body_qd", "particle_q", "particle_qd"):
        assert np.isfinite(getattr(state, field).numpy()).all(), field
    poses = state.body_q.numpy()
    assert poses[carriage, 0] > initial[carriage, 0] + 1.0e-6
    endpoint = np.array(wp.transform_point(wp.transform(*poses[bodies[-1]]), wp.vec3(0.0, 0.0, 0.025)))
    assert np.linalg.norm(endpoint - np.asarray(points[-1])) < 0.02

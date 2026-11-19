"""Keep a gravity-loaded ball chain connected while hard contact and body-block solves interact."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")
from newton.solvers import coupled_high_accuracy as s3d

pytestmark = [
    pytest.mark.skipif(not wp.is_cuda_available(), reason="Ball-chain body blocks require CUDA"),
]


@pytest.mark.parametrize("capture", (False, True))
@pytest.mark.parametrize("contact_block", (False, True))
def test_ball_chain_body_blocks_with_ground_and_external_load(capture, contact_block):
    builder = s3d.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    shape = newton.ModelBuilder.ShapeConfig(ke=8.0e4, kd=20.0, mu=0.0)
    bodies, joints = [], []
    for i in range(3):
        body = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.12 + i * 0.12), wp.quat_identity()),
            mass=0.2,
            inertia=wp.mat33(np.eye(3) * 0.001),
            lock_inertia=True,
        )
        builder.add_shape_capsule(body=body, radius=0.025, half_height=0.035, cfg=shape)
        if i == 0:
            joints.append(builder.add_joint_free(child=body))
        else:
            joints.append(
                builder.add_joint_ball(
                    parent=bodies[-1],
                    child=body,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.06), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.06), wp.quat_identity()),
                )
            )
        bodies.append(body)
    builder.add_articulation(joints)
    builder.add_ground_plane(cfg=shape)
    model = builder.finalize(device="cuda:0")
    solver = s3d.NewtonSolver(model)
    solver.config.affine_body_block_preconditioner = True
    solver.config.affine_body_contact_block_preconditioner = contact_block
    solver.config.rigid_contact_hard = True
    solver.config.nonlinear_iterations = 2
    solver.config.linear_iterations = 10
    state, other, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    solver.sync_affine_body_state(state)
    initial = state.body_q.numpy().copy()
    pipeline = newton.CollisionPipeline(model, soft_contact_max=0)
    contacts = pipeline.contacts()
    solver.prepare_contacts(contacts)
    force = np.zeros_like(state.body_f.numpy())
    force[:, 0] = 0.2
    external_load = wp.array(force, dtype=wp.spatial_vector, device=model.device)

    def two_substeps():
        for current, following in ((state, other), (other, state)):
            current.clear_forces()
            wp.copy(current.body_f, external_load)
            pipeline.collide(current, contacts)
            solver.step(current, following, control, contacts, 1.0 / 480.0)

    if capture:
        with wp.ScopedCapture(device=model.device) as captured:
            two_substeps()
        for _ in range(60):
            wp.capture_launch(captured.graph)
    else:
        for _ in range(60):
            two_substeps()
    wp.synchronize_device(model.device)
    for field in ("body_q", "body_qd", "particle_q", "particle_qd"):
        assert np.isfinite(getattr(state, field).numpy()).all(), field
    poses = state.body_q.numpy()
    assert poses[bodies[0], 0] > initial[bodies[0], 0] + 1.0e-4
    assert poses[bodies[0], 2] > 0.04
    assert contacts.rigid_contact_count.numpy()[0] > 0
    for parent, child in zip(bodies, bodies[1:]):
        a = wp.transform_point(wp.transform(*poses[parent]), wp.vec3(0.0, 0.0, 0.06))
        b = wp.transform_point(wp.transform(*poses[child]), wp.vec3(0.0, 0.0, -0.06))
        assert np.linalg.norm(np.asarray(a) - np.asarray(b)) < 0.015

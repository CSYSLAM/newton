"""A grain contacting a driven hinged body must receive and return the contact impulse."""

from types import SimpleNamespace

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")
from newton.solvers import coupled_high_accuracy as s3d

pytestmark = [
    pytest.mark.skipif(not wp.is_cuda_available(), reason="Articulated granular materials require CUDA"),
]


def scene(contact):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(np.eye(3, dtype=np.float32) * 0.01)
    base = builder.add_link(mass=2.0, inertia=inertia, lock_inertia=True)
    builder.body_flags[base] = int(newton.BodyFlags.KINEMATIC)
    body = builder.add_link(mass=2.0, inertia=inertia, lock_inertia=True)
    root = builder.add_joint_fixed(parent=-1, child=base)
    joint = builder.add_joint_revolute(
        parent=base,
        child=body,
        axis=(0.0, 0.0, 1.0),
        target_ke=1.0e-6,
        target_kd=0.0,
        damping=0.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    builder.add_articulation([root, joint])
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    builder.add_shape_box(
        body=body, xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), hx=0.04, hy=0.1, hz=0.08, cfg=cfg
    )
    builder.add_particles(
        pos=np.array([[0.1, 0.135, 0.0], [0.1, 0.195, 0.0]], dtype=np.float32),
        vel=np.zeros((2, 3), dtype=np.float32),
        mass=np.ones(2),
        radius=np.full(2, 0.05),
    )
    model = builder.finalize(device="cuda:0")
    model.soft_contact_ke, model.soft_contact_kd = 1.0e4, 0.0
    model.soft_contact_kf, model.soft_contact_mu = 0.0, 0.0
    s3d.configure_granular(
        model,
        [0, 1],
        contact_stiffness=1.0e-6,
        cohesion=20.0,
        bond_radius=0.065,
        break_strain=1.0,
        friction=0.0,
        max_compaction=0.0,
    )
    solver = s3d.NewtonSolver(model)
    solver.config.affine_body_rigidity_model = s3d.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
    solver.config.joint_constraint_mode = s3d.JointConstraintMode.AUGMENTED_LAGRANGIAN
    solver.config.nonlinear_solver = s3d.NonlinearSolver.NEWTON
    solver.config.nonlinear_iterations, solver.config.linear_iterations = 10, 40
    solver.config.enable_collision_hessian_non_diagonal = True
    a, b, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, a)
    solver.sync_affine_body_state(a)
    b.assign(a)
    contacts = SimpleNamespace(
        rigid_contact_max=0,
        soft_contact_max=1,
        soft_contact_count=wp.array([int(contact)], dtype=wp.int32, device=model.device),
        soft_contact_particle=wp.array([0], dtype=wp.int32, device=model.device),
        soft_contact_shape=wp.array([0], dtype=wp.int32, device=model.device),
        soft_contact_body_pos=wp.array([wp.vec3(0.1, 0.1, 0.0)], dtype=wp.vec3, device=model.device),
        soft_contact_body_vel=wp.zeros(1, dtype=wp.vec3, device=model.device),
        soft_contact_normal=wp.array([wp.vec3(0.0, 1.0, 0.0)], dtype=wp.vec3, device=model.device),
    )
    solver.prepare_contacts(contacts)
    return model, solver, a, b, control, contacts, body


@pytest.mark.parametrize("capture", [False, True])
def test_grain_contact_reacts_on_driven_body_and_preserves_hinge(capture):
    results = []
    for contact in [False, True]:
        model, solver, a, b, control, contacts, body = scene(contact)
        initial = a.particle_q.numpy()
        base_pose = a.body_q.numpy()[0].copy()

        def advance():
            for current, following in [(a, b), (b, a)]:
                current.clear_forces()
                solver.step(current, following, control, contacts, 1.0 / 240.0)

        if capture:
            with wp.ScopedCapture(device=model.device) as captured:
                advance()
            wp.capture_launch(captured.graph)
        else:
            advance()
        wp.synchronize_device(model.device)
        assert np.isfinite(a.particle_q.numpy()).all()
        assert np.isfinite(a.particle_qd.numpy()).all()
        np.testing.assert_array_equal(a.body_q.numpy()[0], base_pose)
        assert np.linalg.norm(a.body_q.numpy()[body, :3]) < 0.005
        results.append((a.particle_q.numpy()[0, 1] - initial[0, 1], a.body_qd.numpy()[body, 5]))
    assert abs(results[0][0]) < 1.0e-6
    assert abs(results[0][1]) < 1.0e-5
    assert results[1][0] > 1.0e-5
    assert results[1][1] < -1.0e-4


@pytest.mark.parametrize("option", ["pd", "body_block", "hard_contact", "self_collision", "joint_penalty"])
def test_articulated_granular_runs_other_solver_configurations(option):
    _, solver, a, b, control, contacts, _ = scene(True)
    if option == "pd":
        solver.config.nonlinear_solver = s3d.NonlinearSolver.PD_NEWTON
    elif option == "body_block":
        solver.config.affine_body_block_preconditioner = True
    elif option == "hard_contact":
        solver.config.rigid_contact_hard = True
    elif option == "self_collision":
        solver.config.enable_self_collision = True
    else:
        solver.config.joint_constraint_mode = s3d.JointConstraintMode.PENALTY
    solver.step(a, b, control, contacts, 1.0 / 240.0)
    wp.synchronize_device("cuda:0")
    assert np.isfinite(b.particle_q.numpy()).all()
    assert np.isfinite(b.body_q.numpy()).all()

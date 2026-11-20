"""Finite-radius capsule friction must react to material-axis spin."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton.solvers import coupled_high_accuracy as cha

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Cable contact simulation requires CUDA")


@pytest.mark.parametrize("framework", [cha.NonlinearSolver.NEWTON, cha.NonlinearSolver.PD_NEWTON])
@pytest.mark.parametrize("friction_epsilon", [0.0, 0.01])
def test_capsule_material_spin_uses_surface_friction_anchor(framework, friction_epsilon):
    def simulate(friction, ground_friction=None):
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.kf = 1.0e4
        builder.default_shape_cfg.mu = friction
        # Local +Z is world -Y, so a Y-axis spin moves the bottom surface in X
        # while every centerline point has zero spin-induced velocity.
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.045), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi)),
            mass=1.0,
            inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1),
            lock_inertia=True,
        )
        builder.add_shape_capsule(body=body, radius=0.05, half_height=0.1)
        if ground_friction is not None:
            builder.default_shape_cfg.mu = ground_friction
        builder.add_ground_plane()
        model = builder.finalize(device="cuda:0")
        solver = cha.NewtonSolver(model)
        solver.config.nonlinear_solver = framework
        solver.config.nonlinear_iterations = 5
        solver.config.rigid_contact_friction_epsilon = friction_epsilon
        state_in, state_out = model.state(), model.state()
        velocities = state_in.body_qd.numpy()
        velocities[body, 4] = 10.0
        state_in.body_qd.assign(velocities)
        solver.sync_affine_body_state(state_in)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state_in, contacts)
        assert int(contacts.rigid_contact_count.numpy()[0]) > 0
        count = int(contacts.rigid_contact_count.numpy()[0])
        offsets = np.concatenate(
            (contacts.rigid_contact_offset0.numpy()[:count], contacts.rigid_contact_offset1.numpy()[:count])
        )
        assert np.max(np.linalg.norm(offsets, axis=-1)) > 0.04
        state_in.clear_forces()
        solver.step(state_in, state_out, model.control(), contacts, 1.0e-3)
        return state_out.body_qd.numpy()[body]

    free = simulate(0.0)
    resisted = simulate(1.0)
    assert abs(free[0]) < 1.0e-5
    assert resisted[0] > 1.0e-3
    assert resisted[4] < free[4] - 1.0e-4
    assert np.isfinite(resisted).all()
    if friction_epsilon > 0.0:
        # A frictionless pulley flange must remain frictionless against a
        # high-friction cable: VBD uses the geometric mean of the two materials.
        frictionless_surface = simulate(1.0, ground_friction=0.0)
        np.testing.assert_allclose(frictionless_surface, free, atol=1.0e-5)


@pytest.mark.parametrize("builder_type", [newton.ModelBuilder, cha.ModelBuilder])
@pytest.mark.parametrize("framework", [cha.NonlinearSolver.NEWTON, cha.NonlinearSolver.PD_NEWTON])
@pytest.mark.parametrize("hard_contact", [False, True])
def test_prescribed_kinematic_surface_transmits_traction(builder_type, framework, hard_contact):
    builder = builder_type(gravity=wp.vec3(0.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=1.0e4, kd=0.0, kf=1.0e4, mu=1.0)
    inertia = wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1)
    surface = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
        mass=1.0,
        inertia=inertia,
        lock_inertia=True,
    )
    # A positive authored mass must not make a prescribed body dynamic.
    builder.body_flags[surface] = int(newton.BodyFlags.KINEMATIC)
    builder.add_shape_box(body=surface, hx=1.0, hy=1.0, hz=0.1, cfg=cfg)
    sphere = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.09), wp.quat_identity()),
        mass=1.0,
        inertia=inertia,
        lock_inertia=True,
    )
    builder.add_shape_sphere(body=sphere, radius=0.1, cfg=cfg)
    model = builder.finalize(device="cuda:0")
    solver = cha.NewtonSolver(model)
    solver.config.nonlinear_solver = framework
    solver.config.nonlinear_iterations = 3
    solver.config.linear_iterations = 20
    solver.config.rigid_contact_hard = hard_contact
    solver.config.rigid_contact_friction_epsilon = 0.01
    body_ids = solver._affine_body_ids.numpy()
    affine = int(np.flatnonzero(body_ids == surface)[0])
    assert solver._fixed_affine_body_flags.numpy()[affine] == 1
    assert not np.any(solver._particle_flags.numpy()[solver._affine_body_indices.numpy()[affine]] & 1)

    state_in, state_out = model.state(), model.state()
    poses = state_in.body_q.numpy()
    poses[surface, 0] += 1.0e-3
    state_in.body_q.assign(poses)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    assert int(contacts.rigid_contact_count.numpy()[0]) > 0
    state_in.clear_forces()
    solver.step(state_in, state_out, model.control(), contacts, 1.0e-3)
    np.testing.assert_allclose(state_out.body_q.numpy()[surface], poses[surface], atol=1.0e-7)
    assert state_out.body_qd.numpy()[sphere, 0] > 1.0e-3

"""Cable material, prescribed-endpoint, and solver-framework regressions."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton._src.solvers.coupled_high_accuracy.cable_material import register_cable_attributes
from newton._src.solvers.coupled_high_accuracy.solver_coupled_high_accuracy import NewtonSolver
from newton.solvers import coupled_high_accuracy as s3d

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Cable simulation requires CUDA")


def _cable_model(builder_type=s3d.ModelBuilder, *, curved=False, dahl=False):
    builder = builder_type(gravity=wp.vec3(0.0))
    if dahl:
        register_cable_attributes(builder)
    points = [wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.1), wp.vec3(0.02 if curved else 0.0, 0.0, 0.2)]
    quaternions = newton.Rod(points).quaternions
    bodies, joints = builder.add_rod(
        rod=newton.Rod(points, quaternions=quaternions, radius=0.01),
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
    builder.color()
    model = builder.finalize(device="cuda:0")
    if dahl:
        model.vbd.dahl_eps_max.fill_(0.1)
        model.vbd.dahl_tau.fill_(0.01)
    return model, bodies, joints


@pytest.mark.parametrize("builder_type", [s3d.ModelBuilder, newton.ModelBuilder])
@pytest.mark.parametrize("framework", [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON])
def test_curved_cable_rest_shape_is_force_free(builder_type, framework):
    model, bodies, _ = _cable_model(builder_type, curved=True)
    solver = NewtonSolver(model)
    solver.config.nonlinear_solver = framework
    state_in, state_out = model.state(), model.state()
    initial = state_in.body_q.numpy()
    for _ in range(3):
        state_in.clear_forces()
        solver.step(state_in, state_out, model.control(), None, 1.0e-3)
        state_in, state_out = state_out, state_in
    np.testing.assert_allclose(state_in.body_q.numpy(), initial, atol=2.0e-5)
    assert solver._fixed_affine_body_ids == [0]
    assert np.isfinite(state_in.particle_q.numpy()).all()
    assert int(model.body_count) == len(bodies)


@pytest.mark.parametrize("framework", [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON])
def test_cable_prescribed_endpoint_updates_and_pull_response(framework):
    model, bodies, _ = _cable_model()
    solver = NewtonSolver(model)
    solver.config.nonlinear_solver = framework
    solver.config.nonlinear_iterations = 12
    state_in, state_out = model.state(), model.state()
    initial = state_in.body_q.numpy()
    prescribed = initial.copy()
    prescribed[bodies[0], 2] += 0.002
    state_in.body_q.assign(prescribed)
    for _ in range(5):
        state_in.clear_forces()
        solver.step(state_in, state_out, model.control(), None, 1.0e-3)
        state_in, state_out = state_out, state_in
    final = state_in.body_q.numpy()
    np.testing.assert_allclose(final[bodies[0]], prescribed[bodies[0]], atol=1.0e-7)
    assert final[bodies[-1], 2] > initial[bodies[-1], 2] + 1.0e-4
    assert np.isfinite(final).all()


def test_cable_dahl_history_tracks_material_twist():
    model, bodies, joints = _cable_model(dahl=True)
    solver = NewtonSolver(model)
    state_in, state_out = model.state(), model.state()
    solver.step(state_in, state_out, model.control(), None, 1.0e-3)
    poses = state_out.body_q.numpy()
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.05)
    poses[bodies[-1], 3:] = np.asarray(rotation)
    state_out.body_q.assign(poses)
    solver.sync_affine_body_state(state_out)
    state_out.clear_forces()
    solver.step(state_out, state_in, model.control(), None, 1.0e-4)
    assert np.max(np.abs(solver.joint_kappa_prev.numpy()[joints])) > 0.01
    assert np.max(np.abs(solver.joint_sigma_prev.numpy()[joints])) > 0.0
    assert np.isfinite(solver.joint_sigma_prev.numpy()).all()


@pytest.mark.parametrize("framework", [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON])
@pytest.mark.parametrize("captured", [False, True])
def test_cable_solve_and_material_history_do_not_call_vbd(monkeypatch, framework, captured):
    def reject_reference(*args, **kwargs):
        raise AssertionError("A SynReal cable run must not use SolverVBD")

    monkeypatch.setattr(newton.solvers.SolverVBD, "__init__", reject_reference)
    monkeypatch.setattr(newton.solvers.SolverVBD, "register_custom_attributes", reject_reference)
    original_launch = wp.launch

    def native_material_launch(kernel, *args, **kwargs):
        assert not kernel.module.name.startswith("newton._src.solvers.vbd"), kernel.module.name
        return original_launch(kernel, *args, **kwargs)

    monkeypatch.setattr(wp, "launch", native_material_launch)
    model, bodies, joints = _cable_model(dahl=True)
    solver = NewtonSolver(model)
    solver.config.nonlinear_solver = framework
    state_in, state_out = model.state(), model.state()
    solver.step(state_in, state_out, model.control(), None, 1.0e-3)
    poses = state_out.body_q.numpy()
    poses[bodies[-1], 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.05))
    state_out.body_q.assign(poses)
    solver.sync_affine_body_state(state_out)
    state_out.clear_forces()
    if captured:
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(state_out, state_in, model.control(), None, 1.0e-3)
        wp.capture_launch(capture.graph)
    else:
        solver.step(state_out, state_in, model.control(), None, 1.0e-3)
    final = state_in.body_q.numpy()
    assert isinstance(solver.native_solver, s3d.Solver)
    assert np.isfinite(final).all()
    assert np.max(np.abs(final[bodies[-1]] - poses[bodies[-1]])) > 1.0e-6
    assert np.max(np.abs(solver.joint_sigma_prev.numpy()[joints])) > 0.0


@pytest.mark.parametrize("is_proxy", [False, True])
def test_effective_inertia_can_lock_a_coupled_proxy_rotation(is_proxy):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_body()
    builder.add_shape_sphere(body=body, radius=0.05)
    model = builder.finalize(device="cuda:0")
    solver = NewtonSolver(model)
    if is_proxy:
        model.body_flags.assign(model.body_flags.numpy() | int(newton.BodyFlags.PROXY))
    model.body_inertia.zero_()
    model.body_inv_inertia.zero_()
    # Changing angular mobility also requires rebuilding the authored masks.
    flags = newton.ModelFlags.BODY_INERTIAL_PROPERTIES | newton.ModelFlags.BODY_PROPERTIES
    if not is_proxy:
        with pytest.raises(ValueError, match="singular second moment"):
            solver.notify_model_changed(flags)
        return

    solver.notify_model_changed(flags)
    indices = solver._affine_body_indices.numpy()[0]
    flags = solver._particle_flags.numpy()[indices]
    assert flags[0] & int(newton.ParticleFlags.ACTIVE)
    assert not np.any(flags[1:] & int(newton.ParticleFlags.ACTIVE))
    state_in, state_out = model.state(), model.state()
    state_in.body_f.assign(np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    solver.step(state_in, state_out, model.control(), None, 1.0e-3)
    pose = state_out.body_q.numpy()[body]
    assert pose[0] > 0.0
    np.testing.assert_allclose(pose[3:], [0.0, 0.0, 0.0, 1.0], atol=1.0e-7)
    assert np.isfinite(state_out.body_qd.numpy()).all()


@pytest.mark.parametrize("framework", [s3d.NonlinearSolver.NEWTON, s3d.NonlinearSolver.PD_NEWTON])
@pytest.mark.parametrize("surface", ["translate_box", "rotate_box", "spin_capsule"])
def test_prescribed_affine_surface_transfers_friction(framework, surface):
    def simulate(friction):
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.81))
        builder.default_shape_cfg.mu = friction
        builder.default_shape_cfg.kf = 1.0e4
        capsule = surface == "spin_capsule"
        rotate = surface != "translate_box"
        initial_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), np.pi / 2) if capsule else wp.quat_identity()
        floor_z = 0.0 if capsule else -0.1
        floor = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, floor_z), initial_rotation))
        if capsule:
            builder.add_shape_capsule(floor, radius=0.1, half_height=0.5)
        else:
            builder.add_shape_box(floor, hx=1.0, hy=1.0, hz=0.1)
        builder.body_mass[floor] = 0.0
        builder.body_inv_mass[floor] = 0.0
        builder.body_inertia[floor] = wp.mat33(0.0)
        builder.body_inv_inertia[floor] = wp.mat33(0.0)
        block_position = wp.vec3(0.4 if surface == "rotate_box" else 0.0, 0.0, 0.198 if capsule else 0.098)
        block = builder.add_body(xform=wp.transform(block_position, wp.quat_identity()))
        builder.add_shape_box(block, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cuda:0")
        solver = NewtonSolver(model)
        solver.config.nonlinear_solver = framework
        solver.config.nonlinear_iterations = 5
        state_in, state_out = model.state(), model.state()
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        dt = 1.0 / 240.0
        for step in range(30):
            poses = state_in.body_q.numpy()
            if capsule:
                spin = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -(step + 1) * 5.0 * dt)
                poses[floor, 3:] = np.asarray(spin * initial_rotation)
            elif rotate:
                poses[floor, 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), (step + 1) * 1.25 * dt))
            else:
                poses[floor, 0] = (step + 1) * 0.5 * dt
            state_in.body_q.assign(poses)
            state_in.clear_forces()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, model.control(), contacts, dt)
            state_in, state_out = state_out, state_in
        # Axial capsule spin leaves the centerline stationary. Its radial contact
        # offset must participate in friction to move the supported block.
        assert int(contacts.rigid_contact_count.numpy()[0]) > 0
        np.testing.assert_allclose(state_in.body_q.numpy()[floor], poses[floor], atol=1.0e-7)
        return float(state_in.body_q.numpy()[block, 1 if rotate else 0])

    assert abs(simulate(0.0)) < 1.0e-4
    assert simulate(1.0) > 0.02

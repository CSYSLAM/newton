"""The fused, concurrent PCG schedule matches the reference kernel order up to floating-point summation order."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton.solvers import coupled_high_accuracy as s3d

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="The linear schedule requires CUDA")


def _run(reference: bool, capture: bool, reinitialize: bool = False):
    # A tetrahedral block pressed by a driven sphere onto the ground: it exercises the tetrahedron, SO(3), drive,
    # self-collision, surface and soft contact tangents that the default schedule runs concurrently.
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    inertia = wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02)
    body = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
    joint = builder.add_joint_prismatic(
        parent=-1,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.15, 0.15, 0.32), wp.quat_identity()),
        target_q=-0.04,
        target_ke=2000.0,
        target_kd=20.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    builder.add_articulation([joint])
    builder.add_shape_sphere(body, radius=0.1)
    builder.add_ground_plane()
    builder.add_soft_grid(
        pos=wp.vec3(0.0, 0.0, 0.006),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=3,
        dim_y=3,
        dim_z=2,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.06,
        density=500.0,
        k_mu=2.0e4,
        k_lambda=2.0e4,
        k_damp=0.01,
        particle_radius=0.005,
    )
    model = builder.finalize(device="cuda:0")
    solver = s3d.NewtonSolver(model)
    config = solver.config
    config.reference_linear_schedule = reference
    config.affine_body_rigidity_model = s3d.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
    config.joint_constraint_mode = s3d.JointConstraintMode.AUGMENTED_LAGRANGIAN
    config.inertia_mode = s3d.InertiaMode.NONE
    config.nonlinear_iterations = 3
    config.linear_iterations = 60
    config.enable_self_collision = True
    config.self_collision_radius = 0.004
    config.enable_collision_hessian_non_diagonal = True
    state, other = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    solver.sync_affine_body_state(state)
    control = model.control()
    wp.copy(control.joint_target_q, model.joint_target_q)
    wp.copy(control.joint_target_qd, model.joint_target_qd)
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)

    def two_substeps():
        for current, following in ((state, other), (other, state)):
            current.clear_forces()
            pipeline.collide(current, contacts)
            solver.step(current, following, control, contacts, 1.0 / 240.0)

    if capture:
        with wp.ScopedCapture() as recorded:
            two_substeps()
        for _ in range(10):
            wp.capture_launch(recorded.graph)
    else:
        for _ in range(10):
            two_substeps()
    if reinitialize:
        # A model change re-creates the native device context; caches sized for the old one must not be reused.
        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
        for _ in range(10):
            two_substeps()
    wp.synchronize_device(model.device)
    launches = dict(solver.native_solver.source_launch_counts)
    return state.particle_q.numpy(), state.body_q.numpy(), int(contacts.soft_contact_count.numpy()[0]), launches


@pytest.mark.parametrize("capture", (False, True))
def test_fused_schedule_matches_reference_order(capture):
    reference_q, reference_body, reference_contacts, reference_launches = _run(True, capture)
    fused_q, fused_body, fused_contacts, fused_launches = _run(False, capture)
    assert reference_contacts > 0 and fused_contacts > 0
    assert np.isfinite(fused_q).all() and np.isfinite(fused_body).all()
    # Each schedule ran its own kernels: the reference never fuses, the default never falls back.
    assert reference_launches.get("fused_update", 0) == 0 and reference_launches["update"] > 0
    assert fused_launches["fused_update"] > 0 and fused_launches["surface_tangent_cached"] > 0
    assert fused_launches.get("surface_tangent", 0) == 0
    # Only the summation order differs, so both trajectories agree far below the 5 mm particle radius.
    np.testing.assert_allclose(fused_q, reference_q, atol=2.0e-5)
    np.testing.assert_allclose(fused_body, reference_body, atol=2.0e-5)


def test_reinitialized_solver_does_not_reuse_released_caches():
    reference_q, reference_body, _, _ = _run(True, False, reinitialize=True)
    fused_q, fused_body, fused_contacts, launches = _run(False, False, reinitialize=True)
    assert fused_contacts > 0 and launches["surface_tangent_cached"] > 0
    np.testing.assert_allclose(fused_q, reference_q, atol=2.0e-5)
    np.testing.assert_allclose(fused_body, reference_body, atol=2.0e-5)

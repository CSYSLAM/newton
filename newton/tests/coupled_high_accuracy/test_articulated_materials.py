"""Exercise material restoration alongside driven joints, self collision, and surface contacts."""

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton.solvers import coupled_high_accuracy as s3d

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="Articulated materials require CUDA")


@pytest.mark.parametrize("material,mode", (("cloth", "NEWTON"), ("cloth", "PD_NEWTON"), ("tetrahedron", "NEWTON")))
@pytest.mark.parametrize("capture", (False, True))
@pytest.mark.parametrize("joint_kind", ("revolute", "revolute_prismatic"))
def test_material_restoration_with_driven_joint(material, mode, capture, joint_kind):
    from newton.solvers import style3d

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    inertia = wp.mat33(0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.02)
    parent = -1
    parent_xform = wp.transform(wp.vec3(2.0, 0.0, 1.0), wp.quat_identity())
    joints = []
    if joint_kind == "revolute_prismatic":
        # Match an arm plus sliding finger: both joint springs and a point-line constraint must be present.
        parent = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
        joints.append(
            builder.add_joint_revolute(
                parent=-1,
                child=parent,
                axis=(0.0, 0.0, 1.0),
                parent_xform=parent_xform,
                target_ke=20.0,
                target_kd=1.0,
                actuator_mode=newton.JointTargetMode.POSITION,
            )
        )
        parent_xform = wp.transform_identity()
    body = builder.add_link(mass=1.0, inertia=inertia, lock_inertia=True)
    add_joint = builder.add_joint_prismatic if joint_kind == "revolute_prismatic" else builder.add_joint_revolute
    joint = add_joint(
        parent=parent,
        child=body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=parent_xform,
        target_q=0.1,
        target_ke=20.0,
        target_kd=1.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    builder.add_articulation(joints + [joint])
    builder.add_shape_sphere(body, radius=0.1)
    builder.add_ground_plane()
    if material == "cloth":
        style3d.add_cloth_grid(
            builder,
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            particle_radius=0.005,
            tri_aniso_ke=wp.vec3(1.0e4, 1.0e4, 1.0e4),
        )
    else:
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=1,
            dim_y=1,
            dim_z=1,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=1000.0,
            k_mu=1.0e4,
            k_lambda=1.0e4,
            k_damp=0.0,
            particle_radius=0.005,
        )
    material_count = builder.particle_count
    model = builder.finalize(device="cuda:0")
    solver = s3d.NewtonSolver(model)
    solver.config.affine_body_rigidity_model = s3d.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
    solver.config.joint_constraint_mode = s3d.JointConstraintMode.AUGMENTED_LAGRANGIAN
    solver.config.nonlinear_solver = getattr(s3d.NonlinearSolver, mode)
    solver.config.inertia_mode = s3d.InertiaMode.NONE
    solver.config.nonlinear_iterations = 4
    solver.config.linear_iterations = 20
    solver.config.enable_self_collision = True
    solver.config.self_collision_radius = 0.005
    solver.config.enable_collision_hessian_non_diagonal = True
    state, other = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    solver.sync_affine_body_state(state)
    positions = state.particle_q.numpy()
    positions[:material_count, 0] *= 1.1
    initial_width = np.ptp(positions[:material_count, 0])
    state.particle_q.assign(positions)
    control = model.control()
    wp.copy(control.joint_target_q, model.joint_target_q)
    wp.copy(control.joint_target_qd, model.joint_target_qd)
    pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
    contacts = pipeline.contacts()
    # Compile each generated collision specialization before entering CUDA graph capture.
    pipeline.collide(state, contacts)

    def two_substeps():
        for current, following in ((state, other), (other, state)):
            current.clear_forces()
            pipeline.collide(current, contacts)
            solver.step(current, following, control, contacts, 0.002)

    if capture:
        with wp.ScopedCapture() as recorded:
            two_substeps()
        for _ in range(3):
            wp.capture_launch(recorded.graph)
    else:
        for _ in range(3):
            two_substeps()
    wp.synchronize_device(model.device)
    for field in ("particle_q", "particle_qd", "body_q", "body_qd"):
        assert np.isfinite(getattr(state, field).numpy()).all(), field
    assert np.ptp(state.particle_q.numpy()[:material_count, 0]) < initial_width
    assert np.linalg.norm(state.body_qd.numpy()) > 0.0
    if getattr(solver.native_solver, "source_launch_counts_available", True):
        launches = solver.native_solver.source_launch_counts
        assert launches["stretch_force" if material == "cloth" else "tetrahedron_force"] > 0
        assert launches["drive_load"] > 0
    if material == "tetrahedron":
        # PD with tetrahedra and driven joints uses the general schedule rather than the coupled-material one.
        solver.config.nonlinear_solver = s3d.NonlinearSolver.PD_NEWTON
        solver.step(state, other, control, contacts, 0.002)
        wp.synchronize_device(model.device)
        for field in ("particle_q", "body_q"):
            assert np.isfinite(getattr(other, field).numpy()).all(), field

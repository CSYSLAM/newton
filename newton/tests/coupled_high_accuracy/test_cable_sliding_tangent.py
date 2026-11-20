"""Hard-contact sliding uses an optional PSD derivative of the Coulomb disk."""

import numpy as np
import pytest


def _disk_force(displacement, normal, stiffness, dual, initial, alpha, limit):
    projector = np.eye(3) - np.outer(normal, normal)
    trial = stiffness * (projector @ displacement + (1.0 - alpha) * projector @ initial) + projector @ dual
    trial_length = np.linalg.norm(trial)
    return trial if trial_length <= limit else trial * (limit / trial_length)


def _disk_metric(displacement, normal, stiffness, dual, initial, alpha, limit):
    projector = np.eye(3) - np.outer(normal, normal)
    trial = stiffness * (projector @ displacement + (1.0 - alpha) * projector @ initial) + projector @ dual
    trial_length = np.linalg.norm(trial)
    if trial_length <= limit:
        return stiffness * projector, stiffness
    direction = trial / trial_length
    bound = stiffness * limit / trial_length
    return bound * (projector - np.outer(direction, direction)), bound


def _finite_difference(function, value, epsilon=1.0e-7):
    return np.column_stack(
        [
            (function(value + epsilon * axis) - function(value - epsilon * axis)) / (2.0 * epsilon)
            for axis in np.eye(value.size)
        ]
    )


@pytest.mark.parametrize("alpha", [0.0, 0.95])
@pytest.mark.parametrize("trial_fraction", [0.0, 0.4, 3.0])
def test_cpu_coulomb_disk_derivative_and_affine_blocks(alpha, trial_fraction):
    rng = np.random.default_rng(20261025)
    normal = rng.normal(size=3)
    normal /= np.linalg.norm(normal)
    projector = np.eye(3) - np.outer(normal, normal)
    direction = projector @ rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    stiffness, limit = 70000.0, 8.5
    dual, initial = rng.normal(size=3), rng.normal(size=3) * 0.01
    displacement = (trial_fraction * limit * direction - projector @ dual) / stiffness
    displacement -= (1.0 - alpha) * projector @ initial
    force = lambda value: _disk_force(value, normal, stiffness, dual, initial, alpha, limit)
    metric, bound = _disk_metric(displacement, normal, stiffness, dual, initial, alpha, limit)
    np.testing.assert_allclose(_finite_difference(force, displacement), metric, rtol=1.0e-6, atol=1.0e-3)
    np.testing.assert_allclose(metric, metric.T, atol=1.0e-10)
    assert np.linalg.eigvalsh(metric).min() >= -1.0e-9
    assert np.linalg.eigvalsh(bound * np.eye(3) - metric).min() >= -1.0e-9

    # Both surface anchors include COM and radial offsets; their signed affine weights form the full contact Jacobian.
    centers = rng.normal(size=(2, 3)) * 0.1
    points = centers + rng.normal(size=(2, 3)) * 0.03
    weights = np.concatenate(
        [sign * np.r_[1.0, point - center] for sign, point, center in zip((-1.0, 1.0), points, centers, strict=True)]
    )
    jacobian = np.concatenate([weight * np.eye(3) for weight in weights], axis=1)
    base = np.linalg.lstsq(-jacobian, displacement, rcond=None)[0]
    affine_force = lambda value: jacobian.T @ force(-jacobian @ value)
    assembled = jacobian.T @ metric @ jacobian
    np.testing.assert_allclose(_finite_difference(affine_force, base), -assembled, rtol=1.0e-6, atol=1.0e-3)

    # Scalar diagonal bounds, anisotropic corrections and cross blocks must reconstruct that same operator.
    reconstructed = np.zeros_like(assembled)
    for row, row_weight in enumerate(weights):
        for column, column_weight in enumerate(weights):
            block = row_weight * column_weight * metric
            if row == column:
                scalar = row_weight**2 * bound * np.eye(3)
                block = scalar + (block - scalar)
            reconstructed[3 * row : 3 * row + 3, 3 * column : 3 * column + 3] = block
    np.testing.assert_allclose(reconstructed, assembled, atol=1.0e-10)
    if trial_fraction > 1.0:
        assert abs(direction @ metric @ direction) < 1.0e-8
        assert direction @ (stiffness * projector) @ direction > 0.99 * stiffness


def test_sliding_tangent_config_defaults_to_legacy():
    from newton.solvers import coupled_high_accuracy as cha

    config = cha.Config()
    assert config.rigid_contact_sliding_tangent is False
    config.rigid_contact_sliding_tangent = True
    assert config.rigid_contact_sliding_tangent is True


@pytest.mark.parametrize("framework_name", ["NEWTON", "PD_NEWTON"])
@pytest.mark.parametrize("sliding_tangent", [False, True])
@pytest.mark.parametrize("speed", [0.01, 0.5])
def test_translation_only_hard_contact_matches_selected_disk_tangent(
    framework_name, sliding_tangent, speed, monkeypatch
):
    newton = pytest.importorskip("newton")
    wp = pytest.importorskip("warp")
    if not wp.is_cuda_available():
        pytest.skip("Hard contact simulation requires CUDA")
    from newton.solvers import coupled_high_accuracy as cha

    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 0.0
    builder.default_shape_cfg.mu = 0.5
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.09), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1),
        lock_inertia=True,
    )
    builder.add_shape_sphere(
        body=body,
        radius=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, ke=1.0e4, kd=0.0, mu=0.5),
    )
    builder.add_ground_plane()
    model = builder.finalize(device="cuda:0")
    solver = cha.NewtonSolver(model)
    solver.config.nonlinear_solver = getattr(cha.NonlinearSolver, framework_name)
    solver.config.nonlinear_iterations = 1
    solver.config.linear_iterations = 40
    solver.config.affine_body_block_preconditioner = True
    solver.config.affine_body_contact_block_preconditioner = True
    solver.config.enable_collision_hessian_non_diagonal = True
    solver.config.inertia_mode = cha.InertiaMode.NONE
    solver.config.rigid_contact_hard = True
    solver.config.rigid_contact_history = False
    solver.config.rigid_contact_sliding_tangent = sliding_tangent
    # A translation-only proxy removes rotational compliance so the exact position increment is a scalar equation.
    model.body_flags.assign(model.body_flags.numpy() | int(newton.BodyFlags.PROXY))
    model.body_inertia.zero_()
    model.body_inv_inertia.zero_()
    solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES | newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
    state_in, state_out = model.state(), model.state()
    velocity = np.zeros_like(state_in.body_qd.numpy())
    velocity[body, 0] = speed
    state_in.body_qd.assign(velocity)
    solver.sync_affine_body_state(state_in)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    assert int(contacts.rigid_contact_count.numpy()[0]) == 1
    initial_height = float(state_in.body_q.numpy()[body, 2])
    state_in.clear_forces()
    dt, stiffness, mass = 0.01, 1.0e4, float(model.body_mass.numpy()[body])
    solver.step(state_in, state_out, model.control(), contacts, dt)
    normal_load = stiffness * (1.0 - solver.config.rigid_contact_alpha) * (0.1 - initial_height)
    limit = 0.5 * normal_load
    trial_length = stiffness * dt * speed
    clipped_force = min(trial_length, limit)
    radial_stiffness = 0.0 if sliding_tangent and trial_length > limit else stiffness
    expected_x = dt * speed - clipped_force / (mass / dt**2 + radial_stiffness)
    expected_z = initial_height + normal_load / (mass / dt**2 + stiffness)
    np.testing.assert_allclose(state_out.body_q.numpy()[body, :3], [expected_x, 0.0, expected_z], rtol=0.0, atol=2.0e-7)

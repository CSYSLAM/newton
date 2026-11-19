from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")

from newton._src.solvers.coupled_high_accuracy import ModelBuilder, NewtonSolver, NonlinearSolver

pytestmark = pytest.mark.skipif(not wp.is_cuda_available(), reason="NewtonSolver contact tests require CUDA")


def _simulate_contacts(
    builder: ModelBuilder,
    *,
    steps: int,
    dt: float = 1.0 / 240.0,
    configure_model: Callable[[Any], None] | None = None,
    apply_forces: Callable[[Any, int], None] | None = None,
    full_surface: bool = False,
    configure_solver: Callable[[NewtonSolver], None] | None = None,
) -> tuple[Any, Any, Any, NewtonSolver]:
    model = builder.finalize(device="cuda:0")
    if configure_model is not None:
        configure_model(model)
    state_in = model.state()
    state_out = model.state()
    solver = NewtonSolver(model)
    if configure_solver is not None:
        configure_solver(solver)
    collision_pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=full_surface)
    contacts = collision_pipeline.contacts()

    for step in range(steps):
        state_in.clear_forces()
        if apply_forces is not None:
            apply_forces(state_in, step)
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in

    wp.synchronize_device(model.device)
    return model, state_in, contacts, solver


def _add_rigid_primitive(
    builder: ModelBuilder,
    shape: str,
    *,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[int, float]:
    body = builder.add_body(
        xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
        label=f"contact_{shape}",
    )
    builder.body_qd[body] = wp.spatial_vector(*velocity, 0.0, 0.0, 0.0)
    if shape == "sphere":
        support_height = 0.25
        builder.add_shape_sphere(body, radius=support_height)
    elif shape == "box":
        support_height = 0.2
        builder.add_shape_box(body, hx=0.3, hy=0.25, hz=support_height)
    else:
        raise ValueError(f"Unsupported test primitive: {shape}")
    return body, support_height


@pytest.mark.parametrize("shape", ("sphere", "box"))
def test_rigid_primitives_settle_on_ground(shape: str) -> None:
    builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.default_shape_cfg.mu = 0.5
    builder.add_ground_plane()
    body, expected_height = _add_rigid_primitive(builder, shape, position=(0.0, 0.0, 0.8))

    _, state, contacts, _ = _simulate_contacts(builder, steps=240)
    pose = state.body_q.numpy()[body]
    velocity = state.body_qd.numpy()[body]

    assert int(contacts.rigid_contact_count.numpy()[0]) > 0
    assert np.isfinite(pose).all()
    assert np.isfinite(velocity).all()
    assert np.isclose(pose[2], expected_height, atol=3.0e-2)
    assert abs(float(velocity[2])) < 0.1


def _run_rigid_slide(friction: float) -> tuple[float, int]:
    builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.default_shape_cfg.mu = friction
    builder.default_shape_cfg.kf = 1.0e4
    builder.add_ground_plane()
    body, _ = _add_rigid_primitive(
        builder,
        "box",
        position=(0.0, 0.0, 0.18),
        velocity=(2.0, 0.0, 0.0),
    )

    _, state, contacts, _ = _simulate_contacts(builder, steps=20)
    speed = abs(float(state.body_qd.numpy()[body, 0]))
    count = int(contacts.rigid_contact_count.numpy()[0])
    return speed, count


def test_rigid_contact_friction_reduces_tangential_speed() -> None:
    frictionless_speed, frictionless_contacts = _run_rigid_slide(0.0)
    friction_speed, friction_contacts = _run_rigid_slide(1.0)

    assert frictionless_contacts > 0
    assert friction_contacts > 0
    assert friction_speed < 0.8 * frictionless_speed


def _run_box_under_horizontal_load(friction: float) -> float:
    builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.default_shape_cfg.mu = friction
    builder.default_shape_cfg.kf = 1.0e4
    builder.add_ground_plane()
    body, _ = _add_rigid_primitive(builder, "box", position=(0.0, 0.0, 0.2))
    body_force = wp.array(
        (wp.spatial_vector(100.0, 0.0, 0.0, 0.0, 0.0, 0.0),),
        dtype=wp.spatial_vector,
        device="cuda:0",
    )

    def apply_force(state: Any, _step: int) -> None:
        wp.copy(state.body_f, body_force)

    _, state, _, _ = _simulate_contacts(builder, steps=120, apply_forces=apply_force)
    return float(state.body_q.numpy()[body, 0])


def test_static_friction_resists_subthreshold_horizontal_load() -> None:
    frictionless_displacement = abs(_run_box_under_horizontal_load(0.0))
    friction_displacement = abs(_run_box_under_horizontal_load(1.0))

    assert frictionless_displacement > 2.0e-2
    assert friction_displacement < 0.25 * frictionless_displacement


def test_undamped_rigid_contact_produces_rebound() -> None:
    builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 0.0
    builder.default_shape_cfg.mu = 0.0
    builder.add_ground_plane()
    body, _ = _add_rigid_primitive(
        builder,
        "sphere",
        position=(0.0, 0.0, 0.2),
        velocity=(0.0, 0.0, -1.0),
    )

    _, state, contacts, _ = _simulate_contacts(builder, steps=4)
    velocity = state.body_qd.numpy()[body]

    assert int(contacts.rigid_contact_count.numpy()[0]) > 0
    assert np.isfinite(velocity).all()
    assert velocity[2] > 0.0


def test_dynamic_rigid_contact_preserves_pair_momentum() -> None:
    builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    builder.default_shape_cfg.mu = 0.0
    body0, _ = _add_rigid_primitive(
        builder,
        "sphere",
        position=(-0.24, 0.0, 1.0),
        velocity=(1.0, 0.0, 0.0),
    )
    body1, _ = _add_rigid_primitive(
        builder,
        "sphere",
        position=(0.24, 0.0, 1.0),
        velocity=(-1.0, 0.0, 0.0),
    )

    _, state, contacts, _ = _simulate_contacts(builder, steps=4)
    velocities = state.body_qd.numpy()[[body0, body1], :3]

    assert int(contacts.rigid_contact_count.numpy()[0]) > 0
    assert np.isfinite(velocities).all()
    assert velocities[1, 0] - velocities[0, 0] > 0.0
    assert np.allclose(velocities.sum(axis=0), 0.0, atol=2.0e-4)


def test_two_sphere_stack_reaches_stable_rest() -> None:
    builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.default_shape_cfg.mu = 0.75
    builder.add_ground_plane()
    lower, _ = _add_rigid_primitive(builder, "sphere", position=(0.0, 0.0, 0.24))
    upper, _ = _add_rigid_primitive(builder, "sphere", position=(0.0, 0.0, 0.73))

    _, state, contacts, _ = _simulate_contacts(builder, steps=240)
    poses = state.body_q.numpy()[[lower, upper]]
    velocities = state.body_qd.numpy()[[lower, upper]]

    assert int(contacts.rigid_contact_count.numpy()[0]) >= 2
    assert np.isfinite(poses).all()
    assert np.isfinite(velocities).all()
    assert np.isclose(poses[0, 2], 0.25, atol=4.0e-2)
    assert np.isclose(poses[1, 2], 0.75, atol=6.0e-2)
    assert np.max(np.abs(velocities[:, 2])) < 0.15


def _run_soft_particle_contact(friction: float) -> tuple[np.ndarray, int]:
    builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    builder.default_shape_cfg.mu = friction
    builder.add_ground_plane()
    builder.add_particle(
        pos=(0.0, 0.0, 0.04),
        vel=(2.0, 0.0, -1.0),
        mass=1.0,
        radius=0.05,
    )

    def configure_model(model: Any) -> None:
        model.soft_contact_ke = 1.0e5
        model.soft_contact_kd = 1.0e3
        model.soft_contact_mu = friction

    _, state, contacts, _ = _simulate_contacts(builder, steps=8, configure_model=configure_model)
    return state.particle_qd.numpy()[0].copy(), int(contacts.soft_contact_count.numpy()[0])


def test_soft_contact_reverses_impact_and_applies_friction() -> None:
    frictionless_velocity, frictionless_contacts = _run_soft_particle_contact(0.0)
    friction_velocity, friction_contacts = _run_soft_particle_contact(1.0)

    assert frictionless_contacts > 0
    assert friction_contacts > 0
    assert frictionless_velocity[2] > -1.0
    assert friction_velocity[2] > -1.0
    assert abs(float(friction_velocity[0])) < abs(float(frictionless_velocity[0]))


def test_replicated_cloth_rigid_soft_contacts_remain_world_local() -> None:
    from newton.solvers import style3d

    world_count = 2
    scene = ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    newton.solvers.SolverStyle3D.register_custom_attributes(scene)
    style3d.add_cloth_grid(
        scene,
        pos=wp.vec3(0.0, 0.0, 0.22),
        rot=wp.quat_identity(),
        dim_x=1,
        dim_y=1,
        cell_x=0.2,
        cell_y=0.2,
        vel=wp.vec3(0.0),
        mass=0.1,
        particle_radius=0.05,
    )
    body = scene.add_body(
        xform=wp.transform(wp.vec3(0.1, 0.1, 0.0), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    scene.add_shape_box(body, hx=0.3, hy=0.3, hz=0.2)

    builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    builder.replicate(scene, world_count)

    def configure_model(model: Any) -> None:
        model.soft_contact_ke = 1.0e4
        model.soft_contact_kd = 1.0e2

    model, state, contacts, solver = _simulate_contacts(
        builder,
        steps=1,
        configure_model=configure_model,
    )
    particles = state.particle_q.numpy()
    particles_per_world = solver.synreal_model.particle_count

    assert int(contacts.soft_contact_count.numpy()[0]) >= world_count
    assert solver.synreal_model.world_count == world_count
    assert np.isfinite(particles).all()
    assert np.allclose(
        particles[:particles_per_world],
        particles[particles_per_world:],
        atol=1.0e-6,
    )


@pytest.mark.parametrize("world_count", (2, 4))
def test_replicated_rigid_contacts_produce_identical_world_states(
    world_count: int,
) -> None:
    scene = ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    scene.default_shape_cfg.mu = 0.5
    _add_rigid_primitive(
        scene,
        "sphere",
        position=(0.0, 0.0, 0.2),
        velocity=(0.25, 0.0, -0.5),
    )
    builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.default_shape_cfg.mu = 0.5
    builder.replicate(scene, world_count)
    builder.add_ground_plane()

    model, state, contacts, solver = _simulate_contacts(builder, steps=8)
    poses = state.body_q.numpy()
    velocities = state.body_qd.numpy()
    bodies_per_world = model.body_count // world_count

    assert int(contacts.rigid_contact_count.numpy()[0]) >= world_count
    assert solver.synreal_model.world_count == world_count
    for world in range(1, world_count):
        start = world * bodies_per_world
        assert np.allclose(
            poses[:bodies_per_world],
            poses[start : start + bodies_per_world],
            atol=1.0e-6,
        )
        assert np.allclose(
            velocities[:bodies_per_world],
            velocities[start : start + bodies_per_world],
            atol=1.0e-6,
        )


def test_replicated_dynamic_contacts_remain_world_local() -> None:
    world_count = 3
    scene = ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    scene.default_shape_cfg.mu = 0.0
    _add_rigid_primitive(
        scene,
        "sphere",
        position=(-0.24, 0.0, 1.0),
        velocity=(1.0, 0.0, 0.0),
    )
    _add_rigid_primitive(
        scene,
        "sphere",
        position=(0.24, 0.0, 1.0),
        velocity=(-1.0, 0.0, 0.0),
    )
    builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    builder.default_shape_cfg.mu = 0.0
    builder.replicate(scene, world_count)

    model, state, contacts, _ = _simulate_contacts(builder, steps=4)
    velocities = state.body_qd.numpy()[:, :3]
    bodies_per_world = model.body_count // world_count
    reference = velocities[:bodies_per_world]

    assert int(contacts.rigid_contact_count.numpy()[0]) >= world_count
    assert reference[1, 0] - reference[0, 0] > 0.0
    assert np.allclose(reference.sum(axis=0), 0.0, atol=2.0e-4)
    for world in range(1, world_count):
        start = world * bodies_per_world
        assert np.allclose(
            reference,
            velocities[start : start + bodies_per_world],
            atol=1.0e-6,
        )


@pytest.mark.parametrize("contact_kind", ("rigid", "particle", "surface"))
@pytest.mark.parametrize("method", (NonlinearSolver.NEWTON, NonlinearSolver.PD_NEWTON))
def test_friction_gain_is_independent_of_normal_damping(contact_kind, method):
    from newton.solvers import style3d

    def slide(mu, gain):
        builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 0.0
        builder.default_shape_cfg.kf = gain
        builder.default_shape_cfg.mu = mu
        builder.add_ground_plane()
        if contact_kind == "rigid":
            body, _ = _add_rigid_primitive(builder, "box", position=(0.0, 0.0, 0.18), velocity=(0.5, 0.0, 0.0))
        elif contact_kind == "particle":
            builder.add_particle(pos=(0.0, 0.0, 0.01), vel=(0.5, 0.0, 0.0), mass=0.1, radius=0.02)
        else:
            newton.solvers.SolverStyle3D.register_custom_attributes(builder)
            style3d.add_cloth_grid(
                builder,
                pos=wp.vec3(0.0, 0.0, 0.01),
                rot=wp.quat_identity(),
                dim_x=1,
                dim_y=1,
                cell_x=0.1,
                cell_y=0.1,
                vel=wp.vec3(0.5, 0.0, 0.0),
                mass=0.1,
                particle_radius=0.02,
            )

        def configure(model):
            model.soft_contact_ke = 1.0e4
            model.soft_contact_kd = 0.0
            model.soft_contact_kf = gain
            model.soft_contact_mu = mu

        def configure_solver(solver):
            solver.config.nonlinear_solver = method
            solver.config.enable_collision_hessian_non_diagonal = True

        _, state, contacts, _ = _simulate_contacts(
            builder,
            steps=1,
            dt=0.005,
            configure_model=configure,
            full_surface=contact_kind == "surface",
            configure_solver=configure_solver,
        )
        if contact_kind == "rigid":
            assert contacts.rigid_contact_count.numpy()[0] > 0
            return state.body_qd.numpy()[body, 0]
        assert contacts.soft_contact_count.numpy()[0] > 0
        if contact_kind == "surface":
            assert np.any(contacts.soft_contact_indices.numpy()[:, 1] >= 0)
        return state.particle_qd.numpy()[:, 0].mean()

    frictionless = slide(0.0, 1000.0)
    disabled = slide(1.0, 0.0)
    friction = slide(1.0, 1000.0)
    np.testing.assert_allclose(disabled, frictionless, atol=1.0e-4)
    assert 0.0 <= friction < frictionless - 0.005


@pytest.mark.parametrize("full_surface", (False, True))
def test_cloth_friction_transfers_momentum_to_dynamic_rigid_body(full_surface):
    from newton.solvers import style3d

    builder = ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    style3d.add_cloth_grid(
        builder,
        pos=wp.vec3(-0.05, -0.05, 0.01),
        rot=wp.quat_identity(),
        dim_x=1,
        dim_y=1,
        cell_x=0.1,
        cell_y=0.1,
        vel=wp.vec3(0.5, 0.0, 0.0),
        mass=0.1,
        particle_radius=0.02,
    )
    cloth_count = builder.particle_count
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.2), wp.quat_identity()),
        mass=2.0,
        inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        lock_inertia=True,
    )
    cfg = builder.default_shape_cfg.copy()
    cfg.density, cfg.ke, cfg.kd, cfg.kf, cfg.mu = 0.0, 1.0e4, 0.0, 1000.0, 1.0
    builder.add_shape_box(body, hx=0.3, hy=0.3, hz=0.2, cfg=cfg)

    def configure(model):
        model.soft_contact_ke = 1.0e4
        model.soft_contact_kd = 0.0
        model.soft_contact_kf = 1000.0
        model.soft_contact_mu = 1.0

    def configure_solver(solver):
        solver.config.nonlinear_solver = NonlinearSolver.NEWTON
        solver.config.enable_collision_hessian_non_diagonal = True
        solver.config.nonlinear_iterations = 20
        solver.config.linear_iterations = 30

    model, state, contacts, _ = _simulate_contacts(
        builder,
        steps=1,
        dt=0.005,
        configure_model=configure,
        full_surface=full_surface,
        configure_solver=configure_solver,
    )
    masses = model.particle_mass.numpy()[:cloth_count]
    cloth_velocity = state.particle_qd.numpy()[:cloth_count, 0]
    rigid_velocity = state.body_qd.numpy()[body, 0]
    assert contacts.soft_contact_count.numpy()[0] > 0
    assert cloth_velocity.mean() < 0.49
    assert rigid_velocity > 0.001
    momentum = np.dot(masses, cloth_velocity) + model.body_mass.numpy()[body] * rigid_velocity
    np.testing.assert_allclose(momentum, masses.sum() * 0.5, atol=5.0e-4)

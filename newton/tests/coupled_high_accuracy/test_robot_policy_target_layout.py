"""Policy actions must respect Newton's position-target layout and CUDA graph buffer lifetimes."""

import importlib.util
import sys
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

newton = pytest.importorskip("newton")
wp = pytest.importorskip("warp")
pytest.importorskip("warp_nn.runtime")

ROOT = Path(__file__).resolve().parents[2] / "examples" / "coupled_high_accuracy"


def load_example_module(name):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.fixture(params=["example_cha_robot_policy", "example_cha_robot_cloth_policy"])
def example_module(request):
    return load_example_module(request.param)


def make_model(device):
    # Newton main always shapes joint position targets like joint_q (the legacy DOF layout was removed).
    builder = newton.ModelBuilder()
    inertia = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    root, frame, hinge, slide = [builder.add_link(mass=1.0, inertia=inertia) for _ in range(4)]
    joints = [
        builder.add_joint_free(child=root),
        builder.add_joint_fixed(parent=root, child=frame),
        builder.add_joint_revolute(parent=frame, child=hinge),
        builder.add_joint_prismatic(parent=hinge, child=slide),
    ]
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def target_launch(module, model, control, device):
    indices = module.policy_target_indices(model, 2, slice(7, None))
    actions = wp.array([[1.5, -2.0]], dtype=wp.float32, device=device)
    initial = wp.array([0.2, -0.4], dtype=wp.float32, device=device)
    reorder = wp.array([1, 0], dtype=wp.int32, device=device)
    targets = wp.array(indices, dtype=wp.int32, device=device)

    def launch():
        wp.launch(
            module._build_joint_target_q_kernel,
            dim=2,
            inputs=[actions, initial, reorder, 0.5, targets, control.joint_target_q],
            device=device,
        )

    return indices, launch


def test_policy_actions_preserve_the_free_base_targets(example_module):
    model = make_model("cpu")
    control = model.control()
    before = np.arange(len(control.joint_target_q), dtype=np.float32) + 10
    control.joint_target_q.assign(before)
    indices, launch = target_launch(example_module, model, control, "cpu")
    np.testing.assert_array_equal(indices, [7, 8])
    launch()
    expected = before.copy()
    expected[indices] = [-0.8, 0.35]
    np.testing.assert_allclose(control.joint_target_q.numpy(), expected)


@pytest.mark.skipif(not wp.is_cuda_available(), reason="CUDA graph capture requires CUDA")
def test_capture_keeps_the_model_control_buffer(example_module):
    model = make_model("cuda:0")
    example = example_module.Example.__new__(example_module.Example)
    example.device = wp.get_device("cuda:0")
    example.config = {"num_dofs": 2}
    example.control = model.control()
    before = np.arange(len(example.control.joint_target_q), dtype=np.float32) + 10
    example.control.joint_target_q.assign(before)
    indices, launch = target_launch(example_module, model, example.control, example.device)
    example.simulate = launch
    original = example.control.joint_target_q
    # Warm up compilation before capture, then restore the authored targets.
    launch()
    example.control.joint_target_q.assign(before)
    example.capture()
    assert example.control.joint_target_q is original
    assert example.control.joint_target_q.shape == model.joint_target_q.shape
    wp.capture_launch(example.graph)
    expected = before.copy()
    expected[indices] = [-0.8, 0.35]
    np.testing.assert_allclose(example.control.joint_target_q.numpy(), expected)


@pytest.mark.parametrize("fault", [None, "height", "linear_speed", "angular_speed", "cloth_speed", "nonfinite"])
def test_cloth_policy_rejects_finite_but_explosive_motion(fault):
    module = load_example_module("example_cha_robot_cloth_policy")
    bodies = np.array([[0, 0, 0.76, 0, 0, 0, 1], [0, 0, 0.03, 0, 0, 0, 1]], dtype=np.float32)
    velocities = np.zeros((2, 6), dtype=np.float32)
    cloth = np.array([[0, 0, 1], [0.015, 0, 1]], dtype=np.float32)
    cloth_velocity = np.zeros((2, 3), dtype=np.float32)
    if fault == "height":
        bodies[0, 2] = 42.66
    elif fault == "linear_speed":
        velocities[0, 0] = 22501.0
    elif fault == "angular_speed":
        velocities[0, 3] = 98639864.0
    elif fault == "cloth_speed":
        cloth_velocity[0, 0] = 101.0
    elif fault == "nonfinite":
        bodies[0, 0] = np.nan
    example = module.Example.__new__(module.Example)
    example.cloth_particle_count = 2
    example.state_0 = SimpleNamespace(
        **{
            key: SimpleNamespace(numpy=lambda array=array: array)
            for key, array in {
                "body_q": bodies,
                "body_qd": velocities,
                "particle_q": cloth,
                "particle_qd": cloth_velocity,
            }.items()
        }
    )
    if fault is None:
        example.test_post_step()
    else:
        with pytest.raises(ValueError):
            example.test_post_step()


@pytest.mark.parametrize("clearance", [0.05, 0.12])
def test_initial_sheet_clears_posed_collision_and_visual_geometry(clearance):
    module = load_example_module("example_cha_robot_cloth_policy")
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = builder.add_joint_free(child=body)
    builder.add_articulation([joint])
    rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.35)
    builder.joint_q[:7] = [1.2, -0.6, 0.7, *rotation]
    shape_rotation = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.4)
    shape_pose = wp.transform(wp.vec3(0.1, 0.05, 0.4), shape_rotation)
    collision = builder.default_shape_cfg.copy()
    collision.margin = 0.02
    builder.add_shape_box(body, xform=shape_pose, hx=0.2, hy=0.1, hz=0.6, cfg=collision)
    visual = builder.default_shape_cfg.copy()
    visual.has_shape_collision, visual.has_particle_collision, visual.density = False, False, 0.0
    vertices = np.array([[0, 0, 0], [0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.7]], dtype=np.float32)
    mesh = newton.Mesh(vertices, [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
    mesh_pose = wp.transform(wp.vec3(0, 0, 0.9), wp.quat_identity())
    builder.add_shape_mesh(body, xform=mesh_pose, mesh=mesh, scale=(0.5, 0.7, 1.3), cfg=visual)
    builder.add_ground_plane()
    for point in [(0, 0, 0), (0.6, 0, 0), (0, 0.3, 0), (0.6, 0.3, 0), (9, 9, 9)]:
        builder.add_particle(pos=wp.vec3(*point), vel=wp.vec3(0), mass=0.1, radius=0.005)
    model = builder.finalize(device="cpu")
    newton.eval_fk(model, model.joint_q, model.joint_qd, model)
    before = model.particle_q.numpy()
    lower, upper = module.place_cloth_above_robot(model, 4, clearance=clearance)
    after = model.particle_q.numpy()
    pose = wp.transform(*model.body_q.numpy()[body])
    collision_pose = wp.transform_multiply(pose, shape_pose)
    corners = np.asarray(list(product((-1, 1), repeat=3))) * [0.2, 0.1, 0.6]
    collision_world = np.asarray([wp.transform_point(collision_pose, wp.vec3(*point)) for point in corners])
    visual_pose = wp.transform_multiply(pose, mesh_pose)
    visual_world = np.asarray(
        [wp.transform_point(visual_pose, wp.vec3(*point)) for point in vertices * [0.5, 0.7, 1.3]]
    )
    assert upper[2] >= collision_world[:, 2].max() + collision.margin - 1e-6
    assert upper[2] >= visual_world[:, 2].max() - 1e-6
    assert after[:4, 2].min() - 0.005 >= upper[2] + clearance - 1e-6
    np.testing.assert_allclose(after[:4, :2].mean(axis=0), [1.2, -0.6], atol=1e-6)
    np.testing.assert_allclose(np.ptp(after[:4], axis=0), np.ptp(before[:4], axis=0), atol=1e-6)
    np.testing.assert_array_equal(after[4:], before[4:])


def test_cloth_shell_covers_scaled_visuals_without_changing_rigid_dynamics():
    from scipy.spatial import ConvexHull

    module = load_example_module("example_cha_robot_cloth_policy")
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3)))
    fallback_body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    rigid = builder.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02)
    fallback = builder.add_shape_sphere(fallback_body, radius=0.01)
    visual = builder.default_shape_cfg.copy()
    visual.density, visual.has_shape_collision, visual.has_particle_collision = 0.0, False, False
    # More than 64 extreme points catches a hull vertex limit that would cut into the render surface.
    angle = np.linspace(0, 2 * np.pi, 80, endpoint=False)
    vertices = np.vstack(
        [np.column_stack((0.08 * np.cos(angle), 0.08 * np.sin(angle), np.full(80, z))) for z in (-0.06, 0.06)]
    ).astype(np.float32)
    source = newton.Mesh(vertices, ConvexHull(vertices).simplices.ravel(), compute_inertia=False)
    pose = wp.transform(wp.vec3(0.1, -0.2, 0.3), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), 0.4))
    visible = builder.add_shape_mesh(body, xform=pose, mesh=source, scale=(0.7, 1.3, 1.2), cfg=visual)
    masses, inertia, centers = [
        np.array(a, copy=True) for a in (builder.body_mass, builder.body_inertia, builder.body_com)
    ]
    rigid_flags = builder.shape_flags[rigid]
    shells = module.add_cloth_collision_shells(builder)
    assert len(shells) == 1
    shell = shells[0]
    assert builder.shape_flags[shell] == newton.ShapeFlags.COLLIDE_PARTICLES
    assert builder.shape_flags[visible] == newton.ShapeFlags.VISIBLE
    assert builder.shape_source[visible] is source
    assert builder.shape_flags[rigid] == rigid_flags & ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
    assert builder.shape_flags[fallback] & newton.ShapeFlags.COLLIDE_PARTICLES
    assert builder.shape_force_sdf[shell]
    np.testing.assert_array_equal(builder.shape_transform[shell], pose)
    np.testing.assert_array_equal(builder.shape_scale[shell], (1, 1, 1))
    for after, before in zip((builder.body_mass, builder.body_inertia, builder.body_com), (masses, inertia, centers)):
        np.testing.assert_array_equal(after, before)
    hull = ConvexHull(builder.shape_source[shell].vertices)
    signed = (vertices * [0.7, 1.3, 1.2]) @ hull.equations[:, :3].T + hull.equations[:, 3]
    assert signed.max() < 1e-7


@pytest.mark.skipif(not wp.is_cuda_available(), reason="Coupled cloth contact requires CUDA")
@pytest.mark.parametrize("use_shell", [False, True])
def test_falling_sheet_stays_outside_visible_shell(use_shell):
    from scipy.spatial import ConvexHull

    from newton.solvers import coupled_high_accuracy as s3d
    from newton.solvers import style3d

    module = load_example_module("example_cha_robot_cloth_policy")
    with wp.ScopedDevice("cuda:0"):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), is_kinematic=True)
        cfg = builder.default_shape_cfg.copy()
        cfg.density, cfg.is_visible = 0.0, False
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.08, cfg=cfg)
        cfg.has_shape_collision, cfg.has_particle_collision, cfg.is_visible = False, False, True
        vertices = np.asarray(list(product((-0.1, 0.1), (-0.1, 0.1), (-0.12, 0.12))), dtype=np.float32)
        mesh = newton.Mesh(vertices, ConvexHull(vertices).simplices.ravel(), compute_inertia=False)
        builder.add_shape_mesh(body, mesh=mesh, cfg=cfg)
        if use_shell:
            module.add_cloth_collision_shells(builder)
        style3d.add_cloth_grid(
            builder,
            pos=wp.vec3(-0.04, -0.04, 0.17),
            rot=wp.quat_identity(),
            vel=wp.vec3(0),
            dim_x=2,
            dim_y=2,
            cell_x=0.04,
            cell_y=0.04,
            mass=0.001,
            particle_radius=0.005,
            tri_aniso_ke=wp.vec3(1000, 1000, 100),
        )
        count = builder.particle_count
        model = builder.finalize()
        model.soft_contact_ke, model.soft_contact_kd = 2000.0, 20.0
        solver = s3d.SolverCoupledHighAccuracy(model)
        solver.config.nonlinear_iterations, solver.config.linear_iterations = 5, 50
        solver.config.enable_collision_hessian_non_diagonal = True
        pipeline = newton.CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True, soft_contact_gap=0.006)
        contacts, control = pipeline.contacts(), model.control()
        first, second = model.state(), model.state()

        def step_pair():
            for current, following in ((first, second), (second, first)):
                current.clear_forces()
                pipeline.collide(current, contacts)
                solver.step(current, following, control, contacts, 0.001)

        step_pair()
        with wp.ScopedCapture() as captured:
            step_pair()
        for _ in range(150):
            wp.capture_launch(captured.graph)
        z = first.particle_q.numpy()[:count, 2]
        assert np.isfinite(z).all()
        # The unmodified imported proxy is 4 cm inside the visible shell, reproducing the original failure.
        if use_shell:
            assert z.min() > 0.12 - 0.001
        else:
            assert z.max() < 0.10

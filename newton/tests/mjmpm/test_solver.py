# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small physical acceptance cases for the independent MJMPM backend."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM, SolverMJMPM


def _box(builder, position, *, half=0.1):
    body = builder.add_body(xform=wp.transform(wp.vec3(position), wp.quat_identity()))
    builder.add_shape_box(body, hx=half, hy=half, hz=half)
    return body


def _particles(builder, *, velocity=(0.0, 0.0, 0.0), position=(0.025, 0.025, 0.025), custom_attributes=None):
    builder.add_particle_grid(
        pos=wp.vec3(position),
        rot=wp.quat_identity(),
        vel=wp.vec3(velocity),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.05,
        cell_y=0.05,
        cell_z=0.05,
        mass=0.01,
        jitter=0.0,
        radius_mean=0.025,
        custom_attributes={"mpm:young_modulus": 1.0e4, "mpm:poisson_ratio": 0.2} | (custom_attributes or {}),
    )


def _options():
    return {
        "grid_type": "dense",
        "voxel_size": 0.1,
        "max_iterations": 10,
        "solver": "jacobi",
        "tolerance": 0.0,
        "warmstart_mode": "none",
        "air_drag": 0.0,
    }


def _loaded_tray(*, device="cpu", particles=True):
    """Build the inclined tray with either grains or an equivalent external load."""
    slope = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(np.deg2rad(20.0)))
    builder = newton.ModelBuilder()
    SolverMJMPM.register_custom_attributes(builder)
    cfg = newton.ModelBuilder.ShapeConfig(density=500.0, mu=0.8)
    builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), slope), hx=0.5, hy=0.3, hz=0.05, cfg=cfg)
    pose = wp.transform(wp.quat_rotate(slope, wp.vec3(0.0, 0.0, 0.0201)), slope)
    tray = builder.add_body(xform=pose)
    joint = builder.joint_count - 1
    builder.add_shape_box(tray, hx=0.08, hy=0.08, hz=0.02, cfg=cfg)
    spacing = 0.02
    if particles:
        for i in range(6):
            for j in range(6):
                for k in range(2):
                    local = wp.vec3(-0.05 + i * spacing, -0.05 + j * spacing, 0.03 + k * spacing)
                    builder.add_particle(
                        wp.transform_point(pose, local), wp.vec3(0.0), 1600.0 * spacing**3, 0.5 * spacing
                    )
    return builder.finalize(device=device), tray, joint, np.asarray(wp.quat_rotate(slope, wp.vec3(1.0, 0.0, 0.0)))


def _box_on_heavy_material(mass, drop, *, device="cuda:0"):
    """Build a light four-centimeter box over a five-kilogram material bed."""
    builder = newton.ModelBuilder()
    SolverMJMPM.register_custom_attributes(builder)
    builder.add_ground_plane()
    builder.add_particle_grid(
        pos=wp.vec3(-0.09, -0.09, 0.01),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=10,
        dim_y=10,
        dim_z=5,
        cell_x=0.02,
        cell_y=0.02,
        cell_z=0.02,
        mass=0.01,
        jitter=0.0,
        radius_mean=0.01,
        custom_attributes={"mpm:friction": 0.5},
    )
    box = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.12 + drop), wp.quat_identity()),
        mass=mass,
        inertia=wp.mat33(np.eye(3) * mass * 0.04**2 / 6.0),
    )
    joint = builder.joint_count - 1
    builder.add_shape_box(box, hx=0.02, hy=0.02, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5))
    return builder.finalize(device=device), box, joint


class TestMJMPMRigid(unittest.TestCase):
    def test_constrained_response_matches_scalar_compliance(self):
        """Match the exact one-dimensional response to hard, soft, and redundant support rows."""
        from newton._src.solvers.mjmpm.rigid import response  # noqa: PLC0415

        for mass in (0.0016, 0.512):
            for softness in (0.0, 2.0, 1000.0):
                for rows in (1, 4):
                    with self.subTest(mass=mass, softness=softness, rows=rows):
                        device = "cpu"
                        enabled = wp.array([1], dtype=int, device=device)
                        pose = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
                        inverse_mass = wp.array([1.0 / mass], dtype=float, device=device)
                        inverse_inertia = wp.array([np.eye(3)], dtype=wp.mat33, device=device)
                        count = wp.array([rows], dtype=int, device=device)
                        bodies = wp.array([[-1, 0]] * rows, dtype=wp.vec2i, device=device)
                        radius = wp.zeros(rows, dtype=wp.vec3, device=device)
                        normal = wp.array([[0.0, 0.0, 1.0]] * rows, dtype=wp.vec3, device=device)
                        params = wp.array([[0.0, softness, 1.0 / mass, 0.0]] * rows, dtype=wp.vec4, device=device)
                        impulses = wp.array([[0.0, 0.0, 1.0]] * rows, dtype=wp.vec3, device=device)
                        matrix = wp.empty(1, dtype=response.mat66, device=device)
                        wp.launch(
                            response.build_response,
                            dim=1,
                            inputs=[
                                enabled,
                                pose,
                                inverse_mass,
                                inverse_inertia,
                                count,
                                bodies,
                                radius,
                                radius,
                                normal,
                                params,
                                impulses,
                            ],
                            outputs=[matrix],
                            device=device,
                        )
                        result = matrix.numpy()[0]
                        expected = softness / (mass * softness + rows)
                        self.assertAlmostEqual(float(result[2, 2]), expected, delta=max(1.0e-6, expected * 1.0e-5))
                        self.assertAlmostEqual(float(result[0, 0]), 1.0 / mass, delta=1.0e-4)
                        np.testing.assert_allclose(result[3:, 3:], np.eye(3), atol=1.0e-6)

    def test_compliant_contact_balances_weight_with_indentation(self):
        """Balance gravity with shape stiffness instead of a hard rigid contact constraint."""
        builder = newton.ModelBuilder()
        SolverMJMPM.register_custom_attributes(builder)
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, ke=1.0e4, kd=100.0)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.11), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        builder.add_ground_plane(cfg=cfg)
        model = builder.finalize(device="cpu")
        solver = SolverMJMPM(model, rigid_contact_hard=False)
        state = model.state()
        dt = 1.0 / 240.0
        for _ in range(180):
            solver.step(state, state, None, None, dt)
        weight = float(model.body_mass.numpy()[body]) * 9.81
        # The four bottom vertices each support one compliant contact.
        expected_height = 0.1 - weight / (4 * cfg.ke)
        self.assertAlmostEqual(float(state.body_q.numpy()[body, 2]), expected_height, delta=2.0e-4)
        self.assertAlmostEqual(float(solver.environment.rigid_impulses.numpy()[body, 2]) / dt, weight, delta=0.05)
        self.assertLess(float(np.linalg.norm(state.body_qd.numpy())), 0.001)

    def test_free_fall_rotation_and_reset(self):
        """Advance a free body without constructing MuJoCo and restore its initial state."""
        builder = newton.ModelBuilder()
        SolverMJMPM.register_custom_attributes(builder)
        body = _box(builder, (0.0, 0.0, 1.0))
        model = builder.finalize(device="cpu")
        solver = SolverMJMPM(model)
        self.assertEqual(solver.features.backend, "pure_mpm")
        self.assertEqual(solver.features.mujoco_body_count, 0)
        self.assertEqual(solver.ownership.mpm_bodies, (body,))
        state = model.state()
        velocities = state.body_qd.numpy()
        velocities[body, 5] = 2.0
        state.body_qd.assign(velocities)
        dt = 0.01
        for _ in range(10):
            solver.step(state, state, None, None, dt)
        expected_z = 1.0 - 9.81 * dt**2 * 10 * 11 / 2
        self.assertAlmostEqual(float(state.body_q.numpy()[body, 2]), expected_z, delta=2.0e-6)
        self.assertAlmostEqual(float(state.body_qd.numpy()[body, 5]), 2.0, delta=1.0e-5)
        self.assertGreater(abs(float(state.body_q.numpy()[body, 5])), 0.09)
        solver.reset(state)
        np.testing.assert_array_equal(state.body_q.numpy(), model.body_q.numpy())
        np.testing.assert_array_equal(state.body_qd.numpy(), model.body_qd.numpy())

    def test_boxes_stack_on_plane(self):
        """Stack two falling rigid boxes on a plane in the MPM-owned domain."""
        builder = newton.ModelBuilder()
        SolverMJMPM.register_custom_attributes(builder)
        body = _box(builder, (0.0, 0.0, 0.3))
        upper = _box(builder, (0.0, 0.0, 0.6))
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        solver = SolverMJMPM(model)
        state = model.state()
        for _ in range(180):
            solver.step(state, state, None, None, 1.0 / 240.0)
        self.assertAlmostEqual(float(state.body_q.numpy()[body, 2]), 0.1, delta=0.004)
        self.assertAlmostEqual(float(state.body_q.numpy()[upper, 2]), 0.3, delta=0.004)
        self.assertLess(float(np.linalg.norm(state.body_qd.numpy())), 0.02)

    def test_warm_started_friction_holds_pushed_box(self):
        """Hold a box against a sideways push below the friction limit with two sweeps per step.

        Restarting the contact impulses from zero each step leaves the friction
        short of the push after two sweeps, and the box creeps (about 2 mm in
        1.5 s); warm starts converge the friction over steps instead.
        """
        builder = newton.ModelBuilder()
        SolverMJMPM.register_custom_attributes(builder)
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5)
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        builder.add_ground_plane(cfg=cfg)
        model = builder.finalize(device="cpu")
        solver = SolverMJMPM(model, rigid_iterations=2, rigid_warm_start=True)
        state = model.state()
        dt = 1.0 / 240.0
        for _ in range(120):
            solver.step(state, state, None, None, dt)
        start = float(state.body_q.numpy()[body, 0])
        forces = state.body_f.numpy()
        forces[body, 0] = 0.3 * float(model.body_mass.numpy()[body]) * 9.81
        state.body_f.assign(forces)
        for _ in range(360):
            solver.step(state, state, None, None, dt)
        self.assertLess(abs(float(state.body_q.numpy()[body, 0]) - start), 5.0e-5)
        self.assertLess(float(np.linalg.norm(state.body_qd.numpy()[body])), 1.0e-5)

    def test_jacobi_rigid_contacts_hold_and_stack(self):
        """Converge parallel mass-split contact sweeps to resting and sticking contact.

        Mass splitting converges more slowly than sequential sweeps: this stack
        needs about 40 sweeps per step, where Gauss-Seidel holds with 8.
        """
        builder = newton.ModelBuilder()
        SolverMJMPM.register_custom_attributes(builder)
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5)
        lower = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(lower, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        upper = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()))
        builder.add_shape_box(upper, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        builder.add_ground_plane(cfg=cfg)
        model = builder.finalize(device="cpu")
        solver = SolverMJMPM(model, rigid_iterations=40, rigid_warm_start=True, rigid_solver="jacobi")
        state = model.state()
        dt = 1.0 / 240.0
        for _ in range(120):
            solver.step(state, state, None, None, dt)
        start = state.body_q.numpy()[:, :3].copy()
        # Push the stack sideways below the friction limit of the ground contact.
        forces = state.body_f.numpy()
        forces[lower, 0] = 0.3 * float(model.body_mass.numpy().sum()) * 9.81
        state.body_f.assign(forces)
        for _ in range(240):
            solver.step(state, state, None, None, dt)
        np.testing.assert_allclose(state.body_q.numpy()[:, :3], start, atol=2.0e-4)
        np.testing.assert_allclose(start[:, 2], (0.1, 0.3), atol=2.0e-3)
        self.assertLess(float(np.linalg.norm(state.body_qd.numpy())), 1.0e-3)

    def test_two_body_collision_conserves_momentum(self):
        """Exchange equal/opposite rigid contact impulses without a MuJoCo solver."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMJMPM.register_custom_attributes(builder)
        a = _box(builder, (-0.105, 0.0, 0.0))
        b = _box(builder, (0.105, 0.0, 0.0))
        model = builder.finalize(device="cpu")
        solver = SolverMJMPM(model)
        state = model.state()
        velocity = state.body_qd.numpy()
        velocity[a, 0] = 1.0
        state.body_qd.assign(velocity)
        masses = model.body_mass.numpy()
        initial = np.sum(masses[:, None] * velocity[:, :3], axis=0)
        for _ in range(20):
            solver.step(state, state, None, None, 0.002)
        velocity = state.body_qd.numpy()
        np.testing.assert_allclose(np.sum(masses[:, None] * velocity[:, :3], axis=0), initial, atol=2.0e-6)
        self.assertGreater(float(velocity[b, 0]), 0.1)
        self.assertGreater(float(state.body_q.numpy()[b, 0] - state.body_q.numpy()[a, 0]), 0.195)

    def test_masked_rigid_reset_preserves_other_world(self):
        """Reset one world's rigid state while retaining the other world's trajectory."""
        local = newton.ModelBuilder()
        _box(local, (0.0, 0.0, 1.0))
        builder = newton.ModelBuilder()
        SolverMJMPM.register_custom_attributes(builder)
        builder.add_world(local)
        builder.add_world(local, xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()))
        model = builder.finalize(device="cpu")
        solver = SolverMJMPM(model)
        state = model.state()
        solver.step(state, state, None, None, 0.01)
        previous = state.body_q.numpy().copy()
        solver.reset(state, world_mask=wp.array((True, False, False), dtype=wp.bool, device="cpu"))
        actual = state.body_q.numpy()
        np.testing.assert_array_equal(actual[0], model.body_q.numpy()[0])
        np.testing.assert_array_equal(actual[1], previous[1])

    def test_world_parallel_contacts_match_sequential_reference(self):
        """Reproduce the single-threaded contact sweeps when each world runs on its own thread."""
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            for hard in (True, False):
                with self.subTest(device=device, hard=hard):
                    local = newton.ModelBuilder()
                    cfg = newton.ModelBuilder.ShapeConfig(density=500.0, ke=2.0e4, kd=50.0, mu=0.6)
                    lower = local.add_body(
                        xform=wp.transform(wp.vec3(0.0, 0.0, 0.11), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.3))
                    )
                    local.add_shape_box(lower, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
                    upper = local.add_body(xform=wp.transform(wp.vec3(0.03, 0.01, 0.32), wp.quat_identity()))
                    local.add_shape_box(upper, hx=0.08, hy=0.08, hz=0.08, cfg=cfg)
                    builder = newton.ModelBuilder()
                    SolverMJMPM.register_custom_attributes(builder)
                    for world in range(3):
                        builder.add_world(local, xform=wp.transform(wp.vec3(world * 1.0, 0.0, 0.0), wp.quat_identity()))
                    builder.add_ground_plane(cfg=cfg)
                    model = builder.finalize(device=device)
                    states = []
                    for reference in (True, False):
                        solver = SolverMJMPM(model, rigid_contact_hard=hard)
                        env = solver.environment
                        self.assertFalse(env._single_bucket)
                        env._use_reference_rigid_solve = reference
                        state = model.state()
                        velocity = state.body_qd.numpy()
                        velocity[:, 0] = np.linspace(-0.4, 0.4, model.body_count)
                        velocity[:, 5] = np.linspace(1.0, -1.0, model.body_count)
                        state.body_qd.assign(velocity)
                        for _ in range(40):
                            solver.step(state, state, None, None, 1.0 / 240.0)
                        states.append((state.body_q.numpy(), state.body_qd.numpy(), env.rigid_impulses.numpy()))
                    self.assertGreater(float(np.abs(states[0][2]).max()), 1.0e-3)
                    # CUDA may contract the two kernels' arithmetic into different
                    # fused multiply-adds; sliding stacks amplify that rounding.
                    atol = 1.0e-5 if device == "cpu" else 1.0e-4
                    for expected, actual in zip(states[0], states[1], strict=True):
                        np.testing.assert_allclose(actual, expected, atol=atol)

    def test_local_contact_buckets_match_reference(self):
        """Reproduce the reference sweeps bitwise on bucket-local body state, and fall back when crowded."""

        def stack(worlds, boxes, device):
            local = newton.ModelBuilder()
            cfg = newton.ModelBuilder.ShapeConfig(density=500.0, ke=2.0e4, kd=50.0, mu=0.6)
            for i in range(boxes):
                body = local.add_body(
                    xform=wp.transform(
                        wp.vec3(0.03 * (i % 2), 0.25 * (i // 6), 0.06 + 0.11 * (i % 6)),
                        wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.2 * i),
                    )
                )
                local.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
            builder = newton.ModelBuilder()
            SolverMJMPM.register_custom_attributes(builder)
            for world in range(worlds):
                builder.add_world(local, xform=wp.transform(wp.vec3(world * 2.0, 0.0, 0.0), wp.quat_identity()))
            builder.add_ground_plane(cfg=cfg)
            return builder.finalize(device=device)

        def run(model, steps, dt, **kwargs):
            # Contact order from CUDA detection varies between runs, so compare
            # the two sweeps on the same inputs within each solve.
            solver = SolverMJMPM(model, **kwargs)
            env = solver.environment
            sweep = env._solve_contacts_gauss_seidel
            outputs = (env.contact_impulses, env.rigid_impulses)
            stats = {"solves": 0, "fallback": 0, "impulse": 0.0}

            def compare(work):
                saved = [wp.clone(a) for a in (work.body_qd, *outputs)]
                env._use_local_rigid_buckets = False
                sweep(work)
                expected = [a.numpy() for a in (work.body_qd, *outputs)]
                for array, value in zip((work.body_qd, *outputs), saved, strict=True):
                    array.assign(value)
                env._use_local_rigid_buckets = True
                sweep(work)
                for array, value in zip((work.body_qd, *outputs), expected, strict=True):
                    np.testing.assert_array_equal(array.numpy(), value)
                stats["solves"] += 1
                stats["fallback"] = max(stats["fallback"], int(env._bucket_fallback.numpy().max()))
                stats["impulse"] = max(stats["impulse"], float(np.abs(expected[1]).max()))

            env._solve_contacts_gauss_seidel = compare
            state = model.state()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            for _ in range(steps):
                solver.step(state, state, None, None, dt)
            self.assertGreater(stats["solves"], 0)
            self.assertGreater(stats["impulse"], 1.0e-4)
            return stats["fallback"]

        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            for worlds, boxes, crowded in ((1, 4, False), (3, 4, False), (1, 40, True)):
                for hard in (True, False):
                    with self.subTest(device=device, worlds=worlds, boxes=boxes, hard=hard):
                        fallback = run(stack(worlds, boxes, device), 40, 1.0 / 240.0, rigid_contact_hard=hard)
                        self.assertEqual(bool(fallback), crowded)
            with self.subTest(device=device, articulated=True):
                model, _box_body = _arm_and_box(device=device)
                fallback = run(model, 40, 0.001, relaxation=0.5, articulated_response=True)
                self.assertEqual(fallback, 0)


class TestMJMPMMaterial(unittest.TestCase):
    def test_rigid_coupling_assembly_matches_triplets(self):
        """Preserve sorted blocks, zero pruning, and operator values as captured contacts change."""
        import warp.sparse as sparse  # noqa: PLC0415

        from newton._src.solvers.mjmpm.mpm import rasterized_collisions as rc  # noqa: PLC0415

        limit = rc._RIGID_ASSEMBLY_MAX_WORK
        for device in ("cpu", "cuda:0"):
            if device == "cuda:0" and not wp.is_cuda_available():
                continue
            with self.subTest(device=device), wp.ScopedDevice(device):
                collider = rc.MJMPMCollider()
                collider.collider_body_index = wp.array([0, 1, 2, -1], dtype=int)
                collider.collider_particle_offsets = wp.zeros(5, dtype=int)
                collider.collider_particle_ids = wp.empty(0, dtype=int)
                collider.body_com = wp.array([(0, 0, 0), (0.01, 0.02, 0.03), (0, 0, 0)], dtype=wp.vec3)
                poses = [
                    wp.transform_identity(),
                    wp.transform(wp.vec3(-0.2, 0.3, 0.1), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.7)),
                    wp.transform_identity(),
                ]
                body_q = wp.array(poses, dtype=wp.transform)
                mass = wp.array([0.0016, 2.0, 0.0], dtype=float)
                inverse = wp.array([np.diag([3, 4, 5]), np.zeros((3, 3)), np.eye(3)], dtype=wp.mat33)
                center = np.asarray(wp.transform_point(poses[1], wp.vec3(0.01, 0.02, 0.03)))
                points = wp.array(
                    [
                        (0, 0, 0),
                        (float(np.nextafter(np.float32(0), np.float32(1))), 0, 0),
                        (0, 0.03, 0),
                        center,
                        np.add(center, (0.02, -0.05, 0.1)),
                        (0, 0, 0),
                        (0.2, 0.1, 0.4),
                        (-0.1, 0, 0.2),
                    ],
                    dtype=wp.vec3,
                )
                ids = wp.array([0, 0, 0, 1, 1, 2, 3, -1], dtype=int)
                original_ids = ids.numpy()
                volumes = wp.ones(8, dtype=float)
                particles = wp.empty(0, dtype=float)

                data = [volumes, points, collider, body_q, mass, inverse, particles, ids]

                def build(cell_volume, data=data):
                    return rc.build_rigidity_operator(cell_volume, *data)

                def reference(cell_volume, build=build):
                    rc._RIGID_ASSEMBLY_MAX_WORK = 0
                    try:
                        return build(cell_volume)
                    finally:
                        rc._RIGID_ASSEMBLY_MAX_WORK = limit

                def compare(actual, expected):
                    for a, b in zip(actual, expected, strict=True):
                        offsets = b.offsets.numpy()
                        count = int(offsets[-1])
                        np.testing.assert_array_equal(a.offsets.numpy(), offsets)
                        np.testing.assert_array_equal(a.columns.numpy()[:count], b.columns.numpy()[:count])
                        np.testing.assert_array_equal(
                            a.values.numpy()[:count].view(np.uint32), b.values.numpy()[:count].view(np.uint32)
                        )
                        x = wp.array(np.arange(b.ncol * 3, dtype=np.float32).reshape(-1, 3), dtype=wp.vec3)
                        np.testing.assert_array_equal(sparse.bsr_mv(a, x=x).numpy(), sparse.bsr_mv(b, x=x).numpy())

                for cell_volume in (0.001, 0.0):
                    compare(build(cell_volume), reference(cell_volume))
                mass.assign(np.array([1e15, 2.0, 0.0], dtype=np.float32))
                compare(build(0.001), reference(0.001))
                mass.assign(np.array([0.0016, 2.0, 0.0], dtype=np.float32))
                if device == "cuda:0":
                    with wp.ScopedCapture(force_module_load=False) as capture:
                        captured = build(0.001)
                    for topology in (np.full(8, -1), original_ids, np.array([0, -1, -1, -1, -1, -1, -1, -1])):
                        ids.assign(topology.astype(np.int32))
                        wp.capture_launch(capture.graph)
                        compare(captured, reference(0.001))

                # Exercise both sides of each bounded fast-path dimension.
                saved_data = data.copy()
                saved_com = collider.body_com
                for node_count, body_count in ((0, 3), (8, 64), (8, 65), (4096, 3), (4097, 3), (2048, 64), (2049, 64)):
                    data[0] = wp.ones(node_count, dtype=float)
                    data[1] = wp.full(node_count, wp.vec3(0.03, 0.02, 0.01), dtype=wp.vec3)
                    data[3] = wp.array(poses + [wp.transform_identity()] * (body_count - 3), dtype=wp.transform)
                    data[4] = wp.array([0.0016, 2.0, 0.0] + [0.0] * (body_count - 3), dtype=float)
                    data[5] = wp.array(
                        np.concatenate((inverse.numpy(), np.zeros((body_count - 3, 3, 3)))), dtype=wp.mat33
                    )
                    data[7] = wp.zeros(node_count, dtype=int)
                    collider.body_com = wp.array(
                        np.concatenate((saved_com.numpy(), np.zeros((body_count - 3, 3)))), dtype=wp.vec3
                    )
                    compare(build(0.001), reference(0.001))
                data[:] = saved_data
                collider.body_com = saved_com

                # A deformable collider must keep the generic barycentric path.
                mesh = wp.Mesh(
                    points=wp.array([(0, 0, 0), (1, 0, 0), (0, 1, 0)], dtype=wp.vec3),
                    indices=wp.array([0, 1, 2], dtype=int),
                )
                collider.collider_body_index = wp.array([-1], dtype=int)
                collider.collider_particle_offsets = wp.array([0, 3], dtype=int)
                collider.collider_particle_ids = wp.array([0, 1, 2], dtype=int)
                collider.collider_mesh = wp.array([mesh.id], dtype=wp.uint64)
                collider.collider_max_thickness = wp.zeros(1, dtype=float)
                collider.query_max_dist = 1.0
                data[6] = wp.array([1.0, 0.0, 2.0], dtype=float)
                ids.fill_(0)
                compare(build(0.001), reference(0.001))

    def test_free_material_crosses_grid_planes_without_air_drag(self):
        """Keep elastic free fall finite when grid-aligned particles leave empty grid nodes."""
        builder = newton.ModelBuilder()
        SolverMJMPM.register_custom_attributes(builder)
        builder.add_particle_grid(
            pos=wp.vec3(0.405, -0.275, 0.867),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=6,
            dim_y=6,
            dim_z=8,
            cell_x=0.01,
            cell_y=0.01,
            cell_z=0.01,
            mass=0.00025,
            radius_mean=0.005,
            jitter=0.0,
            custom_attributes={
                "mpm:young_modulus": 1.6285714e4,
                "mpm:poisson_ratio": 0.35714286,
                "mpm:yield_stress": 1.0e6,
                "mpm:tensile_yield_ratio": 1.0,
            },
        )
        model = builder.finalize(device="cuda:0" if wp.is_cuda_available() else "cpu")
        solver = SolverMJMPM(
            model,
            mpm_options={
                "grid_type": "fixed",
                "voxel_size": 0.015,
                "grid_padding": 3,
                "max_active_cell_count": 512,
                "max_iterations": 50,
                "solver": "gs",
                "air_drag": 0.0,
            },
        )
        state = model.state()
        initial = state.particle_q.numpy().copy()
        dt = 1.0 / 600.0
        for step in range(10):
            solver.step(state, state, None, None, dt)
            self.assertTrue(np.isfinite(state.particle_q.numpy()).all(), f"Nonfinite material on step {step}")
        expected = initial.copy()
        expected[:, 2] -= 9.81 * dt**2 * 10 * 11 / 2
        # Internal stress must not change center-of-mass free fall. Individual
        # boundary nodes are less well conditioned in this float32 fine grid.
        np.testing.assert_allclose(
            state.particle_q.numpy().mean(axis=0, dtype=np.float64),
            expected.mean(axis=0, dtype=np.float64),
            atol=3.0e-6,
        )
        np.testing.assert_allclose(state.particle_q.numpy(), expected, atol=5.0e-5)

    def test_coupled_material_retains_history_between_steps(self):
        """Retain elastic history across articulation/MPM state distribution and reconciliation."""
        for in_place in (True, False):
            with self.subTest(in_place=in_place):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                SolverMJMPM.register_custom_attributes(builder)
                pose = wp.transform(wp.vec3(5.0, 0.0, 0.0), wp.quat_identity())
                link = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
                builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
                joint = builder.add_joint_revolute(parent=-1, child=link, axis=(0.0, 0.0, 1.0), parent_xform=pose)
                builder.add_articulation([joint])
                _particles(builder)
                model = builder.finalize(device="cpu")
                options = {**_options(), "max_iterations": 50}
                solver = SolverMJMPM(model, mpm_options=options, mujoco_options={"use_mujoco_cpu": True})
                config = SolverImplicitMPM.Config()
                for key, value in options.items():
                    setattr(config, key, value)
                reference = SolverImplicitMPM(model, config=config)
                state, expected = model.state(), model.state()
                output = state if in_place else model.state()
                velocity = state.particle_q.numpy() * np.array([-0.2, -0.1, -0.05], dtype=np.float32)
                state.particle_qd.assign(velocity)
                expected.particle_qd.assign(velocity)
                for step in range(3):
                    solver.step(state, output, None, None, 0.005)
                    state, output = output, state
                    reference.step(expected, expected, None, None, 0.005)
                    entry = solver.backend._entries["mpm"]
                    for name in ("particle_elastic_strain", "particle_stress", "particle_Jp", "particle_qd_grad"):
                        actual = getattr(state.mpm, name).numpy()
                        np.testing.assert_array_equal(
                            actual, getattr(entry.state_1.mpm, name).numpy(), err_msg=f"{name} was lost on step {step}"
                        )
                        np.testing.assert_allclose(
                            actual, getattr(expected.mpm, name).numpy(), rtol=1.0e-6, atol=1.0e-6
                        )
                    np.testing.assert_allclose(state.particle_q.numpy(), expected.particle_q.numpy(), atol=1.0e-7)
                self.assertGreater(float(np.linalg.norm(state.mpm.particle_stress.numpy())), 1.0e-5)

    def test_private_material_matches_reference(self):
        """Preserve implicit MPM particle and material history during private source migration."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMJMPM.register_custom_attributes(builder)
        _particles(builder, velocity=(0.1, -0.2, 0.0))
        model = builder.finalize(device="cpu")
        options = {**_options(), "max_iterations": 50}
        solver = SolverMJMPM(model, mpm_options=options)
        config = SolverImplicitMPM.Config()
        for name, value in options.items():
            setattr(config, name, value)
        reference = SolverImplicitMPM(model, config=config)
        actual, expected = model.state(), model.state()
        velocity = actual.particle_q.numpy() * np.array([-0.2, -0.1, -0.05], dtype=np.float32)
        actual.particle_qd.assign(velocity)
        expected.particle_qd.assign(velocity)
        for _ in range(3):
            solver.step(actual, actual, None, None, 0.005)
            reference.step(expected, expected, None, None, 0.005)
        for name in ("particle_q", "particle_qd"):
            np.testing.assert_allclose(getattr(actual, name).numpy(), getattr(expected, name).numpy(), atol=1.0e-7)
        for name in ("particle_elastic_strain", "particle_stress", "particle_Jp", "particle_qd_grad"):
            np.testing.assert_allclose(
                getattr(actual.mpm, name).numpy(), getattr(expected.mpm, name).numpy(), atol=1.0e-6
            )
        self.assertGreater(float(np.linalg.norm(actual.mpm.particle_stress.numpy())), 1.0e-5)

    def test_material_body_collision_conserves_momentum(self):
        """Apply material contact impulses once to an MPM-owned finite-mass body."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMJMPM.register_custom_attributes(builder)
        body = _box(builder, (0.3, 0.05, 0.05), half=0.2)
        _particles(builder, velocity=(1.0, 0.0, 0.0))
        model = builder.finalize(device="cpu")
        options = {**_options(), "voxel_size": 0.05}
        solver = SolverMJMPM(model, mpm_options=options)
        state = model.state()
        body_mass = model.body_mass.numpy()
        particle_mass = model.particle_mass.numpy()

        def momentum():
            return body_mass @ state.body_qd.numpy()[:, :3] + particle_mass @ state.particle_qd.numpy()

        initial = momentum()
        for _ in range(12):
            solver.step(state, state, None, None, 0.005)
        np.testing.assert_allclose(momentum(), initial, atol=2.0e-4)
        self.assertGreater(float(state.body_qd.numpy()[body, 0]), 1.0e-4)

    def test_lagged_material_coupling_holds_loaded_tray_on_slope(self):
        """Hold a friction-supported tray under a layer of grains with lagged rigid-material coupling.

        With implicit coupling the grains' load reaches the tray after its
        contact solve each step, and the tray creeps down the slope (about
        11 mm in 1 s); lagged coupling applies it before the contact solve.
        """
        model, tray, _joint, downhill = _loaded_tray()
        options = {
            "grid_type": "dense",
            "voxel_size": 0.04,
            "max_iterations": 20,
            "solver": "jacobi",
            "warmstart_mode": "none",
        }
        solver = SolverMJMPM(model, rigid_material_coupling="lagged", mpm_options=options)
        state = model.state()
        start = state.body_q.numpy()[tray, :3].copy()
        for _ in range(240):
            solver.step(state, state, None, None, 1.0 / 240.0)
        # Motion along the slope; the tray settles about 3 mm normal to it as the grains land.
        self.assertLess(abs(float((state.body_q.numpy()[tray, :3] - start) @ downhill)), 2.0e-4)
        self.assertLess(float(np.linalg.norm(state.body_qd.numpy()[tray])), 1.0e-3)
        self.assertTrue(np.isfinite(state.particle_q.numpy()).all())

    def test_constrained_material_coupling_holds_loaded_tray_on_slope(self):
        """Hold a loaded tray using same-step contact response instead of delaying material impulses."""
        model, tray, _joint, downhill = _loaded_tray()
        options = {
            "grid_type": "dense",
            "voxel_size": 0.04,
            "max_iterations": 20,
            "solver": "jacobi",
            "warmstart_mode": "none",
        }
        solver = SolverMJMPM(model, rigid_material_coupling="constrained", rigid_warm_start=True, mpm_options=options)
        state = model.state()
        start = state.body_q.numpy()[tray, :3].copy()
        for _ in range(240):
            solver.step(state, state, None, None, 1.0 / 240.0)
        # Motion along the slope; the tray settles about 3 mm normal to it as the grains land.
        self.assertLess(abs(float((state.body_q.numpy()[tray, :3] - start) @ downhill)), 2.0e-4)
        self.assertLess(float(np.linalg.norm(state.body_qd.numpy()[tray])), 1.0e-3)
        self.assertTrue(np.isfinite(state.particle_q.numpy()).all())
        self.assertLess(float(np.max(np.abs(solver.environment.material_velocity_residual.numpy()))), 1.0e-5)
        self.assertLess(float(np.max(np.abs(solver.environment.momentum_residual.numpy()))), 1.0e-5)
        self.assertLess(float(np.max(np.abs(solver.environment.constraint_velocity_residual.numpy()))), 1.0e-4)

    def test_stall_exit_keeps_resting_material(self):
        """End stalled Jacobi solves early without moving resting material."""

        def run(stall_tolerance):
            builder = newton.ModelBuilder()
            SolverMJMPM.register_custom_attributes(builder)
            builder.add_ground_plane()
            _particles(builder, position=(0.025, 0.025, 0.026))
            model = builder.finalize(device="cpu")
            options = {**_options(), "max_iterations": 60, "tolerance": 1.0e-6, "stall_tolerance": stall_tolerance}
            solver = SolverMJMPM(model, mpm_options=options)
            state = model.state()
            for _ in range(40):
                solver.step(state, state, None, None, 0.005)
            return state

        full, stalled = run(0.0), run(1.0e-2)
        np.testing.assert_allclose(stalled.particle_q.numpy(), full.particle_q.numpy(), atol=1.0e-4)
        self.assertLess(float(np.abs(stalled.particle_qd.numpy()).max()), 0.05)

    def test_parallel_collider_query_matches_sequential(self):
        """Query node-collider pairs in parallel without changing any rasterized or stepped value."""

        def run(parallel):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            SolverMJMPM.register_custom_attributes(builder)
            # Four spinning boxes close in on the particle block from its sides,
            # so several moving colliders compete for the same samples.
            sides = ((1, 0), (-1, 0), (0, 1), (0, -1))
            for x, y in sides:
                body = builder.add_body(
                    xform=wp.transform(
                        wp.vec3(0.05 + 0.09 * x, 0.05 + 0.09 * y, 0.05),
                        wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.3 * (x + 2 * y)),
                    )
                )
                builder.add_shape_box(body, hx=0.04, hy=0.04, hz=0.06)
            builder.add_ground_plane()
            _particles(builder)
            model = builder.finalize(device="cpu")
            solver = SolverMJMPM(model, mpm_options={**_options(), "voxel_size": 0.05})
            mpm_model = solver.environment.material._mpm_model
            self.assertGreater(mpm_model.max_collider_query_count, 1)
            if not parallel:
                mpm_model.max_collider_query_count = 0
            state = model.state()
            velocity = np.zeros((model.body_count, 6), dtype=np.float32)
            for body, (x, y) in enumerate(sides):
                velocity[body] = (-0.5 * x, -0.5 * y, 0.0, 0.0, 0.0, 3.0)
            state.body_qd.assign(velocity)
            for _ in range(12):
                solver.step(state, state, None, None, 0.005)
            return state

        sequential, parallel = run(False), run(True)
        for name in ("particle_q", "particle_qd", "body_q", "body_qd"):
            np.testing.assert_array_equal(getattr(parallel, name).numpy(), getattr(sequential, name).numpy())
        # The boxes reached the material and pushed it.
        self.assertGreater(float(np.abs(sequential.particle_qd.numpy()).max()), 0.05)

    def test_collider_bound_culling_preserves_rasterization(self):
        """Skip only colliders beyond their query distance, leaving sampled SDF data unchanged."""
        from newton._src.solvers.mjmpm.mpm.rasterized_collisions import (  # noqa: PLC0415
            _ALL_COLLIDER_WORLDS,
            MJMPMCollider,
            collision_sdf,
        )

        all_worlds = _ALL_COLLIDER_WORLDS

        @wp.kernel
        def sample(
            points: wp.array[wp.vec3],
            collider: MJMPMCollider,
            body_q: wp.array[wp.transform],
            body_qd: wp.array[wp.spatial_vector],
            body_q_prev: wp.array[wp.transform],
            sdf: wp.array[float],
            grad: wp.array[wp.vec3],
            vel: wp.array[wp.vec3],
            ids: wp.array[int],
        ):
            i = wp.tid()
            d, g, v, collider_id, _material = collision_sdf(
                points[i], all_worlds, collider, body_q, body_qd, body_q_prev, 0.01
            )
            sdf[i] = d
            grad[i] = g
            vel[i] = v
            ids[i] = collider_id

        builder = newton.ModelBuilder()
        SolverMJMPM.register_custom_attributes(builder)
        rng = np.random.default_rng(3)
        for k in range(6):
            axis = rng.normal(size=3)
            body = builder.add_body(
                xform=wp.transform(
                    wp.vec3(*rng.uniform(-0.4, 0.4, size=3)),
                    wp.quat_from_axis_angle(wp.vec3(*(axis / np.linalg.norm(axis))), float(rng.uniform(0, 3))),
                )
            )
            builder.add_shape_box(body, hx=0.05 + 0.02 * k, hy=0.04, hz=0.03)
            builder.add_shape_sphere(body, xform=wp.transform(wp.vec3(0.06, 0.0, 0.0), wp.quat_identity()), radius=0.03)
        builder.add_ground_plane()
        _particles(builder)
        model = builder.finalize(device="cpu")
        solver = SolverMJMPM(model, mpm_options={**_options(), "voxel_size": 0.05})
        mpm_model = solver.environment.material._mpm_model
        collider = mpm_model.collider
        state = model.state()
        state.body_qd.assign(rng.normal(size=(model.body_count, 6)).astype(np.float32))
        # The collider arrays live on the model device; keep every array and the launch there.
        device = model.device
        points = wp.array(rng.uniform(-0.7, 0.7, size=(20000, 3)).astype(np.float32), dtype=wp.vec3, device=device)

        def evaluate():
            out = (
                wp.zeros(points.shape[0], dtype=float, device=device),
                wp.zeros(points.shape[0], dtype=wp.vec3, device=device),
                wp.zeros(points.shape[0], dtype=wp.vec3, device=device),
                wp.zeros(points.shape[0], dtype=int, device=device),
            )
            wp.launch(
                sample,
                dim=points.shape[0],
                inputs=[points, collider, state.body_q, state.body_qd, None],
                outputs=out,
                device=device,
            )
            return [array.numpy() for array in out]

        culled = evaluate()
        lower, upper = collider.collider_aabb_lower, collider.collider_aabb_upper
        unbounded = np.finfo(np.float32).max
        self.assertTrue(np.isfinite(lower.numpy()).all() and (np.abs(lower.numpy()) < unbounded).all())
        collider.collider_aabb_lower = wp.full(lower.shape[0], wp.vec3(-unbounded), dtype=wp.vec3, device=device)
        collider.collider_aabb_upper = wp.full(upper.shape[0], wp.vec3(unbounded), dtype=wp.vec3, device=device)
        exhaustive = evaluate()
        self.assertGreater(int(np.count_nonzero(exhaustive[3] >= 0)), 1000)
        for actual, expected in zip(culled, exhaustive, strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_point_basis_capacity_reports_crowded_cells(self):
        """Grow a plain particle basis to cover crowded cells and flag an explicit basis that cannot."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMJMPM.register_custom_attributes(builder)
        # Eight particles share each 0.1 m cell.
        _particles(builder)
        model = builder.finalize(device="cpu")
        adaptive = SolverMJMPM(model, mpm_options=_options())
        explicit = SolverMJMPM(model, mpm_options={**_options(), "collider_basis": "pic4"})
        # Construction sizes the capacity from the initial particles.
        self.assertEqual(adaptive.environment.material.point_basis_capacity(), 16)
        for solver in (adaptive, explicit):
            state = model.state()
            solver.step(state, state, None, None, 0.005)
        material = adaptive.environment.material
        self.assertEqual(material.point_basis_capacity(), 16)
        adaptive.check_capacity_status()
        with self.assertRaisesRegex(RuntimeError, "held 8 particles"):
            explicit.check_capacity_status()


class TestMJMPMArticulation(unittest.TestCase):
    def test_material_contact_starts_at_particle_surface(self):
        """Transfer no impulse across a visible gap, then react at the particle surface."""
        for voxel_size in (0.075, 0.1):
            for gap in (0.06, 0.02, -0.001):
                with self.subTest(voxel_size=voxel_size, gap=gap):
                    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                    SolverMJMPM.register_custom_attributes(builder)
                    # The upper particles are at z=0.075 with radius 0.025.
                    pose = wp.transform(wp.vec3(0.05, 0.05, 0.125 + gap), wp.quat_identity())
                    link = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
                    builder.add_shape_box(
                        link, hx=0.2, hy=0.2, hz=0.025, cfg=newton.ModelBuilder.ShapeConfig(density=0.0)
                    )
                    joint = builder.add_joint_prismatic(parent=-1, child=link, axis=(0.0, 0.0, 1.0), parent_xform=pose)
                    builder.add_articulation([joint])
                    builder.joint_qd[0] = -0.2
                    _particles(builder)
                    model = builder.finalize(device="cpu")
                    solver = SolverMJMPM(
                        model,
                        mpm_options={
                            "grid_type": "fixed",
                            "voxel_size": voxel_size,
                            "grid_padding": 3,
                            "max_active_cell_count": 64,
                            "max_iterations": 20,
                            "solver": "jacobi",
                        },
                        mujoco_options={"use_mujoco_cpu": True},
                    )
                    state = model.state()
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    solver.step(state, state, None, None, 0.001)
                    impulse = solver.environment.material_impulses.numpy()[0]
                    velocity = state.particle_qd.numpy()
                    self.assertTrue(np.isfinite(impulse).all())
                    if gap > 0.0:
                        self.assertLess(float(np.linalg.norm(impulse)), 1.0e-9)
                        # Float32 strain recovery can leave sub-micrometer/s drift.
                        self.assertLess(float(np.abs(velocity).max()), 1.0e-6)
                    else:
                        self.assertGreater(float(impulse[2]), 1.0e-7)
                        self.assertLess(float(velocity[:, 2].min()), -1.0e-6)

    def test_press_example_holds_before_commanded_motion(self):
        """Keep the press still and retain elastic support during the one-second hold."""
        from argparse import Namespace  # noqa: PLC0415

        from newton.examples.mpm.example_mjmpm_press import Example  # noqa: PLC0415
        from newton.viewer import ViewerNull  # noqa: PLC0415

        with wp.ScopedDevice("cpu"):
            example = Example(ViewerNull(), Namespace(coupling="two_way"))
            initial = example.state.body_q.numpy()[0, :3].copy()
            initial_height = float(np.ptp(example.state.particle_q.numpy()[:, 2]))
            for _ in range(60):
                example.step()
                example.test_post_step()
            height = float(np.ptp(example.state.particle_q.numpy()[:, 2]))
            self.assertGreater(height, 0.9 * initial_height)
            self.assertGreater(float(np.linalg.norm(example.state.mpm.particle_stress.numpy())), 1.0)
            displacement = example.state.body_q.numpy()[0, :3] - initial
            self.assertLess(float(np.linalg.norm(displacement)), 1.0e-4)
            self.assertLess(float(np.linalg.norm(example.solver.environment.material_impulses.numpy()[0])), 1.0e-8)

    def test_owns_closed_tree_and_leaves_free_joint_in_mpm(self):
        """Integrate a revolute mechanism and an independent free-joint body in distinct domains."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMJMPM.register_custom_attributes(builder)
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_revolute(parent=-1, child=body, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([joint])
        free = builder.add_link(
            xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=wp.mat33(np.eye(3))
        )
        free_joint = builder.add_joint_free(child=free)
        builder.add_articulation([free_joint])
        builder.add_shape_box(free, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        solver = SolverMJMPM(model, mujoco_options={"use_mujoco_cpu": True})
        self.assertIsInstance(solver, newton.solvers.SolverBase)
        self.assertEqual(solver.ownership.mujoco_joints, (joint,))
        self.assertEqual(solver.ownership.mujoco_bodies, (body,))
        self.assertEqual(solver.ownership.mpm_bodies, (free,))
        state = model.state()
        forces = state.body_f.numpy()
        forces[free, 0] = 2.0
        state.body_f.assign(forces)
        control = model.control()
        control.joint_f.assign(np.array([1.0] + [0.0] * 6, dtype=np.float32))
        solver.step(state, state, control, None, 0.01)
        self.assertGreater(float(state.joint_q.numpy()[0]), 0.0)
        self.assertAlmostEqual(
            float(state.body_qd.numpy()[free, 0]), 0.02 / float(model.body_mass.numpy()[free]), delta=1.0e-6
        )
        np.testing.assert_allclose(state.joint_q.numpy()[1:4], state.body_q.numpy()[free, :3], atol=1.0e-6)
        self.assertAlmostEqual(float(state.joint_qd.numpy()[1]), float(state.body_qd.numpy()[free, 0]), delta=1.0e-6)

    def test_environment_rigid_impulses_push_back_articulation(self):
        """Return free-body contact impulses to a moving prismatic link."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMJMPM.register_custom_attributes(builder)
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.01)
        body = builder.add_link(
            xform=wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=wp.mat33(np.eye(3))
        )
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        joint = builder.add_joint_prismatic(
            parent=-1,
            child=body,
            axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([joint])
        builder.joint_qd[0] = 1.0
        free = builder.add_body(
            xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), mass=1.0, inertia=wp.mat33(np.eye(3))
        )
        builder.add_shape_box(free, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        model = builder.finalize(device="cpu")
        states = []
        for mode in (SolverMJMPM.Coupling.ONE_WAY, SolverMJMPM.Coupling.TWO_WAY):
            solver = SolverMJMPM(model, coupling=mode, relaxation=0.5, mujoco_options={"use_mujoco_cpu": True})
            state = model.state()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            for _ in range(150):
                solver.step(state, state, None, None, 0.001)
            states.append(state)
        one_way, two_way = states
        self.assertGreater(float(one_way.joint_qd.numpy()[0]), 0.99)
        self.assertLess(float(two_way.joint_qd.numpy()[0]), 0.8)
        self.assertGreater(float(two_way.body_qd.numpy()[free, 0]), 0.2)
        momentum = float(two_way.joint_qd.numpy()[0] + two_way.body_qd.numpy()[free, 0])
        self.assertAlmostEqual(momentum, 1.0, delta=0.005)

    def test_proxy_static_pairs_are_not_detected(self):
        """Exclude articulation-static pairs from environment detection without changing the motion."""
        builder = newton.ModelBuilder()
        SolverMJMPM.register_custom_attributes(builder)
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5)
        pose = wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity())
        link = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
        link_shape = builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        joint = builder.add_joint_prismatic(parent=-1, child=link, axis=(1.0, 0.0, 0.0), parent_xform=pose)
        builder.add_articulation([joint])
        builder.joint_qd[0] = 0.5
        free = _box(builder, (0.205, 0.0, 0.1))
        ground = builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        results = []
        for filtered in (True, False):
            solver = SolverMJMPM(model, mujoco_options={"use_mujoco_cpu": True})
            env = solver.environment
            if not filtered:
                original = model.shape_collision_filter_pairs_array()
                env.detector.shape_pairs_excluded = wp.array(original, dtype=wp.vec2i, device="cpu")
                env.detector.shape_pairs_excluded_count = original.shape[0]
            state = model.state()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            pairs = set()
            peak_push = 0.0
            for _ in range(60):
                solver.step(state, state, None, None, 0.002)
                peak_push = max(peak_push, float(state.body_qd.numpy()[free, 0]))
                count = int(env.contacts.rigid_contact_count.numpy()[0])
                shape0 = env.contacts.rigid_contact_shape0.numpy()[:count]
                shape1 = env.contacts.rigid_contact_shape1.numpy()[:count]
                pairs |= {(int(min(a, b)), int(max(a, b))) for a, b in zip(shape0, shape1, strict=True)}
            results.append((pairs, peak_push, state.body_q.numpy(), state.body_qd.numpy(), state.joint_qd.numpy()))
        static_pair = (min(link_shape, ground), max(link_shape, ground))
        self.assertNotIn(static_pair, results[0][0])
        self.assertIn(static_pair, results[1][0])
        self.assertGreater(results[0][1], 0.05)
        for expected, actual in zip(results[1][2:], results[0][2:], strict=True):
            np.testing.assert_allclose(actual, expected, atol=1.0e-6)

    def test_material_impulses_push_articulation(self):
        """Return MPM grid impulses to a prismatic joint while preserving one-way behavior."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMJMPM.register_custom_attributes(builder)
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
        link = builder.add_link(
            xform=wp.transform(wp.vec3(0.3, 0.05, 0.05), wp.quat_identity()), mass=1.0, inertia=wp.mat33(np.eye(3))
        )
        builder.add_shape_box(link, hx=0.2, hy=0.2, hz=0.2, cfg=cfg)
        joint = builder.add_joint_prismatic(
            parent=-1,
            child=link,
            axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.3, 0.05, 0.05), wp.quat_identity()),
        )
        builder.add_articulation([joint])
        _particles(builder, velocity=(1.0, 0.0, 0.0))
        model = builder.finalize(device="cpu")
        velocities = []
        for mode in (SolverMJMPM.Coupling.ONE_WAY, SolverMJMPM.Coupling.TWO_WAY):
            solver = SolverMJMPM(
                model,
                coupling=mode,
                mpm_options={**_options(), "voxel_size": 0.05},
                mujoco_options={"use_mujoco_cpu": True},
            )
            state = model.state()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            for _ in range(16):
                solver.step(state, state, None, None, 0.005)
            velocities.append(float(state.joint_qd.numpy()[0]))
        self.assertAlmostEqual(velocities[0], 0.0, delta=1.0e-7)
        self.assertGreater(velocities[1], 0.001)


def _prismatic_collision(free_mass=1.0, device="cpu"):
    """Return a prismatic link moving at 1 m/s toward a resting free box without gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverMJMPM.register_custom_attributes(builder)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.01)
    pose = wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity())
    link = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    joint = builder.add_joint_prismatic(parent=-1, child=link, axis=(1.0, 0.0, 0.0), parent_xform=pose)
    builder.add_articulation([joint])
    builder.joint_qd[0] = 1.0
    free = builder.add_body(
        xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), mass=free_mass, inertia=wp.mat33(np.eye(3))
    )
    builder.add_shape_box(free, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    return builder.finalize(device=device), free


class TestMJMPMSameStep(unittest.TestCase):
    def test_iterations_resolve_contact_within_step(self):
        """Converge link/box contact in the step it begins and exchange equal impulses."""
        outcomes = {}
        for iterations in (1, 8):
            model, free = _prismatic_collision()
            solver = SolverMJMPM(
                model, relaxation=0.5, coupling_iterations=iterations, mujoco_options={"use_mujoco_cpu": True}
            )
            state = model.state()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            contact = None
            for _ in range(150):
                solver.step(state, state, None, None, 0.001)
                residuals = solver.coupling_residuals.numpy().copy()
                if contact is None and residuals.max() > 1.0e-6:
                    contact = (float(state.joint_qd.numpy()[0]), float(state.body_qd.numpy()[free, 0]), residuals)
                momentum = float(state.joint_qd.numpy()[0] + state.body_qd.numpy()[free, 0])
                if iterations > 1:
                    # Both domains apply each step's contact impulse in that step.
                    self.assertAlmostEqual(momentum, 1.0, delta=1.0e-4)
            self.assertAlmostEqual(momentum, 1.0, delta=1.0e-4)
            outcomes[iterations] = contact
        link, box, _residuals = outcomes[1]
        # One pass applies the first contact impulse to the link one step late.
        self.assertAlmostEqual(link, 1.0, delta=1.0e-4)
        self.assertGreater(box, 0.7)
        link, box, residuals = outcomes[8]
        # Equal masses share the momentum in the contact step itself.
        self.assertAlmostEqual(link, 0.5, delta=0.005)
        self.assertAlmostEqual(box, 0.5, delta=0.005)
        self.assertTrue(np.all(np.diff(residuals[:, 0]) < 0.0), residuals[:, 0])
        self.assertLess(float(residuals[-1, 0]), 1.0e-3 * float(residuals[0, 0]))

    def test_repeated_passes_restore_step_history(self):
        """Repeat contact-free passes without accumulating material, rigid, or MuJoCo history."""
        builder = newton.ModelBuilder()
        SolverMJMPM.register_custom_attributes(builder)
        pose = wp.transform(wp.vec3(3.0, 0.0, 1.0), wp.quat_identity())
        link = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_revolute(parent=-1, child=link, axis=(0.0, 1.0, 0.0), parent_xform=pose)
        builder.add_articulation([joint])
        builder.joint_qd[0] = 2.0
        _box(builder, (0.6, 0.0, 0.049), half=0.05)
        builder.add_ground_plane()
        _particles(builder)
        model = builder.finalize(device="cpu")
        options = {**_options(), "max_iterations": 6, "solver": "gs", "warmstart_mode": "auto", "voxel_size": 0.05}
        results = []
        for iterations in (1, 3):
            solver = SolverMJMPM(
                model, coupling_iterations=iterations, mpm_options=options, mujoco_options={"use_mujoco_cpu": True}
            )
            state = model.state()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            for _ in range(12):
                solver.step(state, state, None, None, 0.005)
            self.assertEqual(solver.coupling_passes, iterations)
            results.append(
                [
                    state.joint_q.numpy(),
                    state.joint_qd.numpy(),
                    state.body_q.numpy(),
                    state.body_qd.numpy(),
                    state.particle_q.numpy(),
                    state.particle_qd.numpy(),
                    state.mpm.particle_Jp.numpy(),
                    state.mpm.particle_elastic_strain.numpy(),
                    state.mpm.particle_stress.numpy(),
                ]
            )
        self.assertGreater(float(np.abs(results[0][8]).max()), 1.0)
        for expected, actual in zip(results[0], results[1], strict=True):
            np.testing.assert_array_equal(actual, expected)

    def test_tolerance_stops_converged_steps_early(self):
        """Stop after one pass without contact and iterate while contact feedback changes."""
        model, _free = _prismatic_collision()
        solver = SolverMJMPM(
            model,
            relaxation=0.5,
            coupling_iterations=10,
            coupling_tolerance=1.0e-3,
            mujoco_options={"use_mujoco_cpu": True},
        )
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        passes = []
        for _ in range(60):
            solver.step(state, state, None, None, 0.001)
            passes.append(solver.coupling_passes)
        self.assertEqual(passes[0], 1)
        self.assertGreater(max(passes), 2)
        self.assertLess(max(passes), 10)

    def test_iteration_options_require_two_way_articulation(self):
        """Reject same-step iteration settings that would be ignored."""
        model, _free = _prismatic_collision()
        with self.assertRaisesRegex(ValueError, "two-way"):
            SolverMJMPM(model, coupling=SolverMJMPM.Coupling.ONE_WAY, coupling_iterations=2)
        builder = newton.ModelBuilder()
        SolverMJMPM.register_custom_attributes(builder)
        _box(builder, (0.0, 0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "articulation"):
            SolverMJMPM(builder.finalize(device="cpu"), coupling_iterations=2)
        with self.assertRaisesRegex(ValueError, "update_data_interval"):
            SolverMJMPM(model, coupling_iterations=2, mujoco_options={"update_data_interval": 2})


def _two_link_arm(builder, cfg, *, base=(0.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), length=0.2):
    """Add a two-link revolute arm about z extending along ``axis`` from ``base``."""
    base = np.asarray(base, dtype=float)
    axis = np.asarray(axis, dtype=float)
    half = np.where(axis != 0.0, length, 0.03)
    l1 = builder.add_link(
        xform=wp.transform(wp.vec3(*(base + length * axis)), wp.quat_identity()),
        mass=2.0,
        inertia=wp.mat33(np.diag([0.01, 0.03, 0.03])),
    )
    builder.add_shape_box(l1, hx=half[0], hy=half[1], hz=0.03, cfg=cfg)
    j1 = builder.add_joint_revolute(
        parent=-1,
        child=l1,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(*base), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(*(-length * axis)), wp.quat_identity()),
    )
    l2 = builder.add_link(
        xform=wp.transform(wp.vec3(*(base + 3.0 * length * axis)), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(np.diag([0.005, 0.015, 0.015])),
    )
    builder.add_shape_box(l2, hx=half[0], hy=half[1], hz=0.03, cfg=cfg)
    j2 = builder.add_joint_revolute(
        parent=l1,
        child=l2,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(*(length * axis)), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(*(-length * axis)), wp.quat_identity()),
    )
    builder.add_articulation([j1, j2])


def _arm_and_box(device="cpu"):
    """Return a spinning two-link arm whose outer link strikes a free box without gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverMJMPM.register_custom_attributes(builder)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.3)
    _two_link_arm(builder, cfg)
    builder.joint_qd[0] = 1.5
    box = builder.add_body(
        xform=wp.transform(wp.vec3(0.7, 0.115, 0.0), wp.quat_identity()), mass=1.5, inertia=wp.mat33(np.eye(3) * 0.005)
    )
    builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    return builder.finalize(device=device), box


def _run_coupled(model, steps, dt, record, **kwargs):
    solver = SolverMJMPM(model, relaxation=0.5, **kwargs)
    state = model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    rows = []
    for _ in range(steps):
        solver.step(state, state, None, None, dt)
        rows.append(record(state))
    return np.array(rows)


class TestMJMPMArticulatedResponse(unittest.TestCase):
    def test_mujoco_owned_free_body_material_momentum(self):
        """Transfer material momentum to a MuJoCo-owned free body without double applying feedback."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMJMPM.register_custom_attributes(builder)
        box = builder.add_body(
            xform=wp.transform(wp.vec3(0.3, 0.05, 0.05), wp.quat_identity()),
            mass=0.0016,
            inertia=wp.mat33(np.eye(3) * 1.0e-5),
        )
        joint = builder.joint_count - 1
        builder.add_shape_box(box, hx=0.2, hy=0.2, hz=0.2, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        _particles(builder, velocity=(1.0, 0.0, 0.0))
        model = builder.finalize(device="cpu")
        solver = SolverMJMPM(
            model,
            mujoco_joints=[joint],
            articulated_response=True,
            mpm_options={**_options(), "voxel_size": 0.05, "max_iterations": 20},
        )
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        body_mass, particle_mass = model.body_mass.numpy(), model.particle_mass.numpy()

        def momentum():
            return body_mass @ state.body_qd.numpy()[:, :3] + particle_mass @ state.particle_qd.numpy()

        initial = momentum()
        for _ in range(12):
            solver.step(state, state, None, None, 0.005)
        np.testing.assert_allclose(momentum(), initial, atol=2.0e-6)
        self.assertGreater(float(state.body_qd.numpy()[box, 0]), 0.1)
        self.assertEqual(solver.ownership.mujoco_bodies, (box,))
        self.assertEqual(solver.ownership.mpm_bodies, ())
        proxy = solver.backend._proxy_mappings[0]
        destination = solver.backend._entries["mpm"]
        np.testing.assert_allclose(
            destination.state_1.body_qd.numpy()[proxy.proxy_ids_local.numpy()],
            state.body_qd.numpy()[[box]],
            atol=1.0e-5,
        )

    def test_mujoco_owned_loaded_tray(self):
        """Hold a MuJoCo-owned free tray against the weight of MPM grains."""
        model, tray, joint, downhill = _loaded_tray()
        solver = SolverMJMPM(
            model,
            mujoco_joints=[joint],
            articulated_response=True,
            mujoco_options={"impratio": 100.0},
            mpm_options={**_options(), "voxel_size": 0.04, "max_iterations": 20},
        )
        self.assertEqual(solver.ownership.mujoco_bodies, (tray,))
        self.assertEqual(solver.ownership.mpm_bodies, ())
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        start = state.body_q.numpy()[tray, :3].copy()
        for _ in range(240):
            solver.step(state, state, None, None, 1.0 / 240.0)
        slip = abs(float((state.body_q.numpy()[tray, :3] - start) @ downhill))
        self.assertLess(slip, 2.0e-4)
        self.assertTrue(np.isfinite(state.particle_q.numpy()).all())
        self.assertEqual(int(solver.environment.contacts.rigid_contact_count.numpy()[0]), 0)

    def test_link_mobility_matches_mujoco_step_response(self):
        """Predict every link's one-step twist change for an impulse on any link of a damped chain."""
        from newton._src.solvers.mjmpm.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMuJoCo.register_custom_attributes(builder)
        base = builder.add_link(mass=2.0, inertia=wp.mat33(np.diag([0.02, 0.03, 0.04])))
        builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.05)
        j0 = builder.add_joint_prismatic(parent=-1, child=base, axis=(1.0, 0.0, 0.0))
        l1 = builder.add_link(mass=1.0, inertia=wp.mat33(np.diag([0.01, 0.012, 0.005])))
        builder.add_shape_capsule(l1, radius=0.03, half_height=0.15)
        j1 = builder.add_joint_revolute(
            parent=base,
            child=l1,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.15), wp.quat_identity()),
            target_ke=50.0,
            target_kd=3.0,
        )
        l2 = builder.add_link(mass=0.5, inertia=wp.mat33(np.diag([0.004, 0.006, 0.002])))
        builder.add_shape_capsule(l2, radius=0.02, half_height=0.12)
        j2 = builder.add_joint_revolute(
            parent=l1,
            child=l2,
            axis=(1.0, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.12), wp.quat_identity()),
        )
        builder.add_articulation([j0, j1, j2])
        builder.joint_q[1:3] = [0.4, -0.7]
        builder.joint_qd[:] = [0.3, -0.5, 0.8]
        matrix = self._check_mobility_prediction(builder.finalize(device="cpu"))
        # The coupled chain moves links that were not pushed.
        self.assertGreater(float(np.abs(matrix[0:6, 12:18]).max()), 1.0e-2)

    def test_link_mobility_includes_active_joint_limit(self):
        """Predict the one-step response of a chain whose prismatic base presses into its joint limit."""
        from newton._src.solvers.mjmpm.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMuJoCo.register_custom_attributes(builder)
        base = builder.add_link(mass=2.0, inertia=wp.mat33(np.diag([0.02, 0.03, 0.04])))
        builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.05)
        j0 = builder.add_joint_prismatic(parent=-1, child=base, axis=(1.0, 0.0, 0.0), limit_lower=-0.1, limit_upper=0.1)
        arm = builder.add_link(mass=1.0, inertia=wp.mat33(np.diag([0.01, 0.012, 0.005])))
        builder.add_shape_capsule(arm, radius=0.03, half_height=0.15)
        j1 = builder.add_joint_revolute(
            parent=base,
            child=arm,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.15), wp.quat_identity()),
        )
        builder.add_articulation([j0, j1])
        builder.joint_q[:] = [0.1005, 0.4]
        builder.joint_qd[:] = [0.3, -0.5]
        matrix = self._check_mobility_prediction(builder.finalize(device="cpu"))
        # The limit absorbs most of a push along the slide, but not the arm's swing.
        self.assertLess(matrix[0, 0], 0.2 * (1.0 / 3.0))
        self.assertGreater(matrix[10, 10], 1.0)

    def test_link_mobility_includes_mujoco_contact(self):
        """Predict the one-step response of a chain whose base rests in MuJoCo contact with the ground."""
        from newton._src.solvers.mjmpm.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415

        builder = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(builder)
        builder.add_ground_plane()
        base = builder.add_link(mass=2.0, inertia=wp.mat33(np.diag([0.02, 0.03, 0.04])))
        builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.05)
        j0 = builder.add_joint_prismatic(parent=-1, child=base, axis=(0.0, 0.0, 1.0))
        arm = builder.add_link(mass=1.0, inertia=wp.mat33(np.diag([0.01, 0.012, 0.005])))
        builder.add_shape_capsule(arm, radius=0.03, half_height=0.15)
        j1 = builder.add_joint_revolute(
            parent=base,
            child=arm,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.18), wp.quat_identity()),
        )
        builder.add_articulation([j0, j1])
        builder.joint_q[:] = [0.049, 0.4]
        builder.joint_qd[:] = [-0.2, -0.5]
        matrix = self._check_mobility_prediction(builder.finalize(device="cpu"))
        # The ground carries a vertical push on the base, but not the arm's swing.
        self.assertLess(matrix[2, 2], 0.2 * (1.0 / 3.0))
        self.assertGreater(matrix[10, 10], 1.0)

    def _check_mobility_prediction(self, model):
        """Compare the mobility with MuJoCo's one-step twist change for a wrench on each link."""
        from newton._src.solvers.mjmpm.coupling.articulated import LinkMobility  # noqa: PLC0415
        from newton._src.solvers.mjmpm.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415

        solver = SolverMuJoCo(model, use_mujoco_cpu=False)
        start = model.state()
        newton.eval_fk(model, start.joint_q, start.joint_qd, start)
        control = model.control()
        dt = 0.002

        def step(force=None):
            state_in, state_out = model.state(), model.state()
            state_in.assign(start)
            if force is not None:
                state_in.body_f.assign(force)
            solver.step(state_in, state_out, control, None, dt)
            return state_out.body_qd.numpy()

        free = step()
        mobility = LinkMobility(solver)
        mobility.update()
        matrix = mobility.mobility.numpy()[0]
        links = mobility.link_body.numpy()[0]
        rng = np.random.default_rng(0)
        for k, body in enumerate(links):
            wrench = rng.normal(size=6).astype(np.float32)
            force = np.zeros((model.body_count, 6), dtype=np.float32)
            force[body] = wrench
            actual = step(force) - free
            predicted = np.stack(
                [matrix[6 * l : 6 * l + 6, 6 * k : 6 * k + 6] @ (dt * wrench) for l in range(len(links))]
            )
            # The floor covers float32 differencing of link velocities near 1 m/s.
            np.testing.assert_allclose(predicted, actual[links], atol=max(5.0e-3 * float(np.abs(actual).max()), 2.0e-7))
        return matrix

    def test_single_pass_resolves_heavy_box_collision(self):
        """Stop a light jointed link on a heavy box in the contact step instead of rebounding."""
        outcomes = {}
        for articulated in (False, True):
            model, free = _prismatic_collision(free_mass=8.0)
            rows = _run_coupled(
                model,
                150,
                0.001,
                lambda state, free=free: (float(state.joint_qd.numpy()[0]), float(state.body_qd.numpy()[free, 0])),
                articulated_response=articulated,
            )
            contact = int(np.argmax(np.abs(rows[:, 1]) > 1.0e-4))
            outcomes[articulated] = rows[contact], rows[-1]
        # Independent proxy inertia misses the joint constraint and throws the link back.
        self.assertLess(outcomes[False][1][0], -1.0)
        link, box = outcomes[True][0]
        self.assertAlmostEqual(link, 1.0 / 9.0, delta=0.01)
        self.assertAlmostEqual(box, 1.0 / 9.0, delta=0.01)
        final_link, final_box = outcomes[True][1]
        self.assertAlmostEqual(final_link + 8.0 * final_box, 1.0, delta=1.0e-4)

    def test_single_pass_matches_converged_arm_collision(self):
        """Reproduce the converged same-step solution, including the push transmitted to the inner link."""

        def run(**kwargs):
            model, box = _arm_and_box()
            return _run_coupled(
                model,
                40,
                0.001,
                lambda state, box=box: np.concatenate(
                    (state.joint_qd.numpy()[:2], state.body_qd.numpy()[box, [0, 1, 5]])
                ),
                **kwargs,
            )

        reference = run(coupling_iterations=40)
        articulated = run(articulated_response=True)
        proxy = run()
        self.assertGreater(float(reference[-1, 0]), 1.6)
        self.assertLess(float(np.abs(articulated - reference).max()), 0.02)
        self.assertGreater(float(np.abs(proxy - reference).max()), 1.0)

    def test_single_pass_matches_converged_material_push(self):
        """Move a jointed arm by MPM material contact through the articulated response."""

        def run(**kwargs):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            SolverMJMPM.register_custom_attributes(builder)
            _two_link_arm(
                builder,
                newton.ModelBuilder.ShapeConfig(density=0.0),
                base=(0.2, -0.45, 0.05),
                axis=(0.0, 1.0, 0.0),
            )
            _particles(builder, velocity=(1.0, 0.0, 0.0))
            model = builder.finalize(device="cpu")
            return _run_coupled(
                model,
                30,
                0.005,
                lambda state: state.joint_qd.numpy().copy(),
                mpm_options={**_options(), "voxel_size": 0.05},
                **kwargs,
            )

        reference = run(coupling_iterations=30)
        articulated = run(articulated_response=True)
        proxy = run()
        self.assertGreater(float(np.abs(reference[-1]).max()), 0.1)
        self.assertLess(float(np.abs(articulated - reference).max()), 0.005)
        self.assertGreater(float(np.abs(proxy - reference).max()), 0.1)

    def test_single_pass_matches_converged_impact_on_joint_limit(self):
        """Stop a box dropped on a link that gravity holds against its joint limit, as the converged solution does."""

        def run(**kwargs):
            builder = newton.ModelBuilder()
            SolverMJMPM.register_custom_attributes(builder)
            cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.01)
            link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01))
            builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
            joint = builder.add_joint_prismatic(
                parent=-1, child=link, axis=(0.0, 0.0, 1.0), limit_lower=0.0, limit_upper=1.0
            )
            builder.add_articulation([joint])
            # The soft limit settles before the box arrives.
            box = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(np.eye(3) * 0.01),
            )
            builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
            builder.joint_qd[builder.joint_qd_start[-1] + 2] = -1.0
            model = builder.finalize(device="cpu")
            return _run_coupled(
                model,
                150,
                0.001,
                lambda state, box=box: (float(state.joint_qd.numpy()[0]), float(state.body_qd.numpy()[box, 2])),
                **kwargs,
            )

        reference = run(coupling_iterations=40)
        articulated = run(articulated_response=True)
        contact = int(np.argmax(np.abs(reference[:, 1] + 1.0 + 9.81 * 0.001 * np.arange(1, 151)) > 0.05))
        # The limit absorbs the 1.7 m/s impact without throwing the box back up.
        self.assertLess(float(reference[contact - 1, 1]), -1.5)
        self.assertLess(float(reference[contact:, 1].max()), 0.01)
        # The unconstrained mobility lets the box bounce off at 0.4 m/s.
        self.assertLess(float(np.abs(articulated - reference).max()), 0.01)

    def test_material_push_leaves_environment_links_at_mujoco_velocity(self):
        """End MPM-side link velocities at MuJoCo's through the mobility after material contact."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMJMPM.register_custom_attributes(builder)
        _two_link_arm(
            builder,
            newton.ModelBuilder.ShapeConfig(density=0.0),
            base=(0.2, -0.45, 0.05),
            axis=(0.0, 1.0, 0.0),
        )
        _particles(builder, velocity=(1.0, 0.0, 0.0))
        model = builder.finalize(device="cpu")
        solver = SolverMJMPM(
            model, relaxation=0.5, articulated_response=True, mpm_options={**_options(), "voxel_size": 0.05}
        )
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        pushed = False
        for _ in range(30):
            solver.step(state, state, None, None, 0.005)
            pushed |= float(np.abs(solver.environment.material_impulses.numpy()).max()) > 1.0e-4
            np.testing.assert_allclose(
                solver.environment.work.body_qd.numpy()[:2], state.body_qd.numpy()[:2], atol=1.0e-4
            )
        self.assertTrue(pushed)

    def test_articulated_response_keeps_worlds_independent(self):
        """Evaluate per-world link mobility and contact propagation without cross-world coupling."""
        single, free = _prismatic_collision(free_mass=8.0)
        expected = _run_coupled(
            single,
            80,
            0.001,
            lambda state: np.concatenate((state.joint_qd.numpy()[:1], state.body_qd.numpy()[free, :1])),
            articulated_response=True,
        )
        local = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMJMPM.register_custom_attributes(local)
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.01)
        pose = wp.transform(wp.vec3(-0.15, 0.0, 0.0), wp.quat_identity())
        link = local.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
        local.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        joint = local.add_joint_prismatic(parent=-1, child=link, axis=(1.0, 0.0, 0.0), parent_xform=pose)
        local.add_articulation([joint])
        box = local.add_body(
            xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), mass=8.0, inertia=wp.mat33(np.eye(3))
        )
        local.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMJMPM.register_custom_attributes(builder)
        speeds = (1.0, 0.0, -0.5)
        for world, speed in enumerate(speeds):
            builder.add_world(local, xform=wp.transform(wp.vec3(0.0, 2.0 * world, 0.0), wp.quat_identity()))
            builder.joint_qd[builder.joint_qd_start[-2]] = speed
        model = builder.finalize(device="cpu")
        # Each world holds one prismatic dof followed by the box's free-joint dofs.
        prismatic = model.joint_qd_start.numpy()[:-1][model.joint_type.numpy() == newton.JointType.PRISMATIC]
        boxes = [box + world * local.body_count for world in range(len(speeds))]
        rows = _run_coupled(
            model,
            80,
            0.001,
            lambda state: np.concatenate((state.joint_qd.numpy()[prismatic], state.body_qd.numpy()[boxes, 0])),
            articulated_response=True,
        )
        np.testing.assert_allclose(rows[:, [0, 3]], expected, atol=1.0e-5)
        # The world without motion stays still, and the receding link never touches its box.
        np.testing.assert_allclose(
            rows[:, [1, 2, 4, 5]], 0.0 * rows[:, [1, 2, 4, 5]] + [[0.0, -0.5, 0.0, 0.0]], atol=1.0e-6
        )

    def test_articulated_response_validates_backend(self):
        """Reject articulated response where it cannot be evaluated or would be ignored."""
        model, _free = _prismatic_collision()
        with self.assertRaisesRegex(ValueError, "MuJoCo Warp"):
            SolverMJMPM(model, articulated_response=True, mujoco_options={"use_mujoco_cpu": True})
        with self.assertRaisesRegex(ValueError, "two-way"):
            SolverMJMPM(model, coupling=SolverMJMPM.Coupling.ONE_WAY, articulated_response=True)


@unittest.skipUnless(wp.is_cuda_available(), "CUDA device required")
class TestMJMPMCUDA(unittest.TestCase):
    def test_constrained_light_boxes_on_heavy_material(self):
        """Keep contact-aware 1.6-to-512-gram free boxes stable on five kilograms of material in a CUDA graph."""
        for mass in (0.0016, 0.077, 0.512):
            for drop in (0.0, 0.05):
                with self.subTest(mass=mass, drop=drop):
                    model, box, _joint = _box_on_heavy_material(mass, drop)
                    solver = SolverMJMPM(
                        model,
                        rigid_material_coupling="constrained",
                        rigid_warm_start=True,
                        mpm_options={
                            "grid_type": "fixed",
                            "voxel_size": 0.025,
                            "grid_padding": 12,
                            "max_active_cell_count": 4096,
                            "collider_basis": "pic48",
                            "strain_basis": "P0",
                            "max_iterations": 50,
                            "solver": "gs",
                            "air_drag": 0.0,
                            "tolerance": 0.0,
                        },
                    )
                    state = model.state()
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    dt = 1.0 / 360.0
                    solver.step(state, state, None, None, dt)
                    with wp.ScopedCapture(device=model.device) as capture:
                        for _ in range(6):
                            solver.step(state, state, None, None, dt)
                    peak = 0.0
                    for _ in range(90):
                        wp.capture_launch(capture.graph)
                        velocity = state.body_qd.numpy()[box]
                        self.assertTrue(np.isfinite(velocity).all())
                        self.assertTrue(np.isfinite(state.particle_q.numpy()).all())
                        peak = max(peak, float(np.linalg.norm(velocity[:3])))
                    self.assertLess(peak, 3.0)
                    self.assertGreater(float(state.body_q.numpy()[box, 2]), 0.04)
                    solver.check_capacity_status()

    def test_fixed_grid_graph_matches_eager(self):
        """Replay coupled articulation/rigid/material stepping in a fixed-grid CUDA graph."""
        for collider_basis, iterations in (("pic8", 1), ("pic", 1), ("pic", 3)):
            with self.subTest(collider_basis=collider_basis, iterations=iterations):
                self._check_graph_matches_eager(collider_basis, iterations)

    def test_same_step_iterations_replay_in_graph(self):
        """Capture repeated MuJoCo/MPM passes and their final source solve through contact."""
        model, free = _prismatic_collision(device="cuda:0")
        solvers = [SolverMJMPM(model, relaxation=0.5, coupling_iterations=4) for _ in range(2)]
        states = [model.state(), model.state()]
        for solver, state in zip(solvers, states, strict=True):
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            solver.step(state, state, None, None, 0.001)
        with wp.ScopedCapture(device=model.device, force_module_load=False) as capture:
            solvers[1].step(states[1], states[1], None, None, 0.001)
        for _ in range(80):
            solvers[0].step(states[0], states[0], None, None, 0.001)
            wp.capture_launch(capture.graph)
        eager, captured = states
        self.assertGreater(float(captured.body_qd.numpy()[free, 0]), 0.4)
        momentum = float(captured.joint_qd.numpy()[0] + captured.body_qd.numpy()[free, 0])
        self.assertAlmostEqual(momentum, 1.0, delta=1.0e-3)
        np.testing.assert_allclose(captured.joint_qd.numpy(), eager.joint_qd.numpy(), atol=1.0e-5)
        np.testing.assert_allclose(captured.body_qd.numpy(), eager.body_qd.numpy(), atol=1.0e-5)
        np.testing.assert_allclose(captured.body_q.numpy(), eager.body_q.numpy(), atol=1.0e-6)
        self.assertEqual(solvers[1].coupling_passes, 4)

    def test_articulated_response_replays_in_graph(self):
        """Capture the link mobility update, articulated contacts, and the MuJoCo re-solve."""
        model, box = _arm_and_box(device="cuda:0")
        solvers = [SolverMJMPM(model, relaxation=0.5, articulated_response=True) for _ in range(2)]
        states = [model.state(), model.state()]
        for solver, state in zip(solvers, states, strict=True):
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            solver.step(state, state, None, None, 0.001)
        with wp.ScopedCapture(device=model.device, force_module_load=False) as capture:
            solvers[1].step(states[1], states[1], None, None, 0.001)
        for _ in range(40):
            solvers[0].step(states[0], states[0], None, None, 0.001)
            wp.capture_launch(capture.graph)
        eager, captured = states
        self.assertGreater(float(np.abs(captured.body_qd.numpy()[box]).max()), 0.05)
        np.testing.assert_allclose(captured.joint_qd.numpy(), eager.joint_qd.numpy(), atol=1.0e-4)
        np.testing.assert_allclose(captured.body_qd.numpy(), eager.body_qd.numpy(), atol=1.0e-4)

    def test_reordered_gauss_seidel_matches_gauss_seidel(self):
        """Solve the several strain nodes of a P1d cell in order in the reordered layout."""
        results = {}
        for solver_name in ("gs", "gs-soa"):
            builder = newton.ModelBuilder()
            SolverMJMPM.register_custom_attributes(builder)
            builder.add_particle_grid(
                pos=wp.vec3(0.405, -0.275, 0.0055),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=6,
                dim_y=6,
                dim_z=8,
                cell_x=0.01,
                cell_y=0.01,
                cell_z=0.01,
                mass=0.00025,
                radius_mean=0.005,
                jitter=0.0,
                custom_attributes={"mpm:young_modulus": 1.6e4, "mpm:poisson_ratio": 0.36, "mpm:yield_stress": 1.0e6},
            )
            builder.add_ground_plane()
            model = builder.finalize(device="cuda:0")
            solver = SolverMJMPM(
                model,
                mpm_options={
                    "grid_type": "fixed",
                    "voxel_size": 0.015,
                    "grid_padding": 3,
                    "max_active_cell_count": 512,
                    "max_iterations": 50,
                    "strain_basis": "P1d",
                    "solver": solver_name,
                    "air_drag": 0.0,
                },
            )
            state = model.state()
            for _ in range(30):
                solver.step(state, state, None, None, 1.0 / 600.0)
            results[solver_name] = state.particle_q.numpy()
        self.assertTrue(np.isfinite(results["gs-soa"]).all())
        np.testing.assert_allclose(results["gs-soa"], results["gs"], atol=1.0e-5)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_single_node_eigenbasis_matches_assembled(self):
        """Integrate the P0 strain mass diagonal directly with the assembled matrix's eigenbasis result."""
        from newton._src.solvers.mjmpm.mpm.solver_implicit_mpm import SolverImplicitMPM as Material  # noqa: PLC0415

        build = Material._build_strain_eigenbasis
        stats = {"calls": 0, "nodes": 0}

        def compare(material, pic, scratch, inv_cell_volume):
            volume = scratch.strain_node_particle_volume
            before = wp.clone(volume)
            _ev, rotated = material._assemble_strain_eigenbasis(pic, scratch, inv_cell_volume)
            expected = volume.numpy().copy(), rotated.numpy().copy()
            volume.assign(before)
            result = build(material, pic, scratch, inv_cell_volume)
            self.assertIsNone(result[0])
            count = result[1].shape[0]
            np.testing.assert_allclose(volume.numpy()[:count], expected[0][:count], rtol=1.0e-6, atol=0.0)
            np.testing.assert_allclose(result[1].numpy(), expected[1][:count], rtol=1.0e-6, atol=0.0)
            stats["calls"] += 1
            regular = (expected[0][:count] != 0.0) & (expected[0][:count] != 1.0)
            stats["nodes"] = max(stats["nodes"], int(regular.sum()))
            return result

        Material._build_strain_eigenbasis = compare
        try:
            builder = newton.ModelBuilder()
            SolverMJMPM.register_custom_attributes(builder)
            builder.add_particle_grid(
                pos=wp.vec3(-0.05, -0.05, 0.005),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.3, 0.0, 0.0),
                dim_x=8,
                dim_y=8,
                dim_z=8,
                cell_x=0.01,
                cell_y=0.01,
                cell_z=0.01,
                mass=0.0016,
                jitter=0.0,
                radius_mean=0.005,
                custom_attributes={"mpm:friction": 0.5},
            )
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.14), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.03, hy=0.03, hz=0.03, cfg=newton.ModelBuilder.ShapeConfig(density=500.0))
            builder.add_ground_plane()
            model = builder.finalize(device="cuda:0")
            solver = SolverMJMPM(
                model,
                mpm_options={
                    "grid_type": "fixed",
                    "grid_padding": 6,
                    "max_active_cell_count": 2048,
                    "voxel_size": 0.02,
                    "max_iterations": 10,
                    "solver": "gs",
                    "strain_basis": "P0",
                },
            )
            state = model.state()
            for _ in range(20):
                solver.step(state, state, None, None, 1.0 / 240.0)
        finally:
            Material._build_strain_eigenbasis = build
        self.assertGreater(stats["calls"], 0)
        self.assertGreater(stats["nodes"], 0)

    def _check_graph_matches_eager(self, collider_basis, iterations):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMJMPM.register_custom_attributes(builder)
        pose = wp.transform(wp.vec3(5.0, 0.0, 0.0), wp.quat_identity())
        link = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        joint = builder.add_joint_revolute(parent=-1, child=link, axis=(0.0, 0.0, 1.0), parent_xform=pose)
        builder.add_articulation([joint])
        _box(builder, (0.3, 0.05, 0.05), half=0.2)
        _particles(
            builder,
            velocity=(1.0, 0.0, 0.0),
            custom_attributes={"mpm:yield_stress": 1.0e6, "mpm:tensile_yield_ratio": 1.0},
        )
        model = builder.finalize(device="cuda:0")
        options = {
            **_options(),
            "grid_type": "fixed",
            "voxel_size": 0.05,
            "grid_padding": 3,
            "max_active_cell_count": 64,
            "collider_basis": collider_basis,
        }
        eager = SolverMJMPM(model, mpm_options=options, coupling_iterations=iterations)
        captured = SolverMJMPM(model, mpm_options=options, coupling_iterations=iterations)
        a, b = model.state(), model.state()
        for solver, state in ((eager, a), (captured, b)):
            velocity = state.particle_qd.numpy()
            velocity += state.particle_q.numpy() * np.array([-0.2, -0.1, -0.05], dtype=np.float32)
            state.particle_qd.assign(velocity)
            solver.step(state, state, None, None, 0.005)
        with wp.ScopedCapture(device=model.device, force_module_load=False) as capture:
            captured.step(b, b, None, None, 0.005)
        for _ in range(8):
            eager.step(a, a, None, None, 0.005)
            wp.capture_launch(capture.graph)
        captured.check_capacity_status()
        np.testing.assert_allclose(a.body_q.numpy(), b.body_q.numpy(), atol=2.0e-6)
        np.testing.assert_allclose(a.body_qd.numpy(), b.body_qd.numpy(), atol=2.0e-6)
        np.testing.assert_allclose(a.particle_q.numpy(), b.particle_q.numpy(), atol=2.0e-6)
        for name in ("particle_elastic_strain", "particle_stress", "particle_Jp", "particle_qd_grad"):
            # Atomic grid reductions amplify float32 rounding in recovered stress.
            atol = 5.0e-3 if name == "particle_stress" else 2.0e-5
            np.testing.assert_allclose(
                getattr(a.mpm, name).numpy(), getattr(b.mpm, name).numpy(), rtol=1.0e-4, atol=atol, err_msg=name
            )
        self.assertGreater(float(np.linalg.norm(b.mpm.particle_stress.numpy())), 1.0e-5)


if __name__ == "__main__":
    unittest.main()

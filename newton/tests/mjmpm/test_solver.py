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


class TestMJMPMRigid(unittest.TestCase):
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


class TestMJMPMMaterial(unittest.TestCase):
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


@unittest.skipUnless(wp.is_cuda_available(), "CUDA device required")
class TestMJMPMCUDA(unittest.TestCase):
    def test_fixed_grid_graph_matches_eager(self):
        """Replay coupled articulation/rigid/material stepping in a fixed-grid CUDA graph."""
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
            "collider_basis": "pic8",
        }
        eager = SolverMJMPM(model, mpm_options=options)
        captured = SolverMJMPM(model, mpm_options=options)
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

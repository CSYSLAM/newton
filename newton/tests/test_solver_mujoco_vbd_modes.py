# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mode and MJVBD_V2 parity tests for :class:`SolverMuJoCoVBD`."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mujoco_vbd.dispatch import MuJoCoVBDBackendKind
from newton.solvers import SolverMJVBDV2, SolverMuJoCoVBD


def _register(builder: newton.ModelBuilder) -> None:
    SolverMuJoCoVBD.register_custom_attributes(builder)


def _static_particle_model(device="cpu"):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    _register(builder)
    builder.add_ground_plane()
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.01), vel=wp.vec3(), mass=0.01, radius=0.02)
    builder.color()
    return builder.finalize(device=device)


def _free_body_model(device="cpu"):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    _register(builder)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.color()
    return builder.finalize(device=device)


def _kinematic_particle_model(device="cpu", particle_x=0.0, particle_count=1):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    _register(builder)
    link = builder.add_link(label="kinematic_link")
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
    )
    articulation = builder.articulation_count
    builder.add_articulation([joint])
    for index in range(particle_count):
        builder.add_particle(
            pos=wp.vec3(particle_x, 0.03 * index, 1.11),
            vel=wp.vec3(),
            mass=0.01,
            radius=0.02,
        )
    builder.color()
    return builder.finalize(device=device), articulation


def _dynamic_finger_soft_block_model(device="cpu"):
    """Small crossing-contact scene used for deterministic A/B acceptance."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _register(builder)
    finger = builder.add_link(label="dynamic_finger")
    builder.add_shape_box(
        finger,
        hx=0.09,
        hy=0.09,
        hz=0.1,
        cfg=newton.ModelBuilder.ShapeConfig(density=100.0, ke=2.0e4, kd=50.0, mu=0.5),
    )
    joint = builder.add_joint_prismatic(
        parent=-1,
        child=finger,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
    )
    articulation = builder.articulation_count
    builder.add_articulation([joint])
    builder.add_soft_grid(
        pos=wp.vec3(-0.1, -0.1, 1.145),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
        density=100.0,
        k_mu=2.0e3,
        k_lambda=3.0e3,
        k_damp=10.0,
        particle_radius=0.01,
    )
    builder.color()
    return builder.finalize(device=device), articulation


def _multiworld_kinematic_particle_model(device="cpu", particle_x=0.08):
    world = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    _register(world)
    link = world.add_link(label="dynamic_link")
    world.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    joint = world.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
    )
    world.add_articulation([joint])
    world.add_particle(
        pos=wp.vec3(particle_x, 0.0, 1.11),
        vel=wp.vec3(),
        mass=0.01,
        radius=0.02,
    )
    world.color()

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    _register(builder)
    builder.add_world(world)
    builder.add_world(world)
    builder.color()
    return builder.finalize(device=device), (0, 1)


def _partition_model(device="cpu"):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    _register(builder)
    builder.add_ground_plane()
    free_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.65, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_box(free_body, hx=0.08, hy=0.08, hz=0.08)
    link = builder.add_link(label="mujoco_link")
    builder.add_shape_box(link, hx=0.2, hy=0.05, hz=0.05)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    articulation = builder.articulation_count
    builder.add_articulation([joint])
    builder.color()
    return builder.finalize(device=device), articulation


def _filtered_partition_model(device="cpu"):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    _register(builder)
    free_body = builder.add_body(xform=wp.transform(wp.vec3(0.1, 0.0, 1.0), wp.quat_identity()), mass=1.0)
    free_shape = builder.add_shape_box(free_body, hx=0.08, hy=0.08, hz=0.08)
    link = builder.add_link(label="mujoco_link")
    robot_shape = builder.add_shape_box(link, hx=0.2, hy=0.05, hz=0.05)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    articulation = builder.articulation_count
    builder.add_articulation([joint])
    builder.add_shape_collision_filter_pair(free_shape, robot_shape)
    builder.color()
    return builder.finalize(device=device), articulation


def _robot_with_ground_particle_model(device="cpu"):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    _register(builder)
    builder.add_ground_plane()
    link = builder.add_link(label="mujoco_link")
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
    )
    articulation = builder.articulation_count
    builder.add_articulation([joint])
    builder.add_particle(pos=wp.vec3(1.0, 0.0, 0.01), vel=wp.vec3(), mass=0.01, radius=0.02)
    builder.color()
    return builder.finalize(device=device), articulation


def _free_proxy_with_coarse_cloth_model(device="cpu"):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    _register(builder)
    proxy = builder.add_link(label="free_proxy")
    builder.add_shape_box(proxy, hx=0.5, hy=0.5, hz=0.5)
    joint = builder.add_joint_free(child=proxy)
    articulation = builder.articulation_count
    builder.add_articulation([joint])
    # Every corner lies outside the box margin; only the diagonal edge and
    # triangle interiors cross its upper face.
    builder.add_cloth_grid(
        pos=wp.vec3(-1.0, -1.0, 0.45),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=1,
        dim_y=1,
        cell_x=2.0,
        cell_y=2.0,
        mass=0.1,
        particle_radius=0.01,
    )
    builder.color()
    return builder.finalize(device=device), articulation


def _particle_constraint_model(kind: str, device="cpu"):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    _register(builder)
    positions = ((0.0, 0.0, 1.0), (0.2, 0.0, 1.0), (0.0, 0.2, 1.0), (0.0, 0.0, 1.2))
    count = 2 if kind == "spring" else 4
    for position in positions[:count]:
        builder.add_particle(pos=wp.vec3(*position), vel=wp.vec3(), mass=0.1, radius=0.01)
    if kind == "spring":
        builder.add_spring(0, 1, ke=100.0, kd=1.0, control=0.0)
    elif kind == "tet":
        builder.add_tetrahedron(0, 1, 2, 3)
    else:
        raise ValueError(kind)
    builder.color()
    return builder.finalize(device=device)


def _step(solver, model, dt=1.0 / 120.0):
    state_in = model.state()
    state_out = model.state()
    solver.step(state_in, state_out, model.control(), None, dt)
    wp.synchronize()
    return state_in, state_out


def _assert_states_close(test: unittest.TestCase, left, right) -> None:
    for name in ("body_q", "body_qd", "joint_q", "joint_qd", "particle_q", "particle_qd"):
        a = getattr(left, name, None)
        b = getattr(right, name, None)
        if a is None or b is None:
            test.assertIs(a, b, name)
        else:
            np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1.0e-6, atol=1.0e-7, err_msg=name)


class TestMuJoCoVBDModes(unittest.TestCase):
    def test_public_solver_export(self):
        self.assertIs(newton.solvers.SolverMuJoCoVBD, SolverMuJoCoVBD)

    def test_registered_builder_can_import_mjcf(self):
        builder = newton.ModelBuilder()
        SolverMuJoCoVBD.register_custom_attributes(builder)
        builder.add_mjcf(
            """
            <mujoco>
              <worldbody>
                <body name="root">
                  <geom type="sphere" size="0.05"/>
                  <body name="link" pos="0 0 0.1">
                    <joint name="hinge" type="hinge"/>
                    <geom type="sphere" size="0.05"/>
                  </body>
                </body>
              </worldbody>
              <actuator>
                <position name="hinge_drive" joint="hinge"/>
              </actuator>
            </mujoco>
            """,
            floating=False,
        )
        model = builder.finalize(device="cpu")
        self.assertEqual(model.body_count, 2)
        self.assertEqual(model.custom_frequency_counts["mujoco:actuator"], 1)

    def test_pure_vbd_soft_matches_mjvbd_v2_baseline(self):
        model = _static_particle_model()
        options = {"contact_mode": "soft", "vbd_options": {"iterations": 2}}
        new_solver = SolverMuJoCoVBD(model, **options)
        old_solver = SolverMJVBDV2(model, **options)
        _, new_state = _step(new_solver, model)
        _, old_state = _step(old_solver, model)
        self.assertEqual(new_solver.backend_kind, MuJoCoVBDBackendKind.PURE_VBD_SOFT)
        _assert_states_close(self, new_state, old_state)

    def test_pure_vbd_full_matches_mjvbd_v2_baseline(self):
        model = _free_body_model()
        options = {"contact_mode": "full", "vbd_options": {"iterations": 2}}
        new_solver = SolverMuJoCoVBD(model, **options)
        old_solver = SolverMJVBDV2(model, **options)
        _, new_state = _step(new_solver, model)
        _, old_state = _step(old_solver, model)
        self.assertEqual(new_solver.backend_kind, MuJoCoVBDBackendKind.PURE_VBD_FULL)
        _assert_states_close(self, new_state, old_state)

    def test_one_way_kinematic_soft_matches_mjvbd_v2_baseline(self):
        model, articulation = _kinematic_particle_model()
        common = {
            "mujoco_articulations": [articulation],
            "joint_mode": "kinematic",
            "contact_mode": "soft",
            "vbd_options": {"iterations": 2},
        }
        new_solver = SolverMuJoCoVBD(
            model,
            coupling_mode="one_way",
            **common,
        )
        old_solver = SolverMJVBDV2(model, **common)
        _, new_state = _step(new_solver, model)
        _, old_state = _step(old_solver, model)
        self.assertEqual(new_solver.backend_kind, MuJoCoVBDBackendKind.ONE_WAY_KINEMATIC_SOFT)
        self.assertFalse(new_solver.features.feedback_enabled)
        _assert_states_close(self, new_state, old_state)

    def test_one_way_kinematic_full_matches_mjvbd_v2_baseline(self):
        model, articulation = _partition_model()
        common = {
            "mujoco_articulations": [articulation],
            "joint_mode": "kinematic",
            "contact_mode": "full",
            "vbd_options": {"iterations": 2},
        }
        new_solver = SolverMuJoCoVBD(model, coupling_mode="one_way", **common)
        old_solver = SolverMJVBDV2(model, **common)
        _, new_state = _step(new_solver, model)
        _, old_state = _step(old_solver, model)
        self.assertEqual(new_solver.backend_kind, MuJoCoVBDBackendKind.ONE_WAY_KINEMATIC_FULL)
        _assert_states_close(self, new_state, old_state)

    def test_pure_vbd_tet_and_spring_are_finite(self):
        for kind in ("tet", "spring"):
            with self.subTest(kind=kind):
                model = _particle_constraint_model(kind)
                solver = SolverMuJoCoVBD(model, vbd_options={"iterations": 2})
                _, state_out = _step(solver, model)
                self.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())
                self.assertTrue(np.isfinite(state_out.particle_qd.numpy()).all())

    def test_private_vbd_particle_reset_restores_defaults(self):
        model = _static_particle_model()
        solver = SolverMuJoCoVBD(model, contact_mode="soft", vbd_options={"iterations": 1})
        state = model.state()
        state.particle_q.assign([[1.0, 2.0, 3.0]])
        state.particle_qd.assign([[4.0, 5.0, 6.0]])
        solver.reset(state)
        wp.synchronize()
        np.testing.assert_array_equal(state.particle_q.numpy(), model.particle_q.numpy())
        np.testing.assert_array_equal(state.particle_qd.numpy(), model.particle_qd.numpy())

    def test_one_way_proxy_is_immovable_and_keeps_public_source(self):
        """Keep a one-way robot proxy on its prescribed trajectory."""
        model, articulation = _kinematic_particle_model()
        solver = SolverMuJoCoVBD(
            model,
            mujoco_articulations=[articulation],
            joint_mode="kinematic",
            coupling_mode="one_way",
            contact_mode="full",
            vbd_options={"iterations": 2},
        )
        state_in, state_out = _step(solver, model)
        self.assertEqual(solver.backend_kind, MuJoCoVBDBackendKind.ONE_WAY_KINEMATIC_FULL)
        proxy_ids = list(solver.ownership.proxy_bodies)
        self.assertTrue((solver.vbd_solver.model.body_inv_mass.numpy()[proxy_ids] == 0.0).all())
        np.testing.assert_array_equal(state_out.body_q.numpy(), state_in.body_q.numpy())
        np.testing.assert_array_equal(state_out.body_qd.numpy(), state_in.body_qd.numpy())
        self.assertIsNone(solver.mujoco_solver)
        self.assertFalse(solver.features.feedback_enabled)

    def test_two_way_smoke_allocates_transaction_state(self):
        model, articulation = _kinematic_particle_model()
        solver = SolverMuJoCoVBD(
            model,
            mujoco_articulations=[articulation],
            joint_mode="dynamic",
            coupling_mode="two_way",
            contact_mode="soft",
            mujoco_options={"use_mujoco_cpu": True},
            vbd_options={"iterations": 1},
        )
        _, state_out = _step(solver, model)
        self.assertEqual(solver.backend_kind, MuJoCoVBDBackendKind.TWO_WAY)
        self.assertTrue(solver.features.feedback_enabled)
        self.assertTrue(solver.features.iteration_transaction_enabled)
        self.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())
        self.assertTrue(np.isfinite(state_out.body_q.numpy()).all())

    def test_two_way_reset_restores_particle_state(self):
        model, articulation = _kinematic_particle_model(particle_x=0.08)
        solver = SolverMuJoCoVBD(
            model,
            mujoco_articulations=[articulation],
            joint_mode="dynamic",
            coupling_mode="two_way",
            contact_mode="soft",
            coupling_options={"iterations": 2},
            mujoco_options={"use_mujoco_cpu": True},
            vbd_options={"iterations": 1},
        )
        state = model.state()
        state.particle_q.assign([[1.0, 2.0, 3.0]])
        state.particle_qd.assign([[4.0, 5.0, 6.0]])
        solver.reset(state)
        wp.synchronize()
        np.testing.assert_array_equal(state.particle_q.numpy(), model.particle_q.numpy())
        np.testing.assert_array_equal(state.particle_qd.numpy(), model.particle_qd.numpy())

    def test_two_way_outer_iterations_resolve_one_substep(self):
        outputs = []
        for iterations in (1, 4):
            model, articulation = _partition_model()
            solver = SolverMuJoCoVBD(
                model,
                mujoco_articulations=[articulation],
                joint_mode="dynamic",
                coupling_mode="two_way",
                contact_mode="full",
                coupling_options={"iterations": iterations, "relaxation": "fixed"},
                mujoco_options={"use_mujoco_cpu": True},
                vbd_options={"iterations": 1},
            )
            state_in = model.state()
            state_before = model.state()
            state_before.assign(state_in)
            state_out = model.state()
            solver.step(state_in, state_out, model.control(), None, 1.0 / 120.0)
            wp.synchronize()

            self.assertEqual(solver.mujoco_solver._step, 1)
            _assert_states_close(self, state_in, state_before)
            outputs.append(state_out)

        # With no M-V contact, extra fixed-point rounds must not advance either
        # core through additional physical timesteps.
        _assert_states_close(self, outputs[0], outputs[1])

    def test_two_way_contact_feedback_moves_mujoco_joint(self):
        model, articulation = _kinematic_particle_model(particle_x=0.08)
        solver = SolverMuJoCoVBD(
            model,
            mujoco_articulations=[articulation],
            joint_mode="dynamic",
            coupling_mode="two_way",
            contact_mode="soft",
            coupling_options={"iterations": 2, "relaxation": "fixed"},
            mujoco_options={"use_mujoco_cpu": True},
            vbd_options={"iterations": 1},
        )
        state_in = model.state()
        state_before = model.state()
        state_before.assign(state_in)
        state_out = model.state()
        solver.step(state_in, state_out, model.control(), None, 1.0 / 120.0)
        wp.synchronize()

        wrench = solver.diagnostics.feedback_wrench_raw.numpy()[0]
        self.assertEqual(int(solver.contacts.soft_contact_count.numpy()[0]), 1)
        self.assertGreater(abs(float(wrench[2])), 1.0)
        self.assertGreater(abs(float(wrench[4])), 1.0e-3)
        self.assertGreater(abs(float(state_out.joint_qd.numpy()[0])), 1.0e-4)
        self.assertEqual(solver.mujoco_solver._step, 1)
        _assert_states_close(self, state_in, state_before)

    def test_two_way_speculative_contact_and_al_acceptance_scene(self):
        """A crossing dynamic finger must not miss a deformable block for one whole step."""

        def run_first_step(speculative_distance, augmented_lagrangian):
            model, articulation = _dynamic_finger_soft_block_model()
            solver = SolverMuJoCoVBD(
                model,
                mujoco_articulations=[articulation],
                joint_mode="dynamic",
                coupling_mode="two_way",
                contact_mode="soft",
                coupling_options={
                    "iterations": 4,
                    "relaxation": "fixed",
                    "soft_contact_speculative_distance": speculative_distance,
                    "soft_contact_augmented_lagrangian": augmented_lagrangian,
                },
                mujoco_options={"use_mujoco_cpu": True},
                vbd_options={"iterations": 4},
            )
            state_in = model.state()
            state_in.joint_qd.assign([3.0])
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            initial_particles = state_in.particle_q.numpy().copy()
            state_out = model.state()
            solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
            wp.synchronize()
            return model, solver, state_out, initial_particles

        _, legacy, legacy_out, legacy_particles = run_first_step(0.0, False)
        speculative_model, speculative, speculative_out, _ = run_first_step(0.08, False)
        model, robust, robust_out, robust_particles = run_first_step(0.08, True)

        self.assertEqual(int(legacy.contacts.soft_contact_count.numpy()[0]), 0)
        self.assertGreaterEqual(int(robust.contacts.soft_contact_count.numpy()[0]), 1)
        legacy_displacement = float(np.max(np.abs(legacy_out.particle_q.numpy() - legacy_particles)))
        robust_displacement = float(np.max(np.abs(robust_out.particle_q.numpy() - robust_particles)))
        self.assertLess(legacy_displacement, 1.0e-6)
        self.assertGreater(robust_displacement, 1.0e-4)

        def max_bottom_penetration(state):
            finger_top = float(state.body_q.numpy()[0, 2]) + 0.1
            bottom_surface = state.particle_q.numpy()[:9, 2] - 0.01
            return max(0.0, finger_top - float(np.min(bottom_surface)))

        self.assertLess(max_bottom_penetration(robust_out), max_bottom_penetration(legacy_out))
        history = robust._backend.vbd_solver.body_particle_contact_lambda_history.numpy()
        self.assertGreater(float(np.max(history)), 1.0e-3)

        # A second physical substep consumes the persisted multiplier and sends
        # a nonzero equal-and-opposite reaction back through the MuJoCo joint.
        next_out = model.state()
        robust.step(robust_out, next_out, model.control(), None, 1.0 / 60.0)
        speculative_next = speculative_model.state()
        speculative.step(
            speculative_out,
            speculative_next,
            speculative_model.control(),
            None,
            1.0 / 60.0,
        )
        wp.synchronize()
        self.assertGreater(abs(float(robust.diagnostics.feedback_wrench_raw.numpy()[0, 2])), 1.0)
        self.assertLess(float(next_out.joint_qd.numpy()[0]), 2.5)
        self.assertLess(float(next_out.joint_qd.numpy()[0]), float(speculative_next.joint_qd.numpy()[0]) - 0.5)

    def test_two_way_multiworld_cpu_falls_back_to_warp_and_advances_every_world(self):
        model, articulations = _multiworld_kinematic_particle_model()
        with self.assertWarnsRegex(RuntimeWarning, "template world"):
            solver = SolverMuJoCoVBD(
                model,
                mujoco_articulations=articulations,
                joint_mode="dynamic",
                coupling_mode="two_way",
                contact_mode="soft",
                coupling_options={"iterations": 2, "relaxation": "fixed"},
                mujoco_options={"use_mujoco_cpu": True, "separate_worlds": True},
                vbd_options={"iterations": 1},
            )

        _, state_out = _step(solver, model)
        joint_qd = state_out.joint_qd.numpy()
        self.assertFalse(solver.mujoco_solver.use_mujoco_cpu)
        self.assertEqual(int(solver.contacts.soft_contact_count.numpy()[0]), 2)
        self.assertTrue((np.abs(joint_qd) > 1.0e-4).all())
        np.testing.assert_allclose(joint_qd[0], joint_qd[1], rtol=1.0e-5, atol=1.0e-7)

    def test_two_way_nonfinite_feedback_aborts_transaction(self):
        model, articulation = _kinematic_particle_model(particle_x=0.08)
        solver = SolverMuJoCoVBD(
            model,
            mujoco_articulations=[articulation],
            joint_mode="dynamic",
            coupling_mode="two_way",
            contact_mode="soft",
            coupling_options={"iterations": 2, "relaxation": "fixed"},
            mujoco_options={"use_mujoco_cpu": True},
            vbd_options={"iterations": 1},
        )
        state_in = model.state()
        state_before = model.state()
        state_before.assign(state_in)
        state_out = model.state()
        warm_start_before = np.full_like(solver._backend.runtime.wrench_relaxed.numpy(), 0.25)
        solver._backend.runtime.wrench_relaxed.assign(warm_start_before)
        solver._backend.runtime.wrench_previous.assign(warm_start_before)

        feedback = solver._backend.feedback
        original_harvest = feedback.harvest

        def poison_feedback(*args, **kwargs):
            result = original_harvest(*args, **kwargs)
            poisoned = feedback.out_wrench.numpy()
            poisoned[0, 0] = np.nan
            feedback.out_wrench.assign(poisoned)
            return result

        feedback.harvest = poison_feedback
        with self.assertRaises(FloatingPointError):
            solver.step(state_in, state_out, model.control(), None, 1.0 / 120.0)
        wp.synchronize()

        self.assertEqual(solver.mujoco_solver._step, 0)
        np.testing.assert_array_equal(solver._backend.runtime.wrench_relaxed.numpy(), warm_start_before)
        np.testing.assert_array_equal(solver._backend.runtime.wrench_previous.numpy(), warm_start_before)
        _assert_states_close(self, state_in, state_before)
        _assert_states_close(self, state_out, state_before)

    def test_two_way_soft_contact_overflow_aborts_transaction(self):
        model, articulation = _kinematic_particle_model(particle_x=0.08, particle_count=2)
        solver = SolverMuJoCoVBD(
            model,
            mujoco_articulations=[articulation],
            joint_mode="dynamic",
            coupling_mode="two_way",
            contact_mode="soft",
            coupling_options={"iterations": 2, "relaxation": "fixed", "fail_on_overflow": True},
            collision_options={"soft_contact_max": 1},
            mujoco_options={"use_mujoco_cpu": True},
            vbd_options={"iterations": 1},
        )
        state_in = model.state()
        state_before = model.state()
        state_before.assign(state_in)
        state_out = model.state()

        with self.assertRaisesRegex(RuntimeError, "soft_contact_overflow"):
            solver.step(state_in, state_out, model.control(), None, 1.0 / 120.0)
        wp.synchronize()

        self.assertEqual(int(solver.contacts.soft_contact_count.numpy()[0]), 2)
        self.assertEqual(int(solver.diagnostics.soft_contact_overflow.numpy()[0]), 1)
        self.assertEqual(solver.mujoco_solver._step, 0)
        _assert_states_close(self, state_in, state_before)
        _assert_states_close(self, state_out, state_before)

    def test_two_way_rigid_contact_feedback_moves_both_sides(self):
        model, articulation = _partition_model()
        solver = SolverMuJoCoVBD(
            model,
            mujoco_articulations=[articulation],
            joint_mode="dynamic",
            coupling_mode="two_way",
            contact_mode="full",
            coupling_options={"iterations": 2, "relaxation": "fixed"},
            mujoco_options={"use_mujoco_cpu": True},
            vbd_options={"iterations": 2},
        )
        state_in = model.state()
        body_q = state_in.body_q.numpy()
        body_q[0, 0] = 0.42
        state_in.body_q.assign(body_q)
        state_before = model.state()
        state_before.assign(state_in)
        state_out = model.state()
        solver.step(state_in, state_out, model.control(), None, 1.0 / 120.0)
        wp.synchronize()

        proxy_body = solver.ownership.proxy_bodies[0]
        wrench = solver.diagnostics.feedback_wrench_raw.numpy()[proxy_body]
        backend = solver._backend
        self.assertGreater(int(solver.contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertGreater(float(np.linalg.norm(wrench)), 1.0)
        self.assertGreater(abs(float(state_out.body_q.numpy()[0, 0]) - 0.42), 1.0e-5)
        self.assertGreater(abs(float(state_out.joint_qd.numpy()[-1])), 1.0e-4)
        self.assertEqual(solver.mujoco_solver._step, 1)
        self.assertEqual(backend.collision_pipeline._pipeline.broad_phase_mode, "explicit")
        self.assertFalse(backend.overlays.vbd.joint_enabled.numpy()[list(solver.ownership.mujoco_joints)].any())
        mujoco_shape_flags = backend.overlays.mujoco.shape_flags.numpy()
        for shape in solver.ownership.vbd_shapes:
            self.assertEqual(int(mujoco_shape_flags[shape]) & int(newton.ShapeFlags.COLLIDE_SHAPES), 0)
        _assert_states_close(self, state_in, state_before)

    def test_two_way_routing_honors_finalized_collision_filters(self):
        model, articulation = _filtered_partition_model()
        solver = SolverMuJoCoVBD(
            model,
            mujoco_articulations=[articulation],
            joint_mode="dynamic",
            coupling_mode="two_way",
            contact_mode="full",
            coupling_options={"iterations": 2},
            mujoco_options={"use_mujoco_cpu": True},
            vbd_options={"iterations": 1},
        )
        self.assertEqual(solver._backend.routing.cross_pair_count, 0)
        _, state_out = _step(solver, model)
        self.assertEqual(int(solver.contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertTrue(np.isfinite(state_out.body_q.numpy()).all())

    def test_two_way_mujoco_static_owner_keeps_vbd_particle_ground_contact(self):
        model, articulation = _robot_with_ground_particle_model()
        solver = SolverMuJoCoVBD(
            model,
            mujoco_articulations=[articulation],
            joint_mode="dynamic",
            coupling_mode="two_way",
            contact_mode="soft",
            coupling_options={"iterations": 2, "static_contact_owner": "mujoco"},
            mujoco_options={"use_mujoco_cpu": True},
            vbd_options={"iterations": 1},
        )
        _, state_out = _step(solver, model)
        self.assertEqual(int(solver.contacts.soft_contact_count.numpy()[0]), 1)
        self.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())

    def test_two_way_shape_flag_update_refreshes_collision_overlay(self):
        model, articulation = _kinematic_particle_model(particle_x=0.08)
        solver = SolverMuJoCoVBD(
            model,
            mujoco_articulations=[articulation],
            joint_mode="dynamic",
            coupling_mode="two_way",
            contact_mode="soft",
            coupling_options={"iterations": 2},
            mujoco_options={"use_mujoco_cpu": True},
            vbd_options={"iterations": 1},
        )
        robot_shape = solver.ownership.mujoco_shapes[0]
        flags = model.shape_flags.numpy()
        flags[robot_shape] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        model.shape_flags.assign(flags)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

        _, state_out = _step(solver, model)
        overlay_flags = solver._backend.overlays.vbd.shape_flags.numpy()
        self.assertEqual(int(overlay_flags[robot_shape]) & int(newton.ShapeFlags.COLLIDE_PARTICLES), 0)
        self.assertEqual(int(solver.contacts.soft_contact_count.numpy()[0]), 0)
        self.assertTrue(np.isfinite(state_out.particle_q.numpy()).all())

    def test_two_way_rejects_collision_topology_growth_without_rebuild(self):
        model, articulation = _kinematic_particle_model(particle_x=0.08)
        flags = model.shape_flags.numpy()
        flags[0] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        model.shape_flags.assign(flags)
        solver = SolverMuJoCoVBD(
            model,
            mujoco_articulations=[articulation],
            joint_mode="dynamic",
            coupling_mode="two_way",
            contact_mode="soft",
            mujoco_options={"use_mujoco_cpu": True},
            vbd_options={"iterations": 1},
        )

        flags[0] |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
        model.shape_flags.assign(flags)
        with self.assertRaisesRegex(RuntimeError, "rebuild SolverMuJoCoVBD"):
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

    def test_two_way_full_surface_edge_face_feedback_moves_mujoco_body(self):
        model, articulation = _free_proxy_with_coarse_cloth_model()
        solver = SolverMuJoCoVBD(
            model,
            mujoco_articulations=[articulation],
            joint_mode="dynamic",
            coupling_mode="two_way",
            contact_mode="full",
            coupling_options={"iterations": 2, "relaxation": "fixed"},
            collision_options={
                "enable_rigid_soft_full_surface_contact": True,
                "soft_contact_margin": 0.1,
            },
            mujoco_options={"use_mujoco_cpu": True},
            vbd_options={"iterations": 1},
        )
        _, state_out = _step(solver, model)

        count = int(solver.contacts.soft_contact_count.numpy()[0])
        indices = solver.contacts.soft_contact_indices.numpy()[:count]
        self.assertEqual(int(np.count_nonzero(indices[:, 1] < 0)), 0)
        self.assertGreater(int(np.count_nonzero((indices[:, 1] >= 0) & (indices[:, 2] < 0))), 0)
        self.assertGreater(int(np.count_nonzero(indices[:, 2] >= 0)), 0)
        self.assertGreater(float(np.linalg.norm(solver.diagnostics.feedback_wrench_raw.numpy()[0])), 1.0)
        self.assertGreater(float(np.linalg.norm(state_out.joint_qd.numpy())), 1.0e-5)

    def test_two_way_contact_matcher_restarts_each_outer_iteration(self):
        model, articulation = _partition_model()
        solver = SolverMuJoCoVBD(
            model,
            mujoco_articulations=[articulation],
            joint_mode="dynamic",
            coupling_mode="two_way",
            contact_mode="full",
            coupling_options={"iterations": 2, "relaxation": "fixed"},
            collision_options={"contact_matching": "latest"},
            mujoco_options={"use_mujoco_cpu": True},
            vbd_options={"iterations": 1},
        )
        state_in = model.state()
        body_q = state_in.body_q.numpy()
        body_q[0, 0] = 0.42
        state_in.body_q.assign(body_q)
        state_out = model.state()

        pipeline = solver._backend.collision_pipeline._pipeline
        matcher = pipeline._contact_matcher
        previous_counts = []
        original_collide = pipeline.collide

        def record_previous_count(*args, **kwargs):
            previous_counts.append(int(matcher.prev_contact_count.numpy()[0]))
            return original_collide(*args, **kwargs)

        pipeline.collide = record_previous_count
        solver.step(state_in, state_out, model.control(), None, 1.0 / 120.0)
        wp.synchronize()

        self.assertEqual(previous_counts, [0, 0])
        self.assertGreater(int(matcher.prev_contact_count.numpy()[0]), 0)

    @unittest.skipUnless(wp.is_cuda_available(), "Two-way CUDA graph capture requires CUDA")
    def test_two_way_cuda_graph_capture_and_replay(self):
        model, articulation = _kinematic_particle_model(device="cuda:0", particle_x=0.08)
        solver = SolverMuJoCoVBD(
            model,
            mujoco_articulations=[articulation],
            joint_mode="dynamic",
            coupling_mode="two_way",
            contact_mode="soft",
            coupling_options={"iterations": 2, "relaxation": "fixed"},
            collision_options={"contact_matching": "latest"},
            vbd_options={"iterations": 1},
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        # Warm up every private MuJoCo/VBD/collision kernel before capture.
        solver.step(state_0, state_1, control, None, 1.0 / 120.0)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(state_1, state_0, control, None, 1.0 / 120.0)

        wp.capture_launch(capture.graph)
        wp.capture_launch(capture.graph)
        wp.synchronize()

        self.assertEqual(int(solver.contacts.soft_contact_count.numpy()[0]), 1)
        self.assertEqual(int(solver.diagnostics.soft_contact_overflow.numpy()[0]), 0)
        self.assertEqual(int(solver.diagnostics.rigid_contact_overflow.numpy()[0]), 0)
        self.assertGreater(abs(float(state_0.joint_qd.numpy()[0])), 1.0e-4)
        self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state_0.particle_q.numpy()).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)

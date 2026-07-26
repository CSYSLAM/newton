# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for SolverMJVBDV2 ownership and stepping."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjvbd_v2.ownership import resolve_ownership
from newton.solvers import SolverMJVBD, SolverMJVBDV2


def _build_partition_model(device):
    builder = newton.ModelBuilder()
    SolverMJVBDV2.register_custom_attributes(builder)
    builder.add_ground_plane()

    free_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.65, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
        label="vbd_free_body",
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
        label="mujoco_joint",
    )
    articulation = builder.articulation_count
    builder.add_articulation([joint], label="mujoco_articulation")
    builder.color()
    return builder.finalize(device=device), free_body, link, joint, articulation


def _build_pure_vbd_model(device):
    builder = newton.ModelBuilder()
    SolverMJVBDV2.register_custom_attributes(builder)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
    builder.color()
    return builder.finalize(device=device), body


def _build_kinematic_particle_model(device):
    builder = newton.ModelBuilder()
    SolverMJVBDV2.register_custom_attributes(builder)
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
    builder.add_particle(
        pos=wp.vec3(0.0, 0.0, 1.11),
        vel=wp.vec3(),
        mass=0.01,
        radius=0.02,
    )
    builder.color()
    return builder.finalize(device=device), articulation


def _build_kinematic_tet_model(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverMJVBDV2.register_custom_attributes(builder)
    link = builder.add_link(label="kinematic_tet_link", is_kinematic=True)
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
    )
    articulation = builder.articulation_count
    builder.add_articulation([joint])
    for position in ((0.0, 0.0, 1.12), (0.08, 0.0, 1.12), (0.0, 0.08, 1.12), (0.0, 0.0, 1.20)):
        builder.add_particle(pos=wp.vec3(*position), vel=wp.vec3(), mass=0.01, radius=0.01)
    builder.add_tetrahedron(0, 1, 2, 3)
    builder.color()
    return builder.finalize(device=device), articulation


def _build_two_rigid_body_model(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverMJVBDV2.register_custom_attributes(builder)
    left = builder.add_body(
        xform=wp.transform(wp.vec3(-0.045, 0.0, 0.0), wp.quat_identity()),
        mass=0.5,
        label="left_body",
    )
    right = builder.add_body(
        xform=wp.transform(wp.vec3(0.045, 0.0, 0.0), wp.quat_identity()),
        mass=0.5,
        label="right_body",
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0)
    builder.add_shape_sphere(left, radius=0.05, cfg=cfg)
    builder.add_shape_sphere(right, radius=0.05, cfg=cfg)
    builder.color()
    return builder.finalize(device=device), left, right


def _build_particle_module_model(device, element):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverMJVBDV2.register_custom_attributes(builder)
    positions = (
        (0.0, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.0, 0.2, 0.0),
        (0.0, 0.0, 0.2),
    )
    particle_count = 3 if element == "triangle" else 4
    for position in positions[:particle_count]:
        builder.add_particle(
            pos=wp.vec3(*position),
            vel=wp.vec3(),
            mass=0.1,
            radius=0.01,
        )
    if element == "triangle":
        builder.add_triangle(0, 1, 2)
    elif element == "tetrahedron":
        builder.add_tetrahedron(0, 1, 2, 3)
    else:
        raise ValueError(f"Unsupported element type: {element}")
    builder.color()
    return builder.finalize(device=device)


class TestMJVBDV2(unittest.TestCase):
    def test_ownership_partition(self):
        model, free_body, link, joint, articulation = _build_partition_model("cpu")
        ownership = resolve_ownership(
            model,
            mujoco_articulations=[articulation],
        )

        self.assertEqual(ownership.mujoco_joints, (joint,))
        self.assertEqual(ownership.mujoco_bodies, (link,))
        self.assertIn(free_body, ownership.vbd_bodies)
        self.assertTrue(ownership.has_vbd_dynamic_bodies)

    def test_cpu_step_keeps_free_body_in_vbd(self):
        try:
            model, free_body, link, _, articulation = _build_partition_model("cpu")
            state_0 = model.state()
            state_1 = model.state()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
            newton.eval_fk(model, model.joint_q, model.joint_qd, state_1)
            solver = SolverMJVBDV2(
                model,
                mujoco_articulations=[articulation],
                vbd_options={"iterations": 2},
                mujoco_options={"use_mujoco_cpu": True},
                collision_options={"broad_phase": "nxn"},
            )
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(f"MuJoCo is unavailable: {error}")

        initial_z = float(state_0.body_q.numpy()[free_body, 2])
        control = model.control()
        for _ in range(3):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 120.0)
            state_0, state_1 = state_1, state_0

        body_q = state_0.body_q.numpy()
        self.assertTrue(np.all(np.isfinite(body_q)))
        self.assertLess(float(body_q[free_body, 2]), initial_z)
        self.assertTrue(np.all(np.isfinite(body_q[link])))

    def test_no_joint_model_dispatches_to_pure_vbd(self):
        model, body = _build_pure_vbd_model("cpu")
        state_0 = model.state()
        state_1 = model.state()
        solver = SolverMJVBDV2(
            model,
            vbd_options={"iterations": 2},
            collision_options={"broad_phase": "nxn"},
        )

        self.assertEqual(solver.features.backend, "pure_vbd")
        self.assertEqual(solver.features.mujoco_joint_count, 0)
        self.assertIsNone(solver.mujoco_solver)
        initial_z = float(state_0.body_q.numpy()[body, 2])
        solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)
        self.assertLess(float(state_1.body_q.numpy()[body, 2]), initial_z)

    def test_kinematic_particle_model_dispatches_to_mjvbd_fast_path(self):
        model, articulation = _build_kinematic_particle_model("cpu")
        solver = SolverMJVBDV2(
            model,
            mujoco_articulations=[articulation],
            joint_mode="kinematic",
            vbd_options={"iterations": 2},
            collision_options={"soft_contact_margin": 0.01},
        )

        self.assertEqual(solver.features.backend, "mjvbd_kinematic_soft")
        self.assertEqual(solver.features.vbd_dynamic_body_count, 0)
        self.assertIsNone(solver.mujoco_solver)

    def test_dynamic_particle_model_uses_optimized_external_vbd(self):
        model, articulation = _build_kinematic_particle_model("cpu")
        state_in = model.state()
        state_out = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_out)
        try:
            solver = SolverMJVBDV2(
                model,
                mujoco_articulations=[articulation],
                joint_mode="dynamic",
                vbd_options={"iterations": 2},
                mujoco_options={"use_mujoco_cpu": True},
                collision_options={"soft_contact_margin": 0.01},
            )
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(f"MuJoCo is unavailable: {error}")

        self.assertEqual(solver.features.backend, "coupled")
        self.assertTrue(solver.vbd_solver.__class__.__module__.endswith(".vbd_soft.solver_vbd"))
        solver.step(state_in, state_out, model.control(), None, 1.0 / 120.0)
        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))

    def test_kinematic_soft_backend_matches_original_mjvbd(self):
        model, articulation = _build_kinematic_particle_model("cpu")
        reference_in = model.state()
        reference_out = model.state()
        v2_in = model.state()
        v2_out = model.state()
        for state in (reference_in, reference_out, v2_in, v2_out):
            newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        reference_out.particle_q.assign(reference_in.particle_q)
        reference_out.particle_qd.assign(reference_in.particle_qd)
        v2_out.particle_q.assign(v2_in.particle_q)
        v2_out.particle_qd.assign(v2_in.particle_qd)

        options = {"iterations": 2, "particle_enable_self_contact": False}
        reference = SolverMJVBD(
            model,
            rigid_mode="external",
            soft_contact_margin=0.01,
            vbd_options=options,
        )
        v2 = SolverMJVBDV2(
            model,
            mujoco_articulations=[articulation],
            joint_mode="kinematic",
            contact_mode="soft",
            collision_options={"soft_contact_margin": 0.01},
            vbd_options=options,
        )
        reference.step(reference_in, reference_out, model.control(), reference.contacts, 1.0 / 120.0)
        v2.step(v2_in, v2_out, model.control(), v2.contacts, 1.0 / 120.0)

        np.testing.assert_allclose(v2_out.particle_q.numpy(), reference_out.particle_q.numpy(), atol=1.0e-7)
        np.testing.assert_allclose(v2_out.particle_qd.numpy(), reference_out.particle_qd.numpy(), atol=1.0e-7)

    def test_kinematic_tetrahedron_uses_optimized_soft_backend(self):
        model, articulation = _build_kinematic_tet_model("cpu")
        state_in = model.state()
        state_out = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_out)
        state_out.particle_q.assign(state_in.particle_q)
        state_out.particle_qd.assign(state_in.particle_qd)
        solver = SolverMJVBDV2(
            model,
            mujoco_articulations=[articulation],
            joint_mode="kinematic",
            vbd_options={"iterations": 2},
            collision_options={"soft_contact_margin": 0.01},
        )

        self.assertEqual(solver.features.backend, "mjvbd_kinematic_soft")
        self.assertTrue(solver.features.tetrahedron_solve_enabled)
        solver.step(state_in, state_out, model.control(), solver.contacts, 1.0 / 120.0)
        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))

    def test_kinematic_model_with_free_body_uses_full_vbd_without_mujoco(self):
        model, free_body, link, _, articulation = _build_partition_model("cpu")
        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        solver = SolverMJVBDV2(
            model,
            mujoco_articulations=[articulation],
            joint_mode="kinematic",
            vbd_options={"iterations": 2},
            collision_options={"broad_phase": "nxn"},
        )

        self.assertEqual(solver.features.backend, "vbd_kinematic_full")
        self.assertIsNone(solver.mujoco_solver)
        self.assertGreater(float(solver.backend.view.body_inv_mass.numpy()[free_body]), 0.0)
        self.assertEqual(float(solver.backend.view.body_inv_mass.numpy()[link]), 0.0)
        initial_z = float(state_0.body_q.numpy()[free_body, 2])
        solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)
        self.assertLess(float(state_1.body_q.numpy()[free_body, 2]), initial_z)
        np.testing.assert_allclose(state_1.joint_q.numpy(), state_0.joint_q.numpy(), atol=0.0, rtol=0.0)

    def test_dynamic_joint_receives_no_vbd_contact_feedback(self):
        try:
            model, free_body, link, _, articulation = _build_partition_model("cpu")
            near_0 = model.state()
            near_1 = model.state()
            far_0 = model.state()
            far_1 = model.state()
            for state in (near_0, near_1, far_0, far_1):
                newton.eval_fk(model, model.joint_q, model.joint_qd, state)

            near_q = near_0.body_q.numpy()
            near_q[free_body] = near_q[link]
            near_q[free_body, 1] += 0.10
            near_0.body_q.assign(near_q)
            near_1.assign(near_0)
            far_q = far_0.body_q.numpy()
            far_q[free_body] = far_q[link]
            far_q[free_body, 1] += 5.0
            far_0.body_q.assign(far_q)
            far_1.assign(far_0)

            options = {
                "mujoco_articulations": [articulation],
                "vbd_options": {"iterations": 3},
                "mujoco_options": {"use_mujoco_cpu": True},
                "collision_options": {"broad_phase": "nxn"},
            }
            near_solver = SolverMJVBDV2(model, **options)
            far_solver = SolverMJVBDV2(model, **options)
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(f"MuJoCo is unavailable: {error}")

        control = model.control()
        for _ in range(3):
            near_0.clear_forces()
            far_0.clear_forces()
            near_solver.step(near_0, near_1, control, None, 1.0 / 120.0)
            far_solver.step(far_0, far_1, control, None, 1.0 / 120.0)
            near_0, near_1 = near_1, near_0
            far_0, far_1 = far_1, far_0

        np.testing.assert_allclose(near_0.joint_q.numpy(), far_0.joint_q.numpy(), atol=1.0e-7, rtol=0.0)
        np.testing.assert_allclose(near_0.joint_qd.numpy(), far_0.joint_qd.numpy(), atol=1.0e-7, rtol=0.0)
        self.assertFalse(
            np.allclose(
                near_0.body_q.numpy()[free_body, :3],
                far_0.body_q.numpy()[free_body, :3],
            )
        )
        for mapping in near_solver.backend._proxy_mappings:
            self.assertTrue(np.all(mapping.coupling_forces.numpy() == 0.0))

    def test_vbd_rigid_contact_changes_both_bodies(self):
        model, left, right = _build_two_rigid_body_model("cpu")
        state_0 = model.state()
        state_1 = model.state()
        velocity = state_0.body_qd.numpy()
        velocity[left, 0] = 0.25
        velocity[right, 0] = -0.25
        state_0.body_qd.assign(velocity)
        initial_velocity = velocity.copy()
        solver = SolverMJVBDV2(
            model,
            vbd_options={"iterations": 5},
            collision_options={"broad_phase": "nxn"},
        )

        solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)
        solved_velocity = state_1.body_qd.numpy()
        self.assertFalse(np.isclose(solved_velocity[left, 0], initial_velocity[left, 0]))
        self.assertFalse(np.isclose(solved_velocity[right, 0], initial_velocity[right, 0]))
        self.assertGreater(float(solved_velocity[right, 0] - solved_velocity[left, 0]), 0.0)

    def test_rigid_only_model_prunes_particle_modules(self):
        model, _ = _build_pure_vbd_model("cpu")
        solver = SolverMJVBDV2(model, vbd_options={"iterations": 1})

        self.assertTrue(solver.features.rigid_solve_enabled)
        self.assertFalse(solver.features.particle_solve_enabled)
        self.assertFalse(solver.features.triangle_solve_enabled)
        self.assertFalse(solver.features.bending_solve_enabled)
        self.assertFalse(solver.features.tetrahedron_solve_enabled)
        self.assertFalse(solver.features.spring_solve_enabled)
        self.assertFalse(hasattr(solver.vbd_solver, "particle_forces"))
        self.assertTrue(hasattr(solver.vbd_solver, "body_forces"))

    def test_particle_constraint_modules_are_pruned_by_scene(self):
        expected = {
            "triangle": (True, False),
            "tetrahedron": (False, True),
        }
        for element, (has_triangles, has_tetrahedra) in expected.items():
            with self.subTest(element=element):
                model = _build_particle_module_model("cpu", element)
                state_0 = model.state()
                state_1 = model.state()
                solver = SolverMJVBDV2(model, vbd_options={"iterations": 1})

                self.assertFalse(solver.features.rigid_solve_enabled)
                self.assertTrue(solver.features.particle_solve_enabled)
                self.assertEqual(solver.features.triangle_solve_enabled, has_triangles)
                self.assertFalse(solver.features.bending_solve_enabled)
                self.assertEqual(solver.features.tetrahedron_solve_enabled, has_tetrahedra)
                self.assertFalse(solver.features.spring_solve_enabled)
                self.assertTrue(hasattr(solver.vbd_solver, "particle_forces"))
                self.assertFalse(hasattr(solver.vbd_solver, "body_forces"))

                solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)
                self.assertTrue(np.all(np.isfinite(state_1.particle_q.numpy())))


if __name__ == "__main__":
    unittest.main()

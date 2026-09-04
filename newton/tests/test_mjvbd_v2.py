# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for SolverMJVBDV2 ownership and stepping."""

import unittest
import warnings
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjvbd_v2.ownership import resolve_ownership
from newton._src.solvers.mjvbd_v2.vbd_soft.solver_vbd import SolverVBD as SolverVBDSoft
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


def _build_static_ground_particle_model(device):
    """Build a particle colliding with a world shape without rigid bodies."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverMJVBDV2.register_custom_attributes(builder)
    builder.add_ground_plane()
    builder.add_particle(
        pos=wp.vec3(0.0, 0.0, 0.01),
        vel=wp.vec3(),
        mass=0.01,
        radius=0.02,
    )
    builder.color()
    return builder.finalize(device=device)


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


def _build_self_contact_cloth_model(device, cell_size):
    """Build a cloth whose spacing controls self-contact candidate activity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverMJVBDV2.register_custom_attributes(builder)
    builder.add_cloth_grid(
        pos=wp.vec3(),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=9,
        dim_y=9,
        cell_x=cell_size,
        cell_y=cell_size,
        mass=0.001,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
    )
    builder.color()
    return builder.finalize(device=device)


def _build_full_vbd_cloth_model(device):
    """Build a cloth beside a free body so MJVBDV2 selects full VBD."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverMJVBDV2.register_custom_attributes(builder)
    builder.add_cloth_grid(
        pos=wp.vec3(),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=9,
        dim_y=9,
        cell_x=0.02,
        cell_y=0.02,
        mass=0.001,
        tri_ke=1.0e3,
        tri_ka=1.0e3,
    )
    body = builder.add_body(
        xform=wp.transform(wp.vec3(5.0, 0.0, 0.0), wp.quat_identity()),
        mass=1.0,
    )
    builder.add_shape_sphere(body, radius=0.05)
    builder.color()
    return builder.finalize(device=device)


def _build_pneumatic_shell_builder(*, add_dynamic_body=False):
    """Build an unfinished sealed shell and optionally add a free rigid body."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    positions = (
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
    )
    for position in positions:
        builder.add_particle(position, wp.vec3(), 1.0)
    for triangle in ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)):
        builder.add_triangle(*triangle, tri_ke=2.0e3, tri_ka=2.0e3, tri_kd=5.0)

    handle = newton.solvers.add_pneumatic_cavity(
        builder,
        range(4),
        config=newton.solvers.PneumaticConfig(
            mode=newton.solvers.PneumaticMode.PRESCRIBED_GAUGE_PRESSURE,
            ambient_pressure=100_000.0,
            prescribed_gauge_pressure=2_000.0,
        ),
    )
    body = None
    if add_dynamic_body:
        body = builder.add_body(
            xform=wp.transform(wp.vec3(3.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_shape_box(
            body,
            hx=0.05,
            hy=0.05,
            hz=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(density=1_000.0),
        )
    return builder, handle, body


def _build_pneumatic_shell_model(device, *, add_dynamic_body=False):
    """Build a sealed tetrahedral shell and optionally add a free rigid body."""
    builder, handle, body = _build_pneumatic_shell_builder(add_dynamic_body=add_dynamic_body)
    builder.color()
    return builder.finalize(device=device), handle, body


def _build_multiworld_pneumatic_model(device):
    """Build two independent pneumatic shells for masked-reset tests."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    first, _, _ = _build_pneumatic_shell_builder()
    second, _, _ = _build_pneumatic_shell_builder()
    builder.add_world(first)
    builder.add_world(
        second,
        xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.color()
    return builder.finalize(device=device)


def _build_multiworld_external_body_model(device):
    """Build two external rigid-body worlds beside particles."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for x in (0.0, 2.0):
        world = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        world.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 1.0), wp.quat_identity()),
            mass=0.0,
        )
        world.add_particle(
            pos=wp.vec3(x, 0.0, 1.1),
            vel=wp.vec3(),
            mass=0.01,
        )
        builder.add_world(world)
    builder.color()
    return builder.finalize(device=device)


def _build_kinematic_pneumatic_model(device):
    """Build a prescribed joint beside a sealed particle shell."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverMJVBDV2.register_custom_attributes(builder)
    link = builder.add_link(label="kinematic_link")
    builder.add_shape_box(link, hx=0.05, hy=0.05, hz=0.05)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(3.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([joint])
    for position in (
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
    ):
        builder.add_particle(position, wp.vec3(), 1.0)
    for triangle in ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)):
        builder.add_triangle(*triangle, tri_ke=2.0e3, tri_ka=2.0e3, tri_kd=5.0)
    newton.solvers.add_pneumatic_cavity(builder, range(4))
    builder.color()
    return builder.finalize(device=device)


def _build_joint_chain_model(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverMJVBDV2.register_custom_attributes(builder)
    parent = builder.add_link(label="parent_link")
    child = builder.add_link(label="child_link")
    builder.add_shape_box(parent, hx=0.1, hy=0.03, hz=0.03)
    builder.add_shape_box(child, hx=0.1, hy=0.03, hz=0.03)
    root_joint = builder.add_joint_revolute(
        parent=-1,
        child=parent,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
    )
    child_joint = builder.add_joint_revolute(
        parent=parent,
        child=child,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
    )
    articulation = builder.articulation_count
    builder.add_articulation([root_joint, child_joint], label="joint_chain")
    builder.color()
    return builder.finalize(device=device), root_joint, child_joint, articulation


def _build_falling_articulation_model(device):
    """Build a free articulation above a static ground plane."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverMJVBDV2.register_custom_attributes(builder)
    builder.add_ground_plane()
    root = builder.add_link(label="falling_root")
    child = builder.add_link(label="falling_child")
    builder.add_shape_box(root, hx=0.1, hy=0.05, hz=0.05)
    builder.add_shape_box(child, hx=0.1, hy=0.05, hz=0.05)
    free_joint = builder.add_joint_free(
        child=root,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()),
    )
    hinge_joint = builder.add_joint_revolute(
        parent=root,
        child=child,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
    )
    articulation = builder.articulation_count
    builder.add_articulation([free_joint, hinge_joint], label="falling_articulation")
    builder.color()
    return builder.finalize(device=device), root, articulation


class TestMJVBDV2(unittest.TestCase):
    def test_backends_use_rod_joint_name(self):
        """Avoid the deprecated cable-joint alias in both VBD paths."""
        full_model, _, _, _, full_articulation = _build_partition_model("cpu")
        soft_model, soft_articulation = _build_kinematic_particle_model("cpu")

        for label, model, articulation in (
            ("full", full_model, full_articulation),
            ("soft", soft_model, soft_articulation),
        ):
            with self.subTest(backend=label), warnings.catch_warnings():
                warnings.filterwarnings(
                    "error",
                    message=r".*JointType\.CABLE.*",
                    category=DeprecationWarning,
                )
                SolverMJVBDV2(
                    model,
                    mujoco_articulations=[articulation],
                    joint_mode="kinematic",
                    vbd_options={"iterations": 1},
                )

    def test_full_contact_backend_uses_private_pipeline(self):
        """Keep the full-contact collision implementation inside MJVBDV2."""
        model, _ = _build_pure_vbd_model("cpu")
        solver = SolverMJVBDV2(model, contact_mode="full", vbd_options={"iterations": 1})

        self.assertEqual(
            type(solver.backend.pipeline).__module__,
            "newton._src.solvers.mjvbd_v2.full_contact_pipeline",
        )

    def test_ownership_partition(self):
        """Partition selected articulation links away from VBD bodies."""
        model, free_body, link, joint, articulation = _build_partition_model("cpu")
        ownership = resolve_ownership(
            model,
            mujoco_articulations=[articulation],
        )

        self.assertEqual(ownership.mujoco_joints, (joint,))
        self.assertEqual(ownership.mujoco_bodies, (link,))
        self.assertIn(free_body, ownership.vbd_bodies)
        self.assertTrue(ownership.has_vbd_dynamic_bodies)

    def test_partial_joint_tree_is_rejected(self):
        """Reject ancestor- or descendant-incomplete MuJoCo joint trees."""
        model, root_joint, child_joint, _ = _build_joint_chain_model("cpu")

        with self.assertRaisesRegex(ValueError, "closed joint tree"):
            resolve_ownership(model, mujoco_joints=[root_joint])
        with self.assertRaisesRegex(ValueError, "ancestor-closed"):
            resolve_ownership(model, mujoco_joints=[child_joint])

    def test_cpu_step_keeps_free_body_in_vbd(self):
        """Keep an unselected free body dynamic in VBD during a CPU step."""
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

        proxy_pipeline = solver.backend._proxy_collision_configs[("mujoco", "vbd")].pipeline
        self.assertEqual(type(proxy_pipeline).__module__, "newton._src.solvers.mjvbd_v2.full_contact_pipeline")
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

    def test_dynamic_joint_only_model_dispatches_to_pure_mujoco(self):
        """Dispatch a dynamic articulation-only model without constructing VBD."""
        model, _, _, articulation = _build_joint_chain_model("cpu")
        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_1)
        try:
            solver = SolverMJVBDV2(
                model,
                mujoco_articulations=[articulation],
                vbd_options={"iterations": 7},
                mujoco_options={"use_mujoco_cpu": True},
                collision_options={"broad_phase": "nxn"},
            )
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(f"MuJoCo is unavailable: {error}")

        self.assertEqual(solver.features.backend, "pure_mujoco")
        self.assertTrue(solver.features.mujoco_solve_enabled)
        self.assertFalse(solver.features.vbd_solve_enabled)
        self.assertIsNone(solver.vbd_solver)
        self.assertIsNotNone(solver.mujoco_solver)
        self.assertFalse(solver.mujoco_solver.enable_sleeping)
        self.assertEqual(solver.backend.coupled_solver.entry_names(), ("mujoco",))
        self.assertEqual(solver.backend.coupled_solver.view("mujoco").particle_count, 0)

        solver.step(state_0, state_1, model.control(), solver.contacts, 1.0 / 120.0)
        self.assertTrue(np.all(np.isfinite(state_1.joint_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_1.body_q.numpy())))

    def test_pure_mujoco_resolves_ground_contacts(self):
        """Keep a pure-MuJoCo articulation above its static ground plane."""
        model, root, articulation = _build_falling_articulation_model("cpu")
        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_1)
        try:
            solver = SolverMJVBDV2(
                model,
                mujoco_articulations=[articulation],
                mujoco_options={"use_mujoco_cpu": True},
            )
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(f"MuJoCo is unavailable: {error}")

        self.assertEqual(solver.features.backend, "pure_mujoco")
        for _ in range(120):
            state_0.clear_forces()
            solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0

        self.assertGreater(float(state_0.body_q.numpy()[root, 2]), 0.04)

    @unittest.skipUnless(wp.is_cuda_available(), "MuJoCo sleeping requires CUDA")
    def test_dynamic_joint_only_model_can_enable_sleeping(self):
        """Enable sleeping only on the private dynamic MuJoCo backend."""
        model, _, _, articulation = _build_joint_chain_model("cuda:0")
        state_0 = model.state()
        state_1 = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_1)
        try:
            solver = SolverMJVBDV2(
                model,
                mujoco_articulations=[articulation],
                mujoco_options={"enable_sleeping": True},
                collision_options={"broad_phase": "nxn"},
            )
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(f"MuJoCo is unavailable: {error}")

        self.assertEqual(solver.features.backend, "pure_mujoco")
        self.assertTrue(solver.mujoco_solver.enable_sleeping)
        solver.step(state_0, state_1, model.control(), solver.contacts, 1.0 / 120.0)
        self.assertTrue(np.all(np.isfinite(state_1.joint_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_1.body_q.numpy())))

    def test_coupled_backend_rejects_sleeping(self):
        """Reject sleeping when VBD contacts cannot wake MuJoCo bodies."""
        model = _build_kinematic_pneumatic_model("cpu")

        with self.assertRaisesRegex(ValueError, "VBD contacts cannot wake MuJoCo bodies"):
            SolverMJVBDV2(
                model,
                mujoco_options={"enable_sleeping": True, "use_mujoco_cpu": True},
                vbd_options={"iterations": 1},
            )

    def test_kinematic_joint_only_model_dispatches_to_passthrough(self):
        """Preserve externally authored output in a kinematic joint-only scene."""
        model, _, _, articulation = _build_joint_chain_model("cpu")
        state_in = model.state()
        state_out = model.state()
        target_q = state_out.joint_q.numpy()
        target_q[:] = (0.15, -0.25)
        state_out.joint_q.assign(target_q)
        newton.eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)
        expected_joint_q = state_out.joint_q.numpy().copy()
        expected_body_q = state_out.body_q.numpy().copy()
        solver = SolverMJVBDV2(
            model,
            mujoco_articulations=[articulation],
            joint_mode="kinematic",
            vbd_options={"iterations": 7},
            collision_options={"broad_phase": "nxn"},
        )

        self.assertEqual(solver.features.backend, "kinematic_passthrough")
        self.assertFalse(solver.features.mujoco_solve_enabled)
        self.assertFalse(solver.features.vbd_solve_enabled)
        self.assertIsNone(solver.mujoco_solver)
        self.assertIsNone(solver.vbd_solver)
        solver.step(state_in, state_out, model.control(), solver.contacts, 1.0 / 120.0)
        np.testing.assert_array_equal(state_out.joint_q.numpy(), expected_joint_q)
        np.testing.assert_array_equal(state_out.body_q.numpy(), expected_body_q)

    def test_no_joint_model_dispatches_to_pure_vbd(self):
        """Dispatch a joint-free rigid model directly to VBD."""
        model, body = _build_pure_vbd_model("cpu")
        state_0 = model.state()
        state_1 = model.state()
        solver = SolverMJVBDV2(
            model,
            vbd_options={"iterations": 2},
            collision_options={"broad_phase": "nxn"},
        )

        self.assertEqual(solver.features.backend, "pure_vbd")
        self.assertFalse(solver.features.mujoco_solve_enabled)
        self.assertTrue(solver.features.vbd_solve_enabled)
        self.assertEqual(solver.features.mujoco_joint_count, 0)
        self.assertIsNone(solver.mujoco_solver)
        initial_z = float(state_0.body_q.numpy()[body, 2])
        solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)
        self.assertLess(float(state_1.body_q.numpy()[body, 2]), initial_z)

    def test_kinematic_particle_model_dispatches_to_mjvbd_fast_path(self):
        """Dispatch kinematic links and particles to the optimized soft path."""
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
        """Use optimized external-rigid VBD for dynamic joints and particles."""
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
        self.assertEqual(type(solver.backend).__name__, "SolverMJVBDV2")
        self.assertTrue(solver.vbd_solver.__class__.__module__.endswith(".vbd_soft.solver_vbd"))
        for mapping in solver.backend._proxy_mappings:
            self.assertIsNone(mapping.coupling_forces_previous)
            self.assertIsNone(mapping.proxy_qd_before)
        solver.step(state_in, state_out, model.control(), None, 1.0 / 120.0)
        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))

    def test_kinematic_soft_backend_matches_original_mjvbd(self):
        """Match the original MJVBD particle result on the kinematic soft path."""
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
        """Retain tetrahedron solving on the optimized kinematic soft path."""
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
        """Use full VBD for kinematic links plus a dynamic free body."""
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
        self.assertEqual(
            type(solver.backend.pipeline).__module__,
            "newton._src.solvers.mjvbd_v2.full_contact_pipeline",
        )
        self.assertIsNone(solver.mujoco_solver)
        self.assertGreater(float(solver.backend.view.body_inv_mass.numpy()[free_body]), 0.0)
        self.assertEqual(float(solver.backend.view.body_inv_mass.numpy()[link]), 0.0)
        initial_z = float(state_0.body_q.numpy()[free_body, 2])
        solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)
        self.assertLess(float(state_1.body_q.numpy()[free_body, 2]), initial_z)
        np.testing.assert_allclose(state_1.joint_q.numpy(), state_0.joint_q.numpy(), atol=0.0, rtol=0.0)

    def test_dynamic_joint_receives_no_vbd_contact_feedback(self):
        """Prevent VBD contacts from feeding back into dynamic joints."""
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
        """Change both VBD rigid bodies during a bidirectional collision."""
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
        """Prune all particle constraint modules from a rigid-only model."""
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
        """Enable only particle constraint modules present in each scene."""
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

    def test_soft_only_particle_collides_with_world_shape(self):
        """Resolve a static-world contact when no rigid-body state exists."""
        model = _build_static_ground_particle_model("cpu")
        state_0 = model.state()
        state_1 = model.state()
        solver = SolverMJVBDV2(model, vbd_options={"iterations": 2})

        self.assertEqual(model.body_count, 0)
        self.assertIsNone(state_0.body_q)
        self.assertTrue(solver.vbd_solver.__class__.__module__.endswith(".vbd_soft.solver_vbd"))
        solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)

        self.assertEqual(int(solver.contacts.soft_contact_count.numpy()[0]), 1)
        self.assertTrue(np.all(np.isfinite(state_1.particle_q.numpy())))

    def test_device_soft_contact_material_source_selects_runtime_row(self):
        """Select cached soft-contact materials through a device row index."""
        model = _build_static_ground_particle_model("cpu")
        state_0 = model.state()
        state_1 = model.state()
        solver = SolverMJVBDV2(model, vbd_options={"iterations": 1})
        materials = wp.array(
            [wp.vec3(100.0, 1.0, 0.1), wp.vec3(900.0, 3.0, 0.9)],
            dtype=wp.vec3,
            device=model.device,
        )
        material_index = wp.zeros(1, dtype=wp.int32, device=model.device)
        solver.vbd_solver.set_soft_contact_material_source(materials, material_index)

        solver.step(state_0, state_1, model.control(), None, 1.0 / 120.0)
        shape = int(solver.contacts.soft_contact_shape.numpy()[0])
        shape_ke = float(model.shape_material_ke.numpy()[shape])
        shape_kd = float(model.shape_material_kd.numpy()[shape])
        shape_mu = float(model.shape_material_mu.numpy()[shape])
        self.assertAlmostEqual(
            float(solver.vbd_solver.body_particle_contact_material_ke.numpy()[0]), 0.5 * (100.0 + shape_ke)
        )
        self.assertAlmostEqual(
            float(solver.vbd_solver.body_particle_contact_material_kd.numpy()[0]), 0.5 * (1.0 + shape_kd)
        )
        self.assertAlmostEqual(
            float(solver.vbd_solver.body_particle_contact_material_mu.numpy()[0]), np.sqrt(0.1 * shape_mu)
        )

        material_index.fill_(1)
        solver.step(state_1, state_0, model.control(), None, 1.0 / 120.0)
        self.assertAlmostEqual(
            float(solver.vbd_solver.body_particle_contact_material_ke.numpy()[0]), 0.5 * (900.0 + shape_ke)
        )
        self.assertAlmostEqual(
            float(solver.vbd_solver.body_particle_contact_material_kd.numpy()[0]), 0.5 * (3.0 + shape_kd)
        )
        self.assertAlmostEqual(
            float(solver.vbd_solver.body_particle_contact_material_mu.numpy()[0]), np.sqrt(0.9 * shape_mu)
        )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_device_soft_contact_material_source_updates_one_cuda_graph(self):
        """Update cached soft-contact materials while replaying one CUDA graph."""
        model = _build_static_ground_particle_model("cuda:0")
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        solver = SolverMJVBDV2(model, vbd_options={"iterations": 1})
        materials = wp.array(
            [wp.vec3(100.0, 1.0, 0.1), wp.vec3(900.0, 3.0, 0.9)],
            dtype=wp.vec3,
            device=model.device,
        )
        material_index = wp.zeros(1, dtype=wp.int32, device=model.device)
        solver.vbd_solver.set_soft_contact_material_source(materials, material_index)

        solver.step(state_0, state_1, control, None, 1.0 / 120.0)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(state_1, state_0, control, None, 1.0 / 120.0)

        shape = int(solver.contacts.soft_contact_shape.numpy()[0])
        shape_ke = float(model.shape_material_ke.numpy()[shape])
        material_index.fill_(1)
        wp.capture_launch(capture.graph)
        self.assertAlmostEqual(
            float(solver.vbd_solver.body_particle_contact_material_ke.numpy()[0]),
            0.5 * (900.0 + shape_ke),
        )

        material_index.fill_(0)
        wp.capture_launch(capture.graph)
        self.assertAlmostEqual(
            float(solver.vbd_solver.body_particle_contact_material_ke.numpy()[0]),
            0.5 * (100.0 + shape_ke),
        )

    @unittest.skipUnless(wp.is_cuda_available(), "Tiled VBD requires CUDA")
    def test_small_particle_color_groups_use_scalar_solve(self):
        """Use the scalar elasticity kernel for undersubscribed CUDA color groups."""
        model = _build_particle_module_model("cuda:0", "triangle")
        state_0 = model.state()
        state_1 = model.state()
        solver = SolverMJVBDV2(model, vbd_options={"iterations": 2})

        self.assertTrue(solver.vbd_solver.__class__.__module__.endswith(".vbd_soft.solver_vbd"))
        self.assertFalse(solver.vbd_solver.use_particle_tile_solve)
        solver.step(state_0, state_1, model.control(), None, 1.0e-4)
        self.assertTrue(np.all(np.isfinite(state_1.particle_q.numpy())))

    def test_particle_output_is_copied_once_per_step(self):
        """Copy solved particle positions to the output only after all VBD iterations."""
        soft_model = _build_particle_module_model("cpu", "triangle")
        full_model, _, _ = _build_pneumatic_shell_model("cpu")
        cases = (
            ("soft", soft_model, SolverVBDSoft(soft_model, iterations=4)),
            ("full", full_model, SolverMJVBDV2(full_model, vbd_options={"iterations": 4}).vbd_solver),
        )

        for label, model, solver in cases:
            with self.subTest(backend=label):
                state_in = model.state()
                state_out = model.state()
                original_copy = wp.copy
                output_copy_count = 0

                def count_output_copy(
                    dest,
                    src,
                    *args,
                    state_out_q=state_out.particle_q,
                    state_in_q=state_in.particle_q,
                    copy_fn=original_copy,
                    **kwargs,
                ):
                    nonlocal output_copy_count
                    if dest is state_out_q and src is state_in_q:
                        output_copy_count += 1
                    return copy_fn(dest, src, *args, **kwargs)

                with mock.patch.object(wp, "copy", side_effect=count_output_copy):
                    solver.step(state_in, state_out, model.control(), None, 1.0e-3)

                self.assertEqual(output_copy_count, 1)
                np.testing.assert_allclose(state_out.particle_q.numpy(), state_in.particle_q.numpy(), atol=0.0)

    def test_full_vbd_builds_surface_elasticity_groups(self):
        """Partition a pneumatic shell into surface-only elasticity groups."""
        model, _, _ = _build_pneumatic_shell_model("cpu")
        solver = SolverMJVBDV2(model, vbd_options={"iterations": 1})

        self.assertTrue(solver.vbd_solver.__class__.__module__.endswith(".vbd.solver_vbd"))
        self.assertEqual(
            sum(group.size for group in solver.vbd_solver.surface_particle_color_groups), model.particle_count
        )
        self.assertEqual(sum(group.size for group in solver.vbd_solver.volumetric_particle_color_groups), 0)

    @unittest.skipUnless(wp.is_cuda_available(), "Tiled VBD requires CUDA")
    def test_full_vbd_uses_surface_tile_solve(self):
        """Run the surface-specialized tile path for full VBD cloth scenes."""
        model = _build_full_vbd_cloth_model("cuda:0")
        state_in = model.state()
        state_out = model.state()
        solver = SolverMJVBDV2(model, vbd_options={"iterations": 2})

        self.assertEqual(solver.features.backend, "pure_vbd")
        self.assertTrue(solver.vbd_solver.use_particle_tile_solve)
        self.assertEqual(
            sum(group.size for group in solver.vbd_solver.surface_particle_color_groups), model.particle_count
        )
        self.assertEqual(sum(group.size for group in solver.vbd_solver.volumetric_particle_color_groups), 0)
        solver.step(state_in, state_out, model.control(), None, 1.0e-3)
        self.assertTrue(np.all(np.isfinite(state_out.particle_q.numpy())))

    def test_pneumatic_pressure_uses_pure_vbd(self):
        """Expand a sealed shell without entering MuJoCo or tetrahedron paths."""
        model, handle, _ = _build_pneumatic_shell_model("cpu")
        state_0 = model.state()
        state_1 = model.state()
        solver = SolverMJVBDV2(model, vbd_options={"iterations": 4})

        self.assertEqual(solver.features.backend, "pure_vbd")
        self.assertFalse(solver.features.mujoco_solve_enabled)
        self.assertTrue(solver.features.pneumatic_solve_enabled)
        self.assertFalse(solver.features.tetrahedron_solve_enabled)
        self.assertEqual(solver.features.pneumatic_cavity_count, 1)
        self.assertTrue(solver.vbd_solver._pneumatic_enabled)

        solver.step(state_0, state_1, model.control(), None, 1.0e-3)

        volume = state_1.pneumatic.volume.numpy()[handle.cavity_index]
        pressure = state_1.pneumatic.absolute_pressure.numpy()[handle.cavity_index]
        self.assertGreater(volume, handle.rest_volume)
        self.assertAlmostEqual(float(pressure), 102_000.0, places=2)
        self.assertTrue(np.all(np.isfinite(state_1.particle_q.numpy())))

    def test_pneumatic_reset_restores_cavity_history(self):
        """Restore pneumatic observables and previous-volume history on reset."""
        model, handle, _ = _build_pneumatic_shell_model("cpu")
        state_0 = model.state()
        state_1 = model.state()
        solver = SolverMJVBDV2(model, vbd_options={"iterations": 4})

        solver.step(state_0, state_1, model.control(), None, 1.0e-3)
        self.assertGreater(float(state_1.pneumatic.volume.numpy()[handle.cavity_index]), handle.rest_volume)
        solver.reset(state_1, flags=0)

        self.assertAlmostEqual(
            float(state_1.pneumatic.volume.numpy()[handle.cavity_index]),
            handle.rest_volume,
            places=6,
        )
        self.assertAlmostEqual(
            float(state_1.pneumatic.absolute_pressure.numpy()[handle.cavity_index]),
            float(model.pneumatic.reference_absolute_pressure.numpy()[handle.cavity_index]),
            places=3,
        )
        self.assertEqual(float(state_1.pneumatic.volume_rate.numpy()[handle.cavity_index]), 0.0)
        self.assertEqual(int(state_1.pneumatic.clamp_flags.numpy()[handle.cavity_index]), 0)
        self.assertAlmostEqual(
            float(solver.vbd_solver._pneumatic_previous_volume.numpy()[handle.cavity_index]),
            handle.rest_volume,
            places=6,
        )

    def test_pneumatic_reset_honors_world_mask(self):
        """Reset only pneumatic cavities selected by the world mask."""
        model = _build_multiworld_pneumatic_model("cpu")
        state_0 = model.state()
        state_1 = model.state()
        solver = SolverMJVBDV2(model, vbd_options={"iterations": 4})
        rest_volume = state_0.pneumatic.volume.numpy().copy()

        solver.step(state_0, state_1, model.control(), None, 1.0e-3)
        stepped_volume = state_1.pneumatic.volume.numpy().copy()
        self.assertTrue(np.all(np.abs(stepped_volume - rest_volume) > 1.0e-7))
        solver.reset(
            state_1,
            world_mask=wp.array([True, False], dtype=wp.bool, device="cpu"),
            flags=0,
        )

        reset_volume = state_1.pneumatic.volume.numpy()
        self.assertAlmostEqual(float(reset_volume[0]), float(rest_volume[0]), places=6)
        self.assertAlmostEqual(float(reset_volume[1]), float(stepped_volume[1]), places=6)
        np.testing.assert_allclose(
            solver.vbd_solver._pneumatic_previous_volume.numpy(),
            reset_volume,
            atol=1.0e-7,
        )

    def test_pneumatic_reset_masks_global_cavity(self):
        """Reset a global cavity only through the optional global mask slot."""
        model, handle, _ = _build_pneumatic_shell_model("cpu")
        state_0 = model.state()
        state_1 = model.state()
        solver = SolverMJVBDV2(model, vbd_options={"iterations": 4})

        solver.step(state_0, state_1, model.control(), None, 1.0e-3)
        stepped_volume = float(state_1.pneumatic.volume.numpy()[handle.cavity_index])
        solver.reset(
            state_1,
            world_mask=wp.array([True], dtype=wp.bool, device="cpu"),
            flags=0,
        )
        self.assertAlmostEqual(
            float(state_1.pneumatic.volume.numpy()[handle.cavity_index]),
            stepped_volume,
            places=6,
        )

        solver.reset(
            state_1,
            world_mask=wp.array([False, True], dtype=wp.bool, device="cpu"),
            flags=0,
        )
        self.assertAlmostEqual(
            float(state_1.pneumatic.volume.numpy()[handle.cavity_index]),
            handle.rest_volume,
            places=6,
        )

    def test_external_body_history_reset_honors_world_mask(self):
        """Rebaseline external body history only in selected worlds."""
        model = _build_multiworld_external_body_model("cpu")
        state = model.state()
        solver = SolverVBDSoft(
            model,
            iterations=1,
            integrate_with_external_rigid_solver=True,
            external_rigid_state_from_input=True,
        )
        history = solver._external_body_q_prev.numpy()
        history[:, 2] = (-1.0, -2.0)
        solver._external_body_q_prev.assign(history)
        current = state.body_q.numpy()
        current[:, 2] = (1.0, 2.0)
        state.body_q.assign(current)

        solver.reset(
            state,
            world_mask=wp.array([True, False], dtype=wp.bool, device="cpu"),
            flags=0,
        )

        np.testing.assert_array_equal(
            solver._external_body_q_prev.numpy()[:, 2],
            np.array([1.0, -2.0], dtype=np.float32),
        )

    def test_pneumatic_rigid_scene_prunes_unneeded_solvers(self):
        """Keep a pneumatic cloth-and-rigid scene on the full pure-VBD backend."""
        model, _, body = _build_pneumatic_shell_model("cpu", add_dynamic_body=True)
        solver = SolverMJVBDV2(model, vbd_options={"iterations": 1})

        self.assertIsNotNone(body)
        self.assertEqual(solver.features.backend, "pure_vbd")
        self.assertIsNone(solver.mujoco_solver)
        self.assertTrue(solver.features.rigid_solve_enabled)
        self.assertTrue(solver.features.triangle_solve_enabled)
        self.assertFalse(solver.features.tetrahedron_solve_enabled)
        self.assertTrue(solver.features.pneumatic_solve_enabled)

    def test_non_pneumatic_scene_allocates_no_cavity_state(self):
        """Leave the pneumatic module completely dormant for ordinary VBD scenes."""
        model = _build_particle_module_model("cpu", "triangle")
        solver = SolverMJVBDV2(model, vbd_options={"iterations": 1})

        self.assertNotIn("pneumatic:cavity", model.custom_frequency_counts)
        self.assertFalse(solver.features.pneumatic_solve_enabled)
        self.assertEqual(solver.features.pneumatic_cavity_count, 0)
        self.assertTrue(solver.vbd_solver.__class__.__module__.endswith(".vbd_soft.solver_vbd"))
        self.assertFalse(hasattr(solver.vbd_solver, "_pneumatic_enabled"))
        self.assertFalse(hasattr(solver.vbd_solver, "_pneumatic_volume"))
        self.assertFalse(hasattr(solver.vbd_solver, "_pneumatic_kernels"))
        self.assertNotIn(
            "pneumatic_kernels", {module.__name__.rsplit(".", 1)[-1] for module in solver.vbd_solver._module_options}
        )

    def test_self_contact_activity_stays_on_device(self):
        """Select active and inactive self-contact branches without a host readback."""
        for cell_size, expected_active in ((0.01, 1), (0.1, 0)):
            with self.subTest(cell_size=cell_size):
                model = _build_self_contact_cloth_model("cpu", cell_size)
                state_0 = model.state()
                state_1 = model.state()
                solver = SolverMJVBDV2(
                    model,
                    vbd_options={
                        "iterations": 2,
                        "particle_enable_self_contact": True,
                        "particle_self_contact_radius": 0.02,
                        "particle_self_contact_margin": 0.03,
                    },
                )

                solver.step(state_0, state_1, model.control(), None, 1.0e-4)

                self.assertEqual(int(solver.vbd_solver.has_active_self_contact.numpy()[0]), expected_active)
                self.assertTrue(np.all(np.isfinite(state_1.particle_q.numpy())))

    def test_kinematic_soft_contact_uses_full_vbd_for_pneumatics(self):
        """Select full VBD pressure kernels while retaining soft contact generation."""
        model = _build_kinematic_pneumatic_model("cpu")
        solver = SolverMJVBDV2(
            model,
            joint_mode="kinematic",
            contact_mode="soft",
            vbd_options={"iterations": 1},
        )

        self.assertEqual(solver.features.backend, "mjvbd_kinematic_soft")
        self.assertTrue(solver.features.pneumatic_solve_enabled)
        self.assertTrue(solver.vbd_solver._pneumatic_enabled)
        self.assertTrue(solver.vbd_solver.__class__.__module__.endswith(".vbd.solver_vbd"))

        state_0 = model.state()
        state_1 = model.state()
        solver.step(state_0, state_1, model.control(), None, 1.0e-3)
        self.assertTrue(np.all(np.isfinite(state_1.pneumatic.volume.numpy())))

    def test_coupled_external_rigid_path_uses_full_vbd_for_pneumatics(self):
        """Synchronize pneumatic history beside externally integrated joint bodies."""
        model = _build_kinematic_pneumatic_model("cpu")
        try:
            solver = SolverMJVBDV2(
                model,
                mujoco_options={"use_mujoco_cpu": True},
                vbd_options={"iterations": 4},
            )
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(f"MuJoCo is unavailable: {error}")

        self.assertEqual(solver.features.backend, "coupled")
        self.assertEqual(type(solver.backend).__name__, "_SolverMJVBDV2Pneumatic")
        self.assertTrue(solver.features.pneumatic_solve_enabled)
        self.assertTrue(solver.vbd_solver._pneumatic_enabled)
        self.assertTrue(solver.vbd_solver.__class__.__module__.endswith(".vbd.solver_vbd"))

        state_0 = model.state()
        state_1 = model.state()
        joint_q = state_0.joint_q.numpy()
        joint_q[0] = 0.25
        state_0.joint_q.assign(joint_q)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        initial_external_pose = solver.vbd_solver._external_body_q_prev.numpy().copy()
        rest_volume = state_0.pneumatic.volume.numpy().copy()

        solver.step(state_0, state_1, model.control(), None, 1.0e-3)
        first_volume = state_1.pneumatic.volume.numpy().copy()
        vbd_input = solver.backend.entry_state("vbd", phase="input")
        vbd_output = solver.backend.entry_state("vbd", phase="output")
        np.testing.assert_allclose(first_volume, vbd_output.pneumatic.volume.numpy(), atol=1.0e-7)
        self.assertTrue(np.any(np.abs(first_volume - rest_volume) > 1.0e-7))
        np.testing.assert_allclose(
            solver.vbd_solver._external_body_q_prev.numpy(),
            vbd_input.body_q.numpy(),
            atol=1.0e-7,
        )
        self.assertFalse(np.allclose(initial_external_pose, solver.vbd_solver._external_body_q_prev.numpy()))

        state_0, state_1 = state_1, state_0
        solver.step(state_0, state_1, model.control(), None, 1.0e-3)
        np.testing.assert_allclose(
            solver.backend.entry_state("vbd", phase="input").pneumatic.volume.numpy(),
            first_volume,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            state_1.pneumatic.volume.numpy(),
            solver.backend.entry_state("vbd", phase="output").pneumatic.volume.numpy(),
            atol=1.0e-7,
        )

        solver.reset(state_1, flags=0)
        np.testing.assert_allclose(state_1.pneumatic.volume.numpy(), rest_volume, atol=1.0e-7)

    def test_open_pneumatic_surface_is_rejected(self):
        """Reject pneumatic faces that do not form a closed two-manifold shell."""
        builder = newton.ModelBuilder()
        for position in (
            wp.vec3(),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
            wp.vec3(-1.0, 0.0, 0.0),
            wp.vec3(0.0, -1.0, 0.0),
        ):
            builder.add_particle(position, wp.vec3(), 1.0)
        for triangle in ((0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1)):
            builder.add_triangle(*triangle)

        with self.assertRaisesRegex(ValueError, "closed two-manifold"):
            newton.solvers.add_pneumatic_cavity(builder, range(4))


if __name__ == "__main__":
    unittest.main()

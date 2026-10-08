# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-way MuJoCo/VBD coupling in SolverMJVBDV2."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMJVBDV2

TABLE_TOP = 0.5
LINK_HALF = 0.2
PIVOT_HEIGHT = 0.25
TARGET = 0.9
# Two-way feedback reaches MuJoCo one substep late, so stiff contacts need short substeps.
DT = 1.0 / 600.0
# Realistic contact stiffness; the 2.5e3 N/m default lets loaded contacts sink by centimeters.
CONTACT_KE = 2.0e4


def _shape_cfg(density=0.0):
    return newton.ModelBuilder.ShapeConfig(density=density, ke=CONTACT_KE, kd=100.0)


def _device():
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def _mujoco_options(device):
    return {"use_mujoco_cpu": wp.get_device(device).is_cpu}


def _tip_height(body_q, link):
    """Height [m] of the lowest leading corner of the swinging link above the table."""
    transform = wp.transform(*body_q[link])
    return float(wp.transform_point(transform, wp.vec3(LINK_HALF, 0.0, -0.02))[2]) - TABLE_TOP


def _build_arm_model(device, obstacle):
    """Build a PD-driven revolute link that swings down into a static table.

    ``obstacle`` selects what VBD owns: ``"none"`` keeps only the robot and the
    table, ``"far_particle"`` adds one distant particle so that the coupled
    backend is selected without touching the link, and ``"rigid"`` and
    ``"soft"`` place a VBD box or tetrahedral block under the link.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverMJVBDV2.register_custom_attributes(builder)
    builder.add_shape_box(
        -1,
        xform=wp.transform(wp.vec3(0.2, 0.0, TABLE_TOP - 0.05), wp.quat_identity()),
        hx=0.6,
        hy=0.3,
        hz=0.05,
        cfg=_shape_cfg(),
    )
    link = builder.add_link(label="arm")
    builder.add_shape_box(link, hx=LINK_HALF, hy=0.03, hz=0.02, cfg=_shape_cfg(500.0))
    # Positive rotation about +Y swings the +X tip downward into the table.
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, TABLE_TOP + PIVOT_HEIGHT), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-LINK_HALF, 0.0, 0.0), wp.quat_identity()),
        target_ke=40.0,
        target_kd=2.0,
        armature=0.01,
    )
    builder.add_articulation([joint])
    box = None
    if obstacle == "far_particle":
        builder.add_particle(pos=wp.vec3(5.0, 5.0, TABLE_TOP + 0.01), vel=wp.vec3(), mass=0.01, radius=0.01)
    elif obstacle == "rigid":
        box = builder.add_body(xform=wp.transform(wp.vec3(0.3, 0.0, TABLE_TOP + 0.05), wp.quat_identity()))
        builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=_shape_cfg(500.0))
    elif obstacle == "soft":
        builder.add_soft_grid(
            pos=wp.vec3(0.25, -0.05, TABLE_TOP + 0.005),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=4,
            dim_y=4,
            dim_z=4,
            cell_x=0.025,
            cell_y=0.025,
            cell_z=0.025,
            density=300.0,
            k_mu=2.0e4,
            k_lambda=2.0e4,
            k_damp=1.0e-3,
            particle_radius=0.005,
        )
    elif obstacle != "none":
        raise ValueError(obstacle)
    builder.color()
    return builder.finalize(device=device), link, box


def _simulate_arm(obstacle, coupling, *, steps=900):
    """Drive the link toward a target inside the obstacle and return the final state."""
    device = _device()
    model, link, box = _build_arm_model(device, obstacle)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_1)
    control = model.control()
    control.joint_target_q.assign(np.array([TARGET], dtype=np.float32))
    solver = SolverMJVBDV2(
        model,
        coupling=coupling,
        vbd_options={"iterations": 10},
        mujoco_options=_mujoco_options(device),
        collision_options={"broad_phase": "nxn"} if obstacle == "rigid" else {"soft_contact_margin": 0.01},
    )
    lowest_tip = np.inf
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
        lowest_tip = min(lowest_tip, _tip_height(state_0.body_q.numpy(), link))
    return solver, state_0, link, box, lowest_tip


class TestMJVBDV2TwoWayConfiguration(unittest.TestCase):
    def test_rejects_invalid_coupling_configuration(self):
        """Reject two-way options that cannot apply to the requested scene."""
        model, _, _ = _build_arm_model("cpu", "far_particle")
        with self.assertRaisesRegex(ValueError, "coupling must be"):
            SolverMJVBDV2(model, coupling="mutual")
        with self.assertRaisesRegex(ValueError, "requires joint_mode='dynamic'"):
            SolverMJVBDV2(model, joint_mode="kinematic", coupling="two_way")
        with self.assertRaisesRegex(ValueError, "only used with coupling='two_way'"):
            SolverMJVBDV2(model, coupling_options={"iterations": 2})
        with self.assertRaisesRegex(ValueError, "Unsupported coupling_options"):
            SolverMJVBDV2(model, coupling="two_way", coupling_options={"relaxation": 0.5})

    def test_rejects_full_surface_contacts(self):
        """Reject full-surface rigid-soft contacts whose wrenches are not harvested."""
        model, _, _ = _build_arm_model("cpu", "soft")
        try:
            with self.assertRaisesRegex(ValueError, "full-surface"):
                SolverMJVBDV2(
                    model,
                    coupling="two_way",
                    contact_mode="full",
                    mujoco_options={"use_mujoco_cpu": True},
                    collision_options={"enable_rigid_soft_full_surface_contact": True},
                )
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(f"MuJoCo is unavailable: {error}")

    def test_two_way_wires_feedback_and_mujoco_contacts(self):
        """Give proxies MuJoCo inertia, harvest momentum, and keep MuJoCo contacts."""
        try:
            import mujoco

            model, link, _ = _build_arm_model("cpu", "far_particle")
            one_way = SolverMJVBDV2(model, mujoco_options={"use_mujoco_cpu": True})
            two_way = SolverMJVBDV2(model, coupling="two_way", mujoco_options={"use_mujoco_cpu": True})
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(f"MuJoCo is unavailable: {error}")

        contact_disabled = int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        self.assertEqual(one_way.features.backend, "coupled")
        self.assertFalse(one_way.features.two_way_coupling_enabled)
        self.assertTrue(one_way.mujoco_solver.mj_model.opt.disableflags & contact_disabled)
        self.assertEqual(float(one_way.backend._entries["vbd"].view.body_inv_mass.numpy()[link]), 0.0)

        self.assertEqual(two_way.features.backend, "coupled")
        self.assertTrue(two_way.features.two_way_coupling_enabled)
        self.assertFalse(two_way.mujoco_solver.mj_model.opt.disableflags & contact_disabled)
        self.assertGreater(float(two_way.backend._entries["vbd"].view.body_inv_mass.numpy()[link]), 0.0)
        vbd = two_way.vbd_solver
        self.assertTrue(type(vbd).__module__.endswith(".mjvbd_v2.vbd.solver_vbd"))
        self.assertFalse(vbd.one_way_proxy_bodies)
        self.assertFalse(vbd.integrate_with_external_rigid_solver)
        self.assertEqual(vbd.proxy_body_feedback, "momentum")

    def test_vbd_rejects_unknown_proxy_feedback(self):
        """Validate the private VBD proxy feedback mode."""
        from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD  # noqa: PLC0415

        model, _, _ = _build_arm_model("cpu", "rigid")
        with self.assertRaisesRegex(ValueError, "proxy_body_feedback"):
            SolverVBD(model, proxy_body_feedback="impulse")


class TestMJVBDV2TwoWayDynamics(unittest.TestCase):
    def _simulate(self, obstacle, coupling, **kwargs):
        try:
            return _simulate_arm(obstacle, coupling, **kwargs)
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(f"MuJoCo is unavailable: {error}")

    def test_link_stops_on_static_table_like_pure_mujoco(self):
        """Resolve link-vs-table contact in MuJoCo exactly as the pure MuJoCo backend does."""
        reference, reference_state, _, _, reference_tip = self._simulate("none", "two_way")
        _, two_way_state, link, _, two_way_tip = self._simulate("far_particle", "two_way")
        _, one_way_state, _, _, one_way_tip = self._simulate("far_particle", "one_way")

        self.assertEqual(reference.features.backend, "pure_mujoco")
        np.testing.assert_allclose(two_way_state.joint_q.numpy(), reference_state.joint_q.numpy(), atol=1.0e-3)
        self.assertAlmostEqual(two_way_tip, reference_tip, delta=1.0e-3)
        self.assertGreater(_tip_height(two_way_state.body_q.numpy(), link), -0.005)
        # One-way coupling leaves MuJoCo contact-free, so the link reaches its target inside the table.
        self.assertGreater(float(one_way_state.joint_q.numpy()[0]), TARGET - 0.05)
        self.assertLess(one_way_tip, -0.05)

    def test_link_rests_on_vbd_rigid_box(self):
        """Stop a driven link on a VBD rigid box and keep the box on the table."""
        two_way, state, link, box, _ = self._simulate("rigid", "two_way")
        _, one_way_state, _, _, one_way_tip = self._simulate("rigid", "one_way")

        body_q = state.body_q.numpy()
        box_top = float(body_q[box, 2]) + 0.05 - TABLE_TOP
        # The link rests on the box's far top edge; its lower face is at local z = -0.02.
        edge_world = wp.transform_point(wp.transform(*body_q[box]), wp.vec3(0.05, 0.0, 0.05))
        edge = wp.transform_point(wp.transform_inverse(wp.transform(*body_q[link])), edge_world)
        self.assertLess(float(edge[2]), -0.015)
        np.testing.assert_allclose(body_q[box, :2], (0.3, 0.0), atol=0.02)
        self.assertGreater(box_top, 0.09)
        self.assertGreater(float(body_q[box, 2]) - 0.05, TABLE_TOP - 0.01)
        self.assertLess(float(state.joint_q.numpy()[0]), 0.5)
        self.assertGreater(float(np.abs(two_way.backend._proxy_mappings[0].coupling_forces.numpy()).max()), 0.0)
        self.assertGreater(float(one_way_state.joint_q.numpy()[0]), TARGET - 0.05)
        self.assertLess(one_way_tip, -0.05)

    def test_link_rests_on_vbd_soft_block(self):
        """Stop a driven link on a tetrahedral block without piercing it."""
        _, state, _, _, lowest_tip = self._simulate("soft", "two_way")
        _, one_way_state, _, _, one_way_tip = self._simulate("soft", "one_way")

        particles = state.particle_q.numpy()
        self.assertTrue(np.all(np.isfinite(particles)))
        self.assertGreater(float(particles[:, 2].min()), TABLE_TOP - 0.01)
        self.assertLess(float(state.joint_q.numpy()[0]), 0.5)
        self.assertGreater(lowest_tip, 0.03)
        self.assertGreater(float(one_way_state.joint_q.numpy()[0]), TARGET - 0.05)
        self.assertLess(one_way_tip, -0.05)


BOX_HALF = 0.025
FINGER_HALF = 0.01
FINGER_START = 0.08


def _build_gripper_model(device):
    """Build two armatured prismatic fingers on either side of a VBD box on the floor."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverMJVBDV2.register_custom_attributes(builder)
    builder.add_ground_plane(cfg=_shape_cfg())
    for sign in (-1.0, 1.0):
        finger = builder.add_link(label=f"finger_{int(sign)}")
        builder.add_shape_box(finger, hx=FINGER_HALF, hy=0.02, hz=0.015, cfg=_shape_cfg(500.0))
        joint = builder.add_joint_prismatic(
            parent=-1,
            child=finger,
            axis=wp.vec3(-sign, 0.0, 0.0),
            parent_xform=wp.transform(wp.vec3(sign * FINGER_START, 0.0, BOX_HALF + 0.005), wp.quat_identity()),
            limit_lower=0.0,
            limit_upper=0.1,
            target_ke=1000.0,
            target_kd=20.0,
            armature=0.3,
            effort_limit=10.0,
        )
        builder.add_articulation([joint])
    box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, BOX_HALF), wp.quat_identity()))
    builder.add_shape_box(
        box,
        hx=BOX_HALF,
        hy=BOX_HALF,
        hz=BOX_HALF,
        cfg=newton.ModelBuilder.ShapeConfig(density=500.0, ke=CONTACT_KE, kd=100.0, mu=1.0),
    )
    builder.color()
    return builder.finalize(device=device), box


class TestMJVBDV2TwoWayGrip(unittest.TestCase):
    def test_shut_fingers_stop_on_vbd_box(self):
        """Stop light, armatured fingers commanded to close through a VBD box on the floor."""
        device = _device()
        model, box = _build_gripper_model(device)
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_1)
        control = model.control()
        # Fully closed is 7 cm of travel per finger, past the box faces.
        closed = FINGER_START - FINGER_HALF
        control.joint_target_q.assign(np.full(2, closed, dtype=np.float32))
        try:
            solver = SolverMJVBDV2(
                model,
                coupling="two_way",
                vbd_options={"iterations": 10},
                mujoco_options=_mujoco_options(device),
                collision_options={"broad_phase": "nxn"},
            )
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(f"MuJoCo is unavailable: {error}")
        travel = []
        steps = int(round(2.0 / DT))
        for step in range(steps):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, DT)
            state_0, state_1 = state_1, state_0
            if step >= steps // 2:
                # The two prismatic coordinates precede the box's free joint.
                travel.append(state_0.joint_q.numpy()[:2].copy())
        travel = np.asarray(travel)
        body_q = state_0.body_q.numpy()
        self.assertTrue(np.all(np.isfinite(body_q)))
        # The box may settle off-center between equal finger forces, but each
        # finger face must rest on the box face it closes toward.
        box_x = float(body_q[box, 0])
        finger_faces = (-(FINGER_START - FINGER_HALF) + travel[-1, 0], FINGER_START - FINGER_HALF - travel[-1, 1])
        np.testing.assert_allclose(finger_faces, (box_x - BOX_HALF, box_x + BOX_HALF), atol=0.004)
        self.assertLess(float(travel.std(axis=0).max()), 0.001)
        self.assertLess(abs(box_x), 0.03)
        self.assertAlmostEqual(float(body_q[box, 2]), BOX_HALF, delta=0.003)


if __name__ == "__main__":
    unittest.main()

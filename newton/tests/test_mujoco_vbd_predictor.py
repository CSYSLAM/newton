# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the smooth MuJoCo predictor used by MuJoCo--VBD."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.contacts import Contacts
from newton._src.solvers.mujoco.solver_mujoco import SolverMuJoCo
from newton._src.solvers.mujoco_vbd.articulation_corrector import ArticulationCorrector
from newton._src.solvers.mujoco_vbd.articulation_predictor import MuJoCoSmoothPredictor
from newton._src.solvers.mujoco_vbd.articulation_q_block import ArticulationQBlock
from newton._src.solvers.mujoco_vbd.canonical_contacts import CanonicalRigidContacts
from newton._src.solvers.mujoco_vbd.canonical_soft_contacts import CanonicalSoftContacts
from newton._src.solvers.mujoco_vbd.contact_energy import evaluate_canonical_rigid_contact
from newton._src.solvers.mujoco_vbd.contact_jacobian import ArticulationContactJacobian
from newton._src.solvers.mujoco_vbd.gap_verifier import EndpointGapVerifier, SweptStateProbe
from newton._src.solvers.mujoco_vbd.ownership_partition import MuJoCoVBDOwnershipPartition
from newton._src.solvers.mujoco_vbd.q_limits import ArticulationLimitProjector
from newton._src.solvers.mujoco_vbd.solver_mujoco_vbd import SolverMuJoCoVBD
from newton._src.solvers.mujoco_vbd.trial_workspace import MuJoCoVBDTrialWorkspace
from newton._src.solvers.vbd.rigid_vbd_kernels import evaluate_rigid_contact_from_collision
from newton._src.solvers.vbd.solver_vbd import SolverVBD


@wp.kernel(enable_backward=False)
def _compare_canonical_contact_energy(
    body_q: wp.array[wp.transform],
    body_q_previous: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    canonical_gradient: wp.array[wp.vec3],
    canonical_hessian: wp.array[wp.mat33],
    canonical_force0: wp.array[wp.vec3],
    canonical_force1: wp.array[wp.vec3],
    vbd_force0: wp.array[wp.vec3],
    vbd_force1: wp.array[wp.vec3],
    vbd_hessian: wp.array[wp.mat33],
):
    gradient, hessian, force0, force1 = evaluate_canonical_rigid_contact(
        body_q[0],
        body_q[1],
        body_q_previous[0],
        body_q_previous[1],
        body_com[0],
        body_com[1],
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(1.0, 0.0, 0.0),
        0.02,
        100.0,
        40.0,
        3.0,
        wp.vec3(0.0),
        0.5,
        1.0e-2,
        0,
        1.0 / 120.0,
        wp.vec3(0.0),
    )
    result = evaluate_rigid_contact_from_collision(
        0,
        1,
        body_q,
        body_q_previous,
        body_com,
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(0.0),
        wp.vec3(1.0, 0.0, 0.0),
        0.02,
        100.0,
        40.0,
        3.0,
        wp.vec3(0.0),
        0.5,
        1.0e-2,
        0,
        1.0 / 120.0,
        wp.vec3(0.0),
    )
    canonical_gradient[0] = gradient
    canonical_hessian[0] = hessian
    canonical_force0[0] = force0
    canonical_force1[0] = force1
    vbd_force0[0] = result[0]
    vbd_force1[0] = result[5]
    vbd_hessian[0] = result[2]


class TestMuJoCoSmoothPredictor(unittest.TestCase):
    def _make_model(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(np.eye(3)),
        )
        builder.add_shape_sphere(body=body, radius=0.1)
        return builder.finalize(device="cpu")

    def test_euler_endpoint_matches_contact_free_mujoco(self):
        """The predictor must reproduce MuJoCo's smooth Euler endpoint."""
        model = self._make_model()
        state_in = model.state()
        state_reference = model.state()
        control = model.control()
        dt = 1.0 / 120.0

        predictor = MuJoCoSmoothPredictor(model)
        result = predictor.predict(state_in, control, dt)

        reference = SolverMuJoCo(model, integrator="euler", disable_contacts=True)
        reference.step(state_in, state_reference, control, None, dt)

        np.testing.assert_allclose(result.state_hat.joint_q.numpy(), state_reference.joint_q.numpy(), atol=2.0e-6)
        np.testing.assert_allclose(result.state_hat.joint_qd.numpy(), state_reference.joint_qd.numpy(), atol=2.0e-6)
        np.testing.assert_allclose(result.state_hat.body_q.numpy(), state_reference.body_q.numpy(), atol=2.0e-6)
        self.assertTrue(np.all(np.isfinite(result.smooth_acceleration.numpy())))

    def test_rejects_non_euler_integrator(self):
        """A different predictor integrator would change the endpoint metric."""
        with self.assertRaisesRegex(ValueError, "integrator='euler'"):
            MuJoCoSmoothPredictor(self._make_model(), mujoco_options={"integrator": "implicitfast"})

    def test_endpoint_gap_verifier_detects_rigid_penetration(self):
        """DAT acceptance rejects a reported rigid contact below its gap tolerance."""
        model = self._make_model()
        contacts = Contacts(1, 0, device=model.device)
        contacts.contact_counters.assign([1, 0])
        contacts.rigid_contact_shape0.assign([0])
        contacts.rigid_contact_shape1.assign([0])
        contacts.rigid_contact_point0.assign([wp.vec3(0.1, 0.0, 0.0)])
        contacts.rigid_contact_point1.assign([wp.vec3(0.0, 0.0, 0.0)])
        contacts.rigid_contact_normal.assign([wp.vec3(1.0, 0.0, 0.0)])
        contacts.rigid_contact_margin0.assign([0.02])
        contacts.rigid_contact_margin1.assign([0.02])
        verifier = EndpointGapVerifier(model)
        self.assertTrue(verifier.has_violation(model.state(), contacts))
        contacts.rigid_contact_point1.assign([wp.vec3(0.2, 0.0, 0.0)])
        self.assertFalse(verifier.has_violation(model.state(), contacts))

    def test_swept_probe_rejects_midpoint_rigid_pass_through(self):
        """An interior collision rejects a pass-through with clear endpoints."""
        builder = newton.ModelBuilder()
        moving = builder.add_body(
            xform=wp.transform(p=wp.vec3(-0.5, 0.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        builder.add_shape_sphere(moving, radius=0.1)
        builder.add_shape_sphere(-1, radius=0.1)
        model = builder.finalize(device="cpu")
        start = model.state()
        end = model.state()
        end.body_q.assign([wp.transform(p=wp.vec3(0.5, 0.0, 0.0), q=wp.quat_identity())])
        probe = SweptStateProbe(model, EndpointGapVerifier(model), samples=1)
        self.assertTrue(probe.has_violation(start, end))

    def test_q_block_uses_the_full_reduced_mass_matrix(self):
        """A mass-only q-block solve equals MuJoCo's packed-mass solve."""
        model = self._make_model()
        state = model.state()
        predictor = MuJoCoSmoothPredictor(model)
        dt = 1.0 / 120.0
        result = predictor.predict(state, model.control(), dt)
        q_block = ArticulationQBlock(predictor)
        q_block.initialize(result, dt)

        gradient = np.linspace(0.2, 1.1, q_block.dof_count, dtype=np.float32)[None, :]
        q_block.gradient.assign(gradient)
        actual = q_block.solve().numpy().copy()

        rhs = wp.array(-gradient * dt * dt, dtype=float, device=model.device)
        expected = wp.empty_like(rhs)
        predictor.solve_mass(rhs, expected)
        expected_host = expected.numpy().copy()
        self.assertLess(float(np.max(np.abs(actual - expected_host))), 2.0e-9, (actual, expected_host))

    def test_tangent_delta_retracts_through_mujoco_qpos(self):
        """q-block corrections update qpos and qvel on the MuJoCo manifold."""
        model = self._make_model()
        predictor = MuJoCoSmoothPredictor(model)
        dt = 1.0 / 120.0
        result = predictor.predict(model.state(), model.control(), dt)
        q_hat = result.state_hat.joint_q.numpy().copy()
        qd_hat = result.state_hat.joint_qd.numpy().copy()
        delta = np.zeros((1, predictor.qd_count), dtype=np.float32)
        delta[0, 0] = 0.01

        trial = predictor.apply_tangent_delta(wp.array(delta, dtype=float, device=model.device), dt)
        np.testing.assert_allclose(trial.joint_q.numpy()[0], q_hat[0] + delta[0, 0], atol=2.0e-6)
        np.testing.assert_allclose(trial.joint_qd.numpy()[0], qd_hat[0] + delta[0, 0] / dt, atol=2.0e-5)

    def test_q_block_pulls_contact_terms_back_through_jacobian(self):
        """Contact-space f and K must assemble as J.T f and J.T K J."""
        model = self._make_model()
        predictor = MuJoCoSmoothPredictor(model)
        dt = 1.0 / 120.0
        result = predictor.predict(model.state(), model.control(), dt)
        q_block = ArticulationQBlock(predictor)
        q_block.initialize(result, dt)
        hessian_before = q_block.hessian.numpy().copy()

        jacobian = np.zeros((1, 3, q_block.dof_count), dtype=np.float32)
        jacobian[0, 0, 0] = 1.0
        jacobian[0, 1, 1] = 1.0
        jacobian[0, 2, 2] = 1.0
        q_block.accumulate_contact_terms(
            wp.array([1], dtype=wp.int32, device=model.device),
            wp.array([0], dtype=wp.int32, device=model.device),
            wp.array(jacobian, dtype=float, device=model.device),
            wp.array([wp.vec3(1.0, 2.0, 3.0)], dtype=wp.vec3, device=model.device),
            wp.array([wp.mat33(4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0)], dtype=wp.mat33, device=model.device),
        )

        expected_gradient = np.zeros((1, q_block.dof_count), dtype=np.float32)
        expected_gradient[0, :3] = (1.0, 2.0, 3.0)
        expected_hessian = hessian_before
        expected_hessian[0, 0, 0] += 4.0
        expected_hessian[0, 1, 1] += 5.0
        expected_hessian[0, 2, 2] += 6.0
        np.testing.assert_allclose(q_block.gradient.numpy(), expected_gradient)
        np.testing.assert_allclose(q_block.hessian.numpy(), expected_hessian)

    def test_ownership_partition_excludes_vbd_free_dofs_from_q_block(self):
        """The compact MuJoCo model must contain only its complete robot tree."""
        builder = newton.ModelBuilder()
        robot = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="robot_link")
        robot_joint = builder.add_joint_revolute(parent=-1, child=robot, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([robot_joint], label="robot")
        payload = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), label="payload")
        builder.add_shape_sphere(body=robot, radius=0.1)
        builder.add_shape_sphere(body=payload, radius=0.1)
        builder.color()
        model = builder.finalize(device="cpu")

        partition = MuJoCoVBDOwnershipPartition(
            model,
            articulation_bodies=[robot],
            articulation_joints=[robot_joint],
            vbd_bodies=[payload],
            vbd_particles=[],
            articulation_solver=lambda view: SolverMuJoCo(
                view,
                integrator="euler",
                disable_contacts=True,
                use_mujoco_contacts=False,
            ),
            vbd_solver=lambda view: SolverVBD(view, iterations=1),
        )

        articulation = partition.articulation_entry
        self.assertEqual(articulation.view.body_count, 1)
        self.assertEqual(articulation.view.joint_dof_count, 1)
        self.assertEqual(articulation.joint_dof_local_to_global.numpy().tolist(), [0])
        self.assertEqual(articulation.body_local_to_global.numpy().tolist(), [robot])
        self.assertEqual(partition.vbd_entry.body_local_to_global.numpy().tolist(), [robot, payload])
        self.assertEqual(float(partition.vbd_entry.view.body_inv_mass.numpy()[0]), 0.0)
        self.assertGreater(float(partition.vbd_entry.view.body_inv_mass.numpy()[1]), 0.0)

        predictor = MuJoCoSmoothPredictor(articulation.view, solver=articulation.solver)
        self.assertEqual(predictor.qd_count, 1)

    def test_contact_jacobian_uses_relative_q_endpoint_motion(self):
        """An articulated contact endpoint contributes its exact relative Jacobian."""
        builder = newton.ModelBuilder()
        robot = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="robot_link")
        robot_joint = builder.add_joint_revolute(parent=-1, child=robot, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([robot_joint], label="robot")
        payload = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)), label="payload")
        robot_shape = builder.add_shape_sphere(body=robot, radius=0.1)
        payload_shape = builder.add_shape_sphere(body=payload, radius=0.1)
        builder.color()
        model = builder.finalize(device="cpu")
        partition = MuJoCoVBDOwnershipPartition(
            model,
            articulation_bodies=[robot],
            articulation_joints=[robot_joint],
            vbd_bodies=[payload],
            vbd_particles=[],
            articulation_solver=lambda view: SolverMuJoCo(
                view, integrator="euler", disable_contacts=True, use_mujoco_contacts=False
            ),
            vbd_solver=lambda view: SolverVBD(view, iterations=1),
        )
        partition._distribute_state(model.state())
        articulation = partition.articulation_entry
        predictor = MuJoCoSmoothPredictor(articulation.view, solver=articulation.solver)
        state_hat = predictor.predict(articulation.state_0, articulation.control, 1.0 / 120.0).state_hat
        contacts = Contacts(1, 0, device=model.device)
        contacts.contact_counters.assign([1, 0])
        contacts.rigid_contact_shape0.assign([robot_shape])
        contacts.rigid_contact_shape1.assign([payload_shape])
        contacts.rigid_contact_point0.assign([wp.vec3(1.0, 0.0, 0.0)])
        contacts.rigid_contact_point1.assign([wp.vec3(0.0, 0.0, 0.0)])

        evaluator = ArticulationContactJacobian(predictor, articulation)
        jacobian = evaluator.evaluate(state_hat, contacts).numpy()
        self.assertEqual(evaluator.active.numpy().tolist(), [1])
        self.assertEqual(evaluator.endpoint0_q_body.numpy().tolist(), [0])
        self.assertEqual(evaluator.endpoint1_q_body.numpy().tolist(), [-1])
        self.assertGreater(abs(float(jacobian[0, 1, 0])), 0.9)

    def test_copied_canonical_contact_energy_matches_vbd(self):
        """The self-contained joint contact law preserves VBD's force and K."""
        body_q = wp.array(
            [
                wp.transform(p=wp.vec3(0.0), q=wp.quat_identity()),
                wp.transform(p=wp.vec3(0.01, 0.0, 0.0), q=wp.quat_identity()),
            ],
            dtype=wp.transform,
            device="cpu",
        )
        body_q_previous = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device="cpu")
        body_com = wp.zeros(2, dtype=wp.vec3, device="cpu")
        canonical_gradient = wp.empty(1, dtype=wp.vec3, device="cpu")
        canonical_hessian = wp.empty(1, dtype=wp.mat33, device="cpu")
        canonical_force0 = wp.empty(1, dtype=wp.vec3, device="cpu")
        canonical_force1 = wp.empty(1, dtype=wp.vec3, device="cpu")
        vbd_force0 = wp.empty(1, dtype=wp.vec3, device="cpu")
        vbd_force1 = wp.empty(1, dtype=wp.vec3, device="cpu")
        vbd_hessian = wp.empty(1, dtype=wp.mat33, device="cpu")
        wp.launch(
            _compare_canonical_contact_energy,
            dim=1,
            inputs=[body_q, body_q_previous, body_com],
            outputs=[
                canonical_gradient,
                canonical_hessian,
                canonical_force0,
                canonical_force1,
                vbd_force0,
                vbd_force1,
                vbd_hessian,
            ],
            device="cpu",
        )
        np.testing.assert_allclose(canonical_force0.numpy(), vbd_force0.numpy())
        np.testing.assert_allclose(canonical_force1.numpy(), vbd_force1.numpy())
        np.testing.assert_allclose(canonical_hessian.numpy(), vbd_hessian.numpy())
        np.testing.assert_allclose(canonical_gradient.numpy(), -canonical_force1.numpy())

    def test_canonical_contact_batch_assembles_into_q_block(self):
        """The copied VBD contact energy feeds the exact q-space pullback."""
        builder = newton.ModelBuilder()
        robot = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), label="robot_link")
        robot_joint = builder.add_joint_revolute(parent=-1, child=robot, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([robot_joint], label="robot")
        payload = builder.add_body(
            xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
            label="payload",
        )
        robot_shape = builder.add_shape_sphere(body=robot, radius=0.1)
        payload_shape = builder.add_shape_sphere(body=payload, radius=0.1)
        builder.color()
        model = builder.finalize(device="cpu")
        partition = MuJoCoVBDOwnershipPartition(
            model,
            articulation_bodies=[robot],
            articulation_joints=[robot_joint],
            vbd_bodies=[payload],
            vbd_particles=[],
            articulation_solver=lambda view: SolverMuJoCo(
                view, integrator="euler", disable_contacts=True, use_mujoco_contacts=False
            ),
            vbd_solver=lambda view: SolverVBD(view, iterations=1),
        )
        state = model.state()
        partition._distribute_state(state)
        articulation = partition.articulation_entry
        predictor = MuJoCoSmoothPredictor(articulation.view, solver=articulation.solver)
        dt = 1.0 / 120.0
        result = predictor.predict(articulation.state_0, articulation.control, dt)
        contacts = Contacts(1, 0, device=model.device)
        contacts.contact_counters.assign([1, 0])
        contacts.rigid_contact_shape0.assign([robot_shape])
        contacts.rigid_contact_shape1.assign([payload_shape])
        contacts.rigid_contact_point0.assign([wp.vec3(1.0, 0.0, 0.0)])
        contacts.rigid_contact_point1.assign([wp.vec3(0.0, 0.0, 0.0)])
        contacts.rigid_contact_normal.assign([wp.vec3(0.0, 1.0, 0.0)])
        contacts.rigid_contact_margin0.assign([0.02])
        contacts.rigid_contact_margin1.assign([0.02])

        canonical = CanonicalRigidContacts(model)
        canonical.evaluate(state, state, contacts, dt)
        contact_jacobian = ArticulationContactJacobian(predictor, articulation)
        jacobian = contact_jacobian.evaluate(result.state_hat, contacts)
        q_block = ArticulationQBlock(predictor)
        q_block.initialize(result, dt)
        q_block.accumulate_contact_terms(
            contacts.rigid_contact_count,
            contact_jacobian.contact_world,
            jacobian,
            canonical.gradient,
            canonical.hessian,
            contact_jacobian.active,
        )
        self.assertGreater(abs(float(q_block.gradient.numpy()[0, 0])), 1.0e-6)
        self.assertTrue(np.all(np.isfinite(q_block.hessian.numpy())))

    def test_q_block_skips_vbd_only_contacts_in_global_buffer(self):
        """A VBD-only contact must not access an invalid q-block world id."""
        model = self._make_model()
        predictor = MuJoCoSmoothPredictor(model)
        dt = 1.0 / 120.0
        result = predictor.predict(model.state(), model.control(), dt)
        q_block = ArticulationQBlock(predictor)
        q_block.initialize(result, dt)
        jacobian = np.zeros((2, 3, q_block.dof_count), dtype=np.float32)
        jacobian[0, 0, 0] = 1.0
        q_block.accumulate_contact_terms(
            wp.array([2], dtype=int, device=model.device),
            wp.array([0, -1], dtype=int, device=model.device),
            wp.array(jacobian, dtype=float, device=model.device),
            wp.array([wp.vec3(2.0, 0.0, 0.0), wp.vec3(100.0, 0.0, 0.0)], dtype=wp.vec3, device=model.device),
            wp.array(
                [
                    wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                    wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                ],
                dtype=wp.mat33,
                device=model.device,
            ),
            wp.array([1, 0], dtype=int, device=model.device),
        )
        self.assertAlmostEqual(float(q_block.gradient.numpy()[0, 0]), 2.0)

    def test_q_limit_projector_clips_hinge_correction(self):
        """Joint contact corrections cannot cross an authored scalar limit."""
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_revolute(
            parent=-1,
            child=body,
            axis=(0.0, 0.0, 1.0),
            limit_lower=-0.2,
            limit_upper=0.2,
        )
        builder.add_articulation([joint])
        model = builder.finalize(device="cpu")
        predictor = MuJoCoSmoothPredictor(model)
        result = predictor.predict(model.state(), model.control(), 1.0 / 120.0)
        delta = wp.array([[0.5]], dtype=float, device=model.device)
        alpha = ArticulationLimitProjector(predictor).compute_alpha(result.state_hat, delta)
        self.assertAlmostEqual(float(alpha.numpy()[0]), 0.4, places=6)

    def test_trial_workspace_scatter_and_rollback_preserve_ownership(self):
        """q and VBD trial writes affect only their explicitly owned state."""
        builder = newton.ModelBuilder()
        robot = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        robot_joint = builder.add_joint_revolute(parent=-1, child=robot, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([robot_joint])
        payload = builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.color()
        model = builder.finalize(device="cpu")
        partition = MuJoCoVBDOwnershipPartition(
            model,
            articulation_bodies=[robot],
            articulation_joints=[robot_joint],
            vbd_bodies=[payload],
            vbd_particles=[],
            articulation_solver=lambda view: SolverMuJoCo(
                view, integrator="euler", disable_contacts=True, use_mujoco_contacts=False
            ),
            vbd_solver=lambda view: SolverVBD(view, iterations=1),
        )
        workspace = MuJoCoVBDTrialWorkspace(partition)
        state = model.state()
        workspace.begin(state)
        q_state = partition.articulation_entry.state_0
        q_state.joint_q.assign([0.3])
        q_state.joint_qd.assign([0.4])
        workspace.scatter_articulation(q_state)
        self.assertAlmostEqual(float(workspace.trial.joint_q.numpy()[0]), 0.3)
        self.assertAlmostEqual(float(workspace.trial.joint_qd.numpy()[0]), 0.4)
        np.testing.assert_allclose(workspace.trial.body_q.numpy()[payload], state.body_q.numpy()[payload])
        workspace.rollback()
        np.testing.assert_allclose(workspace.trial.joint_q.numpy(), state.joint_q.numpy())

    def test_articulation_corrector_applies_canonical_contact_to_q(self):
        """A canonical robot-payload contact produces a direct q correction."""
        builder = newton.ModelBuilder()
        robot = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        robot_joint = builder.add_joint_revolute(parent=-1, child=robot, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([robot_joint])
        payload = builder.add_body(
            xform=wp.transform(p=wp.vec3(1.0, 0.0, 0.0), q=wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        robot_shape = builder.add_shape_sphere(body=robot, radius=0.1)
        payload_shape = builder.add_shape_sphere(body=payload, radius=0.1)
        builder.color()
        model = builder.finalize(device="cpu")
        partition = MuJoCoVBDOwnershipPartition(
            model,
            articulation_bodies=[robot],
            articulation_joints=[robot_joint],
            vbd_bodies=[payload],
            vbd_particles=[],
            articulation_solver=lambda view: SolverMuJoCo(
                view, integrator="euler", disable_contacts=True, use_mujoco_contacts=False
            ),
            vbd_solver=lambda view: SolverVBD(view, iterations=1),
        )
        workspace = MuJoCoVBDTrialWorkspace(partition)
        workspace.begin(model.state())
        partition._distribute_state(workspace.accepted)
        contacts = Contacts(1, 0, device=model.device)
        contacts.contact_counters.assign([1, 0])
        contacts.rigid_contact_shape0.assign([robot_shape])
        contacts.rigid_contact_shape1.assign([payload_shape])
        contacts.rigid_contact_point0.assign([wp.vec3(1.0, 0.0, 0.0)])
        contacts.rigid_contact_point1.assign([wp.vec3(0.0, 0.0, 0.0)])
        contacts.rigid_contact_normal.assign([wp.vec3(0.0, 1.0, 0.0)])
        contacts.rigid_contact_margin0.assign([0.02])
        contacts.rigid_contact_margin1.assign([0.02])

        ArticulationCorrector(partition).correct(
            workspace,
            model.control(),
            contacts,
            CanonicalRigidContacts(model),
            CanonicalSoftContacts(model),
            1.0 / 120.0,
        )
        self.assertGreater(abs(float(workspace.trial.joint_q.numpy()[0])), 1.0e-7)
        np.testing.assert_allclose(workspace.trial.body_q.numpy()[payload], workspace.accepted.body_q.numpy()[payload])

    def test_articulation_corrector_receives_soft_contact_reaction(self):
        """A cloth/soft contact contributes a direct articulated q reaction."""
        builder = newton.ModelBuilder()
        robot = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        robot_joint = builder.add_joint_revolute(parent=-1, child=robot, axis=(0.0, 0.0, 1.0))
        builder.add_articulation([robot_joint])
        robot_shape = builder.add_shape_sphere(body=robot, radius=0.1)
        particle = builder.add_particle(pos=(1.0, 0.01, 0.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.1)
        builder.color()
        model = builder.finalize(device="cpu")
        partition = MuJoCoVBDOwnershipPartition(
            model,
            articulation_bodies=[robot],
            articulation_joints=[robot_joint],
            vbd_bodies=[],
            vbd_particles=[particle],
            articulation_solver=lambda view: SolverMuJoCo(
                view, integrator="euler", disable_contacts=True, use_mujoco_contacts=False
            ),
            vbd_solver=lambda view: SolverVBD(view, iterations=1),
        )
        workspace = MuJoCoVBDTrialWorkspace(partition)
        workspace.begin(model.state())
        partition._distribute_state(workspace.accepted)
        contacts = Contacts(0, 1, device=model.device)
        contacts.contact_counters.assign([0, 1])
        contacts.soft_contact_indices.assign([wp.vec3i(particle, -1, -1)])
        contacts.soft_contact_barycentric.assign([wp.vec3(1.0, 0.0, 0.0)])
        contacts.soft_contact_shape.assign([robot_shape])
        contacts.soft_contact_body_pos.assign([wp.vec3(1.0, 0.0, 0.0)])
        contacts.soft_contact_body_vel.assign([wp.vec3(0.0)])
        contacts.soft_contact_normal.assign([wp.vec3(0.0, 1.0, 0.0)])

        ArticulationCorrector(partition).correct(
            workspace,
            model.control(),
            contacts,
            CanonicalRigidContacts(model),
            CanonicalSoftContacts(model),
            1.0 / 120.0,
        )
        self.assertGreater(abs(float(workspace.trial.joint_q.numpy()[0])), 1.0e-7)

    def test_solver_steps_q_and_vbd_owned_state_without_admm(self):
        """The new solver advances an owned robot/payload scene end to end."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        robot = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        robot_joint = builder.add_joint_revolute(parent=-1, child=robot, axis=(0.0, 1.0, 0.0))
        builder.add_articulation([robot_joint])
        payload = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 1.0), q=wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3)),
        )
        builder.add_shape_sphere(body=robot, radius=0.1)
        builder.add_shape_sphere(body=payload, radius=0.1)
        builder.color()
        model = builder.finalize(device="cpu")
        solver = SolverMuJoCoVBD(
            model,
            articulation_bodies=[robot],
            articulation_joints=[robot_joint],
            vbd_bodies=[payload],
            vbd_particles=[],
            vbd_options={"iterations": 1},
        )

        def _nested_vbd_step_is_forbidden(*args, **kwargs):
            raise AssertionError("MuJoCo-VBD must schedule VBD blocks directly, not call SolverVBD.step()")

        solver.partition.vbd_entry.solver.step = _nested_vbd_step_is_forbidden
        events = []
        iterate = solver.vbd_block.iterate
        correct = solver.corrector.correct_from_prediction

        def _trace_vbd(*args, **kwargs):
            events.append("vbd")
            return iterate(*args, **kwargs)

        def _trace_q(*args, **kwargs):
            events.append("q")
            return correct(*args, **kwargs)

        solver.vbd_block.iterate = _trace_vbd
        solver.corrector.correct_from_prediction = _trace_q
        state_in = model.state()
        state_out = model.state()
        solver.step(state_in, state_out, model.control(), None, 1.0 / 120.0)
        self.assertGreater(events.index("q"), events.index("vbd"))
        self.assertTrue(np.all(np.isfinite(state_out.joint_q.numpy())))
        self.assertTrue(np.all(np.isfinite(state_out.body_q.numpy())))


if __name__ == "__main__":
    unittest.main(verbosity=2)

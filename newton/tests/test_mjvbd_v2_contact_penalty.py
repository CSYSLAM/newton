# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check independent body-body penalty growth without changing legacy defaults."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjvbd_v2.vbd.rigid_vbd_kernels import update_duals_body_particle_contacts
from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD as FullVBD
from newton._src.solvers.mjvbd_v2.vbd_soft.solver_vbd import SolverVBD as SoftVBD


class TestContactPenalty(unittest.TestCase):
    def test_fixed_soft_penalty_skips_redundant_updates(self):
        """Keep fixed contact stiffness without launching per-sweep ramp updates."""
        for dynamic in (False, True):
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            body = builder.add_body() if dynamic else -1
            builder.add_shape_sphere(body=body, radius=0.1)
            builder.add_particle(pos=wp.vec3(0, 0, 0.105), vel=wp.vec3(), mass=0.01, radius=0.01)
            builder.color()
            model = builder.finalize(device="cpu")
            for beta in (0.0, 100.0):
                solver = FullVBD(model, iterations=3, rigid_avbd_linear_beta=beta)
                state, output = model.state(), model.state()
                pipeline = newton.CollisionPipeline(model)
                contacts = pipeline.contacts()
                pipeline.collide(state, contacts)
                count = int(contacts.soft_contact_count.numpy()[0])
                self.assertGreater(count, 0)
                with mock.patch.object(wp, "launch", wraps=wp.launch) as launch:
                    solver.step(state, output, model.control(), contacts, 1 / 480)
                updates = [
                    call
                    for call in launch.call_args_list
                    if call.kwargs.get("kernel") is update_duals_body_particle_contacts
                ]
                self.assertEqual(len(updates), 0 if beta == 0 else 3)
                if beta == 0:
                    np.testing.assert_array_equal(
                        solver.body_particle_contact_penalty_k.numpy()[:count],
                        solver.body_particle_contact_material_ke.numpy()[:count],
                    )

    @staticmethod
    def model():
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.095), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.1)
        builder.add_ground_plane()
        builder.color()
        return builder.finalize(device="cpu")

    def test_override_does_not_change_particle_or_joint_penalties(self):
        """Separate hard rigid penalty growth from deformable and joint stiffness."""
        for backend in (FullVBD, SoftVBD):
            for linear, contact in ((0.0, 1e5), (1e5, 0.0), (23.0, None)):
                with self.subTest(backend=backend.__module__, linear=linear, contact=contact):
                    solver = backend(self.model(), rigid_avbd_linear_beta=linear, rigid_avbd_contact_beta=contact)
                    expected = linear if contact is None else contact
                    self.assertEqual(solver.rigid_linear_beta, linear)
                    self.assertEqual(solver.rigid_contact_beta, expected)
                    self.assertEqual(solver.rigid_contact_k_start_value, -1.0 if linear == 0 else 100.0)
                    self.assertEqual(solver.rigid_body_contact_k_start_value, -1.0 if expected == 0 else 100.0)

    def test_none_override_preserves_legacy_trajectory(self):
        """Match omitted and explicit inherited beta exactly on rigid contact."""
        for backend in (FullVBD, SoftVBD):
            for beta in (0.0, 1e5):
                model = self.model()
                pipeline = newton.CollisionPipeline(model)
                results = []
                for options in ({}, {"rigid_avbd_contact_beta": beta}, {"rigid_avbd_contact_beta": None}):
                    solver = backend(model, iterations=4, rigid_avbd_beta=beta, **options)
                    state, output = model.state(), model.state()
                    contacts, control = pipeline.contacts(), model.control()
                    for _ in range(12):
                        state.clear_forces()
                        pipeline.collide(state, contacts)
                        solver.step(state, output, control, contacts, 1 / 480)
                        state, output = output, state
                    results.append((state.body_q.numpy(), state.body_qd.numpy()))
                for actual in results[1:]:
                    for before, after in zip(results[0], actual, strict=True):
                        np.testing.assert_array_equal(before, after)

    def test_invalid_contact_beta_is_rejected(self):
        """Reject negative and nonfinite growth rates before allocating contacts."""
        for backend in (FullVBD, SoftVBD):
            for value in (-1.0, float("nan"), float("inf")):
                with self.assertRaisesRegex(ValueError, "rigid_avbd_contact_beta"):
                    backend(self.model(), rigid_avbd_contact_beta=value)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare frozen-color fusion with the unfused mixed-contact solver."""

import unittest
from unittest import mock

import numpy as np
import warp as wp
from rigid_fusion_probe import RigidFusionAdapter, prune_fused_resets

import newton
from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD


class TestRigidFusion(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Fused tile solve requires CUDA")
    def test_mixed_contact_step_matches_unfused(self):
        """Compare fused and reset-pruned contact updates with the original solve."""
        builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
        builder.add_ground_plane()
        for x in (-0.047, 0.047):
            body = builder.add_body(xform=wp.transform(wp.vec3(x, 0, 0.049), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.05)
            builder.add_particle(wp.vec3(x, 0, 0.103), wp.vec3(0.01, 0, 0), mass=0.001, radius=0.008)
        builder.color()
        model = builder.finalize(device="cuda:0")
        results = []
        for fused in (0, 1, 2):
            solver = SolverVBD(model, iterations=4)
            state, output = model.state(), model.state()
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            self.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
            self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
            adapter = RigidFusionAdapter(wp.launch, model)
            skipped = prune_fused_resets(solver) if fused == 2 else None
            with mock.patch.object(wp, "launch", adapter if fused else wp.launch):
                solver.step(state, output, model.control(), contacts, 1 / 960)
            results.append([array.numpy().copy() for array in (output.body_q, output.body_qd, output.particle_q)])
            if skipped is not None:
                self.assertEqual(skipped[0], 20)
        for candidate in results[1:]:
            for expected, actual in zip(results[0], candidate, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare contact/elasticity fusion from identical mixed-contact states."""

import unittest
from unittest import mock

import numpy as np
import warp as wp
from particle_fusion_probe import ParticleFusionAdapter

import newton
from newton._src.solvers.mjvbd_v2.full_contact_pipeline import MJVBDV2CollisionPipeline
from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD


class TestParticleFusion(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Tile solve requires CUDA")
    def test_contact_surface_step_matches_unfused(self):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        body = builder.add_body()
        builder.add_shape_sphere(body, radius=0.055)
        builder.add_cloth_grid(
            pos=wp.vec3(-0.08, -0.08, 0.052),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=16,
            dim_y=16,
            cell_x=0.01,
            cell_y=0.01,
            mass=0.001,
            tri_ke=1e4,
            tri_ka=1e4,
            tri_kd=0.01,
            edge_ke=0.001,
            particle_radius=0.003,
        )
        builder.color()
        model = builder.finalize(device="cuda:0")
        results = []
        for fused in (False, True):
            solver = SolverVBD(model, iterations=2, rigid_body_particle_contact_buffer_size=1024)
            state, output = model.state(), model.state()
            pipeline = MJVBDV2CollisionPipeline(model, enable_rigid_soft_full_surface_contact=True)
            contacts = pipeline.contacts()
            pipeline.collide(state, contacts)
            self.assertGreater(int(contacts.soft_contact_count.numpy()[0]), 0)
            count = int(contacts.soft_contact_count.numpy()[0])
            self.assertTrue(np.any(contacts.soft_contact_indices.numpy()[:count, 1] >= 0))
            adapter = ParticleFusionAdapter(wp.launch, model)
            with mock.patch.object(wp, "launch", adapter if fused else wp.launch):
                solver.step(state, output, model.control(), contacts, 1 / 960)
            if fused:
                self.assertGreater(adapter.fused_launches, 0)
            results.append([a.numpy().copy() for a in (output.body_q, output.particle_q, output.particle_qd)])
        for expected, actual in zip(*results, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


if __name__ == "__main__":
    unittest.main()

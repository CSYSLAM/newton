# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare periodic graph loops against explicitly unrolled sweep graphs."""

import unittest

import numpy as np
import warp as wp
from iteration_loop_probe import install_iteration_loop

import newton
from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD


class TestIterationLoop(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Conditional graphs require CUDA")
    def test_replay_matches_unrolled(self):
        """Preserve contact solves, coarse checkpoints and first/last relaxation."""
        builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
        builder.add_ground_plane()
        builder.add_cloth_grid(
            pos=wp.vec3(-0.02, -0.02, 0.002),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.01, 0, 0),
            dim_x=4,
            dim_y=4,
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
        for looped in (False, True):
            solver = SolverVBD(
                model,
                iterations=16,
                particle_collision_detection_interval=0,
                particle_enable_multilevel_correction=True,
                particle_multilevel_operator="galerkin",
                particle_multilevel_cluster_size=8,
                particle_multilevel_checkpoints=(4, 8, 12, 16),
                particle_multilevel_fallback_iterations=16,
                particle_surface_relaxation=1.1,
            )
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            state, output = model.state(), model.state()
            pipeline.collide(state, contacts)
            control = model.control()
            if looped:
                install_iteration_loop(solver)
            solver.step(state, output, control, contacts, 1 / 960)
            state, output = model.state(), model.state()
            with wp.ScopedCapture(device=model.device) as capture:
                solver.step(state, output, control, contacts, 1 / 960)
            frames = []
            for _ in range(3):
                wp.capture_launch(capture.graph)
                frames.append(output.particle_q.numpy().copy())
            results.append(frames)
        np.testing.assert_allclose(results[1], results[0], rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()

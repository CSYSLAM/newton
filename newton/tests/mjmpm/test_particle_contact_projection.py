# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify residual particle contact stabilization and its rigid feedback."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjmpm.mpm.rasterized_collisions import project_outside_collider
from newton._src.solvers.mjmpm.mpm.solver_implicit_mpm import SolverImplicitMPM


class TestParticleContactProjection(unittest.TestCase):
    def test_moving_wall_reaction_and_graph(self):
        """Correct penetration against a moving wall and return the opposite impulse."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device), wp.ScopedDevice(device):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                SolverImplicitMPM.register_custom_attributes(builder)
                body = builder.add_body()
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.01, cfg=builder.ShapeConfig(mu=0.0))
                for z, mass, flags in ((0.009, 2.0, 1), (0.009, 2.0, 0), (0.009, 0.0, 1), (0.03, 1.0, 1)):
                    builder.add_particle((0.0, 0.0, z), (0.0, 0.0, -1.0), mass, 0.001, flags=flags)
                model = builder.finalize(device=device)
                solver = SolverImplicitMPM(model, SolverImplicitMPM.Config(voxel_size=0.01))
                state, output = model.state(), model.state()
                state.body_qd.assign(
                    wp.array([wp.spatial_vector(0.1, 0.0, 0.2, 0.0, 0.0, 0.0)], dtype=wp.spatial_vector)
                )
                reaction = wp.full(4, wp.vec3(999.0), dtype=wp.vec3)
                points = wp.empty(4, dtype=wp.vec3)
                ids = wp.full(4, 999, dtype=int)
                command = wp.launch(
                    project_outside_collider,
                    dim=4,
                    inputs=[
                        state.particle_q,
                        state.particle_qd,
                        state.mpm.particle_qd_grad,
                        model.particle_flags,
                        model.particle_mass,
                        None,
                        solver._mpm_model.collider,
                        state.body_q,
                        state.body_qd,
                        None,
                        0.001,
                    ],
                    outputs=[output.particle_q, output.particle_qd, output.mpm.particle_qd_grad, reaction, points, ids],
                    record_cmd=True,
                )
                command.launch()
                if model.device.is_cuda:
                    with wp.ScopedCapture() as capture:
                        command.launch()
                    wp.capture_launch(capture.graph)
                impulse = reaction.numpy()
                delta = (output.particle_qd.numpy() - state.particle_qd.numpy()) * model.particle_mass.numpy()[:, None]
                np.testing.assert_allclose(impulse, -delta, atol=1.0e-6)
                np.testing.assert_allclose(impulse[0], [0.0, 0.0, -2.4], atol=1.0e-5)
                self.assertGreater(float(output.particle_q.numpy()[0, 2]), 0.01)
                np.testing.assert_array_equal(ids.numpy()[1:], -1)
                np.testing.assert_array_equal(output.particle_q.numpy()[1:], state.particle_q.numpy()[1:])
                # The existing explicit projection API still works without feedback buffers.
                legacy = model.state()
                solver.project_outside(state, legacy, 0.001)
                np.testing.assert_allclose(legacy.particle_q.numpy(), output.particle_q.numpy(), atol=1.0e-7)

    def test_step_collects_projection_feedback(self):
        """Include stabilization reactions in step feedback for coupled consumers."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device), wp.ScopedDevice(device):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                SolverImplicitMPM.register_custom_attributes(builder)
                builder.add_shape_box(-1, hx=0.1, hy=0.1, hz=0.01)
                builder.add_particle((0.0, 0.0, 0.009), (0.0, 0.0, -1.0), 1.0, 0.001)
                model = builder.finalize(device=device)
                config = SolverImplicitMPM.Config(
                    voxel_size=0.01,
                    max_iterations=1,
                    collider_basis="S2",
                    project_particle_contacts=True,
                    grid_type="sparse" if model.device.is_cuda else "dense",
                    max_active_cell_count=64,
                    grid_padding=0 if model.device.is_cuda else 1,
                )
                solver = SolverImplicitMPM(model, config)
                state, output = model.state(), model.state()
                solver.step(state, output, None, None, 0.001)
                collected = solver.collect_collider_impulses(output)
                for values, expected in zip(collected, output.particle_contact_reactions, strict=True):
                    np.testing.assert_array_equal(values.numpy()[-1:], expected.numpy())
                self.assertGreaterEqual(float(output.particle_q.numpy()[0, 2]), 0.0098)
                if model.device.is_cuda:
                    with wp.ScopedCapture() as capture:
                        solver.step(state, output, None, None, 0.001)
                        captured = solver.collect_collider_impulses(output)
                    wp.capture_launch(capture.graph)
                    for values, expected in zip(captured, output.particle_contact_reactions, strict=True):
                        np.testing.assert_array_equal(values.numpy()[-1:], expected.numpy())

    def test_free_body_receives_stabilization_impulse(self):
        """Preserve linear momentum when a free body receives projected particle contact."""
        from newton.solvers import SolverMJMPM  # noqa: PLC0415

        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device), wp.ScopedDevice(device):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                SolverMJMPM.register_custom_attributes(builder)
                body = builder.add_body()
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.01, cfg=builder.ShapeConfig(mu=0.0))
                builder.add_particle((0.0, 0.0, 0.009), (0.0, 0.0, -1.0), 1.0, 0.001)
                model = builder.finalize(device=device)
                solver = SolverMJMPM(
                    model,
                    mpm_options={
                        "voxel_size": 0.01,
                        "max_iterations": 1,
                        "collider_basis": "S2",
                        "project_particle_contacts": True,
                        "air_drag": 0.0,
                    },
                )
                state, output = model.state(), model.state()
                solver.step(state, output, model.control(), None, 0.001)
                reactions = solver.environment.work.particle_contact_reactions[0].numpy()
                self.assertGreater(float(np.linalg.norm(reactions)), 1.0e-6)
                momentum = (output.particle_qd.numpy() * model.particle_mass.numpy()[:, None]).sum(axis=0)
                momentum += (output.body_qd.numpy()[:, :3] * model.body_mass.numpy()[:, None]).sum(axis=0)
                np.testing.assert_allclose(momentum, [0.0, 0.0, -1.0], atol=2.0e-5)


if __name__ == "__main__":
    unittest.main()

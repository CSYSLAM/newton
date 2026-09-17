# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preserve pressure and IK results while caching conveyor computation."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from newton.examples.mjvbdv2.example_mjvbd_v2_conveyor_sorting import Example
from newton.solvers import SolverMJVBDV2
from newton.tests.test_mjvbd_v2 import _build_pneumatic_shell_builder


def _mixed_model(device):
    builder, handle, _ = _build_pneumatic_shell_builder()
    for i in range(600):
        builder.add_particle(wp.vec3(3.0 + i * 0.01, 0.0, 0.0), wp.vec3(), 1.0)
    builder.color()
    model = builder.finalize(device=device)
    model.pneumatic.mode.fill_(2)
    model.pneumatic.target_volume.fill_(handle.rest_volume * 1.05)
    model.pneumatic.volume_stiffness.fill_(2.0e4)
    model.pneumatic.bulk_damping.fill_(5.0)
    # A valid extra color containing only non-cavity vertices exercises empty
    # pressure rows without dropping their place in the elasticity schedule.
    colors = model.particle_colors.numpy()
    colors[:4] += 1
    colors[4:] = 0
    model.particle_colors.assign(colors)
    model.particle_color_groups = [
        wp.array(np.flatnonzero(colors == color), dtype=int, device=device) for color in range(int(colors.max()) + 1)
    ]
    return model


class TestConveyorPerformance(unittest.TestCase):
    def test_pressure_dispatch_excludes_unrelated_particles(self):
        """Preserve force buffers while dispatching only vertices on cavity faces."""
        model = _mixed_model("cpu")
        solver = SolverMJVBDV2(model).vbd_solver
        compact = solver._pneumatic_particle_color_groups
        ids = np.concatenate([group.numpy() for group in compact])
        np.testing.assert_array_equal(np.sort(ids), np.arange(4))
        for full, selected in zip(model.particle_color_groups, compact, strict=True):
            np.testing.assert_array_equal(selected.numpy(), full.numpy()[full.numpy() < 4])
        solver._pneumatic_gauge_pressure.fill_(2000.0)
        solver._pneumatic_curvature.fill_(1200.0)
        forces = wp.full(model.particle_count, wp.vec3(1.0), dtype=wp.vec3, device="cpu")
        hessians = wp.full(
            model.particle_count, wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), dtype=wp.mat33, device="cpu"
        )
        expected_f, expected_h = wp.clone(forces), wp.clone(hessians)
        for full, selected in zip(model.particle_color_groups, compact, strict=True):
            solver._accumulate_pneumatic_forces(model.particle_q, full, expected_f, expected_h)
            solver._accumulate_pneumatic_forces(model.particle_q, selected, forces, hessians)
        np.testing.assert_array_equal(forces.numpy(), expected_f.numpy())
        np.testing.assert_array_equal(hessians.numpy(), expected_h.numpy())
        np.testing.assert_array_equal(forces.numpy()[4:], np.ones((600, 3)))

    @unittest.skipUnless(wp.is_cuda_available(), "Fused pressure dispatch requires CUDA")
    def test_small_cavity_fusion_in_large_color(self):
        """Match full-color pressure evaluation when unrelated vertices exceed the fusion limit."""
        model = _mixed_model("cuda:0")
        options = {"iterations": 4, "pneumatic_enable_incremental_volume": True, "particle_enable_self_contact": False}
        optimized = SolverMJVBDV2(model, vbd_options=options)
        reference = SolverMJVBDV2(model, vbd_options=options)
        self.assertGreater(max(g.size for g in model.particle_color_groups), 512)
        self.assertTrue(optimized.vbd_solver._pneumatic_single_cavity_force_fusion_enabled)
        reference.vbd_solver._pneumatic_single_cavity_force_fusion_enabled = False
        reference.vbd_solver._pneumatic_particle_color_groups = model.particle_color_groups
        states = [(model.state(), model.state()), (model.state(), model.state())]
        control = model.control()
        for _ in range(12):
            for i, solver in enumerate((reference, optimized)):
                state_in, state_out = states[i]
                solver.step(state_in, state_out, control, None, 0.001)
                states[i] = state_out, state_in
        for actual, expected in (
            (states[1][0].particle_q, states[0][0].particle_q),
            (states[1][0].pneumatic.volume, states[0][0].pneumatic.volume),
            (states[1][0].pneumatic.absolute_pressure, states[0][0].pneumatic.absolute_pressure),
        ):
            np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-6, atol=1e-7)

    @unittest.skipUnless(wp.is_cuda_available(), "IK graphs require CUDA")
    def test_ik_graph_reads_updated_targets_and_distinct_seeds(self):
        """Match uncaptured IK after objective changes for both persistent seed buffers."""
        builder = newton.ModelBuilder()
        body = builder.add_link(mass=1.0)
        joint = builder.add_joint_prismatic(-1, body, axis=wp.vec3(1.0, 0.0, 0.0))
        builder.add_articulation([joint])
        model = builder.finalize(device="cuda:0")
        objective = ik.IKObjectivePosition(body, wp.vec3(), wp.zeros(1, dtype=wp.vec3, device=model.device))
        limits = ik.IKObjectiveJointLimit(
            wp.array([-1.0], dtype=float, device=model.device),
            wp.array([1.0], dtype=float, device=model.device),
            weight=30.0,
        )
        example = Example.__new__(Example)
        example.model = model
        example.args = SimpleNamespace(compute_cache=True, ik_iterations=24)
        example.ik_graphs = {}
        example.ik_solver = ik.IKSolver(model, 1, [objective, limits], jacobian_mode=ik.IKJacobianType.ANALYTIC)
        reference = ik.IKSolver(model, 1, [objective, limits], jacobian_mode=ik.IKJacobianType.ANALYTIC)
        seeds = [wp.clone(model.joint_q).reshape((1, -1)) for _ in range(2)]
        expected = [wp.clone(seed) for seed in seeds]
        for target in (0.2, -0.1, 0.4):
            objective.set_target_position(0, wp.vec3(target, 0.0, 0.0))
            limits.joint_limit_upper.fill_(0.15 if target > 0.3 else 1.0)
            for actual, before in zip(seeds, expected, strict=True):
                reference.step(before, before, iterations=24)
                example._solve_ik(actual)
                np.testing.assert_allclose(actual.numpy(), before.numpy(), atol=1e-6)
        self.assertEqual(len(example.ik_graphs), 2)


if __name__ == "__main__":
    unittest.main()

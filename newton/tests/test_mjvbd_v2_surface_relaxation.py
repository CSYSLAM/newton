# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the experimental contact-free surface relaxation in MJVBDV2."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjvbd_v2.vbd import particle_vbd_kernels as complete_kernels
from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD as SolverVBDComplete
from newton._src.solvers.mjvbd_v2.vbd_soft import particle_vbd_kernels as soft_kernels
from newton._src.solvers.mjvbd_v2.vbd_soft.solver_vbd import SolverVBD as SolverVBDSoft


def _cloth(device, *, requires_grad=False):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_cloth_grid(
        pos=wp.vec3(),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=16,
        dim_y=4,
        cell_x=0.02,
        cell_y=0.02,
        mass=0.001,
        tri_ke=1.0e5,
        tri_ka=1.0e5,
        tri_kd=0.0,
        edge_ke=0.0,
        fix_left=True,
    )
    builder.color()
    return builder.finalize(device=device, requires_grad=requires_grad)


class TestMJVBDV2SurfaceRelaxation(unittest.TestCase):
    def test_reject_invalid_factors(self):
        """Reject nonfinite and non-SOR factors in both backends."""
        model = _cloth("cpu")
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            for value in (0.0, 0.9, 2.0, float("inf"), float("nan")):
                with self.subTest(solver=solver_type.__module__, factor=value):
                    with self.assertRaisesRegex(ValueError, "particle_surface_relaxation"):
                        solver_type(model, particle_surface_relaxation=value)

    def test_cpu_keeps_ordinary_path(self):
        """Leave scalar CPU solves unchanged without requiring capture_if."""
        model = _cloth("cpu")
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            results = []
            for factor in (1.0, 1.3):
                solver = solver_type(model, iterations=8, particle_surface_relaxation=factor)
                state_in, state_out = model.state(), model.state()
                force = np.zeros_like(model.particle_q.numpy())
                force[-1, 0] = 0.1
                state_in.particle_f.assign(force)
                with mock.patch.object(wp, "capture_if", side_effect=AssertionError("unexpected capture_if")):
                    solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
                results.append(state_out.particle_q.numpy())
            np.testing.assert_array_equal(*results)

    @unittest.skipUnless(wp.is_cuda_available(), "Surface tiles require CUDA")
    def test_preserve_deterministic_and_differentiable_modes(self):
        """Disable the experimental relaxation outside validated execution modes."""
        model = _cloth("cuda:0")
        grad_model = _cloth("cuda:0", requires_grad=True)
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            deterministic = solver_type(
                model, particle_surface_relaxation=1.3, deterministic=wp.DeterministicMode.RUN_TO_RUN
            )
            self.assertEqual(deterministic.particle_surface_relaxation, 1.0)
            differentiable = solver_type(grad_model, particle_surface_relaxation=1.3)
            self.assertEqual(differentiable.particle_surface_relaxation, 1.0)

    @unittest.skipUnless(wp.is_cuda_available(), "Surface tiles require CUDA")
    def test_contact_rows_and_anchors_are_not_relaxed(self):
        """Scale only free local updates and preserve constrained updates exactly."""
        model = _cloth("cuda:0")
        positions = model.particle_q.numpy()
        inertia = wp.array(
            positions + np.array([0.001, 0.0, 0.0], dtype=np.float32), dtype=wp.vec3, device=model.device
        )
        ids = wp.array(np.arange(model.particle_count), dtype=wp.int32, device=model.device)
        forces = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        hessians_np = np.zeros((model.particle_count, 3, 3), dtype=np.float32)
        hessians_np[1, 0, 0] = 1.0e4
        hessians_np[2, 0, 1] = 100.0  # Nonzero off-diagonal also excludes relaxation.
        hessians = wp.array(hessians_np, dtype=wp.mat33, device=model.device)
        for solver_type, kernels in ((SolverVBDComplete, complete_kernels), (SolverVBDSoft, soft_kernels)):
            solver = solver_type(model, particle_surface_relaxation=1.3)
            outputs = []
            for factor in (1.0, 1.3):
                delta = wp.zeros_like(model.particle_q)
                wp.launch(
                    kernels.solve_surface_elasticity_tile,
                    dim=model.particle_count * kernels.TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                    block_dim=kernels.TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE,
                    inputs=[
                        1.0 / 60.0,
                        ids,
                        model.particle_q,
                        model.particle_q,
                        model.particle_mass,
                        inertia,
                        model.particle_flags,
                        model.tri_indices,
                        model.tri_poses,
                        model.tri_materials,
                        model.tri_areas,
                        model.edge_indices,
                        model.edge_rest_angle,
                        model.edge_rest_length,
                        model.edge_bending_properties,
                        solver.particle_adjacency,
                        forces,
                        hessians,
                        0,
                        0,
                        factor,
                    ],
                    outputs=[delta],
                    device=model.device,
                )
                outputs.append(delta.numpy())
            plain, relaxed = outputs
            fixed = model.particle_mass.numpy() == 0.0
            np.testing.assert_array_equal(relaxed[fixed], 0.0)
            np.testing.assert_array_equal(relaxed[1:3], plain[1:3])
            free = ~fixed
            free[1:3] = False
            np.testing.assert_allclose(relaxed[free], 1.3 * plain[free], rtol=1.0e-6, atol=1.0e-9)

    @unittest.skipUnless(wp.is_cuda_available(), "Surface tiles require CUDA")
    def test_schedule_and_graph_replay(self):
        """Use ordinary warm-up and final sweeps and replay the same CUDA result."""
        model = _cloth("cuda:0")
        for solver_type, kernels in ((SolverVBDComplete, complete_kernels), (SolverVBDSoft, soft_kernels)):
            solver = solver_type(model, iterations=8, particle_surface_relaxation=1.3)
            self.assertTrue(solver.use_particle_tile_solve)
            control = model.control()
            force = np.zeros_like(model.particle_q.numpy())
            force[-1, 0] = 0.1
            eager_in, eager_out = model.state(), model.state()
            eager_in.particle_f.assign(force)
            with mock.patch.object(wp, "launch", wraps=wp.launch) as launches:
                solver.step(eager_in, eager_out, control, None, 1.0 / 60.0)
            factors = [
                call.kwargs["inputs"][-1]
                for call in launches.call_args_list
                if call.kwargs.get("kernel") is kernels.solve_surface_elasticity_tile
            ]
            colors = len(model.particle_color_groups)
            expected = np.repeat([1.0, 1.3, 1.3, 1.3, 1.3, 1.0, 1.0, 1.0], colors)
            np.testing.assert_allclose(factors, expected)
            graph_in, graph_out = model.state(), model.state()
            graph_in.particle_f.assign(force)
            with wp.ScopedCapture(device=model.device) as capture:
                solver.step(graph_in, graph_out, control, None, 1.0 / 60.0)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(graph_out.particle_q.numpy(), eager_out.particle_q.numpy())


if __name__ == "__main__":
    unittest.main()

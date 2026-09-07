# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check fused Jacobi writes against the original two-kernel update."""

import unittest
from contextlib import contextmanager
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2 import particle_surface_cache
from newton._src.solvers.mjvbd_v2.vbd_soft import particle_vbd_kernels
from newton._src.solvers.mjvbd_v2.vbd_soft.particle_vbd_kernels import apply_particle_jacobi_correction
from newton._src.solvers.mjvbd_v2.vbd_soft.solver_vbd import SolverVBD
from newton.tests.test_mjvbd_v2_surface_relaxation import _cloth


@contextmanager
def unfused_jacobi(solver, *, verify_each_update=False):
    """Reconstruct the pre-fusion launch sequence, including its scratch clear."""
    scratch = wp.zeros(solver.model.particle_count, dtype=wp.vec3, device=solver.device)
    launch = wp.launch
    iteration = solver._solve_particle_jacobi_iteration

    def solve(*args, **kwargs):
        scratch.zero_()
        return iteration(*args, **kwargs)

    def dispatch(*args, **kwargs):
        if kwargs.get("kernel") is not solver._surface_jacobi_kernel:
            return launch(*args, **kwargs)
        inputs = list(kwargs["inputs"])
        relaxation = inputs[-3]
        inputs[-3] = 1.0
        legacy = dict(kwargs, kernel=solver._surface_cached_kernel, inputs=inputs, outputs=[scratch])
        launch(*args, **legacy)
        expected = wp.clone(kwargs["outputs"][0]) if verify_each_update else kwargs["outputs"][0]
        launch(
            apply_particle_jacobi_correction,
            dim=inputs[1].size,
            inputs=[inputs[1], scratch, relaxation],
            outputs=[expected],
            device=solver.device,
        )
        if verify_each_update:
            result = launch(*args, **kwargs)
            np.testing.assert_allclose(kwargs["outputs"][0].numpy(), expected.numpy(), rtol=2e-6, atol=2e-9)
            return result
        return None

    with (
        mock.patch.object(solver, "_solve_particle_jacobi_iteration", solve),
        mock.patch.object(wp, "launch", dispatch),
    ):
        yield scratch


class TestMJVBDV2JacobiFusion(unittest.TestCase):
    def test_increment_preserves_frozen_positions_and_inactive_rows(self):
        """Keep contact weighting, nonzero anchors, and singular-row behavior."""
        for device in ("cpu", "cuda:0") if wp.is_cuda_available() else ("cpu",):
            model = _cloth(device)
            solver = SolverVBD(model)
            count = model.particle_count
            rng = np.random.default_rng(349)
            q = model.particle_q.numpy()
            q[:, 2] += rng.normal(size=count).astype(np.float32) * 0.001
            pos = wp.array(q, dtype=wp.vec3, device=device)
            inertia = wp.array(q + 0.0003, dtype=wp.vec3, device=device)
            ids = wp.array(np.arange(count), dtype=wp.int32, device=device)
            forces = wp.array(rng.normal(size=(count, 3)), dtype=wp.vec3, device=device)
            hessians = wp.array(np.tile(np.eye(3) * 50.0, (count, 1, 1)), dtype=wp.mat33, device=device)
            anchors = wp.empty(model.edge_count, dtype=wp.vec2, device=device)
            wp.launch(
                particle_surface_cache._prepare_anchor_angles,
                dim=model.edge_count,
                inputs=[model.particle_q, model.edge_indices],
                outputs=[anchors],
                device=device,
            )
            membrane = particle_vbd_kernels.evaluate_neo_hookean_membrane_force_hessian
            ordinary = particle_surface_cache.make_surface_kernel(membrane)
            fused = particle_surface_cache.make_surface_kernel(membrane, jacobi_update=True)
            initial = rng.normal(size=(count, 3)).astype(np.float32) * 0.0001
            for singular in (False, True):
                if singular:
                    model.tri_materials.zero_()
                    model.particle_mass.fill_(1.0e-15)
                    hessians.zero_()
                for relaxation in (0.5, 1.0):
                    with self.subTest(device=device, singular=singular, relaxation=relaxation):
                        inputs = [
                            1.0 / 600.0,
                            ids,
                            model.particle_q,
                            pos,
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
                            1.0,
                            anchors,
                            None,
                        ]
                        scratch = wp.zeros(count, dtype=wp.vec3, device=device)
                        expected = wp.array(initial, dtype=wp.vec3, device=device)
                        actual = wp.array(initial, dtype=wp.vec3, device=device)
                        wp.launch(
                            ordinary, dim=count * 16, block_dim=16, inputs=inputs, outputs=[scratch], device=device
                        )
                        wp.launch(
                            apply_particle_jacobi_correction,
                            dim=count,
                            inputs=[ids, scratch, relaxation],
                            outputs=[expected],
                            device=device,
                        )
                        inputs[-3] = relaxation
                        wp.launch(fused, dim=count * 16, block_dim=16, inputs=inputs, outputs=[actual], device=device)
                        np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=2e-6, atol=2e-9)
                        np.testing.assert_array_equal(pos.numpy(), q)
                        if singular:
                            np.testing.assert_array_equal(actual.numpy(), initial)

    @unittest.skipUnless(wp.is_cuda_available(), "Jacobi solver integration requires CUDA")
    def test_fused_update_matches_legacy_and_removes_apply_launch(self):
        model = _cloth("cuda:0")
        for relaxation in (0.5, 1.0):
            for self_contact in (False, True):
                with self.subTest(relaxation=relaxation, self_contact=self_contact):
                    options = {
                        "iterations": 4,
                        "particle_enable_self_contact": self_contact,
                        "particle_enable_surface_cache": True,
                        "particle_enable_truncation_cache": True,
                        "particle_enable_batched_jacobi": True,
                        "particle_jacobi_batch_count": 2,
                        "particle_jacobi_relaxation": relaxation,
                        "particle_chebyshev_spectral_radius": 0.8,
                    }
                    solver = SolverVBD(model, **options)
                    self.assertIsNotNone(getattr(solver, "_surface_jacobi_kernel", None))
                    self.assertIsNone(getattr(solver, "_particle_jacobi_corrections", None))
                    force = np.zeros_like(model.particle_q.numpy())
                    force[-1] = (0.1, -0.02, 0.04)
                    control = model.control()

                    def run(solver=solver, force=force, control=control):
                        state_in, state_out = model.state(), model.state()
                        for _ in range(12):
                            state_in.particle_f.assign(force)
                            solver.step(state_in, state_out, control, None, 1.0 / 600.0)
                            state_in, state_out = state_out, state_in
                        return state_in.particle_q.numpy(), state_in.particle_qd.numpy()

                    with mock.patch.object(wp, "launch", wraps=wp.launch) as launches:
                        actual = run()
                    self.assertFalse(
                        any(
                            call.kwargs.get("kernel") is apply_particle_jacobi_correction
                            for call in launches.call_args_list
                        )
                    )
                    if self_contact:
                        # Self-contact candidate ordering can differ between two
                        # complete trajectories. Compare each update using the
                        # exact same positions and accumulated contact forces.
                        with unfused_jacobi(solver, verify_each_update=True):
                            run()
                        continue
                    with unfused_jacobi(solver):
                        expected = run()
                    for current, reference in zip(actual, expected, strict=True):
                        np.testing.assert_allclose(current, reference, rtol=2e-6, atol=2e-7)

                    graph_in, graph_out = model.state(), model.state()
                    graph_in.particle_f.assign(force)

                    def graph_step(graph_in=graph_in, graph_out=graph_out, solver=solver, control=control):
                        graph_in.particle_q.assign(model.particle_q)
                        graph_in.particle_qd.zero_()
                        solver.step(graph_in, graph_out, control, None, 1.0 / 600.0)

                    with wp.ScopedCapture(device=model.device) as capture:
                        graph_step()
                    fused_graph = capture.graph
                    # Retain the scratch array for the entire graph lifetime.
                    with unfused_jacobi(solver) as reference_scratch:
                        with wp.ScopedCapture(device=model.device) as capture:
                            graph_step()
                        for _ in range(3):
                            wp.capture_launch(fused_graph)
                            actual = graph_out.particle_q.numpy()
                            wp.capture_launch(capture.graph)
                            np.testing.assert_allclose(actual, graph_out.particle_q.numpy(), rtol=2e-6, atol=2e-7)
                        self.assertEqual(reference_scratch.size, model.particle_count)


if __name__ == "__main__":
    unittest.main()

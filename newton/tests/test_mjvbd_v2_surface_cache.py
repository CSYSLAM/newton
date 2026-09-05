# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate experimental surface caches without changing scalar execution."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2 import particle_surface_cache as cache
from newton._src.solvers.mjvbd_v2.vbd import particle_vbd_kernels as original
from newton.tests.test_mjvbd_v2_surface_relaxation import SolverVBDComplete, SolverVBDSoft, _cloth


@wp.kernel
def _compare_bending(
    pos: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    edges: wp.array2d[wp.int32],
    rest: wp.array[float],
    lengths: wp.array[float],
    anchors: wp.array[wp.vec2],
    forces: wp.array[wp.vec3],
    hessians: wp.array[wp.mat33],
    old_forces: wp.array[wp.vec3],
    old_hessians: wp.array[wp.mat33],
    gradients: wp.array[wp.vec3],
):
    tid = wp.tid()
    edge = tid // 4
    order = tid % 4
    force, hessian = cache._evaluate_bending(
        edge, order, pos, edges, rest, lengths, 1.2, 0.1, 1.0 / 600.0, anchors[edge]
    )
    old_force, old_hessian = original.evaluate_dihedral_angle_based_bending_force_hessian(
        edge, order, pos, previous, edges, rest, lengths, 1.2, 0.1, 1.0 / 600.0
    )
    elastic, _hessian = cache._evaluate_bending(
        edge, order, pos, edges, rest, lengths, 1.0, 0.0, 1.0 / 600.0, anchors[edge]
    )
    theta = cache._angle(pos, edges, edge)[0]
    gradient = wp.vec3(0.0)
    if wp.abs(theta - rest[edge]) > 1.0e-6:
        gradient = -elastic / (lengths[edge] * (theta - rest[edge]))
    forces[tid] = force
    hessians[tid] = hessian
    old_forces[tid] = old_force
    old_hessians[tid] = old_hessian
    gradients[tid] = gradient


def _angle_numpy(positions):
    n1 = np.cross(positions[2] - positions[0], positions[3] - positions[0])
    n2 = np.cross(positions[3] - positions[1], positions[2] - positions[1])
    edge = positions[3] - positions[2]
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    edge /= np.linalg.norm(edge)
    return np.arctan2(np.dot(np.cross(n1, n2), edge), np.dot(n1, n2))


class TestMJVBDV2SurfaceCache(unittest.TestCase):
    def test_cpu_preserves_scalar_path(self):
        """Keep CPU output unchanged and avoid conditional graph APIs."""
        model = _cloth("cpu")
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            results = []
            for enabled in (False, True):
                solver = solver_type(
                    model, iterations=6, particle_enable_surface_cache=enabled, particle_enable_truncation_cache=enabled
                )
                self.assertIsNone(solver.surface_anchor_angles)
                self.assertIsNone(solver._particle_truncation_cache)
                state_in, state_out = model.state(), model.state()
                force = np.zeros_like(model.particle_q.numpy())
                force[-1, 0] = 0.1
                state_in.particle_f.assign(force)
                with mock.patch.object(wp, "capture_if", side_effect=AssertionError("unexpected capture_if")):
                    solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
                results.append(state_out.particle_q.numpy())
            np.testing.assert_array_equal(*results)

    def test_bending_force_and_gradient(self):
        """Match original forces and a double-precision finite-difference angle gradient."""
        rng = np.random.default_rng(617)
        points = rng.normal(size=(256, 3)).astype(np.float32) * 0.02
        pos = wp.array(points, dtype=wp.vec3, device="cpu")
        previous = wp.array(
            points + rng.normal(size=points.shape).astype(np.float32) * 1.0e-4, dtype=wp.vec3, device="cpu"
        )
        edges = wp.array(np.arange(256).reshape((-1, 4)), dtype=wp.int32, device="cpu")
        rest = wp.zeros(64, dtype=float, device="cpu")
        lengths = wp.ones(64, dtype=float, device="cpu")
        anchors = wp.zeros(64, dtype=wp.vec2, device="cpu")
        wp.launch(cache._prepare_anchor_angles, dim=64, inputs=[previous, edges], outputs=[anchors], device="cpu")
        force = wp.zeros(256, dtype=wp.vec3, device="cpu")
        old_force, gradient = wp.zeros_like(force), wp.zeros_like(force)
        hessian = wp.zeros(256, dtype=wp.mat33, device="cpu")
        old_hessian = wp.zeros_like(hessian)
        wp.launch(
            _compare_bending,
            dim=256,
            inputs=[pos, previous, edges, rest, lengths, anchors],
            outputs=[force, hessian, old_force, old_hessian, gradient],
            device="cpu",
        )
        for current, reference in ((force.numpy(), old_force.numpy()), (hessian.numpy(), old_hessian.numpy())):
            error = np.linalg.norm((current - reference).reshape((256, -1)), axis=1)
            scale = np.maximum(1.0e-6, np.linalg.norm(reference.reshape((256, -1)), axis=1))
            self.assertLess(float(np.max(error / scale)), 1.0e-4)

        exact = np.zeros_like(points, dtype=np.float64)
        epsilon = 1.0e-7
        for edge, quad in enumerate(points.astype(np.float64).reshape((-1, 4, 3))):
            for vertex in range(4):
                for axis in range(3):
                    perturbed = quad.copy()
                    perturbed[vertex, axis] += epsilon
                    plus = _angle_numpy(perturbed)
                    perturbed[vertex, axis] -= 2.0 * epsilon
                    minus = _angle_numpy(perturbed)
                    difference = (plus - minus + np.pi) % (2.0 * np.pi) - np.pi
                    exact[4 * edge + vertex, axis] = difference / (2.0 * epsilon)
        relative = np.linalg.norm(gradient.numpy() - exact, axis=1) / np.maximum(np.linalg.norm(exact, axis=1), 1.0e-6)
        self.assertLess(float(relative.max()), 1.0e-4)

    def test_bending_degeneracy_and_wrapped_damping(self):
        """Keep boundary/degenerate hinges and the +/-pi damping wrap unchanged."""
        quad = np.array([[0, 0.1, 0], [0, 0.1, 0.001], [0, 0, 0], [0.1, 0, 0]], dtype=np.float32)
        points = np.tile(quad, (4, 1))
        previous = points.copy()
        previous[1, 2] = -0.001  # Cross the atan2 branch cut.
        points[4:8] = 0.0  # Current hinge is degenerate.
        previous[8:12] = 0.0  # Invalid damping anchor; elastic bending remains.
        edges = np.arange(16, dtype=np.int32).reshape((4, 4))
        edges[3, 0] = -1  # Boundary edge has no bending contribution.
        device = "cpu"
        pos = wp.array(points, dtype=wp.vec3, device=device)
        prev = wp.array(previous, dtype=wp.vec3, device=device)
        indices = wp.array(edges, dtype=wp.int32, device=device)
        anchors = wp.zeros(4, dtype=wp.vec2, device=device)
        wp.launch(cache._prepare_anchor_angles, dim=4, inputs=[prev, indices], outputs=[anchors], device=device)
        rest = wp.zeros(4, dtype=float, device=device)
        lengths = wp.ones(4, dtype=float, device=device)
        force = wp.zeros(16, dtype=wp.vec3, device=device)
        hessian = wp.zeros(16, dtype=wp.mat33, device=device)
        old_force, old_hessian, gradient = wp.zeros_like(force), wp.zeros_like(hessian), wp.zeros_like(force)
        wp.launch(
            _compare_bending,
            dim=16,
            inputs=[pos, prev, indices, rest, lengths, anchors],
            outputs=[force, hessian, old_force, old_hessian, gradient],
            device=device,
        )
        np.testing.assert_allclose(force.numpy(), old_force.numpy(), rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(hessian.numpy(), old_hessian.numpy(), rtol=1.0e-5, atol=1.0e-4)
        np.testing.assert_array_equal(force.numpy()[4:8], 0.0)
        np.testing.assert_array_equal(force.numpy()[12:16], 0.0)

    @unittest.skipUnless(wp.is_cuda_available(), "Surface tiles require CUDA")
    def test_refresh_anchors_and_replay_graph(self):
        """Refresh substep anchors and reproduce eager cached solves through CUDA replay."""
        model = _cloth("cuda:0")
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            solver = solver_type(model, iterations=6, particle_enable_surface_cache=True)
            self.assertIsNotNone(solver.surface_anchor_angles)
            initial = model.particle_q.numpy()
            initial[-1, 2] += 0.005
            state_in, state_out = model.state(), model.state()
            state_in.particle_q.assign(initial)
            solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
            eager = state_out.particle_q.numpy()
            expected = wp.zeros_like(solver.surface_anchor_angles)
            wp.launch(
                cache._prepare_anchor_angles,
                dim=model.edge_count,
                inputs=[wp.array(initial, dtype=wp.vec3, device=model.device), model.edge_indices],
                outputs=[expected],
                device=model.device,
            )
            np.testing.assert_array_equal(solver.surface_anchor_angles.numpy(), expected.numpy())
            graph_in, graph_out = model.state(), model.state()
            graph_in.particle_q.assign(initial)
            control = model.control()
            with wp.ScopedCapture(device=model.device) as capture:
                solver.step(graph_in, graph_out, control, None, 1.0 / 60.0)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(graph_out.particle_q.numpy(), eager)

            old_anchor = solver.surface_anchor_angles.numpy().copy()
            initial[-1, 2] += 0.005
            graph_in.particle_q.assign(initial)
            wp.capture_launch(capture.graph)
            self.assertGreater(float(np.max(np.abs(solver.surface_anchor_angles.numpy() - old_anchor))), 1.0e-4)

    @unittest.skipUnless(wp.is_cuda_available(), "Check CUDA opt-out policy")
    def test_preserve_grad_and_deterministic_paths(self):
        """Disable experimental caches outside validated execution modes."""
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            for model, options in (
                (_cloth("cuda:0", requires_grad=True), {}),
                (_cloth("cuda:0"), {"deterministic": wp.DeterministicMode.RUN_TO_RUN}),
            ):
                solver = solver_type(
                    model, particle_enable_surface_cache=True, particle_enable_truncation_cache=True, **options
                )
                self.assertIsNone(solver.surface_anchor_angles)
                self.assertIsNone(solver._particle_truncation_cache)


if __name__ == "__main__":
    unittest.main()

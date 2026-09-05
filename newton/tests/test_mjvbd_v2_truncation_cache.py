# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare cached DAT against the unchanged collision kernels in both backends."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2.particle_truncation_cache import ParticleTruncationCache
from newton._src.solvers.mjvbd_v2.vbd import particle_vbd_kernels as complete
from newton._src.solvers.mjvbd_v2.vbd_soft import particle_vbd_kernels as soft
from newton.tests.test_mjvbd_v2_surface_relaxation import SolverVBDComplete, SolverVBDSoft, _cloth


def _fixture(kernels, device):
    rng = np.random.default_rng(413)
    count = 32
    positions = rng.normal(size=(count * 4, 3)).astype(np.float32) * 0.02
    # Collinear coincident edges exercise robust-normal and zero-distance paths.
    positions[:4] = [[0, 0, 0], [0.03, 0, 0], [0.01, 0, 0], [0.04, 0, 0]]
    positions[4:8] = [[0, 0, 0], [0, 0, 0], [0.03, 0, 0], [0, 0.03, 0]]

    def array(values, dtype=wp.int32):
        return wp.array(values, dtype=dtype, device=device)

    q = array(positions, wp.vec3)
    delta = array(rng.normal(size=positions.shape).astype(np.float32) * 0.025, wp.vec3)
    triangles = array(np.arange(count * 4).reshape((count, 4))[:, 1:].copy())
    edge_indices = np.zeros((count * 2, 4), dtype=np.int32)
    edge_indices[:, 2:] = np.arange(count * 4).reshape((-1, 2))
    edges = array(edge_indices)
    info = kernels.TriMeshCollisionInfo()
    info.edge_colliding_edges_count = wp.ones(count * 2, dtype=wp.int32, device=device)
    info.edge_colliding_edges_buffer_sizes = wp.ones(count * 2, dtype=wp.int32, device=device)
    info.edge_colliding_edges_offsets = array(np.arange(count * 2 + 1))
    pairs = np.column_stack([np.arange(count * 2), np.arange(count * 2) ^ 1])
    pairs[-1, 1] = -1  # Removed candidates must never read uninitialized cache slots.
    info.edge_colliding_edges = array(pairs.ravel())
    vertex_counts = np.zeros(count * 4, dtype=np.int32)
    vertex_counts[::4] = 1
    info.vertex_colliding_triangles_count = array(vertex_counts)
    info.vertex_colliding_triangles_buffer_sizes = wp.ones(count * 4, dtype=wp.int32, device=device)
    info.vertex_colliding_triangles_offsets = array(np.arange(count * 4 + 1))
    info.vertex_colliding_triangles = array(np.column_stack([np.arange(count * 4), np.arange(count * 4) // 4]).ravel())
    info_array = wp.array([info], dtype=kernels.TriMeshCollisionInfo, device=device)
    solver = SimpleNamespace(
        device=wp.get_device(device),
        model=SimpleNamespace(
            particle_count=count * 4,
            tri_indices=triangles,
            edge_indices=edges,
            particle_color_groups=[array(np.arange(0, count * 4, 2))],
        ),
        pos_prev_collision_detection=q,
        particle_displacements=delta,
        trimesh_collision_info=info_array,
        # A device struct contains pointers, not ownership of its array fields.
        collision_info_host=info,
        trimesh_collision_detector=SimpleNamespace(
            edge_colliding_edges=info.edge_colliding_edges,
            vertex_colliding_triangles=info.vertex_colliding_triangles,
            edge_edge_parallel_epsilon=1.0e-6,
        ),
        particle_self_contact_evaluation_kernel_launch_size=count * 16,
        particle_conservative_bound_relaxation=0.8,
        particle_self_contact_margin=0.1,
        truncation_ts=wp.ones(count * 4, dtype=float, device=device),
    )
    if kernels is soft:
        solver.has_active_self_contact = wp.ones(1, dtype=wp.int32, device=device)
    return solver


def _reference_factors(solver, kernels, displacements):
    factors = wp.ones(solver.model.particle_count, dtype=float, device=solver.device)
    inputs = [
        solver.pos_prev_collision_detection,
        displacements,
        solver.model.tri_indices,
        solver.model.edge_indices,
        solver.trimesh_collision_info,
        solver.trimesh_collision_detector.edge_edge_parallel_epsilon,
        solver.particle_conservative_bound_relaxation,
    ]
    if kernels is soft:
        inputs.append(solver.has_active_self_contact)
    wp.launch(
        kernels.apply_planar_truncation_parallel_by_collision,
        dim=solver.particle_self_contact_evaluation_kernel_launch_size,
        inputs=inputs,
        outputs=[factors],
        device=solver.device,
    )
    return factors


class TestMJVBDV2TruncationCache(unittest.TestCase):
    def _check(self, device):
        for kernels in (complete, soft):
            solver = _fixture(kernels, device)
            cache = ParticleTruncationCache(solver, kernels)
            cache.rebuild(solver)
            out = wp.zeros_like(solver.pos_prev_collision_detection)
            initial_delta = solver.particle_displacements.numpy()
            for _refresh in range(3):
                # Rebuild on changes to the collision snapshot, not on every color.
                q = solver.pos_prev_collision_detection.numpy()
                q[8::4, 2] += 0.001
                solver.pos_prev_collision_detection.assign(q)
                cache.rebuild(solver)
                for repetition in range(3):
                    delta = initial_delta * (0.5 + repetition)
                    solver.particle_displacements.assign(delta)
                    expected_t = _reference_factors(solver, kernels, solver.particle_displacements)
                    inputs = [
                        solver.pos_prev_collision_detection,
                        solver.particle_displacements,
                        solver.model.tri_indices,
                        solver.model.edge_indices,
                        solver.trimesh_collision_info,
                        1.0e-6,
                        0.8,
                        cache._active,
                        cache.geometry,
                    ]
                    wp.launch(
                        cache._truncate,
                        dim=solver.particle_self_contact_evaluation_kernel_launch_size,
                        inputs=inputs,
                        outputs=[solver.truncation_ts],
                        device=device,
                    )
                    np.testing.assert_allclose(
                        solver.truncation_ts.numpy(), expected_t.numpy(), rtol=2.0e-5, atol=1.0e-6
                    )
                    reference_delta = wp.zeros_like(solver.particle_displacements)
                    reference_out = wp.zeros_like(out)
                    wp.launch(
                        kernels.apply_truncation_ts,
                        dim=solver.model.particle_count,
                        inputs=[solver.pos_prev_collision_detection, solver.particle_displacements, expected_t, 0.04],
                        outputs=[reference_delta, reference_out],
                        device=device,
                    )
                    # Re-running the factor min is idempotent; apply must reset every consumed slot.
                    cache.apply(solver, out, None)
                    np.testing.assert_allclose(out.numpy(), reference_out.numpy(), rtol=2.0e-5, atol=1.0e-7)
                    np.testing.assert_allclose(
                        solver.particle_displacements.numpy(), reference_delta.numpy(), rtol=2.0e-5, atol=1.0e-7
                    )
                    np.testing.assert_array_equal(solver.truncation_ts.numpy(), 1.0)

            if kernels is soft:
                # Exercise active -> inactive selected-color updates -> active.
                solver.has_active_self_contact.zero_()
                cache.rebuild(solver)
                solver.particle_displacements.assign(initial_delta)
                out.zero_()
                ids = solver.model.particle_color_groups[0]
                cache.apply(solver, out, ids)
                np.testing.assert_array_equal(out.numpy()[1::2], 0.0)
                np.testing.assert_array_equal(solver.particle_displacements.numpy()[1::2], initial_delta[1::2])
                np.testing.assert_array_equal(solver.truncation_ts.numpy(), 1.0)
                solver.has_active_self_contact.fill_(1)
                cache.rebuild(solver)

            if solver.device.is_cuda:
                solver.particle_displacements.assign(initial_delta)
                cache.rebuild(solver)
                cache.apply(solver, out, None)
                expected = out.numpy()
                solver.particle_displacements.assign(initial_delta)
                with wp.ScopedCapture(device=solver.device) as capture:
                    cache.rebuild(solver)
                    cache.apply(solver, out, None)
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(out.numpy(), expected)
                # Replaying again with changed frozen positions must refresh the geometry.
                changed = solver.pos_prev_collision_detection.numpy()
                changed[:, 0] += 0.002
                solver.pos_prev_collision_detection.assign(changed)
                solver.particle_displacements.assign(initial_delta)
                wp.capture_launch(capture.graph)
                replay = out.numpy()
                solver.particle_displacements.assign(initial_delta)
                cache.rebuild(solver)
                cache.apply(solver, out, None)
                np.testing.assert_array_equal(out.numpy(), replay)

    def test_cpu_reference(self):
        """Match ordinary DAT factors and updates on CPU."""
        self._check("cpu")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required")
    def test_cuda_reference_and_replay(self):
        """Match ordinary DAT and refresh cached geometry during CUDA replay."""
        self._check("cuda:0")

    def test_capacity_guards(self):
        """Reject oversized caches and capture-time buffer growth."""
        solver = _fixture(soft, "cpu")
        cache = ParticleTruncationCache(solver, soft)
        solver.trimesh_collision_detector.edge_colliding_edges = SimpleNamespace(size=20_000_000)
        with self.assertRaisesRegex(ValueError, "256 MiB"):
            cache._ensure_capacity(solver)
        solver.device = SimpleNamespace(is_capturing=True)
        with self.assertRaisesRegex(RuntimeError, "outside CUDA capture"):
            cache._ensure_capacity(solver)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA is required")
    def test_native_collision_refresh(self):
        """Refresh cached planes at every detection, including inside an iteration."""
        model = _cloth("cuda:0")
        initial = model.particle_q.numpy()
        right = initial[:, 0] > 0.16
        initial[right, 0] = 0.32 - initial[right, 0]
        initial[right, 2] = 0.002
        control = model.control()
        for solver_type in (SolverVBDComplete, SolverVBDSoft):
            outputs = []
            for enabled in (False, True):
                solver = solver_type(
                    model,
                    iterations=3,
                    particle_enable_self_contact=True,
                    particle_collision_detection_interval=1,
                    particle_self_contact_radius=0.003,
                    particle_self_contact_margin=0.01,
                    particle_enable_truncation_cache=enabled,
                )
                state_in, state_out = model.state(), model.state()
                state_in.particle_q.assign(initial)
                solver.step(state_in, state_out, control, None, 1.0 / 600.0)
                outputs.append(state_out.particle_q.numpy())
                self.assertTrue(np.all(np.isfinite(outputs[-1])))
                if enabled:
                    self.assertIsNotNone(solver._particle_truncation_cache)
                    np.testing.assert_array_equal(solver.truncation_ts.numpy(), 1.0)
                    # Kernels are warm; graph capture must not allocate or reuse
                    # geometry from a preceding substep's collision snapshot.
                    state_in.particle_q.assign(initial)
                    with wp.ScopedCapture(device=model.device) as capture:
                        solver.step(state_in, state_out, control, None, 1.0 / 600.0)
                    wp.capture_launch(capture.graph)
                    np.testing.assert_allclose(state_out.particle_q.numpy(), outputs[-1], atol=1.0e-5, rtol=1.0e-5)
            np.testing.assert_allclose(*outputs, atol=1.0e-5, rtol=1.0e-5)


if __name__ == "__main__":
    unittest.main()

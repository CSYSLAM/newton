# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check independent PCG step lengths on uncoupled frozen systems."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2.contact_projection import ContactProjection, project_records
from newton._src.solvers.mjvbd_v2.coupled_free_body.component_pcg import (
    ComponentPCG,
    _classify_components,
    _initialize_components,
    _join_contacts,
)
from newton._src.solvers.mjvbd_v2.coupled_free_body.two_level import TwoLevelPCG


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA tiles")
class TestComponentPCG(unittest.TestCase):
    def test_contact_connectivity(self):
        """Merge all contact paths without connecting detached rigid islands."""
        device = "cuda:0"
        rng = np.random.default_rng(518)
        count, particles = 193, 7
        edges = np.concatenate(
            [
                np.column_stack((np.arange(6, 92), np.arange(7, 93))),
                rng.integers(93, count, (450, 2)),
            ]
        ).astype(np.int32)
        data = ContactProjection(count, len(edges), device).data
        data.edge_keys.fill_(-1)
        wp.copy(data.edge_keys, wp.array(np.arange(len(edges)), dtype=wp.int64, device=device))
        wp.copy(data.edge_clusters, wp.array(edges, dtype=wp.vec2i, device=device))
        parent = wp.empty(count, dtype=int, device=device)
        component = wp.empty_like(parent)
        with wp.ScopedCapture(device=device) as capture:
            wp.launch(_initialize_components, count, [particles, parent], device=device)
            wp.launch(_join_contacts, data.edge_keys.size, [data, parent], device=device)
            wp.launch(_classify_components, count, [parent, component], device=device)
        expected = np.ones(count, dtype=np.int32)
        expected[:93] = 0
        for _ in range(8):
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(component.numpy(), expected)

    def test_detached_load_does_not_change_particle_solve(self):
        """Match isolated particle PCG despite large detached rigid residuals."""
        device = wp.get_device("cuda:0")
        particles, count = 16, 64
        rng = np.random.default_rng(834)
        diagonal = np.exp(rng.uniform(0, 2, (count, 3))).astype(np.float32)
        diagonal[particles:] *= 1000
        rhs = rng.normal(size=(count, 3)).astype(np.float32)
        matrix = np.zeros((count, count, 3, 3), dtype=np.float32)
        for row in range(count):
            matrix[row, row] = np.diag(diagonal[row])
        for row in range(count - 1):
            if row == particles - 1:
                continue
            spring = np.diag(np.exp(rng.uniform(1, 6, 3))).astype(np.float32)
            matrix[row, row] += spring
            matrix[row + 1, row + 1] += spring
            matrix[row, row + 1] = -spring
            matrix[row + 1, row] = -spring

        def run(size, partitioned, load_scale=1.0, *, bridge=False):
            groups = np.full(size, -1, dtype=np.int32)
            groups[:particles] = 0
            ritz = SimpleNamespace(
                model=SimpleNamespace(device=device),
                dofs=6,
                groups=wp.array(groups, dtype=int, device=device),
                q=wp.zeros(size, dtype=wp.vec3, device=device),
                centers=wp.zeros(1, dtype=wp.vec3, device=device),
                symmetric=wp.array(np.eye(6, dtype=np.float32), device=device),
                project=lambda *args: None,
            )
            solver = ComponentPCG(ritz, size, particles) if partitioned else TwoLevelPCG(ritz, inverse_factor=True)
            contacts = ContactProjection(size, size, device)
            contacts.reset()
            current_matrix = matrix.copy()
            if bridge:
                records = np.full((size - particles, 4), -1, dtype=np.int32)
                records[:, 0] = 0
                records[:, 1] = np.arange(particles, size)
                wp.launch(
                    project_records,
                    len(records),
                    [
                        wp.array(records, dtype=wp.vec4i, device=device),
                        wp.array(np.tile([1, -1, 0, 0], (len(records), 1)), dtype=wp.vec4, device=device),
                        wp.array(np.tile(np.eye(3), (len(records), 1, 1)), dtype=wp.mat33, device=device),
                        wp.array(np.arange(size), dtype=int, device=device),
                        contacts.data,
                    ],
                    device=device,
                )
                current_matrix[0, 0] += len(records) * np.eye(3)
                for row in range(particles, size):
                    current_matrix[row, row] += np.eye(3)
            forces = rhs[:size].copy()
            forces[particles:] *= load_scale
            offsets, columns, blocks, slots = [0], [], [], []
            for row in range(size):
                for col in range(size):
                    if np.any(current_matrix[row, col]):
                        if row == col:
                            slots.append(len(columns))
                        columns.append(col)
                        blocks.append(current_matrix[row, col])
                offsets.append(len(columns))
            inputs = [
                size,
                wp.array(offsets, dtype=int, device=device),
                wp.array(columns, dtype=int, device=device),
                wp.array(slots, dtype=int, device=device),
                wp.array(blocks, dtype=wp.mat33, device=device),
                wp.array(forces, dtype=wp.vec3, device=device),
                8,
                False,
                0.0,
                contacts.data,
            ]
            outputs = [wp.zeros(size, dtype=wp.vec3, device=device) for _ in range(5)]
            outputs += [
                wp.empty(size, dtype=wp.mat33, device=device),
                wp.zeros(1, dtype=int, device=device),
                wp.zeros(16, device=device),
                wp.zeros(2, dtype=int, device=device),
            ]
            solver.solve(inputs, outputs)
            active = wp.ones(1, dtype=int, device=device)
            with wp.ScopedCapture(device=device) as capture:
                wp.capture_if(active, on_true=lambda: solver.solve(inputs, outputs))
            wp.capture_launch(capture.graph)
            self.assertEqual(outputs[6].numpy()[0], 0)
            return outputs[0].numpy()

        isolated = run(particles, False)
        first = run(count, True)
        second = run(count, True, 1000.0)
        np.testing.assert_array_equal(first[:particles], isolated)
        np.testing.assert_array_equal(second[:particles], isolated)
        np.testing.assert_array_equal(run(count, True, bridge=True), run(count, False, bridge=True))
        dense = matrix.transpose(0, 2, 1, 3).reshape(3 * count, 3 * count).astype(np.float64)
        exact = np.linalg.solve(dense, rhs.ravel()).reshape(count, 3)
        global_result = run(count, False)
        independent_error = (first - exact).ravel()
        global_error = (global_result - exact).ravel()
        self.assertLess(independent_error @ dense @ independent_error, global_error @ dense @ global_error)


if __name__ == "__main__":
    unittest.main()

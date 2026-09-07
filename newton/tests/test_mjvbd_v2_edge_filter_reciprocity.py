# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check that built-in EE filtering preserves reciprocal contact candidates."""

import unittest

import warp as wp

from newton._src.solvers.mjvbd_v2.vbd import tri_mesh_collision as complete
from newton._src.solvers.mjvbd_v2.vbd_soft import tri_mesh_collision as soft
from newton.tests.test_mjvbd_v2_particle_multilevel import _build_cloth


class TestMJVBDV2EdgeFilterReciprocity(unittest.TestCase):
    def test_detected_pairs_are_reciprocal(self):
        devices = ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else [])
        for device in devices:
            model = _build_cloth(device, dim_x=8, dim_y=8)
            for backend in (soft, complete):
                with self.subTest(device=device, backend=backend.__name__):
                    detector = backend.TriMeshCollisionDetector(
                        model,
                        topological_contact_filter_threshold=2,
                        edge_collision_buffer_pre_alloc=model.edge_count,
                        edge_collision_buffer_max_alloc=model.edge_count,
                    )
                    detector.edge_edge_collision_detection(max_query_radius=10.0)
                    counts = detector.edge_colliding_edges_count.numpy()
                    offsets = detector.edge_colliding_edges_offsets.numpy()
                    pairs = detector.edge_colliding_edges.numpy().reshape(-1, 2)
                    rows = [set(pairs[offsets[i] : offsets[i] + counts[i], 1]) for i in range(model.edge_count)]
                    self.assertGreater(sum(map(len, rows)), 0)
                    for edge, row in enumerate(rows):
                        self.assertLessEqual(counts[edge], model.edge_count)
                        for other in row:
                            self.assertIn(edge, rows[other])

    def test_external_directed_filters_are_not_rewritten(self):
        model = _build_cloth("cpu", dim_x=3, dim_y=3)
        for backend in (soft, complete):
            with self.subTest(backend=backend.__name__):
                detector = backend.TriMeshCollisionDetector(
                    model, topological_contact_filter_threshold=1, external_edge_edge_filtering_map={0: {2}}
                )
                offsets = detector.edge_filtering_list_offsets.numpy()
                values = detector.edge_filtering_list.numpy()
                self.assertEqual(values.tolist(), [2])
                self.assertEqual(int(offsets[1]), 1)
                self.assertEqual(int(offsets[3] - offsets[2]), 0)

    def test_filters_preserve_legacy_candidates_and_reciprocity(self):
        """Restore missing directed counterparts without removing legacy contacts."""
        model = _build_cloth("cpu", dim_x=8, dim_y=8)
        stencils = model.edge_indices.numpy()
        endpoints = stencils[:, 2:]
        neighbors = [set() for _ in range(model.particle_count)]
        for a, b in endpoints:
            neighbors[a].add(int(b))
            neighbors[b].add(int(a))
        for backend in (soft, complete):
            for rings in (2, 3):
                with self.subTest(backend=backend.__name__, rings=rings):
                    detector = backend.TriMeshCollisionDetector(model, topological_contact_filter_threshold=rings)
                    indices = detector.edge_filtering_list.numpy()
                    offsets = detector.edge_filtering_list_offsets.numpy()
                    rows = [set(indices[offsets[i] : offsets[i + 1]]) for i in range(model.edge_count)]
                    legacy_rows = []
                    for a, b in endpoints:
                        reached = {int(a), int(b)}
                        for _ in range(rings - 1):
                            reached |= {neighbor for vertex in tuple(reached) for neighbor in neighbors[vertex]}
                        legacy_rows.append(
                            {
                                other
                                for other, (c, d) in enumerate(endpoints)
                                if a not in stencils[other]
                                and b not in stencils[other]
                                and (c in reached or d in reached)
                            }
                        )
                    self.assertTrue(
                        any(edge not in legacy_rows[other] for edge, row in enumerate(legacy_rows) for other in row)
                    )
                    for edge, row in enumerate(rows):
                        expected = {other for other in legacy_rows[edge] if edge in legacy_rows[other]}
                        self.assertSetEqual(rows[edge], expected)
                        self.assertTrue(row <= legacy_rows[edge])
                        for other in row:
                            self.assertIn(edge, rows[other])


if __name__ == "__main__":
    unittest.main()

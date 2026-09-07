# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check contact-row layout conversion independently of contact assembly."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2 import contact_projection as cp
from newton.tests.test_mjvbd_v2_contact_projection import _multiply


class TestContactRows(unittest.TestCase):
    def test_compaction_preserves_products_and_reset(self):
        """Preserve linked-row summation order and invalidate packed rows on reset."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device):
                count, clusters = 2048, 129
                rng = np.random.default_rng(19)
                particles = rng.integers(0, clusters - 1, size=(count, 4), dtype=np.int32)
                weights = rng.normal(size=(count, 4)).astype(np.float32)
                weights -= weights.mean(axis=1)[:, None]
                inputs = [
                    wp.array(particles, dtype=wp.vec4i, device=device),
                    wp.array(weights, dtype=wp.vec4, device=device),
                    wp.array(np.tile(np.eye(3, dtype=np.float32), (count, 1, 1)), dtype=wp.mat33, device=device),
                    wp.array(np.arange(clusters), dtype=wp.int32, device=device),
                ]
                op = cp.ContactProjection(clusters, count, device)
                diagonal = wp.zeros(clusters, dtype=wp.mat33, device=device)
                vector = wp.array(rng.normal(size=(clusters, 3)), dtype=wp.vec3, device=device)
                linked = wp.empty_like(vector)
                packed = wp.empty_like(vector)
                op.compact()  # Reserve workspace outside capture; empty rows are valid.

                def run(
                    op=op,
                    count=count,
                    inputs=inputs,
                    clusters=clusters,
                    diagonal=diagonal,
                    vector=vector,
                    linked=linked,
                    packed=packed,
                    device=device,
                ):
                    op.reset()
                    wp.launch(cp.project_records, dim=count, inputs=[*inputs, op.data], device=device)
                    wp.launch(
                        _multiply, dim=clusters, inputs=[diagonal, vector, op.data], outputs=[linked], device=device
                    )
                    op.compact()
                    wp.launch(
                        _multiply, dim=clusters, inputs=[diagonal, vector, op.data], outputs=[packed], device=device
                    )

                run()
                self.assertEqual(int(op.data.overflow.numpy()[0]), 0)
                np.testing.assert_array_equal(linked.numpy(), packed.numpy())
                if device == "cuda:0":
                    with wp.ScopedCapture(device=device) as capture:
                        run()
                    for _ in range(3):
                        wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(linked.numpy(), packed.numpy())
                op.reset()
                op.compact()
                wp.launch(_multiply, dim=clusters, inputs=[diagonal, vector, op.data], outputs=[packed], device=device)
                np.testing.assert_array_equal(packed.numpy(), np.zeros((clusters, 3)))


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check split PCG against the original single-block recurrence."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2 import contact_projection as cp
from newton._src.solvers.mjvbd_v2 import particle_multilevel as ml
from newton._src.solvers.mjvbd_v2.coarse_pcg_split import SplitCoarsePCG
from newton._src.solvers.mjvbd_v2.contact_projection import ContactProjectionData
from newton.tests.test_mjvbd_v2_particle_multilevel import _build_cloth


def system(count, device, mode="normal"):
    rng = np.random.default_rng(7)
    identity = np.eye(3, dtype=np.float32)
    columns = np.array([[(i - 1) % count, i, (i + 1) % count] for i in range(count)], dtype=np.int32).ravel()
    blocks = np.tile(np.array([-identity, 4 * identity, -identity]), (count, 1, 1))
    rhs = rng.normal(size=(count, 3)).astype(np.float32)
    if mode == "zero":
        rhs[:] = 0
    if mode == "negative":
        blocks *= -1
    if mode == "nonfinite":
        rhs[0, 0] = np.nan
    contacts = ContactProjectionData()
    if mode == "overflow":
        contacts.overflow = wp.array([8], dtype=wp.int32, device=device)
    inputs = [
        count,
        wp.array(np.arange(count + 1) * 3, dtype=wp.int32, device=device),
        wp.array(columns, dtype=wp.int32, device=device),
        wp.array(np.arange(count) * 3 + 1, dtype=wp.int32, device=device),
        wp.array(blocks, dtype=wp.mat33, device=device),
        wp.array(rhs, dtype=wp.vec3, device=device),
        8,
        True,
        1e-4,
        contacts,
    ]
    outputs = [
        *[wp.zeros(count, dtype=wp.vec3, device=device) for _ in range(5)],
        wp.zeros(count, dtype=wp.mat33, device=device),
        wp.zeros(1, dtype=wp.int32, device=device),
        wp.zeros(13, dtype=float, device=device),
        wp.zeros(2, dtype=wp.int32, device=device),
    ]
    return inputs, outputs


class TestCoarsePCGSplit(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "Split scheduling requires CUDA")
    def test_projected_contact_recurrence(self):
        """Preserve every output with linked and packed projected contact rows."""
        device, count = "cuda:0", 1025
        inputs, expected = system(count, device)
        projection = cp.ContactProjection(count, count, device)
        projection.reset()
        particles = np.column_stack((np.arange(count), (np.arange(count) + 29) % count, np.full((count, 2), -1)))
        wp.launch(
            cp.project_records,
            dim=count,
            inputs=[
                wp.array(particles, dtype=wp.vec4i, device=device),
                wp.array(np.tile([1, -1, 0, 0], (count, 1)), dtype=wp.vec4, device=device),
                wp.array(np.tile(0.5 * np.eye(3), (count, 1, 1)), dtype=wp.mat33, device=device),
                wp.array(np.arange(count), dtype=wp.int32, device=device),
                projection.data,
            ],
            device=device,
        )
        self.assertEqual(int(projection.data.overflow.numpy()[0]), 0)
        # The base diagonal is strictly dominant even after adding these
        # symmetric contact off-diagonals; all variants share this matrix.
        inputs[-1] = projection.data
        wp.launch(
            ml._solve_energy_galerkin_pcg_persistent,
            dim=256,
            block_dim=256,
            inputs=inputs,
            outputs=expected,
            device=device,
        )
        solver = SplitCoarsePCG(device)
        for packed in (False, True):
            with self.subTest(packed=packed):
                if packed:
                    projection.compact()
                actual = [wp.zeros_like(array) for array in expected]
                solver.solve(inputs, actual)
                with wp.ScopedCapture(device=device) as capture:
                    solver.solve(inputs, actual)
                for _ in range(3):
                    wp.capture_launch(capture.graph)
                for result, reference in zip(actual, expected, strict=True):
                    np.testing.assert_array_equal(result.numpy(), reference.numpy())

    @unittest.skipUnless(wp.is_cuda_available(), "Automatic split dispatch requires CUDA")
    def test_large_surface_dispatch_preserves_correction(self):
        """Select the large-system route and preserve correction and graph replay."""
        model = _build_cloth("cuda:0", dim_x=64, dim_y=64)
        model.tri_materials.zero_()
        model.edge_bending_properties.zero_()
        options = {
            "operator": "galerkin",
            "cluster_size": 4,
            "coarse_iterations": 8,
            "coupling": 0.5,
            "relaxation": 0.1,
            "max_radius_fraction": 0.05,
            "minimum_residual_reduction": 1e-4,
            "max_clamp_fraction": 0.5,
        }
        correction = ml.ParticleMultilevelCorrection(model, **options)
        self.assertTrue(correction.coarse_use_split_pcg)
        small = ml.ParticleMultilevelCorrection(model, **{**options, "coarse_iterations": 3})
        self.assertFalse(small.coarse_use_split_pcg)
        correction.local_hessians.fill_(wp.mat33(np.eye(3)))
        correction.local_correction.assign(np.random.default_rng(12).normal(size=(model.particle_count, 3)) * 1e-5)
        expected = wp.zeros_like(model.particle_q)
        actual = wp.zeros_like(model.particle_q)
        correction.coarse_use_split_pcg = False
        correction.restrict_and_prolong(model, model.particle_q, expected, 0.01)
        metrics = correction.runtime_metrics.numpy()
        correction.coarse_use_split_pcg = True
        correction.restrict_and_prolong(model, model.particle_q, actual, 0.01)
        with wp.ScopedCapture(device=model.device) as capture:
            actual.zero_()
            correction.restrict_and_prolong(model, model.particle_q, actual, 0.01)
        for _ in range(3):
            wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(expected.numpy(), actual.numpy())
        np.testing.assert_array_equal(metrics, correction.runtime_metrics.numpy())

    def test_recurrence_and_capture(self):
        """Preserve PCG outputs across block boundaries and CUDA graph replay."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            for count in (1, 127, 257, 1025):
                with self.subTest(device=device, count=count):
                    inputs, reference = system(count, device)
                    candidate = [wp.zeros_like(array) for array in reference]
                    wp.launch(
                        ml._solve_energy_galerkin_pcg_persistent,
                        dim=256,
                        block_dim=256,
                        inputs=inputs,
                        outputs=reference,
                        device=device,
                    )
                    solver = SplitCoarsePCG(device)
                    solver.solve(inputs, candidate)
                    if device == "cuda:0":
                        with wp.ScopedCapture(device=device) as capture:
                            solver.solve(inputs, candidate)
                        for _ in range(3):
                            wp.capture_launch(capture.graph)
                    for actual, expected in zip(candidate, reference, strict=True):
                        np.testing.assert_array_equal(actual.numpy(), expected.numpy())

    def test_rejection_and_zero_rhs(self):
        """Retain overflow, curvature and nonfinite rejection without host branching."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            for mode in ("overflow", "negative", "nonfinite", "zero"):
                with self.subTest(device=device, mode=mode):
                    inputs, reference = system(257, device, mode)
                    candidate = [wp.zeros_like(array) for array in reference]
                    # Rejected contacts must preserve the old kernel's
                    # untouched scratch, including buffers from a prior solve.
                    for index in range(1, 5):
                        reference[index].fill_(wp.vec3(7.0))
                        candidate[index].fill_(wp.vec3(7.0))
                    wp.launch(
                        ml._solve_energy_galerkin_pcg_persistent,
                        dim=256,
                        block_dim=256,
                        inputs=inputs,
                        outputs=reference,
                        device=device,
                    )
                    SplitCoarsePCG(device).solve(inputs, candidate)
                    for actual, expected in zip(candidate, reference, strict=True):
                        np.testing.assert_array_equal(actual.numpy(), expected.numpy())


if __name__ == "__main__":
    unittest.main()

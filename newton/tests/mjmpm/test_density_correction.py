# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Acceptance tests for opt-in liquid density feedback in private MJMPM."""

import unittest
from dataclasses import replace
from itertools import product

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjmpm.mpm.solver_implicit_mpm import SolverImplicitMPM


def _model(device):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverImplicitMPM.register_custom_attributes(builder)
    # Two cell volumes of reference material initially occupy one cell.
    for point in product((0.01, 0.02, 0.03, 0.04), (0.01, 0.02), (0.01, 0.02)):
        builder.add_particle(
            point,
            (0.0, 0.0, 0.0),
            0.125,
            0.025,
            custom_attributes={"mpm:friction": 0.0, "mpm:tensile_yield_ratio": 1.0},
        )
    return builder.finalize(device=device)


def _config(device):
    return SolverImplicitMPM.Config(
        voxel_size=0.1,
        grid_type="sparse" if wp.get_device(device).is_cuda else "dense",
        max_active_cell_count=64,
        grid_padding=0,
        strain_basis="P0",
        critical_fraction=1.0,
        density_correction=0.1,
        warmstart_mode="none",
        solver="gs",
        max_iterations=128,
        transfer_scheme="pic",
    )


class TestDensityCorrection(unittest.TestCase):
    def test_signed_offset_is_opt_in(self):
        """Do not reinterpret existing negative offsets unless density feedback is enabled."""
        from newton._src.solvers.mjmpm.mpm import rheology_solver_kernels as kernels  # noqa: PLC0415

        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device), wp.ScopedDevice(device):
                offsets = wp.array([-0.2, 0.0, 0.2], dtype=float)
                results = []
                for enabled in (False, True):
                    rhs = wp.zeros(3, dtype=kernels.vec6)
                    wp.launch(
                        kernels.preprocess_stress_and_strain,
                        dim=3,
                        inputs=[
                            offsets,
                            rhs,
                            wp.zeros(3, dtype=kernels.vec6),
                            wp.zeros(3, dtype=kernels.YieldParamVec),
                            enabled,
                        ],
                    )
                    results.append(rhs.numpy())
                np.testing.assert_array_equal(results[0][0], 0.0)
                np.testing.assert_array_equal(results[0][1:], results[1][1:])
                self.assertGreater(float(np.linalg.norm(results[1][0])), 0.0)

    def test_overfilled_cell_expands_without_net_impulse(self):
        """Density feedback corrects stationary compression that div(v)=0 cannot detect."""
        for device in ["cpu"] + (["cuda:0"] if wp.is_cuda_available() else []):
            with self.subTest(device=device), wp.ScopedDevice(device):
                model = _model(device)
                initial = model.particle_q.numpy()
                config = _config(device)
                off = SolverImplicitMPM(model, replace(config, density_correction=0.0))
                unchanged, unchanged_out = model.state(), model.state()
                off.step(unchanged, unchanged_out, None, None, 0.002)
                np.testing.assert_array_equal(unchanged_out.particle_q.numpy(), initial)

                solver = SolverImplicitMPM(model, config)
                state, output = model.state(), model.state()
                solver.step(state, output, None, None, 0.002)
                expected = output.particle_q.numpy().copy()
                velocity = output.particle_qd.numpy()
                self.assertTrue(np.isfinite(expected).all())
                self.assertGreater(float(np.sum((expected - initial) * (initial - initial.mean(axis=0)))), 1.0e-8)
                np.testing.assert_allclose(velocity.sum(axis=0), 0.0, atol=2.0e-4)
                if wp.get_device(device).is_cuda:
                    with wp.ScopedCapture() as capture:
                        solver.step(state, output, None, None, 0.002)
                    wp.capture_launch(capture.graph)
                    np.testing.assert_allclose(output.particle_q.numpy(), expected, atol=1.0e-6)

    def test_rejects_unsupported_settings(self):
        with wp.ScopedDevice("cpu"):
            model = _model("cpu")
            config = _config("cpu")
            for changes in (
                {"density_correction": -0.1},
                {"density_correction": 1.1},
                {"density_correction": float("nan")},
                {"density_correction": float("inf")},
                {"strain_basis": "P1d"},
                {"critical_fraction": 0.0},
                {"warmstart_mode": "particles"},
            ):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    SolverImplicitMPM(model, replace(config, **changes))
            model.mpm.young_modulus.fill_(1.0e4)
            with self.assertRaisesRegex(ValueError, "non-compliant"):
                SolverImplicitMPM(model, config)
            model.mpm.young_modulus.fill_(1.0e15)
            model.mpm.hardening.fill_(1.0)
            with self.assertRaisesRegex(ValueError, "without hardening"):
                SolverImplicitMPM(model, config)


if __name__ == "__main__":
    unittest.main()

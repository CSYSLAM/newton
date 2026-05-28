# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import unittest

import numpy as np
import warp as wp

import newton


def _build_tet_model() -> newton.Model:
    builder = newton.ModelBuilder()
    builder.add_soft_mesh(
        pos=(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0.0, 0.0, 0.0),
        vertices=[
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ],
        indices=[0, 1, 2, 3],
        density=1.0,
        k_mu=1.0e3,
        k_lambda=2.0e3,
        k_damp=0.0,
    )
    builder.add_shape_plane(body=-1)
    return builder.finalize(device="cpu")


class TestSolverIPC(unittest.TestCase):
    def test_public_symbol_is_exported(self):
        self.assertIs(newton.solvers.SolverIPC, newton._src.solvers.SolverIPC)

    def test_constructor_requires_uipc_or_instantiates(self):
        model = _build_tet_model()
        has_uipc = importlib.util.find_spec("uipc") is not None

        if not has_uipc:
            with self.assertRaises(ImportError) as exc_info:
                newton.solvers.SolverIPC(model)
            self.assertIn("uipc", str(exc_info.exception))
            return

        solver = newton.solvers.SolverIPC(model)
        self.assertIsInstance(solver, newton.solvers.SolverIPC)

    def test_rejects_triangle_only_models(self):
        builder = newton.ModelBuilder()
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2], dtype=np.int32)
        builder.add_cloth_mesh(
            pos=(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            density=1.0,
            tri_ke=1.0,
            tri_ka=1.0,
            tri_kd=0.0,
        )
        model = builder.finalize(device="cpu")

        if importlib.util.find_spec("uipc") is None:
            with self.assertRaises(ImportError):
                newton.solvers.SolverIPC(model)
        else:
            with self.assertRaises(ValueError):
                newton.solvers.SolverIPC(model)

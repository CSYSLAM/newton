# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Eliminate exactly masked LM columns while preserving the full IK iteration."""

from types import SimpleNamespace

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def gather_columns(jacobian: wp.array3d[float], active: wp.array[int], compact: wp.array3d[float]):
    batch, row, column = wp.tid()
    compact[batch, row, column] = jacobian[batch, row, active[column]]


@wp.kernel(enable_backward=False)
def scatter_delta(compact: wp.array2d[float], inverse: wp.array[int], full: wp.array2d[float]):
    batch, column = wp.tid()
    index = inverse[column]
    value = float(0.0)
    if index >= 0:
        value = compact[batch, index]
    full[batch, column] = value


class CompactLMSolve:
    def __init__(self, optimizer):
        from .ik_lm_optimizer import IKOptimizerLM  # noqa: PLC0415 - Avoid a circular optimizer import.

        if optimizer.joint_dof_mask is None:
            raise ValueError("Compaction requires an explicit immutable DOF mask")
        active = np.flatnonzero(optimizer.joint_dof_mask.numpy())
        if not 0 < len(active) < optimizer.n_dofs:
            raise ValueError("Compaction requires both active and masked DOFs")
        self.active = wp.array(active, dtype=int, device=optimizer.device)
        inverse = np.full(optimizer.n_dofs, -1, dtype=np.int32)
        inverse[active] = np.arange(len(active))
        self.inverse = wp.array(inverse, dtype=int, device=optimizer.device)
        self.jacobian = wp.empty((optimizer.n_batch, optimizer.n_residuals, len(active)), device=optimizer.device)
        self.delta = wp.empty((optimizer.n_batch, len(active)), device=optimizer.device)
        specialized = IKOptimizerLM._build_specialized((len(active), optimizer.n_residuals, "cuda"))
        self.owner = SimpleNamespace(n_batch=optimizer.n_batch, device=optimizer.device, TILE_THREADS=32)
        self.solve = specialized._solve_tiled
        self.device = optimizer.device

    def __call__(self, jacobian, residuals, lambdas, delta, predicted):
        wp.launch(
            gather_columns, dim=self.jacobian.shape, inputs=[jacobian, self.active, self.jacobian], device=self.device
        )
        self.solve(self.owner, self.jacobian, residuals, lambdas, self.delta, predicted)
        wp.launch(scatter_delta, dim=delta.shape, inputs=[self.delta, self.inverse, delta], device=self.device)

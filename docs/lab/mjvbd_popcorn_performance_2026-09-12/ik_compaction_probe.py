# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Eliminate exactly masked LM columns while preserving the full IK iteration."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

from newton._src.sim.ik.ik_lm_optimizer import IKOptimizerLM


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


def install_compact_ik(example, lock_kernel, *, shadow=False, serial_objectives=False):
    reference_graph = example.ik_graph
    optimizer = example.ik_solver._impl
    if serial_objectives:
        if optimizer.n_batch != 1:
            raise ValueError("Serial objective scheduling is being tested for one robot only")

        def evaluate_objectives(fn, *extra):
            for objective, offset in zip(optimizer.objectives, optimizer.residual_offsets, strict=True):
                fn(objective, offset, *extra)

        optimizer._parallel_for_objectives = evaluate_objectives
    compact = CompactLMSolve(optimizer)
    # Compile before graph capture; these work arrays are overwritten on the
    # next LM iteration and no joint coordinates are changed by this solve.
    compact(
        optimizer.jacobian, optimizer.residuals_3d, optimizer.lambda_values, optimizer.dq_dof, optimizer.pred_reduction
    )
    optimizer._solve_tiled = compact
    with wp.ScopedCapture(device=example.model.device) as capture:
        example.ik_solver.step(example.ik_q, example.ik_q, iterations=32)
        wp.launch(lock_kernel, len(example.lock_indices), [example.lock_indices, example.lock_values, example.ik_q])
    example.ik_graph = capture.graph
    if shadow:
        original_solve = example._solve_wrist_ik
        original_launch = wp.capture_launch
        counter = [0]

        def launch(graph, *args, **kwargs):
            if graph is not example.ik_graph or counter[0] % 120:
                return original_launch(graph, *args, **kwargs)
            initial = example.ik_q.numpy().copy()
            original_launch(reference_graph, *args, **kwargs)
            expected = example.ik_q.numpy().copy()
            example.ik_q.assign(initial)
            result = original_launch(graph, *args, **kwargs)
            actual = example.ik_q.numpy()
            error = float(np.max(np.abs(expected - actual)))
            print(f"Compact IK shadow at {example.sim_time:.3f}s: max joint difference {error:.9g} rad", flush=True)
            if error > 1e-4 or not np.isfinite(error):
                raise AssertionError("Compact IK differs from the full 32-iteration reference")
            return result

        def solve(targets):
            with mock.patch.object(wp, "capture_launch", launch):
                result = original_solve(targets)
            counter[0] += 1
            return result

        example._solve_wrist_ik = solve

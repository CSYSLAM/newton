# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Analytic LM batching and exact fixed-point elision with instance-owned state."""

import hashlib
from functools import cache

import warp as wp
from warp._src.context import Kernel

from .analytic_row_kernels import LAYOUTS
from .ik_objectives import IKObjectiveJointLimit, IKObjectivePosition, IKObjectiveRotation


@wp.func_native("union { float f; int i; } bits; bits.f = value; return bits.i;")
def _bits(value: float) -> int: ...


@wp.kernel(enable_backward=False)
def _changed(
    q: wp.array[float],
    previous_q: wp.array[float],
    residual: wp.array[float],
    previous_residual: wp.array[float],
    damping: wp.array[float],
    previous_damping: wp.array[float],
    active: wp.array[int],
):
    i = wp.tid()
    different = False
    if i < q.shape[0]:
        different = not wp.isfinite(q[i]) or _bits(q[i]) != _bits(previous_q[i])
    if i < residual.shape[0]:
        different = different or not wp.isfinite(residual[i]) or _bits(residual[i]) != _bits(previous_residual[i])
    if i < damping.shape[0]:
        different = different or not wp.isfinite(damping[i]) or _bits(damping[i]) != _bits(previous_damping[i])
    if different:
        wp.atomic_max(active, 0, 1)


@cache
def _build_batch(layout):
    """Specialize only the typed argument list; objective math is static source.

    The immutable cache contains kernels, never target arrays or solver buffers.
    Source generation is needed for Warp's statically typed heterogeneous
    argument list. It does not inspect or rewrite any installed Python method.
    """
    namespace = {"wp": wp, "__name__": __name__}
    branches, parameters = [], []
    offset = 0
    for index, (original, width) in enumerate(layout):
        function, structure, fields = LAYOUTS[original]
        namespace[f"function_{index}"] = function
        namespace[f"Inputs_{index}"] = structure
        parameters.append(f"input_{index}: Inputs_{index}")
        arguments = ", ".join([f"input_{index}.{field}" for field in fields] + ["batch", f"task - {offset}"])
        branches.append(f"    if task >= {offset} and task < {offset + width}:\n        function_{index}({arguments})")
        offset += width
    name = "analytic_batch"
    source = f"def {name}({', '.join(parameters)}):\n    batch, task = wp.tid()\n" + "\n".join(branches) + "\n"
    exec(compile(source, "<analytic_batch>", "exec"), namespace)
    signature = "_".join(f"{original.key}_{width}" for original, width in layout)
    # Warp includes the module name in cache filenames. Bound its length even
    # when a robot has many objectives, keeping distinct layouts in separate modules.
    signature_digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    kernel = Kernel(
        namespace[name],
        key=name,
        module=wp.get_module(f"{__name__}.batch_{signature_digest}"),
        source=source,
        options={"enable_backward": False},
    )
    return kernel, offset


class CudaFastLM:
    """Own one optimizer's batching and exact-cycle scratch buffers."""

    @staticmethod
    def supports(optimizer):
        allowed = (IKObjectivePosition, IKObjectiveRotation, IKObjectiveJointLimit)
        return (
            optimizer.device.is_cuda
            and not optimizer.model.requires_grad
            and wp.config.deterministic == wp.DeterministicMode.NOT_GUARANTEED
            and not optimizer.has_autodiff_objective
            and not optimizer.parallel_objectives
            and bool(optimizer.objectives)
            and all(type(obj) in allowed for obj in optimizer.objectives)
        )

    def __init__(self, optimizer):
        self.previous_q = wp.empty(optimizer.n_batch * optimizer.n_coords, device=optimizer.device)
        self.previous_residual = wp.empty(optimizer.n_batch * optimizer.n_residuals, device=optimizer.device)
        self.previous_damping = wp.empty_like(optimizer.lambda_values)
        self.active = wp.ones(1, dtype=int, device=optimizer.device)

    @staticmethod
    def _emit(optimizer, calls):
        layout = tuple((kernel, dim[1] if isinstance(dim, (tuple, list)) else 1) for kernel, dim, values in calls)
        kernel, width = _build_batch(layout)
        inputs = []
        for original, _dim, values in calls:
            _, structure, fields = LAYOUTS[original]
            record = structure()
            for field, value in zip(fields, values, strict=True):
                setattr(record, field, value)
            inputs.append(record)
        wp.launch(kernel, dim=(optimizer.n_batch, width), inputs=inputs, device=optimizer.device)

    def residuals(self, optimizer, ctx):
        calls = []

        def collect(kernel, *, dim, inputs, outputs, **kwargs):
            calls.append((kernel, dim, [*inputs, *outputs]))

        for obj, offset in zip(optimizer.objectives, optimizer.residual_offsets, strict=True):
            obj.compute_residuals(
                ctx.fk_body_q,
                ctx.joint_q,
                optimizer.model,
                ctx.residuals,
                offset,
                problem_idx=ctx.problem_idx,
                _launch=collect,
            )
        self._emit(optimizer, calls)

    def jacobian(self, optimizer, ctx):
        calls = []

        def collect(kernel, *, dim, inputs, outputs, **kwargs):
            calls.append((kernel, dim, [*inputs, *outputs]))

        for obj, offset in zip(optimizer.objectives, optimizer.residual_offsets, strict=True):
            obj.compute_jacobian_analytic(
                ctx.fk_body_q,
                ctx.joint_q,
                optimizer.model,
                ctx.jacobian_out,
                ctx.motion_subspace,
                offset,
                _launch=collect,
            )
        self._emit(optimizer, calls)

    def step(self, optimizer, joint_q_in, joint_q_out, iterations, step_size):
        if joint_q_in.shape != (optimizer.n_batch, optimizer.n_coords):
            raise ValueError("joint_q_in has incompatible shape")
        if joint_q_out.shape != (optimizer.n_batch, optimizer.n_coords):
            raise ValueError("joint_q_out has incompatible shape")
        if joint_q_in.ptr != joint_q_out.ptr:
            wp.copy(joint_q_out, joint_q_in)
        optimizer.lambda_values.fill_(optimizer.lambda_initial)
        self.active.fill_(1)
        for iteration in range(4):
            optimizer._step(joint_q_out, step_size=step_size, iteration=iteration)
        q = joint_q_out.flatten()
        residual = optimizer.residuals.flatten()
        for start in range(4, iterations - 3, 4):

            def block(start=start):
                wp.copy(self.previous_q, q)
                wp.copy(self.previous_residual, residual)
                wp.copy(self.previous_damping, optimizer.lambda_values)
                for iteration in range(start, start + 4):
                    optimizer._step(joint_q_out, step_size=step_size, iteration=iteration)
                self.active.zero_()
                wp.launch(
                    _changed,
                    max(q.size, residual.size, optimizer.n_batch),
                    [
                        q,
                        self.previous_q,
                        residual,
                        self.previous_residual,
                        optimizer.lambda_values,
                        self.previous_damping,
                        self.active,
                    ],
                    device=optimizer.device,
                )

            wp.capture_if(self.active, on_true=block)
        for iteration in range((iterations // 4) * 4, iterations):
            optimizer._step(joint_q_out, step_size=step_size, iteration=iteration)

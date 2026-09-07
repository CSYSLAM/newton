# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Split surface PCG: parallel sparse products, original block-ordered reductions."""

import warp as wp

from .contact_projection import ContactProjectionData, off_diagonal_product


@wp.kernel(enable_backward=False)
def _initialize(
    count: int,
    slots: wp.array[wp.int32],
    blocks: wp.array[wp.mat33],
    rhs: wp.array[wp.vec3],
    contacts: ContactProjectionData,
    solution: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
    preconditioned: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    inverse: wp.array[wp.mat33],
    status: wp.array[wp.int32],
    metrics: wp.array[float],
    counters: wp.array[wp.int32],
    rz_state: wp.array[float],
):
    lane = wp.tid()
    stride = wp.block_dim()
    invalid = wp.bool(False)
    if contacts.overflow:
        invalid = contacts.overflow[0] != 0
    for index in range(lane, metrics.shape[0], stride):
        metrics[index] = 0.0
    rz_local = float(0.0)
    norm_local = float(0.0)
    nonfinite_local = float(0.0)
    for row in range(lane, count, stride):
        solution[row] = wp.vec3(0.0)
        if not invalid:
            block = blocks[slots[row]]
            block = 0.5 * (block + wp.transpose(block))
            inv = wp.inverse(block + 1e-9 * wp.identity(n=3, dtype=float))
            r = rhs[row]
            z = inv * r
            residual[row] = r
            preconditioned[row] = z
            direction[row] = z
            inverse[row] = inv
            rz_local += wp.dot(r, z)
            norm_local += wp.dot(r, r)
            if not (wp.isfinite(r[0]) and wp.isfinite(r[1]) and wp.isfinite(r[2])):
                nonfinite_local += 1.0
    rz = wp.tile_sum(wp.tile(rz_local))[0]
    norm = wp.tile_sum(wp.tile(norm_local))[0]
    nonfinite = wp.tile_sum(wp.tile(nonfinite_local))[0]
    if lane == 0:
        flags = int(0)
        if invalid:
            flags = 32
        else:
            if norm > 1e-30 and rz <= 0.0:
                flags = flags | 16
            if nonfinite > 0.0 or not wp.isfinite(norm):
                flags = flags | 1
        status[0] = flags
        metrics[0] = norm
        metrics[4] = rz
        rz_state[0] = rz
        counters[0] = 0
        counters[1] = 0


@wp.kernel(enable_backward=False)
def _product(
    offsets: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    blocks: wp.array[wp.mat33],
    contacts: ContactProjectionData,
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    status: wp.array[wp.int32],
):
    row = wp.tid()
    if status[0] & 32:
        return
    value = wp.vec3(0.0)
    for slot in range(offsets[row], offsets[row + 1]):
        value += blocks[slot] * direction[columns[slot]]
    product[row] = value + off_diagonal_product(row, direction, contacts)


@wp.kernel(enable_backward=False)
def _update(
    count: int,
    iteration: int,
    solution: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
    preconditioned: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    inverse: wp.array[wp.mat33],
    status: wp.array[wp.int32],
    metrics: wp.array[float],
    rz_state: wp.array[float],
):
    if status[0] & 32:
        return
    lane = wp.tid()
    stride = wp.block_dim()
    rz = rz_state[0]
    failed = (status[0] & 16) != 0
    product_local = float(0.0)
    for row in range(lane, count, stride):
        product_local += wp.dot(direction[row], product[row])
    direction_product = wp.tile_sum(wp.tile(product_local))[0]
    if rz > 1e-20 and direction_product <= 0.0:
        failed = True
    alpha = float(0.0)
    if not failed and direction_product > 1e-20:
        alpha = rz / direction_product
    new_rz_local = float(0.0)
    for row in range(lane, count, stride):
        solution[row] += alpha * direction[row]
        r = residual[row] - alpha * product[row]
        z = inverse[row] * r
        residual[row] = r
        preconditioned[row] = z
        new_rz_local += wp.dot(r, z)
    new_rz = wp.tile_sum(wp.tile(new_rz_local))[0]
    if new_rz < 0.0:
        failed = True
    beta = float(0.0)
    if rz > 1e-20:
        beta = new_rz / rz
    norm_local = float(0.0)
    for row in range(lane, count, stride):
        direction[row] = preconditioned[row] + beta * direction[row]
        norm_local += wp.dot(direction[row], direction[row])
    norm = wp.tile_sum(wp.tile(norm_local))[0]
    if lane == 0:
        metrics[5 + iteration] = new_rz
        rz_state[0] = new_rz
        if norm <= 1e-30:
            rz_state[0] = 0.0
        if failed:
            status[0] = status[0] | 16


@wp.kernel(enable_backward=False)
def _finish(
    count: int,
    validate: bool,
    minimum_reduction: float,
    solution: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
    status: wp.array[wp.int32],
    metrics: wp.array[float],
):
    if status[0] & 32:
        return
    lane = wp.tid()
    norm_local = float(0.0)
    nonfinite_local = float(0.0)
    for row in range(lane, count, wp.block_dim()):
        r = residual[row]
        x = solution[row]
        norm_local += wp.dot(r, r)
        if not (
            wp.isfinite(r[0])
            and wp.isfinite(r[1])
            and wp.isfinite(r[2])
            and wp.isfinite(x[0])
            and wp.isfinite(x[1])
            and wp.isfinite(x[2])
        ):
            nonfinite_local += 1.0
    norm = wp.tile_sum(wp.tile(norm_local))[0]
    nonfinite = wp.tile_sum(wp.tile(nonfinite_local))[0]
    if lane == 0:
        flags = status[0]
        if nonfinite > 0.0 or not wp.isfinite(norm):
            flags = flags | 2
        initial = metrics[0]
        if validate and initial > 1e-30 and wp.isfinite(norm):
            if 1.0 - norm / initial < minimum_reduction:
                flags = flags | 4
        status[0] = flags
        metrics[1] = norm


class SplitCoarsePCG:
    """Reuse fixed work storage across captures without host convergence reads."""

    def __init__(self, device):
        self.device = device
        self.rz = wp.zeros(1, dtype=float, device=device)

    def solve(self, inputs, outputs):
        if wp.get_device(self.device).is_cpu:
            # Split scheduling is CUDA-only; preserve the established
            # synchronous recurrence on the CPU tile backend.
            from .particle_multilevel import _solve_energy_galerkin_pcg_persistent  # noqa: PLC0415

            wp.launch(
                _solve_energy_galerkin_pcg_persistent,
                dim=256,
                block_dim=256,
                inputs=inputs,
                outputs=outputs,
                device=self.device,
            )
            return
        count, offsets, columns, slots, blocks, rhs, iterations, validate, minimum, contacts = inputs
        solution, residual, preconditioned, direction, product, inverse, status, metrics, counters = outputs
        wp.launch(
            _initialize,
            dim=256,
            block_dim=256,
            inputs=[
                count,
                slots,
                blocks,
                rhs,
                contacts,
                solution,
                residual,
                preconditioned,
                direction,
                inverse,
                status,
                metrics,
                counters,
                self.rz,
            ],
            device=self.device,
        )
        for iteration in range(iterations):
            wp.launch(
                _product,
                dim=count,
                block_dim=64,
                inputs=[offsets, columns, blocks, contacts, direction, product, status],
                device=self.device,
            )
            wp.launch(
                _update,
                dim=256,
                block_dim=256,
                inputs=[
                    count,
                    iteration,
                    solution,
                    residual,
                    preconditioned,
                    direction,
                    product,
                    inverse,
                    status,
                    metrics,
                    self.rz,
                ],
                device=self.device,
            )
        wp.launch(
            _finish,
            dim=256,
            block_dim=256,
            inputs=[count, validate, minimum, solution, residual, status, metrics],
            device=self.device,
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Additive Ritz preconditioning of the full particle/free-body translation system."""

from functools import cache

import warp as wp

from .. import coarse_pcg_split as split

PCG_BLOCK_DIM = 256


@cache
def _build_kernels(width: int):
    # Closure-specialized kernels prevent one solver's basis size from changing
    # another solver's captured graph or scratch-buffer layout.
    WIDTH = wp.constant(width)
    Vector = wp.types.vector(length=width, dtype=wp.float32)

    @wp.kernel(enable_backward=False, module="unique")
    def factor_preconditioner(matrix: wp.array2d[float], factor: wp.array2d[float]):
        wp.tile_store(factor, wp.tile_cholesky(wp.tile_load(matrix, shape=(WIDTH, WIDTH))))

    @wp.kernel(enable_backward=False, module="unique")
    def update_precondition_direction(
        count: int,
        iteration: int,
        q: wp.array[wp.vec3],
        centers: wp.array[wp.vec3],
        groups: wp.array[int],
        matrix: wp.array2d[float],
        solution: wp.array[wp.vec3],
        residual: wp.array[wp.vec3],
        preconditioned: wp.array[wp.vec3],
        direction: wp.array[wp.vec3],
        product: wp.array[wp.vec3],
        inverse: wp.array[wp.mat33],
        status: wp.array[int],
        metrics: wp.array[float],
        rz: wp.array[float],
    ):
        if status[0] != 0:
            return
        lane = wp.tid()
        previous = rz[0]
        dot_local = float(0.0)
        for row in range(lane, count, wp.block_dim()):
            if iteration >= 0:
                dot_local += wp.dot(direction[row], product[row])
        curvature = wp.tile_sum(wp.tile(dot_local))[0]
        alpha = float(0.0)
        if iteration >= 0 and previous > 1e-20:
            if curvature <= 0.0:
                if lane == 0:
                    status[0] = 16
                return
            alpha = previous / curvature
        local = Vector(0.0)
        for row in range(lane, count, wp.block_dim()):
            if iteration >= 0:
                solution[row] += alpha * direction[row]
                residual[row] -= alpha * product[row]
            group = groups[row]
            if group >= 0:
                r = residual[row]
                moment = wp.cross((q[row] - centers[group]) / 0.05, r)
                for axis in range(3):
                    local[6 * group + axis] += r[axis]
                    local[6 * group + 3 + axis] += moment[axis]
        total = wp.tile_reduce(wp.add, wp.tile(local, preserve_type=True))[0]
        value = float(0.0)
        if lane < WIDTH:
            value = total[lane]
        # A shared tile carries the complete coarse RHS across the CTA, without
        # a global-memory round trip or a second kernel launch.
        rhs_tile = wp.tile_view(wp.tile(value), offset=(0,), shape=(WIDTH,))
        factor = wp.tile_load(matrix, shape=(WIDTH, WIDTH))
        coarse = wp.tile_cholesky_solve(factor, rhs_tile)
        local_rz = float(0.0)
        for row in range(lane, count, wp.block_dim()):
            group = groups[row]
            z = inverse[row] * residual[row]
            if group >= 0:
                translation = wp.vec3(coarse[6 * group], coarse[6 * group + 1], coarse[6 * group + 2])
                rotation = wp.vec3(coarse[6 * group + 3], coarse[6 * group + 4], coarse[6 * group + 5])
                z += translation + wp.cross(rotation, (q[row] - centers[group]) / 0.05)
            preconditioned[row] = z
            local_rz += wp.dot(residual[row], z)
        new_rz = wp.tile_sum(wp.tile(local_rz))[0]
        beta = float(0.0)
        if iteration >= 0 and previous > 1e-20:
            beta = new_rz / previous
        for row in range(lane, count, wp.block_dim()):
            direction[row] = preconditioned[row] + beta * direction[row]
        if lane == 0:
            rz[0] = new_rz
            metrics[5 + iteration] = new_rz
            if not wp.isfinite(new_rz) or new_rz < 0.0:
                status[0] = 16

    return factor_preconditioner, update_precondition_direction


class TwoLevelPCG:
    def __init__(self, ritz):
        self.ritz = ritz
        self.device = ritz.model.device
        self.width = ritz.dofs
        self.factor_kernel, self.update_kernel = _build_kernels(self.width)
        self.rz = wp.zeros(1, device=self.device)
        self.factor = wp.empty((self.width, self.width), dtype=float, device=self.device)

    def solve(self, inputs, outputs):
        count, offsets, columns, slots, blocks, rhs, iterations, validate, minimum, contacts = inputs
        solution, residual, preconditioned, direction, product, inverse, status, metrics, counters = outputs
        ritz = self.ritz
        ritz.project(wp.launch, inputs)
        wp.launch(
            self.factor_kernel,
            dim=PCG_BLOCK_DIM,
            block_dim=PCG_BLOCK_DIM,
            inputs=[ritz.symmetric, self.factor],
            device=self.device,
        )
        wp.launch(
            split._initialize,
            dim=PCG_BLOCK_DIM,
            block_dim=PCG_BLOCK_DIM,
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
        for iteration in range(-1, iterations):
            if iteration >= 0:
                wp.launch(
                    split._product,
                    dim=count,
                    block_dim=64,
                    inputs=[offsets, columns, blocks, contacts, direction, product, status],
                    device=self.device,
                )
            wp.launch(
                self.update_kernel,
                dim=PCG_BLOCK_DIM,
                block_dim=PCG_BLOCK_DIM,
                inputs=[
                    count,
                    iteration,
                    ritz.q,
                    ritz.centers,
                    ritz.groups,
                    self.factor,
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
            split._finish,
            dim=PCG_BLOCK_DIM,
            block_dim=PCG_BLOCK_DIM,
            inputs=[count, validate, minimum, solution, residual, status, metrics],
            device=self.device,
        )

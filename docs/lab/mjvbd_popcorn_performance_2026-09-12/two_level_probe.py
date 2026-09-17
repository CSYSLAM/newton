# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Use an additive rigid Ritz correction as a full-space PCG preconditioner."""

import warp as wp

from newton._src.solvers.mjvbd_v2 import coarse_pcg_split as split

Vec32 = wp.types.vector(length=32, dtype=wp.float32)
RITZ_WIDTH = wp.constant(32)
PCG_BLOCK_DIM = 256


@wp.kernel(enable_backward=False)
def factor_preconditioner(matrix: wp.array2d[float], factor: wp.array2d[float]):
    wp.tile_store(factor, wp.tile_cholesky(wp.tile_load(matrix, shape=(RITZ_WIDTH, RITZ_WIDTH))))


@wp.kernel(enable_backward=False)
def update_and_restrict(
    count: int,
    initial: bool,
    q: wp.array[wp.vec3],
    centers: wp.array[wp.vec3],
    groups: wp.array[int],
    solution: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    status: wp.array[int],
    rz: wp.array[float],
    coarse_rhs: wp.array[float],
):
    if status[0] != 0:
        return
    lane = wp.tid()
    dot_local = float(0.0)
    for row in range(lane, count, wp.block_dim()):
        if not initial:
            dot_local += wp.dot(direction[row], product[row])
    curvature = wp.tile_sum(wp.tile(dot_local))[0]
    alpha = float(0.0)
    if not initial and rz[0] > 1e-20:
        if curvature <= 0.0:
            if lane == 0:
                status[0] = 16
            return
        alpha = rz[0] / curvature
    local = Vec32(0.0)
    for row in range(lane, count, wp.block_dim()):
        if not initial:
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
    if lane < RITZ_WIDTH:
        coarse_rhs[lane] = total[lane]


@wp.kernel(enable_backward=False)
def precondition_and_direction(
    count: int,
    iteration: int,
    q: wp.array[wp.vec3],
    centers: wp.array[wp.vec3],
    groups: wp.array[int],
    matrix: wp.array2d[float],
    coarse_rhs: wp.array[float],
    residual: wp.array[wp.vec3],
    preconditioned: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    inverse: wp.array[wp.mat33],
    status: wp.array[int],
    metrics: wp.array[float],
    rz: wp.array[float],
):
    if status[0] != 0:
        return
    factor = wp.tile_load(matrix, shape=(RITZ_WIDTH, RITZ_WIDTH))
    coarse = wp.tile_cholesky_solve(factor, wp.tile_load(coarse_rhs, shape=(RITZ_WIDTH,)))
    lane = wp.tid()
    local = float(0.0)
    previous = rz[0]
    for row in range(lane, count, wp.block_dim()):
        group = groups[row]
        z = inverse[row] * residual[row]
        if group >= 0:
            translation = wp.vec3(coarse[6 * group], coarse[6 * group + 1], coarse[6 * group + 2])
            rotation = wp.vec3(coarse[6 * group + 3], coarse[6 * group + 4], coarse[6 * group + 5])
            z += translation + wp.cross(rotation, (q[row] - centers[group]) / 0.05)
        preconditioned[row] = z
        local += wp.dot(residual[row], z)
    new_rz = wp.tile_sum(wp.tile(local))[0]
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


@wp.kernel(enable_backward=False)
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
    local = Vec32(0.0)
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
    if lane < RITZ_WIDTH:
        value = total[lane]
    # A shared tile carries the complete coarse RHS across the CTA, without
    # a global-memory round trip or a second kernel launch.
    rhs_tile = wp.tile_view(wp.tile(value), offset=(0,), shape=(RITZ_WIDTH,))
    factor = wp.tile_load(matrix, shape=(RITZ_WIDTH, RITZ_WIDTH))
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


class TwoLevelPCG:
    def __init__(self, ritz):
        if ritz.dofs > RITZ_WIDTH:
            raise ValueError("Ritz factor width must cover every coarse basis column")
        self.ritz = ritz
        self.device = ritz.model.device
        self.rz = wp.zeros(1, device=self.device)
        self.factor = wp.empty((RITZ_WIDTH, RITZ_WIDTH), dtype=float, device=self.device)

    def solve(self, inputs, outputs):
        count, offsets, columns, slots, blocks, rhs, iterations, validate, minimum, contacts = inputs
        solution, residual, preconditioned, direction, product, inverse, status, metrics, counters = outputs
        ritz = self.ritz
        ritz.project(wp.launch, inputs)
        wp.launch(
            factor_preconditioner,
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
                update_precondition_direction,
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

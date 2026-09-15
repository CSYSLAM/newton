# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solve particle-connected and disconnected free-body PCG blocks independently."""

from functools import cache

import warp as wp

from .. import coarse_pcg_split as split
from ..contact_projection import ContactProjectionData

PCG_BLOCK_DIM = 256


@wp.kernel(enable_backward=False)
def _initialize_components(particle_count: int, parent: wp.array[int]):
    row = wp.tid()
    # The Ritz basis may couple particle clusters. Merge them conservatively.
    parent[row] = 0 if row < particle_count else row


@wp.func
def _root(row: int, parent: wp.array[int]):
    result = row
    while parent[result] != result:
        result = parent[result]
    return result


@wp.kernel(enable_backward=False)
def _join_contacts(data: ContactProjectionData, parent: wp.array[int]):
    slot = wp.tid()
    if data.edge_keys[slot] == wp.int64(-1):
        return
    edge = data.edge_clusters[slot]
    a, b = edge[0], edge[1]
    while True:
        ra, rb = _root(a, parent), _root(b, parent)
        if ra == rb:
            break
        lower, upper = wp.min(ra, rb), wp.max(ra, rb)
        # Links strictly decrease: concurrent unions cannot create a cycle.
        previous = wp.atomic_cas(parent, upper, upper, lower)
        if previous == upper:
            break


@wp.kernel(enable_backward=False)
def _classify_components(parent: wp.array[int], components: wp.array[int]):
    row = wp.tid()
    components[row] = int(_root(row, parent) != 0)


@cache
def _build_kernels(width: int, inverse_factor: bool = True):
    # Closure-specialized kernels prevent one solver's basis size from changing
    # another solver's captured graph or scratch-buffer layout.
    WIDTH = wp.constant(width)
    Vector = wp.types.vector(length=width, dtype=wp.float32)

    @wp.func
    def identity_value(index: int):
        return float(index // WIDTH == index % WIDTH)

    @wp.kernel(enable_backward=False, module="unique")
    def factor_preconditioner(matrix: wp.array2d[float], factor: wp.array2d[float]):
        factor_tile = wp.tile_cholesky(wp.tile_load(matrix, shape=(WIDTH, WIDTH)))
        if wp.static(inverse_factor):
            indices = wp.tile_arange(0, wp.static(WIDTH * WIDTH), dtype=int)
            identity = wp.tile_reshape(wp.tile_map(identity_value, indices), shape=(WIDTH, WIDTH))
            wp.tile_store(factor, wp.tile_lower_solve(factor_tile, identity))
        else:
            wp.tile_store(factor, factor_tile)

    @wp.func
    def _update_component(
        thread_index: int,
        count: int,
        iteration: int,
        component_ids: wp.array[int],
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
        failed = wp.tile_sum(wp.tile(int(status[0] != 0)))[0]
        if failed != 0:
            return
        lane = thread_index % PCG_BLOCK_DIM
        component = thread_index // PCG_BLOCK_DIM
        previous = rz[component]
        dot_local = float(0.0)
        for row in range(lane, count, PCG_BLOCK_DIM):
            if component_ids[row] != component:
                continue
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
        if component != 0:
            # Detached bodies have no particle Ritz basis. Their preconditioner
            # is exactly block Jacobi; avoid the zero coarse RHS and solve.
            local_rz = float(0.0)
            for row in range(lane, count, PCG_BLOCK_DIM):
                if component_ids[row] != component:
                    continue
                if iteration >= 0:
                    solution[row] += alpha * direction[row]
                    residual[row] -= alpha * product[row]
                z = inverse[row] * residual[row]
                preconditioned[row] = z
                local_rz += wp.dot(residual[row], z)
            new_rz = wp.tile_sum(wp.tile(local_rz))[0]
            beta = float(0.0)
            if iteration >= 0 and previous > 1e-20:
                beta = new_rz / previous
            for row in range(lane, count, PCG_BLOCK_DIM):
                if component_ids[row] == component:
                    direction[row] = preconditioned[row] + beta * direction[row]
            if lane == 0:
                rz[component] = new_rz
                wp.atomic_add(metrics, 5 + iteration, new_rz)
                if not wp.isfinite(new_rz) or new_rz < 0.0:
                    status[0] = 16
            return
        local = Vector(0.0)
        for row in range(lane, count, PCG_BLOCK_DIM):
            if component_ids[row] != component:
                continue
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
        if wp.static(inverse_factor):
            transformed_value = float(0.0)
            if lane < WIDTH:
                for column in range(WIDTH):
                    transformed_value += matrix[lane, column] * total[column]
            transformed = wp.tile_view(wp.tile(transformed_value), offset=(0,), shape=(WIDTH,))
            coarse_value = float(0.0)
            if lane < WIDTH:
                for column in range(WIDTH):
                    coarse_value += matrix[column, lane] * transformed[column]
            coarse = wp.tile_view(wp.tile(coarse_value), offset=(0,), shape=(WIDTH,))
        else:
            rhs_tile = wp.tile_view(wp.tile(value), offset=(0,), shape=(WIDTH,))
            factor = wp.tile_load(matrix, shape=(WIDTH, WIDTH))
            coarse = wp.tile_cholesky_solve(factor, rhs_tile)
        local_rz = float(0.0)
        for row in range(lane, count, PCG_BLOCK_DIM):
            if component_ids[row] != component:
                continue
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
        for row in range(lane, count, PCG_BLOCK_DIM):
            if component_ids[row] != component:
                continue
            direction[row] = preconditioned[row] + beta * direction[row]
        if lane == 0:
            rz[component] = new_rz
            wp.atomic_add(metrics, 5 + iteration, new_rz)
            if not wp.isfinite(new_rz) or new_rz < 0.0:
                status[0] = 16

    @wp.kernel(enable_backward=False, module="unique")
    def update_precondition_direction(
        count: int,
        iteration: int,
        component_ids: wp.array[int],
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

        _update_component(
            wp.tid(),
            count,
            iteration,
            component_ids,
            q,
            centers,
            groups,
            matrix,
            solution,
            residual,
            preconditioned,
            direction,
            product,
            inverse,
            status,
            metrics,
            rz,
        )

    return factor_preconditioner, update_precondition_direction


class ComponentPCG:
    """Keep the Ritz-connected particle domain separate from detached rigid islands."""

    def __init__(self, ritz, count: int, particle_count: int):
        self.ritz = ritz
        self.device = ritz.model.device
        self.width = ritz.dofs
        self.factor_kernel, self.update_kernel = _build_kernels(self.width, True)
        self.rz = wp.zeros(2, device=self.device)
        self.parent = wp.empty(count, dtype=int, device=self.device)
        self.component_ids = wp.empty(count, dtype=int, device=self.device)
        self.particle_count = particle_count
        self.factor = wp.empty((self.width, self.width), dtype=float, device=self.device)

    def solve(self, inputs, outputs):
        count, offsets, columns, slots, blocks, rhs, iterations, validate, minimum, contacts = inputs
        solution, residual, preconditioned, direction, product, inverse, status, metrics, counters = outputs
        ritz = self.ritz
        wp.launch(_initialize_components, count, [self.particle_count, self.parent], device=self.device)
        wp.launch(_join_contacts, contacts.edge_keys.size, [contacts, self.parent], device=self.device)
        wp.launch(_classify_components, count, [self.parent, self.component_ids], device=self.device)
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
        # The initial block-Jacobi metric is replaced by the Ritz metric.
        metrics[4:5].zero_()
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
                dim=2 * PCG_BLOCK_DIM,
                block_dim=PCG_BLOCK_DIM,
                inputs=[
                    count,
                    iteration,
                    self.component_ids,
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

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Project the assembled surface operator, rather than repurposing tet stiffness."""

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2 import particle_multilevel as ml
from newton._src.solvers.mjvbd_v2.contact_projection import ContactProjectionData
from newton._src.solvers.mjvbd_v2.particle_multilevel import _build_clusters


@wp.func
def basis(particle: int, column: int, groups: wp.array[int], q: wp.array[wp.vec3], centers: wp.array[wp.vec3]):
    group = groups[particle]
    result = wp.vec3(0.0)
    if group == column // 6:
        result = ml._cluster_basis_column(column % 6, (q[particle] - centers[group]) / 0.05, True)
    return result


@wp.kernel(enable_backward=False)
def project_operator(
    offsets: wp.array[int],
    members: wp.array[int],
    groups: wp.array[int],
    centers: wp.array[wp.vec3],
    q: wp.array[wp.vec3],
    matrix_offsets: wp.array[int],
    matrix_columns: wp.array[int],
    blocks: wp.array[wp.mat33],
    forces: wp.array[wp.vec3],
    contacts: ContactProjectionData,
    dofs: int,
    matrix: wp.array2d[float],
    rhs: wp.array[float],
):
    pair = wp.tid() // 64
    group = pair // dofs
    column = pair % dofs
    lane = wp.tid() % 64
    if column == 0:
        force_sum = ml._vec6f(0.0)
        for slot in range(offsets[group] + lane, offsets[group + 1], 64):
            particle = members[slot]
            force = forces[particle]
            moment = wp.cross((q[particle] - centers[group]) / 0.05, force)
            for axis in range(3):
                force_sum[axis] += force[axis]
                force_sum[3 + axis] += moment[axis]
        force_total = wp.tile_reduce(wp.add, wp.tile(force_sum, preserve_type=True))[0]
        if lane == 0:
            for row in range(6):
                rhs[group * 6 + row] = force_total[row]

    accumulated = ml._vec6f(0.0)
    for slot in range(offsets[group] + lane, offsets[group + 1], 64):
        particle = members[slot]
        product = wp.vec3(0.0)
        for entry in range(matrix_offsets[particle], matrix_offsets[particle + 1]):
            neighbor = matrix_columns[entry]
            product += blocks[entry] * basis(neighbor, column, groups, q, centers)
        if contacts.row_heads:
            entry = contacts.row_heads[particle]
            while entry >= 0:
                edge = entry // 2
                neighbor = contacts.edge_clusters[edge][1 - entry % 2]
                product += contacts.edge_blocks[edge] * basis(neighbor, column, groups, q, centers)
                entry = contacts.row_next[entry]
        moment = wp.cross((q[particle] - centers[group]) / 0.05, product)
        for axis in range(3):
            accumulated[axis] += product[axis]
            accumulated[3 + axis] += moment[axis]
    total = wp.tile_reduce(wp.add, wp.tile(accumulated, preserve_type=True))[0]
    if lane == 0:
        for row in range(6):
            matrix[group * 6 + row, column] = total[row]


@wp.kernel(enable_backward=False)
def symmetrize(matrix: wp.array2d[float], dofs: int, symmetric: wp.array2d[float]):
    row, col = wp.tid()
    value = float(0.0)
    if row < dofs and col < dofs:
        value = 0.5 * (matrix[row, col] + matrix[col, row])
    elif row == col:
        value = 1.0
    symmetric[row, col] = value


@wp.kernel(enable_backward=False)
def solve_dense(
    matrix: wp.array2d[float],
    rhs: wp.array[float],
    solution: wp.array[float],
    status: wp.array[int],
    counters: wp.array[int],
    metrics: wp.array[float],
):
    a = wp.tile_load(matrix, shape=(32, 32))
    b = wp.tile_load(rhs, shape=(32,))
    factor = wp.tile_cholesky(a)
    x = wp.tile_cholesky_solve(factor, b)
    wp.tile_store(solution, x)
    if wp.tid() == 0:
        status[0] = 0
        counters[0] = 0
        counters[1] = 0
        metrics[0] = 0.0
        metrics[1] = 0.0


@wp.kernel(enable_backward=False)
def assess_solution(
    matrix: wp.array2d[float],
    rhs: wp.array[float],
    solution: wp.array[float],
    status: wp.array[int],
    metrics: wp.array[float],
):
    row = wp.tid()
    residual = -rhs[row]
    for column in range(32):
        residual += matrix[row, column] * solution[column]
    norm = wp.tile_reduce(wp.add, wp.tile(rhs[row] * rhs[row]))[0]
    error = wp.tile_reduce(wp.add, wp.tile(residual * residual))[0]
    if row == 0:
        metrics[0] = norm
        metrics[1] = error
        if not wp.isfinite(error + norm) or error > 1.0e-6 * wp.max(norm, 1.0e-30):
            status[0] = 8


@wp.kernel(enable_backward=False)
def prolong(
    groups: wp.array[int],
    centers: wp.array[wp.vec3],
    q: wp.array[wp.vec3],
    solution: wp.array[float],
    status: wp.array[int],
    result: wp.array[wp.vec3],
):
    particle = wp.tid()
    group = groups[particle]
    delta = wp.vec3(0.0)
    for column in range(6):
        delta += basis(particle, 6 * group + column, groups, q, centers) * solution[6 * group + column]
    if not (wp.isfinite(delta[0]) and wp.isfinite(delta[1]) and wp.isfinite(delta[2])):
        wp.atomic_or(status, 0, 1)
        delta = wp.vec3(0.0)
    result[particle] = delta


class RigidRitz:
    def __init__(self, model, correction, cluster_size=400):
        if not np.array_equal(correction.fine_to_coarse.numpy(), np.arange(model.particle_count)):
            raise ValueError("Ritz probe requires singleton, unpinned fine rows")
        grouping = _build_clusters(model, cluster_size)
        self.group_count = len(grouping[1]) - 1
        self.dofs = 6 * self.group_count
        if self.dofs > 32:
            raise ValueError("Ritz probe supports at most five rigid clusters")
        self.groups = wp.array(grouping[0], dtype=int, device=model.device)
        self.offsets = wp.array(grouping[1], dtype=int, device=model.device)
        self.members = wp.array(grouping[2], dtype=int, device=model.device)
        self.centers = wp.zeros(self.group_count, dtype=wp.vec3, device=model.device)
        self.matrix = wp.zeros((32, 32), dtype=float, device=model.device)
        self.symmetric = wp.empty((32, 32), dtype=float, device=model.device)
        self.rhs = wp.zeros(32, dtype=float, device=model.device)
        self.solution = wp.zeros(32, dtype=float, device=model.device)
        self.model = model
        self.q = None
        correction.coarse_use_split_pcg = False

    def project(self, launch, inputs):
        launch(
            ml._compute_cluster_centroids,
            dim=self.group_count,
            inputs=[self.offsets, self.members, self.q, self.model.particle_mass],
            outputs=[self.centers],
            device=self.model.device,
        )
        launch(
            project_operator,
            dim=self.group_count * self.dofs * 64,
            block_dim=64,
            inputs=[
                self.offsets,
                self.members,
                self.groups,
                self.centers,
                self.q,
                inputs[1],
                inputs[2],
                inputs[4],
                inputs[5],
                inputs[9],
                self.dofs,
            ],
            outputs=[self.matrix, self.rhs],
            device=self.model.device,
        )
        launch(
            symmetrize,
            dim=(32, 32),
            inputs=[self.matrix, self.dofs],
            outputs=[self.symmetric],
            device=self.model.device,
        )

    def solve(self, launch, inputs, outputs):
        self.project(launch, inputs)
        launch(
            solve_dense,
            dim=64,
            block_dim=64,
            inputs=[self.symmetric, self.rhs],
            outputs=[self.solution, outputs[6], outputs[8], outputs[7]],
            device=self.model.device,
        )
        launch(
            assess_solution,
            dim=32,
            block_dim=32,
            inputs=[self.symmetric, self.rhs, self.solution],
            outputs=[outputs[6], outputs[7]],
            device=self.model.device,
        )
        launch(
            prolong,
            dim=self.model.particle_count,
            inputs=[self.groups, self.centers, self.q, self.solution],
            outputs=[outputs[6], outputs[0]],
            device=self.model.device,
        )

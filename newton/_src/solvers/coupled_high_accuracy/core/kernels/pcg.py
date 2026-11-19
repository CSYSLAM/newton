# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Preconditioned conjugate gradient kernels (native ``kernels/pcg.cu`` and ``revisions/pcg_23fb797.cu``).

CG coefficients are accumulated separately per coefficient group (world). The reducing kernels are launched tiled and
sum a block's rows before one atomic when they share a group (native kernels do the same per warp with shuffles);
this only changes the floating-point summation order, which the native code already leaves to unordered atomics.
"""

import warp as wp

from .common import REDUCE_BLOCK, block_world_add, coefficient_world, guarded_ratio

# The solver never differentiates through its kernels; skipping the adjoints halves compile time.
wp.set_module_options({"enable_backward": False})


@wp.kernel
def prepare_inverse_preconditioner(
    diagonal: wp.array[float],
    mass: wp.array[float],
    flags: wp.array[wp.int32],
    inverse: wp.array[float],
    particles_per_world: int,
):
    """1 / d for active particles with d > 0, else 0."""
    i = wp.tid()
    particle = i % particles_per_world
    d = diagonal[i]
    active = mass[particle] > 0.0
    if flags.shape[0] > 0:
        active = active and (flags[particle] & 1) != 0
    if active and d > 0.0:
        inverse[i] = 1.0 / d
    else:
        inverse[i] = 0.0


@wp.kernel
def invert_block_preconditioner(
    blocks: wp.array[wp.mat33],
    mass: wp.array[float],
    flags: wp.array[wp.int32],
    inverse: wp.array[wp.mat33],
    particles_per_world: int,
):
    """Block-Jacobi preconditioner: inverse of each symmetrised diagonal block for active particles, else zero.

    When the determinant is not a finite value > 1e-30, the reciprocals of the positive diagonal entries are used.
    """
    i = wp.tid()
    particle = i % particles_per_world
    active = mass[particle] > 0.0
    if flags.shape[0] > 0:
        active = active and (flags[particle] & 1) != 0
    if not active:
        inverse[i] = wp.mat33(0.0)
        return
    m = blocks[i]
    s01 = (m[1, 0] + m[0, 1]) * 0.5
    s02 = (m[2, 0] + m[0, 2]) * 0.5
    s12 = (m[2, 1] + m[1, 2]) * 0.5
    s00 = m[0, 0]
    s11 = m[1, 1]
    s22 = m[2, 2]
    c00 = s11 * s22 - s12 * s12
    c01 = s02 * s12 - s01 * s22
    c02 = s01 * s12 - s11 * s02
    det = s00 * c00 + s01 * c01 + s02 * c02
    c11 = s00 * s22 - s02 * s02
    c12 = s01 * s02 - s00 * s12
    c22 = s00 * s11 - s01 * s01
    if det > 1.0e-30 and wp.isfinite(det):
        inverse_det = 1.0 / det
        v00 = c00 * inverse_det
        v01 = c01 * inverse_det
        v02 = c02 * inverse_det
        v11 = c11 * inverse_det
        v12 = c12 * inverse_det
        v22 = c22 * inverse_det
        inverse[i] = wp.mat33(v00, v01, v02, v01, v11, v12, v02, v12, v22)
    else:
        a00 = float(0.0)
        a11 = float(0.0)
        a22 = float(0.0)
        if s00 > 0.0:
            a00 = 1.0 / s00
        if s11 > 0.0:
            a11 = 1.0 / s11
        if s22 > 0.0:
            a22 = 1.0 / s22
        inverse[i] = wp.mat33(a00, 0.0, 0.0, 0.0, a11, 0.0, 0.0, 0.0, a22)


@wp.kernel
def initialize_block_non_diagonal(
    entry_columns: wp.array[wp.int32],
    entry_values: wp.array[float],
    row_count: wp.array[wp.int32],
    rows_per_world: int,
    rows: int,
    block_columns: wp.array[wp.int32],
    block_values: wp.array[wp.mat33],
):
    """Block-ELL copy of one world's scalar ELL matrix for all worlds, columns shifted by each world's first row.

    Slots past a row's count stay unset.
    """
    i = wp.tid()
    slot = i // rows
    row = i % rows
    local_row = row % rows_per_world
    if slot >= row_count[local_row]:
        return
    world_offset = row - local_row
    entry = slot * rows_per_world + local_row
    block_columns[i] = entry_columns[entry] + world_offset
    block_values[i] = entry_values[entry] * wp.identity(n=3, dtype=float)


@wp.func
def ell_product_scalar(
    row_count: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[float],
    max_count: int,
    v: wp.array[wp.vec3],
    i: int,
    count: int,
    per_world: int,
):
    """Off-diagonal scalar ELL product of row i with in-world columns; out-of-world columns are skipped."""
    total = wp.vec3(0.0)
    if max_count == 0:
        return total
    local = i
    if per_world != 0:
        local = i % per_world
    world_base = i - local
    slots = wp.min(wp.max(row_count[local], 0), max_count)
    for k in range(slots):
        column = columns[k * per_world + local]
        if column >= 0 and column < per_world:
            j = world_base + column
            if j < count:
                total = total + values[k * per_world + local] * v[j]
    return total


@wp.func
def ell_product_block(
    row_count: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.mat33],
    max_count: int,
    v: wp.array[wp.vec3],
    i: int,
    count: int,
    per_world: int,
):
    """Off-diagonal block-ELL product of row i; see ell_product_scalar."""
    total = wp.vec3(0.0)
    if max_count == 0:
        return total
    local = i
    if per_world != 0:
        local = i % per_world
    world_base = i - local
    slots = wp.min(wp.max(row_count[local], 0), max_count)
    for k in range(slots):
        column = columns[k * per_world + local]
        if column >= 0 and column < per_world:
            j = world_base + column
            if j < count:
                total = total + values[k * per_world + local] * v[j]
    return total


@wp.kernel
def precondition_and_reduce_block(
    inverse: wp.array[wp.mat33],
    residual: wp.array[wp.vec3],
    z: wp.array[wp.vec3],
    precomputed: int,
    world: wp.array[wp.int32],
    particles_per_world: int,
    numerator: wp.array[float],
    count: int,
):
    """z = M r (or keep a precomputed z) and add z . r to the row's coefficient group numerator."""
    block, lane = wp.tid()
    i = block * REDUCE_BLOCK + lane
    valid = i < count
    w = int(0)
    dot = float(0.0)
    if valid:
        value = wp.vec3(0.0)
        if precomputed != 0:
            value = z[i]
        else:
            value = inverse[i] * residual[i]
        z[i] = value
        w = coefficient_world(i, world, particles_per_world)
        dot = wp.dot(value, residual[i])
    block_world_add(numerator, w, dot, valid, lane)


@wp.kernel
def precondition_and_reduce_scalar(
    inverse: wp.array[float],
    residual: wp.array[wp.vec3],
    z: wp.array[wp.vec3],
    precomputed: int,
    world: wp.array[wp.int32],
    particles_per_world: int,
    numerator: wp.array[float],
    count: int,
):
    """Scalar-preconditioner variant of precondition_and_reduce_block."""
    block, lane = wp.tid()
    i = block * REDUCE_BLOCK + lane
    valid = i < count
    w = int(0)
    dot = float(0.0)
    if valid:
        value = wp.vec3(0.0)
        if precomputed != 0:
            value = z[i]
        else:
            value = inverse[i] * residual[i]
        z[i] = value
        w = coefficient_world(i, world, particles_per_world)
        dot = wp.dot(value, residual[i])
    block_world_add(numerator, w, dot, valid, lane)


@wp.kernel
def update_cg_direction(
    iteration: int,
    z: wp.array[wp.vec3],
    world: wp.array[wp.int32],
    particles_per_world: int,
    numerator: wp.array[float],
    previous_numerator: wp.array[float],
    direction: wp.array[wp.vec3],
):
    """p = z + beta p with the row's group beta = num / num_previous; the first iteration takes p = z."""
    i = wp.tid()
    value = z[i]
    if iteration > 0:
        w = coefficient_world(i, world, particles_per_world)
        value = value + guarded_ratio(numerator[w], previous_numerator[w]) * direction[i]
    direction[i] = value


@wp.kernel
def fused_cg_direction(
    iteration: int,
    z: wp.array[wp.vec3],
    world: wp.array[wp.int32],
    particles_per_world: int,
    numerator: wp.array[float],
    previous_numerator: wp.array[float],
    direction: wp.array[wp.vec3],
    tangent: wp.array[wp.vec3],
):
    """update_cg_direction that also zeroes the matrix-free tangent accumulator for the coming product."""
    i = wp.tid()
    value = z[i]
    if iteration > 0:
        w = coefficient_world(i, world, particles_per_world)
        value = value + guarded_ratio(numerator[w], previous_numerator[w]) * direction[i]
    direction[i] = value
    if tangent.shape[0] > 0:
        tangent[i] = wp.vec3(0.0)


@wp.kernel
def update_solution_and_residual(
    world: wp.array[wp.int32],
    particles_per_world: int,
    numerator: wp.array[float],
    denominator: wp.array[float],
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    solution: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
):
    """One CG step with the row's group alpha = num / den: r -= alpha A p, x += alpha p."""
    i = wp.tid()
    w = coefficient_world(i, world, particles_per_world)
    alpha = guarded_ratio(numerator[w], denominator[w])
    residual[i] = residual[i] - alpha * product[i]
    solution[i] = solution[i] + alpha * direction[i]


@wp.kernel
def fused_update_and_precondition(
    world: wp.array[wp.int32],
    particles_per_world: int,
    numerator: wp.array[float],
    denominator: wp.array[float],
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    solution: wp.array[wp.vec3],
    residual: wp.array[wp.vec3],
    inverse: wp.array[wp.mat33],
    z: wp.array[wp.vec3],
    next_numerator: wp.array[float],
    count: int,
):
    """update_solution_and_residual of one iteration followed by the block preconditioning of the next."""
    block, lane = wp.tid()
    i = block * REDUCE_BLOCK + lane
    valid = i < count
    w = int(0)
    dot = float(0.0)
    if valid:
        w = coefficient_world(i, world, particles_per_world)
        alpha = guarded_ratio(numerator[w], denominator[w])
        r = residual[i] - alpha * product[i]
        residual[i] = r
        solution[i] = solution[i] + alpha * direction[i]
        value = inverse[i] * r
        z[i] = value
        dot = wp.dot(value, r)
    block_world_add(next_numerator, w, dot, valid, lane)


@wp.kernel
def multiply_block_matrix_and_reduce(
    row_count: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.mat33],
    max_count: int,
    diagonal: wp.array[wp.mat33],
    direction: wp.array[wp.vec3],
    extra: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    world: wp.array[wp.int32],
    particles_per_world: int,
    denominator: wp.array[float],
    count: int,
    rows_per_world: int,
):
    """A p = D p + ELL p (+ extra) for the block matrix, adding p . A p to the row's group denominator."""
    block, lane = wp.tid()
    i = block * REDUCE_BLOCK + lane
    valid = i < count
    w = int(0)
    dot = float(0.0)
    if valid:
        value = diagonal[i] * direction[i] + ell_product_block(
            row_count, columns, values, max_count, direction, i, count, rows_per_world
        )
        if extra.shape[0] > 0:
            value = value + extra[i]
        product[i] = value
        w = coefficient_world(i, world, particles_per_world)
        dot = wp.dot(value, direction[i])
    block_world_add(denominator, w, dot, valid, lane)


@wp.kernel
def multiply_scalar_matrix_and_reduce(
    row_count: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[float],
    max_count: int,
    diagonal: wp.array[float],
    direction: wp.array[wp.vec3],
    extra: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    world: wp.array[wp.int32],
    particles_per_world: int,
    denominator: wp.array[float],
    count: int,
    rows_per_world: int,
):
    """multiply_block_matrix_and_reduce with a scalar ELL matrix and a scalar diagonal."""
    block, lane = wp.tid()
    i = block * REDUCE_BLOCK + lane
    valid = i < count
    w = int(0)
    dot = float(0.0)
    if valid:
        value = diagonal[i] * direction[i] + ell_product_scalar(
            row_count, columns, values, max_count, direction, i, count, rows_per_world
        )
        if extra.shape[0] > 0:
            value = value + extra[i]
        product[i] = value
        w = coefficient_world(i, world, particles_per_world)
        dot = wp.dot(value, direction[i])
    block_world_add(denominator, w, dot, valid, lane)


@wp.kernel
def multiply_scalar_block_matrix_and_reduce(
    row_count: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[float],
    max_count: int,
    diagonal: wp.array[wp.mat33],
    direction: wp.array[wp.vec3],
    extra: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    world: wp.array[wp.int32],
    particles_per_world: int,
    denominator: wp.array[float],
    count: int,
    rows_per_world: int,
):
    """multiply_block_matrix_and_reduce with a scalar ELL matrix and 3x3 diagonal blocks."""
    block, lane = wp.tid()
    i = block * REDUCE_BLOCK + lane
    valid = i < count
    w = int(0)
    dot = float(0.0)
    if valid:
        value = diagonal[i] * direction[i] + ell_product_scalar(
            row_count, columns, values, max_count, direction, i, count, rows_per_world
        )
        if extra.shape[0] > 0:
            value = value + extra[i]
        product[i] = value
        w = coefficient_world(i, world, particles_per_world)
        dot = wp.dot(value, direction[i])
    block_world_add(denominator, w, dot, valid, lane)


@wp.kernel
def apply_affine_body_preconditioner(
    body_particles: wp.array[wp.vec4i],
    inverse: wp.array[float],
    valid: wp.array[wp.int32],
    residual: wp.array[wp.vec3],
    z: wp.array[wp.vec3],
    bodies_per_world: int,
    particles_per_world: int,
):
    """z rows of a body with a valid inverted 12x12 block (row-major) become M r; other rows keep their z."""
    i = wp.tid()
    if valid[i] == 0:
        return
    world = i // bodies_per_world
    body = i % bodies_per_world
    offset = world * particles_per_world
    p = body_particles[body]
    indices = wp.vec4i(p[0] + offset, p[1] + offset, p[2] + offset, p[3] + offset)
    base = 144 * i
    for row in range(12):
        total = float(0.0)
        for k in range(12):
            total += inverse[base + row * 12 + k] * residual[indices[k // 3]][k % 3]
        target = indices[row // 3]
        value = z[target]
        value[row % 3] = total
        z[target] = value

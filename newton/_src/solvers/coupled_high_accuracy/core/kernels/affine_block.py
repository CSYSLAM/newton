# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Twelve-DOF affine-body block preconditioner (native ``kernels/affine_block.cu``).

Each body's 12x12 block (its four state rows: diagonal blocks, off-diagonal ELL entries between its own rows, and the
SO(3) Gauss-Newton term and the material tangents of the body's cables) is inverted by a scaled Cholesky
factorisation. Inverses are stored row-major, 144 floats per body state.
"""

import warp as wp

from .cables import CableView

# The solver never differentiates through its kernels; skipping the adjoints halves compile time.
wp.set_module_options({"enable_backward": False})

mat1212 = wp.types.matrix(shape=(12, 12), dtype=wp.float32)
mat312 = wp.types.matrix(shape=(3, 12), dtype=wp.float32)
vec12 = wp.types.vector(length=12, dtype=wp.float32)


@wp.func
def row_active(mass: wp.array[float], flags: wp.array[wp.int32], local_particle: int):
    if flags.shape[0] > 0:
        return mass[local_particle] > 0.0 and (flags[local_particle] & 1) != 0
    return mass[local_particle] > 0.0


@wp.func
def invert_affine_body(matrix: mat1212, active: vec12, output: wp.array[float], base: int):
    """Scaled Cholesky inverse with explicit symmetrisation; inactive rows/columns are identity. Returns 0 on
    failure (partial writes preserved, as natively)."""
    scale = vec12()
    lower = mat1212()
    for i in range(12):
        value = float(1.0)
        if active[i] != 0.0:
            value = matrix[i, i]
        if not (value > 0.0) or not wp.isfinite(value):
            return 0
        scale[i] = 1.0 / wp.sqrt(value)
    for i in range(12):
        for j in range(i + 1):
            value = float(0.0)
            if i == j:
                value = 1.0
            if active[i] != 0.0 and active[j] != 0.0:
                value = (scale[i] * ((matrix[i, j] + matrix[j, i]) * 0.5)) * scale[j]
            for k in range(j):
                value = value - lower[i, k] * lower[j, k]
            if i == j:
                if not (value > 0.0) or not wp.isfinite(value):
                    return 0
                lower[i, j] = wp.sqrt(value)
            else:
                lower[i, j] = value / lower[j, j]
    for col in range(12):
        y = vec12()
        x = vec12()
        for i in range(12):
            value = float(0.0)
            if i == col:
                value = 1.0
            for k in range(i):
                value = value - lower[i, k] * y[k]
            y[i] = value / lower[i, i]
        for ii in range(12):
            i = 11 - ii
            value = y[i]
            for k in range(i + 1, 12):
                value = value - lower[k, i] * x[k]
            x[i] = value / lower[i, i]
        for i in range(12):
            value = float(0.0)
            if active[i] != 0.0 and active[col] != 0.0:
                value = (scale[i] * x[i]) * scale[col]
            if not wp.isfinite(value):
                return 0
            output[base + i * 12 + col] = value
    for i in range(12):
        for j in range(i):
            value = (output[base + i * 12 + j] + output[base + j * 12 + i]) * 0.5
            output[base + i * 12 + j] = value
            output[base + j * 12 + i] = value
    return 1


@wp.func
def assemble_rigid_part(
    body_indices: wp.array[wp.vec4i],
    weight: wp.array[float],
    mass: wp.array[float],
    flags: wp.array[wp.int32],
    diagonal: wp.array[wp.mat33],
    row_count: wp.array[wp.int32],
    max_count: int,
    tile: int,
    rotation: wp.array[wp.mat33],
    alpha: float,
    cache: wp.array[float],
    body: int,
    per_world: int,
    particles_per_world: int,
):
    """Diagonal blocks and the SO(3) term; the ELL entries are added by the caller (scalar or block entries)."""
    local = body % per_world
    world = body // per_world
    p = body_indices[local]
    offset = world * particles_per_world
    ids = wp.vec4i(p[0] + offset, p[1] + offset, p[2] + offset, p[3] + offset)
    matrix = mat1212()
    if cache.shape[0] > 0:
        for at in range(144):
            matrix[at // 12, at % 12] = cache[body * 144 + at]
    for i in range(4):
        particle = ids[i]
        block = diagonal[particle]
        for r in range(3):
            for c in range(3):
                matrix[3 * i + r, 3 * i + c] = block[r, c]
    return matrix, ids


@wp.func
def add_so3_term(matrix: mat1212, r: wp.mat33, factor: float):
    for j in range(3):
        for k in range(3):
            for a in range(3):
                for b in range(3):
                    value = r[a, k] * r[b, j]
                    if j == k:
                        if a == 2 and b == 2:
                            value = r[2, k] * r[2, j] + r[2, 0] * r[2, 0]
                            value = r[2, 1] * r[2, 1] + value
                            value = r[2, 2] * r[2, 2] + value
                        else:
                            for axis in range(3):
                                value = r[a, axis] * r[b, axis] + value
                    row = 3 + 3 * j + a
                    col = 3 + 3 * k + b
                    matrix[row, col] = factor * value + matrix[row, col]
    return matrix


@wp.func
def add_cable_terms(
    matrix: mat1212,
    ids: wp.vec4i,
    body: int,
    per_world: int,
    cables: CableView,
    cable_offsets: wp.array[wp.int64],
    cable_ids: wp.array[wp.int32],
    inv_dt: float,
):
    """Add J^T W J of every active cable touching this body, restricted to the body's own twelve coordinates."""
    local = body % per_world
    world = body // per_world
    begin = int(cable_offsets[local])
    end = int(cable_offsets[local + 1])
    for at in range(begin, end):
        cable = cable_ids[at] + world * cables.per_world
        local_cable = cable % cables.per_world
        if cables.active.shape[0] > 0:
            if cables.active[local_cable] == 0:
                continue
        linear = mat312()
        angular = mat312()
        for node in range(cables.count[cable]):
            particle = cables.indices[cable * 8 + node]
            for i in range(4):
                if particle == ids[i]:
                    jl = cables.linear[cable * 8 + node]
                    ja = cables.angular[cable * 8 + node]
                    for r in range(3):
                        for c in range(3):
                            linear[r, 3 * i + c] = jl[r, c] + linear[r, 3 * i + c]
                            angular[r, 3 * i + c] = ja[r, c] + angular[r, 3 * i + c]
        penalty = cables.base[local_cable]
        if cables.penalty.shape[0] > 0:
            penalty = cables.penalty[cable]
        damping = cables.damping[local_cable]
        stiffness = damping * inv_dt + penalty
        lin = wp.vec3(stiffness[1], stiffness[1], stiffness[0])
        ang = wp.vec3(stiffness[2], stiffness[2], stiffness[3])
        if cables.bias.shape[0] > 0:
            ang = ang + cables.bias[cable]
        for row in range(12):
            for col in range(12):
                total = matrix[row, col]
                for axis in range(3):
                    term = (
                        lin[axis] * linear[axis, row] * linear[axis, col]
                        + (ang[axis] * angular[axis, row]) * angular[axis, col]
                    )
                    total = total + term
                matrix[row, col] = total
    return matrix


@wp.func
def active_rows(ids: wp.vec4i, mass: wp.array[float], flags: wp.array[wp.int32], particles_per_world: int):
    active = vec12()
    for i in range(4):
        enabled = row_active(mass, flags, ids[i] % particles_per_world)
        for r in range(3):
            if enabled:
                active[3 * i + r] = 1.0
    return active


@wp.kernel
def build_affine_body_preconditioner_block(
    body_indices: wp.array[wp.vec4i],
    weight: wp.array[float],
    mass: wp.array[float],
    flags: wp.array[wp.int32],
    diagonal: wp.array[wp.mat33],
    row_count: wp.array[wp.int32],
    entry_columns: wp.array[wp.int32],
    entry_values: wp.array[wp.mat33],
    max_count: int,
    tile: int,
    rotation: wp.array[wp.mat33],
    alpha: float,
    cables: CableView,
    cable_offsets: wp.array[wp.int64],
    cable_ids: wp.array[wp.int32],
    inv_dt: float,
    enabled: wp.array[wp.int32],
    cache: wp.array[float],
    output: wp.array[float],
    valid: wp.array[wp.int32],
    per_world: int,
    particles_per_world: int,
):
    """Assemble and invert each enabled body's 12x12 block with block-ELL off-diagonal entries."""
    body = wp.tid()
    valid[body] = 0
    if enabled[body % per_world] == 0:
        return
    matrix, ids = assemble_rigid_part(
        body_indices,
        weight,
        mass,
        flags,
        diagonal,
        row_count,
        max_count,
        tile,
        rotation,
        alpha,
        cache,
        body,
        per_world,
        particles_per_world,
    )
    if max_count > 0:
        for i in range(4):
            particle = ids[i]
            row = particle % tile
            bound = wp.min(max_count, wp.max(row_count[row], 0))
            for slot in range(bound):
                column = entry_columns[slot * tile + row]
                if column < 0 or column >= tile:
                    continue
                target = particle - row + column
                for j in range(4):
                    if target == ids[j]:
                        value = entry_values[slot * tile + row]
                        for r in range(3):
                            for c in range(3):
                                matrix[3 * i + r, 3 * j + c] = matrix[3 * i + r, 3 * j + c] + value[r, c]
    if alpha > 0.0:
        matrix = add_so3_term(matrix, rotation[body], (weight[body % per_world] * alpha) * 0.5)
    matrix = add_cable_terms(matrix, ids, body, per_world, cables, cable_offsets, cable_ids, inv_dt)
    valid[body] = invert_affine_body(matrix, active_rows(ids, mass, flags, particles_per_world), output, body * 144)


@wp.kernel
def build_affine_body_preconditioner_scalar(
    body_indices: wp.array[wp.vec4i],
    weight: wp.array[float],
    mass: wp.array[float],
    flags: wp.array[wp.int32],
    diagonal: wp.array[wp.mat33],
    row_count: wp.array[wp.int32],
    entry_columns: wp.array[wp.int32],
    entry_values: wp.array[float],
    max_count: int,
    tile: int,
    rotation: wp.array[wp.mat33],
    alpha: float,
    cables: CableView,
    cable_offsets: wp.array[wp.int64],
    cable_ids: wp.array[wp.int32],
    inv_dt: float,
    enabled: wp.array[wp.int32],
    cache: wp.array[float],
    output: wp.array[float],
    valid: wp.array[wp.int32],
    per_world: int,
    particles_per_world: int,
):
    """build_affine_body_preconditioner_block with scalar ELL entries expanded to identity blocks (the zero
    off-diagonal additions are kept for their signed-zero effect)."""
    body = wp.tid()
    valid[body] = 0
    if enabled[body % per_world] == 0:
        return
    matrix, ids = assemble_rigid_part(
        body_indices,
        weight,
        mass,
        flags,
        diagonal,
        row_count,
        max_count,
        tile,
        rotation,
        alpha,
        cache,
        body,
        per_world,
        particles_per_world,
    )
    if max_count > 0:
        for i in range(4):
            particle = ids[i]
            row = particle % tile
            bound = wp.min(max_count, wp.max(row_count[row], 0))
            for slot in range(bound):
                column = entry_columns[slot * tile + row]
                if column < 0 or column >= tile:
                    continue
                target = particle - row + column
                for j in range(4):
                    if target == ids[j]:
                        value = entry_values[slot * tile + row]
                        for r in range(3):
                            for c in range(3):
                                add = float(0.0)
                                if r == c:
                                    add = value
                                matrix[3 * i + r, 3 * j + c] = matrix[3 * i + r, 3 * j + c] + add
    if alpha > 0.0:
        matrix = add_so3_term(matrix, rotation[body], (weight[body % per_world] * alpha) * 0.5)
    matrix = add_cable_terms(matrix, ids, body, per_world, cables, cable_offsets, cable_ids, inv_dt)
    valid[body] = invert_affine_body(matrix, active_rows(ids, mass, flags, particles_per_world), output, body * 144)


@wp.kernel
def apply_block_diagonal_preconditioner(blocks: wp.array[wp.mat33], r: wp.array[wp.vec3], z: wp.array[wp.vec3]):
    """z = M r with the 3x3 diagonal blocks of the preconditioner."""
    i = wp.tid()
    z[i] = blocks[i] * r[i]

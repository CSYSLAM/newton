# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Device helpers shared by the solver kernels (native ``common.cuh`` and ``types.cuh``)."""

import warp as wp

# The solver never differentiates through its kernels; skipping the adjoints halves compile time.
wp.set_module_options({"enable_backward": False})


@wp.func
def guarded_ratio(num: float, den: float):
    """num / den for CG coefficients, or 0 for a tiny (|den| <= 1e-30) denominator or a NaN operand."""
    if wp.abs(den) > 1.0e-30 and not wp.isnan(den) and not wp.isnan(num):
        return num / den
    return 0.0


@wp.func
def normalized_or_zero(v: wp.vec3, length: float, threshold: float):
    """v / length when length > threshold (NaN lengths fail the test), else zero."""
    if length > threshold:
        return v / length
    return wp.vec3(0.0)


@wp.func
def coefficient_world(i: int, world: wp.array[wp.int32], particles_per_world: int):
    """CG coefficient group of row i: an explicit group map, or contiguous groups of particles_per_world rows."""
    if world.shape[0] > 0:
        return world[i]
    return i // particles_per_world


@wp.func
def atomic_add_block(
    row_count: wp.array[wp.int32],
    columns: wp.array[wp.int32],
    values: wp.array[wp.mat33],
    max_count: int,
    rows: int,
    row: int,
    column: int,
    value: wp.mat33,
):
    """Atomically add a 3x3 block at (row, column) of a slot-major block-ELL matrix; unstored columns are dropped.

    The first min(row_count[row], max_count) slots of a row are sorted by column, so a lower-bound search finds it.
    """
    count = wp.min(row_count[row], max_count)
    lower = int(0)
    upper = count
    while upper > lower:
        middle = lower + ((upper - lower) >> 1)
        if columns[middle * rows + row] < column:
            lower = middle + 1
        else:
            upper = middle
    if lower >= count:
        return
    if columns[lower * rows + row] != column:
        return
    wp.atomic_add(values, lower * rows + row, value)


# Threads per block of the tiled reduction kernels.
REDUCE_BLOCK = wp.constant(256)


@wp.func
def block_world_add(sums: wp.array[float], w: int, value: float, valid: bool, lane: int):
    """Add value into sums[w] with one atomic per block when the block's valid rows share one coefficient group.

    Native kernels reduce a warp with shuffles before their atomic (``warp_world_add``); per-row atomics into one
    address serialise and dominate large systems. Every thread of the block must call this; rows past the end pass
    ``valid = False``. Lane 0 is always a valid row, since blocks start below the row count.
    """
    first = wp.tile_extract(wp.tile(w), 0)
    mismatch = 0
    if valid and w != first:
        mismatch = 1
    if wp.tile_extract(wp.tile_max(wp.tile(mismatch)), 0) == 0:
        contribution = float(0.0)
        if valid:
            contribution = value
        total = wp.tile_extract(wp.tile_sum(wp.tile(contribution)), 0)
        if lane == 0:
            wp.atomic_add(sums, first, total)
    elif valid:
        wp.atomic_add(sums, w, value)

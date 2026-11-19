# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Particle-particle cloth springs (native ``kernels/springs.cu``)."""

import warp as wp

from .common import atomic_add_block, normalized_or_zero

# The solver never differentiates through its kernels; skipping the adjoints halves compile time.
wp.set_module_options({"enable_backward": False})


@wp.kernel
def evaluate_spring_force(
    x: wp.array[wp.vec3],
    springs: wp.array[wp.vec2i],
    stiffness: wp.array[float],
    rest_length: wp.array[float],
    force: wp.array[wp.vec3],
    springs_per_world: int,
    particles_per_world: int,
):
    """Spring force k (d - rest d/|d|) on the first endpoint and its negation on the second, d = x[b] - x[a].

    Thread i handles spring i % springs_per_world of world i / springs_per_world. A degenerate spring has no direction
    and pulls with k d only.
    """
    i = wp.tid()
    world = i // springs_per_world
    spring = i % springs_per_world
    offset = world * particles_per_world
    a = springs[spring][0] + offset
    b = springs[spring][1] + offset
    d = x[b] - x[a]
    k = stiffness[spring]
    rest = rest_length[spring]
    length = wp.length(d)
    f = k * (d - rest * normalized_or_zero(d, length, 0.0))
    wp.atomic_add(force, a, f)
    wp.atomic_add(force, b, -f)


@wp.kernel
def add_spring_gauss_newton_hessian(
    x: wp.array[wp.vec3],
    springs: wp.array[wp.vec2i],
    stiffness: wp.array[float],
    rest_length: wp.array[float],
    diagonal: wp.array[wp.mat33],
    row_count: wp.array[wp.int32],
    entry_columns: wp.array[wp.int32],
    entry_values: wp.array[wp.mat33],
    max_count: int,
    springs_per_world: int,
    particles_per_world: int,
    rows: int,
):
    """Gauss-Newton correction B = k (H - I) of a spring on top of the host's constant k I.

    H = n n^T + max(0, 1 - rest/|d|) (I - n n^T), or I for |d| <= 1e-8. B goes to both endpoint diagonals and -B to
    the off-diagonal blocks (a, b) and (b, a), or to the diagonals when a == b. Springs with k <= 0 or rest <= 0 are
    skipped.
    """
    i = wp.tid()
    world = i // springs_per_world
    spring = i % springs_per_world
    k = stiffness[spring]
    rest = rest_length[spring]
    if not (k > 0.0) or rest <= 0.0:
        return
    offset = world * particles_per_world
    a = springs[spring][0] + offset
    b = springs[spring][1] + offset
    d = x[b] - x[a]
    length = wp.length(d)
    identity = wp.identity(n=3, dtype=float)
    h = identity
    if length > 1.0e-8:
        n = d / length
        nn = wp.outer(n, n)
        h = (identity - nn) * wp.max(0.0, 1.0 - rest / length) + nn
    block = k * (h - identity)
    wp.atomic_add(diagonal, a, block)
    wp.atomic_add(diagonal, b, block)
    negative = -block
    if a == b:
        wp.atomic_add(diagonal, a, negative)
        wp.atomic_add(diagonal, b, negative)
    else:
        atomic_add_block(row_count, entry_columns, entry_values, max_count, rows, a, b, negative)
        atomic_add_block(row_count, entry_columns, entry_values, max_count, rows, b, a, negative)

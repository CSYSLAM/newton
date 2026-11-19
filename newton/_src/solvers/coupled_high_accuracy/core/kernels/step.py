# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-substep Newton bookkeeping (native ``kernels/step.cu``)."""

import warp as wp

# The solver never differentiates through its kernels; skipping the adjoints halves compile time.
wp.set_module_options({"enable_backward": False})


@wp.kernel
def apply_nonlinear_step(x: wp.array[wp.vec3], x_out: wp.array[wp.vec3], dx: wp.array[wp.vec3]):
    """x_out = x + dx after a Newton iteration; dx is cleared for the next linear solve."""
    i = wp.tid()
    x_out[i] = x[i] + dx[i]
    dx[i] = wp.vec3(0.0)


@wp.kernel
def update_velocity(inverse_dt: float, x_previous: wp.array[wp.vec3], x: wp.array[wp.vec3], v: wp.array[wp.vec3]):
    """v = (x - x_previous) / h with the host-computed reciprocal 1 / h."""
    i = wp.tid()
    v[i] = (x[i] - x_previous[i]) * inverse_dt


@wp.kernel
def initialize_block_diagonal(mass: wp.array[float], diagonal: wp.array[wp.mat33], copy: wp.array[wp.mat33]):
    """Diagonal blocks m * I, scaled as a matrix so NaN and -0 reach the off-diagonal entries as natively."""
    i = wp.tid()
    block = mass[i] * wp.identity(n=3, dtype=float)
    diagonal[i] = block
    if copy.shape[0] > 0:
        copy[i] = block


@wp.kernel
def combine_system_non_diagonal(
    scale: float,
    w1: float,
    w2: float,
    w3: float,
    w4: float,
    w5: float,
    a0: wp.array[float],
    a1: wp.array[float],
    a2: wp.array[float],
    a3: wp.array[float],
    a4: wp.array[float],
    a5: wp.array[float],
    a6: wp.array[float],
    out: wp.array[float],
):
    """Off-diagonal ELL values A0 + scale * (A1 + w1 A2 + ... + w5 A6) of templates sharing A0's columns."""
    i = wp.tid()
    out[i] = (a1[i] + a2[i] * w1 + a3[i] * w2 + a4[i] * w3 + a5[i] * w4 + a6[i] * w5) * scale + a0[i]


@wp.kernel
def refresh_inertial_cache_nonzeros(
    values_a: wp.array[float],
    offsets_a: wp.array[wp.int64],
    ids_a: wp.array[wp.int32],
    values_b: wp.array[float],
    offsets_b: wp.array[wp.int64],
    ids_b: wp.array[wp.int32],
    entries_a: wp.array[float],
    entries_b: wp.array[float],
):
    """Refresh ELL values from per-entry contribution lists: A gets the sum, B the negated sum (both from +0).

    Entries with an empty list keep their value.
    """
    i = wp.tid()
    begin_a = int(offsets_a[i])
    end_a = int(offsets_a[i + 1])
    if begin_a != end_a:
        total = float(0.0)
        for k in range(begin_a, end_a):
            total += values_a[ids_a[k]]
        entries_a[i] = total
    begin_b = int(offsets_b[i])
    end_b = int(offsets_b[i + 1])
    if begin_b != end_b:
        total = float(0.0)
        for k in range(begin_b, end_b):
            total -= values_b[ids_b[k]]
        entries_b[i] = total


@wp.kernel
def refresh_inertial_cache_diagonals(
    values: wp.array[float],
    offsets: wp.array[wp.int64],
    ids: wp.array[wp.int32],
    scaled: wp.array[float],
    total: wp.array[float],
):
    """Refresh the inertial diagonal: s = sum of the row's contributions; stores 3 s and s."""
    i = wp.tid()
    begin = int(offsets[i])
    end = int(offsets[i + 1])
    if begin == end:
        return
    s = float(0.0)
    for k in range(begin, end):
        s += values[ids[k]]
    scaled[i] = s * 3.0
    total[i] = s


@wp.kernel
def initialize_step(
    dt: float,
    gravity: wp.array[wp.vec3],
    particle_world: wp.array[wp.int32],
    velocity: wp.array[wp.vec3],
    x: wp.array[wp.vec3],
    host_diagonal: wp.array[float],
    d1: wp.array[float],
    c1: float,
    d2: wp.array[float],
    c2: float,
    d3: wp.array[float],
    c3: float,
    d4: wp.array[float],
    c4: float,
    d5: wp.array[float],
    c5: float,
    mass: wp.array[float],
    flags: wp.array[wp.int32],
    kind: wp.array[wp.int32],
    previous: wp.array[wp.vec3],
    x_hat: wp.array[wp.vec3],
    diagonal: wp.array[float],
    per_world: int,
):
    """Start a substep for particle i (in-world index p = i % per_world).

    previous = x. Inactive particles (mass <= 0 or no ACTIVE flag) keep x_hat = x with a zero diagonal; affine-body
    proxies among them (kind != 0) get previous = x - h v so the velocity update reproduces their velocity. Active
    particles get diagonal = host_diagonal[p] + (m + sum_k D_k[p] c_k) / h^2 and x_hat = x + h v + h^2 g, also written
    back to x. g is the gravity of the particle's world (clamped to the gravity array), zero for affine columns
    (kind 2) or without gravity.
    """
    i = wp.tid()
    particle = i % per_world
    position = x[i]
    m = mass[particle]
    active = m > 0.0
    if flags.shape[0] > 0:
        active = active and (flags[particle] & 1) != 0
    previous[i] = position
    if not active:
        if kind.shape[0] > 0:
            if kind[particle] != 0:
                previous[i] = position - velocity[i] * dt
        x_hat[i] = position
        diagonal[i] = 0.0
        return
    world = i // per_world
    if particle_world.shape[0] > 0:
        world = particle_world[particle]
    affine_column = False
    if kind.shape[0] > 0:
        affine_column = kind[particle] == 2
    g = wp.vec3(0.0)
    gravity_count = gravity.shape[0]
    if not affine_column and gravity_count > 0:
        g = gravity[wp.min(wp.max(world, 0), gravity_count - 1)]
    v = velocity[i]
    inertia = m
    inertia += d1[particle] * c1
    inertia += d2[particle] * c2
    inertia += d3[particle] * c3
    inertia += d4[particle] * c4
    inertia += d5[particle] * c5
    dt2 = dt * dt
    diagonal[i] = host_diagonal[particle] + inertia / dt2
    prediction = position + v * dt + dt2 * g
    x_hat[i] = prediction
    x[i] = prediction


@wp.kernel
def initialize_rhs(
    inverse_dt2: float,
    x: wp.array[wp.vec3],
    target: wp.array[wp.vec3],
    mass: wp.array[float],
    row_count: wp.array[wp.int32],
    entry_columns: wp.array[wp.int32],
    entry_values: wp.array[float],
    max_count: int,
    base: wp.array[wp.vec3],
    mask: wp.array[wp.int32],
    rhs: wp.array[wp.vec3],
    count: int,
    per_world: int,
):
    """rhs = (m (target - x) + sum_j M_ij (target_j - x_j)) / h^2 + base, zero for inactive particles.

    The optional off-diagonal mass rows (affine-body second moments) are scalar ELL entries of one world with in-world
    columns; columns outside the world or past the particle count are ignored.
    """
    i = wp.tid()
    particle = i % per_world
    m = mass[particle]
    inactive = not (m > 0.0)
    if mask.shape[0] > 0:
        inactive = inactive or (mask[particle] & 1) == 0
    if inactive:
        rhs[i] = wp.vec3(0.0)
        return
    inertia = m * (target[i] - x[i])
    coupling = wp.vec3(0.0)
    if max_count > 0:
        world_base = i - particle
        slots = wp.min(wp.max(row_count[particle], 0), max_count)
        for k in range(slots):
            column = entry_columns[k * per_world + particle]
            j = world_base + column
            if column >= 0 and column < per_world and j < count:
                coupling = coupling + entry_values[k * per_world + particle] * (target[j] - x[j])
    value = (inertia + coupling) * inverse_dt2
    if base.shape[0] > 0:
        value = value + base[i]
    rhs[i] = value

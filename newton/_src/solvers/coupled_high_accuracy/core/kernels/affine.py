# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Affine bodies (native ``kernels/affine.cu`` and ``affine_geometry.cuh``).

Each body is four state particles: the translation x and the columns of the affine matrix A = [y z w]. The
rigidity models keep A close to a rotation: SO(3) augmented Lagrangian (rotation R, multiplier Lambda), six springs,
or a constitutive energy (ABD orthogonality or co-rotated). Matrices are ordinary row/column math here; the native
``m[j]`` (column j) is ``column(m, j)``.
"""

import warp as wp

from .common import atomic_add_block, normalized_or_zero


@wp.func
def column(m: wp.mat33, j: int):
    return wp.vec3(m[0, j], m[1, j], m[2, j])


@wp.func
def body_rows(body_particles: wp.array[wp.vec4i], i: int, bodies_per_world: int, particles_per_world: int):
    """State particle rows of body state i: the body's template rows shifted to its world."""
    offset = (i // bodies_per_world) * particles_per_world
    p = body_particles[i % bodies_per_world]
    return wp.vec4i(p[0] + offset, p[1] + offset, p[2] + offset, p[3] + offset)


@wp.func
def normalize_or_zero(v: wp.vec3):
    return normalized_or_zero(v, wp.length(v), 0.0)


@wp.func
def nearest_rotation(a_in: wp.vec3, b_in: wp.vec3, c_in: wp.vec3):
    """Rotation closest to [a b c]: scale-normalise, up to eight Newton polar iterations X <- (X + cof(X)/det X)/2
    (stopping when det X is tiny or not finite), then Gram-Schmidt with the third axis oriented like c.
    Degenerate inputs give the identity."""
    a = a_in
    b = b_in
    c = c_in
    scale = wp.sqrt((wp.dot(a, a) + wp.dot(b, b) + wp.dot(c, c)) / 3.0)
    if not (scale > 1.0e-8) or not wp.isfinite(scale):
        return wp.identity(n=3, dtype=float)
    inverse_scale = 1.0 / scale
    a = a * inverse_scale
    b = b * inverse_scale
    c = c * inverse_scale
    for _iteration in range(8):
        bc = wp.cross(b, c)
        determinant = wp.dot(a, bc)
        if wp.abs(determinant) <= 1.0e-10 or not wp.isfinite(determinant):
            break
        inverse_determinant = 1.0 / determinant
        ca = wp.cross(c, a)
        ab = wp.cross(a, b)
        a = (bc * inverse_determinant + a) * 0.5
        b = (ca * inverse_determinant + b) * 0.5
        c = (ab * inverse_determinant + c) * 0.5
    n0 = normalize_or_zero(a)
    n1 = normalize_or_zero(b - wp.dot(b, n0) * n0)
    if wp.length(n0) <= 1.0e-8 or wp.length(n1) <= 1.0e-8:
        return wp.identity(n=3, dtype=float)
    n2 = normalize_or_zero(wp.cross(n0, n1))
    if wp.dot(c, n2) < 0.0:
        n1 = -n1
        n2 = -n2
    return wp.matrix_from_cols(n0, n1, n2)


@wp.func
def affine_matrix(x: wp.array[wp.vec3], rows: wp.vec4i):
    return wp.matrix_from_cols(x[rows[1]], x[rows[2]], x[rows[3]])


@wp.kernel
def evaluate_affine_body_so3_augmented_lagrangian_force(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    weight: wp.array[float],
    rotation: wp.array[wp.mat33],
    multiplier: wp.array[wp.mat33],
    alpha: float,
    force: wp.array[wp.vec3],
    bodies_per_world: int,
    particles_per_world: int,
):
    """Rigidity force -(w alpha) R sym(G^T R) on the affine columns, G = (A - R) + Lambda."""
    i = wp.tid()
    rows = body_rows(body_particles, i, bodies_per_world, particles_per_world)
    r = rotation[i]
    lam = multiplier[i]
    scale = weight[i % bodies_per_world] * alpha
    g = wp.matrix_from_cols(
        x[rows[1]] - column(r, 0) + column(lam, 0),
        x[rows[2]] - column(r, 1) + column(lam, 1),
        x[rows[3]] - column(r, 2) + column(lam, 2),
    )
    s = wp.transpose(g) * r
    y = (s + wp.transpose(s)) * 0.5
    f = r * y * -scale
    wp.atomic_add(force, rows[0], wp.vec3(0.0))
    wp.atomic_add(force, rows[1], column(f, 0))
    wp.atomic_add(force, rows[2], column(f, 1))
    wp.atomic_add(force, rows[3], column(f, 2))


@wp.kernel
def multiply_affine_body_so3_augmented_lagrangian_tangent(
    p: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    weight: wp.array[float],
    rotation: wp.array[wp.mat33],
    alpha: float,
    product: wp.array[wp.vec3],
    bodies_per_world: int,
    particles_per_world: int,
):
    """Rigidity tangent times a direction: the affine columns receive (w alpha) R sym(D^T R)."""
    i = wp.tid()
    rows = body_rows(body_particles, i, bodies_per_world, particles_per_world)
    d = affine_matrix(p, rows)
    r = rotation[i]
    scale = weight[i % bodies_per_world] * alpha
    s = wp.transpose(d) * r
    y = (s + wp.transpose(s)) * 0.5
    f = r * y * scale
    wp.atomic_add(product, rows[1], column(f, 0))
    wp.atomic_add(product, rows[2], column(f, 1))
    wp.atomic_add(product, rows[3], column(f, 2))


@wp.kernel
def initialize_affine_body_so3_state(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    rotation: wp.array[wp.mat33],
    multiplier: wp.array[wp.mat33],
    bodies_per_world: int,
    particles_per_world: int,
):
    """Start of a step: R = nearest rotation to A, Lambda = 0."""
    i = wp.tid()
    rows = body_rows(body_particles, i, bodies_per_world, particles_per_world)
    rotation[i] = nearest_rotation(x[rows[1]], x[rows[2]], x[rows[3]])
    multiplier[i] = wp.mat33(0.0)


@wp.kernel
def update_affine_body_so3_augmented_lagrangian_state(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    rotation: wp.array[wp.mat33],
    multiplier: wp.array[wp.mat33],
    update_multiplier: int,
    bodies_per_world: int,
    particles_per_world: int,
):
    """B = A + Lambda, R = nearest rotation to B and, when due, Lambda = R sym((B - R)^T R)."""
    i = wp.tid()
    rows = body_rows(body_particles, i, bodies_per_world, particles_per_world)
    b = affine_matrix(x, rows) + multiplier[i]
    r = nearest_rotation(column(b, 0), column(b, 1), column(b, 2))
    rotation[i] = r
    if update_multiplier == 0:
        return
    s = wp.transpose(b - r) * r
    multiplier[i] = r * ((s + wp.transpose(s)) * 0.5)


@wp.kernel
def add_affine_body_so3_augmented_lagrangian_preconditioner(
    body_particles: wp.array[wp.vec4i],
    weight: wp.array[float],
    rotation: wp.array[wp.mat33],
    alpha: float,
    preconditioner: wp.array[wp.mat33],
    bodies_per_world: int,
    particles_per_world: int,
):
    """Diagonal blocks of the rigidity tangent: the tangent of the unit direction e_c in affine column j gives block
    column c of column j's particle (multiplications by zero kept so NaN/inf in R propagate)."""
    i = wp.tid()
    rows = body_rows(body_particles, i, bodies_per_world, particles_per_world)
    scale = weight[i % bodies_per_world] * alpha
    r = rotation[i]
    for affine_column in range(3):
        block = wp.mat33(0.0)
        for c in range(3):
            d = wp.mat33(0.0)
            d[c, affine_column] = 1.0
            s = wp.transpose(d) * r
            t = column(scale * (r * ((s + wp.transpose(s)) * 0.5)), affine_column)
            block[0, c] = t[0]
            block[1, c] = t[1]
            block[2, c] = t[2]
        wp.atomic_add(preconditioner, rows[affine_column + 1], block)


@wp.func
def rigidity_spring_force(v: wp.vec3, k: float, rest: float, snap: float):
    """v (k e) / |v| with strain e = |v| - rest, snapped to zero when |e| <= snap; no force for |v| <= 0."""
    length = wp.sqrt(v[2] * v[2] + (v[1] * v[1] + v[0] * v[0]))
    if not (length > 0.0):
        return wp.vec3(0.0)
    strain = length - rest
    if wp.abs(strain) <= snap:
        strain = 0.0
    return v * (k * strain) / length


@wp.kernel
def evaluate_affine_body_spring_force(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    weight: wp.array[float],
    alpha: float,
    force: wp.array[wp.vec3],
    bodies_per_world: int,
    particles_per_world: int,
):
    """SIX_SPRINGS rigidity: springs on the affine columns (rest 1) and their differences (rest sqrt 2)."""
    i = wp.tid()
    rows = body_rows(body_particles, i, bodies_per_world, particles_per_world)
    k = weight[i % bodies_per_world] * alpha
    a0 = x[rows[1]]
    a1 = x[rows[2]]
    a2 = x[rows[3]]
    unit_snap = 9.5367431640625e-07
    diagonal_snap = 1.3486991523486091e-06
    sqrt2 = 1.41421356
    f0 = wp.vec3(0.0) - rigidity_spring_force(a0, k, 1.0, unit_snap)
    f1 = wp.vec3(0.0) - rigidity_spring_force(a1, k, 1.0, unit_snap)
    f2 = wp.vec3(0.0) - rigidity_spring_force(a2, k, 1.0, unit_snap)
    f10 = rigidity_spring_force(a1 - a0, k, sqrt2, diagonal_snap)
    f0 = f0 + f10
    f1 = f1 - f10
    f20 = rigidity_spring_force(a2 - a0, k, sqrt2, diagonal_snap)
    f0 = f0 + f20
    f2 = f2 - f20
    f21 = rigidity_spring_force(a2 - a1, k, sqrt2, diagonal_snap)
    f1 = f1 + f21
    f2 = f2 - f21
    wp.atomic_add(force, rows[0], wp.vec3(0.0))
    wp.atomic_add(force, rows[1], f0)
    wp.atomic_add(force, rows[2], f1)
    wp.atomic_add(force, rows[3], f2)


@wp.func
def rigidity_spring_correction(v: wp.vec3, k: float, unit_column: int):
    """k (M - I) with M = n n^T + s (I - n n^T), s = max(0, 1 - rest/|v|) (rest 1 or sqrt 2); M = I for tiny v."""
    length = wp.length(v)
    identity = wp.identity(n=3, dtype=float)
    m = identity
    if length > 1.0e-8:
        n = v / length
        nn = wp.outer(n, n)
        s = float(0.0)
        if unit_column != 0:
            s = wp.max(0.0, 1.0 - 1.0 / length)
        else:
            s = wp.max(0.0, -1.41421356 / length + 1.0)
        m = (identity - nn) * s + nn
    return k * (m - identity)


@wp.kernel
def add_affine_body_spring_gauss_newton_hessian(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    weight: wp.array[float],
    alpha: float,
    diagonal: wp.array[wp.mat33],
    row_count: wp.array[wp.int32],
    entry_columns: wp.array[wp.int32],
    entry_values: wp.array[wp.mat33],
    max_count: int,
    bodies_per_world: int,
    particles_per_world: int,
    rows_total: int,
):
    """SIX_SPRINGS Gauss-Newton Hessian: column springs into the column diagonals, then each column pair's
    correction into both diagonals and its negation into the off-diagonal blocks (k > 0 only)."""
    i = wp.tid()
    rows = body_rows(body_particles, i, bodies_per_world, particles_per_world)
    k = weight[i % bodies_per_world] * alpha
    for c in range(1, 4):
        wp.atomic_add(diagonal, rows[c], rigidity_spring_correction(x[rows[c]], k, 1))
    if not (k > 0.0):
        return
    for a in range(1, 4):
        for b in range(a + 1, 4):
            ci = rows[a]
            cj = rows[b]
            block = rigidity_spring_correction(x[cj] - x[ci], k, 0)
            wp.atomic_add(diagonal, ci, block)
            wp.atomic_add(diagonal, cj, block)
            negative = -block
            if ci == cj:
                wp.atomic_add(diagonal, ci, negative)
                wp.atomic_add(diagonal, cj, negative)
            else:
                atomic_add_block(row_count, entry_columns, entry_values, max_count, rows_total, ci, cj, negative)
                atomic_add_block(row_count, entry_columns, entry_values, max_count, rows_total, cj, ci, negative)


@wp.func
def dot_x_first(r: wp.vec3, a: wp.vec3):
    return r[2] * a[2] + (r[1] * a[1] + r[0] * a[0])


@wp.func
def constitutive_gradient(a: wp.mat33, model: int):
    """Column gradients of the rigidity energy: (A A^T - I) A for ABD_ORTHOGONALITY (1), else co-rotated R P with
    R the nearest rotation, S = R^T A and P = (S + S^T - 2 I) + (tr S - 3) I."""
    identity = wp.identity(n=3, dtype=float)
    if model == 1:
        return (a * wp.transpose(a) - identity) * a
    r = nearest_rotation(column(a, 0), column(a, 1), column(a, 2))
    s = wp.mat33(0.0)
    for j in range(3):
        for i in range(3):
            s[i, j] = dot_x_first(column(r, i), column(a, j))
    e = s + wp.transpose(s) - 2.0 * identity
    trace = s[0, 0] + s[1, 1] + s[2, 2] - 3.0
    p = e + trace * identity
    r0 = column(r, 0)
    r1 = column(r, 1)
    r2 = column(r, 2)
    # The second column leads with its diagonal term, as natively.
    return wp.matrix_from_cols(
        r0 * p[0, 0] + r1 * p[1, 0] + r2 * p[2, 0],
        r1 * p[1, 1] + r0 * p[0, 1] + r2 * p[2, 1],
        r0 * p[0, 2] + r1 * p[1, 2] + r2 * p[2, 2],
    )


@wp.func
def constitutive_tangent(a: wp.mat33, d: wp.mat33, model: int):
    """Directional derivative of the column gradients along D: (A D^T + D A^T) A, or R (T + T^T + tr T I) with
    T = R^T D for the co-rotated model (R frozen)."""
    if model == 1:
        m = a * wp.transpose(d)
        return (m + wp.transpose(m)) * a
    r = nearest_rotation(column(a, 0), column(a, 1), column(a, 2))
    t = wp.transpose(r) * d
    trace = t[0, 0] + t[1, 1] + t[2, 2]
    return r * (t + wp.transpose(t) + trace * wp.identity(n=3, dtype=float))


@wp.kernel
def evaluate_affine_body_constitutive_force(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    weight: wp.array[float],
    alpha: float,
    model: int,
    force: wp.array[wp.vec3],
    bodies_per_world: int,
    particles_per_world: int,
):
    """Constitutive rigidity force: column j receives -(w alpha) G_j."""
    i = wp.tid()
    rows = body_rows(body_particles, i, bodies_per_world, particles_per_world)
    g = constitutive_gradient(affine_matrix(x, rows), model)
    scale = weight[i % bodies_per_world] * alpha
    wp.atomic_add(force, rows[0], wp.vec3(scale * -0.0))
    wp.atomic_add(force, rows[1], column(g, 0) * -scale)
    wp.atomic_add(force, rows[2], column(g, 1) * -scale)
    wp.atomic_add(force, rows[3], column(g, 2) * -scale)


@wp.kernel
def multiply_affine_body_constitutive_tangent(
    x: wp.array[wp.vec3],
    p: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    weight: wp.array[float],
    alpha: float,
    model: int,
    product: wp.array[wp.vec3],
    bodies_per_world: int,
    particles_per_world: int,
):
    """Constitutive rigidity tangent times a direction: column j receives (w alpha) G_j(D)."""
    i = wp.tid()
    rows = body_rows(body_particles, i, bodies_per_world, particles_per_world)
    g = constitutive_tangent(affine_matrix(x, rows), affine_matrix(p, rows), model)
    scale = weight[i % bodies_per_world] * alpha
    wp.atomic_add(product, rows[0], wp.vec3(scale * 0.0))
    wp.atomic_add(product, rows[1], scale * column(g, 0))
    wp.atomic_add(product, rows[2], scale * column(g, 1))
    wp.atomic_add(product, rows[3], scale * column(g, 2))


@wp.kernel
def add_affine_body_constitutive_preconditioner(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    weight: wp.array[float],
    alpha: float,
    model: int,
    preconditioner: wp.array[wp.mat33],
    bodies_per_world: int,
    particles_per_world: int,
):
    """Diagonal blocks of the constitutive tangent: block column k of column j's particle is (w alpha) G_j of the
    unit direction e_k in column j."""
    i = wp.tid()
    rows = body_rows(body_particles, i, bodies_per_world, particles_per_world)
    a = affine_matrix(x, rows)
    scale = weight[i % bodies_per_world] * alpha
    wp.atomic_add(preconditioner, rows[0], wp.mat33(scale * 0.0))
    for affine_column in range(3):
        block = wp.mat33(0.0)
        for k in range(3):
            d = wp.mat33(0.0)
            d[k, affine_column] = 1.0
            t = column(scale * constitutive_tangent(a, d, model), affine_column)
            block[0, k] = t[0]
            block[1, k] = t[1]
            block[2, k] = t[2]
        wp.atomic_add(preconditioner, rows[affine_column + 1], block)

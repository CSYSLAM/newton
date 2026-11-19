# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Volumetric tetrahedra with optional strain-rate damping (native ``kernels/tetrahedra.cu`` and the force of
``revisions/tetrahedra_23fb797.cu``).

Inverse rest shapes are native ``glm::mat3`` spans read as ``wp.mat33`` over the same memory, so the native
``a[i][j]`` is ``a[i, j]`` here and the native column ``a[k]`` is the row ``a[k]``.
"""

import warp as wp

# The solver never differentiates through its kernels; skipping the adjoints halves compile time.
wp.set_module_options({"enable_backward": False})


@wp.struct
class Deformation:
    c0: wp.vec3
    c1: wp.vec3
    c2: wp.vec3


@wp.func
def deformation(x: wp.array[wp.vec3], v0: int, v1: int, v2: int, v3: int, a: wp.mat33):
    """Deformation gradient columns F_j = sum_i e_i a[i][j] with e_i = x_i - x_0."""
    x0 = x[v0]
    e1 = x[v1] - x0
    e2 = x[v2] - x0
    e3 = x[v3] - x0
    f = Deformation()
    f.c0 = e1 * a[0, 0] + e2 * a[1, 0] + e3 * a[2, 0]
    f.c1 = e1 * a[0, 1] + e2 * a[1, 1] + e3 * a[2, 1]
    f.c2 = e1 * a[0, 2] + e2 * a[1, 2] + e3 * a[2, 2]
    return f


@wp.func
def vertex_gradient(a: wp.mat33, k: int):
    """Rest-shape gradient h_k of vertex k: -a[0] - a[1] - a[2] for k = 0, else a[k - 1]."""
    if k == 0:
        return -a[0] - a[1] - a[2]
    return a[k - 1]


@wp.kernel
def evaluate_tetrahedron_force(
    x: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    tetrahedra: wp.array[wp.vec4i],
    inverse_rest: wp.array[wp.mat33],
    materials: wp.array[wp.vec3],
    volume: wp.array[float],
    activation: wp.array[float],
    inverse_dt: float,
    force: wp.array[wp.vec3],
    tetrahedra_per_world: int,
    particles_per_world: int,
):
    """Elastic force -V P h_k with P = mu F + (mu + lambda) (activation + det F - target) cof F.

    target = 1 + mu / (mu + lambda) (1 when mu + lambda <= 0). With damping c > 0 the strain-rate term
    -c V sum_ab g_ab t_ab is added, g = (F^T F - F_prev^T F_prev) / h (off-diagonal entries doubled) and t_ab the
    derivative of (F^T F)_ab along h_k. Activations are per tetrahedron state (index i).
    """
    i = wp.tid()
    world = i // tetrahedra_per_world
    tet = i % tetrahedra_per_world
    local = tetrahedra[tet]
    offset = world * particles_per_world
    v = wp.vec4i(local[0] + offset, local[1] + offset, local[2] + offset, local[3] + offset)
    a = inverse_rest[tet]
    material = materials[tet]
    mu = material[0]
    damping = material[2]
    vol = volume[tet]
    f = deformation(x, v[0], v[1], v[2], v[3], a)
    k12 = wp.cross(f.c1, f.c2)
    determinant = wp.dot(f.c0, k12)
    bulk = mu + material[1]
    activation_value = float(0.0)
    if activation.shape[0] > 0:
        activation_value = activation[i]
    target = float(1.0)
    if bulk > 0.0:
        target = mu / bulk + 1.0
    scale = bulk * (activation_value + (determinant - target))
    k20 = wp.cross(f.c2, f.c0)
    k01 = wp.cross(f.c0, f.c1)
    p0 = mu * f.c0 + k12 * scale
    p1 = mu * f.c1 + k20 * scale
    p2 = mu * f.c2 + k01 * scale
    g00 = float(0.0)
    g01 = float(0.0)
    g02 = float(0.0)
    g11 = float(0.0)
    g12 = float(0.0)
    g22 = float(0.0)
    if damping > 0.0:
        r = deformation(previous, v[0], v[1], v[2], v[3], a)
        d00 = wp.dot(f.c0, f.c0) - wp.dot(r.c0, r.c0)
        d01 = wp.dot(f.c0, f.c1) - wp.dot(r.c0, r.c1)
        d02 = wp.dot(f.c0, f.c2) - wp.dot(r.c0, r.c2)
        d11 = wp.dot(f.c1, f.c1) - wp.dot(r.c1, r.c1)
        d12 = wp.dot(f.c1, f.c2) - wp.dot(r.c1, r.c2)
        d22 = wp.dot(f.c2, f.c2) - wp.dot(r.c2, r.c2)
        g00 = d00 * inverse_dt
        g11 = d11 * inverse_dt
        g22 = d22 * inverse_dt
        g01 = d01 * inverse_dt + d01 * inverse_dt
        g02 = d02 * inverse_dt + d02 * inverse_dt
        g12 = d12 * inverse_dt + d12 * inverse_dt
    weight = damping * vol
    negative_volume = -vol
    for k in range(4):
        h = vertex_gradient(a, k)
        h0 = h[0]
        h1 = h[1]
        h2 = h[2]
        total = (p0 * h0 + p1 * h1 + p2 * h2) * negative_volume
        if damping > 0.0:
            t00 = f.c0 * (h0 + h0)
            t01 = f.c1 * h0 + f.c0 * h1
            t02 = f.c2 * h0 + f.c0 * h2
            t11 = f.c1 * (h1 + h1)
            t12 = f.c2 * h1 + f.c1 * h2
            t22 = f.c2 * (h2 + h2)
            rate = g00 * t00 + g01 * t01 + g02 * t02 + g11 * t11 + g12 * t12 + g22 * t22
            total = total - weight * rate
        wp.atomic_add(force, v[k], total)


@wp.kernel
def multiply_tetrahedron_tangent(
    x: wp.array[wp.vec3],
    p: wp.array[wp.vec3],
    tetrahedra: wp.array[wp.vec4i],
    inverse_rest: wp.array[wp.mat33],
    materials: wp.array[wp.vec3],
    volume: wp.array[float],
    inverse_dt: float,
    product: wp.array[wp.vec3],
    tetrahedra_per_world: int,
    particles_per_world: int,
):
    """Matrix-free tangent: V (dP h_k) with dP = mu dF + (mu + lambda) (cof F : dF) cof F, plus the Gauss-Newton
    strain-rate damping (c V / h) sum_ab dS_ab t_ab with dS = d(F^T F) along p (c > 0 only)."""
    i = wp.tid()
    world = i // tetrahedra_per_world
    tet = i % tetrahedra_per_world
    local = tetrahedra[tet]
    offset = world * particles_per_world
    v = wp.vec4i(local[0] + offset, local[1] + offset, local[2] + offset, local[3] + offset)
    a = inverse_rest[tet]
    f = deformation(x, v[0], v[1], v[2], v[3], a)
    d = deformation(p, v[0], v[1], v[2], v[3], a)
    k12 = wp.cross(f.c1, f.c2)
    k20 = wp.cross(f.c2, f.c0)
    k01 = wp.cross(f.c0, f.c1)
    material = materials[tet]
    mu = material[0]
    damping = material[2]
    scale = (mu + material[1]) * (wp.dot(k12, d.c0) + wp.dot(k20, d.c1) + wp.dot(k01, d.c2))
    dp0 = mu * d.c0 + k12 * scale
    dp1 = mu * d.c1 + k20 * scale
    dp2 = mu * d.c2 + k01 * scale
    s00 = wp.dot(f.c0, d.c0)
    s00 = s00 + s00
    s01 = wp.dot(f.c1, d.c0) + wp.dot(f.c0, d.c1)
    s01 = s01 + s01
    s02 = wp.dot(f.c2, d.c0) + wp.dot(f.c0, d.c2)
    s02 = s02 + s02
    s11 = wp.dot(f.c1, d.c1)
    s11 = s11 + s11
    s12 = wp.dot(f.c2, d.c1) + wp.dot(f.c1, d.c2)
    s12 = s12 + s12
    s22 = wp.dot(f.c2, d.c2)
    s22 = s22 + s22
    vol = volume[tet]
    weight = damping * vol * inverse_dt
    for k in range(4):
        h = vertex_gradient(a, k)
        h0 = h[0]
        h1 = h[1]
        h2 = h[2]
        value = vol * (dp0 * h0 + dp1 * h1 + dp2 * h2)
        if damping > 0.0:
            t00 = f.c0 * (h0 + h0)
            t01 = f.c1 * h0 + f.c0 * h1
            t02 = f.c2 * h0 + f.c0 * h2
            t11 = f.c1 * (h1 + h1)
            t12 = f.c2 * h1 + f.c1 * h2
            t22 = f.c2 * (h2 + h2)
            total = s00 * t00 + s01 * t01 + s02 * t02 + s11 * t11 + s12 * t12 + s22 * t22
            value = weight * total + value
        wp.atomic_add(product, v[k], value)


@wp.kernel
def add_tetrahedron_preconditioner(
    x: wp.array[wp.vec3],
    tetrahedra: wp.array[wp.vec4i],
    inverse_rest: wp.array[wp.mat33],
    materials: wp.array[wp.vec3],
    volume: wp.array[float],
    inverse_dt: float,
    preconditioner: wp.array[wp.mat33],
    tetrahedra_per_world: int,
    particles_per_world: int,
):
    """Diagonal tangent blocks mu V |h|^2 I + (mu + lambda) V cg cg^T with cg = cof F h, plus (c V / h) times the
    strain-rate basis outer products (off-diagonal basis terms weighted twice) when the damping c > 0."""
    i = wp.tid()
    world = i // tetrahedra_per_world
    tet = i % tetrahedra_per_world
    local = tetrahedra[tet]
    offset = world * particles_per_world
    v = wp.vec4i(local[0] + offset, local[1] + offset, local[2] + offset, local[3] + offset)
    a = inverse_rest[tet]
    f = deformation(x, v[0], v[1], v[2], v[3], a)
    k12 = wp.cross(f.c1, f.c2)
    k20 = wp.cross(f.c2, f.c0)
    k01 = wp.cross(f.c0, f.c1)
    material = materials[tet]
    mu = material[0]
    damping = material[2]
    vol = volume[tet]
    w_mu = mu * vol
    w_bulk = (mu + material[1]) * vol
    w_damp = damping * vol * inverse_dt
    for k in range(4):
        h = vertex_gradient(a, k)
        h0 = h[0]
        h1 = h[1]
        h2 = h[2]
        cg = k12 * h0 + k20 * h1 + k01 * h2
        iso = w_mu * wp.dot(h, h)
        block = iso * wp.identity(n=3, dtype=float) + w_bulk * wp.outer(cg, cg)
        if damping > 0.0:
            t00 = f.c0 * (h0 + h0)
            t01 = f.c1 * h0 + f.c0 * h1
            t02 = f.c2 * h0 + f.c0 * h2
            t11 = f.c1 * (h1 + h1)
            t12 = f.c2 * h1 + f.c1 * h2
            t22 = f.c2 * (h2 + h2)
            for column in range(3):
                for row in range(column, 3):
                    total = t01[row] * t01[column] + t01[row] * t01[column]
                    total = t00[row] * t00[column] + total
                    total = (t02[row] * t02[column]) * 2.0 + total
                    total = t11[row] * t11[column] + total
                    total = (t12[row] * t12[column]) * 2.0 + total
                    total = t22[row] * t22[column] + total
                    value = w_damp * total + block[row, column]
                    block[row, column] = value
                    block[column, row] = value
        wp.atomic_add(preconditioner, v[k], block)

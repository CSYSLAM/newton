# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Style3D anisotropic membrane and cotangent bending (native ``kernels/cloth.cu``).

Triangle rest poses are native ``glm::mat2`` spans read as ``wp.mat22`` over the same memory, so the native
``pose[i][j]`` is ``pose[i, j]`` here.
"""

import warp as wp

from .common import atomic_add_block, normalized_or_zero

# The solver never differentiates through its kernels; skipping the adjoints halves compile time.
wp.set_module_options({"enable_backward": False})


@wp.func
def hinge_vector(x: wp.array[wp.vec3], v0: int, v1: int, v2: int, v3: int, w1: float, w2: float, w3: float):
    """Bending hinge vector q = sum_k w_k (x_k - x_0) over the edge's other three vertices, from zero."""
    base = x[v0]
    q = wp.vec3(0.0)
    q += w1 * (x[v1] - base)
    q += w2 * (x[v2] - base)
    q += w3 * (x[v3] - base)
    return q


@wp.func
def bending_edge_valid(e: wp.vec4i, area: float):
    """An edge bends when both opposite vertices exist, all four vertices differ and the hinge area is positive."""
    return not (
        e[0] < 0
        or e[1] < 0
        or e[0] == e[1]
        or e[0] == e[2]
        or e[0] == e[3]
        or e[1] == e[2]
        or e[1] == e[3]
        or e[2] == e[3]
        or area <= 0.0
    )


@wp.kernel
def evaluate_stretch_force(
    x: wp.array[wp.vec3],
    area: wp.array[float],
    poses: wp.array[wp.mat22],
    triangles: wp.array[wp.vec3i],
    stiffness: wp.array[wp.vec3],
    force: wp.array[wp.vec3],
    triangles_per_world: int,
    particles_per_world: int,
):
    """Anisotropic membrane force: ku (|a| - 1), kv (|b| - 1) along the warp/weft directions plus ks times the shear.

    a = e1 m00 + e2 m10 and b = e1 m01 + e2 m11; vertex k has weights (-m00 - m10, -m01 - m11), (m00, m01),
    (m10, m11) and receives -area times the weighted gradient.
    """
    i = wp.tid()
    world = i // triangles_per_world
    triangle = i % triangles_per_world
    offset = world * particles_per_world
    local = triangles[triangle]
    v0 = local[0] + offset
    v1 = local[1] + offset
    v2 = local[2] + offset
    e1 = x[v1] - x[v0]
    e2 = x[v2] - x[v0]
    pose = poses[triangle]
    m00 = pose[0, 0]
    m01 = pose[0, 1]
    m10 = pose[1, 0]
    m11 = pose[1, 1]
    a = e1 * m00 + e2 * m10
    b = e1 * m01 + e2 * m11
    a2 = wp.dot(a, a)
    a_length = wp.sqrt(a2)
    a_unit = normalized_or_zero(a, a_length, 0.0)
    b2 = wp.dot(b, b)
    b_length = wp.sqrt(b2)
    b_unit = normalized_or_zero(b, b_length, 0.0)
    wu0 = -m00 - m10
    wv0 = -m01 - m11
    a_regular = wp.sqrt(a2 + 1.0e-12)
    b_regular = wp.sqrt(b2 + 1.0e-12)
    aa = a / a_regular
    bb = b / b_regular
    shear = wp.dot(aa, bb)
    da = (bb - aa * shear) / a_regular
    db = (aa - bb * shear) / b_regular
    k = stiffness[triangle]
    ca = k[0] * (a_length - 1.0)
    cb = k[1] * (b_length - 1.0)
    cs = k[2] * shear
    scale = -area[triangle]
    wp.atomic_add(force, v0, ((wu0 * a_unit) * ca + (wv0 * b_unit) * cb + (da * wu0 + db * wv0) * cs) * scale)
    wp.atomic_add(force, v1, ((m00 * a_unit) * ca + (m01 * b_unit) * cb + (da * m00 + db * m01) * cs) * scale)
    wp.atomic_add(force, v2, ((m10 * a_unit) * ca + (m11 * b_unit) * cb + (da * m10 + db * m11) * cs) * scale)


@wp.kernel
def evaluate_bending_force(
    x: wp.array[wp.vec3],
    x_previous: wp.array[wp.vec3],
    area: wp.array[float],
    cotangents: wp.array[wp.vec4],
    edges: wp.array[wp.vec4i],
    properties: wp.array[wp.vec2],
    inverse_dt: float,
    force: wp.array[wp.vec3],
    edges_per_world: int,
    particles_per_world: int,
):
    """Cotangent bending w_k (d - (3/A) kb q) with the hinge-rate damping d = -(3/A) kd (q_rate . n) n."""
    i = wp.tid()
    world = i // edges_per_world
    edge = i % edges_per_world
    e = edges[edge]
    a = area[edge]
    if not bending_edge_valid(e, a):
        return
    offset = world * particles_per_world
    v0 = e[0] + offset
    v1 = e[1] + offset
    v2 = e[2] + offset
    v3 = e[3] + offset
    c = cotangents[edge]
    w1 = -c[1] - c[3]
    w2 = c[2] + c[3]
    w3 = c[0] + c[1]
    prop = properties[edge]
    q = hinge_vector(x, v0, v1, v2, v3, w1, w2, w3)
    scale = 3.0 / a
    damping = scale * prop[1]
    d = wp.vec3(0.0)
    if damping > 0.0:
        n = normalized_or_zero(q, wp.length(q), 1.0e-12)
        rate = (q - hinge_vector(x_previous, v0, v1, v2, v3, w1, w2, w3)) * inverse_dt
        d = n * -(damping * wp.dot(rate, n))
    w0 = -c[0] - c[2]
    total = d - q * (scale * prop[0])
    wp.atomic_add(force, v0, total * w0)
    wp.atomic_add(force, v1, total * w1)
    wp.atomic_add(force, v2, total * w2)
    wp.atomic_add(force, v3, total * w3)


@wp.kernel
def add_bending_damping_preconditioner(
    x: wp.array[wp.vec3],
    area: wp.array[float],
    cotangents: wp.array[wp.vec4],
    edges: wp.array[wp.vec4i],
    properties: wp.array[wp.vec2],
    inverse_dt: float,
    preconditioner: wp.array[wp.mat33],
    edges_per_world: int,
    particles_per_world: int,
):
    """w_k^2 c n n^T on vertex k's diagonal block, c = kd (3/A) / h, for valid hinges with kd > 0."""
    i = wp.tid()
    edge = i % edges_per_world
    e = edges[edge]
    a = area[edge]
    kd = properties[edge][1]
    if not bending_edge_valid(e, a) or kd <= 0.0:
        return
    offset = (i // edges_per_world) * particles_per_world
    v0 = e[0] + offset
    v1 = e[1] + offset
    v2 = e[2] + offset
    v3 = e[3] + offset
    c = cotangents[edge]
    w1 = -c[1] - c[3]
    w2 = c[2] + c[3]
    w3 = c[0] + c[1]
    q = hinge_vector(x, v0, v1, v2, v3, w1, w2, w3)
    n = normalized_or_zero(q, wp.length(q), 1.0e-12)
    coefficient = kd * (3.0 / a) * inverse_dt
    scaled = coefficient * wp.outer(n, n)
    w0 = -c[0] - c[2]
    wp.atomic_add(preconditioner, v0, scaled * (w0 * w0))
    wp.atomic_add(preconditioner, v1, scaled * (w1 * w1))
    wp.atomic_add(preconditioner, v2, scaled * (w2 * w2))
    wp.atomic_add(preconditioner, v3, scaled * (w3 * w3))


@wp.kernel
def multiply_bending_damping_tangent(
    x: wp.array[wp.vec3],
    p: wp.array[wp.vec3],
    area: wp.array[float],
    cotangents: wp.array[wp.vec4],
    edges: wp.array[wp.vec4i],
    properties: wp.array[wp.vec2],
    inverse_dt: float,
    product: wp.array[wp.vec3],
    edges_per_world: int,
    particles_per_world: int,
):
    """Bending-damping tangent times p: vertex k receives w_k c (dq . n) n with dq the hinge vector of p."""
    i = wp.tid()
    edge = i % edges_per_world
    e = edges[edge]
    a = area[edge]
    kd = properties[edge][1]
    if not bending_edge_valid(e, a) or kd <= 0.0:
        return
    offset = (i // edges_per_world) * particles_per_world
    v0 = e[0] + offset
    v1 = e[1] + offset
    v2 = e[2] + offset
    v3 = e[3] + offset
    c = cotangents[edge]
    w1 = -c[1] - c[3]
    w2 = c[2] + c[3]
    w3 = c[0] + c[1]
    q = hinge_vector(x, v0, v1, v2, v3, w1, w2, w3)
    n = normalized_or_zero(q, wp.length(q), 1.0e-12)
    dq = hinge_vector(p, v0, v1, v2, v3, w1, w2, w3)
    coefficient = kd * (3.0 / a) * inverse_dt
    g = coefficient * wp.dot(dq, n) * n
    w0 = -c[0] - c[2]
    wp.atomic_add(product, v0, g * w0)
    wp.atomic_add(product, v1, g * w1)
    wp.atomic_add(product, v2, g * w2)
    wp.atomic_add(product, v3, g * w3)


@wp.func
def stretch_projector(v: wp.vec3, length: float):
    """n n^T + max(0, 1 - 1/|v|) (I - n n^T) of a membrane direction v; the identity when |v| <= 1e-8."""
    identity = wp.identity(n=3, dtype=float)
    m = identity
    if length > 1.0e-8:
        n = v / length
        nn = wp.outer(n, n)
        m = (identity - nn) * wp.max(0.0, 1.0 - 1.0 / length) + nn
    return m


@wp.kernel
def add_cloth_membrane_gauss_newton_hessian(
    x: wp.array[wp.vec3],
    triangles: wp.array[wp.vec3i],
    poses: wp.array[wp.mat22],
    area: wp.array[float],
    stiffness: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
    row_count: wp.array[wp.int32],
    entry_columns: wp.array[wp.int32],
    entry_values: wp.array[wp.mat33],
    max_count: int,
    triangles_per_world: int,
    particles_per_world: int,
    rows: int,
):
    """Membrane Gauss-Newton blocks area (ku u_i u_j Pa + kv v_i v_j Pb + ks t_i t_j^T - iso I) per vertex pair.

    The isotropic part iso = (ku + ks) u_i u_j + (kv + ks) v_i v_j lives in the host's constant matrix and is
    removed here. Blocks go to the diagonal (same particle) or to the block-ELL entry (row v_i, column v_j).
    """
    i = wp.tid()
    world = i // triangles_per_world
    triangle = i % triangles_per_world
    offset = world * particles_per_world
    local = triangles[triangle]
    vertices = wp.vec3i(local[0] + offset, local[1] + offset, local[2] + offset)
    e1 = x[vertices[1]] - x[vertices[0]]
    e2 = x[vertices[2]] - x[vertices[0]]
    pose = poses[triangle]
    m00 = pose[0, 0]
    m01 = pose[0, 1]
    m10 = pose[1, 0]
    m11 = pose[1, 1]
    a = e1 * m00 + e2 * m10
    b = e1 * m01 + e2 * m11
    a2 = wp.dot(a, a)
    b2 = wp.dot(b, b)
    a_regular = wp.sqrt(a2 + 1.0e-12)
    b_regular = wp.sqrt(b2 + 1.0e-12)
    aa = a / a_regular
    bb = b / b_regular
    shear = wp.dot(aa, bb)
    da = (bb - aa * shear) / a_regular
    db = (aa - bb * shear) / b_regular
    wu = wp.vec3(-m00 - m10, m00, m10)
    wv = wp.vec3(-m01 - m11, m01, m11)
    pa = stretch_projector(a, wp.sqrt(a2))
    pb = stretch_projector(b, wp.sqrt(b2))
    k = stiffness[triangle]
    kua = k[0] + k[2]
    kvb = k[1] + k[2]
    s = area[triangle]
    identity = wp.identity(n=3, dtype=float)
    for vi in range(3):
        ti = da * wu[vi] + db * wv[vi]
        for vj in range(3):
            tj = da * wu[vj] + db * wv[vj]
            uu = wu[vi] * wu[vj]
            vv = wv[vi] * wv[vj]
            iso = kua * uu + kvb * vv
            block = s * ((k[0] * uu) * pa + (k[1] * vv) * pb + k[2] * wp.outer(ti, tj) - iso * identity)
            row = vertices[vi]
            column = vertices[vj]
            if row == column:
                wp.atomic_add(diagonal, row, block)
            else:
                atomic_add_block(row_count, entry_columns, entry_values, max_count, rows, row, column, block)

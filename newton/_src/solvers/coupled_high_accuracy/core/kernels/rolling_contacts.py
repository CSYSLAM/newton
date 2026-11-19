# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rolling and torsional friction of body contacts (native ``kernels/rolling_contacts.cu``).

The relative angular velocity of the two bodies is measured through the affine columns with each body's inverse
inertia; rolling (tangential) and spinning (normal) components are resisted by torques of magnitude
radius mu fn, regularised below ``epsilon``.
"""

import warp as wp

from .contact_geometry import (
    contact_body_slot,
    contact_dot,
    contact_dot_x_first,
    contact_normalize,
    contact_point,
    contact_point_velocity,
    contact_shape,
    contact_sphere_offset,
    contact_surface_velocity,
)

# The solver never differentiates through its kernels; skipping the adjoints halves compile time.
wp.set_module_options({"enable_backward": False})


vec6i = wp.types.vector(length=6, dtype=wp.int32)
mat63 = wp.types.matrix(shape=(6, 3), dtype=wp.float32)


@wp.struct
class RollingContactView:
    count: wp.array[wp.int32]
    capacity: int
    shape0: wp.array[wp.int32]
    shape1: wp.array[wp.int32]
    point0: wp.array[wp.vec3]
    point1: wp.array[wp.vec3]
    offset0: wp.array[wp.vec3]
    offset1: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    margin0: wp.array[float]
    margin1: wp.array[float]
    shape_body: wp.array[wp.int32]
    shape_type: wp.array[wp.int32]
    transform: wp.array[wp.transform]
    shape_scale: wp.array[wp.vec3]
    ke: wp.array[float]
    kd: wp.array[float]
    rolling: wp.array[float]
    torsional: wp.array[float]
    local_shape_count: int
    shapes_per_world: int
    worlds: int
    body_ids: wp.array[wp.int32]
    body_indices: wp.array[wp.vec4i]
    body_com: wp.array[wp.vec3]
    inertia: wp.array[float]
    particles_per_world: int
    x: wp.array[wp.vec3]
    previous: wp.array[wp.vec3]
    inv_dt: float
    epsilon: float


@wp.struct
class RollingContact:
    entries: int
    targets: vec6i
    # Gradient rows of the three angular directions (tangent 1, tangent 2, normal), one row per affine column.
    g0: mat63
    g1: mat63
    g2: mat63
    omega: wp.vec3
    coefficient: wp.vec3
    torque: wp.vec3


@wp.struct
class RollingInertia:
    valid: int
    b0: wp.vec3
    b1: wp.vec3
    b2: wp.vec3
    i0: wp.vec3
    i1: wp.vec3
    i2: wp.vec3


@wp.func
def rolling_inertia(ids: wp.vec4i, inertia: wp.array[float], base: int, x: wp.array[wp.vec3]):
    """B_k = A row k of J = tr(I)/2 I - I and the inverse rows of M = tr(W) I - W, W = A B^T (row-major input)."""
    r = RollingInertia()
    r.valid = 0
    a0 = x[ids[1]]
    a1 = x[ids[2]]
    a2 = x[ids[3]]
    half = ((inertia[base + 0] + inertia[base + 4]) + inertia[base + 8]) * 0.5
    j = wp.mat33(0.0)
    for row in range(3):
        for col in range(3):
            diagonal = half * 0.0
            if row == col:
                diagonal = half
            j[row, col] = diagonal - inertia[base + 3 * row + col]
    r.b0 = j[0, 2] * a2 + (j[0, 0] * a0 + a1 * j[0, 1])
    r.b1 = j[1, 2] * a2 + (j[1, 0] * a0 + a1 * j[1, 1])
    r.b2 = j[2, 2] * a2 + (j[2, 0] * a0 + a1 * j[2, 1])
    w = wp.outer(a2, r.b2) + (wp.outer(a0, r.b0) + wp.outer(a1, r.b1))
    trace = w[2, 2] + (w[0, 0] + w[1, 1])
    m = trace * wp.identity(n=3, dtype=float) - w
    c00 = m[1, 1] * m[2, 2] - m[2, 1] * m[1, 2]
    c01 = m[1, 2] * m[2, 0] - m[2, 2] * m[1, 0]
    c02 = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]
    det = m[0, 2] * c02 + (m[0, 0] * c00 + m[0, 1] * c01)
    if not (det > 1.0e-30) or not wp.isfinite(det):
        return r
    inverse_det = 1.0 / det
    r.i0 = wp.vec3(c00, c01, c02) * inverse_det
    r.i1 = (
        wp.vec3(
            m[2, 1] * m[0, 2] - m[2, 2] * m[0, 1],
            m[2, 2] * m[0, 0] - m[0, 2] * m[2, 0],
            m[2, 0] * m[0, 1] - m[0, 0] * m[2, 1],
        )
        * inverse_det
    )
    r.i2 = (
        wp.vec3(
            m[0, 1] * m[1, 2] - m[1, 1] * m[0, 2],
            m[0, 2] * m[1, 0] - m[0, 0] * m[1, 2],
            m[0, 0] * m[1, 1] - m[1, 0] * m[0, 1],
        )
        * inverse_det
    )
    r.valid = 1
    return r


@wp.func
def finite3(v: wp.vec3):
    return wp.isfinite(v[0]) and wp.isfinite(v[1]) and wp.isfinite(v[2])


@wp.func
def gradient_row(c: RollingContact, entry: int, d: int):
    if d == 0:
        return c.g0[entry]
    if d == 1:
        return c.g1[entry]
    return c.g2[entry]


@wp.func
def rolling_contact(v: RollingContactView, i: int):
    c = RollingContact()
    c.entries = 0
    if v.count.shape[0] == 0 or i >= v.capacity or i >= wp.max(v.count[0], 0):
        return c
    ok0, world0, s0 = contact_shape(v.shape0[i], v.local_shape_count, v.shapes_per_world, v.worlds)
    if ok0 == 0:
        return c
    ok1, world1, s1 = contact_shape(v.shape1[i], v.local_shape_count, v.shapes_per_world, v.worlds)
    if ok1 == 0:
        return c
    body0 = v.shape_body[s0]
    slot0 = contact_body_slot(body0, v.body_ids)
    if (body0 >= 0 and slot0 < 0) or (slot0 >= 0 and world0 < 0):
        return c
    body1 = v.shape_body[s1]
    slot1 = contact_body_slot(body1, v.body_ids)
    if (body1 >= 0 and slot1 < 0) or (slot1 >= 0 and world1 < 0):
        return c
    if slot0 == slot1:
        return c
    if world0 >= 0 and world1 >= 0 and world0 != world1:
        return c
    mu_r = float(0.0)
    if v.rolling.shape[0] > 0:
        mu_r = wp.max(v.rolling[s0], v.rolling[s1])
    mu_t = float(0.0)
    if v.torsional.shape[0] > 0:
        mu_t = wp.max(v.torsional[s0], v.torsional[s1])
    if not (mu_r > 0.0) and not (mu_t > 0.0):
        return c
    ids0 = wp.vec4i(0)
    ids1 = wp.vec4i(0)
    if slot0 >= 0:
        base0 = v.body_indices[slot0]
        o0 = world0 * v.particles_per_world
        ids0 = wp.vec4i(base0[0] + o0, base0[1] + o0, base0[2] + o0, base0[3] + o0)
    if slot1 >= 0:
        base1 = v.body_indices[slot1]
        o1 = world1 * v.particles_per_world
        ids1 = wp.vec4i(base1[0] + o1, base1[1] + o1, base1[2] + o1, base1[3] + o1)
    x = v.x
    n = contact_normalize(v.normal[i])
    sphere0 = v.shape_type[s0] == 3
    sphere1 = v.shape_type[s1] == 3
    raw0 = v.point0[i]
    if sphere0:
        raw0 = wp.transform_get_translation(v.transform[s0])
    raw1 = v.point1[i]
    if sphere1:
        raw1 = wp.transform_get_translation(v.transform[s1])
    wp0 = raw0
    if slot0 >= 0:
        wp0 = contact_point(raw0, ids0, v.body_com[slot0], x)
    wp1 = raw1
    if slot1 >= 0:
        wp1 = contact_point(raw1, ids1, v.body_com[slot1], x)
    candidate = wp.vec3(0.0)
    if sphere0 and sphere1:
        candidate = contact_normalize(wp1 - wp0)
    normal_squared = contact_dot_x_first(candidate, candidate)
    if normal_squared >= 0.5:
        n = candidate
    else:
        normal_squared = contact_dot(n, n)
    if not (normal_squared >= 0.5):
        return c
    adjusted0 = raw0
    if sphere0:
        radius0 = v.shape_scale[s0][0]
        if slot0 >= 0:
            adjusted0 = raw0 + contact_sphere_offset(radius0 * n, ids0, x)
        else:
            adjusted0 = radius0 * n + raw0
    if v.offset0.shape[0] > 0:
        adjusted0 = v.point0[i] + v.offset0[i]
    adjusted1 = raw1
    if sphere1:
        radius1 = v.shape_scale[s1][0]
        if slot1 >= 0:
            adjusted1 = raw1 + contact_sphere_offset(-(radius1 * n), ids1, x)
        else:
            adjusted1 = -radius1 * n + raw1
    if v.offset1.shape[0] > 0:
        adjusted1 = v.point1[i] + v.offset1[i]
    depth = (v.margin0[i] + v.margin1[i]) - contact_dot(wp1 - wp0, n)
    if not (depth > 0.0):
        return c
    velocity0 = wp.vec3(0.0)
    velocity1 = wp.vec3(0.0)
    radius = float(0.0)
    if slot0 >= 0:
        center = v.body_com[slot0]
        velocity0 = contact_point_velocity(raw0, ids0, center, x, v.previous, v.previous, v.inv_dt)
        lever = contact_surface_velocity(adjusted0 - center, ids0, x)
        length = wp.sqrt(contact_dot(lever, lever))
        if not (length > 0.0) or not wp.isfinite(length):
            return c
        radius = length
    if slot1 >= 0:
        center = v.body_com[slot1]
        velocity1 = contact_point_velocity(raw1, ids1, center, x, v.previous, v.previous, v.inv_dt)
        lever = contact_surface_velocity(adjusted1 - center, ids1, x)
        length = wp.sqrt(contact_dot(lever, lever))
        if not (length > 0.0) or not wp.isfinite(length):
            return c
        if radius > 0.0:
            radius = wp.min(radius, length)
        else:
            radius = length
    vn = contact_dot(velocity1 - velocity0, n)
    ke = wp.max(v.ke[s0], v.ke[s1])
    kd = wp.max(v.kd[s0], v.kd[s1])
    fn = wp.max(depth * ke - vn * kd, 0.0)
    tau_r = radius * (mu_r * fn)
    tau_t = radius * (mu_t * fn)
    if not (fn > 0.0) or not wp.isfinite(tau_r) or not wp.isfinite(tau_t):
        return c
    ax = float(0.0)
    ay = float(1.0)
    if wp.abs(n[0]) < 0.9:
        ax = 1.0
        ay = 0.0
    tangent1 = contact_normalize(wp.vec3(n[1] * 0.0 - ay * n[2], ax * n[2] - n[0] * 0.0, ay * n[0] - ax * n[1]))
    tangent2 = wp.vec3(
        -n[2] * tangent1[1] + n[1] * tangent1[2],
        n[2] * tangent1[0] - n[0] * tangent1[2],
        -n[1] * tangent1[0] + n[0] * tangent1[1],
    )
    entries = int(0)
    omega = wp.vec3(0.0)
    for side in range(2):
        slot = slot0
        ids = ids0
        sign = float(1.0)
        if side == 1:
            slot = slot1
            ids = ids1
            sign = -1.0
        if slot < 0:
            continue
        inertia = rolling_inertia(ids, v.inertia, 9 * slot, x)
        if inertia.valid == 0:
            return c
        for d in range(3):
            direction = n
            if d == 0:
                direction = tangent1
            elif d == 1:
                direction = tangent2
            u = sign * wp.vec3(
                contact_dot(inertia.i0, direction),
                contact_dot(inertia.i1, direction),
                contact_dot(inertia.i2, direction),
            )
            if not finite3(u):
                return c
            for k in range(3):
                column_b = inertia.b0
                if k == 1:
                    column_b = inertia.b1
                elif k == 2:
                    column_b = inertia.b2
                row = wp.cross(u, column_b)
                if d == 0:
                    c.g0[entries + k] = row
                elif d == 1:
                    c.g1[entries + k] = row
                else:
                    c.g2[entries + k] = row
        for k in range(3):
            particle = ids[k + 1]
            velocity = (x[particle] - v.previous[particle]) * v.inv_dt
            c.targets[entries] = particle
            omega = wp.vec3(
                contact_dot(velocity, c.g0[entries]) + omega[0],
                contact_dot(velocity, c.g1[entries]) + omega[1],
                contact_dot(velocity, c.g2[entries]) + omega[2],
            )
            entries += 1
    roll = wp.sqrt(omega[0] * omega[0] + omega[1] * omega[1])
    spin = wp.abs(omega[2])
    if not wp.isfinite(roll) or not wp.isfinite(spin):
        return c
    roll_coefficient = tau_r / wp.max(v.epsilon, roll)
    spin_coefficient = tau_t / wp.max(v.epsilon, spin)
    if not wp.isfinite(roll_coefficient) or not wp.isfinite(spin_coefficient):
        return c
    c.coefficient = wp.vec3(roll_coefficient, roll_coefficient, spin_coefficient)
    c.omega = omega
    scale = -wp.cw_mul(c.coefficient, omega)
    c.torque = scale[2] * n + (scale[0] * tangent1 + scale[1] * tangent2)
    c.entries = entries
    return c


@wp.func
def rolling_scatter(c: RollingContact, scale: wp.vec3, output: wp.array[wp.vec3]):
    for at in range(c.entries):
        wp.atomic_add(output, c.targets[at], scale[2] * c.g2[at] + (scale[0] * c.g0[at] + scale[1] * c.g1[at]))


@wp.kernel
def evaluate_rolling_contact_force(
    v: RollingContactView, force: wp.array[wp.vec3], contact_force: wp.array[wp.spatial_vector]
):
    i = wp.tid()
    c = rolling_contact(v, i)
    if contact_force.shape[0] > 0 and i < v.capacity:
        old = contact_force[i]
        contact_force[i] = wp.spatial_vector(wp.spatial_top(old), wp.spatial_bottom(old) + c.torque)
    if force.shape[0] == 0 or c.entries == 0:
        return
    rolling_scatter(c, -wp.cw_mul(c.coefficient, c.omega), force)


@wp.kernel
def add_rolling_contact_preconditioner(v: RollingContactView, blocks: wp.array[wp.mat33]):
    i = wp.tid()
    c = rolling_contact(v, i)
    if c.entries == 0:
        return
    sr = v.inv_dt * c.coefficient[0]
    st = v.inv_dt * c.coefficient[2]
    for at in range(c.entries):
        g0 = c.g0[at]
        g1 = c.g1[at]
        g2 = c.g2[at]
        block = st * wp.outer(g2, g2) + (sr * wp.outer(g0, g0) + sr * wp.outer(g1, g1))
        wp.atomic_add(blocks, c.targets[at], block)


@wp.kernel
def multiply_rolling_contact_tangent(v: RollingContactView, direction: wp.array[wp.vec3], product: wp.array[wp.vec3]):
    i = wp.tid()
    c = rolling_contact(v, i)
    if c.entries == 0:
        return
    total = wp.vec3(0.0)
    for at in range(c.entries):
        p = direction[c.targets[at]]
        total = wp.vec3(
            contact_dot(c.g0[at], p) + total[0],
            contact_dot(c.g1[at], p) + total[1],
            contact_dot(c.g2[at], p) + total[2],
        )
    scale = wp.vec3(
        (v.inv_dt * c.coefficient[0]) * total[0],
        (v.inv_dt * c.coefficient[1]) * total[1],
        (v.inv_dt * c.coefficient[2]) * total[2],
    )
    rolling_scatter(c, scale, product)

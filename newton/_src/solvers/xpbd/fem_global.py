# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental global compliant-constraint solve with a matrix-free operator.

Eliminate the multiplier increment from the XPBD KKT system, retaining the
primal residual after DAT. Solve (M + J.T alpha_tilde^-1 J) dx by PCG.
No VBD local vertex solver, assembled global matrix, or step-time CPU readback.
The five membrane rows are condensed into triangle-local blocks once per
nonlinear linearization; PCG never reuses blocks from a previous time step.
"""

import numpy as np
import warp as wp

from . import fem_contacts as collision
from . import fem_kernels as elastic


@wp.kernel
def contact_link_counts(
    count: wp.array[int], slots: wp.array[int], pairs: wp.array[wp.vec4i], workers: int, sizes: wp.array[int]
):
    for slot in range(wp.tid(), wp.min(count[0], slots.shape[0]), workers):
        c = slots[slot]
        v = pairs[c]
        for j in range(4):
            wp.atomic_add(sizes, v[j], 1)


@wp.kernel
def fill_contact_links(
    count: wp.array[int],
    slots: wp.array[int],
    pairs: wp.array[wp.vec4i],
    workers: int,
    offsets: wp.array[int],
    cursor: wp.array[int],
    links: wp.array[int],
):
    for slot in range(wp.tid(), wp.min(count[0], slots.shape[0]), workers):
        c = slots[slot]
        v = pairs[c]
        for j in range(4):
            k = wp.atomic_add(cursor, v[j], 1)
            links[offsets[v[j]] + k] = 4 * c + j


@wp.func
def store_row(
    row: int,
    v: wp.vec4i,
    g0: wp.vec3,
    g1: wp.vec3,
    g2: wp.vec3,
    g3: wp.vec3,
    value: float,
    stiffness: float,
    damping: float,
    dt: float,
    pos: wp.array[wp.vec3],
    prev: wp.array[wp.vec3],
    gradients: wp.array[wp.vec3],
    weights: wp.array[float],
    bias: wp.array[float],
):
    motion = wp.dot(g0, pos[v[0]] - prev[v[0]]) + wp.dot(g1, pos[v[1]] - prev[v[1]])
    motion += wp.dot(g2, pos[v[2]] - prev[v[2]]) + wp.dot(g3, pos[v[3]] - prev[v[3]])
    gamma = wp.max(damping, 0.0) / dt
    weights[row] = dt * dt * stiffness * (1.0 + gamma)
    bias[row] = dt * dt * stiffness * (value + gamma * motion)
    gradients[4 * row] = g0
    gradients[4 * row + 1] = g1
    gradients[4 * row + 2] = g2
    gradients[4 * row + 3] = g3


@wp.kernel
def build_elastic(
    pos: wp.array[wp.vec3],
    prev: wp.array[wp.vec3],
    triangles: wp.array2d[int],
    poses: wp.array[wp.mat22],
    areas: wp.array[float],
    materials: wp.array2d[float],
    edges: wp.array2d[int],
    angles: wp.array[float],
    bending: wp.array2d[float],
    dt: float,
    gradients: wp.array[wp.vec3],
    weights: wp.array[float],
    bias: wp.array[float],
):
    e = wp.tid()
    if e < triangles.shape[0]:
        i, j, k = int(triangles[e, 0]), int(triangles[e, 1]), int(triangles[e, 2])
        a, b = pos[j] - pos[i], pos[k] - pos[i]
        q = poses[e]
        f0, f1 = q[0, 0] * a + q[1, 0] * b, q[0, 1] * a + q[1, 1] * b
        n = wp.cross(f0, f1)
        jac = wp.length(n)
        aa, bb, ab = wp.length_sq(f0), wp.length_sq(f1), wp.dot(f0, f1)
        d0, d1, d2 = wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)
        s0, s1, s2 = wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)
        v0, v1, v2 = wp.vec3(0.0), wp.vec3(0.0), wp.vec3(0.0)
        dev = float(0.0)
        shear = float(0.0)
        if jac > 1e-12:
            n /= jac
            h0, h1 = wp.cross(f1, n), wp.cross(n, f0)
            denominator = wp.sqrt(aa + bb + 2.0 * jac)
            dev, shear = (aa - bb) / denominator, 2.0 * ab / denominator
            v1, v2 = q[0, 0] * h0 + q[0, 1] * h1, q[1, 0] * h0 + q[1, 1] * h1
            v0 = -v1 - v2
            z0, z1 = (f0 + h0) / (denominator * denominator), (f1 + h1) / (denominator * denominator)
            h0, h1 = 2.0 * f0 / denominator - dev * z0, -2.0 * f1 / denominator - dev * z1
            d1, d2 = q[0, 0] * h0 + q[0, 1] * h1, q[1, 0] * h0 + q[1, 1] * h1
            d0 = -d1 - d2
            h0, h1 = 2.0 * f1 / denominator - shear * z0, 2.0 * f0 / denominator - shear * z1
            s1, s2 = q[0, 0] * h0 + q[0, 1] * h1, q[1, 0] * h0 + q[1, 1] * h1
            s0 = -s1 - s2
        ids = wp.vec4i(i, j, k, i)
        store_row(
            5 * e,
            ids,
            d0,
            d1,
            d2,
            wp.vec3(0.0),
            dev,
            areas[e] * materials[e, 0],
            materials[e, 2],
            dt,
            pos,
            prev,
            gradients,
            weights,
            bias,
        )
        store_row(
            5 * e + 1,
            ids,
            s0,
            s1,
            s2,
            wp.vec3(0.0),
            shear,
            areas[e] * materials[e, 0],
            materials[e, 2],
            dt,
            pos,
            prev,
            gradients,
            weights,
            bias,
        )
        store_row(
            5 * e + 2,
            ids,
            v0,
            v1,
            v2,
            wp.vec3(0.0),
            jac - 1.0,
            areas[e] * (materials[e, 0] + materials[e, 1]),
            materials[e, 2],
            dt,
            pos,
            prev,
            gradients,
            weights,
            bias,
        )
        # The strain Jacobian has no normal component. Add the PSD normal
        # geometric Hessian of the same energy, not additional elastic forces.
        h00, h01, h11 = float(0.0), float(0.0), float(0.0)
        if jac > 1e-12:
            mu, bulk = materials[e, 0], materials[e, 0] + materials[e, 1]
            factor = (bulk * (jac - 1.0) - mu) / jac
            h00, h01, h11 = mu + factor * bb, -factor * ab, mu + factor * aa
        spread = wp.sqrt(0.25 * (h00 - h11) * (h00 - h11) + h01 * h01)
        high, low = 0.5 * (h00 + h11) + spread, 0.5 * (h00 + h11) - spread
        eigen_axis = wp.vec2(1.0, 0.0)
        if wp.abs(h01) > 1e-8 * wp.max(wp.abs(high), 1.0):
            eigen_axis = wp.normalize(wp.vec2(h01, high - h00))
        elif h11 > h00:
            eigen_axis = wp.vec2(0.0, 1.0)
        for mode in range(2):
            eigenvalue = high
            direction = eigen_axis
            if mode == 1:
                eigenvalue = low
                direction = wp.vec2(-eigen_axis[1], eigen_axis[0])
            g1 = n * (q[0, 0] * direction[0] + q[0, 1] * direction[1])
            g2 = n * (q[1, 0] * direction[0] + q[1, 1] * direction[1])
            store_row(
                5 * e + 3 + mode,
                ids,
                -g1 - g2,
                g1,
                g2,
                wp.vec3(0.0),
                0.0,
                areas[e] * wp.max(eigenvalue, 0.0),
                materials[e, 2],
                dt,
                pos,
                prev,
                gradients,
                weights,
                bias,
            )
    if e < edges.shape[0]:
        row = 5 * triangles.shape[0] + e
        weights[row] = 0.0
        bias[row] = 0.0
        for j in range(4):
            gradients[4 * row + j] = wp.vec3(0.0)
        i, j, k, l = int(edges[e, 0]), int(edges[e, 1]), int(edges[e, 2]), int(edges[e, 3])
        if i >= 0 and j >= 0 and bending[e, 0] > 0.0:
            x0, x1, x2, x3 = pos[i], pos[j], pos[k], pos[l]
            a, b, c, d, axis = x2 - x0, x3 - x0, x2 - x1, x3 - x1, x3 - x2
            length = wp.length(axis)
            n0, n1 = wp.cross(a, b), wp.cross(d, c)
            l0, l1 = wp.length(n0), wp.length(n1)
            if wp.min(length, wp.min(l0, l1)) > 1e-12:
                n0 /= l0
                n1 /= l1
                axis /= length
                value = wp.atan2(wp.dot(wp.cross(n0, n1), axis), wp.dot(n0, n1)) - angles[e]
                if value > wp.pi:
                    value -= 2.0 * wp.pi
                elif value < -wp.pi:
                    value += 2.0 * wp.pi
                g0, g1 = -length * n0 / l0, -length * n1 / l1
                g2 = wp.dot(b, axis) * n0 / l0 + wp.dot(d, axis) * n1 / l1
                g3 = -wp.dot(a, axis) * n0 / l0 - wp.dot(c, axis) * n1 / l1
                store_row(
                    row,
                    wp.vec4i(i, j, k, l),
                    g0,
                    g1,
                    g2,
                    g3,
                    value,
                    bending[e, 0],
                    bending[e, 1],
                    dt,
                    pos,
                    prev,
                    gradients,
                    weights,
                    bias,
                )


@wp.kernel
def build_contacts(
    pos: wp.array[wp.vec3],
    prev: wp.array[wp.vec3],
    count: wp.array[int],
    pairs: wp.array[wp.vec4i],
    kinds: wp.array[int],
    radius: float,
    stiffness: float,
    damping: float,
    friction: float,
    dt: float,
    workers: int,
    bary: wp.array[wp.vec4],
    matrices: wp.array[wp.mat33],
    bias: wp.array[wp.vec3],
    viscous: wp.array[wp.mat33],
    active_count: wp.array[int],
    active_slots: wp.array[int],
):
    for c in range(wp.tid(), wp.min(count[0], pairs.shape[0]), workers):
        ids = pairs[c]
        bary[c] = wp.vec4(0.0)
        matrices[c] = wp.mat33(0.0)
        bias[c] = wp.vec3(0.0)
        viscous[c] = wp.mat33(0.0)
        difference, b = collision.contact_geometry(pos, ids, kinds[c])
        bary[c] = b
        distance = wp.length(difference)
        if distance > 1e-12 and distance < radius:
            slot = wp.atomic_add(active_count, 0, 1)
            active_slots[slot] = c
            n = difference / distance
            motion = wp.vec3(0.0)
            for j in range(4):
                motion += b[j] * (pos[ids[j]] - prev[ids[j]])
            force = stiffness * (radius - distance)
            curvature = stiffness
            tau = 0.5 * radius
            minimum = wp.min(1e-5, 0.1 * tau)
            if distance < tau:
                if distance > minimum:
                    force = stiffness * tau * tau / distance
                    curvature = force / distance
                else:
                    curvature = stiffness * tau * tau / (minimum * minimum)
                    force = curvature * (2.0 * minimum - distance)
            normal_motion = wp.dot(n, motion)
            kd = float(0.0)
            if normal_motion < -1e-6 * dt:
                kd = damping / dt
            tangent = motion - normal_motion * n
            kt = friction * force / wp.max(wp.length(tangent), 0.01 * dt)
            nn = wp.outer(n, n)
            matrices[c] = dt * dt * ((curvature + kd) * nn + kt * (wp.identity(n=3, dtype=float) - nn))
            bias[c] = dt * dt * ((-force + kd * normal_motion) * n + kt * tangent)
            viscous[c] = dt * dt * (kd * nn + kt * (wp.identity(n=3, dtype=float) - nn))


@wp.kernel
def initialize_system(
    pos: wp.array[wp.vec3],
    target: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    offsets: wp.array[int],
    links: wp.array[int],
    element_diagonal: wp.array[wp.mat33],
    element_bias: wp.array[wp.vec3],
    contact_offsets: wp.array[int],
    contact_links: wp.array[int],
    bary: wp.array[wp.vec4],
    matrices: wp.array[wp.mat33],
    contact_bias: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
    r: wp.array[wp.vec3],
    direction: wp.array[wp.vec3],
    solution: wp.array[wp.vec3],
    dots: wp.array[wp.float64],
):
    i = wp.tid()
    w = elastic.effective_mass(i, inv_mass, flags)
    rhs, diag = wp.vec3(0.0), wp.identity(n=3, dtype=float)
    if w > 0.0:
        mass = 1.0 / w
        rhs = -mass * (pos[i] - target[i])
        diag = mass * wp.identity(n=3, dtype=float)
        for k in range(offsets[i], offsets[i + 1]):
            entry = links[k]
            rhs -= element_bias[entry]
            diag += element_diagonal[entry]
        for k in range(contact_offsets[i], contact_offsets[i + 1]):
            entry = contact_links[k]
            row = entry // 4
            b = bary[row][entry % 4]
            a = matrices[row]
            rhs -= b * contact_bias[row]
            diag += b * b * a
    scale = wp.max(diag[0, 0], wp.max(diag[1, 1], diag[2, 2]))
    inverse = wp.mat33(1.0 / diag[0, 0], 0.0, 0.0, 0.0, 1.0 / diag[1, 1], 0.0, 0.0, 0.0, 1.0 / diag[2, 2])
    normalized = diag / scale
    if wp.determinant(normalized) > 1e-20:
        inverse = wp.inverse(normalized) / scale
    z = inverse * rhs
    diagonal[i] = inverse
    r[i] = rhs
    direction[i] = z
    solution[i] = wp.vec3(0.0)
    wp.tile_atomic_add(dots, wp.tile_sum(wp.tile(wp.float64(wp.dot(rhs, z)))), offset=0)


@wp.kernel
def multiply_rows(
    direction: wp.array[wp.vec3],
    ids: wp.array[wp.vec4i],
    gradients: wp.array[wp.vec3],
    weights: wp.array[float],
    count: wp.array[int],
    slots: wp.array[int],
    pairs: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    matrices: wp.array[wp.mat33],
    workers: int,
    values: wp.array[float],
    contact_values: wp.array[wp.vec3],
):
    for row in range(wp.tid(), ids.shape[0], workers):
        v = ids[row]
        value = float(0.0)
        for j in range(4):
            value += wp.dot(gradients[4 * row + j], direction[v[j]])
        values[row] = weights[row] * value
    for slot in range(wp.tid(), wp.min(count[0], slots.shape[0]), workers):
        row = slots[slot]
        v, b = wp.vec4i(pairs[row]), wp.vec4(bary[row])
        contact_value = wp.vec3(0.0)
        for j in range(4):
            contact_value += b[j] * direction[v[j]]
        contact_values[row] = matrices[row] * contact_value


@wp.kernel
def build_element_blocks(
    gradients: wp.array[wp.vec3],
    weights: wp.array[float],
    bias: wp.array[float],
    triangles: int,
    blocks: wp.array[wp.mat33],
    diagonal: wp.array[wp.mat33],
    forces: wp.array[wp.vec3],
):
    """Cache the exact row sum, including damping and normal geometric stiffness."""
    t = wp.tid()
    if t < triangles:
        # Every triangle row has g0 = -g1-g2. Three 3x3 blocks represent its
        # 6x6 operator on relative displacements (dx1-dx0, dx2-dx0), with no
        # low-rank approximation or change to the compliant constraints.
        a, b, c = wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)
        f1, f2 = wp.vec3(0.0), wp.vec3(0.0)
        for row in range(5 * t, 5 * t + 5):
            g1, g2 = gradients[4 * row + 1], gradients[4 * row + 2]
            a += weights[row] * wp.outer(g1, g1)
            b += weights[row] * wp.outer(g1, g2)
            c += weights[row] * wp.outer(g2, g2)
            f1 += bias[row] * g1
            f2 += bias[row] * g2
        blocks[3 * t] = a
        blocks[3 * t + 1] = b
        blocks[3 * t + 2] = c
        diagonal[3 * t] = a + b + wp.transpose(b) + c
        diagonal[3 * t + 1] = a
        diagonal[3 * t + 2] = c
        forces[3 * t] = -f1 - f2
        forces[3 * t + 1] = f1
        forces[3 * t + 2] = f2
    row = 5 * triangles + t
    if row < weights.shape[0]:
        for j in range(4):
            g = gradients[4 * row + j]
            entry = 3 * triangles + 4 * t + j
            diagonal[entry] = weights[row] * wp.outer(g, g)
            forces[entry] = bias[row] * g


@wp.kernel
def multiply_elements(
    direction: wp.array[wp.vec3],
    triangles: wp.array2d[int],
    ids: wp.array[wp.vec4i],
    gradients: wp.array[wp.vec3],
    weights: wp.array[float],
    blocks: wp.array[wp.mat33],
    count: wp.array[int],
    slots: wp.array[int],
    pairs: wp.array[wp.vec4i],
    bary: wp.array[wp.vec4],
    matrices: wp.array[wp.mat33],
    workers: int,
    values: wp.array[wp.vec3],
    contact_values: wp.array[wp.vec3],
):
    nt = triangles.shape[0]
    for t in range(wp.tid(), nt, workers):
        i, j, k = int(triangles[t, 0]), int(triangles[t, 1]), int(triangles[t, 2])
        d1, d2 = direction[j] - direction[i], direction[k] - direction[i]
        h11, h12, h22 = blocks[3 * t], blocks[3 * t + 1], blocks[3 * t + 2]
        v1, v2 = h11 * d1 + h12 * d2, wp.transpose(h12) * d1 + h22 * d2
        values[3 * t] = -v1 - v2
        values[3 * t + 1] = v1
        values[3 * t + 2] = v2
    for row in range(5 * nt + wp.tid(), ids.shape[0], workers):
        v = ids[row]
        value = float(0.0)
        for j in range(4):
            value += wp.dot(gradients[4 * row + j], direction[v[j]])
        for j in range(4):
            values[3 * nt + 4 * (row - 5 * nt) + j] = weights[row] * value * gradients[4 * row + j]
    for slot in range(wp.tid(), wp.min(count[0], slots.shape[0]), workers):
        row = slots[slot]
        v, b = wp.vec4i(pairs[row]), wp.vec4(bary[row])
        value_contact = wp.vec3(0.0)
        for j in range(4):
            value_contact += b[j] * direction[v[j]]
        contact_values[row] = matrices[row] * value_contact


@wp.kernel
def gather_element_product(
    direction: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    offsets: wp.array[int],
    links: wp.array[int],
    values: wp.array[wp.vec3],
    contact_offsets: wp.array[int],
    contact_links: wp.array[int],
    bary: wp.array[wp.vec4],
    contact_values: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    dots: wp.array[wp.float64],
):
    i = wp.tid()
    w = elastic.effective_mass(i, inv_mass, flags)
    value = wp.vec3(0.0)
    if w > 0.0:
        value = direction[i] / w
        for k in range(offsets[i], offsets[i + 1]):
            value += values[links[k]]
        for k in range(contact_offsets[i], contact_offsets[i + 1]):
            entry = contact_links[k]
            value += bary[entry // 4][entry % 4] * contact_values[entry // 4]
    product[i] = value
    wp.tile_atomic_add(dots, wp.tile_sum(wp.tile(wp.float64(wp.dot(direction[i], value)))), offset=2)


@wp.kernel
def gather_product(
    direction: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    offsets: wp.array[int],
    links: wp.array[int],
    gradients: wp.array[wp.vec3],
    values: wp.array[float],
    contact_offsets: wp.array[int],
    contact_links: wp.array[int],
    bary: wp.array[wp.vec4],
    contact_values: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    dots: wp.array[wp.float64],
):
    i = wp.tid()
    w = elastic.effective_mass(i, inv_mass, flags)
    value = wp.vec3(0.0)
    if w > 0.0:
        value = direction[i] / w
        for k in range(offsets[i], offsets[i + 1]):
            entry = links[k]
            value += gradients[entry] * values[entry // 4]
        for k in range(contact_offsets[i], contact_offsets[i + 1]):
            entry = contact_links[k]
            value += bary[entry // 4][entry % 4] * contact_values[entry // 4]
    product[i] = value
    wp.tile_atomic_add(dots, wp.tile_sum(wp.tile(wp.float64(wp.dot(direction[i], value)))), offset=2)


@wp.kernel
def pcg_update(
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
    old_slot: int,
    new_slot: int,
    dots: wp.array[wp.float64],
    r: wp.array[wp.vec3],
    solution: wp.array[wp.vec3],
):
    i = wp.tid()
    alpha = float(0.0)
    if dots[2] > wp.float64(1e-30):
        alpha = float(dots[old_slot] / dots[2])
    residual = r[i] - alpha * product[i]
    r[i] = residual
    solution[i] += alpha * direction[i]
    wp.tile_atomic_add(
        dots, wp.tile_sum(wp.tile(wp.float64(wp.dot(residual, diagonal[i] * residual)))), offset=new_slot
    )


@wp.kernel
def pcg_direction(
    r: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
    old_slot: int,
    new_slot: int,
    dots: wp.array[wp.float64],
    direction: wp.array[wp.vec3],
):
    i = wp.tid()
    beta = float(0.0)
    if dots[old_slot] > wp.float64(1e-30):
        beta = float(dots[new_slot] / dots[old_slot])
    direction[i] = diagonal[i] * r[i] + beta * direction[i]


@wp.kernel
def clear_dot_slots(new_slot: int, dots: wp.array[wp.float64]):
    dots[new_slot] = wp.float64(0.0)
    dots[2] = wp.float64(0.0)


@wp.kernel
def make_candidate(pos: wp.array[wp.vec3], solution: wp.array[wp.vec3], candidate: wp.array[wp.vec3]):
    i = wp.tid()
    candidate[i] = pos[i] + solution[i]


@wp.kernel
def merit_energy(
    pos: wp.array[wp.vec3],
    prev: wp.array[wp.vec3],
    target: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    triangles: wp.array2d[int],
    poses: wp.array[wp.mat22],
    areas: wp.array[float],
    materials: wp.array2d[float],
    edges: wp.array2d[int],
    angles: wp.array[float],
    bending: wp.array2d[float],
    row_ids: wp.array[wp.vec4i],
    gradients: wp.array[wp.vec3],
    weights: wp.array[float],
    count: wp.array[int],
    pairs: wp.array[wp.vec4i],
    kinds: wp.array[int],
    bary: wp.array[wp.vec4],
    viscous: wp.array[wp.mat33],
    radius: float,
    stiffness: float,
    dt: float,
    workers: int,
    slot: int,
    active: wp.array[int],
    energy: wp.array[wp.float64],
):
    if active[0] == 0:
        return
    total = wp.float64(0.0)
    for i in range(wp.tid(), pos.shape[0], workers):
        w = elastic.effective_mass(i, inv_mass, flags)
        if w > 0.0:
            total += wp.float64(0.5 * wp.length_sq(pos[i] - target[i]) / w)
    for t in range(wp.tid(), triangles.shape[0], workers):
        i, j, k = int(triangles[t, 0]), int(triangles[t, 1]), int(triangles[t, 2])
        a, b = pos[j] - pos[i], pos[k] - pos[i]
        q = poses[t]
        f0, f1 = q[0, 0] * a + q[1, 0] * b, q[0, 1] * a + q[1, 1] * b
        jac = wp.length(wp.cross(f0, f1))
        invariant = wp.length_sq(f0) + wp.length_sq(f1)
        value = areas[t] * (
            materials[t, 0] * wp.max(invariant - 2.0 * jac, 0.0)
            + (materials[t, 0] + materials[t, 1]) * (jac - 1.0) * (jac - 1.0)
        )
        for row in range(5 * t, 5 * t + 5):
            v = wp.vec4i(row_ids[row])
            motion = float(0.0)
            for j in range(4):
                motion += wp.dot(gradients[4 * row + j], pos[v[j]] - prev[v[j]])
            gamma = wp.max(materials[t, 2], 0.0) / dt
            total += wp.float64(0.5 * weights[row] * gamma / (1.0 + gamma) * motion * motion)
        total += wp.float64(0.5 * dt * dt * value)
    for e in range(wp.tid(), edges.shape[0], workers):
        v = wp.vec4i(row_ids[5 * triangles.shape[0] + e])
        if edges[e, 0] >= 0 and edges[e, 1] >= 0 and bending[e, 0] > 0.0:
            x0, x1, x2, x3 = pos[v[0]], pos[v[1]], pos[v[2]], pos[v[3]]
            normal0 = wp.normalize(wp.cross(x2 - x0, x3 - x0))
            normal1 = wp.normalize(wp.cross(x3 - x1, x2 - x1))
            axis = wp.normalize(x3 - x2)
            angle = wp.atan2(wp.dot(wp.cross(normal0, normal1), axis), wp.dot(normal0, normal1)) - angles[e]
            if angle > wp.pi:
                angle -= 2.0 * wp.pi
            elif angle < -wp.pi:
                angle += 2.0 * wp.pi
            motion = float(0.0)
            for j in range(4):
                motion += wp.dot(gradients[4 * (5 * triangles.shape[0] + e) + j], pos[v[j]] - prev[v[j]])
            total += wp.float64(0.5 * dt * dt * bending[e, 0] * (angle * angle + bending[e, 1] / dt * motion * motion))
    for c in range(wp.tid(), wp.min(count[0], pairs.shape[0]), workers):
        v = wp.vec4i(pairs[c])
        difference, _b = collision.contact_geometry(pos, v, kinds[c])
        distance = wp.length(difference)
        value = float(0.0)
        if distance < radius:
            tau = 0.5 * radius
            minimum = wp.min(1e-5, 0.1 * tau)
            value = 0.5 * stiffness * (radius - distance) * (radius - distance)
            if distance < tau:
                value = stiffness * tau * tau * (0.5 - wp.log(wp.max(distance, minimum) / tau))
                if distance < minimum:
                    x = distance - minimum
                    value += stiffness * tau * tau * (-x / minimum + 0.5 * x * x / (minimum * minimum))
        contact_motion = wp.vec3(0.0)
        for j in range(4):
            contact_motion += bary[c][j] * (pos[v[j]] - prev[v[j]])
        total += wp.float64(dt * dt * value + 0.5 * wp.dot(contact_motion, viscous[c] * contact_motion))
    wp.tile_atomic_add(energy, wp.tile_sum(wp.tile(total)), offset=slot)


@wp.kernel
def begin_backtrack(energy: wp.array[wp.float64], active: wp.array[int]):
    active[0] = 0
    if energy[1] > energy[0] + wp.float64(1e-12) or not wp.isfinite(energy[1]):
        active[0] = 1
        energy[1] = wp.float64(0.0)


@wp.kernel
def shorten_step(base: wp.array[wp.vec3], active: wp.array[int], pos: wp.array[wp.vec3]):
    i = wp.tid()
    if active[0] != 0:
        pos[i] = wp.vec3(wp.float64(0.5) * (wp.vec3d(base[i]) + wp.vec3d(pos[i])))


@wp.kernel
def reject_step(base: wp.array[wp.vec3], energy: wp.array[wp.float64], pos: wp.array[wp.vec3]):
    i = wp.tid()
    if energy[1] > energy[0] + wp.float64(1e-12) or not wp.isfinite(energy[1]):
        pos[i] = base[i]


class GlobalConstraints:
    """Own fixed-capacity sparse incidence and matrix-free XPBD workspaces."""

    def __init__(self, fem, cg_iterations=4):
        if isinstance(cg_iterations, bool) or not isinstance(cg_iterations, int) or cg_iterations < 1:
            raise ValueError("particle_fem_linear_iterations must be a positive integer")
        self.fem = fem
        model = fem.model
        self.cg_iterations = cg_iterations
        self.workers = 4096
        faces = model.tri_indices.numpy()
        edges = model.edge_indices.numpy()
        ids = []
        for a, b, c in faces:
            ids.extend(((a, b, c, a),) * 5)
        ids.extend(tuple(row) if min(row) >= 0 else (0, 0, 0, 0) for row in edges)
        ids = np.asarray(ids, dtype=np.int32)
        links = [[] for _ in range(model.particle_count)]
        for row, v in enumerate(ids):
            for j, i in enumerate(v):
                links[i].append(4 * row + j)
        self.ids = wp.array(ids, dtype=wp.vec4i, device=model.device)
        # Retain the row CSR for the independent reference operator used by
        # diagnostics. The runtime PCG and initialization use the element CSR.
        self.offsets = wp.array(np.cumsum([0] + [len(row) for row in links]), dtype=int, device=model.device)
        self.links = wp.array([k for row in links for k in row], dtype=int, device=model.device)
        element_links = [[] for _ in range(model.particle_count)]
        for t, face in enumerate(faces):
            for j, i in enumerate(face):
                element_links[i].append(3 * t + j)
        for e, v in enumerate(ids[5 * len(faces) :]):
            for j, i in enumerate(v):
                element_links[i].append(3 * len(faces) + 4 * e + j)
        self.element_offsets = wp.array(
            np.cumsum([0] + [len(row) for row in element_links]), dtype=int, device=model.device
        )
        self.element_links = wp.array([k for row in element_links for k in row], dtype=int, device=model.device)
        self.triangle_blocks = wp.empty(3 * len(faces), dtype=wp.mat33, device=model.device)
        self.element_values = wp.empty(3 * len(faces) + 4 * len(edges), dtype=wp.vec3, device=model.device)
        self.element_bias = wp.empty_like(self.element_values)
        self.element_diagonal = wp.empty(len(self.element_values), dtype=wp.mat33, device=model.device)
        self.gradients = wp.empty(4 * len(ids), dtype=wp.vec3, device=model.device)
        self.weights = wp.empty(len(ids), device=model.device)
        self.bias = wp.empty_like(self.weights)
        self.values = wp.empty_like(self.weights)
        capacity = len(fem.pairs)
        self.bary = wp.empty(capacity, dtype=wp.vec4, device=model.device)
        self.matrices = wp.empty(capacity, dtype=wp.mat33, device=model.device)
        self.contact_bias = wp.empty(capacity, dtype=wp.vec3, device=model.device)
        self.contact_values = wp.empty_like(self.contact_bias)
        self.viscous = wp.empty_like(self.matrices)
        self.contact_offsets = wp.zeros(model.particle_count + 1, dtype=int, device=model.device)
        self.cursor = wp.zeros_like(self.contact_offsets)
        self.contact_links = wp.empty(4 * capacity, dtype=int, device=model.device)
        self.active_count = wp.zeros(1, dtype=int, device=model.device)
        self.active_slots = wp.empty(capacity, dtype=int, device=model.device)
        self.target = wp.empty_like(fem.pos)
        self.diagonal = wp.empty(model.particle_count, dtype=wp.mat33, device=model.device)
        self.r = wp.empty_like(fem.pos)
        self.direction = wp.empty_like(fem.pos)
        self.product = wp.empty_like(fem.pos)
        self.solution = wp.empty_like(fem.pos)
        self.dots = wp.zeros(3, dtype=wp.float64, device=model.device)
        self.backup = wp.empty_like(fem.pos)
        self.energy = wp.zeros(2, dtype=wp.float64, device=model.device)
        self.backtrack_active = wp.ones(1, dtype=int, device=model.device)

    def _energy(self, dt, slot):
        f, m = self.fem, self.fem.model
        self._launch(
            merit_energy,
            self.workers,
            [
                f.pos,
                f.prev,
                self.target,
                m.particle_inv_mass,
                m.particle_flags,
                m.tri_indices,
                m.tri_poses,
                m.tri_areas,
                m.tri_materials,
                m.edge_indices,
                m.edge_rest_angle,
                m.edge_bending_properties,
                self.ids,
                self.gradients,
                self.weights,
                f.count,
                f.pairs,
                f.kinds,
                self.bary,
                self.viscous,
                f.radius,
                m.soft_contact_ke,
                dt,
                self.workers,
                slot,
                self.backtrack_active,
                self.energy,
            ],
        )

    def step(self, state_in, state_out, dt):
        f = self.fem
        m = f.model
        n = m.particle_count
        launch = self._launch
        wp.copy(f.pos, state_in.particle_q)
        wp.copy(f.velocity, state_in.particle_qd)
        wp.copy(f.prev, f.pos)
        if f.radius > 0.0:
            f._detect()
        else:
            wp.copy(f.base, f.pos)
        launch(
            elastic.predict,
            n,
            [
                f.pos,
                f.velocity,
                state_in.particle_f,
                m.particle_inv_mass,
                m.particle_flags,
                m.particle_world,
                m.gravity,
                f.targets,
                1,
                dt,
                f.candidate,
            ],
        )
        wp.copy(self.target, f.candidate)
        f._commit()
        for _ in range(f.iterations):
            launch(
                build_elastic,
                max(m.tri_count, m.edge_count),
                [
                    f.pos,
                    f.prev,
                    m.tri_indices,
                    m.tri_poses,
                    m.tri_areas,
                    m.tri_materials,
                    m.edge_indices,
                    m.edge_rest_angle,
                    m.edge_bending_properties,
                    dt,
                    self.gradients,
                    self.weights,
                    self.bias,
                ],
            )
            self.active_count.zero_()
            launch(
                build_contacts,
                self.workers,
                [
                    f.pos,
                    f.prev,
                    f.count,
                    f.pairs,
                    f.kinds,
                    f.radius,
                    m.soft_contact_ke,
                    m.soft_contact_kd,
                    m.soft_contact_mu,
                    dt,
                    self.workers,
                    self.bary,
                    self.matrices,
                    self.contact_bias,
                    self.viscous,
                    self.active_count,
                    self.active_slots,
                ],
            )
            self.cursor.zero_()
            launch(
                contact_link_counts,
                self.workers,
                [self.active_count, self.active_slots, f.pairs, self.workers, self.cursor],
            )
            wp.utils.array_scan(self.cursor, self.contact_offsets, inclusive=False)
            self.cursor.zero_()
            launch(
                fill_contact_links,
                self.workers,
                [
                    self.active_count,
                    self.active_slots,
                    f.pairs,
                    self.workers,
                    self.contact_offsets,
                    self.cursor,
                    self.contact_links,
                ],
            )
            self.dots.zero_()
            launch(
                build_element_blocks,
                max(m.tri_count, m.edge_count),
                [
                    self.gradients,
                    self.weights,
                    self.bias,
                    m.tri_count,
                    self.triangle_blocks,
                    self.element_diagonal,
                    self.element_bias,
                ],
            )
            launch(
                initialize_system,
                n,
                [
                    f.pos,
                    self.target,
                    m.particle_inv_mass,
                    m.particle_flags,
                    self.element_offsets,
                    self.element_links,
                    self.element_diagonal,
                    self.element_bias,
                    self.contact_offsets,
                    self.contact_links,
                    self.bary,
                    self.matrices,
                    self.contact_bias,
                    self.diagonal,
                    self.r,
                    self.direction,
                    self.solution,
                    self.dots,
                ],
            )
            wp.copy(self.backup, f.pos)
            self.energy.zero_()
            self.backtrack_active.fill_(1)
            self._energy(dt, 0)
            for iteration in range(self.cg_iterations):
                old, new = iteration % 2, 1 - iteration % 2
                launch(clear_dot_slots, 1, [new, self.dots])
                self._multiply()
                launch(
                    pcg_update,
                    n,
                    [self.direction, self.product, self.diagonal, old, new, self.dots, self.r, self.solution],
                )
                if iteration + 1 < self.cg_iterations:
                    launch(pcg_direction, n, [self.r, self.diagonal, old, new, self.dots, self.direction])
            launch(make_candidate, n, [f.pos, self.solution, f.candidate])
            f._commit()
            self._energy(dt, 1)
            for _ in range(4):
                launch(begin_backtrack, 1, [self.energy, self.backtrack_active])
                launch(shorten_step, n, [self.backup, self.backtrack_active, f.pos])
                self._energy(dt, 1)
            launch(reject_step, n, [self.backup, self.energy, f.pos])
        launch(elastic.update_velocity, n, [f.prev, f.pos, dt, f.velocity])
        wp.copy(state_out.particle_q, f.pos)
        wp.copy(state_out.particle_qd, f.velocity)

    def _multiply(self):
        """Apply the current condensed element and active-contact operator."""
        f, m = self.fem, self.fem.model
        launch, n = self._launch, m.particle_count
        launch(
            multiply_elements,
            self.workers,
            [
                self.direction,
                m.tri_indices,
                self.ids,
                self.gradients,
                self.weights,
                self.triangle_blocks,
                self.active_count,
                self.active_slots,
                f.pairs,
                self.bary,
                self.matrices,
                self.workers,
                self.element_values,
                self.contact_values,
            ],
        )
        launch(
            gather_element_product,
            n,
            [
                self.direction,
                m.particle_inv_mass,
                m.particle_flags,
                self.element_offsets,
                self.element_links,
                self.element_values,
                self.contact_offsets,
                self.contact_links,
                self.bary,
                self.contact_values,
                self.product,
                self.dots,
            ],
        )

    def _multiply_rows(self):
        """Apply the uncondensed reference operator for numerical diagnostics only."""
        f, m = self.fem, self.fem.model
        launch, n = self._launch, m.particle_count
        launch(
            multiply_rows,
            self.workers,
            [
                self.direction,
                self.ids,
                self.gradients,
                self.weights,
                self.active_count,
                self.active_slots,
                f.pairs,
                self.bary,
                self.matrices,
                self.workers,
                self.values,
                self.contact_values,
            ],
        )
        launch(
            gather_product,
            n,
            [
                self.direction,
                m.particle_inv_mass,
                m.particle_flags,
                self.offsets,
                self.links,
                self.gradients,
                self.values,
                self.contact_offsets,
                self.contact_links,
                self.bary,
                self.contact_values,
                self.product,
                self.dots,
            ],
        )

    def _launch(self, kernel, dim, inputs):
        # Small vertex domains need enough blocks to populate the GPU. Block
        # reductions still accumulate global dot products/energy in float64.
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.fem.model.device, block_dim=64)

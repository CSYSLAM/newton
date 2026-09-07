# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-FileCopyrightText: Copyright (c) 2008-2025 NVIDIA Corporation
# SPDX-FileCopyrightText: Copyright (c) 2004-2008 AGEIA Technologies, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2001-2004 NovodeX AG
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of NVIDIA CORPORATION nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""PhysX 5.6.1 FEM surface constraint equations, translated to Warp.

Reference: PhysX FEMClothUtil.cuh, membraneEnergySolvePerTriangle and
bendingEnergySolvePerTrianglePair. Use the TGS (zero elastic multiplier)
branch with one constraint sweep per temporal substep. Material mapping:
tri_ke = mu * thickness, tri_ka = lambda * thickness, edge_ke = 1/kInv.
Elasticity and bending damping run in a separate velocity stage, as in
FEMCloth.cu. Their material coefficients are rates [1/s], not VBD damping.
"""

import warp as wp

from ...geometry import ParticleFlags


@wp.func
def effective_mass(i: int, inv_mass: wp.array[float], flags: wp.array[int]):
    w = float(0.0)
    if flags[i] & ParticleFlags.ACTIVE:
        w = inv_mass[i]
    return w


@wp.func
def polar_rotation(f0: wp.vec2, f1: wp.vec2):
    a = wp.dot(f0, f0)
    b = wp.dot(f1, f1)
    c = wp.dot(f0, f1)
    det = a * b - c * c
    r0 = wp.vec2(1.0, 0.0)
    r1 = wp.vec2(0.0, 1.0)
    if det >= 1.0e-14:
        s = wp.sqrt(det)
        t = wp.sqrt(a + b + 2.0 * s)
        sa = (a + s) / t
        sb = (b + s) / t
        sc = c / t
        sd = sa * sb - sc * sc
        if sd >= 1.0e-14:
            r0 = (sb * f0 - sc * f1) / sd
            r1 = (sa * f1 - sc * f0) / sd
    return r0, r1


@wp.func
def membrane_update(
    p0: wp.vec3,
    p1: wp.vec3,
    p2: wp.vec3,
    w0: float,
    w1: float,
    w2: float,
    q: wp.mat22,
    area: float,
    mu_h: float,
    lambda_h: float,
    dt: float,
):
    x01 = p1 - p0
    x02 = p2 - p0
    axis0 = wp.normalize(x01)
    axis1 = wp.normalize(wp.cross(wp.cross(x01, x02), axis0))
    a = wp.vec2(wp.dot(axis0, x01), wp.dot(axis1, x01))
    b = wp.vec2(wp.dot(axis0, x02), wp.dot(axis1, x02))
    d0 = wp.vec2(0.0)
    d1 = wp.vec2(0.0)
    d2 = wp.vec2(0.0)
    if mu_h > 1.0e-14 and area > 0.0:
        f0 = q[0, 0] * a + q[1, 0] * b
        f1 = q[0, 1] * a + q[1, 1] * b
        r0, r1 = polar_rotation(f0, f1)
        e0 = f0 - r0
        e1 = f1 - r1
        c = wp.sqrt(wp.dot(e0, e0) + wp.dot(e1, e1))
        if c > 1.0e-14:
            g1 = (q[0, 0] * e0 + q[0, 1] * e1) / c
            g2 = (q[1, 0] * e0 + q[1, 1] * e1) / c
            g0 = -g1 - g2
            alpha = 1.0 / (2.0 * mu_h * area * dt * dt)
            dl = -c / (w0 * wp.length_sq(g0) + w1 * wp.length_sq(g1) + w2 * wp.length_sq(g2) + alpha)
            d0 = w0 * dl * g0
            d1 = w1 * dl * g1
            d2 = w2 * dl * g2
        if lambda_h > 1.0e-14:
            # Re-evaluate area after the ARAP update, as in PhysX.
            a += d1 - d0
            b += d2 - d0
            c = 0.5 * (a[0] * b[1] - a[1] * b[0]) / area - 1.0
            g1 = wp.vec2(b[1], -b[0]) * (0.5 / area)
            g2 = wp.vec2(-a[1], a[0]) * (0.5 / area)
            g0 = -g1 - g2
            alpha = 1.0 / (lambda_h * area * dt * dt)
            dl = -c / (w0 * wp.length_sq(g0) + w1 * wp.length_sq(g1) + w2 * wp.length_sq(g2) + alpha)
            d0 += w0 * dl * g0
            d1 += w1 * dl * g1
            d2 += w2 * dl * g2
    return (p0 + d0[0] * axis0 + d0[1] * axis1, p1 + d1[0] * axis0 + d1[1] * axis1, p2 + d2[0] * axis0 + d2[1] * axis1)


@wp.kernel
def solve_membrane(
    ids: wp.array[int],
    triangles: wp.array2d[int],
    rest: wp.array[wp.mat22],
    areas: wp.array[float],
    materials: wp.array2d[float],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    dt: float,
    pos: wp.array[wp.vec3],
):
    tri = ids[wp.tid()]
    i, j, k = triangles[tri, 0], triangles[tri, 1], triangles[tri, 2]
    p0, p1, p2 = membrane_update(
        pos[i],
        pos[j],
        pos[k],
        effective_mass(i, inv_mass, flags),
        effective_mass(j, inv_mass, flags),
        effective_mass(k, inv_mass, flags),
        rest[tri],
        areas[tri],
        materials[tri, 0],
        materials[tri, 1],
        dt,
    )
    pos[i] = p0
    pos[j] = p1
    pos[k] = p2


@wp.func
def bending_update(
    x0: wp.vec3,
    x1: wp.vec3,
    x2: wp.vec3,
    x3: wp.vec3,
    w0: float,
    w1: float,
    w2: float,
    w3: float,
    rest_angle: float,
    stiffness: float,
    dt: float,
):
    if stiffness <= 0.0:
        return x0, x1, x2, x3
    x02, x03, x12, x13, x23 = x2 - x0, x3 - x0, x2 - x1, x3 - x1, x3 - x2
    length = wp.length(x23)
    sn0, sn1 = wp.cross(x02, x03), wp.cross(x13, x12)
    n0_len, n1_len = wp.length(sn0), wp.length(sn1)
    if wp.min(length, wp.min(n0_len, n1_len)) < 1.0e-14:
        return x0, x1, x2, x3
    n0, n1, axis = sn0 / n0_len, sn1 / n1_len, x23 / length
    angle = wp.atan2(wp.dot(wp.cross(n0, n1), axis), wp.dot(n0, n1))
    c = angle - rest_angle
    if wp.abs(c + 2.0 * wp.pi) < wp.abs(c):
        c += 2.0 * wp.pi
    elif wp.abs(c - 2.0 * wp.pi) < wp.abs(c):
        c -= 2.0 * wp.pi
    c = wp.clamp(c, -0.5 * wp.pi, 0.5 * wp.pi)
    t0, t1 = n0 / n0_len, n1 / n1_len
    g0, g1 = -length * t0, -length * t1
    g2 = wp.dot(x03, axis) * t0 + wp.dot(x13, axis) * t1
    g3 = -(wp.dot(x02, axis) * t0 + wp.dot(x12, axis) * t1)
    alpha = 1.0 / (stiffness * dt * dt)
    dl = -c / (w0 * wp.length_sq(g0) + w1 * wp.length_sq(g1) + w2 * wp.length_sq(g2) + w3 * wp.length_sq(g3) + alpha)
    return x0 + w0 * dl * g0, x1 + w1 * dl * g1, x2 + w2 * dl * g2, x3 + w3 * dl * g3


@wp.kernel
def solve_bending(
    ids: wp.array[int],
    edges: wp.array2d[int],
    rest_angle: wp.array[float],
    materials: wp.array2d[float],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    dt: float,
    pos: wp.array[wp.vec3],
):
    e = ids[wp.tid()]
    i, j, k, l = edges[e, 0], edges[e, 1], edges[e, 2], edges[e, 3]
    a, b, c, d = bending_update(
        pos[i],
        pos[j],
        pos[k],
        pos[l],
        effective_mass(i, inv_mass, flags),
        effective_mass(j, inv_mass, flags),
        effective_mass(k, inv_mass, flags),
        effective_mass(l, inv_mass, flags),
        rest_angle[e],
        materials[e, 0],
        dt,
    )
    pos[i] = a
    pos[j] = b
    pos[k] = c
    pos[l] = d


@wp.kernel
def solve_pair_partition(
    start: int,
    ids: wp.array[int],
    vertices: wp.array[wp.vec4i],
    faces: wp.array[wp.vec2i],
    poses: wp.array[wp.mat22],
    areas: wp.array[float],
    remap: wp.array[int],
    offsets: wp.array[int],
    copies: wp.array[wp.vec3],
    valid: wp.array[int],
    materials: wp.array2d[float],
    bending: wp.array2d[float],
    rest_angles: wp.array[float],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    shared: bool,
    dt: float,
    pos: wp.array[wp.vec3],
):
    slot = start + wp.tid()
    count = ids.shape[0]
    v = vertices[slot]
    x0, x1, x2, x3 = wp.vec3(pos[v[0]]), wp.vec3(pos[v[1]]), wp.vec3(pos[v[2]]), wp.vec3(pos[v[3]])
    if valid[slot]:
        x0 = wp.vec3(copies[slot])
    if valid[slot + count]:
        x1 = wp.vec3(copies[slot + count])
    if valid[slot + 2 * count]:
        x2 = wp.vec3(copies[slot + 2 * count])
    if valid[slot + 3 * count]:
        x3 = wp.vec3(copies[slot + 3 * count])
    w = wp.vec4(0.0)
    for j in range(4):
        w[j] = effective_mass(v[j], inv_mass, flags) * float(offsets[v[j] + 1] - offsets[v[j]])
    if shared:
        tri = faces[slot]
        x2, x3, x0 = membrane_update(
            x2,
            x3,
            x0,
            w[2],
            w[3],
            w[0],
            poses[2 * slot],
            areas[2 * slot],
            materials[tri[0], 0],
            materials[tri[0], 1],
            dt,
        )
        x2, x3, x1 = membrane_update(
            x2,
            x3,
            x1,
            w[2],
            w[3],
            w[1],
            poses[2 * slot + 1],
            areas[2 * slot + 1],
            materials[tri[1], 0],
            materials[tri[1], 1],
            dt,
        )
    x0, x1, x2, x3 = bending_update(
        x0, x1, x2, x3, w[0], w[1], w[2], w[3], rest_angles[ids[slot]], bending[ids[slot], 0], dt
    )
    copies[remap[slot]] = x0
    copies[remap[slot + count]] = x1
    copies[remap[slot + 2 * count]] = x2
    copies[remap[slot + 3 * count]] = x3
    for j in range(4):
        valid[remap[slot + j * count]] = 1


@wp.kernel
def average_pair_copies(
    count: int,
    offsets: wp.array[int],
    copies: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    pos: wp.array[wp.vec3],
):
    i = wp.tid()
    begin, end = offsets[i], offsets[i + 1]
    if end > begin and effective_mass(i, inv_mass, flags) > 0.0:
        total = wp.vec3(0.0)
        for j in range(begin, end):
            total += copies[4 * count + j] - pos[i]
        pos[i] += total / float(end - begin)


@wp.kernel
def membrane_damping(
    triangles: wp.array2d[int],
    materials: wp.array2d[float],
    pos: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    dt: float,
    delta: wp.array[wp.vec3],
):
    t = wp.tid()
    factor = wp.clamp(materials[t, 2] * dt, 0.0, 1.0)
    if factor == 0.0:
        return
    i, j, k = triangles[t, 0], triangles[t, 1], triangles[t, 2]
    a, b, c = pos[i], pos[j], pos[k]
    normal = wp.cross(b - a, c - a)
    if wp.length(normal) < 1.0e-6:
        return
    normal = wp.normalize(normal)
    center = (a + b + c) / 3.0
    linear = (velocity[i] + velocity[j] + velocity[k]) / 3.0
    u, v, w = velocity[i] - linear, velocity[j] - linear, velocity[k] - linear
    u -= normal * wp.dot(normal, u)
    v -= normal * wp.dot(normal, v)
    w -= normal * wp.dot(normal, w)
    ra, rb, rc = wp.cross(normal, a - center), wp.cross(normal, b - center), wp.cross(normal, c - center)
    if wp.min(wp.length_sq(ra), wp.min(wp.length_sq(rb), wp.length_sq(rc))) < 1.0e-20:
        return
    spin = (
        wp.dot(ra, u) / wp.length_sq(ra) + wp.dot(rb, v) / wp.length_sq(rb) + wp.dot(rc, w) / wp.length_sq(rc)
    ) / 3.0
    da, db, dc = spin * ra - u, spin * rb - v, spin * rc - w
    mean = (da + db + dc) / 3.0
    wa, wb, wc = (
        effective_mass(i, inv_mass, flags),
        effective_mass(j, inv_mass, flags),
        effective_mass(k, inv_mass, flags),
    )
    if wa + wb + wc > 0.0:
        factor /= wa + wb + wc
        wp.atomic_add(delta, i, (da - mean) * wa * factor)
        wp.atomic_add(delta, j, (db - mean) * wb * factor)
        wp.atomic_add(delta, k, (dc - mean) * wc * factor)


@wp.kernel
def bending_damping(
    edges: wp.array2d[int],
    materials: wp.array2d[float],
    pos: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    dt: float,
    delta: wp.array[wp.vec3],
):
    e = wp.tid()
    i, j, a, b = edges[e, 0], edges[e, 1], edges[e, 2], edges[e, 3]
    factor = wp.clamp(materials[e, 1] * dt, 0.0, 1.0)
    if i < 0 or j < 0 or factor == 0.0 or materials[e, 0] <= 0.0:
        return
    axis = pos[b] - pos[a]
    if wp.length(axis) < 1.0e-6:
        return
    axis = wp.normalize(axis)
    linear = (velocity[a] + velocity[b]) * 0.5
    ri, rj = wp.cross(axis, pos[i] - pos[a]), wp.cross(axis, pos[j] - pos[a])
    if wp.min(wp.length_sq(ri), wp.length_sq(rj)) < 1.0e-20:
        return
    spin = (
        wp.dot(rj, velocity[j] - linear) / wp.length_sq(rj) - wp.dot(ri, velocity[i] - linear) / wp.length_sq(ri)
    ) * factor
    di, dj = spin * ri, -spin * rj
    mean = (di + dj) * 0.25
    wi, wj, wa, wb = (
        effective_mass(i, inv_mass, flags),
        effective_mass(j, inv_mass, flags),
        effective_mass(a, inv_mass, flags),
        effective_mass(b, inv_mass, flags),
    )
    if wi + wj + wa + wb > 0.0:
        weight = 1.0 / (wi + wj + wa + wb)
        wp.atomic_add(delta, i, (di - mean) * wi * weight)
        wp.atomic_add(delta, j, (dj - mean) * wj * weight)
        wp.atomic_add(delta, a, -mean * wa * weight)
        wp.atomic_add(delta, b, -mean * wb * weight)


@wp.kernel
def apply_damping(delta: wp.array[wp.vec3], velocity: wp.array[wp.vec3]):
    i = wp.tid()
    velocity[i] += delta[i]


@wp.kernel
def predict(
    pos: wp.array[wp.vec3],
    vel: wp.array[wp.vec3],
    force: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    worlds: wp.array[int],
    gravity: wp.array[wp.vec3],
    targets: wp.array[wp.vec3],
    remaining_steps: int,
    dt: float,
    candidate: wp.array[wp.vec3],
):
    i = wp.tid()
    w = effective_mass(i, inv_mass, flags)
    if w > 0.0:
        candidate[i] = pos[i] + dt * (vel[i] + dt * (gravity[worlds[i]] + w * force[i]))
    else:
        candidate[i] = pos[i] + (targets[i] - pos[i]) / float(remaining_steps)


@wp.kernel
def update_velocity(
    prev: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    dt: float,
    velocity: wp.array[wp.vec3],
):
    i = wp.tid()
    velocity[i] = (pos[i] - prev[i]) / dt

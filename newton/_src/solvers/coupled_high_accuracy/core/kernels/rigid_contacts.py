# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Penalty contacts between shapes on affine bodies and static shapes (native ``kernels/rigid_contacts.cu`` and
``rigid_contact_geometry.cuh``).

Raw points (sphere centres) determine the gap and the normal; adjusted points (sphere surfaces or supplied
offsets) determine friction velocities and reaction levers. With ``epsilon > 0`` the smooth contact model averages
the material parameters and regularises friction below epsilon.
"""

import warp as wp

from .common import normalized_or_zero
from .contact_geometry import (
    contact_body_slot,
    contact_dot,
    contact_dot_x_first,
    contact_normalize,
    contact_point,
    contact_point_velocity,
    contact_shape,
    contact_sphere_offset,
    contact_wrench,
)

vec8i = wp.types.vector(length=8, dtype=wp.int32)
vec8f = wp.types.vector(length=8, dtype=wp.float32)


@wp.struct
class RigidContactView:
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
    kf: wp.array[float]
    mu: wp.array[float]
    epsilon: float
    local_shape_count: int
    shapes_per_world: int
    worlds: int
    body_ids: wp.array[wp.int32]
    body_indices: wp.array[wp.vec4i]
    body_com: wp.array[wp.vec3]
    particles_per_world: int
    x: wp.array[wp.vec3]
    velocity: wp.array[wp.vec3]
    previous: wp.array[wp.vec3]
    inv_dt: float


@wp.struct
class RigidContact:
    valid: int
    shape0: int
    shape1: int
    slot0: int
    slot1: int
    ids0: wp.vec4i
    ids1: wp.vec4i
    com0: wp.vec3
    com1: wp.vec3
    raw0: wp.vec3
    raw1: wp.vec3
    adjusted0: wp.vec3
    adjusted1: wp.vec3
    normal: wp.vec3
    tangent: wp.vec3
    fn: float
    eta: float
    normal_stiffness: float
    friction_stiffness: float
    separate: int


@wp.func
def contact_active(v: RigidContactView, i: int):
    if i >= v.capacity or v.count.shape[0] == 0:
        return False
    return i < wp.max(v.count[0], 0)


@wp.func
def rigid_contact(v: RigidContactView, i: int):
    """Geometry and penalty forces of rigid contact i; ``valid`` is 0 when it is skipped."""
    c = RigidContact()
    c.valid = 0
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
    if slot0 < 0 and slot1 < 0:
        return c
    if world0 >= 0 and world1 >= 0 and world0 != world1:
        return c
    c.shape0 = s0
    c.shape1 = s1
    c.slot0 = slot0
    c.slot1 = slot1
    if slot0 >= 0:
        b = v.body_indices[slot0]
        o = world0 * v.particles_per_world
        c.ids0 = wp.vec4i(b[0] + o, b[1] + o, b[2] + o, b[3] + o)
        c.com0 = v.body_com[slot0]
    if slot1 >= 0:
        b = v.body_indices[slot1]
        o = world1 * v.particles_per_world
        c.ids1 = wp.vec4i(b[0] + o, b[1] + o, b[2] + o, b[3] + o)
        c.com1 = v.body_com[slot1]
    x = v.x
    supplied = contact_normalize(v.normal[i])
    sphere0 = v.shape_type[s0] == 3
    sphere1 = v.shape_type[s1] == 3
    c.raw0 = v.point0[i]
    if sphere0:
        c.raw0 = wp.transform_get_translation(v.transform[s0])
    c.raw1 = v.point1[i]
    if sphere1:
        c.raw1 = wp.transform_get_translation(v.transform[s1])
    world_point0 = c.raw0
    if slot0 >= 0:
        world_point0 = contact_point(c.raw0, c.ids0, c.com0, x)
    world_point1 = c.raw1
    if slot1 >= 0:
        world_point1 = contact_point(c.raw1, c.ids1, c.com1, x)
    candidate = wp.vec3(0.0)
    if sphere0 and sphere1:
        diff = world_point1 - world_point0
        candidate = normalized_or_zero(diff, wp.sqrt(contact_dot_x_first(diff, diff)), 0.0)
    n = supplied
    if contact_dot(candidate, candidate) >= 0.5:
        n = candidate
    if not (contact_dot(n, n) >= 0.5):
        return c
    c.normal = n
    has_offset0 = v.offset0.shape[0] > 0
    has_offset1 = v.offset1.shape[0] > 0
    c.separate = int(has_offset0 or has_offset1 or sphere0 or sphere1)
    c.adjusted0 = c.raw0
    if sphere0:
        offset = v.shape_scale[s0][0] * n
        if slot0 >= 0:
            offset = contact_sphere_offset(offset, c.ids0, x)
        c.adjusted0 = c.raw0 + offset
    if has_offset0:
        c.adjusted0 = v.point0[i] + v.offset0[i]
    c.adjusted1 = c.raw1
    if sphere1:
        offset = -(v.shape_scale[s1][0] * n)
        if slot1 >= 0:
            offset = contact_sphere_offset(offset, c.ids1, x)
        c.adjusted1 = c.raw1 + offset
    if has_offset1:
        c.adjusted1 = v.point1[i] + v.offset1[i]
    depth = (v.margin0[i] + v.margin1[i]) - contact_dot(world_point1 - world_point0, n)
    if depth <= 0.0:
        return c
    nv0 = wp.vec3(0.0)
    tv0 = wp.vec3(0.0)
    if slot0 >= 0:
        nv0 = contact_point_velocity(c.raw0, c.ids0, c.com0, x, v.previous, v.velocity, v.inv_dt)
        tv0 = contact_point_velocity(c.adjusted0, c.ids0, c.com0, x, v.previous, v.velocity, v.inv_dt)
    nv1 = wp.vec3(0.0)
    tv1 = wp.vec3(0.0)
    if slot1 >= 0:
        nv1 = contact_point_velocity(c.raw1, c.ids1, c.com1, x, v.previous, v.velocity, v.inv_dt)
        tv1 = contact_point_velocity(c.adjusted1, c.ids1, c.com1, x, v.previous, v.velocity, v.inv_dt)
    smooth = v.epsilon > 0.0
    ke = wp.max(v.ke[s0], v.ke[s1])
    kd = wp.max(v.kd[s0], v.kd[s1])
    mu = wp.max(v.mu[s0], v.mu[s1])
    if smooth:
        ke = (v.ke[s0] + v.ke[s1]) * 0.5
        kd = (v.kd[s0] + v.kd[s1]) * 0.5
        mu = wp.sqrt(wp.max(v.mu[s0] * v.mu[s1], 0.0))
    vn = contact_dot_x_first(nv1 - nv0, n)
    elastic = depth * ke
    effective_kd = kd
    if smooth and vn >= 0.0:
        effective_kd = 0.0
    raw_force = elastic - vn * effective_kd
    c.fn = wp.max(raw_force, 0.0)
    if smooth:
        c.fn = raw_force
    dv = tv1 - tv0
    c.tangent = dv - contact_dot_x_first(dv, n) * n
    speed = wp.sqrt(contact_dot(c.tangent, c.tangent))
    bound = c.fn * mu
    if smooth:
        bound = elastic * mu
    eta = float(0.0)
    if smooth:
        if bound > 0.0 and speed > 0.0:
            if not (speed <= v.epsilon):
                eta = bound / speed
            else:
                eta = ((2.0 - speed / v.epsilon) * bound) / v.epsilon
    elif bound > 0.0:
        eta = wp.min(wp.max(v.kf[s0], v.kf[s1]), bound / wp.max(speed, 1.0e-8))
    c.eta = eta
    c.normal_stiffness = effective_kd * v.inv_dt + ke
    c.friction_stiffness = eta * v.inv_dt
    c.valid = 1
    return c


@wp.func
def rigid_scatter(force: wp.array[wp.vec3], ids: wp.vec4i, value: wp.vec3, lever: wp.vec3, positive: bool):
    s = float(-1.0)
    if positive:
        s = 1.0
    wp.atomic_add(force, ids[0], s * value)
    wp.atomic_add(force, ids[1], s * (value * lever[0]))
    wp.atomic_add(force, ids[2], s * (value * lever[1]))
    wp.atomic_add(force, ids[3], s * (value * lever[2]))


@wp.func
def rigid_scatter_scalar(scalar: wp.array[float], ids: wp.vec4i, stiffness: float, lever: wp.vec3):
    wp.atomic_add(scalar, ids[0], stiffness)
    wp.atomic_add(scalar, ids[1], lever[0] * (stiffness * lever[0]))
    wp.atomic_add(scalar, ids[2], lever[1] * (stiffness * lever[1]))
    wp.atomic_add(scalar, ids[3], lever[2] * (stiffness * lever[2]))


@wp.func
def side_scatter(
    c: RigidContact,
    force: wp.array[wp.vec3],
    scalar: wp.array[float],
    fnv: wp.vec3,
    ftv: wp.vec3,
    ids: wp.vec4i,
    com: wp.vec3,
    raw: wp.vec3,
    adjusted: wp.vec3,
    positive: bool,
):
    rn = raw - com
    rt = adjusted - com
    if force.shape[0] > 0:
        if c.separate != 0:
            rigid_scatter(force, ids, fnv, rn, positive)
            rigid_scatter(force, ids, ftv, rt, positive)
        else:
            rigid_scatter(force, ids, fnv + ftv, rn, positive)
    if scalar.shape[0] > 0:
        if c.separate != 0:
            rigid_scatter_scalar(scalar, ids, c.normal_stiffness, rn)
            if c.friction_stiffness > 0.0:
                rigid_scatter_scalar(scalar, ids, c.friction_stiffness, rt)
        else:
            rigid_scatter_scalar(scalar, ids, wp.max(c.normal_stiffness, c.friction_stiffness), rn)


@wp.kernel
def evaluate_contact_force(
    v: RigidContactView,
    force: wp.array[wp.vec3],
    scalar: wp.array[float],
    contact_force: wp.array[wp.spatial_vector],
):
    """Penalty contact forces on the body proxies, their scalar diagonal stiffness and the reported wrench."""
    i = wp.tid()
    if not contact_active(v, i):
        return
    c = rigid_contact(v, i)
    if c.valid == 0:
        return
    fnv = c.normal * -c.fn
    ftv = c.tangent * c.eta
    if contact_force.shape[0] > 0:
        adjusted = c.raw0
        if c.separate != 0:
            adjusted = c.adjusted0
        contact_force[i] = contact_wrench(c.slot0, c.ids0, c.com0, v.x, c.raw0, adjusted, fnv, ftv)
    if c.slot0 >= 0:
        side_scatter(c, force, scalar, fnv, ftv, c.ids0, c.com0, c.raw0, c.adjusted0, True)
    if c.slot1 >= 0:
        side_scatter(c, force, scalar, fnv, ftv, c.ids1, c.com1, c.raw1, c.adjusted1, False)


@wp.kernel
def evaluate_contact_tangent(
    v: RigidContactView,
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    system: wp.array[wp.mat33],
    precond: wp.array[wp.mat33],
):
    """Off-diagonal Jacobian products and the per-particle corrections to the isotropic diagonal split."""
    i = wp.tid()
    if not contact_active(v, i):
        return
    c = rigid_contact(v, i)
    if c.valid == 0:
        return
    n = c.normal
    nn = wp.outer(n, n)
    proj = wp.identity(n=3, dtype=float) - nn
    cn_stiffness = float(0.0)
    if c.fn > 0.0:
        cn_stiffness = c.normal_stiffness
    f = c.friction_stiffness
    targets = vec8i()
    normal_weights = vec8f()
    friction_weights = vec8f()
    entries = int(0)
    if c.slot0 >= 0:
        rn = c.raw0 - c.com0
        rf = c.adjusted0 - c.com0
        for k in range(4):
            targets[k] = c.ids0[k]
        normal_weights[0] = -1.0
        friction_weights[0] = -1.0
        for k in range(3):
            normal_weights[k + 1] = -rn[k]
            friction_weights[k + 1] = -rf[k]
        entries = 4
    if c.slot1 >= 0:
        rn = c.raw1 - c.com1
        rf = c.adjusted1 - c.com1
        for k in range(4):
            targets[entries + k] = c.ids1[k]
        normal_weights[entries] = 1.0
        friction_weights[entries] = 1.0
        for k in range(3):
            normal_weights[entries + k + 1] = rn[k]
            friction_weights[entries + k + 1] = rf[k]
        entries += 4
    if direction.shape[0] > 0 and product.shape[0] > 0:
        for part in range(2):
            matrix = cn_stiffness * nn
            if part == 1:
                matrix = f * proj
            for a in range(entries):
                total = wp.vec3(0.0)
                for b in range(entries):
                    if targets[a] == targets[b]:
                        continue
                    w = normal_weights[b]
                    if part == 1:
                        w = friction_weights[b]
                    total = w * direction[targets[b]] + total
                wa = normal_weights[a]
                if part == 1:
                    wa = friction_weights[a]
                wp.atomic_add(product, targets[a], (matrix * total) * wa)
    if system.shape[0] == 0 and precond.shape[0] == 0:
        return
    identity = wp.identity(n=3, dtype=float)
    for a in range(entries):
        n2 = normal_weights[a] * normal_weights[a]
        f2 = friction_weights[a] * friction_weights[a]
        sf = f * f2
        iso = n2 * wp.max(c.normal_stiffness, f)
        if c.separate != 0:
            iso = c.normal_stiffness * n2 + sf
        block = (cn_stiffness * n2) * nn + sf * proj - iso * identity
        if system.shape[0] > 0:
            wp.atomic_add(system, targets[a], block)
        if precond.shape[0] > 0:
            wp.atomic_add(precond, targets[a], block)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Particle-shape penalty contacts (native ``kernels/soft_contacts.cu``).

Static shapes and shapes on affine bodies push particles out along the contact normal with stiffness
min(model, shape) ke, damping capped at 2 sqrt(ke m), and a regularised Coulomb friction. Normal reactions on a body
use the raw contact lever, friction the sphere-surface lever.
"""

import warp as wp

from .contact_geometry import (
    contact_body_slot,
    contact_dot,
    contact_dot_x_first,
    contact_point,
    contact_point_velocity,
    contact_shape,
    contact_sphere_offset,
    contact_surface_velocity,
    contact_wrench,
)

# The solver never differentiates through its kernels; skipping the adjoints halves compile time.
wp.set_module_options({"enable_backward": False})


vec5i = wp.types.vector(length=5, dtype=wp.int32)
vec5f = wp.types.vector(length=5, dtype=wp.float32)


@wp.struct
class SoftContactView:
    count: wp.array[wp.int32]
    capacity: int
    particle: wp.array[wp.int32]
    shape: wp.array[wp.int32]
    body_pos: wp.array[wp.vec3]
    body_vel: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    radius: wp.array[float]
    mass: wp.array[float]
    particle_kind: wp.array[wp.int32]
    particles_per_world: int
    total_particles: int
    shape_body: wp.array[wp.int32]
    shape_type: wp.array[wp.int32]
    shape_transform: wp.array[wp.transform]
    shape_scale: wp.array[wp.vec3]
    shape_margin: wp.array[float]
    shape_ke: wp.array[float]
    shape_kd: wp.array[float]
    shape_kf: wp.array[float]
    shape_mu: wp.array[float]
    local_shape_count: int
    shapes_per_world: int
    worlds: int
    global_ke: float
    global_kd: float
    global_kf: float
    global_mu: float
    body_ids: wp.array[wp.int32]
    body_indices: wp.array[wp.vec4i]
    body_com: wp.array[wp.vec3]
    x: wp.array[wp.vec3]
    velocity: wp.array[wp.vec3]
    previous: wp.array[wp.vec3]
    inv_dt: float
    shape_filter: wp.array[wp.int32]


@wp.struct
class SoftContact:
    valid: int
    p: int
    local_p: int
    slot: int
    ids: wp.vec4i
    com: wp.vec3
    sphere: int
    raw: wp.vec3
    adjusted: wp.vec3
    n: wp.vec3
    ke: float
    kd: float
    fn: float
    eta: float
    tangent: wp.vec3


@wp.func
def evaluate_soft_contact(v: SoftContactView, i: int, bodies_only: int):
    """Geometry and forces of soft contact i; ``valid`` is 0 when the contact is skipped."""
    c = SoftContact()
    c.valid = 0
    active = int(0)
    if v.count.shape[0] > 0:
        active = wp.max(v.count[0], 0)
    if i >= active or i >= v.capacity:
        return c
    raw_shape = v.shape[i]
    if v.shape_filter.shape[0] > 0:
        if raw_shape < 0 or raw_shape >= v.shape_filter.shape[0]:
            return c
        if v.shape_filter[raw_shape] == 0:
            return c
    p = v.particle[i]
    if p < 0 or p >= v.total_particles or v.particles_per_world == 0:
        return c
    particle_world = p // v.particles_per_world
    ok, shape_world, s = contact_shape(raw_shape, v.local_shape_count, v.shapes_per_world, v.worlds)
    if ok == 0:
        return c
    if shape_world >= 0 and shape_world != particle_world:
        return c
    local_p = p % v.particles_per_world
    particle_mass = v.mass[local_p]
    if not (particle_mass > 0.0):
        return c
    if v.particle_kind.shape[0] > 0:
        if v.particle_kind[local_p] != 0:
            return c
    body = v.shape_body[s]
    slot = contact_body_slot(body, v.body_ids)
    if (body >= 0 and slot < 0) or (slot >= 0 and shape_world < 0):
        return c
    if bodies_only != 0 and slot < 0:
        return c
    ids = wp.vec4i(0)
    com = wp.vec3(0.0)
    if slot >= 0:
        base = v.body_indices[slot]
        offset = shape_world * v.particles_per_world
        ids = wp.vec4i(base[0] + offset, base[1] + offset, base[2] + offset, base[3] + offset)
        com = v.body_com[slot]
    x = v.x
    sphere = v.shape_type[s] == 3
    raw = v.body_pos[i]
    if sphere:
        raw = wp.transform_get_translation(v.shape_transform[s])
    world_point = raw
    if slot >= 0:
        world_point = contact_point(raw, ids, com, x)
    input_normal = v.normal[i]
    normal_length = wp.sqrt(contact_dot(input_normal, input_normal))
    n = wp.vec3(0.0)
    if normal_length > 0.0:
        n = input_normal / normal_length
    xp = x[p]
    adjusted = raw
    if sphere:
        r = v.shape_scale[s][0]
        radial = xp - world_point
        radial_length = wp.sqrt(contact_dot(radial, radial))
        alternative = wp.vec3(0.0)
        if radial_length > 0.0:
            alternative = radial / radial_length
        if contact_dot(alternative, alternative) >= 0.5:
            n = alternative
        normal_offset = r * n
        if slot >= 0:
            adjusted = raw + contact_sphere_offset(normal_offset, ids, x)
        else:
            adjusted = normal_offset + raw
    nx2 = n[0] * n[0]
    ny2 = n[1] * n[1]
    nz2 = n[2] * n[2]
    if not (nz2 + (ny2 + nx2) > 0.0):
        return c
    if sphere:
        world_point = v.shape_scale[s][0] * n + world_point
    depth = (v.radius[local_p] + v.shape_margin[s]) - contact_dot(xp - world_point, n)
    if depth <= 0.0:
        return c
    vp = v.velocity[p]
    if v.previous.shape[0] > 0:
        vp = (xp - v.previous[p]) * v.inv_dt
    v_raw = v.body_vel[i]
    v_adjusted = v_raw
    if slot >= 0:
        bv = contact_surface_velocity(v_raw, ids, x)
        v_raw = contact_point_velocity(raw, ids, com, x, v.previous, v.velocity, v.inv_dt) + bv
        v_adjusted = contact_point_velocity(adjusted, ids, com, x, v.previous, v.velocity, v.inv_dt) + bv
    r1 = vp - v_raw
    r2 = vp - v_adjusted
    ke = wp.min(v.global_ke, v.shape_ke[s])
    root = wp.sqrt(ke * particle_mass)
    kd = wp.min(wp.min(v.global_kd, v.shape_kd[s]), root + root)
    vn = contact_dot(r1, n)
    fn = wp.max(depth * ke - vn * kd, 0.0)
    tangent = r2 - contact_dot_x_first(r2, n) * n
    limit = fn * wp.min(v.global_mu, v.shape_mu[s])
    eta = float(0.0)
    if limit > 0.0:
        eta = wp.min(wp.min(v.global_kf, v.shape_kf[s]), limit / wp.max(wp.sqrt(contact_dot(tangent, tangent)), 1.0e-8))
    c.valid = 1
    c.p = p
    c.local_p = local_p
    c.slot = slot
    c.ids = ids
    c.com = com
    c.sphere = int(sphere)
    c.raw = raw
    c.adjusted = adjusted
    c.n = n
    c.ke = ke
    c.kd = kd
    c.fn = fn
    c.eta = eta
    c.tangent = tangent
    return c


@wp.kernel
def evaluate_soft_contact_force(
    v: SoftContactView,
    force: wp.array[wp.vec3],
    scalar: wp.array[float],
    system: wp.array[wp.mat33],
    precond: wp.array[wp.mat33],
    contact_force: wp.array[wp.spatial_vector],
):
    """Contact force, its scalar diagonal stiffness, the anisotropic block corrections and the reported wrench."""
    i = wp.tid()
    c = evaluate_soft_contact(v, i, 0)
    if c.valid == 0:
        return
    p = c.p
    ids = c.ids
    n = c.n
    normal_force = c.fn * n
    friction_force = -c.eta * c.tangent
    if contact_force.shape[0] > 0:
        contact_force[i] = contact_wrench(c.slot, ids, c.com, v.x, c.raw, c.adjusted, -normal_force, -friction_force)
    inv_dt = v.inv_dt
    if force.shape[0] > 0:
        wp.atomic_add(force, p, normal_force + friction_force)
        if c.slot >= 0:
            d = c.raw - c.com
            wp.atomic_add(force, ids[0], -normal_force)
            wp.atomic_add(force, ids[1], -(normal_force * d[0]))
            wp.atomic_add(force, ids[2], -(normal_force * d[1]))
            wp.atomic_add(force, ids[3], -(normal_force * d[2]))
            e = c.adjusted - c.com
            wp.atomic_add(force, ids[0], -friction_force)
            wp.atomic_add(force, ids[1], -(friction_force * e[0]))
            wp.atomic_add(force, ids[2], -(friction_force * e[1]))
            wp.atomic_add(force, ids[3], -(friction_force * e[2]))
    stiff = c.kd * inv_dt + c.ke
    fric = c.eta * inv_dt
    if scalar.shape[0] > 0:
        iso = stiff
        if c.sphere == 0:
            iso = wp.max(stiff, fric)
        extra = float(0.0)
        if c.sphere != 0:
            extra = fric
        wp.atomic_add(scalar, p, extra + iso)
        if c.slot >= 0:
            d = c.raw - c.com
            wp.atomic_add(scalar, ids[0], iso)
            wp.atomic_add(scalar, ids[1], d[0] * (iso * d[0]))
            wp.atomic_add(scalar, ids[2], d[1] * (iso * d[1]))
            wp.atomic_add(scalar, ids[3], d[2] * (iso * d[2]))
            if c.sphere != 0 and fric > 0.0:
                e = c.adjusted - c.com
                wp.atomic_add(scalar, ids[0], fric)
                wp.atomic_add(scalar, ids[1], e[0] * (fric * e[0]))
                wp.atomic_add(scalar, ids[2], e[1] * (fric * e[1]))
                wp.atomic_add(scalar, ids[3], e[2] * (fric * e[2]))
    if system.shape[0] == 0 and precond.shape[0] == 0:
        return
    targets = vec5i(p, 0, 0, 0, 0)
    normal_weights = vec5f(1.0, 0.0, 0.0, 0.0, 0.0)
    friction_weights = vec5f(1.0, 0.0, 0.0, 0.0, 0.0)
    count = 1
    if c.slot >= 0:
        d = c.raw - c.com
        e = c.adjusted - c.com
        targets = vec5i(p, ids[0], ids[1], ids[2], ids[3])
        normal_weights = vec5f(1.0, -1.0, -d[0], -d[1], -d[2])
        friction_weights = vec5f(1.0, -1.0, -e[0], -e[1], -e[2])
        count = 5
    nn = wp.outer(n, n)
    identity = wp.identity(n=3, dtype=float)
    for j in range(count):
        wn2 = normal_weights[j] * normal_weights[j]
        wf2 = friction_weights[j] * friction_weights[j]
        g = fric * wf2
        iso_j = float(0.0)
        if c.sphere != 0:
            iso_j = stiff * wn2 + g
        else:
            iso_j = wn2 * wp.max(stiff, fric)
        a = float(0.0)
        if c.fn > 0.0:
            a = stiff * wn2
        # a n n^T + g (I - n n^T) - iso I, the part not already in the scalar diagonal.
        block = a * nn + g * (identity - nn) - iso_j * identity
        if system.shape[0] > 0:
            wp.atomic_add(system, targets[j], block)
        if precond.shape[0] > 0:
            wp.atomic_add(precond, targets[j], block)


@wp.struct
class SoftTangentCache:
    """Per-contact product data fixed during a Newton iteration's linear solve: five targets with their normal and
    friction weights (flat, ``contact * 5 + entry``), the normal and the (normal, friction) stiffnesses. ``valid``
    is 0 for skipped contacts."""

    valid: wp.array[wp.int32]
    targets: wp.array[wp.int32]
    normal_weights: wp.array[float]
    friction_weights: wp.array[float]
    normal: wp.array[wp.vec3]
    stiffness: wp.array[wp.vec2]


@wp.kernel
def cache_soft_contact_tangent(v: SoftContactView, stride: int, cache: SoftTangentCache):
    """Evaluate each contact's geometry once per Newton iteration (``soft_contact_product`` re-evaluated it in every
    CG iteration)."""
    for i in range(wp.tid(), wp.min(v.capacity, wp.max(v.count[0], 0)), stride):
        c = evaluate_soft_contact(v, i, 1)
        cache.valid[i] = c.valid
        if c.valid != 0:
            stiff = float(0.0)
            if c.fn > 0.0:
                stiff = c.kd * v.inv_dt + c.ke
            cache.stiffness[i] = wp.vec2(stiff, c.eta * v.inv_dt)
            cache.normal[i] = c.n
            d = c.raw - c.com
            e = c.adjusted - c.com
            base = i * 5
            cache.targets[base] = c.p
            cache.normal_weights[base] = 1.0
            cache.friction_weights[base] = 1.0
            cache.normal_weights[base + 1] = -1.0
            cache.friction_weights[base + 1] = -1.0
            for k in range(4):
                cache.targets[base + 1 + k] = c.ids[k]
            for k in range(3):
                cache.normal_weights[base + 2 + k] = -d[k]
                cache.friction_weights[base + 2 + k] = -e[k]


@wp.kernel
def multiply_cached_soft_contact_tangent(
    count: wp.array[wp.int32],
    capacity: int,
    stride: int,
    cache: SoftTangentCache,
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
):
    """The arithmetic of ``soft_contact_product`` from the cached data."""
    for i in range(wp.tid(), wp.min(capacity, wp.max(count[0], 0)), stride):
        if cache.valid[i] != 0:
            n = cache.normal[i]
            stiffness = cache.stiffness[i]
            nn = wp.outer(n, n)
            normal_block = stiffness[0] * nn
            friction_block = stiffness[1] * (wp.identity(n=3, dtype=float) - nn)
            base = i * 5
            for a in range(5):
                target = cache.targets[base + a]
                normal_total = wp.vec3(0.0)
                friction_total = wp.vec3(0.0)
                for b in range(5):
                    other = cache.targets[base + b]
                    if b != a and other != target:
                        p = direction[other]
                        normal_total = cache.normal_weights[base + b] * p + normal_total
                        friction_total = cache.friction_weights[base + b] * p + friction_total
                value = (normal_block * normal_total) * cache.normal_weights[base + a] + (
                    friction_block * friction_total
                ) * cache.friction_weights[base + a]
                wp.atomic_add(product, target, value)

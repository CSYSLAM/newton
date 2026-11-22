# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Full-surface soft contacts: a point on a cloth/soft-body triangle (barycentric combination of three vertices)
against a static or body-attached shape (native ``kernels/surface_contacts.cu``).

Each contact keeps two Jacobian lists (normal and friction): the three triangle vertices with their barycentric
weights, followed by the four body proxies with weights (-1, -lever) when the shape is on an affine body.
"""

import warp as wp

from .common import add_tangent_product, normalized_or_zero
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


vec7i = wp.types.vector(length=7, dtype=wp.int32)
vec7f = wp.types.vector(length=7, dtype=wp.float32)


@wp.struct
class SurfaceContactView:
    count: wp.array[wp.int32]
    capacity: int
    # Launched threads: surface kernels grid-stride over the contacts with at most SM count x 256 threads, as natively,
    # instead of one mostly idle thread per capacity slot.
    stride: int
    indices: wp.array[wp.vec3i]
    barycentric: wp.array[wp.vec3]
    shape: wp.array[wp.int32]
    body_pos: wp.array[wp.vec3]
    body_vel: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    particle_only: wp.array[wp.int32]
    radius: wp.array[float]
    mass: wp.array[float]
    flags: wp.array[wp.int32]
    kind: wp.array[wp.int32]
    particles_per_world: int
    total_particles: int
    shape_body: wp.array[wp.int32]
    shape_type: wp.array[wp.int32]
    transform: wp.array[wp.transform]
    shape_scale: wp.array[wp.vec3]
    shape_margin: wp.array[float]
    shape_ke: wp.array[float]
    shape_kd: wp.array[float]
    shape_kf: wp.array[float]
    shape_mu: wp.array[float]
    local_shape_count: int
    shapes_per_world: int
    worlds: int
    ke: float
    kd: float
    kf: float
    mu: float
    body_ids: wp.array[wp.int32]
    body_indices: wp.array[wp.vec4i]
    body_com: wp.array[wp.vec3]
    x: wp.array[wp.vec3]
    previous: wp.array[wp.vec3]
    inv_dt: float


@wp.struct
class SurfaceTangent:
    """Jacobian lists and matrices of one contact, fixed during a Newton iteration's linear solve."""

    normal_targets: vec7i
    normal_weights: vec7f
    normal_entries: int
    friction_targets: vec7i
    friction_weights: vec7f
    friction_entries: int
    normal_matrix: wp.mat33
    friction_matrix: wp.mat33
    normal: wp.vec3
    normal_scale: float
    friction_scale: float


@wp.struct
class CachedSurfaceTangent:
    """The part of a SurfaceTangent the linear solve reads, in about half the bytes: the friction list shares the
    normal list's targets (both are built together), and the matrices are rebuilt from the normal and scales."""

    targets: vec7i
    normal_weights: vec7f
    friction_weights: vec7f
    entries: int
    normal: wp.vec3
    normal_scale: float
    friction_scale: float


@wp.struct
class SurfaceContact:
    tangent: SurfaceTangent
    normal_force: wp.vec3
    friction_force: wp.vec3
    wrench: wp.spatial_vector
    stiffness_sum: float


@wp.func
def surface_matrix(n: wp.vec3, scale: float, friction: int):
    nn = wp.outer(n, n)
    if friction != 0:
        return (wp.identity(n=3, dtype=float) - nn) * scale
    return nn * scale


@wp.func
def surface_contact(v: SurfaceContactView, i: int):
    """Contact i with both Jacobian lists; the lists stay empty when the contact is skipped."""
    c = SurfaceContact()
    t = SurfaceTangent()
    t.normal_entries = 0
    t.friction_entries = 0
    c.tangent = t
    if v.particles_per_world == 0:
        return c
    raw_shape = v.shape[i]
    if v.particle_only.shape[0] > 0 and raw_shape >= 0 and raw_shape < v.particle_only.shape[0]:
        if v.particle_only[raw_shape] != 0:
            return c
    vertices = v.indices[i]
    weights = v.barycentric[i]
    if vertices[0] < 0 or vertices[0] >= v.total_particles:
        return c
    world = vertices[0] // v.particles_per_world
    if world >= v.worlds:
        return c
    ok, shape_world, s = contact_shape(raw_shape, v.local_shape_count, v.shapes_per_world, v.worlds)
    if ok == 0:
        return c
    if shape_world >= 0 and shape_world != world:
        return c
    x = v.x
    point = wp.vec3(0.0)
    velocity = wp.vec3(0.0)
    radius = float(0.0)
    inverse_mass = float(0.0)
    total = float(0.0)
    entries = int(0)
    for j in range(3):
        vertex = vertices[j]
        if vertex < 0:
            continue
        b = weights[j]
        if not wp.isfinite(b) or b < 0.0 or vertex >= v.total_particles:
            return c
        if vertex // v.particles_per_world != world:
            return c
        local = vertex % v.particles_per_world
        if v.kind.shape[0] > 0:
            if v.kind[local] != 0:
                return c
        radius = wp.max(radius, v.radius[local])
        if b == 0.0:
            continue
        t.normal_targets[entries] = vertex
        t.normal_weights[entries] = b
        t.friction_targets[entries] = vertex
        t.friction_weights[entries] = b
        entries += 1
        total = total + b
        point = b * x[vertex] + point
        velocity = (b * v.inv_dt) * (x[vertex] - v.previous[vertex]) + velocity
        mass = v.mass[local]
        active = mass > 0.0
        if v.flags.shape[0] > 0:
            active = active and (v.flags[local] & 1) != 0
        if active:
            inverse_mass = (b * b) / mass + inverse_mass
    if not wp.isfinite(radius) or entries == 0 or wp.abs(total - 1.0) > 1.0e-4:
        return c
    body = v.shape_body[s]
    slot = contact_body_slot(body, v.body_ids)
    if (body >= 0 and slot < 0) or (slot >= 0 and shape_world < 0):
        return c
    ids = wp.vec4i(0)
    com = wp.vec3(0.0)
    if slot >= 0:
        base = v.body_indices[slot]
        o = shape_world * v.particles_per_world
        ids = wp.vec4i(base[0] + o, base[1] + o, base[2] + o, base[3] + o)
        com = v.body_com[slot]
    n = contact_normalize(v.normal[i])
    raw = v.body_pos[i]
    world_point = raw
    if slot >= 0:
        world_point = contact_point(raw, ids, com, x)
    adjusted = raw
    if v.shape_type[s] == 3:
        raw = wp.transform_get_translation(v.transform[s])
        center = raw
        if slot >= 0:
            center = contact_point(raw, ids, com, x)
        d = point - center
        alternative = normalized_or_zero(d, wp.sqrt(contact_dot_x_first(d, d)), 0.0)
        if contact_dot(alternative, alternative) >= 0.5:
            n = alternative
        rn = v.shape_scale[s][0] * n
        if slot >= 0:
            adjusted = raw + contact_sphere_offset(rn, ids, x)
        else:
            adjusted = raw + rn
        world_point = rn + center
    if not (n[2] * n[2] + (n[1] * n[1] + n[0] * n[0]) >= 0.5):
        return c
    raw_velocity = v.body_vel[i]
    adjusted_velocity = raw_velocity
    normal_entries = entries
    friction_entries = entries
    if slot >= 0:
        rn = raw - com
        rt = adjusted - com
        t.normal_targets[normal_entries] = ids[0]
        t.normal_weights[normal_entries] = -1.0
        t.friction_targets[friction_entries] = ids[0]
        t.friction_weights[friction_entries] = -1.0
        for k in range(3):
            t.normal_targets[normal_entries + k + 1] = ids[k + 1]
            t.normal_weights[normal_entries + k + 1] = -rn[k]
            t.friction_targets[friction_entries + k + 1] = ids[k + 1]
            t.friction_weights[friction_entries + k + 1] = -rt[k]
        normal_entries += 4
        friction_entries += 4
        body_velocity = contact_surface_velocity(raw_velocity, ids, x)
        raw_velocity = body_velocity + contact_point_velocity(raw, ids, com, x, v.previous, v.previous, v.inv_dt)
        adjusted_velocity = body_velocity + contact_point_velocity(
            adjusted, ids, com, x, v.previous, v.previous, v.inv_dt
        )
    relative = point - world_point
    r1 = velocity - raw_velocity
    r2 = velocity - adjusted_velocity
    for k in range(3):
        if not wp.isfinite(relative[k]) or not wp.isfinite(r1[k]) or not wp.isfinite(r2[k]):
            return c
    depth = (radius + v.shape_margin[s]) - contact_dot(relative, n)
    if not wp.isfinite(depth) or depth <= 0.0:
        return c
    ke = wp.min(v.ke, v.shape_ke[s])
    kd = wp.min(v.kd, v.shape_kd[s])
    if inverse_mass > 0.0:
        root = wp.sqrt(ke / inverse_mass)
        kd = wp.min(kd, root + root)
    vn = contact_dot(r1, n)
    if not wp.isfinite(vn):
        return c
    kdh = kd * v.inv_dt
    ahead = wp.max(depth + vn / v.inv_dt, 0.0)
    fn = wp.max(depth * ke + kdh * (depth - ahead), 0.0)
    if fn <= 0.0:
        return c
    vn2 = contact_dot(r2, n)
    tangent = r2 - vn2 * n
    speed = wp.sqrt(contact_dot(tangent, tangent))
    if not wp.isfinite(speed):
        return c
    limit = fn * wp.min(v.mu, v.shape_mu[s])
    eta = float(0.0)
    if limit > 0.0:
        eta = wp.min(wp.min(v.kf, v.shape_kf[s]), limit / wp.max(speed, 1.0e-8))
    c.normal_force = fn * n
    c.friction_force = -eta * tangent
    origin = wp.vec3(0.0)
    normal_point = raw
    friction_point = adjusted
    if slot >= 0:
        origin = contact_point(com, ids, com, x)
        normal_point = contact_point(raw, ids, com, x)
        friction_point = contact_point(adjusted, ids, com, x)
    c.wrench = wp.spatial_vector(
        -c.friction_force - c.normal_force,
        wp.cross(c.normal_force, normal_point - origin) + wp.cross(c.friction_force, friction_point - origin),
    )
    c.stiffness_sum = kd * v.inv_dt + ke
    t.normal_entries = normal_entries
    t.friction_entries = friction_entries
    t.normal_matrix = surface_matrix(n, c.stiffness_sum, 0)
    t.friction_matrix = surface_matrix(n, eta * v.inv_dt, 1)
    t.normal = n
    t.normal_scale = c.stiffness_sum
    t.friction_scale = eta * v.inv_dt
    c.tangent = t
    return c


@wp.func
def contact_count(v: SurfaceContactView):
    return wp.min(v.capacity, wp.max(v.count[0], 0))


@wp.kernel
def evaluate_surface_contact_force(
    v: SurfaceContactView, force: wp.array[wp.vec3], contact_force: wp.array[wp.spatial_vector]
):
    for i in range(wp.tid(), contact_count(v), v.stride):
        c = surface_contact(v, i)
        if contact_force.shape[0] > 0:
            contact_force[i] = c.wrench
        if force.shape[0] > 0:
            t = c.tangent
            for j in range(t.normal_entries):
                wp.atomic_add(force, t.normal_targets[j], t.normal_weights[j] * c.normal_force)
            for j in range(t.friction_entries):
                wp.atomic_add(force, t.friction_targets[j], t.friction_weights[j] * c.friction_force)


@wp.kernel
def add_surface_contact_stiffness_scale(v: SurfaceContactView, scale: wp.array[float]):
    for i in range(wp.tid(), contact_count(v), v.stride):
        c = surface_contact(v, i)
        t = c.tangent
        for j in range(t.normal_entries):
            w = t.normal_weights[j]
            wp.atomic_add(scale, t.normal_targets[j], (w * w) * c.stiffness_sum)


@wp.kernel
def add_surface_contact_preconditioner(v: SurfaceContactView, blocks: wp.array[wp.mat33]):
    for i in range(wp.tid(), contact_count(v), v.stride):
        t = surface_contact(v, i).tangent
        for j in range(t.normal_entries):
            w = t.normal_weights[j]
            wp.atomic_add(blocks, t.normal_targets[j], t.normal_matrix * (w * w))
        for j in range(t.friction_entries):
            w = t.friction_weights[j]
            wp.atomic_add(blocks, t.friction_targets[j], t.friction_matrix * (w * w))


@wp.func
def surface_tangent_product(
    t: SurfaceTangent,
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    proxy_slot: wp.array[wp.int32],
    replicas: wp.array[wp.vec3],
    contact: int,
):
    # Constant trip counts let the compiler unroll and keep the seven-entry lists in registers; dynamic indexing
    # would spill them to local memory.
    normal_total = wp.vec3(0.0)
    for j in range(7):
        if j < t.normal_entries:
            normal_total = t.normal_weights[j] * direction[t.normal_targets[j]] + normal_total
    friction_total = wp.vec3(0.0)
    for j in range(7):
        if j < t.friction_entries:
            friction_total = t.friction_weights[j] * direction[t.friction_targets[j]] + friction_total
    normal_value = t.normal_matrix * normal_total
    friction_value = t.friction_matrix * friction_total
    # With identical target lists (the usual case) each target gets one atomic instead of two; contacts on a body
    # share its proxies, so the product is bound by atomic contention.
    shared = t.normal_entries == t.friction_entries
    for j in range(7):
        if j < t.normal_entries and t.normal_targets[j] != t.friction_targets[j]:
            shared = False
    for j in range(7):
        if j < t.normal_entries:
            value = t.normal_weights[j] * normal_value
            if shared:
                value = value + t.friction_weights[j] * friction_value
            add_tangent_product(product, proxy_slot, replicas, contact, t.normal_targets[j], value)
    if not shared:
        for j in range(7):
            if j < t.friction_entries:
                add_tangent_product(
                    product,
                    proxy_slot,
                    replicas,
                    contact,
                    t.friction_targets[j],
                    t.friction_weights[j] * friction_value,
                )


@wp.kernel
def multiply_surface_contact_tangent(
    v: SurfaceContactView,
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    proxy_slot: wp.array[wp.int32],
    replicas: wp.array[wp.vec3],
):
    for i in range(wp.tid(), contact_count(v), v.stride):
        surface_tangent_product(surface_contact(v, i).tangent, direction, product, proxy_slot, replicas, i)


@wp.kernel
def cache_surface_contact_tangent(v: SurfaceContactView, cache: wp.array[CachedSurfaceTangent]):
    """Evaluate each contact's Jacobians and matrices once per Newton iteration (not in the reference SDK)."""
    for i in range(wp.tid(), contact_count(v), v.stride):
        t = surface_contact(v, i).tangent
        cached = CachedSurfaceTangent()
        cached.targets = t.normal_targets
        cached.normal_weights = t.normal_weights
        cached.friction_weights = t.friction_weights
        cached.entries = t.normal_entries
        cached.normal = t.normal
        cached.normal_scale = t.normal_scale
        cached.friction_scale = t.friction_scale
        cache[i] = cached


@wp.kernel
def multiply_cached_surface_contact_tangent(
    count: wp.array[wp.int32],
    capacity: int,
    stride: int,
    cache: wp.array[CachedSurfaceTangent],
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    proxy_slot: wp.array[wp.int32],
    replicas: wp.array[wp.vec3],
):
    for i in range(wp.tid(), wp.min(capacity, wp.max(count[0], 0)), stride):
        c = cache[i]
        t = SurfaceTangent()
        t.normal_targets = c.targets
        t.normal_weights = c.normal_weights
        t.normal_entries = c.entries
        t.friction_targets = c.targets
        t.friction_weights = c.friction_weights
        t.friction_entries = c.entries
        t.normal_matrix = surface_matrix(c.normal, c.normal_scale, 0)
        t.friction_matrix = surface_matrix(c.normal, c.friction_scale, 1)
        surface_tangent_product(t, direction, product, proxy_slot, replicas, i)

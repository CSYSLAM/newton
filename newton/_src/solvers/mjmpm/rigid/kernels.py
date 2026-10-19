# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid reference integration and projected contact impulses for MJMPM."""

import warp as wp

from newton._src.sim import BodyFlags
from newton._src.sim.contacts import contact_surface_separation
from newton._src.solvers.mjmpm.base import integrate_rigid_body


@wp.kernel
def kick(
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    forces: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3],
    mass_inv: wp.array[float],
    inertia: wp.array[wp.mat33],
    inertia_inv: wp.array[wp.mat33],
    flags: wp.array[int],
    worlds: wp.array[int],
    gravity: wp.array[wp.vec3],
    dt: float,
):
    i = wp.tid()
    if mass_inv[i] > 0.0 and (flags[i] & BodyFlags.KINEMATIC) == 0:
        if (flags[i] & BodyFlags.PROXY) != 0:
            # MuJoCo already integrated gyroscopic motion; rewind only interface wrenches.
            rotation = wp.transform_get_rotation(q[i])
            angular = wp.quat_rotate(
                rotation, inertia_inv[i] * wp.quat_rotate_inv(rotation, wp.spatial_bottom(forces[i]))
            )
            linear = mass_inv[i] * wp.spatial_top(forces[i]) + gravity[worlds[i]]
            velocity = qd[i] + dt * wp.spatial_vector(linear, angular)
        else:
            _unused, velocity = integrate_rigid_body(
                q[i], qd[i], forces[i], com[i], inertia[i], mass_inv[i], inertia_inv[i], gravity[worlds[i]], 0.0, dt
            )
        qd[i] = velocity


@wp.kernel
def drift(
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3],
    mass_inv: wp.array[float],
    flags: wp.array[int],
    dt: float,
):
    i = wp.tid()
    if mass_inv[i] > 0.0 and (flags[i] & BodyFlags.KINEMATIC) == 0:
        rotation = wp.transform_get_rotation(q[i])
        center = wp.transform_point(q[i], com[i]) + dt * wp.spatial_top(qd[i])
        rotation = wp.normalize(rotation + 0.5 * dt * wp.quat(wp.spatial_bottom(qd[i]), 0.0) * rotation)
        q[i] = wp.transform(center - wp.quat_rotate(rotation, com[i]), rotation)


@wp.func
def world_inverse_inertia(q: wp.transform, inv: wp.mat33):
    rotation = wp.quat_to_matrix(wp.transform_get_rotation(q))
    return rotation * inv * wp.transpose(rotation)


@wp.func
def point_velocity(velocity: wp.spatial_vector, radius: wp.vec3):
    return wp.spatial_top(velocity) + wp.cross(wp.spatial_bottom(velocity), radius)


@wp.func
def impulse_mass(mass_inv: float, inv: wp.mat33, radius: wp.vec3, direction: wp.vec3):
    angular = wp.cross(radius, direction)
    return mass_inv + wp.dot(angular, inv * angular)


@wp.kernel(enable_backward=False)
def solve_contacts(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    thickness0: wp.array[float],
    thickness1: wp.array[float],
    shape_body: wp.array[int],
    friction: wp.array[float],
    stiffness: wp.array[float],
    damping: wp.array[float],
    hard_contact: bool,
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3],
    mass_inv: wp.array[float],
    inertia_inv: wp.array[wp.mat33],
    flags: wp.array[int],
    impulses: wp.array[wp.vec3],
    body_impulse: wp.array[wp.spatial_vector],
    iterations: int,
    dt: float,
):
    # Single-threaded reference solve; ``solve_contact_buckets`` is the production path.
    for _sweep in range(iterations):
        for c in range(wp.min(count[0], shape0.shape[0])):
            s0 = shape0[c]
            s1 = shape1[c]
            if s0 < 0 or s1 < 0:
                continue
            a = shape_body[s0]
            b = shape_body[s1]
            if a == b:
                continue
            qa = wp.transform_identity()
            qb = wp.transform_identity()
            va = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
            vb = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
            ma = float(0.0)
            mb = float(0.0)
            ia = wp.mat33(0.0)
            ib = wp.mat33(0.0)
            ca = wp.vec3(0.0)
            cb = wp.vec3(0.0)
            if a >= 0:
                qa = q[a]
                va = qd[a]
                ca = wp.transform_point(qa, com[a])
                if (flags[a] & BodyFlags.KINEMATIC) == 0:
                    ma = mass_inv[a]
                    ia = world_inverse_inertia(qa, inertia_inv[a])
            if b >= 0:
                qb = q[b]
                vb = qd[b]
                cb = wp.transform_point(qb, com[b])
                if (flags[b] & BodyFlags.KINEMATIC) == 0:
                    mb = mass_inv[b]
                    ib = world_inverse_inertia(qb, inertia_inv[b])
            # MuJoCo already owns link-link and link-static contact.
            proxy_a = a >= 0 and (flags[a] & BodyFlags.PROXY) != 0
            proxy_b = b >= 0 and (flags[b] & BodyFlags.PROXY) != 0
            if (proxy_a and (proxy_b or mb == 0.0)) or (proxy_b and ma == 0.0):
                continue
            if ma + mb <= 0.0:
                continue
            pa = wp.transform_point(qa, point0[c])
            pb = wp.transform_point(qb, point1[c])
            n = normals[c]
            # Use the same world application point for equal/opposite angular impulses.
            point = 0.5 * (pa + pb)
            ra = point - ca
            rb = point - cb
            d = contact_surface_separation(pa, pb, n, thickness0[c], thickness1[c])
            if not hard_contact and d > 0.0:
                # Spring/damper contact is local to touching surfaces; a
                # speculative hard constraint here would suppress indentation.
                continue
            relative = point_velocity(vb, rb) - point_velocity(va, ra)
            denom = impulse_mass(ma, ia, ra, n) + impulse_mass(mb, ib, rb, n)
            target = -d / dt
            softness = float(0.0)
            if hard_contact:
                if d < 0.0:
                    target = -0.2 * d / dt
            else:
                # Implicit Kelvin-Voigt contact: J = dt * (-ke*d_next - kd*v_next).
                ke = 0.5 * (stiffness[s0] + stiffness[s1])
                kd = 0.5 * (damping[s0] + damping[s1])
                response = wp.max(kd + dt * ke, 1.0e-12)
                softness = 1.0 / (dt * response)
                target = -ke * d / response
            old = impulses[c]
            normal_old = wp.dot(old, n)
            normal_new = wp.max(
                0.0,
                normal_old + (target - wp.dot(relative, n) - softness * normal_old) / wp.max(denom + softness, 1.0e-12),
            )
            delta_n = (normal_new - normal_old) * n
            relative = (
                relative
                + (ma + mb) * delta_n
                + wp.cross(ib * wp.cross(rb, delta_n), rb)
                + wp.cross(ia * wp.cross(ra, delta_n), ra)
            )
            tangent_velocity = relative - n * wp.dot(relative, n)
            tangent = old - normal_old * n
            speed = wp.length(tangent_velocity)
            if speed > 1.0e-9:
                direction = tangent_velocity / speed
                tangent_mass = impulse_mass(ma, ia, ra, direction) + impulse_mass(mb, ib, rb, direction)
                tangent = tangent - tangent_velocity / wp.max(tangent_mass, 1.0e-12)
            limit = 0.5 * (friction[s0] + friction[s1]) * normal_new
            if wp.length(tangent) > limit:
                tangent = wp.normalize(tangent) * limit
            new = normal_new * n + tangent
            delta = new - old
            impulses[c] = new
            if a >= 0 and ma > 0.0:
                qd[a] = va - wp.spatial_vector(ma * delta, ia * wp.cross(ra, delta))
                body_impulse[a] = body_impulse[a] - wp.spatial_vector(delta, wp.cross(ra, delta))
            if b >= 0 and mb > 0.0:
                qd[b] = vb + wp.spatial_vector(mb * delta, ib * wp.cross(rb, delta))
                body_impulse[b] = body_impulse[b] + wp.spatial_vector(delta, wp.cross(rb, delta))


@wp.kernel
def prepare_contact_bodies(
    q: wp.array[wp.transform],
    com: wp.array[wp.vec3],
    mass_inv: wp.array[float],
    inertia_inv: wp.array[wp.mat33],
    flags: wp.array[int],
    out_center: wp.array[wp.vec3],
    out_mass_inv: wp.array[float],
    out_inertia_inv: wp.array[wp.mat33],
):
    """Cache world centers and effective inverse inertia for one contact solve."""
    i = wp.tid()
    out_center[i] = wp.transform_point(q[i], com[i])
    if (flags[i] & BodyFlags.KINEMATIC) == 0:
        out_mass_inv[i] = mass_inv[i]
        out_inertia_inv[i] = world_inverse_inertia(q[i], inertia_inv[i])
    else:
        out_mass_inv[i] = 0.0
        out_inertia_inv[i] = wp.mat33(0.0)


@wp.kernel(enable_backward=False)
def prepare_contacts(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    thickness0: wp.array[float],
    thickness1: wp.array[float],
    shape_body: wp.array[int],
    friction: wp.array[float],
    stiffness: wp.array[float],
    damping: wp.array[float],
    hard_contact: bool,
    q: wp.array[wp.transform],
    body_world: wp.array[int],
    flags: wp.array[int],
    center: wp.array[wp.vec3],
    mass_inv: wp.array[float],
    inertia_inv: wp.array[wp.mat33],
    world_count: int,
    single_bucket: bool,
    dt: float,
    out_bodies: wp.array[wp.vec2i],
    out_ra: wp.array[wp.vec3],
    out_rb: wp.array[wp.vec3],
    out_normal: wp.array[wp.vec3],
    out_params: wp.array[wp.vec4],
    out_keys: wp.array[int],
    out_ids: wp.array[int],
    impulses: wp.array[wp.vec3],
):
    """Evaluate pose-dependent contact constants once, ahead of the sequential sweeps.

    ``out_params`` holds (target, softness, normal mass, friction coefficient).
    Inactive contacts receive the sort key ``world_count + 1``.
    """
    c = wp.tid()
    out_ids[c] = c
    impulses[c] = wp.vec3(0.0)
    out_keys[c] = world_count + 1
    out_bodies[c] = wp.vec2i(-1, -1)
    if c >= wp.min(count[0], shape0.shape[0]):
        return
    s0 = shape0[c]
    s1 = shape1[c]
    if s0 < 0 or s1 < 0:
        return
    a = shape_body[s0]
    b = shape_body[s1]
    if a == b:
        return
    qa = wp.transform_identity()
    qb = wp.transform_identity()
    ma = float(0.0)
    mb = float(0.0)
    ia = wp.mat33(0.0)
    ib = wp.mat33(0.0)
    ca = wp.vec3(0.0)
    cb = wp.vec3(0.0)
    if a >= 0:
        qa = q[a]
        ca = center[a]
        ma = mass_inv[a]
        ia = inertia_inv[a]
    if b >= 0:
        qb = q[b]
        cb = center[b]
        mb = mass_inv[b]
        ib = inertia_inv[b]
    # MuJoCo already owns link-link and link-static contact.
    proxy_a = a >= 0 and (flags[a] & BodyFlags.PROXY) != 0
    proxy_b = b >= 0 and (flags[b] & BodyFlags.PROXY) != 0
    if (proxy_a and (proxy_b or mb == 0.0)) or (proxy_b and ma == 0.0):
        return
    if ma + mb <= 0.0:
        return
    pa = wp.transform_point(qa, point0[c])
    pb = wp.transform_point(qb, point1[c])
    n = normals[c]
    # Use the same world application point for equal/opposite angular impulses.
    point = 0.5 * (pa + pb)
    ra = point - ca
    rb = point - cb
    d = contact_surface_separation(pa, pb, n, thickness0[c], thickness1[c])
    if not hard_contact and d > 0.0:
        # Spring/damper contact is local to touching surfaces; a
        # speculative hard constraint here would suppress indentation.
        return
    target = -d / dt
    softness = float(0.0)
    if hard_contact:
        if d < 0.0:
            target = -0.2 * d / dt
    else:
        # Implicit Kelvin-Voigt contact: J = dt * (-ke*d_next - kd*v_next).
        ke = 0.5 * (stiffness[s0] + stiffness[s1])
        kd = 0.5 * (damping[s0] + damping[s1])
        response = wp.max(kd + dt * ke, 1.0e-12)
        softness = 1.0 / (dt * response)
        target = -ke * d / response
    denom = impulse_mass(ma, ia, ra, n) + impulse_mass(mb, ib, rb, n)
    out_bodies[c] = wp.vec2i(a, b)
    out_ra[c] = ra
    out_rb[c] = rb
    out_normal[c] = n
    out_params[c] = wp.vec4(target, softness, denom, 0.5 * (friction[s0] + friction[s1]))
    key = int(0)
    if not single_bucket:
        world = int(-1)
        if a >= 0 and ma > 0.0:
            world = body_world[a]
        elif b >= 0:
            world = body_world[b]
        key = world_count if world < 0 else world
    out_keys[c] = key


@wp.kernel
def contact_bucket_ranges(
    keys: wp.array[int],
    count: int,
    bucket_count: int,
    begin: wp.array[int],
    end: wp.array[int],
):
    """Locate each bucket's run in contacts sorted by bucket key."""
    i = wp.tid()
    key = keys[i]
    if key >= bucket_count:
        return
    if i == 0 or keys[i - 1] != key:
        begin[key] = i
    if i == count - 1 or keys[i + 1] != key:
        end[key] = i + 1


@wp.kernel
def single_bucket_range(count: wp.array[int], capacity: int, begin: wp.array[int], end: wp.array[int]):
    """Cover all detected contacts with one bucket in detection order."""
    begin[0] = 0
    end[0] = wp.min(count[0], capacity)


@wp.kernel(enable_backward=False)
def solve_contact_buckets(
    begin: wp.array[int],
    end: wp.array[int],
    ids: wp.array[int],
    bodies: wp.array[wp.vec2i],
    ra_list: wp.array[wp.vec3],
    rb_list: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    params: wp.array[wp.vec4],
    mass_inv: wp.array[float],
    inertia_inv: wp.array[wp.mat33],
    qd: wp.array[wp.spatial_vector],
    impulses: wp.array[wp.vec3],
    body_impulse: wp.array[wp.spatial_vector],
    iterations: int,
):
    """Run sequential projected-impulse sweeps independently for each world.

    Contacts of a world keep their detection order, so each world reproduces
    the single-threaded reference ordering. Worlds share no dynamic bodies.
    """
    bucket = wp.tid()
    first = begin[bucket]
    last = end[bucket]
    for _sweep in range(iterations):
        for k in range(first, last):
            c = ids[k]
            pair = bodies[c]
            a = pair[0]
            b = pair[1]
            if a < 0 and b < 0:
                continue
            va = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
            vb = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
            ma = float(0.0)
            mb = float(0.0)
            ia = wp.mat33(0.0)
            ib = wp.mat33(0.0)
            if a >= 0:
                va = qd[a]
                ma = mass_inv[a]
                ia = inertia_inv[a]
            if b >= 0:
                vb = qd[b]
                mb = mass_inv[b]
                ib = inertia_inv[b]
            ra = ra_list[c]
            rb = rb_list[c]
            n = normals[c]
            p = params[c]
            target = p[0]
            softness = p[1]
            denom = p[2]
            relative = point_velocity(vb, rb) - point_velocity(va, ra)
            old = impulses[c]
            normal_old = wp.dot(old, n)
            normal_new = wp.max(
                0.0,
                normal_old + (target - wp.dot(relative, n) - softness * normal_old) / wp.max(denom + softness, 1.0e-12),
            )
            delta_n = (normal_new - normal_old) * n
            relative = (
                relative
                + (ma + mb) * delta_n
                + wp.cross(ib * wp.cross(rb, delta_n), rb)
                + wp.cross(ia * wp.cross(ra, delta_n), ra)
            )
            tangent_velocity = relative - n * wp.dot(relative, n)
            tangent = old - normal_old * n
            speed = wp.length(tangent_velocity)
            if speed > 1.0e-9:
                direction = tangent_velocity / speed
                tangent_mass = impulse_mass(ma, ia, ra, direction) + impulse_mass(mb, ib, rb, direction)
                tangent = tangent - tangent_velocity / wp.max(tangent_mass, 1.0e-12)
            limit = p[3] * normal_new
            if wp.length(tangent) > limit:
                tangent = wp.normalize(tangent) * limit
            new = normal_new * n + tangent
            delta = new - old
            impulses[c] = new
            if a >= 0 and ma > 0.0:
                qd[a] = va - wp.spatial_vector(ma * delta, ia * wp.cross(ra, delta))
                body_impulse[a] = body_impulse[a] - wp.spatial_vector(delta, wp.cross(ra, delta))
            if b >= 0 and mb > 0.0:
                qd[b] = vb + wp.spatial_vector(mb * delta, ib * wp.cross(rb, delta))
                body_impulse[b] = body_impulse[b] + wp.spatial_vector(delta, wp.cross(rb, delta))


@wp.kernel
def collect_material_impulses(
    impulses: wp.array[wp.vec3],
    points: wp.array[wp.vec3],
    collider_ids: wp.array[int],
    collider_body: wp.array[int],
    q: wp.array[wp.transform],
    com: wp.array[wp.vec3],
    body_impulse: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    collider = collider_ids[i]
    if collider >= 0 and collider < collider_body.shape[0]:
        body = collider_body[collider]
        if body >= 0:
            radius = points[i] - wp.transform_point(q[body], com[body])
            wp.atomic_add(body_impulse, body, wp.spatial_vector(impulses[i], wp.cross(radius, impulses[i])))


@wp.kernel
def apply_material_impulses(
    impulse: wp.array[wp.spatial_vector],
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
    mass_inv: wp.array[float],
    inertia_inv: wp.array[wp.mat33],
    flags: wp.array[int],
):
    i = wp.tid()
    if mass_inv[i] > 0.0 and (flags[i] & BodyFlags.KINEMATIC) == 0:
        p = impulse[i]
        qd[i] = qd[i] + wp.spatial_vector(
            mass_inv[i] * wp.spatial_top(p), world_inverse_inertia(q[i], inertia_inv[i]) * wp.spatial_bottom(p)
        )


@wp.kernel
def harvest_feedback(
    mapping: wp.array[int],
    impulses: wp.array[wp.spatial_vector],
    material: wp.array[wp.spatial_vector],
    dt: float,
    out: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    proxy = mapping[i]
    if proxy >= 0:
        out[proxy] = (impulses[i] + material[i]) / dt

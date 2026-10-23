# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cooperate within each rigid contact while preserving sequential GS contact order."""

import warp as wp

from newton._src.solvers.mjmpm.rigid import kernels as rk

PackedContact = rk.PackedContact
mat63 = rk.mat63
LOCAL_BODIES = rk.LOCAL_BODIES
_local_bodies = rk._local_bodies
_local_links = rk._local_links
_uses_link_responses = rk._uses_link_responses
point_velocity = rk.point_velocity
_packed_point_response = rk._packed_point_response


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    auto result = value;
    for (int i = 0; i < 3; ++i)
        result[i] = __shfl_sync(0xffffffff, value[i], source);
    return result;
#else
    return value;
#endif
""")
def _shuffle_point(value: wp.vec3, source: int) -> wp.vec3:
    """Broadcast a contact point velocity within a full 32-lane CUDA warp."""
    ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
    __syncwarp(0xffffffff);
#endif
""")
def _sync_warp():
    """Order packed-contact and contact-impulse writes between participating lanes."""
    ...


@wp.kernel(enable_backward=False, module="unique")
def solve_contact_buckets_cooperative(
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
    link_slot: wp.array[int],
    link_body: wp.array2d[int],
    mobility: wp.array3d[float],
    contacted_links: wp.array2d[int],
    contacted_count: wp.array[int],
    point_mobility: wp.array[wp.mat33],
    link_responses: wp.array2d[mat63],
    qd: wp.array[wp.spatial_vector],
    impulses: wp.array[wp.vec3],
    body_impulse: wp.array[wp.spatial_vector],
    iterations: int,
    packed: wp.array[PackedContact],
    fallback: wp.array[int],
    skip: wp.array[int] = None,
):
    """Sweep contacts in order, distributing independent body responses over one CUDA warp."""
    bucket, lane = wp.tid()
    first = begin[bucket]
    last = end[bucket]
    if lane == 0:
        fallback[bucket] = 0
    if skip and skip[0]:
        return
    link_count = int(0)
    if link_body:
        link_count = link_body.shape[1]

    table = _local_bodies()
    count = int(0)
    world = int(-1)
    ok = bool(True)
    packed_end = first
    for k in range(first, last):
        c = ids[k]
        pair = bodies[c]
        if pair[0] < 0 and pair[1] < 0:
            continue
        e = PackedContact()
        e.c = c
        e.a = pair[0]
        e.b = pair[1]
        e.la = -1
        e.lb = -1
        e.slot_a = -1
        e.slot_b = -1
        e.ma = 0.0
        e.mb = 0.0
        e.ia = wp.mat33(0.0)
        e.ib = wp.mat33(0.0)
        e.ra = ra_list[c]
        e.rb = rb_list[c]
        e.normal = normals[c]
        e.params = params[c]
        e.responses_a = 0
        e.responses_b = 0
        e.point_mobility = wp.mat33(0.0)
        for side in range(2):
            body = pair[side]
            if body >= 0 and ok:
                slot = int(-1)
                if link_slot:
                    slot = link_slot[body]
                if slot >= 0:
                    if not _uses_link_responses(slot, link_count, contacted_count, point_mobility):
                        ok = False
                    elif world < 0:
                        world = slot // link_count
                    elif world != slot // link_count:
                        ok = False
                found = int(-1)
                for j in range(count):
                    if table[j] == body:
                        found = j
                if found < 0:
                    if count < LOCAL_BODIES:
                        table[count] = body
                        found = count
                        count += 1
                    else:
                        ok = False
                if slot >= 0:
                    e.point_mobility = point_mobility[c]
                if side == 0:
                    e.responses_a = wp.where(slot >= 0, 1, 0)
                    e.la = found
                    e.slot_a = slot
                    e.ma = mass_inv[body]
                    e.ia = inertia_inv[body]
                else:
                    e.responses_b = wp.where(slot >= 0, 1, 0)
                    e.lb = found
                    e.slot_b = slot
                    e.mb = mass_inv[body]
                    e.ib = inertia_inv[body]
        combined = wp.mat33(0.0)
        for col in range(3):
            axis = wp.vec3(0.0)
            axis[col] = 1.0
            value = _packed_point_response(
                e.responses_b, e.point_mobility, e.b, e.mb, e.ib, e.rb, axis
            ) + _packed_point_response(e.responses_a, e.point_mobility, e.a, e.ma, e.ia, e.ra, axis)
            for row in range(3):
                combined[row, col] = value[row]
        e.point_mobility = combined
        if lane == 0:
            packed[packed_end] = e
        packed_end += 1
    links = _local_links()
    link_total = int(0)
    if world >= 0 and ok:
        link_total = contacted_count[world]
        for k in range(link_total):
            body = link_body[world, contacted_links[world, k]]
            found = int(-1)
            for j in range(count):
                if table[j] == body:
                    found = j
            if found < 0:
                if count < LOCAL_BODIES:
                    table[count] = body
                    found = count
                    count += 1
                else:
                    ok = False
            links[k] = found
    if not ok:
        if lane == 0:
            fallback[bucket] = 1
        return

    body = int(-1)
    v = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
    w = wp.spatial_vector(wp.vec3(0.0), wp.vec3(0.0))
    if lane < count:
        body = table[lane]
        v = qd[body]
        w = body_impulse[body]
    response_index = int(-1)
    for k in range(link_total):
        if links[k] == lane:
            response_index = k

    for phase in range(iterations + 1):
        for k in range(first, packed_end):
            _sync_warp()
            e = packed[k]
            # Prepared contacts always join distinct bodies. Each lane needs
            # only its own side's point velocity before the two broadcasts.
            radius = e.rb
            if lane == e.la:
                radius = e.ra
            point = point_velocity(v, radius)
            va = _shuffle_point(point, wp.max(e.la, 0))
            vb = _shuffle_point(point, wp.max(e.lb, 0))
            if e.a < 0:
                va = wp.vec3(0.0)
            if e.b < 0:
                vb = wp.vec3(0.0)
            old = impulses[e.c]
            delta = old
            if phase > 0:
                n = e.normal
                p = e.params
                target = p[0]
                softness = p[1]
                denom = p[2]
                relative = vb - va
                normal_old = wp.dot(old, n)
                normal_new = wp.max(
                    0.0,
                    normal_old
                    + (target - wp.dot(relative, n) - softness * normal_old) / wp.max(denom + softness, 1.0e-12),
                )
                tangent = wp.vec3(0.0)
                if normal_new > 0.0 and p[3] > 0.0:
                    delta_n = (normal_new - normal_old) * n
                    relative = relative + e.point_mobility * delta_n
                    tangent_velocity = relative - n * wp.dot(relative, n)
                    tangent = old - normal_old * n
                    speed_squared = wp.dot(tangent_velocity, tangent_velocity)
                    if speed_squared > 1.0e-18:
                        response_squared = wp.dot(tangent_velocity, e.point_mobility * tangent_velocity)
                        scale = speed_squared / wp.max(response_squared, 1.0e-12 * speed_squared)
                        tangent = tangent - scale * tangent_velocity
                    limit = p[3] * normal_new
                    tangent_squared = wp.dot(tangent, tangent)
                    if tangent_squared > limit * limit:
                        tangent = wp.normalize(tangent) * limit
                new = normal_new * n + tangent
                delta = new - old
                if lane == 0:
                    impulses[e.c] = new
            if delta == wp.vec3(0.0):
                continue
            if e.a >= 0 and e.ma > 0.0:
                if e.slot_a >= 0:
                    if response_index >= 0:
                        change = link_responses[e.c, response_index] * (-delta)
                        v = v + wp.spatial_vector(change[0], change[1], change[2], change[3], change[4], change[5])
                elif lane == e.la:
                    v = v - wp.spatial_vector(e.ma * delta, e.ia * wp.cross(e.ra, delta))
                if lane == e.la:
                    w = w + wp.spatial_vector(-delta, -wp.cross(e.ra, delta))
            if e.b >= 0 and e.mb > 0.0:
                if e.slot_b >= 0:
                    if response_index >= 0:
                        change = link_responses[e.c, response_index] * delta
                        v = v + wp.spatial_vector(change[0], change[1], change[2], change[3], change[4], change[5])
                elif lane == e.lb:
                    v = v + wp.spatial_vector(e.mb * delta, e.ib * wp.cross(e.rb, delta))
                if lane == e.lb:
                    w = w + wp.spatial_vector(delta, wp.cross(e.rb, delta))
    if lane < count:
        qd[body] = v
        body_impulse[body] = w

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared contact geometry (native ``kernels/contact_geometry.cuh`` and ``contact_wrench.cuh``).

Shape transforms are read as ``wp.transform`` (translation first, as the native 28-byte Transform). Body-attached
points are evaluated through the body's four affine proxies: x_0 + d_x x_1 + d_y x_2 + d_z x_3 with d = p - com.
"""

import warp as wp

from .affine import column, nearest_rotation
from .common import normalized_or_zero

# The solver never differentiates through its kernels; skipping the adjoints halves compile time.
wp.set_module_options({"enable_backward": False})


@wp.func
def contact_dot(a: wp.vec3, b: wp.vec3):
    return a[2] * b[2] + (a[0] * b[0] + a[1] * b[1])


@wp.func
def contact_dot_x_first(a: wp.vec3, b: wp.vec3):
    return a[2] * b[2] + (a[1] * b[1] + a[0] * b[0])


@wp.func
def contact_normalize(a: wp.vec3):
    return normalized_or_zero(a, wp.sqrt(contact_dot(a, a)), 0.0)


@wp.func
def contact_body_slot(body: int, ids: wp.array[wp.int32]):
    """Affine slot of a Newton body id: the identity slot when it matches, else a first-match search."""
    if body < 0:
        return -1
    count = ids.shape[0]
    if body < count:
        if ids[body] == body:
            return body
    for i in range(count):
        if ids[i] == body:
            return i
    return -1


@wp.func
def contact_point(point: wp.vec3, ids: wp.vec4i, com: wp.vec3, x: wp.array[wp.vec3]):
    d = point - com
    return d[2] * x[ids[3]] + (d[1] * x[ids[2]] + (d[0] * x[ids[1]] + x[ids[0]]))


@wp.func
def contact_velocity(
    index: int, x: wp.array[wp.vec3], previous: wp.array[wp.vec3], velocity: wp.array[wp.vec3], inv_dt: float
):
    if previous.shape[0] > 0:
        return (x[index] - previous[index]) * inv_dt
    return velocity[index]


@wp.func
def contact_point_velocity(
    point: wp.vec3,
    ids: wp.vec4i,
    com: wp.vec3,
    x: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    inv_dt: float,
):
    d = point - com
    v0 = contact_velocity(ids[0], x, previous, velocity, inv_dt)
    v1 = contact_velocity(ids[1], x, previous, velocity, inv_dt)
    v2 = contact_velocity(ids[2], x, previous, velocity, inv_dt)
    v3 = contact_velocity(ids[3], x, previous, velocity, inv_dt)
    return d[2] * v3 + (d[1] * v2 + (d[0] * v1 + v0))


@wp.func
def contact_surface_velocity(v: wp.vec3, ids: wp.vec4i, x: wp.array[wp.vec3]):
    return v[2] * x[ids[3]] + (v[0] * x[ids[1]] + v[1] * x[ids[2]])


@wp.func
def contact_sphere_offset(v: wp.vec3, ids: wp.vec4i, x: wp.array[wp.vec3]):
    """Sphere lever through A^-1 when A is nonsingular, else through the transpose of its polar rotation."""
    a = x[ids[1]]
    b = x[ids[2]]
    c = x[ids[3]]
    bc = wp.cross(b, c)
    det = contact_dot(a, bc)
    if wp.abs(det) > 1.0e-10 and wp.isfinite(det):
        inverse = 1.0 / det
        return wp.vec3(
            contact_dot(v, bc * inverse),
            contact_dot(v, wp.cross(c, a) * inverse),
            contact_dot(v, wp.cross(a, b) * inverse),
        )
    r = nearest_rotation(a, b, c)
    return wp.vec3(contact_dot(v, column(r, 0)), contact_dot(v, column(r, 1)), contact_dot(v, column(r, 2)))


@wp.func
def contact_shape(shape: int, local_count: int, per_world: int, worlds: int):
    """(valid, world, local) of a global shape id: replicated shapes precede the shared-static suffix (world -1)."""
    if shape < 0:
        return 0, -1, -1
    replicated = per_world * worlds
    if shape < replicated:
        return 1, shape // per_world, shape % per_world
    shared = shape - replicated + per_world
    if shared >= local_count:
        return 0, -1, -1
    return 1, -1, shared


@wp.func
def contact_wrench(
    slot: int,
    ids: wp.vec4i,
    com: wp.vec3,
    x: wp.array[wp.vec3],
    raw: wp.vec3,
    adjusted: wp.vec3,
    normal_force: wp.vec3,
    friction_force: wp.vec3,
):
    """Reaction wrench on the contact's first shape; moments about its world COM (world origin when static)."""
    origin = wp.vec3(0.0)
    pn = raw
    pt = adjusted
    if slot >= 0:
        origin = contact_point(com, ids, com, x)
        pn = contact_point(raw, ids, com, x)
        pt = contact_point(adjusted, ids, com, x)
    return wp.spatial_vector(
        normal_force + friction_force, wp.cross(pn - origin, normal_force) + wp.cross(pt - origin, friction_force)
    )

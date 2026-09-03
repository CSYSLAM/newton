# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MJVBDV2-private particle--shape contact kernels."""

import warp as wp

from ...geometry.flags import ParticleFlags, ShapeFlags
from ...geometry.kernels import (
    counter_increment,
    mesh_query_point_sign,
    resolve_mesh_sign_method,
    sdf_box,
    sdf_box_grad,
    sdf_capsule,
    sdf_capsule_grad,
    sdf_cone,
    sdf_cone_grad,
    sdf_cylinder,
    sdf_cylinder_grad,
    sdf_ellipsoid,
    sdf_ellipsoid_grad,
    sdf_plane,
    sdf_sphere,
    sdf_sphere_grad,
)
from ...geometry.types import Axis, GeoType
from ...utils.heightfield import HeightfieldData, sample_sdf_grad_heightfield

__all__ = [
    "compute_shape_world_aabbs",
    "create_soft_contacts_with_aabb",
    "create_speculative_soft_contacts_with_aabb",
]


@wp.kernel(enable_backward=False)
def compute_shape_world_aabbs(
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    shape_collision_aabb_lower: wp.array[wp.vec3],
    shape_collision_aabb_upper: wp.array[wp.vec3],
    # outputs
    shape_aabb_lower: wp.array[wp.vec3],
    shape_aabb_upper: wp.array[wp.vec3],
):
    """Transform precomputed local collision bounds into world-space AABBs."""
    shape_index = wp.tid()
    body_index = shape_body[shape_index]

    X_ws = shape_transform[shape_index]
    if body_index >= 0:
        X_ws = wp.transform_multiply(body_q[body_index], X_ws)

    position = wp.transform_get_translation(X_ws)
    rotation = wp.transform_get_rotation(X_ws)
    geo_type = shape_type[shape_index]
    scale = shape_scale[shape_index]
    margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0

    # A plane with either unbounded tangent axis uses the infinite-plane SDF.
    # Keep it in the candidate set; the signed half-space cannot be represented
    # by a finite AABB without a scene-wide spatial bound.
    if geo_type == GeoType.PLANE and (scale[0] <= 0.0 or scale[1] <= 0.0):
        extent = 1.0e30
        shape_aabb_lower[shape_index] = wp.vec3(-extent, -extent, -extent)
        shape_aabb_upper[shape_index] = wp.vec3(extent, extent, extent)
        return

    local_lower = shape_collision_aabb_lower[shape_index]
    local_upper = shape_collision_aabb_upper[shape_index]
    if geo_type == GeoType.PLANE:
        local_lower = wp.vec3(-0.5 * scale[0], -0.5 * scale[1], 0.0)
        local_upper = wp.vec3(0.5 * scale[0], 0.5 * scale[1], 0.0)

    local_center = 0.5 * (local_lower + local_upper)
    local_half = 0.5 * (local_upper - local_lower)
    world_center = wp.quat_rotate(rotation, local_center) + position

    axis_x = wp.quat_rotate(rotation, wp.vec3(1.0, 0.0, 0.0))
    axis_y = wp.quat_rotate(rotation, wp.vec3(0.0, 1.0, 0.0))
    axis_z = wp.quat_rotate(rotation, wp.vec3(0.0, 0.0, 1.0))
    world_half = wp.vec3(
        wp.abs(axis_x[0]) * local_half[0] + wp.abs(axis_y[0]) * local_half[1] + wp.abs(axis_z[0]) * local_half[2],
        wp.abs(axis_x[1]) * local_half[0] + wp.abs(axis_y[1]) * local_half[1] + wp.abs(axis_z[1]) * local_half[2],
        wp.abs(axis_x[2]) * local_half[0] + wp.abs(axis_y[2]) * local_half[1] + wp.abs(axis_z[2]) * local_half[2],
    )
    margin_vector = wp.vec3(margin, margin, margin)
    shape_aabb_lower[shape_index] = world_center - world_half - margin_vector
    shape_aabb_upper[shape_index] = world_center + world_half + margin_vector


@wp.kernel
def create_soft_contacts_with_aabb(
    soft_rigid_contact_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[int],
    body_q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_mesh_properties: wp.array[wp.int32],
    shape_world: wp.array[int],
    margin: float,
    shape_margin: wp.array[float],
    soft_contact_max: int,
    shape_flags: wp.array[wp.int32],
    shape_heightfield_index: wp.array[wp.int32],
    heightfield_data: wp.array[HeightfieldData],
    heightfield_elevations: wp.array[wp.float32],
    shape_aabb_lower: wp.array[wp.vec3],
    shape_aabb_upper: wp.array[wp.vec3],
    # outputs
    soft_contact_count: wp.array[int],
    soft_contact_particle: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[int],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    soft_contact_tids: wp.array[int],
):
    """Create particle--shape contacts after a conservative AABB rejection."""
    tid = wp.tid()
    pair = soft_rigid_contact_pairs[tid]
    particle_index = pair[0]
    shape_index = pair[1]
    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return

    particle_world_id = particle_world[particle_index]
    shape_world_id = shape_world[shape_index]
    if particle_world_id != -1 and shape_world_id != -1 and particle_world_id != shape_world_id:
        return

    px = particle_q[particle_index]
    radius = particle_radius[particle_index]
    broad_phase_margin = margin + radius
    aabb_lower = shape_aabb_lower[shape_index]
    aabb_upper = shape_aabb_upper[shape_index]
    if (
        px[0] < aabb_lower[0] - broad_phase_margin
        or px[0] > aabb_upper[0] + broad_phase_margin
        or px[1] < aabb_lower[1] - broad_phase_margin
        or px[1] > aabb_upper[1] + broad_phase_margin
        or px[2] < aabb_lower[2] - broad_phase_margin
        or px[2] > aabb_upper[2] + broad_phase_margin
    ):
        return

    rigid_index = shape_body[shape_index]
    X_wb = wp.transform_identity()
    if rigid_index >= 0:
        X_wb = body_q[rigid_index]

    X_bs = shape_transform[shape_index]
    X_ws = wp.transform_multiply(X_wb, X_bs)
    X_sw = wp.transform_inverse(X_ws)
    x_local = wp.transform_point(X_sw, px)

    geo_type = shape_type[shape_index]
    geo_scale = shape_scale[shape_index]
    s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0

    d = 1.0e6
    n = wp.vec3()
    v = wp.vec3()

    if geo_type == GeoType.SPHERE:
        d = sdf_sphere(x_local, geo_scale[0])
        n = sdf_sphere_grad(x_local, geo_scale[0])

    if geo_type == GeoType.BOX:
        d = sdf_box(x_local, geo_scale[0], geo_scale[1], geo_scale[2])
        n = sdf_box_grad(x_local, geo_scale[0], geo_scale[1], geo_scale[2])

    if geo_type == GeoType.CAPSULE:
        d = sdf_capsule(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))
        n = sdf_capsule_grad(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))

    if geo_type == GeoType.CYLINDER:
        d = sdf_cylinder(x_local, geo_scale[0], geo_scale[1], int(Axis.Z), -1.0, geo_scale[2])
        n = sdf_cylinder_grad(x_local, geo_scale[0], geo_scale[1], int(Axis.Z), -1.0, geo_scale[2])

    if geo_type == GeoType.CONE:
        d = sdf_cone(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))
        n = sdf_cone_grad(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))

    if geo_type == GeoType.ELLIPSOID:
        d = sdf_ellipsoid(x_local, geo_scale)
        n = sdf_ellipsoid_grad(x_local, geo_scale)

    if geo_type == GeoType.MESH or geo_type == GeoType.CONVEX_MESH:
        mesh = shape_source_ptr[shape_index]
        min_scale = wp.min(wp.min(wp.abs(geo_scale[0]), wp.abs(geo_scale[1])), wp.abs(geo_scale[2]))
        query = mesh_query_point_sign(
            mesh,
            wp.cw_div(x_local, geo_scale),
            margin + s_margin / min_scale + radius / min_scale,
            resolve_mesh_sign_method(shape_mesh_properties[shape_index]),
        )
        if query.result:
            shape_p = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            shape_v = wp.mesh_eval_velocity(mesh, query.face, query.u, query.v)
            shape_p = wp.cw_mul(shape_p, geo_scale)
            shape_v = wp.cw_mul(shape_v, geo_scale)
            delta = x_local - shape_p
            d = wp.length(delta) * query.sign
            n = wp.normalize(delta) * query.sign
            v = shape_v

    if geo_type == GeoType.PLANE:
        d = sdf_plane(x_local, geo_scale[0] * 0.5, geo_scale[1] * 0.5)
        n = wp.vec3(0.0, 0.0, 1.0)

    if geo_type == GeoType.HFIELD:
        hfd = heightfield_data[shape_heightfield_index[shape_index]]
        d, n = sample_sdf_grad_heightfield(hfd, heightfield_elevations, x_local)

    if d < margin + s_margin + radius:
        index = counter_increment(soft_contact_count, 0, soft_contact_tids, tid)
        if index < soft_contact_max:
            body_pos = wp.transform_point(X_bs, x_local - n * d)
            body_vel = wp.transform_vector(X_bs, v)
            world_normal = wp.transform_vector(X_ws, n)

            soft_contact_shape[index] = shape_index
            soft_contact_body_pos[index] = body_pos
            soft_contact_body_vel[index] = body_vel
            soft_contact_particle[index] = particle_index
            soft_contact_indices[index] = wp.vec3i(particle_index, -1, -1)
            soft_contact_barycentric[index] = wp.vec3(1.0, 0.0, 0.0)
            soft_contact_normal[index] = world_normal


@wp.kernel
def create_speculative_soft_contacts_with_aabb(
    soft_rigid_contact_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    particle_flags: wp.array[wp.int32],
    particle_world: wp.array[int],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    shape_body: wp.array[int],
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_source_ptr: wp.array[wp.uint64],
    shape_mesh_properties: wp.array[wp.int32],
    shape_world: wp.array[int],
    shape_owner: wp.array[wp.int8],
    dt: float,
    max_speculative_distance: float,
    margin: float,
    shape_margin: wp.array[float],
    soft_contact_max: int,
    shape_flags: wp.array[wp.int32],
    shape_heightfield_index: wp.array[wp.int32],
    heightfield_data: wp.array[HeightfieldData],
    heightfield_elevations: wp.array[wp.float32],
    shape_aabb_lower: wp.array[wp.vec3],
    shape_aabb_upper: wp.array[wp.vec3],
    # outputs
    soft_contact_count: wp.array[int],
    soft_contact_particle: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[int],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    soft_contact_tids: wp.array[int],
):
    """Append only separated M-V pairs predicted to cross during ``dt``.

    Actual contacts are emitted by the normal collision pipeline.  Keeping the
    two predicates mutually exclusive preserves the fixed one-record-per-pair
    capacity and stable candidate TID used by contact warm starting.
    """
    tid = wp.tid()
    pair = soft_rigid_contact_pairs[tid]
    particle_index = pair[0]
    shape_index = pair[1]
    # OWNER_MUJOCO == 1.  Static and VBD-owned shapes keep the legacy path.
    if shape_owner[shape_index] != wp.int8(1):
        return
    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return
    if (shape_flags[shape_index] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return

    particle_world_id = particle_world[particle_index]
    shape_world_id = shape_world[shape_index]
    if particle_world_id != -1 and shape_world_id != -1 and particle_world_id != shape_world_id:
        return

    px = particle_q[particle_index]
    radius = particle_radius[particle_index]
    broad_phase_margin = margin + radius + max_speculative_distance
    aabb_lower = shape_aabb_lower[shape_index]
    aabb_upper = shape_aabb_upper[shape_index]
    if (
        px[0] < aabb_lower[0] - broad_phase_margin
        or px[0] > aabb_upper[0] + broad_phase_margin
        or px[1] < aabb_lower[1] - broad_phase_margin
        or px[1] > aabb_upper[1] + broad_phase_margin
        or px[2] < aabb_lower[2] - broad_phase_margin
        or px[2] > aabb_upper[2] + broad_phase_margin
    ):
        return

    rigid_index = shape_body[shape_index]
    X_wb = wp.transform_identity()
    if rigid_index >= 0:
        X_wb = body_q[rigid_index]

    X_bs = shape_transform[shape_index]
    X_ws = wp.transform_multiply(X_wb, X_bs)
    X_sw = wp.transform_inverse(X_ws)
    x_local = wp.transform_point(X_sw, px)

    geo_type = shape_type[shape_index]
    geo_scale = shape_scale[shape_index]
    s_margin = shape_margin[shape_index] if shape_margin.shape[0] > 0 else 0.0

    d = 1.0e6
    n = wp.vec3()
    v = wp.vec3()

    if geo_type == GeoType.SPHERE:
        d = sdf_sphere(x_local, geo_scale[0])
        n = sdf_sphere_grad(x_local, geo_scale[0])
    if geo_type == GeoType.BOX:
        d = sdf_box(x_local, geo_scale[0], geo_scale[1], geo_scale[2])
        n = sdf_box_grad(x_local, geo_scale[0], geo_scale[1], geo_scale[2])
    if geo_type == GeoType.CAPSULE:
        d = sdf_capsule(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))
        n = sdf_capsule_grad(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))
    if geo_type == GeoType.CYLINDER:
        d = sdf_cylinder(x_local, geo_scale[0], geo_scale[1], int(Axis.Z), -1.0, geo_scale[2])
        n = sdf_cylinder_grad(x_local, geo_scale[0], geo_scale[1], int(Axis.Z), -1.0, geo_scale[2])
    if geo_type == GeoType.CONE:
        d = sdf_cone(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))
        n = sdf_cone_grad(x_local, geo_scale[0], geo_scale[1], int(Axis.Z))
    if geo_type == GeoType.ELLIPSOID:
        d = sdf_ellipsoid(x_local, geo_scale)
        n = sdf_ellipsoid_grad(x_local, geo_scale)
    if geo_type == GeoType.MESH or geo_type == GeoType.CONVEX_MESH:
        mesh = shape_source_ptr[shape_index]
        min_scale = wp.min(wp.min(wp.abs(geo_scale[0]), wp.abs(geo_scale[1])), wp.abs(geo_scale[2]))
        query = mesh_query_point_sign(
            mesh,
            wp.cw_div(x_local, geo_scale),
            (margin + max_speculative_distance + s_margin + radius) / min_scale,
            resolve_mesh_sign_method(shape_mesh_properties[shape_index]),
        )
        if query.result:
            shape_p = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
            shape_v = wp.mesh_eval_velocity(mesh, query.face, query.u, query.v)
            shape_p = wp.cw_mul(shape_p, geo_scale)
            shape_v = wp.cw_mul(shape_v, geo_scale)
            delta = x_local - shape_p
            d = wp.length(delta) * query.sign
            n = wp.normalize(delta) * query.sign
            v = shape_v
    if geo_type == GeoType.PLANE:
        d = sdf_plane(x_local, geo_scale[0] * 0.5, geo_scale[1] * 0.5)
        n = wp.vec3(0.0, 0.0, 1.0)
    if geo_type == GeoType.HFIELD:
        hfd = heightfield_data[shape_heightfield_index[shape_index]]
        d, n = sample_sdf_grad_heightfield(hfd, heightfield_elevations, x_local)

    threshold = margin + s_margin + radius
    if d < threshold or d >= threshold + max_speculative_distance:
        return

    surface_local = x_local - n * d
    surface_world = wp.transform_point(X_ws, surface_local)
    world_normal = wp.transform_vector(X_ws, n)
    body_velocity = wp.transform_vector(X_ws, v)
    if rigid_index >= 0:
        body_twist = body_qd[rigid_index]
        com_world = wp.transform_point(X_wb, body_com[rigid_index])
        body_velocity += wp.spatial_top(body_twist) + wp.cross(wp.spatial_bottom(body_twist), surface_world - com_world)
    relative_normal_velocity = wp.dot(world_normal, particle_qd[particle_index] - body_velocity)
    extension = wp.min(max_speculative_distance, wp.max(0.0, -relative_normal_velocity * dt))
    if d >= threshold + extension:
        return

    index = counter_increment(soft_contact_count, 0, soft_contact_tids, tid)
    if index < soft_contact_max:
        soft_contact_shape[index] = shape_index
        soft_contact_body_pos[index] = wp.transform_point(X_bs, surface_local)
        soft_contact_body_vel[index] = wp.transform_vector(X_bs, v)
        soft_contact_particle[index] = particle_index
        soft_contact_indices[index] = wp.vec3i(particle_index, -1, -1)
        soft_contact_barycentric[index] = wp.vec3(1.0, 0.0, 0.0)
        soft_contact_normal[index] = world_normal

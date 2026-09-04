# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MJVBDV2-private full collision pipeline."""

from __future__ import annotations

import numpy as np
import warp as wp

from ...geometry.flags import ShapeFlags
from ...geometry.sdf_texture import TextureSDFData
from ...geometry.soft_contacts_sdf import (
    SDF_EDGE_ITERS,
    SDF_FACE_ITERS,
    SDF_LS_ITERS,
    _emit_soft_ef_contact,
    _is_analytic,
    _shape_frames,
    eval_shape_sdf,
    optimize_edge_sdf,
    optimize_face_sdf,
)
from ...sim import CollisionPipeline, Contacts, Model, State

__all__ = ["MJVBDV2CollisionPipeline"]


_COMPACT_SOFT_SURFACE_BLOCK_DIM = 128
_COMPACT_SOFT_SURFACE_BLOCKS_PER_SM = 2
_ENABLE_COMPACT_SOFT_SURFACE_PAIRS = True
_ENABLE_TEMPORAL_SOFT_FACE_CACHE = True
_SOFT_SURFACE_AABB_SAFETY_MARGIN = 1.0e-6
_SOFT_FACE_CACHE_BYTES_PER_PAIR = 13
_SOFT_FACE_CACHE_MAX_BYTES = 64 * 1024 * 1024
_SOFT_FACE_CACHE_CONTACT_SLOP = 5.0e-4
_SOFT_FACE_CACHE_REFINEMENT_ITERS = 2
_SOFT_FACE_CACHE_REUSE_LIMIT = 3
_SOFT_FACE_CACHE_STATIONARITY_TOLERANCE = 2.5e-4


def _shape_major_pairs(pairs: wp.array[wp.vec2i]) -> wp.array[wp.vec2i]:
    """Return the same candidate set grouped by rigid shape."""
    if pairs.shape[0] < 2:
        return pairs
    pair_data = np.asarray(pairs.numpy(), dtype=np.int32)
    order = np.argsort(pair_data[:, 1], kind="stable")
    return wp.array(pair_data[order], dtype=wp.vec2i, device=pairs.device)


@wp.func
def _aabb_overlap(
    lower_a: wp.vec3,
    upper_a: wp.vec3,
    lower_b: wp.vec3,
    upper_b: wp.vec3,
):
    return (
        lower_a[0] <= upper_b[0]
        and upper_a[0] >= lower_b[0]
        and lower_a[1] <= upper_b[1]
        and upper_a[1] >= lower_b[1]
        and lower_a[2] <= upper_b[2]
        and upper_a[2] >= lower_b[2]
    )


@wp.func
def _refine_cached_face_sdf(
    geo: wp.int32,
    scale: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    barycentric: wp.vec3,
    shape_sdf_index: wp.int32,
    texture_sdf_table: wp.array[TextureSDFData],
    ls_iters: wp.int32,
):
    """Refine a cached face point and evaluate its Frank-Wolfe stationarity gap."""
    for _i in range(_SOFT_FACE_CACHE_REFINEMENT_ITERS):
        x = barycentric[0] * a + barycentric[1] * b + barycentric[2] * c
        _phi_lower, _phi, grad = eval_shape_sdf(geo, scale, x, shape_sdf_index, texture_sdf_table)
        da = wp.dot(grad, a)
        db = wp.dot(grad, b)
        dc = wp.dot(grad, c)
        target_barycentric = wp.vec3(1.0, 0.0, 0.0)
        if db <= da and db <= dc:
            target_barycentric = wp.vec3(0.0, 1.0, 0.0)
        elif dc <= da and dc <= db:
            target_barycentric = wp.vec3(0.0, 0.0, 1.0)
        target = target_barycentric[0] * a + target_barycentric[1] * b + target_barycentric[2] * c
        gamma, _line_x, _line_phi, _line_grad = optimize_edge_sdf(
            geo,
            scale,
            x,
            target,
            shape_sdf_index,
            texture_sdf_table,
            ls_iters,
        )
        barycentric = (1.0 - gamma) * barycentric + gamma * target_barycentric

    x = barycentric[0] * a + barycentric[1] * b + barycentric[2] * c
    _phi_lower, phi, grad = eval_shape_sdf(geo, scale, x, shape_sdf_index, texture_sdf_table)
    linear_x = wp.dot(grad, x)
    linear_min = wp.min(wp.dot(grad, a), wp.min(wp.dot(grad, b), wp.dot(grad, c)))
    stationarity_gap = wp.max(linear_x - linear_min, 0.0)
    return barycentric, x, phi, grad, stationarity_gap


@wp.kernel(enable_backward=False)
def _mark_soft_edge_pairs_active(
    edge_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    edge_indices: wp.array2d[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_aabb_lower: wp.array[wp.vec3],
    shape_aabb_upper: wp.array[wp.vec3],
    shape_gap: wp.array[float],
    margin: float,
    compact_pairs: bool,
    compact_pair_indices: wp.array[wp.int32],
    compact_counts: wp.array[wp.int32],
    compact_count_index: wp.int32,
    active: wp.array[wp.uint8],
):
    tid = wp.tid()
    pair = edge_pairs[tid]
    edge = pair[0]
    shape = pair[1]
    if (shape_flags[shape] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        active[tid] = wp.uint8(0)
        return

    v0 = edge_indices[edge, 2]
    v1 = edge_indices[edge, 3]
    p = particle_q[v0]
    q = particle_q[v1]
    padding = margin + wp.max(particle_radius[v0], particle_radius[v1])
    padding_vector = wp.vec3(padding, padding, padding)
    # The shared rigid broad phase expands these AABBs by shape_gap, but
    # full-surface soft contact only uses shape_margin. Remove that unrelated
    # positive expansion while leaving a small conservative safety margin.
    aabb_shrink = wp.max(shape_gap[shape] - _SOFT_SURFACE_AABB_SAFETY_MARGIN, 0.0)
    shrink_vector = wp.vec3(aabb_shrink, aabb_shrink, aabb_shrink)
    is_active = _aabb_overlap(
        wp.min(p, q) - padding_vector,
        wp.max(p, q) + padding_vector,
        shape_aabb_lower[shape] + shrink_vector,
        shape_aabb_upper[shape] - shrink_vector,
    )
    active[tid] = wp.uint8(is_active)
    if compact_pairs and is_active:
        compact_index = wp.atomic_add(compact_counts, compact_count_index, 1)
        compact_pair_indices[compact_index] = tid


@wp.kernel(enable_backward=False)
def _mark_soft_face_pairs_active(
    face_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_aabb_lower: wp.array[wp.vec3],
    shape_aabb_upper: wp.array[wp.vec3],
    shape_gap: wp.array[float],
    margin: float,
    compact_pairs: bool,
    compact_pair_indices: wp.array[wp.int32],
    compact_counts: wp.array[wp.int32],
    compact_count_index: wp.int32,
    use_temporal_cache: bool,
    cache_state: wp.array[wp.uint8],
    active: wp.array[wp.uint8],
):
    tid = wp.tid()
    pair = face_pairs[tid]
    face = pair[0]
    shape = pair[1]
    if (shape_flags[shape] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        active[tid] = wp.uint8(0)
        if use_temporal_cache:
            cache_state[tid] = wp.uint8(0)
        return

    v0 = tri_indices[face, 0]
    v1 = tri_indices[face, 1]
    v2 = tri_indices[face, 2]
    p = particle_q[v0]
    q = particle_q[v1]
    r = particle_q[v2]
    padding = margin + wp.max(particle_radius[v0], wp.max(particle_radius[v1], particle_radius[v2]))
    padding_vector = wp.vec3(padding, padding, padding)
    aabb_shrink = wp.max(shape_gap[shape] - _SOFT_SURFACE_AABB_SAFETY_MARGIN, 0.0)
    shrink_vector = wp.vec3(aabb_shrink, aabb_shrink, aabb_shrink)
    is_active = _aabb_overlap(
        wp.min(p, wp.min(q, r)) - padding_vector,
        wp.max(p, wp.max(q, r)) + padding_vector,
        shape_aabb_lower[shape] + shrink_vector,
        shape_aabb_upper[shape] - shrink_vector,
    )
    active[tid] = wp.uint8(is_active)
    if use_temporal_cache and not is_active:
        cache_state[tid] = wp.uint8(0)
    if compact_pairs and is_active:
        compact_index = wp.atomic_add(compact_counts, compact_count_index, 1)
        compact_pair_indices[compact_index] = tid


@wp.func
def _create_soft_edge_contact_at_pair(
    tid: wp.int32,
    edge_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    edge_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    shape_margin: wp.array[float],
    sdf_edge_iters: wp.int32,
    margin: float,
    tid_base: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_tids: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    pair = edge_pairs[tid]
    edge = pair[0]
    shape = pair[1]
    if (shape_flags[shape] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        return
    geo = shape_type[shape]
    sdf_index = shape_sdf_index[shape]
    if (not _is_analytic(geo)) and sdf_index < 0:
        return

    v0 = edge_indices[edge, 2]
    v1 = edge_indices[edge, 3]
    radius = wp.max(particle_radius[v0], particle_radius[v1])
    X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape)
    p = wp.transform_point(X_sw, particle_q[v0])
    q = wp.transform_point(X_sw, particle_q[v1])
    scale = shape_scale[shape]
    shape_contact_margin = shape_margin[shape] if shape_margin.shape[0] > 0 else 0.0
    threshold = margin + shape_contact_margin + radius

    midpoint = 0.5 * (p + q)
    phi_midpoint, _phi_midpoint_accurate, _grad_midpoint = eval_shape_sdf(
        geo, scale, midpoint, sdf_index, texture_sdf_table
    )
    if phi_midpoint > threshold + 0.5 * wp.length(q - p):
        return

    u, x, phi, grad = optimize_edge_sdf(geo, scale, p, q, sdf_index, texture_sdf_table, sdf_edge_iters)
    if phi < threshold:
        y = x - phi * grad
        _emit_soft_ef_contact(
            tid,
            tid_base,
            soft_contact_max,
            soft_contact_count,
            soft_contact_tids,
            soft_contact_particle,
            soft_contact_indices,
            soft_contact_barycentric,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
            wp.vec3i(v0, v1, -1),
            wp.vec3(1.0 - u, u, 0.0),
            shape,
            wp.transform_point(X_bs, y),
            wp.vec3(0.0, 0.0, 0.0),
            wp.transform_vector(X_ws, grad),
        )


@wp.kernel
def _create_soft_edge_contacts(
    active: wp.array[wp.uint8],
    edge_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    edge_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    shape_margin: wp.array[float],
    sdf_edge_iters: wp.int32,
    margin: float,
    tid_base: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_tids: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    tid = wp.tid()
    if active[tid] == wp.uint8(0):
        return
    _create_soft_edge_contact_at_pair(
        tid,
        edge_pairs,
        particle_q,
        particle_radius,
        edge_indices,
        shape_body,
        shape_type,
        shape_flags,
        shape_transform,
        shape_scale,
        body_q,
        shape_sdf_index,
        texture_sdf_table,
        shape_margin,
        sdf_edge_iters,
        margin,
        tid_base,
        soft_contact_max,
        soft_contact_count,
        soft_contact_tids,
        soft_contact_particle,
        soft_contact_indices,
        soft_contact_barycentric,
        soft_contact_shape,
        soft_contact_body_pos,
        soft_contact_body_vel,
        soft_contact_normal,
    )


@wp.kernel(grid_stride=False)
def _create_compact_soft_edge_contacts(
    compact_pair_indices: wp.array[wp.int32],
    compact_counts: wp.array[wp.int32],
    compact_count_index: wp.int32,
    worker_count: wp.int32,
    edge_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    edge_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    shape_margin: wp.array[float],
    sdf_edge_iters: wp.int32,
    margin: float,
    tid_base: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_tids: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    worker = wp.tid()
    compact_index = worker
    compact_count = wp.min(compact_counts[compact_count_index], compact_pair_indices.shape[0])
    while compact_index < compact_count:
        tid = compact_pair_indices[compact_index]
        _create_soft_edge_contact_at_pair(
            tid,
            edge_pairs,
            particle_q,
            particle_radius,
            edge_indices,
            shape_body,
            shape_type,
            shape_flags,
            shape_transform,
            shape_scale,
            body_q,
            shape_sdf_index,
            texture_sdf_table,
            shape_margin,
            sdf_edge_iters,
            margin,
            tid_base,
            soft_contact_max,
            soft_contact_count,
            soft_contact_tids,
            soft_contact_particle,
            soft_contact_indices,
            soft_contact_barycentric,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
        )
        compact_index += worker_count


@wp.func
def _create_soft_face_contact_at_pair(
    tid: wp.int32,
    face_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    shape_margin: wp.array[float],
    sdf_face_iters: wp.int32,
    sdf_ls_iters: wp.int32,
    use_temporal_cache: bool,
    cached_barycentric: wp.array[wp.vec3],
    cache_state: wp.array[wp.uint8],
    margin: float,
    tid_base: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_tids: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    pair = face_pairs[tid]
    face = pair[0]
    shape = pair[1]
    if (shape_flags[shape] & ShapeFlags.COLLIDE_PARTICLES) == 0:
        if use_temporal_cache:
            cache_state[tid] = wp.uint8(0)
        return
    geo = shape_type[shape]
    sdf_index = shape_sdf_index[shape]
    if (not _is_analytic(geo)) and sdf_index < 0:
        if use_temporal_cache:
            cache_state[tid] = wp.uint8(0)
        return

    v0 = tri_indices[face, 0]
    v1 = tri_indices[face, 1]
    v2 = tri_indices[face, 2]
    radius = wp.max(particle_radius[v0], wp.max(particle_radius[v1], particle_radius[v2]))
    X_bs, X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape)
    p = wp.transform_point(X_sw, particle_q[v0])
    q = wp.transform_point(X_sw, particle_q[v1])
    r = wp.transform_point(X_sw, particle_q[v2])
    scale = shape_scale[shape]
    shape_contact_margin = shape_margin[shape] if shape_margin.shape[0] > 0 else 0.0
    threshold = margin + shape_contact_margin + radius

    centroid = (p + q + r) / 3.0
    phi_centroid, _phi_centroid_accurate, _grad_centroid = eval_shape_sdf(
        geo, scale, centroid, sdf_index, texture_sdf_table
    )
    reach = wp.max(wp.length(p - centroid), wp.max(wp.length(q - centroid), wp.length(r - centroid)))
    if phi_centroid > threshold + reach:
        if use_temporal_cache:
            cache_state[tid] = wp.uint8(0)
        return

    barycentric = wp.vec3(0.0)
    x = wp.vec3(0.0)
    phi = float(0.0)
    grad = wp.vec3(0.0)
    cache_hit = False
    if (
        use_temporal_cache
        and cache_state[tid] != wp.uint8(0)
        and cache_state[tid] <= wp.uint8(_SOFT_FACE_CACHE_REUSE_LIMIT)
        and not _is_analytic(geo)
    ):
        barycentric = cached_barycentric[tid]
        barycentric, x, phi, grad, stationarity_gap = _refine_cached_face_sdf(
            geo,
            scale,
            p,
            q,
            r,
            barycentric,
            sdf_index,
            texture_sdf_table,
            sdf_ls_iters,
        )
        cache_hit = (
            phi < threshold - _SOFT_FACE_CACHE_CONTACT_SLOP
            and stationarity_gap <= _SOFT_FACE_CACHE_STATIONARITY_TOLERANCE
        )
    if not cache_hit:
        barycentric, x, phi, grad = optimize_face_sdf(
            geo,
            scale,
            p,
            q,
            r,
            sdf_index,
            texture_sdf_table,
            sdf_face_iters,
            sdf_ls_iters,
        )
    if use_temporal_cache:
        cached_barycentric[tid] = barycentric
        if phi < threshold:
            cache_state[tid] = wp.uint8(1)
            if cache_hit:
                cache_state[tid] += wp.uint8(1)
        else:
            cache_state[tid] = wp.uint8(0)
    if phi < threshold:
        y = x - phi * grad
        _emit_soft_ef_contact(
            tid,
            tid_base,
            soft_contact_max,
            soft_contact_count,
            soft_contact_tids,
            soft_contact_particle,
            soft_contact_indices,
            soft_contact_barycentric,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
            wp.vec3i(v0, v1, v2),
            barycentric,
            shape,
            wp.transform_point(X_bs, y),
            wp.vec3(0.0, 0.0, 0.0),
            wp.transform_vector(X_ws, grad),
        )


@wp.kernel
def _create_soft_face_contacts(
    active: wp.array[wp.uint8],
    face_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    shape_margin: wp.array[float],
    sdf_face_iters: wp.int32,
    sdf_ls_iters: wp.int32,
    use_temporal_cache: bool,
    cached_barycentric: wp.array[wp.vec3],
    cache_state: wp.array[wp.uint8],
    margin: float,
    tid_base: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_tids: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    tid = wp.tid()
    if active[tid] == wp.uint8(0):
        return
    _create_soft_face_contact_at_pair(
        tid,
        face_pairs,
        particle_q,
        particle_radius,
        tri_indices,
        shape_body,
        shape_type,
        shape_flags,
        shape_transform,
        shape_scale,
        body_q,
        shape_sdf_index,
        texture_sdf_table,
        shape_margin,
        sdf_face_iters,
        sdf_ls_iters,
        use_temporal_cache,
        cached_barycentric,
        cache_state,
        margin,
        tid_base,
        soft_contact_max,
        soft_contact_count,
        soft_contact_tids,
        soft_contact_particle,
        soft_contact_indices,
        soft_contact_barycentric,
        soft_contact_shape,
        soft_contact_body_pos,
        soft_contact_body_vel,
        soft_contact_normal,
    )


@wp.kernel(grid_stride=False)
def _create_compact_soft_face_contacts(
    compact_pair_indices: wp.array[wp.int32],
    compact_counts: wp.array[wp.int32],
    compact_count_index: wp.int32,
    worker_count: wp.int32,
    face_pairs: wp.array[wp.vec2i],
    particle_q: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    tri_indices: wp.array2d[wp.int32],
    shape_body: wp.array[wp.int32],
    shape_type: wp.array[wp.int32],
    shape_flags: wp.array[wp.int32],
    shape_transform: wp.array[wp.transform],
    shape_scale: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    shape_sdf_index: wp.array[wp.int32],
    texture_sdf_table: wp.array[TextureSDFData],
    shape_margin: wp.array[float],
    sdf_face_iters: wp.int32,
    sdf_ls_iters: wp.int32,
    use_temporal_cache: bool,
    cached_barycentric: wp.array[wp.vec3],
    cache_state: wp.array[wp.uint8],
    margin: float,
    tid_base: wp.int32,
    soft_contact_max: wp.int32,
    soft_contact_count: wp.array[wp.int32],
    soft_contact_tids: wp.array[wp.int32],
    soft_contact_particle: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_barycentric: wp.array[wp.vec3],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_body_vel: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
):
    worker = wp.tid()
    compact_index = worker
    compact_count = wp.min(compact_counts[compact_count_index], compact_pair_indices.shape[0])
    while compact_index < compact_count:
        tid = compact_pair_indices[compact_index]
        _create_soft_face_contact_at_pair(
            tid,
            face_pairs,
            particle_q,
            particle_radius,
            tri_indices,
            shape_body,
            shape_type,
            shape_flags,
            shape_transform,
            shape_scale,
            body_q,
            shape_sdf_index,
            texture_sdf_table,
            shape_margin,
            sdf_face_iters,
            sdf_ls_iters,
            use_temporal_cache,
            cached_barycentric,
            cache_state,
            margin,
            tid_base,
            soft_contact_max,
            soft_contact_count,
            soft_contact_tids,
            soft_contact_particle,
            soft_contact_indices,
            soft_contact_barycentric,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_body_vel,
            soft_contact_normal,
        )
        compact_index += worker_count


def _launch_soft_surface_contacts(
    model: Model,
    state: State,
    contacts: Contacts,
    margin: float,
    edge_pairs: wp.array[wp.vec2i],
    face_pairs: wp.array[wp.vec2i],
    edge_active: wp.array[wp.uint8],
    face_active: wp.array[wp.uint8],
    edge_compact_pair_indices: wp.array[wp.int32],
    face_compact_pair_indices: wp.array[wp.int32],
    compact_counts: wp.array[wp.int32],
    use_compact_pairs: bool,
    edge_worker_count: int,
    face_worker_count: int,
    particle_pair_count: int,
    use_face_temporal_cache: bool,
    face_cached_barycentric: wp.array[wp.vec3],
    face_cache_state: wp.array[wp.uint8],
) -> None:
    """Generate edge and face contacts after conservative AABB pruning."""
    edge_pair_count = int(edge_pairs.shape[0])
    face_pair_count = int(face_pairs.shape[0])
    shape_args = [
        model.shape_body,
        model.shape_type,
        model.shape_flags,
        model.shape_transform,
        model.shape_scale,
        state.body_q,
        model._shape_sdf_index,
        model._texture_sdf_data,
        model.shape_margin,
    ]
    outputs = [
        contacts.soft_contact_count,
        contacts.soft_contact_tids,
        contacts.soft_contact_particle,
        contacts.soft_contact_indices,
        contacts.soft_contact_barycentric,
        contacts.soft_contact_shape,
        contacts.soft_contact_body_pos,
        contacts.soft_contact_body_vel,
        contacts.soft_contact_normal,
    ]

    if edge_pair_count > 0:
        if use_compact_pairs:
            wp.launch(
                _create_compact_soft_edge_contacts,
                dim=edge_worker_count,
                block_dim=_COMPACT_SOFT_SURFACE_BLOCK_DIM,
                inputs=[
                    edge_compact_pair_indices,
                    compact_counts,
                    0,
                    edge_worker_count,
                    edge_pairs,
                    state.particle_q,
                    model.particle_radius,
                    model.edge_indices,
                    *shape_args,
                    SDF_EDGE_ITERS,
                    margin,
                    particle_pair_count,
                    contacts.soft_contact_max,
                ],
                outputs=outputs,
                device=model.device,
            )
        else:
            wp.launch(
                _create_soft_edge_contacts,
                dim=edge_pair_count,
                inputs=[
                    edge_active,
                    edge_pairs,
                    state.particle_q,
                    model.particle_radius,
                    model.edge_indices,
                    *shape_args,
                    SDF_EDGE_ITERS,
                    margin,
                    particle_pair_count,
                    contacts.soft_contact_max,
                ],
                outputs=outputs,
                device=model.device,
            )
    if face_pair_count > 0:
        if use_compact_pairs:
            wp.launch(
                _create_compact_soft_face_contacts,
                dim=face_worker_count,
                block_dim=_COMPACT_SOFT_SURFACE_BLOCK_DIM,
                inputs=[
                    face_compact_pair_indices,
                    compact_counts,
                    1,
                    face_worker_count,
                    face_pairs,
                    state.particle_q,
                    model.particle_radius,
                    model.tri_indices,
                    *shape_args,
                    SDF_FACE_ITERS,
                    SDF_LS_ITERS,
                    use_face_temporal_cache,
                    face_cached_barycentric,
                    face_cache_state,
                    margin,
                    particle_pair_count + edge_pair_count,
                    contacts.soft_contact_max,
                ],
                outputs=outputs,
                device=model.device,
            )
        else:
            wp.launch(
                _create_soft_face_contacts,
                dim=face_pair_count,
                inputs=[
                    face_active,
                    face_pairs,
                    state.particle_q,
                    model.particle_radius,
                    model.tri_indices,
                    *shape_args,
                    SDF_FACE_ITERS,
                    SDF_LS_ITERS,
                    use_face_temporal_cache,
                    face_cached_barycentric,
                    face_cache_state,
                    margin,
                    particle_pair_count + edge_pair_count,
                    contacts.soft_contact_max,
                ],
                outputs=outputs,
                device=model.device,
            )


class MJVBDV2CollisionPipeline(CollisionPipeline):
    """Full collision pipeline with V2-local soft-surface broad rejection."""

    def __init__(self, model: Model, **kwargs: object):
        super().__init__(model, **kwargs)
        self._use_soft_surface_aabb = bool(
            model.device.is_cuda
            and self.enable_rigid_soft_full_surface_contact
            and (self.soft_edge_rigid_pairs.shape[0] > 0 or self.soft_face_rigid_pairs.shape[0] > 0)
        )
        if self._use_soft_surface_aabb:
            self.soft_edge_rigid_pairs = _shape_major_pairs(self.soft_edge_rigid_pairs)
            self.soft_face_rigid_pairs = _shape_major_pairs(self.soft_face_rigid_pairs)
        edge_mask_size = self.soft_edge_rigid_pairs.shape[0] if self._use_soft_surface_aabb else 0
        face_mask_size = self.soft_face_rigid_pairs.shape[0] if self._use_soft_surface_aabb else 0
        self._soft_edge_pair_active = wp.empty(edge_mask_size, dtype=wp.uint8, device=model.device)
        self._soft_face_pair_active = wp.empty(face_mask_size, dtype=wp.uint8, device=model.device)
        self._use_soft_surface_compaction = bool(
            _ENABLE_COMPACT_SOFT_SURFACE_PAIRS and self._use_soft_surface_aabb and not self.requires_grad
        )
        compact_count = 2 if self._use_soft_surface_compaction else 0
        edge_compact_size = edge_mask_size if self._use_soft_surface_compaction else 0
        face_compact_size = face_mask_size if self._use_soft_surface_compaction else 0
        self._soft_surface_compact_counts = wp.zeros(compact_count, dtype=wp.int32, device=model.device)
        self._soft_edge_compact_pair_indices = wp.empty(edge_compact_size, dtype=wp.int32, device=model.device)
        self._soft_face_compact_pair_indices = wp.empty(face_compact_size, dtype=wp.int32, device=model.device)
        self._use_soft_face_temporal_cache = bool(
            _ENABLE_TEMPORAL_SOFT_FACE_CACHE
            and self._use_soft_surface_aabb
            and not self.requires_grad
            and face_mask_size * _SOFT_FACE_CACHE_BYTES_PER_PAIR <= _SOFT_FACE_CACHE_MAX_BYTES
        )
        face_cache_size = face_mask_size if self._use_soft_face_temporal_cache else 0
        self._soft_face_cached_barycentric = wp.empty(face_cache_size, dtype=wp.vec3, device=model.device)
        self._soft_face_cache_state = wp.zeros(face_cache_size, dtype=wp.uint8, device=model.device)
        max_workers = (
            model.device.sm_count * _COMPACT_SOFT_SURFACE_BLOCKS_PER_SM * _COMPACT_SOFT_SURFACE_BLOCK_DIM
            if self._use_soft_surface_compaction
            else 0
        )
        self._soft_edge_compact_worker_count = min(edge_compact_size, max_workers)
        self._soft_face_compact_worker_count = min(face_compact_size, max_workers)

    def collide(
        self,
        state: State,
        contacts: Contacts,
        *,
        soft_contact_margin: float | None = None,
    ) -> None:
        """Run collision detection with private CUDA full-surface pruning."""
        if not self._use_soft_surface_aabb or not self.enable_rigid_soft_full_surface_contact:
            super().collide(state, contacts, soft_contact_margin=soft_contact_margin)
            return

        # The shared pipeline has no hook before its full-surface launch. Suppress
        # only that launch, then append the equivalent private masked pass.
        full_surface_enabled = self.enable_rigid_soft_full_surface_contact
        self.enable_rigid_soft_full_surface_contact = False
        try:
            super().collide(state, contacts, soft_contact_margin=soft_contact_margin)
        finally:
            self.enable_rigid_soft_full_surface_contact = full_surface_enabled

        contacts._enable_rigid_soft_full_surface_contact = True
        if not state.particle_q:
            return
        margin = self.soft_contact_margin if soft_contact_margin is None else soft_contact_margin
        model = self.model
        shape_aabb_lower = self.narrow_phase.shape_aabb_lower
        shape_aabb_upper = self.narrow_phase.shape_aabb_upper

        if self._use_soft_surface_compaction:
            self._soft_surface_compact_counts.zero_()

        if self.soft_edge_rigid_pairs.shape[0] > 0:
            wp.launch(
                _mark_soft_edge_pairs_active,
                dim=self.soft_edge_rigid_pairs.shape[0],
                inputs=[
                    self.soft_edge_rigid_pairs,
                    state.particle_q,
                    model.particle_radius,
                    model.edge_indices,
                    model.shape_flags,
                    shape_aabb_lower,
                    shape_aabb_upper,
                    model.shape_gap,
                    margin,
                    self._use_soft_surface_compaction,
                    self._soft_edge_compact_pair_indices,
                    self._soft_surface_compact_counts,
                    0,
                ],
                outputs=[self._soft_edge_pair_active],
                device=model.device,
                record_tape=False,
            )
        if self.soft_face_rigid_pairs.shape[0] > 0:
            wp.launch(
                _mark_soft_face_pairs_active,
                dim=self.soft_face_rigid_pairs.shape[0],
                inputs=[
                    self.soft_face_rigid_pairs,
                    state.particle_q,
                    model.particle_radius,
                    model.tri_indices,
                    model.shape_flags,
                    shape_aabb_lower,
                    shape_aabb_upper,
                    model.shape_gap,
                    margin,
                    self._use_soft_surface_compaction,
                    self._soft_face_compact_pair_indices,
                    self._soft_surface_compact_counts,
                    1,
                    self._use_soft_face_temporal_cache,
                    self._soft_face_cache_state,
                ],
                outputs=[self._soft_face_pair_active],
                device=model.device,
                record_tape=False,
            )
        _launch_soft_surface_contacts(
            model,
            state,
            contacts,
            margin,
            self.soft_edge_rigid_pairs,
            self.soft_face_rigid_pairs,
            self._soft_edge_pair_active,
            self._soft_face_pair_active,
            self._soft_edge_compact_pair_indices,
            self._soft_face_compact_pair_indices,
            self._soft_surface_compact_counts,
            self._use_soft_surface_compaction,
            self._soft_edge_compact_worker_count,
            self._soft_face_compact_worker_count,
            self.soft_rigid_contact_pair_count,
            self._use_soft_face_temporal_cache,
            self._soft_face_cached_barycentric,
            self._soft_face_cache_state,
        )

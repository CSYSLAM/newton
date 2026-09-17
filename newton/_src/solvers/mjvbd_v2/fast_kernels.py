# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cooperative contact and certified surface kernels for the CUDA fast path."""

import warp as wp

import newton._src.geometry.sdf_texture as texture
import newton._src.geometry.soft_contacts_sdf as soft
import newton._src.solvers.mjvbd_v2.vbd.particle_vbd_kernels as kernels
import newton._src.solvers.mjvbd_v2.vbd.rigid_vbd_kernels as rk
from newton._src.geometry.broad_phase_common import binary_search
from newton._src.geometry.flags import ShapeFlags
from newton._src.geometry.kernels import triangle_closest_point, vertex_adjacent_to_triangle
from newton._src.geometry.sdf_texture import TextureSDFData
from newton._src.geometry.soft_contacts_sdf import _emit_soft_ef_contact, _is_analytic, _shape_frames, eval_shape_sdf
from newton._src.solvers.mjvbd_v2.coupled_free_body.rigid_fusion import (
    RigidInputs,
    SoftInputs,
    SolveInputs,
    _solve_lane,
)
from newton._src.solvers.mjvbd_v2.particle_surface_cache import _evaluate_bending


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
    pending_ids: wp.array[wp.int32],
    pending_count: wp.array[wp.int32],
):
    pair = face_pairs[tid]
    face = pair[0]
    shape = pair[1]
    if shape_flags[shape] & ShapeFlags.COLLIDE_PARTICLES == 0:
        if use_temporal_cache:
            cache_state[tid] = wp.uint8(0)
        return
    geo = shape_type[shape]
    sdf_index = shape_sdf_index[shape]
    if not _is_analytic(geo) and sdf_index < 0:
        if use_temporal_cache:
            cache_state[tid] = wp.uint8(0)
        return
    v0 = tri_indices[face, 0]
    v1 = tri_indices[face, 1]
    v2 = tri_indices[face, 2]
    radius = wp.max(particle_radius[v0], wp.max(particle_radius[v1], particle_radius[v2]))
    _X_bs, _X_ws, X_sw = _shape_frames(shape_body, body_q, shape_transform, shape)
    p = wp.transform_point(X_sw, particle_q[v0])
    q = wp.transform_point(X_sw, particle_q[v1])
    r = wp.transform_point(X_sw, particle_q[v2])
    scale = shape_scale[shape]
    shape_contact_margin = shape_margin[shape] if shape_margin.shape[0] > 0 else 0.0
    threshold = margin + shape_contact_margin + radius
    centroid = (p + q + r) / 3.0
    phi_centroid = soft.eval_shape_sdf_lower_bound(geo, scale, centroid, sdf_index, texture_sdf_table)
    reach = wp.max(wp.length(p - centroid), wp.max(wp.length(q - centroid), wp.length(r - centroid)))
    if phi_centroid > threshold + reach:
        if use_temporal_cache:
            cache_state[tid] = wp.uint8(0)
        return
    bucket = int(2)
    if _is_analytic(geo):
        bucket = 0
    elif use_temporal_cache and cache_state[tid] != wp.uint8(0) and (cache_state[tid] <= wp.uint8(3)):
        bucket = 1
    slot = wp.atomic_add(pending_count, bucket, 1)
    pending_ids[bucket * face_pairs.shape[0] + slot] = tid


@wp.kernel(enable_backward=False, grid_stride=False)
def gather_face_pairs(
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
    pending_ids: wp.array[wp.int32],
    pending_count: wp.array[wp.int32],
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
            pending_ids,
            pending_count,
        )
        compact_index += worker_count


@wp.func_native("\n#if defined(__CUDA_ARCH__)\n    return threadIdx.x & 7u;\n#else\n    return 0;\n#endif\n")
def lane_id() -> int: ...


@wp.struct
class CellCache:
    valid: bool
    index: wp.vec3i
    base: wp.vec3i
    coarse: bool
    corners: texture.vec8f
    fractions: wp.vec3


@wp.func
def cached_value(sdf: texture.TextureSDFData, position: wp.vec3, cache: CellCache):
    clamped = wp.vec3(
        wp.clamp(position[0], sdf.sdf_box_lower[0], sdf.sdf_box_upper[0]),
        wp.clamp(position[1], sdf.sdf_box_lower[1], sdf.sdf_box_upper[1]),
        wp.clamp(position[2], sdf.sdf_box_lower[2], sdf.sdf_box_upper[2]),
    )
    diff_mag = wp.length(position - clamped)
    f = wp.cw_mul(clamped - sdf.sdf_box_lower, sdf.inv_sdf_dx)
    maximum = wp.vec3(
        float(sdf.coarse_texture.width - 1) * sdf.subgrid_size_f,
        float(sdf.coarse_texture.height - 1) * sdf.subgrid_size_f,
        float(sdf.coarse_texture.depth - 1) * sdf.subgrid_size_f,
    )
    bounded = wp.vec3(wp.clamp(f[0], 0.0, maximum[0]), wp.clamp(f[1], 0.0, maximum[1]), wp.clamp(f[2], 0.0, maximum[2]))
    hit = cache.valid
    for axis in range(3):
        lower = float(cache.index[axis])
        upper = lower + 1.0
        if cache.coarse:
            lower = float(cache.base[axis]) * sdf.subgrid_size_f
            upper = lower + sdf.subgrid_size_f
        if bounded[axis] < lower or (bounded[axis] >= upper and upper != maximum[axis]):
            hit = False
    tx, ty, tz = (float(0.0), float(0.0), float(0.0))
    if hit:
        tx = bounded[0] - float(cache.index[0])
        ty = bounded[1] - float(cache.index[1])
        tz = bounded[2] - float(cache.index[2])
        if cache.coarse:
            # A demoted subgrid shares all eight corners across its fine cells.
            coarse_f = bounded * sdf.fine_to_coarse
            tx = coarse_f[0] - float(cache.base[0])
            ty = coarse_f[1] - float(cache.base[1])
            tz = coarse_f[2] - float(cache.base[2])
    else:
        loc = texture._locate_cell(sdf, f)
        values, tx, ty, tz = texture._read_cell_corners(sdf, f)
        cache.index = wp.vec3i(loc.ix, loc.iy, loc.iz)
        cache.base = wp.vec3i(loc.x_base, loc.y_base, loc.z_base)
        cache.coarse = loc.start_slot >= texture.SLOT_LINEAR
        cache.corners = values
        cache.valid = True
    cache.fractions = wp.vec3(tx, ty, tz)
    value = texture._trilinear(cache.corners, tx, ty, tz)
    if diff_mag > 0.0:
        value += diff_mag
    return (value, cache)


@wp.func
def cached_shape_eval(
    geo: int, scale: wp.vec3, x: wp.vec3, index: int, table: wp.array[texture.TextureSDFData], cache: CellCache
):
    if soft._is_analytic(geo):
        lower, value, gradient = soft.eval_shape_sdf(geo, scale, x, index, table)
        return lower, value, gradient, cache
    sdf = table[index]
    position = x
    if not sdf.scale_baked:
        position = wp.cw_div(x, scale)
    value, cache = cached_value(sdf, position, cache)
    tx, ty, tz = cache.fractions[0], cache.fractions[1], cache.fractions[2]
    omtx, omty, omtz = 1.0 - tx, 1.0 - ty, 1.0 - tz
    v000, v100, v010, v110 = cache.corners[0], cache.corners[1], cache.corners[2], cache.corners[3]
    v001, v101, v011, v111 = cache.corners[4], cache.corners[5], cache.corners[6], cache.corners[7]
    gx = omty * omtz * (v100 - v000) + ty * omtz * (v110 - v010) + omty * tz * (v101 - v001) + ty * tz * (v111 - v011)
    gy = omtx * omtz * (v010 - v000) + tx * omtz * (v110 - v100) + omtx * tz * (v011 - v001) + tx * tz * (v111 - v101)
    gz = omtx * omty * (v001 - v000) + tx * omty * (v101 - v100) + omtx * ty * (v011 - v010) + tx * ty * (v111 - v110)
    gradient = wp.cw_mul(wp.vec3(gx, gy, gz), sdf.inv_sdf_dx)
    clamped = wp.vec3(
        wp.clamp(position[0], sdf.sdf_box_lower[0], sdf.sdf_box_upper[0]),
        wp.clamp(position[1], sdf.sdf_box_lower[1], sdf.sdf_box_upper[1]),
        wp.clamp(position[2], sdf.sdf_box_lower[2], sdf.sdf_box_upper[2]),
    )
    difference = position - clamped
    distance = wp.length(difference)
    if distance > 0.0:
        gradient = difference / distance
    if sdf.scale_baked:
        return value, value, gradient, cache
    inv_scale = wp.vec3(1.0 / scale[0], 1.0 / scale[1], 1.0 / scale[2])
    gradient_length = wp.length(gradient)
    if gradient_length > 0.0:
        gradient = gradient / gradient_length
    stretch = wp.length(wp.cw_mul(scale, gradient))
    min_scale = wp.min(wp.abs(scale))
    scaled_gradient = wp.cw_mul(gradient, inv_scale)
    scaled_length = wp.length(scaled_gradient)
    if scaled_length > 0.0:
        scaled_gradient = scaled_gradient / scaled_length
    else:
        scaled_gradient = gradient
    return value * min_scale, value * stretch, scaled_gradient, cache


@wp.func
def cached_lower(
    geo: int, scale: wp.vec3, x: wp.vec3, index: int, table: wp.array[texture.TextureSDFData], cache: CellCache
):
    value = float(0.0)
    if soft._is_analytic(geo):
        value = soft.eval_shape_sdf_lower_bound(geo, scale, x, index, table)
    else:
        sdf = table[index]
        if sdf.scale_baked:
            value, cache = cached_value(sdf, x, cache)
        else:
            value, cache = cached_value(sdf, wp.cw_div(x, scale), cache)
            value *= wp.min(wp.abs(scale))
    return (value, cache)


@wp.func_native(
    "\n#if defined(__CUDA_ARCH__)\n    return __shfl_sync(__activemask(), value, lane, 8);\n#else\n    return value;\n#endif\n"
)
def broadcast(value: float, lane: int) -> float: ...


@wp.func
def optimize_edge_sdf(
    geo: int,
    scale: wp.vec3,
    p: wp.vec3,
    q: wp.vec3,
    index: int,
    table: wp.array[TextureSDFData],
    n_iter: int,
    cell_cache: CellCache,
):
    lane = lane_id()
    inv_phi = float(0.6180339887498949)
    lo, hi = (float(0.0), float(1.0))
    c = hi - (hi - lo) * inv_phi
    d = lo + (hi - lo) * inv_phi
    u0 = c if lane % 2 == 0 else d
    value, cell_cache = cached_lower(geo, scale, (1.0 - u0) * p + u0 * q, index, table, cell_cache)
    fc, fd = (broadcast(value, 0), broadcast(value, 1))
    step = int(0)
    while step < n_iter:
        depth, path = (int(0), int(0))
        if lane >= 1 and lane <= 2:
            depth, path = (1, lane - 1)
        elif lane >= 3 and lane <= 6:
            depth, path = (2, lane - 3)
        sl, sh, sc, sd = (float(lo), float(hi), float(c), float(d))
        left = fc < fd
        candidate = float(0.0)
        for level in range(depth + 1):
            if level > 0:
                left = path >> depth - level & 1 == 0
            if left:
                sh = sd
                sd = sc
                sc = sh - (sh - sl) * inv_phi
                candidate = sc
            else:
                sl = sc
                sc = sd
                sd = sl + (sh - sl) * inv_phi
                candidate = sd
        sampled, cell_cache = cached_lower(geo, scale, (1.0 - candidate) * p + candidate * q, index, table, cell_cache)
        node = int(0)
        for _level in range(3):
            if step < n_iter:
                next_value = broadcast(sampled, node)
                if fc < fd:
                    hi = d
                    d = c
                    fd = fc
                    c = hi - (hi - lo) * inv_phi
                    fc = next_value
                else:
                    lo = c
                    c = d
                    fc = fd
                    d = lo + (hi - lo) * inv_phi
                    fd = next_value
                node = 2 * node + 1 + int(not fc < fd)
                step += 1
    u = 0.5 * (lo + hi)
    x = (1.0 - u) * p + u * q
    _lower, phi, grad, cell_cache = cached_shape_eval(geo, scale, x, index, table, cell_cache)
    return (u, x, phi, grad, cell_cache)


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
    cell_cache = CellCache()
    "Refine a cached face point and evaluate its Frank-Wolfe stationarity gap."
    for _i in range(2):
        x = barycentric[0] * a + barycentric[1] * b + barycentric[2] * c
        _phi_lower, _phi, grad, cell_cache = cached_shape_eval(
            geo, scale, x, shape_sdf_index, texture_sdf_table, cell_cache
        )
        da = wp.dot(grad, a)
        db = wp.dot(grad, b)
        dc = wp.dot(grad, c)
        target_barycentric = wp.vec3(1.0, 0.0, 0.0)
        if db <= da and db <= dc:
            target_barycentric = wp.vec3(0.0, 1.0, 0.0)
        elif dc <= da and dc <= db:
            target_barycentric = wp.vec3(0.0, 0.0, 1.0)
        target = target_barycentric[0] * a + target_barycentric[1] * b + target_barycentric[2] * c
        gamma, _line_x, _line_phi, _line_grad, cell_cache = optimize_edge_sdf(
            geo, scale, x, target, shape_sdf_index, texture_sdf_table, ls_iters, cell_cache
        )
        barycentric = (1.0 - gamma) * barycentric + gamma * target_barycentric
    x = barycentric[0] * a + barycentric[1] * b + barycentric[2] * c
    _phi_lower, phi, grad = eval_shape_sdf(geo, scale, x, shape_sdf_index, texture_sdf_table)
    linear_x = wp.dot(grad, x)
    linear_min = wp.min(wp.dot(grad, a), wp.min(wp.dot(grad, b), wp.dot(grad, c)))
    stationarity_gap = wp.max(linear_x - linear_min, 0.0)
    return (barycentric, x, phi, grad, stationarity_gap)


@wp.func
def optimize_face_sdf(
    geo: wp.int32,
    scale: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    shape_sdf_index: wp.int32,
    texture_sdf_table: wp.array[TextureSDFData],
    n_iter: wp.int32,
    ls_iter: wp.int32,
):
    cell_cache = CellCache()
    "argmin phi over the soft triangle by Frank-Wolfe on the barycentric simplex (Macklin sec. 3).\n\n    Each step picks the simplex vertex minimizing the linearized objective ``grad . corner`` (eq. 4)\n    and line-searches phi toward it with :func:`optimize_edge_sdf`. Fixed ``n_iter`` / ``ls_iter``\n    iterations -> graph-capturable. Returns ``(bary, x_local, phi, grad)``.\n    "
    bary = wp.vec3(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    for _i in range(n_iter):
        x = bary[0] * a + bary[1] * b + bary[2] * c
        _phi_l, _phi_x, grad, cell_cache = cached_shape_eval(
            geo, scale, x, shape_sdf_index, texture_sdf_table, cell_cache
        )
        da = wp.dot(grad, a)
        db = wp.dot(grad, b)
        dc = wp.dot(grad, c)
        s = wp.vec3(1.0, 0.0, 0.0)
        if db <= da and db <= dc:
            s = wp.vec3(0.0, 1.0, 0.0)
        elif dc <= da and dc <= db:
            s = wp.vec3(0.0, 0.0, 1.0)
        target = s[0] * a + s[1] * b + s[2] * c
        gamma, _lx, _lphi, _lgrad, cell_cache = optimize_edge_sdf(
            geo, scale, x, target, shape_sdf_index, texture_sdf_table, ls_iter, cell_cache
        )
        updated = (1.0 - gamma) * bary + gamma * s
        fixed = updated[0] == bary[0] and updated[1] == bary[1] and (updated[2] == bary[2])
        bary = updated
        if fixed:
            break
    x = bary[0] * a + bary[1] * b + bary[2] * c
    _phi_l, phi, grad = eval_shape_sdf(geo, scale, x, shape_sdf_index, texture_sdf_table)
    return (bary, x, phi, grad)


@wp.func
def _create_soft_face_contact_at_pair_1(
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
    if shape_flags[shape] & ShapeFlags.COLLIDE_PARTICLES == 0:
        if use_temporal_cache:
            if lane_id() == 0:
                cache_state[tid] = wp.uint8(0)
        return
    geo = shape_type[shape]
    sdf_index = shape_sdf_index[shape]
    if not _is_analytic(geo) and sdf_index < 0:
        if use_temporal_cache:
            if lane_id() == 0:
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
    # gather_face_pairs already applied this bound to the same frozen geometry.
    barycentric = wp.vec3(0.0)
    x = wp.vec3(0.0)
    phi = float(0.0)
    grad = wp.vec3(0.0)
    cache_hit = False
    if (
        use_temporal_cache
        and cache_state[tid] != wp.uint8(0)
        and (cache_state[tid] <= wp.uint8(3))
        and (not _is_analytic(geo))
    ):
        barycentric = cached_barycentric[tid]
        barycentric, x, phi, grad, stationarity_gap = _refine_cached_face_sdf(
            geo, scale, p, q, r, barycentric, sdf_index, texture_sdf_table, sdf_ls_iters
        )
        cache_hit = phi < threshold - 0.0005 and stationarity_gap <= 0.00025
    if not cache_hit:
        barycentric, x, phi, grad = optimize_face_sdf(
            geo, scale, p, q, r, sdf_index, texture_sdf_table, sdf_face_iters, sdf_ls_iters
        )
    if use_temporal_cache:
        if lane_id() == 0:
            cached_barycentric[tid] = barycentric
        if phi < threshold:
            if lane_id() == 0:
                cache_state[tid] = wp.uint8(1)
            if cache_hit:
                if lane_id() == 0:
                    cache_state[tid] += wp.uint8(1)
        elif lane_id() == 0:
            cache_state[tid] = wp.uint8(0)
    if phi < threshold:
        y = x - phi * grad
        if lane_id() == 0:
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


@wp.kernel(enable_backward=False, grid_stride=False)
def search_face_pairs(
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
    worker = wp.tid() // 8
    compact_index = worker
    compact_count = wp.min(compact_counts[compact_count_index], compact_pair_indices.shape[0])
    while compact_index < compact_count:
        tid = compact_pair_indices[compact_index]
        _create_soft_face_contact_at_pair_1(
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


@wp.kernel(enable_backward=False)
def pack_face_pairs(
    capacity: int, bucket_counts: wp.array[int], bucket_ids: wp.array[int], count: wp.array[int], ids: wp.array[int]
):
    first = bucket_counts[0]
    second = first + bucket_counts[1]
    total = second + bucket_counts[2]
    if wp.tid() == 0:
        count[0] = total
    for row in range(wp.tid(), total, 4096):
        source = row
        if row >= second:
            source = 2 * capacity + row - second
        elif row >= first:
            source = capacity + row - first
        ids[row] = bucket_ids[source]


@wp.kernel(enable_backward=False)
def certify_vertex_triangle(
    max_query_radius: float,
    min_query_radius: float,
    bvh_id: wp.uint64,
    bvh_group_roots: wp.array[wp.int32],
    pos: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    particle_world: wp.array[wp.int32],
    world_count: wp.int32,
    vertex_colliding_triangles_offsets: wp.array[wp.int32],
    vertex_colliding_triangles_buffer_sizes: wp.array[wp.int32],
    triangle_colliding_vertices_offsets: wp.array[wp.int32],
    triangle_colliding_vertices_buffer_sizes: wp.array[wp.int32],
    vertex_triangle_filtering_list: wp.array[wp.int32],
    vertex_triangle_filtering_list_offsets: wp.array[wp.int32],
    min_distance_filtering_ref_pos: wp.array[wp.vec3],
    vertex_colliding_triangles: wp.array[wp.int32],
    vertex_colliding_triangles_count: wp.array[wp.int32],
    vertex_colliding_triangles_min_dist: wp.array[float],
    triangle_colliding_vertices: wp.array[wp.int32],
    triangle_colliding_vertices_count: wp.array[wp.int32],
    triangle_colliding_vertices_min_dist: wp.array[float],
    resize_flags: wp.array[wp.int32],
    certificate_active: wp.array[int],
    certificate_clearance: wp.array[float],
    original_query_radius: float,
):
    """Bound separation for eligible pairs in an expanded BVH query.

    Witnesses propose separating planes; they are not assumed to be exact
    distance minima. Preserve the original topology, world, and rest filters.
    Only certificate outputs are modified, never the input contact buffers.
    """
    if certificate_active[0] != 0:
        return
    v_index = wp.tid()
    v = pos[v_index]
    vertex_buffer_offset = vertex_colliding_triangles_offsets[v_index]
    _vertex_buffer_size = vertex_colliding_triangles_offsets[v_index + 1] - vertex_buffer_offset
    lower = wp.vec3(v[0] - max_query_radius, v[1] - max_query_radius, v[2] - max_query_radius)
    upper = wp.vec3(v[0] + max_query_radius, v[1] + max_query_radius, v[2] + max_query_radius)
    tri_index = wp.int32(0)
    _vertex_num_collisions = wp.int32(0)
    _min_dis_to_tris = max_query_radius
    vertex_world = particle_world[v_index]
    for query_pass in range(2):
        run_query = bool(False)
        query_all = bool(False)
        group_root = wp.int32(-1)
        if vertex_world < 0:
            if query_pass == 0:
                run_query = True
                query_all = True
        else:
            if query_pass == 0:
                group_root = bvh_group_roots[vertex_world]
            else:
                group_root = bvh_group_roots[world_count]
            run_query = group_root >= 0
        if run_query:
            if query_all:
                query = wp.bvh_query_aabb(bvh_id, lower, upper)
            else:
                query = wp.bvh_query_aabb(bvh_id, lower, upper, group_root)
            tri_index = wp.int32(0)
            while wp.bvh_query_next(query, tri_index):
                t1 = tri_indices[tri_index, 0]
                t2 = tri_indices[tri_index, 1]
                t3 = tri_indices[tri_index, 2]
                if vertex_adjacent_to_triangle(v_index, t1, t2, t3):
                    continue
                if vertex_triangle_filtering_list:
                    fl_start = vertex_triangle_filtering_list_offsets[v_index]
                    fl_end = vertex_triangle_filtering_list_offsets[v_index + 1]
                    if fl_end > fl_start:
                        first_val = vertex_triangle_filtering_list[fl_start]
                        last_val = vertex_triangle_filtering_list[fl_end - 1]
                        if tri_index >= first_val and tri_index <= last_val:
                            idx = binary_search(vertex_triangle_filtering_list, tri_index, fl_start, fl_end)
                            if idx > fl_start and vertex_triangle_filtering_list[idx - 1] == tri_index:
                                continue
                u1 = pos[t1]
                u2 = pos[t2]
                u3 = pos[t3]
                closest_p, _bary, _feature_type = triangle_closest_point(u1, u2, u3, v)
                if min_distance_filtering_ref_pos and min_query_radius > 0.0:
                    closest_p_ref, _, __ = triangle_closest_point(
                        min_distance_filtering_ref_pos[t1],
                        min_distance_filtering_ref_pos[t2],
                        min_distance_filtering_ref_pos[t3],
                        min_distance_filtering_ref_pos[v_index],
                    )
                    dist_ref = wp.length(closest_p_ref - min_distance_filtering_ref_pos[v_index])
                    if dist_ref < min_query_radius:
                        continue
                direction = closest_p - v
                length = wp.length(direction)
                if length > 1e-20:
                    normal = direction * (0.999999 / length)
                    separation = wp.min(wp.dot(u1 - v, normal), wp.min(wp.dot(u2 - v, normal), wp.dot(u3 - v, normal)))
                    magnitude = wp.max(wp.max(wp.abs(v)), wp.max(wp.abs(u1)))
                    magnitude = wp.max(magnitude, wp.max(wp.max(wp.abs(u2)), wp.max(wp.abs(u3))))
                    rounding = 2e-06 * wp.max(0.001, magnitude)
                    if separation > max_query_radius + rounding:
                        continue
                    clearance = separation - original_query_radius - rounding
                    if clearance > 0.0:
                        wp.atomic_min(certificate_clearance, 0, clearance)
                        continue
                wp.atomic_max(certificate_active, 0, 1)
                wp.atomic_min(certificate_clearance, 0, 0.0)
                return


@wp.kernel(enable_backward=False)
def certify_edge_edge(
    max_query_radius: float,
    min_query_radius: float,
    bvh_id: wp.uint64,
    bvh_group_roots: wp.array[wp.int32],
    pos: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    particle_world: wp.array[wp.int32],
    world_count: wp.int32,
    edge_colliding_edges_offsets: wp.array[wp.int32],
    edge_colliding_edges_buffer_sizes: wp.array[wp.int32],
    edge_edge_parallel_epsilon: float,
    edge_filtering_list: wp.array[wp.int32],
    edge_filtering_list_offsets: wp.array[wp.int32],
    min_distance_filtering_ref_pos: wp.array[wp.vec3],
    edge_colliding_edges: wp.array[wp.int32],
    edge_colliding_edges_count: wp.array[wp.int32],
    edge_colliding_edges_min_dist: wp.array[float],
    resize_flags: wp.array[wp.int32],
    certificate_active: wp.array[int],
    certificate_clearance: wp.array[float],
    original_query_radius: float,
):
    """Bound separation for eligible pairs in an expanded BVH query.

    Witnesses propose separating planes; they are not assumed to be exact
    distance minima. Preserve the original topology, world, and rest filters.
    Only certificate outputs are modified, never the input contact buffers.
    """
    if certificate_active[0] != 0:
        return
    e_index = wp.tid()
    e0_v0 = edge_indices[e_index, 2]
    e0_v1 = edge_indices[e_index, 3]
    e0_v0_pos = pos[e0_v0]
    e0_v1_pos = pos[e0_v1]
    lower = wp.min(e0_v0_pos, e0_v1_pos)
    upper = wp.max(e0_v0_pos, e0_v1_pos)
    lower = wp.vec3(lower[0] - max_query_radius, lower[1] - max_query_radius, lower[2] - max_query_radius)
    upper = wp.vec3(upper[0] + max_query_radius, upper[1] + max_query_radius, upper[2] + max_query_radius)
    colliding_edge_index = wp.int32(0)
    _edge_num_collisions = wp.int32(0)
    _min_dis_to_edges = max_query_radius
    edge_world = particle_world[e0_v0]
    for query_pass in range(2):
        run_query = bool(False)
        query_all = bool(False)
        group_root = wp.int32(-1)
        if edge_world < 0:
            if query_pass == 0:
                run_query = True
                query_all = True
        else:
            if query_pass == 0:
                group_root = bvh_group_roots[edge_world]
            else:
                group_root = bvh_group_roots[world_count]
            run_query = group_root >= 0
        if run_query:
            if query_all:
                query = wp.bvh_query_aabb(bvh_id, lower, upper)
            else:
                query = wp.bvh_query_aabb(bvh_id, lower, upper, group_root)
            colliding_edge_index = wp.int32(0)
            while wp.bvh_query_next(query, colliding_edge_index):
                e1_v0 = edge_indices[colliding_edge_index, 2]
                e1_v1 = edge_indices[colliding_edge_index, 3]
                if e0_v0 == e1_v0 or e0_v0 == e1_v1 or e0_v1 == e1_v0 or (e0_v1 == e1_v1):
                    continue
                if edge_filtering_list:
                    fl_start = edge_filtering_list_offsets[e_index]
                    fl_end = edge_filtering_list_offsets[e_index + 1]
                    if fl_end > fl_start:
                        first_val = edge_filtering_list[fl_start]
                        last_val = edge_filtering_list[fl_end - 1]
                        if colliding_edge_index >= first_val and colliding_edge_index <= last_val:
                            idx = binary_search(edge_filtering_list, colliding_edge_index, fl_start, fl_end)
                            if idx > fl_start and edge_filtering_list[idx - 1] == colliding_edge_index:
                                continue
                e1_v0_pos = pos[e1_v0]
                e1_v1_pos = pos[e1_v1]
                std = wp.closest_point_edge_edge(e0_v0_pos, e0_v1_pos, e1_v0_pos, e1_v1_pos, edge_edge_parallel_epsilon)
                if min_distance_filtering_ref_pos and min_query_radius > 0.0:
                    e0_v0_pos_ref = min_distance_filtering_ref_pos[e0_v0]
                    e0_v1_pos_ref = min_distance_filtering_ref_pos[e0_v1]
                    e1_v0_pos_ref = min_distance_filtering_ref_pos[e1_v0]
                    e1_v1_pos_ref = min_distance_filtering_ref_pos[e1_v1]
                    std_ref = wp.closest_point_edge_edge(
                        e0_v0_pos_ref, e0_v1_pos_ref, e1_v0_pos_ref, e1_v1_pos_ref, edge_edge_parallel_epsilon
                    )
                    dist_ref = std_ref[2]
                    if dist_ref < min_query_radius:
                        continue
                closest0 = (1.0 - std[0]) * e0_v0_pos + std[0] * e0_v1_pos
                closest1 = (1.0 - std[1]) * e1_v0_pos + std[1] * e1_v1_pos
                direction = closest1 - closest0
                length = wp.length(direction)
                if length > 1e-20:
                    normal = direction * (0.999999 / length)
                    a1 = wp.dot(normal, e0_v1_pos - e0_v0_pos)
                    b0 = wp.dot(normal, e1_v0_pos - e0_v0_pos)
                    b1 = wp.dot(normal, e1_v1_pos - e0_v0_pos)
                    separation = wp.max(wp.min(b0, b1) - wp.max(0.0, a1), wp.min(0.0, a1) - wp.max(b0, b1))
                    magnitude = wp.max(wp.max(wp.abs(e0_v0_pos)), wp.max(wp.abs(e0_v1_pos)))
                    magnitude = wp.max(magnitude, wp.max(wp.max(wp.abs(e1_v0_pos)), wp.max(wp.abs(e1_v1_pos))))
                    rounding = 2e-06 * wp.max(0.001, magnitude)
                    if separation > max_query_radius + rounding:
                        continue
                    clearance = separation - original_query_radius - rounding
                    if clearance > 0.0:
                        wp.atomic_min(certificate_clearance, 0, clearance)
                        continue
                wp.atomic_max(certificate_active, 0, 1)
                wp.atomic_min(certificate_clearance, 0, 0.0)
                return


@wp.kernel(enable_backward=False)
def need_query(
    q: wp.array[wp.vec3],
    reference: wp.array[wp.vec3],
    valid: wp.array[int],
    clearance: wp.array[float],
    needed: wp.array[int],
):
    i = wp.tid()
    delta = wp.length(q[i] - reference[i])
    finite = wp.isfinite(q[i][0]) and wp.isfinite(q[i][1]) and wp.isfinite(q[i][2])
    if valid[0] == 0 or not finite or (not wp.isfinite(delta)) or (delta >= 0.49 * clearance[0]):
        wp.atomic_max(needed, 0, 1)


@wp.kernel(enable_backward=False)
def save_certificate(active: wp.array[int], valid: wp.array[int], statistics: wp.array[int]):
    valid[0] = int(active[0] == 0)
    statistics[0] += 1
    statistics[1] += valid[0]


@wp.kernel(enable_backward=False)
def record_reuse(statistics: wp.array[int]):
    statistics[2] += 1


@wp.kernel()
def detect_self_contact(info_array: wp.array[kernels.TriMeshCollisionInfo], active: wp.array[int]):
    i = wp.tid()
    info = info_array[0]
    if i < info.edge_colliding_edges_count.shape[0] and info.edge_colliding_edges_count[i] > 0:
        wp.atomic_max(active, 0, 1)
    if i < info.vertex_colliding_triangles_count.shape[0] and info.vertex_colliding_triangles_count[i] > 0:
        wp.atomic_max(active, 0, 1)


@wp.kernel()
def build_adjacency(
    count: wp.array[int],
    maximum: int,
    corners: wp.array[wp.vec3i],
    colors: wp.array[int],
    counts: wp.array[int],
    entries: wp.array[int],
    fallback: wp.array[int],
):
    capacity = entries.shape[0] // counts.shape[0]
    for row in range(wp.tid(), wp.min(count[0], maximum), wp.min(maximum, 4096)):
        ids = corners[row]
        for j in range(3):
            p = ids[j]
            if p >= 0:
                slot = wp.atomic_add(counts, p, 1)
                if slot < capacity:
                    entries[p * capacity + slot] = row * 3 + j
                else:
                    wp.atomic_max(fallback, 0, 1)
                for other in range(j):
                    q = ids[other]
                    if q >= 0 and colors[p] == colors[q]:
                        wp.atomic_max(fallback, 0, 1)


@wp.func
def contact(s: SoftInputs, row: int, corner: int):
    ids = s.soft_contact_indices[row]
    f = wp.vec3(0.0)
    h = wp.mat33(0.0)
    if ids[1] < 0:
        p = ids[0]
        f, h = kernels._eval_body_particle_contact(
            p,
            s.particle_q[p],
            s.particle_q_prev[p],
            row,
            s.body_particle_contact_penalty_k[row],
            s.body_particle_contact_material_kd[row],
            s.body_particle_contact_material_mu[row],
            s.friction_epsilon,
            s.particle_radius,
            s.shape_body,
            s.body_q,
            s.body_q_prev,
            s.body_qd,
            s.body_com,
            s.body_particle_contact_shape,
            s.body_particle_contact_body_pos,
            s.body_particle_contact_body_vel,
            s.body_particle_contact_normal,
            s.shape_margin,
            s.dt,
        )
    else:
        bary = s.soft_contact_barycentric[row]
        ef, eh, _cp = kernels._eval_soft_ef_contact(
            row,
            ids,
            bary,
            s.particle_q,
            s.particle_q_prev,
            s.particle_radius,
            s.body_particle_contact_penalty_k[row],
            s.body_particle_contact_material_kd[row],
            s.body_particle_contact_material_mu[row],
            s.friction_epsilon,
            s.shape_body,
            s.body_q,
            s.body_q_prev,
            s.body_qd,
            s.body_com,
            s.body_particle_contact_shape,
            s.body_particle_contact_body_pos,
            s.body_particle_contact_body_vel,
            s.body_particle_contact_normal,
            s.shape_margin,
            s.dt,
        )
        f = bary[corner] * ef
        h = bary[corner] * bary[corner] * eh
    return (f, h)


@wp.func
def _commit_empty_contact_step(
    p: int, delta: wp.array[wp.vec3], anchor: wp.array[wp.vec3], limit: float, out: wp.array[wp.vec3]
):
    d = delta[p]
    length = wp.length(d)
    if length > limit:
        d = d * limit / length
    delta[p] = d
    out[p] = anchor[p] + d


@wp.kernel(enable_backward=False)
def solve_surface_fused(
    dt: float,
    particle_ids_in_color: wp.array[int],
    pos_prev: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    mass: wp.array[float],
    inertia: wp.array[wp.vec3],
    particle_flags: wp.array[int],
    tri_indices: wp.array2d[int],
    tri_poses: wp.array[wp.mat22],
    tri_materials: wp.array2d[float],
    tri_areas: wp.array[float],
    edge_indices: wp.array2d[int],
    edge_rest_angles: wp.array[float],
    edge_rest_length: wp.array[float],
    edge_bending_properties: wp.array2d[float],
    particle_adjacency: kernels.MeshAdjacencyData,
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
    skip_active_checks: int,
    skip_material_checks: int,
    contact_free_relaxation: float,
    particle_displacements: wp.array[wp.vec3],
    s: SoftInputs,
    counts: wp.array[int],
    entries: wp.array[int],
    anchor: wp.array[wp.vec3],
    limit: float,
    anchor_angles: wp.array[wp.vec2],
    out: wp.array[wp.vec3],
    colors: wp.array[int],
    color: int,
):
    tid = wp.tid()
    elastic_threads = particle_ids_in_color.shape[0] * 16
    if tid >= elastic_threads:
        p = tid - elastic_threads
        if p < pos.shape[0] and colors[p] != color:
            _commit_empty_contact_step(p, particle_displacements, anchor, limit, out)
        return
    lane = tid % 16
    p = particle_ids_in_color[tid // 16]
    cf = wp.vec3(0.0)
    ch = wp.mat33(0.0)
    capacity = entries.shape[0] // counts.shape[0]
    for index in range(lane, counts[p], 16):
        code = entries[p * capacity + index]
        ff, hh = contact(s, code // 3, code % 3)
        cf += ff
        ch += hh
    cforce = wp.tile_reduce(wp.add, wp.tile(cf, preserve_type=True))[0]
    chess = wp.tile_reduce(wp.add, wp.tile(ch, preserve_type=True))[0]
    f = wp.vec3(0.0)
    h = wp.mat33(0.0)
    faces = kernels.get_vertex_num_adjacent_faces(particle_adjacency, p)
    for adj in range(lane, faces, 16):
        tri, order = kernels.get_vertex_adjacent_face_id_order(particle_adjacency, p, adj)
        if skip_material_checks == 1 or tri_materials[tri, 0] > 0.0 or tri_materials[tri, 1] > 0.0:
            ff, hh = kernels.evaluate_neo_hookean_membrane_force_hessian(
                tri,
                order,
                pos,
                pos_prev,
                tri_indices,
                tri_poses[tri],
                tri_areas[tri],
                tri_materials[tri, 0],
                tri_materials[tri, 1],
                tri_materials[tri, 2],
                dt,
            )
            f += ff
            h += hh
    edges = kernels.get_vertex_num_adjacent_edges(particle_adjacency, p)
    for adj in range(lane, edges, 16):
        edge, order = kernels.get_vertex_adjacent_edge_id_order(particle_adjacency, p, adj)
        if skip_material_checks == 1 or edge_bending_properties[edge, 0] > 0.0:
            ff, hh = _evaluate_bending(
                edge,
                order,
                pos,
                edge_indices,
                edge_rest_angles,
                edge_rest_length,
                edge_bending_properties[edge, 0],
                edge_bending_properties[edge, 1],
                dt,
                anchor_angles[edge],
            )
            f += ff
            h += hh
    force = wp.tile_reduce(wp.add, wp.tile(f, preserve_type=True))[0]
    hess = wp.tile_reduce(wp.add, wp.tile(h, preserve_type=True))[0]
    if lane == 0:
        particle_forces[p] = cforce
        particle_hessians[p] = chess
        inv_dt_sq = 1.0 / (dt * dt)
        hess += mass[p] * inv_dt_sq * wp.identity(n=3, dtype=float) + chess
        delta = particle_displacements[p]
        if wp.abs(wp.determinant(hess)) > 1e-08:
            force += mass[p] * (inertia[p] - pos[p]) * inv_dt_sq + cforce
            step = wp.inverse(hess) * force
            if contact_free_relaxation != 1.0 and wp.ddot(chess, chess) == 0.0:
                step *= contact_free_relaxation
            delta += step
        particle_displacements[p] = delta
        _commit_empty_contact_step(p, particle_displacements, anchor, limit, out)


@wp.func
def _soft_lane(
    tid: int,
    dt: float,
    color_group: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    body_q_prev: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    shape_body: wp.array[int],
    friction_epsilon: float,
    body_particle_contact_penalty_k: wp.array[float],
    body_particle_contact_material_ke: wp.array[float],
    body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float],
    body_particle_contact_count: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    body_particle_contact_shape: wp.array[int],
    body_particle_contact_body_pos: wp.array[wp.vec3],
    body_particle_contact_body_vel: wp.array[wp.vec3],
    body_particle_contact_normal: wp.array[wp.vec3],
    soft_contact_barycentric: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    body_particle_contact_buffer_pre_alloc: int,
    body_particle_contact_counts: wp.array[wp.int32],
    body_particle_contact_indices: wp.array[wp.int32],
    dense_contact_threshold: int,
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
):
    """
    Per-body accumulation of body-particle soft contact forces and Hessians on rigid bodies.

    Handles both contact kinds from one per-body adjacency list, dispatching on each record's
    -1-padded ``soft_contact_indices``: a particle record ``(p, -1, -1)`` resolves single-particle
    geometry inline; an edge/face record evaluates the barycentric contact point over its 2-3 soft
    particles via ``rk._eval_soft_ef_contact``. Both apply the shared force law
    ``rk._compute_body_particle_contact_force`` and the equal-and-opposite body reaction. Body surface
    velocity uses the displacement-based path (body_q_prev).

    Notes:
      - Only dynamic bodies (inv_mass > 0) are updated.
      - Hessian contributions are accumulated into body_hessian_ll/al/aa.
      - Uses per-contact effective penalty/material parameters initialized once per step.
    """
    body_idx_in_group = tid // 64
    thread_id_within_body = tid % 64
    if body_idx_in_group >= color_group.shape[0]:
        return (wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0))
    body_id = color_group[body_idx_in_group]
    if body_inv_mass[body_id] <= 0.0:
        return (wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0))
    num_contacts = body_particle_contact_counts[body_id]
    if num_contacts > body_particle_contact_buffer_pre_alloc:
        num_contacts = body_particle_contact_buffer_pre_alloc
    if dense_contact_threshold > 0 and num_contacts >= dense_contact_threshold:
        return (wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0))
    max_contacts = body_particle_contact_count[0]
    X_wb = body_q[body_id]
    X_wb_prev = body_q_prev[body_id]
    com_world = wp.transform_point(X_wb, body_com[body_id])
    force_acc = wp.vec3(0.0)
    torque_acc = wp.vec3(0.0)
    h_ll_acc = wp.mat33(0.0)
    h_al_acc = wp.mat33(0.0)
    h_aa_acc = wp.mat33(0.0)
    i = thread_id_within_body
    while i < num_contacts:
        contact_idx = body_particle_contact_indices[body_id * body_particle_contact_buffer_pre_alloc + i]
        i += 64
        if contact_idx >= max_contacts:
            continue
        force, torque, h_ll, h_al, h_aa = rk._evaluate_body_particle_contact_reaction(
            dt,
            contact_idx,
            X_wb,
            X_wb_prev,
            com_world,
            particle_q,
            particle_q_prev,
            particle_radius,
            body_q_prev,
            body_q,
            body_qd,
            body_com,
            shape_body,
            friction_epsilon,
            body_particle_contact_penalty_k,
            body_particle_contact_material_kd,
            body_particle_contact_material_mu,
            soft_contact_indices,
            body_particle_contact_shape,
            body_particle_contact_body_pos,
            body_particle_contact_body_vel,
            body_particle_contact_normal,
            soft_contact_barycentric,
            shape_margin,
        )
        force_acc += force
        torque_acc += torque
        h_ll_acc += h_ll
        h_al_acc += h_al
        h_aa_acc += h_aa
    return (force_acc, torque_acc, h_ll_acc, h_al_acc, h_aa_acc)


@wp.func
def _rigid_lane(
    tid: int,
    dt: float,
    color_group: wp.array[wp.int32],
    body_q_prev: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_colors: wp.array[int],
    friction_epsilon: float,
    contact_penalty_k: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    avbd_alpha: float,
    hard_contacts: int,
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    shape_body: wp.array[wp.int32],
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array[wp.int32],
    body_contact_indices: wp.array[wp.int32],
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
):
    """
    Per-body augmented-Lagrangian contact accumulation with rk._NUM_RIGID_CONTACT_THREADS_PER_BODY strided threads.
    """
    body_idx_in_group = tid // 64
    thread_id_within_body = tid % 64
    if body_idx_in_group >= color_group.shape[0]:
        return (wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0))
    body_id = color_group[body_idx_in_group]
    if body_inv_mass[body_id] <= 0.0:
        return (wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0))
    num_contacts = body_contact_counts[body_id]
    if num_contacts > body_contact_buffer_pre_alloc:
        num_contacts = body_contact_buffer_pre_alloc
    if thread_id_within_body >= num_contacts:
        return (wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0))
    contact_count = rigid_contact_count[0]
    force_acc = wp.vec3(0.0)
    torque_acc = wp.vec3(0.0)
    h_ll_acc = wp.mat33(0.0)
    h_al_acc = wp.mat33(0.0)
    h_aa_acc = wp.mat33(0.0)
    i = thread_id_within_body
    while i < num_contacts:
        contact_idx = body_contact_indices[body_id * body_contact_buffer_pre_alloc + i]
        if contact_idx >= contact_count:
            i += 64
            continue
        s0 = rigid_contact_shape0[contact_idx]
        s1 = rigid_contact_shape1[contact_idx]
        b0 = shape_body[s0] if s0 >= 0 else -1
        b1 = shape_body[s1] if s1 >= 0 else -1
        if b0 != body_id and b1 != body_id:
            i += 64
            continue
        cp0_local = rigid_contact_point0[contact_idx]
        cp1_local = rigid_contact_point1[contact_idx]
        cp0_offset_local = rigid_contact_offset0[contact_idx]
        cp1_offset_local = rigid_contact_offset1[contact_idx]
        contact_normal = rigid_contact_normal[contact_idx]
        cp0_world = wp.transform_point(body_q[b0], cp0_local) if b0 >= 0 else cp0_local
        cp1_world = wp.transform_point(body_q[b1], cp1_local) if b1 >= 0 else cp1_local
        C_n = -rk.contact_surface_separation(
            cp0_world, cp1_world, contact_normal, rigid_contact_margin0[contact_idx], rigid_contact_margin1[contact_idx]
        )
        lam_n = float(0.0)
        C_eff = C_n
        lam_vec = wp.vec3(0.0)
        k = contact_penalty_k[contact_idx]
        friction_c0 = wp.vec3(0.0)
        if hard_contacts == 1:
            lam_vec = contact_lambda[contact_idx]
            lam_n = wp.dot(lam_vec, contact_normal)
            C0_vec = contact_C0[contact_idx]
            C0_n = wp.dot(contact_normal, C0_vec)
            C_eff = C_n - avbd_alpha * C0_n
            friction_c0 = (1.0 - avbd_alpha) * (C0_vec - contact_normal * C0_n)
        if C_n <= rk._SMALL_LENGTH_EPS and lam_n <= 0.0:
            i += 64
            continue
        f_n_check = k * C_eff + lam_n
        if f_n_check <= 0.0 and lam_n <= 0.0:
            i += 64
            continue
        contact_kd = contact_material_kd[contact_idx]
        contact_mu = contact_material_mu[contact_idx]
        force_0, torque_0, h_ll_0, h_al_0, h_aa_0, force_1, torque_1, h_ll_1, h_al_1, h_aa_1 = (
            rk.evaluate_rigid_contact_from_collision(
                b0,
                b1,
                body_q,
                body_q_prev,
                body_com,
                cp0_local,
                cp1_local,
                cp0_offset_local,
                cp1_offset_local,
                contact_normal,
                C_eff,
                k,
                k,
                contact_kd,
                lam_vec,
                contact_mu,
                friction_epsilon,
                hard_contacts,
                dt,
                friction_c0,
            )
        )
        factor = 1.0
        if hard_contacts == 0 and b0 >= 0 and (b1 >= 0):
            if body_inv_mass[b0] > 0.0 and body_inv_mass[b1] > 0.0:
                if body_colors[b0] == body_colors[b1]:
                    factor = 2.0
        if body_id == b0:
            force_acc += force_0
            torque_acc += torque_0
            h_ll_acc += factor * h_ll_0
            h_al_acc += factor * h_al_0
            h_aa_acc += factor * h_aa_0
        else:
            force_acc += force_1
            torque_acc += torque_1
            h_ll_acc += factor * h_ll_1
            h_al_acc += factor * h_al_1
            h_aa_acc += factor * h_aa_1
        i += 64
    return (force_acc, torque_acc, h_ll_acc, h_al_acc, h_aa_acc)


@wp.kernel(enable_backward=False)
def solve_rigid_64(soft: SoftInputs, rigid: RigidInputs, solve: SolveInputs):
    tid = wp.tid()
    sf, st, sll, sal, saa = _soft_lane(
        tid,
        soft.dt,
        soft.color_group,
        soft.particle_q,
        soft.particle_q_prev,
        soft.particle_radius,
        soft.body_q_prev,
        soft.body_q,
        soft.body_qd,
        soft.body_com,
        soft.body_inv_mass,
        soft.shape_body,
        soft.friction_epsilon,
        soft.body_particle_contact_penalty_k,
        soft.body_particle_contact_material_ke,
        soft.body_particle_contact_material_kd,
        soft.body_particle_contact_material_mu,
        soft.body_particle_contact_count,
        soft.soft_contact_indices,
        soft.body_particle_contact_shape,
        soft.body_particle_contact_body_pos,
        soft.body_particle_contact_body_vel,
        soft.body_particle_contact_normal,
        soft.soft_contact_barycentric,
        soft.shape_margin,
        soft.body_particle_contact_buffer_pre_alloc,
        soft.body_particle_contact_counts,
        soft.body_particle_contact_indices,
        soft.dense_contact_threshold,
        soft.body_forces,
        soft.body_torques,
        soft.body_hessian_ll,
        soft.body_hessian_al,
        soft.body_hessian_aa,
    )
    rf, rt, rll, ral, raa = _rigid_lane(
        tid,
        rigid.dt,
        rigid.color_group,
        rigid.body_q_prev,
        rigid.body_q,
        rigid.body_com,
        rigid.body_inv_mass,
        rigid.body_colors,
        rigid.friction_epsilon,
        rigid.contact_penalty_k,
        rigid.contact_material_ke,
        rigid.contact_material_kd,
        rigid.contact_material_mu,
        rigid.contact_lambda,
        rigid.contact_C0,
        rigid.avbd_alpha,
        rigid.hard_contacts,
        rigid.rigid_contact_count,
        rigid.rigid_contact_shape0,
        rigid.rigid_contact_shape1,
        rigid.rigid_contact_point0,
        rigid.rigid_contact_point1,
        rigid.rigid_contact_offset0,
        rigid.rigid_contact_offset1,
        rigid.rigid_contact_normal,
        rigid.rigid_contact_margin0,
        rigid.rigid_contact_margin1,
        rigid.shape_body,
        rigid.body_contact_buffer_pre_alloc,
        rigid.body_contact_counts,
        rigid.body_contact_indices,
        rigid.body_forces,
        rigid.body_torques,
        rigid.body_hessian_ll,
        rigid.body_hessian_al,
        rigid.body_hessian_aa,
    )
    force = wp.tile_reduce(wp.add, wp.tile(sf + rf, preserve_type=True))[0]
    torque = wp.tile_reduce(wp.add, wp.tile(st + rt, preserve_type=True))[0]
    ll = wp.tile_reduce(wp.add, wp.tile(sll + rll, preserve_type=True))[0]
    al = wp.tile_reduce(wp.add, wp.tile(sal + ral, preserve_type=True))[0]
    aa = wp.tile_reduce(wp.add, wp.tile(saa + raa, preserve_type=True))[0]
    if tid % 64 == 0:
        body = solve.body_ids_in_color[tid // 64]
        solve.external_forces[body] = force
        solve.external_torques[body] = torque
        solve.external_hessian_ll[body] = ll
        solve.external_hessian_al[body] = al
        solve.external_hessian_aa[body] = aa
        _solve_lane(
            tid // 64,
            solve.dt,
            solve.body_ids_in_color,
            solve.body_q,
            solve.body_q_prev,
            solve.body_q_rest,
            solve.body_mass,
            solve.body_inv_mass,
            solve.body_inertia,
            solve.body_inertia_q,
            solve.body_com,
            solve.adjacency,
            solve.joint_type,
            solve.joint_enabled,
            solve.joint_parent,
            solve.joint_child,
            solve.joint_X_p,
            solve.joint_X_c,
            solve.joint_axis,
            solve.joint_qd_start,
            solve.joint_target_q_start,
            solve.joint_constraint_start,
            solve.joint_penalty_k,
            solve.joint_penalty_kd,
            solve.joint_sigma_start,
            solve.joint_C_fric,
            solve.joint_target_ke,
            solve.joint_target_kd,
            solve.joint_target_q,
            solve.joint_target_qd,
            solve.joint_limit_lower,
            solve.joint_limit_upper,
            solve.joint_limit_ke,
            solve.joint_limit_kd,
            solve.joint_lambda_lin,
            solve.joint_lambda_ang,
            solve.joint_C0_lin,
            solve.joint_C0_ang,
            solve.joint_is_hard,
            solve.avbd_alpha,
            solve.joint_dof_dim,
            solve.joint_rest_angle,
            solve.external_forces,
            solve.external_torques,
            solve.external_hessian_ll,
            solve.external_hessian_al,
            solve.external_hessian_aa,
            solve.body_q_new,
        )


@wp.func
def _soft_lane_1(
    tid: int,
    dt: float,
    color_group: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    body_q_prev: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    shape_body: wp.array[int],
    friction_epsilon: float,
    body_particle_contact_penalty_k: wp.array[float],
    body_particle_contact_material_ke: wp.array[float],
    body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float],
    body_particle_contact_count: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    body_particle_contact_shape: wp.array[int],
    body_particle_contact_body_pos: wp.array[wp.vec3],
    body_particle_contact_body_vel: wp.array[wp.vec3],
    body_particle_contact_normal: wp.array[wp.vec3],
    soft_contact_barycentric: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    body_particle_contact_buffer_pre_alloc: int,
    body_particle_contact_counts: wp.array[wp.int32],
    body_particle_contact_indices: wp.array[wp.int32],
    dense_contact_threshold: int,
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
):
    """
    Per-body accumulation of body-particle soft contact forces and Hessians on rigid bodies.

    Handles both contact kinds from one per-body adjacency list, dispatching on each record's
    -1-padded ``soft_contact_indices``: a particle record ``(p, -1, -1)`` resolves single-particle
    geometry inline; an edge/face record evaluates the barycentric contact point over its 2-3 soft
    particles via ``rk._eval_soft_ef_contact``. Both apply the shared force law
    ``rk._compute_body_particle_contact_force`` and the equal-and-opposite body reaction. Body surface
    velocity uses the displacement-based path (body_q_prev).

    Notes:
      - Only dynamic bodies (inv_mass > 0) are updated.
      - Hessian contributions are accumulated into body_hessian_ll/al/aa.
      - Uses per-contact effective penalty/material parameters initialized once per step.
    """
    body_idx_in_group = tid // 128
    thread_id_within_body = tid % 128
    if body_idx_in_group >= color_group.shape[0]:
        return (wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0))
    body_id = color_group[body_idx_in_group]
    if body_inv_mass[body_id] <= 0.0:
        return (wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0))
    num_contacts = body_particle_contact_counts[body_id]
    if num_contacts > body_particle_contact_buffer_pre_alloc:
        num_contacts = body_particle_contact_buffer_pre_alloc
    if dense_contact_threshold > 0 and num_contacts >= dense_contact_threshold:
        return (wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0))
    max_contacts = body_particle_contact_count[0]
    X_wb = body_q[body_id]
    X_wb_prev = body_q_prev[body_id]
    com_world = wp.transform_point(X_wb, body_com[body_id])
    force_acc = wp.vec3(0.0)
    torque_acc = wp.vec3(0.0)
    h_ll_acc = wp.mat33(0.0)
    h_al_acc = wp.mat33(0.0)
    h_aa_acc = wp.mat33(0.0)
    i = thread_id_within_body
    while i < num_contacts:
        contact_idx = body_particle_contact_indices[body_id * body_particle_contact_buffer_pre_alloc + i]
        i += 128
        if contact_idx >= max_contacts:
            continue
        force, torque, h_ll, h_al, h_aa = rk._evaluate_body_particle_contact_reaction(
            dt,
            contact_idx,
            X_wb,
            X_wb_prev,
            com_world,
            particle_q,
            particle_q_prev,
            particle_radius,
            body_q_prev,
            body_q,
            body_qd,
            body_com,
            shape_body,
            friction_epsilon,
            body_particle_contact_penalty_k,
            body_particle_contact_material_kd,
            body_particle_contact_material_mu,
            soft_contact_indices,
            body_particle_contact_shape,
            body_particle_contact_body_pos,
            body_particle_contact_body_vel,
            body_particle_contact_normal,
            soft_contact_barycentric,
            shape_margin,
        )
        force_acc += force
        torque_acc += torque
        h_ll_acc += h_ll
        h_al_acc += h_al
        h_aa_acc += h_aa
    return (force_acc, torque_acc, h_ll_acc, h_al_acc, h_aa_acc)


@wp.func
def _rigid_lane_1(
    tid: int,
    dt: float,
    color_group: wp.array[wp.int32],
    body_q_prev: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    body_colors: wp.array[int],
    friction_epsilon: float,
    contact_penalty_k: wp.array[float],
    contact_material_ke: wp.array[float],
    contact_material_kd: wp.array[float],
    contact_material_mu: wp.array[float],
    contact_lambda: wp.array[wp.vec3],
    contact_C0: wp.array[wp.vec3],
    avbd_alpha: float,
    hard_contacts: int,
    rigid_contact_count: wp.array[int],
    rigid_contact_shape0: wp.array[int],
    rigid_contact_shape1: wp.array[int],
    rigid_contact_point0: wp.array[wp.vec3],
    rigid_contact_point1: wp.array[wp.vec3],
    rigid_contact_offset0: wp.array[wp.vec3],
    rigid_contact_offset1: wp.array[wp.vec3],
    rigid_contact_normal: wp.array[wp.vec3],
    rigid_contact_margin0: wp.array[float],
    rigid_contact_margin1: wp.array[float],
    shape_body: wp.array[wp.int32],
    body_contact_buffer_pre_alloc: int,
    body_contact_counts: wp.array[wp.int32],
    body_contact_indices: wp.array[wp.int32],
    body_forces: wp.array[wp.vec3],
    body_torques: wp.array[wp.vec3],
    body_hessian_ll: wp.array[wp.mat33],
    body_hessian_al: wp.array[wp.mat33],
    body_hessian_aa: wp.array[wp.mat33],
):
    """
    Per-body augmented-Lagrangian contact accumulation with rk._NUM_RIGID_CONTACT_THREADS_PER_BODY strided threads.
    """
    body_idx_in_group = tid // 128
    thread_id_within_body = tid % 128
    if body_idx_in_group >= color_group.shape[0]:
        return (wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0))
    body_id = color_group[body_idx_in_group]
    if body_inv_mass[body_id] <= 0.0:
        return (wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0))
    num_contacts = body_contact_counts[body_id]
    if num_contacts > body_contact_buffer_pre_alloc:
        num_contacts = body_contact_buffer_pre_alloc
    if thread_id_within_body >= num_contacts:
        return (wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0))
    contact_count = rigid_contact_count[0]
    force_acc = wp.vec3(0.0)
    torque_acc = wp.vec3(0.0)
    h_ll_acc = wp.mat33(0.0)
    h_al_acc = wp.mat33(0.0)
    h_aa_acc = wp.mat33(0.0)
    i = thread_id_within_body
    while i < num_contacts:
        contact_idx = body_contact_indices[body_id * body_contact_buffer_pre_alloc + i]
        if contact_idx >= contact_count:
            i += 128
            continue
        s0 = rigid_contact_shape0[contact_idx]
        s1 = rigid_contact_shape1[contact_idx]
        b0 = shape_body[s0] if s0 >= 0 else -1
        b1 = shape_body[s1] if s1 >= 0 else -1
        if b0 != body_id and b1 != body_id:
            i += 128
            continue
        cp0_local = rigid_contact_point0[contact_idx]
        cp1_local = rigid_contact_point1[contact_idx]
        cp0_offset_local = rigid_contact_offset0[contact_idx]
        cp1_offset_local = rigid_contact_offset1[contact_idx]
        contact_normal = rigid_contact_normal[contact_idx]
        cp0_world = wp.transform_point(body_q[b0], cp0_local) if b0 >= 0 else cp0_local
        cp1_world = wp.transform_point(body_q[b1], cp1_local) if b1 >= 0 else cp1_local
        C_n = -rk.contact_surface_separation(
            cp0_world, cp1_world, contact_normal, rigid_contact_margin0[contact_idx], rigid_contact_margin1[contact_idx]
        )
        lam_n = float(0.0)
        C_eff = C_n
        lam_vec = wp.vec3(0.0)
        k = contact_penalty_k[contact_idx]
        friction_c0 = wp.vec3(0.0)
        if hard_contacts == 1:
            lam_vec = contact_lambda[contact_idx]
            lam_n = wp.dot(lam_vec, contact_normal)
            C0_vec = contact_C0[contact_idx]
            C0_n = wp.dot(contact_normal, C0_vec)
            C_eff = C_n - avbd_alpha * C0_n
            friction_c0 = (1.0 - avbd_alpha) * (C0_vec - contact_normal * C0_n)
        if C_n <= rk._SMALL_LENGTH_EPS and lam_n <= 0.0:
            i += 128
            continue
        f_n_check = k * C_eff + lam_n
        if f_n_check <= 0.0 and lam_n <= 0.0:
            i += 128
            continue
        contact_kd = contact_material_kd[contact_idx]
        contact_mu = contact_material_mu[contact_idx]
        force_0, torque_0, h_ll_0, h_al_0, h_aa_0, force_1, torque_1, h_ll_1, h_al_1, h_aa_1 = (
            rk.evaluate_rigid_contact_from_collision(
                b0,
                b1,
                body_q,
                body_q_prev,
                body_com,
                cp0_local,
                cp1_local,
                cp0_offset_local,
                cp1_offset_local,
                contact_normal,
                C_eff,
                k,
                k,
                contact_kd,
                lam_vec,
                contact_mu,
                friction_epsilon,
                hard_contacts,
                dt,
                friction_c0,
            )
        )
        factor = 1.0
        if hard_contacts == 0 and b0 >= 0 and (b1 >= 0):
            if body_inv_mass[b0] > 0.0 and body_inv_mass[b1] > 0.0:
                if body_colors[b0] == body_colors[b1]:
                    factor = 2.0
        if body_id == b0:
            force_acc += force_0
            torque_acc += torque_0
            h_ll_acc += factor * h_ll_0
            h_al_acc += factor * h_al_0
            h_aa_acc += factor * h_aa_0
        else:
            force_acc += force_1
            torque_acc += torque_1
            h_ll_acc += factor * h_ll_1
            h_al_acc += factor * h_al_1
            h_aa_acc += factor * h_aa_1
        i += 128
    return (force_acc, torque_acc, h_ll_acc, h_al_acc, h_aa_acc)


@wp.kernel(enable_backward=False)
def solve_rigid_128(soft: SoftInputs, rigid: RigidInputs, solve: SolveInputs):
    tid = wp.tid()
    sf, st, sll, sal, saa = _soft_lane_1(
        tid,
        soft.dt,
        soft.color_group,
        soft.particle_q,
        soft.particle_q_prev,
        soft.particle_radius,
        soft.body_q_prev,
        soft.body_q,
        soft.body_qd,
        soft.body_com,
        soft.body_inv_mass,
        soft.shape_body,
        soft.friction_epsilon,
        soft.body_particle_contact_penalty_k,
        soft.body_particle_contact_material_ke,
        soft.body_particle_contact_material_kd,
        soft.body_particle_contact_material_mu,
        soft.body_particle_contact_count,
        soft.soft_contact_indices,
        soft.body_particle_contact_shape,
        soft.body_particle_contact_body_pos,
        soft.body_particle_contact_body_vel,
        soft.body_particle_contact_normal,
        soft.soft_contact_barycentric,
        soft.shape_margin,
        soft.body_particle_contact_buffer_pre_alloc,
        soft.body_particle_contact_counts,
        soft.body_particle_contact_indices,
        soft.dense_contact_threshold,
        soft.body_forces,
        soft.body_torques,
        soft.body_hessian_ll,
        soft.body_hessian_al,
        soft.body_hessian_aa,
    )
    rf, rt, rll, ral, raa = _rigid_lane_1(
        tid,
        rigid.dt,
        rigid.color_group,
        rigid.body_q_prev,
        rigid.body_q,
        rigid.body_com,
        rigid.body_inv_mass,
        rigid.body_colors,
        rigid.friction_epsilon,
        rigid.contact_penalty_k,
        rigid.contact_material_ke,
        rigid.contact_material_kd,
        rigid.contact_material_mu,
        rigid.contact_lambda,
        rigid.contact_C0,
        rigid.avbd_alpha,
        rigid.hard_contacts,
        rigid.rigid_contact_count,
        rigid.rigid_contact_shape0,
        rigid.rigid_contact_shape1,
        rigid.rigid_contact_point0,
        rigid.rigid_contact_point1,
        rigid.rigid_contact_offset0,
        rigid.rigid_contact_offset1,
        rigid.rigid_contact_normal,
        rigid.rigid_contact_margin0,
        rigid.rigid_contact_margin1,
        rigid.shape_body,
        rigid.body_contact_buffer_pre_alloc,
        rigid.body_contact_counts,
        rigid.body_contact_indices,
        rigid.body_forces,
        rigid.body_torques,
        rigid.body_hessian_ll,
        rigid.body_hessian_al,
        rigid.body_hessian_aa,
    )
    force = wp.tile_reduce(wp.add, wp.tile(sf + rf, preserve_type=True))[0]
    torque = wp.tile_reduce(wp.add, wp.tile(st + rt, preserve_type=True))[0]
    ll = wp.tile_reduce(wp.add, wp.tile(sll + rll, preserve_type=True))[0]
    al = wp.tile_reduce(wp.add, wp.tile(sal + ral, preserve_type=True))[0]
    aa = wp.tile_reduce(wp.add, wp.tile(saa + raa, preserve_type=True))[0]
    if tid % 128 == 0:
        body = solve.body_ids_in_color[tid // 128]
        solve.external_forces[body] = force
        solve.external_torques[body] = torque
        solve.external_hessian_ll[body] = ll
        solve.external_hessian_al[body] = al
        solve.external_hessian_aa[body] = aa
        _solve_lane(
            tid // 128,
            solve.dt,
            solve.body_ids_in_color,
            solve.body_q,
            solve.body_q_prev,
            solve.body_q_rest,
            solve.body_mass,
            solve.body_inv_mass,
            solve.body_inertia,
            solve.body_inertia_q,
            solve.body_com,
            solve.adjacency,
            solve.joint_type,
            solve.joint_enabled,
            solve.joint_parent,
            solve.joint_child,
            solve.joint_X_p,
            solve.joint_X_c,
            solve.joint_axis,
            solve.joint_qd_start,
            solve.joint_target_q_start,
            solve.joint_constraint_start,
            solve.joint_penalty_k,
            solve.joint_penalty_kd,
            solve.joint_sigma_start,
            solve.joint_C_fric,
            solve.joint_target_ke,
            solve.joint_target_kd,
            solve.joint_target_q,
            solve.joint_target_qd,
            solve.joint_limit_lower,
            solve.joint_limit_upper,
            solve.joint_limit_ke,
            solve.joint_limit_kd,
            solve.joint_lambda_lin,
            solve.joint_lambda_ang,
            solve.joint_C0_lin,
            solve.joint_C0_ang,
            solve.joint_is_hard,
            solve.avbd_alpha,
            solve.joint_dof_dim,
            solve.joint_rest_angle,
            solve.external_forces,
            solve.external_torques,
            solve.external_hessian_ll,
            solve.external_hessian_al,
            solve.external_hessian_aa,
            solve.body_q_new,
        )

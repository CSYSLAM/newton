# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conservative VT/EE streams and Planar-DAT for XPBD surface FEM.

Reference: Divide and Truncate, arXiv:2604.15513, sections 3.3 and 4.
Each record owns both primitives; DAT uses all four vertices, regardless of
which constraint last moved them. Geometry is cached only at detection; the
division-plane offset and truncation ratios are recomputed from displacements.
Float support intervals conservatively filter candidates; narrow gaps use
double closest-feature refinement. Stream membership may be a superset of
the exact radius neighborhood; the physical response recomputes distance.
Status bits are sticky: 1 capacity overflow, 2 nonseparated base pair,
4 nonfinite proposal. Failed transactions never update accepted positions.
"""

import warp as wp

from ...geometry.kernels import triangle_closest_point
from .fem_kernels import effective_mass


@wp.func
def closest_segment(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d):
    direction = b - a
    length2 = wp.length_sq(direction)
    t = wp.float64(0.0)
    if length2 > wp.float64(0.0):
        t = wp.clamp(wp.dot(p - a, direction) / length2, wp.float64(0.0), wp.float64(1.0))
    return a + t * direction


@wp.func
def closest_triangle(p: wp.vec3d, a: wp.vec3d, b: wp.vec3d, c: wp.vec3d):
    # Boundary candidates also cover degenerate triangles. The interior test
    # uses oriented areas, avoiding cancellation in Gram determinants.
    q = closest_segment(p, a, b)
    for edge in range(2):
        other = closest_segment(p, b, c)
        if edge == 1:
            other = closest_segment(p, c, a)
        if wp.length_sq(p - other) < wp.length_sq(p - q):
            q = other
    n = wp.cross(b - a, c - a)
    n2 = wp.length_sq(n)
    if n2 > wp.float64(0.0):
        projection = p - n * (wp.dot(n, p - a) / n2)
        if (
            wp.dot(n, wp.cross(b - a, projection - a)) >= wp.float64(0.0)
            and wp.dot(n, wp.cross(c - b, projection - b)) >= wp.float64(0.0)
            and wp.dot(n, wp.cross(a - c, projection - c)) >= wp.float64(0.0)
        ):
            q = projection
    return q


@wp.func
def closest_edges(a: wp.vec3d, b: wp.vec3d, c: wp.vec3d, d: wp.vec3d):
    p, q = a, closest_segment(a, c, d)
    for endpoint in range(3):
        other_p, other_q = b, closest_segment(b, c, d)
        if endpoint == 1:
            other_p, other_q = closest_segment(c, a, b), c
        elif endpoint == 2:
            other_p, other_q = closest_segment(d, a, b), d
        if wp.length_sq(other_p - other_q) < wp.length_sq(p - q):
            p, q = other_p, other_q
    u, v, r = b - a, d - c, c - a
    n = wp.cross(u, v)
    n2 = wp.length_sq(n)
    if n2 > wp.float64(0.0):
        s = wp.dot(wp.cross(r, v), n) / n2
        t = wp.dot(wp.cross(r, u), n) / n2
        if s >= wp.float64(0.0) and s <= wp.float64(1.0) and t >= wp.float64(0.0) and t <= wp.float64(1.0):
            p, q = a + s * u, c + t * v
    return p, q


@wp.kernel
def primitive_bounds(
    pos: wp.array[wp.vec3],
    triangles: wp.array2d[int],
    edges: wp.array[wp.vec2i],
    tri_lo: wp.array[wp.vec3],
    tri_hi: wp.array[wp.vec3],
    edge_lo: wp.array[wp.vec3],
    edge_hi: wp.array[wp.vec3],
):
    i = wp.tid()
    if i < triangles.shape[0]:
        a, b, c = pos[triangles[i, 0]], pos[triangles[i, 1]], pos[triangles[i, 2]]
        tri_lo[i] = wp.min(a, wp.min(b, c))
        tri_hi[i] = wp.max(a, wp.max(b, c))
    if i < edges.shape[0]:
        e = edges[i]
        edge_lo[i] = wp.min(pos[e[0]], pos[e[1]])
        edge_hi[i] = wp.max(pos[e[0]], pos[e[1]])


@wp.func
def record_float_pair(
    ids: wp.vec4i,
    kind: int,
    a: wp.vec3,
    b: wp.vec3,
    radius: float,
    count: wp.array[int],
    pairs: wp.array[wp.vec4i],
    kinds: wp.array[int],
    normals: wp.array[wp.vec3],
    distances: wp.array[wp.vec4],
    gaps: wp.array[float],
    pos: wp.array[wp.vec3],
    status: wp.array[int],
):
    """Certify a float separating axis; return False to request double refinement."""
    n = wp.normalize(a - b)
    projections = wp.vec4(0.0)
    min_a, max_b = float(wp.inf), float(-wp.inf)
    scale = wp.max(wp.length(a), wp.length(b))
    for j in range(4):
        scale = wp.max(scale, wp.length(pos[ids[j]]))
        projections[j] = wp.dot(n, pos[ids[j]] - b)
        if j == 0 or (kind == 1 and j == 1):
            min_a = wp.min(min_a, projections[j])
        else:
            max_b = wp.max(max_b, projections[j])
    # The support gap is a lower distance bound even for an approximate closest
    # point. Reject only with an outward roundoff allowance, never by |a-b|.
    error = 2.0e-6 * wp.max(scale, 0.01)
    gap = min_a - max_b
    handled = gap > radius + error
    if not handled and gap > 8.0 * error:
        slot = wp.atomic_add(count, 0, 1)
        if slot < pairs.shape[0]:
            pairs[slot] = ids
            kinds[slot] = kind
            normals[slot] = n
            safe_distances = projections - wp.vec4(0.5 * (min_a + max_b))
            for j in range(4):
                if j == 0 or (kind == 1 and j == 1):
                    safe_distances[j] -= error
                else:
                    safe_distances[j] += error
            distances[slot] = safe_distances
            gaps[slot] = gap - 2.0 * error
        else:
            wp.atomic_or(status, 0, 1)
        handled = True
    return handled


@wp.func
def record_pair(
    ids: wp.vec4i,
    kind: int,
    a: wp.vec3d,
    b: wp.vec3d,
    count: wp.array[int],
    pairs: wp.array[wp.vec4i],
    kinds: wp.array[int],
    normals: wp.array[wp.vec3],
    distances: wp.array[wp.vec4],
    gaps: wp.array[float],
    pos: wp.array[wp.vec3],
    status: wp.array[int],
):
    slot = wp.atomic_add(count, 0, 1)
    if slot < pairs.shape[0]:
        pairs[slot] = ids
        kinds[slot] = kind
        if wp.length_sq(a - b) < wp.float64(1.0e-24):
            wp.atomic_or(status, 0, 2)
        n = wp.normalize(a - b)
        projections = wp.vec4d(0.0)
        min_a, max_b = wp.float64(wp.inf), wp.float64(-wp.inf)
        for j in range(4):
            projections[j] = wp.dot(n, wp.vec3d(pos[ids[j]]) - b)
            if j == 0 or (kind == 1 and j == 1):
                min_a = wp.min(min_a, projections[j])
            else:
                max_b = wp.max(max_b, projections[j])
        gap = min_a - max_b
        if gap <= wp.float64(0.0):
            wp.atomic_or(status, 0, 2)
        normals[slot] = wp.vec3(n)
        distances[slot] = wp.vec4(projections - wp.vec4d((min_a + max_b) * wp.float64(0.5)))
        gaps[slot] = float(gap)
    else:
        wp.atomic_or(status, 0, 1)


@wp.kernel
def detect_vt(
    bvh: wp.uint64,
    pos: wp.array[wp.vec3],
    triangles: wp.array2d[int],
    worlds: wp.array[int],
    radius: float,
    count: wp.array[int],
    pairs: wp.array[wp.vec4i],
    kinds: wp.array[int],
    normals: wp.array[wp.vec3],
    distances: wp.array[wp.vec4],
    gaps: wp.array[float],
    status: wp.array[int],
):
    i = wp.tid()
    p = pos[i]
    query = wp.bvh_query_aabb(bvh, p - wp.vec3(radius), p + wp.vec3(radius))
    tri = int(0)
    while wp.bvh_query_next(query, tri):
        a, b, c = triangles[tri, 0], triangles[tri, 1], triangles[tri, 2]
        if i != a and i != b and i != c and worlds[i] == worlds[a]:
            quick, _bary, _feature = triangle_closest_point(pos[a], pos[b], pos[c], p)
            if record_float_pair(
                wp.vec4i(i, a, b, c), 0, p, quick, radius, count, pairs, kinds, normals, distances, gaps, pos, status
            ):
                continue
            q = closest_triangle(wp.vec3d(p), wp.vec3d(pos[a]), wp.vec3d(pos[b]), wp.vec3d(pos[c]))
            if wp.length_sq(q - wp.vec3d(p)) <= wp.float64(radius) * wp.float64(radius):
                record_pair(
                    wp.vec4i(i, a, b, c), 0, wp.vec3d(p), q, count, pairs, kinds, normals, distances, gaps, pos, status
                )


@wp.kernel
def detect_ee(
    bvh: wp.uint64,
    pos: wp.array[wp.vec3],
    edges: wp.array[wp.vec2i],
    worlds: wp.array[int],
    radius: float,
    count: wp.array[int],
    pairs: wp.array[wp.vec4i],
    kinds: wp.array[int],
    normals: wp.array[wp.vec3],
    distances: wp.array[wp.vec4],
    gaps: wp.array[float],
    status: wp.array[int],
):
    e = wp.tid()
    a, b = edges[e][0], edges[e][1]
    query = wp.bvh_query_aabb(bvh, wp.min(pos[a], pos[b]) - wp.vec3(radius), wp.max(pos[a], pos[b]) + wp.vec3(radius))
    other = int(0)
    while wp.bvh_query_next(query, other):
        if other > e:
            c, d = edges[other][0], edges[other][1]
            if a != c and a != d and b != c and b != d and worlds[a] == worlds[c]:
                st = wp.closest_point_edge_edge(pos[a], pos[b], pos[c], pos[d], 1.0e-12)
                quick_p = wp.lerp(pos[a], pos[b], st[0])
                quick_q = wp.lerp(pos[c], pos[d], st[1])
                if record_float_pair(
                    wp.vec4i(a, b, c, d),
                    1,
                    quick_p,
                    quick_q,
                    radius,
                    count,
                    pairs,
                    kinds,
                    normals,
                    distances,
                    gaps,
                    pos,
                    status,
                ):
                    continue
                p, q = closest_edges(wp.vec3d(pos[a]), wp.vec3d(pos[b]), wp.vec3d(pos[c]), wp.vec3d(pos[d]))
                if wp.length_sq(q - p) <= wp.float64(radius) * wp.float64(radius):
                    record_pair(
                        wp.vec4i(a, b, c, d), 1, p, q, count, pairs, kinds, normals, distances, gaps, pos, status
                    )


@wp.kernel
def truncate_pairs(
    base: wp.array[wp.vec3],
    accepted: wp.array[wp.vec3],
    candidate: wp.array[wp.vec3],
    count: wp.array[int],
    pairs: wp.array[wp.vec4i],
    kinds: wp.array[int],
    normals: wp.array[wp.vec3],
    distances: wp.array[wp.vec4],
    gaps: wp.array[float],
    gamma: float,
    workers: int,
    t_out: wp.array[float],
):
    # Persistent workers traverse only the active prefix under a fixed graph.
    for slot in range(wp.tid(), wp.min(count[0], pairs.shape[0]), workers):
        ids = pairs[slot]
        n = normals[slot]
        da, db = float(0.0), float(0.0)
        low, high = -0.5 * gaps[slot], 0.5 * gaps[slot]
        current = wp.vec4(0.0)
        for j in range(4):
            dx = candidate[ids[j]] - accepted[ids[j]]
            current[j] = distances[slot][j] + wp.dot(n, accepted[ids[j]] - base[ids[j]])
            side_a = j == 0 or (kinds[slot] == 1 and j == 1)
            if side_a:
                da = wp.max(da, -wp.dot(n, dx))
                high = wp.min(high, current[j])
            else:
                db = wp.max(db, wp.dot(n, dx))
                low = wp.max(low, current[j])
        ratio = float(0.5)
        if da + db > 0.0:
            ratio = wp.clamp(db / (da + db), 0.05, 0.95)
        # Keep BOTH the detection base and accepted state inside the new cages.
        # Truncate only the proposed increment; never retract an accepted state
        # merely because another stage recomputed a displacement-dependent plane.
        if high < low:
            # Roundoff can exhaust the common separating-plane interval.
            # Reject this increment rather than creating an invalid cage.
            for j in range(4):
                wp.atomic_min(t_out, ids[j], 0.0)
            continue
        shift = low + ratio * (high - low)
        for j in range(4):
            i = ids[j]
            distance = current[j] - shift
            approach = wp.dot(n, candidate[i] - accepted[i])
            reserve = 2.0e-7 * wp.max(wp.length(base[i]), 0.01)
            if distance * approach < 0.0:
                # Leave an absolute rounding reserve when committing float32 positions.
                safe_distance = wp.max(wp.abs(distance) - reserve, 0.0)
                t = wp.clamp(gamma * safe_distance / wp.abs(approach), 0.0, 1.0)
                if t < 1.0:
                    wp.atomic_min(t_out, i, t)


@wp.kernel
def validate_candidate(candidate: wp.array[wp.vec3], status: wp.array[int]):
    p = candidate[wp.tid()]
    if not (wp.isfinite(p[0]) and wp.isfinite(p[1]) and wp.isfinite(p[2])):
        wp.atomic_or(status, 0, 4)


@wp.kernel
def commit(
    base: wp.array[wp.vec3],
    candidate: wp.array[wp.vec3],
    t: wp.array[float],
    max_displacement: float,
    status: wp.array[int],
    accepted: wp.array[wp.vec3],
):
    i = wp.tid()
    if status[0] == 0 and t[i] > 0.0:
        endpoint = wp.vec3d(accepted[i]) + (wp.vec3d(candidate[i]) - wp.vec3d(accepted[i])) * wp.float64(t[i])
        dx = endpoint - wp.vec3d(base[i])
        length = wp.length(dx)
        if length > wp.float64(max_displacement):
            endpoint = wp.vec3d(base[i]) + dx * (wp.float64(max_displacement) / length)
        accepted[i] = wp.vec3(endpoint)


@wp.func
def contact_geometry(pos: wp.array[wp.vec3], ids: wp.vec4i, kind: int):
    p0, p1, p2, p3 = pos[ids[0]], pos[ids[1]], pos[ids[2]], pos[ids[3]]
    bary = wp.vec4(0.0)
    difference = wp.vec3(0.0)
    if kind == 0:
        q, bc, _feature = triangle_closest_point(p1, p2, p3, p0)
        difference = p0 - q
        bary = wp.vec4(1.0, -bc[0], -bc[1], -bc[2])
    else:
        st = wp.closest_point_edge_edge(p0, p1, p2, p3, 1.0e-12)
        difference = wp.lerp(p0, p1, st[0]) - wp.lerp(p2, p3, st[1])
        bary = wp.vec4(1.0 - st[0], st[0], st[1] - 1.0, -st[1])
    return difference, bary


@wp.kernel
def contact_response(
    pos: wp.array[wp.vec3],
    anchor: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    flags: wp.array[int],
    count: wp.array[int],
    pairs: wp.array[wp.vec4i],
    kinds: wp.array[int],
    radius: float,
    stiffness: float,
    friction: float,
    dt: float,
    workers: int,
    delta: wp.array[wp.vec3],
    weights: wp.array[float],
):
    for slot in range(wp.tid(), wp.min(count[0], pairs.shape[0]), workers):
        ids = pairs[slot]
        difference, bary = contact_geometry(pos, ids, kinds[slot])
        distance = wp.length(difference)
        if distance > 1.0e-12 and distance < radius and stiffness > 0.0:
            n = difference / distance
            denom = float(0.0)
            relative = wp.vec3(0.0)
            for j in range(4):
                denom += effective_mass(ids[j], inv_mass, flags) * bary[j] * bary[j]
                relative += bary[j] * (pos[ids[j]] - anchor[ids[j]])
            separation = distance
            if denom > 0.0 and separation < radius:
                # Retain the experimental path's quadratic response. Adding
                # the official logarithmic potential without changing this
                # Jacobi split regressed full-scene quality; see FEM_DAT_NOTES.
                tangent = relative - n * wp.dot(n, relative)
                slip = wp.length(tangent)
                dl = (radius - separation) / (denom + 1.0 / (stiffness * dt * dt))
                correction = dl * n
                if slip > 0.0:
                    friction_dl = wp.min(friction * dl, slip / denom)
                    correction -= friction_dl * tangent / wp.max(slip, 1.0e-2 * dt)
                for j in range(4):
                    w = effective_mass(ids[j], inv_mass, flags)
                    if w > 0.0 and wp.abs(bary[j]) > 1.0e-8:
                        wp.atomic_add(delta, ids[j], w * bary[j] * correction)
                        wp.atomic_add(weights, ids[j], 1.0)


@wp.kernel
def apply_contact_delta(
    pos: wp.array[wp.vec3], delta: wp.array[wp.vec3], weights: wp.array[float], candidate: wp.array[wp.vec3]
):
    i = wp.tid()
    candidate[i] = pos[i] + delta[i] / wp.max(weights[i], 1.0)


@wp.kernel
def count_surface_crossings(
    bvh: wp.uint64,
    pos: wp.array[wp.vec3],
    triangles: wp.array2d[int],
    edges: wp.array[wp.vec2i],
    worlds: wp.array[int],
    crossing_count: wp.array[int],
    crossing_pairs: wp.array[wp.vec2i],
):
    """Independently query current segments against triangles, without contact rows."""
    edge = edges[wp.tid()]
    a, b = pos[edge[0]], pos[edge[1]]
    query = wp.bvh_query_aabb(bvh, wp.min(a, b) - wp.vec3(1.0e-7), wp.max(a, b) + wp.vec3(1.0e-7))
    tri = int(0)
    while wp.bvh_query_next(query, tri):
        i, j, k = triangles[tri, 0], triangles[tri, 1], triangles[tri, 2]
        if (
            edge[0] != i
            and edge[0] != j
            and edge[0] != k
            and edge[1] != i
            and edge[1] != j
            and edge[1] != k
            and worlds[edge[0]] == worlds[i]
        ):
            # Double precision Moller-Trumbore as an independent audit, not the DAT narrow phase.
            origin, direction = wp.vec3d(a), wp.vec3d(b) - wp.vec3d(a)
            p0 = wp.vec3d(pos[i])
            e1, e2 = wp.vec3d(pos[j]) - p0, wp.vec3d(pos[k]) - p0
            h = wp.cross(direction, e2)
            determinant = wp.dot(e1, h)
            intersects = False
            if wp.abs(determinant) > wp.float64(1.0e-18):
                inv = wp.float64(1.0) / determinant
                s = origin - p0
                u = inv * wp.dot(s, h)
                q = wp.cross(s, e1)
                v = inv * wp.dot(direction, q)
                t = inv * wp.dot(e2, q)
                if (
                    u >= wp.float64(0.0)
                    and v >= wp.float64(0.0)
                    and u + v <= wp.float64(1.0)
                    and t > wp.float64(1.0e-7)
                    and t < wp.float64(1.0 - 1.0e-7)
                ):
                    intersects = True
            else:
                normal = wp.cross(e1, e2)
                normal_length = wp.length(normal)
                if (
                    normal_length > wp.float64(0.0)
                    and wp.abs(wp.dot(normal, origin - p0)) <= wp.float64(1.0e-12) * normal_length
                    and wp.abs(wp.dot(normal, origin + direction - p0)) <= wp.float64(1.0e-12) * normal_length
                ):
                    # Clip a coplanar segment against the three oriented edge halfspaces.
                    lower, upper = wp.float64(1.0e-7), wp.float64(1.0 - 1.0e-7)
                    corners = wp.vec3i(i, j, k)
                    for side in range(3):
                        start = wp.vec3d(pos[corners[side]])
                        edge_direction = wp.vec3d(pos[corners[(side + 1) % 3]]) - start
                        value = wp.dot(normal, wp.cross(edge_direction, origin - start))
                        slope = wp.dot(normal, wp.cross(edge_direction, direction))
                        if slope > wp.float64(0.0):
                            lower = wp.max(lower, -value / slope)
                        elif slope < wp.float64(0.0):
                            upper = wp.min(upper, -value / slope)
                        elif value < wp.float64(0.0):
                            upper = wp.float64(-1.0)
                    intersects = lower <= upper
            if intersects:
                slot = wp.atomic_add(crossing_count, 0, 1)
                if slot < crossing_pairs.shape[0]:
                    crossing_pairs[slot] = wp.vec2i(wp.tid(), tri)

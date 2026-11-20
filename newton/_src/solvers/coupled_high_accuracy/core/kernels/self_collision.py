# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cloth/particle self-collision (native ``kernels/lbvh.cu`` and ``kernels/self_collision.cu``).

Linear BVHs (Karras radix trees over Morton codes) are built once over the rest positions and refitted every step.
Each primitive collects at most ``max_slots`` candidates (vertex-face, particle-particle, edge-edge, edge-face) in
the native traversal order, then penalty forces with mass-weighted stiffness are accumulated. Boxes are stored as
(lower, upper) pairs in two arrays.
"""

import warp as wp

# The solver never differentiates through its kernels; skipping the adjoints halves compile time.
wp.set_module_options({"enable_backward": False})

vec5i = wp.types.vector(length=5, dtype=wp.int32)
vec5f = wp.types.vector(length=5, dtype=wp.float32)


@wp.struct
class SelfContact:
    """One contact of the matrix-free tangent: up to five weighted points along a normal."""

    ids: vec5i
    weights: vec5f
    normal: wp.vec3
    stiffness: float
    count: int


# ---------------------------------------------------------------------------------------------------------------------
# LBVH construction


@wp.func
def expand_bits(v: wp.uint32):
    v = (v * wp.uint32(0x00010001)) & wp.uint32(0xFF0000FF)
    v = (v * wp.uint32(0x00000101)) & wp.uint32(0x0F00F00F)
    v = (v * wp.uint32(0x00000011)) & wp.uint32(0xC30C30C3)
    v = (v * wp.uint32(0x00000005)) & wp.uint32(0x49249249)
    return v


@wp.func
def morton_axis(value: float):
    return wp.uint32(wp.min(wp.max(value, 0.0), 0.99999994) * 1024.0)


@wp.kernel
def generate_morton_codes(
    codes: wp.array[wp.int32],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    scene_lower: wp.vec3,
    scene_upper: wp.vec3,
):
    """30-bit Morton code of each box centre normalised to the scene box; a flat axis maps to 0.5. Codes are stored
    as int32 bit patterns (all below 2^30) for the radix sort."""
    i = wp.tid()
    center = (lower[i] + upper[i]) * 0.5
    extent = scene_upper - scene_lower
    unit = wp.vec3(0.5)
    if extent[0] > 0.0:
        unit[0] = (center[0] - scene_lower[0]) / extent[0]
    if extent[1] > 0.0:
        unit[1] = (center[1] - scene_lower[1]) / extent[1]
    if extent[2] > 0.0:
        unit[2] = (center[2] - scene_lower[2]) / extent[2]
    code = expand_bits(morton_axis(unit[0])) * wp.uint32(4) + expand_bits(morton_axis(unit[1])) * wp.uint32(2)
    code = code + expand_bits(morton_axis(unit[2]))
    codes[i] = wp.int32(code)


@wp.func
def clz32(value: wp.uint32):
    """Number of leading zero bits of a 32-bit value (32 for zero)."""
    x = value
    if x == wp.uint32(0):
        return 32
    n = int(0)
    if (x & wp.uint32(0xFFFF0000)) == wp.uint32(0):
        n += 16
        x = x << wp.uint32(16)
    if (x & wp.uint32(0xFF000000)) == wp.uint32(0):
        n += 8
        x = x << wp.uint32(8)
    if (x & wp.uint32(0xF0000000)) == wp.uint32(0):
        n += 4
        x = x << wp.uint32(4)
    if (x & wp.uint32(0xC0000000)) == wp.uint32(0):
        n += 2
        x = x << wp.uint32(2)
    if (x & wp.uint32(0x80000000)) == wp.uint32(0):
        n += 1
    return n


@wp.func
def common_prefix(codes: wp.array[wp.int32], count: int, i: int, j: int):
    """Karras' delta; duplicate codes are disambiguated by the index bits (32 + clz(i ^ j))."""
    if j < 0 or j >= count:
        return -1
    difference = wp.uint32(codes[i]) ^ wp.uint32(codes[j])
    if difference == wp.uint32(0):
        return 32 + clz32(wp.uint32(i) ^ wp.uint32(j))
    return clz32(difference)


@wp.kernel
def build_radix_tree(parent: wp.array[wp.int32], children: wp.array[wp.vec2i], codes: wp.array[wp.int32], n: int):
    i = wp.tid()
    direction = int(1)
    if common_prefix(codes, n, i, i + 1) < common_prefix(codes, n, i, i - 1):
        direction = -1
    prefix_min = common_prefix(codes, n, i, i - direction)
    length_max = int(2)
    while common_prefix(codes, n, i, i + length_max * direction) > prefix_min:
        length_max = length_max << 1
    length = int(0)
    t = length_max >> 1
    while t >= 1:
        if common_prefix(codes, n, i, i + (length + t) * direction) > prefix_min:
            length += t
        t = t >> 1
    other = i + length * direction
    first = wp.min(i, other)
    last = wp.max(i, other)
    prefix_node = common_prefix(codes, n, first, last)
    split = first
    step = last - first
    while True:
        step = (step + 1) >> 1
        candidate = split + step
        if candidate < last and common_prefix(codes, n, first, candidate) > prefix_node:
            split = candidate
        if step <= 1:
            break
    left = split
    if split == first:
        left = split + n - 1
    right = split + 1
    if split + 1 == last:
        right = split + n
    children[i] = wp.vec2i(left, right)
    parent[right] = i
    parent[left] = i


@wp.kernel
def build_primitive_leaf_nodes(leaf_of_primitive: wp.array[wp.int32], sorted_primitive: wp.array[wp.int32], n: int):
    i = wp.tid()
    leaf_of_primitive[sorted_primitive[i]] = i + n - 1


@wp.kernel
def build_escape_nodes(escape: wp.array[wp.int32], parent: wp.array[wp.int32], children: wp.array[wp.vec2i]):
    """Escape links for stackless traversal: climb while the node is its parent's right child, then continue at the
    right sibling; -1 once the root is passed."""
    i = wp.tid()
    node = i
    up = parent[node]
    while up >= 0 and children[up][1] == node:
        node = up
        up = parent[node]
    if up < 0:
        escape[i] = -1
    else:
        escape[i] = children[up][1]


# ---------------------------------------------------------------------------------------------------------------------
# Primitive boxes and refit


@wp.kernel
def build_particle_aabbs(
    x: wp.array[wp.vec3],
    particles: wp.array[wp.int32],
    radius: wp.array[float],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    particles_per_list: int,
    particles_per_world: int,
):
    i = wp.tid()
    world = i // particles_per_list
    particle = particles[i % particles_per_list]
    position = x[world * particles_per_world + particle]
    r = radius[particle]
    lower[i] = position - wp.vec3(r)
    upper[i] = position + wp.vec3(r)


@wp.kernel
def build_edge_aabbs(
    x: wp.array[wp.vec3],
    edges: wp.array[wp.vec4i],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    margin: float,
    edges_per_world: int,
    particles_per_world: int,
):
    i = wp.tid()
    offset = (i // edges_per_world) * particles_per_world
    edge = edges[i % edges_per_world]
    a = x[edge[2] + offset]
    b = x[edge[3] + offset]
    lower[i] = wp.min(a, b) - wp.vec3(margin)
    upper[i] = wp.max(a, b) + wp.vec3(margin)


@wp.kernel
def build_triangle_aabbs(
    x: wp.array[wp.vec3],
    triangles: wp.array[wp.vec3i],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    margin: float,
    triangles_per_world: int,
    particles_per_world: int,
):
    i = wp.tid()
    offset = (i // triangles_per_world) * particles_per_world
    t = triangles[i % triangles_per_world]
    a = x[t[0] + offset]
    b = x[t[1] + offset]
    c = x[t[2] + offset]
    lower[i] = wp.min(a, wp.min(b, c)) - wp.vec3(margin)
    upper[i] = wp.max(a, wp.max(b, c)) + wp.vec3(margin)


@wp.kernel
def copy_leaf_boxes(
    leaf_lower: wp.array[wp.vec3],
    leaf_upper: wp.array[wp.vec3],
    sorted_leaf: wp.array[wp.int32],
    node_lower: wp.array[wp.vec3],
    node_upper: wp.array[wp.vec3],
    leaves_per_world: int,
):
    """Leaf nodes of every world's tree take their primitive boxes (all worlds share one topology)."""
    i = wp.tid()
    world = i // leaves_per_world
    local = i % leaves_per_world
    node = world * (2 * leaves_per_world - 1) + local + leaves_per_world - 1
    source = world * leaves_per_world + sorted_leaf[local]
    node_lower[node] = leaf_lower[source]
    node_upper[node] = leaf_upper[source]


@wp.kernel
def refit_level(
    level_nodes: wp.array[wp.int32],
    level_start: int,
    level_count: int,
    children: wp.array[wp.vec2i],
    node_lower: wp.array[wp.vec3],
    node_upper: wp.array[wp.vec3],
    leaves_per_world: int,
):
    """Merge the children of one height level of internal nodes in every world. Levels run bottom-up, which
    replaces the native atomic-counter refit (it needs device memory fences) with the same min/max boxes."""
    i = wp.tid()
    world = i // level_count
    node = level_nodes[level_start + i % level_count]
    base = world * (2 * leaves_per_world - 1)
    pair = children[node]
    node_lower[base + node] = wp.min(node_lower[base + pair[0]], node_lower[base + pair[1]])
    node_upper[base + node] = wp.max(node_upper[base + pair[0]], node_upper[base + pair[1]])


@wp.kernel
def add_hessian_diagonal(hessian: wp.array[wp.mat33], diagonal: wp.array[wp.mat33]):
    i = wp.tid()
    diagonal[i] = diagonal[i] + hessian[i]


@wp.kernel
def multiply_hessian_diagonal(hessian: wp.array[wp.mat33], v: wp.array[wp.vec3], product: wp.array[wp.vec3]):
    i = wp.tid()
    # Atomic: other tangent terms may add into the same row concurrently (config.tangent_streams).
    wp.atomic_add(product, i, hessian[i] * v[i])


@wp.func
def overlaps(lower: wp.vec3, upper: wp.vec3, box_lower: wp.vec3, box_upper: wp.vec3):
    """Inclusive overlap with ordered comparisons, so any NaN bound rejects the node."""
    return (
        lower[0] <= box_upper[0]
        and upper[0] >= box_lower[0]
        and lower[1] <= box_upper[1]
        and upper[1] >= box_lower[1]
        and lower[2] <= box_upper[2]
        and upper[2] >= box_lower[2]
    )


@wp.func
def group_compatible(groups: wp.array[wp.int32], a: int, b: int):
    """Native equality/sign mask: equal groups, or either negative (unassigned)."""
    if groups.shape[0] == 0:
        return True
    first = groups[a]
    second = groups[b]
    return first == second or (first | second) < 0


# ---------------------------------------------------------------------------------------------------------------------
# Distance helpers


@wp.func
def product_difference(a: float, b: float, c: float, d: float):
    """a b - c d, correctly rounded through float64 (the native code compensates with two FMAs)."""
    return wp.float32(wp.float64(a) * wp.float64(b) - wp.float64(c) * wp.float64(d))


@wp.func
def saturate_parameter(value: float):
    return wp.min(wp.max(value, 0.0), 1.0)


@wp.func
def segment_parameters(p: wp.vec3, q: wp.vec3, a: wp.vec3, b: wp.vec3):
    """Closest-point parameters (s on pq, t on ab) of two segments, with the native degenerate fallbacks."""
    u = q - p
    v = b - a
    r = p - a
    uu = wp.dot(u, u)
    vv = wp.dot(v, v)
    vr = wp.dot(v, r)
    s = float(0.0)
    t = float(0.0)
    if uu <= 1.0e-5 and vv <= 1.0e-5:
        pass
    elif uu <= 1.0e-5:
        t = saturate_parameter(vr / vv)
    else:
        ur = wp.dot(u, r)
        if vv <= 1.0e-5:
            s = saturate_parameter(-ur / uu)
        else:
            uv = wp.dot(u, v)
            determinant = product_difference(uu, vv, uv, uv)
            if determinant > 0.0:
                s = saturate_parameter(product_difference(uv, vr, ur, vv) / determinant)
            t = (uv * s + vr) / vv
            if t < 0.0:
                t = 0.0
                s = saturate_parameter(-ur / uu)
            elif t > 1.0:
                t = 1.0
                s = saturate_parameter((uv - ur) / uu)
    return s, t


@wp.func
def segment_distance2(p: wp.vec3, q: wp.vec3, a: wp.vec3, b: wp.vec3):
    s, t = segment_parameters(p, q, a, b)
    difference = ((q - p) * s + p) - ((b - a) * t + a)
    return wp.dot(difference, difference)


@wp.func
def point_segment_distance2(p: wp.vec3, a: wp.vec3, direction: wp.vec3):
    length2 = wp.dot(direction, direction)
    t = float(0.0)
    if length2 > 0.0:
        t = saturate_parameter(wp.dot(p - a, direction) / length2)
    difference = p - (direction * t + a)
    return wp.dot(difference, difference)


@wp.func
def triangle_distance2(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    """Ericson's closest point on a triangle, with the native degenerate-triangle fallback."""
    ab = b - a
    ac = c - a
    normal = wp.cross(ab, ac)
    if wp.dot(normal, normal) < 1.0e-20:
        return wp.min(
            point_segment_distance2(p, a, ab),
            wp.min(point_segment_distance2(p, b, c - b), point_segment_distance2(p, c, a - c)),
        )
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.dot(ap, ap)
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.dot(bp, bp)
    vc = d1 * d4 - d2 * d3
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        difference = p - (ab * (d1 / (d1 - d3)) + a)
        return wp.dot(difference, difference)
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.dot(cp, cp)
    vb = d2 * d5 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        difference = p - (ac * (d2 / (d2 - d6)) + a)
        return wp.dot(difference, difference)
    va = d3 * d6 - d4 * d5
    edge0 = d4 - d3
    edge1 = d5 - d6
    if va <= 0.0 and edge0 >= 0.0 and edge1 >= 0.0:
        difference = p - ((c - b) * (edge0 / (edge0 + edge1)) + b)
        return wp.dot(difference, difference)
    inverse = 1.0 / (vc + (va + vb))
    point = ac * (vc * inverse) + (ab * (vb * inverse) + a)
    difference = p - point
    return wp.dot(difference, difference)


# ---------------------------------------------------------------------------------------------------------------------
# Candidate queries


@wp.kernel
def query_particle_pairs(
    x: wp.array[wp.vec3],
    particles: wp.array[wp.int32],
    radius: wp.array[float],
    particle_world: wp.array[wp.int32],
    groups: wp.array[wp.int32],
    node_lower: wp.array[wp.vec3],
    node_upper: wp.array[wp.vec3],
    parent: wp.array[wp.int32],
    children: wp.array[wp.vec2i],
    order: wp.array[wp.int32],
    escape: wp.array[wp.int32],
    leaf: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    particles_per_list: int,
    particles_per_world: int,
    max_slots: int,
):
    """Walk sibling subtrees from the particle's leaf to the root (escape links, preorder); keep partners with a
    larger id, compatible groups and overlapping spheres."""
    i = wp.tid()
    world = i // particles_per_list
    entry = i % particles_per_list
    particle = particles[entry]
    position_base = world * particles_per_world
    position = x[position_base + particle]
    r = radius[particle]
    lower = position - wp.vec3(r)
    upper = position + wp.vec3(r)
    internal_count = particles_per_list - 1
    node_base = world * (2 * particles_per_list - 1)
    node = leaf[entry]
    ancestor = parent[node]
    found = int(0)
    while ancestor >= 0 and found < max_slots:
        previous = node
        node = ancestor
        pair = children[node]
        candidate = pair[0]
        if pair[0] == previous:
            candidate = pair[1]
        stop = int(-1)
        if candidate >= 0:
            stop = escape[candidate]
        while candidate >= 0 and candidate != stop and found < max_slots:
            if overlaps(lower, upper, node_lower[node_base + candidate], node_upper[node_base + candidate]):
                if candidate < internal_count:
                    candidate = children[candidate][0]
                    continue
                other = particles[order[candidate - internal_count]]
                same_group = groups[particle] != 0 and groups[particle] == groups[other]
                if other > particle and not same_group and group_compatible(particle_world, particle, other):
                    difference = position - x[position_base + other]
                    reach = r + radius[other]
                    if wp.dot(difference, difference) < reach * reach:
                        slots[i * max_slots + found] = other
                        found += 1
            candidate = escape[candidate]
        ancestor = parent[node]
    counts[i] = found


@wp.func
def next_mesh_node(
    current: int,
    previous: int,
    hit: bool,
    parent: wp.array[wp.int32],
    children: wp.array[wp.vec2i],
    internal: int,
):
    """One step of the parent/left/right stackless walk: descend into the left child of an entered internal node
    that overlaps, otherwise move to the right sibling or back up."""
    ancestor = parent[current]
    next_node = previous
    if previous == ancestor:
        if hit and current < internal:
            next_node = children[current][0]
    elif current >= internal:
        next_node = ancestor
    elif previous == children[current][0]:
        next_node = children[current][1]
    else:
        next_node = ancestor
    return next_node


@wp.kernel
def query_vertex_face(
    x: wp.array[wp.vec3],
    vertices: wp.array[wp.int32],
    vertex_radius: wp.array[float],
    triangles: wp.array[wp.vec3i],
    groups: wp.array[wp.int32],
    node_lower: wp.array[wp.vec3],
    node_upper: wp.array[wp.vec3],
    parent: wp.array[wp.int32],
    children: wp.array[wp.vec2i],
    order: wp.array[wp.int32],
    radius: float,
    extra: float,
    skip_incident: int,
    counts: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    vertices_per_world: int,
    triangles_per_world: int,
    particles_per_world: int,
    max_slots: int,
):
    i = wp.tid()
    world = i // vertices_per_world
    vertex = vertices[i % vertices_per_world]
    offset = world * particles_per_world
    point = x[offset + vertex]
    r = radius
    if vertex_radius.shape[0] > 0:
        r = vertex_radius[vertex]
    reach = (r + extra) * 1.5
    limit = reach * reach
    lower = point - wp.vec3(r)
    upper = point + wp.vec3(r)
    node_base = world * (2 * triangles_per_world - 1)
    internal = triangles_per_world - 1
    found = int(0)
    if max_slots >= 1:
        previous = int(-1)
        current = int(0)
        while True:
            hit = False
            if previous == parent[current]:
                hit = overlaps(lower, upper, node_lower[node_base + current], node_upper[node_base + current])
                if hit and current >= internal:
                    primitive = order[current - internal]
                    tri = triangles[primitive]
                    accepted = True
                    if skip_incident != 0 and (vertex == tri[0] or vertex == tri[1] or vertex == tri[2]):
                        accepted = False
                    if accepted:
                        accepted = (
                            group_compatible(groups, vertex, tri[0])
                            and group_compatible(groups, vertex, tri[1])
                            and group_compatible(groups, vertex, tri[2])
                        )
                    if accepted:
                        distance = triangle_distance2(point, x[tri[0] + offset], x[tri[1] + offset], x[tri[2] + offset])
                        accepted = distance < limit
                    if accepted:
                        slots[i * max_slots + found] = primitive
                        found += 1
            next_node = next_mesh_node(current, previous, hit, parent, children, internal)
            previous = current
            current = next_node
            if found >= max_slots or current < 0:
                break
    counts[i] = found


@wp.kernel
def query_edge_edge(
    x: wp.array[wp.vec3],
    edges: wp.array[wp.vec4i],
    groups: wp.array[wp.int32],
    node_lower: wp.array[wp.vec3],
    node_upper: wp.array[wp.vec3],
    parent: wp.array[wp.int32],
    children: wp.array[wp.vec2i],
    order: wp.array[wp.int32],
    radius: float,
    counts: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    edges_per_world: int,
    particles_per_world: int,
    max_slots: int,
):
    i = wp.tid()
    world = i // edges_per_world
    edge = i % edges_per_world
    offset = world * particles_per_world
    a = edges[edge][2]
    b = edges[edge][3]
    p = x[a + offset]
    q = x[b + offset]
    lower = wp.min(p, q) - wp.vec3(radius)
    upper = wp.max(p, q) + wp.vec3(radius)
    limit = (radius * 9.0) * radius
    node_base = world * (2 * edges_per_world - 1)
    internal = edges_per_world - 1
    found = int(0)
    if max_slots >= 1:
        previous = int(-1)
        current = int(0)
        while True:
            hit = False
            if previous == parent[current]:
                hit = overlaps(lower, upper, node_lower[node_base + current], node_upper[node_base + current])
                if hit and current >= internal:
                    other = order[current - internal]
                    c = edges[other][2]
                    d = edges[other][3]
                    accepted = not (edge >= other or a == c or a == d or b == c or b == d)
                    if accepted:
                        accepted = (
                            group_compatible(groups, a, c)
                            and group_compatible(groups, a, d)
                            and group_compatible(groups, b, c)
                            and group_compatible(groups, b, d)
                        )
                    if accepted:
                        accepted = segment_distance2(p, q, x[c + offset], x[d + offset]) < limit
                    if accepted:
                        slots[i * max_slots + found] = other
                        found += 1
            next_node = next_mesh_node(current, previous, hit, parent, children, internal)
            previous = current
            current = next_node
            if found >= max_slots or current < 0:
                break
    counts[i] = found


@wp.kernel
def query_edge_face(
    x: wp.array[wp.vec3],
    edges: wp.array[wp.vec4i],
    triangles: wp.array[wp.vec3i],
    groups: wp.array[wp.int32],
    node_lower: wp.array[wp.vec3],
    node_upper: wp.array[wp.vec3],
    parent: wp.array[wp.int32],
    children: wp.array[wp.vec2i],
    order: wp.array[wp.int32],
    radius: float,
    counts: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    edges_per_world: int,
    triangles_per_world: int,
    particles_per_world: int,
    max_slots: int,
):
    """Broad phase only: reject incident or incompatible triangles of a 2r-expanded edge box."""
    i = wp.tid()
    world = i // edges_per_world
    edge = i % edges_per_world
    a = edges[edge][2]
    b = edges[edge][3]
    offset = world * particles_per_world
    p = x[offset + a]
    q = x[offset + b]
    grow = radius + radius
    lower = wp.min(p, q) - wp.vec3(grow)
    upper = wp.vec3(grow) + wp.max(p, q)
    node_base = world * (2 * triangles_per_world - 1)
    internal = triangles_per_world - 1
    found = int(0)
    if max_slots > 0:
        previous = int(-1)
        current = int(0)
        while True:
            hit = False
            if previous == parent[current]:
                hit = overlaps(lower, upper, node_lower[node_base + current], node_upper[node_base + current])
                if hit and current >= internal:
                    primitive = order[current - internal]
                    tri = triangles[primitive]
                    accepted = not (
                        tri[0] == a or tri[1] == a or tri[2] == a or tri[0] == b or tri[1] == b or tri[2] == b
                    )
                    if accepted and groups.shape[0] > 0:
                        accepted = (
                            group_compatible(groups, a, tri[0])
                            and group_compatible(groups, a, tri[1])
                            and group_compatible(groups, a, tri[2])
                            and group_compatible(groups, b, tri[0])
                            and group_compatible(groups, b, tri[1])
                            and group_compatible(groups, b, tri[2])
                        )
                    if accepted:
                        slots[i * max_slots + found] = primitive
                        found += 1
            next_node = next_mesh_node(current, previous, hit, parent, children, internal)
            previous = current
            current = next_node
            if found >= max_slots or current < 0:
                break
    counts[i] = found


# ---------------------------------------------------------------------------------------------------------------------
# Accumulation


@wp.func
def finite_vector(v: wp.vec3):
    return wp.isfinite(v[0]) and wp.isfinite(v[1]) and wp.isfinite(v[2])


@wp.func
def finite_matrix(m: wp.mat33):
    ok = True
    for r in range(3):
        for c in range(3):
            if not wp.isfinite(m[r, c]):
                ok = False
    return ok


@wp.func
def contact_stiffness(first: float, second: float, stiffness: float, scaled: float):
    """Series combination of the two sides' stiffness scales, with the native zero-mass and overflow fallbacks."""
    if first == 0.0 or second == 0.0:
        return (first + second) * stiffness
    numerator = scaled * second
    denominator = first + second
    if wp.isfinite(numerator) and wp.isfinite(denominator):
        return numerator / denominator
    lower = wp.min(first, second)
    upper = wp.max(first, second)
    return (lower / (lower / upper + 1.0)) * stiffness


@wp.func
def record_contact(
    contacts: wp.array[SelfContact],
    counter: wp.array[wp.int32],
    ids: vec5i,
    weights: vec5f,
    normal: wp.vec3,
    k: float,
    count: int,
):
    if contacts.shape[0] == 0:
        return
    index = wp.atomic_add(counter, 0, 1)
    c = SelfContact()
    c.ids = ids
    c.weights = weights
    c.normal = normal
    c.stiffness = k
    c.count = count
    contacts[index] = c


@wp.func
def barycentric_weights(a: wp.vec3, b: wp.vec3, c: wp.vec3, point: wp.vec3):
    """(valid, wa, wb, wc) of point in triangle abc (relative to c), invalid for a near-singular triangle."""
    v0 = a - c
    v1 = b - c
    v2 = point - c
    d00 = wp.dot(v0, v0)
    d01 = wp.dot(v0, v1)
    d02 = wp.dot(v0, v2)
    d11 = wp.dot(v1, v1)
    d12 = wp.dot(v1, v2)
    denominator = d00 * d11 - d01 * d01
    if wp.abs(denominator) < 1.0e-12:
        return 0, 0.0, 0.0, 0.0
    inverse = 1.0 / denominator
    wa = inverse * (d11 * d02 - d01 * d12)
    wb = inverse * (d00 * d12 - d01 * d02)
    wc = (1.0 - wa) - wb
    return 1, wa, wb, wc


@wp.kernel
def accumulate_vertex_face(
    x: wp.array[wp.vec3],
    vertices: wp.array[wp.int32],
    vertex_radius: wp.array[float],
    triangles: wp.array[wp.vec3i],
    counts: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    mass: wp.array[float],
    radius: float,
    extra: float,
    skip_incident: int,
    stiffness: float,
    force: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
    contacts: wp.array[SelfContact],
    counter: wp.array[wp.int32],
    vertices_per_world: int,
    particles_per_world: int,
    max_slots: int,
):
    """Vertex-face penalty along the triangle normal; the query vertex sums locally, triangle vertices scatter."""
    i = wp.tid()
    world = i // vertices_per_world
    local_vertex = vertices[i % vertices_per_world]
    offset = world * particles_per_world
    vertex = local_vertex + offset
    point = x[vertex]
    vertex_mass = mass[vertex]
    r = radius
    if vertex_radius.shape[0] > 0:
        r = vertex_radius[local_vertex]
    limit = r + extra
    if not finite_vector(point) or not wp.isfinite(vertex_mass) or not wp.isfinite(r) or not wp.isfinite(limit):
        return
    scaled = vertex_mass * stiffness
    accumulated_force = wp.vec3(0.0)
    accumulated_diagonal = wp.mat33(0.0)
    for slot in range(counts[i]):
        triangle = triangles[slots[i * max_slots + slot]]
        if skip_incident != 0 and (
            local_vertex == triangle[0] or local_vertex == triangle[1] or local_vertex == triangle[2]
        ):
            continue
        ids = wp.vec3i(triangle[0] + offset, triangle[1] + offset, triangle[2] + offset)
        a = x[ids[0]]
        b = x[ids[1]]
        c = x[ids[2]]
        if not finite_vector(a) or not finite_vector(b) or not finite_vector(c):
            continue
        cross = wp.cross(b - a, c - a)
        length = wp.sqrt(wp.dot(cross, cross))
        normal = wp.vec3(0.0)
        if not (length < 1.0e-12):
            normal = cross / length
        if not finite_vector(normal):
            continue
        xx = normal[0] * normal[0]
        yy = normal[1] * normal[1]
        zz = normal[2] * normal[2]
        if zz + (xx + yy) == 0.0:
            continue
        distance = wp.dot(point - a, normal)
        if not wp.isfinite(distance) or wp.abs(distance) > limit:
            continue
        valid, wa, wb, wc = barycentric_weights(a, b, c, point - distance * normal)
        if valid == 0:
            continue
        if not wp.isfinite(wa) or not wp.isfinite(wb) or not wp.isfinite(wc) or wa < 0.0 or wb < 0.0 or wc < 0.0:
            continue
        face_mass = ((mass[ids[1]] + mass[ids[0]]) + mass[ids[2]]) / 3.0
        k = contact_stiffness(vertex_mass, face_mass, stiffness, scaled)
        sign = float(0.0)
        if distance > 0.0:
            sign = 1.0
        elif distance < 0.0:
            sign = -1.0
        f = (sign * ((limit - wp.abs(distance)) * k)) * normal
        h = wp.outer(normal, normal) * k
        if not finite_vector(f) or not finite_matrix(h):
            continue
        accumulated_force = accumulated_force + f
        accumulated_diagonal = accumulated_diagonal + h
        record_contact(
            contacts, counter, vec5i(vertex, ids[0], ids[1], ids[2], 0), vec5f(1.0, -wa, -wb, -wc, 0.0), normal, k, 4
        )
        wp.atomic_add(force, ids[0], f * -wa)
        wp.atomic_add(diagonal, ids[0], (wa * wa) * h)
        wp.atomic_add(force, ids[1], f * -wb)
        wp.atomic_add(diagonal, ids[1], (wb * wb) * h)
        wp.atomic_add(force, ids[2], f * -wc)
        wp.atomic_add(diagonal, ids[2], (wc * wc) * h)
    wp.atomic_add(force, vertex, accumulated_force)
    wp.atomic_add(diagonal, vertex, accumulated_diagonal)


@wp.kernel
def accumulate_particle_pairs(
    x: wp.array[wp.vec3],
    particles: wp.array[wp.int32],
    radius: wp.array[float],
    counts: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    mass: wp.array[float],
    stiffness: float,
    force: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
    contacts: wp.array[SelfContact],
    counter: wp.array[wp.int32],
    particles_per_list: int,
    particles_per_world: int,
    max_slots: int,
):
    i = wp.tid()
    world = i // particles_per_list
    local_particle = particles[i % particles_per_list]
    offset = world * particles_per_world
    particle = local_particle + offset
    position = x[particle]
    first_mass = mass[particle]
    if not finite_vector(position) or not wp.isfinite(first_mass):
        return
    scaled = first_mass * stiffness
    for slot in range(counts[i]):
        local_other = slots[i * max_slots + slot]
        other = local_other + offset
        other_position = x[other]
        other_radius = radius[local_other]
        if not finite_vector(other_position) or not wp.isfinite(other_radius):
            continue
        difference = position - other_position
        distance = wp.sqrt(wp.dot(difference, difference))
        reach = radius[local_particle] + other_radius
        if not wp.isfinite(distance) or not wp.isfinite(reach) or distance >= reach:
            continue
        normal = wp.vec3(0.0)
        if distance > 1.0e-8:
            normal = difference / distance
        else:
            axis = (local_other ^ local_particle) % 3
            normal[axis] = 1.0
        k = contact_stiffness(first_mass, mass[other], stiffness, scaled)
        if not wp.isfinite(k):
            continue
        f = ((reach - distance) * k) * normal
        h = wp.outer(normal, normal) * k
        if not finite_vector(f) or not finite_matrix(h):
            continue
        wp.atomic_add(force, particle, f)
        wp.atomic_add(force, other, -f)
        wp.atomic_add(diagonal, particle, h)
        wp.atomic_add(diagonal, other, h)
        record_contact(
            contacts, counter, vec5i(particle, other, 0, 0, 0), vec5f(1.0, -1.0, 0.0, 0.0, 0.0), normal, k, 2
        )


@wp.kernel
def multiply_contact_tangents(
    contacts: wp.array[SelfContact],
    counter: wp.array[wp.int32],
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    stride: int,
):
    """Grid-stride over the recorded contacts with at most 256 x 256 threads, as natively; one thread per capacity
    slot (pairs x 31) would launch millions of idle threads every CG iteration."""
    for i in range(wp.tid(), counter[0], stride):
        c = contacts[i]
        total = wp.vec3(0.0)
        for point in range(c.count):
            total = c.weights[point] * direction[c.ids[point]] + total
        tangent = (c.stiffness * wp.dot(total, c.normal)) * c.normal
        for point in range(c.count):
            wp.atomic_add(product, c.ids[point], tangent * c.weights[point])


@wp.func
def opposite_contains(offsets: wp.array[wp.int32], opposite: wp.array[wp.int32], edge: int, vertex: int):
    for entry in range(offsets[edge], offsets[edge + 1]):
        if opposite[entry] == vertex:
            return True
    return False


@wp.kernel
def accumulate_edge_edge(
    x: wp.array[wp.vec3],
    edges: wp.array[wp.vec4i],
    opposite_offsets: wp.array[wp.int32],
    opposite: wp.array[wp.int32],
    counts: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    mass: wp.array[float],
    thickness: float,
    stiffness: float,
    force: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
    contacts: wp.array[SelfContact],
    counter: wp.array[wp.int32],
    edges_per_world: int,
    particles_per_world: int,
    max_slots: int,
):
    """Edge-edge penalty for interior closest points; opposite-vertex adjacency can reduce the margin."""
    i = wp.tid()
    world = i // edges_per_world
    edge = i % edges_per_world
    offset = world * particles_per_world
    za = edges[edge][2]
    wa = edges[edge][3]
    first = za + offset
    second = wa + offset
    p = x[first]
    q = x[second]
    length = wp.length(p - q)
    edge_mass = (mass[first] + mass[second]) * 0.5
    if not finite_vector(p) or not finite_vector(q) or not wp.isfinite(length) or not wp.isfinite(edge_mass):
        return
    scaled = edge_mass * stiffness
    first_force = wp.vec3(0.0)
    second_force = wp.vec3(0.0)
    first_diagonal = wp.mat33(0.0)
    second_diagonal = wp.mat33(0.0)
    for slot in range(counts[i]):
        other = slots[i * max_slots + slot]
        if edge >= other:
            continue
        zc = edges[other][2]
        wc = edges[other][3]
        if za == zc or za == wc or wa == zc or wa == wc:
            continue
        third = zc + offset
        fourth = wc + offset
        a = x[third]
        b = x[fourth]
        if not finite_vector(a) or not finite_vector(b):
            continue
        s, t = segment_parameters(p, q, a, b)
        if not wp.isfinite(s) or not wp.isfinite(t) or s <= 0.0 or s >= 1.0 or t <= 0.0 or t >= 1.0:
            continue
        difference = ((q - p) * s + p) - ((b - a) * t + a)
        distance = wp.length(difference)
        other_length = wp.length(a - b)
        mean = (length + other_length) * 0.5
        if not wp.isfinite(distance) or not wp.isfinite(mean):
            continue
        limit = thickness
        if (
            opposite_contains(opposite_offsets, opposite, edge, zc)
            or opposite_contains(opposite_offsets, opposite, edge, wc)
            or opposite_contains(opposite_offsets, opposite, other, za)
            or opposite_contains(opposite_offsets, opposite, other, wa)
        ):
            limit = wp.min(thickness, mean * 0.5)
        if distance >= limit or distance <= 1.0e-6:
            continue
        normal = difference / distance
        other_mass = (mass[third] + mass[fourth]) * 0.5
        k = contact_stiffness(edge_mass, other_mass, stiffness, scaled)
        if not wp.isfinite(k):
            continue
        f = normal * ((limit - distance) * k)
        h = wp.outer(normal, normal) * k
        if not finite_vector(f) or not finite_matrix(h):
            continue
        one_s = 1.0 - s
        one_t = 1.0 - t
        record_contact(
            contacts, counter, vec5i(first, second, third, fourth, 0), vec5f(one_s, s, -one_t, -t, 0.0), normal, k, 4
        )
        first_force = f * one_s + first_force
        first_diagonal = h * (one_s * one_s) + first_diagonal
        wp.atomic_add(force, third, f * -one_t)
        wp.atomic_add(diagonal, third, (one_t * one_t) * h)
        second_force = f * s + second_force
        second_diagonal = h * (s * s) + second_diagonal
        wp.atomic_add(force, fourth, f * -t)
        wp.atomic_add(diagonal, fourth, (t * t) * h)
    wp.atomic_add(force, first, first_force)
    wp.atomic_add(diagonal, first, first_diagonal)
    wp.atomic_add(force, second, second_force)
    wp.atomic_add(diagonal, second, second_diagonal)


@wp.kernel
def accumulate_edge_face(
    x: wp.array[wp.vec3],
    edges: wp.array[wp.vec4i],
    opposite_offsets: wp.array[wp.int32],
    opposite: wp.array[wp.int32],
    triangles: wp.array[wp.vec3i],
    counts: wp.array[wp.int32],
    slots: wp.array[wp.int32],
    mass: wp.array[float],
    thickness: float,
    stiffness: float,
    force: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
    contacts: wp.array[SelfContact],
    counter: wp.array[wp.int32],
    edges_per_world: int,
    particles_per_world: int,
    max_slots: int,
):
    """Edge-face intersections pushed apart along a direction folded from the edge's adjacent faces."""
    i = wp.tid()
    world = i // edges_per_world
    edge = i % edges_per_world
    offset = world * particles_per_world
    z = edges[edge][2]
    w = edges[edge][3]
    first = z + offset
    second = w + offset
    xz = x[first]
    xw = x[second]
    e = xz - xw
    length = wp.length(e)
    if not finite_vector(xz) or not finite_vector(xw) or not wp.isfinite(length) or length < 0.0005:
        return
    direction = wp.vec3(0.0)
    if length > 0.0:
        direction = e / length
    edge_mass = (mass[first] + mass[second]) * 0.5
    if not wp.isfinite(edge_mass):
        return
    scaled = edge_mass * stiffness
    twice = thickness + thickness
    first_force = wp.vec3(0.0)
    second_force = wp.vec3(0.0)
    first_diagonal = wp.mat33(0.0)
    second_diagonal = wp.mat33(0.0)
    for slot in range(counts[i]):
        tri = triangles[slots[i * max_slots + slot]]
        if z == tri[0] or z == tri[1] or z == tri[2] or w == tri[0] or w == tri[1] or w == tri[2]:
            continue
        ids = wp.vec3i(tri[0] + offset, tri[1] + offset, tri[2] + offset)
        a = x[ids[0]]
        b = x[ids[1]]
        c = x[ids[2]]
        if not finite_vector(a) or not finite_vector(b) or not finite_vector(c):
            continue
        cross = wp.cross(b - a, c - b)
        area = wp.length(cross)
        if not wp.isfinite(area) or area < 1.0e-8:
            continue
        tn = cross / area
        dz = wp.dot(xz - a, tn)
        dw = wp.dot(xw - a, tn)
        if not wp.isfinite(dz) or not wp.isfinite(dw) or dz * dw >= 0.0:
            continue
        az = wp.abs(dz)
        aw = wp.abs(dw)
        total = az + aw
        if not wp.isfinite(total) or total <= 0.0:
            continue
        point = (xw * az + xz * aw) / total
        valid, wa, wb, wc = barycentric_weights(a, b, c, point)
        if valid == 0:
            continue
        if not wp.isfinite(wa) or not wp.isfinite(wb) or not wp.isfinite(wc) or wa < 0.01 or wb < 0.01 or wc < 0.01:
            continue
        accumulated = wp.vec3(0.0)
        tn_direction = wp.dot(tn, direction)
        for entry in range(opposite_offsets[edge], opposite_offsets[edge + 1]):
            xo = x[opposite[entry] + offset]
            face_cross = wp.cross(xw - xz, xo - xz)
            face_length = wp.length(face_cross)
            face_normal = wp.vec3(0.0)
            if not (face_length < 1.0e-12):
                face_normal = face_cross / face_length
            if not finite_vector(face_normal):
                continue
            folded_cross = wp.cross(tn, face_normal)
            folded_length = wp.length(folded_cross)
            g = wp.vec3(0.0)
            if not (folded_length < 1.0e-6):
                g = folded_cross / folded_length
            hh = wp.cross(direction, g)
            m = wp.cross(direction, xo - point)
            if wp.dot(m, hh) < 0.0:
                g = -g
            if wp.abs(tn_direction) > 1.0e-6:
                g = g - ((tn + tn) * wp.dot(direction, g)) / tn_direction
            accumulated = accumulated + g
        direction_length = wp.length(accumulated)
        if not wp.isfinite(direction_length) or direction_length < 1.0e-12:
            continue
        normal = accumulated / direction_length
        face_mass = ((mass[ids[1]] + mass[ids[0]]) + mass[ids[2]]) / 3.0
        k = contact_stiffness(edge_mass, face_mass, stiffness, scaled)
        if not wp.isfinite(k):
            continue
        f = normal * (twice * k)
        h = wp.outer(normal, normal) * k
        if not finite_vector(f) or not finite_matrix(h):
            continue
        wz = aw / total
        ww = az / total
        record_contact(
            contacts, counter, vec5i(first, second, ids[0], ids[1], ids[2]), vec5f(wz, ww, -wa, -wb, -wc), normal, k, 5
        )
        first_force = f * wz + first_force
        first_diagonal = h * (wz * wz) + first_diagonal
        second_force = f * ww + second_force
        second_diagonal = h * (ww * ww) + second_diagonal
        wp.atomic_add(force, ids[0], f * -wa)
        wp.atomic_add(diagonal, ids[0], (wa * wa) * h)
        wp.atomic_add(force, ids[1], f * -wb)
        wp.atomic_add(diagonal, ids[1], (wb * wb) * h)
        wp.atomic_add(force, ids[2], f * -wc)
        wp.atomic_add(diagonal, ids[2], (wc * wc) * h)
    wp.atomic_add(force, first, first_force)
    wp.atomic_add(diagonal, first, first_diagonal)
    wp.atomic_add(force, second, second_force)
    wp.atomic_add(diagonal, second, second_diagonal)

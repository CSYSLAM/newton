# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Augmented-Lagrangian ("hard") rigid contacts with warm-started duals (native ``kernels/hard_contacts.cu``,
``hard_evaluate.cu`` and ``hard_contact_geometry.cuh``).

Each contact keeps a penalty, a dual (normal and tangential multiplier) and a displacement anchor. Duals are projected
onto the Coulomb cone after every Newton iteration; matched contacts of the previous step warm-start the next.
"""

import warp as wp

from .common import normalized_or_zero
from .contact_geometry import (
    contact_body_slot,
    contact_dot,
    contact_dot_x_first,
    contact_normalize,
    contact_point,
    contact_point_velocity,
    contact_shape,
    contact_sphere_offset,
    contact_wrench,
)

# The solver never differentiates through its kernels; skipping the adjoints halves compile time.
wp.set_module_options({"enable_backward": False})


vec8i = wp.types.vector(length=8, dtype=wp.int32)
vec8f = wp.types.vector(length=8, dtype=wp.float32)


@wp.struct
class HardContactView:
    count: wp.array[wp.int32]
    capacity: int
    shape0: wp.array[wp.int32]
    shape1: wp.array[wp.int32]
    point0: wp.array[wp.vec3]
    point1: wp.array[wp.vec3]
    offset0: wp.array[wp.vec3]
    offset1: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    margin0: wp.array[float]
    margin1: wp.array[float]
    match: wp.array[wp.int32]
    shape_body: wp.array[wp.int32]
    shape_type: wp.array[wp.int32]
    shape_transform: wp.array[wp.transform]
    shape_scale: wp.array[wp.vec3]
    shape_ke: wp.array[float]
    shape_kd: wp.array[float]
    shape_mu: wp.array[float]
    local_shape_count: int
    shapes_per_world: int
    worlds: int
    body_ids: wp.array[wp.int32]
    body_indices: wp.array[wp.vec4i]
    body_com: wp.array[wp.vec3]
    body_count: int
    particles_per_world: int
    x: wp.array[wp.vec3]
    previous: wp.array[wp.vec3]
    inv_dt: float
    alpha: float
    beta: float
    gamma: float
    start: float
    history: int
    exact_friction: int
    reset: int
    penalty: wp.array[float]
    dual: wp.array[wp.vec3]
    anchor: wp.array[wp.vec3]
    previous_penalty: wp.array[float]
    previous_dual: wp.array[wp.vec3]
    previous_normal: wp.array[wp.vec3]
    previous_count: wp.array[wp.int32]


@wp.struct
class HardGeometry:
    valid: int
    shape0: int
    shape1: int
    slot0: int
    slot1: int
    ids0: wp.vec4i
    ids1: wp.vec4i
    com0: wp.vec3
    com1: wp.vec3
    adjusted0: wp.vec3
    adjusted1: wp.vec3
    normal: wp.vec3
    depth: float


@wp.func
def hard_active(v: HardContactView, i: int):
    if i >= v.capacity or v.count.shape[0] == 0:
        return False
    return i < wp.max(v.count[0], 0)


@wp.func
def body_point(point: wp.vec3, slot: int, ids: wp.vec4i, com: wp.vec3, x: wp.array[wp.vec3]):
    if slot >= 0:
        return contact_point(point, ids, com, x)
    return point


@wp.func
def hard_geometry(v: HardContactView, i: int):
    """Depth keeps the supplied normal and raw pipeline points; the sphere-centre direction only affects the
    adjusted friction levers, and explicit offsets override them."""
    g = HardGeometry()
    g.valid = 0
    ok0, world0, s0 = contact_shape(v.shape0[i], v.local_shape_count, v.shapes_per_world, v.worlds)
    if ok0 == 0:
        return g
    ok1, world1, s1 = contact_shape(v.shape1[i], v.local_shape_count, v.shapes_per_world, v.worlds)
    if ok1 == 0:
        return g
    body0 = v.shape_body[s0]
    slot0 = contact_body_slot(body0, v.body_ids)
    if (body0 >= 0 and slot0 < 0) or (slot0 >= 0 and world0 < 0):
        return g
    body1 = v.shape_body[s1]
    slot1 = contact_body_slot(body1, v.body_ids)
    if (body1 >= 0 and slot1 < 0) or (slot1 >= 0 and world1 < 0):
        return g
    if slot0 < 0 and slot1 < 0:
        return g
    if world0 >= 0 and world1 >= 0 and world0 != world1:
        return g
    g.shape0 = s0
    g.shape1 = s1
    g.slot0 = slot0
    g.slot1 = slot1
    if slot0 >= 0:
        b0 = v.body_indices[slot0]
        o0 = world0 * v.particles_per_world
        g.ids0 = wp.vec4i(b0[0] + o0, b0[1] + o0, b0[2] + o0, b0[3] + o0)
        g.com0 = v.body_com[slot0]
    if slot1 >= 0:
        b1 = v.body_indices[slot1]
        o1 = world1 * v.particles_per_world
        g.ids1 = wp.vec4i(b1[0] + o1, b1[1] + o1, b1[2] + o1, b1[3] + o1)
        g.com1 = v.body_com[slot1]
    x = v.x
    point0 = v.point0[i]
    point1 = v.point1[i]
    sphere0 = v.shape_type[s0] == 3
    sphere1 = v.shape_type[s1] == 3
    raw0 = point0
    if sphere0:
        raw0 = wp.transform_get_translation(v.shape_transform[s0])
    raw1 = point1
    if sphere1:
        raw1 = wp.transform_get_translation(v.shape_transform[s1])
    supplied = contact_normalize(v.normal[i])
    direction = wp.vec3(0.0)
    if sphere0 and sphere1:
        direction = contact_normalize(
            body_point(raw1, slot1, g.ids1, g.com1, x) - body_point(raw0, slot0, g.ids0, g.com0, x)
        )
    if not (contact_dot(direction, direction) >= 0.5):
        direction = supplied
    if not (contact_dot(direction, direction) >= 0.5):
        return g
    g.adjusted0 = raw0
    if sphere0:
        radius0 = v.shape_scale[s0][0]
        if slot0 >= 0:
            g.adjusted0 = raw0 + contact_sphere_offset(radius0 * direction, g.ids0, x)
        else:
            g.adjusted0 = raw0 + radius0 * direction
    if v.offset0.shape[0] > 0:
        g.adjusted0 = point0 + v.offset0[i]
    g.adjusted1 = raw1
    if sphere1:
        radius1 = v.shape_scale[s1][0]
        if slot1 >= 0:
            g.adjusted1 = raw1 + contact_sphere_offset(-(radius1 * direction), g.ids1, x)
        else:
            g.adjusted1 = raw1 - radius1 * direction
    if v.offset1.shape[0] > 0:
        g.adjusted1 = point1 + v.offset1[i]
    g.normal = supplied
    n = g.normal
    if contact_dot_x_first(n, n) < 0.5:
        return g
    p0 = body_point(point0, slot0, g.ids0, g.com0, x)
    p1 = body_point(point1, slot1, g.ids1, g.com1, x)
    g.depth = (v.margin0[i] + v.margin1[i]) - contact_dot(p1 - p0, n)
    g.valid = 1
    return g


@wp.kernel
def snapshot_hard_contacts(v: HardContactView):
    """Store the clamped count (thread 0) and every active contact's penalty, dual and unit normal."""
    i = wp.tid()
    if i == 0:
        count = int(0)
        if v.count.shape[0] > 0:
            count = wp.min(wp.max(v.count[0], 0), v.capacity)
        v.previous_count[0] = count
    if not hard_active(v, i):
        return
    v.previous_penalty[i] = v.penalty[i]
    v.previous_dual[i] = v.dual[i]
    normal = v.normal[i]
    v.previous_normal[i] = normalized_or_zero(normal, wp.sqrt(wp.dot(normal, normal)), 0.0)


@wp.kernel
def initialize_hard_contacts(v: HardContactView):
    """Initialise penalties, rotate matched duals into the current normal and record the displacement anchor."""
    i = wp.tid()
    if not hard_active(v, i):
        return
    g = hard_geometry(v, i)
    if g.valid == 0:
        v.penalty[i] = 0.0
        v.dual[i] = wp.vec3(0.0)
        v.anchor[i] = wp.vec3(0.0)
        return
    upper = (v.shape_ke[g.shape0] + v.shape_ke[g.shape1]) * 0.5
    lower = upper
    if v.beta > 0.0:
        lower = wp.min(v.start, upper)
    penalty = lower
    dual = wp.vec3(0.0)
    if v.reset == 0 and v.previous_count[0] != 0:
        penalty = v.penalty[i]
        dual = v.dual[i]
    else:
        match = int(-1)
        if v.history != 0 and v.match.shape[0] > 0:
            match = v.match[i]
        if match >= 0 and match < v.previous_count[0]:
            penalty = wp.min(wp.max(v.previous_penalty[match], lower), upper)
            old_n = v.previous_normal[match]
            old_dual = v.previous_dual[match]
            normal_part = contact_dot(old_n, old_dual)
            tangent = old_dual - old_n * normal_part
            dual = normal_part * g.normal + tangent - contact_dot(tangent, g.normal) * g.normal
    v.penalty[i] = wp.min(wp.max(penalty * v.gamma, lower), upper)
    if v.history == 0 and v.reset != 0:
        v.dual[i] = wp.vec3(0.0)
    else:
        v.dual[i] = dual * (v.alpha * v.gamma)
    p0 = body_point(g.adjusted0, g.slot0, g.ids0, g.com0, v.x)
    p1 = body_point(g.adjusted1, g.slot1, g.ids1, g.com1, v.x)
    diff = p1 - p0
    tangent = diff - contact_dot(diff, g.normal) * g.normal
    v.anchor[i] = g.depth * g.normal - tangent


@wp.func
def relative_displacement(v: HardContactView, g: HardGeometry):
    now0 = body_point(g.adjusted0, g.slot0, g.ids0, g.com0, v.x)
    now1 = body_point(g.adjusted1, g.slot1, g.ids1, g.com1, v.x)
    old0 = body_point(g.adjusted0, g.slot0, g.ids0, g.com0, v.previous)
    old1 = body_point(g.adjusted1, g.slot1, g.ids1, g.com1, v.previous)
    return ((now1 - old1) - now0) + old0


@wp.kernel
def update_hard_contact_duals(v: HardContactView):
    """Project the dual onto the Coulomb cone and ramp the penalty by beta times the depth."""
    i = wp.tid()
    if not hard_active(v, i):
        return
    g = hard_geometry(v, i)
    if g.valid == 0:
        return
    n = g.normal
    anchor = v.anchor[i]
    dual = v.dual[i]
    an = contact_dot(anchor, n)
    dn = contact_dot(dual, n)
    corrected_depth = g.depth
    if g.depth >= 0.0:
        corrected_depth = g.depth - v.alpha * an
    penalty = v.penalty[i]
    normal_force = wp.max(penalty * corrected_depth + dn, 0.0)
    delta = relative_displacement(v, g)
    correction = contact_dot(delta, n) * n - delta
    correction = (1.0 - v.alpha) * (anchor - an * n) + correction
    tangent = penalty * correction + (dual - dn * n)
    length = wp.sqrt(contact_dot(tangent, tangent))
    mu = wp.sqrt(wp.max(v.shape_mu[g.shape0] * v.shape_mu[g.shape1], 0.0))
    bound = normal_force * mu
    if length > bound and length > 0.0:
        tangent = (bound / length) * tangent
    v.dual[i] = normal_force * n + tangent
    if g.depth > 0.0:
        upper = (v.shape_ke[g.shape0] + v.shape_ke[g.shape1]) * 0.5
        v.penalty[i] = wp.min(upper, v.beta * g.depth + penalty)


@wp.kernel
def evaluate_hard_contacts(
    v: HardContactView,
    force: wp.array[wp.vec3],
    scalar: wp.array[float],
    direction: wp.array[wp.vec3],
    product: wp.array[wp.vec3],
    system: wp.array[wp.mat33],
    precond: wp.array[wp.mat33],
    affine_blocks: wp.array[float],
    contact_force: wp.array[wp.spatial_vector],
):
    """Hard-contact force, scalar diagonal, off-diagonal product, split Hessian blocks and body cross blocks."""
    i = wp.tid()
    if not hard_active(v, i):
        return
    g = hard_geometry(v, i)
    if g.valid == 0:
        return
    n = g.normal
    dual = v.dual[i]
    anchor = v.anchor[i]
    dn = contact_dot(dual, n)
    an = contact_dot(anchor, n)
    gap = g.depth - v.alpha * an
    penalty = v.penalty[i]
    if not (penalty > 0.0) or ((g.depth <= 1.0e-9 or gap <= 1.0e-9) and dn <= 0.0):
        return
    fn = wp.max(penalty * gap + dn, 0.0)
    delta = relative_displacement(v, g)
    mu = wp.sqrt(wp.max(v.shape_mu[g.shape0] * v.shape_mu[g.shape1], 0.0))
    tangent = wp.vec3(0.0)
    unit = wp.vec3(0.0)
    kt = float(0.0)
    sliding = False
    if mu > 0.0 and fn > 0.0:
        correction = contact_dot(delta, n) * n - delta
        correction = (1.0 - v.alpha) * (anchor - an * n) + correction
        tangent = (penalty * correction + dual) - dn * n
        length = wp.sqrt(contact_dot(tangent, tangent))
        bound = fn * mu
        kt = penalty
        if length > bound and length > 0.0:
            factor = bound / length
            if v.exact_friction != 0:
                unit = tangent / length
                kt = penalty * factor
                sliding = True
            tangent = tangent * factor
    raw0 = v.point0[i]
    raw1 = v.point1[i]
    velocity0 = wp.vec3(0.0)
    velocity1 = wp.vec3(0.0)
    if g.slot0 >= 0:
        velocity0 = contact_point_velocity(raw0, g.ids0, g.com0, v.x, v.previous, v.previous, v.inv_dt)
    if g.slot1 >= 0:
        velocity1 = contact_point_velocity(raw1, g.ids1, g.com1, v.x, v.previous, v.previous, v.inv_dt)
    vn = contact_dot(velocity1 - velocity0, n)
    damping = float(0.0)
    if fn > 0.0 and vn < 0.0:
        damping = (v.shape_kd[g.shape0] + v.shape_kd[g.shape1]) * 0.5
    normal_force = fn * n - (vn * damping) * n
    if contact_force.shape[0] > 0:
        contact_force[i] = contact_wrench(g.slot0, g.ids0, g.com0, v.x, raw0, g.adjusted0, -normal_force, -tangent)
    kn = damping * v.inv_dt + penalty
    targets = vec8i()
    normal_weights = vec8f()
    tangent_weights = vec8f()
    entries = int(0)
    for side in range(2):
        slot = g.slot0
        ids = g.ids0
        com = g.com0
        raw = raw0
        adjusted = g.adjusted0
        sign = float(-1.0)
        if side == 1:
            slot = g.slot1
            ids = g.ids1
            com = g.com1
            raw = raw1
            adjusted = g.adjusted1
            sign = 1.0
        if slot < 0:
            continue
        rn = raw - com
        rt = adjusted - com
        for k in range(4):
            targets[entries + k] = ids[k]
        normal_weights[entries] = sign
        tangent_weights[entries] = sign
        for k in range(3):
            normal_weights[entries + k + 1] = sign * rn[k]
            tangent_weights[entries + k + 1] = sign * rt[k]
        if force.shape[0] > 0:
            for part in range(2):
                value = normal_force
                lever = rn
                if part == 1:
                    value = tangent
                    lever = rt
                if side == 0:
                    value = -value
                wp.atomic_add(force, ids[0], value)
                for k in range(3):
                    wp.atomic_add(force, ids[k + 1], lever[k] * value)
        if scalar.shape[0] > 0:
            for part in range(2):
                stiffness = kn
                lever = rn
                if part == 1:
                    stiffness = kt
                    lever = rt
                wp.atomic_add(scalar, ids[0], stiffness)
                for k in range(3):
                    wp.atomic_add(scalar, ids[k + 1], lever[k] * (stiffness * lever[k]))
        entries += 4
    nn = wp.outer(n, n)
    identity = wp.identity(n=3, dtype=float)
    normal_matrix = kn * nn
    tangent_matrix = kt * (identity - nn)
    if sliding:
        tangent_matrix = tangent_matrix - kt * wp.outer(unit, unit)
    if direction.shape[0] > 0 and product.shape[0] > 0:
        for part in range(2):
            matrix = normal_matrix
            if part == 1:
                matrix = tangent_matrix
            for a in range(entries):
                total = wp.vec3(0.0)
                for b in range(entries):
                    if targets[b] == targets[a]:
                        continue
                    weight = normal_weights[b]
                    if part == 1:
                        weight = tangent_weights[b]
                    total = weight * direction[targets[b]] + total
                wa = normal_weights[a]
                if part == 1:
                    wa = tangent_weights[a]
                wp.atomic_add(product, targets[a], (matrix * total) * wa)
    if affine_blocks.shape[0] > 0:
        last_body = int(-1)
        for side in range(2):
            slot = g.slot0
            ids = g.ids0
            shape = v.shape0[i]
            if side == 1:
                slot = g.slot1
                ids = g.ids1
                shape = v.shape1[i]
            if slot < 0:
                continue
            body = slot + (shape // v.shapes_per_world) * v.body_count
            if body == last_body:
                continue
            last_body = body
            wn = wp.vec4(0.0)
            wt = wp.vec4(0.0)
            for a in range(entries):
                for k in range(4):
                    if targets[a] == ids[k]:
                        wn[k] = wn[k] + normal_weights[a]
                        wt[k] = wt[k] + tangent_weights[a]
            for row in range(4):
                for col in range(4):
                    if row == col:
                        continue
                    sa = wn[row] * wn[col]
                    sb = wt[row] * wt[col]
                    for r in range(3):
                        for c in range(3):
                            cross = sa * normal_matrix[r, c] + sb * tangent_matrix[r, c]
                            wp.atomic_add(affine_blocks, body * 144 + (3 * row + r) * 12 + 3 * col + c, cross)
    if system.shape[0] > 0 or precond.shape[0] > 0:
        for a in range(entries):
            n2 = normal_weights[a] * normal_weights[a]
            t2 = tangent_weights[a] * tangent_weights[a]
            iso = kn * n2 + kt * t2
            block = n2 * normal_matrix + t2 * tangent_matrix - iso * identity
            if system.shape[0] > 0:
                wp.atomic_add(system, targets[a], block)
            if precond.shape[0] > 0:
                wp.atomic_add(precond, targets[a], block)

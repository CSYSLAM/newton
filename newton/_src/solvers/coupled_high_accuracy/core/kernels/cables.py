# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cosserat-style cables between affine bodies (native ``kernels/cables.cu``, ``cable_evaluate.cu``,
``cable_tangents.cu`` and the ``cable_*.cuh`` headers).

A cable joins a parent and a child frame (body-attached or fixed). Its strain has a linear part (stretch along the
parent z axis, shear along x/y) and an angular part (discrete bend in x/y and twist about z). Body rotations come
from an inertia-weighted polar decomposition of the affine columns, evaluated in float64; the derivatives with
respect to the affine columns go through the corresponding spin maps. Each cable's Jacobians (up to eight proxy
slots) are prepared once per Newton iteration and reused by the tangent and preconditioner.
"""

import warp as wp

from .affine import column, nearest_rotation
from .common import normalized_or_zero
from .contact_geometry import contact_dot, contact_normalize, contact_point

vec8i = wp.types.vector(length=8, dtype=wp.int32)
PI = wp.constant(3.14159265358979323846)
TWO_PI = wp.constant(6.28318530717958647692)


@wp.struct
class CableView:
    body_indices: wp.array[wp.vec4i]
    body_com: wp.array[wp.vec3]
    inertia: wp.array[float]
    body_pairs: wp.array[wp.vec2i]
    frame0: wp.array[wp.transform]
    frame1: wp.array[wp.transform]
    base: wp.array[wp.vec4]
    penalty: wp.array[wp.vec4]
    damping: wp.array[wp.vec4]
    rest: wp.array[wp.vec3]
    state: wp.array[wp.vec3]
    bias: wp.array[wp.vec3]
    active: wp.array[wp.int32]
    linear: wp.array[wp.mat33]
    angular: wp.array[wp.mat33]
    indices: wp.array[wp.int32]
    count: wp.array[wp.int32]
    per_world: int
    total_count: int
    particles_per_world: int


@wp.func
def authored_basis(frame: wp.transform):
    """Rotation matrix of the frame quaternion: column c = e_c + 2 q x (w e_c + q x e_c)."""
    q4 = wp.transform_get_rotation(frame)
    q = wp.vec3(q4[0], q4[1], q4[2])
    w = q4[3]
    basis = wp.mat33(0.0)
    for c in range(3):
        e = wp.vec3(0.0)
        e[c] = 1.0
        u = w * e + wp.cross(q, e)
        rotated = wp.cross(q, u)
        for r in range(3):
            basis[r, c] = rotated[r] * 2.0 + e[r]
    return basis


# ---------------------------------------------------------------------------------------------------------------------
# Inertia-weighted polar rotation (float64) and spin maps


@wp.struct
class CableSpin:
    rotation: wp.mat33
    map0: wp.mat33
    map1: wp.mat33
    map2: wp.mat33
    weighted: int


@wp.func
def spin_map(spin: CableSpin, k: int):
    if k == 0:
        return spin.map0
    if k == 1:
        return spin.map1
    return spin.map2


@wp.func
def norm_squared_d(m: wp.mat33d):
    total = wp.float64(0.0)
    for c in range(3):
        for r in range(3):
            total = total + m[r, c] * m[r, c]
    return total


@wp.func
def cable_rotation_spin(a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, inertia: wp.array[float], base: int, maps: int):
    """Rotation minimising the inertia-weighted affine distance (scaled Newton polar iteration in float64) and,
    with ``maps``, the spin maps H^-1 [v]x; falls back to the plain nearest rotation in float32."""
    result = CableSpin()
    result.weighted = 0
    a = wp.matrix_from_cols(wp.vec3d(a0), wp.vec3d(a1), wp.vec3d(a2))
    half = (
        (wp.float64(inertia[base + 0]) + wp.float64(inertia[base + 4])) + wp.float64(inertia[base + 8])
    ) * wp.float64(0.5)
    j = wp.mat33d(0.0)
    scale = wp.float64(0.0)
    for r in range(3):
        for c in range(3):
            diagonal = wp.float64(0.0)
            if r == c:
                diagonal = half
            j[r, c] = diagonal - wp.float64(inertia[base + 3 * r + c])
            scale = wp.max(scale, wp.abs(j[r, c]))
    valid = scale > wp.float64(0.0) and wp.isfinite(scale)
    initial = wp.mat33d(0.0)
    current = wp.mat33d(0.0)
    if valid:
        # Column c of the initial matrix is sum_k (j[c][k] / scale) a_k.
        initial = a * wp.transpose(j / scale)
        current = initial
        for _iteration in range(8):
            determinant = wp.determinant(current)
            if not (determinant > wp.float64(1.0e-30)) or not wp.isfinite(determinant):
                valid = False
                break
            c0 = wp.vec3d(current[0, 0], current[1, 0], current[2, 0])
            c1 = wp.vec3d(current[0, 1], current[1, 1], current[2, 1])
            c2 = wp.vec3d(current[0, 2], current[1, 2], current[2, 2])
            inverse = wp.matrix_from_cols(wp.cross(c1, c2), wp.cross(c2, c0), wp.cross(c0, c1)) / determinant
            gamma = wp.sqrt(wp.sqrt(norm_squared_d(inverse) / norm_squared_d(current)))
            next_matrix = (gamma * current + inverse / gamma) * wp.float64(0.5)
            change = norm_squared_d(next_matrix - current)
            current = next_matrix
            if change <= wp.float64(1.0e-24):
                break
        if valid:
            product = initial * wp.transpose(current)
            symmetric = (product + wp.transpose(product)) * wp.float64(0.5)
            trace = (symmetric[0, 0] + symmetric[1, 1]) + symmetric[2, 2]
            hessian = trace * wp.identity(n=3, dtype=wp.float64) - symmetric
            valid = wp.determinant(hessian) > wp.float64(1.0e-30)
            if valid and maps != 0:
                hessian_inverse = wp.inverse(hessian)
                for k in range(3):
                    v = current * wp.vec3d(j[k, 0] / scale, j[k, 1] / scale, j[k, 2] / scale)
                    spin = wp.mat33(hessian_inverse * wp.skew(v))
                    if k == 0:
                        result.map0 = spin
                    elif k == 1:
                        result.map1 = spin
                    else:
                        result.map2 = spin
    if valid:
        result.weighted = 1
        result.rotation = wp.mat33(current)
        return result
    result.rotation = nearest_rotation(a0, a1, a2)
    if maps != 0:
        # The float fallback uses sym(A R^T) without inertia weights, and a diagonal reciprocal if it is singular.
        rot = result.rotation
        p = wp.outer(a0, column(rot, 0)) + wp.outer(a1, column(rot, 1)) + wp.outer(a2, column(rot, 2))
        trace32 = (p[0, 0] + p[1, 1]) + p[2, 2]
        h = trace32 * wp.identity(n=3, dtype=float) - (p + wp.transpose(p)) * 0.5
        det = wp.determinant(h)
        hinv = wp.mat33(0.0)
        if det > 1.0e-30 and wp.isfinite(det):
            hinv = wp.inverse(h)
        else:
            if h[0, 0] > 0.0:
                hinv[0, 0] = 1.0 / h[0, 0]
            if h[1, 1] > 0.0:
                hinv[1, 1] = 1.0 / h[1, 1]
            if h[2, 2] > 0.0:
                hinv[2, 2] = 1.0 / h[2, 2]
        result.map0 = hinv * wp.skew(column(rot, 0))
        result.map1 = hinv * wp.skew(column(rot, 1))
        result.map2 = hinv * wp.skew(column(rot, 2))
    return result


# ---------------------------------------------------------------------------------------------------------------------
# Frames and strain


@wp.struct
class CableFrame:
    point: wp.vec3
    basis: wp.mat33


@wp.struct
class CableEndpoint:
    frame: CableFrame
    spin: CableSpin
    offset: wp.vec3
    ids: wp.vec4i
    valid: int


@wp.func
def cable_endpoint(view: CableView, local: int, world: int, x: wp.array[wp.vec3], parent: int, maps: int):
    """World frame of a cable end (body-attached or fixed) with the body's rotation spin data."""
    e = CableEndpoint()
    transform = view.frame1[local]
    body = view.body_pairs[local][1]
    if parent != 0:
        transform = view.frame0[local]
        body = view.body_pairs[local][0]
    f = CableFrame()
    f.point = wp.transform_get_translation(transform)
    f.basis = authored_basis(transform)
    e.valid = 0
    if body >= 0:
        e.valid = 1
        b = view.body_indices[body]
        o = world * view.particles_per_world
        e.ids = wp.vec4i(b[0] + o, b[1] + o, b[2] + o, b[3] + o)
        com = view.body_com[body]
        e.offset = f.point - com
        f.point = contact_point(f.point, e.ids, com, x)
        e.spin = cable_rotation_spin(x[e.ids[1]], x[e.ids[2]], x[e.ids[3]], view.inertia, 9 * body, maps)
        f.basis = e.spin.rotation * f.basis
    e.frame = f
    return e


@wp.func
def opposite_axis(n: wp.vec3, parent_x: wp.vec3):
    """Reproducible axis in the parent's normal plane at the antiparallel singularity."""
    projected = parent_x - contact_dot(n, parent_x) * n
    length = wp.sqrt(contact_dot(projected, projected))
    if length <= 1.0e-9:
        axis = wp.vec3(1.0, 0.0, 0.0)
        if wp.abs(n[0]) > 0.9:
            axis = wp.vec3(0.0, 1.0, 0.0)
        projected = axis - contact_dot(n, axis) * n
        length = wp.sqrt(contact_dot(projected, projected))
    return normalized_or_zero(projected, length, 0.0)


@wp.struct
class CableStrain:
    linear: wp.vec3
    angular: wp.vec3
    bend: wp.vec3
    twist: float


@wp.func
def cable_strain(parent: CableFrame, child: CableFrame, rest: wp.vec3):
    """Linear strain in the parent frame; discrete bend (capped at 20) minus rest; twist of the parallel-transported
    parent x axis about the child z axis, wrapped to [-pi, pi]."""
    s = CableStrain()
    difference = child.point - parent.point
    px = column(parent.basis, 0)
    py = column(parent.basis, 1)
    n = column(parent.basis, 2)
    m = column(child.basis, 2)
    cx = column(child.basis, 0)
    s.linear = wp.vec3(contact_dot(px, difference), contact_dot(py, difference), contact_dot(n, difference))
    cosine = wp.max(-1.0, wp.min(1.0, contact_dot(n, m)))
    denominator = cosine + 1.0
    cross = wp.cross(n, m)
    bending = wp.vec3(0.0)
    if not (denominator <= 1.0e-12) or not (contact_dot(cross, cross) <= 1.0e-12):
        bending = cross * (2.0 / wp.max(1.0e-12, denominator))
        length = wp.sqrt(contact_dot(bending, bending))
        if length > 20.0:
            bending = bending * (20.0 / length)
    else:
        bending = opposite_axis(n, px) * 20.0
    transported = wp.vec3(0.0)
    if denominator > 1.0e-8:
        transported = px - (n + m) * (contact_dot(m, px) / denominator)
    else:
        length = wp.sqrt(contact_dot(cross, cross))
        if length > 1.0e-7:
            axis = cross / length
            angle = wp.atan2(length, cosine)
            cs = wp.cos(angle)
            sn = wp.sin(angle)
            transported = (contact_dot(axis, px) * (1.0 - cs)) * axis + (cs * px + wp.cross(axis, px) * sn)
        else:
            axis = opposite_axis(n, px)
            projection = contact_dot(px, axis)
            transported = (projection + projection) * axis - px
    transported = contact_normalize(transported)
    twist = wp.atan2(contact_dot(wp.cross(transported, cx), m), contact_dot(cx, transported))
    twist_error = twist - rest[2]
    if twist_error > PI:
        twist_error = twist_error - TWO_PI
    elif twist_error < -PI:
        twist_error = twist_error + TWO_PI
    s.angular = wp.vec3(contact_dot(bending, px) - rest[0], contact_dot(bending, py) - rest[1], twist_error)
    s.bend = bending
    s.twist = twist
    return s


# ---------------------------------------------------------------------------------------------------------------------
# Jacobians


@wp.func
def axis_cross(axis: int, v: wp.vec3):
    e = wp.vec3(0.0)
    e[axis] = 1.0
    return wp.cross(e, v)


@wp.func
def twist_derivative(
    parent: CableFrame, child: CableFrame, denominator: float, dn: wp.vec3, dm: wp.vec3, axis: int, parent_end: int
):
    """Derivative of the twist along a rotation of one endpoint about the world axis."""
    n = column(parent.basis, 2)
    m = column(child.basis, 2)
    if denominator <= 1.0e-8:
        if parent_end != 0:
            return -n[axis]
        return m[axis]
    if denominator > 0.02:
        value = ((n + m) / denominator)[axis]
        if parent_end != 0:
            return -value
        return value
    # In the intermediate antiparallel region, differentiate the normalised transport and atan2 directly.
    px = column(parent.basis, 0)
    cx = column(child.basis, 0)
    dpx = wp.vec3(0.0)
    dcx = wp.vec3(0.0)
    if parent_end != 0:
        dpx = axis_cross(axis, px)
    else:
        dcx = axis_cross(axis, cx)
    numerator = contact_dot(m, px)
    quotient = numerator / denominator
    total = n + m
    transported = px - total * quotient
    length = wp.max(wp.sqrt(contact_dot(transported, transported)), 1.0e-12)
    unit = transported / length
    dc = wp.dot(m, dn) + wp.dot(n, dm)
    dnumerator = wp.dot(m, dpx) + wp.dot(dm, px)
    dquotient = (denominator * dnumerator - numerator * dc) / (denominator * denominator)
    dtransported = (dpx - total * dquotient) - (dn + dm) * quotient
    dunit = (dtransported - unit * contact_dot(unit, dtransported)) / length
    cross = wp.cross(unit, cx)
    sine = contact_dot(m, cross)
    cosine = contact_dot(cx, unit)
    dsine = contact_dot(dm, cross) + contact_dot(m, wp.cross(dunit, cx) + wp.cross(unit, dcx))
    dcosine = contact_dot(dcx, unit) + contact_dot(cx, dunit)
    norm2 = cosine * cosine + sine * sine
    return (cosine * dsine - sine * dcosine) / wp.max(1.0e-12, norm2)


@wp.func
def angular_column(parent: CableFrame, child: CableFrame, bend: wp.vec3, axis: int, parent_end: int):
    n = column(parent.basis, 2)
    m = column(child.basis, 2)
    dn = wp.vec3(0.0)
    dm = wp.vec3(0.0)
    if parent_end != 0:
        dn = axis_cross(axis, n)
    else:
        dm = axis_cross(axis, m)
    cross = wp.cross(n, m)
    denominator = wp.max(-1.0, wp.min(1.0, contact_dot(n, m))) + 1.0
    derivative = wp.vec3(0.0)
    if not (denominator <= 1.0e-12) or not (contact_dot(cross, cross) <= 1.0e-12):
        inverse = 1.0 / wp.max(1.0e-12, denominator)
        scale = inverse + inverse
        uncapped = cross * scale
        derivative = (wp.cross(dn, m) + wp.cross(n, dm)) * scale
        if denominator > 1.0e-12:
            derivative = derivative - cross * ((contact_dot(m, dn) + contact_dot(dm, n)) * (inverse * scale))
        length = wp.sqrt(contact_dot(uncapped, uncapped))
        if length > 20.0:
            derivative = (20.0 / length) * (
                derivative - uncapped * (contact_dot(uncapped, derivative) / (length * length))
            )
    if parent_end != 0:
        derivative = derivative - axis_cross(axis, bend)
    return wp.vec3(
        contact_dot(derivative, column(parent.basis, 0)),
        contact_dot(derivative, column(parent.basis, 1)),
        twist_derivative(parent, child, denominator, dn, dm, axis, parent_end),
    )


@wp.func
def angular_jacobian(parent: CableFrame, child: CableFrame, bend: wp.vec3, parent_end: int):
    return wp.matrix_from_cols(
        angular_column(parent, child, bend, 0, parent_end),
        angular_column(parent, child, bend, 1, parent_end),
        angular_column(parent, child, bend, 2, parent_end),
    )


@wp.func
def linear_rotation_jacobian(parent: CableFrame, child: CableFrame):
    """Rows b_k x (child point - parent point): the parent-frame separation under a parent rotation."""
    delta = child.point - parent.point
    result = wp.mat33(0.0)
    for row in range(3):
        v = wp.cross(column(parent.basis, row), delta)
        result[row, 0] = v[0]
        result[row, 1] = v[1]
        result[row, 2] = v[2]
    return result


@wp.struct
class CableGeometry:
    count: int
    indices: vec8i
    strain: CableStrain


@wp.func
def append_endpoint(
    view: CableView,
    slot_base: int,
    count: int,
    endpoint: CableEndpoint,
    angular: wp.mat33,
    linear_rotation: wp.mat33,
    basis_transpose: wp.mat33,
    parent_end: int,
    write: int,
):
    """Store the four proxy slots of one dynamic endpoint (translation, then the three affine columns)."""
    sign = float(1.0)
    if parent_end != 0:
        sign = -1.0
    for slot in range(4):
        weight = float(1.0)
        if slot > 0:
            weight = endpoint.offset[slot - 1]
        point_jacobian = basis_transpose * (sign * weight)
        lin = point_jacobian
        ang = wp.mat33(0.0)
        if slot > 0:
            spin = spin_map(endpoint.spin, slot - 1)
            ang = angular * spin
            if parent_end != 0:
                lin = linear_rotation * spin + point_jacobian
        if write != 0:
            view.linear[slot_base + count + slot] = lin
            view.angular[slot_base + count + slot] = ang
            view.indices[slot_base + count + slot] = endpoint.ids[slot]
    return count + 4


@wp.func
def cable_geometry(view: CableView, local: int, world: int, x: wp.array[wp.vec3], i: int, write: int):
    """Strain of cable i and, with ``write``, its prepared slot Jacobians."""
    parent = cable_endpoint(view, local, world, x, 1, write)
    child = cable_endpoint(view, local, world, x, 0, write)
    g = CableGeometry()
    g.strain = cable_strain(parent.frame, child.frame, view.rest[local])
    count = int(0)
    if write != 0:
        linear_rotation = linear_rotation_jacobian(parent.frame, child.frame)
        basis_transpose = wp.transpose(parent.frame.basis)
        if parent.valid != 0:
            angular = angular_jacobian(parent.frame, child.frame, g.strain.bend, 1)
            count = append_endpoint(view, i * 8, count, parent, angular, linear_rotation, basis_transpose, 1, 1)
        if child.valid != 0:
            angular = angular_jacobian(parent.frame, child.frame, g.strain.bend, 0)
            count = append_endpoint(view, i * 8, count, child, angular, linear_rotation, basis_transpose, 0, 1)
    g.count = count
    return g


# ---------------------------------------------------------------------------------------------------------------------
# Kernels


@wp.kernel
def prepare_cable_tangents(view: CableView, x: wp.array[wp.vec3]):
    """Prepare every cable's slot Jacobians (the active mask is ignored, as natively)."""
    i = wp.tid()
    g = cable_geometry(view, i % view.per_world, i // view.per_world, x, i, 1)
    view.count[i] = g.count


@wp.kernel
def prepare_cable_penalties(
    base: wp.array[wp.vec4],
    penalty: wp.array[wp.vec4],
    initialized: wp.array[wp.int32],
    per_world: int,
    beta0: float,
    beta1: float,
    scale: float,
    cap0: float,
    cap1: float,
    reset: int,
    rescale: int,
    mask: wp.array[wp.int32],
):
    """Clamp the four penalty channels between the ramp seed and the authored stiffness."""
    i = wp.tid()
    if mask.shape[0] > 0:
        if mask[i] == 0:
            return
    upper = base[i % per_world]
    lower = upper
    if beta0 > 0.0:
        lower[0] = wp.min(cap0, upper[0])
        lower[1] = wp.min(cap0, upper[1])
    if beta1 > 0.0:
        lower[2] = wp.min(cap1, upper[2])
        lower[3] = wp.min(cap1, upper[3])
    value = lower
    if reset == 0 and initialized[i] != 0:
        value = penalty[i]
    factor = float(1.0)
    if rescale != 0:
        factor = scale
    result = wp.vec4(0.0)
    for k in range(4):
        result[k] = wp.min(upper[k], wp.max(lower[k], factor * value[k]))
    penalty[i] = result
    initialized[i] = 1


@wp.kernel
def update_cable_penalties(view: CableView, x: wp.array[wp.vec3], linear_beta: float, angular_beta: float):
    """Increase the penalties by the current stretch/shear/bend/twist residuals, capped at the authored values."""
    i = wp.tid()
    local = i % view.per_world
    if view.active.shape[0] > 0:
        if view.active[local] == 0:
            return
    g = cable_geometry(view, local, i // view.per_world, x, i, 0)
    strain = g.strain
    shear = wp.sqrt(strain.linear[0] * strain.linear[0] + strain.linear[1] * strain.linear[1])
    bend = wp.sqrt(strain.angular[0] * strain.angular[0] + strain.angular[1] * strain.angular[1])
    old = view.penalty[i]
    upper = view.base[local]
    view.penalty[i] = wp.vec4(
        wp.min(upper[0], wp.abs(strain.linear[2]) * linear_beta + old[0]),
        wp.min(upper[1], shear * linear_beta + old[1]),
        wp.min(upper[2], bend * angular_beta + old[2]),
        wp.min(upper[3], wp.abs(strain.angular[2]) * angular_beta + old[3]),
    )


@wp.func
def material_weights(view: CableView, i: int, local: int, inverse_dt: float):
    """Linear and angular stiffness weights: penalty (or authored base) plus damping / h (plus the Dahl Hessian)."""
    penalty = view.base[local]
    if view.penalty.shape[0] > 0:
        penalty = view.penalty[i]
    damping_weight = view.damping[local] * inverse_dt
    linear = wp.vec3(penalty[1] + damping_weight[1], penalty[1] + damping_weight[1], penalty[0] + damping_weight[0])
    bias = wp.vec3(0.0)
    if view.bias.shape[0] > 0:
        bias = view.bias[i]
    angular = wp.vec3(
        (penalty[2] + damping_weight[2]) + bias[0],
        (penalty[2] + damping_weight[2]) + bias[1],
        (penalty[3] + damping_weight[3]) + bias[2],
    )
    return linear, angular


@wp.kernel
def evaluate_cables(
    view: CableView,
    x: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    inverse_dt: float,
    direction: wp.array[wp.vec3],
    force: wp.array[wp.vec3],
    diagonal: wp.array[wp.mat33],
):
    """Force (fresh geometry), tangent product (with a direction) or diagonal blocks (priority) of every cable.
    The product and diagonal modes consume the prepared slot Jacobians."""
    i = wp.tid()
    local = i % view.per_world
    if view.active.shape[0] > 0:
        if view.active[local] == 0:
            return
    linear_weight, angular_weight = material_weights(view, i, local, inverse_dt)
    count = view.count[i]
    base = i * 8
    if diagonal.shape[0] > 0:
        for slot in range(count):
            jl = view.linear[base + slot]
            ja = view.angular[base + slot]
            block = wp.transpose(jl) * wp.diag(linear_weight) * jl + wp.transpose(ja) * wp.diag(angular_weight) * ja
            wp.atomic_add(diagonal, view.indices[base + slot], block)
        return
    linear_force = wp.vec3(0.0)
    angular_force = wp.vec3(0.0)
    if direction.shape[0] > 0:
        linear_delta = wp.vec3(0.0)
        angular_delta = wp.vec3(0.0)
        for slot in range(count):
            v = direction[view.indices[base + slot]]
            linear_delta = linear_delta + view.linear[base + slot] * v
            angular_delta = angular_delta + view.angular[base + slot] * v
        linear_force = wp.cw_mul(linear_weight, linear_delta)
        angular_force = wp.cw_mul(angular_weight, angular_delta)
    else:
        world = i // view.per_world
        # The strain is re-evaluated at x; the slot Jacobians were prepared at the same x this iteration.
        current = cable_geometry(view, local, world, x, i, 0).strain
        penalty = view.base[local]
        if view.penalty.shape[0] > 0:
            penalty = view.penalty[i]
        linear_force = -wp.cw_mul(wp.vec3(penalty[1], penalty[1], penalty[0]), current.linear)
        angular_force = -wp.cw_mul(wp.vec3(penalty[2], penalty[2], penalty[3]), current.angular)
        if view.state.shape[0] > 0:
            angular_force = angular_force - view.state[i]
        damping = view.damping[local]
        if damping[0] > 0.0 or damping[1] > 0.0 or damping[2] > 0.0 or damping[3] > 0.0:
            old = cable_geometry(view, local, world, previous, i, 0).strain
            damping_weight = damping * inverse_dt
            linear_delta = current.linear - old.linear
            angular_delta = current.angular - old.angular
            twist = angular_delta[2]
            if twist > PI:
                twist = twist - TWO_PI
            elif twist < -PI:
                twist = twist + TWO_PI
            angular_delta[2] = twist
            linear_force = linear_force - wp.cw_mul(
                wp.vec3(damping_weight[1], damping_weight[1], damping_weight[0]), linear_delta
            )
            angular_force = angular_force - wp.cw_mul(
                wp.vec3(damping_weight[2], damping_weight[2], damping_weight[3]), angular_delta
            )
    for slot in range(count):
        index = view.indices[base + slot]
        wp.atomic_add(
            force,
            index,
            wp.transpose(view.linear[base + slot]) * linear_force
            + wp.transpose(view.angular[base + slot]) * angular_force,
        )

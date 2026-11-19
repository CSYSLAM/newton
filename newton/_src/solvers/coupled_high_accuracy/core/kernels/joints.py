# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint constraint kernels: body springs, distance constraints, point-lines, drives, armature and angular limits
(native ``kernels/joints.cu``).

Body-attached points are affine combinations of a body's four proxy particles with host weights
(1, p_local - com) stored as ``vec4`` (translation weight first). Ends attached to no body (index < 0) use their fixed
world anchors.
"""

import warp as wp

from .common import atomic_add_block

# The solver never differentiates through its kernels; skipping the adjoints halves compile time.
wp.set_module_options({"enable_backward": False})


@wp.func
def body_point(x: wp.array[wp.vec3], particles: wp.vec4i, weights: wp.vec4):
    """World position of a body-attached point."""
    return (
        weights[0] * x[particles[0]]
        + weights[1] * x[particles[1]]
        + weights[2] * x[particles[2]]
        + weights[3] * x[particles[3]]
    )


@wp.func
def shifted(particles: wp.vec4i, offset: int):
    return wp.vec4i(particles[0] + offset, particles[1] + offset, particles[2] + offset, particles[3] + offset)


@wp.func
def spread(force: wp.array[wp.vec3], particles: wp.vec4i, f: wp.vec3, w: wp.vec4):
    """Add f w_k to the four proxies of a body."""
    wp.atomic_add(force, particles[0], f * w[0])
    wp.atomic_add(force, particles[1], f * w[1])
    wp.atomic_add(force, particles[2], f * w[2])
    wp.atomic_add(force, particles[3], f * w[3])


@wp.kernel
def initialize_joint_body_spring_state(multiplier: wp.array[wp.vec3]):
    """Augmented-Lagrangian multipliers of the joint body springs start every substep at zero."""
    multiplier[wp.tid()] = wp.vec3(0.0)


@wp.kernel
def reset_selected_joint_drive_state(selected: wp.array[wp.int32], state: wp.array[wp.int32]):
    """Clear the drive state of the joints selected by the host."""
    i = wp.tid()
    if selected[i] != 0:
        state[i] = 0


@wp.kernel
def update_joint_body_spring_augmented_lagrangian_state(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    bodies: wp.array[wp.vec2i],
    anchor0: wp.array[wp.vec3],
    anchor1: wp.array[wp.vec3],
    weights0: wp.array[wp.vec4],
    weights1: wp.array[wp.vec4],
    multiplier: wp.array[wp.vec3],
    spring_count: int,
    joints_per_world: int,
    particles_per_world: int,
):
    """Joint springs are the last joints_per_world body springs; lambda = (p1 + lambda) - p0."""
    i = wp.tid()
    world = i // joints_per_world
    spring = i % joints_per_world + (spring_count - joints_per_world)
    offset = world * particles_per_world
    pair = bodies[spring]
    p0 = anchor0[spring]
    p1 = anchor1[spring]
    if pair[0] >= 0:
        p0 = body_point(x, shifted(body_particles[pair[0]], offset), weights0[spring])
    if pair[1] >= 0:
        p1 = body_point(x, shifted(body_particles[pair[1]], offset), weights1[spring])
    multiplier[i] = p1 + multiplier[i] - p0


@wp.kernel
def evaluate_joint_body_spring_augmented_lagrangian_force(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    bodies: wp.array[wp.vec2i],
    anchor0: wp.array[wp.vec3],
    anchor1: wp.array[wp.vec3],
    weights0: wp.array[wp.vec4],
    weights1: wp.array[wp.vec4],
    stiffness: wp.array[float],
    multiplier: wp.array[wp.vec3],
    alpha: float,
    force: wp.array[wp.vec3],
    spring_count: int,
    joints_per_world: int,
    particles_per_world: int,
):
    """f = k alpha ((p1 - p0) + lambda), spread to body 0 proxies as f w and to body 1 proxies as -f w."""
    i = wp.tid()
    world = i // joints_per_world
    spring = i % joints_per_world + (spring_count - joints_per_world)
    offset = world * particles_per_world
    pair = bodies[spring]
    p0 = anchor0[spring]
    p1 = anchor1[spring]
    w0 = weights0[spring]
    w1 = weights1[spring]
    particles0 = wp.vec4i(0)
    particles1 = wp.vec4i(0)
    if pair[0] >= 0:
        particles0 = shifted(body_particles[pair[0]], offset)
        p0 = body_point(x, particles0, w0)
    if pair[1] >= 0:
        particles1 = shifted(body_particles[pair[1]], offset)
        p1 = body_point(x, particles1, w1)
    f = (p1 - p0 + multiplier[i]) * (stiffness[spring] * alpha)
    if pair[0] >= 0:
        spread(force, particles0, f, w0)
    if pair[1] >= 0:
        spread(force, particles1, -f, w1)


@wp.func
def distance_target(d: wp.vec3, rest: float):
    """Target separation along d with length rest; (rest, 0, 0) for a degenerate d, zero when rest <= 0."""
    if rest <= 0.0:
        return wp.vec3(0.0)
    length = wp.sqrt(d[2] * d[2] + (d[1] * d[1] + d[0] * d[0]))
    if length > 1.0e-8:
        return d * (rest / length)
    return wp.vec3(rest, 0.0, 0.0)


@wp.func
def updated_distance_target(e: wp.vec3, previous: wp.vec3, rest: float):
    """Target along e, else along the previous target, else (rest, 0, 0); zero when rest <= 0."""
    if rest <= 0.0:
        return wp.vec3(0.0)
    length = wp.length(e)
    if length > 1.0e-8:
        return e * (rest / length)
    previous_length = wp.length(previous)
    if previous_length > 1.0e-8:
        return previous * (rest / previous_length)
    return wp.vec3(rest, 0.0, 0.0)


@wp.func
def constraint_separation(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    bodies: wp.array[wp.vec2i],
    anchor0: wp.array[wp.vec3],
    anchor1: wp.array[wp.vec3],
    w0: wp.array[wp.vec4],
    w1: wp.array[wp.vec4],
    c: int,
    offset: int,
):
    pair = bodies[c]
    p0 = anchor0[c]
    p1 = anchor1[c]
    if pair[0] >= 0:
        p0 = body_point(x, shifted(body_particles[pair[0]], offset), w0[c])
    if pair[1] >= 0:
        p1 = body_point(x, shifted(body_particles[pair[1]], offset), w1[c])
    return p1 - p0


@wp.kernel
def initialize_body_distance_constraint_state(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    bodies: wp.array[wp.vec2i],
    anchor0: wp.array[wp.vec3],
    anchor1: wp.array[wp.vec3],
    weights0: wp.array[wp.vec4],
    weights1: wp.array[wp.vec4],
    rest: wp.array[float],
    target: wp.array[wp.vec3],
    multiplier: wp.array[wp.vec3],
    constraints_per_world: int,
    particles_per_world: int,
):
    """Start of a step: target along the current separation, multiplier zero."""
    i = wp.tid()
    c = i % constraints_per_world
    offset = (i // constraints_per_world) * particles_per_world
    d = constraint_separation(x, body_particles, bodies, anchor0, anchor1, weights0, weights1, c, offset)
    target[i] = distance_target(d, rest[c])
    multiplier[i] = wp.vec3(0.0)


@wp.kernel
def evaluate_body_distance_constraint_force(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    bodies: wp.array[wp.vec2i],
    anchor0: wp.array[wp.vec3],
    anchor1: wp.array[wp.vec3],
    weights0: wp.array[wp.vec4],
    weights1: wp.array[wp.vec4],
    mass_scale: wp.array[float],
    target: wp.array[wp.vec3],
    multiplier: wp.array[wp.vec3],
    alpha: float,
    force: wp.array[wp.vec3],
    constraints_per_world: int,
    particles_per_world: int,
):
    """f = alpha m_s ((p1 - p0) - r + lambda): body 0 proxies get f w0, body 1 proxies (-f) w1."""
    i = wp.tid()
    c = i % constraints_per_world
    offset = (i // constraints_per_world) * particles_per_world
    pair = bodies[c]
    p0 = anchor0[c]
    p1 = anchor1[c]
    w0 = weights0[c]
    w1 = weights1[c]
    particles0 = wp.vec4i(0)
    particles1 = wp.vec4i(0)
    if pair[0] >= 0:
        particles0 = shifted(body_particles[pair[0]], offset)
        p0 = body_point(x, particles0, w0)
    if pair[1] >= 0:
        particles1 = shifted(body_particles[pair[1]], offset)
        p1 = body_point(x, particles1, w1)
    f = (p1 - p0 - target[i] + multiplier[i]) * (mass_scale[c] * alpha)
    if pair[0] >= 0:
        spread(force, particles0, f, w0)
    if pair[1] >= 0:
        spread(force, particles1, -f, w1)


@wp.kernel
def update_body_distance_constraint_state(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    bodies: wp.array[wp.vec2i],
    anchor0: wp.array[wp.vec3],
    anchor1: wp.array[wp.vec3],
    weights0: wp.array[wp.vec4],
    weights1: wp.array[wp.vec4],
    rest: wp.array[float],
    target: wp.array[wp.vec3],
    multiplier: wp.array[wp.vec3],
    constraints_per_world: int,
    particles_per_world: int,
):
    """e = (p1 - p0) + lambda; the target follows e (or the previous target); lambda = e - r."""
    i = wp.tid()
    c = i % constraints_per_world
    offset = (i // constraints_per_world) * particles_per_world
    d = constraint_separation(x, body_particles, bodies, anchor0, anchor1, weights0, weights1, c, offset)
    e = d + multiplier[i]
    r = updated_distance_target(e, target[i], rest[c])
    multiplier[i] = e - r
    target[i] = r


@wp.kernel
def evaluate_body_spring_force(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    bodies: wp.array[wp.vec2i],
    anchor0: wp.array[wp.vec3],
    anchor1: wp.array[wp.vec3],
    weights0: wp.array[wp.vec4],
    weights1: wp.array[wp.vec4],
    stiffness: wp.array[float],
    rest: wp.array[float],
    alpha: float,
    joint_count: int,
    include_joints: int,
    force: wp.array[wp.vec3],
    springs_per_world: int,
    particles_per_world: int,
):
    """Body springs f = k (d - rest d/|d|), d = p1 - p0. The trailing joint_count joint springs run only in penalty
    mode (include_joints), with stiffness alpha k. Body 0 proxies get w0 f, body 1 proxies w1 (-f)."""
    i = wp.tid()
    spring = i % springs_per_world
    offset = (i // springs_per_world) * particles_per_world
    pair = bodies[spring]
    p0 = anchor0[spring]
    p1 = anchor1[spring]
    w0 = weights0[spring]
    w1 = weights1[spring]
    particles0 = wp.vec4i(0)
    particles1 = wp.vec4i(0)
    if pair[0] >= 0:
        particles0 = shifted(body_particles[pair[0]], offset)
        p0 = body_point(x, particles0, w0)
    if pair[1] >= 0:
        particles1 = shifted(body_particles[pair[1]], offset)
        p1 = body_point(x, particles1, w1)
    d = p1 - p0
    first_joint = springs_per_world - joint_count
    if spring >= first_joint and include_joints == 0:
        return
    scale = float(1.0)
    if spring >= first_joint:
        scale = alpha
    k = scale * stiffness[spring]
    length = wp.sqrt(d[2] * d[2] + (d[1] * d[1] + d[0] * d[0]))
    n = wp.vec3(0.0)
    if length > 0.0:
        n = wp.vec3(d[0] / length, d[1] / length, d[2] / length)
    f = k * (d - rest[spring] * n)
    if pair[0] >= 0:
        spread(force, particles0, f, w0)
    if pair[1] >= 0:
        spread(force, particles1, -f, w1)


@wp.kernel
def evaluate_body_point_line_force(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    bodies: wp.array[wp.vec2i],
    point_weights: wp.array[wp.vec4],
    end0_weights: wp.array[wp.vec4],
    end1_weights: wp.array[wp.vec4],
    end0: wp.array[wp.vec3],
    end1: wp.array[wp.vec3],
    stiffness: wp.array[float],
    alpha: float,
    joint_count: int,
    include_joints: int,
    force: wp.array[wp.vec3],
    lines_per_world: int,
    particles_per_world: int,
):
    """Point-line penalty pulling the point p towards the closest segment point q = a + t (b - a), t in [0, 1].

    f = k (q - p); trailing joint lines only with include_joints (stiffness alpha k); |f|^2 <= 1e-12 is skipped.
    Point body proxies get f w_p, line body proxies wq (-f) with wq = (1 - t) w_a + t w_b.
    """
    i = wp.tid()
    line = i % lines_per_world
    offset = (i // lines_per_world) * particles_per_world
    pair = bodies[line]
    point_particles = shifted(body_particles[pair[0]], offset)
    wpt = point_weights[line]
    p = body_point(x, point_particles, wpt)
    a = end0[line]
    b = end1[line]
    wa = end0_weights[line]
    wb = end1_weights[line]
    line_particles = wp.vec4i(0)
    if pair[1] >= 0:
        line_particles = shifted(body_particles[pair[1]], offset)
        a = body_point(x, line_particles, wa)
        b = body_point(x, line_particles, wb)
    d = b - a
    dd = wp.dot(d, d)
    t = float(0.0)
    if dd > 1.0e-12:
        ratio = wp.dot(p - a, d) / dd
        # A NaN ratio clamps to 0, as the native fminf/fmaxf.
        if not wp.isnan(ratio):
            t = wp.min(wp.max(ratio, 0.0), 1.0)
    wq = wa * (1.0 - t) + t * wb
    diff = d * t + a - p
    first_joint = lines_per_world - joint_count
    if line >= first_joint and include_joints == 0:
        return
    scale = float(1.0)
    if line >= first_joint:
        scale = alpha
    f = diff * (scale * stiffness[line])
    if wp.dot(f, f) <= 1.0e-12:
        return
    g = -f
    for slot in range(4):
        wp.atomic_add(force, point_particles[slot], f * wpt[slot])
        if pair[1] >= 0:
            wp.atomic_add(force, line_particles[slot], wq[slot] * g)


@wp.struct
class PointLineGeometry:
    point: wp.vec3
    middle: wp.vec3
    direction: wp.vec3


@wp.func
def point_line_geometry(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    pair: wp.vec2i,
    point_weights: wp.vec4,
    w0: wp.vec4,
    w1: wp.vec4,
    end0: wp.vec3,
    end1: wp.vec3,
    offset: int,
):
    """The point on body pair.x and the segment midpoint / direction of body pair.y (or of the fixed ends)."""
    g = PointLineGeometry()
    g.point = body_point(x, shifted(body_particles[pair[0]], offset), point_weights)
    if pair[1] >= 0:
        line_particles = shifted(body_particles[pair[1]], offset)
        g.middle = body_point(x, line_particles, (w0 + w1) * 0.5)
        g.direction = body_point(x, line_particles, w1 - w0)
    else:
        g.middle = (end0 + end1) * 0.5
        g.direction = end1 - end0
    return g


@wp.func
def point_line_projection(d: wp.vec3, rel: wp.vec3, x_first: int):
    """Segment parameter s = clamp(rel . d / |d|^2, -1/2, 1/2) and the projector I - d d^T / |d|^2 while s is
    strictly inside (else I); s = 0 and P = I for |d|^2 <= 1e-12."""
    dd = wp.outer(d, d)
    length2 = dd[0, 0] + dd[1, 1] + dd[2, 2]
    identity = wp.identity(n=3, dtype=float)
    if length2 <= 1.0e-12:
        return 0.0, identity
    numerator = float(0.0)
    if x_first != 0:
        numerator = rel[2] * d[2] + (rel[1] * d[1] + rel[0] * d[0])
    else:
        numerator = wp.dot(rel, d)
    t = numerator / length2
    # fminf(0.5, fmaxf(-0.5, t)) maps a NaN t to -0.5.
    s = float(-0.5)
    if not wp.isnan(t):
        s = wp.min(0.5, wp.max(-0.5, t))
    projector = identity
    if t > -0.5 and t < 0.5:
        projector = identity - dd / length2
    return s, projector


@wp.func
def segment_weights(w0: wp.vec4, w1: wp.vec4, s: float):
    """Line-body weights at segment parameter s: (w0 + w1) / 2 + s (w1 - w0)."""
    return (w0 + w1) * 0.5 + s * (w1 - w0)


@wp.kernel
def initialize_joint_body_point_line_state(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    bodies: wp.array[wp.vec2i],
    point_weights: wp.array[wp.vec4],
    end0_weights: wp.array[wp.vec4],
    end1_weights: wp.array[wp.vec4],
    end0: wp.array[wp.vec3],
    end1: wp.array[wp.vec3],
    segment: wp.array[float],
    multiplier: wp.array[wp.vec3],
    projector: wp.array[wp.mat33],
    lines_per_world: int,
    joints_per_world: int,
    particles_per_world: int,
):
    """Start of a step for the joint point-lines (the last joints_per_world lines): s and P from p - mid."""
    i = wp.tid()
    line = i % joints_per_world + (lines_per_world - joints_per_world)
    offset = (i // joints_per_world) * particles_per_world
    g = point_line_geometry(
        x,
        body_particles,
        bodies[line],
        point_weights[line],
        end0_weights[line],
        end1_weights[line],
        end0[line],
        end1[line],
        offset,
    )
    s, p = point_line_projection(g.direction, g.point - g.middle, 1)
    projector[i] = p
    segment[i] = s
    multiplier[i] = wp.vec3(0.0)


@wp.kernel
def evaluate_joint_body_point_line_augmented_lagrangian_force(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    bodies: wp.array[wp.vec2i],
    point_weights: wp.array[wp.vec4],
    end0_weights: wp.array[wp.vec4],
    end1_weights: wp.array[wp.vec4],
    end0: wp.array[wp.vec3],
    end1: wp.array[wp.vec3],
    stiffness: wp.array[float],
    segment: wp.array[float],
    multiplier: wp.array[wp.vec3],
    projector: wp.array[wp.mat33],
    alpha: float,
    force: wp.array[wp.vec3],
    lines_per_world: int,
    joints_per_world: int,
    particles_per_world: int,
):
    """c = (mid + s d - p) + lambda, f = k alpha P c; point proxies get w_p f, line proxies wq (-f)."""
    i = wp.tid()
    line = i % joints_per_world + (lines_per_world - joints_per_world)
    offset = (i // joints_per_world) * particles_per_world
    pair = bodies[line]
    g = point_line_geometry(
        x,
        body_particles,
        pair,
        point_weights[line],
        end0_weights[line],
        end1_weights[line],
        end0[line],
        end1[line],
        offset,
    )
    s = segment[i]
    c = g.direction * s + g.middle - g.point + multiplier[i]
    f = projector[i] * c * (stiffness[line] * alpha)
    point_particles = shifted(body_particles[pair[0]], offset)
    wpt = point_weights[line]
    for slot in range(4):
        wp.atomic_add(force, point_particles[slot], wpt[slot] * f)
    if pair[1] < 0:
        return
    wq = segment_weights(end0_weights[line], end1_weights[line], s)
    line_particles = shifted(body_particles[pair[1]], offset)
    negative = -f
    for slot in range(4):
        wp.atomic_add(force, line_particles[slot], wq[slot] * negative)


@wp.kernel
def update_joint_body_point_line_augmented_lagrangian_state(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    bodies: wp.array[wp.vec2i],
    point_weights: wp.array[wp.vec4],
    end0_weights: wp.array[wp.vec4],
    end1_weights: wp.array[wp.vec4],
    end0: wp.array[wp.vec3],
    end1: wp.array[wp.vec3],
    segment: wp.array[float],
    multiplier: wp.array[wp.vec3],
    projector: wp.array[wp.mat33],
    update_multiplier: int,
    lines_per_world: int,
    joints_per_world: int,
    particles_per_world: int,
):
    """s and P from p - lambda; with the multiplier step lambda = P ((lambda + q) - p) for q = mid + s d, then s
    and P again from p - lambda."""
    i = wp.tid()
    line = i % joints_per_world + (lines_per_world - joints_per_world)
    offset = (i // joints_per_world) * particles_per_world
    g = point_line_geometry(
        x,
        body_particles,
        bodies[line],
        point_weights[line],
        end0_weights[line],
        end1_weights[line],
        end0[line],
        end1[line],
        offset,
    )
    lam = multiplier[i]
    s, p = point_line_projection(g.direction, g.point - lam - g.middle, 0)
    projector[i] = p
    q = g.direction * s + g.middle
    segment[i] = s
    if update_multiplier == 0:
        return
    lam = p * (lam + q - g.point)
    multiplier[i] = lam
    s2, p2 = point_line_projection(g.direction, g.point - lam - g.middle, 0)
    projector[i] = p2
    segment[i] = s2


@wp.kernel
def add_joint_body_point_line_augmented_lagrangian_preconditioner(
    body_particles: wp.array[wp.vec4i],
    bodies: wp.array[wp.vec2i],
    point_weights: wp.array[wp.vec4],
    end0_weights: wp.array[wp.vec4],
    end1_weights: wp.array[wp.vec4],
    stiffness: wp.array[float],
    segment: wp.array[float],
    projector: wp.array[wp.mat33],
    alpha: float,
    preconditioner: wp.array[wp.mat33],
    lines_per_world: int,
    joints_per_world: int,
    particles_per_world: int,
):
    """PD_NEWTON joint point-line preconditioner: w (k w) P on the point proxies, then on the line proxies."""
    i = wp.tid()
    line = i % joints_per_world + (lines_per_world - joints_per_world)
    offset = (i // joints_per_world) * particles_per_world
    pair = bodies[line]
    k = stiffness[line] * alpha
    p = projector[i]
    point_particles = shifted(body_particles[pair[0]], offset)
    wpt = point_weights[line]
    for slot in range(4):
        wp.atomic_add(preconditioner, point_particles[slot], wpt[slot] * (k * wpt[slot]) * p)
    if pair[1] < 0:
        return
    wq = segment_weights(end0_weights[line], end1_weights[line], segment[i])
    line_particles = shifted(body_particles[pair[1]], offset)
    for slot in range(4):
        wp.atomic_add(preconditioner, line_particles[slot], wq[slot] * (k * wq[slot]) * p)


@wp.kernel
def multiply_joint_body_point_line_augmented_lagrangian(
    v: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    bodies: wp.array[wp.vec2i],
    point_weights: wp.array[wp.vec4],
    end0_weights: wp.array[wp.vec4],
    end1_weights: wp.array[wp.vec4],
    stiffness: wp.array[float],
    segment: wp.array[float],
    projector: wp.array[wp.mat33],
    alpha: float,
    product: wp.array[wp.vec3],
    lines_per_world: int,
    joints_per_world: int,
    particles_per_world: int,
):
    """PD_NEWTON joint point-line tangent f = k P (J_p v - J_q v); point proxies get f w_p, line proxies wq (-f)."""
    i = wp.tid()
    line = i % joints_per_world + (lines_per_world - joints_per_world)
    offset = (i // joints_per_world) * particles_per_world
    pair = bodies[line]
    point_particles = shifted(body_particles[pair[0]], offset)
    wpt = point_weights[line]
    wq = segment_weights(end0_weights[line], end1_weights[line], segment[i])
    line_particles = wp.vec4i(0)
    line_motion = wp.vec3(0.0)
    if pair[1] >= 0:
        line_particles = shifted(body_particles[pair[1]], offset)
        line_motion = body_point(v, line_particles, wq)
    f = projector[i] * (body_point(v, point_particles, wpt) - line_motion) * (stiffness[line] * alpha)
    for slot in range(4):
        wp.atomic_add(product, point_particles[slot], f * wpt[slot])
    if pair[1] < 0:
        return
    negative = -f
    for slot in range(4):
        wp.atomic_add(product, line_particles[slot], wq[slot] * negative)


vec8i = wp.types.vector(length=8, dtype=wp.int32)
vec8f = wp.types.vector(length=8, dtype=wp.float32)


@wp.func
def add_pair_blocks(
    diagonal: wp.array[wp.mat33],
    row_count: wp.array[wp.int32],
    entry_columns: wp.array[wp.int32],
    entry_values: wp.array[wp.mat33],
    max_count: int,
    rows: int,
    a: int,
    b: int,
    block: wp.mat33,
    transposed: wp.mat33,
):
    """block at (a, b) and transposed at (b, a), into the diagonal when both entries are the same particle."""
    if a == b:
        wp.atomic_add(diagonal, a, block)
        wp.atomic_add(diagonal, b, transposed)
    else:
        atomic_add_block(row_count, entry_columns, entry_values, max_count, rows, a, b, block)
        atomic_add_block(row_count, entry_columns, entry_values, max_count, rows, b, a, transposed)


@wp.kernel
def add_body_spring_gauss_newton_hessian(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    bodies: wp.array[wp.vec2i],
    anchor0: wp.array[wp.vec3],
    anchor1: wp.array[wp.vec3],
    weights0: wp.array[wp.vec4],
    weights1: wp.array[wp.vec4],
    stiffness: wp.array[float],
    rest_length: wp.array[float],
    diagonal: wp.array[wp.mat33],
    row_count: wp.array[wp.int32],
    entry_columns: wp.array[wp.int32],
    entry_values: wp.array[wp.mat33],
    max_count: int,
    springs_per_world: int,
    particles_per_world: int,
    rows: int,
):
    """Body spring Gauss-Newton correction K = k (M - I) on top of the host's constant k I, spread over the
    Jacobian entries (-w0 on body 0, w1 on body 1)."""
    i = wp.tid()
    spring = i % springs_per_world
    rest = rest_length[spring]
    k = stiffness[spring]
    if rest <= 0.0 or not (k > 0.0):
        return
    offset = (i // springs_per_world) * particles_per_world
    pair = bodies[spring]
    ids = vec8i()
    weights = vec8f()
    n = int(0)
    p0 = anchor0[spring]
    p1 = anchor1[spring]
    if pair[0] >= 0:
        particles = shifted(body_particles[pair[0]], offset)
        w = weights0[spring]
        p0 = body_point(x, particles, w)
        for slot in range(4):
            ids[n] = particles[slot]
            weights[n] = -w[slot]
            n += 1
    if pair[1] >= 0:
        particles = shifted(body_particles[pair[1]], offset)
        w = weights1[spring]
        p1 = body_point(x, particles, w)
        for slot in range(4):
            ids[n] = particles[slot]
            weights[n] = w[slot]
            n += 1
    d = p1 - p0
    length = wp.sqrt(d[2] * d[2] + (d[1] * d[1] + d[0] * d[0]))
    identity = wp.identity(n=3, dtype=float)
    m = identity
    if length > 1.0e-8:
        unit = d / length
        nn = wp.outer(unit, unit)
        m = (identity - nn) * wp.max(0.0, 1.0 - rest / length) + nn
    correction = k * (m - identity)
    for a in range(n):
        wp.atomic_add(diagonal, ids[a], correction * (weights[a] * weights[a]))
        for b in range(a + 1, n):
            block = correction * (weights[a] * weights[b])
            add_pair_blocks(
                diagonal, row_count, entry_columns, entry_values, max_count, rows, ids[a], ids[b], block, block
            )


@wp.kernel
def add_joint_body_point_line_gauss_newton_hessian(
    body_particles: wp.array[wp.vec4i],
    bodies: wp.array[wp.vec2i],
    point_weights: wp.array[wp.vec4],
    end0_weights: wp.array[wp.vec4],
    end1_weights: wp.array[wp.vec4],
    stiffness: wp.array[float],
    segment: wp.array[float],
    projector: wp.array[wp.mat33],
    alpha: float,
    diagonal: wp.array[wp.mat33],
    row_count: wp.array[wp.int32],
    entry_columns: wp.array[wp.int32],
    entry_values: wp.array[wp.mat33],
    max_count: int,
    lines_per_world: int,
    joints_per_world: int,
    particles_per_world: int,
    rows: int,
):
    """Joint point-line Gauss-Newton Hessian over the Jacobian entries -w_p (point body) and wq (line body)."""
    i = wp.tid()
    line = i % joints_per_world + (lines_per_world - joints_per_world)
    k = stiffness[line] * alpha
    if not (k > 0.0):
        return
    offset = (i // joints_per_world) * particles_per_world
    pair = bodies[line]
    wq = segment_weights(end0_weights[line], end1_weights[line], segment[i])
    ids = vec8i()
    weights = vec8f()
    n = int(0)
    point_particles = shifted(body_particles[pair[0]], offset)
    wpt = point_weights[line]
    for slot in range(4):
        ids[n] = point_particles[slot]
        weights[n] = -wpt[slot]
        n += 1
    if pair[1] >= 0:
        line_particles = shifted(body_particles[pair[1]], offset)
        for slot in range(4):
            ids[n] = line_particles[slot]
            weights[n] = wq[slot]
            n += 1
    p = projector[i]
    for a in range(n):
        ka = k * weights[a]
        wp.atomic_add(diagonal, ids[a], p * (weights[a] * ka))
        for b in range(a + 1, n):
            block = p * (ka * weights[b])
            add_pair_blocks(
                diagonal, row_count, entry_columns, entry_values, max_count, rows, ids[a], ids[b], block, block
            )


@wp.kernel
def add_body_point_line_gauss_newton_hessian(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    bodies: wp.array[wp.vec2i],
    point_weights: wp.array[wp.vec4],
    end0_weights: wp.array[wp.vec4],
    end1_weights: wp.array[wp.vec4],
    end0: wp.array[wp.vec3],
    end1: wp.array[wp.vec3],
    stiffness: wp.array[float],
    alpha: float,
    joint_count: int,
    include_joints: int,
    diagonal: wp.array[wp.mat33],
    row_count: wp.array[wp.int32],
    entry_columns: wp.array[wp.int32],
    entry_values: wp.array[wp.mat33],
    max_count: int,
    lines_per_world: int,
    particles_per_world: int,
    rows: int,
):
    """Point-line penalty Gauss-Newton Hessian k (J_a^T J_b - s_a s_b I), the closest-point parameter t
    differentiated while strictly inside the segment; the s s I term removes the host's constant part."""
    i = wp.tid()
    line = i % lines_per_world
    first_joint = lines_per_world - joint_count
    if line >= first_joint and include_joints == 0:
        return
    scale = float(1.0)
    if line >= first_joint:
        scale = alpha
    k = scale * stiffness[line]
    if not (k > 0.0):
        return
    offset = (i // lines_per_world) * particles_per_world
    pair = bodies[line]
    point_particles = shifted(body_particles[pair[0]], offset)
    wpt = point_weights[line]
    p = body_point(x, point_particles, wpt)
    a = end0[line]
    b = end1[line]
    w0 = end0_weights[line]
    w1 = end1_weights[line]
    line_particles = wp.vec4i(0)
    if pair[1] >= 0:
        line_particles = shifted(body_particles[pair[1]], offset)
        a = body_point(x, line_particles, w0)
        b = body_point(x, line_particles, w1)
    r = p - a
    d = b - a
    dd = wp.dot(d, d)
    t_raw = float(0.0)
    if dd > 1.0e-12:
        t_raw = wp.dot(r, d) / dd
    t = float(0.0)
    if not wp.isnan(t_raw):
        t = wp.min(wp.max(t_raw, 0.0), 1.0)
    interior = dd > 1.0e-12 and t_raw > 0.0 and t_raw < 1.0
    g = r - d * (t + t)
    identity = wp.identity(n=3, dtype=float)
    count = 4
    if pair[1] >= 0:
        count = 8
    for ea in range(count):
        ja, sa, ida = point_line_entry(ea, wpt, w0, w1, point_particles, line_particles, d, g, dd, t, interior)
        ja_t = wp.transpose(ja)
        wp.atomic_add(diagonal, ida, k * (ja_t * ja - sa * sa * identity))
        for eb in range(ea + 1, count):
            jb, sb, idb = point_line_entry(eb, wpt, w0, w1, point_particles, line_particles, d, g, dd, t, interior)
            block = k * (ja_t * jb - sa * sb * identity)
            add_pair_blocks(
                diagonal, row_count, entry_columns, entry_values, max_count, rows, ida, idb, block, wp.transpose(block)
            )


@wp.func
def point_line_entry(
    entry: int,
    wpt: wp.vec4,
    w0: wp.vec4,
    w1: wp.vec4,
    point_particles: wp.vec4i,
    line_particles: wp.vec4i,
    d: wp.vec3,
    g: wp.vec3,
    dd: float,
    t: float,
    interior: bool,
):
    """Jacobian, constant scale and particle of Jacobian entry `entry` (point slots 0..3, then line slots 4..7)."""
    identity = wp.identity(n=3, dtype=float)
    if entry < 4:
        w = wpt[entry]
        base = t * 0.0 + 0.0 - w
        jacobian = base * identity
        if interior:
            jacobian = wp.outer(d, (d * w + g * 0.0) / dd) + base * identity
        return jacobian, -w, point_particles[entry]
    slot = entry - 4
    we0 = w0[slot]
    we1 = w1[slot]
    dw = we1 - we0
    base = t * dw + we0
    jacobian = base * identity
    if interior:
        jacobian = wp.outer(d, (d * (0.0 - we0) + g * dw) / dd) + base * identity
    return jacobian, (we0 + we1) * 0.5, line_particles[slot]


# ---------------------------------------------------------------------------------------------------------------------
# Joint drives


@wp.struct
class AngleFrame:
    valid: int
    u: wp.vec3
    length: float
    projection_parent: float
    projection_child: float
    norm_parent: float
    norm_child: float
    parent: wp.vec3
    child: wp.vec3
    sin: float
    cos: float


@wp.func
def angle_frame(
    parent_origin: wp.vec3,
    parent_axis: wp.vec3,
    parent_reference: wp.vec3,
    child_origin: wp.vec3,
    child_reference: wp.vec3,
):
    """Rotation frame of a revolute drive: axis u (parent origin -> axis point), the parent and child reference
    directions projected perpendicular to u and normalised, and the child angle (sin, cos) relative to the parent.
    ``valid`` is 0 for degenerate or non-finite frames."""
    f = AngleFrame()
    f.valid = 0
    d = parent_axis - parent_origin
    f.length = wp.length(d)
    if not (f.length > 1.0e-8) or not wp.isfinite(f.length):
        return f
    f.u = d / f.length
    ep = parent_reference - parent_origin
    f.projection_parent = wp.dot(ep, f.u)
    perpendicular_parent = ep - f.u * f.projection_parent
    ec = child_reference - child_origin
    f.projection_child = wp.dot(f.u, ec)
    perpendicular_child = ec - f.u * f.projection_child
    f.norm_parent = wp.length(perpendicular_parent)
    f.norm_child = wp.length(perpendicular_child)
    if not (f.norm_parent > 1.0e-8) or not (f.norm_child > 1.0e-8):
        return f
    if not wp.isfinite(f.norm_parent) or not wp.isfinite(f.norm_child):
        return f
    f.parent = perpendicular_parent / f.norm_parent
    f.child = perpendicular_child / f.norm_child
    f.sin = wp.dot(f.u, wp.cross(f.parent, f.child))
    f.cos = wp.dot(f.parent, f.child)
    magnitude = f.cos * f.cos + f.sin * f.sin
    if magnitude > 1.0e-12 and wp.isfinite(magnitude):
        f.valid = 1
    return f


@wp.struct
class DrivePoints:
    parent_origin: wp.vec3
    parent_axis: wp.vec3
    parent_reference: wp.vec3
    child_origin: wp.vec3
    child_reference: wp.vec3


@wp.func
def drive_points(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    pair: wp.vec2i,
    po: wp.vec3,
    pa: wp.vec3,
    pr: wp.vec3,
    w_po: wp.vec4,
    w_co: wp.vec4,
    w_pa: wp.vec4,
    w_pr: wp.vec4,
    w_cr: wp.vec4,
    offset: int,
):
    """Parent origin/axis/reference (parent body, or fixed in world space) and child origin/reference points."""
    points = DrivePoints()
    if pair[0] >= 0:
        parent = shifted(body_particles[pair[0]], offset)
        points.parent_origin = body_point(x, parent, w_po)
        points.parent_axis = body_point(x, parent, w_pa)
        points.parent_reference = body_point(x, parent, w_pr)
    else:
        points.parent_origin = po
        points.parent_axis = pa
        points.parent_reference = pr
    child = shifted(body_particles[pair[1]], offset)
    points.child_origin = body_point(x, child, w_co)
    points.child_reference = body_point(x, child, w_cr)
    return points


TWO_PI = wp.constant(6.28318548)


@wp.func
def unwrap_angle(reference: float, theta: float):
    """theta shifted by whole turns to within half a turn of reference (C roundf: halves away from zero)."""
    return wp.round((reference - theta) / TWO_PI) * TWO_PI + theta


@wp.kernel
def update_joint_drive_angle_state(
    x: wp.array[wp.vec3],
    angle: wp.array[float],
    initialized: wp.array[wp.int32],
    body_particles: wp.array[wp.vec4i],
    types: wp.array[wp.int32],
    bodies: wp.array[wp.vec2i],
    parent_origin: wp.array[wp.vec3],
    parent_axis: wp.array[wp.vec3],
    parent_reference: wp.array[wp.vec3],
    w_po: wp.array[wp.vec4],
    w_co: wp.array[wp.vec4],
    w_pa: wp.array[wp.vec4],
    w_pr: wp.array[wp.vec4],
    w_cr: wp.array[wp.vec4],
    drives_per_world: int,
    particles_per_world: int,
):
    """After the velocity update: store revolute (type 1) angles unwrapped against the stored angle."""
    i = wp.tid()
    drive = i % drives_per_world
    if types[drive] != 1:
        return
    offset = (i // drives_per_world) * particles_per_world
    points = drive_points(
        x,
        body_particles,
        bodies[drive],
        parent_origin[drive],
        parent_axis[drive],
        parent_reference[drive],
        w_po[drive],
        w_co[drive],
        w_pa[drive],
        w_pr[drive],
        w_cr[drive],
        offset,
    )
    f = angle_frame(
        points.parent_origin,
        points.parent_axis,
        points.parent_reference,
        points.child_origin,
        points.child_reference,
    )
    if f.valid == 0:
        return
    theta = wp.atan2(f.sin, f.cos)
    reference = theta
    if initialized[i] != 0:
        reference = angle[i]
    angle[i] = unwrap_angle(reference, theta)
    initialized[i] = 1


@wp.func
def drive_load(
    q0: float,
    dq: float,
    v0: float,
    dof: int,
    modes: wp.array[wp.int32],
    target_q: float,
    target_qd: float,
    ke: wp.array[float],
    kd: wp.array[float],
    effort: wp.array[float],
    damping: wp.array[float],
    feed: float,
    inverse_dt: float,
):
    """Drive law: load = clamp(ke (q* - q0) [position modes] + kd (v* - v0) [velocity modes] or -kd v0 [mode 1],
    +-effort); tau = feed + load - c v0 - (kd_eff + c) (dq / h - v0). Modes (mode | 2) == 3 enable the position term,
    (mode | 1) == 3 the target velocity; without modes every drive is mode 3."""
    mode = int(3)
    if modes.shape[0] > 0:
        mode = modes[dof]
    load = float(0.0)
    if (mode | 2) == 3:
        load += (target_q - q0) * ke[dof]
    if mode == 1:
        load -= v0 * kd[dof]
    elif (mode | 1) == 3:
        load += (target_qd - v0) * kd[dof]
    load = wp.min(effort[dof], wp.max(-effort[dof], load))
    c = float(0.0)
    if damping.shape[0] > 0:
        c = damping[dof]
    tau = feed + load - v0 * c
    kd_effective = float(0.0)
    if modes.shape[0] == 0 or (mode >= 1 and mode <= 3):
        kd_effective = kd[dof]
    rate = dq * inverse_dt - v0
    tau -= rate * (kd_effective + c)
    return tau


@wp.kernel
def evaluate_joint_drive_load(
    x: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    qd: wp.array[float],
    angle: wp.array[float],
    initialized: wp.array[wp.int32],
    body_particles: wp.array[wp.vec4i],
    types: wp.array[wp.int32],
    bodies: wp.array[wp.vec2i],
    parent_origin: wp.array[wp.vec3],
    parent_axis: wp.array[wp.vec3],
    parent_reference: wp.array[wp.vec3],
    w_po: wp.array[wp.vec4],
    w_co: wp.array[wp.vec4],
    w_pa: wp.array[wp.vec4],
    w_pr: wp.array[wp.vec4],
    w_cr: wp.array[wp.vec4],
    dofs: wp.array[wp.int32],
    targets: wp.array[wp.int32],
    ke: wp.array[float],
    kd: wp.array[float],
    effort: wp.array[float],
    modes: wp.array[wp.int32],
    damping: wp.array[float],
    feed: wp.array[float],
    target_q: wp.array[float],
    target_qd: wp.array[float],
    force: wp.array[wp.vec3],
    drives_per_world: int,
    particles_per_world: int,
    dofs_per_world: int,
    targets_per_world: int,
    inverse_dt: float,
):
    """Joint drive loads. Revolute (type != 0): q from the angle frame, unwrapped against the stored start-of-step
    angle, applied through the angle Jacobian. Prismatic (type 0): q is the child-origin displacement along the
    parent axis, applied along the axis to the child and, with a parent body, through the axis-rotation lever."""
    i = wp.tid()
    world = i // drives_per_world
    drive = i % drives_per_world
    offset = world * particles_per_world
    pair = bodies[drive]
    child = shifted(body_particles[pair[1]], offset)
    has_parent = pair[0] >= 0
    parent = wp.vec4i(0)
    if has_parent:
        parent = shifted(body_particles[pair[0]], offset)
    wpo = w_po[drive]
    wco = w_co[drive]
    wpa = w_pa[drive]
    wpr = w_pr[drive]
    wcr = w_cr[drive]
    now = drive_points(
        x,
        body_particles,
        pair,
        parent_origin[drive],
        parent_axis[drive],
        parent_reference[drive],
        wpo,
        wco,
        wpa,
        wpr,
        wcr,
        offset,
    )
    axis = now.parent_axis - now.parent_origin
    length = wp.length(axis)
    u = wp.vec3(0.0)
    if length > 0.0:
        u = axis / length
    if wp.length(u) <= 1.0e-8:
        return
    dof = dofs[drive]
    world_dof = dof + world * dofs_per_world
    world_target = targets[drive] + world * targets_per_world
    copied_target = target_q[world_target]
    copied_target_velocity = target_qd[world_dof]
    copied_feed = feed[world_dof]
    v0 = qd[i]
    before = drive_points(
        previous,
        body_particles,
        pair,
        parent_origin[drive],
        parent_axis[drive],
        parent_reference[drive],
        wpo,
        wco,
        wpa,
        wpr,
        wcr,
        offset,
    )
    if types[drive] == 0:
        d = now.child_origin - now.parent_origin
        displacement = wp.dot(d, u)
        axis0 = before.parent_axis - before.parent_origin
        length0 = wp.sqrt(axis0[2] * axis0[2] + (axis0[1] * axis0[1] + axis0[0] * axis0[0]))
        u0 = wp.vec3(0.0)
        if length0 > 0.0:
            u0 = axis0 / length0
        if wp.length(u0) <= 1.0e-8:
            return
        displacement0 = wp.dot(before.child_origin - before.parent_origin, u0)
        tau = drive_load(
            displacement0,
            displacement - displacement0,
            v0,
            dof,
            modes,
            copied_target,
            copied_target_velocity,
            ke,
            kd,
            effort,
            damping,
            copied_feed,
            inverse_dt,
        )
        if not wp.isfinite(tau) or wp.abs(tau) <= 1.0e-12:
            return
        spread(force, child, tau * u, wco)
        if not has_parent or length <= 1.0e-8:
            return
        lever = (d - displacement * u) / length
        spread(force, parent, (lever + u) * -tau, wpo)
        spread(force, parent, lever * tau, wpa)
        return
    f = angle_frame(now.parent_origin, now.parent_axis, now.parent_reference, now.child_origin, now.child_reference)
    if f.valid == 0:
        return
    upc = wp.cross(f.u, f.parent)
    ucc = wp.cross(f.u, f.child)
    sp = f.projection_parent / f.norm_parent
    sc = f.projection_child / f.norm_child
    axis_term = (sp * upc - sc * ucc) / f.length
    theta = wp.atan2(f.sin, f.cos)
    gp = upc / f.norm_parent
    gp_negative = -upc / f.norm_parent
    gc_negative = -ucc / f.norm_child
    gc = ucc / f.norm_child
    f0 = angle_frame(
        before.parent_origin,
        before.parent_axis,
        before.parent_reference,
        before.child_origin,
        before.child_reference,
    )
    if f0.valid == 0:
        return
    theta0 = wp.atan2(f0.sin, f0.cos)
    if initialized[i] != 0:
        theta0 = angle[i]
    else:
        angle[i] = theta0
        initialized[i] = 1
    q = unwrap_angle(theta0, theta)
    tau = drive_load(
        theta0,
        q - theta0,
        v0,
        dof,
        modes,
        copied_target,
        copied_target_velocity,
        ke,
        kd,
        effort,
        damping,
        copied_feed,
        inverse_dt,
    )
    if not wp.isfinite(tau) or wp.abs(tau) <= 1.0e-12:
        return
    for slot in range(4):
        wp.atomic_add(force, child[slot], tau * (gc_negative * wco[slot] + gc * wcr[slot]))
    if not has_parent:
        return
    difference = gp - axis_term
    for slot in range(4):
        wp.atomic_add(
            force, parent[slot], tau * (difference * wpo[slot] + axis_term * wpa[slot] + gp_negative * wpr[slot])
        )


mat43 = wp.types.matrix(shape=(4, 3), dtype=wp.float32)


@wp.struct
class DriveJacobian:
    valid: int
    child: mat43
    parent: mat43


@wp.func
def angle_jacobian(
    points: DrivePoints, has_parent: bool, w_po: wp.vec4, w_co: wp.vec4, w_pa: wp.vec4, w_pr: wp.vec4, w_cr: wp.vec4
):
    """Angle Jacobian rows of a revolute drive (one row per proxy slot), child then parent."""
    jacobian = DriveJacobian()
    jacobian.valid = 0
    f = angle_frame(
        points.parent_origin,
        points.parent_axis,
        points.parent_reference,
        points.child_origin,
        points.child_reference,
    )
    if f.valid == 0:
        return jacobian
    upc = wp.cross(f.u, f.parent)
    ucc = wp.cross(f.u, f.child)
    sp = f.projection_parent / f.norm_parent
    sc = f.projection_child / f.norm_child
    axis_term = (sp * upc - sc * ucc) / f.length
    gp = upc / f.norm_parent
    gp_negative = -upc / f.norm_parent
    gc_negative = -ucc / f.norm_child
    gc = ucc / f.norm_child
    for slot in range(4):
        jacobian.child[slot] = gc_negative * w_co[slot] + gc * w_cr[slot]
    if has_parent:
        difference = gp - axis_term
        for slot in range(4):
            jacobian.parent[slot] = difference * w_po[slot] + axis_term * w_pa[slot] + gp_negative * w_pr[slot]
    jacobian.valid = 1
    return jacobian


@wp.func
def drive_tangent_scale(
    dof: int, modes: wp.array[wp.int32], damping: wp.array[float], kd: wp.array[float], inverse_dt: float
):
    """(kd [when the mode uses it] + passive damping) / h."""
    kd_effective = float(0.0)
    if modes.shape[0] == 0:
        kd_effective = kd[dof]
    elif modes[dof] >= 1 and modes[dof] <= 3:
        kd_effective = kd[dof]
    c = float(0.0)
    if damping.shape[0] > 0:
        c = damping[dof]
    return (kd_effective + c) * inverse_dt


@wp.kernel
def add_joint_drive_preconditioner(
    x: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    types: wp.array[wp.int32],
    bodies: wp.array[wp.vec2i],
    parent_origin: wp.array[wp.vec3],
    parent_axis: wp.array[wp.vec3],
    parent_reference: wp.array[wp.vec3],
    w_po: wp.array[wp.vec4],
    w_co: wp.array[wp.vec4],
    w_pa: wp.array[wp.vec4],
    w_pr: wp.array[wp.vec4],
    w_cr: wp.array[wp.vec4],
    dofs: wp.array[wp.int32],
    modes: wp.array[wp.int32],
    damping: wp.array[float],
    kd: wp.array[float],
    inverse_dt: float,
    preconditioner: wp.array[wp.mat33],
    drives_per_world: int,
    particles_per_world: int,
):
    """Drive damping preconditioner s g g^T on the diagonal block of every Jacobian row g. Prismatic rows: child
    w_co u, parent w_pa e - w_po (e + u) with the axis-rotation lever e."""
    i = wp.tid()
    drive = i % drives_per_world
    offset = (i // drives_per_world) * particles_per_world
    pair = bodies[drive]
    child = shifted(body_particles[pair[1]], offset)
    has_parent = pair[0] >= 0
    parent = wp.vec4i(0)
    if has_parent:
        parent = shifted(body_particles[pair[0]], offset)
    wpo = w_po[drive]
    wco = w_co[drive]
    wpa = w_pa[drive]
    wpr = w_pr[drive]
    wcr = w_cr[drive]
    points = drive_points(
        x,
        body_particles,
        pair,
        parent_origin[drive],
        parent_axis[drive],
        parent_reference[drive],
        wpo,
        wco,
        wpa,
        wpr,
        wcr,
        offset,
    )
    axis = points.parent_axis - points.parent_origin
    length = wp.length(axis)
    u = wp.vec3(0.0)
    if length > 0.0:
        u = axis / length
    if wp.length(u) <= 1.0e-8:
        return
    s = drive_tangent_scale(dofs[drive], modes, damping, kd, inverse_dt)
    if not (s > 0.0) or not wp.isfinite(s):
        return
    if types[drive] != 0:
        jacobian = angle_jacobian(points, has_parent, wpo, wco, wpa, wpr, wcr)
        if jacobian.valid == 0:
            return
        for slot in range(4):
            row = jacobian.child[slot]
            wp.atomic_add(preconditioner, child[slot], s * wp.outer(row, row))
        if has_parent:
            for slot in range(4):
                row = jacobian.parent[slot]
                wp.atomic_add(preconditioner, parent[slot], s * wp.outer(row, row))
        return
    for slot in range(4):
        row = wco[slot] * u
        wp.atomic_add(preconditioner, child[slot], s * wp.outer(row, row))
    if not has_parent or length <= 1.0e-8:
        return
    d = points.child_origin - points.parent_origin
    lever = (d - wp.dot(d, u) * u) / length
    a = lever + u
    for slot in range(4):
        row = lever * wpa[slot] - a * wpo[slot]
        wp.atomic_add(preconditioner, parent[slot], s * wp.outer(row, row))


@wp.kernel
def multiply_joint_drive_tangent(
    x: wp.array[wp.vec3],
    p: wp.array[wp.vec3],
    body_particles: wp.array[wp.vec4i],
    types: wp.array[wp.int32],
    bodies: wp.array[wp.vec2i],
    parent_origin: wp.array[wp.vec3],
    parent_axis: wp.array[wp.vec3],
    parent_reference: wp.array[wp.vec3],
    w_po: wp.array[wp.vec4],
    w_co: wp.array[wp.vec4],
    w_pa: wp.array[wp.vec4],
    w_pr: wp.array[wp.vec4],
    w_cr: wp.array[wp.vec4],
    dofs: wp.array[wp.int32],
    modes: wp.array[wp.int32],
    damping: wp.array[float],
    kd: wp.array[float],
    inverse_dt: float,
    product: wp.array[wp.vec3],
    drives_per_world: int,
    particles_per_world: int,
):
    """Drive damping tangent J^T (s J p). Revolute: J p over child then parent rows (skipped when s J p is not
    finite). Prismatic: J p = (dOc - dOp) . u [+ (dAp - dOp) . e with a parent]."""
    i = wp.tid()
    drive = i % drives_per_world
    offset = (i // drives_per_world) * particles_per_world
    pair = bodies[drive]
    child = shifted(body_particles[pair[1]], offset)
    has_parent = pair[0] >= 0
    parent = wp.vec4i(0)
    if has_parent:
        parent = shifted(body_particles[pair[0]], offset)
    wpo = w_po[drive]
    wco = w_co[drive]
    wpa = w_pa[drive]
    wpr = w_pr[drive]
    wcr = w_cr[drive]
    points = drive_points(
        x,
        body_particles,
        pair,
        parent_origin[drive],
        parent_axis[drive],
        parent_reference[drive],
        wpo,
        wco,
        wpa,
        wpr,
        wcr,
        offset,
    )
    axis = points.parent_axis - points.parent_origin
    length = wp.length(axis)
    u = wp.vec3(0.0)
    if length > 0.0:
        u = axis / length
    if wp.length(u) <= 1.0e-8:
        return
    s = drive_tangent_scale(dofs[drive], modes, damping, kd, inverse_dt)
    if not (s > 0.0) or not wp.isfinite(s):
        return
    if types[drive] == 0:
        d_child = body_point(p, child, wco)
        d_origin = wp.vec3(0.0)
        if has_parent:
            d_origin = body_point(p, parent, wpo)
        relative = d_child - d_origin
        jp = relative[2] * u[2] + (relative[1] * u[1] + relative[0] * u[0])
        if has_parent:
            if length <= 1.0e-8:
                return
            d_axis = body_point(p, parent, wpa)
            d = points.child_origin - points.parent_origin
            lever = (d - wp.dot(d, u) * u) / length
            jp = jp + wp.dot(d_axis - d_origin, lever)
            y = s * jp
            spread(product, child, y * u, wco)
            spread(product, parent, (lever + u) * -y, wpo)
            spread(product, parent, lever * y, wpa)
            return
        y = s * jp
        spread(product, child, y * u, wco)
        return
    jacobian = angle_jacobian(points, has_parent, wpo, wco, wpa, wpr, wcr)
    if jacobian.valid == 0:
        return
    jp = float(0.0)
    for slot in range(4):
        jp = jp + wp.dot(jacobian.child[slot], p[child[slot]])
    if has_parent:
        for slot in range(4):
            jp = jp + wp.dot(jacobian.parent[slot], p[parent[slot]])
    y = s * jp
    if not wp.isfinite(y):
        return
    for slot in range(4):
        wp.atomic_add(product, child[slot], y * jacobian.child[slot])
    if has_parent:
        for slot in range(4):
            wp.atomic_add(product, parent[slot], y * jacobian.parent[slot])


@wp.struct
class JointCoordinateView:
    """Joint geometry shared by the coordinate velocity, armature and angular-limit kernels."""

    body_particles: wp.array[wp.vec4i]
    types: wp.array[wp.int32]
    bodies: wp.array[wp.vec2i]
    parent_origin: wp.array[wp.vec3]
    parent_axis: wp.array[wp.vec3]
    parent_reference: wp.array[wp.vec3]
    w_po: wp.array[wp.vec4]
    w_co: wp.array[wp.vec4]
    w_pa: wp.array[wp.vec4]
    w_pr: wp.array[wp.vec4]
    w_cr: wp.array[wp.vec4]
    dofs: wp.array[wp.int32]
    joints_per_world: int
    particles_per_world: int


@wp.struct
class AngularLimitView:
    lower: wp.array[float]
    upper: wp.array[float]
    stiffness: wp.array[float]
    damping: wp.array[float]


@wp.func
def view_points(view: JointCoordinateView, x: wp.array[wp.vec3], joint: int, offset: int):
    return drive_points(
        x,
        view.body_particles,
        view.bodies[joint],
        view.parent_origin[joint],
        view.parent_axis[joint],
        view.parent_reference[joint],
        view.w_po[joint],
        view.w_co[joint],
        view.w_pa[joint],
        view.w_pr[joint],
        view.w_cr[joint],
        offset,
    )


@wp.func
def coordinate_rows(view: JointCoordinateView, joint: int, points: DrivePoints, has_parent: bool):
    """Jacobian rows of the joint coordinate (prismatic or revolute) and its current value; valid = 0 when
    degenerate. The coordinate is returned in the Jacobian's ``child[0]``-independent extra field."""
    jacobian = DriveJacobian()
    jacobian.valid = 0
    coordinate = float(0.0)
    if view.types[joint] == 0:
        axis = points.parent_axis - points.parent_origin
        length = wp.sqrt(axis[2] * axis[2] + (axis[0] * axis[0] + axis[1] * axis[1]))
        if length <= 1.0e-8:
            return jacobian, coordinate
        u = axis / length
        d = points.child_origin - points.parent_origin
        coordinate = wp.dot(d, u)
        lever = (d - coordinate * u) / length
        for slot in range(4):
            jacobian.child[slot] = u * view.w_co[joint][slot]
            if has_parent:
                jacobian.parent[slot] = lever * view.w_pa[joint][slot] - (u + lever) * view.w_po[joint][slot]
        jacobian.valid = 1
        return jacobian, coordinate
    frame = angle_frame(
        points.parent_origin,
        points.parent_axis,
        points.parent_reference,
        points.child_origin,
        points.child_reference,
    )
    if frame.valid == 0:
        return jacobian, coordinate
    coordinate = wp.atan2(frame.sin, frame.cos)
    upc = wp.cross(frame.u, frame.parent)
    ucc = wp.cross(frame.u, frame.child)
    sp = frame.projection_parent / frame.norm_parent
    sc = frame.projection_child / frame.norm_child
    axis_term = (sp * upc - sc * ucc) / frame.length
    gp = upc / frame.norm_parent
    gp_negative = -upc / frame.norm_parent
    gc = ucc / frame.norm_child
    gc_negative = -ucc / frame.norm_child
    for slot in range(4):
        jacobian.child[slot] = gc_negative * view.w_co[joint][slot] + gc * view.w_cr[joint][slot]
        if has_parent:
            jacobian.parent[slot] = (
                (gp - axis_term) * view.w_po[joint][slot]
                + axis_term * view.w_pa[joint][slot]
                + gp_negative * view.w_pr[joint][slot]
            )
    jacobian.valid = 1
    return jacobian, coordinate


@wp.func
def joint_coordinate(points: DrivePoints, joint_type: int):
    """Joint coordinate only (valid flag, value)."""
    if joint_type == 0:
        axis = points.parent_axis - points.parent_origin
        length = wp.sqrt(axis[2] * axis[2] + (axis[0] * axis[0] + axis[1] * axis[1]))
        if length <= 1.0e-8:
            return 0, 0.0
        return 1, wp.dot(points.child_origin - points.parent_origin, axis / length)
    frame = angle_frame(
        points.parent_origin,
        points.parent_axis,
        points.parent_reference,
        points.child_origin,
        points.child_reference,
    )
    if frame.valid == 0:
        return 0, 0.0
    return 1, wp.atan2(frame.sin, frame.cos)


@wp.kernel
def initialize_joint_coordinate_velocities(
    view: JointCoordinateView,
    x: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    coordinate_velocity: wp.array[float],
):
    """Project particle velocities onto the joint coordinate; invalid geometry writes zero."""
    i = wp.tid()
    joint = i % view.joints_per_world
    offset = (i // view.joints_per_world) * view.particles_per_world
    pair = view.bodies[joint]
    has_parent = pair[0] >= 0
    points = view_points(view, x, joint, offset)
    jacobian = DriveJacobian()
    if view.types[joint] != 0:
        jacobian = angle_jacobian(
            points,
            has_parent,
            view.w_po[joint],
            view.w_co[joint],
            view.w_pa[joint],
            view.w_pr[joint],
            view.w_cr[joint],
        )
        if jacobian.valid == 0:
            coordinate_velocity[i] = 0.0
            return
    else:
        axis = points.parent_axis - points.parent_origin
        length = wp.length(axis)
        if length <= 1.0e-8:
            coordinate_velocity[i] = 0.0
            return
        u = axis / length
        d = points.child_origin - points.parent_origin
        lever = (d - wp.dot(d, u) * u) / length
        a = u + lever
        for slot in range(4):
            jacobian.child[slot] = u * view.w_co[joint][slot]
            if has_parent:
                jacobian.parent[slot] = lever * view.w_pa[joint][slot] - a * view.w_po[joint][slot]
    child = shifted(view.body_particles[pair[1]], offset)
    parent = wp.vec4i(0)
    if has_parent:
        parent = shifted(view.body_particles[pair[0]], offset)
    projection = float(0.0)
    for slot in range(4):
        projection = projection + wp.dot(jacobian.child[slot], velocity[child[slot]])
        if has_parent:
            projection = projection + wp.dot(jacobian.parent[slot], velocity[parent[slot]])
    coordinate_velocity[i] = projection


@wp.func
def scatter_rows(
    force: wp.array[wp.vec3], child: wp.vec4i, parent: wp.vec4i, has_parent: bool, tau: float, jacobian: DriveJacobian
):
    for slot in range(4):
        wp.atomic_add(force, child[slot], tau * jacobian.child[slot])
    if has_parent:
        for slot in range(4):
            wp.atomic_add(force, parent[slot], tau * jacobian.parent[slot])


@wp.func
def scatter_blocks(
    block: wp.array[wp.mat33], child: wp.vec4i, parent: wp.vec4i, has_parent: bool, s: float, jacobian: DriveJacobian
):
    for slot in range(4):
        wp.atomic_add(block, child[slot], s * wp.outer(jacobian.child[slot], jacobian.child[slot]))
        if has_parent:
            wp.atomic_add(block, parent[slot], s * wp.outer(jacobian.parent[slot], jacobian.parent[slot]))


@wp.func
def project_rows(
    direction: wp.array[wp.vec3], child: wp.vec4i, parent: wp.vec4i, has_parent: bool, jacobian: DriveJacobian
):
    projection = float(0.0)
    for slot in range(4):
        projection = projection + wp.dot(jacobian.child[slot], direction[child[slot]])
        if has_parent:
            projection = projection + wp.dot(jacobian.parent[slot], direction[parent[slot]])
    return projection


@wp.kernel
def evaluate_joint_armature(
    view: JointCoordinateView,
    armature: wp.array[float],
    x: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    qd: wp.array[float],
    inverse_dt: float,
    direction: wp.array[wp.vec3],
    force: wp.array[wp.vec3],
    block: wp.array[wp.mat33],
):
    """Reflected joint inertia with s = a / h^2: diagonal s J^T J (block output has priority), tangent s J^T J p
    (with a direction), or force s J^T (qd h - delta q)."""
    i = wp.tid()
    joint = i % view.joints_per_world
    a = armature[view.dofs[joint]]
    if not (a > 0.0):
        return
    offset = (i // view.joints_per_world) * view.particles_per_world
    pair = view.bodies[joint]
    has_parent = pair[0] >= 0
    points = view_points(view, x, joint, offset)
    jacobian, coordinate = coordinate_rows(view, joint, points, has_parent)
    if jacobian.valid == 0:
        return
    child = shifted(view.body_particles[pair[1]], offset)
    parent = wp.vec4i(0)
    if has_parent:
        parent = shifted(view.body_particles[pair[0]], offset)
    s = (a * inverse_dt) * inverse_dt
    if block.shape[0] > 0:
        scatter_blocks(block, child, parent, has_parent, s, jacobian)
        return
    delta = float(0.0)
    if direction.shape[0] > 0:
        delta = project_rows(direction, child, parent, has_parent, jacobian)
    else:
        before = view_points(view, previous, joint, offset)
        valid, previous_coordinate = joint_coordinate(before, view.types[joint])
        if valid == 0:
            return
        delta = qd[i] / inverse_dt - (coordinate - previous_coordinate)
        if view.types[joint] == 1:
            delta = delta - wp.round(delta / TWO_PI) * TWO_PI
    tau = s * delta
    if not wp.isfinite(tau):
        return
    scatter_rows(force, child, parent, has_parent, tau, jacobian)


@wp.kernel
def evaluate_angular_limits(
    view: JointCoordinateView,
    limits: AngularLimitView,
    x: wp.array[wp.vec3],
    previous: wp.array[wp.vec3],
    angle: wp.array[float],
    initialized: wp.array[wp.int32],
    inverse_dt: float,
    direction: wp.array[wp.vec3],
    force: wp.array[wp.vec3],
    block: wp.array[wp.mat33],
):
    """One-sided angular stops (revolute only). The diagonal output has priority over the direction."""
    i = wp.tid()
    joint = i % view.joints_per_world
    if view.types[joint] != 1:
        return
    dof = view.dofs[joint]
    ke = limits.stiffness[dof]
    kd = limits.damping[dof]
    if ke == 0.0 and kd == 0.0:
        return
    offset = (i // view.joints_per_world) * view.particles_per_world
    pair = view.bodies[joint]
    has_parent = pair[0] >= 0
    points = view_points(view, x, joint, offset)
    jacobian, theta = coordinate_rows(view, joint, points, has_parent)
    if jacobian.valid == 0:
        return
    theta0 = float(0.0)
    if initialized[i] != 0:
        theta0 = angle[i]
    else:
        before = view_points(view, previous, joint, offset)
        valid, value = joint_coordinate(before, 1)
        if valid == 0:
            return
        theta0 = value
    q = unwrap_angle(theta0, theta)
    velocity = (q - theta0) * inverse_dt
    tau = float(0.0)
    if q < limits.lower[dof]:
        tau = wp.max(0.0, ke * (limits.lower[dof] - q) - kd * velocity)
    elif q > limits.upper[dof]:
        tau = wp.min(0.0, ke * (limits.upper[dof] - q) - kd * velocity)
    else:
        return
    if tau == 0.0 or not wp.isfinite(tau):
        return
    s = kd * inverse_dt + ke
    child = shifted(view.body_particles[pair[1]], offset)
    parent = wp.vec4i(0)
    if has_parent:
        parent = shifted(view.body_particles[pair[0]], offset)
    if block.shape[0] > 0:
        scatter_blocks(block, child, parent, has_parent, s, jacobian)
        return
    if direction.shape[0] > 0:
        tau = s * project_rows(direction, child, parent, has_parent, jacobian)
    scatter_rows(force, child, parent, has_parent, tau, jacobian)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-soft DAT primitives ported from Newton PR #4180, b8eac31540d0.

The upstream interval trajectory tests and division-plane rules are preserved.
Kinematic bodies are excluded from pose application in the MJVBDV2 adapter.
"""

import warp as wp

from .dat_interval_arithmetic import (
    rigid_point_plane_signed_distance_derivative_interval,
    rigid_point_plane_signed_distance_interval,
)

wp.set_module_options({"enable_backward": False})
_SMALL_ANGLE_EPS = wp.constant(1.0e-7)
DAT_TRAJECTORY_SAMPLES = wp.constant(8)
DAT_BISECTION_ITERATIONS = wp.constant(16)
DAT_SEPARATION_EPS = wp.constant(1.0e-6)
_FLOAT32_EPS = wp.constant(1.1920929e-7)
_FLOAT32_MIN_NORMAL = wp.constant(1.17549435e-38)
DAT_ULP_FACTOR = wp.constant(4.0)


@wp.func
def dat_separation_epsilon(coordinate_scale: float):
    """Return a representable DAT half-band at the given coordinate scale."""
    return wp.max(DAT_SEPARATION_EPS, DAT_ULP_FACTOR * _FLOAT32_EPS * coordinate_scale)


@wp.func
def place_dat_division_plane(
    n: wp.vec3,
    negative_support: wp.vec3,
    gap: float,
    positive_approach: float,
    negative_approach: float,
    separation_eps: float = DAT_SEPARATION_EPS,
):
    """Place a DAT plane between the supports while reserving clearance on both sides.

    ``n`` points from ``negative_support`` toward the positive-side primitive.
    The approach values are the largest motions of the corresponding primitive
    toward the other side. Each side keeps at least 5 % of the gap and at least
    ``separation_eps``.
    """
    lmbd = float(0.5)
    if gap >= 2.0 * separation_eps:
        total_approach = positive_approach + negative_approach
        if total_approach > 0.0:
            lmbd = negative_approach / total_approach

        # Clamp the adaptive placement so each side keeps a fraction of the gap (the
        # rigid trajectory is evaluated at absolute float32 positions and needs real
        # clearance to its boundary) and, for small gaps, at least the (-eps, eps)
        # band that keeps the two primitive supports strictly separated.
        minimum_fraction = wp.max(0.05, separation_eps / gap)
        lmbd = wp.clamp(lmbd, minimum_fraction, 1.0 - minimum_fraction)

    # When the gap is smaller than 2*eps, lambda remains 0.5: the midpoint
    # maximizes the available clearance even though the full band cannot fit.
    plane_distance = lmbd * gap
    plane_point = negative_support + plane_distance * n
    return plane_point, lmbd


@wp.func
def planar_truncation_t(
    v: wp.vec3,
    delta_v: wp.vec3,
    n: wp.vec3,
    d: wp.vec3,
    gamma_r: float,
    minimum_signed_distance: float = 0.0,
):
    """Keep a straight vertex trajectory in the positive plane half-space.

    The allowed side satisfies
    ``dot(n, x - d) >= minimum_signed_distance``. Endpoint signs, rather than
    an absolute displacement tolerance, determine whether the segment crosses
    that boundary. A wrong-side start may move toward the allowed side but may
    not make its signed distance more negative.
    """
    s0 = wp.dot(n, v - d) - minimum_signed_distance
    normal_displacement = wp.dot(n, delta_v)
    s1 = s0 + normal_displacement

    if s0 < 0.0:
        if s1 >= s0:
            return 1.0
        return 0.0

    if s1 >= 0.0:
        return 1.0

    # s0 >= 0 and s1 < 0 imply a unique crossing on the segment.
    t = s0 / (s0 - s1)
    t = wp.clamp(t * gamma_r, 0.0, 1.0)
    return t


@wp.func
def rigid_pose_delta(q_ref: wp.transform, q_cur: wp.transform, com: wp.vec3):
    """Decompose the update from ``q_ref`` to ``q_cur`` into a COM translation and a
    world-frame rotation vector (shortest arc) about the COM.

    Returns (c0, dx, axis, angle): reference world COM, COM translation, and the
    axis-angle of the relative rotation.
    """
    c0 = wp.transform_point(q_ref, com)
    c1 = wp.transform_point(q_cur, com)
    q_rel = wp.transform_get_rotation(q_cur) * wp.quat_inverse(wp.transform_get_rotation(q_ref))
    q_rel = wp.normalize(q_rel)
    if q_rel[3] < 0.0:
        q_rel = wp.quat(-q_rel[0], -q_rel[1], -q_rel[2], -q_rel[3])
    axis, angle = wp.quat_to_axis_angle(q_rel)
    return c0, c1 - c0, axis, angle


@wp.func
def rigid_point_trajectory(
    t: float, c0: wp.vec3, dx: wp.vec3, axis: wp.vec3, angle: float, offset0: wp.vec3
) -> wp.vec3:
    """Position at ``t`` under linear translation and Rodrigues rotation.

    ``axis`` must be a unit world-space axis when ``angle`` is nonzero. A zero
    axis is permitted for the zero-angle identity trajectory.
    """
    ta = t * angle
    parallel = axis * wp.dot(axis, offset0)
    perpendicular = offset0 - parallel
    rotated = parallel + wp.cos(ta) * perpendicular + wp.sin(ta) * wp.cross(axis, offset0)
    return c0 + t * dx + rotated


@wp.func
def _rigid_trajectory_prefix_is_interval_safe(
    t: float,
    n: wp.vec3,
    d: wp.vec3,
    c0: wp.vec3,
    dx: wp.vec3,
    axis: wp.vec3,
    angle: float,
    offset0: wp.vec3,
    s0: float,
    maximum_signed_distance: float,
) -> bool:
    """Certify that the complete prefix trajectory ``[0, t]`` stays safe."""

    trajectory_range = rigid_point_plane_signed_distance_interval(0.0, t, n, d, c0, dx, axis, angle, offset0)
    if trajectory_range.upper < maximum_signed_distance:
        return True

    if s0 <= 0.0:
        # If the trajectory starts on or behind its assigned rigid-side boundary
        # and its signed plane distance is nonincreasing, the entire prefix is safe.
        derivative_range = rigid_point_plane_signed_distance_derivative_interval(0.0, t, n, dx, axis, angle, offset0)
        return derivative_range.upper <= 0.0

    return False


@wp.func
def rigid_trajectory_truncation_t(
    n: wp.vec3,
    d: wp.vec3,
    c0: wp.vec3,
    dx: wp.vec3,
    axis: wp.vec3,
    angle: float,
    offset0: wp.vec3,
    gamma_r: float,
    gamma_min: float = 1e-3,
    use_interval_arithmetic: bool = False,
    trajectory_samples: int = DAT_TRAJECTORY_SAMPLES,
    maximum_signed_distance: float = 0.0,
):
    """Return a backed-off interpolation parameter before a rigid point crosses a plane.

    Stage 1 always samples the trajectory and bisects the first bracketed
    crossing. The optional interval-arithmetic path additionally runs Stage 2:
    certify the complete prefix arc ``[0, t*]`` and shorten it by prefix
    bisection when needed.

    Args:
        n: World-space plane normal away from the rigid side. The allowed rigid
            side satisfies
            ``dot(n, x - d) <= maximum_signed_distance``.
        d: A world-space point on the division plane.
        c0: Body center of mass in world space at the reference pose (``t = 0``).
        dx: Proposed world-space COM displacement from the reference pose to the
            current pose. The trajectory translates the COM as ``c0 + t * dx``.
        axis: Unit world-space axis of the shortest-arc rotation from the
            reference orientation to the current orientation. It may be zero
            only when ``angle`` is zero.
        angle: Total shortest-arc rotation angle in radians. At parameter ``t``,
            the point has rotated by ``t * angle`` about ``axis``.
        offset0: World-space vector from ``c0`` to the body-fixed point at the
            reference pose.
        gamma_r: Multiplicative DAT safety factor applied to the last certified
            pre-crossing parameter.
        gamma_min: Additive parameter-space backoff from that parameter. The
            returned value uses the more conservative of ``gamma_r * t`` and
            ``t - gamma_min``.
        use_interval_arithmetic: Run the experimental interval-arithmetic Stage
            2 after the common sampling and bisection Stage 1.
        trajectory_samples: Number of uniform Stage-1 endpoint samples. The
            production default is ``DAT_TRAJECTORY_SAMPLES``; exposing it here
            allows focused tests to demonstrate sampling-parity failures.
        maximum_signed_distance: Signed boundary assigned to the rigid side.
            Rigid-soft DAT passes ``-epsilon`` to keep the rigid primitive
            outside the negative edge of the empty band.

    Returns:
        A truncation parameter in ``[0, 1]``. ``1`` accepts the complete proposed
        rigid update, while ``0`` blocks it at the reference pose.
    """
    s0 = wp.dot(n, rigid_point_trajectory(0.0, c0, dx, axis, angle, offset0) - d) - maximum_signed_distance
    candidate_t = float(1.0)
    crossed = bool(False)
    if s0 > 0.0:
        # Algorithm 1 assumes a valid starting half-space. A point already
        # inside the forbidden band may move only monotonically away from it.
        s_end = wp.dot(n, rigid_point_trajectory(1.0, c0, dx, axis, angle, offset0) - d) - maximum_signed_distance
        # An endpoint-only test is insufficient for rotation: an arc can first
        # worsen and then recover. Requiring the derivative to be nonpositive.
        derivative_range = rigid_point_plane_signed_distance_derivative_interval(0.0, 1.0, n, dx, axis, angle, offset0)
        if s_end <= s0 and derivative_range.upper <= _FLOAT32_MIN_NORMAL:
            return 1.0
        return 0.0
    else:
        # DAT Paper Algorithm 1, Stage 1: locate the first sampled sign change, then refine
        # that pointwise root bracket by ordinary bisection.
        t_lo = float(0.0)
        t_hi = float(1.0)
        for k in range(trajectory_samples):
            t_k = float(k + 1) / float(trajectory_samples)
            s_k = wp.dot(n, rigid_point_trajectory(t_k, c0, dx, axis, angle, offset0) - d) - maximum_signed_distance
            if s_k > 0.0:
                t_lo = float(k) / float(trajectory_samples)
                t_hi = t_k
                crossed = True
                break

        if crossed:
            for _j in range(DAT_BISECTION_ITERATIONS):
                t_mid = 0.5 * (t_lo + t_hi)
                s_mid = (
                    wp.dot(n, rigid_point_trajectory(t_mid, c0, dx, axis, angle, offset0) - d) - maximum_signed_distance
                )
                if s_mid <= 0.0:
                    t_lo = t_mid
                else:
                    t_hi = t_mid
            candidate_t = t_lo

    if use_interval_arithmetic:
        if not _rigid_trajectory_prefix_is_interval_safe(
            candidate_t, n, d, c0, dx, axis, angle, offset0, s0, maximum_signed_distance
        ):
            # DAT Paper Algorithm 1, Stage 2: prefix safety is monotone. Search for the
            # largest t whose complete trajectory prefix can be certified safe.
            t_lo = float(0.0)
            t_hi = candidate_t
            for _j in range(DAT_BISECTION_ITERATIONS):
                t_mid = 0.5 * (t_lo + t_hi)
                if _rigid_trajectory_prefix_is_interval_safe(
                    t_mid, n, d, c0, dx, axis, angle, offset0, s0, maximum_signed_distance
                ):
                    t_lo = t_mid
                else:
                    t_hi = t_mid
            candidate_t = t_lo
            crossed = True

    if crossed:
        return wp.clamp(wp.min(candidate_t * gamma_r, candidate_t - gamma_min), 0.0, 1.0)
    return 1.0


@wp.func
def _truncate_contact(
    contact_index: int,
    # inputs
    soft_contact_count: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    soft_contact_barycentric: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    pos_prev_collision_detection: wp.array[wp.vec3],
    particle_displacements: wp.array[wp.vec3],
    body_q_ref: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    gamma: float,
    use_interval_arithmetic: bool,
    # outputs
    truncation_ts: wp.array[float],
    body_truncation_ts: wp.array[float],
):
    """Joint DAT truncation for one rigid-soft contact row.

    Each row (particle, edge, or face record from the collision pipeline) defines one
    division plane through its stored rigid surface point, oriented by its stored contact
    normal, both taken at the detection-time reference configuration. Both sides of the
    row are constrained against that same plane within a single thread: every vertex of
    the soft record along its straight accumulated displacement, and the rigid body
    along the curved trajectory of the stored surface point. Truncation scalars are
    atomically min-reduced per particle and per body.

    A soft vertex already on the wrong side of its plane may still move toward the
    allowed side but not deeper; the rigid side follows the same rule along its arc.
    """

    if contact_index >= soft_contact_count[0]:
        return

    indices = soft_contact_indices[contact_index]
    if indices[0] < 0:
        return
    bary = soft_contact_barycentric[contact_index]

    # Stored contact point on the soft feature at the reference (detection) state.
    x_ref = bary[0] * pos_prev_collision_detection[indices[0]]
    for i in range(1, 3):
        vi = indices[i]
        if vi >= 0:
            x_ref += bary[i] * pos_prev_collision_detection[vi]

    shape_index = soft_contact_shape[contact_index]
    body_index = shape_body[shape_index]

    # Contact anchor on the rigid surface at the reference pose (world frame for statics).
    X_wb_ref = wp.transform_identity()
    if body_index >= 0:
        X_wb_ref = body_q_ref[body_index]
    bx0 = wp.transform_point(X_wb_ref, soft_contact_body_pos[contact_index])

    # Use a one-micrometer band around meter-scale scenes. At larger
    # world-coordinate magnitudes, increase it so the band remains several
    # representable float32 steps wide.
    coordinate_scale = float(1.0)
    for i in range(3):
        vi = indices[i]
        if vi >= 0:
            coordinate_scale = wp.max(coordinate_scale, wp.max(wp.abs(pos_prev_collision_detection[vi])))
    coordinate_scale = wp.max(coordinate_scale, wp.max(wp.abs(bx0)))
    separation_eps = dat_separation_epsilon(coordinate_scale)

    # The stored normal points from the rigid surface toward the soft feature. Clamp an
    # existing penetration to a zero plane gap so DAT does not construct a deeper target.
    n = soft_contact_normal[contact_index]
    pair_delta = x_ref - bx0
    gap = wp.max(wp.dot(n, pair_delta), 0.0)

    # Rigid-body update accumulated since the reference pose.
    c0 = wp.vec3(0.0)
    dx_body = wp.vec3(0.0)
    rot_axis = wp.vec3(0.0)
    rot_angle = float(0.0)
    body_is_moving = bool(False)
    if body_index >= 0:
        c0, dx_body, rot_axis, rot_angle = rigid_pose_delta(X_wb_ref, body_q[body_index], body_com[body_index])
        body_is_moving = wp.length_sq(dx_body) > 0.0 or rot_angle != 0.0

    # Adaptive plane placement: each side's approach is its largest normal motion
    # toward the other side. ``n`` points from the rigid surface toward the soft feature.
    delta_soft = float(0.0)
    for i in range(3):
        vi = indices[i]
        if vi >= 0:
            delta_soft = wp.max(delta_soft, -wp.dot(n, particle_displacements[vi]))
    delta_rigid = float(0.0)
    if body_is_moving:
        anchor_end = wp.transform_point(body_q[body_index], soft_contact_body_pos[contact_index])
        delta_rigid = wp.max(wp.dot(n, anchor_end - bx0), 0.0)

    plane_point, _lmbd = place_dat_division_plane(n, bx0, gap, delta_soft, delta_rigid, separation_eps)

    # Soft side: every vertex of the record stays in the positive half-space, which
    # begins ``separation_eps`` beyond the division plane.
    for i in range(3):
        vi = indices[i]
        if vi >= 0:
            x_v = pos_prev_collision_detection[vi]
            t_v = planar_truncation_t(
                x_v,
                particle_displacements[vi],
                n,
                plane_point,
                gamma,
                separation_eps,
            )
            if t_v < 1.0:
                wp.atomic_min(truncation_ts, vi, t_v)

    # Rigid side: the stored surface point follows the body's curved trajectory and
    # must stay ``separation_eps`` on the negative side of the plane.
    if body_is_moving:
        t_b = rigid_trajectory_truncation_t(
            n,
            plane_point,
            c0,
            dx_body,
            rot_axis,
            rot_angle,
            bx0 - c0,
            gamma,
            1.0e-3,
            use_interval_arithmetic,
            DAT_TRAJECTORY_SAMPLES,
            -separation_eps,
        )
        if t_b < 1.0:
            wp.atomic_min(body_truncation_ts, body_index, t_b)


@wp.kernel
def apply_rigid_soft_truncation(
    # inputs
    soft_contact_count: wp.array[wp.int32],
    soft_contact_indices: wp.array[wp.vec3i],
    soft_contact_shape: wp.array[wp.int32],
    soft_contact_body_pos: wp.array[wp.vec3],
    soft_contact_normal: wp.array[wp.vec3],
    soft_contact_barycentric: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    pos_prev_collision_detection: wp.array[wp.vec3],
    particle_displacements: wp.array[wp.vec3],
    body_q_ref: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    gamma: float,
    use_interval_arithmetic: bool,
    # outputs
    truncation_ts: wp.array[float],
    body_truncation_ts: wp.array[float],
):
    """Process all stored rows with a fixed 4096-thread grid and strided loops."""
    for contact_index in range(wp.tid(), wp.min(soft_contact_count[0], soft_contact_indices.shape[0]), 4096):
        _truncate_contact(
            contact_index,
            soft_contact_count,
            soft_contact_indices,
            soft_contact_shape,
            soft_contact_body_pos,
            soft_contact_normal,
            soft_contact_barycentric,
            shape_body,
            pos_prev_collision_detection,
            particle_displacements,
            body_q_ref,
            body_q,
            body_com,
            gamma,
            use_interval_arithmetic,
            truncation_ts,
            body_truncation_ts,
        )


@wp.kernel
def apply_body_truncation_ts(
    # inputs
    body_q_ref: wp.array[wp.transform],
    body_flags: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    body_truncation_ts: wp.array[float],
    rigid_dat_body_bounding_radius: wp.array[float],
    rigid_dat_body_max_displacement: wp.array[float],
    # input/output
    body_q: wp.array[wp.transform],
):
    """Scale each body's accumulated pose update (reference -> candidate) by its truncation
    scalar, interpolating translation and rotation about the COM.

    Also applies the conservative isotropic bound: no point of the body may move farther
    than its rigid-soft budget (0.5 * gamma * soft-contact query gap) since the
    last collision detection, using
    |dx| + |angle| * bounding_radius as an upper bound of the largest point motion.
    """
    b = wp.tid()
    if (body_flags[b] & 2) != 0:
        return

    q_cur = body_q[b]
    q_ref = body_q_ref[b]
    com = body_com[b]
    c0, dx, axis, angle = rigid_pose_delta(q_ref, q_cur, com)

    t = body_truncation_ts[b]

    motion_bound = wp.length(dx) + wp.abs(angle) * rigid_dat_body_bounding_radius[b]
    max_point_displacement = rigid_dat_body_max_displacement[b]
    if motion_bound > max_point_displacement:
        # For any represented collision point,
        # ||x(t) - x(0)|| <= t * (||dx|| + |angle| * bounding_radius)
        #                  = t * motion_bound.
        # Therefore, t <= max_point_displacement / motion_bound guarantees
        # ||x(t) - x(0)|| <= max_point_displacement.
        t = wp.min(t, max_point_displacement / motion_bound)

    if t < 1.0:
        c_new = c0 + t * dx
        q_rot = wp.transform_get_rotation(q_ref)
        ta = t * angle
        if wp.abs(ta) > _SMALL_ANGLE_EPS:
            q_new = wp.normalize(wp.quat_from_axis_angle(axis, ta) * q_rot)
        else:
            half_w = axis * (ta * 0.5)
            q_new = wp.normalize(wp.quat(half_w[0], half_w[1], half_w[2], 1.0) * q_rot)
        body_q[b] = wp.transform(c_new - wp.quat_rotate(q_new, com), q_new)

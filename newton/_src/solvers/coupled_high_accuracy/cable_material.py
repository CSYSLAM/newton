# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Material geometry and history for the native SynReal cable solve.

DER geometry and Dahl constitutive formulas are adapted from Newton 1.5.0.
These kernels initialize rest strains and update material history only. Cable
forces and Hessian products are assembled in solver_newton_kernel_cable.cuh
and solved by SynReal's existing Newton/PD_NEWTON global linear systems.
There is no vertex/block descent, body integration or constraint solve here.
"""

import warp as wp

from newton import JointType, Model, ModelBuilder

# Newton's rod joint DOF layout: linear (shear x, shear y, stretch z), then angular (bend x, bend y, twist z).
# The native cable material packs one value per family as (stretch, shear, bend, twist).
ROD_SHEAR_DOF = 0
ROD_STRETCH_DOF = 2
ROD_BEND_DOF = 3
ROD_TWIST_DOF = 5
ROD_MATERIAL_DOFS = (ROD_STRETCH_DOF, ROD_SHEAR_DOF, ROD_BEND_DOF, ROD_TWIST_DOF)

wp.set_module_options({"enable_backward": False})

_SMALL_ANGLE_EPS = wp.constant(1.0e-7)


_SMALL_LENGTH_EPS = wp.constant(1.0e-9)


_DAHL_KAPPADOT_DEADBAND = wp.constant(1.0e-6)


_CABLE_KB_FOLD_EPS = wp.constant(1.0e-12)


_CABLE_KB_CURVATURE_CAP = wp.constant(20.0)


_CABLE_TRANSPORT_DENOM_EPS = wp.constant(1.0e-8)


@wp.func
def _world_selected(world: int, mask: wp.array[wp.bool]):
    """Query an internal world mask whose final entry selects global entities."""
    mask_index = world
    if world < 0:
        mask_index = mask.shape[0] - 1
    return mask[mask_index]


@wp.func
def _quat_rotate_local_z(q: wp.quat) -> wp.vec3:
    """Rotate local +Z by a unit quaternion; the third rotation-matrix column."""
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    return wp.vec3(2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y))


@wp.func
def _quat_rotate_local_x(q: wp.quat) -> wp.vec3:
    """Rotate local +X by a unit quaternion; the first rotation-matrix column."""
    x = q[0]
    y = q[1]
    z = q[2]
    w = q[3]
    return wp.vec3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w))


@wp.func
def _normalize_with_fallback(v: wp.vec3, fallback: wp.vec3) -> wp.vec3:
    """Normalize v, then fallback, with a final fixed axis if both are tiny."""
    v_len = wp.length(v)
    if v_len > _SMALL_LENGTH_EPS:
        return v / v_len
    fb_len = wp.length(fallback)
    if fb_len > _SMALL_LENGTH_EPS:
        return fallback / fb_len
    return wp.vec3(1.0, 0.0, 0.0)


@wp.func
def _project_perp(v: wp.vec3, axis: wp.vec3) -> wp.vec3:
    """Project v onto the plane orthogonal to an already-unit axis."""
    return v - axis * wp.dot(v, axis)


@wp.func
def _perpendicular_axis_with_fallback(axis: wp.vec3, preferred: wp.vec3) -> wp.vec3:
    """Return a unit vector perpendicular to axis, preferring preferred when possible."""
    axis = _normalize_with_fallback(axis, wp.vec3(0.0, 0.0, 1.0))
    perp = _project_perp(preferred, axis)
    perp_len = wp.length(perp)
    if perp_len > _SMALL_LENGTH_EPS:
        return perp / perp_len

    alt = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(axis[0]) > 0.9:
        alt = wp.vec3(0.0, 1.0, 0.0)
    return _normalize_with_fallback(_project_perp(alt, axis), wp.vec3(0.0, 0.0, 1.0))


@wp.func
def _bishop_transport_quat(t0: wp.vec3, t1: wp.vec3, fallback_axis: wp.vec3) -> wp.quat:
    """Minimal no-twist rotation that transports unit tangent t0 to unit tangent t1."""
    c = wp.clamp(wp.dot(t0, t1), -1.0, 1.0)
    v = wp.cross(t0, t1)
    s = wp.length(v)
    if s > _SMALL_ANGLE_EPS:
        return wp.quat_from_axis_angle(v / s, wp.atan2(s, c))
    if c > 0.0:
        return wp.quat_identity()

    axis = _perpendicular_axis_with_fallback(t0, fallback_axis)
    return wp.quat_from_axis_angle(axis, wp.pi)


@wp.func
def _finite_curvature_binormal(t0: wp.vec3, t1: wp.vec3, fallback_axis: wp.vec3) -> wp.vec3:
    """Capped DER finite curvature binormal with a true-fold fallback."""
    tangent_dot = wp.clamp(wp.dot(t0, t1), -1.0, 1.0)
    tangent_cross = wp.cross(t0, t1)
    denom = 1.0 + tangent_dot
    cross_sq = wp.dot(tangent_cross, tangent_cross)

    # Use exact DER while cross(t0, t1) gives a direction. At an exact fold the
    # cross product vanishes, so choose a stable perpendicular cap direction.
    if denom <= _CABLE_KB_FOLD_EPS and cross_sq <= _CABLE_KB_FOLD_EPS:
        axis = _perpendicular_axis_with_fallback(t0, fallback_axis)
        return _CABLE_KB_CURVATURE_CAP * axis

    kb = (2.0 / wp.max(_CABLE_KB_FOLD_EPS, denom)) * tangent_cross

    kb_len = wp.length(kb)
    if kb_len > _CABLE_KB_CURVATURE_CAP:
        kb = (_CABLE_KB_CURVATURE_CAP / kb_len) * kb
    return kb


@wp.func
def _transport_material_axis(t0: wp.vec3, t1: wp.vec3, m0: wp.vec3, fallback_axis: wp.vec3) -> wp.vec3:
    """Parallel-transport a material normal from tangent t0 to tangent t1."""
    c = wp.clamp(wp.dot(t0, t1), -1.0, 1.0)
    denom = 1.0 + c
    # Closed-form minimal rotation t0 -> t1; assumes m0 perpendicular to t0.
    if denom > _CABLE_TRANSPORT_DENOM_EPS:
        w = t0 + t1
        transported = m0 - (wp.dot(m0, t1) / denom) * w
        return _normalize_with_fallback(transported, fallback_axis)

    # Anti-parallel tangents (~180 deg): use a Bishop transport quaternion.
    B = _bishop_transport_quat(t0, t1, fallback_axis)
    return _normalize_with_fallback(wp.quat_rotate(B, m0), fallback_axis)


@wp.func
def _transported_twist_angle_from_material_axes(
    t0: wp.vec3,
    t1: wp.vec3,
    m0: wp.vec3,
    m1: wp.vec3,
    fallback_axis: wp.vec3,
) -> float:
    """Signed twist from transported parent material normal to child material normal."""
    m0_transport = _transport_material_axis(t0, t1, m0, fallback_axis)
    sin_theta = wp.dot(t1, wp.cross(m0_transport, m1))
    cos_theta = wp.dot(m0_transport, m1)
    return wp.atan2(sin_theta, cos_theta)


@wp.struct
class CableBendTwistMeasure:
    """Live bend/twist geometry for one cable joint.

    Measured once per force/Hessian evaluation and reused by the residual and
    analytic Jacobian columns.
    """

    t0: wp.vec3
    t1: wp.vec3
    m0: wp.vec3
    m1: wp.vec3
    kb_world: wp.vec3
    twist: float


@wp.func
def _measure_cable_bend_twist_z(q_wp: wp.quat, q_wc: wp.quat) -> CableBendTwistMeasure:
    """Measure bend/twist for cable material frames, whose material tangent is local +Z.

    The fixed cable material basis is local ``+X, +Y, +Z``.
    The adapter supplies normalized body rotations, so rotated basis axes are already
    orthonormal.
    """
    t0 = _quat_rotate_local_z(q_wp)
    t1 = _quat_rotate_local_z(q_wc)
    m0 = _quat_rotate_local_x(q_wp)
    m1 = _quat_rotate_local_x(q_wc)

    # DER-style split: bend comes from the finite curvature binormal of the two
    # tangents; twist is material spin after no-twist/Bishop transport.
    measure = CableBendTwistMeasure()
    measure.t0 = t0
    measure.t1 = t1
    measure.m0 = m0
    measure.m1 = m1
    measure.twist = _transported_twist_angle_from_material_axes(t0, t1, m0, m1, m0)
    measure.kb_world = _finite_curvature_binormal(t0, t1, m0)
    return measure


@wp.func
def _wrap_principal_angle(angle: float) -> float:
    """Wrap a bounded angular difference into ``[-pi, pi]``.

    Args:
        angle: Difference of two principal angles. The caller guarantees a
            value in ``[-2*pi, 2*pi]``.

    Returns:
        Equivalent principal angular difference.
    """
    if angle > wp.pi:
        angle -= 2.0 * wp.pi
    elif angle < -wp.pi:
        angle += 2.0 * wp.pi
    return angle


@wp.func
def _assemble_geometric_cable_kappa_z(
    q_wp: wp.quat,
    kb_now_world: wp.vec3,
    twist_now: float,
    kb_rest_local: wp.vec3,
    twist_rest: float,
) -> wp.vec3:
    """Assemble [bend_x, bend_y, twist_z] for local +Z cables."""
    bend_now_local = wp.quat_rotate(wp.quat_inverse(q_wp), kb_now_world)
    bend_residual_local = bend_now_local - kb_rest_local
    # In the local +Z convention, parent-frame x/y are bend and z is twist.
    # Wrap the twist delta of two atan2 angles into [-pi, pi] to avoid a ~2*pi
    # jump when rest/current straddle the branch cut (one revolution per joint).
    twist_residual = _wrap_principal_angle(twist_now - twist_rest)
    return wp.vec3(bend_residual_local[0], bend_residual_local[1], twist_residual)


@wp.func
def _cable_bend_twist_delta(kappa: wp.vec3, kappa_prev: wp.vec3) -> wp.vec3:
    """Return a temporal cable bend/twist strain increment.

    Args:
        kappa: Current ``[bend_x, bend_y, twist_z]`` strain.
        kappa_prev: Previous strain in the same representation.

    Returns:
        Bend increments from ordinary subtraction and the shortest signed
        principal-angle increment for twist.
    """
    return wp.vec3(
        kappa[0] - kappa_prev[0],
        kappa[1] - kappa_prev[1],
        _wrap_principal_angle(kappa[2] - kappa_prev[2]),
    )


@wp.func
def compute_geometric_cable_kappa_cached_z(
    q_wp: wp.quat,
    q_wc: wp.quat,
    kb_rest_local: wp.vec3,
    twist_rest: float,
) -> wp.vec3:
    """Geometric cable strain residual for fixed local +Z cables."""
    measure = _measure_cable_bend_twist_z(q_wp, q_wc)
    return _assemble_geometric_cable_kappa_z(q_wp, measure.kb_world, measure.twist, kb_rest_local, twist_rest)


@wp.kernel
def init_cable_rest_bend_twist(
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    joint_cable_rest_kb_local: wp.array[wp.vec3],
    joint_cable_rest_twist: wp.array[float],
):
    """Precompute cable rest angular deformation invariants."""
    j = wp.tid()
    joint_cable_rest_kb_local[j] = wp.vec3(0.0)
    joint_cable_rest_twist[j] = 0.0

    if joint_type[j] != JointType.ROD:
        return

    child = joint_child[j]
    if child < 0:
        return

    parent = joint_parent[j]
    if parent >= 0:
        X_wp_rest = body_q_rest[parent] * joint_X_p[j]
    else:
        X_wp_rest = joint_X_p[j]
    X_wc_rest = body_q_rest[child] * joint_X_c[j]

    q_wp_rest = wp.transform_get_rotation(X_wp_rest)
    q_wc_rest = wp.transform_get_rotation(X_wc_rest)

    # Rest DER bend (parent-local curvature binormal) and rest twist (transported
    # material spin), measured once so a pre-curved rest yields zero strain.
    rest_measure = _measure_cable_bend_twist_z(q_wp_rest, q_wc_rest)
    joint_cable_rest_kb_local[j] = wp.quat_rotate(wp.quat_inverse(q_wp_rest), rest_measure.kb_world)
    joint_cable_rest_twist[j] = rest_measure.twist


@wp.func
def _cable_dahl_active_stiffness(
    c_start: int,
    joint_penalty_k: wp.array[float],
    joint_is_hard: wp.array[wp.int32],
) -> wp.vec3:
    """Current stiffness for soft cable bend/twist modes; hard modes return zero."""
    bend_idx = c_start + ROD_BEND_DOF
    twist_idx = c_start + ROD_TWIST_DOF

    k_bend = float(0.0)
    k_bend_active = joint_penalty_k[bend_idx]
    if joint_is_hard[bend_idx] == 0 and k_bend_active > 0.0:
        k_bend = k_bend_active

    k_twist = float(0.0)
    k_twist_active = joint_penalty_k[twist_idx]
    if joint_is_hard[twist_idx] == 0 and k_twist_active > 0.0:
        k_twist = k_twist_active

    return wp.vec3(k_bend, k_bend, k_twist)


@wp.func
def _dahl_axis_direction(d_kappa: float, d_kappa_prev: float) -> float:
    """Loading direction for one scalar Dahl component."""
    direction = float(1.0)
    if d_kappa > _DAHL_KAPPADOT_DEADBAND:
        direction = 1.0
    elif d_kappa < -_DAHL_KAPPADOT_DEADBAND:
        direction = -1.0
    else:
        direction = 1.0 if d_kappa_prev >= 0.0 else -1.0
    return direction


@wp.func
def _advance_dahl_axis(
    d_kappa: float,
    d_kappa_prev: float,
    sigma_prev: float,
    sigma_max: float,
    tau: float,
):
    """Advance one scalar Dahl component from a supplied strain increment.

    Args:
        d_kappa: Current strain increment.
        d_kappa_prev: Previous increment used inside the direction deadband.
        sigma_prev: Previous Dahl stress.
        sigma_max: Magnitude of the Dahl stress envelope.
        tau: Dahl transition length in strain units.

    Returns:
        Updated Dahl stress and loading direction.
    """
    direction = _dahl_axis_direction(d_kappa, d_kappa_prev)
    exp_term = wp.exp(-direction * d_kappa / tau)
    sigma = direction * sigma_max * (1.0 - exp_term) + sigma_prev * exp_term
    return sigma, direction


@wp.kernel
def compute_cable_dahl_parameters(
    # Inputs
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_world: wp.array[wp.int32],
    pose_rebaseline_mask: wp.array[wp.bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_constraint_start: wp.array[int],
    joint_penalty_k: wp.array[float],
    joint_is_hard: wp.array[wp.int32],
    joint_cable_rest_kb_local: wp.array[wp.vec3],
    joint_cable_rest_twist: wp.array[float],
    body_q: wp.array[wp.transform],
    joint_sigma_prev: wp.array[wp.vec3],
    joint_kappa_prev: wp.array[wp.vec3],
    joint_dkappa_prev: wp.array[wp.vec3],
    joint_eps_max: wp.array[float],
    joint_tau: wp.array[float],
    # Outputs
    joint_sigma_start: wp.array[wp.vec3],
    joint_C_fric: wp.array[wp.vec3],
):
    """
    Compute shared cable Dahl hysteresis parameters (sigma0, C_fric) from
    the current bend/twist strain and the stored previous Dahl state.

    The outputs are:
      - sigma0: linearized friction stress at the start of the step (per component)
      - C_fric: tangent stiffness d(sigma)/d(kappa) (per component)

    Dahl eps_max/tau remain per-joint scalars for compatibility with Newton's
    custom attributes. Bend and twist still get separate envelopes through live
    active stiffness. Hard or inactive subspaces produce zero Dahl stress and
    tangent stiffness.

    On a selected first/reset step, curvature is rebased to the current
    start-of-step pose and the stored stress and curvature increment are
    cleared. This pre-solve rebaseline covers enabled cables; a disabled cable
    refreshes its history in ``update_cable_dahl_state`` (the end-of-step
    finalizer) instead, so a reset while disabled is applied there.
    """
    j = wp.tid()
    zero = wp.vec3(0.0)

    # Default to no friction; the success path overwrites both outputs below.
    joint_sigma_start[j] = zero
    joint_C_fric[j] = zero

    # Only cable joints own Dahl state. Disabled cables are not solved, and
    # the finalizer refreshes their Dahl history every step, so they need no
    # begin-of-step rebaseline.
    if not joint_enabled[j] or joint_type[j] != JointType.ROD:
        return

    parent = joint_parent[j]
    child = joint_child[j]
    # World-parent joints are valid; child body must exist.
    if child < 0:
        return

    rebaseline = _world_selected(joint_world[j], pose_rebaseline_mask)

    eps_max = joint_eps_max[j]
    tau = joint_tau[j]
    c_start = joint_constraint_start[j]
    k_dahl = _cable_dahl_active_stiffness(c_start, joint_penalty_k, joint_is_hard)
    # A gated joint (no Dahl this step) still owes state clearing on a
    # rebaseline step, so stale pre-reset stress can never resurface when a
    # subspace later reactivates.
    dahl_active = tau > 0.0 and eps_max > 0.0 and (k_dahl[0] > 0.0 or k_dahl[1] > 0.0 or k_dahl[2] > 0.0)
    if not dahl_active and not rebaseline:
        return

    # Compute joint frames in world space and the current bend/twist strain.
    if parent >= 0:
        X_wp = body_q[parent] * joint_X_p[j]
    else:
        X_wp = joint_X_p[j]
    X_wc = body_q[child] * joint_X_c[j]
    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)
    kappa_now = compute_geometric_cable_kappa_cached_z(
        q_wp,
        q_wc,
        joint_cable_rest_kb_local[j],
        joint_cable_rest_twist[j],
    )

    # Previous Dahl state (from last converged timestep).
    kappa_prev = joint_kappa_prev[j]
    d_kappa_prev = joint_dkappa_prev[j]
    sigma_prev = joint_sigma_prev[j]
    if rebaseline:
        kappa_prev = kappa_now
        d_kappa_prev = wp.vec3(0.0)
        sigma_prev = wp.vec3(0.0)
        joint_kappa_prev[j] = kappa_prev
        joint_dkappa_prev[j] = d_kappa_prev
        joint_sigma_prev[j] = sigma_prev

    if not dahl_active:
        return

    d_kappa = _cable_bend_twist_delta(kappa_now, kappa_prev)
    sigma_out = zero
    C_fric_out = zero
    for axis in range(3):
        sigma_max = k_dahl[axis] * eps_max
        if sigma_max <= 0.0:
            continue

        sigma0_i, direction = _advance_dahl_axis(
            d_kappa[axis],
            d_kappa_prev[axis],
            sigma_prev[axis],
            sigma_max,
            tau,
        )
        sigma0_i = wp.clamp(sigma0_i, -sigma_max, sigma_max)

        # Tangent stiffness K = (sigma_max - dir*sigma0) / (tau + |d_kappa|).
        numerator = sigma_max - direction * sigma0_i
        denominator = tau + wp.abs(d_kappa[axis])
        sigma_out[axis] = sigma0_i
        C_fric_out[axis] = wp.max(numerator / denominator, 0.0)

    joint_sigma_start[j] = sigma_out
    joint_C_fric[j] = C_fric_out


@wp.kernel
def update_cable_dahl_state(
    # Joint geometry
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_constraint_start: wp.array[int],
    joint_penalty_k: wp.array[float],
    joint_is_hard: wp.array[wp.int32],
    joint_cable_rest_kb_local: wp.array[wp.vec3],
    joint_cable_rest_twist: wp.array[float],
    # Body states (final, after solver convergence)
    body_q: wp.array[wp.transform],
    # Dahl model parameters (PER-JOINT arrays, isotropic)
    joint_eps_max: wp.array[float],
    joint_tau: wp.array[float],
    # Dahl state (inputs - from previous timestep, outputs - to next timestep) - component-wise (vec3)
    joint_sigma_prev: wp.array[wp.vec3],  # input/output
    joint_kappa_prev: wp.array[wp.vec3],  # input/output
    joint_dkappa_prev: wp.array[wp.vec3],  # input/output (stores Delta kappa)
):
    """
    Persist cable Dahl hysteresis state after solver convergence.

    State is diagonal in [bend_x, bend_y, twist_z]. Only soft modes with active
    stiffness are advanced; inactive modes clear stress and use final strain as
    the next baseline.
    """
    j = wp.tid()
    zero = wp.vec3(0.0)

    if joint_type[j] != JointType.ROD:
        return

    parent = joint_parent[j]
    child = joint_child[j]
    if child < 0:
        return

    if parent >= 0:
        X_wp = body_q[parent] * joint_X_p[j]
    else:
        X_wp = joint_X_p[j]
    X_wc = body_q[child] * joint_X_c[j]

    q_wp = wp.transform_get_rotation(X_wp)
    q_wc = wp.transform_get_rotation(X_wc)

    kappa_final = compute_geometric_cable_kappa_cached_z(
        q_wp,
        q_wc,
        joint_cable_rest_kb_local[j],
        joint_cable_rest_twist[j],
    )

    c_start = joint_constraint_start[j]
    k_dahl = _cable_dahl_active_stiffness(c_start, joint_penalty_k, joint_is_hard)

    # Inactive modes clear stress and use the current strain as the next baseline.
    if not joint_enabled[j] or (k_dahl[0] <= 0.0 and k_dahl[1] <= 0.0 and k_dahl[2] <= 0.0):
        joint_kappa_prev[j] = kappa_final
        joint_sigma_prev[j] = zero
        joint_dkappa_prev[j] = zero
        return

    # Stored Dahl state from the previous converged timestep.
    kappa_old = joint_kappa_prev[j]
    d_kappa_old = joint_dkappa_prev[j]
    sigma_old = joint_sigma_prev[j]
    d_kappa = _cable_bend_twist_delta(kappa_final, kappa_old)

    eps_max = joint_eps_max[j]  # Maximum persistent strain [rad]
    tau = joint_tau[j]  # Memory decay length [rad]

    if eps_max <= 0.0 or tau <= 0.0:
        joint_sigma_prev[j] = zero
        joint_kappa_prev[j] = kappa_final
        joint_dkappa_prev[j] = d_kappa
        return

    sigma_final_out = zero
    d_kappa_out = zero

    for axis in range(3):
        sigma_max = k_dahl[axis] * eps_max  # [N*m]
        if sigma_max <= 0.0:
            continue

        sigma_i_next, _direction = _advance_dahl_axis(
            d_kappa[axis],
            d_kappa_old[axis],
            sigma_old[axis],
            sigma_max,
            tau,
        )
        sigma_final_out[axis] = sigma_i_next
        d_kappa_out[axis] = d_kappa[axis]

    # Store final vector state for next timestep: [bend_x, bend_y, twist_z].
    joint_sigma_prev[j] = sigma_final_out
    joint_kappa_prev[j] = kappa_final
    joint_dkappa_prev[j] = d_kappa_out


def register_cable_attributes(builder: ModelBuilder) -> None:
    """Register authored cable material fields without constructing a reference solver."""
    # Keep the imported scene schema so the same model can be used by the VBD
    # reference. The namespace is data compatibility, not solver dispatch.
    for name, dtype, default in (
        ("dahl_eps_max", wp.float32, 0.0),
        ("dahl_tau", wp.float32, 0.0),
        ("joint_is_hard", wp.int32, 1),
    ):
        builder.add_custom_attribute(
            ModelBuilder.CustomAttribute(
                name=name,
                frequency=Model.AttributeFrequency.JOINT,
                assignment=Model.AttributeAssignment.MODEL,
                dtype=dtype,
                default=default,
                namespace="vbd",
            )
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic per-body fusion, evaluated against a frozen color pose snapshot."""

import warp as wp

from newton._src.solvers.mjvbd_v2.vbd import rigid_vbd_kernels as rk


@wp.struct
class SoftInputs:
    dt: float
    color_group: wp.array[wp.int32]
    particle_q: wp.array[wp.vec3]
    particle_q_prev: wp.array[wp.vec3]
    particle_radius: wp.array[float]
    body_q_prev: wp.array[wp.transform]
    body_q: wp.array[wp.transform]
    body_qd: wp.array[wp.spatial_vector]
    body_com: wp.array[wp.vec3]
    body_inv_mass: wp.array[float]
    shape_body: wp.array[int]
    friction_epsilon: float
    body_particle_contact_penalty_k: wp.array[float]
    body_particle_contact_material_ke: wp.array[float]
    body_particle_contact_material_kd: wp.array[float]
    body_particle_contact_material_mu: wp.array[float]
    body_particle_contact_count: wp.array[int]
    soft_contact_indices: wp.array[wp.vec3i]
    body_particle_contact_shape: wp.array[int]
    body_particle_contact_body_pos: wp.array[wp.vec3]
    body_particle_contact_body_vel: wp.array[wp.vec3]
    body_particle_contact_normal: wp.array[wp.vec3]
    soft_contact_barycentric: wp.array[wp.vec3]
    shape_margin: wp.array[float]
    body_particle_contact_buffer_pre_alloc: int
    body_particle_contact_counts: wp.array[wp.int32]
    body_particle_contact_indices: wp.array[wp.int32]
    dense_contact_threshold: int
    body_forces: wp.array[wp.vec3]
    body_torques: wp.array[wp.vec3]
    body_hessian_ll: wp.array[wp.mat33]
    body_hessian_al: wp.array[wp.mat33]
    body_hessian_aa: wp.array[wp.mat33]


@wp.struct
class RigidInputs:
    dt: float
    color_group: wp.array[wp.int32]
    body_q_prev: wp.array[wp.transform]
    body_q: wp.array[wp.transform]
    body_com: wp.array[wp.vec3]
    body_inv_mass: wp.array[float]
    body_colors: wp.array[int]
    friction_epsilon: float
    contact_penalty_k: wp.array[float]
    contact_material_ke: wp.array[float]
    contact_material_kd: wp.array[float]
    contact_material_mu: wp.array[float]
    contact_lambda: wp.array[wp.vec3]
    contact_C0: wp.array[wp.vec3]
    avbd_alpha: float
    hard_contacts: int
    rigid_contact_count: wp.array[int]
    rigid_contact_shape0: wp.array[int]
    rigid_contact_shape1: wp.array[int]
    rigid_contact_point0: wp.array[wp.vec3]
    rigid_contact_point1: wp.array[wp.vec3]
    rigid_contact_offset0: wp.array[wp.vec3]
    rigid_contact_offset1: wp.array[wp.vec3]
    rigid_contact_normal: wp.array[wp.vec3]
    rigid_contact_margin0: wp.array[float]
    rigid_contact_margin1: wp.array[float]
    shape_body: wp.array[wp.int32]
    body_contact_buffer_pre_alloc: int
    body_contact_counts: wp.array[wp.int32]
    body_contact_indices: wp.array[wp.int32]
    body_forces: wp.array[wp.vec3]
    body_torques: wp.array[wp.vec3]
    body_hessian_ll: wp.array[wp.mat33]
    body_hessian_al: wp.array[wp.mat33]
    body_hessian_aa: wp.array[wp.mat33]


@wp.struct
class SolveInputs:
    dt: float
    body_ids_in_color: wp.array[wp.int32]
    body_q: wp.array[wp.transform]
    body_q_prev: wp.array[wp.transform]
    body_q_rest: wp.array[wp.transform]
    body_mass: wp.array[float]
    body_inv_mass: wp.array[float]
    body_inertia: wp.array[wp.mat33]
    body_inertia_q: wp.array[wp.transform]
    body_com: wp.array[wp.vec3]
    adjacency: rk.RigidForceElementAdjacencyInfo
    joint_type: wp.array[int]
    joint_enabled: wp.array[bool]
    joint_parent: wp.array[int]
    joint_child: wp.array[int]
    joint_X_p: wp.array[wp.transform]
    joint_X_c: wp.array[wp.transform]
    joint_axis: wp.array[wp.vec3]
    joint_qd_start: wp.array[int]
    joint_target_q_start: wp.array[int]
    joint_constraint_start: wp.array[int]
    joint_penalty_k: wp.array[float]
    joint_penalty_kd: wp.array[float]
    joint_sigma_start: wp.array[wp.vec3]
    joint_C_fric: wp.array[wp.vec3]
    joint_target_ke: wp.array[float]
    joint_target_kd: wp.array[float]
    joint_target_q: wp.array[float]
    joint_target_qd: wp.array[float]
    joint_limit_lower: wp.array[float]
    joint_limit_upper: wp.array[float]
    joint_limit_ke: wp.array[float]
    joint_limit_kd: wp.array[float]
    joint_lambda_lin: wp.array[wp.vec3]
    joint_lambda_ang: wp.array[wp.vec3]
    joint_C0_lin: wp.array[wp.vec3]
    joint_C0_ang: wp.array[wp.vec3]
    joint_is_hard: wp.array[wp.int32]
    avbd_alpha: float
    joint_dof_dim: wp.array2d[int]
    joint_rest_angle: wp.array[float]
    external_forces: wp.array[wp.vec3]
    external_torques: wp.array[wp.vec3]
    external_hessian_ll: wp.array[wp.mat33]
    external_hessian_al: wp.array[wp.mat33]
    external_hessian_aa: wp.array[wp.mat33]
    body_q_new: wp.array[wp.transform]


@wp.func
def _soft_lane(
    tid: int,
    dt: float,
    color_group: wp.array[wp.int32],
    # Particle state
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    # Rigid body state
    body_q_prev: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    body_inv_mass: wp.array[float],
    shape_body: wp.array[int],
    # AVBD body-particle soft contact penalties and material properties
    friction_epsilon: float,
    body_particle_contact_penalty_k: wp.array[float],
    body_particle_contact_material_ke: wp.array[float],
    body_particle_contact_material_kd: wp.array[float],
    body_particle_contact_material_mu: wp.array[float],
    # Soft contact data (body-particle)
    body_particle_contact_count: wp.array[int],
    soft_contact_indices: wp.array[wp.vec3i],
    body_particle_contact_shape: wp.array[int],
    body_particle_contact_body_pos: wp.array[wp.vec3],
    body_particle_contact_body_vel: wp.array[wp.vec3],
    body_particle_contact_normal: wp.array[wp.vec3],
    # Barycentric weights on each record's soft particles; (1, 0, 0) for a particle contact.
    soft_contact_barycentric: wp.array[wp.vec3],
    shape_margin: wp.array[float],
    # Per-body soft-contact adjacency (body-particle)
    body_particle_contact_buffer_pre_alloc: int,
    body_particle_contact_counts: wp.array[wp.int32],
    body_particle_contact_indices: wp.array[wp.int32],
    dense_contact_threshold: int,
    # Outputs
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
    body_idx_in_group = tid // rk._NUM_RIGID_CONTACT_THREADS_PER_BODY
    thread_id_within_body = tid % rk._NUM_RIGID_CONTACT_THREADS_PER_BODY

    if body_idx_in_group >= color_group.shape[0]:
        return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)
    body_id = color_group[body_idx_in_group]
    if body_inv_mass[body_id] <= 0.0:
        return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)
    num_contacts = body_particle_contact_counts[body_id]
    if num_contacts > body_particle_contact_buffer_pre_alloc:
        num_contacts = body_particle_contact_buffer_pre_alloc
    if dense_contact_threshold > 0 and num_contacts >= dense_contact_threshold:
        return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)
    max_contacts = body_particle_contact_count[0]  # single total soft-contact count

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
        i += rk._NUM_RIGID_CONTACT_THREADS_PER_BODY
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

    return force_acc, torque_acc, h_ll_acc, h_al_acc, h_aa_acc


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
    body_idx_in_group = tid // rk._NUM_RIGID_CONTACT_THREADS_PER_BODY
    thread_id_within_body = tid % rk._NUM_RIGID_CONTACT_THREADS_PER_BODY

    if body_idx_in_group >= color_group.shape[0]:
        return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)
    body_id = color_group[body_idx_in_group]
    if body_inv_mass[body_id] <= 0.0:
        return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)
    num_contacts = body_contact_counts[body_id]
    if num_contacts > body_contact_buffer_pre_alloc:
        num_contacts = body_contact_buffer_pre_alloc

    # Sparse bodies need no zero-valued atomic updates from unused lanes.
    if thread_id_within_body >= num_contacts:
        return wp.vec3(0.0), wp.vec3(0.0), wp.mat33(0.0), wp.mat33(0.0), wp.mat33(0.0)
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
            i += rk._NUM_RIGID_CONTACT_THREADS_PER_BODY
            continue

        s0 = rigid_contact_shape0[contact_idx]
        s1 = rigid_contact_shape1[contact_idx]
        b0 = shape_body[s0] if s0 >= 0 else -1
        b1 = shape_body[s1] if s1 >= 0 else -1

        if b0 != body_id and b1 != body_id:
            i += rk._NUM_RIGID_CONTACT_THREADS_PER_BODY
            continue

        cp0_local = rigid_contact_point0[contact_idx]
        cp1_local = rigid_contact_point1[contact_idx]
        cp0_offset_local = rigid_contact_offset0[contact_idx]
        cp1_offset_local = rigid_contact_offset1[contact_idx]
        contact_normal = rigid_contact_normal[contact_idx]
        # Normal C_n uses the unprojected (skeleton) points: ``thickness`` already accounts
        # for the radial extent, so adding the offset here would double-count it.
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
            # Hard-contact stabilization: normal uses C_n - alpha*C0_n; tangent caches
            # (1 - alpha)*C0_t for the later tangential update.
            C_eff = C_n - avbd_alpha * C0_n
            friction_c0 = (1.0 - avbd_alpha) * (C0_vec - contact_normal * C0_n)

        if C_n <= rk._SMALL_LENGTH_EPS and lam_n <= 0.0:
            i += rk._NUM_RIGID_CONTACT_THREADS_PER_BODY
            continue

        f_n_check = k * C_eff + lam_n
        if f_n_check <= 0.0 and lam_n <= 0.0:
            i += rk._NUM_RIGID_CONTACT_THREADS_PER_BODY
            continue

        contact_kd = contact_material_kd[contact_idx]
        contact_mu = contact_material_mu[contact_idx]

        (
            force_0,
            torque_0,
            h_ll_0,
            h_al_0,
            h_aa_0,
            force_1,
            torque_1,
            h_ll_1,
            h_al_1,
            h_aa_1,
        ) = rk.evaluate_rigid_contact_from_collision(
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

        # For simultaneous soft-contact endpoints, 2*diag(H00,H11) majorizes
        # the linearized pair energy. Keep the hard-contact AL update unchanged.
        factor = 1.0
        if hard_contacts == 0 and b0 >= 0 and b1 >= 0:
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

        i += rk._NUM_RIGID_CONTACT_THREADS_PER_BODY

    return force_acc, torque_acc, h_ll_acc, h_al_acc, h_aa_acc


@wp.func
def _solve_lane(
    tid: int,
    dt: float,
    body_ids_in_color: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_q_rest: wp.array[wp.transform],
    body_mass: wp.array[float],
    body_inv_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_inertia_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    adjacency: rk.RigidForceElementAdjacencyInfo,
    # Joint data
    joint_type: wp.array[int],
    joint_enabled: wp.array[bool],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_qd_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_constraint_start: wp.array[int],
    # AVBD per-constraint penalty state (scalar constraints indexed via joint_constraint_start)
    joint_penalty_k: wp.array[float],
    joint_penalty_kd: wp.array[float],
    # Dahl hysteresis parameters (frozen for this timestep, component-wise vec3 per joint)
    joint_sigma_start: wp.array[wp.vec3],
    joint_C_fric: wp.array[wp.vec3],
    # Drive parameters (DOF-indexed via joint_qd_start)
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_q: wp.array[float],
    joint_target_qd: wp.array[float],
    # Limit parameters (DOF-indexed via joint_qd_start)
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_limit_ke: wp.array[float],
    joint_limit_kd: wp.array[float],
    joint_lambda_lin: wp.array[wp.vec3],
    joint_lambda_ang: wp.array[wp.vec3],
    joint_C0_lin: wp.array[wp.vec3],
    joint_C0_ang: wp.array[wp.vec3],
    joint_is_hard: wp.array[wp.int32],
    avbd_alpha: float,
    joint_dof_dim: wp.array2d[int],
    joint_rest_angle: wp.array[float],
    external_forces: wp.array[wp.vec3],
    external_torques: wp.array[wp.vec3],
    external_hessian_ll: wp.array[wp.mat33],  # Linear-linear block from rigid contacts
    external_hessian_al: wp.array[wp.mat33],  # Angular-linear coupling block from rigid contacts
    external_hessian_aa: wp.array[wp.mat33],  # Angular-angular block from rigid contacts
    # Output
    body_q_new: wp.array[wp.transform],
):
    """
    AVBD solve step for rigid bodies.

    Assembles inertial, joint, and collision contributions into a 6x6 SPD
    block system and solves via direct LDL^T.

    Algorithm:
      1. Compute inertial forces/Hessians
      2. Accumulate external forces/Hessians from rigid contacts
      3. Accumulate joint forces/Hessians from adjacent joints
      4. Solve 6x6 system via LDL^T
      5. Update pose: rotation from angular increment, position from linear increment

    Args:
        dt: Time step.
        body_ids_in_color: Body indices in current color group (for parallel coloring).
        body_q_prev: Previous body transforms (for damping and friction).
        body_q_rest: Rest transforms (for joint targets).
        body_mass: Body masses.
        body_inv_mass: Inverse masses (0 for kinematic bodies).
        body_inertia: Inertia tensors (local body frame).
        body_inertia_q: Inertial target transforms (from forward integration).
        body_com: Center of mass offsets (local body frame).
        adjacency: Body-joint adjacency (CSR format).
        joint_*: Joint configuration arrays.
        joint_penalty_k: AVBD per-constraint penalty stiffness (one scalar per solver constraint component).
        joint_sigma_start: Dahl hysteresis state at start of step.
        joint_C_fric: Dahl friction configuration per joint.
        external_forces: External linear forces from rigid contacts.
        external_torques: External angular torques from rigid contacts.
        external_hessian_ll: Linear-linear Hessian block (3x3) from rigid contacts.
        external_hessian_al: Angular-linear coupling Hessian block (3x3) from rigid contacts.
        external_hessian_aa: Angular-angular Hessian block (3x3) from rigid contacts.
        body_q: Current body transforms (input).
        body_q_new: Updated body transforms (output) for the current solve sweep.

    Note:
      - All forces, torques, and Hessian blocks are expressed in the world frame.
    """
    body_index = body_ids_in_color[tid]

    q_current = body_q[body_index]

    # Early exit for kinematic bodies
    if body_inv_mass[body_index] == 0.0:
        body_q_new[body_index] = q_current
        return

    # Inertial force and Hessian
    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # Read body properties
    q_inertial = body_inertia_q[body_index]
    body_com_local = body_com[body_index]
    m = body_mass[body_index]
    I_body = body_inertia[body_index]

    # Extract poses
    pos_current = wp.transform_get_translation(q_current)
    rot_current = wp.transform_get_rotation(q_current)
    pos_star = wp.transform_get_translation(q_inertial)
    rot_star = wp.transform_get_rotation(q_inertial)

    # Compute COM positions
    com_current = pos_current + wp.quat_rotate(rot_current, body_com_local)
    com_star = pos_star + wp.quat_rotate(rot_star, body_com_local)

    # Linear inertial force and Hessian
    inertial_coeff = m * dt_sqr_reciprocal
    f_lin = (com_star - com_current) * inertial_coeff

    # Compute relative rotation via quaternion difference
    # dq = q_current^-1 * q_star
    q_delta = wp.mul(wp.quat_inverse(rot_current), rot_star)

    # Enforce shortest path (w > 0) to avoid double-cover ambiguity
    if q_delta[3] < 0.0:
        q_delta = wp.quat(-q_delta[0], -q_delta[1], -q_delta[2], -q_delta[3])

    # Rotation vector
    axis_body, angle_body = wp.quat_to_axis_angle(q_delta)
    theta_body = axis_body * angle_body

    # Angular inertial torque
    tau_body = I_body * (theta_body * dt_sqr_reciprocal)
    tau_world = wp.quat_rotate(rot_current, tau_body)

    # Angular Hessian in world frame: use full inertia (supports off-diagonal products of inertia)
    R_cur = wp.quat_to_matrix(rot_current)
    I_world = R_cur * I_body * wp.transpose(R_cur)
    angular_hessian = dt_sqr_reciprocal * I_world

    # Accumulate external forces (rigid contacts)
    # Read external contributions
    ext_torque = external_torques[body_index]
    ext_force = external_forces[body_index]
    ext_h_aa = external_hessian_aa[body_index]
    ext_h_al = external_hessian_al[body_index]
    ext_h_ll = external_hessian_ll[body_index]

    f_torque = tau_world + ext_torque
    f_force = f_lin + ext_force

    h_aa = angular_hessian + ext_h_aa
    h_al = ext_h_al
    h_ll = wp.mat33(
        ext_h_ll[0, 0] + inertial_coeff,
        ext_h_ll[0, 1],
        ext_h_ll[0, 2],
        ext_h_ll[1, 0],
        ext_h_ll[1, 1] + inertial_coeff,
        ext_h_ll[1, 2],
        ext_h_ll[2, 0],
        ext_h_ll[2, 1],
        ext_h_ll[2, 2] + inertial_coeff,
    )

    # This prototype is limited to joint-free VBD views (checked by the adapter).

    # Regularize angular Hessian
    trA = wp.trace(h_aa) / 3.0
    epsA = 1.0e-9 * (trA + 1.0)
    h_aa[0, 0] = h_aa[0, 0] + epsA
    h_aa[1, 1] = h_aa[1, 1] + epsA
    h_aa[2, 2] = h_aa[2, 2] + epsA

    # Solve 6x6 system via direct LDL^T
    x_inc, w_world = rk.ldlt6_solve(h_ll, h_aa, h_al, f_force, f_torque)

    # Update pose from increments
    # Convert angular increment to quaternion
    if rk._USE_SMALL_ANGLE_APPROX:
        half_w = w_world * 0.5
        dq_world = wp.quat(half_w[0], half_w[1], half_w[2], 1.0)
        dq_world = wp.normalize(dq_world)
    else:
        ang_mag = wp.length(w_world)
        if ang_mag > rk._SMALL_ANGLE_EPS:
            dq_world = wp.quat_from_axis_angle(w_world / ang_mag, ang_mag)
        else:
            half_w = w_world * 0.5
            dq_world = wp.quat(half_w[0], half_w[1], half_w[2], 1.0)
            dq_world = wp.normalize(dq_world)

    # Apply rotation
    rot_new = wp.mul(dq_world, rot_current)
    rot_new = wp.normalize(rot_new)

    # Update position
    com_new = com_current + x_inc
    pos_new = com_new - wp.quat_rotate(rot_new, body_com_local)

    body_q_new[body_index] = wp.transform(pos_new, rot_new)


@wp.kernel(enable_backward=False)
def fused_rigid_color(soft: SoftInputs, rigid: RigidInputs, solve: SolveInputs):
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
    if tid % 32 == 0:
        body = solve.body_ids_in_color[tid // 32]
        solve.external_forces[body] = force
        solve.external_torques[body] = torque
        solve.external_hessian_ll[body] = ll
        solve.external_hessian_al[body] = al
        solve.external_hessian_aa[body] = aa
        _solve_lane(
            tid // 32,
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


class RigidFusionAdapter:
    """Intercept one complete mixed-contact color solve using a frozen pose view."""

    def __init__(self, original, model):
        solved = {int(body) for group in model.body_color_groups for body in group.numpy()}
        if model.joint_count:
            for kind, parent, child in zip(
                model.joint_type.numpy(), model.joint_parent.numpy(), model.joint_child.numpy(), strict=True
            ):
                if int(kind) != int(rk.JointType.FREE) and (int(parent) in solved or int(child) in solved):
                    raise ValueError("The fused prototype requires free-body solve groups")
        self.original = original
        self.inputs = {}
        self.frozen_q = wp.clone(model.body_q)

    def __call__(self, *positional, **kwargs):
        kernel = kwargs.get("kernel", positional[0] if positional else None)
        if kernel in (rk.accumulate_body_particle_contacts_per_body, rk.accumulate_body_body_contacts_per_body):
            self.inputs[kernel] = [*kwargs["inputs"], *kwargs["outputs"]]
            return None
        if kernel in (
            rk.accumulate_body_particle_contact_dense_partials,
            rk.accumulate_body_particle_contact_dense_reduction,
            rk.accumulate_body_particle_contact_dense_single,
        ):
            return None
        if kernel is rk.solve_rigid_body:
            values = [*kwargs["inputs"], *kwargs["outputs"]]
            current_q = values[2]
            wp.copy(self.frozen_q, current_q)
            structures = []
            lists = [
                self.inputs[rk.accumulate_body_particle_contacts_per_body],
                self.inputs[rk.accumulate_body_body_contacts_per_body],
                values,
            ]
            for index, (kind, fields, entries) in enumerate(
                zip((SoftInputs, RigidInputs, SolveInputs), FIELDS, lists, strict=True)
            ):
                record = kind()
                for field, value in zip(fields, entries, strict=True):
                    selected = value
                    if field == "body_q":
                        selected = self.frozen_q
                    if index == 0 and field == "dense_contact_threshold":
                        selected = 0
                    setattr(record, field, selected)
                structures.append(record)
            self.last_structures = structures
            return self.original(
                fused_rigid_color,
                dim=values[1].size * 32,
                block_dim=32,
                inputs=structures,
                device=kwargs["device"],
            )
        return self.original(*positional, **kwargs)


FIELDS = [
    [
        "dt",
        "color_group",
        "particle_q",
        "particle_q_prev",
        "particle_radius",
        "body_q_prev",
        "body_q",
        "body_qd",
        "body_com",
        "body_inv_mass",
        "shape_body",
        "friction_epsilon",
        "body_particle_contact_penalty_k",
        "body_particle_contact_material_ke",
        "body_particle_contact_material_kd",
        "body_particle_contact_material_mu",
        "body_particle_contact_count",
        "soft_contact_indices",
        "body_particle_contact_shape",
        "body_particle_contact_body_pos",
        "body_particle_contact_body_vel",
        "body_particle_contact_normal",
        "soft_contact_barycentric",
        "shape_margin",
        "body_particle_contact_buffer_pre_alloc",
        "body_particle_contact_counts",
        "body_particle_contact_indices",
        "dense_contact_threshold",
        "body_forces",
        "body_torques",
        "body_hessian_ll",
        "body_hessian_al",
        "body_hessian_aa",
    ],
    [
        "dt",
        "color_group",
        "body_q_prev",
        "body_q",
        "body_com",
        "body_inv_mass",
        "body_colors",
        "friction_epsilon",
        "contact_penalty_k",
        "contact_material_ke",
        "contact_material_kd",
        "contact_material_mu",
        "contact_lambda",
        "contact_C0",
        "avbd_alpha",
        "hard_contacts",
        "rigid_contact_count",
        "rigid_contact_shape0",
        "rigid_contact_shape1",
        "rigid_contact_point0",
        "rigid_contact_point1",
        "rigid_contact_offset0",
        "rigid_contact_offset1",
        "rigid_contact_normal",
        "rigid_contact_margin0",
        "rigid_contact_margin1",
        "shape_body",
        "body_contact_buffer_pre_alloc",
        "body_contact_counts",
        "body_contact_indices",
        "body_forces",
        "body_torques",
        "body_hessian_ll",
        "body_hessian_al",
        "body_hessian_aa",
    ],
    [
        "dt",
        "body_ids_in_color",
        "body_q",
        "body_q_prev",
        "body_q_rest",
        "body_mass",
        "body_inv_mass",
        "body_inertia",
        "body_inertia_q",
        "body_com",
        "adjacency",
        "joint_type",
        "joint_enabled",
        "joint_parent",
        "joint_child",
        "joint_X_p",
        "joint_X_c",
        "joint_axis",
        "joint_qd_start",
        "joint_target_q_start",
        "joint_constraint_start",
        "joint_penalty_k",
        "joint_penalty_kd",
        "joint_sigma_start",
        "joint_C_fric",
        "joint_target_ke",
        "joint_target_kd",
        "joint_target_q",
        "joint_target_qd",
        "joint_limit_lower",
        "joint_limit_upper",
        "joint_limit_ke",
        "joint_limit_kd",
        "joint_lambda_lin",
        "joint_lambda_ang",
        "joint_C0_lin",
        "joint_C0_ang",
        "joint_is_hard",
        "avbd_alpha",
        "joint_dof_dim",
        "joint_rest_angle",
        "external_forces",
        "external_torques",
        "external_hessian_ll",
        "external_hessian_al",
        "external_hessian_aa",
        "body_q_new",
    ],
]

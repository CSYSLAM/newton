# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic single-block dense reduction; production remains unchanged."""

import warp as wp

from newton._src.solvers.mjvbd_v2.vbd.rigid_vbd_kernels import (
    _BODY_PARTICLE_CONTACT_BLOCK_DIM,
    _evaluate_body_particle_contact_reaction,
)


@wp.kernel
def accumulate_body_particle_contact_dense_single(
    dt: float,
    color_group: wp.array[wp.int32],
    chunks_per_body: int,
    dense_contact_threshold: int,
    particle_q: wp.array[wp.vec3],
    particle_q_prev: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    body_q_prev: wp.array[wp.transform],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    shape_body: wp.array[int],
    friction_epsilon: float,
    body_particle_contact_penalty_k: wp.array[float],
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
    partial_forces: wp.array[wp.vec3],
    partial_torques: wp.array[wp.vec3],
    partial_hessian_ll: wp.array[wp.mat33],
    partial_hessian_al: wp.array[wp.mat33],
    partial_hessian_aa: wp.array[wp.mat33],
):
    """Reduce all active contacts with one block per dense body."""
    tid = wp.tid()
    block_idx = tid // _BODY_PARTICLE_CONTACT_BLOCK_DIM
    lane = tid % _BODY_PARTICLE_CONTACT_BLOCK_DIM
    body_idx_in_group = block_idx
    body_id = color_group[body_idx_in_group]

    num_contacts = body_particle_contact_counts[body_id]
    if num_contacts > body_particle_contact_buffer_pre_alloc:
        num_contacts = body_particle_contact_buffer_pre_alloc
    if num_contacts < dense_contact_threshold:
        return
    partial_idx = body_id

    force = wp.vec3(0.0)
    torque = wp.vec3(0.0)
    h_ll = wp.mat33(0.0)
    h_al = wp.mat33(0.0)
    h_aa = wp.mat33(0.0)
    for contact_offset in range(lane, num_contacts, _BODY_PARTICLE_CONTACT_BLOCK_DIM):
        contact_idx = body_particle_contact_indices[body_id * body_particle_contact_buffer_pre_alloc + contact_offset]
        if contact_idx < body_particle_contact_count[0]:
            X_wb = body_q[body_id]
            X_wb_prev = body_q_prev[body_id]
            com_world = wp.transform_point(X_wb, body_com[body_id])
            f, t, ll, al, aa = _evaluate_body_particle_contact_reaction(
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
            force += f
            torque += t
            h_ll += ll
            h_al += al
            h_aa += aa

    force_total = wp.tile_reduce(wp.add, wp.tile(force, preserve_type=True))[0]
    torque_total = wp.tile_reduce(wp.add, wp.tile(torque, preserve_type=True))[0]
    h_ll_total = wp.tile_reduce(wp.add, wp.tile(h_ll, preserve_type=True))[0]
    h_al_total = wp.tile_reduce(wp.add, wp.tile(h_al, preserve_type=True))[0]
    h_aa_total = wp.tile_reduce(wp.add, wp.tile(h_aa, preserve_type=True))[0]
    if lane == 0:
        wp.atomic_add(partial_forces, partial_idx, force_total)
        wp.atomic_add(partial_torques, partial_idx, torque_total)
        wp.atomic_add(partial_hessian_ll, partial_idx, h_ll_total)
        wp.atomic_add(partial_hessian_al, partial_idx, h_al_total)
        wp.atomic_add(partial_hessian_aa, partial_idx, h_aa_total)

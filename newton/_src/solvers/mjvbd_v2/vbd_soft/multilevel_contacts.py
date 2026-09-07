# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Extract the same frozen contact stencils used by the soft VBD solver."""

import warp as wp

from ..contact_projection import ContactProjection, ContactProjectionData, append_contact
from . import particle_vbd_kernels as k


@wp.kernel(enable_backward=False)
def project_self_contacts(
    dt: float,
    pos_prev: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    triangles: wp.array2d[wp.int32],
    edges: wp.array2d[wp.int32],
    collision_info: wp.array[k.TriMeshCollisionInfo],
    radius: float,
    stiffness: float,
    damping: float,
    friction: float,
    materials: wp.array[wp.vec3],
    material_index: wp.array[int],
    use_materials: bool,
    friction_epsilon: float,
    parallel_epsilon: float,
    has_contact: wp.array[wp.int32],
    fine_to_coarse: wp.array[wp.int32],
    data: ContactProjectionData,
):
    primitive = wp.tid()
    if has_contact[0] == 0:
        return
    info = collision_info[0]
    material = k._select_soft_contact_material(stiffness, damping, friction, materials, material_index, use_materials)
    if primitive < info.vertex_colliding_triangles_buffer_sizes.shape[0]:
        if info.vertex_colliding_triangles_count[primitive] > info.vertex_colliding_triangles_buffer_sizes[primitive]:
            wp.atomic_or(data.overflow, 0, 8)
        for slot in range(k.get_vertex_colliding_triangles_count(info, primitive)):
            tri = info.vertex_colliding_triangles[2 * (info.vertex_colliding_triangles_offsets[primitive] + slot) + 1]
            if tri >= 0:
                active, _f0, _f1, _f2, _f3, _h0, _h1, _h2, hessian = (
                    k.evaluate_vertex_triangle_collision_force_hessian_4_vertices(
                        primitive,
                        tri,
                        pos,
                        pos_prev,
                        triangles,
                        radius,
                        material[0],
                        material[1],
                        material[2],
                        friction_epsilon,
                        dt,
                    )
                )
                if active:
                    a, b, c = triangles[tri, 0], triangles[tri, 1], triangles[tri, 2]
                    _point, bary, _feature = k.triangle_closest_point(pos[a], pos[b], pos[c], pos[primitive])
                    append_contact(
                        wp.vec4i(a, b, c, primitive),
                        wp.vec4(-bary[0], -bary[1], -bary[2], 1.0),
                        hessian,
                        fine_to_coarse,
                        data,
                    )
    if primitive < info.edge_colliding_edges_buffer_sizes.shape[0]:
        if info.edge_colliding_edges_count[primitive] > info.edge_colliding_edges_buffer_sizes[primitive]:
            wp.atomic_or(data.overflow, 0, 8)
        for slot in range(k.get_edge_colliding_edges_count(info, primitive)):
            other = info.edge_colliding_edges[2 * (info.edge_colliding_edges_offsets[primitive] + slot) + 1]
            if other >= 0 and other != primitive:
                active, _f0, _f1, h0, h1 = k.evaluate_edge_edge_contact_2_vertices(
                    primitive,
                    other,
                    pos,
                    pos_prev,
                    edges,
                    radius,
                    material[0],
                    material[1],
                    material[2],
                    friction_epsilon,
                    dt,
                    parallel_epsilon,
                )
                if active:
                    # Fine VBD assembles the two directed halves separately. A
                    # missing reciprocal record must reject this coarse solve.
                    reciprocal = wp.bool(False)
                    for reverse_slot in range(k.get_edge_colliding_edges_count(info, other)):
                        reverse = info.edge_colliding_edges[
                            2 * (info.edge_colliding_edges_offsets[other] + reverse_slot) + 1
                        ]
                        if reverse == primitive:
                            reciprocal = True
                    if not reciprocal:
                        wp.atomic_or(data.overflow, 0, 16)
                    if primitive < other:
                        e0, e1, e2, e3 = edges[primitive, 2], edges[primitive, 3], edges[other, 2], edges[other, 3]
                        st = wp.closest_point_edge_edge(pos[e0], pos[e1], pos[e2], pos[e3], parallel_epsilon)
                        weights = wp.vec4(1.0 - st[0], st[0], st[1] - 1.0, -st[1])
                        # One weight on the first edge is at least 1/2: avoid
                        # dividing by a small barycentric coefficient.
                        hessian = wp.mat33(0.0)
                        if weights[0] >= weights[1]:
                            hessian = h0 / (weights[0] * weights[0])
                        else:
                            hessian = h1 / (weights[1] * weights[1])
                        append_contact(wp.vec4i(e0, e1, e2, e3), weights, hessian, fine_to_coarse, data)


@wp.kernel(enable_backward=False)
def project_body_contacts(
    dt: float,
    pos_prev: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    particle_radius: wp.array[float],
    indices: wp.array[wp.vec3i],
    count: wp.array[int],
    maximum: int,
    penalty: wp.array[float],
    damping: wp.array[float],
    friction: wp.array[float],
    friction_epsilon: float,
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_prev: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    shape: wp.array[int],
    body_pos: wp.array[wp.vec3],
    body_vel: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin: wp.array[float],
    barycentric: wp.array[wp.vec3],
    fine_to_coarse: wp.array[wp.int32],
    data: ContactProjectionData,
):
    contact = wp.tid()
    if contact == 0 and count[0] > maximum:
        wp.atomic_or(data.overflow, 0, 32)
    if contact < wp.min(count[0], maximum):
        corners = indices[contact]
        if corners[0] >= 0 and corners[1] >= 0:
            bary = barycentric[contact]
            _force, hessian, _point = k._eval_soft_ef_contact(
                contact,
                corners,
                bary,
                pos,
                pos_prev,
                particle_radius,
                penalty[contact],
                damping[contact],
                friction[contact],
                friction_epsilon,
                shape_body,
                body_q,
                body_q_prev,
                body_qd,
                body_com,
                shape,
                body_pos,
                body_vel,
                normal,
                margin,
                dt,
            )
            append_contact(
                wp.vec4i(corners[0], corners[1], corners[2], -1),
                wp.vec4(bary[0], bary[1], bary[2], 0.0),
                hessian,
                fine_to_coarse,
                data,
            )


def prepare(solver, state, contacts, body_q, body_q_prev, body_qd, dt):
    """Rebuild only the opt-in surface Galerkin contact operator on device."""
    correction = solver.particle_multilevel
    if not correction.contact_projection_enabled or correction.operator != "galerkin" or correction.use_rigid_basis:
        correction.contact_projection = None
        return
    if not solver.particle_enable_self_contact and (contacts is None or solver._soft_contact_launch_dim <= 0):
        correction.contact_projection = None
        return
    if correction.contact_projection is None:
        # Bound workspace by model size. Dense cases reject safely on overflow;
        # never grow storage or read a device contact count inside CUDA capture.
        capacity = max(1, 64 * correction.active_particle_count)
        correction.contact_projection = ContactProjection(correction.cluster_count, capacity, solver.device)
    projection = correction.contact_projection
    projection.reset()
    model = solver.model
    if solver.particle_enable_self_contact:
        wp.launch(
            project_self_contacts,
            dim=max(model.particle_count, model.edge_count),
            inputs=[
                dt,
                solver.particle_q_prev,
                state.particle_q,
                model.tri_indices,
                model.edge_indices,
                solver.trimesh_collision_info,
                solver.particle_self_contact_radius,
                model.soft_contact_ke,
                model.soft_contact_kd,
                model.soft_contact_mu,
                solver._soft_contact_materials,
                solver._soft_contact_material_index,
                solver._use_soft_contact_material_source,
                solver.friction_epsilon,
                solver.trimesh_collision_detector.edge_edge_parallel_epsilon,
                solver.has_active_self_contact,
                correction.fine_to_coarse,
                projection.data,
            ],
            device=solver.device,
        )
    if contacts is not None and solver._soft_contact_launch_dim > 0:
        wp.launch(
            project_body_contacts,
            dim=solver._soft_contact_launch_dim,
            inputs=[
                dt,
                solver.particle_q_prev,
                state.particle_q,
                model.particle_radius,
                contacts.soft_contact_indices,
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                solver.body_particle_contact_penalty_k,
                solver.body_particle_contact_material_kd,
                solver.body_particle_contact_material_mu,
                solver.friction_epsilon,
                model.shape_body,
                body_q,
                body_q_prev,
                body_qd,
                model.body_com,
                contacts.soft_contact_shape,
                contacts.soft_contact_body_pos,
                contacts.soft_contact_body_vel,
                contacts.soft_contact_normal,
                model.shape_margin,
                contacts.soft_contact_barycentric,
                correction.fine_to_coarse,
                projection.data,
            ],
            device=solver.device,
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnostic barycentric contact gather fused into surface elasticity."""

import warp as wp

from newton._src.solvers.mjvbd_v2.vbd import particle_vbd_kernels as pk
from newton._src.solvers.mjvbd_v2.vbd import rigid_vbd_kernels as rk


@wp.struct
class ContactInputs:
    dt: float
    current_color: int
    pos_anchor: wp.array[wp.vec3]
    pos: wp.array[wp.vec3]
    particle_colors: wp.array[int]
    contact_color_masks: wp.array[wp.uint32]
    use_contact_color_masks: bool
    friction_epsilon: float
    particle_radius: wp.array[float]
    body_particle_contact_indices: wp.array[wp.vec3i]
    body_particle_contact_count: wp.array[int]
    body_particle_contact_max: int
    body_particle_contact_penalty_k: wp.array[float]
    body_particle_contact_material_ke: wp.array[float]
    body_particle_contact_material_kd: wp.array[float]
    body_particle_contact_material_mu: wp.array[float]
    shape_body: wp.array[int]
    body_q: wp.array[wp.transform]
    body_q_prev: wp.array[wp.transform]
    body_qd: wp.array[wp.spatial_vector]
    body_com: wp.array[wp.vec3]
    contact_shape: wp.array[int]
    contact_body_pos: wp.array[wp.vec3]
    contact_body_vel: wp.array[wp.vec3]
    contact_normal: wp.array[wp.vec3]
    shape_margin: wp.array[float]
    contact_barycentric: wp.array[wp.vec3]


CONTACT_FIELDS = [
    "dt",
    "current_color",
    "pos_anchor",
    "pos",
    "particle_colors",
    "contact_color_masks",
    "use_contact_color_masks",
    "friction_epsilon",
    "particle_radius",
    "body_particle_contact_indices",
    "body_particle_contact_count",
    "body_particle_contact_max",
    "body_particle_contact_penalty_k",
    "body_particle_contact_material_ke",
    "body_particle_contact_material_kd",
    "body_particle_contact_material_mu",
    "shape_body",
    "body_q",
    "body_q_prev",
    "body_qd",
    "body_com",
    "contact_shape",
    "contact_body_pos",
    "contact_body_vel",
    "contact_normal",
    "shape_margin",
    "contact_barycentric",
]


@wp.kernel(enable_backward=False)
def build_corner_links(c: ContactInputs, heads: wp.array[int], links: wp.array[int]):
    for contact in range(
        wp.tid(),
        min(c.body_particle_contact_count[0], c.body_particle_contact_max),
        min(c.body_particle_contact_max, 4096),
    ):
        corners = c.body_particle_contact_indices[contact]
        for corner in range(3):
            if corners[corner] >= 0:
                slot = 3 * contact + corner
                links[slot] = wp.atomic_exch(heads, corners[corner], slot)


@wp.func
def gather_corner_contacts(particle: int, lane: int, c: ContactInputs, heads: wp.array[int], links: wp.array[int]):
    entry = heads[particle]
    for _skip in range(lane):
        if entry >= 0:
            entry = links[entry]
    force = wp.vec3(0.0)
    hessian = wp.mat33(0.0)
    while entry >= 0:
        contact = entry // 3
        corner = entry % 3
        bary = c.contact_barycentric[contact]
        f, h, _point = rk._eval_soft_ef_contact(
            contact,
            c.body_particle_contact_indices[contact],
            bary,
            c.pos,
            c.pos_anchor,
            c.particle_radius,
            c.body_particle_contact_penalty_k[contact],
            c.body_particle_contact_material_kd[contact],
            c.body_particle_contact_material_mu[contact],
            c.friction_epsilon,
            c.shape_body,
            c.body_q,
            c.body_q_prev,
            c.body_qd,
            c.body_com,
            c.contact_shape,
            c.contact_body_pos,
            c.contact_body_vel,
            c.contact_normal,
            c.shape_margin,
            c.dt,
        )
        weight = bary[corner]
        force += weight * f
        hessian += weight * weight * h
        for _skip in range(16):
            if entry >= 0:
                entry = links[entry]
    return force, hessian


@wp.kernel(enable_backward=False)
def fused_surface_elasticity(
    dt: float,
    particle_ids_in_color: wp.array[wp.int32],
    pos_prev: wp.array[wp.vec3],
    pos: wp.array[wp.vec3],
    mass: wp.array[float],
    inertia: wp.array[wp.vec3],
    particle_flags: wp.array[wp.int32],
    tri_indices: wp.array2d[wp.int32],
    tri_poses: wp.array[wp.mat22],
    tri_materials: wp.array2d[float],
    tri_areas: wp.array[float],
    edge_indices: wp.array2d[wp.int32],
    edge_rest_angles: wp.array[float],
    edge_rest_length: wp.array[float],
    edge_bending_properties: wp.array2d[float],
    particle_adjacency: pk.MeshAdjacencyData,
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
    skip_active_checks: int,
    skip_material_checks: int,
    contact_free_relaxation: float,
    contacts: ContactInputs,
    heads: wp.array[int],
    links: wp.array[int],
    particle_displacements: wp.array[wp.vec3],
):
    """Solve membrane and bending elasticity for particles without adjacent tetrahedra."""
    tid = wp.tid()
    block_idx = tid // pk.TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    thread_idx = tid % pk.TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
    particle = particle_ids_in_color[block_idx]
    if skip_active_checks == 0 and (not particle_flags[particle] & pk.ParticleFlags.ACTIVE or mass[particle] == 0.0):
        if thread_idx == 0:
            particle_displacements[particle] = wp.vec3(0.0)
        return

    cf, ch = gather_corner_contacts(particle, thread_idx, contacts, heads, links)
    contact_f = wp.tile_reduce(wp.add, wp.tile(cf, preserve_type=True))[0]
    contact_h = wp.tile_reduce(wp.add, wp.tile(ch, preserve_type=True))[0]
    f = wp.vec3(0.0)
    h = wp.mat33(0.0)
    counter = wp.int32(0)
    faces = pk.get_vertex_num_adjacent_faces(particle_adjacency, particle)
    while counter + thread_idx < faces:
        adjacent = counter + thread_idx
        counter += pk.TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
        tri, order = pk.get_vertex_adjacent_face_id_order(particle_adjacency, particle, adjacent)
        if skip_material_checks == 1 or tri_materials[tri, 0] > 0.0 or tri_materials[tri, 1] > 0.0:
            f_tri, h_tri = pk.evaluate_neo_hookean_membrane_force_hessian(
                tri,
                order,
                pos,
                pos_prev,
                tri_indices,
                tri_poses[tri],
                tri_areas[tri],
                tri_materials[tri, 0],
                tri_materials[tri, 1],
                tri_materials[tri, 2],
                dt,
            )
            f += f_tri
            h += h_tri

    counter = wp.int32(0)
    edges = pk.get_vertex_num_adjacent_edges(particle_adjacency, particle)
    while counter + thread_idx < edges:
        adjacent = counter + thread_idx
        counter += pk.TILE_SIZE_TRI_MESH_ELASTICITY_SOLVE
        edge, order = pk.get_vertex_adjacent_edge_id_order(particle_adjacency, particle, adjacent)
        if skip_material_checks == 1 or edge_bending_properties[edge, 0] > 0.0:
            f_edge, h_edge = pk.evaluate_dihedral_angle_based_bending_force_hessian(
                edge,
                order,
                pos,
                pos_prev,
                edge_indices,
                edge_rest_angles,
                edge_rest_length,
                edge_bending_properties[edge, 0],
                edge_bending_properties[edge, 1],
                dt,
            )
            f += f_edge
            h += h_edge

    f_total = wp.tile_reduce(wp.add, wp.tile(f, preserve_type=True))[0]
    h_total = wp.tile_reduce(wp.add, wp.tile(h, preserve_type=True))[0]
    if thread_idx == 0:
        inv_dt_sq = 1.0 / (dt * dt)
        h_total += mass[particle] * inv_dt_sq * wp.identity(n=3, dtype=float) + (
            particle_hessians[particle] + contact_h
        )
        if abs(wp.determinant(h_total)) > 1.0e-8:
            f_total += mass[particle] * (inertia[particle] - pos[particle]) * inv_dt_sq + (
                particle_forces[particle] + contact_f
            )
            delta = wp.inverse(h_total) * f_total
            if contact_free_relaxation != 1.0:
                contact_hessian = particle_hessians[particle] + contact_h
                # Do not extrapolate constrained rows, including off-diagonal contributions.
                if wp.ddot(contact_hessian, contact_hessian) == 0.0:
                    delta *= contact_free_relaxation
            particle_displacements[particle] += delta


class ParticleFusionAdapter:
    """Build complete corner adjacency once per contact refresh, never truncate it."""

    def __init__(self, original, model):
        self.original = original
        self.heads = wp.empty(model.particle_count, dtype=int, device=model.device)
        self.links = None
        self.contacts = None
        self.rebuild = True
        self.device = model.device
        self.fused_launches = 0

    def __call__(self, *positional, **kwargs):
        kernel = kwargs.get("kernel", positional[0] if positional else None)
        if kernel is rk.init_body_particle_contacts:
            self.rebuild = True
        if kernel is pk.accumulate_particle_body_contact_force_and_hessian and kwargs["inputs"][1] >= 0:
            self.contacts = ContactInputs()
            for field, value in zip(CONTACT_FIELDS, kwargs["inputs"], strict=True):
                setattr(self.contacts, field, value)
            capacity = 3 * self.contacts.body_particle_contact_max
            if self.links is None or self.links.size < capacity:
                if self.device.is_capturing:
                    raise RuntimeError("Initialize corner capacity outside CUDA Graph capture")
                self.links = wp.empty(capacity, dtype=int, device=self.device)
                self.rebuild = True
            if self.rebuild:
                self.heads.fill_(-1)
                self.original(
                    build_corner_links,
                    dim=min(self.contacts.body_particle_contact_max, 4096),
                    inputs=[self.contacts, self.heads, self.links],
                    device=self.device,
                )
                self.rebuild = False
            return None
        if kernel is pk.solve_surface_elasticity_tile and self.contacts is not None:
            self.fused_launches += 1
            return self.original(
                fused_surface_elasticity,
                dim=kwargs["dim"],
                block_dim=kwargs["block_dim"],
                inputs=[*kwargs["inputs"], self.contacts, self.heads, self.links],
                outputs=kwargs["outputs"],
                device=self.device,
            )
        return self.original(*positional, **kwargs)

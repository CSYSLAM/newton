# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Augment the fine surface Newton solve with free-body translation unknowns."""

import numpy as np
import warp as wp
from rigid_fusion_probe import FIELDS, RigidInputs, SoftInputs, SolveInputs
from two_level_probe import TwoLevelPCG

from newton._src.solvers.mjvbd_v2.contact_projection import ContactProjection, ContactProjectionData, append_contact
from newton._src.solvers.mjvbd_v2.vbd import rigid_vbd_kernels as rk


@wp.kernel(enable_backward=False)
def initialize_bodies(
    data: SolveInputs,
    body_ids: wp.array[int],
    particle_count: int,
    diagonal_slots: wp.array[int],
    blocks: wp.array[wp.mat33],
    rhs: wp.array[wp.vec3],
):
    index = wp.tid()
    body = body_ids[index]
    row = particle_count + index
    mass_term = data.body_mass[body] / (data.dt * data.dt)
    current = wp.transform_point(data.body_q_new[body], data.body_com[body])
    target = wp.transform_point(data.body_inertia_q[body], data.body_com[body])
    rhs[row] = mass_term * (target - current)
    blocks[diagonal_slots[row]] = mass_term * wp.identity(n=3, dtype=float)


@wp.kernel(enable_backward=False)
def assemble_soft(
    data: SoftInputs,
    body_rows: wp.array[int],
    identity: wp.array[int],
    slots: wp.array[int],
    blocks: wp.array[wp.mat33],
    rhs: wp.array[wp.vec3],
    contacts: ContactProjectionData,
):
    for record in range(
        wp.tid(), wp.min(data.body_particle_contact_count[0], data.soft_contact_indices.shape[0]), 4096
    ):
        corners = data.soft_contact_indices[record]
        if corners[0] < 0:
            continue
        bary = data.soft_contact_barycentric[record]
        force, hessian, _point = rk._eval_soft_ef_contact(
            record,
            corners,
            bary,
            data.particle_q,
            data.particle_q_prev,
            data.particle_radius,
            data.body_particle_contact_penalty_k[record],
            data.body_particle_contact_material_kd[record],
            data.body_particle_contact_material_mu[record],
            data.friction_epsilon,
            data.shape_body,
            data.body_q,
            data.body_q_prev,
            data.body_qd,
            data.body_com,
            data.body_particle_contact_shape,
            data.body_particle_contact_body_pos,
            data.body_particle_contact_body_vel,
            data.body_particle_contact_normal,
            data.shape_margin,
            data.dt,
        )
        shape = data.body_particle_contact_shape[record]
        body = data.shape_body[shape]
        row = int(-1)
        if body >= 0:
            row = body_rows[body]
        if row >= 0:
            wp.atomic_add(rhs, row, -force)
            wp.atomic_add(blocks, slots[row], hessian)
        append_contact(
            wp.vec4i(corners[0], corners[1], corners[2], row),
            wp.vec4(bary[0], bary[1], bary[2], -1.0),
            hessian,
            identity,
            contacts,
        )


@wp.kernel(enable_backward=False)
def assemble_rigid(
    data: RigidInputs,
    body_rows: wp.array[int],
    identity: wp.array[int],
    slots: wp.array[int],
    blocks: wp.array[wp.mat33],
    rhs: wp.array[wp.vec3],
    contacts: ContactProjectionData,
):
    for record in range(wp.tid(), wp.min(data.rigid_contact_count[0], data.rigid_contact_shape0.shape[0]), 4096):
        s0, s1 = data.rigid_contact_shape0[record], data.rigid_contact_shape1[record]
        b0 = data.shape_body[s0] if s0 >= 0 else -1
        b1 = data.shape_body[s1] if s1 >= 0 else -1
        row0 = body_rows[b0] if b0 >= 0 else -1
        row1 = body_rows[b1] if b1 >= 0 else -1
        if row0 < 0 and row1 < 0:
            continue
        p0, p1 = data.rigid_contact_point0[record], data.rigid_contact_point1[record]
        world0 = wp.transform_point(data.body_q[b0], p0) if b0 >= 0 else p0
        world1 = wp.transform_point(data.body_q[b1], p1) if b1 >= 0 else p1
        normal = data.rigid_contact_normal[record]
        penetration = -rk.contact_surface_separation(
            world0, world1, normal, data.rigid_contact_margin0[record], data.rigid_contact_margin1[record]
        )
        if penetration <= rk._SMALL_LENGTH_EPS:
            continue
        k = data.contact_penalty_k[record]
        f0, _t0, h0, _a0, _aa0, f1, _t1, h1, _a1, _aa1 = rk.evaluate_rigid_contact_from_collision(
            b0,
            b1,
            data.body_q,
            data.body_q_prev,
            data.body_com,
            p0,
            p1,
            data.rigid_contact_offset0[record],
            data.rigid_contact_offset1[record],
            normal,
            penetration,
            k,
            k,
            data.contact_material_kd[record],
            wp.vec3(0.0),
            data.contact_material_mu[record],
            data.friction_epsilon,
            0,
            data.dt,
            wp.vec3(0.0),
        )
        if row0 >= 0:
            wp.atomic_add(rhs, row0, f0)
            wp.atomic_add(blocks, slots[row0], h0)
        if row1 >= 0:
            wp.atomic_add(rhs, row1, f1)
            wp.atomic_add(blocks, slots[row1], h1)
        append_contact(wp.vec4i(row0, row1, -1, -1), wp.vec4(1.0, -1.0, 0.0, 0.0), h0, identity, contacts)


@wp.kernel(enable_backward=False)
def bound_step(
    particle_count: int,
    radius: wp.array[float],
    radius_fraction: float,
    delta: wp.array[wp.vec3],
    scale: wp.array[float],
):
    row = wp.tid()
    limit = float(0.001)
    if row < particle_count:
        limit = radius_fraction * radius[row]
    length = wp.length(delta[row])
    if not wp.isfinite(length):
        wp.atomic_min(scale, 0, 0.0)
    elif length > limit:
        wp.atomic_min(scale, 0, limit / length)


@wp.kernel(enable_backward=False)
def copy_fine(particle_count: int, scale: wp.array[float], delta: wp.array[wp.vec3], fine: wp.array[wp.vec3]):
    row = wp.tid()
    delta[row] *= scale[0]
    if row < particle_count:
        fine[row] = delta[row]


@wp.kernel(enable_backward=False)
def commit_bodies(
    particle_count: int,
    body_ids: wp.array[int],
    status: wp.array[int],
    delta: wp.array[wp.vec3],
    q: wp.array[wp.transform],
):
    if status[0] == 0:
        index = wp.tid()
        body = body_ids[index]
        pose = q[body]
        q[body] = wp.transform(
            wp.transform_get_translation(pose) + delta[particle_count + index], wp.transform_get_rotation(pose)
        )


class CoupledTranslationPCG:
    def __init__(self, solver, correction, ritz, fusion):
        if solver.rigid_contact_hard:
            raise ValueError("Coupled translation diagnostic supports soft contacts only")
        self.solver, self.correction, self.ritz, self.fusion = solver, correction, ritz, fusion
        self.device = solver.device
        self.particle_count = solver.model.particle_count
        dynamic = np.concatenate([group.numpy() for group in solver.model.body_color_groups])
        if np.any(solver.body_inv_mass_effective.numpy()[dynamic] <= 0):
            raise ValueError("Coupled diagnostic requires movable free-body groups")
        self.body_ids = wp.array(dynamic, dtype=int, device=self.device)
        self.count = self.particle_count + len(dynamic)
        rows = np.full(solver.model.body_count, -1, dtype=np.int32)
        rows[dynamic] = np.arange(self.particle_count, self.count)
        self.body_rows = wp.array(rows, dtype=int, device=self.device)
        self.identity = wp.array(np.arange(self.count), dtype=int, device=self.device)
        original_offsets = correction.coarse_matrix_offsets.numpy()
        original_columns = correction.coarse_matrix_columns.numpy()
        offsets = np.concatenate([original_offsets, original_offsets[-1] + np.arange(1, len(dynamic) + 1)])
        columns = np.concatenate([original_columns, np.arange(self.particle_count, self.count)])
        slots = np.concatenate(
            [correction.coarse_diagonal_slots.numpy(), len(original_columns) + np.arange(len(dynamic))]
        )
        self.offsets = wp.array(offsets, dtype=int, device=self.device)
        self.columns = wp.array(columns, dtype=int, device=self.device)
        self.slots = wp.array(slots, dtype=int, device=self.device)
        self.blocks = wp.empty(len(columns), dtype=wp.mat33, device=self.device)
        self.rhs = wp.empty(self.count, dtype=wp.vec3, device=self.device)
        self.contacts = ContactProjection(self.count, 1000000, self.device)
        groups = np.concatenate([ritz.groups.numpy(), np.full(len(dynamic), -1)])
        ritz.groups = wp.array(groups, dtype=int, device=self.device)
        self.pcg = TwoLevelPCG(ritz)
        self.work = [wp.zeros(self.count, dtype=wp.vec3, device=self.device) for _ in range(5)]
        self.work.append(wp.empty(self.count, dtype=wp.mat33, device=self.device))
        self.scale = wp.ones(1, dtype=float, device=self.device)
        self.status = None
        self.body_q = None

    def solve(self, inputs, outputs):
        records = []
        for kind, fields, original in zip(
            (SoftInputs, RigidInputs, SolveInputs), FIELDS, self.fusion.last_structures, strict=True
        ):
            record = kind()
            for field in fields:
                setattr(record, field, getattr(original, field))
            records.append(record)
        soft, rigid, body = records
        soft.body_q = body.body_q_new
        soft.particle_q = self.ritz.q
        rigid.body_q = body.body_q_new
        self.body_q = body.body_q_new
        self.status = outputs[6]
        wp.copy(self.blocks, inputs[4], count=inputs[4].size)
        wp.copy(self.rhs, inputs[5], count=self.particle_count)
        self.contacts.reset()
        wp.launch(
            initialize_bodies,
            dim=self.body_ids.size,
            inputs=[body, self.body_ids, self.particle_count, self.slots, self.blocks, self.rhs],
            device=self.device,
        )
        wp.launch(
            assemble_soft,
            dim=4096,
            inputs=[soft, self.body_rows, self.identity, self.slots, self.blocks, self.rhs, self.contacts.data],
            device=self.device,
        )
        wp.launch(
            assemble_rigid,
            dim=4096,
            inputs=[rigid, self.body_rows, self.identity, self.slots, self.blocks, self.rhs, self.contacts.data],
            device=self.device,
        )
        self.contacts.correct_diagonal(self.blocks, self.slots)
        augmented = [
            self.count,
            self.offsets,
            self.columns,
            self.slots,
            self.blocks,
            self.rhs,
            inputs[6],
            inputs[7],
            inputs[8],
            self.contacts.data,
        ]
        self.pcg.solve(augmented, [*self.work, *outputs[6:]])
        self.scale.fill_(1.0)
        wp.launch(
            bound_step,
            dim=self.count,
            inputs=[
                self.particle_count,
                self.solver.model.particle_radius,
                self.correction.max_radius_fraction,
                self.work[0],
                self.scale,
            ],
            device=self.device,
        )
        wp.launch(
            copy_fine,
            dim=self.count,
            inputs=[self.particle_count, self.scale, self.work[0], outputs[0]],
            device=self.device,
        )

    def commit(self):
        wp.launch(
            commit_bodies,
            dim=self.body_ids.size,
            inputs=[self.particle_count, self.body_ids, self.status, self.work[0], self.body_q],
            device=self.device,
        )

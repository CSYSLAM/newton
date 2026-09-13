# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Instance-owned CUDA scheduling; contact laws and iteration budgets are unchanged."""

import numpy as np
import warp as wp

from . import fast_kernels as k
from .coupled_free_body.rigid_fusion import SoftInputs


class FaceSearch:
    """Compact compatible query work without sharing buffers between pipelines."""

    def __init__(self, model, capacity):
        self.model = model
        self.capacity = capacity
        self.ids = wp.empty(capacity * 3, dtype=int, device=model.device)
        self.counts = wp.zeros(3, dtype=int, device=model.device)
        self.packed = wp.empty(capacity, dtype=int, device=model.device)
        self.total = wp.zeros(1, dtype=int, device=model.device)

    def launch(self, _kernel, **kwargs):
        values = kwargs["inputs"]
        self.counts.zero_()
        first = dict(kwargs)
        first["inputs"] = [*values, *first.pop("outputs", []), self.ids, self.counts]
        wp.launch(k.gather_face_pairs, **first)
        wp.launch(
            k.pack_face_pairs,
            4096,
            [self.capacity, self.counts, self.ids, self.total, self.packed],
            device=self.model.device,
        )
        workers = min(values[3], self.model.device.sm_count * 64)
        kwargs["inputs"] = [self.packed, self.total, 0, workers, *values[4:]]
        kwargs["dim"] = workers * 8
        kwargs["block_dim"] = 32
        wp.launch(k.search_face_pairs, **kwargs)


class SurfaceFastPath:
    """Select an empty-self-contact fused block with a conservative fallback."""

    def __init__(self, solver):
        m = solver.model
        self.eligible = bool(
            m.device.is_cuda
            and m.particle_count
            and m.particle_count * 128 * 4 <= 256 * 1024 * 1024
            and not m.requires_grad
            and not m.tet_count
            and not m.spring_count
            and not solver._pneumatic_enabled
            and not solver.particle_chebyshev_enabled
            and solver.use_particle_tile_solve
            and solver._surface_cached_kernel is None
            and solver.surface_anchor_angles is None
            and solver._particle_truncation_cache is not None
            and solver.particle_collision_detection_interval == 0
            and np.all(m.particle_mass.numpy() > 0)
            and np.all((m.particle_flags.numpy() & 1) != 0)
        )
        if self.eligible:
            colors = m.particle_colors.numpy()
            for row in m.tri_indices.numpy():
                ids = row[row >= 0]
                if len(np.unique(colors[ids])) != len(ids):
                    self.eligible = False
                    break
        self.fallback = wp.zeros(1, dtype=int, device=m.device)
        self.counts = wp.zeros(m.particle_count if self.eligible else 0, dtype=int, device=m.device)
        self.entries = wp.empty(m.particle_count * 128 if self.eligible else 0, dtype=int, device=m.device)

    def after_detection(self, solver):
        if self.eligible:
            self.fallback.zero_()
            wp.launch(
                k.detect_self_contact,
                max(solver.model.particle_count, solver.model.edge_count),
                [solver.trimesh_collision_info, self.fallback],
                device=solver.device,
            )

    def iterations(self, solver, state_in, state_out, control, contacts, dt):
        def run(*, fused=False, prepared=False):
            for iteration in range(solver.iterations):
                solver._solve_rigid_body_iteration(state_in, state_out, control, contacts, dt)
                solver._solve_particle_iteration(
                    state_in, state_out, control, contacts, dt, iteration, _fused=fused, _skip_detection=prepared
                )

        if not self.eligible or not solver.device.is_capturing or contacts is None:
            run()
            return
        if solver.particle_enable_self_contact:
            solver._collision_detection_penetration_free(state_in)
        else:
            self.fallback.zero_()
        self.counts.zero_()
        wp.launch(
            k.build_adjacency,
            min(contacts.soft_contact_max, 4096),
            [
                contacts.soft_contact_count,
                contacts.soft_contact_max,
                contacts.soft_contact_indices,
                solver.model.particle_colors,
                self.counts,
                self.entries,
                self.fallback,
            ],
            device=solver.device,
        )
        wp.capture_if(
            self.fallback, on_true=lambda: run(prepared=True), on_false=lambda: run(fused=True, prepared=True)
        )


def surface_contact_inputs(solver, state, contacts, dt, body_q, body_q_prev, body_qd):
    """Use precisely the same contact operands as the unfused particle solve."""
    s = SoftInputs()
    s.dt = dt
    s.particle_q_prev = solver.particle_q_prev
    s.particle_q = state.particle_q
    s.friction_epsilon = solver.friction_epsilon
    s.particle_radius = solver.model.particle_radius
    s.soft_contact_indices = contacts.soft_contact_indices
    s.body_particle_contact_count = contacts.soft_contact_count
    for field in (
        "body_particle_contact_penalty_k",
        "body_particle_contact_material_ke",
        "body_particle_contact_material_kd",
        "body_particle_contact_material_mu",
    ):
        setattr(s, field, getattr(solver, field))
    s.shape_body = solver.model.shape_body
    s.body_q, s.body_q_prev, s.body_qd = body_q, body_q_prev, body_qd
    s.body_com = solver.model.body_com
    for field in ("shape", "body_pos", "body_vel", "normal"):
        setattr(s, "body_particle_contact_" + field, getattr(contacts, "soft_contact_" + field))
    s.shape_margin = solver.model.shape_margin
    s.soft_contact_barycentric = contacts.soft_contact_barycentric
    return s


class SelfContactCertificate:
    """Reuse an empty candidate set only while its separation proof holds."""

    def __init__(self, solver):
        device = solver.device
        self.reference = wp.clone(solver.model.particle_q)
        self.valid = wp.zeros(1, dtype=int, device=device)
        self.needed = wp.ones(1, dtype=int, device=device)
        self.active = wp.ones(1, dtype=int, device=device)
        self.clearance = wp.zeros(1, device=device)
        self.statistics = wp.zeros(3, dtype=int, device=device)
        self.signature = None

    def invalidate(self):
        self.valid.zero_()

    def detect(self, solver, state):
        if solver.particle_self_contact_margin <= 0:
            solver._collision_detection_penetration_free_uncached(state)
            return
        detector = solver.trimesh_collision_detector
        signature = (
            solver.particle_self_contact_margin,
            solver.particle_rest_shape_contact_exclusion_radius,
            solver.particle_q_rest.ptr,
            detector.edge_edge_parallel_epsilon,
            getattr(detector.vertex_triangle_filtering_list, "ptr", 0),
            getattr(detector.edge_filtering_list, "ptr", 0),
        )
        if signature != self.signature:
            self.invalidate()
            self.signature = signature
        skin = solver.particle_self_contact_margin * 0.5
        self.needed.zero_()
        wp.launch(
            k.need_query,
            solver.model.particle_count,
            [state.particle_q, self.reference, self.valid, self.clearance, self.needed],
            device=solver.device,
        )

        def rebuild():
            solver._collision_detection_penetration_free_uncached(state)
            self.active.zero_()
            self.clearance.fill_(skin)
            self._query(
                detector,
                solver.particle_self_contact_margin + skin,
                solver.particle_rest_shape_contact_exclusion_radius,
                solver.particle_q_rest,
                solver.particle_self_contact_margin,
            )
            wp.copy(self.reference, state.particle_q)
            wp.launch(k.save_certificate, 1, [self.active, self.valid, self.statistics], device=solver.device)

        def reuse():
            solver.pos_prev_collision_detection.assign(state.particle_q)
            solver.particle_displacements.zero_()
            detector.refit(state.particle_q)
            if solver._particle_truncation_cache is not None:
                solver.truncation_ts.fill_(1.0)
            wp.launch(k.record_reuse, 1, [self.statistics], device=solver.device)

        if solver.device.is_capturing:
            wp.capture_if(self.needed, on_true=rebuild, on_false=reuse)
        elif self.needed.numpy()[0]:
            rebuild()
        else:
            reuse()

    def _query(self, detector, radius, exclusion_radius, rest, original_radius):
        shared = [self.active, self.clearance, original_radius]
        wp.launch(
            k.certify_vertex_triangle,
            dim=detector.model.particle_count,
            block_dim=detector.collision_detection_block_size,
            device=detector.model.device,
            inputs=[
                radius,
                exclusion_radius,
                detector.bvh_tris.id,
                detector.bvh_tris_group_roots,
                detector.vertex_positions,
                detector.model.tri_indices,
                detector.model.particle_world,
                detector.model.world_count,
                detector.vertex_colliding_triangles_offsets,
                detector.vertex_colliding_triangles_buffer_sizes,
                detector.triangle_colliding_vertices_offsets,
                detector.triangle_colliding_vertices_buffer_sizes,
                detector.vertex_triangle_filtering_list,
                detector.vertex_triangle_filtering_list_offsets,
                rest,
                detector.vertex_colliding_triangles,
                detector.vertex_colliding_triangles_count,
                detector.vertex_colliding_triangles_min_dist,
                detector.triangle_colliding_vertices,
                detector.triangle_colliding_vertices_count,
                detector.triangle_colliding_vertices_min_dist,
                detector.resize_flags,
                *shared,
            ],
        )
        wp.launch(
            k.certify_edge_edge,
            dim=detector.model.edge_count,
            block_dim=detector.collision_detection_block_size,
            device=detector.model.device,
            inputs=[
                radius,
                exclusion_radius,
                detector.bvh_edges.id,
                detector.bvh_edges_group_roots,
                detector.vertex_positions,
                detector.model.edge_indices,
                detector.model.particle_world,
                detector.model.world_count,
                detector.edge_colliding_edges_offsets,
                detector.edge_colliding_edges_buffer_sizes,
                detector.edge_edge_parallel_epsilon,
                detector.edge_filtering_list,
                detector.edge_filtering_list_offsets,
                rest,
                detector.edge_colliding_edges,
                detector.edge_colliding_edges_count,
                detector.edge_colliding_edges_min_dist,
                detector.resize_flags,
                *shared,
            ],
        )

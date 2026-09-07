# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private PhysX-equation surface path owned by SolverXPBD.

One sweep per TGS time slice, with shared/nonshared pair partitions and
PhysX-style copy/remap chains followed by terminal-copy averaging.
DAT is a transaction boundary around prediction, elasticity and contacts.
Uncommitted elastic candidates may leave the feasible region; only accepted
states are exposed to subsequent contact stages and velocity reconstruction.
An optional global compliant-constraint solver reuses this collision storage
and DAT transaction boundary, without using the temporal copy/remap solve.
"""

import math

import numpy as np
import warp as wp

from . import fem_contacts as collision
from . import fem_kernels as elastic
from .fem_cooking import PairBatch, categorize, color_elements


class FEMSurface:
    """Preallocate FEM, BVH and DAT storage; perform no step-time allocation."""

    def __init__(self, model, iterations, radius, margin, capacity):
        if model.requires_grad:
            raise ValueError("XPBD FEM surface currently requires requires_grad=False")
        if model.body_count or model.shape_count or model.tet_count or model.spring_count:
            raise ValueError(
                "XPBD FEM surface currently supports triangle cloth without bodies, shapes, tets or springs"
            )
        if model.tri_count == 0:
            raise ValueError("XPBD FEM surface requires a triangle mesh")
        if iterations < 1 or int(iterations) != iterations:
            raise ValueError("FEM iterations must be a positive integer")
        if not math.isfinite(radius) or radius < 0.0:
            raise ValueError("self-contact radius must be finite and nonnegative")
        if not math.isfinite(margin) or (radius > 0.0 and margin <= radius):
            raise ValueError("self-contact margin must be finite and larger than radius")
        if capacity < 1 or int(capacity) != capacity:
            raise ValueError("self-contact capacity must be a positive integer")
        faces = model.tri_indices.numpy()
        if np.unique(faces).size != model.particle_count:
            raise ValueError("Every FEM particle must belong to a triangle")
        if np.any(model.tri_areas.numpy() <= 0.0):
            raise ValueError("FEM rest triangles must have positive area")
        if np.any(model.tri_materials.numpy()[:, 3:] != 0.0):
            raise ValueError("PhysX FEM surface currently requires zero aerodynamic drag/lift")
        self.model, self.iterations = model, int(iterations)
        self.external_bvh_rebuild = False
        self.global_constraints = None
        self.radius, self.margin = float(radius), float(margin)
        self.workers = min(4096, int(capacity))
        bending = model.edge_indices.numpy()
        shared, nonshared, singles, pair_faces = categorize(faces, bending)
        singles = np.asarray(singles, dtype=np.int32)
        self.membrane_groups = [
            wp.array(singles[group], dtype=int, device=model.device) for group in color_elements(faces[singles])
        ]
        self.shared_pairs = PairBatch(model, shared, pair_faces)
        self.nonshared_pairs = PairBatch(model, nonshared, pair_faces)
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        self.edges = wp.array(edges, dtype=wp.vec2i, device=model.device)
        self.base = wp.clone(model.particle_q)
        self.pos = wp.clone(model.particle_q)
        self.prev = wp.clone(model.particle_q)
        self.candidate = wp.clone(model.particle_q)
        self.targets = wp.clone(model.particle_q)
        self.velocity = wp.zeros_like(model.particle_q)
        self.delta = wp.zeros_like(model.particle_q)
        self.weights = wp.zeros(model.particle_count, dtype=float, device=model.device)
        self.t = wp.ones_like(self.weights)
        self.status = wp.zeros(1, dtype=int, device=model.device)
        self.count = wp.zeros(1, dtype=int, device=model.device)
        self.crossing_count = wp.zeros(1, dtype=int, device=model.device)
        self.crossing_pairs = wp.empty(16, dtype=wp.vec2i, device=model.device)
        self.pairs = wp.empty(capacity, dtype=wp.vec4i, device=model.device)
        self.kinds = wp.empty(capacity, dtype=int, device=model.device)
        self.normals = wp.empty(capacity, dtype=wp.vec3, device=model.device)
        self.distances = wp.empty(capacity, dtype=wp.vec4, device=model.device)
        self.gaps = wp.empty(capacity, dtype=float, device=model.device)
        self.tri_lo = wp.empty(model.tri_count, dtype=wp.vec3, device=model.device)
        self.tri_hi = wp.empty_like(self.tri_lo)
        self.edge_lo = wp.empty(len(edges), dtype=wp.vec3, device=model.device)
        self.edge_hi = wp.empty_like(self.edge_lo)
        self._bounds()
        self.tri_bvh = wp.Bvh(self.tri_lo, self.tri_hi)
        self.edge_bvh = wp.Bvh(self.edge_lo, self.edge_hi)
        if self.radius > 0.0:
            self.validate()

    def _launch(self, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.model.device)

    def _bounds(self):
        self._launch(
            collision.primitive_bounds,
            max(self.model.tri_count, len(self.edges)),
            [
                self.base,
                self.model.tri_indices,
                self.edges,
                self.tri_lo,
                self.tri_hi,
                self.edge_lo,
                self.edge_hi,
            ],
        )

    def _detect(self):
        wp.copy(self.base, self.pos)
        self.count.zero_()
        self._bounds()
        if self.external_bvh_rebuild:
            self.tri_bvh.refit()
            self.edge_bvh.refit()
        else:
            self.tri_bvh.rebuild()
            self.edge_bvh.rebuild()
        output = [self.count, self.pairs, self.kinds, self.normals, self.distances, self.gaps, self.status]
        self._launch(
            collision.detect_vt,
            self.model.particle_count,
            [
                self.tri_bvh.id,
                self.base,
                self.model.tri_indices,
                self.model.particle_world,
                self.margin,
                *output,
            ],
        )
        self._launch(
            collision.detect_ee,
            len(self.edges),
            [
                self.edge_bvh.id,
                self.base,
                self.edges,
                self.model.particle_world,
                self.margin,
                *output,
            ],
        )

    def rebuild_bvh(self, state):
        wp.copy(self.base, state.particle_q)
        self._bounds()
        self.tri_bvh.rebuild()
        self.edge_bvh.rebuild()
        self.external_bvh_rebuild = True

    def _commit(self):
        self.t.fill_(1.0)
        if self.radius > 0.0:
            self._launch(
                collision.truncate_pairs,
                self.workers,
                [
                    self.base,
                    self.pos,
                    self.candidate,
                    self.count,
                    self.pairs,
                    self.kinds,
                    self.normals,
                    self.distances,
                    self.gaps,
                    0.85,
                    self.workers,
                    self.t,
                ],
            )
        self._launch(collision.validate_candidate, self.model.particle_count, [self.candidate, self.status])
        self._launch(
            collision.commit,
            self.model.particle_count,
            [
                self.base,
                self.candidate,
                self.t,
                0.425 * self.margin if self.radius > 0.0 else wp.inf,
                self.status,
                self.pos,
            ],
        )

    def _solve_shell(self, h):
        model = self.model
        for group in self.membrane_groups:
            self._launch(
                elastic.solve_membrane,
                len(group),
                [
                    group,
                    model.tri_indices,
                    model.tri_poses,
                    model.tri_areas,
                    model.tri_materials,
                    model.particle_inv_mass,
                    model.particle_flags,
                    h,
                    self.candidate,
                ],
            )
        for shared, batch in ((True, self.shared_pairs), (False, self.nonshared_pairs)):
            if not len(batch.ids):
                continue
            batch.valid.zero_()
            start = 0
            for end in batch.ends:
                if end > start:
                    self._launch(
                        elastic.solve_pair_partition,
                        end - start,
                        [
                            start,
                            batch.ids,
                            batch.vertices,
                            batch.faces,
                            batch.poses,
                            batch.areas,
                            batch.remap,
                            batch.offsets,
                            batch.copies,
                            batch.valid,
                            model.tri_materials,
                            model.edge_bending_properties,
                            model.edge_rest_angle,
                            model.particle_inv_mass,
                            model.particle_flags,
                            shared,
                            h,
                            self.candidate,
                        ],
                    )
                start = end
            self._launch(
                elastic.average_pair_copies,
                model.particle_count,
                [
                    len(batch.ids),
                    batch.offsets,
                    batch.copies,
                    model.particle_inv_mass,
                    model.particle_flags,
                    self.candidate,
                ],
            )

    def step(self, state_in, state_out, dt):
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("FEM step dt must be finite and positive")
        if self.global_constraints is not None:
            self.global_constraints.step(state_in, state_out, dt)
            return
        model = self.model
        wp.copy(self.pos, state_in.particle_q)
        wp.copy(self.velocity, state_in.particle_qd)
        if self.radius > 0.0:
            self._detect()
        else:
            wp.copy(self.base, self.pos)
        h = dt / self.iterations
        for iteration in range(self.iterations):
            wp.copy(self.prev, self.pos)
            self._launch(
                elastic.predict,
                model.particle_count,
                [
                    self.pos,
                    self.velocity,
                    state_in.particle_f,
                    model.particle_inv_mass,
                    model.particle_flags,
                    model.particle_world,
                    model.gravity,
                    self.targets,
                    self.iterations - iteration,
                    h,
                    self.candidate,
                ],
            )
            self._commit()
            # PhysX cloth accumulates external contact deltas on the prediction,
            # solves shell energy, then applies the external deltas. Keep DAT
            # boundaries around all three stages: fused trials failed safety
            # or complete-motion acceptance in the full twist scene.
            if self.radius > 0.0:
                self.delta.zero_()
                self.weights.zero_()
                self._launch(
                    collision.contact_response,
                    self.workers,
                    [
                        self.pos,
                        self.prev,
                        model.particle_inv_mass,
                        model.particle_flags,
                        self.count,
                        self.pairs,
                        self.kinds,
                        self.radius,
                        model.soft_contact_ke,
                        model.soft_contact_mu,
                        h,
                        self.workers,
                        self.delta,
                        self.weights,
                    ],
                )
            wp.copy(self.candidate, self.pos)
            self._solve_shell(h)
            self._commit()
            if self.radius > 0.0:
                self._launch(
                    collision.apply_contact_delta,
                    model.particle_count,
                    [self.pos, self.delta, self.weights, self.candidate],
                )
                self._commit()
            self._launch(elastic.update_velocity, model.particle_count, [self.prev, self.pos, h, self.velocity])
            self.delta.zero_()
            self._launch(
                elastic.membrane_damping,
                model.tri_count,
                [
                    model.tri_indices,
                    model.tri_materials,
                    self.pos,
                    self.velocity,
                    model.particle_inv_mass,
                    model.particle_flags,
                    h,
                    self.delta,
                ],
            )
            self._launch(
                elastic.bending_damping,
                model.edge_count,
                [
                    model.edge_indices,
                    model.edge_bending_properties,
                    self.pos,
                    self.velocity,
                    model.particle_inv_mass,
                    model.particle_flags,
                    h,
                    self.delta,
                ],
            )
            self._launch(elastic.apply_damping, model.particle_count, [self.delta, self.velocity])
        wp.copy(state_out.particle_q, self.pos)
        wp.copy(state_out.particle_qd, self.velocity)

    def validate(self):
        code = int(self.status.numpy()[0])
        if code:
            raise RuntimeError(
                f"XPBD FEM contact transaction rejected: status={code} (1 overflow, 2 nonseparated base, 4 nonfinite)"
            )
        if self.radius > 0.0:
            self._launch(
                collision.primitive_bounds,
                max(self.model.tri_count, len(self.edges)),
                [
                    self.pos,
                    self.model.tri_indices,
                    self.edges,
                    self.tri_lo,
                    self.tri_hi,
                    self.edge_lo,
                    self.edge_hi,
                ],
            )
            self.tri_bvh.refit()
            self.crossing_count.zero_()
            self._launch(
                collision.count_surface_crossings,
                len(self.edges),
                [
                    self.tri_bvh.id,
                    self.pos,
                    self.model.tri_indices,
                    self.edges,
                    self.model.particle_world,
                    self.crossing_count,
                    self.crossing_pairs,
                ],
            )
            crossings = int(self.crossing_count.numpy()[0])
            if crossings:
                raise RuntimeError(f"XPBD FEM surface audit found {crossings} segment-triangle crossings")

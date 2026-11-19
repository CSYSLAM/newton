# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host precomputation for affine bodies and their constraints (native ``precompute_bodies`` and
``precompute_affine_blocks``).

Bodies use the "inertial templates": a combined off-diagonal ELL pattern whose channels hold the particle material
Laplacian (L0), the non-diagonal masses (L1), the six-spring rigidity coupling (L2), the joint springs (L4), the joint
point-lines (L5) and the distance constraints (L6). Each step combines them with their weights. Host arithmetic is
float32 in the native operation and accumulation order.
"""

from __future__ import annotations

import numpy as np
import warp as wp

f32 = np.float32


class _Entry:
    __slots__ = ("a", "b", "l0", "l4", "l5", "l6")

    def __init__(self) -> None:
        self.a: list[int] = []
        self.b: list[int] = []
        self.l0 = f32(0.0)
        self.l4 = f32(0.0)
        self.l5 = f32(0.0)
        self.l6 = f32(0.0)


class _Rows:
    """``std::vector<std::map<int, Entry>>``: entries created on first access, traversed by ascending column."""

    def __init__(self, rows: int) -> None:
        self.rows: list[dict[int, _Entry]] = [{} for _ in range(rows)]

    def __call__(self, row: int, column: int) -> _Entry:
        entries = self.rows[row]
        entry = entries.get(column)
        if entry is None:
            entry = _Entry()
            entries[column] = entry
        return entry


def _weight(body: int, point: np.ndarray, com: np.ndarray, body_count: int, what: str) -> np.ndarray:
    """Host weights (1, p_local - com) of a body-attached point; zero for world ends."""
    if body < 0:
        return np.zeros(4, dtype=f32)
    if body >= body_count:
        raise ValueError(f"{what} references an invalid body")
    return np.array([1.0, point[0] - com[body][0], point[1] - com[body][1], point[2] - com[body][2]], dtype=f32)


class BodyPrecompute:
    """Mixin of :class:`~.solver.Solver` with the body/constraint host setup."""

    def _accumulate(
        self, rows: _Rows, ends: list[tuple[int, np.float32]], k: np.float32, diagonal: np.ndarray, member: str
    ) -> None:
        """Constant template k s s^T of one element: each slot pair a <= b evaluated once as (k s_a) s_b."""
        k = f32(k)
        for a in range(len(ends)):
            i, wi = ends[a]
            for b in range(a, len(ends)):
                j, wj = ends[b]
                v = f32(f32(k * wi) * wj)
                if i == j:
                    diagonal[i] = f32(diagonal[i] + v)
                    if a != b:
                        diagonal[i] = f32(diagonal[i] + v)
                else:
                    entry = rows(i, j)
                    setattr(entry, member, f32(getattr(entry, member) + v))
                    entry = rows(j, i)
                    setattr(entry, member, f32(getattr(entry, member) + v))

    @staticmethod
    def _body_ends(body: int, bodies: np.ndarray, weights: np.ndarray, sign: float) -> list[tuple[int, np.float32]]:
        return [(int(bodies[body][j]), f32(f32(sign) * weights[j])) for j in range(4)]

    def _precompute_affine_blocks(self) -> None:
        model = self.model
        self.affine_blocks = self._empty(wp.float32)
        self.affine_valid = self._empty(wp.int32)
        self.affine_enabled = self._empty(wp.int32)
        self.body_rotation = self._empty(wp.mat33)
        self.body_multiplier = self._empty(wp.mat33)
        self.affine_cable_offsets = self._upload(np.zeros(1, dtype=np.int64), wp.int64)
        self.affine_cable_ids = self._empty(wp.int32)
        if not self.body_count:
            return
        self._require(model.affine_body_indices, self.body_count, "affine_body_indices")
        if self.body_state_count != self.body_count * (self.count // self.per_world):
            raise ValueError("Body state count must contain whole replicated worlds")
        self.affine_blocks = self._zeros(self.body_state_count * 144, wp.float32)
        self.affine_valid = self._zeros(self.body_state_count, wp.int32)
        bodies = self._read(model.affine_body_indices, wp.vec4i, self.body_count)
        owner = np.full(self.per_world, -1, dtype=np.int64)
        enabled = np.ones(self.body_count, dtype=np.int32)
        for body in range(self.body_count):
            for value in bodies[body]:
                particle = int(value)
                if particle < 0 or particle >= self.per_world:
                    raise ValueError("affine body references an invalid particle index")
                if owner[particle] >= 0:
                    # Sharing even one proxy coordinate disables both overlapping twelve-DOF blocks.
                    enabled[body] = 0
                    enabled[owner[particle]] = 0
                else:
                    owner[particle] = body
        self.affine_enabled = self._upload(enabled, wp.int32)
        self._require(model.cable_body_indices, model.cable_count, "cable_body_indices")
        pairs = self._read(model.cable_body_indices, wp.vec2i, model.cable_count)
        adjacency: list[list[int]] = [[] for _ in range(self.body_count)]
        for cable, pair in enumerate(pairs):
            a, b = int(pair[0]), int(pair[1])
            for body in (a, b):
                if body < -1 or body >= self.body_count:
                    raise ValueError("cable references an invalid body")
            if a >= 0:
                adjacency[a].append(cable)
            if b >= 0 and b != a:
                adjacency[b].append(cable)
        offsets = np.zeros(self.body_count + 1, dtype=np.int64)
        ids: list[int] = []
        for body in range(self.body_count):
            ids.extend(adjacency[body])
            offsets[body + 1] = len(ids)
        self.affine_cable_offsets = self._upload(offsets, wp.int64)
        self.affine_cable_ids = self._upload(np.asarray(ids, dtype=np.int32), wp.int32)
        self.body_rotation = self._zeros(self.body_state_count, wp.mat33)
        self.body_multiplier = self._zeros(self.body_state_count, wp.mat33)

    def _precompute_bodies(self) -> None:
        model = self.model
        per_world = self.per_world
        copies = self.count // per_world
        self.body_count = model.body_count
        self.body_state_count = model.body_state_count if model.body_state_count else model.body_count * copies
        self.body_spring_total = self.joint_spring_total = self.plain_spring_count = 0
        self.drive_total = 0
        self.angular_limits = self.armature = False
        self.point_line_total = self.joint_line_total = self.plain_line_count = 0
        self.distance_total = 0
        empty_v4 = self._empty(wp.vec4)
        self.body_spring_w0 = self.body_spring_w1 = empty_v4
        self.point_line_wp = self.point_line_we0 = self.point_line_we1 = empty_v4
        self.distance_w0 = self.distance_w1 = empty_v4
        self.drive_w_po = self.drive_w_co = self.drive_w_pa = self.drive_w_pr = self.drive_w_cr = empty_v4
        if not self._inertial_templates():
            return
        self._require(model.affine_body_indices, self.body_count, "affine_body_indices")
        self._require(model.body_masses, self.body_count, "body_masses")
        if self.body_state_count != self.body_count * copies:
            raise ValueError("Body state count must contain whole replicated worlds")
        bodies = self._read(model.affine_body_indices, wp.vec4i, self.body_count).astype(np.int64)
        mass_count = model.particle_mass_non_diagonal_count
        if mass_count:
            self._require(model.particle_mass_non_diagonal_indices, mass_count, "particle_mass_non_diagonal_indices")
            self._require(model.particle_mass_non_diagonal_values, mass_count, "particle_mass_non_diagonal_values")
        pairs = self._read(model.particle_mass_non_diagonal_indices, wp.vec2i, mass_count).astype(np.int64)

        def valid(p: int) -> bool:
            return 0 <= p < per_world

        rows = _Rows(per_world)
        host_diagonal = np.zeros(per_world, dtype=f32)
        distance_diagonal = np.zeros(per_world, dtype=f32)
        joint_diagonal = np.zeros(per_world, dtype=f32)
        line_diagonal = np.zeros(per_world, dtype=f32)
        # Mixed templates put the particle material Laplacian in L0 and the host diagonal.
        if self.spring_rows is not None:
            row, column = self.spring_rows.row_columns()
            for r, c, value in zip(row.tolist(), column.tolist(), self.spring_rows.values, strict=True):
                rows(r, c).l0 = f32(value)
            host_diagonal = self.spring_diagonal.astype(f32).copy()
        diagonal_ids: list[list[int]] = [[] for _ in range(per_world)]
        kind = np.zeros(per_world, dtype=np.int32)
        for m, (a, b) in enumerate(pairs.tolist()):
            if not valid(a) or not valid(b) or a == b:
                raise ValueError("mass non-diagonal entry references an invalid particle pair")
            rows(a, b).a.append(m)
            rows(b, a).a.append(m)
        for body in range(self.body_count):
            ids = [int(v) for v in bodies[body]]
            for p in ids:
                if not valid(p):
                    raise ValueError("affine body references an invalid particle index")
            kind[ids[0]] = 1
            for c in range(1, 4):
                kind[ids[c]] = 2
                diagonal_ids[ids[c]].append(body)
                for d in range(1, 4):
                    if c != d:
                        rows(ids[c], ids[d]).b.append(body)

        def com() -> np.ndarray:
            self._require(model.body_centers_of_mass, self.body_count, "body_centers_of_mass")
            return self._read(model.body_centers_of_mass, wp.vec3, self.body_count)

        # Plain body springs go to the host constant (diagonal, L0); joint springs to D3/L4 (weighted alpha).
        if model.body_spring_count:
            n = model.body_spring_count
            if model.joint_body_spring_count > n:
                raise ValueError("joint_body_spring_count exceeds body_spring_count")
            self.plain_spring_count = n - model.joint_body_spring_count
            for field in (
                "body_spring_indices",
                "body_spring_points0",
                "body_spring_points1",
                "body_spring_stiffnesses",
                "body_spring_rest_lengths",
            ):
                self._require(getattr(model, field), n, field)
            centers = com()
            spring_bodies = self._read(model.body_spring_indices, wp.vec2i, n).astype(np.int64)
            points0 = self._read(model.body_spring_points0, wp.vec3, n)
            points1 = self._read(model.body_spring_points1, wp.vec3, n)
            stiffness = self._read(model.body_spring_stiffnesses, wp.float32, n)
            rest = self._read(model.body_spring_rest_lengths, wp.float32, n)
            for s in range(n):
                if not np.isfinite(rest[s]) or rest[s] < 0.0:
                    raise ValueError("body spring rest length must be finite and non-negative")
                if s >= self.plain_spring_count and rest[s] != 0.0:
                    raise ValueError("joint-generated body springs must have zero rest length")
            w0 = np.zeros((n, 4), dtype=f32)
            w1 = np.zeros((n, 4), dtype=f32)
            for s in range(n):
                a, b = int(spring_bodies[s][0]), int(spring_bodies[s][1])
                w0[s] = _weight(a, points0[s], centers, self.body_count, "body spring")
                w1[s] = _weight(b, points1[s], centers, self.body_count, "body spring")
                ends: list[tuple[int, np.float32]] = []
                if a >= 0:
                    ends += self._body_ends(a, bodies, w0[s], 1.0)
                if b >= 0:
                    ends += self._body_ends(b, bodies, w1[s], -1.0)
                joint = s >= self.plain_spring_count
                self._accumulate(
                    rows, ends, stiffness[s], joint_diagonal if joint else host_diagonal, "l4" if joint else "l0"
                )
            self.body_spring_total = n * copies
            self.joint_spring_total = model.joint_body_spring_count * copies
            self.body_spring_w0 = self._upload(w0, wp.vec4)
            self.body_spring_w1 = self._upload(w1, wp.vec4)

        if model.joint_drive_count:
            n, dofs = model.joint_drive_count, model.joint_dof_count
            for field in (
                "joint_drive_types",
                "joint_drive_body_indices",
                "joint_drive_parent_origins",
                "joint_drive_parent_axis_points",
                "joint_drive_parent_reference_points",
                "joint_drive_child_origins",
                "joint_drive_child_reference_points",
                "joint_drive_dof_indices",
                "joint_drive_target_indices",
            ):
                self._require(getattr(model, field), n, field)
            for field in (
                "joint_target_stiffnesses",
                "joint_target_dampings",
                "joint_effort_limits",
                "joint_target_modes",
                "joint_passive_dampings",
            ):
                self._require(getattr(model, field), dofs, field)
            if model.joint_armature.size:
                self._require(model.joint_armature, dofs, "joint_armature")
                self.armature = bool(np.any(self._read(model.joint_armature, wp.float32, dofs) > 0.0))
            centers = com()
            drive_bodies = self._read(model.joint_drive_body_indices, wp.vec2i, n).astype(np.int64)
            po = self._read(model.joint_drive_parent_origins, wp.vec3, n)
            pa = self._read(model.joint_drive_parent_axis_points, wp.vec3, n)
            pr = self._read(model.joint_drive_parent_reference_points, wp.vec3, n)
            co = self._read(model.joint_drive_child_origins, wp.vec3, n)
            cr = self._read(model.joint_drive_child_reference_points, wp.vec3, n)
            weights = {name: np.zeros((n, 4), dtype=f32) for name in ("po", "co", "pa", "pr", "cr")}
            for i in range(n):
                a, b = int(drive_bodies[i][0]), int(drive_bodies[i][1])
                weights["po"][i] = _weight(a, po[i], centers, self.body_count, "joint drive")
                weights["pa"][i] = _weight(a, pa[i], centers, self.body_count, "joint drive")
                weights["pr"][i] = _weight(a, pr[i], centers, self.body_count, "joint drive")
                weights["co"][i] = _weight(b, co[i], centers, self.body_count, "joint drive")
                weights["cr"][i] = _weight(b, cr[i], centers, self.body_count, "joint drive")
            self.drive_total = n * copies
            for name, values in weights.items():
                setattr(self, f"drive_w_{name}", self._upload(values, wp.vec4))
            self.angular_limits = bool(model.joint_limit_stiffnesses.size or model.joint_limit_dampings.size)
            if self.angular_limits:
                for field in (
                    "joint_limit_lower",
                    "joint_limit_upper",
                    "joint_limit_stiffnesses",
                    "joint_limit_dampings",
                ):
                    self._require(getattr(model, field), dofs, field)

        # Plain point-lines: -w_pt on the point body, then 0.5 (w_e0 + w_e1) on the line body, k s_i s_j in line
        # order after the springs. Joint lines go to D4/L5 with their ends sorted by particle id (map order).
        if model.body_point_line_count:
            n = model.body_point_line_count
            if model.joint_body_point_line_count > n:
                raise ValueError("joint_body_point_line_count exceeds body_point_line_count")
            self.plain_line_count = n - model.joint_body_point_line_count
            for field in (
                "body_point_line_indices",
                "body_point_line_points",
                "body_point_line_endpoints0",
                "body_point_line_endpoints1",
                "body_point_line_stiffnesses",
                "body_point_line_dampings",
            ):
                self._require(getattr(model, field), n, field)
            centers = com()
            line_bodies = self._read(model.body_point_line_indices, wp.vec2i, n).astype(np.int64)
            points = self._read(model.body_point_line_points, wp.vec3, n)
            end0 = self._read(model.body_point_line_endpoints0, wp.vec3, n)
            end1 = self._read(model.body_point_line_endpoints1, wp.vec3, n)
            stiffness = self._read(model.body_point_line_stiffnesses, wp.float32, n)
            wpt = np.zeros((n, 4), dtype=f32)
            we0 = np.zeros((n, 4), dtype=f32)
            we1 = np.zeros((n, 4), dtype=f32)
            for j in range(n):
                pb, lb = int(line_bodies[j][0]), int(line_bodies[j][1])
                wpt[j] = _weight(pb, points[j], centers, self.body_count, "point-line")
                we0[j] = _weight(lb, end0[j], centers, self.body_count, "point-line")
                we1[j] = _weight(lb, end1[j], centers, self.body_count, "point-line")
                ends = []
                if pb >= 0:
                    ends += [(int(bodies[pb][k]), f32(-wpt[j][k])) for k in range(4)]
                if lb >= 0:
                    ends += [(int(bodies[lb][k]), f32(f32(0.5) * f32(we0[j][k] + we1[j][k]))) for k in range(4)]
                if j < self.plain_line_count:
                    self._accumulate(rows, ends, stiffness[j], host_diagonal, "l0")
                else:
                    ends.sort(key=lambda end: end[0])
                    self._accumulate(rows, ends, stiffness[j], line_diagonal, "l5")
            self.point_line_total = n * copies
            self.joint_line_total = model.joint_body_point_line_count * copies
            self.point_line_wp = self._upload(wpt, wp.vec4)
            self.point_line_we0 = self._upload(we0, wp.vec4)
            self.point_line_we1 = self._upload(we1, wp.vec4)

        # Distance constraints: D6/L6 = m_s s_i s_j with s = (+w0, -w1), weighted constraint_alpha.
        if model.body_distance_constraint_count:
            n = model.body_distance_constraint_count
            for field in (
                "body_distance_constraint_indices",
                "body_distance_constraint_points0",
                "body_distance_constraint_points1",
                "body_distance_constraint_weights0",
                "body_distance_constraint_weights1",
                "body_distance_constraint_rest_lengths",
                "body_distance_constraint_mass_scales",
            ):
                self._require(getattr(model, field), n, field)
            constraint_bodies = self._read(model.body_distance_constraint_indices, wp.vec2i, n).astype(np.int64)
            w0 = self._read(model.body_distance_constraint_weights0, wp.vec4, n).astype(f32).copy()
            w1 = self._read(model.body_distance_constraint_weights1, wp.vec4, n).astype(f32).copy()
            scale = self._read(model.body_distance_constraint_mass_scales, wp.float32, n)
            for c in range(n):
                ends = []
                for side, weights in ((0, w0), (1, w1)):
                    b = int(constraint_bodies[c][side])
                    if b < 0:
                        weights[c] = 0.0
                        continue
                    if b >= self.body_count:
                        raise ValueError("distance constraint references an invalid body")
                    ends += self._body_ends(b, bodies, weights[c], -1.0 if side else 1.0)
                self._accumulate(rows, ends, scale[c], distance_diagonal, "l6")
            self.distance_total = n * copies
            self.distance_w0 = self._upload(w0, wp.vec4)
            self.distance_w1 = self._upload(w1, wp.vec4)

        counts_local = np.array([len(row) for row in rows.rows], dtype=np.int32)
        self.max_entries = int(counts_local.max(initial=0))
        entries = self.max_entries * per_world
        size = max(entries, 1)
        columns = np.full(size, -1, dtype=np.int32)
        channels = {name: np.zeros(size, dtype=f32) for name in ("l0", "l4", "l5", "l6")}
        lists_a: list[list[int]] = [[] for _ in range(entries)]
        lists_b: list[list[int]] = [[] for _ in range(entries)]
        for i in range(per_world):
            for slot, column in enumerate(sorted(rows.rows[i])):
                entry = rows.rows[i][column]
                e = slot * per_world + i
                columns[e] = column
                for name, values in channels.items():
                    values[e] = getattr(entry, name)
                lists_a[e] = entry.a
                lists_b[e] = entry.b
        offsets_a = np.zeros(entries + 1, dtype=np.int64)
        offsets_b = np.zeros(entries + 1, dtype=np.int64)
        ids_a: list[int] = []
        ids_b: list[int] = []
        for e in range(entries):
            ids_a.extend(lists_a[e])
            ids_b.extend(lists_b[e])
            offsets_a[e + 1] = len(ids_a)
            offsets_b[e + 1] = len(ids_b)
        diagonal_offsets = np.zeros(per_world + 1, dtype=np.int64)
        diagonal_list: list[int] = []
        for i in range(per_world):
            diagonal_list.extend(diagonal_ids[i])
            diagonal_offsets[i + 1] = len(diagonal_list)
        # Empty lists are never dereferenced; keep them non-empty so every kernel argument is valid.
        ids_a = ids_a or [0]
        ids_b = ids_b or [0]
        diagonal_list = diagonal_list or [0]
        body_mass = self._read(model.body_masses, wp.float32, self.body_count)
        mass_values = self._read(model.particle_mass_non_diagonal_values, wp.float32, mass_count)
        if not np.all(np.isfinite(mass_values)):
            raise ValueError("particle mass non-diagonal value must be finite")
        d1 = np.zeros(per_world, dtype=f32)
        d2 = np.zeros(per_world, dtype=f32)
        for i in range(per_world):
            total = f32(0.0)
            for body in diagonal_ids[i]:
                total = f32(total + body_mass[body])
            d1[i] = f32(total * f32(3.0))
            d2[i] = total
        l1 = np.zeros(size, dtype=f32)
        l2 = np.zeros(size, dtype=f32)
        for e in range(entries):
            for m in lists_a[e]:
                l1[e] = f32(l1[e] + mass_values[m])
            for body in lists_b[e]:
                l2[e] = f32(l2[e] - body_mass[body])

        upload = self._upload
        self.host_diagonal = upload(host_diagonal, wp.float32)
        self.body_d1 = upload(d1, wp.float32)
        self.body_d2 = upload(d2, wp.float32)
        self.joint_d3 = upload(joint_diagonal, wp.float32)
        self.line_d4 = upload(line_diagonal, wp.float32)
        self.distance_d5 = upload(distance_diagonal, wp.float32)
        counts = np.tile(counts_local, copies)
        self.local_row_count = upload(counts_local, wp.int32)
        self.row_count = upload(counts, wp.int32)
        self.scalar_template_columns = upload(columns, wp.int32)
        self.scalar_template_values = self._zeros(size, wp.float32)
        self.body_l0 = upload(channels["l0"], wp.float32)
        self.body_l1 = upload(l1, wp.float32)
        self.body_l2 = upload(l2, wp.float32)
        self.body_zero_ell = self._zeros(size, wp.float32)
        self.joint_l4 = upload(channels["l4"], wp.float32)
        self.line_l5 = upload(channels["l5"], wp.float32)
        self.distance_l6 = upload(channels["l6"], wp.float32)
        self.body_offsets_a = upload(offsets_a, wp.int64)
        self.body_ids_a = upload(np.asarray(ids_a, dtype=np.int32), wp.int32)
        self.body_offsets_b = upload(offsets_b, wp.int64)
        self.body_ids_b = upload(np.asarray(ids_b, dtype=np.int32), wp.int32)
        self.body_diagonal_offsets = upload(diagonal_offsets, wp.int64)
        self.body_diagonal_ids = upload(np.asarray(diagonal_list, dtype=np.int32), wp.int32)
        self.body_kind = upload(kind, wp.int32)
        # Without bodies every particle keeps kind 0; contact views read the particle-kind channel.
        if not self.body_count:
            self.particle_kind = self.body_kind
        n = self.max_entries * self.count
        self.block_template_columns = self._zeros(n, wp.int32)
        self.block_template_values = self._zeros(n, wp.mat33)
        self.block_entry_columns = self._zeros(n, wp.int32)
        self.block_entry_values = self._zeros(n, wp.mat33)

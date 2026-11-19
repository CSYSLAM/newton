# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host side of granular material (native ``precompute_granular`` and the accumulate/multiply/commit hooks)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from .kernels import granular as kernels


@dataclass
class _Bonds:
    """One bond group (authored cohesive or dynamic adhesive) and its stiffness."""

    count: int = 0
    pairs: Any = None
    rest: Any = None
    history: Any = None
    active: Any = None
    tangent: Any = None
    stiffness: float = 0.0


class GranularHost:
    """Mixin of :class:`~.solver.Solver` with the granular host logic."""

    def _precompute_granular(self) -> None:
        model = self.model
        self.granular_points = 0
        self.granular_initial = _Bonds()
        self.granular_dynamic = _Bonds()
        local, authored = model.granular_particle_count, model.granular_bond_count
        if not local:
            if authored:
                raise ValueError("granular bonds require granular particles")
            return
        values = (
            model.granular_friction,
            model.granular_friction_damping,
            model.granular_cohesion,
            model.granular_bond_damping,
            model.granular_yield_strain,
            model.granular_max_compaction,
            model.granular_break_strain,
            model.granular_adhesion,
            model.granular_bond_formation_gap,
            model.granular_bond_formation_speed,
        )
        valid = math.isfinite(model.granular_contact_stiffness) and model.granular_contact_stiffness > 0.0
        valid = valid and all(math.isfinite(value) and value >= 0.0 for value in values)
        if (
            not valid
            or model.granular_yield_strain >= 1.0
            or model.granular_max_compaction >= 1.0
            or not 1 <= model.granular_max_dynamic_bonds <= 64
        ):
            raise ValueError(
                "granular parameters must be finite with positive contact stiffness, "
                "non-negative friction/damping/cohesion/adhesion/formation gap/speed/break strain, "
                "yield/compaction in [0, 1), and dynamic bond capacity in [1, 64]"
            )
        if not model.granular_particle_indices.ptr or model.granular_particle_indices.size < local:
            raise ValueError("granular_particle_indices is smaller than its collision count")

        def enough(span, size):
            if size and (not span.ptr or span.size < size):
                raise ValueError("granular model array is smaller than its count")

        per_world = self.per_world
        enough(model.particle_radii, per_world)
        indices = self._read(model.granular_particle_indices, wp.int32, local)
        masses = self._read(model.particle_masses, wp.float32, per_world)
        radii = self._read(model.particle_radii, wp.float32, per_world)
        flags = self._read(model.particle_flags, wp.int32, per_world) if model.particle_flags.size else None
        worlds = (
            self._read(model.particle_world, wp.int32, per_world)
            if model.particle_world.size
            else np.zeros(per_world, dtype=np.int32)
        )
        inverse = np.full(per_world, -1, dtype=np.int64)
        maximum_radius = np.float32(0.0)
        for i, value in enumerate(indices):
            index = int(value)
            if index < 0 or index >= per_world:
                raise ValueError("granular particle index is out of range")
            if inverse[index] >= 0:
                raise ValueError("granular indices must be unique valid particle indices")
            inverse[index] = i
            if (
                not math.isfinite(masses[index])
                or masses[index] <= 0.0
                or not math.isfinite(radii[index])
                or radii[index] <= 0.0
                or (flags is not None and not flags[index] & 1)
            ):
                raise ValueError("granular particles must have positive finite mass/radius and be active")
            maximum_radius = max(maximum_radius, radii[index])
        for field, count_field, dtype in (
            ("triangle_indices", "triangle_count", wp.vec3i),
            ("edge_indices", "edge_count", wp.vec4i),
            ("spring_indices", "spring_count", wp.vec2i),
            ("tetrahedron_indices", "tetrahedron_count", wp.vec4i),
            ("affine_body_indices", "body_count", wp.vec4i),
        ):
            elements = self._read(getattr(model, field), dtype, getattr(model, count_field)).reshape(-1)
            in_range = elements[(elements >= 0) & (elements < per_world)]
            if np.any(inverse[in_range] >= 0):
                raise ValueError("granular particles cannot also belong to elastic or affine topology")
        enough(model.granular_bond_indices, authored)
        enough(model.granular_bond_rest_lengths, authored)
        pairs = self._read(model.granular_bond_indices, wp.vec2i, authored).reshape(-1, 2)
        rest = self._read(model.granular_bond_rest_lengths, wp.float32, authored)
        unique = set()
        for i in range(authored):
            a, b = int(pairs[i, 0]), int(pairs[i, 1])
            key = (min(a, b), max(a, b))
            if (
                a < 0
                or b < 0
                or a >= per_world
                or b >= per_world
                or a == b
                or inverse[a] < 0
                or inverse[b] < 0
                or key in unique
            ):
                raise ValueError("granular bonds must be unique pairs of distinct granular particles")
            unique.add(key)
            if not math.isfinite(rest[i]) or rest[i] <= 0.0:
                raise ValueError("granular bond rest lengths must be positive and finite")
            if worlds[a] != worlds[b]:
                raise ValueError("granular bond endpoints must belong to the same world")
        copies = self.count // per_world
        points = local * copies
        dynamic = points * model.granular_max_dynamic_bonds if model.granular_adhesion > 0.0 else 0
        if points > 0x1FFFFFFF or self.count > 0x7FFFFFFF or authored * copies > 0x7FFFFFFF or dynamic > 0x7FFFFFFF:
            raise ValueError("granular model exceeds supported index range")
        self.granular_points = points
        # float32 arithmetic, as natively.
        cell = np.float32(maximum_radius + maximum_radius)
        if dynamic:
            cell = np.float32(cell + np.float32(model.granular_bond_formation_gap))
        if not math.isfinite(cell):
            raise ValueError("granular radii exceed supported range")
        self.granular_cell_size = float(cell)
        slots = 1
        while slots < points:
            slots *= 2
        self.granular_slots = slots * 2
        z = self._zeros
        self.granular_head = z(self.granular_slots, wp.int32)
        self.granular_next = z(points, wp.int32)
        self.granular_ids = z(points, wp.int32)
        self.granular_world = z(points, wp.int32)
        self.granular_cells = z(points, wp.vec3i)
        self.granular_positions = z(points, wp.vec3)
        self.granular_previous = z(points, wp.vec3)
        self.granular_radius = z(points, wp.float32)
        self.granular_counts = z(points, wp.int32)
        self.granular_neighbours = z(points * 32, wp.int32)
        self.granular_cache = z(points * 32, wp.vec4)
        self.granular_diagonal = z(self.count, wp.mat33)

        def prepare(bonds, lengths, active, stiffness):
            group = _Bonds(count=len(bonds), stiffness=stiffness)
            if group.count:
                group.pairs = self._upload(np.asarray(bonds, dtype=np.int32).reshape(-1, 2), wp.vec2i)
                group.rest = self._upload(np.asarray(lengths, dtype=np.float32), wp.float32)
                group.history = self._upload(np.asarray(lengths, dtype=np.float32), wp.float32)
                group.active = self._upload(np.full(group.count, active, dtype=np.int32), wp.int32)
                group.tangent = z(group.count, wp.mat33)
            return group

        global_pairs, global_rest = [], []
        adjacency = [[] for _ in range(points)]
        for world in range(copies):
            for i in range(authored):
                a, b = int(pairs[i, 0]), int(pairs[i, 1])
                if model.granular_cohesion > 0.0:
                    global_pairs.append((a + world * per_world, b + world * per_world))
                    global_rest.append(rest[i])
                # Every authored pair is excluded from adhesion, stored once in the smaller grain's row.
                ga = int(inverse[a]) + world * local
                gb = int(inverse[b]) + world * local
                adjacency[min(ga, gb)].append(max(ga, gb))
        self.granular_initial = prepare(global_pairs, global_rest, 1, model.granular_cohesion)
        if dynamic:
            self.granular_dynamic = prepare(np.zeros((dynamic, 2)), np.zeros(dynamic), 0, model.granular_adhesion)
            self.granular_partner = self._upload(np.full(dynamic, -1, dtype=np.int32), wp.int32)
            offsets, columns = [0], []
            for row in adjacency:
                columns.extend(sorted(row))
                offsets.append(len(columns))
            self.granular_offsets = self._upload(np.asarray(offsets, dtype=np.int32), wp.int32)
            self.granular_columns = (
                self._upload(np.asarray(columns, dtype=np.int32), wp.int32) if columns else self._empty(wp.int32)
            )

    def _granular_model(self) -> Any:
        model = self.model
        m = kernels.GranularModel()
        m.particles_per_world = model.particle_count
        m.state_count = model.particle_state_count
        m.radius = self._view(model.particle_radii, wp.float32, self.per_world)
        m.particle_world = self._view(model.particle_world, wp.int32)
        m.granular_count = model.granular_particle_count
        m.granular_indices = self._view(model.granular_particle_indices, wp.int32, model.granular_particle_count)
        m.bond_damping = model.granular_bond_damping
        m.bond_yield_strain = model.granular_yield_strain
        m.bond_max_compaction = model.granular_max_compaction
        m.bond_break_strain = model.granular_break_strain
        m.formation_gap = model.granular_bond_formation_gap
        m.formation_speed = model.granular_bond_formation_speed
        m.max_dynamic_bonds = model.granular_max_dynamic_bonds
        return m

    def _granular_view(self, dt: float) -> Any:
        model = self.model
        v = kernels.GranularView()
        v.head, v.next, v.ids, v.world = self.granular_head, self.granular_next, self.granular_ids, self.granular_world
        v.cells, v.positions, v.previous = self.granular_cells, self.granular_positions, self.granular_previous
        v.radius, v.counts, v.neighbours, v.cache = (
            self.granular_radius,
            self.granular_counts,
            self.granular_neighbours,
            self.granular_cache,
        )
        v.count, v.slots, v.cell_size = self.granular_points, self.granular_slots, self.granular_cell_size
        v.stiffness = model.granular_contact_stiffness
        v.friction = model.granular_friction
        v.friction_damping = model.granular_friction_damping
        v.dt = dt
        return v

    def _build_granular_grid(self, x: wp.array, dt: float, reset_diagonal: bool = False) -> None:
        self.granular_head.fill_(-1)
        if reset_diagonal:
            self.granular_diagonal.zero_()
        self._launch(
            "granular_10build_grid",
            kernels.build_grid,
            self.granular_points,
            [
                self._granular_model(),
                x,
                self.previous,
                self._granular_view(dt),
                self.granular_positions,
                self.granular_previous,
                self.granular_radius,
                self.granular_ids,
                self.granular_world,
                self.granular_cells,
                self.granular_head,
                self.granular_next,
            ],
        )

    def _granular_groups(self) -> tuple[_Bonds, ...]:
        return tuple(group for group in (self.granular_initial, self.granular_dynamic) if group.count)

    def _accumulate_granular(self, x: wp.array, dt: float) -> None:
        self._build_granular_grid(x, dt, True)
        self._launch(
            "granular_19accumulate_contacts",
            kernels.accumulate_contacts,
            self.granular_points,
            [self._granular_view(dt), self.rhs, self.granular_diagonal],
        )
        for group in self._granular_groups():
            self._launch(
                "granular_16accumulate_bonds",
                kernels.accumulate_bonds,
                group.count,
                [
                    self._granular_model(),
                    x,
                    self.previous,
                    dt,
                    group.pairs,
                    group.rest,
                    group.history,
                    group.active,
                    group.stiffness,
                    self.rhs,
                    self.granular_diagonal,
                    group.tangent,
                ],
            )

    def _multiply_granular(self, dt: float, direction: wp.array, product: wp.array) -> None:
        self._launch(
            "granular_17multiply_contacts",
            kernels.multiply_contacts,
            self.granular_points,
            [self._granular_view(dt), direction, product],
        )
        for group in self._granular_groups():
            self._launch(
                "granular_14multiply_bonds",
                kernels.multiply_bonds,
                group.count,
                [group.pairs, group.active, group.tangent, direction, product],
            )

    def _commit_granular(self, x: wp.array, dt: float) -> None:
        for group in self._granular_groups():
            self._launch(
                "granular_12commit_bonds",
                kernels.commit_bonds,
                group.count,
                [self._granular_model(), x, group.pairs, group.rest, group.history, group.active],
            )
        group = self.granular_dynamic
        if group.count:
            self._build_granular_grid(x, dt)
            self._launch(
                "granular_10form_bonds",
                kernels.form_bonds,
                self.granular_points,
                [
                    self._granular_model(),
                    self._granular_view(dt),
                    self.granular_offsets,
                    self.granular_columns,
                    self.granular_partner,
                    group.pairs,
                    group.rest,
                    group.history,
                    group.active,
                ],
            )

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host side of the particle fluid (native ``precompute_fluid``, ``accumulate_fluid`` and ``limit_fluid``)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import warp as wp

from .kernels import fluid as kernels


class FluidHost:
    """Mixin of :class:`~.solver.Solver` with the fluid host logic."""

    def _precompute_fluid(self) -> None:
        model = self.model
        self.fluid_rows = 0
        if not model.fluid_particle_count:
            return
        local, boundaries = model.fluid_particle_count, model.fluid_boundary_count
        copies = self.count // self.per_world
        for value, positive in (
            (model.fluid_rest_density, True),
            (model.fluid_smoothing_radius, True),
            (model.fluid_bulk_modulus, False),
            (model.fluid_viscosity, False),
        ):
            if not math.isfinite(value) or (value <= 0.0 if positive else value < 0.0):
                raise ValueError(
                    "fluid density/radius must be positive; bulk modulus/viscosity non-negative and finite"
                )
        self._require(model.fluid_particle_indices, local, "fluid_particle_indices")
        indices = self._read(model.fluid_particle_indices, wp.int32, local)
        masses = self._read(model.particle_masses, wp.float32, self.per_world)
        flags = self._read(model.particle_flags, wp.int32, self.per_world) if model.particle_flags.size else None
        selected = np.zeros(self.per_world, dtype=bool)
        for value in indices:
            index = int(value)
            if index < 0 or index >= self.per_world:
                raise ValueError("fluid particle index is out of range")
            if selected[index]:
                raise ValueError("fluid indices must be unique valid particle indices")
            selected[index] = True
            if not math.isfinite(masses[index]) or masses[index] <= 0.0 or (flags is not None and not flags[index] & 1):
                raise ValueError("fluid particles must have positive finite mass and be active")
        for field, count_field, dtype in (
            ("triangle_indices", "triangle_count", wp.vec3i),
            ("edge_indices", "edge_count", wp.vec4i),
            ("spring_indices", "spring_count", wp.vec2i),
            ("tetrahedron_indices", "tetrahedron_count", wp.vec4i),
            ("affine_body_indices", "body_count", wp.vec4i),
        ):
            elements = self._read(getattr(model, field), dtype, getattr(model, count_field)).reshape(-1)
            valid = elements[(elements >= 0) & (elements < self.per_world)]
            if np.any(selected[valid]):
                raise ValueError("fluid particles cannot also belong to elastic or affine topology")
        self._require(model.fluid_boundary_positions, boundaries, "fluid_boundary_positions")
        self._require(model.fluid_boundary_volumes, boundaries, "fluid_boundary_volumes")
        positions = self._read(model.fluid_boundary_positions, wp.vec3, boundaries)
        volumes = self._read(model.fluid_boundary_volumes, wp.float32, boundaries)
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
            raise ValueError("fluid boundary positions must be finite and volumes positive and finite")
        if local + boundaries > 0x1FFFFFFF // copies or self.count > 0x7FFFFFFF:
            raise ValueError("fluid grid is too large")
        self.fluid_stride = local + boundaries
        self.fluid_points = (local + boundaries) * copies
        self.fluid_rows = local * copies
        slots = 1
        while slots < self.fluid_points:
            slots *= 2
        self.fluid_slots = slots * 2
        self.fluid_worlds = copies
        if copies == 1 and model.particle_world.size:
            worlds = self._read(model.particle_world, wp.int32, model.particle_world.size)
            if np.any(worlds >= 0x1FFFFFFF):
                raise ValueError("fluid world index is too large")
            self.fluid_worlds = max(self.fluid_worlds, int(worlds.max(initial=0)) + 1)
        z = self._zeros
        self.fluid_head = z(self.fluid_slots, wp.int32)
        self.fluid_next = z(self.fluid_points, wp.int32)
        self.fluid_ids = z(self.fluid_points, wp.int32)
        self.fluid_world = z(self.fluid_points, wp.int32)
        self.fluid_cells = z(self.fluid_points, wp.vec3i)
        self.fluid_positions = z(self.fluid_points, wp.vec3)
        self.fluid_volumes = z(self.fluid_points, wp.float32)
        self.fluid_counts = z(self.fluid_rows, wp.int32)
        self.fluid_neighbours = z(self.fluid_rows * 64, wp.int32)
        self.fluid_cache = z(self.fluid_rows * 64, wp.vec4)
        self.fluid_gradients = z(self.fluid_rows, wp.vec3)
        self.fluid_residuals = z(self.fluid_rows, wp.float32)
        self.fluid_weights = z(self.fluid_rows, wp.float32)
        self.fluid_diagonal = z(self.count, wp.mat33)
        self.fluid_scale = z(self.fluid_worlds, wp.float32)

    def _fluid_model(self) -> Any:
        model = self.model
        m = kernels.FluidModel()
        m.particles_per_world = model.particle_count
        m.state_count = model.particle_state_count
        m.mass = self._view(model.particle_masses, wp.float32, self.per_world)
        m.particle_world = self._view(model.particle_world, wp.int32)
        m.fluid_count = model.fluid_particle_count
        m.fluid_indices = self._view(model.fluid_particle_indices, wp.int32, model.fluid_particle_count)
        m.rest_density = model.fluid_rest_density
        m.smoothing_radius = model.fluid_smoothing_radius
        m.boundary_count = model.fluid_boundary_count
        m.boundary_positions = self._view(model.fluid_boundary_positions, wp.vec3, model.fluid_boundary_count)
        m.boundary_volumes = self._view(model.fluid_boundary_volumes, wp.float32, model.fluid_boundary_count)
        return m

    def _fluid_view(self, dt: float) -> Any:
        model = self.model
        v = kernels.FluidView()
        v.head, v.next, v.ids, v.world = self.fluid_head, self.fluid_next, self.fluid_ids, self.fluid_world
        v.cells, v.positions, v.volumes = self.fluid_cells, self.fluid_positions, self.fluid_volumes
        v.counts, v.neighbours, v.cache = self.fluid_counts, self.fluid_neighbours, self.fluid_cache
        v.slots, v.stride, v.fluid_count, v.rows = (
            self.fluid_slots,
            self.fluid_stride,
            model.fluid_particle_count,
            self.fluid_rows,
        )
        v.smoothing_radius = model.fluid_smoothing_radius
        v.bulk_modulus = model.fluid_bulk_modulus
        v.viscosity = model.fluid_viscosity
        v.dt = dt
        return v

    def _accumulate_fluid(self, x: wp.array, dt: float) -> None:
        self.fluid_head.fill_(-1)
        self.fluid_diagonal.zero_()
        self._launch(
            "fluid_10build_grid",
            kernels.build_grid,
            self.fluid_points,
            [
                self._fluid_model(),
                x,
                self.fluid_positions,
                self.fluid_volumes,
                self.fluid_ids,
                self.fluid_world,
                self.fluid_cells,
                self.fluid_head,
                self.fluid_next,
                self.fluid_slots,
            ],
        )
        view = self._fluid_view(dt)
        self._launch(
            "fluid_13evaluate_rows",
            kernels.evaluate_rows,
            self.fluid_rows,
            [
                view,
                self.fluid_gradients,
                self.fluid_residuals,
                self.fluid_weights,
            ],
        )
        self._launch(
            "fluid_15accumulate_rows",
            kernels.accumulate_rows,
            self.fluid_rows,
            [
                view,
                self.fluid_gradients,
                self.fluid_residuals,
                self.fluid_weights,
                self.previous,
                self.rhs,
                self.fluid_diagonal,
            ],
        )

    def _limit_fluid(self, dt: float) -> None:
        config = self._config
        self._launch("fluid_16initialize_scale", kernels.initialize_scale, self.fluid_worlds, [self.fluid_scale, 1.0])
        self._launch(
            "fluid_11bound_scale",
            kernels.bound_scale,
            self.fluid_rows,
            [
                self._fluid_view(dt),
                self.solution,
                config.fluid_relaxation,
                config.fluid_max_step_ratio * self.model.fluid_smoothing_radius,
                self.fluid_scale,
            ],
        )
        self._launch(
            "fluid_10scale_step",
            kernels.scale_step,
            self.count,
            [
                self._fluid_model(),
                self.solution,
                self.fluid_scale,
            ],
        )

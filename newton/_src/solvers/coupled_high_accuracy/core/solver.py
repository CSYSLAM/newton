# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp implementation of the native ``Solver`` (``native/source_solver.cpp``).

The host logic mirrors the native class method for method: ``initialize`` validates the model and precomputes the
matrix templates and caches; ``step`` runs the implicit Newton iterations with a block-Jacobi preconditioned CG
solve. Buffer names follow the native members without the trailing underscore. Features that have not been ported
yet are rejected in ``initialize`` with ``NotImplementedError`` rather than silently skipped.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from contextlib import nullcontext
from typing import Any, ClassVar

import numpy as np
import warp as wp

from .api import (
    AffineBodyRigidityModel,
    Config,
    Contact,
    Control,
    JointConstraintMode,
    Model,
    NonlinearSolver,
    Span,
    State,
)
from .bodies import BodyPrecompute
from .cables import CableHost
from .fluid import FluidHost
from .granular import GranularHost
from .kernels import affine as affine_kernels
from .kernels import affine_block as affine_block_kernels
from .kernels import cables as cable_kernels
from .kernels import cloth as cloth_kernels
from .kernels import drag as drag_kernels
from .kernels import fluid as fluid_kernels
from .kernels import granular as granular_kernels
from .kernels import hard_contacts as hard_contact_kernels
from .kernels import joints as joint_kernels
from .kernels import pcg as pcg_kernels
from .kernels import rigid_contacts as rigid_contact_kernels
from .kernels import rolling_contacts as rolling_contact_kernels
from .kernels import self_collision as self_collision_kernels
from .kernels import soft_contacts as soft_contact_kernels
from .kernels import springs as spring_kernels
from .kernels import step as step_kernels
from .kernels import surface_contacts as surface_contact_kernels
from .kernels import tetrahedra as tetrahedron_kernels
from .self_collision import SelfCollisionHost
from .templates import SparseRows

__all__ = ["Solver"]

# Maximum PCG iteration count the per-iteration numerator/denominator history is sized for, as natively.
_MAX_LINEAR_ITERATIONS = 999

_NUMPY_TYPES = {
    wp.float32: (np.float32, ()),
    wp.int32: (np.int32, ()),
    wp.uint32: (np.uint32, ()),
    wp.int64: (np.int64, ()),
    wp.vec2i: (np.int32, (2,)),
    wp.vec3i: (np.int32, (3,)),
    wp.vec4i: (np.int32, (4,)),
    wp.vec2: (np.float32, (2,)),
    wp.vec3: (np.float32, (3,)),
    wp.vec4: (np.float32, (4,)),
    wp.mat22: (np.float32, (2, 2)),
    wp.mat33: (np.float32, (3, 3)),
    wp.transform: (np.float32, (7,)),
}


_GRAPH_ENABLED = os.environ.get("NEWTON_CHA_GRAPH", "1") != "0"


def _resolve_stream(stream: Any, device: wp.context.Device) -> wp.Stream | None:
    if stream is None:
        return None
    if isinstance(stream, wp.Stream):
        return stream
    if isinstance(stream, int):
        return wp.Stream(device, cuda_stream=stream) if stream else None
    if hasattr(stream, "__cuda_stream__"):
        protocol = stream.__cuda_stream__()
        if len(protocol) != 2 or protocol[0] != 0:
            raise ValueError("Invalid CUDA Stream Protocol result")
        return wp.Stream(device, cuda_stream=protocol[1]) if protocol[1] else None
    raise TypeError("Expected a CUDA stream integer, CUDA Stream Protocol object, or None")


class Solver(BodyPrecompute, SelfCollisionHost, CableHost, FluidHost, GranularHost):
    """Implicit Newton solver for particles, cloth, soft bodies and affine rigid bodies."""

    def __init__(self) -> None:
        self._config = Config()
        self._initialized = False
        self._launches: Counter[str] = Counter()
        self._graphs: dict[tuple, tuple] = {}
        self._graph_seen: set[tuple] = set()
        self._views: dict[tuple[int, int, Any], wp.array] = {}
        self._null: dict[Any, wp.array] = {}
        self.device: wp.context.Device | None = None
        self.count = 0

    # ------------------------------------------------------------------------------------------------------------
    # Public API (matches the native bindings)

    @property
    def config(self) -> Config:
        return self._config

    @config.setter
    def config(self, value: Config) -> None:
        if not isinstance(value, Config):
            raise TypeError("config requires a Config")
        self._config = value._copy()

    @property
    def source_launch_counts(self) -> dict[str, int]:
        return dict(self._launches)

    def reserve_rigid_contact_capacity(self, capacity: int) -> None:
        """Grow the hard-contact history, keeping existing entries and zeroing the new tail."""
        if not self._initialized or capacity <= self.hard_capacity:
            return
        for name, dtype in self._HARD_BUFFERS:
            old = getattr(self, name)
            new = self._zeros(capacity, dtype)
            if self.hard_capacity:
                wp.copy(new, old, count=self.hard_capacity)
            setattr(self, name, new)
        self.hard_capacity = capacity

    _HARD_BUFFERS = (
        ("hard_penalty", wp.float32),
        ("hard_dual", wp.vec3),
        ("hard_anchor", wp.vec3),
        ("hard_previous_penalty", wp.float32),
        ("hard_previous_dual", wp.vec3),
        ("hard_previous_normal", wp.vec3),
    )

    def reset_joint_drive_state(self, stream: Any = None, mask: Span = Span()) -> None:  # noqa: B008
        # External affine-state edits invalidate hard-contact history even without joint drives.
        drive_total = self.drive_total if self._initialized else 0
        if mask.size and mask.size != drive_total:
            raise ValueError("joint drive reset mask must match the drive state count")
        if not self._initialized or not self.count:
            return
        with self._stream_scope(stream):
            self.hard_previous_count.zero_()
            if not self.drive_total:
                return
            if not mask.size:
                self.drive_initialized.zero_()
            else:
                self._require(mask, self.drive_total, "mask")
                self._launch(
                    "drive_reset",
                    joint_kernels.reset_selected_joint_drive_state,
                    self.drive_total,
                    [self._view(mask, wp.int32, self.drive_total), self.drive_initialized],
                )

    def step(self, state_in: State, state_out: State, *args: Any, **kwargs: Any) -> None:
        """``step(in, out, dt[, stream])``, ``step(in, out, contact, dt[, stream])`` or
        ``step(in, out, control, contact, dt[, stream])``, as the native overloads."""
        control, contact, dt, stream = self._parse_step_arguments(args, kwargs)
        self._step(state_in, state_out, control, contact, dt, stream)

    # ------------------------------------------------------------------------------------------------------------
    # Memory helpers

    def _empty(self, dtype: Any) -> wp.array:
        array = self._null.get(dtype)
        if array is None:
            array = wp.empty(0, dtype=dtype, device=self.device)
            self._null[dtype] = array
        return array

    def _view(self, span: Span, dtype: Any, count: int | None = None) -> wp.array:
        """Zero-copy Warp array over a native span (an empty array for a null span)."""
        n = span.size if count is None else count
        if not span.ptr or not n:
            return self._empty(dtype)
        key = (span.ptr, n, dtype)
        array = self._views.get(key)
        if array is None:
            array = wp.array(ptr=span.ptr, dtype=dtype, shape=(n,), device=self.device)
            self._views[key] = array
        return array

    def _read(self, span: Span, dtype: Any, count: int) -> np.ndarray:
        """Host copy of a span, served from the constructor snapshot when one was captured."""
        scalar, inner = _NUMPY_TYPES[dtype]
        if not count:
            return np.zeros((0, *inner), dtype=scalar)
        snapshot = self._constructor_reads.get((span.ptr, count, dtype))
        if snapshot is not None:
            return snapshot.copy()
        return self._view(span, dtype, count).numpy().reshape((count, *inner)).astype(scalar, copy=False)

    def _capture(self, span: Span, count: int, dtype: Any, name: str) -> None:
        """Snapshot a span at construction time; later reads in the same initialize see this copy."""
        if not count:
            return
        self._require(span, count, name)
        self._constructor_reads.pop((span.ptr, count, dtype), None)
        self._constructor_reads[(span.ptr, count, dtype)] = self._read(span, dtype, count)

    def _upload(self, values: Any, dtype: Any) -> wp.array:
        values = np.asarray(values)
        if not values.size:
            return self._empty(dtype)
        return wp.array(values, dtype=dtype, device=self.device)

    def _zeros(self, count: int, dtype: Any) -> wp.array:
        if not count:
            return self._empty(dtype)
        return wp.zeros(count, dtype=dtype, device=self.device)

    @staticmethod
    def _require(span: Span, count: int, field: str) -> None:
        if count and (not span.ptr or span.size < count):
            raise ValueError(f"{field} span is missing or too small")

    def _launch(self, key: str, kernel: Any, dim: int, inputs: list[Any]) -> None:
        if dim <= 0:
            return
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)
        self._launches[key] += 1

    # ------------------------------------------------------------------------------------------------------------
    # Initialization

    def _reject_model(self, model: Model) -> None:
        has_bodies = model.body_count or model.body_state_count
        if (
            (model.cable_count and not has_bodies)
            or (model.body_spring_count and not has_bodies)
            or (model.body_distance_constraint_count and not has_bodies)
            or (model.body_point_line_count and not has_bodies)
            or (model.joint_drive_count and not has_bodies)
            or (model.joint_body_spring_count and not has_bodies)
            or (model.joint_body_point_line_count and not has_bodies)
        ):
            raise RuntimeError("Requested source model topology has not been implemented")

    def initialize(self, model: Model) -> None:
        if not isinstance(model, Model):
            raise TypeError("initialize requires a Model")
        if model.particle_state_count and not model.particle_count:
            raise ValueError("particle_state_count requires a nonzero particle_count")
        if model.body_state_count and not model.body_count:
            raise ValueError("body_state_count requires a nonzero body_count")
        if model.cable_count:
            for span in (
                model.cable_body_indices,
                model.cable_parent_transforms,
                model.cable_child_transforms,
                model.cable_stiffnesses,
                model.cable_dampings,
                model.cable_rest_bend_twist,
            ):
                if not span.ptr or span.size < model.cable_count:
                    raise ValueError("cable metadata is smaller than cable_count")
        self._reject_model(model)
        count = model.particle_state_count if model.particle_state_count else model.particle_count
        if count and (not model.particle_count or count % model.particle_count):
            raise ValueError("Particle state count must contain whole replicated worlds")
        self._require(model.particle_masses, model.particle_count, "particle_masses")
        if model.particle_rest_positions.size:
            self._require(model.particle_rest_positions, count, "particle_rest_positions")
        if model.particle_flags.size and model.particle_flags.size < model.particle_count:
            raise ValueError("particle_flags is smaller than particle_count")
        if model.particle_world.size and model.particle_world.size < model.particle_count:
            raise ValueError("particle_world is smaller than particle_count")
        if model.gravity.size:
            self._require(model.gravity, model.gravity.size, "gravity")
        for count_field, fields in (
            ("spring_count", ("spring_indices", "spring_stiffnesses", "spring_rest_lengths")),
            (
                "triangle_count",
                ("triangle_indices", "triangle_inv_dms", "triangle_rest_areas", "triangle_aniso_stiffnesses"),
            ),
            ("edge_count", ("edge_indices", "edge_rest_areas", "edge_bending_cots", "edge_bending_properties")),
            ("tetrahedron_count", ("tetrahedron_indices", "tetrahedron_inverse_dms", "tetrahedron_materials")),
        ):
            n = getattr(model, count_field)
            if n:
                for field in fields:
                    self._require(getattr(model, field), n, field)
        if model.shape_count:
            if model.shape_replicated_count > model.shape_count:
                raise ValueError("shape_replicated_count exceeds shape_count")
            self._require(model.particle_radii, model.particle_count, "particle_radii")
            for field in (
                "shape_bodies",
                "shape_types",
                "shape_transforms",
                "shape_scales",
                "shape_margins",
                "shape_contact_stiffnesses",
                "shape_contact_dampings",
                "shape_contact_friction_gains",
                "shape_frictions",
            ):
                self._require(getattr(model, field), model.shape_count, field)

        self.device = wp.get_device() if wp.get_device().is_cuda else wp.get_cuda_device()
        self._views.clear()
        self._null.clear()
        self._launches.clear()
        self._graphs.clear()
        self._graph_seen.clear()
        self._constructor_reads: dict[tuple[int, int, Any], np.ndarray] = {}
        self._initialized = False
        self.model = model._copy()
        self.count = count
        self.per_world = model.particle_count
        self.surface_tangent_cache = wp.zeros(0, dtype=surface_contact_kernels.SurfaceTangent, device=self.device)
        self.rigid_tangent_capacity = 0
        self.hard_capacity = 0
        for name, dtype in self._HARD_BUFFERS:
            setattr(self, name, self._empty(dtype))
        self.drive_total = 0
        self.cable_count = 0
        self.fluid_rows = 0
        self.granular_points = 0
        if not count:
            self._initialized = True
            return

        masses = self._read(model.particle_masses, wp.float32, self.per_world)
        if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
            raise ValueError("particle masses must be finite and non-negative")
        if model.gravity.size:
            if not np.all(np.isfinite(self._read(model.gravity, wp.vec3, model.gravity.size))):
                raise ValueError("gravity values must be finite")
        if model.particle_world.size:
            if np.any(self._read(model.particle_world, wp.int32, self.per_world) < -1):
                raise ValueError("particle_world contains an invalid world index")

        if model.shape_count:
            n = model.shape_count
            if np.any(self._read(model.shape_bodies, wp.int32, n) < -1):
                raise ValueError("shape_bodies contains an invalid affine body id")
            types = self._read(model.shape_types, wp.int32, n)
            transforms = self._read(model.shape_transforms, wp.transform, n)
            scales = self._read(model.shape_scales, wp.vec3, n)
            for i in np.flatnonzero(types == 3):
                if not np.all(np.isfinite(transforms[i][:3])) or not math.isfinite(scales[i][0]) or scales[i][0] <= 0.0:
                    raise ValueError("sphere centers must be finite and sphere radii must be finite and positive")
            for field in (
                "shape_contact_stiffnesses",
                "shape_contact_dampings",
                "shape_frictions",
                "shape_contact_friction_gains",
                "shape_margins",
            ):
                values = self._read(getattr(model, field), wp.float32, n)
                if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                    raise ValueError("shape contact materials and margins must be finite and non-negative")
            for field in ("shape_rolling_frictions", "shape_torsional_frictions"):
                span = getattr(model, field)
                if span.size:
                    values = self._read(span, wp.float32, n)
                    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                        raise ValueError(f"{field} must be finite and non-negative")
        # Radii are only checked when shapes or collision particles use them.
        if model.shape_count or model.collision_particle_count:
            self._require(model.particle_radii, self.per_world, "particle_radii")
            radii = self._read(model.particle_radii, wp.float32, self.per_world)
            if np.any(np.isnan(radii)) or np.any(radii < 0.0):
                raise ValueError("particle radii must be non-negative and not NaN")

        # Body-owned shapes only matter for contacts, which need affine bodies.
        if model.shape_count and not (model.body_count or model.body_state_count):
            if np.any(self._read(model.shape_bodies, wp.int32, model.shape_count) >= 0):
                raise RuntimeError("Source shape host currently supports static shapes only")
        self.solution = self._zeros(count, wp.vec3)
        self.rhs = self._zeros(count, wp.vec3)
        self.previous = self._zeros(count, wp.vec3)
        self.target = self._zeros(count, wp.vec3)
        self.hard_previous_count = self._zeros(1, wp.int32)
        self.body_count = model.body_count
        self._precompute_cables()
        self._precompute_drag()
        copies = count // self.per_world
        empty_f, empty_i = self._empty(wp.float32), self._empty(wp.int32)
        self.drive_angle, self.drive_initialized, self.drive_qd = empty_f, empty_i, empty_f
        if model.joint_drive_count:
            states = model.joint_drive_count * copies
            self.drive_angle = self._zeros(states, wp.float32)
            self.drive_initialized = self._zeros(states, wp.int32)
            self.drive_qd = self._zeros(states, wp.float32)
        self.base_diagonal = self._zeros(count, wp.float32)
        self.diagonal = self._zeros(count, wp.float32)
        self.collision_stiffness = self._zeros(count, wp.float32)
        self.scalar_inverse = self._zeros(count, wp.float32)
        self.blocks = self._zeros(count, wp.mat33)
        self.preconditioner_blocks = self._zeros(count, wp.mat33)
        # The native solver inverts the preconditioner blocks in place.
        self.inverse = self.preconditioner_blocks
        self.particle_kind = self._empty(wp.int32)
        self.body_kind = self._empty(wp.int32)
        self.body_state_count = model.body_state_count if model.body_state_count else self.body_count * copies
        empty_v, empty_m = self._empty(wp.vec3), self._empty(wp.mat33)
        self.joint_multiplier = self._zeros(model.joint_body_spring_count * copies, wp.vec3)
        self.distance_target = self._zeros(model.body_distance_constraint_count * copies, wp.vec3)
        self.distance_multiplier = self._zeros(model.body_distance_constraint_count * copies, wp.vec3)
        line_states = model.joint_body_point_line_count * copies
        self.line_segment = self._zeros(line_states, wp.float32)
        self.line_projector = self._zeros(line_states, wp.mat33)
        self.line_multiplier = self._zeros(line_states, wp.vec3)
        del empty_v, empty_m
        self._precompute_affine_blocks()
        self.rolling = False
        if self.body_count:
            # Rolling/torsional kernels run only when some shape has such friction.
            for span in (model.shape_rolling_frictions, model.shape_torsional_frictions):
                if span.ptr and span.size >= model.shape_count:
                    self.rolling = self.rolling or bool(np.any(self._read(span, wp.float32, model.shape_count) != 0.0))
            if self.rolling:
                self._require(model.body_inertia_tensors, self.body_count, "body_inertia_tensors")

        self._precompute_springs()
        self._precompute_cloth()
        self._precompute_bodies()
        # Material kernels need their state count and rest-volume cache even alongside affine joints.
        self._precompute_tetrahedra()
        self._constructor_reads.clear()

        # Remap arbitrary world labels (including -1) in first-seen order; each replicated template is offset by its
        # number of distinct local labels. Coefficient groups are contiguous unless the labels interleave.
        local_ids = np.zeros(self.per_world, dtype=np.int64)
        if model.particle_world.size:
            local_ids = self._read(model.particle_world, wp.int32, self.per_world).astype(np.int64)
        labels: dict[int, int] = {}
        for index, value in enumerate(local_ids):
            local_ids[index] = labels.setdefault(int(value), len(labels))
        self.group_count = copies * len(labels)
        self.particles_per_group = count // self.group_count
        rows = np.arange(count)
        ids = (rows // self.per_world) * len(labels) + local_ids[rows % self.per_world]
        contiguous = count % self.group_count == 0 and np.array_equal(ids, rows // self.particles_per_group)
        self.coefficient_world = self._empty(wp.int32) if contiguous else self._upload(ids.astype(np.int32), wp.int32)
        if self.row_count is None:
            self.row_count = self._upload(np.zeros(count, dtype=np.int32), wp.int32)

        self.residual = self._zeros(count, wp.vec3)
        self._precompute_collision_bvhs()
        self.direction = self._zeros(count, wp.vec3)
        self.z = self._zeros(count, wp.vec3)
        self.product = self._zeros(count, wp.vec3)
        self.contact_product = self._zeros(count, wp.vec3)
        self.denominator = self._zeros(_MAX_LINEAR_ITERATIONS * self.group_count, wp.float32)
        self.numerator = self._zeros(_MAX_LINEAR_ITERATIONS * self.group_count, wp.float32)
        # Built even with self-collision disabled, as natively.
        self._precompute_self_particle_pairs()
        self._precompute_fluid()
        self._precompute_granular()
        wp.synchronize_device(self.device)
        self._initialized = True

    def _precompute_collision_bvhs(self) -> None:
        """Validate the self-collision primitives; the BVHs are built only once self-collision is ported."""
        model = self.model
        self.self_collision_primitives = bool(
            model.collision_triangle_count or model.collision_edge_count or model.collision_particle_count
        )
        if not model.collision_particle_count:
            return
        self._require(model.collision_particle_indices, model.collision_particle_count, "collision_particle_indices")
        self._require(model.particle_radii, self.per_world, "particle_radii")
        indices = self._read(model.collision_particle_indices, wp.int32, model.collision_particle_count)
        radii = self._read(model.particle_radii, wp.float32, self.per_world)
        seen: set[int] = set()
        for value in indices:
            index = int(value)
            if index < 0 or index >= self.per_world:
                raise ValueError("collision_particle_indices contains an invalid particle index")
            if index in seen:
                raise ValueError("collision_particle_indices contains a duplicate particle index")
            seen.add(index)
            if not math.isfinite(radii[index]) or radii[index] <= 0.0:
                raise ValueError("collision particle radii must be finite and positive")

    def _precompute_drag(self) -> None:
        model = self.model
        self.drag_count = self.drag_local_count = 0
        empty_f, empty_v, empty_i = self._empty(wp.float32), self._empty(wp.vec3), self._empty(wp.vec3i)
        self.drag_indices, self.drag_coefficients = empty_i, empty_f
        self.drag_normal, self.drag_area, self.drag_scalar = empty_v, empty_f, empty_f
        if not model.collision_triangle_count or not model.triangle_drag_coefficients.size:
            return
        n = model.collision_triangle_count
        self._require(model.triangle_drag_coefficients, n, "triangle_drag_coefficients")
        self._require(model.collision_triangle_indices, n, "collision_triangle_indices")
        indices = self._read(model.collision_triangle_indices, wp.vec3i, n)
        coefficients = self._read(model.triangle_drag_coefficients, wp.float32, n)
        if not np.all(np.isfinite(coefficients)) or np.any(coefficients < 0.0):
            raise ValueError("triangle drag coefficients must be finite and non-negative")
        keep = coefficients > 0.0
        if np.any((indices[keep] < 0) | (indices[keep] >= self.per_world)):
            raise ValueError("drag triangle references an invalid particle index")
        self.drag_local_count = int(np.count_nonzero(keep))
        self.drag_count = self.drag_local_count * (self.count // self.per_world)
        if not self.drag_count:
            return
        self.drag_indices = self._upload(indices[keep], wp.vec3i)
        self.drag_coefficients = self._upload(coefficients[keep], wp.float32)
        self.drag_normal = self._zeros(self.drag_count, wp.vec3)
        self.drag_area = self._zeros(self.drag_count, wp.float32)
        self.drag_scalar = self._zeros(self.drag_count, wp.float32)

    def _inertial_templates(self) -> bool:
        """Non-diagonal particle masses use the body inertial cache even without affine bodies."""
        return bool(self.model.body_count or self.model.particle_mass_non_diagonal_count)

    def _precompute_particle_matrices(
        self, diagonal: np.ndarray, counts: np.ndarray, columns: np.ndarray, values: np.ndarray
    ) -> None:
        """Upload the scalar ELL template (slot-major, one world) and its zero-valued channel copies."""
        per_world = self.per_world
        self.host_diagonal = self._upload(np.asarray(diagonal, dtype=np.float32), wp.float32)
        self.particle_diagonal_templates = [self._zeros(per_world, wp.float32) for _ in range(5)]
        self.local_row_count = self._upload(np.asarray(counts[:per_world], dtype=np.int32), wp.int32)
        self.row_count = self._upload(np.asarray(counts, dtype=np.int32), wp.int32)
        self.scalar_template_columns = self._upload(columns.astype(np.int32), wp.int32)
        self.scalar_template_values = self._upload(values.astype(np.float32), wp.float32)
        # The inertial channel shares the template's columns with zero values (including -1 padding columns).
        self.body_l1 = self._upload(np.zeros(len(values), dtype=np.float32), wp.float32)
        self.particle_kind = self._upload(np.zeros(per_world, dtype=np.int32), wp.int32)
        if self.max_entries:
            n = self.max_entries * self.count
            self.block_template_columns = self._zeros(n, wp.int32)
            self.block_template_values = self._zeros(n, wp.mat33)
            self.block_entry_columns = self._zeros(n, wp.int32)
            self.block_entry_values = self._zeros(n, wp.mat33)

    def _template_row_counts(self) -> wp.array:
        return self.local_row_count if self.local_row_count.shape[0] else self.row_count

    def _precompute_springs(self) -> None:
        model = self.model
        per_world = self.per_world
        self.max_entries = 0
        self.row_count = None
        self.local_row_count = self._empty(wp.int32)
        self.block_template_columns = self.block_entry_columns = self._empty(wp.int32)
        self.block_template_values = self.block_entry_values = self._empty(wp.mat33)
        self.spring_rows: SparseRows | None = None
        self.spring_diagonal: np.ndarray | None = None
        self.spring_count = model.spring_count * (self.count // per_world)
        if not self.spring_count:
            if not self._inertial_templates() and not model.triangle_count and not model.edge_count:
                self._precompute_particle_matrices(
                    np.zeros(per_world, dtype=np.float32),
                    np.zeros(self.count, dtype=np.int32),
                    np.zeros(0, dtype=np.int32),
                    np.zeros(0, dtype=np.float32),
                )
            return
        indices = self._read(model.spring_indices, wp.vec2i, model.spring_count).astype(np.int64)
        stiffness = self._read(model.spring_stiffnesses, wp.float32, model.spring_count)
        rest = self._read(model.spring_rest_lengths, wp.float32, model.spring_count)
        a, b = indices[:, 0], indices[:, 1]
        checks = (
            (~((stiffness >= 0.0) & np.isfinite(stiffness)), "spring stiffness must be finite and non-negative"),
            (~((rest > 0.0) & np.isfinite(rest)), "spring rest length must be finite and positive"),
            ((a < 0) | (b < 0) | (a >= per_world) | (b >= per_world), "spring references an invalid particle index"),
            (a == b, "spring endpoints must reference different particles"),
        )
        bad = np.zeros(len(a), dtype=bool)
        for mask, _message in checks:
            bad |= mask
        if bad.any():
            # Report the first failing spring's first failing condition, as the native per-spring loop.
            first = int(np.argmax(bad))
            raise ValueError(next(message for mask, message in checks if mask[first]))
        rows = SparseRows(per_world)
        # Duplicate columns accumulate in spring order; each spring adds (a, b) then (b, a).
        rows.add(np.stack([a, b], 1).ravel(), np.stack([b, a], 1).ravel(), np.repeat(stiffness, 2))
        diagonal = np.zeros(per_world, dtype=np.float32)
        np.add.at(diagonal, np.stack([a, b], 1).ravel(), np.repeat(stiffness, 2))
        if self._inertial_templates() or model.triangle_count or model.edge_count:
            # Negating once after the duplicate summation keeps the native rounding and signed zeros.
            rows.negate()
            self.spring_rows, self.spring_diagonal = rows, diagonal
            return
        columns, values, self.max_entries = rows.ell(-rows.values)
        counts = np.tile(rows.counts(), self.count // per_world)
        self._precompute_particle_matrices(diagonal, counts, columns, values)

    def _precompute_cloth(self) -> None:
        model = self.model
        per_world = self.per_world
        copies = self.count // per_world
        self.triangle_count = model.triangle_count * copies
        self.edge_count = model.edge_count * copies
        self.bending_damping = False
        if not self.triangle_count and not self.edge_count:
            return
        f32 = np.float32
        indices = self._read(model.triangle_indices, wp.vec3i, model.triangle_count).astype(np.int64)
        poses = self._read(model.triangle_inv_dms, wp.mat22, model.triangle_count).reshape(-1, 4)
        stiffness = self._read(model.triangle_aniso_stiffnesses, wp.vec3, model.triangle_count)
        areas = self._read(model.triangle_rest_areas, wp.float32, model.triangle_count)
        rows = self.spring_rows if self.spring_rows is not None else SparseRows(per_world)
        diagonal = self.spring_diagonal if self.spring_diagonal is not None else np.zeros(per_world, dtype=f32)
        self.spring_rows = self.spring_diagonal = None
        if np.any((indices < 0) | (indices >= per_world)):
            raise ValueError("triangle references an invalid particle index")
        # Float32 host arithmetic in the native operation order (no fused multiply-add).
        m0, m1, m2, m3 = (poses[:, k].astype(f32) for k in range(4))
        u = np.stack([-m0 - m2, m0, m2], 1)
        v = np.stack([-m1 - m3, m1, m3], 1)
        ku = stiffness[:, 0] + stiffness[:, 2]
        kv = stiffness[:, 2] + stiffness[:, 1]
        pairs = [(i, j) for i in range(3) for j in range(i, 3)]
        diagonal_rows: list = []
        diagonal_values: list = []
        off_rows: list = []
        off_columns: list = []
        off_values: list = []
        values = np.stack([((u[:, i] * ku) * u[:, j] + (v[:, i] * kv) * v[:, j]) * areas for i, j in pairs], 1)
        self._scatter_pairs(indices, pairs, values, diagonal_rows, diagonal_values, off_rows, off_columns, off_values)

        edges = self._read(model.edge_indices, wp.vec4i, model.edge_count).astype(np.int64)
        cots = self._read(model.edge_bending_cots, wp.vec4, model.edge_count)
        properties = self._read(model.edge_bending_properties, wp.vec2, model.edge_count)
        edge_areas = self._read(model.edge_rest_areas, wp.float32, model.edge_count)
        valid = edge_areas > 0.0
        for i in range(4):
            valid &= edges[:, i] >= 0
            for j in range(i + 1, 4):
                valid &= edges[:, i] != edges[:, j]
        if np.any(edges[valid] >= per_world):
            raise ValueError("bending edge references an invalid particle index")
        edges, cots, properties, edge_areas = edges[valid], cots[valid], properties[valid], edge_areas[valid]
        self.bending_damping = bool(np.any(properties[:, 1] > 0.0))
        c = cots.astype(f32)
        w = np.stack([-c[:, 0] - c[:, 2], -c[:, 1] - c[:, 3], c[:, 2] + c[:, 3], c[:, 0] + c[:, 1]], 1)
        scale = (f32(3.0) / edge_areas) * properties[:, 0]
        edge_pairs = [(i, j) for i in range(4) for j in range(i, 4)]
        edge_values = np.stack([(w[:, i] * scale) * w[:, j] for i, j in edge_pairs], 1).astype(f32)
        self._scatter_pairs(
            edges, edge_pairs, edge_values, diagonal_rows, diagonal_values, off_rows, off_columns, off_values
        )
        np.add.at(diagonal, np.concatenate(diagonal_rows), np.concatenate(diagonal_values))
        rows.add(np.concatenate(off_rows), np.concatenate(off_columns), np.concatenate(off_values))
        if self._inertial_templates():
            # Kept for the body CSR/ELL assembly.
            self.spring_rows, self.spring_diagonal = rows, diagonal
            return
        columns, values, self.max_entries = rows.ell()
        counts = np.tile(rows.counts(), copies)
        self._precompute_particle_matrices(diagonal, counts, columns, values)

    @staticmethod
    def _scatter_pairs(
        indices: np.ndarray,
        pairs: list[tuple[int, int]],
        values: np.ndarray,
        diagonal_rows: list,
        diagonal_values: list,
        off_rows: list,
        off_columns: list,
        off_values: list,
    ) -> None:
        """Element-major contribution lists: per element, per (a, b >= a) pair, the diagonal or (a, b), (b, a)."""
        diagonal_mask = np.array([i == j for i, j in pairs])
        first = np.array([i for i, _ in pairs])
        second = np.array([j for _, j in pairs])
        rows = indices[:, first]
        columns = indices[:, second]
        diagonal_rows.append(rows[:, diagonal_mask].ravel())
        diagonal_values.append(values[:, diagonal_mask].ravel())
        off_r, off_c, off_v = rows[:, ~diagonal_mask], columns[:, ~diagonal_mask], values[:, ~diagonal_mask]
        off_rows.append(np.stack([off_r, off_c], 2).ravel())
        off_columns.append(np.stack([off_c, off_r], 2).ravel())
        off_values.append(np.stack([off_v, off_v], 2).ravel())

    def _precompute_tetrahedra(self) -> None:
        model = self.model
        self.tetrahedron_count = model.tetrahedron_count * (self.count // self.per_world)
        self.tetrahedron_volumes = self._empty(wp.float32)
        if not self.tetrahedron_count:
            return
        indices = self._read(model.tetrahedron_indices, wp.vec4i, model.tetrahedron_count)
        if np.any((indices < 0) | (indices >= self.per_world)):
            raise ValueError("tetrahedron references an invalid particle index")
        m = self._read(model.tetrahedron_inverse_dms, wp.mat33, model.tetrahedron_count).reshape(-1, 9)
        m = m.astype(np.float32)
        a = m[:, 8] * m[:, 4] - m[:, 5] * m[:, 7]
        b = m[:, 8] * m[:, 3] - m[:, 5] * m[:, 6]
        c = m[:, 3] * m[:, 7] - m[:, 6] * m[:, 4]
        determinant = (m[:, 0] * a - m[:, 1] * b) + m[:, 2] * c
        with np.errstate(divide="ignore"):
            volumes = np.float32(1.0) / (determinant * np.float32(6.0))
        self.tetrahedron_volumes = self._upload(volumes.astype(np.float32), wp.float32)

    # ------------------------------------------------------------------------------------------------------------
    # Step

    @staticmethod
    def _parse_step_arguments(args: tuple, kwargs: dict) -> tuple[Control, Contact, float, Any]:
        names = ("control", "contact", "dt", "stream")
        values = dict(kwargs)
        positional = list(args)
        # Native overloads: (dt[, stream]), (contact, dt[, stream]), (control, contact, dt[, stream]).
        if positional and isinstance(positional[0], Control):
            order = names
        elif positional and isinstance(positional[0], Contact):
            order = names[1:]
        else:
            order = names[2:]
        for name, value in zip(order, positional, strict=False):
            if name in values:
                raise TypeError(f"step() got multiple values for {name!r}")
            values[name] = value
        if "dt" not in values:
            raise TypeError("step() missing required argument 'dt'")
        control = values.get("control") or Control()
        contact = values.get("contact") or Contact()
        if not isinstance(control, Control) or not isinstance(contact, Contact):
            raise TypeError("step() received an invalid control or contact")
        return control, contact, float(np.float32(values["dt"])), values.get("stream")

    def _stream_scope(self, stream: Any):
        device = self.device if self.device is not None else wp.get_cuda_device()
        resolved = _resolve_stream(stream, device)
        return nullcontext() if resolved is None else wp.ScopedStream(resolved, sync_enter=False)

    def _step(self, state_in: State, state_out: State, control: Control, contact: Contact, dt: float, stream: Any):
        if not self._initialized:
            raise RuntimeError("initialize() must be called before step()")
        with self._stream_scope(stream):
            key = self._graph_key(state_in, state_out, control, contact, dt)
            if key is None:
                self._step_on_stream(state_in, state_out, control, contact, dt)
            else:
                self._step_graph(key, state_in, state_out, control, contact, dt)

    # A step issues hundreds of small launches whose Python overhead dominates; replaying a captured CUDA graph removes
    # it. The graph bakes in every pointer, scalar and host branch, so it is keyed on all step inputs (buffer addresses,
    # scalars, dt and the configuration) and on the addresses of the solver's own arrays, which grow lazily.
    _GRAPH_CACHE_SIZE = 16

    def _graph_key(self, state_in: State, state_out: State, control: Control, contact: Contact, dt: float):
        if not _GRAPH_ENABLED or self.device is None or not self.device.is_cuda:
            return None
        # A caller capturing its own graph records the launches directly.
        if self.device.is_capturing or not wp.is_mempool_enabled(self.device):
            return None

        def record(value: Any) -> tuple:
            fields = []
            for name, _kind, _default in value._FIELDS:
                field = getattr(value, name)
                fields.append((field.ptr, field.size) if isinstance(field, Span) else field)
            return tuple(fields)

        stream = wp.get_stream(self.device)
        return (
            record(state_in),
            record(state_out),
            record(control),
            record(contact),
            float(dt),
            record(self._config),
            stream.cuda_stream,
        )

    def _array_fingerprint(self) -> tuple:
        # Only owned allocations: per-step views of the caller's buffers are covered by the key.
        return tuple(
            (name, value.ptr)
            for name, value in vars(self).items()
            if isinstance(value, wp.array) and value.deleter is not None
        )

    def _step_graph(self, key: tuple, state_in: State, state_out: State, control: Control, contact: Contact, dt: float):
        cached = self._graphs.get(key)
        if cached is not None:
            graph, launches, fingerprint = cached
            if fingerprint == self._array_fingerprint():
                wp.capture_launch(graph)
                self._launches.update(launches)
                return
            self._graphs.clear()
        # The first step with a key runs eagerly: it validates the inputs, loads modules and grows lazy buffers.
        if key not in self._graph_seen:
            if len(self._graph_seen) >= 4 * self._GRAPH_CACHE_SIZE:
                self._graph_seen.clear()
            self._graph_seen.add(key)
            self._step_on_stream(state_in, state_out, control, contact, dt)
            return
        before = Counter(self._launches)
        fingerprint = self._array_fingerprint()
        with wp.ScopedCapture(device=self.device) as capture:
            self._step_on_stream(state_in, state_out, control, contact, dt)
        launches = Counter(self._launches)
        launches.subtract(before)
        # Capture records without executing; run this step from the graph.
        wp.capture_launch(capture.graph)
        if fingerprint == self._array_fingerprint():
            if len(self._graphs) >= self._GRAPH_CACHE_SIZE:
                self._graphs.clear()
            self._graphs[key] = (capture.graph, +launches, fingerprint)

    def _step_on_stream(self, state_in: State, state_out: State, control: Control, contact: Contact, dt: float):
        model, config = self.model, self._config
        count = self.count
        if contact.contact_force.size:
            if contact.max_count < 0 or contact.soft_max_count < 0:
                raise ValueError("Contact capacity must be non-negative")
            capacity = contact.max_count + contact.soft_max_count
            if contact.contact_force.size != capacity:
                raise ValueError("contact_force must cover rigid and soft contact capacity")
            self._view(contact.contact_force, wp.vec3, capacity * 2).zero_()
        if not count:
            return
        if not (dt > 0.0) or not math.isfinite(dt):
            raise ValueError("dt must be finite and positive")
        self._validate_step_config()
        surface = bool(contact.soft_max_count and contact.soft_indices.size and contact.soft_barycentric.size)
        particle_contact = bool(contact.soft_max_count and (not surface or contact.soft_particle_only_shapes.size))
        contact_product = particle_contact and config.enable_collision_hessian_non_diagonal
        rigid = bool(self.body_count and contact.max_count and contact.count.size)
        hard = rigid and config.rigid_contact_hard
        rolling = rigid and self.rolling
        if hard and config.rigid_contact_history and not contact.match_index.size:
            raise ValueError("hard-contact history requires collision match indices")
        self._rigid = rigid
        self._hard = hard
        self._rolling = rolling
        if contact.soft_max_count:
            if not surface and (
                contact.soft_indices.size or contact.soft_barycentric.size or contact.soft_particle_only_shapes.size
            ):
                raise RuntimeError("Partial surface-contact inputs have not been implemented")
            if not model.shape_count:
                raise ValueError("Soft contacts require model shapes")
            self._require(contact.soft_count, 1, "soft_count")
            if particle_contact and contact.soft_particle.size < contact.soft_max_count:
                raise ValueError("soft contact arrays are smaller than soft_max_count")
            for field in ("soft_shape", "soft_body_pos", "soft_body_vel", "soft_normal"):
                self._require(getattr(contact, field), contact.soft_max_count, field)
        self._particle_contact = particle_contact
        self._contact = contact
        self._surface = surface
        self._qd_in = self._view(state_in.particle_qd, wp.vec3, count)
        self._contact_product = contact_product
        self_collision = bool(config.enable_self_collision and self.self_capacity)
        self._self = self_collision
        self._dt = dt
        self._validate_cable_config()

        pd = config.nonlinear_solver == NonlinearSolver.PD_NEWTON
        drag = bool(self.drag_count) and config.air_density > 0.0
        bodies = self.body_count != 0
        rigidity = config.affine_body_rigidity_model
        so3 = bodies and rigidity == AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
        six_springs = bodies and rigidity == AffineBodyRigidityModel.SIX_SPRINGS
        constitutive = bodies and not so3 and not six_springs
        joint_al = config.joint_constraint_mode == JointConstraintMode.AUGMENTED_LAGRANGIAN
        alpha = config.constraint_alpha
        # AL joint terms apply without a constraint_alpha gate, as in the latest SDK.
        joint_al_active = joint_al
        body_tangent = (
            (bodies and alpha > 0.0 and not six_springs)
            or rigid
            or bool(self.cable_count)
            or bool(self.drive_total)
            or (pd and joint_al_active and bool(self.joint_line_total))
        )
        block_matrix = not pd
        block = (
            block_matrix
            or self_collision
            or surface
            or contact_product
            or self.bending_damping
            or bool(self.tetrahedron_count)
            or drag
            or body_tangent
            or bool(self.fluid_rows)
            or bool(self.granular_points)
        )
        # The PD branch ignores the body-block option when it selects the scalar matrix.
        body_block = bodies and block and config.affine_body_block_preconditioner
        coupled_materials = (
            self._articulated_materials()
            and so3
            and joint_al
            and not self.drag_count
            and not config.rigid_contact_hard
            and not config.affine_body_block_preconditioner
            and (not pd or not self.tetrahedron_count)
        )
        extra_product = (
            self_collision
            or surface
            or contact_product
            or self.bending_damping
            or bool(self.tetrahedron_count)
            or drag
            or body_tangent
            or bool(self.fluid_rows)
            or bool(self.granular_points)
        )
        reference_pcg = config.reference_linear_schedule
        # Joint-owned cable models (every particle an affine proxy) evaluate cables with the drive terms.
        m = model
        joint_cable_step = bool(
            m.cable_count
            and m.body_count
            and m.particle_count == m.body_count * 4
            and m.joint_drive_count
            and m.joint_body_spring_count
            and m.body_spring_count == m.joint_body_spring_count
            and m.body_point_line_count == m.joint_body_point_line_count
            and not m.body_distance_constraint_count
            and not m.spring_count
            and not m.triangle_count
            and not m.edge_count
            and not m.tetrahedron_count
            and not m.fluid_particle_count
            and not m.granular_particle_count
            and so3
            and joint_al
            and not pd
            and not self_collision
            and not self.drag_count
            and not contact.soft_max_count
            and not config.rigid_contact_hard
            and count == self.body_state_count * 4
        )
        self._joint_cable_step = joint_cable_step
        # Surface contact Jacobians are evaluated once per Newton iteration instead of once per CG iteration.
        self._cached_surface = surface and extra_product and not reference_pcg
        # The matrix-free rigid contact product reuses geometry cached once per Newton iteration.
        self._cached_rigid = rigid and not hard and config.enable_collision_hessian_non_diagonal
        if self._cached_rigid and self.rigid_tangent_capacity < contact.max_count:
            n = contact.max_count
            self.rigid_tangent_capacity = n
            self.rigid_tangent_targets = self._zeros(8 * n, wp.int32)
            self.rigid_tangent_normal_weights = self._zeros(8 * n, wp.float32)
            self.rigid_tangent_friction_weights = self._zeros(8 * n, wp.float32)
            self.rigid_tangent_entries = self._zeros(n, wp.int32)
            self.rigid_tangent_normal = self._zeros(n, wp.vec3)
            self.rigid_tangent_stiffness = self._zeros(n, wp.vec2)
        if self._cached_surface and self.surface_tangent_cache.shape[0] < contact.soft_max_count:
            self.surface_tangent_cache = wp.zeros(
                contact.soft_max_count, dtype=surface_contact_kernels.SurfaceTangent, device=self.device
            )
        fused_pcg = block_matrix and not body_block and not reference_pcg
        if control.tetrahedron_activations.size and control.tetrahedron_activations.size < self.tetrahedron_count:
            raise ValueError("tetrahedron activations are smaller than the tetrahedron state count")
        if self.drive_total:
            worlds_total = count // self.per_world
            if control.joint_drive_velocities.size:
                self._require(control.joint_drive_velocities, self.drive_total, "control.joint_drive_velocities")
            self._require(control.joint_f, model.joint_dof_count * worlds_total, "control.joint_f")
            self._require(control.joint_target_q, model.joint_target_count * worlds_total, "control.joint_target_q")
            self._require(control.joint_target_qd, model.joint_dof_count * worlds_total, "control.joint_target_qd")
        self._require(state_in.particle_q, count, "state_in.particle_q")
        self._require(state_in.particle_qd, count, "state_in.particle_qd")
        if state_in.particle_f.size and state_in.particle_f.size < count:
            raise ValueError("input state particle_f is smaller than particle_count")
        self._require(state_out.particle_q, count, "state_out.particle_q")
        self._require(state_out.particle_qd, count, "state_out.particle_qd")

        per_world = self.per_world
        q_in = self._view(state_in.particle_q, wp.vec3, count)
        qd_in = self._view(state_in.particle_qd, wp.vec3, count)
        q_out = self._view(state_out.particle_q, wp.vec3, count)
        qd_out = self._view(state_out.particle_qd, wp.vec3, count)
        force = self._view(state_in.particle_f, wp.vec3, count)
        mass = self._view(model.particle_masses, wp.float32, per_world)
        flags = self._view(model.particle_flags, wp.int32, per_world)
        # Homogeneous replicated models (world_count <= 1) pass their world map; otherwise worlds follow the row.
        worlds = (
            self._view(model.particle_world, wp.int32, per_world)
            if model.world_count <= 1 and model.particle_world.size
            else self._empty(wp.int32)
        )
        gravity = self._view(model.gravity, wp.vec3)
        inverse_dt = 1.0 / dt
        inverse_dt2 = 1.0 / (dt * dt)
        so3_alpha = alpha * inverse_dt2
        spring_alpha = alpha if six_springs else 0.0
        body_so3_alpha = so3_alpha if so3 else 0.0
        # AL joint lines give the constant D4/L5 channel zero weight; penalty joint lines weight it by alpha.
        line_matrix_alpha = 0.0 if joint_al else alpha
        empty_v = self._empty(wp.vec3)
        empty_f = self._empty(wp.float32)
        empty_m = self._empty(wp.mat33)
        v = self._model_views()
        activations = self._view(control.tetrahedron_activations, wp.float32)
        launch = self._launch
        group_count = self.group_count
        per_group = self.particles_per_group
        bodies_per_world = self.body_count
        body_states = self.body_state_count
        body_indices = v["affine_body_indices"]
        body_masses = v["body_masses"]
        rows_total = count
        body_block_flag = bodies and block and config.affine_body_block_preconditioner
        self._contact_block = hard and body_block_flag and config.affine_body_contact_block_preconditioner
        # Query the pre-prediction positions once per step, before the prediction mutates them.
        if self_collision:
            self._begin_self(q_in)
        if self.cable_count and self._cable_ramping():
            self._prepare_cable_penalties(False, True)
        if hard:
            self.reserve_rigid_contact_capacity(contact.max_count)
            launch(
                "hard_initialize",
                hard_contact_kernels.initialize_hard_contacts,
                contact.max_count,
                [self._hard_view(contact, q_in, inverse_dt)],
            )

        if self._inertial_templates():
            entries = self.max_entries * per_world
            launch(
                "refresh_nonzeros",
                step_kernels.refresh_inertial_cache_nonzeros,
                entries,
                [
                    v["particle_mass_non_diagonal_values"],
                    self.body_offsets_a,
                    self.body_ids_a,
                    body_masses,
                    self.body_offsets_b,
                    self.body_ids_b,
                    self.body_l1,
                    self.body_l2,
                ],
            )
            launch(
                "refresh_diagonals",
                step_kernels.refresh_inertial_cache_diagonals,
                per_world,
                [
                    body_masses,
                    self.body_diagonal_offsets,
                    self.body_diagonal_ids,
                    self.body_d1,
                    self.body_d2,
                ],
            )
            # Only SIX_SPRINGS weights D1/L2; constitutive models use their matrix-free tangent, not D2.
            launch(
                "start",
                step_kernels.initialize_step,
                count,
                [
                    dt,
                    gravity,
                    worlds,
                    qd_in,
                    q_in,
                    self.host_diagonal,
                    self.body_d1,
                    spring_alpha,
                    self.body_d2,
                    0.0,
                    self.joint_d3,
                    alpha,
                    self.line_d4,
                    line_matrix_alpha,
                    self.distance_d5,
                    alpha,
                    mass,
                    flags,
                    self.body_kind,
                    self.previous,
                    self.target,
                    self.diagonal,
                    per_world,
                ],
            )
            # Preserve the native graph order for the supplied joint velocity and diagonal snapshots.
            if self.drive_total and control.joint_drive_velocities.size:
                wp.copy(self.drive_qd, self._view(control.joint_drive_velocities, wp.float32, self.drive_total))
            wp.copy(self.base_diagonal, self.diagonal, count=count)
            # The distance targets start from the prediction, before the SO(3) state.
            if self.distance_total:
                launch(
                    "distance_initialize",
                    joint_kernels.initialize_body_distance_constraint_state,
                    self.distance_total,
                    [
                        q_in,
                        body_indices,
                        v["body_distance_constraint_indices"],
                        v["body_distance_constraint_points0"],
                        v["body_distance_constraint_points1"],
                        self.distance_w0,
                        self.distance_w1,
                        v["body_distance_constraint_rest_lengths"],
                        self.distance_target,
                        self.distance_multiplier,
                        model.body_distance_constraint_count,
                        per_world,
                    ],
                )
            # Derive velocities from the saved pre-prediction positions when the optional control span is absent.
            if self.drive_total and not control.joint_drive_velocities.size:
                launch(
                    "drive_velocity",
                    joint_kernels.initialize_joint_coordinate_velocities,
                    self.drive_total,
                    [
                        self._joint_view(v),
                        self.previous,
                        qd_in,
                        self.drive_qd,
                    ],
                )
            if so3:
                launch(
                    "so3_initialize",
                    affine_kernels.initialize_affine_body_so3_state,
                    body_states,
                    [
                        q_in,
                        body_indices,
                        self.body_rotation,
                        self.body_multiplier,
                        bodies_per_world,
                        per_world,
                    ],
                )
            if joint_al and self.joint_spring_total:
                launch(
                    "joint_spring_initialize",
                    joint_kernels.initialize_joint_body_spring_state,
                    self.joint_spring_total,
                    [self.joint_multiplier],
                )
            if joint_al and self.joint_line_total:
                launch(
                    "joint_line_initialize",
                    joint_kernels.initialize_joint_body_point_line_state,
                    self.joint_line_total,
                    [
                        q_in,
                        body_indices,
                        v["body_point_line_indices"],
                        self.point_line_wp,
                        self.point_line_we0,
                        self.point_line_we1,
                        v["body_point_line_endpoints0"],
                        v["body_point_line_endpoints1"],
                        self.line_segment,
                        self.line_multiplier,
                        self.line_projector,
                        model.body_point_line_count,
                        model.joint_body_point_line_count,
                        per_world,
                    ],
                )
            launch(
                "combine",
                step_kernels.combine_system_non_diagonal,
                entries,
                [
                    inverse_dt2,
                    spring_alpha,
                    0.0,
                    alpha,
                    line_matrix_alpha,
                    alpha,
                    self.body_l0,
                    self.body_l1,
                    self.body_l2,
                    self.body_zero_ell,
                    self.joint_l4,
                    self.line_l5,
                    self.distance_l6,
                    self.scalar_template_values,
                ],
            )
        else:
            templates = self.particle_diagonal_templates
            launch(
                "start",
                step_kernels.initialize_step,
                count,
                [
                    dt,
                    gravity,
                    worlds,
                    qd_in,
                    q_in,
                    self.host_diagonal,
                    templates[0],
                    0.0,
                    templates[1],
                    0.0,
                    templates[2],
                    alpha,
                    templates[3],
                    0.0,
                    templates[4],
                    alpha,
                    mass,
                    flags,
                    self.particle_kind,
                    self.previous,
                    self.target,
                    self.diagonal,
                    per_world,
                ],
            )
            wp.copy(self.base_diagonal, self.diagonal, count=count)
        if drag:
            launch(
                "drag_start",
                drag_kernels.initialize_triangle_drag,
                self.drag_count,
                [
                    self.previous,
                    self.drag_indices,
                    self.drag_normal,
                    self.drag_area,
                    self.drag_local_count,
                    per_world,
                ],
            )
        # The native solver initializes drag before rebuilding the block template.
        if block_matrix and self.max_entries:
            launch(
                "block_entries",
                pcg_kernels.initialize_block_non_diagonal,
                self.max_entries * count,
                [
                    self.scalar_template_columns,
                    self.scalar_template_values,
                    self._template_row_counts(),
                    per_world,
                    count,
                    self.block_template_columns,
                    self.block_template_values,
                ],
            )

        for iteration in range(config.nonlinear_iterations):
            if self.cable_count:
                launch(
                    "cable_prepare", cable_kernels.prepare_cable_tangents, self.cable_count, [self._cable_view(), q_in]
                )
            # The native solver restores this cache on every iteration, including contact-free models.
            wp.copy(self.diagonal, self.base_diagonal, count=count)
            # With bodies the inertia couples through the unscaled mass non-diagonal cache (L1).
            launch(
                "rhs",
                step_kernels.initialize_rhs,
                count,
                [
                    inverse_dt2,
                    q_in,
                    self.target,
                    mass,
                    self._template_row_counts(),
                    self.scalar_template_columns,
                    self.body_l1 if (bodies or self.max_entries) else empty_f,
                    self.max_entries,
                    force,
                    flags,
                    self.rhs,
                    count,
                    per_world,
                ],
            )
            if self.fluid_rows:
                self._accumulate_fluid(q_in, dt)
            if self.granular_points:
                self._accumulate_granular(q_in, dt)
            if drag:
                launch(
                    "drag_force",
                    drag_kernels.evaluate_triangle_drag,
                    self.drag_count,
                    [
                        q_in,
                        self.previous,
                        self.drag_indices,
                        self.drag_coefficients,
                        self.drag_normal,
                        self.drag_area,
                        config.air_density,
                        wp.vec3(*config.wind_velocity),
                        inverse_dt,
                        self.rhs,
                        self.drag_scalar,
                        self.drag_local_count,
                        per_world,
                    ],
                )
            if hard:
                self._hard_evaluate(contact, q_in, inverse_dt, force=self.rhs, scalar=self.diagonal)
            elif rigid:
                self._rigid_contact(contact, q_in, qd_in, inverse_dt, force=self.rhs, scalar=self.diagonal)
            if rolling:
                self._launch_rolling(
                    "rolling_force",
                    rolling_contact_kernels.evaluate_rolling_contact_force,
                    contact,
                    q_in,
                    inverse_dt,
                    [self.rhs, self._empty(wp.spatial_vector)],
                )
            # Rigid and rolling forces precede surface contacts, as in the native bag/balls schedules.
            if surface:
                self._surface_launch(
                    "surface_force",
                    surface_contact_kernels.evaluate_surface_contact_force,
                    contact,
                    q_in,
                    inverse_dt,
                    [self.rhs, self._empty(wp.spatial_vector)],
                )
            if particle_contact:
                self._soft_contact(contact, q_in, qd_in, inverse_dt, surface, force=self.rhs, scalar=self.diagonal)
            if self_collision:
                # Self penalties scale with this iteration's diagonal including the contact increments.
                wp.copy(self.collision_stiffness, self.diagonal, count=count)
                if surface:
                    self._surface_launch(
                        "surface_stiffness",
                        surface_contact_kernels.add_surface_contact_stiffness_scale,
                        contact,
                        q_in,
                        inverse_dt,
                        [self.collision_stiffness],
                    )
            # Force order: plain body springs (the kernel skips joint springs), distance constraints, AL joint
            # springs, point-lines, joint drive load, angular limits; finally SO(3).
            if self.body_spring_total:
                launch(
                    "body_spring_force",
                    joint_kernels.evaluate_body_spring_force,
                    self.body_spring_total,
                    [
                        q_in,
                        body_indices,
                        v["body_spring_indices"],
                        v["body_spring_points0"],
                        v["body_spring_points1"],
                        self.body_spring_w0,
                        self.body_spring_w1,
                        v["body_spring_stiffnesses"],
                        v["body_spring_rest_lengths"],
                        so3_alpha,
                        model.joint_body_spring_count,
                        int(not joint_al),
                        self.rhs,
                        model.body_spring_count,
                        per_world,
                    ],
                )
            if self.distance_total:
                launch(
                    "distance_force",
                    joint_kernels.evaluate_body_distance_constraint_force,
                    self.distance_total,
                    [
                        q_in,
                        body_indices,
                        v["body_distance_constraint_indices"],
                        v["body_distance_constraint_points0"],
                        v["body_distance_constraint_points1"],
                        self.distance_w0,
                        self.distance_w1,
                        v["body_distance_constraint_mass_scales"],
                        self.distance_target,
                        self.distance_multiplier,
                        so3_alpha,
                        self.rhs,
                        model.body_distance_constraint_count,
                        per_world,
                    ],
                )
            if joint_al_active and self.joint_spring_total:
                launch(
                    "joint_spring_force",
                    joint_kernels.evaluate_joint_body_spring_augmented_lagrangian_force,
                    self.joint_spring_total,
                    [
                        q_in,
                        body_indices,
                        v["body_spring_indices"],
                        v["body_spring_points0"],
                        v["body_spring_points1"],
                        self.body_spring_w0,
                        self.body_spring_w1,
                        v["body_spring_stiffnesses"],
                        self.joint_multiplier,
                        so3_alpha,
                        self.rhs,
                        model.body_spring_count,
                        model.joint_body_spring_count,
                        per_world,
                    ],
                )
            if self.point_line_total:
                launch(
                    "point_line_force",
                    joint_kernels.evaluate_body_point_line_force,
                    self.point_line_total,
                    [
                        q_in,
                        body_indices,
                        v["body_point_line_indices"],
                        self.point_line_wp,
                        self.point_line_we0,
                        self.point_line_we1,
                        v["body_point_line_endpoints0"],
                        v["body_point_line_endpoints1"],
                        v["body_point_line_stiffnesses"],
                        so3_alpha,
                        model.joint_body_point_line_count,
                        int(not joint_al),
                        self.rhs,
                        model.body_point_line_count,
                        per_world,
                    ],
                )
            if joint_al_active and self.joint_line_total:
                launch(
                    "joint_line_force",
                    joint_kernels.evaluate_joint_body_point_line_augmented_lagrangian_force,
                    self.joint_line_total,
                    [
                        q_in,
                        body_indices,
                        v["body_point_line_indices"],
                        self.point_line_wp,
                        self.point_line_we0,
                        self.point_line_we1,
                        v["body_point_line_endpoints0"],
                        v["body_point_line_endpoints1"],
                        v["body_point_line_stiffnesses"],
                        self.line_segment,
                        self.line_multiplier,
                        self.line_projector,
                        so3_alpha,
                        self.rhs,
                        model.body_point_line_count,
                        model.joint_body_point_line_count,
                        per_world,
                    ],
                )
            if self.drive_total:
                launch(
                    "drive_load",
                    joint_kernels.evaluate_joint_drive_load,
                    self.drive_total,
                    [
                        q_in,
                        self.previous,
                        self.drive_qd,
                        self.drive_angle,
                        self.drive_initialized,
                        body_indices,
                        v["joint_drive_types"],
                        v["joint_drive_body_indices"],
                        v["joint_drive_parent_origins"],
                        v["joint_drive_parent_axis_points"],
                        v["joint_drive_parent_reference_points"],
                        self.drive_w_po,
                        self.drive_w_co,
                        self.drive_w_pa,
                        self.drive_w_pr,
                        self.drive_w_cr,
                        v["joint_drive_dof_indices"],
                        v["joint_drive_target_indices"],
                        v["joint_target_stiffnesses"],
                        v["joint_target_dampings"],
                        v["joint_effort_limits"],
                        v["joint_target_modes"],
                        v["joint_passive_dampings"],
                        self._view(control.joint_f, wp.float32),
                        self._view(control.joint_target_q, wp.float32),
                        self._view(control.joint_target_qd, wp.float32),
                        self.rhs,
                        model.joint_drive_count,
                        per_world,
                        model.joint_dof_count,
                        model.joint_target_count,
                        inverse_dt,
                    ],
                )
                if self.armature:
                    self._joint_armature(v, q_in, inverse_dt, empty_v, self.rhs, empty_m)
                if joint_cable_step and self.cable_count:
                    self._evaluate_cables(q_in, inverse_dt, force=self.rhs)
                if self.angular_limits:
                    self._angular_limits(v, q_in, inverse_dt, empty_v, self.rhs, empty_m)
            if self_collision:
                self._accumulate_self(q_in)
            # Joint-owned cable schedules evaluate the cable force inside the drive block instead.
            if self.cable_count and not joint_cable_step:
                self._evaluate_cables(q_in, inverse_dt, force=self.rhs)
            if so3 and so3_alpha > 0.0:
                launch(
                    "so3_force",
                    affine_kernels.evaluate_affine_body_so3_augmented_lagrangian_force,
                    body_states,
                    [
                        q_in,
                        body_indices,
                        body_masses,
                        self.body_rotation,
                        self.body_multiplier,
                        so3_alpha,
                        self.rhs,
                        bodies_per_world,
                        per_world,
                    ],
                )
            if six_springs and so3_alpha > 0.0:
                launch(
                    "rigidity_spring_force",
                    affine_kernels.evaluate_affine_body_spring_force,
                    body_states,
                    [
                        q_in,
                        body_indices,
                        body_masses,
                        so3_alpha,
                        self.rhs,
                        bodies_per_world,
                        per_world,
                    ],
                )
            if constitutive and so3_alpha > 0.0:
                launch(
                    "constitutive_force",
                    affine_kernels.evaluate_affine_body_constitutive_force,
                    body_states,
                    [
                        q_in,
                        body_indices,
                        body_masses,
                        so3_alpha,
                        int(rigidity),
                        self.rhs,
                        bodies_per_world,
                        per_world,
                    ],
                )
            if self.spring_count:
                launch(
                    "spring_force",
                    spring_kernels.evaluate_spring_force,
                    self.spring_count,
                    [
                        q_in,
                        v["spring_indices"],
                        v["spring_stiffnesses"],
                        v["spring_rest_lengths"],
                        self.rhs,
                        model.spring_count,
                        per_world,
                    ],
                )
            if self.triangle_count:
                launch(
                    "stretch_force",
                    cloth_kernels.evaluate_stretch_force,
                    self.triangle_count,
                    [
                        q_in,
                        v["triangle_rest_areas"],
                        v["triangle_inv_dms"],
                        v["triangle_indices"],
                        v["triangle_aniso_stiffnesses"],
                        self.rhs,
                        model.triangle_count,
                        per_world,
                    ],
                )
            if self.tetrahedron_count:
                launch(
                    "tetrahedron_force",
                    tetrahedron_kernels.evaluate_tetrahedron_force,
                    self.tetrahedron_count,
                    [
                        q_in,
                        self.previous,
                        v["tetrahedron_indices"],
                        v["tetrahedron_inverse_dms"],
                        v["tetrahedron_materials"],
                        self.tetrahedron_volumes,
                        activations,
                        inverse_dt,
                        self.rhs,
                        model.tetrahedron_count,
                        per_world,
                    ],
                )
            if self.edge_count:
                launch(
                    "bending_force",
                    cloth_kernels.evaluate_bending_force,
                    self.edge_count,
                    [
                        q_in,
                        self.previous,
                        v["edge_rest_areas"],
                        v["edge_bending_cots"],
                        v["edge_indices"],
                        v["edge_bending_properties"],
                        inverse_dt,
                        self.rhs,
                        model.edge_count,
                        per_world,
                    ],
                )

            if not block:
                # The contact-free PD path keeps the constant scalar matrix and uses 1 / D.
                launch(
                    "scalar_inverse",
                    pcg_kernels.prepare_inverse_preconditioner,
                    count,
                    [
                        self.diagonal,
                        mass,
                        flags,
                        self.scalar_inverse,
                        per_world,
                    ],
                )
            else:
                self._assemble_blocks(
                    v,
                    q_in,
                    mass,
                    flags,
                    inverse_dt,
                    so3_alpha,
                    body_so3_alpha,
                    pd,
                    block_matrix,
                    body_block,
                    so3,
                    six_springs,
                    constitutive,
                    rigidity,
                    joint_al,
                    joint_al_active,
                    drag,
                )

            if self._cached_rigid:
                launch(
                    "rigid_tangent_cache",
                    rigid_contact_kernels.cache_contact_tangent,
                    contact.max_count,
                    [self._rigid_view(contact, q_in, self._empty(wp.vec3), inverse_dt), self._rigid_tangent_cache()],
                )
            if self._cached_surface:
                self._surface_launch(
                    "surface_tangent_cache",
                    surface_contact_kernels.cache_surface_contact_tangent,
                    contact,
                    q_in,
                    inverse_dt,
                    [self.surface_tangent_cache],
                )
            wp.copy(self.residual, self.rhs, count=count)
            self.solution.zero_()
            self.numerator.zero_()
            self.denominator.zero_()
            iterations = config.linear_iterations
            for k in range(iterations):
                numerator = self.numerator[k * group_count : (k + 1) * group_count]
                previous_numerator = (
                    self.numerator[(k - 1) * group_count : k * group_count] if k else self._empty(wp.float32)
                )
                denominator = self.denominator[k * group_count : (k + 1) * group_count]
                if body_block:
                    launch(
                        "block_apply",
                        affine_block_kernels.apply_block_diagonal_preconditioner,
                        count,
                        [
                            self.inverse,
                            self.residual,
                            self.z,
                        ],
                    )
                    launch(
                        "affine_apply",
                        pcg_kernels.apply_affine_body_preconditioner,
                        body_states,
                        [
                            body_indices,
                            self.affine_blocks,
                            self.affine_valid,
                            self.residual,
                            self.z,
                            bodies_per_world,
                            per_world,
                        ],
                    )
                # With the fused PCG the previous iteration's update already preconditioned this residual.
                if not fused_pcg or k == 0:
                    if block:
                        launch(
                            "precondition",
                            pcg_kernels.precondition_and_reduce_block,
                            count,
                            [
                                self.inverse,
                                self.residual,
                                self.z,
                                int(body_block),
                                self.coefficient_world,
                                per_group,
                                numerator,
                            ],
                        )
                    else:
                        launch(
                            "scalar_precondition",
                            pcg_kernels.precondition_and_reduce_scalar,
                            count,
                            [
                                self.scalar_inverse,
                                self.residual,
                                self.z,
                                0,
                                self.coefficient_world,
                                per_group,
                                numerator,
                            ],
                        )
                if fused_pcg:
                    launch(
                        "fused_direction",
                        pcg_kernels.fused_cg_direction,
                        count,
                        [
                            k,
                            self.z,
                            self.coefficient_world,
                            per_group,
                            numerator,
                            previous_numerator,
                            self.direction,
                            self.contact_product if extra_product else empty_v,
                        ],
                    )
                else:
                    launch(
                        "direction",
                        pcg_kernels.update_cg_direction,
                        count,
                        [
                            k,
                            self.z,
                            self.coefficient_world,
                            per_group,
                            numerator,
                            previous_numerator,
                            self.direction,
                        ],
                    )
                    if extra_product:
                        self.contact_product.zero_()
                self._multiply_tangents(
                    v,
                    q_in,
                    inverse_dt,
                    so3_alpha,
                    pd,
                    so3,
                    constitutive,
                    rigidity,
                    coupled_materials,
                    joint_al_active,
                    drag,
                )
                extra = self.contact_product if extra_product else empty_v
                if block_matrix:
                    launch(
                        "fused_multiply" if fused_pcg else "multiply",
                        pcg_kernels.multiply_block_matrix_and_reduce,
                        count,
                        [
                            self.row_count,
                            self.block_entry_columns,
                            self.block_entry_values,
                            self.max_entries,
                            self.blocks,
                            self.direction,
                            extra,
                            self.product,
                            self.coefficient_world,
                            per_group,
                            denominator,
                            count,
                            count,
                        ],
                    )
                elif block:
                    launch(
                        "scalar_block_multiply",
                        pcg_kernels.multiply_scalar_block_matrix_and_reduce,
                        count,
                        [
                            self._template_row_counts(),
                            self.scalar_template_columns,
                            self.scalar_template_values,
                            self.max_entries,
                            self.blocks,
                            self.direction,
                            extra,
                            self.product,
                            self.coefficient_world,
                            per_group,
                            denominator,
                            count,
                            per_world,
                        ],
                    )
                else:
                    launch(
                        "scalar_multiply",
                        pcg_kernels.multiply_scalar_matrix_and_reduce,
                        count,
                        [
                            self._template_row_counts(),
                            self.scalar_template_columns,
                            self.scalar_template_values,
                            self.max_entries,
                            self.diagonal,
                            self.direction,
                            extra,
                            self.product,
                            self.coefficient_world,
                            per_group,
                            denominator,
                            count,
                            per_world,
                        ],
                    )
                if fused_pcg and k + 1 < iterations:
                    next_numerator = self.numerator[(k + 1) * group_count : (k + 2) * group_count]
                    launch(
                        "fused_update",
                        pcg_kernels.fused_update_and_precondition,
                        count,
                        [
                            self.coefficient_world,
                            per_group,
                            numerator,
                            denominator,
                            self.direction,
                            self.product,
                            self.solution,
                            self.residual,
                            self.inverse,
                            self.z,
                            next_numerator,
                        ],
                    )
                else:
                    launch(
                        "update",
                        pcg_kernels.update_solution_and_residual,
                        count,
                        [
                            self.coefficient_world,
                            per_group,
                            numerator,
                            denominator,
                            self.direction,
                            self.product,
                            self.solution,
                            self.residual,
                        ],
                    )
            if self.fluid_rows:
                self._limit_fluid(dt)
            # The native host updates state_in.q in place, even when state_out owns a separate allocation.
            launch("apply", step_kernels.apply_nonlinear_step, count, [q_in, q_out, self.solution])
            if self.cable_count and self._cable_ramping():
                launch(
                    "cable_penalty_update",
                    cable_kernels.update_cable_penalties,
                    self.cable_count,
                    [self._cable_view(), q_out, config.cable_linear_beta, config.cable_angular_beta],
                )
            if hard:
                launch(
                    "hard_update",
                    hard_contact_kernels.update_hard_contact_duals,
                    contact.max_count,
                    [self._hard_view(contact, q_out, inverse_dt)],
                )
            if self.distance_total:
                launch(
                    "distance_update",
                    joint_kernels.update_body_distance_constraint_state,
                    self.distance_total,
                    [
                        q_out,
                        body_indices,
                        v["body_distance_constraint_indices"],
                        v["body_distance_constraint_points0"],
                        v["body_distance_constraint_points1"],
                        self.distance_w0,
                        self.distance_w1,
                        v["body_distance_constraint_rest_lengths"],
                        self.distance_target,
                        self.distance_multiplier,
                        model.body_distance_constraint_count,
                        per_world,
                    ],
                )
            if so3 and so3_alpha > 0.0:
                # The multiplier advances only after iteration c (from 1) == primal iterations, never the last one.
                c = iteration + 1
                advance = c < config.nonlinear_iterations and c == config.affine_body_rigidity_primal_iterations
                launch(
                    "so3_update",
                    affine_kernels.update_affine_body_so3_augmented_lagrangian_state,
                    body_states,
                    [
                        q_out,
                        body_indices,
                        self.body_rotation,
                        self.body_multiplier,
                        int(advance),
                        bodies_per_world,
                        per_world,
                    ],
                )
            # Joint multipliers advance after iterations whose count is a multiple of the primal iteration count,
            # never after the last one.
            c = iteration + 1
            primal = max(config.joint_constraint_primal_iterations, 1)
            joint_advance = c < config.nonlinear_iterations and c % primal == 0
            if joint_al_active and self.joint_spring_total and joint_advance:
                launch(
                    "joint_spring_update",
                    joint_kernels.update_joint_body_spring_augmented_lagrangian_state,
                    self.joint_spring_total,
                    [
                        q_out,
                        body_indices,
                        v["body_spring_indices"],
                        v["body_spring_points0"],
                        v["body_spring_points1"],
                        self.body_spring_w0,
                        self.body_spring_w1,
                        self.joint_multiplier,
                        model.body_spring_count,
                        model.joint_body_spring_count,
                        per_world,
                    ],
                )
            if joint_al_active and self.joint_line_total:
                launch(
                    "joint_line_update",
                    joint_kernels.update_joint_body_point_line_augmented_lagrangian_state,
                    self.joint_line_total,
                    [
                        q_out,
                        body_indices,
                        v["body_point_line_indices"],
                        self.point_line_wp,
                        self.point_line_we0,
                        self.point_line_we1,
                        v["body_point_line_endpoints0"],
                        v["body_point_line_endpoints1"],
                        self.line_segment,
                        self.line_multiplier,
                        self.line_projector,
                        int(joint_advance),
                        model.body_point_line_count,
                        model.joint_body_point_line_count,
                        per_world,
                    ],
                )
            wp.copy(q_in, q_out, count=count)

        if not hard:
            self.hard_previous_count.zero_()
        if not config.nonlinear_iterations:
            wp.copy(q_out, q_in, count=count)
        if self.granular_points:
            self._commit_granular(q_out, dt)
        # Report contact wrenches from the final solved pose before the velocity update.
        if contact.contact_force.size and (rigid or particle_contact):
            wrenches = self._view(contact.contact_force, wp.spatial_vector)
        if contact.contact_force.size and hard:
            self._hard_evaluate(contact, q_in, inverse_dt, contact_force=wrenches[: contact.max_count])
        elif contact.contact_force.size and rigid:
            self._rigid_contact(contact, q_in, qd_in, inverse_dt, contact_force=wrenches[: contact.max_count])
        if contact.contact_force.size and rolling:
            self._launch_rolling(
                "rolling_force",
                rolling_contact_kernels.evaluate_rolling_contact_force,
                contact,
                q_in,
                inverse_dt,
                [self._empty(wp.vec3), wrenches[: contact.max_count]],
            )
        if contact.contact_force.size and surface:
            self._surface_launch(
                "surface_force",
                surface_contact_kernels.evaluate_surface_contact_force,
                contact,
                q_in,
                inverse_dt,
                [self._empty(wp.vec3), wrenches[contact.max_count :]],
            )
        if contact.contact_force.size and particle_contact:
            self._soft_contact(contact, q_in, qd_in, inverse_dt, surface, contact_force=wrenches[contact.max_count :])
        if hard:
            launch(
                "hard_snapshot",
                hard_contact_kernels.snapshot_hard_contacts,
                contact.max_count,
                [self._hard_view(contact, q_in, inverse_dt)],
            )
        launch("velocity", step_kernels.update_velocity, count, [inverse_dt, self.previous, q_out, qd_out])
        if self.drive_total:
            launch(
                "drive_angle_update",
                joint_kernels.update_joint_drive_angle_state,
                self.drive_total,
                [
                    q_out,
                    self.drive_angle,
                    self.drive_initialized,
                    body_indices,
                    v["joint_drive_types"],
                    v["joint_drive_body_indices"],
                    v["joint_drive_parent_origins"],
                    v["joint_drive_parent_axis_points"],
                    v["joint_drive_parent_reference_points"],
                    self.drive_w_po,
                    self.drive_w_co,
                    self.drive_w_pa,
                    self.drive_w_pr,
                    self.drive_w_cr,
                    model.joint_drive_count,
                    per_world,
                ],
            )
        del rows_total

    def _soft_view(self, contact: Contact, x: wp.array, velocity: wp.array, inverse_dt: float, filtered: bool):
        model = self.model
        view = soft_contact_kernels.SoftContactView()
        view.count = self._view(contact.soft_count, wp.int32, 1)
        view.capacity = contact.soft_max_count
        view.particle = self._view(contact.soft_particle, wp.int32)
        view.shape = self._view(contact.soft_shape, wp.int32)
        view.body_pos = self._view(contact.soft_body_pos, wp.vec3)
        view.body_vel = self._view(contact.soft_body_vel, wp.vec3)
        view.normal = self._view(contact.soft_normal, wp.vec3)
        view.radius = self._view(model.particle_radii, wp.float32, self.per_world)
        view.mass = self._view(model.particle_masses, wp.float32, self.per_world)
        view.particle_kind = self.body_kind if self.body_count else self.particle_kind
        view.particles_per_world = self.per_world
        view.total_particles = self.count
        shapes = model.shape_count
        view.shape_body = self._view(model.shape_bodies, wp.int32, shapes)
        view.shape_type = self._view(model.shape_types, wp.int32, shapes)
        view.shape_transform = self._view(model.shape_transforms, wp.transform, shapes)
        view.shape_scale = self._view(model.shape_scales, wp.vec3, shapes)
        view.shape_margin = self._view(model.shape_margins, wp.float32, shapes)
        view.shape_ke = self._view(model.shape_contact_stiffnesses, wp.float32, shapes)
        view.shape_kd = self._view(model.shape_contact_dampings, wp.float32, shapes)
        view.shape_kf = self._view(model.shape_contact_friction_gains, wp.float32, shapes)
        view.shape_mu = self._view(model.shape_frictions, wp.float32, shapes)
        view.local_shape_count = shapes
        view.shapes_per_world = model.shape_replicated_count
        view.worlds = self.count // self.per_world
        view.global_ke = model.soft_contact_stiffness
        view.global_kd = model.soft_contact_damping
        view.global_kf = model.soft_contact_friction_gain
        view.global_mu = model.soft_contact_friction
        if self.body_count:
            view.body_ids = self._view(model.affine_body_ids, wp.int32, self.body_count)
            view.body_indices = self._view(model.affine_body_indices, wp.vec4i, self.body_count)
            view.body_com = self._view(model.body_centers_of_mass, wp.vec3, self.body_count)
        else:
            view.body_ids = self._empty(wp.int32)
            view.body_indices = self._empty(wp.vec4i)
            view.body_com = self._empty(wp.vec3)
        view.x = x
        view.velocity = velocity
        view.previous = self.previous
        view.inv_dt = inverse_dt
        view.shape_filter = (
            self._view(contact.soft_particle_only_shapes, wp.int32) if filtered else self._empty(wp.int32)
        )
        return view

    def _soft_contact(
        self,
        contact,
        x,
        velocity,
        inverse_dt,
        filtered,
        force=None,
        scalar=None,
        system=None,
        precond=None,
        contact_force=None,
    ) -> None:
        self._launch(
            "soft_contact",
            soft_contact_kernels.evaluate_soft_contact_force,
            contact.soft_max_count,
            [
                self._soft_view(contact, x, velocity, inverse_dt, filtered),
                force if force is not None else self._empty(wp.vec3),
                scalar if scalar is not None else self._empty(wp.float32),
                system if system is not None else self._empty(wp.mat33),
                precond if precond is not None else self._empty(wp.mat33),
                contact_force if contact_force is not None else self._empty(wp.spatial_vector),
            ],
        )

    def _rigid_view(self, contact: Contact, x: wp.array, velocity: wp.array, inverse_dt: float):
        model = self.model
        view = rigid_contact_kernels.RigidContactView()
        view.count = self._view(contact.count, wp.int32, 1)
        view.capacity = contact.max_count
        n = contact.max_count
        view.shape0 = self._view(contact.shape0, wp.int32, n)
        view.shape1 = self._view(contact.shape1, wp.int32, n)
        view.point0 = self._view(contact.point0, wp.vec3, n)
        view.point1 = self._view(contact.point1, wp.vec3, n)
        view.offset0 = self._view(contact.offset0, wp.vec3)
        view.offset1 = self._view(contact.offset1, wp.vec3)
        view.normal = self._view(contact.normal, wp.vec3, n)
        view.margin0 = self._view(contact.margin0, wp.float32, n)
        view.margin1 = self._view(contact.margin1, wp.float32, n)
        shapes = model.shape_count
        view.shape_body = self._view(model.shape_bodies, wp.int32, shapes)
        view.shape_type = self._view(model.shape_types, wp.int32, shapes)
        view.transform = self._view(model.shape_transforms, wp.transform, shapes)
        view.shape_scale = self._view(model.shape_scales, wp.vec3, shapes)
        view.ke = self._view(model.shape_contact_stiffnesses, wp.float32, shapes)
        view.kd = self._view(model.shape_contact_dampings, wp.float32, shapes)
        view.kf = self._view(model.shape_contact_friction_gains, wp.float32, shapes)
        view.mu = self._view(model.shape_frictions, wp.float32, shapes)
        view.epsilon = self._config.rigid_contact_friction_epsilon
        view.local_shape_count = shapes
        view.shapes_per_world = model.shape_replicated_count
        view.worlds = self.count // self.per_world
        view.body_ids = self._view(model.affine_body_ids, wp.int32, self.body_count)
        view.body_indices = self._view(model.affine_body_indices, wp.vec4i, self.body_count)
        view.body_com = self._view(model.body_centers_of_mass, wp.vec3, self.body_count)
        view.particles_per_world = self.per_world
        view.x = x
        view.velocity = velocity
        view.previous = self.previous
        view.inv_dt = inverse_dt
        return view

    def _surface_launch(self, key: str, kernel: Any, contact: Contact, x: wp.array, inverse_dt: float, outputs: list):
        model = self.model
        soft = self._soft_view(contact, x, self._empty(wp.vec3), inverse_dt, True)
        view = surface_contact_kernels.SurfaceContactView()
        view.count = soft.count
        view.capacity = contact.soft_max_count
        view.indices = self._view(contact.soft_indices, wp.vec3i, contact.soft_max_count)
        view.barycentric = self._view(contact.soft_barycentric, wp.vec3, contact.soft_max_count)
        for name in (
            "shape",
            "body_pos",
            "body_vel",
            "normal",
            "radius",
            "mass",
            "particles_per_world",
            "total_particles",
            "shape_body",
            "shape_type",
            "shape_scale",
            "shape_margin",
            "shape_ke",
            "shape_kd",
            "shape_kf",
            "shape_mu",
            "local_shape_count",
            "shapes_per_world",
            "worlds",
            "body_ids",
            "body_indices",
            "body_com",
            "x",
            "previous",
            "inv_dt",
        ):
            setattr(view, name, getattr(soft, name))
        view.particle_only = self._view(contact.soft_particle_only_shapes, wp.int32)
        view.flags = self._view(model.particle_flags, wp.int32, self.per_world)
        view.kind = soft.particle_kind
        view.transform = soft.shape_transform
        view.ke, view.kd, view.kf, view.mu = soft.global_ke, soft.global_kd, soft.global_kf, soft.global_mu
        view.stride = self._surface_threads(contact.soft_max_count)
        self._launch(key, kernel, view.stride, [view, *outputs])

    def _surface_threads(self, capacity: int) -> int:
        """Grid cap shared by every surface kernel: max(1, SM count) x 256 threads."""
        return min(capacity, max(1, self.device.sm_count) * 256)

    def _launch_rolling(
        self, key: str, kernel: Any, contact: Contact, x: wp.array, inverse_dt: float, outputs: list
    ) -> None:
        model = self.model
        view = rolling_contact_kernels.RollingContactView()
        rigid = self._rigid_view(contact, x, self._empty(wp.vec3), inverse_dt)
        for name in (
            "count",
            "capacity",
            "shape0",
            "shape1",
            "point0",
            "point1",
            "offset0",
            "offset1",
            "normal",
            "margin0",
            "margin1",
            "shape_body",
            "shape_type",
            "transform",
            "shape_scale",
            "ke",
            "kd",
            "local_shape_count",
            "shapes_per_world",
            "worlds",
            "body_ids",
            "body_indices",
            "body_com",
            "particles_per_world",
            "x",
            "previous",
            "inv_dt",
        ):
            setattr(view, name, getattr(rigid, name))
        shapes = model.shape_count
        view.rolling = self._view(model.shape_rolling_frictions, wp.float32, shapes)
        view.torsional = self._view(model.shape_torsional_frictions, wp.float32, shapes)
        view.inertia = self._view(model.body_inertia_tensors, wp.float32, 9 * self.body_count)
        view.epsilon = self._config.rolling_friction_velocity_epsilon
        self._launch(key, kernel, contact.max_count, [view, *outputs])

    def _hard_view(self, contact: Contact, x: wp.array, inverse_dt: float):
        rigid = self._rigid_view(contact, x, self._empty(wp.vec3), inverse_dt)
        view = hard_contact_kernels.HardContactView()
        for name in (
            "count",
            "capacity",
            "shape0",
            "shape1",
            "point0",
            "point1",
            "offset0",
            "offset1",
            "normal",
            "margin0",
            "margin1",
            "shape_body",
            "shape_type",
            "shape_scale",
            "local_shape_count",
            "shapes_per_world",
            "worlds",
            "body_ids",
            "body_indices",
            "body_com",
            "particles_per_world",
            "x",
            "previous",
            "inv_dt",
        ):
            setattr(view, name, getattr(rigid, name))
        view.shape_transform = rigid.transform
        view.shape_ke = rigid.ke
        view.shape_kd = rigid.kd
        view.shape_mu = rigid.mu
        view.match = self._view(contact.match_index, wp.int32, contact.max_count)
        view.body_count = self.body_count
        config = self._config
        view.alpha = config.rigid_contact_alpha
        view.beta = config.rigid_contact_beta
        view.gamma = config.rigid_contact_gamma
        view.start = config.rigid_contact_k_start
        view.history = int(config.rigid_contact_history)
        view.exact_friction = int(config.rigid_contact_sliding_tangent)
        view.reset = int(contact.history_refresh)
        view.penalty = self.hard_penalty
        view.dual = self.hard_dual
        view.anchor = self.hard_anchor
        view.previous_penalty = self.hard_previous_penalty
        view.previous_dual = self.hard_previous_dual
        view.previous_normal = self.hard_previous_normal
        view.previous_count = self.hard_previous_count
        return view

    def _hard_evaluate(
        self,
        contact,
        x,
        inverse_dt,
        force=None,
        scalar=None,
        direction=None,
        product=None,
        system=None,
        precond=None,
        affine_blocks=None,
        contact_force=None,
    ) -> None:
        def arg(value, dtype):
            return value if value is not None else self._empty(dtype)

        self._launch(
            "hard_evaluate",
            hard_contact_kernels.evaluate_hard_contacts,
            contact.max_count,
            [
                self._hard_view(contact, x, inverse_dt),
                arg(force, wp.vec3),
                arg(scalar, wp.float32),
                arg(direction, wp.vec3),
                arg(product, wp.vec3),
                arg(system, wp.mat33),
                arg(precond, wp.mat33),
                arg(affine_blocks, wp.float32),
                arg(contact_force, wp.spatial_vector),
            ],
        )

    def _rigid_contact(self, contact, x, velocity, inverse_dt, force=None, scalar=None, contact_force=None) -> None:
        self._launch(
            "rigid_force",
            rigid_contact_kernels.evaluate_contact_force,
            contact.max_count,
            [
                self._rigid_view(contact, x, velocity, inverse_dt),
                force if force is not None else self._empty(wp.vec3),
                scalar if scalar is not None else self._empty(wp.float32),
                contact_force if contact_force is not None else self._empty(wp.spatial_vector),
            ],
        )

    def _rigid_tangent_cache(self) -> Any:
        cache = rigid_contact_kernels.RigidTangentCache()
        cache.targets = self.rigid_tangent_targets
        cache.normal_weights = self.rigid_tangent_normal_weights
        cache.friction_weights = self.rigid_tangent_friction_weights
        cache.entries = self.rigid_tangent_entries
        cache.normal = self.rigid_tangent_normal
        cache.stiffness = self.rigid_tangent_stiffness
        return cache

    def _rigid_tangent(self, contact, x, inverse_dt, direction=None, product=None, system=None, precond=None):
        self._launch(
            "rigid_tangent",
            rigid_contact_kernels.evaluate_contact_tangent,
            contact.max_count,
            [
                self._rigid_view(contact, x, self._empty(wp.vec3), inverse_dt),
                direction if direction is not None else self._empty(wp.vec3),
                product if product is not None else self._empty(wp.vec3),
                system if system is not None else self._empty(wp.mat33),
                precond if precond is not None else self._empty(wp.mat33),
            ],
        )

    def _soft_contact_product(self, contact, x, velocity, inverse_dt, filtered) -> None:
        self._launch(
            "soft_contact_product",
            soft_contact_kernels.multiply_soft_contact_hessian_off_diagonal,
            contact.soft_max_count,
            [self._soft_view(contact, x, velocity, inverse_dt, filtered), self.direction, self.contact_product],
        )

    def _articulated_materials(self) -> bool:
        """Only joint-owned constraints share the mixed cloth/tetrahedron schedule of the robot demos."""
        m = self.model
        return bool(
            m.body_count
            and (m.triangle_count or m.tetrahedron_count)
            and m.joint_drive_count
            and m.joint_body_spring_count
            and m.body_spring_count == m.joint_body_spring_count
            and m.body_point_line_count == m.joint_body_point_line_count
            and not m.body_distance_constraint_count
            and not m.spring_count
            and not m.cable_count
            and not m.fluid_particle_count
            and not m.granular_particle_count
        )

    _MODEL_SPANS: ClassVar[dict[str, tuple[Any, str]]] = {
        "affine_body_indices": (wp.vec4i, "body_count"),
        "body_masses": (wp.float32, "body_count"),
        "particle_mass_non_diagonal_values": (wp.float32, "particle_mass_non_diagonal_count"),
        "spring_indices": (wp.vec2i, "spring_count"),
        "spring_stiffnesses": (wp.float32, "spring_count"),
        "spring_rest_lengths": (wp.float32, "spring_count"),
        "triangle_indices": (wp.vec3i, "triangle_count"),
        "triangle_inv_dms": (wp.mat22, "triangle_count"),
        "triangle_rest_areas": (wp.float32, "triangle_count"),
        "triangle_aniso_stiffnesses": (wp.vec3, "triangle_count"),
        "edge_indices": (wp.vec4i, "edge_count"),
        "edge_rest_areas": (wp.float32, "edge_count"),
        "edge_bending_cots": (wp.vec4, "edge_count"),
        "edge_bending_properties": (wp.vec2, "edge_count"),
        "tetrahedron_indices": (wp.vec4i, "tetrahedron_count"),
        "tetrahedron_inverse_dms": (wp.mat33, "tetrahedron_count"),
        "tetrahedron_materials": (wp.vec3, "tetrahedron_count"),
        "body_spring_indices": (wp.vec2i, "body_spring_count"),
        "body_spring_points0": (wp.vec3, "body_spring_count"),
        "body_spring_points1": (wp.vec3, "body_spring_count"),
        "body_spring_stiffnesses": (wp.float32, "body_spring_count"),
        "body_spring_rest_lengths": (wp.float32, "body_spring_count"),
        "body_distance_constraint_indices": (wp.vec2i, "body_distance_constraint_count"),
        "body_distance_constraint_points0": (wp.vec3, "body_distance_constraint_count"),
        "body_distance_constraint_points1": (wp.vec3, "body_distance_constraint_count"),
        "body_distance_constraint_mass_scales": (wp.float32, "body_distance_constraint_count"),
        "body_distance_constraint_rest_lengths": (wp.float32, "body_distance_constraint_count"),
        "body_point_line_indices": (wp.vec2i, "body_point_line_count"),
        "body_point_line_endpoints0": (wp.vec3, "body_point_line_count"),
        "body_point_line_endpoints1": (wp.vec3, "body_point_line_count"),
        "body_point_line_stiffnesses": (wp.float32, "body_point_line_count"),
        "body_point_line_dampings": (wp.float32, "body_point_line_count"),
        "joint_drive_types": (wp.int32, "joint_drive_count"),
        "joint_drive_body_indices": (wp.vec2i, "joint_drive_count"),
        "joint_drive_parent_origins": (wp.vec3, "joint_drive_count"),
        "joint_drive_parent_axis_points": (wp.vec3, "joint_drive_count"),
        "joint_drive_parent_reference_points": (wp.vec3, "joint_drive_count"),
        "joint_drive_dof_indices": (wp.int32, "joint_drive_count"),
        "joint_drive_target_indices": (wp.int32, "joint_drive_count"),
        "joint_target_stiffnesses": (wp.float32, "joint_dof_count"),
        "joint_target_dampings": (wp.float32, "joint_dof_count"),
        "joint_effort_limits": (wp.float32, "joint_dof_count"),
        "joint_target_modes": (wp.int32, "joint_dof_count"),
        "joint_passive_dampings": (wp.float32, "joint_dof_count"),
        "joint_armature": (wp.float32, "joint_dof_count"),
        "joint_limit_lower": (wp.float32, "joint_dof_count"),
        "joint_limit_upper": (wp.float32, "joint_dof_count"),
        "joint_limit_stiffnesses": (wp.float32, "joint_dof_count"),
        "joint_limit_dampings": (wp.float32, "joint_dof_count"),
    }

    def _model_views(self) -> dict[str, wp.array]:
        """Typed views of the model spans used by the step (empty arrays for absent spans)."""
        model = self.model
        return {
            name: self._view(getattr(model, name), dtype, getattr(model, count_field) or None)
            for name, (dtype, count_field) in self._MODEL_SPANS.items()
        }

    def _joint_view(self, v: dict[str, wp.array]) -> Any:
        view = joint_kernels.JointCoordinateView()
        view.body_particles = v["affine_body_indices"]
        view.types = v["joint_drive_types"]
        view.bodies = v["joint_drive_body_indices"]
        view.parent_origin = v["joint_drive_parent_origins"]
        view.parent_axis = v["joint_drive_parent_axis_points"]
        view.parent_reference = v["joint_drive_parent_reference_points"]
        view.w_po = self.drive_w_po
        view.w_co = self.drive_w_co
        view.w_pa = self.drive_w_pa
        view.w_pr = self.drive_w_pr
        view.w_cr = self.drive_w_cr
        view.dofs = v["joint_drive_dof_indices"]
        view.joints_per_world = self.model.joint_drive_count
        view.particles_per_world = self.per_world
        return view

    def _limit_view(self, v: dict[str, wp.array]) -> Any:
        view = joint_kernels.AngularLimitView()
        view.lower = v["joint_limit_lower"]
        view.upper = v["joint_limit_upper"]
        view.stiffness = v["joint_limit_stiffnesses"]
        view.damping = v["joint_limit_dampings"]
        return view

    def _joint_armature(self, v, x, inverse_dt, direction, force, block) -> None:
        self._launch(
            "joint_armature",
            joint_kernels.evaluate_joint_armature,
            self.drive_total,
            [
                self._joint_view(v),
                v["joint_armature"],
                x,
                self.previous,
                self.drive_qd,
                inverse_dt,
                direction,
                force,
                block,
            ],
        )

    def _angular_limits(self, v, x, inverse_dt, direction, force, block) -> None:
        self._launch(
            "angular_limits",
            joint_kernels.evaluate_angular_limits,
            self.drive_total,
            [
                self._joint_view(v),
                self._limit_view(v),
                x,
                self.previous,
                self.drive_angle,
                self.drive_initialized,
                inverse_dt,
                direction,
                force,
                block,
            ],
        )

    def _assemble_blocks(
        self,
        v,
        q_in,
        mass,
        flags,
        inverse_dt,
        so3_alpha,
        body_so3_alpha,
        pd,
        block_matrix,
        body_block,
        so3,
        six_springs,
        constitutive,
        rigidity,
        joint_al,
        joint_al_active,
        drag,
    ) -> None:
        """Diagonal blocks, block-ELL entries and the block preconditioner, in the native launch order."""
        model, launch, count, per_world = self.model, self._launch, self.count, self.per_world
        empty_m = self._empty(wp.mat33)
        body_indices, body_masses = v["affine_body_indices"], v["body_masses"]
        body_states, bodies_per_world = self.body_state_count, self.body_count
        ell = [self.row_count, self.block_entry_columns, self.block_entry_values, self.max_entries]
        launch(
            "blocks",
            step_kernels.initialize_block_diagonal,
            count,
            [
                self.diagonal,
                self.blocks,
                empty_m if block_matrix else self.preconditioner_blocks,
            ],
        )
        if block_matrix and self.max_entries:
            wp.copy(self.block_entry_columns, self.block_template_columns)
            wp.copy(self.block_entry_values, self.block_template_values)
        if not pd and self.triangle_count:
            launch(
                "membrane_hessian",
                cloth_kernels.add_cloth_membrane_gauss_newton_hessian,
                self.triangle_count,
                [
                    q_in,
                    v["triangle_indices"],
                    v["triangle_inv_dms"],
                    v["triangle_rest_areas"],
                    v["triangle_aniso_stiffnesses"],
                    self.blocks,
                    *ell,
                    model.triangle_count,
                    per_world,
                    count,
                ],
            )
        if not pd and self.spring_count:
            launch(
                "spring_hessian",
                spring_kernels.add_spring_gauss_newton_hessian,
                self.spring_count,
                [
                    q_in,
                    v["spring_indices"],
                    v["spring_stiffnesses"],
                    v["spring_rest_lengths"],
                    self.blocks,
                    *ell,
                    model.spring_count,
                    per_world,
                    count,
                ],
            )
        if block_matrix and six_springs and so3_alpha > 0.0:
            launch(
                "rigidity_spring_hessian",
                affine_kernels.add_affine_body_spring_gauss_newton_hessian,
                body_states,
                [
                    q_in,
                    body_indices,
                    body_masses,
                    so3_alpha,
                    self.blocks,
                    *ell,
                    bodies_per_world,
                    per_world,
                    count,
                ],
            )
        if not pd and self.plain_spring_count:
            launch(
                "body_spring_hessian",
                joint_kernels.add_body_spring_gauss_newton_hessian,
                self.body_spring_total,
                [
                    q_in,
                    body_indices,
                    v["body_spring_indices"],
                    v["body_spring_points0"],
                    v["body_spring_points1"],
                    self.body_spring_w0,
                    self.body_spring_w1,
                    v["body_spring_stiffnesses"],
                    v["body_spring_rest_lengths"],
                    self.blocks,
                    *ell,
                    model.body_spring_count,
                    per_world,
                    count,
                ],
            )
        if not pd and (self.plain_line_count or (not joint_al and self.joint_line_total)):
            launch(
                "point_line_hessian",
                joint_kernels.add_body_point_line_gauss_newton_hessian,
                self.point_line_total,
                [
                    q_in,
                    body_indices,
                    v["body_point_line_indices"],
                    self.point_line_wp,
                    self.point_line_we0,
                    self.point_line_we1,
                    v["body_point_line_endpoints0"],
                    v["body_point_line_endpoints1"],
                    v["body_point_line_stiffnesses"],
                    so3_alpha,
                    model.joint_body_point_line_count,
                    int(not joint_al or not self.joint_line_total),
                    self.blocks,
                    *ell,
                    model.body_point_line_count,
                    per_world,
                    count,
                ],
            )
        if not pd and joint_al_active and self.joint_line_total:
            launch(
                "joint_line_hessian",
                joint_kernels.add_joint_body_point_line_gauss_newton_hessian,
                self.joint_line_total,
                [
                    body_indices,
                    v["body_point_line_indices"],
                    self.point_line_wp,
                    self.point_line_we0,
                    self.point_line_we1,
                    v["body_point_line_stiffnesses"],
                    self.line_segment,
                    self.line_projector,
                    so3_alpha,
                    self.blocks,
                    *ell,
                    model.body_point_line_count,
                    model.joint_body_point_line_count,
                    per_world,
                    count,
                ],
            )
        # The native solver retains A and inverts its separate cache in place.
        if block_matrix:
            wp.copy(self.preconditioner_blocks, self.blocks, count=count)
        if self._contact_block:
            self.affine_blocks.zero_()
        if self._hard:
            self._hard_evaluate(
                self._contact,
                q_in,
                inverse_dt,
                system=self.blocks,
                precond=self.preconditioner_blocks,
                affine_blocks=self.affine_blocks if self._contact_block else None,
            )
        elif self._rigid and self._config.enable_collision_hessian_non_diagonal:
            self._rigid_tangent(self._contact, q_in, inverse_dt, system=self.blocks, precond=self.preconditioner_blocks)
        if self._particle_contact and (not pd or self._config.enable_collision_hessian_non_diagonal):
            self._soft_contact(
                self._contact,
                q_in,
                self._qd_in,
                inverse_dt,
                self._surface,
                system=self.blocks,
                precond=self.preconditioner_blocks,
            )
        # Without the full body block, SO(3) precedes tetrahedra/bending.
        if constitutive and so3_alpha > 0.0:
            launch(
                "constitutive_preconditioner",
                affine_kernels.add_affine_body_constitutive_preconditioner,
                body_states,
                [
                    q_in,
                    body_indices,
                    body_masses,
                    so3_alpha,
                    int(rigidity),
                    self.preconditioner_blocks,
                    bodies_per_world,
                    per_world,
                ],
            )
        if so3 and not body_block and so3_alpha > 0.0:
            launch(
                "so3_preconditioner",
                affine_kernels.add_affine_body_so3_augmented_lagrangian_preconditioner,
                body_states,
                [
                    body_indices,
                    body_masses,
                    self.body_rotation,
                    so3_alpha,
                    self.preconditioner_blocks,
                    bodies_per_world,
                    per_world,
                ],
            )
        if self.tetrahedron_count:
            launch(
                "tetrahedron_preconditioner",
                tetrahedron_kernels.add_tetrahedron_preconditioner,
                self.tetrahedron_count,
                [
                    q_in,
                    v["tetrahedron_indices"],
                    v["tetrahedron_inverse_dms"],
                    v["tetrahedron_materials"],
                    self.tetrahedron_volumes,
                    inverse_dt,
                    self.preconditioner_blocks,
                    model.tetrahedron_count,
                    per_world,
                ],
            )
        if self.bending_damping:
            launch(
                "bending_preconditioner",
                cloth_kernels.add_bending_damping_preconditioner,
                self.edge_count,
                [
                    q_in,
                    v["edge_rest_areas"],
                    v["edge_bending_cots"],
                    v["edge_indices"],
                    v["edge_bending_properties"],
                    inverse_dt,
                    self.preconditioner_blocks,
                    model.edge_count,
                    per_world,
                ],
            )
        # Self Hessians precede the fluid and surface contributions.
        if self._self:
            launch(
                "self_preconditioner",
                self_collision_kernels.add_hessian_diagonal,
                count,
                [self.self_diagonal, self.preconditioner_blocks],
            )
        if self.fluid_rows:
            launch(
                "fluid_12add_diagonal",
                fluid_kernels.add_diagonal,
                count,
                [self.fluid_diagonal, self.preconditioner_blocks],
            )
        if self.granular_points:
            launch(
                "granular_12add_diagonal",
                granular_kernels.add_diagonal,
                count,
                [self.granular_diagonal, self.preconditioner_blocks],
            )
        if self._surface:
            self._surface_launch(
                "surface_preconditioner",
                surface_contact_kernels.add_surface_contact_preconditioner,
                self._contact,
                q_in,
                inverse_dt,
                [self.preconditioner_blocks],
            )
        # Joint-line PD and drive terms precede the body-block assembly.
        if pd and joint_al_active and self.joint_line_total and so3_alpha > 0.0:
            launch(
                "joint_line_pd_preconditioner",
                joint_kernels.add_joint_body_point_line_augmented_lagrangian_preconditioner,
                self.joint_line_total,
                [
                    body_indices,
                    v["body_point_line_indices"],
                    self.point_line_wp,
                    self.point_line_we0,
                    self.point_line_we1,
                    v["body_point_line_stiffnesses"],
                    self.line_segment,
                    self.line_projector,
                    so3_alpha,
                    self.preconditioner_blocks,
                    model.body_point_line_count,
                    model.joint_body_point_line_count,
                    per_world,
                ],
            )
        if self.drive_total:
            launch(
                "drive_preconditioner",
                joint_kernels.add_joint_drive_preconditioner,
                self.drive_total,
                [
                    q_in,
                    body_indices,
                    v["joint_drive_types"],
                    v["joint_drive_body_indices"],
                    v["joint_drive_parent_origins"],
                    v["joint_drive_parent_axis_points"],
                    v["joint_drive_parent_reference_points"],
                    self.drive_w_po,
                    self.drive_w_co,
                    self.drive_w_pa,
                    self.drive_w_pr,
                    self.drive_w_cr,
                    v["joint_drive_dof_indices"],
                    v["joint_target_modes"],
                    v["joint_passive_dampings"],
                    v["joint_target_dampings"],
                    inverse_dt,
                    self.preconditioner_blocks,
                    model.joint_drive_count,
                    per_world,
                ],
            )
            if self.armature:
                self._joint_armature(
                    v, q_in, inverse_dt, self._empty(wp.vec3), self._empty(wp.vec3), self.preconditioner_blocks
                )
            if self.angular_limits:
                self._angular_limits(
                    v, q_in, inverse_dt, self._empty(wp.vec3), self._empty(wp.vec3), self.preconditioner_blocks
                )
        if body_block:
            if self._rolling:
                self._launch_rolling(
                    "rolling_preconditioner",
                    rolling_contact_kernels.add_rolling_contact_preconditioner,
                    self._contact,
                    q_in,
                    inverse_dt,
                    [self.preconditioner_blocks],
                )
            # The body assembly adds SO(3) itself, before the per-particle copies do.
            if block_matrix:
                launch(
                    "affine_build",
                    affine_block_kernels.build_affine_body_preconditioner_block,
                    body_states,
                    [
                        body_indices,
                        body_masses,
                        mass,
                        flags,
                        self.preconditioner_blocks,
                        self.row_count,
                        self.block_entry_columns,
                        self.block_entry_values,
                        self.max_entries,
                        count,
                        self.body_rotation,
                        body_so3_alpha,
                        self._cable_view(),
                        self.affine_cable_offsets,
                        self.affine_cable_ids,
                        inverse_dt,
                        self.affine_enabled,
                        self.affine_blocks if self._contact_block else self._empty(wp.float32),
                        self.affine_blocks,
                        self.affine_valid,
                        bodies_per_world,
                        per_world,
                    ],
                )
            else:
                launch(
                    "affine_build_scalar",
                    affine_block_kernels.build_affine_body_preconditioner_scalar,
                    body_states,
                    [
                        body_indices,
                        body_masses,
                        mass,
                        flags,
                        self.preconditioner_blocks,
                        self._template_row_counts(),
                        self.scalar_template_columns,
                        self.scalar_template_values,
                        self.max_entries,
                        per_world,
                        self.body_rotation,
                        body_so3_alpha,
                        self._cable_view(),
                        self.affine_cable_offsets,
                        self.affine_cable_ids,
                        inverse_dt,
                        self.affine_enabled,
                        self.affine_blocks if self._contact_block else self._empty(wp.float32),
                        self.affine_blocks,
                        self.affine_valid,
                        bodies_per_world,
                        per_world,
                    ],
                )
        if so3 and body_block and so3_alpha > 0.0:
            launch(
                "so3_preconditioner",
                affine_kernels.add_affine_body_so3_augmented_lagrangian_preconditioner,
                body_states,
                [
                    body_indices,
                    body_masses,
                    self.body_rotation,
                    so3_alpha,
                    self.preconditioner_blocks,
                    bodies_per_world,
                    per_world,
                ],
            )
        if self.cable_count:
            self._evaluate_cables(q_in, inverse_dt, diagonal=self.preconditioner_blocks)
        if self._rolling and not body_block:
            self._launch_rolling(
                "rolling_preconditioner",
                rolling_contact_kernels.add_rolling_contact_preconditioner,
                self._contact,
                q_in,
                inverse_dt,
                [self.preconditioner_blocks],
            )
        if drag:
            launch(
                "drag_preconditioner",
                drag_kernels.add_triangle_drag_preconditioner,
                self.drag_count,
                [
                    self.drag_indices,
                    self.drag_normal,
                    self.drag_scalar,
                    self.preconditioner_blocks,
                    self.drag_local_count,
                    per_world,
                ],
            )
        launch(
            "inverse",
            pcg_kernels.invert_block_preconditioner,
            count,
            [
                self.preconditioner_blocks,
                mass,
                flags,
                self.inverse,
                per_world,
            ],
        )

    def _multiply_tangents(
        self,
        v,
        q_in,
        inverse_dt,
        so3_alpha,
        pd,
        so3,
        constitutive,
        rigidity,
        coupled_materials,
        joint_al_active,
        drag,
    ) -> None:
        """Matrix-free tangent products added to contact_product, in the native order."""
        model, launch, per_world = self.model, self._launch, self.per_world
        body_indices, body_masses = v["affine_body_indices"], v["body_masses"]
        body_states, bodies_per_world = self.body_state_count, self.body_count
        product = self.contact_product
        joint_line_pd = [
            self.direction,
            body_indices,
            v["body_point_line_indices"],
            self.point_line_wp,
            self.point_line_we0,
            self.point_line_we1,
            v["body_point_line_stiffnesses"],
            self.line_segment,
            self.line_projector,
            so3_alpha,
            product,
            model.body_point_line_count,
            model.joint_body_point_line_count,
            per_world,
        ]
        # Coupled-material schedules add the joint-line PD tangent before SO(3).
        if pd and coupled_materials and self.joint_line_total and so3_alpha > 0.0:
            launch(
                "joint_line_pd_tangent",
                joint_kernels.multiply_joint_body_point_line_augmented_lagrangian,
                self.joint_line_total,
                joint_line_pd,
            )
        if so3 and so3_alpha > 0.0:
            launch(
                "so3_tangent",
                affine_kernels.multiply_affine_body_so3_augmented_lagrangian_tangent,
                body_states,
                [
                    self.direction,
                    body_indices,
                    body_masses,
                    self.body_rotation,
                    so3_alpha,
                    product,
                    bodies_per_world,
                    per_world,
                ],
            )
        if constitutive and so3_alpha > 0.0:
            launch(
                "constitutive_tangent",
                affine_kernels.multiply_affine_body_constitutive_tangent,
                body_states,
                [
                    q_in,
                    self.direction,
                    body_indices,
                    body_masses,
                    so3_alpha,
                    int(rigidity),
                    product,
                    bodies_per_world,
                    per_world,
                ],
            )
        if self.tetrahedron_count:
            launch(
                "tetrahedron_tangent",
                tetrahedron_kernels.multiply_tetrahedron_tangent,
                self.tetrahedron_count,
                [
                    q_in,
                    self.direction,
                    v["tetrahedron_indices"],
                    v["tetrahedron_inverse_dms"],
                    v["tetrahedron_materials"],
                    self.tetrahedron_volumes,
                    inverse_dt,
                    product,
                    model.tetrahedron_count,
                    per_world,
                ],
            )
        if self.bending_damping:
            launch(
                "bending_tangent",
                cloth_kernels.multiply_bending_damping_tangent,
                self.edge_count,
                [
                    q_in,
                    self.direction,
                    v["edge_rest_areas"],
                    v["edge_bending_cots"],
                    v["edge_indices"],
                    v["edge_bending_properties"],
                    inverse_dt,
                    product,
                    model.edge_count,
                    per_world,
                ],
            )
        if self._self:
            self._multiply_self()
        if self._cached_surface:
            launch(
                "surface_tangent_cached",
                surface_contact_kernels.multiply_cached_surface_contact_tangent,
                self._surface_threads(self._contact.soft_max_count),
                [
                    self._view(self._contact.soft_count, wp.int32, 1),
                    self._contact.soft_max_count,
                    self._surface_threads(self._contact.soft_max_count),
                    self.surface_tangent_cache,
                    self.direction,
                    product,
                ],
            )
        elif self._surface:
            self._surface_launch(
                "surface_tangent",
                surface_contact_kernels.multiply_surface_contact_tangent,
                self._contact,
                q_in,
                inverse_dt,
                [self.direction, product],
            )
        if self.fluid_rows:
            launch(
                "fluid_13multiply_rows",
                fluid_kernels.multiply_rows,
                self.fluid_rows,
                [
                    self._fluid_view(self._dt),
                    self.fluid_gradients,
                    self.fluid_weights,
                    self.direction,
                    product,
                ],
            )
        if self.granular_points:
            self._multiply_granular(self._dt, self.direction, product)
        if self._hard:
            self._hard_evaluate(self._contact, q_in, inverse_dt, direction=self.direction, product=self.contact_product)
        elif self._rigid and self._config.enable_collision_hessian_non_diagonal:
            launch(
                "rigid_tangent",
                rigid_contact_kernels.multiply_cached_contact_tangent,
                self._contact.max_count,
                [
                    self._view(self._contact.count, wp.int32, 1),
                    self._contact.max_count,
                    self._rigid_tangent_cache(),
                    self.direction,
                    self.contact_product,
                ],
            )
        if self._contact_product:
            self._soft_contact_product(self._contact, q_in, self._qd_in, inverse_dt, self._surface)
        # The other PD joint-line schedules add it after the contact tangents, ahead of the drive terms.
        if pd and not coupled_materials and joint_al_active and self.joint_line_total and so3_alpha > 0.0:
            launch(
                "joint_line_pd_tangent",
                joint_kernels.multiply_joint_body_point_line_augmented_lagrangian,
                self.joint_line_total,
                joint_line_pd,
            )
        if self.drive_total:
            launch(
                "drive_tangent",
                joint_kernels.multiply_joint_drive_tangent,
                self.drive_total,
                [
                    q_in,
                    self.direction,
                    body_indices,
                    v["joint_drive_types"],
                    v["joint_drive_body_indices"],
                    v["joint_drive_parent_origins"],
                    v["joint_drive_parent_axis_points"],
                    v["joint_drive_parent_reference_points"],
                    self.drive_w_po,
                    self.drive_w_co,
                    self.drive_w_pa,
                    self.drive_w_pr,
                    self.drive_w_cr,
                    v["joint_drive_dof_indices"],
                    v["joint_target_modes"],
                    v["joint_passive_dampings"],
                    v["joint_target_dampings"],
                    inverse_dt,
                    product,
                    model.joint_drive_count,
                    per_world,
                ],
            )
            if self.armature:
                self._joint_armature(v, q_in, inverse_dt, self.direction, product, self._empty(wp.mat33))
            if self._joint_cable_step and self.cable_count:
                self._evaluate_cables(q_in, inverse_dt, direction=self.direction, force=product)
            if self.angular_limits:
                self._angular_limits(v, q_in, inverse_dt, self.direction, product, self._empty(wp.mat33))
        if self.cable_count and not self._joint_cable_step:
            self._evaluate_cables(q_in, inverse_dt, direction=self.direction, force=product)
        if self._rolling:
            self._launch_rolling(
                "rolling_tangent",
                rolling_contact_kernels.multiply_rolling_contact_tangent,
                self._contact,
                q_in,
                inverse_dt,
                [self.direction, self.contact_product],
            )
        if drag:
            launch(
                "drag_tangent",
                drag_kernels.multiply_triangle_drag_tangent,
                self.drag_count,
                [
                    self.direction,
                    self.drag_indices,
                    self.drag_normal,
                    self.drag_scalar,
                    product,
                    self.drag_local_count,
                    per_world,
                ],
            )

    def _validate_step_config(self) -> None:
        config = self._config

        def finite(value: float) -> bool:
            return math.isfinite(value)

        if int(config.nonlinear_solver) > int(NonlinearSolver.PD_NEWTON):
            raise ValueError("config.nonlinear_solver is invalid")
        if int(config.joint_constraint_mode) > int(JointConstraintMode.AUGMENTED_LAGRANGIAN):
            raise ValueError("config.joint_constraint_mode is invalid")
        if int(config.affine_body_rigidity_model) > int(AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN):
            raise ValueError("config.affine_body_rigidity_model is invalid")
        if not 0 <= config.nonlinear_iterations <= _MAX_LINEAR_ITERATIONS:
            raise ValueError("config.nonlinear_iterations must be between 0 and 999")
        if not 1 <= config.linear_iterations <= _MAX_LINEAR_ITERATIONS:
            raise ValueError("config.linear_iterations must be between 1 and 999")
        if not finite(config.constraint_alpha) or config.constraint_alpha < 0.0:
            raise ValueError("config.constraint_alpha must be finite and non-negative")
        if not 1 <= config.joint_constraint_primal_iterations <= _MAX_LINEAR_ITERATIONS:
            raise ValueError("config.joint_constraint_primal_iterations must be between 1 and 999")
        if not 1 <= config.affine_body_rigidity_primal_iterations <= _MAX_LINEAR_ITERATIONS:
            raise ValueError("config.affine_body_rigidity_primal_iterations must be between 1 and 999")
        # Validated even when self-collision is disabled or unavailable, as natively.
        if not finite(config.self_collision_radius) or config.self_collision_radius <= 0.0:
            raise ValueError("config.self_collision_radius must be finite and positive")
        for value in (
            config.self_collision_vertex_face_stiffness,
            config.self_collision_edge_edge_stiffness,
            config.self_collision_edge_face_stiffness,
            config.self_collision_particle_particle_stiffness,
        ):
            if not finite(value) or value < 0.0:
                raise ValueError("self-collision stiffness factors must be finite and non-negative")
        if not finite(config.rigid_contact_friction_epsilon) or config.rigid_contact_friction_epsilon < 0.0:
            raise ValueError("rigid_contact_friction_epsilon must be finite and non-negative")
        if not finite(config.rigid_contact_alpha) or not 0.0 <= config.rigid_contact_alpha <= 1.0:
            raise ValueError("rigid_contact_alpha must be finite and in [0, 1]")
        if not finite(config.rigid_contact_gamma) or not 0.0 <= config.rigid_contact_gamma <= 1.0:
            raise ValueError("rigid_contact_gamma must be finite and in [0, 1]")
        if (
            not finite(config.rigid_contact_beta)
            or config.rigid_contact_beta < 0.0
            or not finite(config.rigid_contact_k_start)
            or config.rigid_contact_k_start < 0.0
        ):
            raise ValueError("hard-contact penalty ramp and seed must be finite and non-negative")
        self._validate_cable_config()
        self._validate_fluid_config()

    def _validate_fluid_config(self) -> None:
        config = self._config
        if self.fluid_rows and (
            not math.isfinite(config.fluid_relaxation)
            or config.fluid_relaxation <= 0.0
            or config.fluid_relaxation > 1.0
            or not math.isfinite(config.fluid_max_step_ratio)
            or config.fluid_max_step_ratio < 0.0
        ):
            raise ValueError("fluid relaxation must be in (0, 1] and maximum step ratio finite and non-negative")

    def _validate_cable_config(self) -> None:
        config = self._config
        if not math.isfinite(config.cable_penalty_gamma) or not 0.0 <= config.cable_penalty_gamma <= 1.0:
            raise ValueError("cable_penalty_gamma must be finite and in [0, 1]")
        for value in (
            config.cable_linear_beta,
            config.cable_angular_beta,
            config.cable_linear_k_start,
            config.cable_angular_k_start,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("cable penalty ramps and seeds must be finite and non-negative")

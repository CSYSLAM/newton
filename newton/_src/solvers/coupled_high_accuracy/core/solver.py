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
from collections import Counter
from contextlib import nullcontext
from typing import Any

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
from .kernels import drag as drag_kernels
from .kernels import pcg as pcg_kernels
from .kernels import springs as spring_kernels
from .kernels import step as step_kernels

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
    wp.vec3: (np.float32, (3,)),
    wp.vec4: (np.float32, (4,)),
    wp.mat22: (np.float32, (2, 2)),
    wp.mat33: (np.float32, (3, 3)),
    wp.transform: (np.float32, (7,)),
}


def _not_ported(feature: str) -> NotImplementedError:
    return NotImplementedError(f"The Warp solver core does not support {feature} yet (NEWTON_CHA_BACKEND=native).")


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


class Solver:
    """Implicit Newton solver for particles, cloth, soft bodies and affine rigid bodies."""

    def __init__(self) -> None:
        self._config = Config()
        self._initialized = False
        self._launches: Counter[str] = Counter()
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
        if not self._initialized or capacity <= self.hard_capacity:
            return
        raise _not_ported("hard rigid contacts")

    def reset_joint_drive_state(self, stream: Any = None, mask: Span = Span()) -> None:  # noqa: B008
        if not self._initialized or not self.drive_total:
            return
        raise _not_ported("joint drives")

    def reset_cable_penalty_state(self, stream: Any = None, mask: Span = Span()) -> None:  # noqa: B008
        if not self._initialized or not self.cable_count:
            return
        raise _not_ported("cables")

    def copy_cable_penalties(self, output: Span, stream: Any = None) -> None:
        if not self._initialized or not self.cable_count:
            return
        raise _not_ported("cables")

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

    def _reject_unported(self, model: Model) -> None:
        unported = {
            "affine bodies": model.body_count or model.body_state_count,
            "non-diagonal particle masses": model.particle_mass_non_diagonal_count,
            "cloth triangles": model.triangle_count,
            "bending edges": model.edge_count,
            "tetrahedra": model.tetrahedron_count,
            "cables": model.cable_count,
            "shapes and contacts": model.shape_count,
            "fluid particles": model.fluid_particle_count,
            "granular particles": model.granular_particle_count or model.granular_bond_count,
            "joint drives": model.joint_drive_count,
        }
        for feature, present in unported.items():
            if present:
                raise _not_ported(feature)

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
        self._constructor_reads: dict[tuple[int, int, Any], np.ndarray] = {}
        self._initialized = False
        self.model = model._copy()
        self.count = count
        self.per_world = model.particle_count
        self.hard_capacity = 0
        self.drive_total = 0
        self.cable_count = 0
        if not count:
            self._initialized = True
            return
        self._reject_unported(model)

        masses = self._read(model.particle_masses, wp.float32, self.per_world)
        if not np.all(np.isfinite(masses)) or np.any(masses < 0.0):
            raise ValueError("particle masses must be finite and non-negative")
        if model.gravity.size:
            if not np.all(np.isfinite(self._read(model.gravity, wp.vec3, model.gravity.size))):
                raise ValueError("gravity values must be finite")
        if model.particle_world.size:
            if np.any(self._read(model.particle_world, wp.int32, self.per_world) < -1):
                raise ValueError("particle_world contains an invalid world index")

        # Radii are only checked when shapes or collision particles use them.
        if model.shape_count or model.collision_particle_count:
            self._require(model.particle_radii, self.per_world, "particle_radii")
            radii = self._read(model.particle_radii, wp.float32, self.per_world)
            if np.any(np.isnan(radii)) or np.any(radii < 0.0):
                raise ValueError("particle radii must be non-negative and not NaN")

        self.solution = self._zeros(count, wp.vec3)
        self.rhs = self._zeros(count, wp.vec3)
        self.previous = self._zeros(count, wp.vec3)
        self.target = self._zeros(count, wp.vec3)
        self._precompute_drag()
        self.base_diagonal = self._zeros(count, wp.float32)
        self.diagonal = self._zeros(count, wp.float32)
        self.scalar_inverse = self._zeros(count, wp.float32)
        self.blocks = self._zeros(count, wp.mat33)
        self.preconditioner_blocks = self._zeros(count, wp.mat33)
        # The native solver inverts the preconditioner blocks in place.
        self.inverse = self.preconditioner_blocks
        self.body_count = 0
        self.body_state_count = 0

        self._precompute_springs()
        self._constructor_reads.clear()

        # Remap arbitrary world labels (including -1) in first-seen order; each replicated template is offset by its
        # number of distinct local labels. Coefficient groups are contiguous unless the labels interleave.
        local_ids = np.zeros(self.per_world, dtype=np.int64)
        if model.particle_world.size:
            local_ids = self._read(model.particle_world, wp.int32, self.per_world).astype(np.int64)
        labels: dict[int, int] = {}
        for index, value in enumerate(local_ids):
            local_ids[index] = labels.setdefault(int(value), len(labels))
        copies = count // self.per_world
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
        self.spring_count = model.spring_count * (self.count // per_world)
        if not self.spring_count:
            self._precompute_particle_matrices(
                np.zeros(per_world, dtype=np.float32),
                np.zeros(self.count, dtype=np.int32),
                np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32),
            )
            return
        indices = self._read(model.spring_indices, wp.vec2i, model.spring_count)
        stiffness = self._read(model.spring_stiffnesses, wp.float32, model.spring_count)
        rest = self._read(model.spring_rest_lengths, wp.float32, model.spring_count)
        rows: list[dict[int, np.float32]] = [{} for _ in range(per_world)]
        diagonal = np.zeros(per_world, dtype=np.float32)
        for (a, b), k, length in zip(indices, stiffness, rest, strict=True):
            if not (k >= 0.0) or not math.isfinite(k):
                raise ValueError("spring stiffness must be finite and non-negative")
            if not (length > 0.0) or not math.isfinite(length):
                raise ValueError("spring rest length must be finite and positive")
            if a < 0 or b < 0 or a >= per_world or b >= per_world:
                raise ValueError("spring references an invalid particle index")
            if a == b:
                raise ValueError("spring endpoints must reference different particles")
            # Accumulate duplicate columns in spring order in float32, as the native std::map<int, float>.
            rows[a][int(b)] = np.float32(rows[a].get(int(b), np.float32(0.0)) + k)
            rows[b][int(a)] = np.float32(rows[b].get(int(a), np.float32(0.0)) + k)
        for (a, b), k in zip(indices, stiffness, strict=True):
            diagonal[a] = np.float32(diagonal[a] + k)
            diagonal[b] = np.float32(diagonal[b] + k)
        counts = np.array([len(rows[i % per_world]) for i in range(self.count)], dtype=np.int32)
        self.max_entries = int(counts.max(initial=0))
        columns = np.full(self.max_entries * per_world, -1, dtype=np.int32)
        values = np.zeros(self.max_entries * per_world, dtype=np.float32)
        for i in range(per_world):
            for slot, column in enumerate(sorted(rows[i])):
                columns[slot * per_world + i] = column
                values[slot * per_world + i] = -rows[i][column]
        self._precompute_particle_matrices(diagonal, counts, columns, values)

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

    def _step(self, state_in: State, state_out: State, control: Control, contact: Contact, dt: float, stream: Any):
        if not self._initialized:
            raise RuntimeError("initialize() must be called before step()")
        device = self.device if self.device is not None else wp.get_cuda_device()
        resolved = _resolve_stream(stream, device)
        scope = nullcontext() if resolved is None else wp.ScopedStream(resolved, sync_enter=False)
        with scope:
            self._step_on_stream(state_in, state_out, control, contact, dt)

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
        if contact.soft_max_count or contact.max_count:
            raise _not_ported("contacts")
        if control.tetrahedron_activations.size:
            raise _not_ported("tetrahedron activations")
        if config.enable_self_collision and self.self_collision_primitives:
            raise _not_ported("self-collision")

        pd = config.nonlinear_solver == NonlinearSolver.PD_NEWTON
        drag = bool(self.drag_count) and config.air_density > 0.0
        bodies = self.body_count != 0
        rigidity = config.affine_body_rigidity_model
        six_springs = bodies and rigidity == AffineBodyRigidityModel.SIX_SPRINGS
        alpha = config.constraint_alpha
        block_matrix = not pd
        block = block_matrix or drag
        body_block = bodies and block and config.affine_body_block_preconditioner
        extra_product = drag
        reference_pcg = config.reference_linear_schedule
        fused_pcg = block_matrix and not body_block and not reference_pcg
        del six_springs, alpha

        self._require(state_in.particle_q, count, "state_in.particle_q")
        self._require(state_in.particle_qd, count, "state_in.particle_qd")
        if state_in.particle_f.size and state_in.particle_f.size < count:
            raise ValueError("input state particle_f is smaller than particle_count")
        self._require(state_out.particle_q, count, "state_out.particle_q")
        self._require(state_out.particle_qd, count, "state_out.particle_qd")

        q_in = self._view(state_in.particle_q, wp.vec3, count)
        qd_in = self._view(state_in.particle_qd, wp.vec3, count)
        q_out = self._view(state_out.particle_q, wp.vec3, count)
        qd_out = self._view(state_out.particle_qd, wp.vec3, count)
        force = self._view(state_in.particle_f, wp.vec3, count)
        mass = self._view(model.particle_masses, wp.float32, self.per_world)
        flags = self._view(model.particle_flags, wp.int32, self.per_world)
        # Homogeneous replicated models (world_count <= 1) pass their world map; otherwise worlds follow the row.
        worlds = (
            self._view(model.particle_world, wp.int32, self.per_world)
            if model.world_count <= 1 and model.particle_world.size
            else self._empty(wp.int32)
        )
        gravity = self._view(model.gravity, wp.vec3)
        inverse_dt = 1.0 / dt
        inverse_dt2 = 1.0 / (dt * dt)
        per_world = self.per_world
        empty_v = self._empty(wp.vec3)
        empty_f = self._empty(wp.float32)
        empty_m = self._empty(wp.mat33)
        empty_i = self._empty(wp.int32)
        templates = self.particle_diagonal_templates

        self._launch(
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
                config.constraint_alpha,
                templates[3],
                0.0,
                templates[4],
                config.constraint_alpha,
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
            self._launch(
                "drag_start",
                drag_kernels.initialize_triangle_drag,
                self.drag_count,
                [self.previous, self.drag_indices, self.drag_normal, self.drag_area, self.drag_local_count, per_world],
            )
        if block_matrix and self.max_entries:
            self._launch(
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

        group_count = self.group_count
        for _iteration in range(config.nonlinear_iterations):
            # The native solver restores this cache on every iteration, including contact-free models.
            wp.copy(self.diagonal, self.base_diagonal, count=count)
            self._launch(
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
            if drag:
                wind = wp.vec3(*config.wind_velocity)
                self._launch(
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
                        wind,
                        inverse_dt,
                        self.rhs,
                        self.drag_scalar,
                        self.drag_local_count,
                        per_world,
                    ],
                )
            if self.spring_count:
                self._launch(
                    "spring_force",
                    spring_kernels.evaluate_spring_force,
                    self.spring_count,
                    [
                        q_in,
                        self._view(model.spring_indices, wp.vec2i, model.spring_count),
                        self._view(model.spring_stiffnesses, wp.float32, model.spring_count),
                        self._view(model.spring_rest_lengths, wp.float32, model.spring_count),
                        self.rhs,
                        model.spring_count,
                        per_world,
                    ],
                )
            if not block:
                # The contact-free PD path keeps the constant scalar spring matrix and uses 1 / D.
                self._launch(
                    "scalar_inverse",
                    pcg_kernels.prepare_inverse_preconditioner,
                    count,
                    [self.diagonal, mass, flags, self.scalar_inverse, per_world],
                )
            else:
                self._launch(
                    "blocks",
                    step_kernels.initialize_block_diagonal,
                    count,
                    [self.diagonal, self.blocks, empty_m if block_matrix else self.preconditioner_blocks],
                )
                if block_matrix and self.max_entries:
                    wp.copy(self.block_entry_columns, self.block_template_columns)
                    wp.copy(self.block_entry_values, self.block_template_values)
                if not pd and self.spring_count:
                    self._launch(
                        "spring_hessian",
                        spring_kernels.add_spring_gauss_newton_hessian,
                        self.spring_count,
                        [
                            q_in,
                            self._view(model.spring_indices, wp.vec2i, model.spring_count),
                            self._view(model.spring_stiffnesses, wp.float32, model.spring_count),
                            self._view(model.spring_rest_lengths, wp.float32, model.spring_count),
                            self.blocks,
                            self.row_count,
                            self.block_entry_columns,
                            self.block_entry_values,
                            self.max_entries,
                            model.spring_count,
                            per_world,
                            count,
                        ],
                    )
                # The native solver retains A and inverts a separate preconditioner copy in place.
                if block_matrix:
                    wp.copy(self.preconditioner_blocks, self.blocks, count=count)
                if drag:
                    self._launch(
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
                self._launch(
                    "inverse",
                    pcg_kernels.invert_block_preconditioner,
                    count,
                    [self.preconditioner_blocks, mass, flags, self.inverse, per_world],
                )

            wp.copy(self.residual, self.rhs, count=count)
            self.solution.zero_()
            iterations = config.linear_iterations
            self.numerator.zero_()
            self.denominator.zero_()
            for k in range(iterations):
                numerator = self.numerator[k * group_count : (k + 1) * group_count]
                previous_numerator = (
                    self.numerator[(k - 1) * group_count : k * group_count] if k else self._empty(wp.float32)
                )
                denominator = self.denominator[k * group_count : (k + 1) * group_count]
                per_group = self.particles_per_group
                if not fused_pcg or k == 0:
                    if block:
                        self._launch(
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
                        self._launch(
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
                    self._launch(
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
                    self._launch(
                        "direction",
                        pcg_kernels.update_cg_direction,
                        count,
                        [k, self.z, self.coefficient_world, per_group, numerator, previous_numerator, self.direction],
                    )
                    if extra_product:
                        self.contact_product.zero_()
                if drag:
                    self._launch(
                        "drag_tangent",
                        drag_kernels.multiply_triangle_drag_tangent,
                        self.drag_count,
                        [
                            self.direction,
                            self.drag_indices,
                            self.drag_normal,
                            self.drag_scalar,
                            self.contact_product,
                            self.drag_local_count,
                            per_world,
                        ],
                    )
                extra = self.contact_product if extra_product else empty_v
                if block_matrix:
                    self._launch(
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
                    self._launch(
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
                    self._launch(
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
                    self._launch(
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
                    self._launch(
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
            # The native host updates state_in.q in place, even when state_out owns a separate allocation.
            self._launch("apply", step_kernels.apply_nonlinear_step, count, [q_in, q_out, self.solution])
            wp.copy(q_in, q_out, count=count)

        if not config.nonlinear_iterations:
            wp.copy(q_out, q_in, count=count)
        self._launch("velocity", step_kernels.update_velocity, count, [inverse_dt, self.previous, q_out, qd_out])
        del empty_i

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

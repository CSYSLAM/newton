# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Data API of the Warp solver core, field for field identical to the native ``_core`` extension.

``Model``, ``State``, ``Control`` and ``Contact`` hold non-owning :class:`Span` views (device address and element
count) of Newton-owned arrays plus scalar settings; ``Config`` holds the runtime options. Field names, types,
defaults and validation follow the native bindings so the adapter and tests run unchanged on either backend.
"""

from __future__ import annotations

import enum
import math
import operator
from typing import Any, ClassVar

import numpy as np

__all__ = [
    "AffineBodyRigidityModel",
    "Config",
    "Contact",
    "Control",
    "InertiaMode",
    "JointConstraintMode",
    "LogLevel",
    "Model",
    "NonlinearSolver",
    "Solver",  # noqa: F822 (resolved lazily by __getattr__)
    "Span",
    "State",
    "set_log_callback",
]


class _NativeEnum(enum.IntEnum):
    """IntEnum that, like a pybind11 enum, can also hold values outside its members (named ``???``)."""

    @classmethod
    def _missing_(cls, value: object) -> _NativeEnum | None:
        if isinstance(value, int) and not isinstance(value, bool):
            member = int.__new__(cls, value)
            member._name_ = "???"
            member._value_ = value
            return member
        return None


class NonlinearSolver(_NativeEnum):
    NEWTON = 0
    PD_NEWTON = 1


class JointConstraintMode(_NativeEnum):
    PENALTY = 0
    AUGMENTED_LAGRANGIAN = 1


class InertiaMode(_NativeEnum):
    EXPLICIT_EULER = 0
    IMPLICIT_EULER = 1
    LIE_GROUP_MIDPOINT = 2
    MIDPOINT_CAYLEY = 3
    NONE = 4
    VBD_SEMI_IMPLICIT = 5


class AffineBodyRigidityModel(_NativeEnum):
    CO_ROTATED = 0
    ABD_ORTHOGONALITY = 1
    SIX_SPRINGS = 2
    SO3_AUGMENTED_LAGRANGIAN = 3


class LogLevel(_NativeEnum):
    INFO = 0
    ERROR = 1
    DEBUG = 2
    WARNING = 4


_ENUMS = {
    "NonlinearSolver": NonlinearSolver,
    "JointConstraintMode": JointConstraintMode,
    "InertiaMode": InertiaMode,
    "AffineBodyRigidityModel": AffineBodyRigidityModel,
}


class Span:
    """Non-owning view of ``size`` elements at device address ``ptr``; never dereferenced on the host."""

    __slots__ = ("_ptr", "_size")

    def __init__(self, ptr: int = 0, size: int = 0) -> None:
        ptr, size = operator.index(ptr), operator.index(size)
        if ptr < 0 or size < 0:
            raise TypeError("Span address and size must be non-negative integers")
        if not ptr and size:
            raise ValueError("a non-empty span requires a nonzero device address")
        self._ptr, self._size = ptr, size

    @property
    def ptr(self) -> int:
        return self._ptr

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"Span(ptr={self._ptr}, size={self._size})"


_MODEL_FIELDS = (
    ("affine_body_ids", "span", None),
    ("affine_body_indices", "span", None),
    ("body_centers_of_mass", "span", None),
    ("body_count", "u32", 0),
    ("body_distance_constraint_count", "u32", 0),
    ("body_distance_constraint_indices", "span", None),
    ("body_distance_constraint_mass_scales", "span", None),
    ("body_distance_constraint_points0", "span", None),
    ("body_distance_constraint_points1", "span", None),
    ("body_distance_constraint_rest_lengths", "span", None),
    ("body_distance_constraint_weights0", "span", None),
    ("body_distance_constraint_weights1", "span", None),
    ("body_inertia_tensors", "span", None),
    ("body_local_transforms", "span", None),
    ("body_masses", "span", None),
    ("body_point_line_count", "u32", 0),
    ("body_point_line_dampings", "span", None),
    ("body_point_line_endpoints0", "span", None),
    ("body_point_line_endpoints1", "span", None),
    ("body_point_line_indices", "span", None),
    ("body_point_line_points", "span", None),
    ("body_point_line_stiffnesses", "span", None),
    ("body_spring_count", "u32", 0),
    ("body_spring_indices", "span", None),
    ("body_spring_points0", "span", None),
    ("body_spring_points1", "span", None),
    ("body_spring_rest_lengths", "span", None),
    ("body_spring_stiffnesses", "span", None),
    ("body_state_count", "u32", 0),
    ("cable_angular_bias", "span", None),
    ("cable_angular_hessian", "span", None),
    ("cable_body_indices", "span", None),
    ("cable_child_transforms", "span", None),
    ("cable_count", "u32", 0),
    ("cable_dampings", "span", None),
    ("cable_enabled", "span", None),
    ("cable_parent_transforms", "span", None),
    ("cable_rest_bend_twist", "span", None),
    ("cable_stiffnesses", "span", None),
    ("collision_edge_count", "u32", 0),
    ("collision_edge_indices", "span", None),
    ("collision_particle_count", "u32", 0),
    ("collision_particle_indices", "span", None),
    ("collision_triangle_count", "u32", 0),
    ("collision_triangle_indices", "span", None),
    ("edge_bending_cots", "span", None),
    ("edge_bending_properties", "span", None),
    ("edge_count", "u32", 0),
    ("edge_indices", "span", None),
    ("edge_rest_areas", "span", None),
    ("fluid_boundary_count", "u32", 0),
    ("fluid_boundary_positions", "span", None),
    ("fluid_boundary_volumes", "span", None),
    ("fluid_bulk_modulus", "f32", 200000.0),
    ("fluid_particle_count", "u32", 0),
    ("fluid_particle_indices", "span", None),
    ("fluid_rest_density", "f32", 1000.0),
    ("fluid_smoothing_radius", "f32", 0.10000000149011612),
    ("fluid_viscosity", "f32", 0.0),
    ("granular_adhesion", "f32", 0.0),
    ("granular_bond_count", "u32", 0),
    ("granular_bond_damping", "f32", 0.0),
    ("granular_bond_formation_gap", "f32", 0.0),
    ("granular_bond_formation_speed", "f32", 1.0),
    ("granular_bond_indices", "span", None),
    ("granular_bond_rest_lengths", "span", None),
    ("granular_break_strain", "f32", 0.10000000149011612),
    ("granular_cohesion", "f32", 200.0),
    ("granular_contact_stiffness", "f32", 2000.0),
    ("granular_friction", "f32", 0.5),
    ("granular_friction_damping", "f32", 20.0),
    ("granular_max_compaction", "f32", 0.4000000059604645),
    ("granular_max_dynamic_bonds", "u32", 12),
    ("granular_particle_count", "u32", 0),
    ("granular_particle_indices", "span", None),
    ("granular_yield_strain", "f32", 0.029999999329447746),
    ("gravity", "span", None),
    ("hinge_count", "u32", 0),
    ("hinge_indices", "span", None),
    ("hinge_rest_angles", "span", None),
    ("hinge_stiffnesses", "span", None),
    ("joint_armature", "span", None),
    ("joint_body_point_line_count", "u32", 0),
    ("joint_body_spring_count", "u32", 0),
    ("joint_constraint_mass_scale", "f32", 1.0),
    ("joint_dof_count", "u32", 0),
    ("joint_drive_body_indices", "span", None),
    ("joint_drive_child_origins", "span", None),
    ("joint_drive_child_reference_points", "span", None),
    ("joint_drive_count", "u32", 0),
    ("joint_drive_dof_indices", "span", None),
    ("joint_drive_parent_axis_points", "span", None),
    ("joint_drive_parent_origins", "span", None),
    ("joint_drive_parent_reference_points", "span", None),
    ("joint_drive_target_indices", "span", None),
    ("joint_drive_types", "span", None),
    ("joint_effort_limits", "span", None),
    ("joint_limit_dampings", "span", None),
    ("joint_limit_lower", "span", None),
    ("joint_limit_stiffnesses", "span", None),
    ("joint_limit_upper", "span", None),
    ("joint_passive_dampings", "span", None),
    ("joint_target_count", "u32", 0),
    ("joint_target_dampings", "span", None),
    ("joint_target_modes", "span", None),
    ("joint_target_stiffnesses", "span", None),
    ("particle_count", "u32", 0),
    ("particle_flags", "span", None),
    ("particle_indices", "span", None),
    ("particle_mass_non_diagonal_count", "u32", 0),
    ("particle_mass_non_diagonal_indices", "span", None),
    ("particle_mass_non_diagonal_values", "span", None),
    ("particle_masses", "span", None),
    ("particle_radii", "span", None),
    ("particle_rest_positions", "span", None),
    ("particle_state_count", "u32", 0),
    ("particle_world", "span", None),
    ("shape_bodies", "span", None),
    ("shape_contact_dampings", "span", None),
    ("shape_contact_friction_gains", "span", None),
    ("shape_contact_stiffnesses", "span", None),
    ("shape_count", "u32", 0),
    ("shape_frictions", "span", None),
    ("shape_margins", "span", None),
    ("shape_replicated_count", "u32", 0),
    ("shape_rolling_frictions", "span", None),
    ("shape_scales", "span", None),
    ("shape_torsional_frictions", "span", None),
    ("shape_transforms", "span", None),
    ("shape_types", "span", None),
    ("soft_contact_damping", "f32", 0.0),
    ("soft_contact_friction", "f32", 0.0),
    ("soft_contact_friction_gain", "f32", 1000.0),
    ("soft_contact_stiffness", "f32", 1000.0),
    ("spring_count", "u32", 0),
    ("spring_indices", "span", None),
    ("spring_rest_lengths", "span", None),
    ("spring_stiffnesses", "span", None),
    ("tetrahedron_count", "u32", 0),
    ("tetrahedron_indices", "span", None),
    ("tetrahedron_inverse_dms", "span", None),
    ("tetrahedron_materials", "span", None),
    ("triangle_aniso_stiffnesses", "span", None),
    ("triangle_count", "u32", 0),
    ("triangle_drag_coefficients", "span", None),
    ("triangle_indices", "span", None),
    ("triangle_inv_dms", "span", None),
    ("triangle_masses", "span", None),
    ("triangle_rest_areas", "span", None),
    ("world_count", "u32", 0),
)

_STATE_FIELDS = (
    ("particle_f", "span", None),
    ("particle_q", "span", None),
    ("particle_qd", "span", None),
)

_CONTROL_FIELDS = (
    ("joint_drive_velocities", "span", None),
    ("joint_f", "span", None),
    ("joint_target_q", "span", None),
    ("joint_target_qd", "span", None),
    ("tetrahedron_activations", "span", None),
)

_CONTACT_FIELDS = (
    ("contact_force", "span", None),
    ("count", "span", None),
    ("history_refresh", "bool", True),
    ("margin0", "span", None),
    ("margin1", "span", None),
    ("match_index", "span", None),
    ("max_count", "u32", 0),
    ("normal", "span", None),
    ("offset0", "span", None),
    ("offset1", "span", None),
    ("point0", "span", None),
    ("point1", "span", None),
    ("shape0", "span", None),
    ("shape1", "span", None),
    ("soft_barycentric", "span", None),
    ("soft_body_pos", "span", None),
    ("soft_body_vel", "span", None),
    ("soft_count", "span", None),
    ("soft_indices", "span", None),
    ("soft_max_count", "u32", 0),
    ("soft_normal", "span", None),
    ("soft_particle", "span", None),
    ("soft_particle_only_shapes", "span", None),
    ("soft_shape", "span", None),
)

_CONFIG_FIELDS = (
    ("affine_body_block_preconditioner", "bool", False),
    ("affine_body_contact_block_preconditioner", "bool", True),
    ("affine_body_rigidity_model", "enum:AffineBodyRigidityModel", 3),
    ("affine_body_rigidity_primal_iterations", "i32", 1),
    ("air_density", "f32", 1.225000023841858),
    ("cable_angular_beta", "f32", 0.0),
    ("cable_angular_k_start", "f32", 10.0),
    ("cable_linear_beta", "f32", 0.0),
    ("cable_linear_k_start", "f32", 100.0),
    ("cable_penalty_gamma", "f32", 0.9990000128746033),
    ("constraint_alpha", "f32", 100.0),
    ("enable_collision_hessian_non_diagonal", "bool", True),
    ("enable_self_collision", "bool", False),
    ("fluid_max_step_ratio", "f32", 0.25),
    ("fluid_relaxation", "f32", 0.5),
    ("inertia_mode", "enum:InertiaMode", 3),
    ("joint_constraint_mode", "enum:JointConstraintMode", 1),
    ("joint_constraint_primal_iterations", "i32", 1),
    ("linear_iterations", "i32", 50),
    ("nonlinear_iterations", "i32", 5),
    ("nonlinear_solver", "enum:NonlinearSolver", 0),
    ("reference_linear_schedule", "bool", False),
    ("rigid_contact_alpha", "f32", 0.949999988079071),
    ("rigid_contact_beta", "f32", 0.0),
    ("rigid_contact_friction_epsilon", "f32", 0.0),
    ("rigid_contact_gamma", "f32", 0.9990000128746033),
    ("rigid_contact_hard", "bool", False),
    ("rigid_contact_history", "bool", False),
    ("rigid_contact_k_start", "f32", 100.0),
    ("rigid_contact_sliding_tangent", "bool", False),
    ("rolling_friction_velocity_epsilon", "f32", 0.10000000149011612),
    ("self_collision_edge_edge_stiffness", "f32", 0.10000000149011612),
    ("self_collision_edge_face_stiffness", "f32", 1.0),
    ("self_collision_particle_particle_stiffness", "f32", 1.0),
    ("self_collision_radius", "f32", 0.003000000026077032),
    ("self_collision_vertex_face_stiffness", "f32", 0.5),
    # Warp backend only: side streams for the PCG matrix-free tangent terms; 1 restores the serial native order.
    ("tangent_streams", "i32", 8),
    ("wind_velocity", "vec3", (0.0, 0.0, 0.0)),
)


def _float32(value: Any) -> float:
    if isinstance(value, (str, bytes)) or value is None:
        raise TypeError(f"expected a float, got {type(value).__name__}")
    return float(np.float32(float(value)))


def _convert(kind: str, name: str, value: Any) -> Any:
    """Coerce ``value`` to the native field type, raising the same exception types as the pybind11 casters."""
    if kind == "span":
        if not isinstance(value, Span):
            raise TypeError(f"{name} requires a Span")
        return value
    if kind == "u32":
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} requires an unsigned 32-bit integer")
        value = int(value)
        if not 0 <= value < 2**32:
            raise TypeError(f"{name} requires an unsigned 32-bit integer")
        return value
    if kind == "i32":
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} requires a 32-bit integer")
        value = int(value)
        if not -(2**31) <= value < 2**31:
            raise TypeError(f"{name} requires a 32-bit integer")
        return value
    if kind == "f32":
        return _float32(value)
    if kind == "bool":
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} requires a bool")
        return bool(value)
    if kind == "vec3":
        try:
            components = tuple(value)
        except TypeError:
            raise TypeError(f"{name} requires a sequence") from None
        if len(components) != 3:
            raise ValueError(f"{name} must contain three finite components")
        parsed = tuple(_float32(component) for component in components)
        if not all(math.isfinite(component) for component in parsed):
            raise ValueError(f"{name} must contain three finite components")
        return parsed
    if kind.startswith("enum:"):
        enum_type = _ENUMS[kind[5:]]
        if not isinstance(value, enum_type):
            raise TypeError(f"{name} requires a {enum_type.__name__}")
        return value
    raise AssertionError(kind)


def _default(kind: str, value: Any) -> Any:
    if kind == "span":
        return Span()
    if kind.startswith("enum:"):
        return _ENUMS[kind[5:]](value)
    if kind == "f32":
        return _float32(value)
    return value


class _Record:
    """Plain field container with native field types; unknown attributes are rejected through ``__slots__``."""

    __slots__ = ()
    _FIELDS: tuple[tuple[str, str, Any], ...] = ()
    _KINDS: ClassVar[dict[str, str]] = {}

    def __init__(self) -> None:
        for name, kind, value in self._FIELDS:
            object.__setattr__(self, name, _default(kind, value))

    def __setattr__(self, name: str, value: Any) -> None:
        kind = self._KINDS.get(name)
        if kind is None:
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        object.__setattr__(self, name, _convert(kind, name, value))

    def _copy(self):
        other = type(self)()
        for name, _kind, _value in self._FIELDS:
            object.__setattr__(other, name, getattr(self, name))
        return other


def _record(name: str, fields: tuple[tuple[str, str, Any], ...]) -> type:
    names = tuple(field[0] for field in fields)
    return type(name, (_Record,), {"__slots__": names, "_FIELDS": fields, "_KINDS": {f[0]: f[1] for f in fields}})


Model = _record("Model", _MODEL_FIELDS)
State = _record("State", _STATE_FIELDS)
Control = _record("Control", _CONTROL_FIELDS)
Contact = _record("Contact", _CONTACT_FIELDS)
_ConfigBase = _record("_ConfigBase", _CONFIG_FIELDS)


class Config(_ConfigBase):
    __slots__ = ()

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "air_density":
            value = _float32(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("air_density must be finite and non-negative")
        super().__setattr__(name, value)


for _cls in (Model, State, Control, Contact, Config):
    _cls.__module__ = __name__


def set_log_callback(callback: Any = None) -> None:
    """Accepted for API compatibility; the Warp core emits no native log messages."""
    del callback


def __getattr__(name: str) -> Any:
    # The solver pulls in Warp kernels; resolve it lazily so the data API stays importable on its own.
    if name == "Solver":
        from .solver import Solver  # noqa: PLC0415

        globals()["Solver"] = Solver
        return Solver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

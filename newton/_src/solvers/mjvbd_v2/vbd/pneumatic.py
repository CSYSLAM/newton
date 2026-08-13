# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pneumatic-cavity authoring helpers for :class:`SolverMJVBDV2`."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
import warp as wp

from ....sim import Model, ModelBuilder

__all__ = [
    "PneumaticCavityHandle",
    "PneumaticConfig",
    "PneumaticMode",
    "add_inflatable_mesh",
    "add_pneumatic_cavity",
    "register_pneumatic_attributes",
]


class PneumaticMode(IntEnum):
    """Pressure laws supported by :class:`SolverMJVBDV2` pneumatic cavities."""

    ISOTHERMAL = 0
    """Use ``p V = constant`` with the configured reference pressure."""

    ADIABATIC = 1
    """Use ``p V**gamma = constant`` with the configured heat-capacity ratio."""

    TARGET_VOLUME = 2
    """Use a quadratic energy that drives the cavity toward a target volume."""

    PRESCRIBED_GAUGE_PRESSURE = 3
    """Apply a directly prescribed pressure difference from the control input."""


@dataclass(frozen=True)
class PneumaticConfig:
    """Parameters for one closed pneumatic cavity.

    The pressure law is evaluated by :class:`SolverMJVBDV2` during every particle
    iteration. ``reference_absolute_pressure`` is the gas pressure at the
    authored rest volume; it therefore includes the ambient component.

    Args:
        mode: Pressure law to evaluate.
        reference_absolute_pressure: Gas pressure at rest volume [Pa].
        ambient_pressure: Pressure outside the bag [Pa].
        heat_capacity_ratio: Adiabatic exponent ``gamma`` [dimensionless].
        target_volume: Target cavity volume for ``TARGET_VOLUME`` [m^3].
            ``None`` uses the authored rest volume.
        volume_stiffness: Target-volume stiffness [Pa/m^3].
        bulk_damping: Volume-rate damping [Pa·s/m^3].
        min_volume_ratio: Lower bound relative to rest volume [dimensionless].
        max_absolute_pressure: Upper pressure clamp [Pa].
        prescribed_gauge_pressure: Default pressure difference for
            ``PRESCRIBED_GAUGE_PRESSURE`` [Pa].
    """

    mode: PneumaticMode = PneumaticMode.ISOTHERMAL
    reference_absolute_pressure: float = 150_000.0
    ambient_pressure: float = 101_325.0
    heat_capacity_ratio: float = 1.4
    target_volume: float | None = None
    volume_stiffness: float = 0.0
    bulk_damping: float = 0.0
    min_volume_ratio: float = 0.05
    max_absolute_pressure: float = 2_000_000.0
    prescribed_gauge_pressure: float = 0.0

    def __post_init__(self) -> None:
        """Validate values that do not depend on the authored mesh."""
        if self.reference_absolute_pressure <= 0.0:
            raise ValueError("reference_absolute_pressure must be positive.")
        if self.ambient_pressure < 0.0:
            raise ValueError("ambient_pressure must be non-negative.")
        if self.heat_capacity_ratio < 1.0:
            raise ValueError("heat_capacity_ratio must be at least 1.0.")
        if self.target_volume is not None and self.target_volume <= 0.0:
            raise ValueError("target_volume must be positive when provided.")
        if self.volume_stiffness < 0.0:
            raise ValueError("volume_stiffness must be non-negative.")
        if self.bulk_damping < 0.0:
            raise ValueError("bulk_damping must be non-negative.")
        if not 0.0 < self.min_volume_ratio <= 1.0:
            raise ValueError("min_volume_ratio must be in (0, 1].")
        if self.max_absolute_pressure <= 0.0:
            raise ValueError("max_absolute_pressure must be positive.")


_DEFAULT_PNEUMATIC_CONFIG = PneumaticConfig()


@dataclass(frozen=True)
class PneumaticCavityHandle:
    """Indices of one pneumatic cavity authored into a :class:`ModelBuilder`.

    Args:
        cavity_index: Index into the ``pneumatic:cavity`` arrays.
        face_start: First index in the ``pneumatic:face`` arrays.
        face_count: Number of closed-surface triangles in the cavity.
        rest_volume: Rest volume [m^3].
    """

    cavity_index: int
    face_start: int
    face_count: int
    rest_volume: float


_CAVITY_FREQUENCY = "pneumatic:cavity"
_FACE_FREQUENCY = "pneumatic:face"
_NAMESPACE = "pneumatic"


def _register_attribute(
    builder: ModelBuilder,
    *,
    name: str,
    dtype: type,
    frequency: str,
    assignment: Model.AttributeAssignment,
    default: Any,
    references: str | None = None,
) -> None:
    builder.add_custom_attribute(
        ModelBuilder.CustomAttribute(
            name=name,
            dtype=dtype,
            frequency=frequency,
            assignment=assignment,
            namespace=_NAMESPACE,
            references=references,
            default=default,
        )
    )


def register_pneumatic_attributes(builder: ModelBuilder) -> None:
    """Register VBD pneumatic-cavity schema on a builder.

    The function is idempotent, so applications may call it before creating
    each independently-authored bag. It is also called automatically by
    :func:`add_pneumatic_cavity` and :func:`add_inflatable_mesh`.

    Args:
        builder: Builder that will own the cavity rows.
    """
    builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="cavity", namespace=_NAMESPACE))
    builder.add_custom_frequency(ModelBuilder.CustomFrequency(name="face", namespace=_NAMESPACE))

    model_fields = (
        ("mode", wp.int32, 0, None),
        ("world", wp.int32, -1, "world"),
        ("anchor_particle", wp.int32, -1, "particle"),
        ("rest_volume", wp.float32, 0.0, None),
        ("reference_absolute_pressure", wp.float32, 0.0, None),
        ("ambient_pressure", wp.float32, 0.0, None),
        ("heat_capacity_ratio", wp.float32, 1.4, None),
        ("target_volume", wp.float32, 0.0, None),
        ("volume_stiffness", wp.float32, 0.0, None),
        ("bulk_damping", wp.float32, 0.0, None),
        ("min_volume", wp.float32, 0.0, None),
        ("max_absolute_pressure", wp.float32, 0.0, None),
    )
    for name, dtype, default, references in model_fields:
        _register_attribute(
            builder,
            name=name,
            dtype=dtype,
            frequency=_CAVITY_FREQUENCY,
            assignment=Model.AttributeAssignment.MODEL,
            default=default,
            references=references,
        )

    state_fields = (
        ("volume", wp.float32, 0.0),
        ("absolute_pressure", wp.float32, 0.0),
        ("volume_rate", wp.float32, 0.0),
        ("clamp_flags", wp.int32, 0),
    )
    for name, dtype, default in state_fields:
        _register_attribute(
            builder,
            name=name,
            dtype=dtype,
            frequency=_CAVITY_FREQUENCY,
            assignment=Model.AttributeAssignment.STATE,
            default=default,
        )

    control_fields = (
        ("pressure_scale", wp.float32, 1.0),
        ("prescribed_gauge_pressure", wp.float32, 0.0),
        ("target_volume_scale", wp.float32, 1.0),
    )
    for name, dtype, default in control_fields:
        _register_attribute(
            builder,
            name=name,
            dtype=dtype,
            frequency=_CAVITY_FREQUENCY,
            assignment=Model.AttributeAssignment.CONTROL,
            default=default,
        )

    face_fields = (
        ("face_cavity", wp.int32, -1, _CAVITY_FREQUENCY),
        ("face_triangle", wp.int32, -1, "triangle"),
        ("face_sign", wp.float32, 1.0, None),
    )
    for name, dtype, default, references in face_fields:
        _register_attribute(
            builder,
            name=name,
            dtype=dtype,
            frequency=_FACE_FREQUENCY,
            assignment=Model.AttributeAssignment.MODEL,
            default=default,
            references=references,
        )


def _edge_direction(i: int, j: int) -> int:
    return 1 if i < j else -1


def _orient_closed_surface(triangles: Sequence[tuple[int, int, int]]) -> list[float]:
    """Return signs that make shared triangle edges oppositely directed."""
    edge_uses: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for face_index, (i, j, k) in enumerate(triangles):
        if len({i, j, k}) != 3:
            raise ValueError(f"Pneumatic face {face_index} is degenerate.")
        for a, b in ((i, j), (j, k), (k, i)):
            edge_uses[(min(a, b), max(a, b))].append((face_index, _edge_direction(a, b)))

    for edge, uses in edge_uses.items():
        if len(uses) != 2:
            raise ValueError(
                "A pneumatic cavity must be a closed two-manifold surface; "
                f"edge {edge} belongs to {len(uses)} selected faces."
            )

    signs: list[int | None] = [None] * len(triangles)
    signs[0] = 1
    queue = deque([0])
    face_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for uses in edge_uses.values():
        (first_face, first_direction), (second_face, second_direction) = uses
        face_edges[first_face].append((second_face, -first_direction * second_direction))
        face_edges[second_face].append((first_face, -second_direction * first_direction))

    while queue:
        face_index = queue.popleft()
        assert signs[face_index] is not None
        for other, multiplier in face_edges[face_index]:
            expected = signs[face_index] * multiplier
            if signs[other] is None:
                signs[other] = expected
                queue.append(other)
            elif signs[other] != expected:
                raise ValueError("Pneumatic cavity surface is not orientable.")

    if any(sign is None for sign in signs):
        raise ValueError("A pneumatic cavity must contain one connected closed surface.")
    return [float(sign) for sign in signs]


def _validate_oriented_surface(triangles: Sequence[tuple[int, int, int]], signs: Sequence[float]) -> None:
    """Validate an explicitly supplied orientation-sign field."""
    _orient_closed_surface(triangles)
    edge_uses: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, ((i, j, k), sign) in enumerate(zip(triangles, signs, strict=True)):
        if sign not in (-1.0, 1.0):
            raise ValueError(f"face_signs[{face_index}] must be either -1 or 1.")
        if len({i, j, k}) != 3:
            raise ValueError(f"Pneumatic face {face_index} is degenerate.")
        for a, b in ((i, j), (j, k), (k, i)):
            edge_uses[(min(a, b), max(a, b))].append(int(sign) * _edge_direction(a, b))
    for edge, directions in edge_uses.items():
        if len(directions) != 2 or directions[0] == directions[1]:
            raise ValueError(
                f"face_signs must orient every shared edge in opposite directions; edge {edge} is invalid."
            )


def _surface_volume(positions: np.ndarray, triangles: Sequence[tuple[int, int, int]], signs: Sequence[float]) -> float:
    """Compute signed closed-surface volume with a local tetrahedron anchor."""
    anchor = positions[triangles[0][0]]
    volume = 0.0
    for triangle, sign in zip(triangles, signs, strict=True):
        p0, p1, p2 = positions[list(triangle)] - anchor
        volume += sign * float(np.dot(p0, np.cross(p1, p2))) / 6.0
    return volume


def add_pneumatic_cavity(
    builder: ModelBuilder,
    triangle_indices: Sequence[int],
    *,
    config: PneumaticConfig = _DEFAULT_PNEUMATIC_CONFIG,
    face_signs: Sequence[float] | None = None,
) -> PneumaticCavityHandle:
    """Add one closed triangular-shell cavity to a model builder.

    The selected triangle elements must form exactly one closed, orientable,
    two-manifold surface. Existing cloth triangles are reused; this function
    does not duplicate particles or material elements.

    Args:
        builder: Builder containing the shell triangles.
        triangle_indices: Indices into ``builder.tri_indices``.
        config: Pressure-law and safety settings.
        face_signs: Optional per-face signs. If omitted, signs are inferred from
            the selected mesh topology and made outward-facing.

    Returns:
        A handle identifying the added cavity and its face rows.
    """
    register_pneumatic_attributes(builder)
    selected_indices = [int(index) for index in triangle_indices]
    if len(selected_indices) < 4:
        raise ValueError("A closed pneumatic cavity requires at least four triangles.")
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("triangle_indices must not contain duplicates.")
    if any(index < 0 or index >= len(builder.tri_indices) for index in selected_indices):
        raise IndexError("triangle_indices contains a triangle that is not present on the builder.")

    triangles = [tuple(builder.tri_indices[index]) for index in selected_indices]
    if face_signs is None:
        signs = _orient_closed_surface(triangles)
    else:
        if len(face_signs) != len(triangles):
            raise ValueError("face_signs must have one value per selected triangle.")
        signs = [float(sign) for sign in face_signs]
        _validate_oriented_surface(triangles, signs)

    positions = np.asarray(builder.particle_q, dtype=np.float64)
    rest_volume = _surface_volume(positions, triangles, signs)
    if face_signs is None and rest_volume < 0.0:
        signs = [-sign for sign in signs]
        rest_volume = -rest_volume
    if rest_volume <= 0.0:
        raise ValueError("Pneumatic cavity has zero or inward-facing rest volume.")

    target_volume = rest_volume if config.target_volume is None else config.target_volume
    assert target_volume is not None
    anchor_particle = min(vertex for triangle in triangles for vertex in triangle)
    cavity_row = builder.add_custom_values(
        **{
            "pneumatic:mode": int(config.mode),
            "pneumatic:world": builder.current_world,
            "pneumatic:anchor_particle": anchor_particle,
            "pneumatic:rest_volume": rest_volume,
            "pneumatic:reference_absolute_pressure": config.reference_absolute_pressure,
            "pneumatic:ambient_pressure": config.ambient_pressure,
            "pneumatic:heat_capacity_ratio": config.heat_capacity_ratio,
            "pneumatic:target_volume": target_volume,
            "pneumatic:volume_stiffness": config.volume_stiffness,
            "pneumatic:bulk_damping": config.bulk_damping,
            "pneumatic:min_volume": rest_volume * config.min_volume_ratio,
            "pneumatic:max_absolute_pressure": config.max_absolute_pressure,
            "pneumatic:volume": rest_volume,
            "pneumatic:absolute_pressure": config.reference_absolute_pressure,
            "pneumatic:volume_rate": 0.0,
            "pneumatic:clamp_flags": 0,
            "pneumatic:pressure_scale": 1.0,
            "pneumatic:prescribed_gauge_pressure": config.prescribed_gauge_pressure,
            "pneumatic:target_volume_scale": 1.0,
        }
    )
    cavity_index = cavity_row["pneumatic:mode"]
    face_start = -1
    for triangle_index, sign in zip(selected_indices, signs, strict=True):
        face_row = builder.add_custom_values(
            **{
                "pneumatic:face_cavity": cavity_index,
                "pneumatic:face_triangle": triangle_index,
                "pneumatic:face_sign": sign,
            }
        )
        if face_start < 0:
            face_start = face_row["pneumatic:face_cavity"]

    return PneumaticCavityHandle(
        cavity_index=cavity_index,
        face_start=face_start,
        face_count=len(selected_indices),
        rest_volume=rest_volume,
    )


def add_inflatable_mesh(
    builder: ModelBuilder,
    *,
    config: PneumaticConfig = _DEFAULT_PNEUMATIC_CONFIG,
    **cloth_mesh_kwargs: Any,
) -> PneumaticCavityHandle:
    """Create a triangular cloth shell and register it as one pneumatic cavity.

    Keyword arguments other than ``config`` are forwarded unchanged to
    :meth:`ModelBuilder.add_cloth_mesh`. The input mesh must be a closed
    two-manifold surface after cloth creation.

    Args:
        builder: Builder that receives particles, triangles, and cavity rows.
        config: Pressure-law and safety settings for the new cavity.
        **cloth_mesh_kwargs: Arguments accepted by
            :meth:`ModelBuilder.add_cloth_mesh`.

    Returns:
        A handle identifying the added cavity.
    """
    triangle_start = len(builder.tri_indices)
    builder.add_cloth_mesh(**cloth_mesh_kwargs)
    return add_pneumatic_cavity(builder, range(triangle_start, len(builder.tri_indices)), config=config)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton 1.7 rod helpers used by the engineRTS cable examples.

Newton main removed ``ModelBuilder.add_rod(positions=...)``, ``add_rod_graph()`` and the four ``newton.utils``
cable helpers in favour of :class:`newton.Rod`. These wrappers keep the examples' authored values and assembly
unchanged: they resolve the released defaults and call the builder's chain/graph assembly that both API generations
share, instead of re-expressing each scene through ``Rod`` material inputs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import NamedTuple

import warp as wp

import newton
from newton._src.geometry.inertia import compute_inertia_capsule
from newton._src.sim.rod import _compute_parallel_transport_quaternions


class CableStiffness(NamedTuple):
    """Per-joint stiffness: stretch [N/m], bend and twist [N·m/rad]."""

    stretch: float
    bend: float
    twist: float


def create_straight_cable_points(start, direction, length: float, num_segments: int) -> list[wp.vec3]:
    if num_segments < 1:
        raise ValueError("num_segments must be >= 1")
    length = float(length)
    if not math.isfinite(length) or length < 0.0:
        raise ValueError("length must be finite and >= 0")
    direction = wp.vec3(direction)
    norm = float(wp.length(direction))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("direction must be finite and non-zero")
    unit = direction / norm
    spacing = length / num_segments
    start = wp.vec3(start)
    return [start + unit * (spacing * i) for i in range(num_segments + 1)]


def create_parallel_transport_cable_quaternions(points: Sequence[wp.vec3], *, twist_total: float = 0.0):
    return _compute_parallel_transport_quaternions(points, twist_total=twist_total)


def create_straight_cable_points_and_quaternions(start, direction, length, num_segments, *, twist_total=0.0):
    points = create_straight_cable_points(start, direction, length, num_segments)
    return points, _compute_parallel_transport_quaternions(points, twist_total=twist_total)


def create_cable_stiffness_from_elastic_moduli(
    youngs_modulus: float,
    radius: float,
    segment_length: float,
    *,
    poissons_ratio: float | None = None,
    shear_modulus: float | None = None,
):
    """``(stretch, bend)``, or :class:`CableStiffness` with a torsional term when a shear input is given."""
    youngs, radius, length = float(youngs_modulus), float(radius), float(segment_length)
    if youngs < 0.0 or radius <= 0.0 or length <= 0.0:
        raise ValueError("youngs_modulus must be >= 0 and radius/segment_length > 0")
    if poissons_ratio is not None and shear_modulus is not None:
        raise ValueError("poissons_ratio and shear_modulus are mutually exclusive")
    stretch = youngs * (math.pi * radius * radius) / length
    bend = youngs * (0.25 * math.pi * radius**4) / length
    if poissons_ratio is None and shear_modulus is None:
        return stretch, bend
    if shear_modulus is None:
        if not -1.0 < float(poissons_ratio) < 0.5:
            raise ValueError("poissons_ratio must satisfy -1 < nu < 0.5")
        shear = youngs / (2.0 * (1.0 + float(poissons_ratio)))
    else:
        shear = float(shear_modulus)
        if shear < 0.0:
            raise ValueError("shear_modulus must be >= 0")
    return CableStiffness(stretch=stretch, bend=bend, twist=shear * 0.5 * math.pi * radius**4 / length)


def _legacy_capsule_mass(builder: newton.ModelBuilder, bodies: list[int], cfg) -> None:
    """Newton 1.7 gave each rod body the mass of its full capsule (cylinder plus hemispherical caps); Newton main
    keeps only the cylinder. Restore the released mass and inertia so the scenes keep their authored loads."""
    density = (cfg or builder.default_shape_cfg).density
    for body in bodies:
        for shape in builder.body_shapes[body]:
            if builder.shape_type[shape] != newton.GeoType.CAPSULE:
                continue
            radius, half_height = float(builder.shape_scale[shape][0]), float(builder.shape_scale[shape][1])
            mass, _com, inertia = compute_inertia_capsule(density, radius, half_height)
            builder.body_mass[body] = mass
            builder.body_inv_mass[body] = 1.0 / mass if mass > 0.0 else 0.0
            builder.body_inertia[body] = inertia
            builder.body_inv_inertia[body] = wp.inverse(inertia) if mass > 0.0 else wp.mat33(0.0)
            break


def _check_origin(body_frame_origin):
    # Newton main always centres rod bodies on the capsule COM, which every ported example requests.
    if body_frame_origin not in (None, "com"):
        raise ValueError(f"body_frame_origin={body_frame_origin!r} is not available in Newton main")


def add_rod(
    builder: newton.ModelBuilder,
    positions=None,
    *,
    quaternions=None,
    radius: float | None = None,
    cfg=None,
    stretch_stiffness=None,
    stretch_damping=None,
    shear_stiffness=None,
    shear_damping=None,
    bend_stiffness=None,
    bend_damping=None,
    twist_stiffness=None,
    twist_damping=None,
    closed: bool | None = None,
    label: str | None = None,
    wrap_in_articulation: bool = True,
    color=None,
    body_frame_origin=None,
    rod=None,
    junction_collision_filter: bool = True,
):
    """Legacy ``add_rod``: either ``rod=`` (forwarded) or an ordered ``positions`` chain."""
    _check_origin(body_frame_origin)
    stiffness = {
        "stretch_stiffness": stretch_stiffness,
        "stretch_damping": stretch_damping,
        "shear_stiffness": shear_stiffness,
        "shear_damping": shear_damping,
        "bend_stiffness": bend_stiffness,
        "bend_damping": bend_damping,
        "twist_stiffness": twist_stiffness,
        "twist_damping": twist_damping,
    }
    if rod is not None:
        return builder.add_rod(
            rod=rod,
            cfg=cfg,
            label=label,
            wrap_in_articulation=wrap_in_articulation,
            color=color,
            junction_collision_filter=junction_collision_filter,
            **stiffness,
        )
    positions = [wp.vec3(p) for p in positions]
    if len(positions) < 3:
        raise ValueError("add_rod: requires at least 2 segments")
    if quaternions is not None and len(quaternions) != len(positions) - 1:
        raise ValueError("add_rod: quaternions must have one element per segment")
    result = builder._add_rod_chain(
        positions,
        quaternions=quaternions,
        radius=0.1 if radius is None else radius,
        cfg=cfg,
        total_mass=None,
        closed=bool(closed),
        label=label,
        wrap_in_articulation=wrap_in_articulation,
        color=color,
        **stiffness,
    )
    _legacy_capsule_mass(builder, result[0], cfg)
    return result


def add_rod_graph(
    builder: newton.ModelBuilder,
    node_positions,
    edges,
    *,
    radius: float = 0.1,
    cfg=None,
    stretch_stiffness=None,
    stretch_damping=None,
    shear_stiffness=None,
    shear_damping=None,
    bend_stiffness=None,
    bend_damping=None,
    twist_stiffness=None,
    twist_damping=None,
    label: str | None = None,
    wrap_in_articulation: bool = True,
    quaternions=None,
    junction_collision_filter: bool = True,
    color=None,
    body_frame_origin=None,
):
    """Legacy ``add_rod_graph`` over explicit node positions and edges."""
    _check_origin(body_frame_origin)
    result = builder._add_rod_graph(
        node_positions=[wp.vec3(p) for p in node_positions],
        edges=[tuple(edge) for edge in edges],
        radius=radius,
        cfg=cfg,
        stretch_stiffness=1.0e5 if stretch_stiffness is None else stretch_stiffness,
        stretch_damping=0.0 if stretch_damping is None else stretch_damping,
        shear_stiffness=shear_stiffness,
        shear_damping=shear_damping,
        bend_stiffness=0.0 if bend_stiffness is None else bend_stiffness,
        bend_damping=0.0 if bend_damping is None else bend_damping,
        twist_stiffness=twist_stiffness,
        twist_damping=twist_damping,
        label=label,
        wrap_in_articulation=wrap_in_articulation,
        quaternions=quaternions,
        junction_collision_filter=junction_collision_filter,
        color=color,
    )
    _legacy_capsule_mass(builder, result[0], cfg)
    return result

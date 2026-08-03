# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Procedural triangle-mesh generators for LBM immersed solids.

Immersed-boundary LBM needs a triangle mesh per rigid body in the body-local
frame. Primitive shapes (box, cylinder, sphere, capsule) are triangulated here
from their ``shape_scale`` half-extents; MESH/CONVEX_MESH shapes fall back to
their Newton mesh geometry.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import warp as wp

from ...geometry.types import GeoType


def _unit_box_faces() -> np.ndarray:
    """Return the 12 triangles of a unit box as a (12, 3) index array."""
    return np.array(
        [
            [0, 1, 2], [2, 1, 3],
            [4, 6, 5], [5, 6, 7],
            [0, 4, 1], [1, 4, 5],
            [2, 3, 6], [6, 3, 7],
            [0, 2, 4], [4, 2, 6],
            [1, 5, 3], [3, 5, 7],
        ],
        dtype=np.int32,
    )


def box_mesh(hx: float, hy: float, hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) of an axis-aligned box with half-extents (hx, hy, hz)."""
    vertices = np.array(
        [
            [-hx, -hy, -hz], [hx, -hy, -hz], [-hx, hy, -hz], [hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [-hx, hy, hz], [hx, hy, hz],
        ],
        dtype=np.float32,
    )
    return vertices, _unit_box_faces()


def cylinder_mesh(radius: float, half_height: float, segments: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) of a cylinder with axis aligned to Z."""
    theta = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    xs = radius * np.cos(theta)
    ys = radius * np.sin(theta)
    # Vertices: ring at -h (0..s-1), ring at +h (s..2s-1), bottom cap center (2s), top cap center (2s+1)
    ring_bottom = np.stack([xs, ys, np.full(segments, -half_height)], axis=1)
    ring_top = np.stack([xs, ys, np.full(segments, half_height)], axis=1)
    vertices = np.concatenate([ring_bottom, ring_top], axis=0).astype(np.float32)
    bottom_center = np.array([[0.0, 0.0, -half_height]], dtype=np.float32)
    top_center = np.array([[0.0, 0.0, half_height]], dtype=np.float32)
    vertices = np.concatenate([vertices, bottom_center, top_center], axis=0)

    faces: list[list[int]] = []
    for i in range(segments):
        j = (i + 1) % segments
        # Side quads (two triangles), outward winding
        faces.append([i, j, j + segments])
        faces.append([i, j + segments, i + segments])
        # Bottom cap (outward normal = -Z)
        faces.append([j, i, 2 * segments])
        # Top cap (outward normal = +Z)
        faces.append([i, j, 2 * segments + 1])
    return vertices, np.array(faces, dtype=np.int32)


def sphere_mesh(radius: float, stacks: int = 8, slices: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) of a UV sphere centered at the origin."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for i in range(stacks + 1):
        phi = np.pi * float(i) / stacks
        y = radius * np.cos(phi)
        r = radius * np.sin(phi)
        for j in range(slices):
            theta = 2.0 * np.pi * float(j) / slices
            vertices.append([r * np.cos(theta), y, r * np.sin(theta)])
    for i in range(stacks):
        for j in range(slices):
            a = i * slices + j
            b = a + slices
            c = b + 1 if (j + 1) < slices else i * slices
            d = a + 1 if (j + 1) < slices else a + 1 - slices
            faces.append([a, b, c])
            faces.append([a, c, d])
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def capsule_mesh(radius: float, half_height: float, stacks: int = 6, slices: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) of a capsule with axis aligned to Z."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    n_side = 4  # side stack bands
    for i in range(n_side + 1):
        h = -half_height + (2.0 * half_height * float(i) / n_side)
        for j in range(slices):
            theta = 2.0 * np.pi * float(j) / slices
            vertices.append([radius * np.cos(theta), h, radius * np.sin(theta)])
    # Top hemisphere
    base = len(vertices)
    for i in range(1, stacks + 1):
        phi = 0.5 * np.pi * float(i) / stacks
        y = half_height + radius * np.sin(phi)
        r = radius * np.cos(phi)
        for j in range(slices):
            theta = 2.0 * np.pi * float(j) / slices
            vertices.append([r * np.cos(theta), y, r * np.sin(theta)])
    # Bottom hemisphere
    for i in range(1, stacks + 1):
        phi = 0.5 * np.pi * float(i) / stacks
        y = -half_height - radius * np.sin(phi)
        r = radius * np.cos(phi)
        for j in range(slices):
            theta = 2.0 * np.pi * float(j) / slices
            vertices.append([r * np.cos(theta), y, r * np.sin(theta)])

    def _ring_quads(start: int, count: int) -> None:
        for j in range(slices):
            a = start + j
            b = a + slices
            c = b + 1 if (j + 1) < slices else start
            d = a + 1 if (j + 1) < slices else start
            faces.append([a, c, b])
            faces.append([a, d, c])

    _ring_quads(0, slices)  # side band 0->1 ... uses (n_side+1)*slices total side verts
    for i in range(1, n_side):
        _ring_quads(i * slices, slices)
    # Connect side top to top hemisphere base
    top_start = (n_side + 1) * slices
    for j in range(slices):
        a = n_side * slices + j
        b = top_start + j
        c = top_start + (j + 1) % slices
        d = n_side * slices + (j + 1) % slices
        faces.append([a, c, b])
        faces.append([a, d, c])
    # Top hemisphere rings (downward facing caps)
    for i in range(stacks - 1):
        _ring_quads(top_start + i * slices, slices)
    # Bottom hemisphere rings
    bot_start = top_start + stacks * slices
    for i in range(stacks - 1):
        _ring_quads(bot_start + i * slices, slices)
    # Connect side bottom to bottom hemisphere top ring (flip winding)
    for j in range(slices):
        a = j
        b = bot_start + (stacks - 1) * slices + j
        c = bot_start + (stacks - 1) * slices + (j + 1) % slices
        d = (j + 1) % slices
        faces.append([a, b, c])
        faces.append([a, c, d])
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


def generate_shape_mesh(
    shape_type: int, shape_scale: Any, shape_source: Any = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Generate (vertices, faces) for one shape in its local frame.

    Args:
        shape_type: :class:`newton.GeoType` member of the shape.
        shape_scale: 3-vector scale (half-extents) of the shape.
        shape_source: Optional Newton geometry (e.g. ``newton.Mesh``) for
            MESH/CONVEX_MESH shapes.

    Returns:
        ``(vertices, faces)`` float32/int32 arrays, or ``None`` if the shape
        has no volumetric geometry usable by the immersed-boundary method.
    """
    scale = np.asarray(shape_scale, dtype=np.float32)
    s0, s1, s2 = float(scale[0]), float(scale[1]), float(scale[2])

    if shape_type == int(GeoType.BOX):
        return box_mesh(s0, s1, s2)
    if shape_type == int(GeoType.CYLINDER):
        return cylinder_mesh(s0, max(s1, 1.0e-6))
    if shape_type == int(GeoType.SPHERE):
        return sphere_mesh(s0)
    if shape_type == int(GeoType.CAPSULE):
        return capsule_mesh(s0, max(s1, 1.0e-6))
    if shape_type in (int(GeoType.MESH), int(GeoType.CONVEX_MESH)):
        if shape_source is None:
            return None
        try:
            vertices = np.asarray(shape_source.vertices, dtype=np.float32)
            faces = np.asarray(shape_source.indices, dtype=np.int32)
        except (AttributeError, ValueError):
            return None
        if vertices.size == 0 or faces.size == 0:
            return None
        # Apply per-shape scale.
        return vertices * scale, faces
    return None

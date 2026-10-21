# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared watertight measuring-cup geometry for collision and Blender export."""

import math
from itertools import pairwise

import numpy as np

CUP_RADIUS = 0.040
CUP_BOTTOM_RADIUS = 0.037
CUP_HEIGHT = 0.085
CUP_WALL = 0.015
HANDLE_CENTER = np.array((-0.060, 0.0, 0.050))
HANDLE_BOXES = (
    ((-0.060, 0.0, 0.062), (0.025, 0.006, 0.003)),
    ((-0.060, 0.0, 0.038), (0.025, 0.006, 0.003)),
    ((-0.082, 0.0, 0.050), (0.003, 0.006, 0.009)),
)


def cup_mesh(*, segments=48):
    """Return a closed tapered cup shell with a pouring lip toward local +y [m]."""
    profile = (
        (0, 0),
        (CUP_BOTTOM_RADIUS, 0),
        (CUP_BOTTOM_RADIUS, 0.008),
        (CUP_RADIUS, CUP_HEIGHT),
        (CUP_RADIUS - CUP_WALL, CUP_HEIGHT),
        (CUP_BOTTOM_RADIUS - CUP_WALL, CUP_WALL),
        (0, CUP_WALL),
    )
    vertices, rings, faces = [], [], []
    for radius, height in profile:
        if radius == 0:
            rings.append([len(vertices)])
            vertices.append((0.0, 0.0, height))
            continue
        ring = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            distance = math.atan2(math.sin(angle - math.pi / 2), math.cos(angle - math.pi / 2))
            lip = math.exp(-((distance / 0.24) ** 2)) if height == CUP_HEIGHT else 0.0
            r = radius + 0.006 * lip
            ring.append(len(vertices))
            vertices.append((r * math.cos(angle), r * math.sin(angle), height - 0.003 * lip))
        rings.append(ring)
    for a, b in pairwise(rings):
        for i in range(segments):
            j = (i + 1) % segments
            if len(a) == 1:
                faces.append((a[0], b[j], b[i]))
            elif len(b) == 1:
                faces.append((a[i], a[j], b[0]))
            else:
                faces.extend(((a[i], a[j], b[j]), (a[i], b[j], b[i])))
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def cup_inner_radius(height):
    """Return the unflared inner radius at a local height [m]."""
    fraction = np.clip((height - CUP_WALL) / (CUP_HEIGHT - CUP_WALL), 0.0, 1.0)
    return CUP_BOTTOM_RADIUS - CUP_WALL + (CUP_RADIUS - CUP_BOTTOM_RADIUS) * fraction


def cup_visual_mesh(*, segments=96):
    """Return a 2 mm display shell sharing the collision cup's inner surface.

    MPM's numerical wall extends outward from this surface. It is deliberately
    thicker than the visible wall, preserving the material boundary and floor.
    """
    vertices, triangles = cup_mesh(segments=segments)
    for ring in range(3):
        selection = slice(1 + ring * segments, 1 + (ring + 1) * segments)
        xy = vertices[selection, :2]
        radius = np.linalg.norm(xy, axis=1, keepdims=True)
        vertices[selection, :2] *= (radius - (CUP_WALL - 0.002)) / radius
    return vertices, triangles

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared forged chef-knife geometry for physics and Blender authoring."""

import numpy as np

HANDLE_HALF_WIDTH = 0.009
# Includes the W1 visual pad overhang, flush rivets and drive tracking margin.
HANDLE_GRIP_OPENING = HANDLE_HALF_WIDTH + 0.0018


def blade_mesh():
    """Closed wedge with a curved cutting tip and a resolved 12 mm spine."""
    outline = (
        (0.060, -0.055, 0.0005),
        (0.190, -0.055, 0.0005),
        (0.205, -0.052, 0.0005),
        (0.218, -0.045, 0.0005),
        (0.230, -0.032, 0.0005),
        (0.240, -0.015, 0.0005),
        (0.225, 0.005, 0.0045),
        (0.200, 0.014, 0.0060),
        (0.060, 0.015, 0.0060),
    )
    count = len(outline)
    vertices = np.array([(x, sign * width, z) for sign in (-1, 1) for x, z, width in outline], dtype=np.float32)
    faces = []
    for i in range(1, count - 1):
        faces.extend(((0, i, i + 1), (count, count + i + 1, count + i)))
    for i in range(count):
        j = (i + 1) % count
        faces.extend(((i, j + count, j), (i, i + count, j + count)))
    return vertices, np.array(faces, dtype=np.int32)

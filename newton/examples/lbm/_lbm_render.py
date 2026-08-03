# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared rendering helpers for the LBM examples."""

from __future__ import annotations

import numpy as np
import warp as wp


def vorticity_to_image(
    projection: wp.array2d, vmax: float | None = None
) -> tuple[np.ndarray, float]:
    """Convert a signed-vorticity projection buffer to a uint8 RGB image.

    Solid cells (value >= 999.0) are drawn mid-gray; fluid cells use a
    blue-white-red map centered on zero vorticity.

    Args:
        projection: 2D float32 buffer from ``SolverLBM.vorticity_projection``.
        vmax: Absolute value used to normalize the color map. If ``None``, the
            maximum absolute fluid value is used.

    Returns:
        ``(rgb_image, vmax)`` where ``rgb_image`` has shape ``(H, W, 3)``.
    """
    raw = np.asarray(projection.numpy(), dtype=np.float32)
    solid = raw >= 999.0
    fluid = raw.copy()
    fluid[solid] = 0.0
    if vmax is None:
        vmax = float(np.max(np.abs(fluid))) if np.any(~solid) else 1.0
    vmax = max(vmax, 1.0e-6)

    # Signed map: blue (negative) -> white (zero) -> red (positive).
    normalized = np.clip((fluid / vmax + 1.0) * 0.5, 0.0, 1.0)
    rgb = np.empty((raw.shape[0], raw.shape[1], 3), dtype=np.uint8)
    r = np.clip(normalized * 2.0 - 1.0, 0.0, 1.0)
    b = np.clip(1.0 - normalized * 2.0, 0.0, 1.0)
    g = np.clip(1.0 - np.abs(normalized * 2.0 - 1.0), 0.0, 1.0)
    rgb[:, :, 0] = (r * 255.0).astype(np.uint8)
    rgb[:, :, 1] = (g * 255.0).astype(np.uint8)
    rgb[:, :, 2] = (b * 255.0).astype(np.uint8)
    rgb[solid] = (120, 120, 120)
    return rgb, vmax


def domain_box_points(nx: int, ny: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the 8 corners and 12 edges of the LBM domain box (lattice units)."""
    corners = np.array(
        [
            [0.0, 0.0, 0.0], [nx, 0.0, 0.0], [0.0, ny, 0.0], [nx, ny, 0.0],
            [0.0, 0.0, nz], [nx, 0.0, nz], [0.0, ny, nz], [nx, ny, nz],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [0, 1], [2, 3], [4, 5], [6, 7],
            [0, 2], [1, 3], [4, 6], [5, 7],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        dtype=np.int32,
    )
    return corners, edges

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small geometry helpers shared by the MPM Lite vertical-slice examples."""

from __future__ import annotations

import meshio
import numpy as np

import newton


def add_particles(
    builder: newton.ModelBuilder,
    points: np.ndarray,
    particle_volume: float,
    density: float,
    particle_radius: float,
) -> None:
    """Add uniformly weighted MPM particles to *builder*.

    Args:
        builder: Destination Newton model builder.
        points: Particle positions [m], shape ``[particle_count, 3]``.
        particle_volume: Rest volume assigned to every particle [m^3].
        density: Material density [kg/m^3].
        particle_radius: Visualization radius [m].
    """
    mass = density * particle_volume
    builder.add_particles(
        pos=points.tolist(),
        vel=np.zeros_like(points).tolist(),
        mass=[mass] * len(points),
        radius=[particle_radius] * len(points),
    )


def sample_tetrahedra(points: np.ndarray, tetrahedra: np.ndarray, count: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample *count* points uniformly from a tetrahedral volume mesh."""
    vertices = points[tetrahedra]
    volumes = (
        np.abs(
            np.einsum(
                "ij,ij->i",
                vertices[:, 1] - vertices[:, 0],
                np.cross(vertices[:, 2] - vertices[:, 0], vertices[:, 3] - vertices[:, 0]),
            )
        )
        / 6.0
    )
    indices = rng.choice(len(tetrahedra), size=count, replace=True, p=volumes / volumes.sum())
    weights = -np.log(rng.random_sample((count, 4)))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.einsum("ni,nij->nj", weights, vertices[indices]).astype(np.float32)


def load_vtk_tetrahedra(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load points and tetrahedra from a VTK volume asset."""
    mesh = meshio.read(path)
    return np.asarray(mesh.points), np.asarray(mesh.cells_dict["tetra"], dtype=np.int64)


def surface_nodes(grid_size: tuple[int, int, int]) -> np.ndarray:
    """Return the unique nodes on the six faces of a rectangular grid."""
    nx, ny, nz = grid_size
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    k, l = np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij")
    m, n = np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij")
    faces = (
        np.column_stack((i.ravel(), j.ravel(), np.zeros(i.size, dtype=np.int32))),
        np.column_stack((i.ravel(), j.ravel(), np.full(i.size, nz - 1, dtype=np.int32))),
        np.column_stack((k.ravel(), np.zeros(k.size, dtype=np.int32), l.ravel())),
        np.column_stack((k.ravel(), np.full(k.size, ny - 1, dtype=np.int32), l.ravel())),
        np.column_stack((np.zeros(m.size, dtype=np.int32), m.ravel(), n.ravel())),
        np.column_stack((np.full(m.size, nx - 1, dtype=np.int32), m.ravel(), n.ravel())),
    )
    return np.unique(np.concatenate(faces).astype(np.int32), axis=0)


def boundary_band_nodes(grid_size: tuple[int, int, int], width: int) -> np.ndarray:
    """Return grid nodes within *width* nodes of a rectangular grid boundary."""
    if width < 1:
        raise ValueError("width must be positive.")
    nx, ny, nz = grid_size
    ij = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij"), axis=-1).reshape(-1, 2)
    ik = np.stack(np.meshgrid(np.arange(nx), np.arange(nz), indexing="ij"), axis=-1).reshape(-1, 2)
    jk = np.stack(np.meshgrid(np.arange(ny), np.arange(nz), indexing="ij"), axis=-1).reshape(-1, 2)
    faces = []
    for offset in range(width):
        faces.extend(
            (
                np.column_stack((ij, np.full(len(ij), offset))),
                np.column_stack((ij, np.full(len(ij), nz - 1 - offset))),
                np.column_stack((ik[:, 0], np.full(len(ik), offset), ik[:, 1])),
                np.column_stack((ik[:, 0], np.full(len(ik), ny - 1 - offset), ik[:, 1])),
                np.column_stack((np.full(len(jk), offset), jk)),
                np.column_stack((np.full(len(jk), nx - 1 - offset), jk)),
            )
        )
    return np.unique(np.concatenate(faces).astype(np.int32), axis=0)

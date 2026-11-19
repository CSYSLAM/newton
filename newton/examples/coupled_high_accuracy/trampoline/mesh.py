# SPDX-License-Identifier: Apache-2.0
"""Local OBJ support and an asset-independent circular trampoline mesh."""

from pathlib import Path

import numpy as np

DEFAULT_OBJ = Path(__file__).resolve().parents[1] / "assets" / "trampoline_v1051.obj"


def load_obj_triangles(path: Path) -> tuple[list[tuple[float, float, float]], list[int]]:
    """Load positions and triangulated faces from a Wavefront OBJ file."""
    vertices, indices = [], []
    with path.open(encoding="utf-8") as obj_file:
        for line_number, line in enumerate(obj_file, start=1):
            fields = line.partition("#")[0].split()
            if not fields:
                continue
            if fields[0] == "v":
                if len(fields) < 4:
                    raise ValueError(f"Invalid vertex at {path}:{line_number}")
                vertices.append(tuple(map(float, fields[1:4])))
            elif fields[0] == "f":
                if len(fields) < 4:
                    raise ValueError(f"Invalid face at {path}:{line_number}")
                face = []
                for field in fields[1:]:
                    obj_index = int(field.split("/", maxsplit=1)[0])
                    vertex_index = obj_index - 1 if obj_index > 0 else len(vertices) + obj_index
                    if obj_index == 0 or vertex_index < 0 or vertex_index >= len(vertices):
                        raise ValueError(f"Invalid vertex index at {path}:{line_number}")
                    face.append(vertex_index)
                for corner in range(1, len(face) - 1):
                    indices.extend((face[0], face[corner], face[corner + 1]))
    if not vertices or not indices:
        raise ValueError(f"OBJ contains no triangle mesh: {path}")
    return vertices, indices


def triangle_mesh_area(vertices, indices) -> float:
    positions = np.asarray(vertices, dtype=np.float64)
    triangles = positions[np.asarray(indices, dtype=np.int32).reshape(-1, 3)]
    double_areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    )
    area = float(0.5 * double_areas.sum())
    if not np.isfinite(area) or area <= 0:
        raise ValueError("The trampoline must have finite, positive surface area")
    return area


def boundary_vertex_indices(indices) -> np.ndarray:
    """Find the fixed rim by topology; imported OBJ vertex order is not a boundary contract."""
    triangles = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("The trampoline mesh has nonmanifold edges")
    boundary = np.unique(edges[counts == 1])
    if not len(boundary):
        raise ValueError("The trampoline mesh needs an open boundary to fix its rim")
    return boundary


def generate_trampoline_mesh(radius: float = 1.0, resolution: int = 16, boundary_segments: int = 64):
    """Triangulate a flat disk with an approximately equilateral interior and a circular rim."""
    from scipy.spatial import Delaunay

    if not np.isfinite(radius) or radius <= 0 or resolution < 2 or boundary_segments < 3:
        raise ValueError("Use a positive radius, resolution >= 2, and at least three boundary segments")
    spacing = radius / resolution
    dy = np.sqrt(3) * spacing / 2
    rows = int(np.ceil(radius / dy))
    row, column = np.meshgrid(np.arange(-rows, rows + 1), np.arange(-resolution, resolution + 1), indexing="ij")
    interior = np.column_stack(((column.ravel() + 0.5 * (row.ravel() % 2)) * spacing, row.ravel() * dy))
    # Leave room between the lattice and the rim to avoid sliver triangles at the circular cut.
    interior = interior[np.linalg.norm(interior, axis=1) < radius - 0.8 * spacing]
    angle = np.arange(boundary_segments) * (2 * np.pi / boundary_segments)
    rim = radius * np.column_stack((np.cos(angle), np.sin(angle)))
    points = np.vstack((rim, interior))
    triangles = Delaunay(points).simplices.copy()
    a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
    reversed_faces = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]) < 0
    triangles[reversed_faces] = triangles[reversed_faces, ::-1]
    vertices = np.column_stack((points, np.zeros(len(points))))
    return [tuple(point) for point in vertices], triangles.ravel().tolist()


def load_trampoline_mesh(path: Path | None = None):
    """Honor explicit assets and use a generated disk when the optional historical OBJ is absent."""
    if path is not None:
        return load_obj_triangles(Path(path).expanduser())
    if DEFAULT_OBJ.is_file():
        return load_obj_triangles(DEFAULT_OBJ)
    return generate_trampoline_mesh()

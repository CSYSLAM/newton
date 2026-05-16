"""Exterior tet removal via ray-casting inside/outside test.

For each tet, cast a ray from its centroid and count intersections
with the surface mesh. Odd count = inside, even = outside.
"""

import numpy as np

from .mesh_data import NEIGHBOR_FACES


def remove_exterior_tets(del_tet, surface_faces: np.ndarray,
                         n_sup: int = 0, verbose: bool = False) -> None:
    """Remove tets whose centroids are outside the surface mesh.

    Uses ray-casting in the +Z direction from each tet centroid.
    Tets with an odd number of ray-surface intersections are inside.

    Args:
        del_tet: DelaunayTetrahedralizer instance
        surface_faces: (M, 3) int32 surface face indices (0-based)
        n_sup: Number of super-vertices to offset surface face indices
        verbose: Print progress
    """
    # Get surface triangles with super-vertex offset
    faces = surface_faces + n_sup

    # Collect active tet centroids
    centroids = []
    tet_ids = []
    for i in range(len(del_tet.tets)):
        if del_tet._is_tet_deleted(i):
            continue
        t = del_tet.tets[i]
        # Skip tets containing super-vertices
        if any(int(v) < n_sup for v in t):
            continue
        c = (del_tet.points[t[0]] + del_tet.points[t[1]] +
             del_tet.points[t[2]] + del_tet.points[t[3]]) * 0.25
        centroids.append(c)
        tet_ids.append(i)

    if not centroids:
        return

    centroids = np.array(centroids)
    n_tets = len(tet_ids)

    if verbose:
        print(f"  Testing {n_tets} tets for inside/outside...")

    # Vectorized ray-triangle intersection
    inside = _batch_ray_cast(centroids, np.array(del_tet.points), faces)

    # Delete exterior tets
    deleted_count = 0
    for idx, tet_id in enumerate(tet_ids):
        if not inside[idx]:
            del_tet._delete_tet(tet_id)
            deleted_count += 1

    if verbose:
        print(f"  Removed {deleted_count} exterior tets, kept {n_tets - deleted_count}")


def _batch_ray_cast(origins: np.ndarray, vertices: np.ndarray,
                     faces: np.ndarray) -> np.ndarray:
    """Batch ray-triangle intersection test.

    Cast rays from origins in +Z direction, count intersections with
    each surface triangle. Return boolean array (True = inside).

    Uses the Moller-Trumbore algorithm vectorized over triangles.
    """
    n_origins = len(origins)
    n_faces = len(faces)

    # Precompute triangle vertices
    v0 = vertices[faces[:, 0]]  # (n_faces, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    edge1 = v1 - v0  # (n_faces, 3)
    edge2 = v2 - v0

    # Ray direction
    direction = np.array([0.0, 0.0, 1.0])

    inside = np.zeros(n_origins, dtype=bool)

    for i in range(n_origins):
        origin = origins[i]
        pvec = np.cross(direction, edge2)  # (n_faces, 3)

        det = np.sum(edge1 * pvec, axis=1)  # (n_faces,)

        valid = np.abs(det) > 1e-12

        inv_det = np.zeros(n_faces)
        inv_det[valid] = 1.0 / det[valid]

        tvec = origin[np.newaxis, :] - v0

        u = np.sum(tvec * pvec, axis=1) * inv_det
        valid &= (u >= -1e-10) & (u <= 1.0 + 1e-10)

        qvec = np.cross(tvec, edge1)
        v = np.sum(direction * qvec, axis=1) * inv_det
        valid &= (v >= -1e-10) & (u + v <= 1.0 + 1e-10)

        t = np.sum(edge2 * qvec, axis=1) * inv_det
        valid &= t > 1e-10  # Intersection in front of origin

        count = int(np.sum(valid))
        inside[i] = (count % 2) == 1

    return inside

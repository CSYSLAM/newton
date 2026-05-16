"""Utility functions for tetrahedral mesh processing.

Includes vertex deduplication, normalization, noise, and mesh cleanup.
"""

import numpy as np


def dedup_vertices(vertices: np.ndarray, faces: np.ndarray, eps: float = 1e-10):
    """Remove duplicate vertices and remap face indices.

    Args:
        vertices: (N, 3) float64 vertex positions
        faces: (M, 3) int32 face indices
        eps: distance threshold for considering vertices identical

    Returns:
        (vertices_dedup, faces_remapped) with duplicates merged
    """
    n = len(vertices)
    if n == 0:
        return vertices, faces

    # Sort vertices by (x, y, z) for grouping
    order = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    remap = np.full(n, -1, dtype=np.int32)

    new_idx = 0
    remap[order[0]] = 0
    for i in range(1, n):
        curr = order[i]
        prev = order[i - 1]
        if np.all(np.abs(vertices[curr] - vertices[prev]) < eps):
            remap[curr] = remap[prev]
        else:
            new_idx += 1
            remap[curr] = new_idx

    # Build deduplicated vertex array
    unique_count = new_idx + 1
    new_vertices = np.zeros((unique_count, 3), dtype=np.float64)
    for i in range(n):
        new_vertices[remap[i]] = vertices[i]

    # Remap faces
    new_faces = remap[faces]

    # Remove degenerate faces (where any two vertices are the same)
    valid = np.ones(len(new_faces), dtype=bool)
    for i in range(3):
        for j in range(i + 1, 3):
            valid &= new_faces[:, i] != new_faces[:, j]

    return new_vertices, new_faces[valid].astype(np.int32)


def normalize_to_unit_cube(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalize vertices to fit within a unit cube.

    Returns:
        (normalized_vertices, offset, scale) where
        original = normalized * scale + offset
    """
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extent = vmax - vmin
    scale = float(np.max(extent))
    if scale < 1e-15:
        scale = 1.0
    normalized = (vertices - vmin) / scale
    return normalized, vmin, scale


def add_noise(vertices: np.ndarray, magnitude: float = 1e-6, seed: int = 42) -> np.ndarray:
    """Add small random noise to break degeneracies.

    Modifies vertices in-place and returns them.
    """
    rng = np.random.RandomState(seed)
    noise = rng.uniform(-magnitude, magnitude, size=vertices.shape)
    vertices += noise
    return vertices


def collect_surface_edges(faces: np.ndarray) -> set[tuple[int, int]]:
    """Collect all edges from surface faces as sorted tuples."""
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edges.add((min(a, b), max(a, b)))
    return edges


def collect_surface_faces_set(faces: np.ndarray) -> set[tuple[int, int, int]]:
    """Collect all surface faces as sorted tuples."""
    face_set = set()
    for f in faces:
        s = sorted([int(f[0]), int(f[1]), int(f[2])])
        face_set.add((s[0], s[1], s[2]))
    return face_set


def ensure_positive_volume(mesh) -> None:
    """Flip tets with negative volume so all have consistent positive orientation.

    Modifies mesh.tets in-place.
    """
    from .predicates import tet_volume_signed

    for i in range(mesh.num_tets()):
        if mesh.is_tet_deleted(i):
            continue
        v = mesh.vertices
        t = mesh.tets[i]
        vol = tet_volume_signed(v[t[0]], v[t[1]], v[t[2]], v[t[3]])
        if vol < 0:
            # Swap vertices 0 and 1 to flip orientation
            mesh.tets[i][0], mesh.tets[i][1] = mesh.tets[i][1], mesh.tets[i][0]
            # Also swap corresponding neighbor entries
            mesh.neighbors[i][0], mesh.neighbors[i][1] = mesh.neighbors[i][1], mesh.neighbors[i][0]


def remove_disconnected_islands(mesh) -> None:
    """Keep only the largest connected component of tetrahedra.

    Uses BFS from the first non-deleted tet.
    """
    from .mesh_data import NEIGHBOR_FACES

    n = mesh.num_tets()
    if n == 0:
        return

    # Find first non-deleted tet
    start = -1
    for i in range(n):
        if not mesh.is_tet_deleted(i):
            start = i
            break
    if start < 0:
        return

    # BFS
    visited = np.zeros(n, dtype=bool)
    queue = [start]
    visited[start] = True
    component = [start]

    while queue:
        tet = queue.pop(0)
        for face_idx in range(4):
            enc = mesh.face_neighbor(tet, face_idx)
            if enc < 0:
                continue
            nbr_tet, _ = mesh.decode_neighbor(enc)
            if nbr_tet >= 0 and not visited[nbr_tet] and not mesh.is_tet_deleted(nbr_tet):
                visited[nbr_tet] = True
                queue.append(nbr_tet)
                component.append(nbr_tet)

    # Delete all tets not in the largest component
    if len(component) < n:
        for i in range(n):
            if not visited[i] and not mesh.is_tet_deleted(i):
                mesh.delete_tet(i)


def compact_mesh(mesh) -> tuple[np.ndarray, np.ndarray]:
    """Remove deleted tets and unused vertices, return compact arrays.

    Returns:
        (vertices, tet_indices) where vertices is (K, 3) float64 and
        tet_indices is (T, 4) int32 with compact vertex indexing
    """
    # Collect active tets
    active_tets = []
    for i in range(mesh.num_tets()):
        if not mesh.is_tet_deleted(i):
            active_tets.append(mesh.tets[i].copy())

    if not active_tets:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 4), dtype=np.int32)

    tet_array = np.array(active_tets, dtype=np.int32)

    # Find used vertices
    used = np.unique(tet_array)
    if len(used) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 4), dtype=np.int32)

    # Build remap
    max_v = int(used.max()) + 1
    remap = np.full(max_v, -1, dtype=np.int32)
    for new_idx, old_idx in enumerate(used):
        remap[old_idx] = new_idx

    # Remap tet indices
    valid_mask = np.all(tet_array < max_v, axis=1)
    tet_array = tet_array[valid_mask]
    tet_remapped = remap[tet_array]

    # Compact vertices
    vertices = mesh.vertices[used]

    return vertices, tet_remapped

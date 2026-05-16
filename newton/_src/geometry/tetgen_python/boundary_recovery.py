"""Conforming Delaunay: boundary edge recovery via Steiner point insertion.

Ensures all surface edges appear as edges in the Delaunay tetrahedralization
by inserting Steiner points at edge midpoints when edges are missing.
"""

import numpy as np

from .mesh_data import NEIGHBOR_FACES, edge_key, sorted_face


def recover_plc(mesh, surface_edges: list[tuple[int, int]],
                surface_faces: np.ndarray, verbose: bool = False) -> bool:
    """Enforce all surface edges in the Delaunay tetrahedralization.

    For each surface edge that doesn't exist in the tet mesh,
    insert Steiner points at the midpoint to subdivide it.

    Args:
        mesh: _TetMeshWrapper instance with vertices, tets, deleted, etc.
        surface_edges: List of (a, b) vertex index pairs (0-based)
        surface_faces: Surface face array (for reference)
        verbose: Print progress

    Returns True if all edges were recovered.
    """
    # Track edges as lists of sub-edges (for Steiner point subdivision)
    all_edges: list[list[int]] = []
    for a, b in surface_edges:
        all_edges.append([a, b])

    max_subdivisions = 10
    for iteration in range(max_subdivisions):
        missing_edges = []
        for edge_list in all_edges:
            for i in range(len(edge_list) - 1):
                a, b = edge_list[i], edge_list[i + 1]
                if not _check_edge_present(mesh, a, b):
                    missing_edges.append((a, b, edge_list, i))

        if not missing_edges:
            if verbose:
                print(f"  All edges recovered after {iteration} iterations")
            return True

        if verbose:
            print(f"  Iteration {iteration}: {len(missing_edges)} missing edges")

        for a, b, edge_list, idx in missing_edges:
            # Insert Steiner point at midpoint
            midpoint = (mesh.vertices[a] + mesh.vertices[b]) * 0.5
            new_idx = mesh.insert_single_point(midpoint)

            # Update edge tracking
            edge_list.insert(idx + 1, new_idx)

    if verbose:
        print(f"  Warning: could not recover all edges after {max_subdivisions} iterations")
    return False


def _check_edge_present(mesh, a: int, b: int) -> bool:
    """Check if edge (a,b) exists as an edge in any tet.

    An edge exists if there's a tet where a and b share a face.
    """
    for i in range(len(mesh.tets)):
        if mesh._is_tet_deleted(i):
            continue
        tet = mesh.tets[i]
        if a not in tet or b not in tet:
            continue
        # Check if a and b share a face
        for fi in range(4):
            f = NEIGHBOR_FACES[fi]
            face_verts = [int(tet[f[0]]), int(tet[f[1]]), int(tet[f[2]])]
            if a in face_verts and b in face_verts:
                return True
    return False
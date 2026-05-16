"""Main pipeline for pure-Python Delaunay tetrahedralization.

Orchestrates: dedup → normalize → convex shortcut → Delaunay (scipy) →
exterior removal → cleanup → scale back.

Provides TetGenPython class with the same interface as the external
tetgen package for drop-in replacement.
"""

import numpy as np

from .mesh_data import NEIGHBOR_FACES, sorted_face
from .utils import (
    dedup_vertices, normalize_to_unit_cube, add_noise,
)
from .predicates import orient3d


def _add_interior_points(vertices: np.ndarray, faces: np.ndarray,
                          quality: float = 2.0,
                          verbose: bool = False) -> np.ndarray:
    """Add interior Steiner points to improve tet mesh quality.

    Creates a regular grid inside the bounding box, filters to keep only
    points inside the surface (via winding numbers), and appends them
    to the vertex array. This produces shorter, more uniform tets similar
    to TetGen's Steiner point insertion.
    """
    from .winding_number import compute_winding_numbers

    # Compute target spacing from average surface edge length
    edges = set()
    for f in faces:
        for i in range(3):
            edges.add(tuple(sorted((int(f[i]), int(f[(i + 1) % 3])))))
    avg_edge = np.mean([np.linalg.norm(vertices[e[1]] - vertices[e[0]]) for e in edges])
    spacing = avg_edge * quality

    # Create grid inside bounding box
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    margin = spacing * 0.5
    lo = bbox_min + margin
    hi = bbox_max - margin
    # Ensure at least 1 grid point per axis
    dims = np.maximum((hi - lo) / spacing, 1).astype(int) + 1
    xs = np.linspace(lo[0], hi[0], dims[0])
    ys = np.linspace(lo[1], hi[1], dims[1])
    zs = np.linspace(lo[2], hi[2], dims[2])
    grid = np.array(np.meshgrid(xs, ys, zs, indexing='ij')).reshape(3, -1).T

    if len(grid) == 0:
        return vertices

    # Filter to inside only using winding numbers
    wn = compute_winding_numbers(grid, vertices, faces, beta=2.0)
    interior_points = grid[wn > 0.5]

    if verbose:
        print(f"  Added {len(interior_points)} interior grid points (spacing={spacing:.4f}, avg_edge={avg_edge:.4f})")

    if len(interior_points) == 0:
        return vertices

    return np.vstack([vertices, interior_points])


class TetGenPython:
    """Drop-in replacement for the external tetgen package.

    Usage matches the tetgen Python API:
        tgen = TetGenPython(points, faces)
        tgen.tetrahedralize()
        vertices = tgen.node  # (N, 3) float64
        tets = tgen.elem      # (M, 4) int32
    """

    def __init__(self, points: np.ndarray, faces: np.ndarray):
        self._points = np.asarray(points, dtype=np.float64).copy()
        self._faces = np.asarray(faces, dtype=np.int32).copy()
        self._node: np.ndarray | None = None
        self._elem: np.ndarray | None = None

    @property
    def node(self) -> np.ndarray:
        """Tet mesh vertices, (N, 3) float64."""
        return self._node

    @property
    def elem(self) -> np.ndarray:
        """Tet mesh elements, (M, 4) int32."""
        return self._elem

    def tetrahedralize(self, quality: float = 2.0, verbose: bool = False):
        """Run the tetrahedralization pipeline."""
        vertices, faces = self._points, self._faces

        if verbose:
            print(f"[TetGenPython] Input: {len(vertices)} vertices, {len(faces)} faces")

        # Step 1: Deduplicate
        vertices, faces = dedup_vertices(vertices, faces)
        if verbose:
            print(f"  After dedup: {len(vertices)} vertices, {len(faces)} faces")

        # Step 2: Normalize to unit cube
        vertices, offset, scale = normalize_to_unit_cube(vertices)

        # Step 3: Add noise to break degeneracies
        add_noise(vertices, magnitude=1e-6, seed=42)

        # Step 4: Try convex shortcut
        result = _try_convex_shortcut(vertices, faces, verbose=verbose)
        if result is not None:
            tet_verts, tet_elems = result
        else:
            # Step 5: Full Delaunay tetrahedralization
            tet_verts, tet_elems = _delaunay_pipeline(vertices, faces, quality=quality, verbose=verbose)

        if len(tet_elems) == 0:
            if verbose:
                print("  Warning: no tetrahedra generated!")
            self._node = np.zeros((0, 3), dtype=np.float64)
            self._elem = np.zeros((0, 4), dtype=np.int32)
            return

        # Scale back to original coordinates
        tet_verts = tet_verts * scale + offset

        # Ensure positive volume
        _ensure_positive_volume(tet_verts, tet_elems)

        if verbose:
            print(f"  Output: {len(tet_verts)} vertices, {len(tet_elems)} tets")

        self._node = tet_verts.astype(np.float64)
        self._elem = tet_elems.astype(np.int32)


def _try_convex_shortcut(vertices: np.ndarray, faces: np.ndarray,
                          verbose: bool = False) -> tuple[np.ndarray, np.ndarray] | None:
    """Try convex shortcut: fan-connect surface triangles to centroid.

    Only succeeds if the mesh is truly convex, verified by checking that
    all surface normals point away from the centroid.
    """
    centroid = vertices.mean(axis=0)
    n = len(vertices)

    # Check convexity: every face normal should point away from centroid
    for f in faces:
        va = vertices[f[0]]
        vb = vertices[f[1]]
        vc = vertices[f[2]]
        normal = np.cross(vb - va, vc - va)
        norm = np.linalg.norm(normal)
        if norm < 1e-30:
            continue
        normal /= norm
        dist = np.dot(centroid - va, normal)
        if dist > 1e-10:
            return None

    all_verts = np.vstack([vertices, centroid.reshape(1, 3)])
    centroid_idx = n

    tets = []
    for f in faces:
        tets.append([int(f[0]), int(f[1]), int(f[2]), centroid_idx])

    tet_array = np.array(tets, dtype=np.int32)

    # Check if all tets have non-zero volume
    for t in tet_array:
        vol = orient3d(
            all_verts[t[0]], all_verts[t[1]], all_verts[t[2]], all_verts[t[3]]
        )
        if abs(vol) < 1e-15:
            return None

    if verbose:
        print("  Convex shortcut succeeded!")
    return all_verts, tet_array


def _delaunay_pipeline(vertices: np.ndarray, faces: np.ndarray,
                        quality: float = 2.0,
                        verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Full Delaunay tetrahedralization pipeline using scipy."""
    # Add interior Steiner points for better mesh quality
    if verbose:
        print("  Adding interior points...")
    tet_verts = _add_interior_points(vertices, faces, quality=quality, verbose=verbose)

    try:
        from scipy.spatial import Delaunay
    except ImportError:
        raise ImportError(
            "scipy is required for the 'python' tetrahedralization backend. "
            "Install it with: pip install scipy"
        )

    if verbose:
        print("  Computing Delaunay with scipy...")

    tri = Delaunay(tet_verts)
    tet_elems = tri.simplices.astype(np.int32)

    if verbose:
        print(f"  scipy Delaunay: {len(tet_elems)} tets")

    if len(tet_elems) == 0:
        return vertices, np.zeros((0, 4), dtype=np.int32)

    # Remove exterior tets using winding numbers
    if verbose:
        print("  Removing exterior tets via winding numbers...")

    from .winding_number import compute_winding_numbers

    # Compute centroids of all tets
    centroids = np.zeros((len(tet_elems), 3), dtype=np.float64)
    for i in range(len(tet_elems)):
        t = tet_elems[i]
        centroids[i] = (tet_verts[t[0]] + tet_verts[t[1]] +
                        tet_verts[t[2]] + tet_verts[t[3]]) * 0.25

    # Compute winding numbers for each tet centroid
    winding_numbers = compute_winding_numbers(centroids, tet_verts, faces, beta=2.0)

    # Global sign correction (PhysX approach)
    winding_sum = np.sum(winding_numbers)
    sign = -1.0 if winding_sum < 0.0 else 1.0

    # Keep tets with winding number > 0.5 (inside the surface)
    interior_mask = (sign * winding_numbers) > 0.5
    tet_elems = tet_elems[interior_mask]

    if verbose:
        removed = len(interior_mask) - len(tet_elems)
        print(f"  Removed {removed} exterior tets, kept {len(tet_elems)}")

    if len(tet_elems) == 0:
        return vertices, np.zeros((0, 4), dtype=np.int32)

    # Remove degenerate tets (zero volume — coplanar vertices)
    keep = np.ones(len(tet_elems), dtype=bool)
    for i in range(len(tet_elems)):
        t = tet_elems[i]
        vol = orient3d(tet_verts[t[0]], tet_verts[t[1]], tet_verts[t[2]], tet_verts[t[3]])
        if abs(vol) < 1e-12:
            keep[i] = False
    removed_deg = len(tet_elems) - int(keep.sum())
    if removed_deg > 0:
        if verbose:
            print(f"  Removed {removed_deg} degenerate tets (zero volume)")
        tet_elems = tet_elems[keep]

    # Remove disconnected islands
    tet_verts, tet_elems = _remove_disconnected_islands(tet_verts, tet_elems, verbose=verbose)

    # Compact vertices
    tet_verts, tet_elems = _compact_mesh(tet_verts, tet_elems)

    return tet_verts, tet_elems


def _ensure_positive_volume(vertices: np.ndarray, tets: np.ndarray) -> None:
    """Ensure tets have consistent orientation matching Newton builder convention.

    Newton's builder computes volume as det([q-p, r-p, s-p]) / 6.0 where
    p,q,r,s are the four vertices. This is positive when orient3d(p,q,r,s) < 0.
    We flip tets where orient3d > 0 so the builder sees positive volume.
    """
    for i in range(len(tets)):
        t = tets[i]
        vol = orient3d(vertices[t[0]], vertices[t[1]], vertices[t[2]], vertices[t[3]])
        if vol > 0:
            tets[i][0], tets[i][1] = tets[i][1], tets[i][0]


def _remove_disconnected_islands(vertices: np.ndarray, tets: np.ndarray,
                                   verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the largest connected component of tetrahedra."""
    if len(tets) == 0:
        return vertices, tets

    # Build face -> tet mapping
    face_to_tet: dict[tuple[int, int, int], list[int]] = {}
    for i in range(len(tets)):
        t = tets[i]
        for fi in range(4):
            f = NEIGHBOR_FACES[fi]
            face = sorted_face(int(t[f[0]]), int(t[f[1]]), int(t[f[2]]))
            if face not in face_to_tet:
                face_to_tet[face] = []
            face_to_tet[face].append(i)

    # Build adjacency list
    adj: list[set[int]] = [set() for _ in range(len(tets))]
    for face, tet_list in face_to_tet.items():
        if len(tet_list) == 2:
            adj[tet_list[0]].add(tet_list[1])
            adj[tet_list[1]].add(tet_list[0])

    # BFS from first tet
    visited = set()
    queue = [0]
    visited.add(0)
    component = [0]

    while queue:
        tet_id = queue.pop(0)
        for nbr in adj[tet_id]:
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)
                component.append(nbr)

    if len(component) == len(tets):
        return vertices, tets

    if verbose:
        print(f"  Removing {len(tets) - len(component)} disconnected tets")

    return vertices, tets[component]


def _compact_mesh(vertices: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove unused vertices and remap indices."""
    if len(tets) == 0:
        return vertices, tets

    used = np.unique(tets)
    if len(used) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 4), dtype=np.int32)

    max_v = int(used.max()) + 1
    remap = np.full(max_v, -1, dtype=np.int32)
    for new_idx, old_idx in enumerate(used):
        remap[old_idx] = new_idx

    tet_remapped = remap[tets]
    compact_verts = vertices[used]

    return compact_verts, tet_remapped


def tetrahedralize_surface_mesh_python(
    vertices: np.ndarray,
    faces: np.ndarray,
    quality: float = 2.0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedralize a closed surface mesh using scipy Delaunay.

    Args:
        vertices: (N, 3) float32/float64 surface vertex positions
        faces: (M, 3) int32 surface triangle indices
        quality: Quality parameter (currently unused)
        verbose: Print progress

    Returns:
        (tet_vertices, tet_indices) where:
        - tet_vertices is (K, 3) float32
        - tet_indices is (4*T,) int32 flattened tet vertex indices
    """
    verts_f64 = np.asarray(vertices, dtype=np.float64)
    faces_i32 = np.asarray(faces, dtype=np.int32)

    tgen = TetGenPython(verts_f64, faces_i32)
    tgen.tetrahedralize(quality=quality, verbose=verbose)

    if tgen.node is None or len(tgen.node) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)

    out_verts = tgen.node.astype(np.float32)
    out_tets = tgen.elem.flatten().astype(np.int32)

    return out_verts, out_tets
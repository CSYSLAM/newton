# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tetrahedral mesh generation from surface meshes using TetGen.

This module provides utilities to convert triangular surface meshes (OBJ, PLY, STL, etc.)
into volumetric tetrahedral meshes suitable for soft body simulation.
"""

from __future__ import annotations

import os

import numpy as np

from .types import TetMesh
from .utils import load_mesh


def remesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    resolution: int = 100,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Remesh a surface mesh using voxel grid + marching cubes.

    Takes a surface mesh with potentially bad topology and produces a clean
    remeshed surface via voxelization + marching cubes isosurface, then
    deduplication, pruning internal surfaces, projection onto input surface,
    and normal computation.

    Args:
        vertices: Vertex positions [m], shape (N, 3), float32.
        faces: Triangle face indices, shape (M, 3), int32.
        resolution: Controls voxel grid density. Higher values produce
            finer remeshed surfaces. Default 100.
        verbose: Whether to print progress information.

    Returns:
        Tuple of (remeshed_vertices, remeshed_faces) where:
        - remeshed_vertices: Vertex positions [m], shape (K, 3), float32
        - remeshed_faces: Triangle face indices, shape (L, 3), uint32

    Raises:
        ImportError: If the native module is not compiled/installed.
    """
    from ..softbody._voxelize_native import remesh_surface_native

    if verbose:
        print(f"Surface remeshing: resolution={resolution}")

    result = remesh_surface_native(
        vertices, faces, resolution=resolution,
    )

    remeshed_verts = result["vertices"]
    remeshed_faces = result["tri_ids"].reshape(-1, 3)

    if verbose:
        stats = result["debug_stats"]
        print(f"Remesh result: {stats['output_vertex_count']} vertices, "
              f"{stats['output_triangle_count']} triangles "
              f"(from {stats['input_vertex_count']} vertices, "
              f"{stats['input_triangle_count']} triangles)")

    return remeshed_verts, remeshed_faces


def compute_volume_embedding(
    tet_verts: np.ndarray,
    tet_indices: np.ndarray,
    render_verts: np.ndarray,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute volume skinning embedding for render vertices into a tet mesh.

    For each render vertex, finds the closest tetrahedron and computes
    4-component barycentric coordinates. At runtime, deformed positions
    are recovered via ``p = p0*w0 + p1*w1 + p2*w2 + p3*w3``.

    Args:
        tet_verts: Tetrahedron vertex positions [m], shape (N, 3), float32.
        tet_indices: Tetrahedron indices, shape (M, 4), int32.
        render_verts: Render vertex positions to embed [m], shape (K, 3), float32.
        verbose: Whether to print progress information.

    Returns:
        Tuple of (tet_idx, bary_weights) where:
        - tet_idx: Tetrahedron index per render vertex, shape (K,), int32
        - bary_weights: Barycentric coordinates, shape (K, 4), float32

    Raises:
        ImportError: If the native module is not compiled/installed.
    """
    from ..softbody._voxelize_native import compute_volume_embedding_native

    result = compute_volume_embedding_native(tet_verts, tet_indices, render_verts)

    if verbose:
        num_verts = len(result["tet_idx"])
        print(f"Volume embedding: {num_verts} render vertices embedded")

    return result["tet_idx"], result["bary_weights"]


def deform_with_embedding(
    deformed_tet_verts: np.ndarray,
    tet_indices: np.ndarray,
    skin_tet_idx: np.ndarray,
    skin_weights: np.ndarray,
) -> np.ndarray:
    """Apply volume skinning deformation to render vertices.

    Args:
        deformed_tet_verts: Deformed tet vertex positions [m], shape (N, 3), float32.
        tet_indices: Tetrahedron indices, shape (M, 4), int32.
        skin_tet_idx: Tetrahedron index per render vertex, shape (K,), int32.
        skin_weights: Barycentric coordinates, shape (K, 4), float32.

    Returns:
        Deformed render vertex positions [m], shape (K, 3), float32.

    Raises:
        ImportError: If the native module is not compiled/installed.
    """
    from ..softbody._voxelize_native import deform_with_embedding_native

    return deform_with_embedding_native(
        deformed_tet_verts, tet_indices, skin_tet_idx, skin_weights,
    )


def tetrahedralize_surface_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    quality: float = 1.5,
    max_volume: float | None = None,
    verbose: bool = False,
    backend: str = "auto",
    resolution: int = 32,
    num_relaxation_iters: int = 5,
    rel_min_tet_volume: float = 0.05,
    surface_dist_ratio: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedralize a triangular surface mesh.

    Converts a closed surface mesh into a volumetric tetrahedral mesh.

    Args:
        vertices: Vertex positions [m], shape (N, 3).
        faces: Triangle face indices, shape (M, 3).
        quality: Maximum radius-edge ratio for mesh quality. Smaller values
            produce higher quality meshes. Default 1.5. Typical values:
            1.1 (high quality) to 2.0 (lower quality). Must be > 1.0.
        max_volume: Maximum volume constraint for tetrahedra. If None, no
            volume constraint is applied.
        verbose: Whether to print tetrahedralization progress.
        backend: Tetrahedralization backend to use. ``"auto"`` tries the
            external TetGen package first, falling back to the Python
            implementation if TetGen is not installed. ``"external"`` uses
            the TetGen C++ package. ``"python"`` uses scipy's Delaunay
            tetrahedralization with ray-casting exterior removal.
            ``"voxel"`` uses the PhysX voxel-based tetrahedralizer with
            density controlled by ``resolution``. Default ``"auto"``.
        resolution: Voxel grid resolution for the ``"voxel"`` backend.
            Higher values produce denser tet meshes. Default 32.
        num_relaxation_iters: Number of relaxation iterations for the
            ``"voxel"`` backend. Default 5.
        rel_min_tet_volume: Minimum relative tet volume for volume
            conservation in the ``"voxel"`` backend. Default 0.05.
        surface_dist_ratio: Ratio controlling how far surface vertices
            are pulled toward the input surface during relaxation in the
            ``"voxel"`` backend. The actual distance is
            ``surface_dist_ratio * gridSpacing``. Default 0.2.

    Returns:
        Tuple of (tet_vertices, tet_indices) where:
        - tet_vertices: Vertex positions [m], shape (K, 3), float32
        - tet_indices: Flattened tetrahedron indices (4 per tet), int32

    Raises:
        ImportError: If backend is "external" and tetgen is not installed,
            or backend is "python" and scipy is not installed.
        ValueError: If tetrahedralization fails.
    """
    if backend == "auto":
        try:
            import tetgen as _tetgen  # noqa: F401
            backend = "external"
        except ImportError:
            backend = "python"

    if backend == "external":
        import tetgen

        # Build kwargs for tetrahedralize
        kwargs = {
            "quality": True,
            "minratio": quality,
            "quiet": not verbose,
        }

        if max_volume is not None:
            kwargs["maxvolume"] = max_volume

        if verbose:
            print(f"TetGen parameters: minratio={quality}, maxvolume={max_volume}")

        # Run TetGen
        t = tetgen.TetGen(vertices, faces)
        t.tetrahedralize(**kwargs)

        # Extract results
        tet_vertices = np.array(t.node, dtype=np.float32)
        tet_indices = np.array(t.elem, dtype=np.int32).flatten()

        if len(tet_vertices) == 0 or len(tet_indices) == 0:
            raise ValueError("Tetrahedralization failed: no vertices or indices generated")

        return tet_vertices, tet_indices

    elif backend == "python":
        from .tetgen_python.api import tetrahedralize_surface_mesh_python

        return tetrahedralize_surface_mesh_python(
            vertices, faces, quality=quality, verbose=verbose
        )

    elif backend == "voxel":
        from ..softbody._voxelize_native import voxelize_soft_body_native

        if verbose:
            print(f"Voxel tetrahedralization: resolution={resolution}, "
                  f"num_relaxation_iters={num_relaxation_iters}, "
                  f"rel_min_tet_volume={rel_min_tet_volume}")

        result = voxelize_soft_body_native(
            vertices, faces,
            resolution=resolution,
            num_relaxation_iters=num_relaxation_iters,
            rel_min_tet_volume=rel_min_tet_volume,
            surface_dist_ratio=surface_dist_ratio,
        )
        return result["tet_vertices"], result["tet_indices"]

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'external', 'python', or 'voxel'.")


def tetrahedralize_obj(
    filename: str,
    quality: float = 1.5,
    max_volume: float | None = None,
    verbose: bool = False,
    method: str | None = None,
    backend: str = "auto",
    resolution: int = 32,
    num_relaxation_iters: int = 5,
    rel_min_tet_volume: float = 0.05,
    surface_dist_ratio: float = 0.2,
) -> TetMesh:
    """Load an OBJ file and convert it to a tetrahedral mesh.

    This is a convenience function that combines loading a surface mesh
    from an OBJ file with tetrahedralization using TetGen.

    Args:
        filename: Path to the OBJ file.
        quality: Maximum radius-edge ratio for mesh quality. Smaller values
            produce higher quality meshes. Default 1.5.
        max_volume: Maximum volume constraint for tetrahedra. If None, no
            volume constraint is applied.
        verbose: Whether to print loading and tetrahedralization progress.
        method: Method to use for loading the mesh ("trimesh", "meshio", "pcu", "openmesh").
            If None, tries all methods.
        backend: Tetrahedralization backend. See :func:`tetrahedralize_surface_mesh`.

    Returns:
        A :class:`~newton.TetMesh` object ready for soft body simulation.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If tetgen or mesh loading library is not installed.
        ValueError: If loading or tetrahedralization fails.

    Example:
        Create a TetMesh from an OBJ file for soft body simulation:

        .. code-block:: python

            import newton

            tet_mesh = newton.utils.tetrahedralize_obj("model.obj", quality=1.2)
            builder.add_soft_mesh(
                pos=(0.0, 0.0, 1.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=(0.0, 0.0, 0.0),
                mesh=tet_mesh,
                density=100.0,
                k_mu=1.0e5,
                k_lambda=1.0e5,
            )
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    # Load surface mesh
    if verbose:
        print(f"Loading surface mesh from {filename}...")
    vertices, faces = load_mesh(filename, method=method)
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32).reshape(-1, 3)

    if verbose:
        print(f"Surface mesh: {len(vertices)} vertices, {len(faces)} faces")

    # Tetrahedralize
    if verbose:
        print("Tetrahedralizing with TetGen...")
    tet_vertices, tet_indices = tetrahedralize_surface_mesh(
        vertices, faces, quality=quality, max_volume=max_volume, verbose=verbose, backend=backend,
        resolution=resolution, num_relaxation_iters=num_relaxation_iters, rel_min_tet_volume=rel_min_tet_volume,
        surface_dist_ratio=surface_dist_ratio,
    )

    if verbose:
        print(f"Tetrahedral mesh: {len(tet_vertices)} vertices, {len(tet_indices) // 4} tetrahedra")

    return TetMesh(tet_vertices, tet_indices)


def tetrahedralize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    quality: float = 1.5,
    max_volume: float | None = None,
    verbose: bool = False,
    backend: str = "auto",
    resolution: int = 32,
    num_relaxation_iters: int = 5,
    rel_min_tet_volume: float = 0.05,
    surface_dist_ratio: float = 0.2,
) -> TetMesh:
    """Convert a triangular surface mesh to a tetrahedral mesh.

    Args:
        vertices: Vertex positions [m], shape (N, 3).
        faces: Triangle face indices, shape (M, 3) or flattened (M*3,).
        quality: Maximum radius-edge ratio for mesh quality. Smaller values
            produce higher quality meshes. Default 1.5.
        max_volume: Maximum volume constraint for tetrahedra. If None, no
            volume constraint is applied.
        verbose: Whether to print tetrahedralization progress.

    Returns:
        A :class:`~newton.TetMesh` object ready for soft body simulation.

    Example:
        Create a TetMesh from raw vertex/face arrays:

        .. code-block:: python

            import numpy as np
            import newton

            # Simple cube surface mesh
            vertices = np.array([
                [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
            ], dtype=np.float32)
            faces = np.array([
                [0, 1, 2], [0, 2, 3],  # bottom
                [4, 5, 6], [4, 6, 7],  # top
                [0, 1, 5], [0, 5, 4],  # front
                [2, 3, 7], [2, 7, 6],  # back
                [0, 3, 7], [0, 7, 4],  # left
                [1, 2, 6], [1, 6, 5],  # right
            ], dtype=np.int32)

            tet_mesh = newton.utils.tetrahedralize_mesh(vertices, faces, quality=1.2)
    """
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32).reshape(-1, 3)

    tet_vertices, tet_indices = tetrahedralize_surface_mesh(
        vertices, faces, quality=quality, max_volume=max_volume, verbose=verbose, backend=backend,
        resolution=resolution, num_relaxation_iters=num_relaxation_iters, rel_min_tet_volume=rel_min_tet_volume,
        surface_dist_ratio=surface_dist_ratio,
    )

    return TetMesh(tet_vertices, tet_indices)

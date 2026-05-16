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


def tetrahedralize_surface_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    quality: float = 1.5,
    max_volume: float | None = None,
    verbose: bool = False,
    backend: str = "auto",
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
            tetrahedralization with ray-casting exterior removal. Default
            ``"auto"``.

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

    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'auto', 'external', or 'python'.")


def tetrahedralize_obj(
    filename: str,
    quality: float = 1.5,
    max_volume: float | None = None,
    verbose: bool = False,
    method: str | None = None,
    backend: str = "auto",
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
        vertices, faces, quality=quality, max_volume=max_volume, verbose=verbose, backend=backend
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
        vertices, faces, quality=quality, max_volume=max_volume, verbose=verbose, backend=backend
    )

    return TetMesh(tet_vertices, tet_indices)

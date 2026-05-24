# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Python wrapper for the native voxel tetrahedralization module."""

from __future__ import annotations

import numpy as np

_NATIVE_AVAILABLE: bool | None = None


def _native_module_available() -> bool:
    """Check if the compiled native module can be imported."""
    global _NATIVE_AVAILABLE
    if _NATIVE_AVAILABLE is None:
        try:
            from . import _voxel_tet  # noqa: F401
            _NATIVE_AVAILABLE = True
        except ImportError:
            _NATIVE_AVAILABLE = False
    return _NATIVE_AVAILABLE


def _validate_voxelize_inputs(
    vertices: np.ndarray,
    triangles: np.ndarray,
    resolution: int,
    num_relaxation_iters: int,
    rel_min_tet_volume: float,
    surface_dist_ratio: float,
) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"triangles must have shape (M, 3), got {triangles.shape}")
    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}")
    if num_relaxation_iters < 0:
        raise ValueError(f"num_relaxation_iters must be >= 0, got {num_relaxation_iters}")
    if rel_min_tet_volume < 0.0:
        raise ValueError(f"rel_min_tet_volume must be >= 0.0, got {rel_min_tet_volume}")
    if surface_dist_ratio < 0.0:
        raise ValueError(f"surface_dist_ratio must be >= 0.0, got {surface_dist_ratio}")


def voxelize_soft_body_native(
    vertices: np.ndarray,
    triangles: np.ndarray,
    resolution: int,
    num_relaxation_iters: int = 5,
    rel_min_tet_volume: float = 0.05,
    surface_dist_ratio: float = 0.2,
) -> dict:
    """Tetrahedralize a surface mesh using voxel-based method (C++ backend).

    Args:
        vertices: Vertex positions, shape (N, 3), float32.
        triangles: Triangle face indices, shape (M, 3), int32.
        resolution: Controls voxel grid density. Higher = more tets.
        num_relaxation_iters: Number of relaxation iterations.
        rel_min_tet_volume: Minimum relative tet volume for volume conservation.
        surface_dist_ratio: Surface offset as ratio of grid spacing.

    Returns:
        Dict with "tet_vertices" (float32, shape K,3),
        "tet_indices" (int32, flattened, 4 per tet), and
        "debug_stats" (dict of mesh diagnostics).

    Raises:
        ImportError: If the native module is not compiled/installed.
    """
    if not _native_module_available():
        raise ImportError(
            "Native voxel tetrahedralization module (_voxel_tet) is not available. "
            "Build it with: python scripts/build_voxel_tet.py --install"
        )

    _validate_voxelize_inputs(
        vertices, triangles, resolution, num_relaxation_iters, rel_min_tet_volume, surface_dist_ratio
    )

    verts = np.ascontiguousarray(vertices, dtype=np.float32)
    tris = np.ascontiguousarray(triangles.reshape(-1, 3), dtype=np.int32)

    from . import _voxel_tet

    return _voxel_tet.voxelize_soft_body(
        verts,
        tris,
        resolution,
        num_relaxation_iters,
        rel_min_tet_volume,
        surface_dist_ratio,
    )


def remesh_surface_native(
    vertices: np.ndarray,
    triangles: np.ndarray,
    resolution: int = 100,
    return_vertex_map: bool = False,
) -> dict:
    """Remesh a surface mesh using voxel grid + marching cubes (C++ backend).

    Args:
        vertices: Vertex positions, shape (N, 3), float32.
        triangles: Triangle face indices, shape (M, 3), int32.
        resolution: Controls voxel grid density. Higher = finer remesh.
        return_vertex_map: If True, compute and return a mapping from output
            vertex indices to input vertex indices.

    Returns:
        Dict with "vertices" (float32, shape K,3),
        "tri_ids" (uint32, shape L,3), and
        "debug_stats" (dict of remesh diagnostics).
        If return_vertex_map is True, also includes "vertex_map"
        (uint32, shape K) mapping output vertices to input vertices.

    Raises:
        ImportError: If the native module is not compiled/installed.
    """
    if not _native_module_available():
        raise ImportError(
            "Native voxel tetrahedralization module (_voxel_tet) is not available. "
            "Build it with: python scripts/build_voxel_tet.py --install"
        )

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"triangles must have shape (M, 3), got {triangles.shape}")
    if resolution <= 0:
        raise ValueError(f"resolution must be > 0, got {resolution}")

    verts = np.ascontiguousarray(vertices, dtype=np.float32)
    tris = np.ascontiguousarray(triangles.reshape(-1, 3), dtype=np.int32)

    from . import _voxel_tet

    result = _voxel_tet.remesh_surface(verts, tris, resolution, return_vertex_map)

    if return_vertex_map:
        out_verts, out_tris, stats, vertex_map = result
        return {
            "vertices": out_verts,
            "tri_ids": out_tris,
            "debug_stats": stats,
            "vertex_map": vertex_map,
        }
    else:
        out_verts, out_tris, stats = result
        return {
            "vertices": out_verts,
            "tri_ids": out_tris,
            "debug_stats": stats,
        }


def compute_volume_embedding_native(
    tet_verts: np.ndarray,
    tet_indices: np.ndarray,
    render_verts: np.ndarray,
    impl: str = "physx_exact_cpu",
) -> dict:
    """Compute volume skinning embedding for render vertices into a tet mesh (C++ backend).

    Args:
        tet_verts: Tetrahedron vertex positions, shape (N, 3), float32.
        tet_indices: Tetrahedron indices, shape (M, 4), int32.
        render_verts: Render vertex positions to embed, shape (K, 3), float32.
        impl: Backend implementation. One of "physx_exact_cpu", "legacy_custom".

    Returns:
        Dict with "tet_idx" (int32, shape K) and "bary_weights" (float32, shape K,4).

    Raises:
        ImportError: If the native module is not compiled/installed.
    """
    if not _native_module_available():
        raise ImportError(
            "Native voxel tetrahedralization module (_voxel_tet) is not available. "
            "Build it with: python scripts/build_voxel_tet.py --install"
        )

    if tet_verts.ndim != 2 or tet_verts.shape[1] != 3:
        raise ValueError(f"tet_verts must have shape (N, 3), got {tet_verts.shape}")
    if tet_indices.ndim != 2 or tet_indices.shape[1] != 4:
        raise ValueError(f"tet_indices must have shape (M, 4), got {tet_indices.shape}")
    if render_verts.ndim != 2 or render_verts.shape[1] != 3:
        raise ValueError(f"render_verts must have shape (K, 3), got {render_verts.shape}")

    tv = np.ascontiguousarray(tet_verts, dtype=np.float32)
    ti = np.ascontiguousarray(tet_indices.reshape(-1, 4), dtype=np.int32)
    rv = np.ascontiguousarray(render_verts, dtype=np.float32)

    from . import _voxel_tet

    tet_idx, bary = _voxel_tet.compute_volume_embedding(tv, ti, rv, impl)

    return {
        "tet_idx": tet_idx,
        "bary_weights": bary,
    }


def deform_with_embedding_native(
    deformed_tet_verts: np.ndarray,
    tet_indices: np.ndarray,
    skin_tet_idx: np.ndarray,
    skin_weights: np.ndarray,
) -> np.ndarray:
    """Apply volume skinning deformation (C++ backend).

    Args:
        deformed_tet_verts: Deformed tet vertex positions, shape (N, 3), float32.
        tet_indices: Tetrahedron indices, shape (M, 4), int32.
        skin_tet_idx: Tet index per render vertex, shape (K,), int32.
        skin_weights: Barycentric weights, shape (K, 4), float32.

    Returns:
        Deformed render vertex positions, shape (K, 3), float32.

    Raises:
        ImportError: If the native module is not compiled/installed.
    """
    if not _native_module_available():
        raise ImportError(
            "Native voxel tetrahedralization module (_voxel_tet) is not available. "
            "Build it with: python scripts/build_voxel_tet.py --install"
        )

    dtv = np.ascontiguousarray(deformed_tet_verts, dtype=np.float32)
    ti = np.ascontiguousarray(tet_indices.reshape(-1, 4), dtype=np.int32)
    sti = np.ascontiguousarray(skin_tet_idx, dtype=np.int32)
    sw = np.ascontiguousarray(skin_weights.reshape(-1, 4), dtype=np.float32)

    from . import _voxel_tet

    return _voxel_tet.deform_with_embedding(dtv, ti, sti, sw)


def summarize_voxelize_result(result: dict) -> dict:
    """Derive stable summary statistics from the native output.

    Args:
        result: Dict returned by :func:`voxelize_soft_body_native`.

    Returns:
        Dict with vertex_count, tet_count, bbox_min, bbox_max, bbox_extent,
        total_signed_volume, min_signed_tet_volume, max_signed_tet_volume.
    """
    tv = result["tet_vertices"]
    ti = result["tet_indices"].reshape(-1, 4)

    vols = np.empty(len(ti))
    for i, (v0, v1, v2, v3) in enumerate(ti):
        a, b, c, d = tv[v0], tv[v1], tv[v2], tv[v3]
        vols[i] = np.dot(a - d, np.cross(b - d, c - d)) / 6.0

    return {
        "vertex_count": len(tv),
        "tet_count": len(ti),
        "bbox_min": tv.min(axis=0).tolist(),
        "bbox_max": tv.max(axis=0).tolist(),
        "bbox_extent": (tv.max(axis=0) - tv.min(axis=0)).tolist(),
        "total_signed_volume": float(vols.sum()),
        "min_signed_tet_volume": float(vols.min()),
        "max_signed_tet_volume": float(vols.max()),
    }

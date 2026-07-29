#!/usr/bin/env python3
"""Generate the Gear Crusher assets used by example_gear_crusher.py.

The Armadillo source mesh is the Stanford Armadillo distributed by the libigl
``libigl-tutorial-data`` repository.  The script simplifies the watertight
surface, tetrahedralizes it with TetGen, rotates it into Newton's Z-up frame,
and builds an 18-tooth crusher drum reconstructed from Fig. 8 of the DAT paper.

Required only when regenerating assets:
    pip install numpy trimesh fast-simplification tetgen matplotlib
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import tetgen
import trimesh

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
SOURCE_OBJ = ASSET_DIR / "source" / "armadillo_stanford_original.obj"


def _positive_tets(vertices: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Return tets with positive signed volume in (a,b,c,d) ordering."""
    out = np.asarray(tets, dtype=np.int32).copy()
    p = vertices[out]
    det = np.einsum(
        "ij,ij->i",
        np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
        p[:, 3] - p[:, 0],
    )
    flip = det < 0.0
    tmp = out[flip, 0].copy()
    out[flip, 0] = out[flip, 1]
    out[flip, 1] = tmp
    return out


def _boundary_triangles(tets: np.ndarray) -> np.ndarray:
    faces = np.concatenate(
        [
            tets[:, [0, 2, 1]],
            tets[:, [0, 1, 3]],
            tets[:, [1, 2, 3]],
            tets[:, [2, 0, 3]],
        ],
        axis=0,
    )
    sorted_faces = np.sort(faces, axis=1)
    _, first, counts = np.unique(sorted_faces, axis=0, return_index=True, return_counts=True)
    return faces[first[counts == 1]].astype(np.int32)


def build_armadillo() -> dict[str, object]:
    source = trimesh.load(SOURCE_OBJ, force="mesh", process=True)
    if not isinstance(source, trimesh.Trimesh) or not source.is_watertight:
        raise RuntimeError("The Stanford Armadillo source must be one watertight triangle mesh")

    # This target gives approximately the same resolution as the paper:
    # about 15k volume vertices and 60k tetrahedra after quality tetrahedralization.
    simplified = source.simplify_quadric_decimation(face_count=6650)
    components = simplified.split(only_watertight=False)
    surface = max(components, key=lambda m: len(m.faces))
    if not surface.is_watertight:
        raise RuntimeError("Simplified Armadillo is not watertight")

    tet = tetgen.TetGen(np.asarray(surface.vertices), np.asarray(surface.faces))
    vertices, tets, _, _ = tet.tetrahedralize(
        quality=True,
        minratio=2.0,
        mindihedral=0.0,
        quiet=True,
    )
    vertices = np.asarray(vertices, dtype=np.float64)
    tets = _positive_tets(vertices, np.asarray(tets, dtype=np.int32))

    # Stanford coordinates: X = left/right, Y = up, Z = front/back.
    # Newton example coordinates: X = left/right, Y = depth, Z = up.
    vertices = vertices[:, [0, 2, 1]]
    # The axis permutation changes handedness; restore positive tet orientation.
    tets = _positive_tets(vertices, tets)

    # Normalize the height to 1.25 m and put the feet at z=0.  Keep the model
    # centered in X/Y so it can enter the crusher gap without guide cubes.
    vertices -= 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    height = float(vertices[:, 2].max() - vertices[:, 2].min())
    vertices *= 1.25 / height
    vertices[:, 2] -= vertices[:, 2].min()

    boundary = _boundary_triangles(tets)
    tet_path = ASSET_DIR / "armadillo_stanford_tet.npz"
    np.savez_compressed(
        tet_path,
        vertices=vertices.astype(np.float32),
        tets=tets.astype(np.int32),
        surface_triangles=boundary.astype(np.int32),
    )
    surface_mesh = trimesh.Trimesh(vertices=vertices, faces=boundary, process=False)
    surface_mesh.export(ASSET_DIR / "armadillo_stanford_surface.obj")

    signed = np.einsum(
        "ij,ij->i",
        np.cross(
            vertices[tets[:, 1]] - vertices[tets[:, 0]],
            vertices[tets[:, 2]] - vertices[tets[:, 0]],
        ),
        vertices[tets[:, 3]] - vertices[tets[:, 0]],
    ) / 6.0
    if np.min(signed) <= 0.0:
        raise RuntimeError("Tetrahedral mesh contains inverted elements")

    return {
        "source_vertices": int(len(source.vertices)),
        "source_triangles": int(len(source.faces)),
        "tet_vertices": int(len(vertices)),
        "tetrahedra": int(len(tets)),
        "surface_triangles": int(len(boundary)),
        "minimum_tet_volume": float(np.min(signed)),
        "bounds": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
    }


def gear_profile(
    tooth_count: int,
    root_radius: float,
    tip_radius: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Create a straight-tooth crusher cross-section.

    Each tooth has a broad flat crest, steep trapezoidal flanks, and a narrow
    root valley.  This matches the long axial ridges and deep grooves visible
    in Fig. 8 more closely than a generic low-poly star or box proxy.
    """
    # Fractions of one pitch and corresponding radii.  Repeating the first
    # root sample avoids an unrealistically sharp V-shaped root.
    fractions = np.array([0.00, 0.10, 0.22, 0.32, 0.68, 0.78, 0.90], dtype=np.float64)
    radii = np.array(
        [root_radius, root_radius, 0.5 * (root_radius + tip_radius), tip_radius,
         tip_radius, 0.5 * (root_radius + tip_radius), root_radius],
        dtype=np.float64,
    )
    points = []
    pitch = 2.0 * math.pi / tooth_count
    for tooth in range(tooth_count):
        angles = phase + (tooth + fractions) * pitch
        points.extend(np.column_stack((radii * np.cos(angles), radii * np.sin(angles))))
    return np.asarray(points, dtype=np.float64)


def build_crusher_gear() -> dict[str, object]:
    tooth_count = 18
    root_radius = 0.50
    tip_radius = 0.62
    half_width = 0.38
    bevel = 0.028
    profile = gear_profile(tooth_count, root_radius, tip_radius, phase=math.pi / tooth_count)
    n = len(profile)

    # The end rings are slightly inset to create the rounded/chamfered end cap
    # visible in the paper instead of a razor-sharp extrusion.
    y_rings = np.array([-half_width, -half_width + bevel, half_width - bevel, half_width])
    radial_scales = np.array([0.965, 1.0, 1.0, 0.965])
    vertices = []
    for y, scale in zip(y_rings, radial_scales, strict=True):
        vertices.extend(np.column_stack((profile[:, 0] * scale, np.full(n, y), profile[:, 1] * scale)))
    vertices = np.asarray(vertices, dtype=np.float64)

    faces: list[tuple[int, int, int]] = []
    for ring in range(len(y_rings) - 1):
        a0, b0 = ring * n, (ring + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces.append((a0 + i, b0 + i, b0 + j))
            faces.append((a0 + i, b0 + j, a0 + j))

    # Solid front/back caps.  The crusher in Fig. 8 has no center bore.
    front_center = len(vertices)
    back_center = front_center + 1
    vertices = np.vstack((vertices, [0.0, -half_width, 0.0], [0.0, half_width, 0.0]))
    front_ring = 0
    back_ring = (len(y_rings) - 1) * n
    for i in range(n):
        j = (i + 1) % n
        faces.append((front_center, front_ring + j, front_ring + i))
        faces.append((back_center, back_ring + i, back_ring + j))

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(faces, dtype=np.int32), process=False)
    mesh.fix_normals(multibody=True)
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise RuntimeError("Generated crusher gear is not a closed, consistently wound mesh")

    mesh.export(ASSET_DIR / "crusher_drum_18t.obj")
    np.savez_compressed(
        ASSET_DIR / "crusher_drum_18t.npz",
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        triangles=np.asarray(mesh.faces, dtype=np.int32),
    )
    return {
        "tooth_count": tooth_count,
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "root_radius": root_radius,
        "tip_radius": tip_radius,
        "half_width": half_width,
        "watertight": bool(mesh.is_watertight),
    }


def build_previews() -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    gear_data = np.load(ASSET_DIR / "crusher_drum_18t.npz")
    gv = gear_data["vertices"]
    gf = gear_data["triangles"]
    arm_data = np.load(ASSET_DIR / "armadillo_stanford_tet.npz")
    av = arm_data["vertices"]
    af = arm_data["surface_triangles"]

    def add_mesh(ax, vertices, faces, offset, color, alpha=1.0):
        verts = vertices[faces] + np.asarray(offset)
        poly = Poly3DCollection(verts, linewidths=0.08, alpha=alpha)
        poly.set_facecolor(color)
        poly.set_edgecolor((0.12, 0.12, 0.12, 0.22))
        ax.add_collection3d(poly)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    add_mesh(ax, gv, gf, (-0.59, 0.0, 0.72), (0.62, 0.64, 0.68, 1.0))
    add_mesh(ax, gv, gf, (0.59, 0.0, 0.72), (0.62, 0.64, 0.68, 1.0))
    arm = av.copy()
    arm[:, 2] += 1.20
    add_mesh(ax, arm, af[::2], (0.0, 0.0, 0.0), (0.05, 0.78, 0.30, 1.0))
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(0.0, 2.55)
    ax.set_box_aspect((2.7, 2.0, 2.55))
    ax.view_init(elev=8, azim=-90)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(ROOT / "previews" / "gear_crusher_scene.png", dpi=190, transparent=False)
    plt.close(fig)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    arm = build_armadillo()
    gear = build_crusher_gear()
    build_previews()
    metadata = {
        "armadillo": arm,
        "crusher_gear": gear,
        "paper_target": {
            "soft_vertices": 15000,
            "tetrahedra": 60000,
            "lambda": 1.0e6,
            "mu": 1.0e5,
            "contact_stiffness": 1.0e6,
            "friction": 0.2,
            "contact_radius_m": 0.005,
            "dt_s": 1.0 / 600.0,
            "vbd_iterations": 10,
        },
    }
    (ASSET_DIR / "asset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    manifest_paths = [
        ASSET_DIR / "source" / "armadillo_stanford_original.obj",
        ASSET_DIR / "armadillo_stanford_surface.obj",
        ASSET_DIR / "armadillo_stanford_tet.npz",
        ASSET_DIR / "crusher_drum_18t.obj",
        ASSET_DIR / "crusher_drum_18t.npz",
        ASSET_DIR / "asset_metadata.json",
    ]
    lines = [f"{sha256(p)}  {p.relative_to(ROOT).as_posix()}" for p in manifest_paths]
    (ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

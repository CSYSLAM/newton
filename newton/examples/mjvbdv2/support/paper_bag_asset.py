# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Load, assemble, and render the deformable paper shopping bag."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton


@dataclass
class PaperBagMesh:
    vertices: np.ndarray
    faces: np.ndarray
    paper_face_count: int
    handles: tuple[np.ndarray, np.ndarray]
    bottom: np.ndarray
    plies: np.ndarray | None = None


ASSET_DIR = Path(__file__).resolve().parents[4] / "assets/piper_paper_bag"


def make_paper_bag() -> PaperBagMesh:
    """Load the 90 x 200 x 260 mm bag authored with Blender MCP."""
    with np.load(ASSET_DIR / "simulation_mesh.npz") as data:
        return PaperBagMesh(
            data["vertices"],
            data["faces"],
            int(data["paper_face_count"]),
            (data["handle_0"], data["handle_1"]),
            data["bottom"],
            data["plies"],
        )


def add_paper_bag(
    builder: newton.ModelBuilder, bag: PaperBagMesh, position, *, membrane_stiffness=5e4, bending_stiffness=100.0
):
    """Add elastic paperboard, bonded reinforcement, and flexible cord handles.

    Stiffness coefficients are an experimental demo material, not a calibrated
    paper grade. Bonded double plies scale membrane stiffness and mass by two
    and bending stiffness by eight. Every vertex remains dynamic.
    """
    first_particle, first_face, first_edge = builder.particle_count, builder.tri_count, len(builder.edge_indices)
    density = 0.18
    builder.add_cloth_mesh(
        pos=wp.vec3(*position),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(),
        vertices=bag.vertices.tolist(),
        indices=bag.faces.ravel().tolist(),
        density=density,
        tri_ke=membrane_stiffness,
        tri_ka=membrane_stiffness,
        tri_kd=0.2,
        edge_ke=bending_stiffness,
        edge_kd=0.02,
        particle_radius=0.0012,
        color=(0.57, 0.37, 0.16),
    )
    plies = bag.plies if bag.plies is not None else np.ones(len(bag.faces), dtype=int)
    edge_plies = {}
    for face_id, (face, ply) in enumerate(zip(bag.faces, plies, strict=True)):
        if face_id >= bag.paper_face_count:
            continue
        if ply > 1:
            i = first_face + face_id
            ke, ka, kd, drag, lift = builder.tri_materials[i]
            builder.tri_materials[i] = (ke * ply, ka * ply, kd * ply, drag, lift)
            for vertex in face:
                builder.particle_mass[first_particle + vertex] += (ply - 1) * density * builder.tri_areas[i] / 3
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = tuple(sorted((int(a), int(b))))
            edge_plies[key] = min(edge_plies.get(key, int(ply)), int(ply))
    paper_end = int(bag.handles[0].min())
    for i in range(first_edge, len(builder.edge_indices)):
        edge = np.asarray(builder.edge_indices[i]) - first_particle
        if edge.max() >= paper_end:
            builder.edge_bending_properties[i] = (0.1, 0.0001)
        else:
            ply = edge_plies.get(tuple(sorted(map(int, edge[2:]))), 1)
            builder.edge_bending_properties[i] = (bending_stiffness * ply**3, 0.02 * ply**3)


def paper_coordinates(bag: PaperBagMesh, positions: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Express world-space points in the fitted frame of the dynamic paper body."""
    end = int(bag.handles[0].min())
    rest = bag.vertices[:end].astype(np.float64)
    current = positions[:end].astype(np.float64)
    rest_center, current_center = rest.mean(axis=0), current.mean(axis=0)
    rest -= rest_center
    current -= current_center
    u, _, vt = np.linalg.svd(rest.T @ current)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)
    return (points - current_center) @ (u @ correction @ vt).T + rest_center


def parcel_is_contained(bag: PaperBagMesh, positions: np.ndarray, points: np.ndarray, *, radius=0.0) -> bool:
    """Require the whole parcel below the rim and above the moving bag bottom.

    This conservative interior envelope is for the nearly undeformed demo bag;
    it is not a general containment query for collapsed cloth surfaces.
    """
    local = paper_coordinates(bag, positions, points)
    lower = np.array((-0.045, -0.1, -0.003))
    upper = np.array((0.045, 0.1, 0.245))
    return bool(np.all(local.min(axis=0) - radius >= lower) and np.all(local.max(axis=0) + radius <= upper))


def measure_paper_shape(bag: PaperBagMesh, positions: np.ndarray) -> dict[str, float]:
    """Measure shell distortion [m] after removing rigid translation/rotation."""
    end = int(bag.handles[0].min())
    local = paper_coordinates(bag, positions, positions[:end])
    error = np.linalg.norm(local - bag.vertices[:end], axis=1)
    return {"shape_rms": float(np.sqrt(np.mean(error**2))), "shape_max": float(error.max())}


@wp.kernel
def deform_render_mesh(
    q: wp.array[wp.vec3],
    indices: wp.array[wp.vec3i],
    weights: wp.array[wp.vec3],
    offsets: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    i = wp.tid()
    tri, w, d = indices[i], weights[i], offsets[i]
    a, b, c = q[tri[0]], q[tri[1]], q[tri[2]]
    tangent = wp.normalize(b - a)
    normal = wp.normalize(wp.cross(b - a, c - a))
    bitangent = wp.cross(normal, tangent)
    output[i] = w[0] * a + w[1] * b + w[2] * c + d[0] * tangent + d[1] * bitangent + d[2] * normal

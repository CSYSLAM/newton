# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Load a closed pneumatic pouch with airless welded tabs in Newton MJVBDV2.

Coordinates are meters. The whole bag is free; there are no internal chips.
NumPy-only rendering helpers also work in Blender without importing Newton.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def bind_render_points(mesh: dict, positions: np.ndarray) -> np.ndarray:
    """Map bag-local simulation positions to the visible geometry [m]."""
    tri = positions[mesh["render_parent_vertices"]]
    result = np.einsum("ij,ijk->ik", mesh["render_weights"], tri)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-12)
    return result + normal * mesh["render_normal_offsets"][:, None]


def bind_render_normals(mesh: dict, positions: np.ndarray) -> np.ndarray:
    """Smooth each film panel separately, retaining the weld crease and cut edge."""
    groups = mesh["render_normal_groups"]
    normals = np.zeros((len(groups), 3), dtype=positions.dtype)
    parents, weights = mesh["render_parent_vertices"], mesh["render_weights"]
    for group in range(4):
        faces = mesh["triangles"][mesh["triangle_groups"] == group]
        face_normals = np.cross(
            positions[faces[:, 1]] - positions[faces[:, 0]], positions[faces[:, 2]] - positions[faces[:, 0]]
        )
        vertex_normals = np.zeros_like(positions)
        for corner in range(3):
            np.add.at(vertex_normals, faces[:, corner], face_normals)
        vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1)[:, None], 1e-12)
        mask = groups == group
        normals[mask] = np.einsum("ij,ijk->ik", weights[mask], vertex_normals[parents[mask]])
    normals *= mesh["render_normal_signs"][:, None]
    if np.any(groups < 0):
        p = bind_render_points(mesh, positions)
        faces = mesh["render_triangles"]
        face_normals = np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]])
        geometric = np.zeros_like(p)
        for corner in range(3):
            np.add.at(geometric, faces[:, corner], face_normals)
        normals[groups < 0] = geometric[groups < 0]
    zero = np.linalg.norm(normals, axis=1) < 1e-12
    if zero.any():
        tri = positions[parents[zero]]
        normals[zero] = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    return normals


@dataclass
class SoftBag:
    """One free package, including its pressure cavity and welded tab particles."""

    particle_start: int
    particle_end: int
    cavity: object
    config: dict
    mesh: dict

    def solver_vbd_options(self) -> dict:
        """Return solver settings; the host owns the timestep and contact pipeline."""
        c = self.config["solver"]
        return {
            "iterations": c["iterations"],
            "pneumatic_enable_incremental_volume": True,
            "pneumatic_enable_color_coupling": self.config["pneumatic"]["mode"] == "target-volume",
            "particle_enable_self_contact": True,
            "particle_self_contact_radius": c["self_contact_radius_m"],
            "particle_self_contact_margin": c["self_contact_margin_m"],
            "rigid_body_particle_contact_buffer_size": 8192,
            "friction_epsilon": 0.0001,
        }

    def bind_render(self, particle_positions: np.ndarray) -> np.ndarray:
        """Return visible vertices from the host's complete particle array [m]."""
        return bind_render_points(self.mesh, particle_positions[self.particle_start : self.particle_end])

    def bind_normals(self, particle_positions: np.ndarray) -> np.ndarray:
        """Return visible unit normals from the host's complete particle array."""
        return bind_render_normals(self.mesh, particle_positions[self.particle_start : self.particle_end])


def add_softbag(builder, *, directory=None, pos=None, rot=None, mass_kg=None) -> SoftBag:
    """Add a free cavity, optionally rescaling the filled package's mass [kg]."""
    import warp as wp  # noqa: PLC0415 -- Keep Blender's binding helpers independent of Newton.

    from newton.solvers import PneumaticConfig, PneumaticMode, add_pneumatic_cavity  # noqa: PLC0415

    directory = (
        Path(directory)
        if directory is not None
        else Path.home() / "下载/oishi_softbag_usd_v3_20260922/oishi_shrimp_softbag"
    )
    config = json.loads((directory / "physics.json").read_text())
    if config["schema_version"] != 2:
        raise ValueError("This loader expects schema 2 with explicit cavity and weld topology.")
    with np.load(directory / "physics_mesh.npz", allow_pickle=False) as data:
        mesh = {key: data[key].copy() for key in data.files}
    if mass_kg is not None:
        if not np.isfinite(mass_kg) or mass_kg <= 0:
            raise ValueError("Soft package mass must be finite and positive")
        mesh["particle_masses"] *= mass_kg / float(mesh["particle_masses"].sum())
        config["density_kg_m2"] *= mass_kg / config["mass_kg"]
        config["mass_kg"] = float(mass_kg)
    start, triangle_start = builder.particle_count, len(builder.tri_indices)
    translation = np.zeros(3) if pos is None else np.asarray(pos)
    rotation = wp.quat_identity() if rot is None else rot
    matrix = np.asarray(wp.quat_to_matrix(rotation)).reshape(3, 3)
    positions = mesh["points"] @ matrix.T + translation
    radius = np.full(len(positions), config["particle_radius_m"])
    radius[mesh["seal_particle_indices"]] = config["seals"]["particle_radius_m"]
    builder.add_particles(
        positions.tolist(),
        [[0.0, 0.0, 0.0]] * len(positions),
        mass=mesh["particle_masses"].tolist(),
        radius=radius.tolist(),
    )
    faces = mesh["triangles"] + start
    scales = np.where(mesh["triangle_groups"] >= 2, config["seals"]["membrane_stiffness_scale"], 1.0)
    m = config["membrane"]
    builder.add_triangles(
        faces[:, 0],
        faces[:, 1],
        faces[:, 2],
        tri_ke=(scales * m["tri_ke"]).tolist(),
        tri_ka=(scales * m["tri_ka"]).tolist(),
        tri_kd=np.full(len(faces), m["tri_kd"]).tolist(),
    )
    edges = np.where(mesh["bending_edges"] >= 0, mesh["bending_edges"] + start, -1)
    scale_table = [1.0, config["seals"]["bending_stiffness_scale"], config["seals"]["weld_bending_stiffness_scale"]]
    scales = np.asarray(scale_table)[mesh["bending_edge_groups"]]
    builder.add_edges(
        edges[:, 0],
        edges[:, 1],
        edges[:, 2],
        edges[:, 3],
        edge_ke=(m["edge_ke"] * scales).tolist(),
        edge_kd=np.full(len(edges), m["edge_kd"]).tolist(),
    )
    pneumatic = dict(config["pneumatic"])
    pneumatic["mode"] = {"target-volume": PneumaticMode.TARGET_VOLUME, "isothermal": PneumaticMode.ISOTHERMAL}[
        pneumatic["mode"]
    ]
    cavity = add_pneumatic_cavity(
        builder, (mesh["cavity_triangle_indices"] + triangle_start).tolist(), config=PneumaticConfig(**pneumatic)
    )
    return SoftBag(start, builder.particle_count, cavity, config, mesh)

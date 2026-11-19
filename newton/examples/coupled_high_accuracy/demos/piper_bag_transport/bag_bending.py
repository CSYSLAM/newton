"""Preserve developed-sheet bending at welded, non-manifold bag seams."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def rebuild_bag_bending(
    builder,
    triangles,
    panel_vertices,
    panel_triangles,
    *,
    particle_start: int,
    edge_start: int,
    edge_aniso_ke,
    edge_kd: float,
) -> dict[str, int]:
    """Replace the just-added cloth's edges without dropping faces at heat seals.

    Call immediately after ``style3d.add_cloth_mesh`` with the same local triangle
    and panel arrays. Panel vertex IDs identify continuous pieces of uncut film;
    matching 3D particle IDs identify welds, which can join more than two pieces.
    """
    triangles = np.asarray(triangles, dtype=np.int32).reshape(-1, 3)
    panel_triangles = np.asarray(panel_triangles, dtype=np.int32).reshape(-1, 3)
    panel_vertices = np.asarray(panel_vertices, dtype=np.float64).reshape(-1, 2)
    if triangles.shape != panel_triangles.shape:
        raise ValueError("The physical and developed-panel triangles must correspond one to one.")
    if not 0 <= edge_start <= len(builder.edge_indices):
        raise ValueError("edge_start must identify the just-added cloth's edge suffix.")
    anisotropy = np.asarray(edge_aniso_ke, dtype=np.float64).reshape(3)
    adjacency = defaultdict(list)
    for face, triangle in enumerate(triangles):
        for opposite in range(3):
            corners = ((opposite + 1) % 3, (opposite + 2) % 3)
            endpoints = tuple(sorted(int(triangle[corner]) for corner in corners))
            panel_edge = tuple(sorted(int(panel_triangles[face, corner]) for corner in corners))
            adjacency[endpoints].append((face, opposite, panel_edge))

    rows, areas, cots, stiffnesses = [], [], [], []
    nonmanifold = sheet_hinges = 0

    def add_row(first, second=None):
        face, opposite, _ = first
        triangle = triangles[face]
        start, end = int(triangle[(opposite + 1) % 3]), int(triangle[(opposite + 2) % 3])
        other = -1 if second is None else int(triangles[second[0], second[1]])
        rows.append((int(triangle[opposite]), other, start, end))
        if second is None or len(set(rows[-1])) < 4:
            areas.append(0.0)
            cots.append((0.0, 0.0, 0.0, 0.0))
            stiffnesses.append(0.0)
            return

        face_cots, face_areas, directions = [], [], []
        for adjacent, tip, _ in (first, second):
            ids = triangles[adjacent]
            developed = panel_vertices[panel_triangles[adjacent]]
            p = developed[int(np.flatnonzero(ids == start)[0])]
            q = developed[int(np.flatnonzero(ids == end)[0])]
            r = developed[tip]
            edge, offset = q - p, r - p
            double_area = abs(float(edge[0] * offset[1] - edge[1] * offset[0]))
            # Match Style3D's panel metric regularization without using the folded 3D rest pose.
            face_cots.append(
                (float(edge @ offset) / (double_area + 1.0e-8), float((-edge) @ (r - q)) / (double_area + 1.0e-8))
            )
            face_areas.append(double_area)
            directions.append(np.arctan2(edge[1], edge[0]))
        angle = 0.5 * sum(directions)
        sine, cosine = np.sin(angle), np.cos(angle)
        stiffnesses.append(float(anisotropy @ (sine**12, cosine**12, 4.0 * sine**2 * cosine**2)))
        areas.append(0.5 * (sum(face_areas) + 1.0e-8))
        cots.append((face_cots[0][0], face_cots[1][0], face_cots[0][1], face_cots[1][1]))

    for adjacent in adjacency.values():
        if len(adjacent) <= 2:
            add_row(*adjacent)
            continue
        nonmanifold += 1
        sheets = defaultdict(list)
        for face in adjacent:
            sheets[face[2]].append(face)
        for continuous in sheets.values():
            if len(continuous) > 2:
                raise ValueError("A developed film edge cannot have more than two incident faces.")
            # A heat seal joins membranes positionally; it does not invent all-pairs
            # flat bending springs between distinct plies of the folded plastic.
            add_row(*continuous)
            sheet_hinges += int(len(continuous) == 2)

    for name in ("edge_indices", "edge_rest_angle", "edge_rest_length", "edge_bending_properties"):
        del getattr(builder, name)[edge_start:]
    for attribute in builder.custom_attributes.values():
        if getattr(attribute.frequency, "name", str(attribute.frequency)).upper() == "EDGE":
            for index in tuple(attribute.values):
                if index >= edge_start:
                    del attribute.values[index]
    indices = np.asarray(rows, dtype=np.int32).reshape(-1, 4)
    indices[indices >= 0] += particle_start
    if len(indices):
        builder.add_edges(
            *indices.T.tolist(),
            edge_ke=stiffnesses,
            edge_kd=[edge_kd] * len(indices),
            custom_attributes={
                "style3d:edge_rest_area": areas,
                "style3d:edge_bending_cot": cots,
                "style3d:aniso_ke": [tuple(anisotropy)] * len(indices),
            },
        )
    return {
        "physical_edge_count": len(adjacency),
        "bending_hinge_count": sum(area > 0.0 for area in areas),
        "boundary_stencil_count": sum(area == 0.0 for area in areas),
        "nonmanifold_edge_count": nonmanifold,
        "nonmanifold_sheet_hinge_count": sheet_hinges,
        "maximum_incident_faces": max(map(len, adjacency.values()), default=0),
        "represented_face_sides": sum(1 if row[1] < 0 else 2 for row in rows),
    }

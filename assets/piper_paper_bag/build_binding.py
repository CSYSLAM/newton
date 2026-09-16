# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Bind Blender render detail to the shell, using only NumPy at export time."""

import json
from pathlib import Path

import numpy as np


def closest_weights(points, triangles):
    """Return closest triangle indices and barycentric coordinates for points."""
    a, b, c = (triangles[:, i].astype(np.float64) for i in range(3))
    ab, ac = b - a, c - a
    d00, d01, d11 = np.sum(ab * ab, axis=1), np.sum(ab * ac, axis=1), np.sum(ac * ac, axis=1)
    delta = points[:, None] - a
    d20, d21 = np.sum(delta * ab, axis=2), np.sum(delta * ac, axis=2)
    determinant = d00 * d11 - d01**2
    u = (d11 * d20 - d01 * d21) / determinant
    v = (d00 * d21 - d01 * d20) / determinant
    weights = np.stack((1 - u - v, u, v), axis=2)
    projected = a + u[:, :, None] * ab + v[:, :, None] * ac
    distance = np.sum((points[:, None] - projected) ** 2, axis=2)
    distance[np.any(weights < 0, axis=2)] = np.inf
    for i, j in ((0, 1), (1, 2), (2, 0)):
        start, end = triangles[:, i], triangles[:, j]
        edge = end - start
        t = np.clip(np.sum((points[:, None] - start) * edge, axis=2) / np.sum(edge * edge, axis=1), 0, 1)
        candidate = start + t[:, :, None] * edge
        candidate_distance = np.sum((points[:, None] - candidate) ** 2, axis=2)
        mask = candidate_distance < distance
        candidate_weights = np.zeros_like(weights)
        candidate_weights[:, :, i], candidate_weights[:, :, j] = 1 - t, t
        weights[mask] = candidate_weights[mask]
        distance[mask] = candidate_distance[mask]
    selected = np.argmin(distance, axis=1)
    return selected, weights[np.arange(len(points)), selected]


def build_binding(directory):
    """Write rest-frame offsets; runtime deformation uses the same local frame."""
    directory = Path(directory)
    with np.load(directory / "simulation_mesh.npz") as sim, np.load(directory / "render_mesh.npz") as render:
        vertices, faces = render["vertices"], render["faces"]
        triangles = sim["vertices"][sim["faces"]]
        split = int(sim["paper_face_count"])
        parts = json.loads((directory / "render_parts.json").read_text())
        indices = np.full((len(vertices), 3), -1, dtype=np.int32)
        weights = np.zeros_like(vertices)
        offsets = np.zeros_like(vertices)
        for kind, subset in (("paper", np.arange(split)), ("cord", np.arange(split, len(triangles)))):
            visual = np.unique(
                np.concatenate([faces[p["start"] : p["end"]].ravel() for p in parts if p["material"] == kind])
            )
            for first in range(0, len(visual), 64):
                ids = visual[first : first + 64]
                selected, barycentric = closest_weights(vertices[ids], triangles[subset])
                selected = subset[selected]
                tri = triangles[selected]
                closest = np.sum(tri * barycentric[:, :, None], axis=1)
                tangent = tri[:, 1] - tri[:, 0]
                tangent /= np.linalg.norm(tangent, axis=1)[:, None]
                normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                normal /= np.linalg.norm(normal, axis=1)[:, None]
                bitangent = np.cross(normal, tangent)
                delta = vertices[ids] - closest
                indices[ids] = sim["faces"][selected]
                weights[ids] = barycentric
                offsets[ids] = np.column_stack([np.sum(delta * axis, axis=1) for axis in (tangent, bitangent, normal)])
        if np.any(indices < 0):
            raise ValueError("Render vertices have no physical surface binding")
        np.savez_compressed(directory / "render_binding.npz", indices=indices, weights=weights, offset=offsets)
        print(f"Bound {len(vertices)} vertices; maximum detail offset {np.linalg.norm(offsets, axis=1).max():.6f} m")


if __name__ == "__main__":
    build_binding(Path(__file__).resolve().parent)

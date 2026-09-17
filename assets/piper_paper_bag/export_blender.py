# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Export the editable bag scene from Blender; then run build_binding.py."""

import json
from pathlib import Path

import bpy
import numpy as np


def export_asset(directory):
    """Export physical topology and evaluated render meshes in metres."""
    directory = Path(directory)
    scene = bpy.data.scenes["PiPER kraft bag asset"]
    physics = scene.objects["Simulation shell (hidden)"]
    vertices = np.asarray([physics.matrix_world @ v.co for v in physics.data.vertices], dtype=np.float32)
    faces = np.asarray([face.vertices[:] for face in physics.data.polygons], dtype=np.int32)
    groups = {}
    for name in ("handle_0", "handle_1", "bottom"):
        group = physics.vertex_groups[name].index
        groups[name] = np.asarray(
            [v.index for v in physics.data.vertices if any(g.group == group for g in v.groups)], dtype=np.int32
        )
    plies = np.asarray([value.value for value in physics.data.attributes["paper_plies"].data], dtype=np.int32)
    np.savez_compressed(
        directory / "simulation_mesh.npz",
        vertices=vertices,
        faces=faces,
        paper_face_count=int(physics["paper_face_count"]),
        plies=plies,
        **groups,
    )
    points, triangles, parts = [], [], []
    point_count = face_count = 0
    graph = bpy.context.evaluated_depsgraph_get()
    for obj in sorted(scene.objects, key=lambda item: item.name):
        if not obj.get("paper_bag_asset"):
            continue
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            mesh.calc_loop_triangles()
            v = np.asarray([obj.matrix_world @ vertex.co for vertex in mesh.vertices], dtype=np.float32)
            f = np.asarray([tri.vertices[:] for tri in mesh.loop_triangles], dtype=np.int32)
            points.append(v)
            triangles.append(f + point_count)
            parts.append(
                {
                    "name": obj.name,
                    "start": face_count,
                    "end": face_count + len(f),
                    "material": "cord" if obj.name.startswith("Braided handle") else "paper",
                }
            )
            point_count += len(v)
            face_count += len(f)
        finally:
            evaluated.to_mesh_clear()
    np.savez_compressed(directory / "render_mesh.npz", vertices=np.concatenate(points), faces=np.concatenate(triangles))
    (directory / "render_parts.json").write_text(json.dumps(parts, indent=2) + "\n")
    print(f"Exported {len(vertices)} physical vertices and {point_count} render vertices")


if __name__ == "__main__":
    export_asset(Path(__file__).resolve().parent)

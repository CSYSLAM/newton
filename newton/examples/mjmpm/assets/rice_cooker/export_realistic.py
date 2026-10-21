# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Export the authored RicePour Blender scene as simulation-sized visual meshes.

Open rice_pour_studio.blend, then run this script in Blender. The exported
archive needs only NumPy at runtime. Staged rice, tabletop and studio lights
are excluded; Newton supplies the moving cup and material particle positions.
"""

import json
import math
import runpy
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

DESTINATION = Path(__file__).resolve().parent


def export():
    """Bake evaluated visual meshes and a material manifest in SI units."""
    source = bpy.data.collections["Rice Pour · product assets"]
    # Evaluate the authored cooker modifiers even when another Blender scene
    # (such as the cup detail study) is currently active.
    linked_source = source.name not in bpy.context.scene.collection.children
    if linked_source:
        bpy.context.scene.collection.children.link(source)
    fitted = bpy.data.collections.new("Newton fitted pitcher export")
    bpy.context.scene.collection.children.link(fitted)
    meshes, manifest = {}, []

    geometry = runpy.run_path(str(DESTINATION.parents[1] / "support/rice_cooker_geometry.py"))
    model = runpy.run_path(str(DESTINATION / "model_measuring_cup.py"))
    model["build_measuring_cup"](fitted, geometry)

    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cooker_rotation = Matrix.Rotation(-math.pi / 2, 4, "Z")
    cooker_scale = Matrix.Diagonal((0.12 / 0.144, 0.12 / 0.144, 0.15 / 0.225, 1))
    objects = [(obj, "cup", Matrix.Identity(4)) for obj in fitted.objects]
    for obj in source.objects:
        if obj.name.startswith(("COOKER", "POT", "LID", "HINGE", "CONTROL", "BRAND")) and obj.type != "EMPTY":
            objects.append((obj, "cooker", cooker_rotation @ cooker_scale))
    objects.append((bpy.data.objects["RICE · single grain instancing master"], "grain", Matrix.Identity(4)))
    for obj, parent, mapping in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        if mesh is None:
            continue
        mesh.calc_loop_triangles()
        transform = mapping @ obj.matrix_world
        normal_transform = transform.to_3x3().inverted().transposed()
        for slot, material in enumerate(mesh.materials):
            triangles = [t for t in mesh.loop_triangles if t.material_index == slot]
            if not triangles:
                continue
            points, normals = [], []
            for tri in triangles:
                for vertex, loop in zip(tri.vertices, tri.loops, strict=True):
                    points.append(tuple(transform @ mesh.vertices[vertex].co))
                    normal = mesh.corner_normals[loop].vector
                    normals.append(tuple((normal_transform @ Vector(normal)).normalized()))
            key = f"part_{len(manifest)}"
            # Weld corners that share a position and normal to keep the archive small.
            corners = np.concatenate([np.asarray(points, np.float32), np.asarray(normals, np.float32)], axis=1)
            unique, indices = np.unique(corners, axis=0, return_inverse=True)
            meshes[key + "_points"] = np.ascontiguousarray(unique[:, :3])
            # Half precision is ample for shading normals.
            meshes[key + "_normals"] = unique[:, 3:].astype(np.float16)
            meshes[key + "_indices"] = indices.reshape(-1).astype(np.int32)
            shader = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
            manifest.append(
                {
                    "key": key,
                    "name": obj.get("newton_asset_name", obj.name),
                    "parent": parent,
                    "color": list(material.get("newton_color", shader.inputs["Base Color"].default_value[:3])),
                    "roughness": shader.inputs["Roughness"].default_value,
                    "metallic": shader.inputs["Metallic"].default_value,
                    "opacity": float(material.get("newton_opacity", 1.0)),
                }
            )
        evaluated.to_mesh_clear()
    meshes["manifest"] = np.asarray(json.dumps(manifest))
    np.savez_compressed(DESTINATION / "realistic_visuals.npz", **meshes)
    # The temporary export geometry is not part of the presentation scene.
    for obj in list(fitted.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(fitted)
    if linked_source:
        bpy.context.scene.collection.children.unlink(source)
    print(f"Exported {len(manifest)} visual parts to {DESTINATION / 'realistic_visuals.npz'}")


if __name__ == "__main__":
    export()

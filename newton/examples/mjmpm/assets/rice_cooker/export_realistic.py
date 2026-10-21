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
    fitted = bpy.data.collections.new("Newton fitted pitcher export")
    bpy.context.scene.collection.children.link(fitted)
    glass = bpy.data.materials["06 · Smoked clear measuring pitcher"]
    ink = bpy.data.materials["07 · Warm grey measurement ink"]
    meshes, manifest = {}, []

    geometry = runpy.run_path(str(DESTINATION.parents[1] / "support/rice_cooker_geometry.py"))
    vertices, faces = geometry["cup_mesh"]()
    mesh = bpy.data.meshes.new("Round tapered pitcher with pouring spout")
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    vessel = bpy.data.objects.new("Newton pitcher shell", mesh)
    fitted.objects.link(vessel)
    vessel.data.materials.append(glass)
    for p in mesh.polygons:
        p.use_smooth = True
    # Three narrow bars leave the handle open; shared dimensions match contact.
    for index, (center, half) in enumerate(geometry["HANDLE_BOXES"]):
        bpy.ops.mesh.primitive_cube_add(size=1, location=center)
        handle = bpy.context.object
        handle.name = f"Newton pitcher open handle {index}"
        handle.scale = tuple(2 * x for x in half)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        handle.data.materials.append(glass)
        bevel = handle.modifiers.new("Molded handle edges", "BEVEL")
        bevel.width, bevel.segments = 0.002, 4
        handle.modifiers.new("Corner normals", "WEIGHTED_NORMAL")
        for col in list(handle.users_collection):
            col.objects.unlink(handle)
        fitted.objects.link(handle)
    for ml in (25, 50, 75, 100):
        # Calibrated against the tapered cavity rather than decorative labels.
        candidates = np.linspace(geometry["CUP_WALL"], geometry["CUP_HEIGHT"], 1000)
        areas = math.pi * geometry["cup_inner_radius"](candidates) ** 2
        volumes = np.cumsum(areas) * (candidates[1] - candidates[0])
        z = float(candidates[np.argmin(abs(volumes - ml * 1e-6))])
        data = bpy.data.curves.new(f"Pitcher {ml} mL", "FONT")
        data.body, data.size, data.align_x = f"{ml} -", 0.0045, "CENTER"
        obj = bpy.data.objects.new(data.name, data)
        fitted.objects.link(obj)
        outer = geometry["CUP_BOTTOM_RADIUS"] + (geometry["CUP_RADIUS"] - geometry["CUP_BOTTOM_RADIUS"]) * (
            z - 0.008
        ) / (geometry["CUP_HEIGHT"] - 0.008)
        obj.location = (0, -outer - 0.0003, z)
        obj.rotation_euler[0] = math.pi / 2
        data.materials.append(ink)

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
            meshes[key + "_normals"] = np.ascontiguousarray(unique[:, 3:])
            meshes[key + "_indices"] = indices.reshape(-1).astype(np.int32)
            shader = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
            manifest.append(
                {
                    "key": key,
                    "name": obj.name,
                    "parent": parent,
                    "color": list(shader.inputs["Base Color"].default_value[:3]),
                    "roughness": shader.inputs["Roughness"].default_value,
                    "metallic": shader.inputs["Metallic"].default_value,
                    "opacity": 0.18
                    if parent == "cup" and shader.inputs["Transmission Weight"].default_value > 0.5
                    else 1.0,
                }
            )
        evaluated.to_mesh_clear()
    meshes["manifest"] = np.asarray(json.dumps(manifest))
    np.savez_compressed(DESTINATION / "realistic_visuals.npz", **meshes)
    # The temporary export geometry is not part of the presentation scene.
    for obj in list(fitted.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(fitted)
    print(f"Exported {len(manifest)} visual parts to {DESTINATION / 'realistic_visuals.npz'}")


if __name__ == "__main__":
    export()

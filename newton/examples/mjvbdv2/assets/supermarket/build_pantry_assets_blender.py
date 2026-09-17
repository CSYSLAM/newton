# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Prepare licensed pantry meshes for Newton, retaining source UVs/normals."""

import json
from pathlib import Path

import bpy
from mathutils import Vector


def build(output_dir):
    """Import the CC0 source in a new scene and export bottom-centered assets."""
    output = Path(output_dir)
    previous = bpy.context.window.scene
    scene = bpy.data.scenes.new("Prepared CC0 pantry assets")
    bpy.context.window.scene = scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    bpy.ops.import_scene.gltf(filepath=str(output / "polyhaven/long_life_food/long_life_food_1k.gltf"))
    groups = {}
    usd_dir = output / "usd"
    usd_dir.mkdir(exist_ok=True)
    for obj in list(scene.objects):
        if obj.type != "MESH":
            continue
        mesh = obj.data
        mesh.calc_loop_triangles()
        world = [obj.matrix_world @ vertex.co for vertex in mesh.vertices]
        low = Vector(tuple(min(v[i] for v in world) for i in range(3)))
        high = Vector(tuple(max(v[i] for v in world) for i in range(3)))
        origin = Vector(((low.x + high.x) / 2, (low.y + high.y) / 2, low.z))
        normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
        item = {"vertices": [], "normals": [], "uvs": [], "indices": [], "dimensions_m": list(high - low)}
        unique = {}
        for triangle in mesh.loop_triangles:
            for loop_id in triangle.loops:
                vertex_id = mesh.loops[loop_id].vertex_index
                normal = tuple(round(v, 6) for v in (normal_matrix @ mesh.corner_normals[loop_id].vector).normalized())
                uv = tuple(mesh.uv_layers.active.data[loop_id].uv)
                key = (vertex_id, normal, uv)
                if key not in unique:
                    unique[key] = len(item["vertices"])
                    item["vertices"].append([round(v, 7) for v in world[vertex_id] - origin])
                    item["normals"].append(normal)
                    item["uvs"].append(uv)
                item["indices"].append(unique[key])
        item["texture"] = "polyhaven/long_life_food/textures/long_life_food_diff_1k.jpg"
        groups[obj.name.split(".")[0]] = item
        obj["source_url"] = "https://polyhaven.com/a/long_life_food"
        obj["license"] = "CC0-1.0"
        obj["simulation_role"] = "static_background_noncolliding"
        # Portable source asset, with a baked bottom-centered metric frame.
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        obj.location -= origin
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.wm.usd_export(
            filepath=str(usd_dir / (obj.name.split(".")[0] + ".usdc")),
            selected_objects_only=True,
            export_materials=True,
            export_textures_mode="NEW",
            export_custom_properties=True,
            root_prim_path="/Asset",
            convert_scene_units="METERS",
        )
        obj.location += origin
    (output / "pantry_meshes.json").write_text(json.dumps(groups, separators=(",", ":")), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "pantry_assets.blend"), copy=True)
    bpy.context.window.scene = previous

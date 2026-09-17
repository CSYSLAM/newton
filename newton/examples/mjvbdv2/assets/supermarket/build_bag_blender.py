# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Author the simulation bag in Blender; run through the Blender MCP bridge."""

import json
import math
from itertools import pairwise
from pathlib import Path

import bpy


def build(output_dir):
    """Create a welded open shopping-bag shell with two integral handles."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices, faces, lookup = [], [], {}
    nx, ny, nz = 16, 12, 16
    width, depth, height = 0.30, 0.18, 0.27

    def vertex(point):
        key = tuple(round(v, 8) for v in point)
        if key not in lookup:
            lookup[key] = len(vertices)
            vertices.append(point)
        return lookup[key]

    def grid(rows, reverse=False, handle_hole=False, hanging_slots=False):
        for k, (a, b) in enumerate(pairwise(rows)):
            for i in range(len(a) - 1):
                if handle_hole and 12 <= k < 15 and 3 <= i < 9:
                    continue
                if hanging_slots and 13 <= k < 15 and i in (1, 14):
                    continue
                triangles = [(a[i], a[i + 1], b[i + 1]), (a[i], b[i + 1], b[i])]
                faces.extend([tuple(reversed(t)) if reverse else t for t in triangles])

    def point(i, j, k):
        u, v, t = i / nx, j / ny, k / nz
        x = width * (u - 0.5)
        y = depth * (v - 0.5)
        # Vest cut: tall integral side shoulders with a broad central neck.
        shoulder = max(0.0, min(1.0, (abs(2 * u - 1) - 0.55) / 0.30))
        shoulder = shoulder * shoulder * (3 - 2 * shoulder)
        z = (height + 0.11 * shoulder) * t
        if j in (0, ny):
            y += 0.0006 * math.sin(8 * math.pi * u) * math.sin(math.pi * t)
        if i in (0, nx):
            x -= math.copysign(0.008 * math.sin(math.pi * v) * math.sin(math.pi * t), x)
            if 3 <= j <= 9 and 12 <= k <= 15 and (j in (3, 9) or k in (12, 15)):
                a, b = (j - 6) / 3, (k - 13.5) / 1.5
                y = 0.045 * a * math.sqrt(1 - b * b / 2)
                z = 0.320625 + 0.035625 * b * math.sqrt(1 - a * a / 2)
        return (x, y, z)

    grid([[vertex(point(i, j, 0)) for i in range(nx + 1)] for j in range(ny + 1)], True)
    for j in (0, ny):
        grid(
            [[vertex(point(i, j, k)) for i in range(nx + 1)] for k in range(nz + 1)],
            j == ny,
            hanging_slots=True,
        )
    handle_tops = []
    for i in (0, nx):
        rows = [[vertex(point(i, j, k)) for j in range(ny + 1)] for k in range(nz + 1)]
        grid(rows, i == 0, handle_hole=True)
    # Remove unused vertices inside the die-cut holes.
    used = sorted({i for face in faces for i in face})
    remap = {old: new for new, old in enumerate(used)}
    vertices = [vertices[i] for i in used]
    faces = [tuple(remap[i] for i in face) for face in faces]
    handle_tops = [remap[i] for i in handle_tops]
    mesh = bpy.data.meshes.new("ShoppingBag_WeldedShell")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("ShoppingBag_Simulation", mesh)
    bpy.context.scene.collection.objects.link(obj)
    material = bpy.data.materials.new("Ivory translucent polyethylene film")
    material.diffuse_color = (0.91, 0.94, 0.88, 1)
    material.use_nodes = True
    bsdf = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = material.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = material.diffuse_color
    bsdf.inputs["Roughness"].default_value = 0.32
    mesh.materials.append(material)
    # Continuous UV print avoids assigning entire coarse triangles to ink.
    image = bpy.data.images.new("Shopping bag leaf print", width=512, height=512)
    pixels = []
    for j in range(512):
        z = 0.40 * j / 511
        for i in range(512):
            x = 0.32 * i / 511 - 0.16
            u, v = x / 0.048, (z - 0.14) / 0.055
            leaf = (u - 0.4 * v) ** 2 / 0.38 + v * v < 1
            vein = abs(u - 0.4 * v) < 0.025
            ink = (leaf and not vein) or (0.060 < z < 0.066)
            pixels.extend((0.10, 0.32, 0.15, 1.0) if ink else (0.94, 0.96, 0.91, 1.0))
    image.pixels.foreach_set(pixels)
    image.filepath_raw = str(output_dir / "bag_print.png")
    image.file_format = "PNG"
    image.save()
    texture = material.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    material.node_tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    uv_layer = mesh.uv_layers.new(name="Film print UV")
    for loop in mesh.loops:
        point = mesh.vertices[loop.vertex_index].co
        uv_layer.data[loop.index].uv = ((point.x + 0.16) / 0.32, point.z / 0.40)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    obj["simulation_surface"] = True
    obj["material_note"] = "Thin shell; thickness belongs to the contact/material model, not a duplicate mesh."
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "shopping_bag.json").write_text(
        json.dumps(
            {
                "vertices": vertices,
                "triangles": faces,
                "handle_support_vertices": handle_tops,
                "support_mode": "contact_only_slotted_handles",
                "units": "m",
                "up_axis": "Z",
                "simulation_material": {
                    "density": 0.035,
                    "tri_ke": 1.0e4,
                    "tri_ka": 1.0e4,
                    "tri_kd": 0.5,
                    "edge_ke": 0.003,
                    "edge_kd": 0.001,
                    "particle_radius": 0.0015,
                },
                "material_calibration": "illustrative_not_measured",
                "rack_rail_centers_local": [[-0.121875, 0, 0.329], [0.121875, 0, 0.329]],
                "dimensions_m": [width, depth, height],
                "author": "Authored using Blender MCP",
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "shopping_bag.blend"), copy=True)
    print(f"Bag exported: {len(vertices)} vertices, {len(faces)} triangles, supports={handle_tops}")
    return obj

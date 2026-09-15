# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Author recognizable checkout fixtures and grocery packaging in Blender."""

import json
import math
from pathlib import Path

import bpy


def build(output_dir):
    """Export static decoration grouped by material, without physics proxies."""
    colors = {
        "steel": (0.48, 0.53, 0.55),
        "black": (0.035, 0.045, 0.045),
        "ivory": (0.86, 0.86, 0.80),
        "green": (0.08, 0.28, 0.15),
        "white": (0.95, 0.95, 0.90),
        "blue": (0.15, 0.42, 0.52),
        "orange": (0.90, 0.39, 0.06),
        "red": (0.55, 0.045, 0.025),
        "gold": (0.71, 0.51, 0.20),
        "screen": (0.04, 0.12, 0.14),
    }
    materials = {}
    for name, color in colors.items():
        material = bpy.data.materials.new(name)
        material.diffuse_color = (*color, 1)
        material["roughness"] = 0.28 if name == "steel" else 0.22 if name == "screen" else 0.58
        material["metallic"] = 0.85 if name == "steel" else 0.0
        materials[name] = material

    def finish(obj, name, material):
        obj.name = name
        obj.data.materials.append(materials[material])
        return obj

    def box(name, pos, size, material, bevel=0.004):
        bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
        obj = bpy.context.object
        obj.scale = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        if bevel:
            mod = obj.modifiers.new("Manufactured rounded edges", "BEVEL")
            mod.width, mod.segments = bevel, 3
        return finish(obj, name, material)

    def cylinder(name, pos, radius, depth, material):
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=depth, location=pos)
        for polygon in bpy.context.object.data.polygons:
            polygon.use_smooth = len(polygon.vertices) == 4
        return finish(bpy.context.object, name, material)

    def text(body, pos, size, material="white"):
        curve = bpy.data.curves.new("Package lettering", "FONT")
        curve.body, curve.size, curve.extrude = body, size, 0.0
        obj = bpy.data.objects.new(body, curve)
        bpy.context.scene.collection.objects.link(obj)
        obj.location, obj.rotation_euler = pos, (math.pi / 2, 0, 0)
        return finish(obj, body, material)

    def wrapped_text(body, center, z, radius, size, material="white"):
        """Wrap ink geometry onto a cylindrical label, not a tangent plane."""
        obj = text(body, (0, 0, 0), size, material)
        obj.data.align_x = "CENTER"
        obj.data.extrude = 0.0
        obj.rotation_euler = (0, 0, 0)
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.convert(target="MESH")
        for vertex in obj.data.vertices:
            x, y = vertex.co.x, vertex.co.y
            angle = x / radius
            sector = 2 * math.pi / 24
            # Match the actual 24-sided label mesh, with only 0.15 mm ink
            # clearance, rather than hovering above its inscribed facets.
            r = radius * math.cos(sector / 2) / math.cos(angle % sector - sector / 2) + 0.00015
            vertex.co = (center[0] + r * math.sin(angle), center[1] - r * math.cos(angle), z + y)
        return obj

    # Under-counter storage and customer-facing trim; worktop physics is in
    # the demo, so the exported decoration contains no colliding duplicates.
    box("Checkout cabinet", (0.40, -0.20, 0.35), (1.20, 0.52, 0.66), "ivory", 0.014)
    box("Dark kickboard", (0.40, -0.20, 0.055), (1.15, 0.48, 0.10), "black")
    for x in (0.08, 0.69):
        box("Front cabinet door", (x, -0.468, 0.36), (0.58, 0.014, 0.53), "green")
        box("Door handle", (x + 0.19, -0.484, 0.46), (0.012, 0.016, 0.12), "steel")
    box("Checkout front trim", (0.4, -0.494, 0.676), (1.28, 0.012, 0.032), "steel")
    text("FRESH MARKET", (-0.02, -0.4752, 0.33), 0.050)

    # Pole-mounted POS, scanner glass, receipt printer, payment keypad.
    cylinder("POS foot", (-0.12, -0.30, 0.732), 0.065, 0.018, "black")
    cylinder("POS stand", (-0.12, -0.30, 0.86), 0.015, 0.25, "steel")
    box("POS bezel", (-0.12, -0.30, 1.00), (0.24, 0.034, 0.16), "black", 0.012)
    box("POS glass", (-0.12, -0.319, 1.00), (0.213, 0.003, 0.136), "screen", 0.002)
    text("CHECKOUT  03", (-0.217, -0.3207, 1.04), 0.013)
    text("FRESH FOOD", (-0.217, -0.3207, 1.002), 0.012, "gold")
    text("TOTAL   12.80", (-0.217, -0.3207, 0.960), 0.014)
    box("Scanner stainless bezel", (0.10, -0.40, 0.728), (0.17, 0.13, 0.016), "steel")
    box("Scanner optical glass", (0.10, -0.40, 0.738), (0.12, 0.09, 0.004), "black")
    box("Scanner line", (0.10, -0.40, 0.741), (0.09, 0.0015, 0.001), "red", 0)
    box("Receipt printer", (0.82, -0.34, 0.758), (0.12, 0.12, 0.072), "black", 0.012)
    box("Receipt paper", (0.82, -0.407, 0.785), (0.065, 0.055, 0.001), "white", 0)
    for n in range(7):
        box("Printed receipt line", (0.82, -0.428 + n * 0.006, 0.786), (0.045, 0.0007, 0.0002), "black", 0)

    # A complete gondola, price rails and different product silhouettes.
    box("Gondola back", (0.50, 1.18, 0.80), (1.95, 0.035, 1.58), "ivory")
    for x in (-0.49, 1.49):
        box("Shelf upright", (x, 1.11, 0.80), (0.035, 0.11, 1.60), "steel")
    for level, z in enumerate((0.22, 0.65, 1.08)):
        box("Retail shelf", (0.50, 1.02, z), (2.00, 0.36, 0.025), "ivory")
        box("Shelf price rail", (0.50, 0.834, z), (2.00, 0.015, 0.036), "black", 0.002)
        for i in range(10):
            x = -0.37 + i * 0.193
            box("Price card", (x, 0.824, z), (0.095, 0.001, 0.024), "white", 0)
            text(("2.49", "3.90", "1.80")[level], (x - 0.036, 0.8233, z - 0.006), 0.016, "black")
            if level == 0:
                # Prepared CC0 pantry meshes are instanced by the demo.
                continue
            elif level == 1:
                # Bottles have a tapered shoulder, neck, screw cap and label.
                cylinder("Juice bottle", (x, 0.99, z + 0.11), 0.045, 0.19, "orange" if i % 2 else "green")
                bpy.ops.mesh.primitive_cone_add(
                    vertices=24, radius1=0.045, radius2=0.017, depth=0.035, location=(x, 0.99, z + 0.2225)
                )
                for polygon in bpy.context.object.data.polygons:
                    polygon.use_smooth = len(polygon.vertices) == 4
                finish(bpy.context.object, "Bottle shoulder", "orange" if i % 2 else "green")
                cylinder("Screw cap", (x, 0.99, z + 0.254), 0.019, 0.027, "white")
                cylinder("Bottle label", (x, 0.99, z + 0.12), 0.0455, 0.095, "ivory")
                wrapped_text("JUICE", (x, 0.99), z + 0.127, 0.0455, 0.018, "green")
                wrapped_text("100% FRUIT", (x, 0.99), z + 0.109, 0.0455, 0.007, "green")
                wrapped_text("250 ml", (x, 0.99), z + 0.085, 0.0455, 0.008, "green")
                for rib in range(3):
                    cylinder("Cap moulded ring", (x, 0.99, z + 0.245 + rib * 0.007), 0.0195, 0.0015, "ivory")
            else:
                if i < 6:
                    continue  # Full-textured milk cartons occupy these bays.
                box("Cereal carton", (x, 1.01, z + 0.15), (0.133, 0.125, 0.27), "gold" if i % 2 else "blue", 0.002)
                box("Carton front label", (x, 0.945, z + 0.155), (0.11, 0.002, 0.19), "ivory", 0)
                text("OATS", (x - 0.049, 0.9438, z + 0.20), 0.032, "green")
                text("WHOLEGRAIN", (x - 0.048, 0.9438, z + 0.16), 0.010, "black")
                text("500 g", (x - 0.03, 0.9438, z + 0.08), 0.016, "black")
    box("Aisle sign", (0.50, 1.16, 1.72), (2.00, 0.05, 0.17), "green")
    text("GROCERY  /  EVERYDAY FRESH", (-0.33, 1.1348, 1.69), 0.075)

    # The demo uses the licensed scanned tile material for the store floor.

    groups = {}
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in list(bpy.context.scene.objects):
        if obj.type not in ("MESH", "FONT"):
            continue
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        key = obj.data.materials[0].name
        group = groups.setdefault(
            key,
            {
                "vertices": [],
                "indices": [],
                "normals": [],
                "color": list(obj.data.materials[0].diffuse_color[:3]),
                "roughness": obj.data.materials[0]["roughness"],
                "metallic": obj.data.materials[0]["metallic"],
            },
        )
        normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
        split_vertices = {}
        for triangle in mesh.loop_triangles:
            for loop_id in triangle.loops:
                vertex_id = mesh.loops[loop_id].vertex_index
                normal = (normal_matrix @ mesh.corner_normals[loop_id].vector).normalized()
                normal = tuple(round(x, 6) for x in normal)
                key = (vertex_id, normal)
                if key not in split_vertices:
                    split_vertices[key] = len(group["vertices"])
                    point = obj.matrix_world @ mesh.vertices[vertex_id].co
                    group["vertices"].append([round(x, 6) for x in point])
                    group["normals"].append(normal)
                group["indices"].append(split_vertices[key])
        evaluated.to_mesh_clear()
    output = Path(output_dir)
    (output / "checkout_decor.json").write_text(json.dumps(groups, separators=(",", ":")), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "checkout_decor.blend"), copy=True)
    print(f"Exported {len(groups)} material batches")

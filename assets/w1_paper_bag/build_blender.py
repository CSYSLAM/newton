# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Author the reference-video bag and snacks in a separate Blender scene."""

import json
import math
from itertools import pairwise
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix


def build(directory):
    """Export a dynamic paper shell and colored, detailed snack render meshes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scene = bpy.data.scenes.new("W1 video paper bag packing")
    bpy.context.window.scene = scene

    def material(name, color, roughness=0.7):
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = (*color, 1)
        mat.use_nodes = True
        node = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if node is None:
            node = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
            output = mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
            mat.node_tree.links.new(node.outputs["BSDF"], output.inputs["Surface"])
        node.inputs["Base Color"].default_value = (*color, 1)
        node.inputs["Roughness"].default_value = roughness
        return mat

    paper = material("W1 warm kraft paper", (0.49, 0.37, 0.20))
    ribbon = material("W1 folded kraft ribbon", (0.42, 0.32, 0.16))
    green = material("W1 green snack label", (0.20, 0.52, 0.045), 0.45)
    red = material("W1 red biscuit carton", (0.50, 0.035, 0.025), 0.5)
    cream = material("W1 cream lettering", (0.96, 0.88, 0.66))
    gold = material("W1 golden biscuit", (0.76, 0.45, 0.11))
    silver = material("W1 aluminum rim", (0.65, 0.68, 0.70), 0.25)

    def mesh_object(name, vertices, faces, mat):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        obj = bpy.data.objects.new(name, mesh)
        scene.collection.objects.link(obj)
        obj.data.materials.append(mat)
        return obj

    vertices, faces, lookup = [], [], {}
    nx, ny, nz = 8, 20, 18
    depth, width, height = 0.13, 0.32, 0.25

    def vertex(i, j, k):
        key = (i, j, k)
        if key not in lookup:
            x, y, z = depth * (i / nx - 0.5), width * (j / ny - 0.5), height * k / nz
            # Shallow inset gussets and an irregular lip, as in the video.
            if j in (0, ny):
                y -= math.copysign(0.016 * (1 - abs(2 * i / nx - 1)) * math.sin(math.pi * k / nz), y)
            if i in (0, nx):
                x += math.copysign(0.0015 * math.sin(6 * math.pi * j / ny + 0.5) * math.sin(math.pi * k / nz), x)
            # Pressed fold above the glued bottom; use a crease, not a rounded box edge.
            if i in (0, nx) and k == 2:
                x -= math.copysign(0.0025 * math.sin(math.pi * j / ny), x)
            lookup[key] = len(vertices)
            vertices.append((x, y, z))
        return lookup[key]

    def grid(rows, reverse=False):
        for a, b in pairwise(rows):
            for i in range(len(a) - 1):
                tri = [(a[i], a[i + 1], b[i + 1]), (a[i], b[i + 1], b[i])]
                faces.extend(t[::-1] if reverse else t for t in tri)

    grid([[vertex(i, j, 0) for j in range(ny + 1)] for i in range(nx + 1)])
    for i in (0, nx):
        grid([[vertex(i, j, k) for j in range(ny + 1)] for k in range(nz + 1)], i == 0)
    for j in (0, ny):
        grid([[vertex(i, j, k) for i in range(nx + 1)] for k in range(nz + 1)], j == ny)
    paper_count = len(vertices)
    roots = {}
    for i in (0, nx):
        for j in (5, 14):
            roots[i, j] = [vertex(i, j, 14), vertex(i, j + 1, 14), vertex(i, j + 1, 15), vertex(i, j, 15)]
    faces = [face for face in faces if not any(set(face) <= set(root) for root in roots.values())]
    paper_faces = len(faces)
    handles = []
    for i in (0, nx):
        rows = [roots[i, 5]]
        for k in range(1, 24):
            t = math.pi * k / 24
            y = -0.072 * math.cos(t)
            z = height * 14.5 / nz - 0.072 * math.sin(t)
            tangent = np.array((0.072 * math.sin(t), -0.072 * math.cos(t)))
            normal = np.array((-tangent[1], tangent[0])) / np.linalg.norm(tangent)
            row = []
            for thick, wide in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                row.append(len(vertices))
                x = (depth / 2 + 0.004 + 0.008 * math.sin(t)) * (-1 if i == 0 else 1)
                vertices.append((x + thick * 0.0013, y + wide * 0.007 * normal[0], z + wide * 0.007 * normal[1]))
            rows.append(row)
        rows.append(roots[i, 14][::-1])
        for a, b in pairwise(rows):
            for j in range(4):
                n = (j + 1) % 4
                faces.extend(((a[j], a[n], b[n]), (a[j], b[n], b[j])))
        handles.append(np.array(rows).ravel())
    obj = mesh_object("Creased kraft bag with folded-down paper handles", vertices, faces, paper)
    obj.data.materials.append(ribbon)
    for polygon in obj.data.polygons[paper_faces:]:
        polygon.material_index = 1
    solidify = obj.modifiers.new("Paper thickness 0.35 mm", "SOLIDIFY")
    solidify.thickness = 0.00035
    solidify.offset = 0
    rest = np.asarray(vertices, dtype=np.float32)
    np.savez_compressed(
        directory / "bag.npz",
        vertices=rest,
        faces=np.asarray(faces, dtype=np.int32),
        paper_count=paper_count,
        paper_faces=paper_faces,
        bottom=np.flatnonzero(rest[:paper_count, 2] < 1e-6),
        rim=np.flatnonzero(rest[:paper_count, 2] > height - 1e-6),
        handle_0=handles[0],
        handle_1=handles[1],
    )
    uv = obj.data.uv_layers.new(name="Kraft grain")
    for loop in obj.data.loops:
        v = obj.data.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = ((v.x + v.y + 0.24) * 2.5, v.z * 3)
    rng = np.random.default_rng(27)
    grain = np.clip(rng.normal(1, 0.045, (512, 512, 1)), 0.82, 1.18)
    pixels = np.ones((512, 512, 4), dtype=np.float32)
    pixels[:, :, :3] = np.array((0.60, 0.47, 0.29)) * grain
    img = bpy.data.images.new("Video bag kraft grain", width=512, height=512)
    img.pixels.foreach_set(pixels.ravel())
    img.filepath_raw = str(directory / "kraft.png")
    img.file_format = "PNG"
    img.save()
    img.pack()
    texture = paper.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = img
    paper.node_tree.links.new(
        texture.outputs["Color"],
        next(n for n in paper.node_tree.nodes if n.type == "BSDF_PRINCIPLED").inputs["Base Color"],
    )

    snack_parts = {"can": [], "carton": []}

    def cylinder(name, radius, length, z, mat):
        bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=radius, depth=length, location=(0, 0, z))
        item = bpy.context.object
        item.name = name
        item.data.materials.append(mat)
        return item

    def cube(name, half, position, mat, bevel=0.0):
        bpy.ops.mesh.primitive_cube_add(size=2, location=position)
        item = bpy.context.object
        item.name = name
        item.scale = half
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        item.data.materials.append(mat)
        if bevel:
            modifier = item.modifiers.new("Folded carton edges", "BEVEL")
            modifier.width, modifier.segments = bevel, 2
        return item

    def text(name, label, z, size, front, mat):
        curve = bpy.data.curves.new(name, "FONT")
        curve.body, curve.size, curve.align_x = label, size, "CENTER"
        curve.extrude = 0.00012
        item = bpy.data.objects.new(name, curve)
        scene.collection.objects.link(item)
        item.location = (front, 0, z)
        # Local label X -> world -Y; local Y -> world Z; face -> -X.
        item.rotation_euler = Matrix(((0, 0, -1), (-1, 0, 0), (0, 1, 0))).to_euler()
        curve.materials.append(mat)
        return item

    snack_parts["can"].append(cylinder("Green stacked-crisp can", 0.032, 0.112, 0, green))
    for z in (-0.057, 0.057):
        snack_parts["can"].append(cylinder("Rolled aluminum rim", 0.0325, 0.002, z, silver))
    snack_parts["can"].append(cylinder("Can foil lid", 0.031, 0.001, 0.058, silver))
    for label, z, size in (("CRISP", 0.018, 0.012), ("POTATO", 0.003, 0.008), ("ORIGINAL", -0.039, 0.006)):
        snack_parts["can"].append(text(label, label, z, size, -0.0325, cream))
    for z in (-0.014, -0.021, -0.028):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, location=(-0.0325, 0, z))
        chip = bpy.context.object
        chip.name = "Golden potato crisp illustration"
        chip.scale = (0.001, 0.015, 0.006)
        chip.data.materials.append(gold)
        snack_parts["can"].append(chip)
    snack_parts["carton"].append(cube("Red biscuit carton", (0.024, 0.030, 0.059), (0, 0, 0), red, 0.001))
    for z in (-0.050, 0.051):
        snack_parts["carton"].append(cube("Carton cream border", (0.0003, 0.028, 0.002), (-0.0245, 0, z), cream))
    for label, z, size in (("BISCUITS", 0.030, 0.010), ("COCOA", 0.014, 0.009), ("CRUNCH", -0.043, 0.007)):
        snack_parts["carton"].append(text(label, label, z, size, -0.025, cream))
    for y, z in ((-0.010, -0.010), (0.009, -0.025)):
        snack_parts["carton"].append(cube("Biscuit illustration", (0.0006, 0.014, 0.010), (-0.025, y, z), gold, 0.002))

    exports, info = {}, {}
    graph = bpy.context.evaluated_depsgraph_get()
    for kind, objects in snack_parts.items():
        info[kind] = []
        for index, item in enumerate(objects):
            evaluated = item.evaluated_get(graph)
            mesh = evaluated.to_mesh()
            mesh.calc_loop_triangles()
            key = f"{kind}_{index}"
            exports[key + "_vertices"] = np.asarray([item.matrix_world @ v.co for v in mesh.vertices], dtype=np.float32)
            exports[key + "_faces"] = np.asarray([p.vertices[:] for p in mesh.loop_triangles], dtype=np.int32)
            info[kind].append({"key": key, "color": list(item.data.materials[0].diffuse_color[:3])})
            evaluated.to_mesh_clear()
        for item in objects:
            item.location.x += 0.22 if kind == "can" else 0.34
            item.location.z += 0.06
    np.savez_compressed(directory / "snacks.npz", **exports)
    (directory / "snacks.json").write_text(json.dumps(info, indent=2) + "\n")
    (directory / "dimensions.json").write_text(
        json.dumps(
            {
                "bag_depth_width_height_m": [depth, width, height],
                "can_radius_height_m": [0.0325, 0.117],
                "carton_size_m": [0.048, 0.060, 0.118],
                "reference": "User video Feishu 20260916-115427, estimated scale",
            },
            indent=2,
        )
        + "\n"
    )
    bpy.data.libraries.write(str(directory / "w1_bag_and_snacks.blend"), {scene}, fake_user=True, compress=True)
    print(f"W1 bag exported: {len(vertices)} vertices, {len(faces)} triangles; two snacks")


if __name__ == "__main__":
    build(Path(__file__).resolve().parent)

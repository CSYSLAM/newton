# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the W1 workstation in Blender and export material-batched meshes.

Run in Blender's Python console with ``exec(compile(open(path).read(), path, 'exec'))``.
Only the named collection is replaced. The Newton demo loads the exported NPZ;
Blender is an authoring tool and is not required to run the simulation.
"""

import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parent
COLLECTION = "W1 Sorting Workstation"
old = bpy.data.collections.get(COLLECTION)
if old:
    for obj in list(old.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(old)
collection = bpy.data.collections.new(COLLECTION)
bpy.context.scene.collection.children.link(collection)
MATERIALS = {
    "aluminum": ((0.55, 0.60, 0.64), 0.32, 0.8),
    "stainless": ((0.46, 0.51, 0.54), 0.28, 0.75),
    "paint": ((0.19, 0.25, 0.28), 0.48, 0.2),
    "rubber": ((0.035, 0.045, 0.050), 0.88, 0.0),
    "dark": ((0.075, 0.09, 0.105), 0.48, 0.1),
    "ivory": ((0.71, 0.73, 0.72), 0.6, 0.0),
    "yellow": ((0.81, 0.56, 0.10), 0.55, 0.0),
    "red": ((0.63, 0.055, 0.035), 0.38, 0.0),
    "green": ((0.12, 0.56, 0.29), 0.3, 0.0),
    "screen": ((0.045, 0.20, 0.26), 0.26, 0.15),
    "wall": ((0.47, 0.50, 0.52), 0.9, 0.0),
    "tray0": ((0.51, 0.34, 0.17), 0.68, 0.0),
    "tray1": ((0.22, 0.39, 0.30), 0.68, 0.0),
    "tray2": ((0.24, 0.36, 0.46), 0.68, 0.0),
    "tray3": ((0.39, 0.28, 0.39), 0.68, 0.0),
}
for name, (color, rough, metal) in MATERIALS.items():
    mat = bpy.data.materials.get("W1_" + name) or bpy.data.materials.new("W1_" + name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    node = next(node for node in mat.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    node.inputs["Base Color"].default_value = (*color, 1)
    node.inputs["Roughness"].default_value = rough
    node.inputs["Metallic"].default_value = metal


def finish(obj, name, material, bevel=0):
    obj.name = name
    for group in list(obj.users_collection):
        group.objects.unlink(obj)
    collection.objects.link(obj)
    obj.data.materials.append(bpy.data.materials["W1_" + material])
    if bevel:
        modifier = obj.modifiers.new("Manufactured edge radius", "BEVEL")
        modifier.width = bevel
        modifier.segments = 2
    if obj.type == "MESH":
        modifier = obj.modifiers.new("Face weighted normals", "WEIGHTED_NORMAL")
        modifier.keep_sharp = True
    return obj


def box(name, center, size, material, bevel=0.003):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return finish(obj, name, material, bevel)


def cylinder(name, center, radius, length, material, axis=(0, 0, 1), vertices=24):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=length, location=center)
    obj = bpy.context.object
    obj.rotation_euler = Vector(axis).to_track_quat("Z", "Y").to_euler()
    for face in obj.data.polygons:
        face.use_smooth = len(face.vertices) == 4
    return finish(obj, name, material, min(0.0015, length / 5))


def cable(name, points, radius, material):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 8
    curve.bevel_depth = radius
    curve.bevel_resolution = 2
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for node, position in zip(spline.bezier_points, points, strict=True):
        node.co = position
        node.handle_left_type = node.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    collection.objects.link(obj)
    obj.data.materials.append(bpy.data.materials["W1_" + material])
    return obj


def rounded_ring(hx, hy, radius, z):
    points = []
    for x, y, start in (
        (hx - radius, hy - radius, 0),
        (-hx + radius, hy - radius, 90),
        (-hx + radius, -hy + radius, 180),
        (hx - radius, -hy + radius, 270),
    ):
        for degrees in np.linspace(start, start + 90, 9):
            a = math.radians(degrees)
            points.append((x + radius * math.cos(a), y + radius * math.sin(a), z))
    return points


def tub(index, x, y, hx, hy):
    # Match the simulation's floor and straight inner walls; round only corners.
    rings = [
        rounded_ring(hx - 0.006, hy - 0.006, 0.014, 0.635),
        rounded_ring(hx, hy, 0.014, 0.724),
        rounded_ring(hx, hy, 0.014, 0.730),
        rounded_ring(hx - 0.020, hy - 0.020, 0.006, 0.730),
        rounded_ring(hx - 0.020, hy - 0.020, 0.006, 0.685),
    ]
    n = len(rings[0])
    vertices = [(px + x, py + y, z) for ring in rings for px, py, z in ring]
    faces = []
    for level in range(4):
        for i in range(n):
            j = (i + 1) % n
            faces.append((level * n + i, level * n + j, (level + 1) * n + j, (level + 1) * n + i))
    faces.extend([tuple(reversed(range(n))), tuple(range(4 * n, 5 * n))])
    mesh = bpy.data.meshes.new(f"Molded tray {index}")
    mesh.from_pydata(vertices, [], faces)
    obj = bpy.data.objects.new(mesh.name, mesh)
    collection.objects.link(obj)
    finish(obj, mesh.name, f"tray{index}", 0.001)
    # Reinforcing ribs and a molded front label recess.
    for xx in np.linspace(x - hx + 0.035, x + hx - 0.035, max(2, int(hx * 25))):
        box("Tray reinforcement rib", (xx, y + hy - 0.002, 0.68), (0.006, 0.006, 0.066), f"tray{index}", 0.002)
    box("Recessed label holder", (x, y - hy - 0.001, 0.704), (min(2 * hx - 0.026, 0.198), 0.004, 0.052), "dark", 0.005)
    for sign in (-1, 1):
        cable(
            "Molded lifting grip",
            [
                (x + sign * (hx + 0.002), y - 0.035, 0.704),
                (x + sign * (hx + 0.008), y - 0.029, 0.719),
                (x + sign * (hx + 0.008), y + 0.029, 0.719),
                (x + sign * (hx + 0.002), y + 0.035, 0.704),
            ],
            0.004,
            f"tray{index}",
        )


# Aluminum side rails with dark T-slots, end plates, and recessed fasteners.
for x in (0.425, 0.775):
    box("Anodized conveyor extrusion", (x, 0.76, 0.70), (0.05, 2.08, 0.16), "aluminum", 0.006)
    outward = -1 if x < 0.6 else 1
    for z in (0.651, 0.699, 0.749):
        box("Recessed extrusion channel", (x + outward * 0.0251, 0.76, z), (0.0008, 2.025, 0.004), "dark", 0.0003)
    for y in (-0.28, 1.8):
        box("Machined rail end cap", (x, y, 0.70), (0.048, 0.004, 0.156), "paint", 0.004)
    for y in (-0.14, 0.20, 1.28, 1.66):
        for z in (0.638, 0.765):
            cylinder("Socket head fastener", (x + outward * 0.027, y, z), 0.005, 0.003, "stainless", (1, 0, 0), 12)
            cylinder("Hex socket", (x + outward * 0.029, y, z), 0.0022, 0.001, "dark", (1, 0, 0), 6)
    for y in (-0.14, 1.66):
        box("Conveyor upright", (x, y, 0.338), (0.045, 0.065, 0.615), "aluminum", 0.004)
        box("Upright slot", (x + outward * 0.023, y, 0.33), (0.001, 0.005, 0.51), "dark", 0.0003)
        box("Gusset mounting plate", (x + outward * 0.026, y, 0.59), (0.004, 0.095, 0.11), "paint", 0.008)
        cylinder("Leveling spindle", (x, y, 0.034), 0.010, 0.06, "stainless")
        cylinder("Rubber leveling foot", (x, y, 0.012), 0.044, 0.024, "rubber")
    box("Lower conveyor brace", (x, 0.76, 0.24), (0.04, 1.8, 0.045), "paint", 0.004)
for y in (-0.14, 1.66):
    box("Cross member", (0.6, y, 0.30), (0.34, 0.045, 0.045), "paint", 0.004)
for y in (-0.24, 1.76):
    for x in (0.413, 0.787):
        box("Bearing pillow block", (x, y, 0.708), (0.019, 0.087, 0.076), "paint", 0.012)
        cylinder("Bearing race", (x, y, 0.708), 0.024, 0.026, "stainless", (1, 0, 0))
# Screw take-ups allow the front drum to be tensioned without moving the frame.
for x in (0.413, 0.787):
    box("Take-up bearing slide", (x, -0.185, 0.707), (0.022, 0.156, 0.071), "paint", 0.005)
    box(
        "Take-up adjustment slot",
        (x + (-0.012 if x < 0.6 else 0.012), -0.16, 0.707),
        (0.001, 0.086, 0.012),
        "dark",
        0.003,
    )
    cylinder(
        "Tensioning screw", (x + (-0.022 if x < 0.6 else 0.022), -0.080, 0.678), 0.004, 0.096, "stainless", (0, 1, 0)
    )
    for y in (-0.050, -0.038):
        cylinder(
            "Take-up lock nut", (x + (-0.022 if x < 0.6 else 0.022), y, 0.678), 0.009, 0.006, "stainless", (0, 1, 0), 6
        )
    box(
        "Take-up support tab",
        (x + (-0.012 if x < 0.6 else 0.012), -0.044, 0.691),
        (0.040, 0.007, 0.045),
        "aluminum",
        0.002,
    )
    for y in (-0.25, 1.80):
        box("Folded end guard", (x, y, 0.690), (0.029, 0.050, 0.103), "paint", 0.009)
# A sheet-metal catch pan protects the lower return run while leaving the ends visible.
box("Return-run guard pan", (0.6, 0.76, 0.626), (0.332, 1.76, 0.003), "paint", 0.001)
for x in (0.435, 0.765):
    box("Folded pan flange", (x, 0.76, 0.636), (0.003, 1.76, 0.022), "paint", 0.001)
for y in (-0.065, 1.585):
    box("Pan end fold", (0.6, y, 0.636), (0.326, 0.003, 0.022), "paint", 0.001)
# Gearmotor, cooling fins, gearbox, flexible conduits and control box.
box("Cast gearbox", (0.839, 1.76, 0.708), (0.084, 0.112, 0.102), "paint", 0.016)
cylinder("Motor housing", (0.941, 1.76, 0.708), 0.052, 0.14, "paint", (1, 0, 0), 48)
for x in np.linspace(0.895, 0.994, 10):
    cylinder("Motor cooling fin", (x, 1.76, 0.708), 0.058, 0.003, "aluminum", (1, 0, 0))
cylinder("Motor end cover", (1.019, 1.76, 0.708), 0.052, 0.017, "dark", (1, 0, 0))
box("Electrical cabinet", (0.874, 0.64, 0.46), (0.13, 0.36, 0.30), "ivory", 0.012)
box("Cabinet door seam", (0.941, 0.64, 0.46), (0.003, 0.338, 0.276), "dark", 0.008)
box("Cabinet door", (0.944, 0.64, 0.46), (0.005, 0.327, 0.267), "ivory", 0.007)
cylinder("Door lock", (0.950, 0.53, 0.47), 0.009, 0.008, "stainless", (1, 0, 0))
for y in np.linspace(0.60, 0.73, 9):
    box("Cabinet louver", (0.948, y, 0.38), (0.004, 0.005, 0.027), "dark", 0.001)
cable(
    "Motor power conduit",
    [(0.947, 1.76, 0.770), (0.965, 1.50, 0.79), (0.87, 1.30, 0.60), (0.865, 0.86, 0.57)],
    0.009,
    "rubber",
)
cable(
    "Control conduit",
    [(0.87, 0.64, 0.305), (0.86, 0.59, 0.22), (0.80, 0.22, 0.20), (0.79, 0.11, 0.61)],
    0.007,
    "rubber",
)
box("Operator pendant", (0.85, 0.095, 0.676), (0.087, 0.17, 0.083), "paint", 0.012)
cylinder("Emergency stop bezel", (0.85, 0.05, 0.723), 0.023, 0.008, "yellow")
cylinder("Emergency stop mushroom", (0.85, 0.05, 0.736), 0.017, 0.022, "red")
cylinder("Start button", (0.85, 0.14, 0.723), 0.010, 0.010, "green")
for x in (0.42, 0.78):
    box("Photoelectric sensor", (x, -0.12, 0.799), (0.022, 0.046, 0.028), "dark", 0.004)
    cylinder("Sensor lens", (x, -0.145, 0.802), 0.006, 0.003, "red", (0, 1, 0))
# HMI display and stack light are outside the robot's manipulation envelope.
cylinder("HMI support", (0.90, 1.13, 0.93), 0.018, 0.44, "stainless")
box("HMI rear housing", (0.90, 1.12, 1.16), (0.23, 0.055, 0.155), "paint", 0.012)
box("HMI glass bezel", (0.90, 1.089, 1.16), (0.209, 0.008, 0.136), "dark", 0.009)
box("HMI screen", (0.90, 1.083, 1.16), (0.185, 0.002, 0.109), "screen", 0.004)
for z, w in ((1.19, 0.12), (1.17, 0.08), (1.15, 0.14)):
    box("HMI process indicator", (0.882, 1.081, z), (w, 0.001, 0.005), "ivory", 0.001)
cylinder("Signal mast", (0.79, 1.64, 0.99), 0.010, 0.48, "stainless")
for z, mat in ((1.20, "green"), (1.244, "yellow"), (1.288, "red")):
    cylinder("Stack light lens", (0.79, 1.64, z), 0.024, 0.038, mat)
    cylinder("Stack light divider", (0.79, 1.64, z + 0.021), 0.025, 0.004, "dark")
# Folded stainless bench top, underside apron, hollow square legs and feet.
outline = [(-0.13, -0.84), (0.8, -0.84), (0.8, -0.40), (0.45, -0.40), (0.45, -0.20), (-0.13, -0.20)]
vertices = [(x, y, z) for z in (0.605, 0.635) for x, y in outline]
faces = [tuple(reversed(range(6))), tuple(range(6, 12))]
faces.extend((i, (i + 1) % 6, (i + 1) % 6 + 6, i + 6) for i in range(6))
mesh = bpy.data.meshes.new("Notched stainless worktop")
mesh.from_pydata(vertices, [], faces)
obj = bpy.data.objects.new(mesh.name, mesh)
collection.objects.link(obj)
finish(obj, mesh.name, "stainless", 0.006)
box("Workbench front apron", (0.335, -0.817, 0.582), (0.88, 0.018, 0.07), "paint", 0.004)
box("Workbench rear left apron", (0.16, -0.223, 0.582), (0.54, 0.018, 0.07), "paint", 0.004)
box("Workbench pickup apron", (0.625, -0.423, 0.582), (0.30, 0.018, 0.07), "paint", 0.004)
box("Workbench notch return", (0.427, -0.312, 0.582), (0.018, 0.20, 0.07), "paint", 0.004)
for x in (-0.105, 0.775):
    rear = -0.223 if x < 0.45 else -0.423
    box("Workbench end apron", (x, (-0.817 + rear) / 2, 0.582), (0.018, rear + 0.817, 0.07), "paint", 0.004)
for x in (-0.045, 0.715):
    rear = -0.255 if x < 0.45 else -0.45
    for y in (-0.785, rear):
        box("Bench tubular leg", (x, y, 0.315), (0.044, 0.044, 0.574), "stainless", 0.004)
        cylinder("Bench leveling screw", (x, y, 0.037), 0.010, 0.045, "stainless")
        cylinder("Bench rubber foot", (x, y, 0.013), 0.035, 0.026, "rubber")
    box("Bench end stretcher", (x, (-0.785 + rear) / 2, 0.21), (0.036, rear + 0.785, 0.036), "stainless", 0.003)
box("Bench rear stretcher", (0.335, -0.45, 0.21), (0.76, 0.035, 0.035), "stainless", 0.003)
for i, (x, y, hx, hy) in enumerate(
    ((0.65, -0.60, 0.125, 0.125), (0.28, -0.34, 0.095, 0.125), (0.04, -0.48, 0.125, 0.20), (0.34, -0.60, 0.125, 0.125))
):
    tub(i, x, y, hx, hy)
# Industrial bay: architectural scale, panel seams and skirting, kept behind the cell.
box("Factory rear wall", (0.2, 2.8, 1.55), (5.0, 0.10, 3.10), "wall", 0.003)
box("Wall impact protection", (0.2, 2.738, 0.38), (5.0, 0.025, 0.46), "paint", 0.005)
for x in np.arange(-2.2, 2.7, 0.6):
    box("Wall panel seam", (x, 2.747, 1.86), (0.006, 0.004, 2.35), "dark", 0.001)
box("Wall skirting", (0.2, 2.724, 0.075), (5.0, 0.04, 0.15), "aluminum", 0.004)
for y in (-1.15, 2.02):
    box("Cell floor marking", (0.30, y, 0.0015), (2.16, 0.026, 0.002), "yellow", 0.001)
for x in (-0.78, 1.38):
    box("Cell floor marking", (x, 0.435, 0.0015), (0.026, 3.17, 0.002), "yellow", 0.001)

# The belt is a closed 4 mm laminate with continuous longitudinal UVs. Its
# geometry stays fixed in Newton; material coordinates circulate with belt travel.
belt_radius = 0.0521
belt_thickness = 0.004
belt_length = 4.0 + 2.0 * math.pi * belt_radius
path_samples = [(1.76, 0.0, 0.0), (-0.24, 0.0, 2.0)]
for angle in np.linspace(0.0, math.pi, 49)[1:]:
    path_samples.append((-0.24, angle, 2.0 + belt_radius * angle))
path_samples.append((1.76, math.pi, 4.0 + math.pi * belt_radius))
for angle in np.linspace(math.pi, 2.0 * math.pi, 49)[1:]:
    path_samples.append((1.76, angle, 4.0 + belt_radius * angle))
vertices, normals, vertex_uvs, faces = [], [], [], []
for strip in range(4):
    start = len(vertices)
    for center_y, angle, distance in path_samples:
        # Outer face, inner face, and the two cut edges have separate normals.
        for edge in range(2):
            if strip < 2:
                x = 0.45 + 0.30 * edge
                radius = belt_radius - strip * belt_thickness
                n = (0.0, -math.sin(angle), math.cos(angle))
                if strip == 1:
                    n = tuple(-value for value in n)
                u = float(edge)
            else:
                x = 0.45 if strip == 2 else 0.75
                radius = belt_radius - edge * belt_thickness
                n = (-1.0, 0.0, 0.0) if strip == 2 else (1.0, 0.0, 0.0)
                u = 0.005 if strip == 2 else 0.995
            vertices.append((x, center_y - radius * math.sin(angle), 0.708 + radius * math.cos(angle)))
            normals.append(n)
            vertex_uvs.append((u, distance / belt_length))
    for segment in range(len(path_samples) - 1):
        a = start + 2 * segment
        face = (a, a + 2, a + 3, a + 1)
        faces.append(tuple(reversed(face)) if strip in (1, 2) else face)
mesh = bpy.data.meshes.new("Continuous rubber belt")
mesh.from_pydata(vertices, [], faces)
mesh.update()
for polygon in mesh.polygons:
    polygon.use_smooth = True
mesh.normals_split_custom_set_from_vertices(normals)
uv_layer = mesh.uv_layers.new(name="Belt travel")
for loop in mesh.loops:
    uv_layer.data[loop.index].uv = vertex_uvs[loop.vertex_index]
belt = bpy.data.objects.new(mesh.name, mesh)
collection.objects.link(belt)
mesh.materials.append(bpy.data.materials["W1_rubber"])
belt["moving_part"] = "belt"

# Rotating drum, hubs and three recessed bolt heads; exported around the X axis.
roller_parts = [
    cylinder("Roller barrel", (0, 0, 0), belt_radius - belt_thickness - 0.0003, 0.298, "rubber", (1, 0, 0), 64),
    cylinder("Roller shaft", (0, 0, 0), 0.010, 0.410, "stainless", (1, 0, 0)),
]
for sign in (-1, 1):
    roller_parts.append(cylinder("Rotating hub", (sign * 0.201, 0, 0), 0.020, 0.006, "stainless", (1, 0, 0), 32))
    for angle in np.linspace(0, 2 * math.pi, 3, endpoint=False):
        roller_parts.append(
            cylinder(
                "Hub bolt recess",
                (sign * 0.205, 0.013 * math.cos(angle), 0.013 * math.sin(angle)),
                0.0025,
                0.001,
                "dark",
                (1, 0, 0),
                6,
            )
        )
for obj in roller_parts:
    obj["moving_part"] = "roller"
    obj.hide_set(True)
    obj.hide_render = True
    for y in (-0.24, 1.76):
        preview = obj.copy()
        collection.objects.link(preview)
        preview["moving_part"] = "preview"
        preview.location += Vector((0.6, y, 0.708))
        preview.hide_set(False)
        preview.hide_render = False

# Export evaluated geometry and UV seams, merging objects by material.
bpy.context.view_layer.update()
depsgraph = bpy.context.evaluated_depsgraph_get()


def export_meshes(filename, objects, **metadata):
    arrays = dict(metadata)
    triangle_count = 0
    for material_index, material in enumerate(MATERIALS):
        positions, normals, triangles, uvs = [], [], [], []
        for obj in objects:
            if not obj.data.materials or obj.data.materials[0].name != "W1_" + material:
                continue
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            mesh.calc_loop_triangles()
            transform = obj.matrix_world
            normal_transform = transform.to_3x3().inverted().transposed()
            lookup = {}
            for tri in mesh.loop_triangles:
                for loop_index in tri.loops:
                    loop = mesh.loops[loop_index]
                    p = transform @ mesh.vertices[loop.vertex_index].co
                    n = (normal_transform @ mesh.corner_normals[loop_index].vector).normalized()
                    uv = tuple(mesh.uv_layers.active.data[loop_index].uv) if mesh.uv_layers.active else (0.0, 0.0)
                    key = tuple(round(v, 6) for v in (*p, *n, *uv))
                    if key not in lookup:
                        lookup[key] = len(positions)
                        positions.append(tuple(p))
                        normals.append(tuple(n))
                        uvs.append(uv)
                    triangles.append(lookup[key])
            evaluated.to_mesh_clear()
        arrays[f"vertices_{material_index}"] = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        arrays[f"normals_{material_index}"] = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
        arrays[f"indices_{material_index}"] = np.asarray(triangles, dtype=np.int32)
        arrays[f"uvs_{material_index}"] = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
        triangle_count += len(triangles) // 3
    arrays["materials"] = np.asarray(
        [(*color, rough, metal) for color, rough, metal in MATERIALS.values()], dtype=np.float32
    )
    np.savez_compressed(ROOT / filename, **arrays)
    print(f"Exported {len(objects)} objects, {triangle_count} triangles to {filename}")


export_meshes("station.npz", [obj for obj in collection.objects if not obj.get("moving_part")])
export_meshes("belt.npz", [belt], path_length=np.float64(belt_length), radius=np.float64(belt_radius))
export_meshes("roller.npz", roller_parts)

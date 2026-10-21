# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Author the thin clear measuring jug and calibrated scale in Blender."""

import math

import bpy
import numpy as np
from mathutils import Vector


def build_measuring_cup(collection, geometry):
    """Build a readable kitchen jug with a rolled lip and continuous soft grip."""

    def material(name, color, roughness, opacity=1.0, metallic=0.0):
        mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
        mat.use_nodes = True
        shader = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        shader.inputs["Base Color"].default_value = (*color, 1)
        shader.inputs["Roughness"].default_value = roughness
        shader.inputs["Metallic"].default_value = metallic
        shader.inputs["Transmission Weight"].default_value = 1.0 if opacity < 1 else 0
        shader.inputs["IOR"].default_value = 1.49
        mat["newton_opacity"] = opacity
        return mat

    clear = material("Newton jug - clear teal copolyester", (0.91, 0.965, 0.955), 0.06, 0.22)
    clear["newton_color"] = (0.45, 0.66, 0.64)
    clear_handle = material("Newton jug - clear molded handle", (0.93, 0.97, 0.96), 0.045, 0.42)
    clear_handle["newton_color"] = (0.55, 0.70, 0.67)
    edge = material("Newton jug - polished clear edge", (0.97, 0.99, 0.99), 0.045, 0.82)
    edge["newton_color"] = (0.55, 0.72, 0.70)
    rubber = material("Newton jug - dark teal comfort grip", (0.025, 0.065, 0.058), 0.38)
    ribs = material("Newton jug - molded grip ribs", (0.08, 0.15, 0.135), 0.48)
    red = material("Newton jug - vermilion printed scale", (0.54, 0.025, 0.012), 0.48)

    def mesh(name, vertices, faces, mat, smooth=True):
        data = bpy.data.meshes.new(name)
        data.from_pydata(vertices, [], faces)
        data.update()
        obj = bpy.data.objects.new(name, data)
        obj["newton_asset_name"] = name
        collection.objects.link(obj)
        data.materials.append(mat)
        for face in data.polygons:
            face.use_smooth = smooth
        return obj

    def curve(name, points, radius, mat, cyclic=False):
        data = bpy.data.curves.new(name, "CURVE")
        data.dimensions = "3D"
        data.bevel_depth = radius
        data.bevel_resolution = 3
        spline = data.splines.new("POLY")
        spline.points.add(len(points) - 1)
        for point, co in zip(spline.points, points, strict=True):
            point.co = (*co, 1)
        spline.use_cyclic_u = cyclic
        obj = bpy.data.objects.new(name, data)
        obj["newton_asset_name"] = name
        collection.objects.link(obj)
        data.materials.append(mat)
        return obj

    vertices, faces = geometry["cup_visual_mesh"]()
    shell = mesh("Newton pitcher shell", vertices.tolist(), faces.tolist(), clear)
    # Preserve the optical normals of the walls: smoothing across the lip or
    # floor bends them toward the cap and makes thin glass look opaque.
    segments = 96
    for edge_data in shell.data.edges:
        a, b = edge_data.vertices
        if 0 < a < len(vertices) - 1 and 0 < b < len(vertices) - 1 and (a - 1) // segments == (b - 1) // segments:
            edge_data.use_edge_sharp = True
    shell.data.update()

    # A bright, narrow rolled edge gives clear plastic a visible silhouette in
    # raster viewers; the flared spout follows the physical opening exactly.
    lip = []
    for i in range(192):
        angle = 2 * math.pi * i / 192
        distance = math.atan2(math.sin(angle - math.pi / 2), math.cos(angle - math.pi / 2))
        spout = math.exp(-((distance / 0.24) ** 2))
        radius = geometry["CUP_RADIUS"] - geometry["CUP_WALL"] + 0.001 + 0.006 * spout
        lip.append((radius * math.cos(angle), radius * math.sin(angle), geometry["CUP_HEIGHT"] - 0.003 * spout))
    curve("Pitcher rolled pouring lip", lip, 0.00085, edge, True)

    # One swept D-shaped grip, with generous fillets rather than separate blocks.
    controls = [
        (-0.024, 0.068),
        (-0.047, 0.068),
        (-0.071, 0.063),
        (-0.081, 0.057),
        (-0.082, 0.050),
        (-0.081, 0.043),
        (-0.071, 0.037),
        (-0.047, 0.032),
        (-0.024, 0.032),
    ]
    controls = [Vector((x, 0, z)) for x, z in controls]
    extended = [controls[0], *controls, controls[-1]]
    path = []
    for i in range(len(controls) - 1):
        a, b, c, d = extended[i : i + 4]
        for t in np.linspace(0, 1, 10, endpoint=False):
            path.append(
                0.5
                * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (-a + 3 * b - 3 * c + d) * t * t * t)
            )
    path.append(controls[-1])
    verts, quads = [], []
    sides = 16
    for i, center in enumerate(path):
        tangent = (path[min(i + 1, len(path) - 1)] - path[max(0, i - 1)]).normalized()
        normal = Vector((tangent.z, 0, -tangent.x))
        for j in range(sides):
            angle = 2 * math.pi * j / sides
            verts.append(tuple(center + normal * (0.003 * math.sin(angle)) + Vector((0, 0.006 * math.cos(angle), 0))))
    for i in range(len(path) - 1):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            quads.append((a, b, b + sides, a + sides))
    quads.extend((tuple(reversed(range(sides))), tuple(range(len(verts) - sides, len(verts)))))
    mesh("Pitcher continuous ergonomic D handle", verts, quads, clear_handle)
    for side in (-1, 1):
        curve(
            "Pitcher polished handle seam",
            [(p.x, side * 0.0048, p.z + 0.0018) for p in path],
            0.00035,
            edge,
        )
    bpy.ops.mesh.primitive_cube_add(size=1, location=(-0.082, 0, 0.050))
    pad = bpy.context.object
    pad.name = "Pitcher small rear silicone grip"
    pad.scale = (0.0065, 0.012, 0.016)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    pad.data.materials.append(rubber)
    bevel = pad.modifiers.new("Rounded comfort pad", "BEVEL")
    bevel.width = 0.0025
    bevel.segments = 5
    pad.modifiers.new("Pad corner normals", "WEIGHTED_NORMAL")
    for col in list(pad.users_collection):
        col.objects.unlink(pad)
    collection.objects.link(pad)
    for i in range(5):
        z = 0.045 + i * 0.0025
        curve("Pitcher comfort grip rib", [(-0.085, -0.005, z), (-0.0856, 0, z), (-0.085, 0.005, z)], 0.0003, ribs)

    # A protective silicone foot also defines the base against a bright worktop.
    radius = geometry["CUP_BOTTOM_RADIUS"] - geometry["CUP_WALL"] + 0.0028
    profile = [
        (0, 0.0003),
        (radius - 0.001, 0.0003),
        (radius, 0.0015),
        (radius, 0.0055),
        (radius - 0.001, 0.0065),
        (0, 0.0065),
    ]
    vs, fs = [], []
    for r, z in profile:
        for i in range(96):
            angle = 2 * math.pi * i / 96
            vs.append((r * math.cos(angle), r * math.sin(angle), z))
    for k in range(len(profile) - 1):
        for i in range(96):
            a = k * 96 + i
            b = k * 96 + (i + 1) % 96
            fs.append((a, b, b + 96, a + 96))
    mesh("Pitcher protective silicone base", vs, fs, rubber)
    curve(
        "Pitcher clear base bead",
        [
            (radius * math.cos(2 * math.pi * i / 128), radius * math.sin(2 * math.pi * i / 128), 0.009)
            for i in range(128)
        ],
        0.0006,
        edge,
        True,
    )

    # Calibrate in millilitres, then wrap the printed markings onto the wall.
    heights = np.linspace(geometry["CUP_WALL"], geometry["CUP_HEIGHT"], 1500)
    volume = np.cumsum(math.pi * geometry["cup_inner_radius"](heights) ** 2) * (heights[1] - heights[0])
    print_vertices, print_faces = [], []

    def surface(x, z):
        r = float(geometry["cup_inner_radius"](z)) + 0.00215
        return (r * math.sin(x / r), -r * math.cos(x / r), z)

    def label(body, x, z, size):
        data = bpy.data.curves.new("Temporary scale lettering", "FONT")
        data.body = body
        data.size = size
        data.align_x = "RIGHT"
        obj = bpy.data.objects.new(data.name, data)
        collection.objects.link(obj)
        bpy.context.view_layer.update()
        evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
        glyph = evaluated.to_mesh()
        glyph.calc_loop_triangles()
        offset = len(print_vertices)
        print_vertices.extend(surface(x + v.co.x, z + v.co.y) for v in glyph.vertices)
        print_faces.extend(tuple(offset + i for i in triangle.vertices) for triangle in glyph.loop_triangles)
        evaluated.to_mesh_clear()
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.curves.remove(data)

    for ml in range(5, 116, 5):
        z = float(heights[np.argmin(abs(volume - ml * 1e-6))])
        major = ml % 25 == 0
        start, end = (0.0005, 0.012) if major else (0.006, 0.012)
        offset = len(print_vertices)
        for x in np.linspace(start, end, 8):
            print_vertices.extend((surface(x, z - 0.00012), surface(x, z + 0.00012)))
        for k in range(7):
            a = offset + 2 * k
            print_faces.append((a, a + 1, a + 3, a + 2))
        if major:
            label(str(ml), -0.003, z - 0.0016, 0.0042)
    label("mL", 0.011, 0.076, 0.0040)
    mesh("Pitcher calibrated vermilion scale", print_vertices, print_faces, red, False)

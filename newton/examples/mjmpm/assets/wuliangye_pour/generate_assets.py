# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Build reference-inspired Wuliangye props in Blender; export triangle meshes.

Run in Blender's Python console with ``exec(compile(open(path).read(), path, 'exec'))``.
Only the named collection is replaced. The simulation uses the exported NPZ,
so Blender and a Chinese font are not runtime dependencies.
"""

from itertools import pairwise
from pathlib import Path

import bpy
import numpy as np

OUTPUT = Path(__file__).resolve().parent
COLLECTION = "Newton_Wuliangye_Props"
old = bpy.data.collections.get(COLLECTION)
if old:
    for obj in list(old.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(old)
collection = bpy.data.collections.new(COLLECTION)
bpy.context.scene.collection.children.link(collection)
export = {}


def mesh_object(name, vertices, faces, color, opacity=1.0, roughness=0.22, metallic=0.0):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if opacity < 1.0:
        bsdf.inputs["Transmission Weight"].default_value = 1.0
        bsdf.inputs["IOR"].default_value = 1.46
    material.diffuse_color = (*color, 1.0)
    obj.data.materials.append(material)
    export[name + "_vertices"] = np.asarray(vertices, dtype=np.float32)
    export[name + "_faces"] = np.asarray(faces, dtype=np.int32)
    export[name + "_color"] = np.asarray(color, dtype=np.float32)
    export[name + "_opacity"] = np.asarray(opacity, dtype=np.float32)
    export[name + "_normals"] = np.asarray([tuple(v.normal) for v in mesh.vertices], dtype=np.float32)
    export[name + "_roughness"] = np.asarray(roughness, dtype=np.float32)
    export[name + "_metallic"] = np.asarray(metallic, dtype=np.float32)
    return obj


def lathe(name, profile, color, opacity=1.0, segments=64):
    vertices = []
    rings = []
    for radius, z in profile:
        ring = []
        for angle in np.arange(segments if radius else 1) * (2 * np.pi / segments):
            ring.append(len(vertices))
            vertices.append((radius * np.cos(angle), radius * np.sin(angle), z))
        rings.append(ring)
    faces = []
    for a, b in pairwise(rings):
        for i in range(segments):
            j = (i + 1) % segments
            if len(a) == 1:
                faces.append((a[0], b[j], b[i]))
            elif len(b) == 1:
                faces.append((a[i], a[j], b[0]))
            else:
                faces.extend(((a[i], a[j], b[i]), (a[j], b[j], b[i])))
    return mesh_object(name, vertices, faces, color, opacity)


lathe(
    "bottle",
    [
        (0, 0),
        (0.037, 0),
        (0.045, 0.012),
        (0.047, 0.07),
        (0.043, 0.10),
        (0.025, 0.125),
        (0.019, 0.14),
        (0.019, 0.18),
        (0.014, 0.18),
        (0.014, 0.14),
        (0.021, 0.125),
        (0.038, 0.099),
        (0.042, 0.07),
        (0.040, 0.015),
        (0, 0.015),
    ],
    (0.79, 0.90, 0.94),
    0.24,
)


# Shared bowl profile: the visible inside and the MPM cavity use the same
# smooth spline. Store it so containment checks follow the actual vessel.
def rounded_profile(knots, samples=5):
    points = np.asarray(knots, dtype=float)
    r, z = points[:, 0], points[:, 1]
    slope = np.gradient(r, z)
    result = []
    for i in range(len(points) - 1):
        dz = z[i + 1] - z[i]
        for t in np.linspace(0, 1, samples, endpoint=False):
            radius = (
                (2 * t**3 - 3 * t**2 + 1) * r[i]
                + (t**3 - 2 * t**2 + t) * dz * slope[i]
                + (-2 * t**3 + 3 * t**2) * r[i + 1]
                + (t**3 - t**2) * dz * slope[i + 1]
            )
            result.append((max(0.0, radius), z[i] + t * dz))
    return [*result, tuple(points[-1])]


glass_inner = rounded_profile(
    [
        (0, 0.095),
        (0.020, 0.097),
        (0.028, 0.103),
        (0.038, 0.115),
        (0.045, 0.135),
        (0.047, 0.150),
        (0.046, 0.168),
        (0.041, 0.188),
        (0.039, 0.195),
    ]
)
glass_outer = [(0, 0.0925)] + [(r + 0.0018, z - 0.0015) for r, z in glass_inner[1:]]
glass_wall = [*glass_outer, (0.0403, 0.195), (0.0396, 0.1955), *list(reversed(glass_inner))]
# The wetted surface is identical. Thicken the invisible exterior to two
# MPM cell so both wall sides remain distinguishable on the collision grid.
collision_outer = [(0, 0.087)] + [(r + 0.008, z - 0.008) for r, z in glass_inner[1:]]
lathe("glass", [*collision_outer, (0.047, 0.195), *list(reversed(glass_inner))], (0.91, 0.97, 0.98), 0.008, segments=96)
export["glass_inner_profile"] = np.asarray(glass_inner, dtype=np.float32)


# Keep the validated collision cavities independent of decorative tessellation.
for name in ("bottle", "glass"):
    collection.objects[name].hide_render = True
    collection.objects[name].hide_viewport = True
    collection.objects[name].hide_set(True)

GLASS = (0.91, 0.97, 0.98)
GOLD = (0.83, 0.55, 0.15)
RED = (0.48, 0.008, 0.018)
outer = [
    (0, 0),
    (0.037, 0),
    (0.042, 0.003),
    (0.045, 0.008),
    (0.045, 0.012),
    (0.0455, 0.020),
    (0.046, 0.032),
    (0.0465, 0.050),
    (0.047, 0.070),
    (0.0462, 0.081),
    (0.0453, 0.091),
    (0.0447, 0.096),
    (0.0435, 0.100),
    (0.042, 0.103),
    (0.040, 0.106),
    (0.0375, 0.109),
    (0.035, 0.112),
    (0.029, 0.119),
    (0.025, 0.125),
    (0.0225, 0.130),
    (0.0203, 0.135),
    (0.0193, 0.139),
    (0.019, 0.143),
    (0.019, 0.174),
    (0.0196, 0.175),
    (0.0196, 0.179),
    (0.019, 0.180),
]
inner = [(0.014, 0.180), (0.014, 0.140), (0.021, 0.125), (0.038, 0.099), (0.042, 0.070), (0.040, 0.015), (0, 0.015)]
bottle = lathe("bottle_visual", outer + inner, GLASS, 0.012, segments=144)
# Shallow twelve-fold pressed-glass flutes, strongest at the heel and shoulder.
# The smooth inner cavity and the neck remain identical to the collider.
v = export["bottle_visual_vertices"].copy()
outer_count = 1 + (len(outer) - 1) * 144
for i in range(outer_count):
    x, y, z = v[i]
    amplitude = 0.0024 * np.exp(-(((z - 0.022) / 0.015) ** 2)) + 0.0018 * np.exp(-(((z - 0.094) / 0.012) ** 2))
    radius = np.hypot(x, y)
    if radius > 0:
        scale = 1 - amplitude * (0.5 + 0.5 * np.cos(12 * np.arctan2(y, x))) / radius
        v[i, :2] *= scale
        bottle.data.vertices[i].co = v[i]
bottle.data.update()
export["bottle_visual_vertices"] = v
export["bottle_visual_normals"] = np.asarray([tuple(v.normal) for v in bottle.data.vertices], dtype=np.float32)

# A turned foot, slender drawn stem, rounded bowl and rolled lip. The wetted
# bowl profile is shared with the simulation; only the exterior is refined.
lathe(
    "glass_stem",
    [
        (0, 0.001),
        (0.027, 0.001),
        (0.031, 0.0015),
        (0.033, 0.003),
        (0.033, 0.004),
        (0.031, 0.0055),
        (0.021, 0.0065),
        (0.010, 0.008),
        (0.005, 0.012),
        (0.003, 0.020),
        (0.0025, 0.045),
        (0.0028, 0.073),
        (0.004, 0.084),
        (0.008, 0.090),
        (0.016, 0.093),
        (0, 0.093),
    ],
    GLASS,
    0.025,
    segments=128,
)
lathe("glass_visual", glass_wall, GLASS, 0.008, segments=128)


def ring(name, radius, z, tube, color=GLASS, opacity=0.025):
    profile = [(radius + tube * np.cos(t), z + tube * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 9)]
    return lathe(name, profile, color, opacity, segments=96)


ring("glass_lip", 0.0397, 0.195, 0.00055)
ring("glass_foot_edge", 0.032, 0.0035, 0.00065, opacity=0.025)
ring("bottle_heel", 0.044, 0.007, 0.001, opacity=0.025)
ring("bottle_lip", 0.0168, 0.1795, 0.0015, opacity=0.025)
lathe("neck_band", [(0.0192, 0.140), (0.0198, 0.140), (0.0198, 0.155), (0.0192, 0.155)], GOLD)
lathe("neck_seal", [(0.0198, 0.154), (0.020, 0.154), (0.020, 0.164), (0.0198, 0.164)], RED)
lathe(
    "cap",
    [(0, 0), (0.020, 0), (0.021, 0.0015), (0.021, 0.042), (0.0205, 0.044), (0.019, 0.045), (0, 0.045)],
    RED,
    segments=96,
)
lathe("cap_band", [(0.021, 0.002), (0.0213, 0.002), (0.0213, 0.010), (0.021, 0.010)], GOLD, segments=96)
for i, z in enumerate((0.013, 0.015, 0.038, 0.040)):
    ring(f"cap_ridge_{i}", 0.021, z, 0.0003, RED, 1.0)


def surface_radius(z):
    return float(np.interp(z, [p[1] for p in outer[1:]], [p[0] for p in outer[1:]]))


def label_patch(name, half_angle, bottom, top, color, offset):
    vertices, faces = [], []
    for z in np.linspace(bottom, top, 12):
        radius = surface_radius(z) + offset
        for theta in np.linspace(-half_angle, half_angle, 65):
            vertices.append((radius * np.sin(theta), -radius * np.cos(theta), z))
    for row in range(11):
        for col in range(64):
            i = row * 65 + col
            faces.extend(((i, i + 1, i + 65), (i + 1, i + 66, i + 65)))
    mesh_object(name, vertices, faces, color, roughness=0.25, metallic=0.65 if color == GOLD else 0.0)


label_patch("label_border", 1.22, 0.0255, 0.0845, GOLD, 0.0004)
label_patch("label", 1.20, 0.026, 0.084, (1.0, 1.0, 1.0), 0.00065)
label = collection.objects["label"]
uvs = np.asarray([(col / 64, row / 11) for row in range(12) for col in range(65)], dtype=np.float32)
export["label_uvs"] = uvs
export["label_roughness"] = np.asarray(0.32, dtype=np.float32)
uv_layer = label.data.uv_layers.new(name="LabelUV")
for loop in label.data.loops:
    uv_layer.data[loop.index].uv = uvs[loop.vertex_index]
material = label.data.materials[0]
bsdf = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
image_node = material.node_tree.nodes.new("ShaderNodeTexImage")
image_node.image = bpy.data.images.load(str(OUTPUT / "label.jpg"), check_existing=False)
image_node.image.pack()
material.node_tree.links.new(image_node.outputs["Color"], bsdf.inputs["Base Color"])

# Shoulder medallion: recessed ruby enamel, concentric brass bezel and a
# five-grain radial motif, all geometry rather than a floating rectangular decal.
for name, radius, color, offset in (("crest_rim", 0.009, GOLD, 0.001), ("crest", 0.0077, RED, 0.0013)):
    vertices = [(0, -surface_radius(0.111) - offset, 0.111)]
    for theta in np.linspace(0, 2 * np.pi, 65)[:-1]:
        x, z = radius * np.cos(theta), 0.111 + 0.72 * radius * np.sin(theta)
        y = -np.sqrt(max(surface_radius(z) ** 2 - x * x, 0)) - offset
        vertices.append((x, y, z))
    mesh_object(
        name,
        vertices,
        [(0, i + 1, (i + 1) % 64 + 1) for i in range(64)],
        color,
        roughness=0.20,
        metallic=0.7 if color == GOLD else 0.0,
    )
vertices, faces = [], []
for theta in np.linspace(0, 2 * np.pi, 6)[:-1]:
    start = len(vertices)
    for radius, angle in ((0.001, theta), (0.0055, theta - 0.23), (0.007, theta), (0.0055, theta + 0.23)):
        x, z = radius * np.sin(angle), 0.111 + 0.72 * radius * np.cos(angle)
        vertices.append((x, -np.sqrt(surface_radius(z) ** 2 - x * x) - 0.0015, z))
    faces.extend(((start, start + 2, start + 1), (start, start + 3, start + 2)))
mesh_object("crest_grains", vertices, faces, GOLD, roughness=0.2, metallic=0.7)

# Turned walnut display rest instead of a placeholder cube.
lathe(
    "rest",
    [
        (0, 0),
        (0.040, 0),
        (0.044, 0.003),
        (0.044, 0.007),
        (0.041, 0.010),
        (0.041, 0.073),
        (0.044, 0.076),
        (0.044, 0.078),
        (0.041, 0.080),
        (0, 0.080),
    ],
    (0.28, 0.16, 0.085),
    segments=96,
)
ring("rest_trim", 0.042, 0.006, 0.0009, GOLD, 1.0)

names = [name.removesuffix("_vertices") for name in export if name.endswith("_vertices")]
export["names"] = np.asarray(names)
for name in names:
    group = (
        "rest"
        if name.startswith("rest")
        else "glass"
        if name.startswith("glass")
        else "cap"
        if name.startswith("cap")
        else "bottle"
    )
    export[name + "_group"] = np.asarray(group)
    if name not in ("bottle", "glass"):
        obj = collection.objects[name]
        if group == "bottle" and name != "bottle_visual":
            obj.parent = bottle
        elif group == "cap" and name != "cap":
            obj.parent = collection.objects["cap"]
        elif group == "rest" and name != "rest":
            obj.parent = collection.objects["rest"]
        elif group == "glass" and name != "glass_visual":
            obj.parent = collection.objects["glass_visual"]
    if name in ("bottle_visual", "glass_visual", "glass_stem"):
        export[name + "_roughness"] = np.asarray(0.08, dtype=np.float32)
    elif "band" in name:
        export[name + "_metallic"] = np.asarray(0.75, dtype=np.float32)
    obj = collection.objects[name]
    bsdf = next(n for n in obj.data.materials[0].node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = float(export[name + "_roughness"])
    bsdf.inputs["Metallic"].default_value = float(export[name + "_metallic"])
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
for name in names:
    if name in ("bottle", "glass"):
        # Invisible colliders need no shading normals. Keep physics positions
        # at full precision; only display data is quantized for distribution.
        del export[name + "_normals"]
    else:
        export[name + "_vertices"] = np.round(export[name + "_vertices"], decimals=6)
        export[name + "_normals"] = export[name + "_normals"].astype(np.float16)
np.savez_compressed(OUTPUT / "props.npz", **export)
collection.objects["rest"].location = (-0.2, 0.0, 0.0)
collection.objects["cap"].location = (-0.10, 0.0, 0.0)
collection.objects["glass_visual"].location = (0.15, 0.0, 0.0)
if "BLEND_OUTPUT" in globals():
    bpy.data.libraries.write(str(BLEND_OUTPUT), {collection}, fake_user=True, compress=True)
print(f"Exported {len(names)} meshes to {OUTPUT / 'props.npz'}")

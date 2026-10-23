# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run inside Blender (also via MCP) to author the bread-cutting assets.

All meshes are original procedural assets. The loaf skin has two embedding
patches at the demonstrated cut plane; these are visuals, not physical pieces.
"""

import json
import runpy
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

OUTPUT = Path("E:/csy_work/CG/Engine/newton/newton/examples/mjmpm/assets/bread_cut")
OUTPUT.mkdir(parents=True, exist_ok=True)
COLLECTION = "MJMPM Bread Cutting"
collection = bpy.data.collections.get(COLLECTION)
if collection is None:
    collection = bpy.data.collections.new(COLLECTION)
    bpy.context.scene.collection.children.link(collection)
else:
    for item in list(collection.objects):
        bpy.data.objects.remove(item, do_unlink=True)


def image_texture(name, kind):
    """Original deterministic crust, crumb and end-grain colour textures."""
    size = 1024
    rng = np.random.default_rng(62)
    yy, xx = np.mgrid[:size, :size] / size
    noise = rng.normal(0.0, 0.018, (size, size))
    if kind == "crust":
        grain = np.sin(xx * 45 + 3 * np.sin(yy * 19)) * 0.035 + np.sin(yy * 31) * 0.025
        color = np.array((0.56, 0.285, 0.09))[None, None, :] * (1 + (noise + grain)[:, :, None] * 3)
        flour = np.zeros_like(xx)
        for center in (0.23, 0.47, 0.71):
            line = yy - center - 0.65 * (xx - 0.25)
            flour += 0.65 * np.exp(-((line / 0.012) ** 2)) * np.exp(-(((xx - 0.25) / 0.14) ** 6))
        flour += np.maximum(noise - 0.032, 0) * 2
        flour = np.clip(flour, 0, 0.8)
        color = color * (1 - flour[:, :, None]) + np.array((0.84, 0.74, 0.54)) * flour[:, :, None]
    elif kind == "crumb":
        color = np.ones((size, size, 3)) * np.array((0.87, 0.76, 0.54))
        color += noise[:, :, None]
        for _ in range(3200):
            cx, cy = rng.uniform(0, 1, 2)
            rx, ry = rng.uniform(0.001, 0.009, 2)
            xlo, xhi = max(0, int((cx - rx) * size)), min(size, int((cx + rx) * size) + 1)
            ylo, yhi = max(0, int((cy - ry) * size)), min(size, int((cy + ry) * size) + 1)
            distance = ((xx[ylo:yhi, xlo:xhi] - cx) / rx) ** 2 + ((yy[ylo:yhi, xlo:xhi] - cy) / ry) ** 2
            pore = np.clip(1 - distance, 0, 1)
            color[ylo:yhi, xlo:xhi] *= 1 - 0.60 * pore[:, :, None]
    elif kind == "steel":
        brushing = np.sin(yy * 5500 + np.sin(xx * 45)) * 0.007 + noise * 0.45
        color = np.array((0.68, 0.71, 0.73))[None, None, :] * (1 + brushing[:, :, None])
    else:
        rings = np.sin(yy * 1800 + np.sin(xx * 20) * 2 + np.sin(xx * 39))
        color = np.array((0.49, 0.30, 0.14))[None, None, :] * (1 + (0.12 * rings + noise)[:, :, None])
    rgba = np.concatenate((np.clip(color, 0, 1), np.ones((size, size, 1))), axis=2).astype(np.float32)
    img = bpy.data.images.new(name, width=size, height=size, alpha=True)
    img.pixels.foreach_set(rgba.ravel())
    img.filepath_raw = str(OUTPUT / f"{kind}.jpg")
    img.file_format = "JPEG"
    img.save()
    return img


def material(name, color, roughness=0.6, metallic=0.0, image=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    if image is not None:
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = image
        mat.node_tree.links.new(tex.outputs["Color"], shader.inputs["Base Color"])
        bump = mat.node_tree.nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = 0.22
        bump.inputs["Distance"].default_value = 0.000004 if metallic > 0.5 else 0.00035
        mat.node_tree.links.new(tex.outputs["Color"], bump.inputs["Height"])
        mat.node_tree.links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    return mat


crust = material("Bread / golden baked crust", (0.58, 0.29, 0.10), image=image_texture("Crust colour", "crust"))
crumb = material("Bread / porous fresh crumb", (0.87, 0.76, 0.54), image=image_texture("Crumb colour", "crumb"))
wood = material("Board / oiled walnut", (0.49, 0.30, 0.14), image=image_texture("Walnut grain", "wood"))
steel = material(
    "Knife / brushed stainless steel",
    (0.62, 0.67, 0.70),
    0.23,
    0.92,
    image=image_texture("Fine brushed steel", "steel"),
)
edge = material("Knife / polished bevel", (0.85, 0.87, 0.88), 0.12, 1.0)
handle = material("Knife / dark walnut handle", (0.07, 0.035, 0.015), 0.38)


def move_collection(obj):
    for owner in list(obj.users_collection):
        owner.objects.unlink(obj)
    collection.objects.link(obj)


def box(name, center, half, mat, parent, bevel):
    bpy.ops.mesh.primitive_cube_add(size=2, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.scale = half
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    obj["simulation_parent"] = parent
    if bevel:
        modifier = obj.modifiers.new("Rounded machined edges", "BEVEL")
        modifier.width = bevel
        modifier.segments = 3
    move_collection(obj)
    return obj


box("Walnut chopping board", (0, 0, -0.009), (0.16, 0.16, 0.009), wood, "board", 0.004)
knife_origin = np.array((-0.15, -0.035, 0.13))
geometry = runpy.run_path(str(OUTPUT.parents[1] / "support/bread_cut_geometry.py"))
handle_width = geometry["HANDLE_HALF_WIDTH"]
box("Knife walnut handle", knife_origin, (0.048, handle_width, 0.012), handle, "knife", 0.005)
blade_mesh = geometry["blade_mesh"]
physical_points, physical_faces = blade_mesh()
blade_vertices = (physical_points + knife_origin).tolist()
blade_faces = physical_faces.tolist()
blade_data = bpy.data.meshes.new("Forged curved chef blade geometry")
blade_data.from_pydata(blade_vertices, [], blade_faces)
blade_data.materials.append(steel)
for face in blade_data.polygons:
    face.use_smooth = True
blade_data.set_sharp_from_angle(angle=0.35)
blade_uv = blade_data.uv_layers.new(name="Brushed steel grain")
for face in blade_data.polygons:
    for li in face.loop_indices:
        position = physical_points[blade_data.loops[li].vertex_index]
        blade_uv.data[li].uv = ((position[0] - 0.060) / 0.180, (position[2] + 0.055) / 0.070)
blade_obj = bpy.data.objects.new("Knife / resolved spine and sharp edge", blade_data)
collection.objects.link(blade_obj)
blade_obj["simulation_parent"] = "knife"
for sign in (-1, 1):
    bevel_points = []
    for x, z in ((0.060, -0.0548), (0.190, -0.0548), (0.190, -0.048), (0.060, -0.048)):
        width = 0.0005 + (z + 0.055) / 0.070 * 0.0055 + 0.000025
        bevel_points.append(tuple(knife_origin + np.array((x, sign * width, z))))
    bevel_data = bpy.data.meshes.new("Ground cutting bevel")
    bevel_data.from_pydata(bevel_points, [], [(0, 1, 2, 3)])
    bevel_data.materials.append(edge)
    bevel_obj = bpy.data.objects.new("Knife / polished cutting bevel", bevel_data)
    collection.objects.link(bevel_obj)
    bevel_obj["simulation_parent"] = "knife"
stamp_mat = material("Knife / laser etched steel", (0.09, 0.10, 0.11), 0.40, 0.7)
bpy.ops.object.text_add(location=tuple(knife_origin + np.array((0.078, -0.005, -0.006))))
stamp = bpy.context.object
stamp.name = "Knife / laser etched INOX mark"
stamp.data.body = "INOX  |  FORGED"
stamp.data.size = 0.0025
stamp.rotation_euler = (np.pi / 2, 0, 0)
stamp.data.materials.append(stamp_mat)
bpy.ops.object.convert(target="MESH")
stamp = bpy.context.object
move_collection(stamp)
stamp["simulation_parent"] = "knife"
for vertex in stamp.data.vertices:
    global_point = stamp.matrix_world @ vertex.co
    local_z = global_point.z - knife_origin[2]
    global_point.y = knife_origin[1] - (0.0005 + (local_z + 0.055) / 0.070 * 0.0055) - 0.00005
    vertex.co = stamp.matrix_world.inverted() @ global_point
box("Knife bolster", knife_origin + np.array((0.050, 0, 0)), (0.006, 0.010, 0.014), steel, "knife", 0.002)
for x in (-0.027, 0.0, 0.027):
    for sign in (-1, 1):
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=16,
            ring_count=8,
            radius=1,
            location=knife_origin + np.array((x, sign * (handle_width - 0.00015), 0)),
        )
        obj = bpy.context.object
        obj.name = "Flush stainless handle rivet"
        obj.scale = (0.0028, 0.00035, 0.0028)
        obj.data.materials.append(steel)
        obj["simulation_parent"] = "knife"
        move_collection(obj)


def loaf_patch(name, ylo, yhi, side):
    rings, segments = 48, 96
    vertices, faces, uv = [], [], []
    for j in range(rings + 1):
        y = ylo + (yhi - ylo) * j / rings
        end = np.sqrt(max(1.0 - (max(abs(y) - 0.060, 0.0) / 0.040) ** 2, 0.001))
        for k in range(segments + 1):
            angle = k / segments * 2 * np.pi
            x = 0.050 * end * np.cos(angle)
            z = max(0.002, 0.027 + 0.038 * end * np.sin(angle))
            irregular = 0.00035 * np.sin(k * 0.71 + j * 1.7) * np.sin(angle) ** 2
            vertices.append((x, y, z + irregular))
            uv.append((k / segments, (y + 0.1) / 0.2))
    for j in range(rings):
        for k in range(segments):
            a = j * (segments + 1) + k
            faces.append((a, a + 1, a + segments + 2, a + segments + 1))
    outer_count = len(faces)
    # Both ends have physical skin surfaces; the internal cut surfaces start
    # coincident and become visible only as their live material moves apart.
    for ring in (0, rings):
        center = len(vertices)
        y = ylo if ring == 0 else yhi
        vertices.append((0, y, 0.027))
        uv.append((0.5, 0.5))
        for k in range(segments):
            a = ring * (segments + 1) + k
            faces.append((center, a + 1, a) if ring == 0 else (center, a, a + 1))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], [face[::-1] for face in faces])
    mesh.materials.append(crust)
    mesh.materials.append(crumb)
    layer = mesh.uv_layers.new(name="Bread UV")
    for face in mesh.polygons:
        face.use_smooth = face.index < outer_count
        cut_cap = face.index >= outer_count and (
            (side < 0 and face.index >= outer_count + segments) or (side > 0 and face.index < outer_count + segments)
        )
        face.material_index = int(cut_cap)
        for li in face.loop_indices:
            vi = mesh.loops[li].vertex_index
            if cut_cap:
                p = vertices[vi]
                layer.data[li].uv = (p[0] / 0.1 + 0.5, p[2] / 0.065)
            else:
                layer.data[li].uv = uv[vi]
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    obj["simulation_parent"] = "bread"
    obj["embedding_side"] = side


loaf_patch("Loaf / slice skin", -0.098, -0.035, -1)
loaf_patch("Loaf / remainder skin", -0.035, 0.098, 1)

# Export evaluated triangles and UVs by material for Newton's live renderer.
archive, manifest = {}, []
origins = {"knife": knife_origin, "bread": np.zeros(3), "board": np.zeros(3)}
depsgraph = bpy.context.evaluated_depsgraph_get()
for obj in list(collection.objects):
    if obj.type != "MESH":
        continue
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    mesh.calc_loop_triangles()
    for material_id, mat in enumerate(mesh.materials):
        triangles = [t for t in mesh.loop_triangles if t.material_index == material_id]
        if not triangles:
            continue
        points, normals, uvs = [], [], []
        normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
        parent = obj["simulation_parent"]
        for tri in triangles:
            for vi, li in zip(tri.vertices, tri.loops, strict=True):
                point = np.array(obj.matrix_world @ mesh.vertices[vi].co) - origins[parent]
                if parent == "bread" and mat is crumb:
                    point[1] += obj["embedding_side"] * 0.0003
                points.append(point)
                normals.append(tuple((normal_matrix @ mesh.corner_normals[li].vector).normalized()))
                uvs.append(tuple(mesh.uv_layers.active.data[li].uv) if mesh.uv_layers.active else (0, 0))
        key = f"part_{len(manifest)}"
        archive[key + "_points"] = np.asarray(points, dtype=np.float32)
        archive[key + "_normals"] = np.asarray(normals, dtype=np.float16)
        archive[key + "_uvs"] = np.asarray(uvs, dtype=np.float32)
        archive[key + "_indices"] = np.arange(len(points), dtype=np.uint16)
        shader = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        texture = next((n.image for n in mat.node_tree.nodes if n.type == "TEX_IMAGE"), None)
        manifest.append(
            {
                "key": key,
                "parent": parent,
                "side": obj.get("embedding_side", 0),
                "name": obj.name,
                "color": list(shader.inputs["Base Color"].default_value[:3]),
                "roughness": shader.inputs["Roughness"].default_value,
                "metallic": shader.inputs["Metallic"].default_value,
                "texture": Path(texture.filepath_raw).name if texture else None,
            }
        )
    evaluated.to_mesh_clear()
archive["manifest"] = np.array(json.dumps(manifest))
np.savez_compressed(OUTPUT / "visuals.npz", **archive)

# A studio preview, without removing objects from the user's existing scene.
for obj in bpy.context.scene.objects:
    obj.hide_render = obj.name not in collection.objects
for location, energy, size in (((0.1, -0.3, 0.5), 18, 0.35), ((-0.3, 0.2, 0.35), 12, 0.30)):
    data = bpy.data.lights.new("Bread studio softbox", "AREA")
    data.energy, data.shape, data.size = energy, "DISK", size
    light = bpy.data.objects.new(data.name, data)
    collection.objects.link(light)
    light.location = location
    light.rotation_euler = (Vector((0, 0, 0.04)) - light.location).to_track_quat("-Z", "Y").to_euler()
camera_data = bpy.data.cameras.new("Bread asset camera")
camera = bpy.data.objects.new("Bread asset camera", camera_data)
collection.objects.link(camera)
camera.location = (0.32, -0.40, 0.31)
camera.rotation_euler = (Vector((0, 0, 0.05)) - camera.location).to_track_quat("-Z", "Y").to_euler()
camera_data.lens = 48
scene = bpy.context.scene
scene.camera = camera
scene.render.resolution_x, scene.render.resolution_y = 1280, 960
scene.render.resolution_percentage = 100
blend = Path("C:/Users/Administrator/Documents/Blender/BreadCut/bread_cut_assets.blend")
blend.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=str(blend))
print(
    json.dumps(
        {
            "parts": len(manifest),
            "triangles": sum(len(v) // 3 for k, v in archive.items() if k.endswith("_indices")),
            "archive": str(OUTPUT / "visuals.npz"),
            "blend": str(blend),
        }
    )
)

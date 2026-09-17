# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build meter-scale popcorn props inside Blender through its MCP addon.

The caller supplies SOURCE (validated shell/pan meshes) and OUTPUT_DIRECTORY.
Only the dedicated asset scene is changed. Collision geometry stays in Newton;
the exported cup preserves the simulation vertex ordering exactly.
"""

import json
import math
from pathlib import Path

if __name__ == "__main__":
    import socket

    from newton.examples.mjvbdv2.example_mjvbd_v2_popcorn import cup_mesh, scoop_bowl_panels

    vertices, faces = cup_mesh()
    source = {
        "cup_vertices": vertices.tolist(),
        "cup_faces": faces.tolist(),
        "panels": [
            {"vertices": m.vertices.tolist(), "faces": m.indices.reshape(-1, 3).tolist()} for m in scoop_bowl_panels()
        ],
    }
    destination = Path(__file__).resolve().parents[1] / "assets/popcorn"
    code = "__name__ = 'blender_asset_build'\nSOURCE = " + repr(source)
    code += "\nOUTPUT_DIRECTORY = " + repr(str(destination)) + "\n"
    code += Path(__file__).read_text(encoding="utf-8")
    with socket.create_connection(("127.0.0.1", 9876), timeout=10) as connection:
        connection.settimeout(60)
        connection.sendall(json.dumps({"type": "execute_code", "params": {"code": code}}).encode())
        response = bytearray()
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                raise RuntimeError("Blender closed the connection before returning a response")
            response.extend(chunk)
            try:
                result = json.loads(response)
                break
            except json.JSONDecodeError:
                pass
    print(json.dumps(result))
    if result.get("status") != "success":
        raise RuntimeError(result)
    raise SystemExit(0)

import bpy
from mathutils import Vector

output = Path(OUTPUT_DIRECTORY)
output.mkdir(parents=True, exist_ok=True)
scene_name = "Newton Popcorn Assets R2"
if scene_name in bpy.data.scenes:
    if len(bpy.data.scenes[scene_name].objects):
        raise RuntimeError("Asset scene already exists; preserve it before rebuilding")
    bpy.data.scenes.remove(bpy.data.scenes[scene_name])
scene = bpy.data.scenes.new(scene_name)
bpy.context.window.scene = scene
scene.unit_settings.system = "METRIC"
scene.unit_settings.scale_length = 1.0
props = []


def material(name, color, metal=0.0, roughness=0.4):
    mat = bpy.data.materials.new("Popcorn/" + name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    bsdf = next(node for node in mat.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*color, 1)
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


red = material("enamel oxblood", (0.48, 0.016, 0.022), 0.45, 0.27)
steel = material("brushed stainless", (0.57, 0.61, 0.64), 0.9, 0.29)
aluminum = material("satin aluminum", (0.66, 0.69, 0.72), 0.78, 0.48)
black = material("phenolic handle", (0.008, 0.010, 0.012), 0.0, 0.70)
next(node for node in black.node_tree.nodes if node.type == "BSDF_PRINCIPLED").inputs[
    "Specular IOR Level"
].default_value = 0.2
paper = material("unbleached paper", (0.88, 0.82, 0.68), 0, 0.88)
ink = material("faded paper print", (0.48, 0.33, 0.23), 0, 0.9)
glass = material("clear tempered glass", (0.74, 0.89, 0.92), 0, 0.1)
next(node for node in glass.node_tree.nodes if node.type == "BSDF_PRINCIPLED").inputs[
    "Transmission Weight"
].default_value = 1


def finish(obj, name, group, mat, bevel=0):
    obj.name = name
    obj.data.materials.append(mat)
    obj["newton_group"] = group
    if bevel:
        modifier = obj.modifiers.new("Manufactured edge radii", "BEVEL")
        modifier.width = bevel
        modifier.segments = 3
    props.append(obj)
    return obj


def box(name, pos, size, mat, bevel=0.001, group="machine"):
    bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
    obj = bpy.context.object
    obj.scale = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return finish(obj, name, group, mat, bevel)


def cylinder(name, pos, radius, depth, mat, group="machine", axis=(0, 0, 1), bevel=0.0005):
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=radius, depth=depth, location=pos)
    obj = bpy.context.object
    obj.rotation_euler = Vector(axis).to_track_quat("Z", "Y").to_euler()
    for polygon in obj.data.polygons:
        polygon.use_smooth = len(polygon.vertices) == 4
    return finish(obj, name, group, mat, bevel)


def mesh(name, vertices, faces, mat, group):
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    scene.collection.objects.link(obj)
    return finish(obj, name, group, mat)


# Machine coordinates: origin at the tray pedestal top, x points away from
# the operator. Dimensions match the existing open-front collision enclosure.
box("Enamel base", (0, 0, 0.011), (0.33, 0.36, 0.022), red, 0.004)
box("Pedestal", (0, 0, -0.03), (0.33, 0.36, 0.06), red, 0.003)
box("Tray floor", (0, 0, 0.025), (0.30, 0.32, 0.008), steel, 0.0006)
box("Rear tray baffle", (0.10125, 0, 0.075), (0.0075, 0.32, 0.09), steel)
box("Rear cabinet", (0.1575, 0, 0.25), (0.012, 0.34, 0.44), steel)
for y in (-0.17, 0.17):
    box("Folded tray side", (0, y, 0.07), (0.315, 0.014, 0.09), steel)
    obj = box("Tempered side glass", (0, y, 0.295), (0.30, 0.004, 0.36), glass, 0.0004)
    obj["opacity"] = 0.16
    box("Rear corner post", (0.15, y, 0.26), (0.0165, 0.022, 0.5), red)
    # Slim glazing rails remain within the existing side-wall envelope.
    for z in (0.119, 0.47):
        box("Glass retaining rail", (0, y, z), (0.30, 0.006, 0.006), steel)
    for x in (-0.145, 0.145):
        for z in (0.13, 0.46):
            cylinder("Glazing rivet", (x, y, z), 0.002, 0.006, steel, axis=(0, 1, 0))
box("Rounded canopy", (0, 0, 0.52), (0.3525, 0.39, 0.07), red, 0.009)
box("Canopy lower folded seam", (0, 0, 0.487), (0.348, 0.387, 0.006), steel)
for y in (-0.12, -0.10, -0.08, -0.06, -0.04, 0.04, 0.06, 0.08, 0.10, 0.12):
    box("Canopy vent inset", (-0.1764, y, 0.524), (0.0004, 0.006, 0.019), black, 0.0002)
cylinder("Kettle body", (0.045, 0, 0.37), 0.075, 0.094, steel, bevel=0.004)
cylinder("Kettle lid rim", (0.045, 0, 0.42), 0.079, 0.01, aluminum)
cylinder("Lid knob", (0.045, 0, 0.435), 0.013, 0.016, black)
box("Kettle handle", (0.1125, 0, 0.395), (0.0825, 0.024, 0.024), black, 0.006)
for y in (-0.058, 0.058):
    box("Kettle hanger", (0.045, y, 0.453), (0.009, 0.006, 0.06), steel)
for y in (-0.145, 0.145):
    box("Canopy lamp diffuser", (-0.01, y, 0.479), (0.23, 0.012, 0.004), paper)

# Preserve the circular grasp envelope and real thin pan collision surface.
cylinder("Round molded grip", (0, 0, 0), 0.014, 0.11, black, "scoop", (1, 0, 0), 0.002)
cylinder("Handle ferrule", (0.052, 0, 0), 0.014, 0.004, steel, "scoop", (1, 0, 0))
cylinder(
    "Bent scoop shank", (0.095, 0, -0.010), 0.005, 0.08246, steel, "scoop", (math.sin(-1.326), 0, math.cos(-1.326))
)
for i, panel in enumerate(SOURCE["panels"]):
    obj = mesh(f"Formed aluminum pan {i:02d}", panel["vertices"], panel["faces"], aluminum, "scoop")
    # Keep exact contact reference meshes in the export, but display one
    # continuous sheet instead of coincident internal panel side walls.
    if i < len(SOURCE["panels"]) - 1:
        obj["display"] = False
        obj.hide_render = True
        obj.hide_set(True)

segments = len(SOURCE["panels"]) - 1
sheet_vertices, sheet_faces = [], []
for x in (0.135, 0.280):
    for i in range(segments + 1):
        angle = -1.30 + 2.60 * i / segments
        for radius in (0.056, 0.0572):
            sheet_vertices.append((x, radius * math.sin(angle), 0.008 - radius * math.cos(angle)))


def sheet_index(end, section, outside):
    return (end * (segments + 1) + section) * 2 + outside


for i in range(segments):
    for outside in (0, 1):
        face = (
            sheet_index(0, i, outside),
            sheet_index(1, i, outside),
            sheet_index(1, i + 1, outside),
            sheet_index(0, i + 1, outside),
        )
        sheet_faces.append(face if outside == 0 else face[::-1])
    sheet_faces.append((sheet_index(1, i, 0), sheet_index(1, i, 1), sheet_index(1, i + 1, 1), sheet_index(1, i + 1, 0)))
for i in (0, segments):
    face = (sheet_index(0, i, 0), sheet_index(0, i, 1), sheet_index(1, i, 1), sheet_index(1, i, 0))
    sheet_faces.append(face if i == 0 else face[::-1])
sheet = mesh("Continuous pressed scoop sheet", sheet_vertices, sheet_faces, aluminum, "scoop")
normals = [None] * len(sheet.data.loops)
for polygon in sheet.data.polygons:
    polygon.use_smooth = True
    for loop in polygon.loop_indices:
        point = sheet.data.vertices[sheet.data.loops[loop].vertex_index].co
        radial = Vector((0, -point.y, 0.008 - point.z)).normalized()
        alignment = polygon.normal.dot(radial)
        normals[loop] = radial * (1 if alignment > 0 else -1) if abs(alignment) > 0.8 else polygon.normal
sheet.data.normals_split_custom_set(normals)
# Outside-only shank reinforcement, like a stamped attachment tab.
box("Shank attachment tab", (0.1322, 0, -0.021), (0.0012, 0.024, 0.024), steel, bevel=0.0005, group="scoop")
for y in (-0.007, 0.007):
    cylinder("Shank flush rivet", (0.1313, y, -0.022), 0.0022, 0.0006, steel, "scoop", (-1, 0, 0), bevel=0.0002)
cylinder("Rear grip end plug", (-0.0552, 0, 0), 0.0115, 0.0004, black, "scoop", (1, 0, 0), bevel=0.0001)

cup = mesh("Paper cup physical shell", SOURCE["cup_vertices"], SOURCE["cup_faces"], paper, "cup")
cup.data.materials.append(ink)
cup.data.materials.append(material("paper lap seam", (0.92, 0.86, 0.73), 0, 0.9))
for poly in cup.data.polygons:
    poly.use_smooth = True
    c = poly.center
    # Restrained print at the lower wall leaves grip dents legible.
    if -0.044 < c.z < -0.023 and abs(math.sin(6 * math.atan2(c.y, c.x))) < 0.5:
        poly.material_index = 1
    if -0.067 < c.z < 0.067 and math.pi / 4 < math.atan2(c.y, c.x) < 0.3 * math.pi:
        poly.material_index = 2

bpy.context.view_layer.update()
export = {"version": 1, "units": "m", "meshes": []}
depsgraph = bpy.context.evaluated_depsgraph_get()
for obj in props:
    evaluated = obj.evaluated_get(depsgraph)
    data = evaluated.to_mesh()
    data.calc_loop_triangles()
    vertices = [list(obj.matrix_world @ v.co) for v in data.vertices]
    bsdf = next(n for n in obj.data.materials[0].node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    export["meshes"].append(
        {
            "name": obj.name,
            "group": obj["newton_group"],
            "vertices": vertices,
            "faces": [list(t.vertices) for t in data.loop_triangles],
            "color": list(obj.data.materials[0].diffuse_color[:3]),
            "opacity": obj.get("opacity", 1.0),
            "display": obj.get("display", True),
            "roughness": bsdf.inputs["Roughness"].default_value,
            "metallic": bsdf.inputs["Metallic"].default_value,
            "material_indices": [t.material_index for t in data.loop_triangles],
        }
    )
    if obj["newton_group"] != "cup":
        loops = [loop for triangle in data.loop_triangles for loop in triangle.loops]
        normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
        export["meshes"][-1].update(
            render_vertices=[vertices[data.loops[loop].vertex_index] for loop in loops],
            render_normals=[list((normal_matrix @ data.corner_normals[loop].vector).normalized()) for loop in loops],
        )
    evaluated.to_mesh_clear()
(output / "props.json").write_text(json.dumps(export, separators=(",", ":")), encoding="utf-8")
# Presentation layout is applied only after exporting the simulation frames.
for obj in props:
    if obj["newton_group"] == "cup":
        obj.location = (-0.30, 0.18, 0.08)
    elif obj["newton_group"] == "scoop":
        obj.location += Vector((-0.44, -0.20, 0.07))
# Save only the dedicated scene; the user's startup scene is not overwritten.
bpy.data.libraries.write(str(output / "popcorn_props.blend"), {scene}, fake_user=True)
print(json.dumps({"objects": len(props), "output": str(output), "cup_vertices": len(cup.data.vertices)}))

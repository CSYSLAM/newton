"""Blender renderer for the W1 snack-basket recording (run with ``blender --background --python render.py -- ...``).

Only simulated geometry is animated. The supplied 800k-triangle basket mesh rides on the simulated basket body;
its box collision proxies and the hidden chips core are not rendered.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT / "assets/w1_snack_basket"
BASKET_USD = ASSETS / "basket/model_basket_7.usdc"
SNACK_TEXTURE = ASSETS / "snack/resource/img/57dd04deef32c53a6a1c3dc3db516907.png"
CRUMPLED_NORMAL = ROOT / "demos/piper_bag_transport/textures/crumpled_normal.png"
HDRI = ROOT / "demos/piper_bag_transport/textures/studio_softboxes.hdr"
BASKET_SCALE = 0.36

CAMERA = dict(position=(1.95, -1.0, 1.8), target=(0.45, 0.02, 1.12), fov=40.0)
LIGHTS = [
    # name, position, target, energy (W), size (m), colour
    ("Key", (1.6, -2.2, 3.2), (0.5, 0.0, 1.0), 380, 1.6, (1.0, 0.95, 0.88)),
    ("Fill", (2.4, 1.8, 1.6), (0.5, 0.0, 1.0), 90, 2.0, (0.85, 0.92, 1.0)),
    ("Rim", (-1.0, -0.6, 2.6), (0.5, 0.0, 1.0), 220, 1.0, (1.0, 1.0, 1.0)),
    ("Top", (0.6, 0.0, 3.0), (0.6, 0.0, 0.9), 120, 1.2, (1.0, 0.98, 0.95)),
]


def principled(name, color, roughness, metallic=0.0, coat=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Coat Weight"].default_value = coat
    return mat, shader


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def snack_material():
    """Printed metallised polypropylene: glossy coated print outside, bare aluminium layer inside."""
    mat = bpy.data.materials.new("Snack bag | printed metallised film")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    outside = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
    output = next(n for n in nodes if n.type == "OUTPUT_MATERIAL")
    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = "LaysPrint"
    print_image = nodes.new("ShaderNodeTexImage")
    print_image.image = bpy.data.images.load(str(SNACK_TEXTURE), check_existing=True)
    print_image.extension = "EXTEND"
    print_image.interpolation = "Cubic"
    links.new(uv.outputs["UV"], print_image.inputs["Vector"])
    # AgX desaturates bright reds toward pink; pre-compensate the ink so the rendered print matches the asset.
    ink = nodes.new("ShaderNodeHueSaturation")
    ink.inputs["Saturation"].default_value = 1.35
    ink.inputs["Value"].default_value = 0.85
    links.new(print_image.outputs["Color"], ink.inputs["Color"])
    links.new(ink.outputs["Color"], outside.inputs["Base Color"])
    # A thin varnish over the print: sharp but weak reflections, so the ink colour is not washed out by the studio.
    outside.inputs["Roughness"].default_value = 0.4
    outside.inputs["Metallic"].default_value = 0.0
    outside.inputs["Coat Weight"].default_value = 0.3
    outside.inputs["Coat Roughness"].default_value = 0.08
    outside.inputs["Specular IOR Level"].default_value = 0.35
    # Fine creases from handling; the large folds come from the simulated film itself.
    crinkle_uv = nodes.new("ShaderNodeUVMap")
    crinkle_uv.uv_map = "Crinkle"
    crinkle = nodes.new("ShaderNodeTexImage")
    crinkle.image = bpy.data.images.load(str(CRUMPLED_NORMAL), check_existing=True)
    crinkle.image.colorspace_settings.name = "Non-Color"
    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.uv_map = "Crinkle"
    normal_map.inputs["Strength"].default_value = 0.35
    links.new(crinkle_uv.outputs["UV"], crinkle.inputs["Vector"])
    links.new(crinkle.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], outside.inputs["Normal"])
    links.new(normal_map.outputs["Normal"], outside.inputs["Coat Normal"])
    inside = nodes.new("ShaderNodeBsdfPrincipled")
    inside.inputs["Base Color"].default_value = (0.8, 0.8, 0.82, 1)
    inside.inputs["Metallic"].default_value = 1.0
    inside.inputs["Roughness"].default_value = 0.22
    links.new(normal_map.outputs["Normal"], inside.inputs["Normal"])
    geometry = nodes.new("ShaderNodeNewGeometry")
    mix = nodes.new("ShaderNodeMixShader")
    links.new(geometry.outputs["Backfacing"], mix.inputs["Fac"])
    links.new(outside.outputs["BSDF"], mix.inputs[1])
    links.new(inside.outputs["BSDF"], mix.inputs[2])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return mat


def counter_material():
    """Light speckled laminate, like a shop counter."""
    mat, shader = principled("Table | speckled laminate", (0.6, 0.6, 0.58), 0.38, coat=0.15)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 220.0
    noise.inputs["Detail"].default_value = 2.0
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position, ramp.color_ramp.elements[1].position = 0.42, 0.62
    ramp.color_ramp.elements[0].color = (0.40, 0.40, 0.39, 1)
    ramp.color_ramp.elements[1].color = (0.62, 0.62, 0.60, 1)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    return mat


def floor_material():
    mat, shader = principled("Floor | polished concrete", (0.42, 0.42, 0.42), 0.55)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 6.0
    noise.inputs["Detail"].default_value = 8.0
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.30, 0.30, 0.31, 1)
    ramp.color_ramp.elements[1].color = (0.50, 0.50, 0.50, 1)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    return mat


def world(scene, folder):
    # Blender rejects this Radiance header with CRLF line endings; rewrite only the header bytes.
    data = HDRI.read_bytes()
    end = data.index(b"\n", data.index(b"-Y ")) + 1
    hdri = folder / "studio_softboxes_lf.hdr"
    hdri.write_bytes(data[:end].replace(b"\r\n", b"\n") + data[end:])
    sky = bpy.data.worlds.new("Studio")
    sky.use_nodes = True
    nodes, links = sky.node_tree.nodes, sky.node_tree.links
    background = next(n for n in nodes if n.type == "BACKGROUND")
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(hdri))
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value = (0, 0, math.radians(110))
    coords = nodes.new("ShaderNodeTexCoord")
    links.new(coords.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    links.new(environment.outputs["Color"], background.inputs["Color"])
    background.inputs["Strength"].default_value = 0.35
    scene.world = sky


def import_basket(body):
    before = set(bpy.data.objects)
    bpy.ops.wm.usd_import(filepath=str(BASKET_USD), import_cameras=False, import_lights=False, set_frame_range=False)
    added = [obj for obj in bpy.data.objects if obj not in before]
    roots = [obj for obj in added if obj.parent is None]
    wire, _ = principled("Basket | white powder-coated wire", (0.86, 0.87, 0.86), 0.32, coat=0.35)
    grip, _ = principled("Basket | red handle grips", (0.62, 0.015, 0.02), 0.38, coat=0.2)
    for obj in added:
        if obj.type == "MESH":
            red = any(m and "mat_54" in m.name for m in obj.data.materials)
            assign(obj, grip if red else wire)
            for face in obj.data.polygons:
                face.use_smooth = True
        elif obj.type not in ("EMPTY",):
            bpy.data.objects.remove(obj)
    for root in roots:
        if root.name not in bpy.data.objects:
            continue
        # The basket body frame is the supplied asset frame scaled to real size; the body carries the rotation.
        root.parent = body
        root.matrix_parent_inverse.identity()
        root.location, root.rotation_euler, root.scale = (0, 0, 0), (0, 0, 0), (BASKET_SCALE,) * 3


def build(folder, args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(
        filepath=str(folder / "simulation.usdc"), import_cameras=False, import_lights=False, set_frame_range=True
    )
    scene = bpy.context.scene
    run = json.loads((folder / "run.json").read_text())
    with np.load(folder / "trajectory.npz") as recording:
        texture_uv = recording["film_texture_uv"]
        panel_uv = recording["film_panel_vertices"]
        panel_faces = recording["film_panel_faces"]
    basket_body = bpy.data.objects[f"body_{run['basket_body']:02d}"]
    for child in list(basket_body.children):
        # Box collision proxies of the wire basket.
        child.hide_render = child.hide_viewport = True
    bpy.data.objects["snack_core"].hide_render = bpy.data.objects["snack_core"].hide_viewport = True
    film = bpy.data.objects["snack_film"]
    if len(film.data.polygons) != len(panel_faces):
        raise ValueError("Recorded film topology does not match the exported mesh.")
    for modifier in film.modifiers:
        if modifier.type == "MESH_SEQUENCE_CACHE":
            # Stream only positions so the UV layers added below survive every frame update.
            modifier.read_data = {"VERT"}
    print_uv = film.data.uv_layers.new(name="LaysPrint")
    print_uv.data.foreach_set("uv", texture_uv[panel_faces].ravel())
    crinkle_uv = film.data.uv_layers.new(name="Crinkle")
    crinkle_uv.data.foreach_set("uv", (panel_uv[panel_faces] / 0.12).ravel())
    assign(film, snack_material())
    for face in film.data.polygons:
        face.use_smooth = True
    subdivision = film.modifiers.new("Render smoothing", "SUBSURF")
    subdivision.levels, subdivision.render_levels = 1, 2
    import_basket(basket_body)
    rubber, _ = principled("Gripper | rubber pads", (0.02, 0.02, 0.02), 0.7)
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.name.startswith("pad_") and obj.parent is not basket_body:
            assign(obj, rubber)
        elif obj.name.startswith("robot_"):
            for mat in obj.data.materials:
                shader = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None) if mat else None
                if shader is not None:
                    # USD preview surfaces import fully rough; a light satin finish reads as machined parts.
                    shader.inputs["Roughness"].default_value = min(shader.inputs["Roughness"].default_value, 0.45)
                    shader.inputs["Specular IOR Level"].default_value = 0.5
            for face in obj.data.polygons:
                face.use_smooth = True
    table_top = counter_material()
    legs, _ = principled("Table | dark steel legs", (0.05, 0.05, 0.055), 0.35, metallic=0.9)
    for obj in bpy.data.objects:
        if obj.name.startswith("table_") and obj.type == "MESH":
            assign(obj, table_top if obj.dimensions.z < 0.1 else legs)
    bpy.ops.mesh.primitive_plane_add(size=30, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "Floor"
    assign(floor, floor_material())
    world(scene, folder)
    for name, position, target, energy, size, color in LIGHTS:
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.size, light.color = energy, size, color
        obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(obj)
        obj.location = position
        aim(obj, target)
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.location = CAMERA["position"]
    aim(camera, CAMERA["target"])
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.angle = math.radians(CAMERA["fov"])
    camera.data.clip_start, camera.data.clip_end = 0.01, 100
    camera.data.dof.use_dof = True
    camera.data.dof.aperture_fstop = 5.6
    camera.data.dof.focus_distance = (Vector((0.52, 0.02, 1.05)) - camera.location).length
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for device_type in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = device_type
            prefs.get_devices()
            if any(d.type == device_type for d in prefs.devices):
                break
        except TypeError:
            continue
    for device in prefs.devices:
        device.use = device.type == prefs.compute_device_type
    scene.cycles.device = "GPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_use_gpu = True
    scene.cycles.max_bounces = 8
    scene.render.use_persistent_data = True
    scene.render.resolution_x, scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.fps = 30
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Punchy"
    scene.frame_start = 0
    scene.frame_set(0)
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "snack_basket.blend"))
    return scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "build/w1-snack-basket")
    parser.add_argument("--frames", default="all", help="Comma-separated frame numbers or 'all'.")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--resolution", type=int, nargs=2, default=(1920, 1080))
    parser.add_argument("--target", default="frames")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    folder = args.output.resolve()
    scene = build(folder, args)
    frames = (
        range(scene.frame_start, scene.frame_end + 1)
        if args.frames == "all"
        else [int(n) for n in args.frames.split(",")]
    )
    target = folder / args.target
    target.mkdir(exist_ok=True)
    for frame in frames:
        path = target / f"frame_{frame:04d}.png"
        if args.frames == "all" and path.exists():
            # Resume an interrupted full render without redoing finished frames.
            continue
        scene.frame_set(frame)
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        print(f"RENDERED_FRAME {frame}", flush=True)


if __name__ == "__main__":
    main()

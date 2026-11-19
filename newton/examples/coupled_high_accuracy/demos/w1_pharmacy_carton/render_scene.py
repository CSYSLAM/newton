"""Render the solver's actual paperboard vertices in a bright neighborhood pharmacy."""

import argparse
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

HERE = Path(__file__).resolve().parent
FONT = HERE.parents[1] / "assets/w1/fonts/NotoSansCJKsc-Regular.otf"
sys.path.insert(0, str(HERE.parent / "w1_courier_transport"))
from workshop_scene import box, material, unify_shell


def aim(obj, point):
    obj.rotation_euler = (Vector(point) - obj.location).to_track_quat("-Z", "Y").to_euler()


def comparison_labels(scene):
    font = bpy.data.fonts.load(str(FONT))
    for x, label, color in [
        (-0.365, "机器人右手 · 加力夹持", (0.24, 0.065, 0.014)),
        (0.07, "机器人左手 · 正常夹持", (0.015, 0.14, 0.095)),
    ]:
        mat = bpy.data.materials.new(label)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        shader = nodes.new("ShaderNodeEmission")
        shader.inputs[0].default_value = (1, 1, 1, 1)
        shader.inputs[1].default_value = 2
        mat.node_tree.links.new(shader.outputs[0], out.inputs["Surface"])
        curve = bpy.data.curves.new(label, "FONT")
        curve.body = label
        curve.font = font
        curve.size = 0.014
        curve.materials.append(mat)
        obj = bpy.data.objects.new(label, curve)
        scene.collection.objects.link(obj)
        obj.parent = scene.camera
        obj.location = (x, 0.205, -1)
        obj.visible_shadow = False
        obj.visible_diffuse = False
        obj.visible_glossy = False
        backing = bpy.data.materials.new(label + " backing")
        backing.use_nodes = True
        nodes = backing.node_tree.nodes
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs[0].default_value = (*color, 1)
        backing.node_tree.links.new(emission.outputs[0], out.inputs[0])
        mesh = bpy.data.meshes.new(label + " panel")
        mesh.from_pydata(
            [(-0.008, -0.009, 0), (0.287, -0.009, 0), (0.287, 0.021, 0), (-0.008, 0.021, 0)], [], [(0, 1, 2, 3)]
        )
        panel = bpy.data.objects.new(label + " panel", mesh)
        scene.collection.objects.link(panel)
        panel.parent = scene.camera
        panel.location = (x, 0.205, -1.001)
        mesh.materials.append(backing)
        panel.visible_shadow = False
        panel.visible_diffuse = False
        panel.visible_glossy = False


def carton_materials():
    materials = []
    for panel in range(6):
        mat = material(f"Carton print {panel}", (0.9, 0.92, 0.9), 0.58)
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        shader = nodes.get("Principled BSDF")
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(HERE / f"assets/carton_panel_{panel}.png"), check_existing=True)
        tex.image.pack()
        links.new(tex.outputs["Color"], shader.inputs["Base Color"])
        materials.append(mat)
    return materials


def dress_carton(obj, data, mats):
    obj.data.materials.clear()
    for mat in mats:
        obj.data.materials.append(mat)
    uv = obj.data.uv_layers.new(name="CartonPrint")
    uv.data.foreach_set("uv", data["uv"].reshape(-1).astype(np.float32))
    for face, panel in zip(obj.data.polygons, data["panel"]):
        face.material_index = int(panel)
    # The surface is directly animated by the physical shell; no shape keys or wrinkle displacement.
    shell = obj.modifiers.new("Paperboard edge thickness", "SOLIDIFY")
    shell.thickness = 0.00045
    shell.offset = -1


def pharmacy(scene):
    white = material("Pharmacy white lacquer", (0.82, 0.86, 0.85), 0.55)
    green = material("Pharmacy green", (0.025, 0.26, 0.19), 0.45)
    floor = material("Warm stone tile", (0.37, 0.39, 0.36), 0.82)
    wall = material("Painted wall", (0.70, 0.75, 0.71), 0.9)
    box("Floor", (0, 0, -0.05), (7, 7, 0.1), floor)
    box("Pharmacy back wall", (-1.2, 0, 1.5), (0.12, 4.5, 3), wall)
    box("Pharmacy side wall", (0, 2.1, 1.5), (4.5, 0.12, 3), wall)
    rng = random.Random(23)
    colors = [(0.74, 0.8, 0.78), (0.16, 0.43, 0.36), (0.16, 0.32, 0.5), (0.75, 0.42, 0.23)]
    stock = [material(f"Pharmacy stock {i}", c, 0.6) for i, c in enumerate(colors)]
    for y in (-0.95, 0.95):
        box("Cabinet back", (-0.98, y, 1.35), (0.07, 1.28, 2.05), white)
        for z in (0.45, 0.85, 1.25, 1.65, 2.05):
            box("Shelf", (-0.77, y, z), (0.47, 1.28, 0.025), white)
            for j in range(13):
                yy = y - 0.56 + j * 0.09
                size = (rng.uniform(0.045, 0.08), 0.065, rng.uniform(0.09, 0.16))
                obj = box(
                    "Medicine stock", (-0.72, yy, z + 0.014 + size[2] / 2), size, stock[rng.randrange(4)], bevel=0.001
                )
                label = box(
                    "Stock white label",
                    (-0.72 + size[0] / 2 + 0.0001, yy, obj.location.z),
                    (0.0005, 0.055, size[2] * 0.65),
                    white,
                    0,
                )
                box(
                    "Stock green band",
                    (label.location.x + 0.0003, yy, obj.location.z + size[2] * 0.18),
                    (0.0006, 0.055, 0.008),
                    green,
                    0,
                )
    box("Counter fascia", (0.49, 0, 0.68), (0.63, 1.42, 0.30), white)
    box("Counter green trim", (0.817, 0, 0.74), (0.012, 1.40, 0.08), green)
    box("Pharmacy sign", (-1.11, 0, 2.25), (0.04, 1.40, 0.32), green)
    for dims in ((0.025, 0.08, 0.24), (0.025, 0.24, 0.08)):
        box("Pharmacy cross", (-1.08, -0.43, 2.25), dims, white, 0)
    for dims in ((0.008, 0.035, 0.11), (0.008, 0.11, 0.035)):
        box("Counter cross", (0.829, -0.44, 0.68), dims, green, 0)
    plaque = box("Pharmacy name plaque", (-0.69, -1.02, 1.46), (0.018, 1.12, 0.20), green, 0.004)
    font = bpy.data.curves.new("Pharmacy sign lettering", "FONT")
    font.body = "青禾药房  PHARMACY"
    font.font = bpy.data.fonts.load(str(FONT))
    font.align_x = "CENTER"
    font.size = 0.08
    text = bpy.data.objects.new("Pharmacy sign lettering", font)
    scene.collection.objects.link(text)
    text.location = (-0.678, -1.02, 1.43)
    text.rotation_euler = (np.pi / 2, 0, np.pi / 2)
    font.materials.append(white)


def build(folder):
    bpy.context.preferences.view.language = "en_US"
    bpy.context.preferences.view.use_translate_new_dataname = False
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.usd_import(filepath=str(folder / "simulation.usdc"), import_cameras=False, import_lights=False)
    scene = bpy.context.scene
    unify_shell()
    data = np.load(folder / "trajectory.npz")
    mats = carton_materials()
    for name in ("carton_left_gentle", "carton_right_crush"):
        dress_carton(bpy.data.objects[name], data, mats)
    pharmacy(scene)
    world = bpy.data.worlds.new("Pharmacy daylight")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.82, 0.9, 1, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.15
    scene.world = world
    for name, pos, power, size in [
        ("Storefront", (1, -2, 2.8), 130, 2),
        ("Ceiling", (-0.4, 0.8, 2.6), 110, 1.5),
        ("Carton rim", (1.5, 1.2, 1.9), 170, 1),
    ]:
        light = bpy.data.lights.new(name, "AREA")
        light.energy = power
        light.shape = "DISK"
        light.size = size
        obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(obj)
        obj.location = pos
        aim(obj, (0.45, 0, 1.1))
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.location = (1.9, -0.9, 1.6)
    aim(camera, (0.25, 0, 1.17))
    camera.data.lens = 38
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "OPTIX"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 8
    scene.cycles.use_denoising = True
    scene.cycles.denoising_use_gpu = True
    scene.cycles.max_bounces = 4
    scene.render.use_persistent_data = True
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.fps = 30
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = -0.5
    comparison_labels(scene)
    return scene


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", default="120,210")
    args = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    folder = args.output.resolve()
    scene = build(folder)
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "pharmacy_scene.blend"))
    (folder / "frames").mkdir(exist_ok=True)
    for frame in map(int, args.frames.split(",")):
        scene.frame_set(frame)
        scene.render.filepath = str(folder / "frames" / f"frame_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)

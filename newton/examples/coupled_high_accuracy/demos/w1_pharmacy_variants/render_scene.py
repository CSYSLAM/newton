"""The new four-carton comparison, rendered from physical states in the pharmacy."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[1] / "assets/w1/pharmacy"
FONT = HERE.parents[1] / "assets/w1/fonts/NotoSansCJKsc-Regular.otf"
sys.path.insert(0, str(HERE))
from catalog import Carton

spec = importlib.util.spec_from_file_location(
    "pharmacy_reference_render", HERE.parent / "w1_pharmacy_carton/render_scene.py"
)
store = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = store
spec.loader.exec_module(store)


def material(name, color, roughness=0.6):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    shader = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    return mat


def carton_materials(case):
    materials = []
    for panel in range(6):
        mat = material(f"{case.key} print {panel}", (0.9, 0.92, 0.9), 0.60)
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        shader = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
        image = nodes.new("ShaderNodeTexImage")
        image.image = bpy.data.images.load(str(ASSETS / f"{case.key}_panel_{panel}.png"), check_existing=True)
        image.image.pack()
        links.new(image.outputs["Color"], shader.inputs["Base Color"])
        materials.append(mat)
    return materials


def dress(obj, asset, case, solidify=True):
    obj.data.materials.clear()
    for mat in carton_materials(case):
        obj.data.materials.append(mat)
    uv = obj.data.uv_layers.new(name="VariantPrint")
    uv.data.foreach_set("uv", asset["uv"].reshape(-1).astype(np.float32))
    for face, panel in zip(obj.data.polygons, asset["panel"]):
        face.material_index = int(panel)
    if solidify:
        modifier = obj.modifiers.new("Paperboard edge thickness", "SOLIDIFY")
        modifier.thickness = case.thickness
        modifier.offset = -1


def captions(scene, cases):
    font = bpy.data.fonts.load(str(FONT))
    # Robot right is screen left from this fixed pharmacy camera.
    for x, case in [(-0.365, cases[1]), (0.07, cases[0])]:
        label = f"{case.name} · {case.behavior}"
        textmat = bpy.data.materials.new(label)
        textmat.use_nodes = True
        nodes = textmat.node_tree.nodes
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs[0].default_value = (1, 1, 1, 1)
        emission.inputs[1].default_value = 2
        textmat.node_tree.links.new(emission.outputs[0], out.inputs[0])
        curve = bpy.data.curves.new(label, "FONT")
        curve.body = label
        curve.font = font
        curve.size = 0.013
        curve.materials.append(textmat)
        obj = bpy.data.objects.new(label, curve)
        scene.collection.objects.link(obj)
        obj.parent = scene.camera
        obj.location = (x, 0.205, -1)
        obj.visible_shadow = False
        obj.visible_diffuse = False
        obj.visible_glossy = False
        mat = bpy.data.materials.new(label + " background")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        out = nodes.new("ShaderNodeOutputMaterial")
        emission = nodes.new("ShaderNodeEmission")
        color = tuple((v / 255) ** 2.2 * 0.7 for v in case.color)
        emission.inputs[0].default_value = (*color, 1)
        mat.node_tree.links.new(emission.outputs[0], out.inputs[0])
        mesh = bpy.data.meshes.new(label + " backing")
        mesh.from_pydata(
            [(-0.008, -0.009, 0), (0.287, -0.009, 0), (0.287, 0.021, 0), (-0.008, 0.021, 0)], [], [(0, 1, 2, 3)]
        )
        panel = bpy.data.objects.new(label + " backing", mesh)
        scene.collection.objects.link(panel)
        panel.parent = scene.camera
        panel.location = (x, 0.205, -1.001)
        mesh.materials.append(mat)
        panel.visible_shadow = False
        panel.visible_diffuse = False
        panel.visible_glossy = False


def build(folder, samples=4):
    bpy.context.preferences.view.language = "en_US"
    bpy.context.preferences.view.use_translate_new_dataname = False
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.wm.usd_import(filepath=str(folder / "simulation.usdc"), import_cameras=False, import_lights=False)
    scene = bpy.context.scene
    store.unify_shell()
    data = np.load(folder / "trajectory.npz")
    report = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    cases = [Carton(**case) for case in report["cases"]]
    for i, case in enumerate(cases):
        dress(bpy.data.objects[f"carton_{i}"], dict(uv=data[f"uv_{i}"], panel=data[f"panel_{i}"]), case)
    store.pharmacy(scene)
    bpy.data.objects["Pharmacy name plaque"].location = (-0.69, 0.78, 1.75)
    bpy.data.objects["Pharmacy sign lettering"].location = (-0.678, 0.78, 1.72)
    world = bpy.data.worlds.new("Dispensary daylight")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.82, 0.9, 1, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.15
    scene.world = world
    for name, pos, power, size in [
        ("Storefront", (1, -2, 2.8), 130, 2),
        ("Ceiling", (-0.4, 0.8, 2.6), 110, 1.5),
        ("Board edge light", (1.5, 1.2, 1.9), 170, 1),
    ]:
        light = bpy.data.lights.new(name, "AREA")
        light.energy = power
        light.shape = "DISK"
        light.size = size
        obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(obj)
        obj.location = pos
        store.aim(obj, (0.45, 0, 1.1))
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.location = (1.9, -0.9, 1.6)
    store.aim(camera, (0.25, 0, 1.17))
    camera.data.lens = 38
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "OPTIX"
    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
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
    captions(scene, cases)
    return scene


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", default="90,180,330")
    p.add_argument("--samples", type=int, default=4)
    args = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    folder = args.output.resolve()
    scene = build(folder, args.samples)
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "pharmacy_variants.blend"))
    (folder / "frames").mkdir(exist_ok=True)
    for frame in map(int, args.frames.split(",")):
        scene.frame_set(frame)
        scene.render.filepath = str(folder / "frames" / f"frame_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)

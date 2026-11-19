"""Import the simulation USD and save an editable Cycles studio project.

Run with Blender: blender -b --python build_render_scene.py -- [--config render_config.json]
The source USD is an external relative cache. Keep it next to the saved .blend file.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from pxr import Usd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "utils"))
from usd_cache import record_usd_sources


def material(name, color, metallic=0.0, roughness=0.3):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color[:3], 1.0)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    result.diffuse_color = (*color[:3], 1.0)
    return result, shader


def source_color(obj):
    if obj.data.materials and obj.data.materials[0]:
        mat = obj.data.materials[0]
        if mat.use_nodes:
            shader = mat.node_tree.nodes.get("Principled BSDF")
            if shader:
                return tuple(shader.inputs["Base Color"].default_value[:3])
        return tuple(mat.diffuse_color[:3])
    attribute = obj.data.color_attributes.get("displayColor")
    if attribute and len(attribute.data):
        return tuple(attribute.data[0].color[:3])
    return (0.55, 0.58, 0.62)


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def linear_rgb(color):
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in color)


def configure_camera_fov(camera, cfg):
    """Use horizontal FOV in degrees, with focal-length compatibility for older configs."""
    camera.sensor_width = 36.0
    if "fov" in cfg:
        fov = float(cfg["fov"])
        if not math.isfinite(fov) or not 0.0 < fov < 180.0:
            raise ValueError("camera.fov must be a finite horizontal angle between 0 and 180 degrees")
        # Fix the horizontal fit so changing the render aspect ratio does not redefine the configured FOV.
        camera.sensor_fit = "HORIZONTAL"
        camera.lens_unit = "FOV"
        camera.angle = math.radians(fov)
    else:
        camera.lens = cfg.get("lens", 48.0)


def apply_color_management(scene, cfg):
    # Use Blender's OCIO output transform so both the viewport and saved PNGs receive the same tone mapping.
    scene.view_settings.view_transform = cfg.get("view_transform", "ACES 2.0")
    scene.view_settings.look = cfg.get("look", "None")
    scene.view_settings.exposure = cfg["exposure"]
    # This is an extra artistic adjustment; the OCIO display transform already handles sRGB output encoding.
    scene.view_settings.gamma = cfg.get("gamma", 1.0)


def select_gpu():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    found = False
    for device in prefs.devices:
        device.use = device.type == "OPTIX"
        found |= device.use
    if not found:
        raise RuntimeError("An OptiX GPU is required for this render preset")
    print("Render devices:", [(d.name, d.type) for d in prefs.devices if d.use], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "render_config.json")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    usd_path = (ROOT / cfg["usd"]).resolve()
    stage = Usd.Stage.Open(str(usd_path))
    source_fps = stage.GetTimeCodesPerSecond()
    # Catch stale export timing before replacing the scene or saving an incompatible Blender project.
    if not math.isclose(cfg["simulation_fps"], source_fps, rel_tol=0.0, abs_tol=1.0e-8):
        raise ValueError(
            f"Sampling rate mismatch: render_config simulation_fps={cfg['simulation_fps']}, "
            f"USD timeCodesPerSecond={source_fps:g}. Re-export with export_simulation.py after changing simulation FPS."
        )
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    apply_color_management(scene, cfg)
    # Preserve simulation timing on import; playback speed changes belong to the video encoder.
    scene.render.fps = round(source_fps)
    scene.render.fps_base = scene.render.fps / source_fps
    bpy.ops.wm.usd_import(filepath=str(usd_path), import_materials=True, set_frame_range=True)
    scene.render.fps = round(source_fps)
    scene.render.fps_base = scene.render.fps / source_fps
    scene.frame_start, scene.frame_end = cfg["frame_start"], cfg["frame_end"]
    studio = bpy.data.collections.new("Studio | camera, lights, floor")
    scene.collection.children.link(studio)

    mats, counts = {}, {"jelly": 0, "robot": 0}
    for obj in list(scene.objects):
        if obj.type != "MESH":
            continue
        color = source_color(obj)
        if obj.name.lower().startswith("jelly_"):
            key = tuple(round(c, 4) for c in color)
            if key not in mats:
                mat, shader = material(
                    f"Jelly | layer color {len(mats) + 1}",
                    linear_rgb(color),
                    roughness=cfg["jelly"]["roughness"],
                )
                shader.inputs["Transmission Weight"].default_value = cfg["jelly"]["transmission"]
                shader.inputs["IOR"].default_value = cfg["jelly"]["ior"]
                shader.inputs["Subsurface Weight"].default_value = cfg["jelly"]["subsurface"]
                shader.inputs["Subsurface Radius"].default_value = (0.12, 0.06, 0.035)
                shader.inputs["Coat Weight"].default_value = cfg["jelly"]["coat"]
                shader.inputs["Coat Roughness"].default_value = 0.12
                absorption = mat.node_tree.nodes.new("ShaderNodeVolumeAbsorption")
                absorption.inputs["Color"].default_value = (*color, 1)
                absorption.inputs["Density"].default_value = cfg["jelly"]["absorption_density"]
                mat.node_tree.links.new(absorption.outputs[0], mat.node_tree.nodes["Material Output"].inputs["Volume"])
                mats[key] = mat
            assign(obj, mats[key])
            # Smooth the render surface after the cache; the exported simulation remains unmodified.
            subd = obj.modifiers.new("Render surface smoothing", "SUBSURF")
            subd.levels = 1
            subd.render_levels = cfg["jelly"]["subdivision"]
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
            counts["jelly"] += 1
        elif "floor" in obj.name.lower():
            obj.hide_render = True
        elif "marking" in obj.name.lower():
            # Hide markings from older USD exports while retaining their objects for cache validation.
            obj.hide_render = True
            obj.hide_set(True)
        else:
            dark = max(color) < 0.3
            # Older configs retain the original per-color roughness until a uniform override is supplied.
            mat, shader = material(
                f"Robot | {obj.name}",
                linear_rgb(color),
                metallic=cfg.get("robot_metallic", 0.05),
                roughness=cfg.get("robot_roughness", 0.30 if dark else 0.26),
            )
            shader.inputs["Coat Weight"].default_value = 0.15
            assign(obj, mat)
            for polygon in obj.data.polygons:
                polygon.use_smooth = True
            counts["robot"] += 1

    floor_mat, floor_shader = material(
        "Studio | graphite satin",
        cfg["floor_color"],
        metallic=cfg.get("floor_metallic", 0.0),
        roughness=cfg.get("floor_roughness", 0.65),
    )
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -0.009))
    floor = bpy.context.object
    floor.name = "Studio floor"
    assign(floor, floor_mat)
    for collection in list(floor.users_collection):
        collection.objects.unlink(floor)
    studio.objects.link(floor)
    world = bpy.data.worlds.new("Studio ambient")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.38, 0.46, 0.62, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = cfg["world_strength"]
    scene.world = world
    for spec in cfg["lights"]:
        light = bpy.data.lights.new(spec["name"], "AREA")
        light.energy, light.size, light.color = spec["energy"], spec["size"], spec["color"]
        obj = bpy.data.objects.new(spec["name"], light)
        studio.objects.link(obj)
        obj.location = spec["position"]
        aim(obj, spec["target"])

    cam_data = bpy.data.cameras.new("Hero camera")
    camera = bpy.data.objects.new("Hero camera", cam_data)
    studio.objects.link(camera)
    scene.camera = camera
    configure_camera_fov(cam_data, cfg["camera"])
    cam_data.clip_end = 500
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = cfg["camera"]["fstop"]
    for key in cfg["camera"]["keys"]:
        camera.location = key["position"]
        aim(camera, key["target"])
        cam_data.dof.focus_distance = (Vector(key["target"]) - camera.location).length
        camera.keyframe_insert(data_path="location", frame=key["frame"])
        camera.keyframe_insert(data_path="rotation_euler", frame=key["frame"])
        cam_data.keyframe_insert(data_path="dof.focus_distance", frame=key["frame"])

    scene.render.engine = "CYCLES"
    select_gpu()
    scene.cycles.device = "GPU"
    scene.cycles.samples = cfg["samples"]
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = cfg["adaptive_threshold"]
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 10
    scene.cycles.glossy_bounces = 6
    scene.cycles.diffuse_bounces = 4
    scene.cycles.transparent_max_bounces = 12
    scene.cycles.sample_clamp_indirect = 4.0
    scene.render.use_persistent_data = True
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = cfg["motion_blur_shutter"]
    scene.render.resolution_x, scene.render.resolution_y = cfg["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.image_settings.compression = 25
    scene.render.filepath = "//frames/frame_"
    scene.frame_set(0)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"
    scene["simulation_cache"] = (
        f"{usd_path.name} | {source_fps:g} Hz physics cache, frames {scene.frame_start} to {scene.frame_end}"
    )
    scene["video_playback"] = (
        f"Encode {scene.frame_end - scene.frame_start + 1} rendered frames at {cfg['video_fps']} fps "
        f"for {source_fps / cfg['video_fps']:.3g}x playback duration"
    )
    scene["render_config"] = args.config.name
    # Establish the project directory before resolving Blender's // relative cache paths.
    (ROOT / cfg["project"]).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / cfg["project"]))
    bpy.ops.file.make_paths_relative()
    record_usd_sources(scene)
    bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / cfg["project"]))
    print("Prepared scene:", counts, "frames", scene.frame_start, scene.frame_end, flush=True)
    if not counts["jelly"] or not counts["robot"]:
        raise RuntimeError(f"USD import did not contain the expected geometry: {counts}")


if __name__ == "__main__":
    main()

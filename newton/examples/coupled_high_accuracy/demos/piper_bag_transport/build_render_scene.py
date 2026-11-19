"""Build an editable Cycles project from the coupling USD. Run with Blender --background --python."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector
from pxr import Usd

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY))
sys.path.insert(0, str(DIRECTORY.parent / "utils"))
from film_uv import FILM_UV, assign_film_uv
from render_frames import select_device
from render_provenance import load_run, validate_scene_run
from usd_cache import record_usd_sources


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def material(name, color, roughness):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1.0)
    shader.inputs["Roughness"].default_value = roughness
    mat.diffuse_color = (*color, 1.0)
    return mat, shader


def bag_material(spec, texture_directory=DIRECTORY):
    mat = bpy.data.materials.new("Bag | clear pale yellow plastic film")
    mat.use_nodes = True
    mat.diffuse_color = (*spec["color"], 1.0)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (970, 160)
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    transparent.label = "Straight-through light; no refraction"
    transparent.location = (-440, 180)
    transparent.inputs["Color"].default_value = (*spec["transmitted_color"], 1.0)
    translucent = nodes.new("ShaderNodeBsdfTranslucent")
    translucent.label = "Subtle diffuse haze in the thin film"
    translucent.location = (-440, 0)
    translucent.inputs["Color"].default_value = (*spec["color"], 1.0)
    transmission = nodes.new("ShaderNodeMixShader")
    transmission.location = (-140, 120)
    transmission.inputs[0].default_value = spec["transparency"]
    links.new(translucent.outputs[0], transmission.inputs[1])
    links.new(transparent.outputs[0], transmission.inputs[2])
    reflection = nodes.new("ShaderNodeBsdfAnisotropic")
    reflection.label = "Crisp highlights along the smooth film and creases"
    reflection.location = (-140, -100)
    reflection.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    reflection.inputs["Roughness"].default_value = spec["roughness"]
    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = FILM_UV
    uv.label = "Developed film UVs follow the cloth deformation"
    uv.location = (-1000, -800)
    normal_texture = nodes.new("ShaderNodeTexImage")
    normal_texture.label = "Crumpled plastic | OpenGL tangent normals"
    normal_texture.location = (-780, -800)
    normal_texture.extension = "REPEAT"
    normal_path = (texture_directory / spec["normal_map"]).resolve()
    normal_texture.image = bpy.data.images.load(str(normal_path), check_existing=True)
    normal_texture.image.colorspace_settings.name = "Non-Color"
    # Pack the data texture so moving the generated Blender project cannot lose its film detail.
    normal_texture.image.pack()
    links.new(uv.outputs["UV"], normal_texture.inputs["Vector"])
    normal = nodes.new("ShaderNodeNormalMap")
    normal.space = "TANGENT"
    normal.uv_map = FILM_UV
    normal.label = "Crumpled film normal strength"
    normal.location = (-360, -800)
    normal.inputs["Strength"].default_value = spec["normal_strength"]
    links.new(normal_texture.outputs["Color"], normal.inputs["Color"])
    wrinkles = nodes.new("ShaderNodeTexNoise")
    wrinkles.label = "Small irregularities in the smooth extruded film"
    wrinkles.location = (-780, -380)
    wrinkles.inputs["Scale"].default_value = spec["wrinkle_scale"] * spec["normal_tile_size_m"]
    wrinkles.inputs["Detail"].default_value = 3.0
    wrinkles.inputs["Roughness"].default_value = 0.65
    links.new(uv.outputs["UV"], wrinkles.inputs["Vector"])
    bump = nodes.new("ShaderNodeBump")
    bump.label = "Crease normal map with subordinate surface waviness"
    bump.location = (-100, -420)
    bump.inputs["Strength"].default_value = spec["wrinkle_strength"]
    bump.inputs["Distance"].default_value = spec["wrinkle_distance"]
    links.new(wrinkles.outputs["Fac"], bump.inputs["Height"])
    links.new(normal.outputs["Normal"], bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], reflection.inputs["Normal"])
    links.new(bump.outputs["Normal"], translucent.inputs["Normal"])
    fresnel = nodes.new("ShaderNodeFresnel")
    fresnel.label = "Polyethylene interface: reflected creases and folded edges"
    fresnel.location = (-100, -660)
    fresnel.inputs["IOR"].default_value = spec["ior"]
    links.new(bump.outputs["Normal"], fresnel.inputs["Normal"])
    geometry = nodes.new("ShaderNodeNewGeometry")
    geometry.location = (-780, -1100)
    interface_ior = nodes.new("ShaderNodeMath")
    interface_ior.operation = "MULTIPLY_ADD"
    interface_ior.location = (-360, -1060)
    interface_ior.inputs[1].default_value = 1.0 / spec["ior"] - spec["ior"]
    interface_ior.inputs[2].default_value = spec["ior"]
    # Both sides of a zero-thickness sheet face air. Cancel Fresnel's automatic backface IOR inversion.
    links.new(geometry.outputs["Backfacing"], interface_ior.inputs[0])
    links.new(interface_ior.outputs[0], fresnel.inputs["IOR"])
    double_interface = nodes.new("ShaderNodeMath")
    double_interface.operation = "MULTIPLY"
    double_interface.location = (100, -640)
    double_interface.inputs[1].default_value = 2.0
    links.new(fresnel.outputs["Fac"], double_interface.inputs[0])
    denominator = nodes.new("ShaderNodeMath")
    denominator.operation = "ADD"
    denominator.location = (100, -800)
    denominator.inputs[1].default_value = 1.0
    links.new(fresnel.outputs["Fac"], denominator.inputs[0])
    reflectance = nodes.new("ShaderNodeMath")
    reflectance.operation = "DIVIDE"
    reflectance.location = (310, -640)
    # An unresolved air/film/air sheet reflects 2F/(1+F); straight transmission avoids a solid-glass appearance.
    links.new(double_interface.outputs[0], reflectance.inputs[0])
    links.new(denominator.outputs[0], reflectance.inputs[1])
    reflection_gain = nodes.new("ShaderNodeMath")
    reflection_gain.operation = "MULTIPLY"
    reflection_gain.use_clamp = True
    reflection_gain.location = (520, -400)
    single_f0 = ((spec["ior"] - 1.0) / (spec["ior"] + 1.0)) ** 2
    reflection_gain.inputs[1].default_value = spec["reflection"] / (2.0 * single_f0 / (1.0 + single_f0))
    links.new(reflectance.outputs[0], reflection_gain.inputs[0])
    film = nodes.new("ShaderNodeMixShader")
    film.location = (740, 160)
    links.new(reflection_gain.outputs[0], film.inputs[0])
    links.new(transmission.outputs[0], film.inputs[1])
    links.new(reflection.outputs[0], film.inputs[2])
    links.new(film.outputs[0], output.inputs["Surface"])
    return mat


def studio_world(cfg, texture_directory):
    """Light and reflect the studio HDRI while camera rays see a quiet solid background."""
    world = bpy.data.worlds.new("Studio | HDR lighting with separate camera background")
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    nodes.clear()
    illumination = nodes.new("ShaderNodeBackground")
    illumination.location = (0, 180)
    illumination.inputs["Color"].default_value = (*cfg["world_color"], 1.0)
    illumination.inputs["Strength"].default_value = cfg["world_strength"]
    environment = cfg.get("hdri")
    if environment and environment.get("file"):
        texture = nodes.new("ShaderNodeTexEnvironment")
        texture.location = (-260, 180)
        texture.image = bpy.data.images.load(
            str((texture_directory / environment["file"]).resolve()), check_existing=True
        )
        texture.image.pack()
        mapping = nodes.new("ShaderNodeMapping")
        mapping.location = (-500, 180)
        mapping.inputs["Rotation"].default_value[2] = math.radians(environment.get("rotation_degrees", 0.0))
        coordinates = nodes.new("ShaderNodeTexCoord")
        coordinates.location = (-720, 180)
        links.new(coordinates.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
        links.new(texture.outputs["Color"], illumination.inputs["Color"])
        illumination.inputs["Strength"].default_value = environment.get("strength", cfg["world_strength"])
    background = nodes.new("ShaderNodeBackground")
    background.label = "Solid background visible to direct camera rays only"
    background.location = (0, -40)
    background.inputs["Color"].default_value = (*cfg["world_color"], 1.0)
    background.inputs["Strength"].default_value = cfg.get("background_strength", 0.5)
    path = nodes.new("ShaderNodeLightPath")
    path.location = (0, -260)
    mix = nodes.new("ShaderNodeMixShader")
    mix.location = (300, 120)
    links.new(path.outputs["Is Camera Ray"], mix.inputs[0])
    links.new(illumination.outputs[0], mix.inputs[1])
    links.new(background.outputs[0], mix.inputs[2])
    output = nodes.new("ShaderNodeOutputWorld")
    output.location = (540, 120)
    links.new(mix.outputs[0], output.inputs["Surface"])
    return world


def environment_material(name, spec):
    mat, shader = material(name, spec["color"], spec["roughness"])
    shader.inputs["Specular IOR Level"].default_value = 0.3
    shader.inputs["Metallic"].default_value = spec.get("metallic", 0.0)
    # Round only the shading normals, leaving the exported contact geometry unchanged.
    bevel = mat.node_tree.nodes.new("ShaderNodeBevel")
    bevel.inputs["Radius"].default_value = spec["edge_radius"]
    bevel.samples = 4
    mat.node_tree.links.new(bevel.outputs["Normal"], shader.inputs["Normal"])
    return mat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DIRECTORY / "render_config.json")
    parser.add_argument("--usd", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--device", choices=("CPU", "OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"))
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    config = args.config.resolve()
    cfg = json.loads(config.read_text(encoding="utf-8"))
    usd = (args.usd or config.parent / cfg["usd"]).resolve()
    project = (args.project or config.parent / cfg["project"]).resolve()
    stage = Usd.Stage.Open(str(usd))
    fps = stage.GetTimeCodesPerSecond()
    if not math.isclose(cfg["simulation_fps"], fps, abs_tol=1.0e-8, rel_tol=0.0):
        raise ValueError(f"USD FPS {fps} does not match simulation_fps {cfg['simulation_fps']}")
    start = cfg.get("frame_start", int(stage.GetStartTimeCode()))
    end = cfg.get("frame_end")
    end = int(stage.GetEndTimeCode()) if end is None else end
    if not stage.GetStartTimeCode() <= start <= end <= stage.GetEndTimeCode():
        raise ValueError("Configured frame range is outside the exported USD animation")
    fov = float(cfg["camera"]["fov"])
    if not math.isfinite(fov) or not 0 < fov < 180:
        raise ValueError("camera.fov must be between 0 and 180 degrees")
    for name in ("transparency", "reflection", "roughness", "wrinkle_strength"):
        if not 0 <= cfg["bag"][name] <= 1:
            raise ValueError(f"bag.{name} must be between 0 and 1")
    for name in ("wrinkle_scale", "wrinkle_distance"):
        if not math.isfinite(cfg["bag"][name]) or cfg["bag"][name] <= 0:
            raise ValueError(f"bag.{name} must be finite and positive")
    if not math.isfinite(cfg["bag"]["ior"]) or cfg["bag"]["ior"] <= 1.0:
        raise ValueError("bag.ior must be finite and greater than one")
    if not math.isfinite(cfg["bag"]["normal_strength"]) or cfg["bag"]["normal_strength"] < 0:
        raise ValueError("bag.normal_strength must be finite and non-negative")
    if not math.isfinite(cfg["bag"]["normal_tile_size_m"]) or cfg["bag"]["normal_tile_size_m"] <= 0:
        raise ValueError("bag.normal_tile_size_m must be finite and positive")
    if not (config.parent / cfg["bag"]["normal_map"]).is_file():
        raise FileNotFoundError(f"Bag normal map not found: {config.parent / cfg['bag']['normal_map']}")
    environment = cfg.get("hdri")
    if environment and environment.get("file") and not (config.parent / environment["file"]).is_file():
        raise FileNotFoundError(f"Studio HDRI not found: {config.parent / environment['file']}")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base = round(fps), round(fps) / fps
    bpy.ops.wm.usd_import(filepath=str(usd), import_materials=True, set_frame_range=True)
    scene.render.fps, scene.render.fps_base = round(fps), round(fps) / fps
    scene.frame_start, scene.frame_end = start, end
    counts = {"bag": 0, "apple": 0, "robot": 0, "environment": 0}
    converted_materials = set()
    environment_names = set()
    for obj in list(scene.objects):
        if obj.type != "MESH":
            continue
        if obj.name == "plastic_bag":
            counts["bag"] += 1
            spec = cfg["bag"]
            assign_film_uv(obj, spec["normal_tile_size_m"])
            obj.data.materials.clear()
            obj.data.materials.append(bag_material(spec, config.parent))
            if spec["subdivision"]:
                subd = obj.modifiers.new("Render-only film smoothing", "SUBSURF")
                subd.levels = subd.render_levels = spec["subdivision"]
        elif obj.name in ("scene_floor", "bag_platform"):
            counts["environment"] += 1
            environment_names.add(obj.name)
            spec = cfg["environment"][obj.name]
            obj.data.materials.clear()
            obj.data.materials.append(environment_material(obj.name, spec))
        else:
            kind = "apple" if obj.name.startswith("apple_visual_") else "robot"
            counts[kind] += 1
            for mat in obj.data.materials:
                if not mat or not mat.use_nodes or mat.name in converted_materials:
                    continue
                converted_materials.add(mat.name)
                shader = mat.node_tree.nodes.get("Principled BSDF")
                if shader:
                    # Preserve the original apple color-texture connections and Piper's individual part colors.
                    shader.inputs["Roughness"].default_value = cfg[f"{kind}_roughness"]
                    shader.inputs["Coat Weight"].default_value = (
                        cfg.get("apple_coat_weight", 0.05) if kind == "apple" else 0.2
                    )
                    if kind == "robot":
                        shader.inputs["Metallic"].default_value = cfg["robot_metallic"]
                        # MJCF part colors are display RGB; Cycles expects linear-light base colors.
                        color = shader.inputs["Base Color"].default_value[:3]
                        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in color]
                        shader.inputs["Base Color"].default_value = (*linear, 1.0)
        for face in obj.data.polygons:
            face.use_smooth = obj.name not in environment_names
    if (
        counts["bag"] != 1
        or counts["apple"] != 5
        or counts["robot"] < 1
        or not {"scene_floor", "bag_platform"}.issubset(environment_names)
    ):
        raise RuntimeError(f"Incomplete USD import: {counts}")
    scene.world = studio_world(cfg, config.parent)
    for spec in cfg["lights"]:
        light = bpy.data.lights.new(spec["name"], "AREA")
        light.energy, light.size, light.color = spec["energy"], spec["size"], spec["color"]
        light.shape = spec.get("shape", "SQUARE")
        if light.shape == "RECTANGLE":
            # Long, narrow emitters form bright streaks on the film's curved folds without tinting them.
            light.size_y = spec["size_y"]
        obj = bpy.data.objects.new(spec["name"], light)
        scene.collection.objects.link(obj)
        obj.location = spec["position"]
        aim(obj, spec["target"])
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.location = cfg["camera"]["position"]
    aim(camera, cfg["camera"]["target"])
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.angle = math.radians(fov)
    camera.data.clip_start, camera.data.clip_end = 0.01, 300
    camera.data.dof.use_dof = True
    camera.data.dof.aperture_fstop = cfg["camera"]["fstop"]
    camera.data.dof.focus_distance = (Vector(cfg["camera"]["target"]) - camera.location).length
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    select_device(scene, args.device)
    scene.cycles.samples = cfg["samples"]
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = cfg["adaptive_threshold"]
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_use_gpu = scene.cycles.device == "GPU"
    scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 8
    # Gussets contain several closely folded film layers; early termination makes them look opaque black.
    scene.cycles.transparent_max_bounces = 32
    scene.render.use_persistent_data = True
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = cfg["motion_blur_shutter"]
    scene.render.resolution_x, scene.render.resolution_y = cfg["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.render.filepath = "//frames/frame_"
    scene.view_settings.view_transform = cfg["view_transform"]
    scene.view_settings.look = cfg["look"]
    scene.view_settings.exposure = cfg["exposure"]
    scene.view_settings.gamma = cfg["gamma"]
    scene.frame_set(start)
    scene["render_config"] = config.name
    scene["simulation_fps"] = fps
    scene["video_fps"] = cfg["video_fps"]
    project.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(project))
    bpy.ops.file.make_paths_relative()
    record_usd_sources(scene)
    run_path = usd.parent / "run_manifest.json"
    if run_path.exists():
        run = load_run(run_path)
        validate_scene_run(run, list(json.loads(scene["usd_source_sha256"]).values()), start, end, fps)
        scene["run_id"] = run["run_id"]
    bpy.ops.wm.save_as_mainfile(filepath=str(project))
    print(f"Prepared {counts}, frames {start}..{end}: {project}", flush=True)


if __name__ == "__main__":
    main()

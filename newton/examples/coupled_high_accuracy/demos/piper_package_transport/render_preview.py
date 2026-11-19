"""Build a scene from baked simulation and render selected frames or the full animation."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

DIRECTORY = Path(__file__).resolve().parent
REFERENCE = DIRECTORY.parents[1] / "demos/piper_bag_transport"
sys.path.insert(0, str(REFERENCE))
from build_render_scene import bag_material, environment_material, studio_world
from film_uv import FILM_PATTERN, FILM_UV, assign_film_uv


def reference_style():
    config = json.loads((REFERENCE / "render_config.json").read_text(encoding="utf-8"))
    files = (
        "render_config.json",
        "build_render_scene.py",
        "film_uv.py",
        config["bag"]["normal_map"],
        config["hdri"]["file"],
    )
    hashes = {name: hashlib.sha256((REFERENCE / name).read_bytes()).hexdigest() for name in files}
    return config, hashes


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def material(name, color, roughness):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    return mat, shader


def film_coordinates(obj, panel_vertices, panel_faces, tile_size_m):
    # The closed envelope shares seam vertices, but each cutting panel needs its own developed UV chart.
    if panel_faces.shape != (len(obj.data.polygons), 3):
        raise ValueError(f"Film pattern does not match recorded topology: {obj.name}")
    pattern = obj.data.uv_layers.new(name=FILM_PATTERN)
    pattern.data.foreach_set("uv", panel_vertices[panel_faces].ravel())
    assign_film_uv(obj, tile_size_m)


def shipping_label(obj):
    """Printed film uses the simulated faces and persistent UVs, so its face marker turns with the parcel."""
    points = np.array([vertex.co[:] for vertex in obj.data.vertices])
    center = points.mean(0)
    xy = points[:, :2] - center[:2]
    size = np.ptp(xy, axis=0)
    coordinates = xy / size + 0.5
    uv = obj.data.uv_layers.new(name="ShippingPrint")
    for face in obj.data.polygons:
        for loop in face.loop_indices:
            uv.data[loop].uv = coordinates[obj.data.loops[loop].vertex_index]
    mat, shader = material("Shipping label | " + obj.name, (0.93, 0.93, 0.88), 0.65)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    texcoord = nodes.new("ShaderNodeUVMap")
    texcoord.uv_map = "ShippingPrint"
    wave = nodes.new("ShaderNodeTexWave")
    wave.bands_direction = "Y"
    wave.inputs["Scale"].default_value = 38
    wave.inputs["Distortion"].default_value = 1.2
    links.new(texcoord.outputs["UV"], wave.inputs["Vector"])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[0].color = (0.03, 0.04, 0.05, 1)
    ramp.color_ramp.elements[1].position = 0.50
    ramp.color_ramp.elements[1].color = (0.93, 0.93, 0.88, 1)
    links.new(wave.outputs[0], ramp.inputs[0])
    links.new(ramp.outputs[0], shader.inputs["Base Color"])
    obj.data.materials.append(mat)
    for face in obj.data.polygons:
        ids = list(face.vertices)
        c = coordinates[ids].mean(0)
        if face.index < len(obj.data.polygons) // 2 and abs(c[0] - 0.5) < 0.25 and abs(c[1] - 0.5) < 0.14:
            face.material_index = 1


def restore_piper_finish(folder):
    """Keep imported colors/UVs and restore MJCF finish fields omitted by Newton's visual importer."""
    appearances = json.loads((folder / "piper_materials.json").read_text())["meshes"]
    for name, appearance in appearances.items():
        obj = bpy.data.objects.get(name)
        if obj is None or len(obj.data.materials) != 1:
            raise ValueError(f"Missing original Piper material binding: {name}")
        shader = next(node for node in obj.active_material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
        shader.inputs["Specular IOR Level"].default_value = appearance["specular"]
    return appearances


def validate_piper_materials(scene, folder, appearances):
    """Check actual render materials after cache evaluation, before spending time rendering the animation."""
    checks = []
    for frame in (scene.frame_start, scene.frame_end // 2, scene.frame_end - 1):
        scene.frame_set(frame)
        graph = bpy.context.evaluated_depsgraph_get()
        for name, appearance in appearances.items():
            obj = bpy.data.objects[name].evaluated_get(graph)
            shader = next(node for node in obj.active_material.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
            if appearance["texture"]:
                images = [node.image for node in obj.active_material.node_tree.nodes if node.type == "TEX_IMAGE"]
                if not images or any(image is None or not image.has_data for image in images):
                    raise ValueError(f"Missing Piper texture: {name}")
            elif not np.allclose(shader.inputs["Base Color"].default_value[:3], appearance["color_linear"], atol=1e-6):
                raise ValueError(f"Piper base color changed: {name}")
            for socket, key in (
                ("Roughness", "roughness"),
                ("Metallic", "metallic"),
                ("Specular IOR Level", "specular"),
            ):
                if abs(shader.inputs[socket].default_value - appearance[key]) > 1e-6:
                    raise ValueError(f"Piper {key} changed: {name}")
        checks.append(frame)
    report = dict(
        passed=True,
        checked_frames=checks,
        mesh_count=len(appearances),
        source_materials=sorted({appearance["name"] for appearance in appearances.values()}),
        display_rgb_decoded_to_linear=True,
        uniform_material_override=False,
    )
    (folder / "material_validation.json").write_text(json.dumps(report, indent=2))


def validate_reference_style(scene, folder, config, film):
    """Inspect evaluated render data so USD cache playback cannot silently erase the film's static UVs."""
    checks = {}
    floor = bpy.data.objects["Floor"]
    shader = floor.active_material.node_tree.nodes.get("Principled BSDF")
    spec = config["environment"]["scene_floor"]
    checks["floor"] = bool(
        np.allclose(floor.dimensions, (4.4, 4.4, 0.04))
        and np.allclose(shader.inputs["Base Color"].default_value[:3], spec["color"])
        and abs(shader.inputs["Metallic"].default_value - spec["metallic"]) < 1e-6
        and abs(shader.inputs["Roughness"].default_value - spec["roughness"]) < 1e-6
    )
    checks["lights"] = sum(obj.type == "LIGHT" for obj in scene.objects) == len(config["lights"])
    for spec in config["lights"]:
        obj = bpy.data.objects[spec["name"]]
        checks[f"light_{spec['name']}"] = bool(
            np.allclose(obj.location, spec["position"])
            and np.allclose(obj.data.color, spec["color"])
            and abs(obj.data.energy - spec["energy"]) < 1e-6
            and abs(obj.data.size - spec["size"]) < 1e-6
            and obj.data.shape == spec.get("shape", "SQUARE")
            and (obj.data.shape != "RECTANGLE" or abs(obj.data.size_y - spec["size_y"]) < 1e-6)
        )
    images = [node.image for node in film.node_tree.nodes if node.type == "TEX_IMAGE"]
    # Packed images load pixels lazily in Blender 5.2; packed bytes and dimensions verify availability without decoding.
    checks["packed_film_normals"] = bool(images and all(image.packed_file and min(image.size) > 0 for image in images))
    environments = [node for node in scene.world.node_tree.nodes if node.type == "TEX_ENVIRONMENT"]
    checks["packed_hdri"] = bool(
        environments and all(node.image.packed_file and min(node.image.size) > 0 for node in environments)
    )
    camera, spec = scene.camera, config["camera"]
    view_direction = camera.rotation_euler.to_matrix() @ Vector((0, 0, -1))
    target_direction = (Vector(spec["target"]) - camera.location).normalized()
    checks["camera"] = bool(
        camera.data.type == "PERSP"
        and np.allclose(camera.location, spec["position"])
        and np.allclose(view_direction, target_direction)
        and abs(math.degrees(camera.data.angle) - spec["fov"]) < 1e-4
        and camera.data.dof.use_dof
        and abs(camera.data.dof.aperture_fstop - spec["fstop"]) < 1e-6
    )
    for frame in (scene.frame_start, scene.frame_end // 2, scene.frame_end - 1):
        scene.frame_set(frame)
        graph = bpy.context.evaluated_depsgraph_get()
        for name in ("film_0", "film_1"):
            obj = bpy.data.objects[name].evaluated_get(graph)
            uv = obj.data.uv_layers.get(FILM_UV)
            checks[f"{name}_uv_frame_{frame}"] = bool(
                uv and len(uv.data) == len(obj.data.loops) and all(np.isfinite(corner.uv).all() for corner in uv.data)
            )
            checks[f"{name}_material_frame_{frame}"] = obj.data.materials[0].original == film
    report = dict(
        passed=all(checks.values()),
        reference="demos/piper_bag_transport",
        checks=checks,
        bag=config["bag"],
        floor=config["environment"]["scene_floor"],
        lights=config["lights"],
        hdri=config["hdri"],
        camera=config["camera"],
    )
    (folder / "render_style_validation.json").write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        raise ValueError(f"Reference render style validation failed: {checks}")


def build(folder, quality="preview"):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(
        filepath=str(folder / "simulation.usdc"), import_cameras=False, import_lights=False, set_frame_range=True
    )
    scene = bpy.context.scene
    config, _ = reference_style()
    film = bag_material(config["bag"], REFERENCE)
    with np.load(folder / "trajectory.npz") as recording:
        panel_vertices = recording["film_panel_vertices"]
        panel_faces = recording["film_panel_faces"]
    appearances = restore_piper_finish(folder)
    for obj in list(bpy.data.objects):
        if obj.type != "MESH":
            continue
        if obj.name in ("film_0", "film_1"):
            obj.data.materials.clear()
            film_coordinates(obj, panel_vertices, panel_faces, config["bag"]["normal_tile_size_m"])
            obj.data.materials.append(film)
            shipping_label(obj)
            if config["bag"]["subdivision"]:
                subd = obj.modifiers.new("Render-only film smoothing", "SUBSURF")
                subd.levels = subd.render_levels = config["bag"]["subdivision"]
        elif obj.name == "jelly":
            obj.data.materials.clear()
            mat, shader = material("Jelly | green", (0.045, 0.37, 0.16), 0.22)
            shader.inputs["Subsurface Weight"].default_value = 0.2
            shader.inputs["Subsurface Radius"].default_value = (0.02, 0.04, 0.025)
            shader.inputs["Coat Weight"].default_value = 0.4
            shader.inputs["Transmission Weight"].default_value = 0.25
            shader.inputs["IOR"].default_value = 1.38
            obj.data.materials.append(mat)
        elif obj.name == "carton":
            obj.data.materials.clear()
            mat, shader = material("Carton | kraft paper", (0.43, 0.23, 0.09), 0.8)
            noise = mat.node_tree.nodes.new("ShaderNodeTexNoise")
            noise.inputs["Scale"].default_value = 180
            bump = mat.node_tree.nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value = 0.15
            bump.inputs["Distance"].default_value = 0.0002
            mat.node_tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
            mat.node_tree.links.new(bump.outputs["Normal"], shader.inputs["Normal"])
            obj.data.materials.append(mat)
        if not obj.name.startswith("robot_"):
            for face in obj.data.polygons:
                face.use_smooth = obj.name != "carton"
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, -0.2, -0.02))
    floor = bpy.context.object
    floor.name = "Floor"
    floor.dimensions = (4.4, 4.4, 0.04)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    floor.data.materials.append(
        environment_material("Floor | piper_bag_transport", config["environment"]["scene_floor"])
    )
    scene.world = studio_world(config, REFERENCE)
    for spec in config["lights"]:
        light = bpy.data.lights.new(spec["name"], "AREA")
        light.energy, light.size, light.color = spec["energy"], spec["size"], spec["color"]
        light.shape = spec.get("shape", "SQUARE")
        if light.shape == "RECTANGLE":
            light.size_y = spec["size_y"]
        obj = bpy.data.objects.new(spec["name"], light)
        scene.collection.objects.link(obj)
        obj.location = spec["position"]
        aim(obj, spec["target"])
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    scene.collection.objects.link(camera)
    scene.camera = camera
    spec = config["camera"]
    camera.location = spec["position"]
    aim(camera, spec["target"])
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.angle = math.radians(spec["fov"])
    camera.data.clip_start, camera.data.clip_end = 0.01, 300
    camera.data.dof.use_dof = True
    camera.data.dof.aperture_fstop = spec["fstop"]
    camera.data.dof.focus_distance = (Vector(spec["target"]) - camera.location).length
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "OPTIX"
    scene.cycles.device = "GPU"
    scene.cycles.samples = config["samples"] if quality == "final" else config["preview_samples"]
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = config["adaptive_threshold"]
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = "OPENIMAGEDENOISE"
    scene.cycles.denoising_use_gpu = True
    scene.cycles.transparent_max_bounces = 32
    scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 8
    scene.render.use_persistent_data = True
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = config["motion_blur_shutter"]
    percentage = 100 if quality == "final" else config["preview_percentage"]
    scene.render.resolution_x, scene.render.resolution_y = [
        int(size * percentage / 100) for size in config["resolution"]
    ]
    scene.render.resolution_percentage = 100
    scene.render.fps = 30
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "16"
    scene.view_settings.view_transform = config["view_transform"]
    scene.view_settings.look = config["look"]
    scene.view_settings.exposure = config["exposure"]
    scene.view_settings.gamma = config["gamma"]
    scene.frame_start = 0
    validate_piper_materials(scene, folder, appearances)
    validate_reference_style(scene, folder, config, film)
    scene.frame_set(0)
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / f"{quality}.blend"))
    return scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", default="0,180,300,390,630,750,899")
    parser.add_argument("--animation", action="store_true")
    parser.add_argument("--quality", choices=("preview", "final"), default="preview")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    folder = args.output.resolve()
    scene = build(folder, args.quality)
    frames = range(scene.frame_end) if args.animation else map(int, args.frames.split(","))
    digest = hashlib.sha256((folder / "simulation.usdc").read_bytes()).hexdigest()
    recipe = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    appearance = hashlib.sha256((folder / "piper_materials.json").read_bytes()).hexdigest()
    _, style_sources = reference_style()
    style = hashlib.sha256(json.dumps(style_sources, sort_keys=True).encode()).hexdigest()
    target = folder / "frames" / f"{digest[:12]}_{recipe[:12]}_{appearance[:12]}_{style[:12]}_{args.quality}"
    target.mkdir(parents=True, exist_ok=True)
    (folder / f"{args.quality}_render.json").write_text(
        json.dumps(
            dict(
                usd_sha256=digest,
                script_sha256=recipe,
                material_sha256=appearance,
                reference_style="demos/piper_bag_transport",
                reference_source_sha256=style_sources,
                frame_directory=str(target.relative_to(folder)),
                quality=args.quality,
                fps=scene.render.fps,
                width=scene.render.resolution_x,
                height=scene.render.resolution_y,
                samples=scene.cycles.samples,
                denoiser=scene.cycles.denoiser,
                frames=scene.frame_end,
            ),
            indent=2,
        )
    )
    for frame in frames:
        path = target / f"frame_{frame:04d}.png"
        if path.exists():
            continue
        scene.frame_set(frame)
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        print(f"{args.quality.upper()}_FRAME {frame}", flush=True)


if __name__ == "__main__":
    main()

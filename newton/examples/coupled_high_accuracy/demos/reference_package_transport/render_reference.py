"""Render the physically simulated package scene with the supplied video framing."""

import argparse
import json
import sys
from pathlib import Path

import bpy

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "piper_package_transport"))
import render_preview as base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE.parents[1] / "build/reference-match-8")
    parser.add_argument("--frames", default="0,90,210,315,450,630,780,899")
    parser.add_argument("--animation", action="store_true")
    parser.add_argument("--engine", choices=("cycles", "eevee"), default="cycles")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    folder = args.output.resolve()
    profile = json.loads((HERE / "reference_style.json").read_text(encoding="utf8"))
    original_style = base.reference_style

    def reference_style():
        config, hashes = original_style()
        config["camera"] = profile["camera"]
        config["resolution"] = profile["resolution"]
        config["preview_percentage"] = 100
        config["preview_samples"] = profile["samples"]
        config["bag"].update(profile.get("film_overrides", {}))
        config["lights"][0]["size"] = profile.get("key_light_size", config["lights"][0]["size"])
        for light in config["lights"]:
            if light["name"].startswith("Film highlights"):
                light["energy"] *= profile.get("strip_light_scale", 1.0)
        config["environment"]["scene_floor"].update(color=profile["floor_color"], metallic=0.0, roughness=0.8)
        # Blender 5.1 rejects this Radiance header with CRLF; preserve all binary pixel bytes.
        hdri = (base.REFERENCE / config["hdri"]["file"]).read_bytes()
        marker = hdri.index(b"-Y ")
        end = hdri.index(b"\n", marker) + 1
        compatible = folder / "studio_softboxes_lf.hdr"
        compatible.write_bytes(hdri[:end].replace(b"\r\n", b"\n") + hdri[end:])
        config["hdri"]["file"] = str(compatible)
        return config, hashes

    base.reference_style = reference_style
    scene = base.build(folder, "preview")
    for name, color in [("Carton | kraft paper", profile["carton_color"]), ("Jelly | green", profile["jelly_color"])]:
        mat = bpy.data.materials.get(name)
        if mat:
            shader = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
            shader.inputs["Base Color"].default_value = (*color, 1)
            if name == "Jelly | green":
                shader.inputs["Roughness"].default_value = profile.get("jelly_roughness", 0.55)
                shader.inputs["Coat Weight"].default_value = profile.get("jelly_coat", 0.03)
                shader.inputs["Transmission Weight"].default_value = profile.get("jelly_transmission", 0.0)
                shader.inputs["Subsurface Weight"].default_value = 0.0
                shader.inputs["Specular IOR Level"].default_value = 0.1
    for mat in bpy.data.materials:
        if mat.name.startswith("Shipping label |"):
            shader = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
            for link in list(shader.inputs["Base Color"].links):
                mat.node_tree.links.remove(link)
            shader.inputs["Base Color"].default_value = (*profile["label_color"], 1)
    scene.camera.data.dof.use_dof = False
    if args.engine == "eevee":
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.color_depth = "8"
    scene.frame_set(0)
    bpy.ops.wm.save_as_mainfile(filepath=str(folder / "reference_scene.blend"))
    frames = range(900) if args.animation else [int(n) for n in args.frames.split(",")]
    target = folder / ("reference_frames" if args.engine == "cycles" else "reference_frames_eevee")
    target.mkdir(exist_ok=True)
    for frame in frames:
        scene.frame_set(frame)
        scene.render.filepath = str(target / f"frame_{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
        print(f"REFERENCE_FRAME {frame}", flush=True)


if __name__ == "__main__":
    main()

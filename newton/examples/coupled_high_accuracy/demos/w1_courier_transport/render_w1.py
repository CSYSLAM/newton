"""Render opaque courier mailers using the reference scene camera and lighting."""

import json
import sys
from pathlib import Path

import bpy

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parents[1] / "assets/w1/courier"
sys.path.insert(0, str(HERE.parent / "reference_package_transport"))
import render_reference as reference


def main():
    hd = "--hd" in sys.argv
    if hd:
        sys.argv.remove("--hd")
    # Reuse the render setup in this process only; the transparent scene files remain untouched.
    reference.HERE = HERE
    profile = json.loads((HERE / "reference_style.json").read_text(encoding="utf8"))
    build = reference.base.build
    restore = reference.base.restore_piper_finish

    def restore_w1(folder):
        appearances = restore(folder)
        for image in bpy.data.images:
            if image.source == "FILE":
                # Force lazy USD texture decoding before inspecting evaluated material copies.
                _ = image.pixels[:4]
                image.pack()
        return appearances

    reference.base.restore_piper_finish = restore_w1

    def build_courier(folder, mode):
        scene = build(folder, mode)
        if hd:
            scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
            scene.cycles.samples = 64
            scene.cycles.adaptive_threshold = 0.02
        bpy.data.objects["Floor"].scale.x = 10
        bpy.data.objects["Floor"].scale.y = 10
        finished_materials = set()
        for obj in scene.objects:
            if obj.type != "MESH" or obj.name not in ("film_0", "film_1"):
                continue
            pattern = obj.data.uv_layers["film_pattern_m"]
            coordinates = [tuple(item.uv) for item in pattern.data]
            low = [min(uv[i] for uv in coordinates) for i in range(2)]
            high = [max(uv[i] for uv in coordinates) for i in range(2)]
            surface_uv = obj.data.uv_layers.new(name="CourierSurface")
            for target, uv in zip(surface_uv.data, coordinates):
                target.uv = tuple((uv[i] - low[i]) / (high[i] - low[i]) for i in range(2))
            label_uv = obj.data.uv_layers.new(name="CourierLabel")
            for target, source in zip(label_uv.data, obj.data.uv_layers["ShippingPrint"].data):
                target.uv = ((source.uv[0] - 0.25) / 0.5, (source.uv[1] - 0.36) / 0.28)
            label = obj.data.materials[1]
            # Keep the reference renderer's plain-label override from replacing the printed waybill.
            label.name = "Courier waybill | " + obj.name
            label_shader = next(node for node in label.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
            bind_texture(label, label_shader, "shipping_label.png", "CourierLabel")
            material = obj.data.materials[0]
            if material.name in finished_materials:
                continue
            material.name = "Courier | opaque gray polyethylene"
            finished_materials.add(material.name)
            nodes, links = material.node_tree.nodes, material.node_tree.links
            output = next(node for node in nodes if node.type == "OUTPUT_MATERIAL")
            for link in list(output.inputs["Surface"].links):
                links.remove(link)
            shader = nodes.new("ShaderNodeBsdfPrincipled")
            shader.inputs["Base Color"].default_value = (*profile["courier_color"], 1.0)
            shader.inputs["Roughness"].default_value = 0.55
            shader.inputs["Transmission Weight"].default_value = 0.0
            shader.inputs["Alpha"].default_value = 1.0
            bind_texture(material, shader, "mailer_surface.png", "CourierSurface")
            bump = next((node for node in nodes if node.type == "BUMP"), None)
            if bump:
                links.new(bump.outputs["Normal"], shader.inputs["Normal"])
            links.new(shader.outputs["BSDF"], output.inputs["Surface"])
            material.diffuse_color = (*profile["courier_color"], 1.0)
        return scene

    reference.base.build = build_courier
    if "--" not in sys.argv:
        sys.argv += ["--"]
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    if not any(arg == "--output" or arg.startswith("--output=") for arg in arguments):
        sys.argv += ["--output", str(HERE.parents[1] / "build/w1-courier")]
    reference.main()


def bind_texture(material, shader, filename, uv_name):
    nodes, links = material.node_tree.nodes, material.node_tree.links
    uv = nodes.new("ShaderNodeUVMap")
    uv.uv_map = uv_name
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = bpy.data.images.load(str(ASSETS / filename), check_existing=True)
    texture.image.pack()
    texture.extension = "EXTEND"
    links.new(uv.outputs["UV"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], shader.inputs["Base Color"])


if __name__ == "__main__":
    main()

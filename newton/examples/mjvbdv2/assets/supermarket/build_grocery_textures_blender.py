# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Author surface-bound bread and milk packaging textures in Blender."""

import math
from pathlib import Path

import bpy


def build(output_dir):
    """Export authored textures; preserve the caller's scene."""
    output = Path(output_dir)
    image = bpy.data.images.new("Baked bread crust", width=768, height=768)
    pixels = []
    for j in range(768):
        v = 2 * j / 767 - 1
        for i in range(768):
            u = 2 * i / 767 - 1
            noise = math.sin(i * 12.9898 + j * 78.233) * 43758.5453
            noise -= math.floor(noise)
            mottling = 0.04 * math.sin(19 * u + 11 * v) * math.sin(12 * v - 8 * u)
            score = min(abs(v - c - 0.35 * u) for c in (-0.55, 0, 0.55))
            cut = max(0, 1 - score / 0.07) * max(0, 1 - (abs(u) / 0.82) ** 8)
            rim = max(0, abs(u) - 0.65) * 0.20
            flour = 0.16 if noise > 0.987 else 0
            pixels.extend(
                (
                    min(1, 0.65 + 0.24 * cut + mottling + flour - rim),
                    min(1, 0.32 + 0.36 * cut + mottling + flour - rim),
                    min(1, 0.105 + 0.25 * cut + mottling * 0.5 + flour),
                    1,
                )
            )
    image.pixels.foreach_set(pixels)
    image.filepath_raw = str(output / "bread_crust.png")
    image.file_format = "PNG"
    image.save()

    previous = bpy.context.window.scene
    scene = bpy.data.scenes.new("Authored milk pouch print")
    bpy.context.window.scene = scene
    materials = {}
    for name, color in {"white": (0.96, 0.97, 0.96), "blue": (0.02, 0.18, 0.39), "light": (0.40, 0.72, 0.88)}.items():
        material = bpy.data.materials.new("Milk print " + name)
        material.use_nodes = True
        material.node_tree.nodes.clear()
        emission = material.node_tree.nodes.new("ShaderNodeEmission")
        emission.inputs["Color"].default_value = (*color, 1)
        out = material.node_tree.nodes.new("ShaderNodeOutputMaterial")
        material.node_tree.links.new(emission.outputs[0], out.inputs[0])
        materials[name] = material

    def panel(x, y, width, height, color, z=0):
        bpy.ops.mesh.primitive_plane_add(size=1, location=(x, y, z))
        obj = bpy.context.object
        obj.scale = (width, height, 1)
        obj.data.materials.append(materials[color])

    def lettering(body, x, y, size, color="blue"):
        curve = bpy.data.curves.new(body, "FONT")
        curve.body, curve.size = body, size
        obj = bpy.data.objects.new(body, curve)
        scene.collection.objects.link(obj)
        obj.location = (x, y, 0.01)
        curve.materials.append(materials[color])

    panel(0, 0, 1.1, 1.6, "white")
    panel(0, 0.56, 1, 0.27, "blue", 0.002)
    panel(0, -0.57, 1, 0.19, "blue", 0.002)
    panel(0, -0.42, 1, 0.025, "light", 0.002)
    lettering("DAIRY FRESH", -0.39, 0.53, 0.092, "white")
    lettering("MILK", -0.40, 0.17, 0.24)
    lettering("PURE WHOLE MILK", -0.39, 0.045, 0.06)
    lettering("3.5%", -0.38, -0.16, 0.10)
    lettering("MILK FAT", -0.38, -0.23, 0.044)
    lettering("KEEP REFRIGERATED", -0.38, -0.36, 0.042)
    lettering("FRESH DAILY     200 ml", -0.39, -0.59, 0.062, "white")
    for i in range(26):
        width = 0.0025 if i % 3 else 0.004
        panel(0.13 + i * 0.009, -0.16, width, 0.14, "blue", 0.003)
    lettering("0 12345 67890", 0.105, -0.275, 0.026)
    # Printed heat-seal bands follow the actual deformable seam in the demo.
    for j in (-1, 1):
        for n in range(5):
            panel(0, j * (0.69 + n * 0.01), 1, 0.002, "light", 0.003)
    bpy.ops.object.camera_add(location=(0, 0, 3))
    scene.camera = bpy.context.object
    scene.camera.data.type = "ORTHO"
    scene.camera.data.ortho_scale = 1.5
    engines = scene.render.bl_rna.properties["engine"].enum_items.keys()
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = 512, 768
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output / "milk_pouch.png")
    bpy.ops.render.render(write_still=True)
    bpy.context.window.scene = previous

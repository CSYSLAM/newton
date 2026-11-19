"""Fast warehouse dressing and consistent yellow W1 shell materials."""

import random

import bpy


def material(name, color, roughness=0.65, metallic=0):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    return mat


def box(name, location, dimensions, mat, bevel=0.008):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    if bevel:
        modifier = obj.modifiers.new("Manufactured edges", "BEVEL")
        modifier.width, modifier.segments = bevel, 2
    return obj


def unify_shell():
    # Preserve baked logos, dark joints and yellow body textures; recolor only chromatic accent regions.
    gripper = material("W1 | black gripper housing and fingers", (0.012, 0.014, 0.017), 0.38, 0.25)
    # Exported visual IDs cover both housings, fingers and pads, excluding the wrist shells.
    gripper_ids = {24, 26, 28, 30, 32, 33, 48, 50, 52, 54, 56, 57}
    seen = set()
    for obj in bpy.context.scene.objects:
        if not obj.name.startswith("robot_") or obj.type != "MESH":
            continue
        if obj.name in {f"robot_{i:03d}" for i in gripper_ids}:
            obj.data.materials.clear()
            obj.data.materials.append(gripper)
            for polygon in obj.data.polygons:
                polygon.material_index = 0
            continue
        for mat in obj.data.materials:
            if not mat or mat.name in seen or not mat.use_nodes:
                continue
            seen.add(mat.name)
            nodes, links = mat.node_tree.nodes, mat.node_tree.links
            for shader in [n for n in nodes if n.type == "BSDF_PRINCIPLED"]:
                socket = shader.inputs["Base Color"]
                if not socket.is_linked:
                    color = socket.default_value
                    if max(color[:3]) - min(color[:3]) > 0.06:
                        socket.default_value = (0.604, 0.413, 0.005, 1)
                    continue
                source = socket.links[0].from_socket
                separate = nodes.new("ShaderNodeSeparateColor")
                separate.mode = "HSV"
                links.new(source, separate.inputs[0])
                hue_mask = nodes.new("ShaderNodeMath")
                hue_mask.operation = "GREATER_THAN"
                hue_mask.inputs[1].default_value = 0.17
                links.new(separate.outputs[0], hue_mask.inputs[0])
                sat_mask = nodes.new("ShaderNodeMath")
                sat_mask.operation = "GREATER_THAN"
                sat_mask.inputs[1].default_value = 0.22
                links.new(separate.outputs[1], sat_mask.inputs[0])
                mask = nodes.new("ShaderNodeMath")
                mask.operation = "MULTIPLY"
                links.new(hue_mask.outputs[0], mask.inputs[0])
                links.new(sat_mask.outputs[0], mask.inputs[1])
                mix = nodes.new("ShaderNodeMixRGB")
                links.new(mask.outputs[0], mix.inputs[0])
                links.new(source, mix.inputs[1])
                mix.inputs[2].default_value = (0.604, 0.413, 0.005, 1)
                links.new(mix.outputs[0], socket)


def soft_gel_material():
    """Make the robot-left rigid-core mailer's outer film readable without changing its motion."""
    film = bpy.data.objects["film_0"]
    mat = material("Soft package | blue translucent elastomer", (0.035, 0.33, 0.55), 0.22)
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Transmission Weight"].default_value = 0.28
    shader.inputs["IOR"].default_value = 1.40
    shader.inputs["Coat Weight"].default_value = 0.22
    shader.inputs["Coat Roughness"].default_value = 0.16
    shader.inputs["Subsurface Weight"].default_value = 0.045
    shader.inputs["Subsurface Radius"].default_value = (0.008, 0.014, 0.02)
    # No baked wrinkle normals: highlights follow the animated mesh surface itself.
    film.data.materials[0] = mat


def decorate(scene):
    unify_shell()
    bpy.data.objects["Floor"].hide_render = True
    concrete = material("Workshop | fine concrete", (0.25, 0.27, 0.28), 0.88)
    nodes, links = concrete.node_tree.nodes, concrete.node_tree.links
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 75
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value, bump.inputs["Distance"].default_value = 0.18, 0.002
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs[0], nodes.get("Principled BSDF").inputs["Normal"])
    wall = material("Workshop | warm plaster", (0.64, 0.62, 0.56), 0.9)
    steel = material("Workshop | graphite powder coat", (0.06, 0.085, 0.10), 0.45, 0.35)
    board = material("Workshop | shelf board", (0.27, 0.19, 0.10))
    kraft = [
        material("Shipping carton " + str(i), (0.34 + i * 0.035, 0.20 + i * 0.025, 0.09 + i * 0.015)) for i in range(4)
    ]
    tape = material("Workshop | parcel tape", (0.47, 0.33, 0.16), 0.38)
    white = material("Workshop | shipping labels", (0.82, 0.81, 0.75))
    box("Concrete floor", (0, 0, -0.055), (10, 10, 0.1), concrete)
    box("Back wall", (-2.2, 0.2, 1.7), (0.12, 8, 3.4), wall)
    box("Side wall", (0.4, 3.0, 1.7), (5.2, 0.12, 3.4), wall)
    box("Back skirting", (-2.12, 0.2, 0.08), (0.035, 8, 0.16), steel, 0.002)
    box("Side skirting", (0.4, 2.92, 0.08), (5.2, 0.035, 0.16), steel, 0.002)
    rng = random.Random(17)
    # Shelving sits behind the robot, outside every recorded collision/contact region.
    for bay_y in (-1.25, 1.25):
        for x in (-1.95, -1.42):
            for y in (bay_y - 0.68, bay_y + 0.68):
                box("Shelf upright", (x, y, 1.12), (0.045, 0.045, 2.24), steel)
        for level in (0.18, 0.77, 1.36, 1.95):
            box("Shelf", (-1.685, bay_y, level), (0.62, 1.45, 0.035), board)
            if level > 1.9:
                continue
            for j in range(3):
                w, h = rng.uniform(0.29, 0.36), rng.uniform(0.25, 0.41)
                y = bay_y - 0.45 + j * 0.44
                box("Stored parcel", (-1.66, y, level + 0.018 + h / 2), (0.43, w, h), rng.choice(kraft))
                box("Carton tape", (-1.66, y, level + 0.02 + h), (0.435, 0.045, 0.002), tape, 0.001)
                box("Parcel label", (-1.438, y, level + h * 0.55), (0.002, w * 0.48, h * 0.28), white, 0.001)
    frame = material("Workshop | window frame", (0.72, 0.74, 0.73), 0.4, 0.2)
    sky = material("Workshop | daylight glazing", (0.61, 0.76, 0.85), 0.28)
    shader = sky.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Emission Color"].default_value = (0.61, 0.76, 0.85, 1)
    shader.inputs["Emission Strength"].default_value = 0.6
    box("Daylight window", (0.7, 2.925, 2.05), (2.1, 0.035, 1.35), sky)
    for x in (-0.38, 0.7, 1.78):
        box("Window mullion", (x, 2.88, 2.05), (0.045, 0.06, 1.43), frame)
    for z in (1.35, 2.05, 2.75):
        box("Window rail", (0.7, 2.88, z), (2.2, 0.06, 0.045), frame)
    scene.camera.location = (3.0, -3.8, 2.65)
    from mathutils import Vector

    scene.camera.rotation_euler = (Vector((0.1, 0, 1.02)) - scene.camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera.data.lens = 43
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.render.use_motion_blur = False
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 8
    scene.cycles.adaptive_threshold = 0.1
    scene.cycles.max_bounces = 4

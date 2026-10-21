# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Model the rice cooker of ``mjmpm_w1_rice_cooker`` in Blender and export its OBJ meshes.

Run inside Blender 5.x (Scripting workspace or ``blender --background --python``).
Units are metres; the origin is the bottom centre of the cooker and the lid
opens toward +x, away from the robot.
"""

import math
import os

import bmesh
import bpy
import mathutils

R_OUT, H = 0.12, 0.15  # body radius and height
R_IN, DEPTH = 0.09, 0.11  # inner pot radius and depth; walls and bottom stay >= 2 MPM cells
SEG = 48
OUT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()


def cylinder(name, radius, depth, z_center):
    bpy.ops.mesh.primitive_cylinder_add(vertices=SEG, radius=radius, depth=depth, location=(0, 0, z_center))
    obj = bpy.context.active_object
    obj.name = name
    return obj


def boolean(target, cutter, operation):
    mod = target.modifiers.new(operation.lower(), "BOOLEAN")
    mod.operation = operation
    mod.object = cutter
    mod.solver = "EXACT"
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

body = cylinder("rice_cooker_body", R_OUT, H, H / 2)
boolean(body, cylinder("cavity", R_IN, DEPTH + 0.02, H - DEPTH / 2 + 0.01), "DIFFERENCE")
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-R_OUT + 0.005, 0.0, 0.06))
panel = bpy.context.active_object
panel.scale = (0.03, 0.08, 0.05)
bpy.ops.object.transform_apply(scale=True)
boolean(body, panel, "UNION")

# Lid: a shallow dome hinged at the back rim and opened 100 degrees.
bpy.ops.mesh.primitive_uv_sphere_add(segments=SEG, ring_count=16, radius=1.0)
lid = bpy.context.active_object
lid.name = "rice_cooker_lid"
bm = bmesh.new()
bm.from_mesh(lid.data)
bmesh.ops.bisect_plane(
    bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=(0, 0, 0), plane_no=(0, 0, 1), clear_inner=True
)
bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary])
bm.to_mesh(lid.data)
bm.free()
lid.scale = (R_OUT + 0.003, R_OUT + 0.003, 0.035)
lid.location = (0.0, 0.0, H)
bpy.ops.object.transform_apply(location=True, scale=True)
hinge = mathutils.Vector((R_OUT, 0.0, H))
lid.data.transform(
    mathutils.Matrix.Translation(hinge)
    @ mathutils.Matrix.Rotation(math.radians(100), 4, "Y")
    @ mathutils.Matrix.Translation(-hinge)
)

for obj, filename in ((body, "body.obj"), (lid, "lid.obj")):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.wm.obj_export(
        filepath=os.path.join(OUT_DIR, filename),
        export_selected_objects=True,
        forward_axis="Y",
        up_axis="Z",
        export_materials=False,
        export_uv=False,
        export_normals=False,
        export_triangulated_mesh=True,
    )

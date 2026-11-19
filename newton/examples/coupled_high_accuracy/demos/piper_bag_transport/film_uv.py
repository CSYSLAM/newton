"""Create static material coordinates for a deforming, fixed-topology film mesh in Blender."""

from __future__ import annotations

import math

import bpy
from mathutils import Vector

FILM_UV = "film_uv"
FILM_PATTERN = "film_pattern_m"


def preserve_film_coordinates(obj):
    """Read animated positions while retaining coordinates authored on the initial film."""
    for modifier in obj.modifiers:
        if modifier.type == "MESH_SEQUENCE_CACHE":
            # Generic USD primvars can also replace UVs/rest positions on newer Blender versions.
            modifier.read_data = modifier.read_data - {"UV", "ATTRIBUTES"}


def assign_film_uv(obj, tile_size_m):
    """Use the developed cutting pattern, or unwrap legacy meshes, at a metric texture scale."""
    if obj.type != "MESH" or not math.isfinite(tile_size_m) or tile_size_m <= 0:
        raise ValueError("Film UV mapping requires a mesh and a finite, positive tile size")
    mesh = obj.data
    pattern = mesh.uv_layers.get(FILM_PATTERN)
    if pattern is not None:
        uv = mesh.uv_layers.get(FILM_UV) or mesh.uv_layers.new(name=FILM_UV)
        # Face-varying coordinates keep welded gussets and multilayer seams in separate flat charts.
        # A geometric unwrap cannot recover the undeformed metric once those panels have been folded.
        for source, target in zip(pattern.data, uv.data):
            if not all(math.isfinite(value) for value in source.uv):
                raise ValueError("Film cutting-pattern UVs contain a non-finite coordinate")
            target.uv = source.uv / tile_size_m
        mesh.uv_layers.active = uv
        uv.active_render = True
        preserve_film_coordinates(obj)
        return FILM_UV
    context = bpy.context
    active = context.view_layer.objects.active
    selected = list(context.selected_objects)
    mode = active.mode if active else "OBJECT"
    if mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    original_seams = [edge.use_seam for edge in mesh.edges]
    original_selection = [[item.select for item in items] for items in (mesh.vertices, mesh.edges, mesh.polygons)]
    try:
        for other in context.selected_objects:
            other.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        uv = mesh.uv_layers.get(FILM_UV) or mesh.uv_layers.new(name=FILM_UV)
        mesh.uv_layers.active = uv
        uv.active_render = True
        adjacency = {}
        for face in mesh.polygons:
            for key in face.edge_keys:
                adjacency.setdefault(key, []).append(face)
        for edge in mesh.edges:
            neighbours = adjacency.get(edge.key, [])
            # Cut the right-angle bag-panel joints, retaining continuous UVs along the curved handles.
            edge.use_seam = (
                edge.use_seam or len(neighbours) != 2 or neighbours[0].normal.dot(neighbours[1].normal) < 0.5
            )
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.select_all(action="SELECT")
        bpy.ops.uv.unwrap(method="ANGLE_BASED", correct_aspect=False, margin=0.005)
        # Unwrap may scale separate charts differently; normalize their texel density before metric scaling.
        bpy.ops.uv.average_islands_scale()
        bpy.ops.object.mode_set(mode="OBJECT")
        uv = mesh.uv_layers[FILM_UV]
        mesh.calc_loop_triangles()
        surface_area = uv_area = 0.0
        for triangle in mesh.loop_triangles:
            points = [obj.matrix_world @ mesh.vertices[index].co for index in triangle.vertices]
            surface_area += (points[1] - points[0]).cross(points[2] - points[0]).length * 0.5
            coordinates = [Vector(uv.data[index].uv) for index in triangle.loops]
            uv_area += abs((coordinates[1] - coordinates[0]).cross(coordinates[2] - coordinates[0])) * 0.5
        if not math.isfinite(uv_area) or min(surface_area, uv_area) <= 0:
            raise RuntimeError("Film unwrap produced an empty or non-finite UV surface")
        scale = math.sqrt(surface_area / uv_area) / tile_size_m
        for corner in uv.data:
            corner.uv *= scale
        preserve_film_coordinates(obj)
    finally:
        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for edge, seam in zip(mesh.edges, original_seams):
            edge.use_seam = seam
        for items, flags in zip((mesh.vertices, mesh.edges, mesh.polygons), original_selection):
            for item, flag in zip(items, flags):
                item.select = flag
        for other in context.selected_objects:
            other.select_set(False)
        for other in selected:
            other.select_set(True)
        context.view_layer.objects.active = active
        if active and mode != "OBJECT":
            bpy.ops.object.mode_set(mode=mode)
    return FILM_UV

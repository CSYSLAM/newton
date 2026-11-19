"""Verify Blender's evaluated USD caches against the original samples without saving the project.

Run: blender -b outputs/scene.blend --python demos/utils/validate_import.py -- --usd outputs/simulation.usdc
USD mesh basenames must match Blender object names; duplicate basenames are rejected.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import bpy
import numpy as np
from pxr import Usd, UsdGeom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from usd_cache import validate_usd_sources

TOLERANCE = 2.0e-6


def world_points(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, required=True, help="Source USD to compare with the loaded Blender project")
    parser.add_argument("--frames", help="Comma-separated sample frames; defaults to four samples across the USD range")
    parser.add_argument("--report", type=Path, help="Report path; defaults to import_validation.json beside the .blend")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    if not bpy.data.filepath:
        raise RuntimeError("Load the saved .blend project before validating its USD import")
    root = Path(bpy.data.filepath).parent.resolve()
    stage = Usd.Stage.Open(str(args.usd.resolve()))
    if stage is None:
        raise RuntimeError(f"Unable to open USD: {args.usd}")
    start, end = stage.GetStartTimeCode(), stage.GetEndTimeCode()
    frames = (
        list(map(float, args.frames.split(","))) if args.frames else sorted(set(np.linspace(start, end, 4).tolist()))
    )
    scene = bpy.context.scene
    validate_usd_sources(scene)
    report = {
        "project": Path(bpy.data.filepath).name,
        "blender": bpy.app.version_string,
        "frames_checked": frames,
        "tolerance_m": TOLERANCE,
        "scene_fps": scene.render.fps,
        "scene_fps_base": scene.render.fps_base,
        "usd_time_codes_per_second": stage.GetTimeCodesPerSecond(),
        "checks": [],
        "errors": [],
        "cache_files": [],
        "external_files": [],
    }
    if scene.render.fps / scene.render.fps_base != stage.GetTimeCodesPerSecond():
        report["errors"].append("Blender and USD sampling rates differ")
    for cache in bpy.data.cache_files:
        report["cache_files"].append(
            {
                "name": cache.name,
                "path": cache.filepath,
                "frame_offset": cache.frame_offset,
                "override_frame": cache.override_frame,
            }
        )
        if cache.frame_offset != 0 or cache.override_frame:
            report["errors"].append(f"Unexpected cache retiming: {cache.name}")
    # These transient changes expose the imported FEM surface; they are never saved to the project.
    for obj in scene.objects:
        for modifier in obj.modifiers:
            if modifier.type == "SUBSURF":
                modifier.show_viewport = False
                modifier.show_render = False
    # Inspect the USD directly so another demo does not need G1's simulation metadata format.
    manifest = [
        {"path": str(prim.GetPath()), "kind": prim.GetCustomDataByKey("kind") or "mesh"}
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh)
    ]
    names = Counter(entry["path"].rsplit("/", 1)[-1] for entry in manifest)
    duplicates = [name for name, count in names.items() if count > 1]
    if not manifest or duplicates:
        raise ValueError(f"Expected meshes with unique USD basenames; duplicates: {duplicates}")
    maximum_point_error = 0.0
    maximum_transform_error = 0.0
    for frame in frames:
        scene.frame_set(math.floor(frame), subframe=frame - math.floor(frame))
        graph = bpy.context.evaluated_depsgraph_get()
        graph.update()
        usd_cache = UsdGeom.XformCache(Usd.TimeCode(frame))
        for entry in manifest:
            name = entry["path"].rsplit("/", 1)[-1]
            obj = bpy.data.objects.get(name)
            if obj is None:
                report["errors"].append(f"Missing Blender object {name}")
                continue
            prim = stage.GetPrimAtPath(entry["path"])
            usd_mesh = UsdGeom.Mesh(prim)
            expected = np.asarray(usd_mesh.GetPointsAttr().Get(frame), dtype=np.float64)
            evaluated = obj.evaluated_get(graph)
            mesh = evaluated.to_mesh()
            actual = np.empty((len(mesh.vertices), 3), dtype=np.float64)
            mesh.vertices.foreach_get("co", actual.ravel())
            usd_matrix = np.asarray(usd_cache.GetLocalToWorldTransform(prim), dtype=np.float64).T
            blender_matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
            transform_error = float(np.max(np.abs(usd_matrix - blender_matrix)))
            maximum_transform_error = max(maximum_transform_error, transform_error)
            record = {
                "frame": frame,
                "path": entry["path"],
                "kind": entry["kind"],
                "usd_vertex_count": len(expected),
                "blender_vertex_count": len(actual),
                "usd_faces": len(usd_mesh.GetFaceVertexCountsAttr().Get(frame)),
                "blender_faces": len(mesh.polygons),
                "maximum_transform_element_error": transform_error,
            }
            if len(expected) != len(actual):
                report["errors"].append(f"Frame {frame}: vertex count mismatch in {name}")
            else:
                point_error = float(np.linalg.norm(actual - expected, axis=1).max(initial=0.0))
                world_error = float(
                    np.linalg.norm(
                        world_points(actual, blender_matrix) - world_points(expected, usd_matrix),
                        axis=1,
                    ).max(initial=0.0)
                )
                record["maximum_local_point_error_m"] = point_error
                record["maximum_world_point_error_m"] = world_error
                maximum_point_error = max(maximum_point_error, world_error)
                if point_error > TOLERANCE or world_error > TOLERANCE:
                    report["errors"].append(f"Frame {frame}: point mismatch in {name}: {world_error:.9g} m")
            if record["usd_faces"] != record["blender_faces"]:
                report["errors"].append(f"Frame {frame}: topology count mismatch in {name}")
            if transform_error > TOLERANCE:
                report["errors"].append(f"Frame {frame}: transform mismatch in {name}: {transform_error:.9g}")
            evaluated.to_mesh_clear()
            report["checks"].append(record)
    for path in bpy.utils.blend_paths(absolute=False, packed=False, local=True):
        resolved = Path(bpy.path.abspath(path)).resolve()
        relative = path.startswith("//")
        inside_project = resolved.is_relative_to(root)
        report["external_files"].append(
            {
                "path": path,
                "relative": relative,
                "exists": resolved.exists(),
                "inside_project": inside_project,
            }
        )
        if not relative or not resolved.exists() or not inside_project:
            report["errors"].append(f"External reference is not portable within the demo directory: {path}")
    for layer in stage.GetUsedLayers():
        if not layer.anonymous and not Path(layer.realPath).resolve().is_relative_to(root):
            report["errors"].append(f"USD has a nonportable layer reference: {layer.realPath}")
    report["summary"] = {
        "passed": not report["errors"],
        "mesh_samples_checked": len(report["checks"]),
        "mesh_counts": dict(Counter(entry["kind"] for entry in manifest)),
        "maximum_world_point_error_m": maximum_point_error,
        "maximum_transform_element_error": maximum_transform_error,
        "relative_external_references_verified": all(
            entry["relative"] and entry["exists"] and entry["inside_project"] for entry in report["external_files"]
        ),
        "project_modified_on_disk": False,
    }
    report_path = args.report or root / "import_validation.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    if report["errors"]:
        raise AssertionError("; ".join(report["errors"][:8]))


if __name__ == "__main__":
    main()

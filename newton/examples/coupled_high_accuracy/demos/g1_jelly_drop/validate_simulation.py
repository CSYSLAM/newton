"""Check every USD animation sample against the preserved NPZ states using only the CPU.

Run from any directory with Python that provides numpy and usd-core:
    python demos/g1_jelly_drop/validate_simulation.py

This validates export fidelity and provenance; it does not rerun or independently validate the solver.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

ROOT = Path(__file__).resolve().parent / "outputs"
REPOSITORY = Path(__file__).resolve().parents[2]
TOLERANCE = 1.0e-7


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate(report):
    metadata = json.loads((ROOT / "simulation_metadata.json").read_text(encoding="utf-8"))
    simulation = metadata["simulation"]
    frames = list(range(simulation["samples"]))
    frame_seconds = simulation["substep_seconds"] * metadata["physics_parameters"]["substeps"]
    require(
        np.isclose(frame_seconds * simulation["fps"], 1.0, rtol=0.0, atol=1.0e-10),
        "Export sampling rate differs from the recorded physics timestep",
    )
    require(
        np.isclose(simulation["duration_seconds"], simulation["frames"] / simulation["fps"]),
        "Export duration differs from its frame count and sampling rate",
    )
    report["source_hashes"] = []
    for relative, expected in metadata["source_sha256"].items():
        # Metadata created on Windows uses backslashes, even when this bundle is checked on another OS.
        path = REPOSITORY / Path(relative.replace("\\", "/"))
        require(path.is_file(), f"Export source was moved or removed: {relative}. Export the simulation again.")
        actual = sha256(path)
        report["source_hashes"].append({"path": relative, "sha256": actual, "matches": actual == expected})
    require(all(item["matches"] for item in report["source_hashes"]), "Example source hashes differ from this export")
    usd_path = ROOT / metadata["files"]["usd"]
    states_path = ROOT / metadata["files"]["raw_states"]
    report["files"] = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in (usd_path, states_path)
    }
    stage = Usd.Stage.Open(str(usd_path))
    require(stage is not None, "Unable to open the USD stage")
    require(stage.GetStartTimeCode() == 0, "Expected the initial state at time code zero")
    require(stage.GetEndTimeCode() == frames[-1] == simulation["frames"], "USD frame range differs from metadata")
    require(stage.GetTimeCodesPerSecond() == simulation["fps"], "USD sampling rate differs from metadata")
    require(stage.GetFramesPerSecond() == simulation["fps"], "USD playback rate differs from metadata")
    require(UsdGeom.GetStageUpAxis(stage) == simulation["up_axis"], "USD up axis differs from metadata")
    require(UsdGeom.GetStageMetersPerUnit(stage) == simulation["meters_per_unit"], "USD length units differ")
    require(stage.GetDefaultPrim().GetPath().pathString == "/World", "Expected /World as the default prim")

    with np.load(states_path, allow_pickle=False) as archive:
        particles = archive["particle_q"]
        bodies = archive["body_q"]
        times = archive["time_seconds"]
    require(
        particles.ndim == 3 and particles.shape[0] == len(frames) and particles.shape[2] == 3,
        "Unexpected raw particle array shape",
    )
    require(bodies.shape == (len(frames), simulation["robot_bodies"], 7), "Unexpected raw body array shape")
    require(
        np.array_equal(times, np.arange(len(frames), dtype=np.float64) / simulation["fps"]),
        "Raw sample timestamps differ from the USD timeline",
    )
    require([entry["frame"] for entry in metadata["timeline"]] == frames, "Metadata timeline has missing frames")
    require(
        np.array_equal([entry["time_seconds"] for entry in metadata["timeline"]], times),
        "Metadata timestamps differ from the raw sample timestamps",
    )
    require(np.isfinite(particles).all() and np.isfinite(bodies).all(), "Raw positions contain nonfinite values")
    report["raw_shapes"] = {"particle_q": list(particles.shape), "body_q": list(bodies.shape)}

    expected_paths = {entry["path"] for entry in metadata["meshes"]}
    actual_paths = {str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)}
    require(actual_paths == expected_paths, "USD meshes differ from the exported mesh manifest")
    maximum_point_error = 0.0
    maximum_pose_error = 0.0
    jelly_samples = robot_samples = 0
    matched_particles = set()
    mesh_counts = {"jelly": 0, "robot": 0, "decoration": 0}
    for entry in metadata["meshes"]:
        path = entry["path"]
        prim = stage.GetPrimAtPath(path)
        mesh = UsdGeom.Mesh(prim)
        require(prim.GetCustomDataByKey("kind") == entry["kind"], f"Mesh kind mismatch: {path}")
        mesh_counts[entry["kind"]] += 1
        counts_attr, indices_attr = mesh.GetFaceVertexCountsAttr(), mesh.GetFaceVertexIndicesAttr()
        require(not counts_attr.GetTimeSamples() and not indices_attr.GetTimeSamples(), f"Animated topology: {path}")
        counts, indices = np.asarray(counts_attr.Get()), np.asarray(indices_attr.Get())
        points_attr = mesh.GetPointsAttr()
        points = np.asarray(points_attr.Get(0))
        require(points.shape == (entry["vertices"], 3), f"Vertex count mismatch: {path}")
        require(len(counts) == entry["triangles"] and np.all(counts == 3), f"Triangle count mismatch: {path}")
        require(
            len(indices) == 3 * len(counts) and np.all((indices >= 0) & (indices < len(points))),
            f"Invalid triangle indices: {path}",
        )
        require(np.isfinite(points).all(), f"Nonfinite mesh vertices: {path}")
        require(mesh.GetSubdivisionSchemeAttr().Get() == UsdGeom.Tokens.none, f"Unexpected subdivision: {path}")
        if entry["kind"] != "jelly":
            require(not points_attr.GetTimeSamples(), f"Unexpected vertex animation on a rigid/static mesh: {path}")
            continue
        require(points_attr.GetTimeSamples() == frames, f"Incomplete jelly animation: {path}")
        require(mesh.GetExtentAttr().GetTimeSamples() == frames, f"Incomplete jelly bounds: {path}")
        # Cube vertices are spatially distinct initially. Recover their raw indices without importing GPU scene code.
        initial_jelly = particles[0, : simulation["jelly_particles"]]
        distance = np.linalg.norm(points[:, None, :] - initial_jelly[None, :, :], axis=-1)
        source = distance.argmin(axis=1)
        require(np.all(distance[np.arange(len(points)), source] <= TOLERANCE), f"Unknown initial jelly vertex: {path}")
        require(len(np.unique(source)) == len(source), f"Ambiguous initial jelly vertex mapping: {path}")
        require(not matched_particles.intersection(source.tolist()), f"Jelly meshes share particles: {path}")
        matched_particles.update(source.tolist())
        triangles = indices.reshape(-1, 3)
        edges = np.sort(np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]])), axis=1)
        require(np.all(np.unique(edges, axis=0, return_counts=True)[1] == 2), f"Jelly surface is not closed: {path}")
        for frame in frames:
            actual = np.asarray(points_attr.Get(frame))
            require(actual.shape == points.shape and np.isfinite(actual).all(), f"Invalid jelly state: {path}, {frame}")
            error = float(np.linalg.norm(actual - particles[frame, source], axis=1).max())
            maximum_point_error = max(maximum_point_error, error)
            require(error <= TOLERANCE, f"Jelly/raw state mismatch: {path}, frame {frame}, {error:g} m")
            extent = np.stack((actual.min(axis=0), actual.max(axis=0)))
            require(
                np.array_equal(np.asarray(mesh.GetExtentAttr().Get(frame)), extent),
                f"Jelly extent mismatch: {path}, frame {frame}",
            )
            jelly_samples += 1

    seen_bodies = set()
    for prim in stage.Traverse():
        index = prim.GetCustomDataByKey("body_index")
        if index is None or not prim.IsA(UsdGeom.Xform):
            continue
        require(index not in seen_bodies and 0 <= index < bodies.shape[1], f"Invalid body index: {index}")
        seen_bodies.add(index)
        require(prim.GetParent().GetPath().pathString == "/World/G1", f"Unexpected body hierarchy: {prim.GetPath()}")
        translate, orient = prim.GetAttribute("xformOp:translate"), prim.GetAttribute("xformOp:orient")
        ops = [op.GetOpName() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]
        require(
            ops == ["xformOp:translate", "xformOp:orient"], f"Unexpected body transform operations: {prim.GetPath()}"
        )
        require(
            translate.GetTimeSamples() == frames and orient.GetTimeSamples() == frames,
            f"Incomplete body animation: {prim.GetPath()}",
        )
        for frame in frames:
            quaternion = orient.Get(frame)
            pose = np.r_[translate.Get(frame), quaternion.GetImaginary(), quaternion.GetReal()]
            error = float(np.max(np.abs(pose - bodies[frame, index])))
            maximum_pose_error = max(maximum_pose_error, error)
            require(error <= TOLERANCE, f"Body/raw state mismatch: {prim.GetPath()}, frame {frame}, {error:g}")
            robot_samples += 1
    require(len(seen_bodies) == simulation["robot_bodies"], "Missing robot body transforms")
    require(mesh_counts["jelly"] == simulation["jelly_cubes"], "Missing jelly meshes")
    require(mesh_counts["robot"] == simulation["visual_meshes"], "Missing robot visual meshes")
    report["summary"] = {
        "frames_checked": len(frames),
        "first_frame": frames[0],
        "last_frame": frames[-1],
        "fps": stage.GetTimeCodesPerSecond(),
        "mesh_counts": mesh_counts,
        "jelly_mesh_samples_checked": jelly_samples,
        "robot_body_samples_checked": robot_samples,
        "maximum_jelly_point_error_m": maximum_point_error,
        "maximum_body_pose_element_error": maximum_pose_error,
        "finite_raw_states": True,
        "source_hashes_match": True,
        "fixed_triangle_topology_verified": True,
        "jelly_extents_verified": True,
        "files_modified": False,
    }


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "tolerance": TOLERANCE, "errors": []}
    try:
        validate(report)
        report["passed"] = True
    except Exception as error:
        report["errors"].append(f"{type(error).__name__}: {error}")
    (ROOT / "simulation_validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

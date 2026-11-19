"""Export the unmodified G1 jelly example to portable, animated USD meshes.

Run from the repository root with: uv run python demos/g1_jelly_drop/export_simulation.py
Generated files are saved in outputs/ by default.
Additional example physics options, such as --layers or --substeps, are forwarded to its own parser.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

import newton
from newton.viewer import ViewerNull

DIRECTORY = Path(__file__).resolve().parent
REPOSITORY = DIRECTORY.parents[1]
if __package__:
    from ..demo.demo_g1_jelly_drop import Example
    from .g1_robot import measure_joint_errors
    from .jelly_stack import signed_six_volumes
else:
    sys.path.insert(0, str(DIRECTORY.parent / "demo"))
    from demo_g1_jelly_drop import Example
    from g1_robot import measure_joint_errors
    from jelly_stack import signed_six_volumes

SOURCE = "demos/demo/demo_g1_jelly_drop.py"


class CaptureViewer(ViewerNull):
    """Capture decorative meshes while keeping simulation and all forces unchanged."""

    def __init__(self, frames):
        self.decorations = {}
        super().__init__(num_frames=frames)

    def log_mesh(self, name, points, indices, *args, **kwargs):
        if name == "studio_floor":
            self.decorations[name] = (points.numpy(), indices.numpy(), kwargs)


def identifier(text):
    return re.sub(r"[^a-zA-Z0-9_]", "_", text)


def vector(value):
    return Gf.Vec3f(*(float(item) for item in value))


def transform_points(points, pose):
    xyz, scalar = pose[3:6], pose[6]
    return points + 2.0 * np.cross(xyz, np.cross(xyz, points) + scalar * points) + pose[:3]


def bounds(points):
    return np.stack((points.min(axis=0), points.max(axis=0)))


def set_pose(translate, orient, pose, frame=None):
    time_code = Usd.TimeCode.Default() if frame is None else Usd.TimeCode(frame)
    translate.Set(Gf.Vec3d(*(float(value) for value in pose[:3])), time_code)
    orient.Set(Gf.Quatf(float(pose[6]), vector(pose[3:6])), time_code)


class SimulationUSD:
    def __init__(self, path, example, decorations, frame_count):
        self.example = example
        # One USD time code represents one output frame, regardless of the number of physics substeps.
        self.fps = 1.0 / example.frame_dt
        self.stage = Usd.Stage.CreateNew(str(path))
        self.stage.SetFramesPerSecond(self.fps)
        self.stage.SetTimeCodesPerSecond(self.fps)
        self.stage.SetStartTimeCode(0)
        self.stage.SetEndTimeCode(frame_count)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        root = UsdGeom.Xform.Define(self.stage, "/World")
        self.stage.SetDefaultPrim(root.GetPrim())
        root.GetPrim().SetCustomDataByKey("source", SOURCE)
        self.materials = {}
        self.bodies = []
        self.robot_meshes = []
        self.jellies = []
        self.mesh_manifest = []
        self.timeline = []
        self.add_robot()
        triangles = example.model.tri_indices.numpy()
        for name, group in example.groups.items():
            first, last = group["triangle_range"]
            source, indices = np.unique(triangles[first:last], return_inverse=True)
            mesh = self.add_mesh(
                f"/World/Jellies/{name}",
                example.initial_particles[source],
                indices,
                group["color"],
                0.19,
                0.03,
                kind="jelly",
                layer=group["layer"],
            )
            self.jellies.append((mesh, source))
        for name, (points, indices, attributes) in decorations.items():
            self.add_mesh(
                f"/World/Stage/{name}",
                points,
                indices,
                attributes["color"],
                attributes.get("roughness", 0.5),
                attributes.get("metallic", 0.0),
                kind="decoration",
            )

    def material(self, color, roughness, metallic):
        key = (*color, roughness, metallic)
        if key not in self.materials:
            material = UsdShade.Material.Define(self.stage, f"/World/Materials/material_{len(self.materials):02d}")
            shader = UsdShade.Shader.Define(self.stage, f"{material.GetPath()}/PreviewSurface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(vector(color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            self.materials[key] = material
        return self.materials[key]

    def add_mesh(self, path, points, indices, color, roughness, metallic, normals=None, **metadata):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        indices = np.asarray(indices, dtype=np.int32).ravel()
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(indices) // 3, 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
        # FEM triangulation is authoritative; a DCC can add cosmetic subdivision in its render project.
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateDisplayColorAttr([vector(color)])
        mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(bounds(points)))
        if normals is not None:
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals, dtype=np.float32)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.material(color, roughness, metallic))
        for key, value in metadata.items():
            mesh.GetPrim().SetCustomDataByKey(key, value)
        self.mesh_manifest.append(
            {
                "path": path,
                "vertices": len(points),
                "triangles": len(indices) // 3,
                "color": [float(v) for v in color],
                **metadata,
            }
        )
        return mesh

    def add_robot(self):
        model = self.example.model
        transforms = model.shape_transform.numpy()
        scales = model.shape_scale.numpy()
        flags = model.shape_flags.numpy()
        parents = model.shape_body.numpy()
        types = model.shape_type.numpy()
        colors = model.shape_color.numpy()
        body_paths = {}
        for index in self.example.robot.body_indices:
            name = identifier(model.body_label[index].rsplit("/", 1)[-1])
            path = f"/World/G1/{name}_{index:02d}"
            body = UsdGeom.Xform.Define(self.stage, path)
            body.GetPrim().SetCustomDataByKey("body_index", int(index))
            self.bodies.append((int(index), body.AddTranslateOp(), body.AddOrientOp()))
            body_paths[index] = path
        for index, source in enumerate(model.shape_source):
            parent = int(parents[index])
            if parent not in body_paths or not flags[index] & int(newton.ShapeFlags.VISIBLE):
                continue
            if types[index] not in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
                raise ValueError(f"Unexpected visible G1 primitive {index}: {types[index]}")
            points = np.asarray(source.vertices, dtype=np.float32) * scales[index]
            normals = source.normals
            if normals is not None:
                normals = np.asarray(normals, dtype=np.float32) / scales[index]
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-20)
            shape_name = identifier(model.shape_label[index].rsplit("/", 1)[-1])
            mesh = self.add_mesh(
                f"{body_paths[parent]}/{shape_name}_{index:03d}",
                points,
                source.indices,
                colors[index],
                source.roughness if source.roughness is not None else 0.5,
                source.metallic if source.metallic is not None else 0.0,
                normals=normals,
                kind="robot",
                body_index=parent,
                shape_index=index,
            )
            xform = UsdGeom.Xformable(mesh)
            set_pose(xform.AddTranslateOp(), xform.AddOrientOp(), transforms[index])
            # Cache geometry in each body's local frame for accurate camera-framing bounds.
            self.robot_meshes.append((parent, transform_points(points, transforms[index])))

    def sample(self, frame, particles, bodies):
        for index, translate, orient in self.bodies:
            set_pose(translate, orient, bodies[index], frame)
        jelly_bounds = []
        for mesh, source in self.jellies:
            points = particles[source]
            extent = bounds(points)
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points), frame)
            mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(extent), frame)
            jelly_bounds.append(extent)
        robot_bounds = bounds(
            np.concatenate([transform_points(points, bodies[index]) for index, points in self.robot_meshes])
        )
        jelly_bounds = bounds(np.concatenate(jelly_bounds))
        self.timeline.append(
            {
                "frame": frame,
                "time_seconds": frame / self.fps,
                "pelvis_position": bodies[self.example.robot.root_body, :3].tolist(),
                "robot_bounds": robot_bounds.tolist(),
                "jelly_bounds": jelly_bounds.tolist(),
                "content_bounds": bounds(np.concatenate((robot_bounds, jelly_bounds))).tolist(),
            }
        )

    def save(self):
        self.stage.GetRootLayer().Save()


def main():
    export_parser = argparse.ArgumentParser(add_help=False)
    export_parser.add_argument("--output", type=Path, default=DIRECTORY / "outputs" / "simulation.usdc")
    export_args, physics_args = export_parser.parse_known_args()
    args = Example.create_parser().parse_args(["--viewer", "null", *physics_args])
    if args.num_frames < 1:
        raise ValueError("Export requires a positive --num-frames")
    export_args.output = export_args.output.resolve()
    export_args.output.parent.mkdir(parents=True, exist_ok=True)
    wp.init()
    wp.set_device(args.device or "cuda:0")
    started = time.perf_counter()
    viewer = CaptureViewer(args.num_frames)
    example = Example(viewer, args)
    archive = SimulationUSD(export_args.output, example, viewer.decorations, args.num_frames)
    particle_samples, body_samples, diagnostic_samples = [], [], []
    validation_errors = []
    minimum_tet_ratio = 1.0
    largest_joint_gap = 0.0
    largest_hinge_angle = 0.0
    for frame in range(args.num_frames + 1):
        if frame:
            example.step()
        wp.synchronize()
        particles, bodies = example.state_0.particle_q.numpy(), example.state_0.body_q.numpy()
        velocities = (example.state_0.particle_qd.numpy(), example.state_0.body_qd.numpy())
        if not all(np.isfinite(array).all() for array in (particles, bodies, *velocities)):
            raise RuntimeError(f"Nonfinite simulation state at frame {frame}; export stopped")
        particle_samples.append(particles)
        body_samples.append(bodies)
        archive.sample(frame, particles, bodies)
        joint_errors = measure_joint_errors(example.model, bodies, example.robot)
        largest_joint_gap = max(largest_joint_gap, joint_errors["max_anchor_separation"])
        largest_hinge_angle = max(largest_hinge_angle, joint_errors["max_hinge_axis_angle"])
        for group in example.groups.values():
            first, last = group["particle_range"]
            ratios = signed_six_volumes(particles[first:last], group["tet_indices"]) / group["initial_signed_volumes"]
            minimum_tet_ratio = min(minimum_tet_ratio, float(ratios.min()))
        if frame % 30 == 0 or frame == args.num_frames:
            try:
                example.test_final()
            except AssertionError as error:
                validation_errors.append({"frame": frame, "error": str(error)})
            diagnostic_samples.append(
                {
                    "frame": frame,
                    "contact_observations": example.observations.numpy().tolist(),
                    "penetration": example.penetration.numpy().tolist(),
                    "joint_errors": joint_errors,
                }
            )
            print(f"Exported frame {frame}/{args.num_frames} ({time.perf_counter() - started:.1f}s)", flush=True)
    archive.save()
    state_path = export_args.output.with_name("simulation_states.npz")
    np.savez_compressed(
        state_path,
        particle_q=np.stack(particle_samples),
        body_q=np.stack(body_samples),
        time_seconds=np.arange(args.num_frames + 1, dtype=np.float64) / archive.fps,
    )
    stage = Usd.Stage.Open(str(export_args.output))
    expected_samples = list(range(args.num_frames + 1))
    for mesh, _ in archive.jellies:
        if stage.GetPrimAtPath(mesh.GetPath()).GetAttribute("points").GetTimeSamples() != expected_samples:
            raise AssertionError(f"Incomplete mesh animation: {mesh.GetPath()}")
    for _, translate, orient in archive.bodies:
        for operation in (translate, orient):
            if stage.GetAttributeAtPath(operation.GetAttr().GetPath()).GetTimeSamples() != expected_samples:
                raise AssertionError(f"Incomplete body animation: {operation.GetAttr().GetPath()}")
    observations = example.observations.numpy()
    penetration = example.penetration.numpy()
    if observations[1] == 0:
        validation_errors.append({"frame": args.num_frames, "error": "G1 did not contact the jelly"})
    source_files = (
        REPOSITORY / SOURCE,
        DIRECTORY / "g1_robot.py",
        DIRECTORY / "jelly_stack.py",
    )
    metadata = {
        "source": SOURCE,
        "physics_parameters": vars(args),
        "source_sha256": {
            path.relative_to(REPOSITORY).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        "simulation": {
            "frames": args.num_frames,
            "samples": args.num_frames + 1,
            "fps": archive.fps,
            "start_time_code": 0,
            "end_time_code": args.num_frames,
            "duration_seconds": args.num_frames / archive.fps,
            "frame_seconds": example.frame_dt,
            "up_axis": "Z",
            "meters_per_unit": 1.0,
            "substep_seconds": example.dt,
            "jelly_cubes": len(example.groups),
            "robot_bodies": len(archive.bodies),
            "visual_meshes": len(archive.robot_meshes),
            "jelly_particles": example.jelly_stack["particle_range"][1],
            "elapsed_seconds": time.perf_counter() - started,
        },
        "validation": {
            "finite_all_frames": True,
            "exact_time_samples_verified": True,
            "actual_robot_jelly_contacts": int(observations[1]),
            "first_contact_seconds": float(observations[2] * example.dt),
            "peak_raw_penetration_m": float(penetration[0]),
            "contact_buffer_overflow": bool(penetration[1]),
            "minimum_tet_volume_ratio_all_frames": minimum_tet_ratio,
            "maximum_joint_gap_m": largest_joint_gap,
            "maximum_hinge_angle_rad": largest_hinge_angle,
            "example_assertion_errors": validation_errors,
        },
        "meshes": archive.mesh_manifest,
        "timeline": archive.timeline,
        "diagnostic_samples": diagnostic_samples,
        "files": {"usd": export_args.output.name, "raw_states": state_path.name},
    }
    metadata_path = export_args.output.with_name("simulation_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Saved {export_args.output} and {metadata_path}", flush=True)
    if validation_errors:
        print(f"WARNING: {len(validation_errors)} simulation checks failed; see preserved metadata", flush=True)


if __name__ == "__main__":
    main()

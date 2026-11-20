"""Bake the Piper, falling apples, and film bag to USD and independently verifiable raw states."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

import newton

if __package__:
    from ..demo import demo_piper_bag_transport as demo
    from .run_artifacts import seal_run
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))
    import demo_piper_bag_transport as demo
    from run_artifacts import seal_run

DIRECTORY = Path(__file__).resolve().parent


# Sources span the demo and Newton's solver package, so they are recorded relative to the Newton repository.
SOURCE_ROOT = Path(__file__).resolve().parents[5]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vec3(value):
    return Gf.Vec3f(*(float(v) for v in value))


def bounds(points):
    return np.stack((points.min(axis=0), points.max(axis=0)))


def box_mesh(half_extents):
    points = np.asarray(
        ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)),
        dtype=np.float32,
    )
    triangles = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ),
        dtype=np.int32,
    )
    return points * half_extents, triangles


def set_pose(translate, orient, pose, frame=Usd.TimeCode.Default()):
    translate.Set(Gf.Vec3d(*(float(v) for v in pose[:3])), frame)
    orient.Set(Gf.Quatf(float(pose[6]), vec3(pose[3:6])), frame)


class SimulationUSD:
    def __init__(self, path, example, frames):
        self.path = Path(path)
        self.stage = Usd.Stage.CreateNew(str(self.path))
        self.fps = 1.0 / example.frame_dt
        self.stage.SetFramesPerSecond(self.fps)
        self.stage.SetTimeCodesPerSecond(self.fps)
        self.stage.SetStartTimeCode(0)
        self.stage.SetEndTimeCode(frames)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.stage.SetDefaultPrim(UsdGeom.Xform.Define(self.stage, "/World").GetPrim())
        self.materials, self.textures = {}, {}
        self.meshes, self.bodies = [], []
        model = example.model
        self.bag = self.add_mesh(
            "/World/Bag/plastic_bag",
            example.state_0.particle_q.numpy()[: example.bag_particle_count],
            model.tri_indices.numpy(),
            demo.BAG_COLOR,
            0.3,
            0.0,
            kind="bag",
        )
        if hasattr(example, "bag_panel_points") and hasattr(example, "bag_panel_indices"):
            pattern = np.asarray(example.bag_panel_points)[np.asarray(example.bag_panel_indices)]
            pattern = np.asarray(pattern, dtype=np.float32).reshape(-1, 2)
            if len(pattern) != len(model.tri_indices.numpy().ravel()):
                raise ValueError("Film pattern coordinates must match every triangle corner")
            primvar = UsdGeom.PrimvarsAPI(self.bag).CreatePrimvar(
                "film_pattern_m",
                Sdf.ValueTypeNames.TexCoord2fArray,
                "faceVarying",
            )
            primvar.Set(Vt.Vec2fArray.FromNumpy(pattern))
            self.meshes[-1]["film_pattern_sha256"] = hashlib.sha256(pattern.tobytes()).hexdigest()
        # Fixed-base MJCF collapse attaches the base's visible shapes directly to the world (body -1).
        body_paths = {-1: "/World/Piper/fixed_base"}
        UsdGeom.Xform.Define(self.stage, body_paths[-1])
        for index in range(model.body_count):
            apple = index in example.bag_bodies
            group, prefix = ("Apples", "apple") if apple else ("Piper", "piper")
            path = f"/World/{group}/{prefix}_body_{index:02d}"
            body = UsdGeom.Xform.Define(self.stage, path)
            body.GetPrim().SetCustomDataByKey("body_index", index)
            body.GetPrim().SetCustomDataByKey("label", model.body_label[index])
            self.bodies.append((index, body.AddTranslateOp(), body.AddOrientOp()))
            body_paths[index] = path
        parents, flags = model.shape_body.numpy(), model.shape_flags.numpy()
        scales, transforms = model.shape_scale.numpy(), model.shape_transform.numpy()
        colors, types = model.shape_color.numpy(), model.shape_type.numpy()
        for index, source in enumerate(model.shape_source):
            parent = int(parents[index])
            # Export the original visual meshes only, never convex colliders or ABD coordinates.
            if parent not in body_paths or not flags[index] & int(newton.ShapeFlags.VISIBLE):
                continue
            if source is None:
                label = model.shape_label[index]
                if types[index] == newton.GeoType.BOX and parent == -1:
                    if label not in ("scene_floor", "bag_platform"):
                        raise ValueError(f"Unexpected visible environment box: {label}")
                    points, triangles = box_mesh(scales[index])
                    path = f"/World/Environment/{label}" if parent == -1 else f"{body_paths[parent]}/{label}"
                    mesh = self.add_mesh(
                        path,
                        points,
                        triangles,
                        colors[index],
                        0.4,
                        0.0,
                        kind="environment",
                        body_index=parent,
                        shape_index=index,
                    )
                    xform = UsdGeom.Xformable(mesh)
                    translate, orient = xform.AddTranslateOp(), xform.AddOrientOp()
                    set_pose(translate, orient, transforms[index])
                    # Keep the primitive parameters so validation can detect geometry or pose changes in the DCC input.
                    self.meshes[-1].update(
                        primitive="box", half_extents=scales[index].tolist(), local_pose=transforms[index].tolist()
                    )
                    continue
                if parent == -1:
                    continue
                raise ValueError(f"Expected a visual mesh for shape {index}")
            kind = "apple" if parent in example.bag_bodies else "robot"
            points = np.asarray(source.vertices, dtype=np.float32) * scales[index]
            normals = None if source.normals is None else np.asarray(source.normals, dtype=np.float32) / scales[index]
            if normals is not None:
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
            mesh = self.add_mesh(
                f"{body_paths[parent]}/{kind}_visual_{index:03d}",
                points,
                source.indices,
                colors[index],
                0.3 if source.roughness is None else source.roughness,
                0.0 if source.metallic is None else source.metallic,
                normals=normals,
                uvs=source.uvs,
                texture=source.texture,
                kind=kind,
                body_index=parent,
                shape_index=index,
            )
            xform = UsdGeom.Xformable(mesh)
            set_pose(xform.AddTranslateOp(), xform.AddOrientOp(), transforms[index])
            if kind == "apple":
                if normals is None or source.uvs is None or not source.texture:
                    raise ValueError("Original apple visuals require normals, UVs, and their color texture")
                self.meshes[-1].update(
                    primitive="mesh",
                    local_pose=transforms[index].tolist(),
                    points_sha256=hashlib.sha256(np.asarray(points, dtype=np.float32).tobytes()).hexdigest(),
                    indices_sha256=hashlib.sha256(np.asarray(source.indices, dtype=np.int32).tobytes()).hexdigest(),
                    normals_sha256=hashlib.sha256(np.asarray(normals, dtype=np.float32).tobytes()).hexdigest(),
                )

    def copy_texture(self, texture):
        if texture is None:
            return None
        if not isinstance(texture, (str, Path)):
            raise ValueError("This scene requires file-backed color textures")
        source = Path(texture).resolve()
        if source not in self.textures:
            digest = sha256(source)
            relative = f"textures/{digest[:16]}{source.suffix.lower()}"
            target = self.path.parent / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or sha256(target) != digest:
                shutil.copy2(source, target)
            self.textures[source] = relative
        return self.textures[source]

    def material(self, color, roughness, metallic, texture):
        key = (*map(float, color), float(roughness), float(metallic), texture)
        if key in self.materials:
            return self.materials[key]
        path = f"/World/Materials/material_{len(self.materials):03d}"
        material = UsdShade.Material.Define(self.stage, path)
        shader = UsdShade.Shader.Define(self.stage, path + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(vec3(color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        if texture:
            reader = UsdShade.Shader.Define(self.stage, path + "/UVReader")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            image = UsdShade.Shader.Define(self.stage, path + "/ColorTexture")
            image.CreateIdAttr("UsdUVTexture")
            image.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture))
            image.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            image.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            image.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.GetInput("diffuseColor").ConnectToSource(image.ConnectableAPI(), "rgb")
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        self.materials[key] = material
        return material

    def add_mesh(self, path, points, indices, color, roughness, metallic, normals=None, uvs=None, texture=None, **data):
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        indices = np.asarray(indices, dtype=np.int32).ravel()
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(indices) // 3, 3, dtype=np.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(bounds(points)))
        mesh.CreateDisplayColorAttr([vec3(color)])
        if normals is not None:
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(normals, dtype=np.float32)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        if uvs is not None:
            uvs = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
            if len(uvs) != len(points):
                raise ValueError(f"Expected per-vertex UVs: {path}")
            primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
            primvar.Set(Vt.Vec2fArray.FromNumpy(uvs))
        texture_path = self.copy_texture(texture)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(self.material(color, roughness, metallic, texture_path))
        for key, value in data.items():
            mesh.GetPrim().SetCustomDataByKey(key, value)
        self.meshes.append(
            {"path": path, "vertices": len(points), "triangles": len(indices) // 3, "texture": texture_path, **data}
        )
        return mesh

    def sample(self, frame, particles, bodies):
        self.bag.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(particles), frame)
        self.bag.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(bounds(particles)), frame)
        for index, translate, orient in self.bodies:
            set_pose(translate, orient, bodies[index], frame)


def release_motion_metadata(source=demo):
    metadata = {
        "release_start_seconds": source.RELEASE_START,
        "release_duration_seconds": source.RELEASE_DURATION,
        "release_joint_targets": list(source.PIPER_RETRACT_Q),
        "release_opening_m": source.RELEASE_OPENING,
        "retraction_delay_seconds": source.RETRACT_DELAY,
        "retraction_duration_seconds": source.RETRACT_DURATION,
        "grip_point_local": list(source.GRIP_POINT_LOCAL),
    }
    pickup_constants = ("GRASP_START", "LIFT_START")
    if any(hasattr(source, key) for key in pickup_constants):
        if not all(hasattr(source, key) for key in pickup_constants):
            raise ValueError("Incomplete pickup configuration")
        pickup_start, lift_start = (getattr(source, key) for key in pickup_constants)
        if not (
            np.isfinite(pickup_start)
            and np.isfinite(lift_start)
            and 0 <= pickup_start <= lift_start < source.RELEASE_START
        ):
            raise ValueError("Invalid pickup timing: require 0 <= pickup start <= lift start < release start")
        metadata.update(pickup_start_seconds=pickup_start, lift_start_seconds=lift_start)
    swing_constants = ("SWING_START", "SWING_PERIOD", "SWING_CYCLES")
    if any(hasattr(source, key) for key in swing_constants):
        if not all(hasattr(source, key) for key in swing_constants):
            raise ValueError("Incomplete swing configuration")
        metadata.update(
            swing_start_seconds=source.SWING_START,
            swing_period_seconds=source.SWING_PERIOD,
            swing_cycles=source.SWING_CYCLES,
        )
        if hasattr(source, "SWING_AMPLITUDE"):
            metadata["swing_yaw_amplitude_radians"] = source.SWING_AMPLITUDE
    for direction in ("horizontal", "vertical"):
        constants = tuple(f"{direction.upper()}_{field}" for field in ("START", "END", "PERIOD", "CYCLES"))
        if any(hasattr(source, key) for key in constants):
            if not all(hasattr(source, key) for key in constants):
                raise ValueError(f"Incomplete {direction} swing configuration")
            values = (getattr(source, key) for key in constants)
            metadata.update(
                {
                    f"{direction}_swing_{field}": value
                    for field, value in zip(("start_seconds", "end_seconds", "period_seconds", "cycles"), values)
                }
            )
    if hasattr(source, "VERTICAL_MOTION_PROFILE"):
        metadata["vertical_motion_profile"] = source.VERTICAL_MOTION_PROFILE
    if hasattr(source, "VERTICAL_FIXED_JOINTS"):
        metadata["vertical_fixed_joint_indices"] = list(source.VERTICAL_FIXED_JOINTS)
    for name, constant in (
        ("vertical_center_height_m", "VERTICAL_CENTER_HEIGHT"),
        ("vertical_rise_duration_seconds", "VERTICAL_RISE_DURATION"),
        ("vertical_drop_duration_seconds", "VERTICAL_DROP_DURATION"),
        ("vertical_wrist_pitch_amplitude_radians", "VERTICAL_WRIST_PITCH"),
    ):
        if hasattr(source, constant):
            metadata[name] = getattr(source, constant)
    if hasattr(source, "HORIZONTAL_MOTION_PROFILE"):
        metadata["horizontal_motion_profile"] = source.HORIZONTAL_MOTION_PROFILE
    place_constants = ("PLACE_START", "PLACE_DURATION", "PLACE_SETTLE_DURATION")
    if any(hasattr(source, key) for key in place_constants):
        if not all(hasattr(source, key) for key in place_constants):
            raise ValueError("Incomplete placement configuration")
        place_start, place_duration, settle_duration = (getattr(source, key) for key in place_constants)
        if (
            not all(np.isfinite(value) for value in (place_start, place_duration, settle_duration))
            or place_start < metadata.get("lift_start_seconds", 0.0)
            or place_duration <= 0
            or settle_duration < 0
            or not np.isclose(place_start + place_duration + settle_duration, source.RELEASE_START, rtol=0, atol=1.0e-9)
            or getattr(source, "AIRBORNE_RELEASE", False)
        ):
            raise ValueError("Invalid placement timing or airborne release")
        metadata.update(
            place_start_seconds=place_start,
            place_duration_seconds=place_duration,
            place_settle_duration_seconds=settle_duration,
        )
    loading_constants = ("APPLE_LOADING_START", "APPLE_LOADING_END")
    if any(hasattr(source, key) for key in loading_constants):
        if not all(hasattr(source, key) for key in loading_constants) or "pickup_start_seconds" not in metadata:
            raise ValueError("Incomplete apple loading configuration")
        metadata.update(
            loading_start_seconds=source.APPLE_LOADING_START,
            loading_end_seconds=source.APPLE_LOADING_END,
            loading_mode="gravity_drop",
        )
    end = max(
        source.RELEASE_START + source.RELEASE_DURATION,
        source.RELEASE_START + source.RETRACT_DELAY + source.RETRACT_DURATION,
    )
    sweep_constants = ("RELEASE_SWEEP_START", "RELEASE_SWEEP_DURATION", "RELEASE_SWEEP_YAW")
    if any(hasattr(source, key) for key in sweep_constants):
        if not all(hasattr(source, key) for key in sweep_constants):
            raise ValueError("Incomplete release sweep configuration")
        metadata.update(
            release_sweep_start_seconds=source.RELEASE_SWEEP_START,
            release_sweep_duration_seconds=source.RELEASE_SWEEP_DURATION,
            release_sweep_yaw_radians=source.RELEASE_SWEEP_YAW,
        )
        end = max(end, source.RELEASE_SWEEP_START + source.RELEASE_SWEEP_DURATION)
    for name in ("toss", "lower", "clear", "unhook"):
        constants = (f"{name.upper()}_START", f"{name.upper()}_DURATION", f"PIPER_{name.upper()}_Q")
        if not any(hasattr(source, key) for key in constants):
            continue
        if not all(hasattr(source, key) for key in constants):
            raise ValueError(f"Incomplete {name} configuration")
        start, duration, targets = (getattr(source, key) for key in constants)
        metadata.update(
            {
                f"{name}_start_seconds": start,
                f"{name}_duration_seconds": duration,
                f"{name}_joint_targets": list(targets),
            }
        )
        # Tossing precedes opening; the later arm stages may continue after the jaw motion ends.
        if name != "toss":
            end = max(end, start + duration)
    for name in ("apple", "floor", "gripper"):
        constant = f"{name.upper()}_FRICTION"
        if hasattr(source, constant):
            metadata[f"{name}_friction"] = getattr(source, constant)
        rolling_constant = f"{name.upper()}_ROLLING_FRICTION"
        if hasattr(source, rolling_constant):
            metadata[f"{name}_rolling_friction"] = getattr(source, rolling_constant)
        torsional_constant = f"{name.upper()}_TORSIONAL_FRICTION"
        if hasattr(source, torsional_constant):
            metadata[f"{name}_torsional_friction"] = getattr(source, torsional_constant)
    for name in ("stiffness", "damping"):
        constant = f"FLOOR_CONTACT_{name.upper()}"
        if hasattr(source, constant):
            metadata[f"floor_contact_{name}"] = getattr(source, constant)
    for name, constant in (
        ("cloth_contact_friction", "CLOTH_CONTACT_FRICTION"),
        ("finger_kp", "PIPER_FINGER_KP"),
        ("finger_kd", "PIPER_FINGER_KD"),
        ("bag_finger_kp", "PIPER_BAG_FINGER_KP"),
        ("bag_finger_kd", "PIPER_BAG_FINGER_KD"),
        ("bag_finger_gain_ramp_seconds", "PIPER_BAG_GAIN_RAMP"),
    ):
        if hasattr(source, constant):
            metadata[name] = getattr(source, constant)
    if hasattr(source, "PIPER_BAG_GAIN_RAMP"):
        metadata["bag_finger_gain_start_seconds"] = source.APPLE_LOADING_END
    if hasattr(source, "DURATION_SECONDS"):
        metadata["duration_seconds"] = source.DURATION_SECONDS
    if hasattr(source, "AIRBORNE_RELEASE"):
        metadata["airborne_release"] = source.AIRBORNE_RELEASE
        metadata["arm_motion_end_seconds"] = end
        # A falling bag is independent of the arm once the opening motion completes.
        if source.AIRBORNE_RELEASE:
            end = source.RELEASE_START + source.RELEASE_DURATION
    return metadata, end


def simulation_phase(time_seconds, motion_metadata, release_end):
    if (
        motion_metadata.get("loading_start_seconds", 0.0)
        <= time_seconds
        < motion_metadata.get("loading_end_seconds", 0.0)
    ):
        return "loading"
    # Approach and jaw closing do not require an already retained handle.
    if time_seconds < motion_metadata.get("lift_start_seconds", 0.0):
        return "pickup"
    if time_seconds < motion_metadata["release_start_seconds"]:
        return "place" if time_seconds >= motion_metadata.get("place_start_seconds", float("inf")) else "carry"
    if time_seconds < release_end:
        return "release"
    return "placed" if "place_start_seconds" in motion_metadata else "drop"


def executed_motion_metadata(example, initial_body_poses, source=demo):
    """Snapshot the final planner state after feedback has updated the executed pickup targets."""
    apple_indices = np.asarray(example.bag_bodies, dtype=int)
    metadata = {
        "apple_body_indices": apple_indices.tolist(),
        "initial_apple_positions": np.asarray(initial_body_poses)[apple_indices, :3].tolist(),
    }
    if hasattr(example, "mouth_vertices"):
        metadata["mouth_vertex_indices"] = np.asarray(example.mouth_vertices, dtype=int).tolist()
    if hasattr(example, "grip_vertices"):
        metadata["grip_vertex_indices"] = np.asarray(example.grip_vertices, dtype=int).tolist()
    if hasattr(example, "wrist_body"):
        metadata.update(wrist_body_index=int(example.wrist_body), finger_tip_local_z=source.FINGER_TIP_LOCAL_Z)
        metadata["arm_body_indices"] = [
            next(i for i, label in enumerate(example.model.body_label) if label.endswith(f"/link{link}"))
            for link in range(1, 7)
        ]
    if hasattr(example, "plan_times") and hasattr(example, "plan_q"):
        targets = np.asarray(example.plan_q)
        metadata.update(
            motion_plan_times_seconds=np.asarray(example.plan_times).tolist(),
            motion_plan_joint_targets=targets.tolist(),
            release_joint_targets=targets[-1].tolist(),
        )
    if hasattr(example, "handle_plan_updated"):
        metadata["handle_plan_updated"] = bool(example.handle_plan_updated)
    if hasattr(example, "handle_grasp_refined"):
        metadata["handle_grasp_refined"] = bool(example.handle_grasp_refined)
    return metadata


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, default=DIRECTORY / "outputs/simulation.usdc")
    export_args, physics_args = parser.parse_known_args()
    physics_parser = demo.Example.create_parser()
    physics_parser.set_defaults(
        num_frames=demo.DEFAULT_NUM_FRAMES,
        viewer="null",
        telemetry_csv=export_args.output.resolve().with_name("motor_telemetry.csv"),
    )
    args = physics_parser.parse_args(physics_args)
    if args.num_frames < 1:
        parser.error("--num-frames must be positive")
    path = export_args.output.resolve()
    if path.suffix.lower() not in (".usd", ".usdc", ".usda"):
        parser.error("--output must be a USD file")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_telemetry and not args.telemetry_csv.resolve().is_relative_to(path.parent):
        parser.error("Keep --telemetry-csv within the export directory for a portable simulation bundle")
    wp.config.kernel_cache_dir = str(demo.REPO_ROOT / "build/warp_effort_cache")
    wp.set_device(args.device or "cuda:0")
    started = time.perf_counter()
    motion_metadata, release_end = release_motion_metadata()
    example = demo.Example(newton.viewer.ViewerNull(), args)
    archive = SimulationUSD(path, example, args.num_frames)
    particles, bodies, body_velocities, timeline, errors = [], [], [], [], []
    for frame in range(args.num_frames + 1):
        if frame:
            example.step()
        points = example.state_0.particle_q.numpy()[: example.bag_particle_count].copy()
        poses = example.state_0.body_q.numpy()
        if not np.isfinite(points).all() or not np.isfinite(poses).all():
            raise RuntimeError(f"Nonfinite state at frame {frame}; export stopped")
        particles.append(points)
        bodies.append(poses)
        body_velocities.append(example.state_0.body_qd.numpy())
        archive.sample(frame, points, poses)
        grip = example.current_grip_point(poses)
        sample_time = frame / archive.fps
        phase = simulation_phase(sample_time, motion_metadata, release_end)
        timeline.append(
            {
                "frame": frame,
                "time_seconds": sample_time,
                "phase": phase,
                "bag_bounds": bounds(points).tolist(),
                "grip_error_m": float(np.linalg.norm(points[example.grip_vertices].mean(axis=0) - grip)),
            }
        )
        try:
            example.test_final()
        except AssertionError as error:
            errors.append({"frame": frame, "error": str(error)})
        if frame % 30 == 0 or frame == args.num_frames:
            print(f"Exported {frame}/{args.num_frames} ({time.perf_counter() - started:.1f}s)", flush=True)
    example.finish_telemetry()
    archive.stage.GetRootLayer().Save()
    state_path = path.with_name(path.stem + "_states.npz")
    np.savez_compressed(
        state_path,
        particle_q=np.stack(particles),
        body_q=np.stack(bodies),
        body_qd=np.stack(body_velocities),
        triangles=example.model.tri_indices.numpy(),
        time_seconds=np.arange(args.num_frames + 1, dtype=np.float64) / archive.fps,
    )
    # The apple is downloaded into the Style3D Sim asset cache, so it is not a repository-relative source.
    sources = [Path(demo.__file__), Path(__file__)]
    # The solver adapter and joint-drive effort live in Newton's coupled solver package.
    import newton._src.solvers.coupled_high_accuracy as solver_package

    sources.extend(
        Path(solver_package.__file__).parent / filename
        for filename in ("solver_coupled_high_accuracy.py", "joint_drive_effort.py")
    )
    for filename in ("piper_motion.py", "bag_pattern.py", "bag_bending.py", "film_uv.py", "motor_telemetry.py"):
        dependency = DIRECTORY / filename
        if dependency.is_file():
            sources.append(dependency)
    # Read the plan after simulation so the archive records sensed-handle corrections, not initial placeholders.
    motion_metadata.update(executed_motion_metadata(example, bodies[0]))
    if "mouth_vertex_indices" in motion_metadata:
        mouth = particles[0][motion_metadata["mouth_vertex_indices"]]
        # The low cut lip defines the mouth plane; the same boundary continues up both U-shaped handles.
        motion_metadata["initial_mouth_height_m"] = float(mouth[:, 2].min())
    metadata = {
        "source_sha256": {p.relative_to(SOURCE_ROOT).as_posix(): sha256(p) for p in sources},
        "physics_parameters": vars(args),
        "simulation": {
            "frames": args.num_frames,
            "samples": args.num_frames + 1,
            "fps": archive.fps,
            "substep_seconds": example.sim_dt,
            "bag_particles": example.bag_particle_count,
            "body_count": example.model.body_count,
            "apple_count": len(example.bag_bodies),
            "apple_collision_vertices": example.apple_collision_vertices.tolist(),
            "apple_collision_margin": demo.APPLE_COLLISION_MARGIN,
            "apple_masses_kg": example.model.body_mass.numpy()[example.bag_bodies].tolist(),
            "gravity_m_s2": example.model.gravity.numpy()[0].tolist(),
            "cloth_contact_radius": demo.PARTICLE_RADIUS,
            "apple_cloth_contact_gap": demo.APPLE_COLLISION_MARGIN + demo.PARTICLE_RADIUS,
            "arm_kp": demo.PIPER_ARM_KP,
            "arm_kd": demo.PIPER_ARM_KD,
            "bounce_amplitude_m": demo.BOUNCE_AMPLITUDE,
            "bounce_frequency_hz": demo.BOUNCE_FREQUENCY,
            **motion_metadata,
            "grip_reference": "finger_midpoint",
            "finger_body_indices": example.gripper_bodies,
            "constraint_alpha": example.solver.config.constraint_alpha,
            "nonlinear_solver": example.solver.config.nonlinear_solver.name,
            "nonlinear_iterations": example.solver.config.nonlinear_iterations,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "meshes": archive.meshes,
        "timeline": timeline,
        # Free-flight separation after opening is expected; measure attachment only while carrying.
        "validation": {
            "example_assertion_errors": errors,
            "maximum_grip_error_m": max((s["grip_error_m"] for s in timeline if s["phase"] == "carry"), default=0.0),
            "maximum_handle_separation_m": max(s["grip_error_m"] for s in timeline),
        },
        "files": {"usd": path.name, "raw_states": state_path.name},
        "file_sha256": {p.name: sha256(p) for p in (path, state_path)},
        "texture_sha256": {relative: sha256(path.parent / relative) for relative in archive.textures.values()},
    }
    if example.telemetry is not None:
        csv_path = example.telemetry.writer.path
        for key, artifact in (
            ("motor_telemetry", csv_path),
            ("motor_telemetry_metadata", csv_path.with_suffix(".json")),
        ):
            relative = artifact.relative_to(path.parent).as_posix()
            metadata["files"][key] = relative
            metadata["file_sha256"][relative] = sha256(artifact)
    metadata_path = path.with_name(path.stem + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Saved {path} and {metadata_path}", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} physics checks failed; inspect {metadata_path}")
    if example.telemetry is not None:
        manifest = seal_run(metadata_path, SOURCE_ROOT)
        print(f"Sealed simulation run {manifest['run_id']}", flush=True)


if __name__ == "__main__":
    main()

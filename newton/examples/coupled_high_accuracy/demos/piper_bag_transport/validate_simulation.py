"""Compare every USD sample with the raw simulation states, without loading the CUDA solver."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdShade

if __package__:
    from .run_artifacts import validate_sources
else:
    from run_artifacts import validate_sources

DIRECTORY = Path(__file__).resolve().parent
# The Newton repository: export sources include the solver package outside the demo tree.
REPOSITORY = DIRECTORY.parents[4]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_hash(path, expected):
    require(path.is_file(), f"Missing file: {path}")
    require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, f"File changed since export: {path}")


def validate_motor_telemetry(root, metadata):
    """Check recorded motor samples independently of the GPU recorder and plotting dependencies."""
    files = metadata["files"]
    if "motor_telemetry" not in files:
        return {}
    require("motor_telemetry_metadata" in files, "Missing motor telemetry metadata")
    csv_path = root / files["motor_telemetry"]
    sidecar = json.loads((root / files["motor_telemetry_metadata"]).read_text(encoding="utf-8"))
    sim = metadata["simulation"]
    fps, samples = sim["fps"], sim["samples"]
    require(np.isfinite(fps) and fps > 0, "Invalid motor telemetry frame rate")
    require(
        sidecar.get("schema_version") == 1 and sidecar.get("csv") == csv_path.name,
        "Wrong motor telemetry schema or CSV reference",
    )
    require(sidecar.get("complete") is True, "Motor telemetry recording is incomplete")
    require(
        sidecar.get("fps") == fps and sidecar.get("frame_count") == samples and sidecar.get("rows_per_frame") == 8,
        "Motor telemetry sample counts or frame rate disagree",
    )
    substeps = metadata.get("physics_parameters", {}).get("substeps")
    require(
        isinstance(substeps, int) and substeps > 0 and sidecar.get("physics_substeps_per_frame") == substeps,
        "Motor telemetry substep count disagrees with the simulation",
    )
    require(
        np.isclose(sim.get("substep_seconds", np.nan) * substeps * fps, 1.0, rtol=0, atol=1.0e-9),
        "Motor telemetry substep duration disagrees with the simulation",
    )
    joints = {
        f"joint{index}": ("revolute", "rad", "Nm") if index <= 6 else ("prismatic", "m", "N") for index in range(1, 9)
    }
    channels = sidecar.get("joints", [])
    require(
        isinstance(channels, list) and len(channels) == 8 and {joint.get("name") for joint in channels} == set(joints),
        "Wrong motor telemetry channel manifest",
    )
    for joint in channels:
        require(
            (joint.get("type"), joint.get("position_unit"), joint.get("effort_unit")) == joints[joint["name"]],
            "Wrong motor telemetry channel units",
        )
    columns = (
        "frame",
        "time_s",
        "joint_name",
        "joint_type",
        "position_unit",
        "effort_unit",
        "target_time_s",
        "target_position",
        "target_velocity",
        "feedforward_effort",
        "feedback_effort",
        "feedback_effort_mean",
        "feedback_effort_min",
        "feedback_effort_max",
        "feedback_valid",
    )
    with csv_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        require(
            reader.fieldnames is not None
            and len(reader.fieldnames) == len(columns)
            and set(reader.fieldnames) == set(columns),
            "Wrong motor telemetry CSV columns",
        )
        rows = list(reader)
    require(len(rows) == samples * 8, "Wrong motor telemetry row count")
    seen = set()
    for row in rows:
        require(None not in row and all(value is not None for value in row.values()), "Malformed motor telemetry row")
        try:
            frame = int(row["frame"])
        except ValueError as error:
            raise ValueError("Invalid motor telemetry frame index") from error
        joint = row["joint_name"]
        require(0 <= frame < samples and joint in joints, "Unexpected motor telemetry frame or joint")
        require((frame, joint) not in seen, "Duplicate motor telemetry frame/joint sample")
        seen.add((frame, joint))
        require(
            (row["joint_type"], row["position_unit"], row["effort_unit"]) == joints[joint],
            "Wrong motor telemetry CSV units",
        )
        for key in ("time_s", "target_time_s", "target_position", "target_velocity", "feedforward_effort"):
            try:
                value = float(row[key])
            except ValueError as error:
                raise ValueError(f"Invalid motor telemetry {key}") from error
            require(np.isfinite(value), f"Nonfinite motor telemetry {key}")
            if key == "time_s":
                require(np.isclose(value, frame / fps, rtol=0, atol=1.0e-10), "Wrong motor telemetry sample time")
        feedback_keys = ("feedback_effort", "feedback_effort_mean", "feedback_effort_min", "feedback_effort_max")
        if frame == 0:
            require(
                row["feedback_valid"] == "0" and all(row[key] == "" for key in feedback_keys),
                "Initial motor telemetry feedback must be unavailable",
            )
            continue
        require(row["feedback_valid"] == "1", "Missing valid motor telemetry feedback")
        try:
            last, mean, low, high = (float(row[key]) for key in feedback_keys)
        except ValueError as error:
            raise ValueError("Invalid motor telemetry feedback") from error
        require(np.isfinite((last, mean, low, high)).all(), "Nonfinite motor telemetry feedback")
        require(low <= last <= high and low <= mean <= high, "Motor telemetry feedback lies outside substep extrema")
    return {"telemetry_rows": len(rows), "telemetry_frames": samples}


def validate_release_timeline(sim, timeline, validation):
    keys = (
        "release_start_seconds",
        "release_duration_seconds",
        "retraction_delay_seconds",
        "retraction_duration_seconds",
        "release_opening_m",
    )
    sweep_keys = ("release_sweep_start_seconds", "release_sweep_duration_seconds", "release_sweep_yaw_radians")
    pickup_keys = ("pickup_start_seconds", "lift_start_seconds")
    loading_keys = ("loading_start_seconds", "loading_end_seconds", "loading_mode")
    swing_keys = ("swing_start_seconds", "swing_period_seconds", "swing_cycles")
    directional_keys = tuple(
        f"{direction}_swing_{field}"
        for direction in ("horizontal", "vertical")
        for field in ("start_seconds", "end_seconds", "period_seconds", "cycles")
    )
    place_keys = ("place_start_seconds", "place_duration_seconds", "place_settle_duration_seconds")
    motions = {
        name: (f"{name}_start_seconds", f"{name}_duration_seconds", f"{name}_joint_targets")
        for name in ("toss", "lower", "clear", "unhook")
    }
    motion_keys = tuple(key for group in motions.values() for key in group)
    if not any(
        key in sim
        for key in (
            *keys,
            *sweep_keys,
            *pickup_keys,
            *loading_keys,
            *swing_keys,
            *place_keys,
            *motion_keys,
            "release_joint_targets",
            "airborne_release",
            "arm_motion_end_seconds",
        )
    ):
        return {}
    require(all(key in sim for key in keys), "Incomplete release parameters")
    require(all(np.isfinite(sim[key]) and sim[key] >= 0 for key in keys), "Invalid release parameters")
    require(
        sim["release_duration_seconds"] > 0 and sim["retraction_duration_seconds"] > 0 and sim["release_opening_m"] > 0,
        "Release duration and opening must be positive",
    )
    start = sim["release_start_seconds"]
    lift_start = 0.0
    if any(key in sim for key in pickup_keys):
        require(all(key in sim for key in pickup_keys), "Incomplete pickup parameters")
        require(all(np.isfinite(sim[key]) for key in pickup_keys), "Invalid pickup timing")
        pickup_start, lift_start = (sim[key] for key in pickup_keys)
        require(
            0 <= pickup_start <= lift_start < start,
            "Invalid pickup timing: require 0 <= pickup start <= lift start < release start",
        )
    swing_end = None
    if any(key in sim for key in swing_keys):
        require(all(key in sim for key in swing_keys), "Incomplete swing parameters")
        require(all(np.isfinite(sim[key]) for key in swing_keys), "Invalid swing timing")
        swing_start, swing_period, swing_cycles = (sim[key] for key in swing_keys)
        require(
            lift_start <= swing_start < start
            and swing_period > 0
            and swing_cycles > 0
            and float(swing_cycles).is_integer(),
            "Invalid swing timing",
        )
        swing_end = swing_start + swing_period * swing_cycles
        require(swing_end <= start + 1.0e-9, "Swing must finish before release")
    if any(key in sim for key in directional_keys):
        require(all(key in sim for key in directional_keys), "Incomplete sequential swing parameters")
        for direction in ("horizontal", "vertical"):
            begin, finish, period, cycles = (
                sim[f"{direction}_swing_{field}"]
                for field in ("start_seconds", "end_seconds", "period_seconds", "cycles")
            )
            require(
                np.isfinite((begin, finish, period, cycles)).all()
                and lift_start <= begin < finish <= start
                and period > 0
                and cycles > 0
                and float(cycles).is_integer()
                and np.isclose(begin + period * cycles, finish, rtol=0, atol=1.0e-8),
                f"Invalid {direction} swing timing",
            )
        require(
            sim["horizontal_swing_end_seconds"] <= sim["vertical_swing_start_seconds"],
            "Horizontal swinging must finish before vertical swinging starts",
        )
        require(
            np.isclose(sim["horizontal_swing_start_seconds"], sim["swing_start_seconds"], rtol=0, atol=1.0e-8),
            "Sequential swinging must start with the horizontal stage",
        )
        swing_end = sim["vertical_swing_end_seconds"]
    place_start = float("inf")
    if any(key in sim for key in place_keys):
        require(all(key in sim for key in place_keys), "Incomplete placement parameters")
        require(all(np.isfinite(sim[key]) for key in place_keys), "Invalid placement timing")
        place_start, place_duration, settle_duration = (sim[key] for key in place_keys)
        require(
            lift_start <= place_start
            and place_duration > 0
            and settle_duration >= 0
            and np.isclose(place_start + place_duration + settle_duration, start, rtol=0, atol=1.0e-9),
            "Invalid placement timing",
        )
        require(not sim.get("airborne_release", False), "Placement must release the bag on the platform")
        if swing_end is not None:
            require(
                np.isclose(swing_end, place_start, rtol=0, atol=1.0e-9), "Placement must start when swinging finishes"
            )
    loading_start = loading_end = 0.0
    if any(key in sim for key in loading_keys):
        require(
            all(key in sim for key in loading_keys) and all(key in sim for key in pickup_keys),
            "Incomplete apple loading parameters",
        )
        loading_start, loading_end = sim["loading_start_seconds"], sim["loading_end_seconds"]
        require(
            np.isfinite(loading_start)
            and np.isfinite(loading_end)
            and loading_start == 0 < loading_end <= sim["pickup_start_seconds"],
            "Invalid apple loading timing",
        )
        require(sim["loading_mode"] == "gravity_drop", "Apples must fall from the initial state under gravity")
    end = max(
        start + sim["release_duration_seconds"],
        start + sim["retraction_delay_seconds"] + sim["retraction_duration_seconds"],
    )
    if any(key in sim for key in sweep_keys):
        require(all(key in sim for key in sweep_keys), "Incomplete release sweep parameters")
        require(all(np.isfinite(sim[key]) for key in sweep_keys), "Invalid release sweep parameters")
        sweep_start, sweep_duration = sim["release_sweep_start_seconds"], sim["release_sweep_duration_seconds"]
        sweep_end = sweep_start + sweep_duration
        require(sweep_start >= 0 and sweep_duration > 0, "Invalid release sweep timing")
        require(sweep_start - 1.0e-9 <= start <= sweep_end + 1.0e-9, "Release must start during its finishing sweep")
        end = max(end, sweep_end)
    if "release_joint_targets" in sim:
        targets = np.asarray(sim["release_joint_targets"])
        require(targets.shape == (8,) and np.isfinite(targets).all(), "Invalid release joint targets")
        require(
            np.allclose(targets[-2:], [sim["release_opening_m"], -sim["release_opening_m"]], rtol=0, atol=1.0e-9),
            "Release joint targets do not match finger opening",
        )
    for name, motion_keys in motions.items():
        if not any(key in sim for key in motion_keys):
            continue
        require(all(key in sim for key in motion_keys), f"Incomplete {name} parameters")
        motion_start, duration = sim[motion_keys[0]], sim[motion_keys[1]]
        require(
            np.isfinite(motion_start) and motion_start >= 0 and np.isfinite(duration) and duration > 0,
            f"Invalid {name} timing",
        )
        targets = np.asarray(sim[motion_keys[2]])
        require(targets.shape == (8,) and np.isfinite(targets).all(), f"Invalid {name} joint targets")
        if name == "toss":
            require(motion_start + duration <= start + 1.0e-9, "Toss must finish before release")
        else:
            if name == "clear":
                require(motion_start >= start - 1.0e-9, "Clearing must start after release")
            elif name == "unhook":
                require(motion_start >= start - 1.0e-9, "Unhooking must start after release")
                if "clear_start_seconds" in sim:
                    clear_end = sim["clear_start_seconds"] + sim["clear_duration_seconds"]
                    require(motion_start >= clear_end - 1.0e-9, "Unhooking must start after clearing finishes")
            end = max(end, motion_start + duration)
    if "arm_motion_end_seconds" in sim:
        require(
            np.isfinite(sim["arm_motion_end_seconds"])
            and np.isclose(sim["arm_motion_end_seconds"], end, rtol=0, atol=1.0e-9),
            "Wrong arm motion end",
        )
    if "airborne_release" in sim:
        require(isinstance(sim["airborne_release"], bool), "Invalid airborne release flag")
        if sim["airborne_release"]:
            end = start + sim["release_duration_seconds"]
    if "duration_seconds" in sim:
        require(np.isfinite(sim["duration_seconds"]) and sim["duration_seconds"] >= end, "Invalid simulation duration")
    if "grip_point_local" in sim:
        point = np.asarray(sim["grip_point_local"])
        require(point.shape == (3,) and np.isfinite(point).all(), "Invalid local grip point")
    for key in (
        "apple_friction",
        "floor_friction",
        "gripper_friction",
        "apple_rolling_friction",
        "floor_rolling_friction",
        "gripper_rolling_friction",
        "apple_torsional_friction",
        "floor_torsional_friction",
        "gripper_torsional_friction",
        "floor_contact_stiffness",
        "floor_contact_damping",
        "cloth_contact_friction",
        "finger_kp",
        "finger_kd",
        "bag_finger_kp",
        "bag_finger_kd",
        "bag_finger_gain_start_seconds",
        "bag_finger_gain_ramp_seconds",
    ):
        if key in sim:
            require(np.isfinite(sim[key]) and sim[key] >= 0, f"Invalid {key}")
    for sample in timeline:
        time_seconds = sample["time_seconds"]
        if loading_start <= time_seconds < loading_end:
            phase = "loading"
        elif time_seconds < lift_start:
            phase = "pickup"
        elif place_start <= time_seconds < start:
            phase = "place"
        else:
            final_phase = "placed" if np.isfinite(place_start) else "drop"
            phase = "carry" if time_seconds < start else "release" if time_seconds < end else final_phase
        require(sample.get("phase") == phase, f"Wrong release phase at frame {sample['frame']}")
        require(
            "grip_error_m" in sample and np.isfinite(sample["grip_error_m"]) and sample["grip_error_m"] >= 0,
            f"Invalid handle separation at frame {sample['frame']}",
        )
    carry_error = max((sample["grip_error_m"] for sample in timeline if sample["phase"] == "carry"), default=0.0)
    handle_separation = max(sample["grip_error_m"] for sample in timeline)
    require(validation["maximum_grip_error_m"] == carry_error, "Grip error must include only carry samples")
    require(validation.get("maximum_handle_separation_m") == handle_separation, "Wrong maximum handle separation")
    return {"maximum_handle_separation_m": handle_separation}


def posed_collider(vertices, pose):
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(pose[3:7]).apply(vertices) + pose[:3]


def validate_placement_support(stage, sim, meshes, particles, times, bodies):
    """Confirm support before opening from geometry, independently of the controller's intended poses."""
    if "place_start_seconds" not in sim or times[-1] < sim["release_start_seconds"]:
        return {}
    prior = np.flatnonzero(times < sim["release_start_seconds"])
    require(len(prior) > 0, "Missing sample before placement release")
    frame = int(prior[-1])
    cache = UsdGeom.XformCache(float(frame))

    def world_points(entry):
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(entry["path"]))
        points = np.asarray(mesh.GetPointsAttr().Get(frame))
        matrix = np.asarray(cache.GetLocalToWorldTransform(mesh.GetPrim()))
        return points @ matrix[:3, :3] + matrix[3, :3]

    platform = next((entry for entry in meshes if entry["path"].endswith("/bag_platform")), None)
    require(platform is not None, "Missing placement platform")
    platform_points = world_points(platform)
    minimum, maximum = platform_points.min(axis=0), platform_points.max(axis=0)
    top = maximum[2]
    film_radius = sim.get("cloth_contact_radius", 0.0)
    bag = particles[frame]
    on_platform = np.all((bag[:, :2] >= minimum[:2]) & (bag[:, :2] <= maximum[:2]), axis=1)
    require(np.any(on_platform), "Bag is outside the platform before release")
    bag_gap = float(bag[on_platform, 2].min() - top)
    require(-0.01 <= bag_gap <= film_radius + 0.01, "Bag must touch the platform before release")
    apple_gaps = []
    for entry in meshes:
        if entry["kind"] != "apple":
            continue
        vertices = np.asarray(sim["apple_collision_vertices"])
        apple = posed_collider(vertices, bodies[frame, entry["body_index"]])
        center = 0.5 * (apple.min(axis=0) + apple.max(axis=0))
        require(
            np.all(center[:2] >= minimum[:2]) and np.all(center[:2] <= maximum[:2]),
            "A placed apple is outside the platform before release",
        )
        gap = float(apple[:, 2].min() - top)
        require(gap >= -0.01, "A placed apple penetrates the platform before release")
        apple_gaps.append(gap)
    # Apples can stack inside the bag, so the lowest collision hull establishes support for the load.
    support_gap = sim.get("apple_collision_margin", 0.0) + film_radius + 0.01
    require(
        bool(apple_gaps) and min(apple_gaps) <= support_gap,
        "The apples must be supported by the platform before release",
    )
    report = {
        "placement_support_frame": frame,
        "placement_bag_gap_m": bag_gap,
        "placement_lowest_apple_gap_m": min(apple_gaps),
    }
    if "vertical_swing_end_seconds" in sim and times[-1] >= sim.get("duration_seconds", float("inf")) - 1e-8:
        final_bag = particles[-1]
        require(
            np.all(final_bag[:, :2] >= minimum[:2] - film_radius)
            and np.all(final_bag[:, :2] <= maximum[:2] + film_radius),
            "The placed bag must remain on the platform after withdrawal",
        )
        final_bag_gap = float(final_bag[:, 2].min() - top)
        require(-0.01 <= final_bag_gap <= film_radius + 0.01, "The bag must remain supported after withdrawal")
        final_gaps = []
        for entry in meshes:
            if entry["kind"] != "apple":
                continue
            apple = posed_collider(np.asarray(sim["apple_collision_vertices"]), bodies[-1, entry["body_index"]])
            require(
                np.all(apple[:, :2] >= minimum[:2]) and np.all(apple[:, :2] <= maximum[:2]),
                "All placed apples must remain on the platform after withdrawal",
            )
            final_gaps.append(float(apple[:, 2].min() - top))
        require(
            min(final_gaps) >= -0.01 and min(final_gaps) <= support_gap,
            "The apples must remain supported after withdrawal",
        )
        report.update(final_placement_bag_gap_m=final_bag_gap, final_placement_lowest_apple_gap_m=min(final_gaps))
    return report


def point_triangle_distances(point, triangles):
    """Return point-to-triangle distance, including projections outside each triangle."""
    a, b, c = np.moveaxis(triangles, 1, 0)
    u, v = b - a, c - a
    normal = np.cross(u, v)
    denominator = np.maximum(np.sum(normal * normal, axis=1), 1.0e-30)
    projected = point - normal * (np.sum((point - a) * normal, axis=1) / denominator)[:, None]
    w = projected - a
    uv, uu, vv = np.sum(u * v, axis=1), np.sum(u * u, axis=1), np.sum(v * v, axis=1)
    uw, vw = np.sum(u * w, axis=1), np.sum(v * w, axis=1)
    determinant = np.maximum(uu * vv - uv * uv, 1.0e-30)
    beta, gamma = (vv * uw - uv * vw) / determinant, (uu * vw - uv * uw) / determinant
    distance = np.linalg.norm(projected - point, axis=1)
    distance[(beta < 0) | (gamma < 0) | (beta + gamma > 1)] = np.inf
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        along = np.clip(np.sum((point - start) * edge, axis=1) / np.maximum(np.sum(edge * edge, axis=1), 1.0e-30), 0, 1)
        distance = np.minimum(distance, np.linalg.norm(start + along[:, None] * edge - point, axis=1))
    return distance


def convex_triangle_distance(vertices, triangle):
    """Solve the convex closest-points problem without replacing an apple with a bounding sphere."""
    from scipy.optimize import LinearConstraint, minimize
    from scipy.spatial import ConvexHull

    origin = vertices.mean(axis=0)
    scale = np.linalg.norm(vertices - origin, axis=1).max()
    hull = ConvexHull((vertices - origin) / scale)
    a, b, c = (triangle - origin) / scale
    u, v = b - a, c - a
    matrix = np.zeros((len(hull.equations) + 3, 5))
    matrix[:-3, :3] = hull.equations[:, :3]
    matrix[-3:, 3:] = ((-1, 0), (0, -1), (1, 1))
    limit = np.r_[-hull.equations[:, 3], 0, 0, 1]

    def objective(value):
        difference = value[:3] - a - value[3] * u - value[4] * v
        return 0.5 * np.dot(difference, difference), np.r_[difference, -np.dot(difference, u), -np.dot(difference, v)]

    result = minimize(
        objective,
        (0, 0, 0, 1 / 3, 1 / 3),
        method="SLSQP",
        jac=True,
        constraints=LinearConstraint(matrix, -np.inf, limit),
        options={"ftol": 1.0e-12, "maxiter": 100},
    )
    require(result.success, f"Unable to verify initial apple clearance: {result.message}")
    return float(np.sqrt(max(0.0, 2.0 * result.fun)) * scale)


def convex_film_distance(vertices, surface):
    center = vertices.mean(axis=0)
    radius = np.linalg.norm(vertices - center, axis=1).max()
    lower_bounds = point_triangle_distances(center, surface) - radius
    best = float("inf")
    # The enclosing sphere only prunes distant triangles; every reported clearance uses the actual collision hull.
    for index in np.argsort(lower_bounds):
        if lower_bounds[index] >= best:
            break
        best = min(best, convex_triangle_distance(vertices, surface[index]))
    return best


def validate_apple_motion(sim, particles, bodies, times, triangles=None):
    """Check the observed drop and retained load independently of arm waypoint targets."""
    if sim.get("loading_mode") != "gravity_drop":
        return {}
    indices = np.asarray(sim.get("apple_body_indices", ()))
    require(
        indices.shape == (5,)
        and np.issubdtype(indices.dtype, np.integer)
        and len(set(indices)) == 5
        and np.all((indices >= 0) & (indices < bodies.shape[1])),
        "The gravity drop must contain five distinct apple bodies",
    )
    vertices = np.asarray(sim.get("apple_collision_vertices", ()))
    require(
        vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) >= 4 and np.isfinite(vertices).all(),
        "Invalid apple collision vertices",
    )
    radius = float(np.linalg.norm(vertices, axis=1).max())
    initial = np.asarray(sim.get("initial_apple_positions", ()))
    centers = bodies[:, indices, :3]
    require(
        initial.shape == (5, 3) and np.allclose(initial, centers[0], rtol=0, atol=1.0e-7),
        "Initial apple positions do not match the raw state",
    )
    mouth = np.asarray(sim.get("mouth_vertex_indices", ()))
    require(
        mouth.ndim == 1
        and len(mouth) >= 3
        and np.issubdtype(mouth.dtype, np.integer)
        and np.all((mouth >= 0) & (mouth < particles.shape[1])),
        "Missing bag mouth vertices",
    )
    lip = particles[0, mouth]
    initial_hulls = np.stack([posed_collider(vertices, pose) for pose in bodies[0, indices]])
    mouth_height = float(lip[:, 2].min())
    if "initial_mouth_height_m" in sim:
        require(
            np.isclose(sim["initial_mouth_height_m"], mouth_height, rtol=0, atol=1.0e-7),
            "Initial mouth height does not match its low cut lip",
        )
    # A U-cut food bag's boundary rises into the handles, above its actual low mouth plane.
    require(
        np.all(initial_hulls[:, :, 2].min(axis=1) >= mouth_height - 1.0e-5),
        "All five apples must start above the bag mouth",
    )
    require(
        np.all(initial_hulls[:, :, :2].min(axis=1) >= lip[:, :2].min(axis=0) - 1.0e-5)
        and np.all(initial_hulls[:, :, :2].max(axis=1) <= lip[:, :2].max(axis=0) + 1.0e-5),
        "All five apples must start over the bag opening",
    )
    for first in range(5):
        for second in range(first):
            a, b = initial_hulls[first], initial_hulls[second]
            require(
                np.any(a.min(axis=0) >= b.max(axis=0) - 1.0e-5) or np.any(b.min(axis=0) >= a.max(axis=0) - 1.0e-5),
                "Initial apples overlap",
            )
    report = {"initial_apple_count": len(indices)}
    if triangles is not None:
        surface = particles[0, np.asarray(triangles)]
        minimum_gap = min(convex_film_distance(hull, surface) for hull in initial_hulls)
        require(
            minimum_gap >= sim.get("apple_cloth_contact_gap", 0.0) - 1.0e-5,
            "Initial apples intersect the film or its contact margin",
        )
        report["initial_apple_film_gap_m"] = minimum_gap
    loaded = np.flatnonzero(times >= sim["loading_end_seconds"])
    if len(loaded):
        frame = int(loaded[0])
        descent = initial[:, 2] - centers[frame, :, 2]
        require(np.all(descent > 0.5 * np.ptp(vertices[:, 2])), "Every apple must fall into the bag before pickup")
        report["minimum_loading_descent_m"] = float(descent.min())
    retained = (times >= sim["loading_end_seconds"]) & (times < sim.get("release_start_seconds", float("inf")))
    if np.any(retained):
        tolerance = radius + sim.get("apple_cloth_contact_gap", 0.0)
        minimum = particles[retained].min(axis=1)[:, None, :] - tolerance
        maximum = particles[retained].max(axis=1)[:, None, :] + tolerance
        require(
            np.all((centers[retained] >= minimum) & (centers[retained] <= maximum)),
            "A loaded apple escaped the bag before release",
        )
        report["retained_apple_samples"] = int(retained.sum())
    return report


def validate_vertical_swing(sim, bodies, times, particles=None):
    """Measure completed jaw and cloth-handle travel, independently of commanded joint angles."""
    if sim.get("bounce_amplitude_m", 0.0) < 0.02 or "wrist_body_index" not in sim:
        return {}
    start = sim.get("vertical_swing_start_seconds", sim["swing_start_seconds"])
    end = sim.get("vertical_swing_end_seconds", start + sim["swing_period_seconds"] * sim["swing_cycles"])
    if times[-1] < end:
        return {}
    selected = (times >= start) & (times <= end)
    wrist = bodies[:, sim["wrist_body_index"]]
    fingers = bodies[:, sim["finger_body_indices"], :3].mean(axis=1)
    offset = np.array((0.0, 0.0, sim["grip_point_local"][2] - sim["finger_tip_local_z"]))
    # The seated handle lies behind the native finger origins; rotate that offset with the completed wrist.
    xyz, scalar = wrist[:, 3:6], wrist[:, 6:7]
    grip = fingers + offset + 2.0 * np.cross(xyz, np.cross(xyz, offset) + scalar * offset)
    heights = grip[selected, 2]
    excursion = float(np.ptp(heights))
    require(excursion >= 0.12, "The gripper must visibly swing up and down by at least 12 cm")
    frames = np.flatnonzero(selected)
    report = {
        "swing_grip_vertical_travel_m": excursion,
        "swing_grip_min_height_m": float(heights.min()),
        "swing_grip_max_height_m": float(heights.max()),
        "swing_grip_min_height_frame": int(frames[heights.argmin()]),
        "swing_grip_max_height_frame": int(frames[heights.argmax()]),
    }
    if "vertical_fixed_joint_indices" in sim:
        from scipy.spatial.transform import Rotation

        fixed = np.asarray(sim["vertical_fixed_joint_indices"])
        links = np.asarray(sim.get("arm_body_indices", ()))
        require(
            fixed.shape == (2,) and np.array_equal(fixed, (4, 5)),
            "The two joints immediately before the gripper must be joint5 and joint6",
        )
        require(
            links.shape == (6,)
            and np.issubdtype(links.dtype, np.integer)
            and np.all((links >= 0) & (links < bodies.shape[1])),
            "Missing arm links for fixed-wrist validation",
        )
        plan_times = np.asarray(sim.get("motion_plan_times_seconds", ()))
        targets = np.asarray(sim.get("motion_plan_joint_targets", ()))
        require(
            targets.shape == (len(plan_times), 8) and np.isfinite(targets).all(), "Missing motor plan for fixed wrist"
        )
        planned = targets[(plan_times >= start - 1e-6) & (plan_times <= end + 1e-6)]
        require(
            len(planned) >= 2 and np.max(np.ptp(planned[:, fixed], axis=0)) < 1e-7,
            "Fixed wrist motor targets must remain constant throughout vertical swinging",
        )
        ranges = []
        for joint in fixed:
            parent = Rotation.from_quat(bodies[selected, links[joint - 1], 3:])
            child = Rotation.from_quat(bodies[selected, links[joint], 3:])
            relative = parent.inv() * child
            # Parent-relative motion distinguishes a held joint from the whole arm's natural change in attitude.
            angles = (relative[0].inv() * relative).as_rotvec()
            variation = float(np.degrees(np.linalg.norm(np.ptp(angles, axis=0))))
            require(
                variation <= 2.0, "The fixed wrist joints must hold their solved relative angles within two degrees"
            )
            ranges.append(variation)
        report["vertical_fixed_joint_range_degrees"] = dict(zip(("joint5", "joint6"), ranges))
    if "vertical_wrist_pitch_amplitude_radians" in sim:
        from scipy.spatial.transform import Rotation

        amplitude = sim["vertical_wrist_pitch_amplitude_radians"]
        require(np.isfinite(amplitude) and 0.0 < amplitude < np.pi / 4, "Invalid vertical wrist pitch amplitude")
        extension = Rotation.from_quat(wrist[selected, 3:]).apply(np.array((0.0, 0.0, 1.0)))
        pitch = np.arctan2(extension[:, 2], np.linalg.norm(extension[:, :2], axis=1))
        pitch_range = float(np.ptp(pitch))
        require(
            1.7 * amplitude <= pitch_range <= 2.6 * amplitude,
            "The vertical jolts must include the commanded wrist pitching in the solved poses",
        )
        report.update(
            vertical_wrist_pitch_range_degrees=float(np.degrees(pitch_range)),
            vertical_wrist_min_pitch_degrees=float(np.degrees(pitch.min())),
            vertical_wrist_max_pitch_degrees=float(np.degrees(pitch.max())),
        )
    if "horizontal_swing_start_seconds" in sim:
        horizontal = (times >= sim["horizontal_swing_start_seconds"]) & (times <= sim["horizontal_swing_end_seconds"])
        h_points, v_points = grip[horizontal], grip[selected]
        horizontal_travel = float(np.ptp(h_points[:, 0]))
        horizontal_height = float(np.ptp(h_points[:, 2]))
        vertical_drift = float(np.linalg.norm(np.ptp(v_points[:, :2], axis=0)))
        require(horizontal_travel >= 0.25, "The horizontal stage must visibly swing side to side")
        require(horizontal_height <= 0.02, "The horizontal stage must keep a steady grip height")
        require(vertical_drift <= 0.02, "The vertical stage must keep a fixed horizontal grip position")
        report.update(
            horizontal_swing_travel_m=horizontal_travel,
            horizontal_swing_height_variation_m=horizontal_height,
            vertical_swing_horizontal_drift_m=vertical_drift,
        )
    if "grip_vertex_indices" in sim:
        require(particles is not None, "Cloth states are required to verify the gripped bag handle")
        indices = np.asarray(sim["grip_vertex_indices"])
        require(
            indices.ndim == 1
            and len(indices) > 0
            and np.issubdtype(indices.dtype, np.integer)
            and len(np.unique(indices)) == len(indices)
            and np.all((indices >= 0) & (indices < particles.shape[1])),
            "Invalid gripped handle vertices",
        )
        # Follow the same material points throughout the motion; a moving jaw alone cannot pass this check.
        handle = particles[:, indices].mean(axis=1)
        handle_heights = handle[selected, 2]
        travel = float(np.ptp(handle_heights))
        drift = float(np.linalg.norm(np.ptp(handle[selected, :2], axis=0)))
        separation = float(np.linalg.norm(handle[selected] - grip[selected], axis=1).max())
        require(
            travel >= max(0.12, 1.7 * sim["bounce_amplitude_m"]),
            "The gripped bag handle itself must visibly move up and down",
        )
        require(drift <= 0.025, "The gripped bag handle must move vertically with little sideways drift")
        require(separation <= 0.025, "The bag handle must remain between the moving jaws")
        report.update(
            handle_vertical_travel_m=travel,
            handle_vertical_horizontal_drift_m=drift,
            handle_vertical_max_jaw_separation_m=separation,
            handle_min_height_m=float(handle_heights.min()),
            handle_max_height_m=float(handle_heights.max()),
            handle_min_height_frame=int(frames[handle_heights.argmin()]),
            handle_max_height_frame=int(frames[handle_heights.argmax()]),
        )
    return report


def validate_coordinated_swing(sim, bodies, times):
    """Verify articulation and gentle pitch in the completed arm poses, not just commanded waypoints."""
    if sim.get("horizontal_motion_profile") != "coordinated_cartesian":
        return {}
    start, end = sim["horizontal_swing_start_seconds"], sim["horizontal_swing_end_seconds"]
    if times[-1] < end:
        return {}
    from scipy.spatial.transform import Rotation

    links = np.asarray(sim.get("arm_body_indices", ()))
    require(
        links.shape == (6,)
        and np.issubdtype(links.dtype, np.integer)
        and len(np.unique(links)) == 6
        and np.all((links >= 0) & (links < bodies.shape[1])),
        "Missing arm links for coordination validation",
    )
    selected = (times >= start) & (times <= end)
    rotations = [Rotation.from_quat(bodies[selected, body, 3:]) for body in links]
    changes = []
    for index in (1, 2):
        # World base rotation cancels in the parent-relative pose, so base-only swinging cannot pass.
        relative = rotations[index - 1].inv() * rotations[index]
        excursion = float(np.linalg.norm(np.ptp((relative[0].inv() * relative).as_rotvec(), axis=0)))
        require(excursion >= 0.10, "Shoulder and elbow must both articulate during horizontal swinging")
        changes.append(excursion)
    extension = rotations[-1].apply(np.array((0.0, 0.0, 1.0)))
    pitch = np.arctan2(extension[:, 2], np.linalg.norm(extension[:, :2], axis=1))
    pitch_range = float(np.degrees(np.ptp(pitch)))
    require(4.0 <= pitch_range <= 20.0, "The coordinated swing must include gentle wrist pitch")
    return {
        "horizontal_shoulder_rotation_range_degrees": float(np.degrees(changes[0])),
        "horizontal_elbow_rotation_range_degrees": float(np.degrees(changes[1])),
        "horizontal_wrist_pitch_range_degrees": pitch_range,
    }


def validate_apple_unloading(sim, velocities, times):
    """Require brief near-freefall intervals in the solved apple velocities during each downward jolt."""
    if sim.get("vertical_motion_profile") != "asymmetric_jolt" or times[-1] < sim["vertical_swing_end_seconds"]:
        return {}
    require(velocities is not None, "Solved apple velocities are required to verify unloading")
    indices = np.asarray(sim["apple_body_indices"], dtype=int)
    masses = np.asarray(sim.get("apple_masses_kg", ()), dtype=float)
    gravity = np.asarray(sim.get("gravity_m_s2", ()), dtype=float)
    require(
        masses.shape == indices.shape and np.isfinite(masses).all() and np.all(masses > 0),
        "Invalid apple masses for unloading validation",
    )
    require(
        gravity.shape == (3,) and np.isfinite(gravity).all() and gravity[2] < 0,
        "Invalid gravity for unloading validation",
    )
    gravity_magnitude = np.linalg.norm(gravity)
    dt = np.diff(times)
    acceleration = np.diff(velocities[:, indices, :3].astype(float), axis=0) / dt[:, None, None]
    residual = acceleration - gravity
    # Summing m*(a-g) cancels internal apple-to-apple forces and measures the net external supporting load.
    net = np.sum(residual * masses[None, :, None], axis=1) / masses.sum() / gravity_magnitude
    near_freefall = (np.linalg.norm(net, axis=1) <= 0.35) & (np.abs(net[:, 2]) <= 0.25)
    individual = np.linalg.norm(residual, axis=2) / gravity_magnitude <= 0.35
    intervals = 0.5 * (times[:-1] + times[1:])
    cycles = []
    for cycle in range(int(sim["vertical_swing_cycles"])):
        begin = sim["vertical_swing_start_seconds"] + cycle * sim["vertical_swing_period_seconds"]
        end = begin + sim["vertical_swing_period_seconds"]
        selected = np.flatnonzero((intervals >= begin) & (intervals < end) & near_freefall)
        groups = np.split(selected, np.flatnonzero(np.diff(selected) != 1) + 1)
        longest = max(groups, key=lambda group: dt[group].sum(), default=np.array([], dtype=int))
        duration = float(dt[longest].sum())
        require(duration >= 0.05 - 1e-8, "Every jolt must briefly unload the apples for at least 50 ms")
        require(
            np.count_nonzero(individual[longest].any(axis=0)) >= 3,
            "A jolt must unload individual apples as well as their combined load",
        )
        cycles.append(
            {
                "cycle": cycle + 1,
                "start_seconds": float(times[longest[0]]),
                "end_seconds": float(times[longest[-1] + 1]),
                "duration_seconds": duration,
                "maximum_net_load_fraction": float(np.linalg.norm(net[longest], axis=1).max()),
                "individually_unloaded_apples": int(np.count_nonzero(individual[longest].any(axis=0))),
            }
        )
    return {"apple_unloading_cycles": cycles}


def validate_airborne_release(sim, particles, bodies, times):
    """Require an unsupported release and visible post-impact travel by an apple that clears the bag."""
    if not sim.get("airborne_release") or sim.get("loading_mode") != "gravity_drop":
        return {}
    before = np.flatnonzero(times <= sim["release_start_seconds"])
    if not len(before) or times[-1] < sim["release_start_seconds"]:
        return {}
    frame = int(before[-1])
    # The authored platform top is 3 cm; the lower floor cannot support a bag clearing this plane.
    support_top = 0.03
    film_radius = sim.get("cloth_contact_radius", 0.0)
    gap = float(particles[frame, :, 2].min() - film_radius - support_top)
    require(gap > 0.05, "The bag must be unsupported in midair when the fingers begin opening")
    report = {"airborne_release_frame": frame, "airborne_bag_clearance_m": gap}
    if times[-1] < sim.get("duration_seconds", float("inf")) - 1e-8:
        return report
    require(particles[-1, :, 2].min() <= support_top + film_radius + 0.01, "The airborne bag did not land")
    indices = np.asarray(sim["apple_body_indices"], dtype=int)
    vertices = np.asarray(sim["apple_collision_vertices"])
    radius = float(np.linalg.norm(vertices, axis=1).max())
    centers = bodies[:, indices, :3]
    outside = np.any(
        (centers[-1] < particles[-1].min(axis=0) - radius) | (centers[-1] > particles[-1].max(axis=0) + radius), axis=1
    )
    require(np.any(outside), "No apple rolled clear of the released bag")
    travels, turns, rolling_apples = [], [], []
    for apple in np.flatnonzero(outside):
        candidates = np.flatnonzero(times >= sim["release_start_seconds"] + sim["release_duration_seconds"])
        near_stage = [
            sample
            for sample in candidates
            if posed_collider(vertices, bodies[sample, indices[apple]])[:, 2].min() <= support_top + 0.025
        ]
        if not near_stage:
            continue
        landed = int(near_stage[0])
        travel = float(np.linalg.norm(centers[-1, apple, :2] - centers[landed, apple, :2]))
        first, last = bodies[landed, indices[apple], 3:], bodies[-1, indices[apple], 3:]
        cosine = float(abs(np.dot(first, last)) / (np.linalg.norm(first) * np.linalg.norm(last)))
        turn = float(2.0 * np.arccos(np.clip(cosine, 0.0, 1.0)))
        travels.append(travel)
        turns.append(turn)
        rolling_apples.append(
            {
                "body_index": int(indices[apple]),
                "ground_contact_frame": landed,
                "ground_travel_m": travel,
                "rotation_radians": turn,
            }
        )
    require(
        any(distance >= 0.05 and angle >= 0.4 for distance, angle in zip(travels, turns)),
        "The spilled apples must visibly translate and rotate after reaching the stage",
    )
    report.update(
        spilled_apple_count=int(outside.sum()),
        maximum_spilled_apple_ground_travel_m=max(travels),
        maximum_spilled_apple_rotation_radians=max(turns),
        spilled_apples=rolling_apples,
    )
    return report


def validate(metadata_path):
    metadata_path = Path(metadata_path).resolve()
    root = metadata_path.parent
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_sources(metadata, root, REPOSITORY)
    for relative, expected in (metadata["file_sha256"] | metadata["texture_sha256"]).items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root), f"Nonportable export dependency: {relative}")
        check_hash(path, expected)
    require(not metadata["validation"]["example_assertion_errors"], "Physics checks failed; inspect export metadata")
    stage = Usd.Stage.Open(str(root / metadata["files"]["usd"]))
    sim = metadata["simulation"]
    frames = list(range(sim["samples"]))
    require(sim["samples"] == sim["frames"] + 1, "Incorrect sample count")
    require(stage.GetStartTimeCode() == 0 and stage.GetEndTimeCode() == frames[-1], "Wrong USD time range")
    require(stage.GetTimeCodesPerSecond() == stage.GetFramesPerSecond() == sim["fps"], "Wrong USD frame rate")
    require(UsdGeom.GetStageUpAxis(stage) == "Z" and UsdGeom.GetStageMetersPerUnit(stage) == 1, "Wrong USD units")
    require(str(stage.GetDefaultPrim().GetPath()) == "/World", "Wrong USD root")
    with np.load(root / metadata["files"]["raw_states"], allow_pickle=False) as raw:
        particles, bodies = raw["particle_q"], raw["body_q"]
        triangles, times = raw["triangles"], raw["time_seconds"]
        velocities = raw["body_qd"] if "body_qd" in raw else None
        if "body_qd" in raw:
            require(
                raw["body_qd"].shape == (sim["samples"], sim["body_count"], 6) and np.isfinite(raw["body_qd"]).all(),
                "Invalid recorded body velocities",
            )
    require(particles.shape == (len(frames), sim["bag_particles"], 3), "Wrong bag state shape")
    require(bodies.shape == (len(frames), sim["body_count"], 7), "Wrong body state shape")
    require(np.isfinite(particles).all() and np.isfinite(bodies).all(), "Nonfinite raw states")
    require(np.array_equal(times, np.arange(len(frames), dtype=np.float64) / sim["fps"]), "Wrong sample times")
    if "nonlinear_iterations" in sim:
        require(sim["nonlinear_iterations"] == 4, "The coupling sequence requires four nonlinear iterations")
    require([s["frame"] for s in metadata["timeline"]] == frames, "Missing timeline samples")
    require(np.array_equal([s["time_seconds"] for s in metadata["timeline"]], times), "Wrong timeline times")
    release_report = validate_release_timeline(sim, metadata["timeline"], metadata["validation"])
    telemetry_report = validate_motor_telemetry(root, metadata)
    expected_paths = {entry["path"] for entry in metadata["meshes"]}
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    require({str(prim.GetPath()) for prim in meshes} == expected_paths, "Mesh manifest mismatch")
    require(len({prim.GetName() for prim in meshes}) == len(meshes), "Duplicate mesh names break Blender validation")
    counts = {"bag": 0, "robot": 0, "apple": 0, "environment": 0}
    environment_names = set()
    max_point_error = max_pose_error = 0.0
    for entry in metadata["meshes"]:
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(entry["path"]))
        require(mesh.GetPrim().GetCustomDataByKey("kind") == entry["kind"], "Mesh kind mismatch")
        counts[entry["kind"]] += 1
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        faces = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        points = np.asarray(mesh.GetPointsAttr().Get(0))
        require(points.shape == (entry["vertices"], 3) and np.isfinite(points).all(), "Invalid mesh points")
        require(len(faces) == entry["triangles"] and np.all(faces == 3), "Invalid triangle counts")
        require(len(indices) == 3 * len(faces) and np.all((indices >= 0) & (indices < len(points))), "Invalid topology")
        require(not mesh.GetFaceVertexIndicesAttr().GetTimeSamples(), "Topology must remain constant")
        require(mesh.GetSubdivisionSchemeAttr().Get() == "none", "USD must retain original simulation geometry")
        material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()
        require(bool(material), f"Unbound material: {entry['path']}")
        if entry["texture"]:
            uv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
            require(uv and len(uv.ComputeFlattened()) == len(points), "Missing texture coordinates")
            require(uv.GetInterpolation() == "vertex", "Wrong UV interpolation")
            shader = UsdShade.Shader(stage.GetPrimAtPath(str(material.GetPath()) + "/ColorTexture"))
            asset = shader.GetInput("file").Get()
            require(asset.path == entry["texture"] and bool(asset.resolvedPath), "Missing portable texture")
        if entry["kind"] != "bag":
            require(not mesh.GetPointsAttr().GetTimeSamples(), "Rigid visuals must use body transforms")
            if entry["kind"] == "apple":
                require(
                    entry.get("primitive") == "mesh" and bool(entry["texture"]),
                    "Apples must export their original textured visual meshes",
                )
                normals = np.asarray(mesh.GetNormalsAttr().Get(), dtype=np.float32)
                require(
                    mesh.GetNormalsInterpolation() == "vertex" and normals.shape == points.shape,
                    "Apple visuals require their original per-vertex normals",
                )
                for name, array in (
                    ("points", points.astype(np.float32)),
                    ("indices", indices.astype(np.int32)),
                    ("normals", normals),
                ):
                    require(
                        hashlib.sha256(array.tobytes()).hexdigest() == entry[f"{name}_sha256"],
                        f"Apple visual {name} changed",
                    )
                translate = mesh.GetPrim().GetAttribute("xformOp:translate")
                orient = mesh.GetPrim().GetAttribute("xformOp:orient")
                require(
                    not translate.GetTimeSamples() and not orient.GetTimeSamples(),
                    "Apple shapes must stay fixed relative to their bodies",
                )
                q = orient.Get()
                pose = np.r_[translate.Get(), q.GetImaginary(), q.GetReal()]
                require(np.allclose(pose, entry["local_pose"], rtol=0, atol=1.0e-7), "Apple local pose changed")
            if entry["kind"] == "environment":
                environment_names.add(mesh.GetPrim().GetName())
                half_extents = np.asarray(entry["half_extents"])
                require(
                    entry["primitive"] == "box" and -1 <= entry["body_index"] < sim["body_count"],
                    "Invalid environment primitive",
                )
                require(half_extents.shape == (3,) and np.all(half_extents > 0), "Invalid environment dimensions")
                require(len(points) == 8 and len(faces) == 12, "Invalid environment box geometry")
                require(
                    np.allclose(np.abs(points), half_extents, rtol=0, atol=1.0e-7), "Environment dimensions changed"
                )
                require(len(np.unique(points, axis=0)) == 8, "Missing environment box corners")
                translate = mesh.GetPrim().GetAttribute("xformOp:translate")
                orient = mesh.GetPrim().GetAttribute("xformOp:orient")
                require(
                    not translate.GetTimeSamples() and not orient.GetTimeSamples(),
                    "Environment shape transform must remain fixed relative to its body",
                )
                q = orient.Get()
                pose = np.r_[translate.Get(), q.GetImaginary(), q.GetReal()]
                require(np.allclose(pose, entry["local_pose"], rtol=0, atol=1.0e-7), "Environment pose changed")
            continue
        if "film_pattern_sha256" in entry:
            pattern = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("film_pattern_m")
            require(
                pattern and pattern.GetInterpolation() == "faceVarying" and not pattern.GetTimeSamples(),
                "Missing static face-varying film pattern coordinates",
            )
            values = np.asarray(pattern.ComputeFlattened(), dtype=np.float32)
            require(
                values.shape == (len(indices), 2)
                and np.isfinite(values).all()
                and hashlib.sha256(values.tobytes()).hexdigest() == entry["film_pattern_sha256"],
                "Film pattern coordinates changed",
            )
        require(np.array_equal(indices, triangles.ravel()), "Bag/raw topology mismatch")
        require(mesh.GetPointsAttr().GetTimeSamples() == frames, "Incomplete bag animation")
        require(mesh.GetExtentAttr().GetTimeSamples() == frames, "Incomplete bag bounds")
        for frame in frames:
            points = np.asarray(mesh.GetPointsAttr().Get(frame))
            error = float(np.max(np.abs(points - particles[frame])))
            max_point_error = max(max_point_error, error)
            require(error < 2.0e-6, f"Bag/raw state mismatch at {frame}")
            require(
                np.array_equal(mesh.GetExtentAttr().Get(frame), [points.min(axis=0), points.max(axis=0)]),
                f"Wrong bag bounds at {frame}",
            )
    seen = set()
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Xform) or prim.GetCustomDataByKey("body_index") is None:
            continue
        index = prim.GetCustomDataByKey("body_index")
        require(index not in seen and 0 <= index < sim["body_count"], "Invalid body index")
        seen.add(index)
        translate, orient = prim.GetAttribute("xformOp:translate"), prim.GetAttribute("xformOp:orient")
        require(translate.GetTimeSamples() == orient.GetTimeSamples() == frames, "Incomplete body animation")
        for frame in frames:
            q = orient.Get(frame)
            pose = np.r_[translate.Get(frame), q.GetImaginary(), q.GetReal()]
            error = float(np.max(np.abs(pose - bodies[frame, index])))
            max_pose_error = max(max_pose_error, error)
            require(error < 2.0e-6, f"Body/raw state mismatch at {frame}: {index}")
    require(len(seen) == sim["body_count"], "Missing animated bodies")
    require(
        counts["bag"] == 1 and counts["apple"] == sim["apple_count"] and counts["robot"] > 0, "Missing scene objects"
    )
    require({"scene_floor", "bag_platform"}.issubset(environment_names), "Missing floor or bag platform")
    require(environment_names == {"scene_floor", "bag_platform"}, "Unexpected environment object; no rack is required")
    placement_report = validate_placement_support(stage, sim, metadata["meshes"], particles, times, bodies)
    apple_report = validate_apple_motion(sim, particles, bodies, times, triangles)
    swing_report = validate_vertical_swing(sim, bodies, times, particles)
    coordinated_report = validate_coordinated_swing(sim, bodies, times)
    unloading_report = validate_apple_unloading(sim, velocities, times)
    airborne_report = validate_airborne_release(sim, particles, bodies, times)
    return {
        "passed": True,
        "samples_checked": len(frames),
        "mesh_counts": counts,
        "maximum_point_error_m": max_point_error,
        "maximum_pose_error": max_pose_error,
        "maximum_grip_error_m": metadata["validation"]["maximum_grip_error_m"],
        **release_report,
        **placement_report,
        **apple_report,
        **swing_report,
        **coordinated_report,
        **unloading_report,
        **airborne_report,
        **telemetry_report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DIRECTORY / "outputs/simulation_metadata.json")
    args = parser.parse_args()
    try:
        report = validate(args.metadata)
    except Exception as error:
        report = {"passed": False, "error": f"{type(error).__name__}: {error}"}
    report_path = args.metadata.with_name(args.metadata.stem.replace("_metadata", "") + "_validation.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

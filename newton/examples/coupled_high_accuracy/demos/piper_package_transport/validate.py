"""Check airborne shaking, prompt release, landing, containment and tetrahedral validity."""

import argparse
import json
from pathlib import Path

import numpy as np
from geometry import sealed_bag, six_volumes
from motion import (
    DROP_RELEASE_END,
    DROP_RELEASE_START,
    LIFT_START,
    PARCEL_STARTS,
    RELEASE_OPENING,
    SHAKE_END,
    SHAKE_START,
)
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation


def inside_surface(points, vertices, faces):
    """Odd-even ray intersections give an independent check of a payload inside the closed film."""
    tri = vertices[faces].astype(float)
    a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    determinant = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    valid = np.abs(determinant) > 1e-12
    tri, a, b, determinant = tri[valid], a[valid], b[valid], determinant[valid]
    delta = points[:, None, :] - tri[None, :, 0]
    u = (delta[:, :, 1] * b[:, 2] - delta[:, :, 2] * b[:, 1]) / determinant
    v = (a[:, 1] * delta[:, :, 2] - a[:, 2] * delta[:, :, 1]) / determinant
    hit_x = tri[:, 0, 0] + u * a[:, 0] + v * b[:, 0]
    crossings = (u >= 0) & (v >= 0) & (u + v < 1) & (hit_x > points[:, None, 0])
    return crossings.sum(axis=1) % 2 == 1


def fitted_normals(rest, frames):
    x = rest - rest.mean(axis=0)
    result = []
    errors = []
    for frame in frames:
        y = frame - frame.mean(axis=0)
        u, _, vt = np.linalg.svd(x.T @ y)
        correction = np.diag((1, 1, np.linalg.det(u @ vt)))
        rotation = u @ correction @ vt
        result.append(rotation[2])
        errors.append(np.sqrt(np.mean(np.sum((y - x @ rotation) ** 2, axis=1))))
    return np.array(result), np.array(errors)


def drop_metrics(film, payload, wrist, grip_local, times, start, seal_patch=None):
    active = (times >= start + SHAKE_START) & (times <= start + SHAKE_END)
    held = (times >= start + LIFT_START) & (times <= start + DROP_RELEASE_START)
    release_frame = np.argmin(np.abs(times - start - DROP_RELEASE_START))
    after_release = (times >= start + DROP_RELEASE_END) & (times <= start + DROP_RELEASE_END + 1.0)
    wrist_rotation = Rotation.from_quat(wrist[:, 3:]).as_matrix()
    grip = np.einsum("tij,j->ti", wrist_rotation, grip_local) + wrist[:, :3]
    # Track the initial seal while the jaws are closed, before the parcel is free to fall.
    rest = film[0]
    seal_y = rest[:, 1].min() + 0.015
    patch = (
        (np.abs(rest[:, 0] - rest[:, 0].mean()) < 0.015) & (np.abs(rest[:, 1] - seal_y) < 0.007)
        if seal_patch is None
        else seal_patch
    )
    distance = np.linalg.norm(film[held][:, patch] - grip[held, None], axis=2).min(axis=1)
    payload_bottom = payload[:, :, 2].min(1)
    payload_center = payload.mean(1)
    shake_wrist = grip[active]
    pitch = np.rad2deg(np.arctan2(-wrist_rotation[active, 2, 2], wrist_rotation[active, 1, 2]))
    payload_after = payload_center[after_release]
    film_bottom = film[:, :, 2].min(1)
    prompt_frame = np.argmin(np.abs(times - start - DROP_RELEASE_END - 0.6))
    detached_frame = np.argmin(np.abs(times - start - DROP_RELEASE_END - 0.2))
    return {
        "lift_m": float(payload[active].mean(1)[:, 2].max() - payload[0].mean(0)[2]),
        "shake_x_span_m": float(np.ptp(shake_wrist[:, 0])),
        "shake_y_span_m": float(np.ptp(shake_wrist[:, 1])),
        "shake_z_span_m": float(np.ptp(shake_wrist[:, 2])),
        "forward_shake_peaks": int(len(find_peaks(shake_wrist[:, 1], prominence=0.012)[0])),
        "backward_shake_peaks": int(len(find_peaks(-shake_wrist[:, 1], prominence=0.012)[0])),
        "shake_pitch_min_degrees": float(pitch.min()),
        "shake_pitch_max_degrees": float(pitch.max()),
        "shake_pitch_span_degrees": float(np.ptp(pitch)),
        "upward_pitch_peaks": int(len(find_peaks(-pitch, prominence=5)[0])),
        "downward_pitch_peaks": int(len(find_peaks(pitch, prominence=5)[0])),
        "pitch_translation_correlation": float(np.corrcoef(pitch, shake_wrist[:, 1])[0, 1]),
        "maximum_closing_axis_vertical_component": float(np.abs(wrist_rotation[held, 2, 1]).max()),
        "minimum_closing_axis_x_component": float(np.abs(wrist_rotation[held, 0, 1]).min()),
        "maximum_closing_axis_error_degrees": float(
            np.rad2deg(np.arccos(np.clip(np.abs(wrist_rotation[held, 0, 1]).min(), 0, 1)))
        ),
        "maximum_held_seal_distance_m": float(distance.max()),
        "payload_bottom_at_release_m": float(payload_bottom[release_frame]),
        "film_bottom_at_release_m": float(film_bottom[release_frame]),
        "minimum_film_clearance_during_shake_m": float(film_bottom[active].min()),
        "payload_center_at_release_m": float(payload_center[release_frame, 2]),
        "payload_peak_height_m": float(payload_center[active, 2].max()),
        "free_drop_distance_m": float(payload_center[release_frame, 2] - payload_after[:, 2].min()),
        "drop_within_600ms_m": float(payload_center[release_frame, 2] - payload_center[prompt_frame, 2]),
        "seal_separation_after_release_m": float(
            np.linalg.norm(film[detached_frame, patch] - grip[detached_frame], axis=1).min()
        ),
        "final_top_height_m": float(payload[-1, :, 2].max()),
        "final_second_displacement_m": float(np.linalg.norm(payload[-30:].mean(1) - payload[-1].mean(0), axis=1).max()),
    }


def validate(folder):
    data = np.load(folder / "trajectory.npz")
    particles, bodies = data["particles"], data["bodies"]
    if len(particles) != 901:
        report = dict(
            duration_seconds=(len(particles) - 1) / 30,
            passed=False,
            checks=dict(full_duration=False),
            error="Drop validation requires the complete 30-second trajectory at 30 fps.",
        )
        (folder / "validation.json").write_text(json.dumps(report, indent=2))
        return report
    ranges = data["group_ranges"]
    film0, film1, jelly = [particles[:, a:b] for a, b in ranges]
    carton = bodies[:, int(data["carton"])]
    ratios = np.array([six_volumes(frame, data["tets"]) / data["rest_volumes"] for frame in jelly])
    carton_vertices = np.einsum("tij,pj->tpi", Rotation.from_quat(carton[:, 3:]).as_matrix(), data["carton_points"])
    carton_vertices += carton[:, None, :3]
    # Topology belongs to the baked run, even when the current geometry defaults have changed.
    faces = data["film_faces"] if "film_faces" in data else sealed_bag()[1]
    samples = sorted({0, len(jelly) - 1, *range(1, len(jelly), 30)})
    inside = [inside_surface(jelly[t], film1[t], faces).mean() for t in samples]
    carton_inside = [inside_surface(carton_vertices[t], film0[t], faces).mean() for t in samples]
    times = np.arange(len(particles)) / 30
    wrist = bodies[:, int(data["wrist"])]
    grip_points = data["grip_local"]
    if grip_points.ndim == 1:
        grip_points = np.tile(grip_points, (2, 1))
    seal_patch = data["seal_patch_indices"] if "seal_patch_indices" in data else None
    carton_turn = drop_metrics(film0, carton_vertices, wrist, grip_points[0], times, PARCEL_STARTS[0], seal_patch)
    jelly_turn = drop_metrics(film1, jelly, wrist, grip_points[1], times, PARCEL_STARTS[1], seal_patch)
    for start, metrics in zip(PARCEL_STARTS, (carton_turn, jelly_turn)):
        frame = np.argmin(np.abs(times - start - DROP_RELEASE_END - 0.1))
        fingers = bodies[frame, data["gripper_bodies"], :3]
        separation = fingers[0] - fingers[1]
        metrics["opened_finger_separation_world_m"] = separation.tolist()
        metrics["opened_finger_x_gap_m"] = float(abs(separation[0]))
        metrics["opened_finger_yz_offset_m"] = float(np.linalg.norm(separation[1:]))
    report = {
        "duration_seconds": (len(particles) - 1) / 30,
        "finite": bool(np.isfinite(particles).all() and np.isfinite(bodies).all()),
        "minimum_cloth_jelly_z_m": float(particles[:, : ranges[-1, 1], 2].min()),
        "minimum_carton_z_m": float(carton_vertices[:, :, 2].min()),
        "minimum_tet_volume_ratio": float(ratios.min()),
        "carton_drop": carton_turn,
        "jelly_drop": jelly_turn,
        "carton_bag_max_center_offset_m": float(np.linalg.norm(film0.mean(1) - carton[:, :3], axis=1).max()),
        "jelly_bag_max_center_offset_m": float(np.linalg.norm(film1.mean(1) - jelly.mean(1), axis=1).max()),
        "minimum_jelly_nodes_inside_fraction": float(min(inside)),
        "minimum_carton_nodes_inside_fraction": float(min(carton_inside)),
    }
    checks = {
        "full_duration": report["duration_seconds"] == 30,
        "finite": report["finite"],
        "floor": min(report["minimum_cloth_jelly_z_m"], report["minimum_carton_z_m"]) > -0.003,
        "positive_tetrahedra": report["minimum_tet_volume_ratio"] > 0.1,
        "carton_contained": report["carton_bag_max_center_offset_m"] < 0.035 and bool(min(carton_inside) > 0.98),
        "jelly_contained": report["jelly_bag_max_center_offset_m"] < 0.035 and bool(min(inside) > 0.98),
    }
    # Verify the baked motor schedule as well as the geometry, so an older closed-jaw cache cannot pass a new release check.
    targets = data["joint_targets"]
    target_fps = (len(targets) - 1) / report["duration_seconds"]
    checks["intentional_release_schedule"] = all(
        np.max(np.abs(targets[round((start + DROP_RELEASE_START) * target_fps), 6:])) < 1e-6
        and np.allclose(
            targets[round((start + DROP_RELEASE_END) * target_fps), 6:], (RELEASE_OPENING, -RELEASE_OPENING), atol=1e-6
        )
        for start in PARCEL_STARTS
    )
    for name, turn in (("carton", carton_turn), ("jelly", jelly_turn)):
        checks.update(
            {
                f"{name}_lifted": turn["lift_m"] > 0.045,
                f"{name}_fore_aft_shake": 0.075 < turn["shake_y_span_m"] < 0.085
                and max(turn["shake_x_span_m"], turn["shake_z_span_m"]) < 0.005,
                f"{name}_three_shake_cycles": turn["forward_shake_peaks"] == turn["backward_shake_peaks"] == 3,
                f"{name}_wrist_flick": 27 < turn["shake_pitch_span_degrees"] < 33
                and turn["pitch_translation_correlation"] < -0.95,
                f"{name}_three_pitch_cycles": turn["upward_pitch_peaks"] == turn["downward_pitch_peaks"] == 3,
                # Contact loads can deflect the physical wrist; a five-degree cone still requires side-to-side clamping.
                f"{name}_horizontal_clamping": turn["maximum_closing_axis_error_degrees"] < 5.0,
                f"{name}_horizontal_opening": turn["opened_finger_x_gap_m"] > 0.06
                and turn["opened_finger_yz_offset_m"] < 0.003,
                f"{name}_airborne_shake": turn["minimum_film_clearance_during_shake_m"] > 0.015,
                f"{name}_airborne_release": turn["payload_bottom_at_release_m"] > 0.025
                and turn["film_bottom_at_release_m"] > 0.015,
                f"{name}_free_drop": turn["free_drop_distance_m"] > 0.025,
                f"{name}_prompt_release": turn["drop_within_600ms_m"] > 0.04,
                f"{name}_detached_from_fingers": turn["seal_separation_after_release_m"] > 0.065,
                f"{name}_seal_retained": turn["maximum_held_seal_distance_m"] < 0.045,
                f"{name}_resting_on_floor": turn["final_top_height_m"] < 0.065
                and turn["final_second_displacement_m"] < 0.01,
            }
        )
    report["checks"], report["passed"] = checks, all(checks.values())
    (folder / "validation.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    report = validate(parser.parse_args().output)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)

"""Check dual-arm payload retention, release, settling and sampled bag containment."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "piper_package_transport"))
from geometry import six_volumes
from validate import inside_surface


def grasp_observations(film, fingers):
    """Measure actual jaw separation and film proximity independently of commanded joint positions."""
    gap = np.linalg.norm(fingers[:, 0, :3] - fingers[:, 1, :3], axis=1)
    # A finger is 78 mm long; its origin need not lie on the film even in a valid fingertip grasp.
    distances = np.linalg.norm(film[:, :, None, :] - fingers[:, None, :, :3], axis=-1)
    distance = distances.min(axis=1).max(axis=1)
    return dict(maximum_jaw_gap_m=float(gap.max()), maximum_film_finger_distance_m=float(distance.max()))


def validate(folder):
    data = np.load(folder / "trajectory.npz")
    p, b = data["particles"], data["bodies"]
    # Native ABD proxy coordinates follow the physical cloth/tet particles in state.particle_q.
    physical = p[:, : int(data["group_ranges"][-1, 1])]
    film0, film1, jelly = [p[:, a:z] for a, z in data["group_ranges"]]
    carton_pose = b[:, int(data["carton"])]
    carton = np.einsum("tij,pj->tpi", Rotation.from_quat(carton_pose[:, 3:]).as_matrix(), data["carton_points"])
    carton += carton_pose[:, None, :3]
    times = np.arange(len(p)) / 30
    held = (times >= 7) & (times <= 10)
    ratios = np.array([six_volumes(frame, data["tets"]) / data["rest_volumes"] for frame in jelly])
    checks = dict(
        full_duration=len(p) == 451,
        finite=bool(np.isfinite(p).all() and np.isfinite(b).all()),
        positive_tetrahedra=bool(ratios.min() > 0),
        table_clearance=bool(physical[:, :, 2].min() > 0.854),
    )
    metrics = dict(min_tet_volume_ratio=float(ratios.min()), minimum_particle_z=float(physical[:, :, 2].min()))
    for name, payload, film in [("carton", carton, film0), ("soft_package", jelly, film1)]:
        center = payload.mean(1)
        inside = [inside_surface(payload[t], film[t], data["film_faces"]).mean() for t in range(0, len(p), 30)]
        lift = float(center[held, 2].min() - center[0, 2])
        drop = float(center[300, 2] - center[360, 2])
        settling = float(np.linalg.norm(center[-30:] - center[-1], axis=1).max())
        checks[name + "_held_during_shake"] = lift > 0.10
        checks[name + "_released"] = drop > 0.10
        checks[name + "_settled"] = settling < 0.01
        checks[name + "_contained_sampled"] = bool(min(inside) >= 0.98)
        metrics[name] = dict(
            minimum_held_lift_m=lift,
            drop_within_two_seconds_m=drop,
            last_second_displacement_m=settling,
            minimum_inside_fraction=float(min(inside)),
        )
    for side, ids in enumerate(np.asarray(data["gripper_bodies"]).reshape(2, 2)):
        x = b[held][:, ids, 0].mean(1)
        peaks = len(find_peaks(x, prominence=0.015)[0])
        checks[f"hand_{side}_three_shakes"] = peaks == 3
        metrics[f"hand_{side}"] = dict(shake_peaks=peaks, x_span_m=float(np.ptp(x)))
        if "gripper_bag_indices" in data:
            film = (film0, film1)[int(data["gripper_bag_indices"][side])]
            observed = grasp_observations(film[held], b[held][:, ids])
            checks[f"hand_{side}_closed_during_shake"] = observed["maximum_jaw_gap_m"] < 0.030
            checks[f"hand_{side}_film_retained"] = observed["maximum_film_finger_distance_m"] < 0.080
            metrics[f"hand_{side}"].update(observed)
    report = dict(
        passed=all(checks.values()),
        checks=checks,
        metrics=metrics,
        scope="Sampled trajectory checks; not continuous collision certification.",
    )
    (folder / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    result = validate(Path(sys.argv[1]))
    raise SystemExit(0 if result["passed"] else 1)

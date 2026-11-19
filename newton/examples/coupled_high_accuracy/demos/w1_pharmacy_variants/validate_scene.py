"""Validate the four physical carton constructions and measure their different deformation grades."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
from gripper_quality import measure_gripper


def deformation(rest, current, outer, size):
    a = rest[:outer] - rest[:outer].mean(0)
    b = current[:outer] - current[:outer].mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    fix = np.eye(3)
    fix[2, 2] = np.linalg.det(u @ vt)
    aligned = b @ (u @ fix @ vt).T
    rms = float(np.sqrt(np.mean(np.sum((aligned - a) ** 2, axis=1))) * 1000)
    patch = (abs(a[:, 0]) < size[0] * 0.32) & (abs(a[:, 2] - (size[2] / 2 - 0.020)) < 0.022)
    front = patch & (a[:, 1] < -size[1] / 2 + 0.0002)
    back = patch & (a[:, 1] > size[1] / 2 - 0.0002)
    gap = float((aligned[back, 1].mean() - aligned[front, 1].mean()) * 1000)
    return dict(
        rms_mm=rms,
        normalized_rms=rms / (size[1] * 1000),
        grip_width_mm=gap,
        compression_fraction=float(1 - gap / (size[1] * 1000)),
    )


def validate_pair(folder):
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    d = np.load(folder / "trajectory.npz")
    q = d["particles"]
    strain = d["plastic_strain"]
    pre = round(1.7 * 30)
    held = min(round(6.1 * 30), len(q) - 1)
    checks = dict(
        full_sequence=run["seconds"] >= 11.5, finite=bool(np.isfinite(q).all() and np.isfinite(d["bodies"]).all())
    )
    metrics = []
    for i, case in enumerate(run["cases"]):
        a, b = d["ranges"][i]
        ta, tb = d["tet_ranges"][i]
        outer = int(d["outer_counts"][i])
        rest = d["rest"][a:b]
        before = deformation(rest, q[min(pre, len(q) - 1), a:b], outer, case["size"])
        loaded = deformation(rest, q[held, a:b], outer, case["size"])
        released = deformation(rest, q[-1, a:b], outer, case["size"])
        center = q[:, a:b].mean(1)
        lift = float(center[:, 2].max() - center[0, 2])
        increment = strain[-1, ta:tb] - strain[min(pre, len(q) - 1), ta:tb]
        metrics.append(
            dict(
                key=case["key"],
                name=case["name"],
                construction=case["construction"],
                lift_m=lift,
                before=before,
                loaded=loaded,
                released=released,
                plastic_increment_max=float(increment.max()),
                yielded_fraction=float((increment > 0.01).mean()),
            )
        )
        checks[case["key"] + "_picked"] = lift > 0.15
        checks[case["key"] + "_initially_undeformed"] = before["rms_mm"] < 0.5
        if case["key"] == "tablets":
            checks["compact_carton_recovers"] = released["rms_mm"] < max(0.3, loaded["rms_mm"] * 0.7)
        if case["key"] in ("granules", "patches"):
            checks[case["key"] + "_has_residual_crease"] = released["rms_mm"] > 0.5 and increment.max() > 0.03
    tets = d["tets"]
    p = d["rest"][tets]
    v0 = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    minimum = 1.0
    inversions = 0
    for frame in range(0, len(q), 3):
        p = q[frame, tets]
        v = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
        ratio = v / v0
        minimum = min(minimum, float(ratio.min()))
        inversions += int((ratio <= 0).sum())
    checks["sampled_no_inversion"] = inversions == 0
    checks["different_loaded_deformation"] = (
        metrics[1]["loaded"]["normalized_rms"] > metrics[0]["loaded"]["normalized_rms"] * 1.10
    )
    gripper = measure_gripper(d, 4.3, 6.3)
    checks["jaw_center_stable"] = gripper["passed"]
    checks = {key: bool(value) for key, value in checks.items()}
    result = dict(
        passed=all(checks.values()),
        checks=checks,
        cartons=metrics,
        gripper=gripper,
        minimum_tet_volume_ratio=minimum,
        sampled_inverted_tets=inversions,
        finger_effort_peak_n=abs(d["forces"]).max(0).tolist(),
        scope="Estimated effective paperboard; 10 Hz tetrahedron checks, no calibrated medicine contents.",
    )
    (folder / "validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def validate(folder):
    if (folder / "trajectory.npz").exists():
        result = validate_pair(folder)
    else:
        pairs = [validate_pair(folder / f"pair_{i}") for i in (1, 2)]
        cartons = [c for pair in pairs for c in pair["cartons"]]
        grades = [c["loaded"]["normalized_rms"] for c in cartons]
        graded = all(b > a * 1.10 for a, b in zip(grades, grades[1:]))
        result = dict(
            passed=all(p["passed"] for p in pairs) and graded,
            graded_deformation=graded,
            normalized_loaded_rms=grades,
            cartons=cartons,
            pairs=[p["passed"] for p in pairs],
        )
        (folder / "validation.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(0 if validate(Path(sys.argv[1]))["passed"] else 1)

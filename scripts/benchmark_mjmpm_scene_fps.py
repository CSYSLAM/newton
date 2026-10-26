# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Measure full MJMPM trajectories with synchronized GL rendering in fresh processes.

    uv run --extra examples scripts/benchmark_mjmpm_scene_fps.py --output-dir artifacts/mjmpm_fps_ab

The compact baseline disables only the parallel mobility rows. Acceptance
checks run outside timed frames. Failed runs are reported, never silently retried.
"""

import argparse
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.mjmpm.mpm import solve_rheology as rheology
from newton.examples.mjmpm.example_mjmpm_teapot_fill import Example as WaterExample
from newton.examples.mjmpm.example_mjmpm_w1_bread_three_pieces import Example as BreadExample
from newton.examples.mjmpm.example_mjmpm_w1_rice_cooker import Example as RiceExample
from newton.viewer import ViewerGL

SCENES = {"water": (WaterExample, 900), "rice": (RiceExample, 990), "bread": (BreadExample, 1200)}
WARMUP = 60


def gpu_info():
    """Record clocks and temperature outside the measured trajectory."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return result.stdout.strip()


def validate_frame(scene, example, previous_bowl_peak):
    """Collect test-only diagnostics after the production frame has completed."""
    if scene == "rice" and example.frame % 60:
        example.args.test = True
        try:
            example.after_step()
        finally:
            example.args.test = False
    if example.tracks_freely(example.sim_time):
        tcp = example.tcp_positions(example.state_0.body_q.numpy())
        example.peak_free_tcp_error = max(
            example.peak_free_tcp_error, float(np.max(np.linalg.norm(tcp - example.targets, axis=1)))
        )
    if scene == "water":
        if example.frame % 30 == 0 and previous_bowl_peak > 100:
            if example.bowl_particle_count < 0.5 * previous_bowl_peak:
                raise AssertionError(
                    f"Water disappeared: {example.bowl_particle_count} after peak {previous_bowl_peak}"
                )
        if example.frame % 10 == 0:
            from newton.examples.mjmpm.support.w1_two_way import TABLE_Z  # noqa: PLC0415

            pose = example.state_0.body_q.numpy()[example.teapot]
            up = np.asarray(wp.quat_rotate(wp.quat(*pose[3:]), wp.vec3(0.0, 0.0, 1.0)))
            example.peak_lift = max(example.peak_lift, float(pose[2] - TABLE_Z))
            example.peak_tilt = max(example.peak_tilt, float(np.arccos(np.clip(up[2], -1.0, 1.0))))
    example.test_post_step()


def worker(options):
    example_class, frames = SCENES[options.scene]
    wp.set_device("cuda:0")
    rheology._MOBILITY_ROW_PARALLEL = options.layout == "rows"
    args = example_class.create_parser().parse_args(["--viewer", "null"])
    args.test = False
    args.num_frames = frames
    viewer = ViewerGL(width=1280, height=720, headless=True, num_frames=frames)
    example = example_class(viewer, args)
    before = gpu_info()
    samples = []
    dispatches = {"compact": 0, "rows": 0}
    original_launch, original_tiled = wp.launch, wp.launch_tiled

    def launch(*a, **kw):
        kernel = kw.get("kernel", a[0] if a else None)
        if getattr(kernel, "key", "") == "apply_mobility_feedback":
            dispatches["compact"] += 1
        return original_launch(*a, **kw)

    def tiled(*a, **kw):
        kernel = kw.get("kernel", a[0] if a else None)
        if getattr(kernel, "key", "") == "apply_mobility_feedback_rows":
            dispatches["rows"] += 1
        return original_tiled(*a, **kw)

    wp.launch, wp.launch_tiled = launch, tiled
    error = None
    validation = None
    try:
        for frame in range(frames):
            previous_peak = getattr(example, "peak_bowl_particle_count", 0)
            start = time.perf_counter_ns()
            example.step()
            example.render()
            viewer.renderer.gl.glFinish()
            wp.synchronize_device(example.model.device)
            samples.append((time.perf_counter_ns() - start) * 1.0e-6)
            validate_frame(options.scene, example, previous_peak)
            if (frame + 1) % 300 == 0:
                print(f"[{options.scene}/{options.layout}] {frame + 1}/{frames}", flush=True)
        example.args.test = True
        example.test_final()
        if options.scene == "water":
            validation = {
                "water_in_bowl": example.water_in_bowl_count(),
                "water_total": example.water_end - example.water_begin,
            }
        elif options.scene == "rice":
            validation = dict(zip(("in_cup", "in_pot", "spilled"), example.rice_locations(), strict=True))
        else:
            validation = example.validation
    except Exception:
        error = traceback.format_exc()
        print(error, file=sys.stderr, flush=True)
    finally:
        wp.launch, wp.launch_tiled = original_launch, original_tiled
        after = gpu_info()
        viewer.close()
    measured = np.asarray(samples[WARMUP:])
    report = {
        "scene": options.scene,
        "layout": options.layout,
        "round": options.round,
        "resolution": [1280, 720],
        "warmup_frames": WARMUP,
        "planned_frames": frames,
        "completed_frames": len(samples),
        "passed": error is None and len(samples) == frames,
        "measurement": "production step + GL render + glFinish + device fence; acceptance checks excluded",
        "warp_version": wp.__version__,
        "device": example.model.device.name,
        "gpu_before": before,
        "gpu_after": after,
        "recorded_dispatches": dispatches,
        "warm_fps": float(1000.0 / measured.mean()) if len(measured) else None,
        "warm_mean_ms": float(measured.mean()) if len(measured) else None,
        "warm_median_ms": float(np.median(measured)) if len(measured) else None,
        "warm_p90_ms": float(np.percentile(measured, 90)) if len(measured) else None,
        "validation": validation,
        "error": error,
        "frame_ms": samples,
    }
    options.output_dir.mkdir(parents=True, exist_ok=True)
    path = options.output_dir / f"{options.scene}_r{options.round}_{options.layout}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("frame_ms", "error")}), flush=True)


def summarize(output_dir, rounds):
    summary = {}
    for scene in SCENES:
        pairs = []
        failed = []
        for iteration in range(1, rounds + 1):
            reports = {}
            for layout in ("compact", "rows"):
                path = output_dir / f"{scene}_r{iteration}_{layout}.json"
                if path.exists():
                    reports[layout] = json.loads(path.read_text(encoding="utf-8"))
                    if not reports[layout]["passed"]:
                        failed.append(path.name)
            if len(reports) == 2 and all(r["passed"] for r in reports.values()):
                a, b = reports["compact"]["warm_fps"], reports["rows"]["warm_fps"]
                pairs.append({"round": iteration, "compact_fps": a, "rows_fps": b, "change_percent": 100 * (b / a - 1)})
        row = {"pairs": pairs, "failed_runs": failed}
        if pairs:
            ratios = np.log([p["rows_fps"] / p["compact_fps"] for p in pairs])
            row.update(
                {
                    "compact_mean_fps": float(np.mean([p["compact_fps"] for p in pairs])),
                    "rows_mean_fps": float(np.mean([p["rows_fps"] for p in pairs])),
                    "paired_geometric_change_percent": 100 * math.expm1(float(ratios.mean())),
                }
            )
            if len(pairs) > 1:
                critical = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}.get(len(pairs) - 1, 2.571)
                half = critical * float(np.std(ratios, ddof=1)) / math.sqrt(len(pairs))
                row["paired_95_percent_ci"] = [
                    100 * math.expm1(float(ratios.mean()) - half),
                    100 * math.expm1(float(ratios.mean()) + half),
                ]
        summary[scene] = row
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3, choices=(2, 3, 4, 5))
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--scene", choices=SCENES)
    parser.add_argument("--layout", choices=("compact", "rows"))
    parser.add_argument("--round", type=int, default=1)
    options = parser.parse_args()
    if options.worker:
        worker(options)
        return
    options.output_dir.mkdir(parents=True, exist_ok=True)
    for iteration in range(1, options.rounds + 1):
        order = ("compact", "rows") if iteration % 2 else ("rows", "compact")
        for scene in SCENES:
            for layout in order:
                path = options.output_dir / f"{scene}_r{iteration}_{layout}.json"
                if path.exists():
                    print(f"Keep recorded run {path.name}", flush=True)
                    continue
                with path.with_suffix(".log").open("w", encoding="utf-8") as log:
                    print(f"Start {scene} round {iteration} {layout}", flush=True)
                    result = subprocess.run(
                        [
                            "uv",
                            "run",
                            "--no-sync",
                            "python",
                            "-u",
                            str(Path(__file__).resolve()),
                            "--worker",
                            "--scene",
                            scene,
                            "--layout",
                            layout,
                            "--round",
                            str(iteration),
                            "--output-dir",
                            str(options.output_dir.resolve()),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                if result.returncode or not path.exists():
                    raise RuntimeError(f"Benchmark process failed; inspect {path.with_suffix('.log')}")
                report = json.loads(path.read_text(encoding="utf-8"))
                fps = report["warm_fps"]
                fps_text = f"{fps:.2f} FPS" if fps is not None else "FPS unavailable"
                print(f"Done {path.name}: {fps_text}, passed={report['passed']}", flush=True)
                summarize(options.output_dir, options.rounds)
    print(json.dumps(summarize(options.output_dir, options.rounds), indent=2), flush=True)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Validate a first-solve warm start for the rice-cooker reaction recovery.

The experimental wrapper is process-local and leaves production code unchanged.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton.examples.mjmpm.example_mjmpm_w1_rice_cooker import Example
from newton.viewer import ViewerNull


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-sweeps", type=int, default=60)
    parser.add_argument("--warm-start", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    opts = parser.parse_args()
    if opts.recovery_sweeps < 1:
        parser.error("--recovery-sweeps must be positive")
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    wp.set_device("cuda:0")
    args = Example.create_parser().parse_args(["--viewer", "null"])
    args.test = True
    viewer = ViewerNull(num_frames=990)
    example = Example(viewer, args)
    env = example.solver.environment
    original = env._solve_contacts_gauss_seidel
    first_impulse = wp.zeros_like(env.contact_impulses)
    calls = 0

    def sweep(work):
        nonlocal calls
        recovery = calls % 2 == 1
        calls += 1
        if recovery and opts.warm_start:
            wp.copy(env.contact_impulses, first_impulse)
        iterations = env.rigid_iterations
        if recovery:
            env.rigid_iterations = opts.recovery_sweeps
        try:
            original(work)
        finally:
            env.rigid_iterations = iterations
        if not recovery and opts.warm_start:
            wp.copy(first_impulse, env.contact_impulses)

    env._solve_contacts_gauss_seidel = sweep
    times = []
    residuals = []
    validation = "PASS"
    try:
        for frame in range(990):
            wp.synchronize_device(example.model.device)
            start = time.perf_counter()
            example.step()
            wp.synchronize_device(example.model.device)
            elapsed = (time.perf_counter() - start) * 1000
            if frame >= 60:
                times.append(elapsed)
            example.test_post_step()
            if frame % 60 == 0:
                residuals.append(
                    {
                        "frame": frame,
                        "material_velocity": env.material_velocity_residual.numpy().tolist(),
                        "momentum": env.momentum_residual.numpy().tolist(),
                        "constraint_velocity": env.constraint_velocity_residual.numpy().tolist(),
                    }
                )
            if frame % 120 == 0:
                print(f"Progress {frame + 1}/990", flush=True)
        example.solver.check_capacity_status()
        try:
            example.test_final()
        except AssertionError as error:
            validation = str(error)
        poses = np.asarray(example.cup_in_grip)
        slip = float(np.max(np.linalg.norm(poses[:, :3] - poses[0, :3], axis=1)))
        turn = float(np.max(2 * np.arccos(np.clip(np.abs(poses[:, 3:] @ poses[0, 3:]), 0, 1))))
        report = {
            "warm_start": opts.warm_start,
            "recovery_sweeps": opts.recovery_sweeps,
            "initial_sweeps": args.rigid_iterations,
            "validation": validation,
            "median_ms": float(np.median(times)),
            "mean_ms": float(np.mean(times)),
            "p90_ms": float(np.percentile(times, 90)),
            "rice_locations": example.rice_locations(),
            "slip_mm": slip * 1000,
            "turn_degrees": float(np.rad2deg(turn)),
            "peak_grain_mps": example.peak_grain,
            "finger_load_N": {k: float(np.mean(v)) for k, v in example.finger_load.items()},
            "residuals": residuals,
            "notes": "One complete task run with per-frame validation. Timed step includes normal test telemetry; correctness checks and residual downloads outside step are excluded. Experimental, not enabled in production.",
        }
        name = f"{'warm' if opts.warm_start else 'cold'}_{opts.recovery_sweeps}"
        (opts.output_dir / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "residuals"}), flush=True)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()

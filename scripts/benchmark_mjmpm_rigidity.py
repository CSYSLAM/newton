# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare unchanged triplet assembly with direct rigid assembly on the full task."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.mjmpm.mpm import rasterized_collisions as rc
from newton._src.solvers.mjmpm.mpm import solver_implicit_mpm as sm
from newton.examples.mjmpm.example_mjmpm_w1_rice_cooker import Example
from newton.viewer import ViewerNull


@wp.kernel
def compare_sparse(
    nrow: int,
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.uint32],
    expected_offsets: wp.array[int],
    expected_columns: wp.array[int],
    expected_values: wp.array[wp.uint32],
    stats: wp.array[int],
):
    i = wp.tid()
    if i == 0:
        wp.atomic_add(stats, 1, 1)
    if i <= nrow and offsets[i] != expected_offsets[i]:
        wp.atomic_add(stats, 0, 1)
    if i < expected_offsets[nrow]:
        if columns[i] != expected_columns[i]:
            wp.atomic_add(stats, 0, 1)
        for k in range(9):
            if values[9 * i + k] != expected_values[9 * i + k]:
                wp.atomic_add(stats, 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", choices=("generic", "direct"), default="direct")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    wp.set_device("cuda:0")
    args = Example.create_parser().parse_args(["--viewer", "null"])
    args.test = True
    viewer = ViewerNull(num_frames=990)
    example = Example(viewer, args)
    original = sm.build_rigidity_operator
    limit = rc._RIGID_ASSEMBLY_MAX_WORK
    stats = wp.zeros(2, dtype=int)
    if opts.assembly == "generic":
        rc._RIGID_ASSEMBLY_MAX_WORK = 0

    def verified_build(*a, **kw):
        result = original(*a, **kw)
        rc._RIGID_ASSEMBLY_MAX_WORK = 0
        try:
            expected = original(*a, **kw)
        finally:
            rc._RIGID_ASSEMBLY_MAX_WORK = limit
        for matrix, reference in zip(result, expected, strict=True):
            wp.launch(
                compare_sparse,
                dim=max(matrix.nrow + 1, reference.nnz),
                inputs=[
                    matrix.nrow,
                    matrix.offsets,
                    matrix.columns,
                    matrix.values.view(dtype=wp.uint32).flatten(),
                    reference.offsets,
                    reference.columns,
                    reference.values.view(dtype=wp.uint32).flatten(),
                    stats,
                ],
            )
        return result

    if opts.verify:
        if opts.assembly != "direct":
            parser.error("--verify requires --assembly direct")
        sm.build_rigidity_operator = verified_build
    times = []
    residuals = []
    env = example.solver.environment
    try:
        for frame in range(990):
            wp.synchronize_device(example.model.device)
            start = time.perf_counter()
            example.step()
            wp.synchronize_device(example.model.device)
            if frame >= 60:
                times.append((time.perf_counter() - start) * 1000)
            example.test_post_step()
            if frame % 60 == 0:
                residuals.append(
                    {
                        "frame": frame,
                        "momentum": env.momentum_residual.numpy().tolist(),
                        "material_velocity": env.material_velocity_residual.numpy().tolist(),
                        "constraint_velocity": env.constraint_velocity_residual.numpy().tolist(),
                    }
                )
            if frame % 120 == 0:
                print(f"Progress {frame + 1}/990", flush=True)
        example.solver.check_capacity_status()
        example.test_final()
        mismatch, compared = map(int, stats.numpy())
        if opts.verify and (mismatch or compared == 0):
            raise AssertionError(f"Operator comparison failed: {mismatch} mismatches in {compared} matrices")
        poses = np.asarray(example.cup_in_grip)
        report = {
            "assembly": opts.assembly,
            "verify": opts.verify,
            "validation": "PASS",
            "median_ms": float(np.median(times)),
            "mean_ms": float(np.mean(times)),
            "p90_ms": float(np.percentile(times, 90)),
            "rice_locations": example.rice_locations(),
            "slip_mm": float(np.max(np.linalg.norm(poses[:, :3] - poses[0, :3], axis=1))) * 1000,
            "turn_degrees": float(
                np.rad2deg(np.max(2 * np.arccos(np.clip(np.abs(poses[:, 3:] @ poses[0, 3:]), 0, 1))))
            ),
            "finger_load_N": {k: float(np.mean(v)) for k, v in example.finger_load.items()},
            "operator_mismatches": mismatch,
            "operators_compared": compared,
            "peak_grain_mps": example.peak_grain,
            "residuals": residuals,
            "notes": "Fixed 6 substeps, 60/60 rigid sweeps, MPM cap 50; test telemetry inside step, extra checks outside. Verification timings are not performance measurements.",
        }
        opts.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "residuals"}), flush=True)
    finally:
        rc._RIGID_ASSEMBLY_MAX_WORK = limit
        sm.build_rigidity_operator = original
        viewer.close()


if __name__ == "__main__":
    main()

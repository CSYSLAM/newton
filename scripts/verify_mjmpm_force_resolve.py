# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shadow force-only MuJoCo resolves with full solves on identical data.

Use --control to measure full-versus-full floating-point nondeterminism.
Verification overhead makes this unsuitable for performance measurements.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import warp as wp

from newton.examples.mjmpm.example_mjmpm_w1_rice_cooker import Example
from newton.viewer import ViewerNull
from scripts.benchmark_mjmpm_sweeps import compare_bits


@wp.kernel
def compare_error(a: wp.array[float], b: wp.array[float], errors: wp.array[float]):
    i = wp.tid()
    error = wp.abs(a[i] - b[i])
    wp.atomic_max(errors, 0, error)
    wp.atomic_max(errors, 1, error / wp.max(1.0, wp.max(wp.abs(a[i]), wp.abs(b[i]))))


def _arrays(obj):
    if isinstance(obj, wp.array):
        if obj.size:
            yield obj
    elif dataclasses.is_dataclass(obj):
        for field in dataclasses.fields(obj):
            yield from _arrays(getattr(obj, field.name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=990)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic MuJoCo force kernels")
    opts = parser.parse_args()
    wp.set_device("cuda:0")
    args = Example.create_parser().parse_args(["--viewer", "null"])
    args.test = True
    viewer = ViewerNull(num_frames=opts.frames)
    example = Example(viewer, args)
    solver = example.solver.environment._link_mobility._solver
    if opts.deterministic:
        from mujoco_warp._src import forward, passive, smooth, support
        from mujoco_warp._src import solver as backend_solver

        from newton._src.solvers.mjmpm.mujoco.solver_mujoco import (  # noqa: PLC0415
            _mujoco_warp_deterministic_max_records,
        )

        # Remove atomic force accumulation noise without recompiling convex
        # collision detection with deterministic records. The production solver
        # keeps its original determinism and collision settings.
        records = _mujoco_warp_deterministic_max_records(solver.mj_model, solver.mjw_data)
        for module in (forward, passive, smooth, backend_solver, support):
            solver._set_module_options(
                {
                    "deterministic": wp.DeterministicMode.RUN_TO_RUN,
                    "deterministic_max_records": records if module is smooth else 0,
                },
                module=module,
            )
    data = solver.mjw_data
    names = (
        "qpos",
        "qvel",
        "qacc",
        "qacc_warmstart",
        "qacc_smooth",
        "qfrc_smooth",
        "qfrc_bias",
        "qfrc_passive",
        "qfrc_actuator",
        "M",
        "qLD",
        "qLDiagInv",
        "solver_niter",
        "time",
        "efc.force",
        "efc.state",
        "nefc",
    )
    outputs = [getattr(data, name) for name in names[:14]] + [data.efc.force, data.efc.state, data.nefc]
    sources = list(_arrays(data))
    before = [wp.empty_like(array) for array in sources]
    expected = [wp.empty_like(array) for array in outputs]
    stats = [wp.zeros(2, dtype=int) for _ in outputs]
    errors = [wp.zeros(2, dtype=float) for _ in outputs]
    original = solver._mujoco_warp_step

    def step():
        if not solver._coupling_reuse_kinematics:
            return original()
        for a, b in zip(before, sources, strict=True):
            wp.copy(a, b)
        if opts.control:
            solver._coupling_reuse_kinematics = False
        original()
        for a, b in zip(expected, outputs, strict=True):
            wp.copy(a, b)
        for a, b in zip(sources, before, strict=True):
            wp.copy(a, b)
        solver._coupling_reuse_kinematics = False
        original()
        for a, b, counter, error in zip(outputs, expected, stats, errors, strict=True):
            count = a.size * wp.types.type_size_in_bytes(a.dtype) // 4
            wp.launch(
                compare_bits, dim=count, inputs=[a.view(wp.uint32).flatten(), b.view(wp.uint32).flatten(), counter]
            )
            if a.dtype == wp.float32:
                wp.launch(compare_error, dim=count, inputs=[a.flatten(), b.flatten(), error])

    solver._mujoco_warp_step = step
    try:
        for frame in range(opts.frames):
            example.step()
            example.test_post_step()
            if frame % 120 == 0:
                print(f"Progress {frame + 1}/{opts.frames}", flush=True)
        if opts.frames >= 990:
            example.test_final()
        report = {
            "control": opts.control,
            "deterministic_force_kernels": opts.deterministic,
            "frames": opts.frames,
            "arrays": {
                name: dict(zip(("different_components", "comparisons"), counter.numpy().tolist(), strict=True))
                | dict(zip(("max_absolute_error", "max_scaled_error"), error.numpy().tolist(), strict=True))
                for name, counter, error in zip(names, stats, errors, strict=True)
            },
        }
        opts.output.parent.mkdir(parents=True, exist_ok=True)
        opts.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report), flush=True)
        assert all(row["comparisons"] > 0 for row in report["arrays"].values())
        if opts.deterministic:
            assert all(row["different_components"] == 0 for row in report["arrays"].values()), report
    finally:
        viewer.close()


if __name__ == "__main__":
    main()

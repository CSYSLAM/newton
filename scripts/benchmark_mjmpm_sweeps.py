# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare rigid GS kernels on identical frozen inputs or on the complete rice task."""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.mjmpm.rigid import kernels as rk
from newton.examples.mjmpm.example_mjmpm_w1_rice_cooker import Example
from newton.viewer import ViewerNull
from scripts.profile_mjmpm_physics import copy_when


@wp.kernel
def compare_bits(a: wp.array[wp.uint32], b: wp.array[wp.uint32], stats: wp.array[int]):
    i = wp.tid()
    if i == 0:
        wp.atomic_add(stats, 1, 1)
    if a[i] != b[i]:
        wp.atomic_add(stats, 0, 1)


@wp.kernel
def compare_roundoff(a: wp.array[float], b: wp.array[float], stats: wp.array[int], maxima: wp.array[float], slot: int):
    i = wp.tid()
    if i == 0:
        wp.atomic_add(stats, 1, 1)
    difference = wp.abs(a[i] - b[i])
    wp.atomic_max(maxima, slot, difference)
    if not wp.isfinite(a[i]) or not wp.isfinite(b[i]) or difference > 1.0e-5 + 1.0e-5 * wp.abs(b[i]):
        wp.atomic_add(stats, 0, 1)


def load_baseline(output, commit):
    """Load the committed reference without changing the checkout."""
    source = subprocess.run(
        ["git", "show", f"{commit}:newton/_src/solvers/mjmpm/rigid/kernels.py"],
        capture_output=True,
        check=True,
    ).stdout
    path = output / "baseline_rigid.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("mjmpm_reference_rigid", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def micro(inputs, baseline, cooperative=False, allow_roundoff=False):
    """Time fresh solves with identical initial velocities and impulses."""
    device = inputs[0].device
    original = [wp.clone(a) for a in inputs[17:20]]
    records = {}
    graphs = {}
    results = {}
    current = rk
    if cooperative:
        from newton._src.solvers.mjmpm.rigid import warp_sweeps  # noqa: PLC0415

        current = warp_sweeps
    for name, module in (("baseline", baseline), ("current", current)):
        packed = wp.zeros(inputs[3].shape[0], dtype=module.PackedContact, device=device)
        fallback = wp.zeros(inputs[0].shape[0], dtype=int, device=device)

        def solve(module=module, packed=packed, fallback=fallback):
            for a, b in zip(inputs[17:20], original, strict=True):
                wp.copy(a, b)
            is_cooperative = cooperative and module is current
            kernel = module.solve_contact_buckets_cooperative if is_cooperative else module.solve_contact_buckets_local
            wp.launch(
                kernel,
                dim=(inputs[0].shape[0], 32) if is_cooperative else inputs[0].shape[0],
                inputs=[*inputs, 60, packed, fallback],
                device=device,
                block_dim=32 if is_cooperative else 256,
            )

        solve()
        results[name] = [a.numpy().copy() for a in inputs[17:20]]
        if fallback.numpy().any():
            raise AssertionError("Frozen probe requires the local path")
        with wp.ScopedCapture(device=device, force_module_load=False) as capture:
            for _ in range(10):
                solve()
        graphs[name] = capture.graph
        records[name] = []
    equal = True
    differences = []
    for a, b in zip(results["baseline"], results["current"], strict=True):
        equal = equal and np.array_equal(a.view(np.uint32), b.view(np.uint32))
        differences.append(float(np.max(np.abs(a - b))))
        if allow_roundoff:
            np.testing.assert_allclose(a, b, rtol=1.0e-5, atol=1.0e-5)
        else:
            np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))
    for _ in range(5):
        for name, graph in graphs.items():
            wp.synchronize_device(device)
            start = time.perf_counter()
            wp.capture_launch(graph)
            wp.synchronize_device(device)
            records[name].append(1000 * (time.perf_counter() - start) / 10)
    ids = inputs[2].numpy()
    bodies = inputs[3].numpy()
    count = active = 0
    for first, last in zip(inputs[0].numpy(), inputs[1].numpy(), strict=True):
        selected = bodies[ids[first:last]]
        count += len(selected)
        active += int(np.any(selected >= 0, axis=1).sum())
    for a, b in zip(inputs[17:20], original, strict=True):
        wp.copy(a, b)
    return {"contacts": count, "active": active, "bitwise_equal": equal, "max_abs_differences": differences} | {
        f"{name}_ms": float(np.median(values)) for name, values in records.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("micro", "frozen", "verify", "feedback", "full", "display"))
    parser.add_argument("--snapshots", type=Path)
    parser.add_argument("--cooperative", action="store_true")
    parser.add_argument("--allow-roundoff", action="store_true")
    parser.add_argument("--variant", choices=("baseline", "current"), default="current")
    parser.add_argument("--reference", default="a84a6331")
    parser.add_argument("--feedback", choices=("reference", "fused"), default="fused")
    parser.add_argument("--column-solver", choices=("reference", "local"), default="local")
    parser.add_argument("--properties", choices=("reference", "cached"), default="cached")
    parser.add_argument("--batch-opaque", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--timing", choices=("synchronized", "native"), default="synchronized")
    parser.add_argument("--output-dir", type=Path, required=True)
    opts = parser.parse_args()
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline(opts.output_dir, opts.reference)
    if opts.mode == "frozen":
        types = (
            int,
            int,
            int,
            wp.vec2i,
            wp.vec3,
            wp.vec3,
            wp.vec3,
            wp.vec4,
            float,
            wp.mat33,
            int,
            int,
            float,
            int,
            int,
            wp.mat33,
            rk.mat63,
            wp.spatial_vector,
            wp.vec3,
            wp.spatial_vector,
        )
        records = []
        for path in sorted(opts.snapshots.glob("*.npz")):
            with np.load(path) as data:
                arrays = [wp.array(data[f"a{i}"], dtype=t, device="cuda:0") for i, t in enumerate(types)]
            record = {"snapshot": path.name} | micro(arrays, baseline, opts.cooperative, opts.allow_roundoff)
            records.append(record)
            print(record, flush=True)
        (opts.output_dir / "report.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        return
    if opts.mode in ("full", "display"):
        from newton._src.solvers.mjmpm.coupling import articulated  # noqa: PLC0415
        from newton._src.solvers.mjmpm.environment import Environment  # noqa: PLC0415
        from newton._src.solvers.mjmpm.mpm import solve_rheology as rheology  # noqa: PLC0415

        initialize = Environment._init_contact_buffers

        def init(env):
            initialize(env)
            env._use_cooperative_rigid_buckets = opts.cooperative and opts.variant == "current"

        Environment._init_contact_buffers = init
        initialize_mobility = articulated.LinkMobility.__init__

        def init_mobility(mobility, *a, **kw):
            initialize_mobility(mobility, *a, **kw)
            if opts.column_solver == "reference" or opts.variant == "baseline":
                mobility._local_columns = None
                mobility._local_in_place = None

        articulated.LinkMobility.__init__ = init_mobility
        rheology._FUSE_RIGIDITY_FEEDBACK = opts.feedback == "fused" and opts.variant == "current"
        initialize_example = Example.__init__
        from newton.examples.mjmpm.support.rice_cooker_visuals import RiceCookerVisuals  # noqa: PLC0415

        initialize_visuals = RiceCookerVisuals.__init__

        def init_visuals(visuals, *a, **kw):
            kw["batch_opaque"] = opts.batch_opaque and opts.variant == "current"
            initialize_visuals(visuals, *a, **kw)

        RiceCookerVisuals.__init__ = init_visuals

        def init_example(example, *a, **kw):
            initialize_example(example, *a, **kw)
            if example.visuals is not None and (opts.properties == "reference" or opts.variant == "baseline"):
                example.visuals._reuse_instance_properties = False

        Example.__init__ = init_example
        if opts.variant == "baseline":
            articulated._factor_matrix = lambda nv, matrix: wp.launch(
                articulated._cholesky, dim=matrix.shape[0], inputs=[nv, matrix]
            )
            rk.solve_contact_buckets_local = baseline.solve_contact_buckets_local
            rk.finish_articulated_contacts = baseline.finish_articulated_contacts
            rk.PackedContact = baseline.PackedContact
        if opts.mode == "full":
            from scripts.benchmark_mjmpm_rigidity import main as benchmark  # noqa: PLC0415

            sys.argv = [sys.argv[0], "--output", str(opts.output_dir / "full.json")]
        else:
            from scripts.diagnose_mjmpm_rice_rendering import main as benchmark  # noqa: PLC0415

            sys.argv = [sys.argv[0], "full", "--output-dir", str(opts.output_dir), "--timing", opts.timing]
            if opts.visible:
                sys.argv.append("--visible")
        benchmark()
        return

    wp.set_device("cuda:0")
    args = Example.create_parser().parse_args(["--viewer", "null"])
    args.test = True
    frames = 516 if opts.mode == "micro" else 990
    viewer = ViewerNull(num_frames=frames)
    example = Example(viewer, args)
    if opts.mode == "feedback":
        from newton._src.solvers.mjmpm.mpm import solve_rheology as rheology  # noqa: PLC0415

        original_feedback = rheology.apply_rigidity_operator
        feedback_stats = wp.zeros(2, dtype=int)
        saved_collider = wp.empty(example.model.particle_count, dtype=wp.vec3)
        saved_accumulated = wp.empty(example.solver.environment.model.body_count, dtype=wp.spatial_vector)
        expected = [
            wp.empty_like(saved_collider),
            wp.empty(6 * example.solver.environment.model.body_count, dtype=float),
            wp.empty_like(saved_accumulated),
        ]

        def feedback(operator, impulse, collider, body_delta):
            independent = operator[3]
            if collider.shape != saved_collider.shape or body_delta.shape != expected[1].shape:
                raise AssertionError("Feedback verification workspace shape differs from this scene")
            wp.copy(saved_collider, collider)
            wp.copy(saved_accumulated, independent.velocity_delta)
            rheology._FUSE_RIGIDITY_FEEDBACK = True
            original_feedback(operator, impulse, collider, body_delta)
            arrays = (collider, body_delta, independent.velocity_delta)
            for source, dest in zip(arrays, expected, strict=True):
                wp.copy(dest, source)
            wp.copy(collider, saved_collider)
            wp.copy(independent.velocity_delta, saved_accumulated)
            rheology._FUSE_RIGIDITY_FEEDBACK = False
            original_feedback(operator, impulse, collider, body_delta)
            for source, dest in zip(arrays, expected, strict=True):
                wp.launch(
                    compare_bits,
                    dim=source.size * wp.types.type_size_in_bytes(source.dtype) // 4,
                    inputs=[source.view(wp.uint32).flatten(), dest.view(wp.uint32).flatten(), feedback_stats],
                )

        rheology.apply_rigidity_operator = feedback
        try:
            for frame in range(frames):
                example.step()
                example.test_post_step()
                if frame % 120 == 0:
                    print(f"Feedback progress {frame + 1}/{frames}", flush=True)
            example.test_final()
            mismatch, compared = map(int, feedback_stats.numpy())
            report = {"mismatches": mismatch, "arrays_compared": compared}
            (opts.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if mismatch or not compared:
                raise AssertionError(report)
        finally:
            rheology.apply_rigidity_operator = original_feedback
            viewer.close()
        return
    env = example.solver.environment
    env._use_cooperative_rigid_buckets = opts.cooperative
    from newton._src.solvers.mjmpm.rigid import warp_sweeps  # noqa: PLC0415

    original_launch = wp.launch
    selected = {180, 300, 515}
    flag = wp.zeros(1, dtype=int)
    snapshots = {}
    call = 0
    stats = wp.zeros(2, dtype=int)
    maxima = wp.zeros(3, dtype=float)
    output_arrays = (env.work.body_qd, env.contact_impulses, env.rigid_impulses)
    saved = [wp.zeros_like(a) for a in output_arrays]
    expected = [wp.zeros_like(a) for a in output_arrays]
    packed = wp.zeros(env.contacts.rigid_contact_max, dtype=baseline.PackedContact)
    fallback = wp.zeros(env._bucket_count, dtype=int)

    def launch(*a, **kw):
        nonlocal call
        kernel = kw.get("kernel", a[0] if a else None)
        if kernel not in (rk.solve_contact_buckets_local, warp_sweeps.solve_contact_buckets_cooperative):
            return original_launch(*a, **kw)
        inputs = kw["inputs"]
        if opts.mode == "verify":
            for dest, source in zip(saved, output_arrays, strict=True):
                wp.copy(dest, source)
            original_launch(*a, **kw)
            for dest, source, initial in zip(expected, output_arrays, saved, strict=True):
                wp.copy(dest, source)
                wp.copy(source, initial)
            original_launch(
                baseline.solve_contact_buckets_local,
                dim=env._bucket_count,
                inputs=[*inputs[:21], packed, fallback],
            )
            for index, (dest, source) in enumerate(zip(expected, output_arrays, strict=True)):
                if opts.allow_roundoff:
                    original_launch(
                        compare_roundoff,
                        dim=source.size * wp.types.type_size_in_bytes(source.dtype) // 4,
                        inputs=[dest.view(float).flatten(), source.view(float).flatten(), stats, maxima, index],
                    )
                else:
                    original_launch(
                        compare_bits,
                        dim=source.size * wp.types.type_size_in_bytes(source.dtype) // 4,
                        inputs=[dest.view(wp.uint32).flatten(), source.view(wp.uint32).flatten(), stats],
                    )
            return None
        if example.model.device.is_capturing:
            if call in (0, 1, 10, 11):
                snapshots[call] = [wp.zeros_like(v) for v in inputs[:20]]
                for dest, source in zip(snapshots[call], inputs[:20], strict=True):
                    original_launch(copy_when, dim=source.size, inputs=[flag, source.flatten(), dest.flatten()])
            call += 1
        return original_launch(*a, **kw)

    wp.launch = launch
    records = []
    try:
        for frame in range(frames):
            flag.fill_(int(frame in selected))
            example.step()
            example.test_post_step()
            if opts.mode == "micro" and frame in selected:
                for index, inputs in snapshots.items():
                    record = {"frame": frame, "solve": index} | micro(
                        inputs, baseline, opts.cooperative, opts.allow_roundoff
                    )
                    records.append(record)
                    np.savez(
                        opts.output_dir / f"frame_{frame}_{index}.npz",
                        **{f"a{i}": a.numpy() for i, a in enumerate(inputs)},
                    )
                    print(record, flush=True)
            if frame % 120 == 0:
                print(f"Progress {frame + 1}/{frames}", flush=True)
        if frames == 990:
            example.test_final()
        mismatch, compared = map(int, stats.numpy())
        report = {
            "mode": opts.mode,
            "mismatches": mismatch,
            "arrays_compared": compared,
            "max_abs_differences": maxima.numpy().tolist(),
            "allow_roundoff": opts.allow_roundoff,
            "micro": records,
        }
        (opts.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if opts.mode == "verify" and (mismatch or compared == 0):
            raise AssertionError(report)
    finally:
        wp.launch = original_launch
        viewer.close()


if __name__ == "__main__":
    main()

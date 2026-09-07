# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark linked versus packed contacts in an identical frozen coarse PCG."""

import argparse
import json
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.mjvbd_v2 import contact_projection as cp
from newton._src.solvers.mjvbd_v2 import particle_multilevel as ml
from newton._src.solvers.mjvbd_v2.coarse_pcg_split import SplitCoarsePCG


def benchmark_rows(correction, model, q, dt, repeats=50):
    """Compare graph replay with identical matrix, RHS, iterations and row order."""
    calls = []
    launch = wp.launch

    def record(kernel, *args, **kwargs):
        if kernel == ml._solve_energy_galerkin_pcg_persistent:
            calls.append((args, kwargs))
        return launch(kernel, *args, **kwargs)

    with patch.object(wp, "launch", record), patch.object(correction, "coarse_use_split_pcg", False):
        correction.restrict_and_prolong(model, q, wp.zeros_like(q), dt)
    _, original = calls[0]
    op = correction.contact_projection
    return benchmark_call(original, op, model.device, repeats)


def benchmark_call(original, op, device, repeats=50):
    """Compare layouts and schedules for one immutable block system."""
    launch = wp.launch
    op.compact()
    variants = {}
    for label in ("linked", "packed", "split_linked", "split_packed"):
        data = cp.ContactProjectionData()
        for field in cp.ContactProjectionData.vars:
            setattr(data, field, getattr(op.data, field))
        if label.endswith("linked"):
            data.packed_ready = None
        inputs = [*original["inputs"][:-1], data]
        outputs = [wp.empty_like(array) for array in original["outputs"]]
        options = {**original, "inputs": inputs, "outputs": outputs}
        split = SplitCoarsePCG(device) if label.startswith("split") else None

        def solve(split=split, options=options):
            if split is None:
                launch(ml._solve_energy_galerkin_pcg_persistent, **options)
            else:
                split.solve(options["inputs"], options["outputs"])

        solve()
        with wp.ScopedCapture(device=device) as capture:
            solve()
        variants[label] = (capture.graph, outputs, data, split)
        if label.endswith("packed"):
            with wp.ScopedCapture(device=device) as packed_capture:
                op.compact()
                solve()
            variants[label + "_total"] = (packed_capture.graph, outputs, data, split)
    for candidate in variants.values():
        for left, right in zip(variants["linked"][1], candidate[1], strict=True):
            np.testing.assert_array_equal(left.numpy(), right.numpy())
    with wp.ScopedCapture(device=device) as capture:
        op.compact()
    graphs = {label: values[0] for label, values in variants.items()}
    graphs["packing"] = capture.graph
    times = {name: [] for name in graphs}
    for graph in graphs.values():
        for _ in range(5):
            wp.capture_launch(graph)
    for trial in range(8):
        order = list(graphs) if trial % 2 == 0 else list(reversed(graphs))
        for label in order:
            start = wp.Event(device, enable_timing=True)
            end = wp.Event(device, enable_timing=True)
            wp.record_event(start)
            for _ in range(repeats):
                wp.capture_launch(graphs[label])
            wp.record_event(end)
            times[label].append(wp.get_event_elapsed_time(start, end) / repeats)
    medians = {label + "_ms": float(np.median(values)) for label, values in times.items()}
    return {
        "clusters": original["inputs"][0],
        "base_blocks": original["inputs"][4].size,
        "contact_pairs": int(op.data.edge_count.numpy()[0]),
        "pcg_iterations": original["inputs"][6],
        "outputs_bitwise_equal": True,
        **medians,
        "net_pcg_and_packing_ratio": medians["packed_total_ms"] / medians["linked_ms"],
        "split_ratio": medians["split_linked_ms"] / medians["linked_ms"],
        "samples_ms": times,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    device = wp.get_device("cuda:0")
    wp.set_device(device)
    wp.config.log_level = wp.LOG_WARNING
    for count, width in ((128, 3), (256, 3), (256, 17), (512, 17), (1024, 17), (2048, 17)):
        for contacting in (False, True):
            identity = np.eye(3, dtype=np.float32)
            shift = np.arange(-(width // 2), width // 2 + 1)
            columns = ((np.arange(count)[:, None] + shift) % count).astype(np.int32).ravel()
            blocks = np.tile(-identity, (count * width, 1, 1))
            slots = np.arange(count, dtype=np.int32) * width + width // 2
            blocks[slots] = width * identity
            op = cp.ContactProjection(count, max(1, count * 8), device)
            op.reset()
            if contacting:
                first = np.repeat(np.arange(count), 8)
                second = (first + np.tile(np.arange(8) * 11 + 29, count)) % count
                particles = np.column_stack((first, second, np.full((count * 8, 2), -1)))
                wp.launch(
                    cp.project_records,
                    dim=count * 8,
                    inputs=[
                        wp.array(particles, dtype=wp.vec4i, device=device),
                        wp.array(np.tile([1, -1, 0, 0], (count * 8, 1)), dtype=wp.vec4, device=device),
                        wp.array(np.tile(10 * identity, (count * 8, 1, 1)), dtype=wp.mat33, device=device),
                        wp.array(np.arange(count), dtype=wp.int32, device=device),
                        op.data,
                    ],
                    device=device,
                )
                blocks[slots] += 160 * identity
            inputs = [
                count,
                wp.array(np.arange(count + 1) * width, dtype=wp.int32, device=device),
                wp.array(columns, dtype=wp.int32, device=device),
                wp.array(slots, dtype=wp.int32, device=device),
                wp.array(blocks, dtype=wp.mat33, device=device),
                wp.array(np.random.default_rng(9).normal(size=(count, 3)), dtype=wp.vec3, device=device),
                args.iterations,
                True,
                1e-4,
                op.data,
            ]
            outputs = [
                *[wp.zeros(count, dtype=wp.vec3, device=device) for _ in range(5)],
                wp.zeros(count, dtype=wp.mat33, device=device),
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.zeros(5 + args.iterations, dtype=float, device=device),
                wp.zeros(2, dtype=wp.int32, device=device),
            ]
            result = benchmark_call(
                {"dim": 256, "block_dim": 256, "inputs": inputs, "outputs": outputs, "device": device},
                op,
                device,
                args.repeats,
            )
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

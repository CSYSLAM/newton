# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compare compact and distributed CUDA layouts for identical rigid feedback."""

import json

import numpy as np
import warp as wp

from newton._src.solvers.mjmpm.mpm.rigidity_feedback import apply_mobility_feedback, apply_mobility_feedback_rows
from newton._src.solvers.mjmpm.rigid.response import mat66


def main():
    wp.set_device("cuda:0")
    rng = np.random.default_rng(2026)
    calls_per_graph = 100
    replays_per_sample = 10
    report = []
    for count in (24, 48, 64):
        for active in (4, 12, count):
            if active > count:
                continue
            offsets = wp.array(np.arange(2 * count + 1, dtype=np.int32))
            slots = wp.array(np.arange(count, dtype=np.int32))
            links = wp.array(np.arange(count, dtype=np.int32).reshape(1, -1))
            active_links = wp.clone(links)
            active_count = wp.array([active], dtype=int)
            basis = rng.normal(size=(6 * count, 6 * count)).astype(np.float32)
            mobility = wp.array((basis @ basis.T)[None])
            enabled = wp.array((np.arange(count) % 3 == 0).astype(np.int32))
            independent_basis = rng.normal(size=(count, 6, 6)).astype(np.float32)
            independent_values = independent_basis @ np.transpose(independent_basis, (0, 2, 1))
            independent = wp.array(independent_values, dtype=mat66)
            wrenches = wp.array(rng.normal(size=6 * count).astype(np.float32))
            accumulated = wp.zeros(count, dtype=wp.spatial_vector)
            twists = wp.zeros(6 * count, dtype=float)
            args = [
                offsets,
                slots,
                links,
                active_links,
                active_count,
                mobility,
                enabled,
                independent,
                wrenches,
                accumulated,
                twists,
            ]
            graphs = {}
            expected = None

            def launch(layout, count=count, args=args):
                if layout == "rows":
                    wp.launch_tiled(apply_mobility_feedback_rows, dim=(count,), inputs=args, block_dim=32)
                else:
                    wp.launch(apply_mobility_feedback, count, args)

            for layout in ("compact", "rows"):
                launch(layout)
                result = twists.numpy().view(np.uint32)
                if expected is None:
                    expected = result.copy()
                else:
                    np.testing.assert_array_equal(result, expected)
                with wp.ScopedCapture() as capture:
                    for _ in range(calls_per_graph):
                        launch(layout)
                graphs[layout] = capture.graph
            times = {"compact": [], "rows": []}
            start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
            stream = wp.get_stream()
            for _ in range(3):
                for layout, graph in graphs.items():
                    wp.capture_launch(graph)
                    stream.record_event(start)
                    for _ in range(replays_per_sample):
                        wp.capture_launch(graph)
                    stream.record_event(end)
                    elapsed_us = wp.get_event_elapsed_time(start, end) * 1000.0
                    times[layout].append(elapsed_us / (calls_per_graph * replays_per_sample))
            row = {
                "bodies": count,
                "active_links": active,
                "compact_us": float(np.median(times["compact"])),
                "rows_us": float(np.median(times["rows"])),
            }
            report.append(row)
            print(json.dumps(row), flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

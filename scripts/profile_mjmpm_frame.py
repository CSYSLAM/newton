# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Measure captured solver phases after optimization, including the overlapping preparation stream."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.mjmpm.coupling import articulated
from newton._src.solvers.mjmpm.mpm import solve_rheology as rheology
from newton._src.solvers.mjmpm.rigid import kernels as rigid
from newton._src.solvers.mjmpm.rigid import warp_sweeps
from newton.examples.mjmpm.example_mjmpm_w1_rice_cooker import Example
from newton.viewer import ViewerNull


class Events:
    def __init__(self, device):
        self.device = device
        self.used = defaultdict(int)
        self.pairs = {}

    def wrap(self, obj, method, name):
        original = getattr(obj, method)
        pairs = [
            (wp.Event(self.device, enable_timing=True), wp.Event(self.device, enable_timing=True)) for _ in range(32)
        ]
        self.pairs[name] = pairs

        def call(*a, **kw):
            if not self.device.is_capturing:
                return original(*a, **kw)
            index = self.used[name]
            self.used[name] += 1
            start, end = pairs[index]
            stream = wp.get_stream(self.device)
            stream.record_event(start)
            result = original(*a, **kw)
            stream.record_event(end)
            return result

        setattr(obj, method, call)

    def read(self):
        return {
            name: sum(wp.get_event_elapsed_time(a, b, synchronize=False) for a, b in pairs[: self.used[name]])
            for name, pairs in self.pairs.items()
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    opts = parser.parse_args()
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    wp.set_device("cuda:0")
    args = Example.create_parser().parse_args(["--viewer", "null"])
    args.test = True
    viewer = ViewerNull(num_frames=520)
    example = Example(viewer, args)
    env = example.solver.environment
    events = Events(example.model.device)
    for obj, method, name in (
        (example, "_simulate_frame", "frame"),
        (example.solver, "step", "coupled_step"),
        (env, "step", "environment"),
        (env.detector, "collide", "collision"),
        (env, "_solve_rigid_contacts", "rigid_contacts"),
        (env, "_prepare_constrained_material", "prepare_response"),
        (env, "_finish_constrained_material", "finish_response"),
        (env.material, "prepare_step", "mpm_prepare_overlapped"),
        (env.material, "step", "mpm_main"),
        (env.material, "_particles_to_cells", "mpm_binning"),
        (env.material, "_rebuild_scratchpad", "mpm_spaces"),
        (env.material, "_assemble_particle_terms", "mpm_particle_terms"),
        (rheology, "_run_solver_loop", "mpm_iterations"),
    ):
        events.wrap(obj, method, name)
    events.wrap(env._link_mobility, "update", "link_mobility")
    events.wrap(env._link_mobility._solver, "step", "mujoco")
    original_launch = wp.launch
    launch_calls = {}
    for name in ("_cholesky", "_cholesky_cooperative", "_solve_columns", "_multiply", "_solve_in_place", "_project"):
        holder = SimpleNamespace(call=original_launch)
        events.wrap(holder, "call", name)
        launch_calls[getattr(articulated, name)] = holder.call
    for name in (
        "prepare_contacts",
        "prepare_link_contact_responses",
        "finish_articulated_contacts",
        "solve_contact_buckets",
    ):
        holder = SimpleNamespace(call=original_launch)
        events.wrap(holder, "call", name)
        launch_calls[getattr(rigid, name)] = holder.call
    holder = SimpleNamespace(call=original_launch)
    events.wrap(holder, "call", "cooperative_sweeps")
    launch_calls[warp_sweeps.solve_contact_buckets_cooperative] = holder.call

    def launch(*a, **kw):
        kernel = kw.get("kernel", a[0] if a else None)
        return launch_calls.get(kernel, original_launch)(*a, **kw)

    wp.launch = launch
    samples = []
    try:
        for frame in range(520):
            example.step()
            example.test_post_step()
            if 280 <= frame < 320 or 500 <= frame < 520:
                wp.synchronize_device(example.model.device)
                samples.append({"index": frame, **events.read()})
            if frame % 120 == 0:
                print(f"Profile {frame + 1}/520", flush=True)
        summary = {}
        for name, start, end in (("carry", 280, 320), ("pour", 500, 520)):
            selected = [r for r in samples if start <= r["index"] < end]
            summary[name] = {k: float(np.median([r[k] for r in selected])) for k in events.pairs}
        report = {
            "phases_ms": summary,
            "calls": dict(events.used),
            "samples": samples,
            "nv": env._link_mobility.nv,
            "links": env._link_mobility.link_count,
        }
        opts.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(summary), flush=True)
    finally:
        wp.launch = original_launch
        viewer.close()


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Profile captured rice-cooker rigidity assembly and probe frozen GS inputs.

Run with UV_OFFLINE=1 and the repository on PYTHONPATH. The solver is not
modified; diagnostic solves use private copies of captured inputs.
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from newton._src.solvers.mjmpm.mpm import solver_implicit_mpm as sm
from newton._src.solvers.mjmpm.rigid import kernels as rk
from newton._src.solvers.mjmpm.rigid import warp_sweeps
from newton.examples.mjmpm.example_mjmpm_w1_rice_cooker import Example
from newton.viewer import ViewerNull


@wp.kernel
def copy_when(flag: wp.array[int], source: wp.array[Any], dest: wp.array[Any]):
    i = wp.tid()
    if flag[0] != 0:
        dest[i] = source[i]


class Events:
    def __init__(self, device):
        self.device = device
        self.pairs = {
            name: [(wp.Event(device, enable_timing=True), wp.Event(device, enable_timing=True)) for _ in range(24)]
            for name in ("rigidity_all", "fill", "J_triplets", "IJtm_triplets")
        }
        self.used = defaultdict(int)

    def call(self, name, function, *args, **kwargs):
        if not self.device.is_capturing:
            return function(*args, **kwargs)
        index = self.used[name]
        self.used[name] += 1
        start, end = self.pairs[name][index]
        stream = wp.get_stream(self.device)
        stream.record_event(start)
        result = function(*args, **kwargs)
        stream.record_event(end)
        return result

    def read(self):
        return {
            name: sum(wp.get_event_elapsed_time(a, b, synchronize=False) for a, b in pairs[: self.used[name]])
            for name, pairs in self.pairs.items()
        }


def projection_residual(args, velocities, impulses):
    """Evaluate the existing contact update on frozen final velocities, without applying it."""
    begin, end, ids, bodies, radii_a, radii_b, normals, params, masses, inertias = [a.numpy() for a in args[:10]]
    slots = args[10].numpy() if args[10] is not None else np.full(len(masses), -1)
    pm = args[15].numpy() if args[15] is not None else None
    normal_max = 0.0
    friction_max = 0.0
    for offset in range(int(begin[0]), int(end[0])):
        c = int(ids[offset])
        a, b = map(int, bodies[c])
        if a < 0 and b < 0:
            continue
        ra, rb = radii_a[c].astype(float), radii_b[c].astype(float)
        n = normals[c].astype(float)
        target, softness, denom, mu = map(float, params[c])

        def point(body, radius):
            if body < 0:
                return np.zeros(3)
            return velocities[body, :3] + np.cross(velocities[body, 3:], radius)

        def response(body, radius, impulse, contact=c):
            if body < 0:
                return np.zeros(3)
            if slots[body] >= 0:
                return pm[contact] @ impulse
            return masses[body] * impulse + np.cross(inertias[body] @ np.cross(radius, impulse), radius)

        relative = point(b, rb) - point(a, ra)
        old = impulses[c].astype(float)
        old_n = float(old @ n)
        new_n = max(0.0, old_n + (target - relative @ n - softness * old_n) / max(denom + softness, 1e-12))
        normal_max = max(normal_max, abs(new_n - old_n) * (denom + softness))
        dn = (new_n - old_n) * n
        relative += response(a, ra, dn) + response(b, rb, dn)
        tangent_v = relative - n * (relative @ n)
        old_t = old - old_n * n
        tangent = old_t.copy()
        speed = np.linalg.norm(tangent_v)
        effective = denom
        if speed > 1e-9:
            direction = tangent_v / speed
            effective = float(direction @ (response(a, ra, direction) + response(b, rb, direction)))
            tangent -= tangent_v / max(effective, 1e-12)
        limit = mu * new_n
        length = np.linalg.norm(tangent)
        if length > limit:
            tangent *= limit / length
        friction_max = max(friction_max, np.linalg.norm(tangent - old_t) * max(effective, 1e-12))
    return normal_max, friction_max


def probe(snapshot, frame, call, dt, warm=None, reference=None):
    args = snapshot
    original_impulse = wp.clone(args[18])
    if warm is not None:
        args[18].assign(warm)
    saved = [wp.clone(a) for a in args[17:20]]
    outputs = {}
    for count in (0, 5, 10, 20, 40, 60, 120):
        for a, b in zip(args[17:20], saved, strict=True):
            a.assign(b)
        wp.launch(rk.solve_contact_buckets, dim=1, inputs=[*args[:20], count, None], device=args[0].device)
        outputs[count] = [a.numpy() for a in args[17:20]]
    if reference is None:
        reference = outputs[120]
    slots = args[10].numpy()
    links = slots >= 0
    records = []
    for count, (velocity, impulse, wrench) in outputs.items():
        nres, tres = projection_residual(args, velocity, impulse)
        dv = velocity - reference[0]
        dw = wrench - reference[2]
        record = {
            "frame": frame,
            "substep": call // 2,
            "solve": "initial" if call % 2 == 0 else "recovery",
            "guess": "first_solve_60" if warm is not None else "current",
            "iterations": count,
            "normal_projection_mps": nres,
            "friction_projection_mps": tres,
            "linear_vs120_mps": float(np.linalg.norm(dv[:, :3], axis=1).max()),
            "angular_vs120_radps": float(np.linalg.norm(dv[:, 3:], axis=1).max()),
            "link_force_vs120_N": float(np.linalg.norm(dw[links, :3], axis=1).max(initial=0) / dt),
            "net_link_force_vs120_N": float(np.linalg.norm(dw[links, :3].sum(axis=0)) / dt),
            "net_link_vertical_vs120_N": float(abs(dw[links, 2].sum()) / dt),
            "link_torque_vs120_Nm": float(np.linalg.norm(dw[links, 3:], axis=1).max(initial=0) / dt),
        }
        records.append(record)
    for a, b in zip(args[17:20], saved, strict=True):
        a.assign(b)
    args[18].assign(original_impulse)
    return records, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=990)
    opts = parser.parse_args()
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    wp.set_device("cuda:0")
    args = Example.create_parser().parse_args(["--viewer", "null"])
    args.test = True
    viewer = ViewerNull(num_frames=opts.frames)
    example = Example(viewer, args)
    env = example.solver.environment
    if not env._single_bucket or env.rigid_iterations != 60:
        raise RuntimeError("This diagnostic expects the one-world, 60-sweep rice-cooker workload")
    events = Events(example.model.device)
    flag = wp.zeros(1, dtype=int, device=example.model.device)
    template = [
        env._bucket_begin,
        env._bucket_end,
        env._contact_ids,
        env._contact_bodies,
        env._contact_ra,
        env._contact_rb,
        env._contact_normal,
        env._contact_params,
        env._contact_mass_inv,
        env._contact_inertia_inv,
        env._link_slot,
        env._link_body,
        env._mobility,
        env._contacted_links,
        env._contacted_count,
        env._point_mobility,
        env._link_responses,
        env.work.body_qd,
        env.contact_impulses,
        env.rigid_impulses,
    ]
    snapshots = {index: [wp.zeros_like(a) for a in template] for index in (0, 1, 10, 11)}
    original_launch = wp.launch
    original_build = sm.build_rigidity_operator
    original_triplets = sm.build_rigidity_operator.__globals__["wps"].bsr_from_triplets
    sparse = sm.build_rigidity_operator.__globals__["wps"]
    active = False
    triplet_index = 0
    sweep_index = 0
    operator_shapes = []

    def build(*a, **kw):
        nonlocal active, triplet_index
        active = True
        triplet_index = 0
        if example.model.device.is_capturing:
            operator_shapes.append(
                {
                    "node_capacity": kw["node_volumes"].shape[0],
                    "bodies": kw["body_q"].shape[0],
                    "deformable_vertices": kw["collider"].collider_particle_ids.shape[0],
                }
            )
        try:
            return events.call("rigidity_all", original_build, *a, **kw)
        finally:
            active = False

    def triplets(*a, **kw):
        nonlocal triplet_index
        if not active:
            return original_triplets(*a, **kw)
        name = "J_triplets" if triplet_index == 0 else "IJtm_triplets"
        triplet_index += 1
        return events.call(name, original_triplets, *a, **kw)

    def launch(*a, **kw):
        nonlocal sweep_index
        kernel = kw.get("kernel", a[0] if a else None)
        if active and kernel is original_build.__globals__["fill_collider_coupling_matrices"]:
            return events.call("fill", original_launch, *a, **kw)
        if (
            kernel in (rk.solve_contact_buckets_local, warp_sweeps.solve_contact_buckets_cooperative)
            and example.model.device.is_capturing
        ):
            index = sweep_index
            sweep_index += 1
            if index in snapshots:
                inputs = kw["inputs"]
                for dest, source in zip(snapshots[index], inputs[:20], strict=True):
                    original_launch(copy_when, dim=source.size, inputs=[flag, source.flatten(), dest.flatten()])
        return original_launch(*a, **kw)

    sm.build_rigidity_operator = build
    sparse.bsr_from_triplets = triplets
    wp.launch = launch
    selected = {120, 180, 240, 300, 360, 420, 480, 510, 540, 600, 720, 840, 960}
    phase_samples = []
    convergence = []
    started = time.perf_counter()
    try:
        for frame in range(opts.frames):
            flag.fill_(int(frame in selected))
            example.step()
            example.test_post_step()
            wp.synchronize_device(example.model.device)
            if frame >= 60 and frame % 15 == 0:
                phase_samples.append({"frame": frame, **events.read()})
            if frame in selected:
                first_outputs = None
                for index, snapshot in snapshots.items():
                    records, outputs = probe(snapshot, frame, index, example.sim_dt)
                    convergence.extend(records)
                    if index % 2 == 0:
                        first_outputs = outputs
                    else:
                        records, _ = probe(
                            snapshot, frame, index, example.sim_dt, warm=first_outputs[60][1], reference=outputs[120]
                        )
                        convergence.extend(records)
                print(f"Captured convergence frame {frame}", flush=True)
            if frame % 120 == 0:
                print(f"Progress {frame + 1}/{opts.frames}", flush=True)
        example.solver.check_capacity_status()
        if opts.frames >= 990:
            example.test_final()
        report = {
            "configuration": {
                "frames": opts.frames,
                "rice": example.rice_end - example.rice_begin,
                "substeps": args.substeps,
                "rigid_iterations": args.rigid_iterations,
                "mpm_iterations": args.iterations,
                "contact_capacity": env.contacts.rigid_contact_max,
                "snapshot_calls": list(snapshots),
                "captured_sweep_calls": sweep_index,
            },
            "event_calls_per_frame": dict(events.used),
            "phase_samples": phase_samples,
            "operator_shapes": operator_shapes,
            "convergence": convergence,
            "rice_locations": example.rice_locations(),
            "diagnostic_wall_seconds": time.perf_counter() - started,
            "notes": "Instrumented phase timings, not production FPS. GS reference uses the identical frozen input at each budget; 120 sweeps is a comparison, not an exact solution. Projection residuals are velocity-scaled frozen contact updates; absolute state differences compare to 120 sweeps. Count 0 applies only the initial warm-start impulses.",
        }
        (opts.output_dir / "physics_profile.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "phase_median_ms": {
                        name: float(np.median([s[name] for s in phase_samples])) for name in events.pairs
                    },
                    "event_calls": dict(events.used),
                    "snapshot_count": len(convergence) // 7,
                }
            ),
            flush=True,
        )
    finally:
        wp.launch = original_launch
        sparse.bsr_from_triplets = original_triplets
        sm.build_rigidity_operator = original_build
        viewer.close()


if __name__ == "__main__":
    main()

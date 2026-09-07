# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A/B benchmark equivalent Jacobi launch sequences on generic surface models.

Both CUDA graphs reset the physical input state before each measured substep.
The unfused reference recreates the original scratch-clear/solve/apply launches;
Python interception happens during capture only, not during timed replay.
"""

import argparse
import json
import time
from contextlib import nullcontext

import numpy as np
import warp as wp

import newton
from newton._src.solvers.mjvbd_v2.vbd_soft.solver_vbd import SolverVBD
from newton.tests.test_mjvbd_v2_jacobi_fusion import unfused_jacobi


def benchmark(size, layers, repeats, coarse_contacts=False):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    for layer in range(layers):
        builder.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, layer * 0.004),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=size,
            dim_y=size,
            cell_x=0.01,
            cell_y=0.01,
            mass=0.001,
            tri_ke=1.0e4,
            tri_ka=1.0e4,
            tri_kd=0.01,
            edge_ke=0.01,
            fix_left=True,
        )
    builder.color()
    model = builder.finalize(device="cuda:0")
    solver = SolverVBD(
        model,
        iterations=8,
        particle_enable_surface_cache=True,
        particle_enable_batched_jacobi=True,
        particle_jacobi_batch_count=2,
        particle_chebyshev_spectral_radius=0.8,
        particle_enable_self_contact=layers > 1,
        particle_enable_truncation_cache=True,
        particle_self_contact_radius=0.006,
        particle_self_contact_margin=0.009,
        particle_enable_multilevel_correction=coarse_contacts,
        particle_multilevel_operator="galerkin",
        particle_multilevel_checkpoints=(6,) if coarse_contacts else None,
    )
    state_in, state_out = model.state(), model.state()
    control = model.control()
    q, qd = wp.clone(state_in.particle_q), wp.clone(state_in.particle_qd)

    def step():
        state_in.particle_q.assign(q)
        state_in.particle_qd.assign(qd)
        solver.step(state_in, state_out, control, None, 1.0 / 600.0)

    step()
    graphs = {}
    graph_owners = {}
    labels = ("uncoupled", "coupled") if coarse_contacts else ("unfused", "fused")
    for label in labels:
        if coarse_contacts:
            solver.particle_multilevel.contact_projection_enabled = label == "coupled"
        with unfused_jacobi(solver) if label == "unfused" else nullcontext() as scratch:
            graph_owners[label] = scratch
            with wp.ScopedCapture(device=model.device) as capture:
                step()
        graphs[label] = capture.graph
        if coarse_contacts:
            graph_owners[label] = solver.particle_multilevel.contact_projection
    for _ in range(10):
        for graph in graphs.values():
            wp.capture_launch(graph)
    timings = {label: [] for label in graphs}
    for trial in range(8):
        order = labels if trial % 2 == 0 else labels[::-1]
        for label in order:
            wp.synchronize_device(model.device)
            start = time.perf_counter()
            for _ in range(repeats):
                wp.capture_launch(graphs[label])
            wp.synchronize_device(model.device)
            timings[label].append((time.perf_counter() - start) * 1000 / repeats)
    if not np.isfinite(state_out.particle_q.numpy()).all():
        raise ValueError("Nonfinite benchmark output")
    medians = {key: float(np.median(value)) for key, value in timings.items()}
    print(
        json.dumps(
            {
                "size": size,
                "layers": layers,
                "particles": model.particle_count,
                "self_contact_active": bool(solver.has_active_self_contact.numpy()[0]) if layers > 1 else False,
                "median_substep_ms": medians,
                "samples_ms": timings,
                "speedup_percent": (medians[labels[0]] / medians[labels[1]] - 1) * 100,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--coarse-contacts", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    wp.config.log_level = wp.LOG_WARNING
    for size, layers in ((8, 1), (32, 1), (64, 1), (32, 2)):
        benchmark(size, layers, args.repeats, args.coarse_contacts)


if __name__ == "__main__":
    main()

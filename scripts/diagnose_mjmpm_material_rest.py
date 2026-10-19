# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure material self-weight deformation before press contact."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverImplicitMPM, SolverMJMPM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("shared", "private"), default="private")
    parser.add_argument("--young", type=float, default=1.0e4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--solver", default="jacobi")
    parser.add_argument("--gravity", type=float, default=-9.81)
    parser.add_argument("--voxel-size", type=float, default=0.075)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--no-ground", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    wp.config.log_level = wp.LOG_WARNING
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, args.gravity))
    SolverMJMPM.register_custom_attributes(builder)
    if not args.no_ground:
        builder.add_ground_plane()
    builder.add_particle_grid(
        pos=wp.vec3(-0.125, -0.125, 0.035),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=6,
        dim_y=6,
        dim_z=4,
        cell_x=0.05,
        cell_y=0.05,
        cell_z=0.05,
        mass=0.05,
        jitter=0.0,
        radius_mean=0.025,
        custom_attributes={
            "mpm:young_modulus": args.young,
            "mpm:poisson_ratio": 0.2,
            "mpm:yield_stress": 1.0e6,
            "mpm:tensile_yield_ratio": 1.0,
        },
    )
    model = builder.finalize(device=args.device)
    options = {
        "grid_type": "fixed",
        "voxel_size": args.voxel_size,
        "grid_padding": 6,
        "max_active_cell_count": 256,
        "max_iterations": args.iterations,
        "solver": args.solver,
        "separate_worlds": True,
    }
    if args.backend == "private":
        solver = SolverMJMPM(model, mpm_options=options)
    else:
        config = SolverImplicitMPM.Config()
        for key, value in options.items():
            setattr(config, key, value)
        solver = SolverImplicitMPM(model, config=config)
    state = model.state()
    rows = []
    for step in range(args.steps + 1):
        if step in (0, 6, 60, 300, 600, args.steps):
            positions = state.particle_q.numpy()
            F = state.mpm.particle_elastic_strain.numpy()
            row = {
                "time": step / 600,
                "extent": np.ptp(positions, axis=0).tolist(),
                "z_range": [float(positions[:, 2].min()), float(positions[:, 2].max())],
                "F_diagonal": np.diagonal(F, axis1=1, axis2=2).mean(axis=0).tolist(),
                "elastic_volume_ratio": float(np.linalg.det(F).mean()),
                "Jp": [float(state.mpm.particle_Jp.numpy().min()), float(state.mpm.particle_Jp.numpy().max())],
                "max_stress": float(np.abs(state.mpm.particle_stress.numpy()).max()),
                "max_speed": float(np.linalg.norm(state.particle_qd.numpy(), axis=1).max()),
            }
            rows.append(row)
            print(row, flush=True)
        if step < args.steps:
            solver.step(state, state, None, None, 1.0 / 600)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"config": vars(args) | {"output": str(args.output)}, "samples": rows}, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

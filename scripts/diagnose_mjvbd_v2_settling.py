# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare settling policies from a common evolved state without changing demos."""

import argparse
import gc
import json
import time
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.viewer
from newton.examples.mjvbdv2 import (
    example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00 as tshirt,
)
from newton.examples.mjvbdv2 import (
    example_mjvbd_v2_cloth_twist as twist,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("twist", "tshirt"), required=True)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--cases", nargs="+")
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    wp.config.log_level = wp.LOG_WARNING
    wp.set_device("cuda:0")
    module = twist if args.scene == "twist" else tshirt
    original = newton.solvers.SolverMJVBDV2.__init__
    constructor = {}

    def configure(self, model, **kwargs):
        options = dict(kwargs["vbd_options"])
        options.pop("particle_jacobi_polish_iterations", None)
        options.update(particle_enable_multilevel_correction=False, particle_multilevel_checkpoints=None)
        kwargs["vbd_options"] = options
        constructor.update(kwargs)
        original(self, model, **kwargs)

    example_args = None if args.scene == "twist" else newton.examples.default_args(module.Example.create_parser())
    with patch.object(newton.solvers.SolverMJVBDV2, "__init__", configure):
        example = module.Example(newton.viewer.ViewerNull(), example_args)
    for _ in range(900):
        example.step()
    snapshots = [example.model.state(), example.model.state()]
    for snapshot, state in zip(snapshots, (example.state_0, example.state_1), strict=True):
        snapshot.assign(state)
    saved_time = example.sim_time
    saved_ik = None if args.scene == "twist" else wp.clone(example.ik_q)
    saved_t = wp.clone(example.t) if args.scene == "twist" else None
    cases = {
        "fast": {},
        "jacobi_half": {"particle_jacobi_relaxation": 0.5},
        "gs_cached": {"particle_enable_batched_jacobi": False, "particle_chebyshev_spectral_radius": None},
        "gs20_cached": {
            "iterations": 20,
            "particle_enable_batched_jacobi": False,
            "particle_chebyshev_spectral_radius": None,
        },
        "gs_uncached": {
            "particle_enable_batched_jacobi": False,
            "particle_chebyshev_spectral_radius": None,
            "particle_enable_surface_cache": False,
            "particle_enable_truncation_cache": False,
        },
        "gs_contact_each": {
            "particle_enable_batched_jacobi": False,
            "particle_chebyshev_spectral_radius": None,
            "particle_collision_detection_interval": 1,
        },
        "gs_no_self_contact": {
            "particle_enable_batched_jacobi": False,
            "particle_chebyshev_spectral_radius": None,
            "particle_enable_self_contact": False,
        },
        "gs_frozen_ik": {
            "particle_enable_batched_jacobi": False,
            "particle_chebyshev_spectral_radius": None,
        },
    }
    if args.scene == "twist":
        del cases["gs_frozen_ik"]
    if args.cases:
        unknown = set(args.cases) - cases.keys()
        if unknown:
            parser.error(f"Unsupported cases for {args.scene}: {sorted(unknown)}")
        cases = {name: cases[name] for name in args.cases}
    ordinary_ik = getattr(example, "_solve_runtime_ik_frame", None)
    for label, overrides in cases.items():
        for state, snapshot in zip((example.state_0, example.state_1), snapshots, strict=True):
            state.assign(snapshot)
        example.sim_time = saved_time
        if saved_ik is not None:
            example.ik_q.assign(saved_ik)
            example.frame_index = 900
            example._solve_runtime_ik_frame = ordinary_ik
            if label == "gs_frozen_ik":
                example.frame_q_start.assign(example.state_0.joint_q)
                example.frame_q_end.assign(example.state_0.joint_q)
                example._solve_runtime_ik_frame = lambda: None
        else:
            example.t.assign(saved_t)
        kwargs = dict(constructor)
        kwargs["vbd_options"] = dict(constructor["vbd_options"], **overrides)
        example.solver = newton.solvers.SolverMJVBDV2(example.model, **kwargs)
        example.contacts = example.solver.contacts
        example.capture()
        speeds, maxima, increments, changes = [], [], [], []
        previous_q = example.state_0.particle_q.numpy()
        previous_delta = None
        wp.synchronize_device(example.model.device)
        start = time.perf_counter()
        for frame in range(args.frames):
            example.step()
            q = example.state_0.particle_q.numpy()
            v = example.state_0.particle_qd.numpy()
            delta = q - previous_q
            if frame >= args.frames // 2:
                speeds.append(float(np.sqrt(np.mean(np.sum(v * v, axis=1)))))
                maxima.append(float(np.linalg.norm(v, axis=1).max()))
                increments.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
                changes.append(float(np.sqrt(np.mean(np.sum((delta - previous_delta) ** 2, axis=1)))))
            previous_delta, previous_q = delta, q
        print(
            json.dumps(
                {
                    "scene": args.scene,
                    "case": label,
                    "rms_speed": np.mean(speeds),
                    "max_speed": np.mean(maxima),
                    "frame_motion_rms": np.mean(increments),
                    "frame_second_difference_rms": np.mean(changes),
                    "ms_including_readback": 1000 * (time.perf_counter() - start) / args.frames,
                }
            ),
            flush=True,
        )
        gc.collect()


if __name__ == "__main__":
    main()

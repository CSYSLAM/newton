# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test one ordinary final sweep without changing the scene or contact policy."""

import argparse
import json
import time
from unittest.mock import patch

import numpy as np
import warp as wp

import newton.examples
import newton.viewer
from newton._src.solvers.mjvbd_v2.vbd_soft.solver_vbd import SolverVBD
from newton.examples.mjvbdv2 import example_mjvbd_v2_tshirt_fold as scene
from newton.solvers import SolverMJVBDV2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fast", "final_gs", "ordinary30", "option", "graph7"), required=True)
    parser.add_argument("--frames", type=int, default=900)
    args = parser.parse_args()
    if args.frames < 92:
        parser.error("--frames must be at least 92")
    wp.config.log_level = wp.LOG_WARNING
    wp.set_device("cuda:0")
    original_init = SolverMJVBDV2.__init__
    original_iteration = SolverVBD._solve_particle_iteration

    def configure(self, model, **kwargs):
        kwargs["vbd_options"] = dict(
            kwargs["vbd_options"], particle_jacobi_polish_iterations=int(args.mode == "option")
        )
        if args.mode == "ordinary30":
            kwargs["vbd_preset"] = None
            kwargs["vbd_options"] = dict(kwargs["vbd_options"], iterations=30)
        if args.mode == "graph7":
            kwargs["vbd_options"].update(
                iterations=7,
                particle_enable_multilevel_correction=True,
                particle_multilevel_operator="graph",
                particle_multilevel_checkpoints=(6,),
                particle_multilevel_relaxation=0.5,
                particle_multilevel_min_residual_reduction=1.0e-4,
                particle_multilevel_max_clamp_fraction=0.5,
                particle_chebyshev_polish_iterations=1,
            )
        original_init(self, model, **kwargs)

    def solve(self, state_in, state_out, contacts, dt, iter_num):
        if args.mode != "final_gs" or not self.particle_enable_batched_jacobi or iter_num != self.iterations - 1:
            return original_iteration(self, state_in, state_out, contacts, dt, iter_num)
        self.particle_enable_batched_jacobi = False
        enabled = self.particle_chebyshev_enabled
        self.particle_chebyshev_enabled = False
        try:
            return original_iteration(self, state_in, state_out, contacts, dt, iter_num)
        finally:
            self.particle_enable_batched_jacobi = True
            self.particle_chebyshev_enabled = enabled

    with (
        patch.object(SolverMJVBDV2, "__init__", configure),
        patch.object(SolverVBD, "_solve_particle_iteration", solve),
    ):
        example = scene.Example(newton.viewer.ViewerNull(), newton.examples.default_args(scene.Example.create_parser()))
        previous_q = previous_delta = None
        speed, peak, jitter = [], [], []
        for frame in range(args.frames):
            example.step()
            if frame >= args.frames - 92:
                q = example.state_0.particle_q.numpy()
                v = example.state_0.particle_qd.numpy()
                if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
                    raise ValueError(f"Nonfinite state at frame {frame + 1}")
                if previous_q is not None:
                    delta = q - previous_q
                    if previous_delta is not None:
                        speed.append(float(np.sqrt(np.mean(np.sum(v**2, axis=1)))))
                        peak.append(float(np.linalg.norm(v, axis=1).max()))
                        jitter.append(float(np.sqrt(np.mean(np.sum((delta - previous_delta) ** 2, axis=1)))))
                    previous_delta = delta
                previous_q = q
        example.test_final()
        wp.synchronize_device(example.model.device)
        start = time.perf_counter()
        for _ in range(120):
            example.step()
        wp.synchronize_device(example.model.device)
        elapsed = time.perf_counter() - start
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "frames": args.frames,
                    "rms_speed": np.mean(speed),
                    "mean_max_speed": np.mean(peak),
                    "second_difference_rms": np.mean(jitter),
                    "end_frame_ms": elapsed * 1000 / 120,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()

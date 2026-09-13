# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Observe the direct-file popcorn entry point without changing physics options."""

import argparse
import json
import runpy
import sys
from pathlib import Path

import numpy as np

import newton.examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=2880)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    original_run = newton.examples.run

    def observe(example, options):
        original_running = example.viewer.is_running
        original_step = example.step
        frames = 0

        def report():
            print(
                json.dumps(
                    {
                        "time": example.sim_time,
                        "trajectory_time": example.trajectory_time,
                        "cup_slip_mm": 1000 * getattr(example, "cup_grip_slip", 0.0),
                        "force_N": example.grasp_force_filtered.tolist(),
                        "finger_offset_deg": np.degrees(example.grasp_joint_offset).tolist(),
                    }
                ),
                flush=True,
            )

        def step():
            nonlocal frames
            original_step()
            frames += 1
            if frames % 60 == 0:
                report()

        example.step = step
        example.viewer.is_running = lambda: frames < args.frames and original_running()
        try:
            original_run(example, options)
            example.test_final()
        finally:
            report()

    newton.examples.run = observe
    path = Path(__file__).resolve().parents[3] / "newton/examples/mjvbdv2/example_mjvbd_v2_popcorn.py"
    sys.argv = [str(path)] + (["--headless"] if args.headless else [])
    try:
        runpy.run_path(str(path), run_name="__main__")
    finally:
        newton.examples.run = original_run


if __name__ == "__main__":
    main()

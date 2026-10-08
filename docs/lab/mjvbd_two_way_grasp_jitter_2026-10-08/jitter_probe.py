# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Measure per-substep finger jitter of the two-way W1 squeeze grasp.

Arguments after the script name are forwarded to the example parser, so the
remedies can be compared directly, for example::

    uv run docs/lab/mjvbd_two_way_grasp_jitter_2026-10-08/jitter_probe.py
    uv run docs/lab/mjvbd_two_way_grasp_jitter_2026-10-08/jitter_probe.py --finger-damping 0 --proxy-mass-scale 1

The probe disables CUDA graph capture so it can read the finger coordinates
after every substep, and reports statistics over the lifted hold
(4.6 s to 7.3 s), when both fingers are commanded fully shut.
"""

import json
import sys

import numpy as np

import newton.viewer
from newton.examples.mjvbdv2 import example_mjvbd_v2_w1_squeeze_grasp as squeeze

HOLD = (4.6, 7.3)
FRAMES = 7 * 60 + 30


def measure(extra_args):
    args = squeeze.Example.create_parser().parse_args(["--viewer", "null", "--no-cuda-graph", *extra_args])
    example = squeeze.Example(newton.viewer.ViewerNull(num_frames=FRAMES), args)
    coords = [side[0] for side in example.finger_coords]
    samples = []
    solver_step = example.solver.step

    def recording_step(state_in, state_out, control, contacts, dt):
        solver_step(state_in, state_out, control, contacts, dt)
        samples.append((example.sim_time, *state_out.joint_q.numpy()[coords]))

    example.solver.step = recording_step
    for _ in range(FRAMES):
        example.step()

    data = np.asarray(samples)
    held = data[(data[:, 0] >= HOLD[0]) & (data[:, 0] <= HOLD[1])]
    substep_rate = 1.0 / example.sim_dt
    result = {"args": list(extra_args)}
    for column, side in ((1, "rigid"), (2, "soft")):
        q = held[:, column] * 1.0e3
        steps = np.diff(q)
        reversals = int(np.sum(np.sign(steps[1:]) * np.sign(steps[:-1]) < 0))
        result[side] = {
            "mean_mm": float(q.mean()),
            "peak_to_peak_mm": float(np.ptp(q)),
            "std_mm": float(q.std()),
            "reversals_per_s": reversals / (len(q) / substep_rate),
        }
    return result


if __name__ == "__main__":
    print(json.dumps(measure(sys.argv[1:]), indent=2))

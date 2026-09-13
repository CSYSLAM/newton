# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Trace payload escape without modifying state, controls or acceptance."""

import json

import numpy as np

from newton.examples.mjvbdv2 import example_mjvbd_v2_popcorn as scene


def install_payload_trace(example):
    original = example._measure
    previous = np.zeros(len(example.popcorn_bodies), dtype=bool)
    last_time = -1.0

    def measure():
        nonlocal previous, last_time
        original()
        if not 10 < example.trajectory_time < 28:
            return
        q = example.state_0.particle_q.numpy()
        bodies = example.state_0.body_q.numpy()
        grains = bodies[example.popcorn_bodies, :3]
        inside = scene._inside_cup(grains, q, example.cup_faces, example.rim_indices)
        exits = previous & ~inside
        rim = q[example.rim_indices].mean(axis=0)
        if exits.any() or example.trajectory_time - last_time >= 0.5:
            print(
                json.dumps(
                    {
                        "payload_trace_time": example.trajectory_time,
                        "in_scoop": example.scoop_count,
                        "inside": int(inside.sum()),
                        "exit_relative_to_rim": (grains[exits] - rim).tolist(),
                        "loaded_outside_relative_to_rim": (grains[example.lifted_payload & ~inside] - rim).tolist(),
                    }
                ),
                flush=True,
            )
            last_time = example.trajectory_time
        previous = inside

    example._measure = measure

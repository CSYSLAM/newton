# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reuse read-only physical-state downloads within the unchanged controller phase."""

from unittest import mock

import warp as wp


def install_state_read_cache(example):
    original_step = example.step
    original_measure = example._measure
    original_simulate = example._simulate
    original_numpy = wp.array.numpy
    snapshots = {}
    targets = set()

    def numpy(array):
        key = (array.ptr, array.shape)
        if key not in targets:
            return original_numpy(array)
        if key not in snapshots:
            value = original_numpy(array)
            value.flags.writeable = False
            snapshots[key] = value
        return snapshots[key]

    def simulate():
        snapshots.clear()
        try:
            return original_simulate()
        finally:
            snapshots.clear()

    def measure():
        # A graph may have just advanced physics, even when _simulate was not
        # invoked in Python. Measurements must always see the new state.
        snapshots.clear()
        return original_measure()

    def step():
        snapshots.clear()
        targets.clear()
        targets.update((array.ptr, array.shape) for array in (example.state_0.body_q, example.state_0.particle_q))
        try:
            with mock.patch.object(wp.array, "numpy", numpy):
                return original_step()
        finally:
            snapshots.clear()
            targets.clear()

    example.step = step
    example._simulate = simulate
    example._measure = measure

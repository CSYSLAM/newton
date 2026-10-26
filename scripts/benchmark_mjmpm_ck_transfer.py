# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Isolated P2G stencil probe for the SIGGRAPH 2025 compact MPM kernel.

This measures transfer cost and conservation only. CK-MPM requires two
staggered grids; this script does not substitute for an implicit MJMPM step.

    uv run --extra examples scripts/benchmark_mjmpm_ck_transfer.py
"""

import argparse
import json

import numpy as np
import warp as wp


@wp.func
def _linear(f: float, i: int) -> float:
    return 1.0 - f if i == 0 else f


@wp.func
def _quadratic(f: float, i: int) -> float:
    if i == 0:
        return 0.5 * (1.5 - f) * (1.5 - f)
    if i == 1:
        return 0.75 - (f - 1.0) * (f - 1.0)
    return 0.5 * (f - 0.5) * (f - 0.5)


@wp.func
def _compact(f: float, i: int) -> float:
    # Liu, Wang, and Li, CK-MPM (SIGGRAPH 2025), author reference code.
    left = 1.0 - f + wp.sin(6.283185307179586 * f) / 6.283185307179586
    return left if i == 0 else 1.0 - left


@wp.kernel
def _p2g(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    mass_grid: wp.array[float],
    momentum_grid: wp.array[wp.vec3],
    n: int,
    inv_h: float,
    method: int,
):
    p = wp.tid()
    pos = positions[p] * inv_h
    vel = velocities[p]
    grid_size = n * n * n
    grid_count = 2 if method == 2 else 1
    for grid in range(grid_count):
        shift = 0.25 * float(2 * grid - 1) if method == 2 else 0.0
        x = pos - wp.vec3(shift, shift, shift)
        base = wp.vec3i(int(wp.floor(x[0])), int(wp.floor(x[1])), int(wp.floor(x[2])))
        if method == 1:
            base = wp.vec3i(int(wp.floor(x[0] - 0.5)), int(wp.floor(x[1] - 0.5)), int(wp.floor(x[2] - 0.5)))
        f = x - wp.vec3(float(base[0]), float(base[1]), float(base[2]))
        width = 3 if method == 1 else 2
        for i in range(width):
            for j in range(width):
                for k in range(width):
                    if method == 0:
                        weight = _linear(f[0], i) * _linear(f[1], j) * _linear(f[2], k)
                    elif method == 1:
                        weight = _quadratic(f[0], i) * _quadratic(f[1], j) * _quadratic(f[2], k)
                    else:
                        weight = _compact(f[0], i) * _compact(f[1], j) * _compact(f[2], k)
                    node = base + wp.vec3i(i, j, k)
                    index = grid * grid_size + (node[2] * n + node[1]) * n + node[0]
                    wp.atomic_add(mass_grid, index, weight)
                    wp.atomic_add(momentum_grid, index, weight * vel)


def _timed_transfer(positions, velocities, mass_grid, momentum_grid, n, method, repeat):
    device = positions.device
    start = wp.Event(device, enable_timing=True)
    end = wp.Event(device, enable_timing=True)
    stream = wp.get_stream(device)
    times = []
    for iteration in range(repeat + 3):
        mass_grid.zero_()
        momentum_grid.zero_()
        stream.record_event(start)
        wp.launch(_p2g, len(positions), (positions, velocities, mass_grid, momentum_grid, n, float(n), method))
        stream.record_event(end)
        if iteration >= 3:
            times.append(wp.get_event_elapsed_time(start, end))
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=40000)
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--repeat", type=int, default=20)
    options = parser.parse_args()
    rng = np.random.default_rng(2026)
    particle_positions = rng.uniform(0.2, 0.8, (options.particles, 3)).astype(np.float32)
    particle_velocities = rng.normal(size=(options.particles, 3)).astype(np.float32)
    device = wp.get_device("cuda:0")
    positions = wp.array(particle_positions, dtype=wp.vec3, device=device)
    velocities = wp.array(particle_velocities, dtype=wp.vec3, device=device)
    mass_grid = wp.zeros(2 * options.grid**3, dtype=float, device=device)
    momentum_grid = wp.zeros(2 * options.grid**3, dtype=wp.vec3, device=device)
    expected_momentum = particle_velocities.sum(axis=0, dtype=np.float64)
    # For CK, each grid alone lacks first-moment consistency; the two-grid
    # average restores it. Check that property independently of mass sums.
    sample = particle_positions[: min(options.particles, 1024)].astype(np.float64) * options.grid
    ck_centroids = []
    for shift in (-0.25, 0.25):
        local = sample - shift
        base = np.floor(local)
        fractional = local - base
        right_weight = fractional - np.sin(2.0 * np.pi * fractional) / (2.0 * np.pi)
        ck_centroids.append(base + right_weight + shift)
    ck_linear_error = float(np.max(np.abs(0.5 * (ck_centroids[0] + ck_centroids[1]) - sample)))
    if ck_linear_error > 1.0e-12:
        raise AssertionError(f"CK dual-grid first moment failed: {ck_linear_error}")
    results = {}
    for method, name, nodes in ((0, "Q1", 8), (1, "B2", 27), (2, "CK", 16)):
        elapsed = _timed_transfer(positions, velocities, mass_grid, momentum_grid, options.grid, method, options.repeat)
        masses = mass_grid.numpy().reshape(2, -1)
        momenta = momentum_grid.numpy().reshape(2, -1, 3)
        grid_count = 2 if method == 2 else 1
        mass_error = float(
            max(abs(masses[g].sum(dtype=np.float64) / options.particles - 1.0) for g in range(grid_count))
        )
        momentum_error = float(
            max(np.linalg.norm(momenta[g].sum(axis=0, dtype=np.float64) - expected_momentum) for g in range(grid_count))
            / options.particles
        )
        if mass_error > 1.0e-5 or momentum_error > 1.0e-5:
            raise AssertionError(f"{name} transfer failed conservation: {mass_error}, {momentum_error}")
        results[name] = {
            "nodes_per_particle": nodes,
            "median_p2g_ms": elapsed,
            "relative_mass_error": mass_error,
            "momentum_error_per_particle": momentum_error,
        }
    print(
        json.dumps(
            {
                "particles": options.particles,
                "grid": options.grid,
                "ck_first_moment_error": ck_linear_error,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scenes for compare_backends.py. Each scene yields the state after every step."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import warp as wp

import newton
from newton.solvers import coupled_high_accuracy as s3d

SCENES: dict[str, Callable[[int, dict[str, str]], Iterator[Any]]] = {}


def scene(function):
    SCENES[function.__name__] = function
    return function


def _framework(options: dict[str, str]):
    return s3d.NonlinearSolver.PD_NEWTON if options.get("framework") == "pd" else s3d.NonlinearSolver.NEWTON


def _run(model, solver, steps: int, dt: float = 1.0 / 240.0) -> Iterator[Any]:
    state_in, state_out = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        state_in.clear_forces()
        solver.step(state_in, state_out, control, None, dt)
        state_in, state_out = state_out, state_in
        yield state_in


@scene
def particles(steps: int, options: dict[str, str]) -> Iterator[Any]:
    """Free particles of different masses (one inactive) under gravity, in two replicated worlds."""
    builder = newton.ModelBuilder()
    for world in range(2):
        builder.begin_world()
        for index, mass in enumerate((1.0, 0.5, 0.0, 2.0)):
            builder.add_particle(pos=wp.vec3(float(index), float(world), 1.0), vel=wp.vec3(0.1, 0.0, 0.5), mass=mass)
        builder.end_world()
    model = builder.finalize(device="cuda:0")
    solver = s3d.SolverCoupledHighAccuracy(model)
    solver.config.nonlinear_solver = _framework(options)
    solver.config.nonlinear_iterations = int(options.get("iterations", 3))
    solver.config.linear_iterations = int(options.get("linear", 20))
    yield from _run(model, solver, steps)


@scene
def springs(steps: int, options: dict[str, str]) -> Iterator[Any]:
    """A pinned 6x6 particle grid with structural and shear springs (options: framework=pd, worlds=N)."""
    builder = newton.ModelBuilder()
    worlds = int(options.get("worlds", 1))
    n = 6
    for world in range(worlds):
        builder.begin_world()
        base = builder.particle_count
        for j in range(n):
            for i in range(n):
                mass = 0.0 if j == n - 1 and i in (0, n - 1) else 0.05
                builder.add_particle(
                    pos=wp.vec3(0.1 * i, 0.0, 1.0 + 0.1 * j + 0.3 * world), vel=wp.vec3(0.0), mass=mass
                )
        for j in range(n):
            for i in range(n):
                index = base + j * n + i
                if i + 1 < n:
                    builder.add_spring(index, index + 1, 500.0, 0.0, 0)
                if j + 1 < n:
                    builder.add_spring(index, index + n, 500.0, 0.0, 0)
                if i + 1 < n and j + 1 < n:
                    builder.add_spring(index, index + n + 1, 200.0, 0.0, 0)
        builder.end_world()
    model = builder.finalize(device="cuda:0")
    solver = s3d.SolverCoupledHighAccuracy(model)
    solver.config.nonlinear_solver = _framework(options)
    solver.config.nonlinear_iterations = int(options.get("iterations", 4))
    solver.config.linear_iterations = int(options.get("linear", 50))
    yield from _run(model, solver, steps)


if __name__ == "__main__":
    print(sorted(SCENES))
    np.set_printoptions(precision=4)

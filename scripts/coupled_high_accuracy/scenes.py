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


def _cloth_model(options: dict[str, str]):
    from newton.solvers import style3d  # noqa: PLC0415

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    for world in range(int(options.get("worlds", 1))):
        builder.begin_world()
        style3d.add_cloth_grid(
            builder,
            pos=wp.vec3(0.0, 0.0, 1.0 + 0.2 * world),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.3),
            dim_x=8,
            dim_y=6,
            cell_x=0.05,
            cell_y=0.05,
            vel=wp.vec3(0.0),
            mass=0.02,
            fix_left=True,
            particle_radius=0.01,
            tri_aniso_ke=wp.vec3(2.0e3, 1.0e3, 3.0e2),
            edge_aniso_ke=wp.vec3(float(options.get("bend", 1.0e-3))),
            edge_kd=float(options.get("bend_damping", 0.0)),
        )
        builder.end_world()
    if "springs" in options:
        builder.add_spring(1, 9, 100.0, 0.0, 0)
    model = builder.finalize(device="cuda:0")
    drag = float(options.get("drag", 0.0))
    if drag:
        materials = model.tri_materials.numpy()
        materials[:, 3] = drag
        model.tri_materials.assign(materials)
    return model


@scene
def cloth(steps: int, options: dict[str, str]) -> Iterator[Any]:
    """Pinned Style3D cloth grid (options: framework=pd, bend=, bend_damping=, drag=, springs=1, worlds=N)."""
    model = _cloth_model(options)
    solver = s3d.SolverCoupledHighAccuracy(model)
    solver.config.nonlinear_solver = _framework(options)
    solver.config.nonlinear_iterations = int(options.get("iterations", 4))
    solver.config.linear_iterations = int(options.get("linear", 50))
    yield from _run(model, solver, steps)


@scene
def tetrahedra(steps: int, options: dict[str, str]) -> Iterator[Any]:
    """A falling-free tetrahedral soft cube pinned on one face (options: framework=pd, damping=, worlds=N)."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    for world in range(int(options.get("worlds", 1))):
        builder.begin_world()
        builder.add_soft_grid(
            pos=wp.vec3(0.0, 0.5 * world, 1.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.2),
            vel=wp.vec3(0.0),
            dim_x=3,
            dim_y=2,
            dim_z=2,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=300.0,
            k_mu=2.0e4,
            k_lambda=8.0e4,
            k_damp=float(options.get("damping", 0.0)),
            fix_left=True,
        )
        builder.end_world()
    model = builder.finalize(device="cuda:0")
    solver = s3d.SolverCoupledHighAccuracy(model)
    solver.config.nonlinear_solver = _framework(options)
    solver.config.nonlinear_iterations = int(options.get("iterations", 4))
    solver.config.linear_iterations = int(options.get("linear", 50))
    yield from _run(model, solver, steps)

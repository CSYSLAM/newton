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
# Set by compare_backends.py --replay: recorded states every step restarts from (one-step comparison).
REPLAY: dict[str, np.ndarray] | None = None


def _restart(state, step: int) -> None:
    """Overwrite the state with the replayed state before step `step` (the recorded state after step - 1)."""
    if REPLAY is None or step == 0:
        return
    for field in ("particle_q", "particle_qd", "body_q", "body_qd"):
        array = getattr(state, field, None)
        if array is not None and field in REPLAY and REPLAY[field].size:
            array.assign(REPLAY[field][step - 1])


def scene(function):
    SCENES[function.__name__] = function
    return function


def _framework(options: dict[str, str]):
    return s3d.NonlinearSolver.PD_NEWTON if options.get("framework") == "pd" else s3d.NonlinearSolver.NEWTON


def _run(model, solver, steps: int, dt: float = 1.0 / 240.0) -> Iterator[Any]:
    state_in, state_out = model.state(), model.state()
    control = model.control()
    for step in range(steps):
        _restart(state_in, step)
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


def _body_solver(model, options: dict[str, str]):
    solver = s3d.SolverCoupledHighAccuracy(model)
    solver.config.nonlinear_solver = _framework(options)
    solver.config.nonlinear_iterations = int(options.get("iterations", 4))
    solver.config.linear_iterations = int(options.get("linear", 50))
    rigidity = options.get("rigidity")
    if rigidity:
        solver.config.affine_body_rigidity_model = getattr(s3d.AffineBodyRigidityModel, rigidity.upper())
    if options.get("body_block"):
        solver.config.affine_body_block_preconditioner = True
    if options.get("penalty"):
        solver.config.joint_constraint_mode = s3d.JointConstraintMode.PENALTY
    return solver


def _run_bodies(model, solver, steps: int, dt: float = 1.0 / 240.0, control=None) -> Iterator[Any]:
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    solver.sync_affine_body_state(state_in)
    control = control or model.control()
    for step in range(steps):
        _restart(state_in, step)
        if REPLAY is not None and step:
            solver.sync_affine_body_state(state_in)
        state_in.clear_forces()
        solver.step(state_in, state_out, control, None, dt)
        state_in, state_out = state_out, state_in
        yield state_in


@scene
def spinning_body(steps: int, options: dict[str, str]) -> Iterator[Any]:
    """A free affine body with anisotropic inertia spinning under gravity (options: framework, rigidity,
    body_block=1)."""
    builder = s3d.ModelBuilder(up_axis=newton.Axis.Z)
    inertia = wp.mat33(0.04, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.07)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), mass=1.5, inertia=inertia, lock_inertia=True
    )
    model = builder.finalize(device="cuda:0")
    qd = model.body_qd.numpy()
    qd[body] = [0.1, 0.0, 0.2, 0.3, 5.0, 0.2]
    model.body_qd.assign(qd)
    model.joint_qd.assign(qd[body])
    yield from _run_bodies(model, _body_solver(model, options), steps)


@scene
def pendulum(steps: int, options: dict[str, str]) -> Iterator[Any]:
    """A three-link revolute pendulum, the first joint driven to a target (options: framework, rigidity,
    body_block=1, penalty=1, prismatic=1, armature=, limit=1)."""
    builder = s3d.ModelBuilder(up_axis=newton.Axis.Z)
    inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.002)
    parent = -1
    armature = float(options.get("armature", 0.0))
    for link in range(3):
        body = builder.add_link(
            xform=wp.transform(wp.vec3(0.2 * link + 0.1, 0.0, 1.0), wp.quat_identity()),
            mass=0.5,
            inertia=inertia,
            lock_inertia=True,
        )
        parent_xform = wp.transform(
            wp.vec3(0.2 * link, 0.0, 1.0) if parent < 0 else wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()
        )
        child_xform = wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity())
        if options.get("prismatic") and link == 0:
            builder.add_joint_prismatic(
                parent,
                body,
                axis=(1.0, 0.0, 0.0),
                parent_xform=parent_xform,
                child_xform=child_xform,
                target_q=0.05,
                target_ke=200.0,
                target_kd=5.0,
                armature=armature,
                actuator_mode=newton.JointTargetMode.POSITION,
            )
        else:
            kwargs = {}
            if link == 0:
                kwargs = {
                    "target_q": 0.4,
                    "target_ke": 50.0,
                    "target_kd": 2.0,
                    "armature": armature,
                    "actuator_mode": newton.JointTargetMode.POSITION,
                }
            if options.get("limit"):
                kwargs.update(limit_lower=-0.2, limit_upper=0.2, limit_ke=100.0, limit_kd=1.0)
            builder.add_joint_revolute(
                parent, body, axis=(0.0, 1.0, 0.0), parent_xform=parent_xform, child_xform=child_xform, **kwargs
            )
        parent = body
    builder.add_articulation(list(range(builder.joint_count)))
    model = builder.finalize(device="cuda:0")
    yield from _run_bodies(model, _body_solver(model, options), steps)


def _run_contacts(model, solver, steps: int, dt: float = 1.0 / 240.0, **pipeline_options) -> Iterator[Any]:
    state_in, state_out = model.state(), model.state()
    if model.joint_count:
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_in)
    if model.body_count:
        solver.sync_affine_body_state(state_in)
    pipeline = newton.CollisionPipeline(model, **pipeline_options)
    contacts = pipeline.contacts()
    control = model.control()
    for step in range(steps):
        _restart(state_in, step)
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, dt)
        state_in, state_out = state_out, state_in
        yield state_in


@scene
def cloth_ground(steps: int, options: dict[str, str]) -> Iterator[Any]:
    """A cloth grid dropped on the ground plane and a static sphere (options: framework, friction=, diagonal=1)."""
    from newton.solvers import style3d  # noqa: PLC0415

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    builder.add_ground_plane()
    builder.add_shape_sphere(body=-1, xform=wp.transform(wp.vec3(0.2, 0.15, 0.05), wp.quat_identity()), radius=0.06)
    style3d.add_cloth_grid(
        builder,
        pos=wp.vec3(0.0, 0.0, 0.12),
        rot=wp.quat_identity(),
        dim_x=8,
        dim_y=6,
        cell_x=0.05,
        cell_y=0.05,
        vel=wp.vec3(0.1, 0.0, -0.5),
        mass=0.02,
        particle_radius=0.01,
        tri_aniso_ke=wp.vec3(2.0e3, 1.0e3, 3.0e2),
        edge_aniso_ke=wp.vec3(1.0e-3),
        edge_kd=0.0,
    )
    model = builder.finalize(device="cuda:0")
    model.soft_contact_ke = 2.0e3
    model.soft_contact_kd = 5.0
    model.soft_contact_mu = float(options.get("friction", 0.5))
    solver = s3d.SolverCoupledHighAccuracy(model)
    solver.config.nonlinear_solver = _framework(options)
    solver.config.nonlinear_iterations = int(options.get("iterations", 4))
    solver.config.linear_iterations = int(options.get("linear", 50))
    if options.get("diagonal"):
        solver.config.enable_collision_hessian_non_diagonal = False
    if options.get("reference"):
        solver.config.reference_linear_schedule = True
    yield from _run_contacts(model, solver, steps, enable_rigid_soft_full_surface_contact=bool(options.get("surface")))


@scene
def particles_on_body(steps: int, options: dict[str, str]) -> Iterator[Any]:
    """Particles hitting a sphere and a box attached to a free affine body, without gravity (options: framework)."""
    builder = s3d.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
    _shape_friction(builder, options)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_sphere(body=body, radius=0.1)
    builder.add_shape_box(
        body=body, xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()), hx=0.05, hy=0.05, hz=0.05
    )
    for index in range(6):
        builder.add_particle(
            pos=wp.vec3(-0.05 + 0.06 * index, 0.01 * index, 0.72), vel=wp.vec3(0.0, 0.0, -1.0), mass=0.05, radius=0.02
        )
    model = builder.finalize(device="cuda:0")
    model.soft_contact_ke = 5.0e3
    model.soft_contact_kd = 2.0
    model.soft_contact_mu = 0.3
    yield from _run_contacts(model, _body_solver(model, options), steps)


def _shape_friction(builder, options: dict[str, str]) -> None:
    rolling = float(options.get("rolling", 0.0))
    builder.default_shape_cfg.mu_rolling = rolling
    builder.default_shape_cfg.mu_torsional = rolling


@scene
def boxes_on_ground(steps: int, options: dict[str, str]) -> Iterator[Any]:
    """Two boxes and a sphere body dropped on the ground and on each other (options: framework, rolling=,
    epsilon=, body_block=1, diagonal=1)."""
    builder = s3d.ModelBuilder(up_axis=newton.Axis.Z)
    _shape_friction(builder, options)
    builder.default_shape_cfg.ke = 2.0e4
    builder.default_shape_cfg.kd = 20.0
    builder.default_shape_cfg.mu = 0.6
    builder.add_ground_plane()
    rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 1.0, 0.0)), 0.3)
    for index, height in enumerate((0.12, 0.32)):
        body = builder.add_body(xform=wp.transform(wp.vec3(0.03 * index, 0.0, height), rotation))
        builder.add_shape_box(body=body, hx=0.08, hy=0.06, hz=0.05)
    ball = builder.add_body(xform=wp.transform(wp.vec3(0.3, 0.0, 0.2), wp.quat_identity()))
    builder.add_shape_sphere(body=ball, radius=0.06)
    model = builder.finalize(device="cuda:0")
    if options.get("perturb"):
        # Shift every body by a tiny amount to measure how sensitive the scene is to round-off.
        q = model.joint_q.numpy()
        q[2::7] += float(options["perturb"])
        model.joint_q.assign(q)
    qd = model.body_qd.numpy()
    qd[ball] = [0.0, 0.0, 0.0, -0.5, 0.0, 0.0]
    model.body_qd.assign(qd)
    model.joint_qd.assign(qd.reshape(-1))
    solver = _body_solver(model, options)
    if options.get("epsilon"):
        solver.config.rigid_contact_friction_epsilon = float(options["epsilon"])
    if options.get("diagonal"):
        solver.config.enable_collision_hessian_non_diagonal = False
    pipeline_options = {}
    if options.get("hard"):
        solver.config.rigid_contact_hard = True
        solver.config.rigid_contact_beta = float(options.get("beta", 0.0))
        solver.config.rigid_contact_sliding_tangent = bool(options.get("sliding"))
        if options.get("history"):
            solver.config.rigid_contact_history = True
            pipeline_options["contact_matching"] = "sticky"
    yield from _run_contacts(model, solver, steps, **pipeline_options)


@scene
def cloth_on_box(steps: int, options: dict[str, str]) -> Iterator[Any]:
    """A cloth draping over a free box body resting on the ground, with full-surface contacts (options: framework,
    rolling=, hard=1, reference=1, point=1 for particle-only contacts)."""
    from newton.solvers import style3d  # noqa: PLC0415

    builder = s3d.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    _shape_friction(builder, options)
    builder.default_shape_cfg.ke = 1.0e4
    builder.default_shape_cfg.kd = 10.0
    builder.add_ground_plane()
    box = builder.add_body(xform=wp.transform(wp.vec3(0.18, 0.12, 0.06), wp.quat_identity()))
    builder.add_shape_box(body=box, hx=0.08, hy=0.06, hz=0.05)
    style3d.add_cloth_grid(
        builder,
        pos=wp.vec3(0.0, 0.0, 0.16),
        rot=wp.quat_identity(),
        dim_x=8,
        dim_y=6,
        cell_x=0.05,
        cell_y=0.05,
        vel=wp.vec3(0.0, 0.0, -0.3),
        mass=0.02,
        particle_radius=0.01,
        tri_aniso_ke=wp.vec3(2.0e3, 1.0e3, 3.0e2),
        edge_aniso_ke=wp.vec3(1.0e-3),
        edge_kd=0.0,
    )
    model = builder.finalize(device="cuda:0")
    model.soft_contact_ke = 2.0e3
    model.soft_contact_kd = 5.0
    model.soft_contact_mu = 0.5
    solver = _body_solver(model, options)
    if options.get("hard"):
        solver.config.rigid_contact_hard = True
    if options.get("reference"):
        solver.config.reference_linear_schedule = True
    yield from _run_contacts(model, solver, steps, enable_rigid_soft_full_surface_contact=not options.get("point"))


@scene
def cloth_self(steps: int, options: dict[str, str]) -> Iterator[Any]:
    """A cloth hanging from one edge that folds onto itself on the ground, with self-collision (options: framework,
    diagonal=1)."""
    from newton.solvers import style3d  # noqa: PLC0415

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverStyle3D.register_custom_attributes(builder)
    builder.add_ground_plane()
    style3d.add_cloth_grid(
        builder,
        pos=wp.vec3(0.0, 0.0, 0.25),
        rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 1.2),
        dim_x=10,
        dim_y=10,
        cell_x=0.03,
        cell_y=0.03,
        vel=wp.vec3(0.0, 0.3, -0.5),
        mass=0.01,
        particle_radius=0.006,
        tri_aniso_ke=wp.vec3(2.0e3, 1.0e3, 3.0e2),
        edge_aniso_ke=wp.vec3(1.0e-4),
        edge_kd=0.0,
    )
    model = builder.finalize(device="cuda:0")
    model.soft_contact_ke = 2.0e3
    model.soft_contact_kd = 5.0
    model.soft_contact_mu = 0.3
    solver = s3d.SolverCoupledHighAccuracy(model)
    solver.config.nonlinear_solver = _framework(options)
    solver.config.nonlinear_iterations = int(options.get("iterations", 4))
    solver.config.linear_iterations = int(options.get("linear", 50))
    solver.config.enable_self_collision = True
    solver.config.self_collision_radius = 0.006
    if options.get("diagonal"):
        solver.config.enable_collision_hessian_non_diagonal = False
    yield from _run_contacts(model, solver, steps)


@scene
def particle_pile(steps: int, options: dict[str, str]) -> Iterator[Any]:
    """Free particles poured into a pile on the ground, with particle-particle self-collision (options:
    framework, diagonal=1)."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()
    rng = np.random.default_rng(3)
    for _ in range(64):
        position = rng.uniform((-0.05, -0.05, 0.02), (0.05, 0.05, 0.2))
        builder.add_particle(pos=wp.vec3(*position), vel=wp.vec3(0.0), mass=0.01, radius=0.012)
    model = builder.finalize(device="cuda:0")
    model.soft_contact_ke = 2.0e3
    model.soft_contact_kd = 2.0
    model.soft_contact_mu = 0.3
    solver = s3d.SolverCoupledHighAccuracy(model)
    solver.config.nonlinear_solver = _framework(options)
    solver.config.nonlinear_iterations = int(options.get("iterations", 4))
    solver.config.linear_iterations = int(options.get("linear", 50))
    solver.config.enable_self_collision = True
    if options.get("diagonal"):
        solver.config.enable_collision_hessian_non_diagonal = False
    yield from _run_contacts(model, solver, steps)


@scene
def cable(steps: int, options: dict[str, str]) -> Iterator[Any]:
    """A curved rod of capsule bodies pinned at its root, swinging under gravity (options: framework, body_block=1,
    ramp=1 for penalty ramping, damping=)."""
    from newton._src.solvers.coupled_high_accuracy.cable_material import register_cable_attributes  # noqa: PLC0415

    builder = s3d.ModelBuilder(up_axis=newton.Axis.Z)
    register_cable_attributes(builder)
    points = [wp.vec3(0.04 * k, 0.01 * k * k, 1.0 - 0.002 * k * k) for k in range(9)]
    damping = float(options.get("damping", 0.0))
    bodies, _ = builder.add_rod(
        rod=newton.Rod(points, radius=0.01),
        stretch_stiffness=2000.0,
        shear_stiffness=1000.0,
        bend_stiffness=0.5,
        twist_stiffness=0.25,
        stretch_damping=damping,
        bend_damping=damping * 1.0e-3,
    )
    root = bodies[0]
    builder.body_mass[root] = 0.0
    builder.body_inv_mass[root] = 0.0
    builder.body_inertia[root] = wp.mat33(0.0)
    builder.body_inv_inertia[root] = wp.mat33(0.0)
    builder.color()
    model = builder.finalize(device="cuda:0")
    solver = _body_solver(model, options)
    if options.get("ramp"):
        solver.config.cable_linear_beta = 100.0
        solver.config.cable_angular_beta = 20.0
        solver.config.cable_penalty_gamma = 0.5
        solver.config.cable_linear_k_start = 10.0
        solver.config.cable_angular_k_start = 5.0
    yield from _run_bodies(model, solver, steps)

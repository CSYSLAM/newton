# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One-way coupling step function for multi-solver simulation.

Provides :func:`step_one_way_coupling` which orchestrates a rigid-body solver
(XPBD, Featherstone, or MuJoCo) and a VBD soft-body solver in a one-way
coupling pattern: the rigid body affects the soft body, but not vice versa.

Typical usage::

    from newton.examples.multiphysics.one_way_coupling import step_one_way_coupling

    for _ in range(substeps):
        state_0.clear_forces()
        state_1.clear_forces()
        step_one_way_coupling(
            rigid_solver=xpbd_solver,
            vbd_solver=vbd_solver,
            collision_pipeline=collision_pipeline,
            model=model,
            state_0=state_0,
            state_1=state_1,
            control=control,
            dt=sim_dt,
            real_particle_count=model.particle_count,
        )
        state_0, state_1 = state_1, state_0
"""

from __future__ import annotations

import warp as wp

# Per-device zero-gravity cache to avoid repeated allocation
_zero_gravity_cache: dict[str, wp.array] = {}


def _get_zero_gravity(model: wp.array) -> wp.array:
    """Return a zero-filled gravity array matching the model's device and shape."""
    device_str = str(model.gravity.device)
    if device_str not in _zero_gravity_cache:
        _zero_gravity_cache[device_str] = wp.zeros_like(model.gravity)
    return _zero_gravity_cache[device_str]


def step_one_way_coupling(
    rigid_solver,
    vbd_solver,
    collision_pipeline,
    model,
    state_0,
    state_1,
    control,
    dt,
    real_particle_count,
    contacts_rigid=None,
    contacts_soft=None,
):
    """Perform one sub-step of one-way coupled rigid + VBD simulation.

    The rigid body solver advances first (unaffected by soft bodies), then
    VBD advances with full collision so the soft body responds to the rigid
    body's new position.

    Supports three rigid solver types:

    - **XPBD** (``SolverXPBD``): Hides particles before collision so no soft
      contacts are generated for the rigid solver.
    - **Featherstone** (``SolverFeatherstone``): Same particle-hiding, plus
      zeroes gravity and shape-contact pairs during the rigid step so the
      articulated body integrates without soft-body interference.
    - **MuJoCo** (``SolverMuJoCo``): Same handling as Featherstone.

    Args:
        rigid_solver: The rigid-body solver instance (XPBD, Featherstone, or MuJoCo).
        vbd_solver: The VBD solver instance (must be created with
            ``integrate_with_external_rigid_solver=True``).
        collision_pipeline: :class:`newton.CollisionPipeline` instance.
        model: The :class:`newton.Model` being simulated.
        state_0: Input state.
        state_1: Output state.
        control: :class:`newton.Control` for the solvers.
        dt: Simulation time step [s].
        real_particle_count: The true ``model.particle_count`` to restore after
            temporarily hiding particles.
        contacts_rigid: :class:`newton.Contacts` for the rigid solver pass.
            If ``None``, a new one is created from the pipeline each call (slow).
        contacts_soft: :class:`newton.Contacts` for the VBD solver pass.
            If ``None``, a new one is created from the pipeline each call (slow).
    """
    # Lazily create contacts if not provided
    if contacts_rigid is None:
        contacts_rigid = collision_pipeline.contacts()
    if contacts_soft is None:
        contacts_soft = collision_pipeline.contacts()

    # Detect solver type for correct handling
    solver_name = type(rigid_solver).__name__
    is_articulated = solver_name in ("SolverFeatherstone", "SolverMuJoCo")

    # ---- Phase 1: Rigid solver step (no soft-body influence) ----

    # Hide particles so collision generates no soft contacts
    model.particle_count = 0

    if is_articulated:
        # Articulated solvers own rigid integration fully — suppress gravity
        # and shape contacts so they don't get confused by soft-body state
        saved_gravity = wp.clone(model.gravity)
        model.gravity.assign(_get_zero_gravity(model))
        saved_shape_pairs = model.shape_contact_pair_count
        model.shape_contact_pair_count = 0

    collision_pipeline.collide(state_0, contacts_rigid)

    if is_articulated:
        model.gravity.assign(saved_gravity)
        model.shape_contact_pair_count = saved_shape_pairs

    rigid_solver.step(state_0, state_1, control, contacts_rigid, dt)

    # ---- Phase 2: VBD solver step (cloth/soft sees rigid body) ----

    model.particle_count = real_particle_count
    state_0.particle_f.zero_()

    collision_pipeline.collide(state_0, contacts_soft)

    vbd_solver.step(state_0, state_1, control, contacts_soft, dt)

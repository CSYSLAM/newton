# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""VBD-owned block scheduling inside the MuJoCo--VBD trial workspace.

This module deliberately owns the timestep orchestration instead of calling
``SolverVBD.step()``.  The numerical particle and free-rigid-body primitives
remain the established VBD kernels while the surrounding joint solver chooses
when each block observes a new articulated trial pose.
"""

from __future__ import annotations

import warp as wp

from ...sim import Contacts, Control, State
from ..vbd import SolverVBD


class VBDOwnedBlock:
    """Run VBD-owned free-rigid and particle blocks without a nested timestep.

    The compact VBD model contains only free bodies and particles as dynamic
    entities.  Articulated links are retained as zero-inverse-mass collision
    boundaries, so this block never advances reduced-coordinate links.

    Args:
        solver: Local VBD solver providing the preallocated kernel workspace.
    """

    def __init__(self, solver: SolverVBD) -> None:
        self.solver = solver
        self._initialized = False
        self._snapshots: dict[str, object] = {}
        self._update_rigid_history = True

    def begin(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts, dt: float) -> None:
        """Initialize one VBD trial timestep without running its iteration loop."""
        solver = self.solver
        self._snapshot_persistent_state()
        solver._apply_module_options()
        refresh = solver._update_rigid_history
        self._update_rigid_history = refresh
        solver._update_rigid_history = True
        if control is None:
            control = solver.model.control(clone_variables=False)
        solver._initialize_rigid_bodies(state_in, control, contacts, dt, refresh)
        solver._initialize_particles(state_in, state_out, dt)
        self._initialized = True

    def iterate(
        self, state_in: State, state_out: State, control: Control | None, contacts: Contacts, dt: float, iteration: int
    ) -> None:
        """Execute one free-rigid/particle VBD block sweep."""
        if not self._initialized:
            raise RuntimeError("VBD-owned block iteration requires begin()")
        solver = self.solver
        if control is None:
            control = solver.model.control(clone_variables=False)
        solver._solve_rigid_body_iteration(state_in, state_out, control, contacts, dt)
        solver._solve_particle_iteration(state_in, state_out, contacts, dt, iteration)

    def finish(self, state_in: State, state_out: State, contacts: Contacts, dt: float) -> None:
        """Finalize velocities and commit local VBD persistent history."""
        if not self._initialized:
            raise RuntimeError("VBD-owned block finish requires begin()")
        solver = self.solver
        solver._snapshot_rigid_contact_history(contacts)
        solver._finalize_rigid_bodies(
            state_in,
            state_out,
            dt,
            apply_stick_deadzone=solver.rigid_contact_hard,
        )
        solver._finalize_particles(state_out, dt)
        self._initialized = False
        self._snapshots.clear()

    def abort(self) -> None:
        """Discard an unaccepted trial before any VBD history is finalized."""
        for name, snapshot in self._snapshots.items():
            value = getattr(self.solver, name)
            if value is not None:
                wp.copy(value, snapshot)
        self.solver._update_rigid_history = self._update_rigid_history
        self._initialized = False
        self._snapshots.clear()

    def _snapshot_persistent_state(self) -> None:
        """Capture VBD arrays that must not survive an unaccepted trial."""
        self._snapshots.clear()
        for name in (
            "body_q_prev",
            "_coupling_body_q_prev_snapshot",
            "_rigid_pose_rebaseline_mask",
            "joint_penalty_k",
            "joint_lambda_lin",
            "joint_lambda_ang",
            "joint_C0_lin",
            "joint_C0_ang",
            "joint_sigma_prev",
            "joint_kappa_prev",
            "joint_dkappa_prev",
            "body_body_contact_penalty_k",
            "body_body_contact_lambda",
            "body_body_contact_C0",
            "body_body_contact_stick_flag",
            "body_particle_contact_penalty_k",
        ):
            value = getattr(self.solver, name, None)
            if value is not None:
                self._snapshots[name] = wp.clone(value)

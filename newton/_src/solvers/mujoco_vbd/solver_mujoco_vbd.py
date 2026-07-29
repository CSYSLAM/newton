# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental direct MuJoCo reduced-coordinate / VBD joint solver."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np

from ...core.types import override
from ...sim import Contacts, Control, Model, State, StateFlags
from ..coupled.interface import CouplingInterface
from ..coupled.solver_coupled import _copy_control_to_entry
from ..mujoco import SolverMuJoCo
from ..solver import SolverBase
from ..vbd import SolverVBD
from .articulation_corrector import ArticulationCorrector
from .canonical_contacts import CanonicalRigidContacts
from .canonical_soft_contacts import CanonicalSoftContacts
from .gap_verifier import EndpointGapVerifier, SweptStateProbe, VBDDATProjector
from .ownership_partition import MuJoCoVBDOwnershipPartition
from .trial_workspace import MuJoCoVBDTrialWorkspace
from .vbd_owned_block import VBDOwnedBlock


def _prepare_smooth_mujoco_view(view: Model) -> None:
    """Make a compact predictor view valid for MuJoCo actuator export.

    Newton permits zero effort limits for direct-FK/kinematic authoring,
    whereas MuJoCo rejects a limited actuator range whose endpoints coincide.
    The smooth predictor does not use native contacts; an effectively unlimited
    range preserves authored target forces without mutating the shared model.
    """
    limits = view.joint_effort_limit.numpy()
    if np.any(limits <= 0.0):
        limits = limits.copy()
        limits[limits <= 0.0] = 1.0e30
        view.joint_effort_limit.assign(limits)


class _KinematicArticulationBoundary(SolverBase, CouplingInterface):
    """Partition placeholder for caller-prescribed articulated boundaries."""

    def step(self, state_in, state_out, control, contacts, dt) -> None:
        raise RuntimeError("kinematic articulation boundaries are advanced by caller FK, not SolverBase.step()")


class SolverMuJoCoVBD(SolverBase):
    """Direct reduced-coordinate articulation and VBD scene coupling.

    .. experimental::

    The articulated system is solved in MuJoCo tangent coordinates and receives
    contact reaction directly through a q-block; free rigid bodies and
    particles are owned by a local VBD view with articulated links retained as
    zero-inverse-mass collision boundaries.  This class never invokes ADMM or
    ``SolverCoupled.step()``.

    The initial execution path establishes the ownership and direct-reaction
    contract.  Swept DAT and copied VBD block kernels are added before this
    solver is used as the default for the high-speed cloth/bag examples.

    Args:
        model: Shared scene model.
        articulation_bodies: Global robot/link body ids.
        articulation_joints: Global joints forming complete articulations.
        vbd_bodies: Global free rigid body ids.
        vbd_particles: Global soft/cloth particle ids.
        articulation_mode: ``"dynamic"`` uses the full MuJoCo q-block
            pullback. ``"kinematic"`` treats caller-FK links as infinite-mass
            VBD collision boundaries and skips unnecessary smooth/q solves.
        vbd_options: Options forwarded to the local VBD implementation.
        mujoco_options: Options forwarded to the compact MuJoCo predictor.
        max_q_dofs: Maximum dense q-block tangent DOFs per world.
        coupling_iterations: Number of VBD boundary-relaxation sweeps after
            the q-space contact update.
        dat_max_subdivisions: Maximum rejected-trial bisection depth.
        dat_gap_tolerance: Allowed negative accepted contact gap [m].
        dat_sweep_samples: Interior geometry probes per trial before acceptance.
        collide: Optional collision callback. It must populate the supplied
            canonical contact buffer for the given global state. Pass
            :meth:`CollisionPipeline.collide` for full-surface soft contact.
    """

    def __init__(
        self,
        model: Model,
        *,
        articulation_bodies: Sequence[int],
        articulation_joints: Sequence[int],
        vbd_bodies: Sequence[int],
        vbd_particles: Sequence[int],
        articulation_mode: Literal["dynamic", "kinematic"] = "dynamic",
        vbd_options: Mapping[str, object] | None = None,
        mujoco_options: Mapping[str, object] | None = None,
        max_q_dofs: int = 96,
        coupling_iterations: int = 1,
        dat_max_subdivisions: int = 4,
        dat_gap_tolerance: float = 1.0e-5,
        dat_sweep_samples: int = 1,
        collide: Callable[[State, Contacts], None] | None = None,
    ) -> None:
        super().__init__(model)
        if coupling_iterations < 1:
            raise ValueError("coupling_iterations must be positive")
        if dat_max_subdivisions < 0:
            raise ValueError("dat_max_subdivisions must be non-negative")
        if articulation_mode not in ("dynamic", "kinematic"):
            raise ValueError("articulation_mode must be 'dynamic' or 'kinematic'")
        vbd_kwargs = dict(vbd_options or {})
        mujoco_kwargs = dict(mujoco_options or {})
        def articulation_solver(view: Model) -> SolverMuJoCo:
            _prepare_smooth_mujoco_view(view)
            return SolverMuJoCo(
                view,
                integrator="euler",
                disable_contacts=True,
                use_mujoco_contacts=False,
                **mujoco_kwargs,
            )

        self.articulation_mode = articulation_mode
        self.partition = MuJoCoVBDOwnershipPartition(
            model,
            articulation_bodies=articulation_bodies,
            articulation_joints=articulation_joints,
            vbd_bodies=vbd_bodies,
            vbd_particles=vbd_particles,
            articulation_solver=(
                articulation_solver if articulation_mode == "dynamic" else lambda view: _KinematicArticulationBoundary(view)
            ),
            vbd_solver=lambda view: SolverVBD(view, **vbd_kwargs),
        )
        self.workspace = MuJoCoVBDTrialWorkspace(self.partition)
        self.corrector = (
            ArticulationCorrector(self.partition, max_dofs=max_q_dofs) if articulation_mode == "dynamic" else None
        )
        self.vbd_block = VBDOwnedBlock(self.partition.vbd_entry.solver)
        if self.vbd_block.solver.rigid_contact_k_start_value >= 0.0:
            raise ValueError(
                "MuJoCo-VBD currently requires fixed VBD contact penalties; "
                "set rigid_avbd_linear_beta=0 until canonical soft-contact state is unified"
            )
        self.canonical_contacts = CanonicalRigidContacts(model) if articulation_mode == "dynamic" else None
        self.canonical_soft_contacts = CanonicalSoftContacts(model) if articulation_mode == "dynamic" else None
        self.gap_verifier = EndpointGapVerifier(model, tolerance=dat_gap_tolerance)
        self._collide = model.collide if collide is None else collide
        self.swept_probe = SweptStateProbe(
            model,
            self.gap_verifier,
            samples=dat_sweep_samples,
            collide=self._collide,
        )
        self.vbd_dat = VBDDATProjector(
            model,
            self.gap_verifier,
            self.swept_probe,
            collide=self._collide,
            max_bisections=dat_max_subdivisions,
        )
        self._contacts = model.contacts()
        self.coupling_iterations = coupling_iterations
        self.dat_max_subdivisions = dat_max_subdivisions
        self._substep_states = [model.state(requires_grad=False) for _ in range(dat_max_subdivisions)]
        self._vbd_dat_start = model.state(requires_grad=False)

    @property
    def contacts(self) -> Contacts:
        """Canonical contact buffer owned by the joint solver."""
        return self._contacts

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        """Advance one direct-coupled trial sweep.

        Args:
            state_in: Accepted input state.
            state_out: Output state.
            control: Global joint/actuator controls.
            contacts: Optional caller-owned canonical contact buffer.
            dt: Timestep [s].
        """
        if dt <= 0.0:
            raise ValueError("MuJoCo-VBD timestep must be positive")
        active_contacts = self._contacts if contacts is None else contacts
        self._advance_adaptive(state_in, state_out, control, active_contacts, dt, 0)

    def _advance_adaptive(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts,
        dt: float,
        depth: int,
    ) -> None:
        """Accept one trial, or retry the same motion as two smaller trials."""
        if self._trial_step(state_in, state_out, control, contacts, dt):
            return
        if depth >= self.dat_max_subdivisions:
            raise RuntimeError(
                "MuJoCo-VBD DAT exhausted its substep budget before reaching a penetration-free endpoint"
            )
        self._reset_rejected_vbd_trial()
        middle = self._substep_states[depth]
        half_dt = 0.5 * dt
        self._advance_adaptive(state_in, middle, control, contacts, half_dt, depth + 1)
        self._advance_adaptive(middle, state_out, control, contacts, half_dt, depth + 1)

    def _trial_step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        active_contacts: Contacts,
        dt: float,
    ) -> bool:
        """Run one uncommitted coupled trial and report endpoint feasibility."""
        self.workspace.begin(state_in)
        self.partition._distribute_state(self.workspace.accepted)

        # MuJoCo first supplies q_hat and its FK. VBD then optimizes the
        # free-rigid/particle endpoints against that articulated boundary;
        # only afterwards is its current contact geometry pulled back through
        # J^T into q-space. Kinematic callers already provided FK in state_in,
        # so no reduced-coordinate solve is meaningful or required.
        result = None
        if self.corrector is not None:
            result = self.corrector.predict(self.workspace, control, dt)
        self._collide(self.workspace.trial, active_contacts)

        # VBD receives q_hat links as read-only collision geometry. Its phases
        # are scheduled here rather than as a nested SolverVBD step, so the
        # q-space pullback and DAT gate share one uncommitted workspace.
        self.partition._distribute_state(self.workspace.trial)
        self._collide(self.workspace.trial, active_contacts)
        vbd_entry = self.partition.vbd_entry
        vbd_control = _copy_control_to_entry(control, vbd_entry)
        # The local VBD ModelView has compact shape/body/particle ids.  Filter
        # and remap canonical global contacts before a VBD kernel sees them;
        # this is pure data ownership, not a coupled solver iteration.
        vbd_contacts = self.partition.entry_contacts("vbd", active_contacts)
        if vbd_contacts is None:
            raise RuntimeError("MuJoCo-VBD requires a local VBD contact buffer")
        self.vbd_block.begin(vbd_entry.state_0, vbd_entry.state_1, vbd_control, vbd_contacts, dt)
        vbd_iterations = vbd_entry.solver.iterations
        for iteration in range(vbd_iterations):
            self.workspace._copy_state(self.workspace.trial, self._vbd_dat_start)
            self.vbd_block.iterate(
                vbd_entry.state_0,
                vbd_entry.state_1,
                vbd_control,
                vbd_contacts,
                dt,
                iteration,
            )
            self.workspace.scatter_vbd_owned(vbd_entry.state_0)
            self.vbd_dat.project(self._vbd_dat_start, self.workspace.trial, active_contacts)
            self.partition._distribute_state(self.workspace.trial)
            vbd_contacts = self.partition.entry_contacts("vbd", active_contacts)
            if vbd_contacts is None:
                raise RuntimeError("MuJoCo-VBD requires a local VBD contact buffer")
        # VBD's current endpoints now define the contact reaction applied to
        # the dynamic robot. MuJoCo's reduced mass metric maps that reaction
        # to q, and its tangent integration restores a valid articulation
        # configuration before a cheap VBD boundary relaxation.
        self.workspace.scatter_vbd_owned(vbd_entry.state_0)
        self._collide(self.workspace.trial, active_contacts)
        if self.corrector is not None:
            assert result is not None
            assert self.canonical_contacts is not None
            assert self.canonical_soft_contacts is not None
            self.corrector.correct_from_prediction(
                self.workspace,
                result,
                active_contacts,
                self.canonical_contacts,
                self.canonical_soft_contacts,
                dt,
            )
            self.partition._distribute_state(self.workspace.trial)
            self._collide(self.workspace.trial, active_contacts)
            vbd_contacts = self.partition.entry_contacts("vbd", active_contacts)
            if vbd_contacts is None:
                raise RuntimeError("MuJoCo-VBD requires a local VBD contact buffer")
            for polish_iteration in range(self.coupling_iterations):
                self.workspace._copy_state(self.workspace.trial, self._vbd_dat_start)
                self.vbd_block.iterate(
                    vbd_entry.state_0,
                    vbd_entry.state_1,
                    vbd_control,
                    vbd_contacts,
                    dt,
                    vbd_iterations + polish_iteration,
                )
                self.workspace.scatter_vbd_owned(vbd_entry.state_0)
                self.vbd_dat.project(self._vbd_dat_start, self.workspace.trial, active_contacts)
                self.partition._distribute_state(self.workspace.trial)
                vbd_contacts = self.partition.entry_contacts("vbd", active_contacts)
                if vbd_contacts is None:
                    raise RuntimeError("MuJoCo-VBD requires a local VBD contact buffer")
        # Body and particle positions live in VBD's input state until its
        # finalizer. Test exactly that endpoint before persistent histories
        # and output velocities are written.
        self.workspace.scatter_vbd_owned(vbd_entry.state_0)
        self._collide(self.workspace.trial, active_contacts)
        if self.gap_verifier.has_violation(self.workspace.trial, active_contacts) or self.swept_probe.has_violation(
            self.workspace.accepted, self.workspace.trial
        ):
            self.vbd_block.abort()
            self.workspace.rollback()
            return False
        vbd_contacts = self.partition.entry_contacts("vbd", active_contacts)
        if vbd_contacts is None:
            raise RuntimeError("MuJoCo-VBD requires a local VBD contact buffer")
        self.vbd_block.finish(vbd_entry.state_0, vbd_entry.state_1, vbd_contacts, dt)
        self.workspace.scatter_vbd_owned(vbd_entry.state_1)
        self.workspace.accept()
        self.workspace._copy_state(self.workspace.trial, state_out)
        return True

    def _reset_rejected_vbd_trial(self) -> None:
        """Restore local input state and cold-start VBD state after rejection."""
        self.partition._distribute_state(self.workspace.accepted)

    @override
    def reset(
        self,
        state: State,
        world_mask=None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Reset local scratch state and the VBD-owned solver state."""
        self.partition._distribute_state(state)
        self.partition.vbd_entry.solver.reset(
            self.partition.vbd_entry.state_0,
            world_mask=world_mask,
            flags=flags,
        )
        self.workspace.begin(state)
        self._contacts.clear()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-way MuJoCo <-> VBD backend with contact-native feedback (``DESIGN.md`` 4.1, 17).

Constructs the full iteration snapshot, effective mass, wrench, and convergence
buffers and runs K rounds of equal-and-opposite contact feedback between the
private MuJoCo and full VBD cores.
"""

from __future__ import annotations

import warp as wp

from ....sim import Contacts, Control, ModelFlags, State, StateFlags
from ..backend_mujoco import MuJoCoCouplingBackend
from ..backend_vbd import VBDCouplingBackend
from ..collision_pipeline import MuJoCoVBDCollisionPipeline
from ..config import MuJoCoVBDResolvedOptions
from ..contact_routing import build_contact_routing, validate_contact_routing
from ..convergence import MuJoCoVBDConvergence
from ..diagnostics import allocate_diagnostics
from ..effective_mass import MuJoCoVBDEffectiveMass
from ..feedback import MuJoCoVBDFeedback
from ..kernels import (
    reconcile_owned_body_state_kernel,
    reconcile_owned_joint_state_kernel,
    update_relaxed_wrench_kernel,
)
from ..model_overlay import build_model_overlays, refresh_model_overlays
from ..ownership import MuJoCoVBDOwnership
from ..state import allocate_runtime
from .base import MuJoCoVBDBackendBase

__all__ = ["TwoWayBackend"]


class TwoWayBackend(MuJoCoVBDBackendBase):
    """K-iteration bidirectional coupling (``DESIGN.md`` section 17)."""

    def __init__(self, model, ownership: MuJoCoVBDOwnership, options: MuJoCoVBDResolvedOptions) -> None:
        super().__init__(model, ownership, options)
        self.coupling = options.coupling

        self.routing = build_contact_routing(
            model,
            ownership,
            collision_options=options.collision_options,
            static_contact_owner=self.coupling.static_contact_owner,
        )
        self.overlays = build_model_overlays(model, ownership, self.routing, self.coupling)
        # A two-way fixed-point round must reproduce the same MuJoCo body state
        # from an identical transaction snapshot.  CUDA's level-parallel FK is
        # mathematically equivalent to the serial kernel, but its last-bit
        # roundoff can select a different stiff contact branch and then be
        # amplified by feedback on later rounds.  Keep the private MuJoCo view
        # on the serial path; the public model remains eligible for parallel FK.
        self.overlays.mujoco._fk_articulation_level_start = None

        self._diagnostics = allocate_diagnostics(model, backend=None, feedback_enabled=True)
        self.runtime = allocate_runtime(model, self.overlays, ownership, self.coupling)

        self.mujoco_backend = MuJoCoCouplingBackend(
            self.overlays.mujoco,
            ownership,
            self.runtime.mujoco_state_in,
            self.runtime.substep_state_snapshot,
            **dict(options.mujoco_options),
        )
        validate_contact_routing(model, ownership, self.routing)
        self.vbd_backend = VBDCouplingBackend(
            self.overlays.vbd,
            ownership,
            self.coupling,
            self.runtime.vbd_state_in,
            self.runtime.substep_state_snapshot,
            self.runtime.proxy_qd_before,
            **dict(options.vbd_options),
        )

        self.effective_mass = MuJoCoVBDEffectiveMass(
            self.mujoco_backend.solver, self.vbd_backend, ownership, self.coupling
        )
        self.collision_pipeline = MuJoCoVBDCollisionPipeline(
            model,
            ownership,
            self.routing,
            static_contact_owner=self.coupling.static_contact_owner,
            soft_contact_speculative_distance=self.coupling.soft_contact_speculative_distance,
            **dict(options.collision_options),
        )
        self.feedback = MuJoCoVBDFeedback(
            model,
            ownership,
            self._diagnostics,
            self.vbd_backend,
            self.collision_pipeline.contacts(),
        )
        self.convergence = MuJoCoVBDConvergence(model, ownership, self.coupling, self._diagnostics)

        self._contacts = self.collision_pipeline.contacts()
        self.vbd_backend.prepare_contact_capacity(self._contacts)

    # -- read-only accessors --

    @property
    def contacts(self) -> Contacts:
        return self._contacts

    @property
    def mujoco_solver(self):
        return self.mujoco_backend.solver

    @property
    def vbd_solver(self):
        return self.vbd_backend.solver

    # -- main step orchestration (DESIGN 17) --

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        selected_contacts = self._contacts if contacts is None else contacts
        if (
            selected_contacts.rigid_contact_max > self._contacts.rigid_contact_max
            or selected_contacts.soft_contact_max > self._contacts.soft_contact_max
        ):
            raise ValueError(
                "Two-way contact capacity exceeds the construction-time capacity; rebuild SolverMuJoCoVBD "
                "with larger collision_options before stepping or CUDA Graph capture."
            )

        self._begin_substep(state_in, control, dt)
        try:
            for iteration in range(self.coupling.iterations):
                self._restore_iteration(iteration)

                self.mujoco_backend.solve_iteration(
                    coupling_wrench=self.runtime.wrench_relaxed,
                    state_out=self.runtime.mujoco_state_out,
                    dt=dt,
                )
                self.effective_mass.update(self.runtime.mujoco_state_out, self.runtime.vbd_state_in)
                self.vbd_backend.sync_proxy_state(
                    self.runtime.mujoco_state_out,
                    self.runtime.wrench_relaxed,
                    dt,
                )

                self.collision_pipeline.collide_iteration(
                    self.runtime.vbd_state_in, selected_contacts, iteration=iteration, dt=dt
                )
                self.collision_pipeline.record_overflow(selected_contacts, self._diagnostics)
                self.vbd_backend.solve_iteration(self.runtime.vbd_state_out, control, selected_contacts, dt)
                self.vbd_backend.record_overflow(self._diagnostics)
                self.convergence.update_velocity_residual(
                    self.runtime.proxy_qd_before,
                    self.runtime.vbd_state_out.body_qd,
                )
                self.feedback.harvest(self.runtime.vbd_state_in, self.runtime.vbd_state_out, selected_contacts, dt)

                self._update_relaxation(iteration)
                self._record_final_iteration_state(iteration)

            self._commit_substep(state_out, selected_contacts)
        except Exception:
            self._abort_substep(state_out)
            raise

    # -- substep transaction lifecycle (DESIGN 17.1 - 17.4) --

    def _begin_substep(self, state_in: State, control: Control | None, dt: float) -> None:
        self.runtime.snapshot_public_state(state_in)
        self.runtime.snapshot_coupling_history()
        self.mujoco_backend.begin_substep(state_in, control, dt)
        self.vbd_backend.begin_substep(state_in, dt)
        self.collision_pipeline.begin_substep()

        warm_start = bool(self.coupling.warm_start_wrench)
        if not warm_start:
            self.runtime.clear_wrench_warm_start()
        self.convergence.begin_substep(warm_start)
        self._diagnostics.clear()

    def _restore_iteration(self, iteration: int) -> None:
        self.mujoco_backend.restore_iteration(iteration)
        self.vbd_backend.restore_iteration(iteration)
        self.collision_pipeline.restore_iteration(iteration)

    def _update_relaxation(self, iteration: int) -> None:
        n_proxy = int(self.ownership.proxy_body_ids.shape[0])
        if n_proxy == 0:
            return
        wp.copy(self.runtime.wrench_raw, self._diagnostics.feedback_wrench_raw)
        wp.launch(
            update_relaxed_wrench_kernel,
            dim=n_proxy,
            inputs=[
                iteration,
                self.coupling.relaxation_mode_int,
                self.coupling.relaxation_initial,
                self.coupling.relaxation_min,
                self.coupling.relaxation_max,
                self.model.body_world,
                self.ownership.proxy_body_ids,
                self.runtime.wrench_raw,
                self.runtime.wrench_previous,
                self.runtime.residual_previous,
                self.convergence.aitken_omega,
                self.runtime.wrench_relaxed,
                self.runtime.residual_current,
            ],
            device=self.device,
        )
        self.convergence.update_aitken_omega(self.runtime.residual_current, self.runtime.residual_previous, iteration)
        self.convergence.detect_failure(
            self.runtime.residual_current,
            self.runtime.residual_previous,
            iteration,
        )
        wp.copy(self._diagnostics.feedback_wrench_relaxed, self.runtime.wrench_relaxed)
        wp.copy(self.runtime.wrench_previous, self.runtime.wrench_relaxed)
        wp.copy(self.runtime.residual_previous, self.runtime.residual_current)

    def _record_final_iteration_state(self, iteration: int) -> None:
        if iteration != self.coupling.iterations - 1:
            return
        self.runtime.final_mujoco_state.assign(self.runtime.mujoco_state_out)
        self.runtime.final_vbd_state.assign(self.runtime.vbd_state_out)

    def _commit_substep(self, state_out: State, contacts: Contacts) -> None:
        self.convergence.finalize(
            self.runtime.residual_current,
            self.runtime.wrench_raw,
            self.runtime.wrench_relaxed,
        )
        self._validate_overflow_and_nonfinite()
        self.mujoco_backend.commit_substep(self.runtime.final_mujoco_state)
        self.vbd_backend.commit_substep(self.runtime.final_vbd_state, contacts)
        self._reconcile_public_state(state_out)

    def _abort_substep(self, state_out: State) -> None:
        self.mujoco_backend.abort_substep()
        self.vbd_backend.abort_substep()
        self.collision_pipeline.abort_substep()
        self.runtime.restore_coupling_history()
        self.runtime.restore_public_snapshot(state_out)

    def _reconcile_public_state(self, state_out: State) -> None:
        mj = self.runtime.final_mujoco_state
        vbd = self.runtime.final_vbd_state
        # VBD owns particles, pneumatic observables and all unselected state
        # namespaces. MuJoCo-owned body/joint rows are overwritten below.
        state_out.assign(vbd)
        if self.model.body_count:
            wp.launch(
                reconcile_owned_body_state_kernel,
                dim=self.model.body_count,
                inputs=[
                    self.ownership.body_owner,
                    mj.body_q,
                    mj.body_qd,
                    vbd.body_q,
                    vbd.body_qd,
                    state_out.body_q,
                    state_out.body_qd,
                ],
                device=self.device,
            )
        if self.model.joint_count and state_out.joint_q is not None:
            wp.launch(
                reconcile_owned_joint_state_kernel,
                dim=self.model.joint_count,
                inputs=[
                    self.ownership.joint_owner,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    self.model.joint_coord_count,
                    self.model.joint_dof_count,
                    mj.joint_q,
                    mj.joint_qd,
                    vbd.joint_q,
                    vbd.joint_qd,
                    state_out.joint_q,
                    state_out.joint_qd,
                ],
                device=self.device,
            )

    def _validate_overflow_and_nonfinite(self) -> None:
        # A captured graph cannot raise from device data. Flags remain exposed
        # through diagnostics and are checked on eager execution/replay control.
        if self.device.is_capturing:
            return
        if self.coupling.fail_on_overflow:
            for name in ("rigid_contact_overflow", "soft_contact_overflow", "body_particle_overflow"):
                flag = getattr(self._diagnostics, name, None)
                if flag is not None and int(flag.numpy()[0]) != 0:
                    raise RuntimeError(
                        f"TwoWayBackend detected {name}; increase contact capacity or reduce scene density "
                        "(DESIGN 21.1)."
                    )
        if self.coupling.fail_on_nonfinite:
            diagnostic_nonfinite = int(self._diagnostics.nonfinite_flag.numpy().sum())
            inertia_nonfinite = int(self.vbd_backend._nonfinite_flag.numpy()[0])
            if diagnostic_nonfinite or inertia_nonfinite:
                raise FloatingPointError(
                    "TwoWayBackend detected non-finite coupling feedback or effective inertia; "
                    "the substep transaction was aborted (DESIGN 14.1)."
                )

    # -- reset, model update, BVH (DESIGN 18) --

    def reset(self, state, world_mask: wp.array | None = None, flags: StateFlags | int | None = None) -> None:
        self.mujoco_backend.reset(state, world_mask, flags)
        self.vbd_backend.reset(state, world_mask, flags)
        self.collision_pipeline.reset(world_mask)
        self.convergence.reset(world_mask)
        self.feedback.clear()
        self.runtime.clear_wrench_warm_start_masked(self.model, world_mask)
        self._diagnostics.clear()

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        if int(flags) & int(ModelFlags.SHAPE_PROPERTIES):
            self.collision_pipeline.validate_runtime_shape_flags()
        refresh_model_overlays(self.model, self.ownership, self.coupling, self.overlays, flags)
        self.mujoco_backend.notify_model_changed(flags)
        self.vbd_backend.notify_model_changed(flags)
        if int(flags) & int(ModelFlags.BODY_INERTIAL_PROPERTIES):
            self.effective_mass.invalidate()

    def rebuild_bvh(self, state: State) -> None:
        self.vbd_backend.rebuild_bvh(state)
        self.collision_pipeline.rebuild_dynamic_bvhs(state)

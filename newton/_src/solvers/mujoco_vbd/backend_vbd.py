# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Transactional VBD backend adapter for :class:`SolverMuJoCoVBD`.

See ``DESIGN.md`` sections 9.2 and 16. Persistent AVBD, contact matching,
Dahl, pose-history, and pneumatic arrays are snapshotted into fixed buffers and
restored before each outer coupling round. The selected final round remains as
the sole committed history update.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ...sim import Contacts, Control, Model, ModelFlags, State
from .config import PROXY_RESPONSE_EFFECTIVE_MASS, MuJoCoVBDCouplingOptions
from .diagnostics import MuJoCoVBDDiagnostics, record_vbd_contact_overflow
from .kernels import (
    gather_proxy_effective_inverse_kernel,
    install_proxy_effective_inertia_kernel,
    sync_and_rewind_proxy_bodies_kernel,
)
from .ownership import MuJoCoVBDOwnership
from .vbd.solver_vbd import SolverVBD, _get_pneumatic_counts

__all__ = ["VBDCouplingBackend"]

_PERSISTENT_VBD_ARRAYS = (
    "particle_q_prev",
    "pos_prev_collision_detection",
    "body_q_prev",
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
    "body_particle_contact_lambda_history",
    "_prev_contact_lambda",
    "_prev_contact_stick_flag",
    "_prev_contact_penalty_k",
    "_prev_contact_point0",
    "_prev_contact_point1",
    "_prev_contact_offset0",
    "_prev_contact_offset1",
    "_prev_contact_normal",
    "_rigid_pose_rebaseline_mask",
    "_contact_history_reset_mask",
    "_contact_history_reset_pending",
    "_pneumatic_previous_volume",
    "_pneumatic_volume",
    "_pneumatic_absolute_pressure",
    "_pneumatic_gauge_pressure",
    "_pneumatic_curvature",
    "_pneumatic_volume_rate",
    "_pneumatic_clamp_flags",
)


class VBDCouplingBackend:
    """VBD begin/restore/solve/commit adapter (``DESIGN.md`` section 16)."""

    def __init__(
        self,
        model: Model,
        ownership: MuJoCoVBDOwnership,
        options: MuJoCoVBDCouplingOptions,
        state_in: State,
        state_snapshot: State,
        proxy_qd_before: wp.array,
        **vbd_options: object,
    ) -> None:
        self.model = model
        self.ownership = ownership
        self.options = options
        self.device = model.device

        kwargs = dict(vbd_options)
        # Two-way proxies must respond to feedback, so proxy bodies are NOT one-way.
        kwargs["integrate_with_external_rigid_solver"] = False
        kwargs.pop("one_way_proxy_bodies", None)
        kwargs["body_particle_contact_augmented_lagrangian"] = options.soft_contact_augmented_lagrangian
        kwargs["body_particle_contact_al_rho_scale"] = options.soft_contact_al_rho_scale
        kwargs["body_particle_contact_lambda_decay"] = options.soft_contact_lambda_decay
        self.solver = SolverVBD(model, **kwargs)
        al_body_mask = np.zeros(model.body_count, dtype=np.uint8)
        if options.soft_contact_augmented_lagrangian and ownership.proxy_bodies:
            al_body_mask[list(ownership.proxy_bodies)] = 1
        self.solver.body_particle_contact_al_body_mask = wp.array(al_body_mask, dtype=wp.uint8, device=self.device)
        self._state_in = state_in
        self._state_snapshot = state_snapshot
        self._proxy_qd_before_cache = proxy_qd_before

        self._pneumatic_cavity_count, _ = _get_pneumatic_counts(model)
        self._proxy_inv_mass = wp.zeros(max(int(ownership.proxy_body_ids.shape[0]), 1), dtype=float, device=self.device)
        self._proxy_inv_inertia = wp.zeros(
            max(int(ownership.proxy_body_ids.shape[0]), 1), dtype=wp.mat33, device=self.device
        )
        self._nonfinite_flag = wp.zeros(1, dtype=wp.int32, device=self.device)
        self._history_snapshot: dict[str, wp.array] = {}
        self._coupling_snapshot_baseline = wp.clone(self.solver._coupling_body_q_prev_snapshot)
        self._update_rigid_history_snapshot = True

    def prepare_contact_capacity(self, contacts: Contacts) -> None:
        """Preallocate VBD contact/history buffers before graph capture."""
        rigid_max = int(getattr(contacts, "rigid_contact_max", 0))
        soft_max = int(getattr(contacts, "soft_contact_max", 0))
        if rigid_max > 0 and self.solver.body_body_contact_penalty_k.shape[0] < rigid_max:
            self.solver._init_body_body_contact_state(rigid_max)
        if soft_max > 0 and self.solver.body_particle_contact_penalty_k.shape[0] < soft_max:
            self.solver._init_body_particle_contact_state(soft_max)
            self.solver._ensure_particle_contact_adjacency_capacity(soft_max)
        soft_tid_count = int(getattr(contacts.soft_contact_tids, "shape", (0,))[0])
        if self.solver.body_particle_contact_augmented_lagrangian and soft_tid_count > 0:
            self.solver._ensure_body_particle_contact_history_capacity(soft_tid_count)
        if self.solver.rigid_contact_history and rigid_max > 0:
            previous = self.solver._prev_contact_lambda
            if previous is None or previous.shape[0] < rigid_max:
                self.solver._init_rigid_contact_warmstart(rigid_max)
        if rigid_max > 0 and self.solver._rigid_contact_body0.shape[0] < rigid_max:
            self.solver._rigid_contact_body0 = wp.full(rigid_max, -1, dtype=wp.int32, device=self.device)
            self.solver._rigid_contact_body1 = wp.full(rigid_max, -1, dtype=wp.int32, device=self.device)
            self.solver._rigid_contact_point0_world = wp.zeros(rigid_max, dtype=wp.vec3, device=self.device)
            self.solver._rigid_contact_point1_world = wp.zeros(rigid_max, dtype=wp.vec3, device=self.device)
        self._allocate_history_snapshot()

    # -- proxy preconditioner installation (DESIGN 11.2) --

    def set_proxy_effective_inertia(self, mass: wp.array, inertia: wp.array) -> None:
        """Update only proxy slots in VBD's effective inverse mass/inertia."""
        n_proxy = int(self.ownership.proxy_body_ids.shape[0])
        if n_proxy == 0:
            return
        inv_mass = getattr(self.solver, "body_inv_mass_effective", None)
        inv_inertia = getattr(self.solver, "body_inv_inertia_effective", None)
        if inv_mass is None or inv_inertia is None:
            return
        wp.launch(
            install_proxy_effective_inertia_kernel,
            dim=n_proxy,
            inputs=[
                self.ownership.proxy_body_ids,
                mass,
                inertia,
                self.options.proxy_mass_scale,
                self.options.proxy_mass_min,
                self.options.proxy_mass_max,
                self.options.proxy_inertia_eigenvalue_min,
                self.options.proxy_inertia_eigenvalue_max,
                PROXY_RESPONSE_EFFECTIVE_MASS,
                inv_mass,
                inv_inertia,
                self._nonfinite_flag,
            ],
            device=self.device,
        )
        wp.launch(
            gather_proxy_effective_inverse_kernel,
            dim=n_proxy,
            inputs=[
                self.ownership.proxy_body_ids,
                inv_mass,
                inv_inertia,
                self._proxy_inv_mass,
                self._proxy_inv_inertia,
            ],
            device=self.device,
        )

    # -- proxy state synchronization (DESIGN 10.2 / 16) --

    def sync_proxy_state(
        self,
        mujoco_state_out: State,
        relaxed_wrench: wp.array,
        dt: float,
    ) -> None:
        """Prepare VBD proxies from MuJoCo solved states without double stepping."""
        n_proxy = int(self.ownership.proxy_body_ids.shape[0])
        if n_proxy == 0:
            return
        vbd_state = self._vbd_input_state()
        if vbd_state is None:
            return
        proxy_qd_before = self._proxy_qd_before()
        wp.launch(
            sync_and_rewind_proxy_bodies_kernel,
            dim=n_proxy,
            inputs=[
                dt,
                self.ownership.proxy_body_ids,
                mujoco_state_out.body_q,
                mujoco_state_out.body_qd,
                self.model.body_com,
                self.model.body_inertia,
                self.model.body_gravity_acceleration
                if hasattr(self.model, "body_gravity_acceleration")
                else wp.zeros(max(int(self.model.body_count), 1), dtype=wp.vec3, device=self.device),
                relaxed_wrench,
                self._proxy_inv_mass,
                PROXY_RESPONSE_EFFECTIVE_MASS,
                vbd_state.body_q,
                vbd_state.body_qd,
                vbd_state.body_f,
                self.solver.body_q_prev,
                self.solver._coupling_body_q_prev_snapshot,
                proxy_qd_before,
            ],
            device=self.device,
        )

    # -- transaction lifecycle (DESIGN 9.2) --

    def begin_substep(self, state_in: State, dt: float) -> None:
        _ = dt
        _ = state_in
        self._nonfinite_flag.zero_()
        self._snapshot_history()
        self.restore_iteration(0)

    def restore_iteration(self, iteration: int) -> None:
        _ = iteration
        self._state_in.assign(self._state_snapshot)
        self._restore_history()

    def solve_iteration(
        self,
        state_out: State,
        control: Control | None,
        contacts: Contacts,
        dt: float,
    ) -> None:
        """Execute one complete VBD solve against the current proxy state."""
        self.solver.step(self._state_in, state_out, control, contacts, dt)

    def record_overflow(self, diagnostics: MuJoCoVBDDiagnostics) -> None:
        """Latch VBD per-body adjacency overflows into coupling diagnostics."""
        rigid_flag = diagnostics.rigid_contact_overflow
        particle_flag = diagnostics.body_particle_overflow
        if rigid_flag is not None and particle_flag is not None:
            record_vbd_contact_overflow(
                self.solver.body_body_contact_overflow_max,
                self.solver.body_body_contact_buffer_pre_alloc,
                self.solver.body_particle_contact_overflow_max,
                self.solver.body_particle_contact_buffer_pre_alloc,
                rigid_flag,
                particle_flag,
            )

    def commit_substep(self, state_out: State, contacts: Contacts) -> None:
        _ = (state_out, contacts)

    def abort_substep(self) -> None:
        self._restore_history()
        wp.copy(self.solver._coupling_body_q_prev_snapshot, self._coupling_snapshot_baseline)

    def reset(self, state: State, world_mask: wp.array | None, flags: object) -> None:
        self.solver.reset(state, world_mask=world_mask, flags=flags)

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        self.solver.notify_model_changed(flags)

    def rebuild_bvh(self, state: State) -> None:
        rebuild = getattr(self.solver, "rebuild_bvh", None)
        if callable(rebuild):
            rebuild(state)

    # -- feedback accessors consumed by feedback.py (DESIGN 13.5) --

    def body_particle_contact_penalty_k(self):
        return getattr(self.solver, "body_particle_contact_penalty_k", None)

    def body_particle_contact_material_kd(self):
        return getattr(self.solver, "body_particle_contact_material_kd", None)

    def body_particle_contact_lambda(self):
        return getattr(self.solver, "body_particle_contact_lambda", None)

    def body_particle_contact_material_mu(self):
        return getattr(self.solver, "body_particle_contact_material_mu", None)

    def particle_q_prev(self):
        prev = getattr(self.solver, "particle_q_prev", None)
        if prev is not None:
            return prev
        return getattr(self, "_state_in", None) and self._state_in.particle_q

    def body_q_prev(self):
        return getattr(self.solver, "_coupling_body_q_prev_snapshot", self.solver.body_q_prev)

    def collect_rigid_contact_forces(self, contacts: Contacts, state_out: State, dt: float):
        """Evaluate the exact AVBD rigid-contact force law at the solved state."""
        return self.solver.collect_rigid_contact_forces(
            state_out.body_q,
            self.body_q_prev(),
            contacts,
            dt,
        )

    # -- internal helpers --

    def _vbd_input_state(self) -> State | None:
        return getattr(self, "_state_in", None)

    def _proxy_qd_before(self) -> wp.array:
        cached = getattr(self, "_proxy_qd_before_cache", None)
        if cached is None:
            cached = wp.zeros(max(int(self.model.body_count), 1), dtype=wp.spatial_vector, device=self.device)
            self._proxy_qd_before_cache = cached
        return cached

    def _allocate_history_snapshot(self) -> None:
        self._history_snapshot.clear()
        for name in _PERSISTENT_VBD_ARRAYS:
            value = getattr(self.solver, name, None)
            if isinstance(value, wp.array):
                self._history_snapshot[name] = wp.clone(value)

    def _snapshot_history(self) -> None:
        for name, snapshot in self._history_snapshot.items():
            source = getattr(self.solver, name, None)
            if isinstance(source, wp.array) and source.shape == snapshot.shape:
                wp.copy(snapshot, source)
        wp.copy(self._coupling_snapshot_baseline, self.solver._coupling_body_q_prev_snapshot)
        self._update_rigid_history_snapshot = bool(self.solver._update_rigid_history)

    def _restore_history(self) -> None:
        for name, snapshot in self._history_snapshot.items():
            target = getattr(self.solver, name, None)
            if isinstance(target, wp.array) and target.shape == snapshot.shape:
                wp.copy(target, snapshot)
        self.solver._update_rigid_history = self._update_rigid_history_snapshot

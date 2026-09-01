# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Transactional MuJoCo backend adapter for :class:`SolverMuJoCoVBD`.

See ``DESIGN.md`` sections 9.2 and 15. Provides a per-substep transaction so a
coupling iteration re-solves the same interval without advancing time, warm
start, sleeping bookkeeping, or step counters more than once.

The adapter restores persistent MuJoCo integration inputs before every outer
coupling iteration, so all iterations re-solve one physical substep. Derived
forward-dynamics buffers are recomputed by MuJoCo and are intentionally not
copied.
"""

from __future__ import annotations

import warnings

import numpy as np
import warp as wp

from ...sim import Control, Model, ModelFlags, State
from .kernels import compose_mujoco_body_force_kernel
from .mujoco.solver_mujoco import SolverMuJoCo
from .ownership import MuJoCoVBDOwnership

__all__ = ["MuJoCoCouplingBackend"]

_PERSISTENT_MUJOCO_FIELDS = (
    "qpos",
    "qvel",
    "act",
    "qacc_warmstart",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "eq_active",
    "mocap_pos",
    "mocap_quat",
    "plugin_state",
    "userdata",
    "time",
)


def _copy_body_state(destination: State, source: State) -> None:
    destination.assign(source)


class MuJoCoCouplingBackend:
    """MuJoCo begin/restore/solve/commit adapter (``DESIGN.md`` section 15)."""

    def __init__(
        self,
        model: Model,
        ownership: MuJoCoVBDOwnership,
        state_in: State,
        state_snapshot: State,
        **mujoco_options: object,
    ) -> None:
        self.model = model
        self.ownership = ownership
        self.device = model.device

        options = dict(mujoco_options)
        # Cross M-V contacts are owned by VBD; keep MuJoCo internal contacts only.
        if bool(options.get("enable_sleeping", False)):
            raise ValueError(
                "Two-way MuJoCo/VBD requires enable_sleeping=False so external feedback cannot leave "
                "a MuJoCo tree asleep."
            )
        update_interval = int(options.get("update_data_interval", 1))
        if update_interval != 1:
            raise ValueError(
                "Two-way MuJoCo/VBD requires update_data_interval=1 so every restored outer round "
                "rebuilds MuJoCo data from the same public input."
            )
        options["enable_sleeping"] = False
        options["update_data_interval"] = 1
        if model.device.is_cpu and model.world_count > 1 and bool(options.get("use_mujoco_cpu", False)):
            warnings.warn(
                "SolverMuJoCoVBD cannot use the native MuJoCo-C backend for a multi-world two-way solve: "
                "one MjData instance advances only the template world. Falling back to MuJoCo Warp on CPU "
                "with separate_worlds=True so every world receives coupling feedback.",
                RuntimeWarning,
                stacklevel=3,
            )
            options["use_mujoco_cpu"] = False
            options["separate_worlds"] = True
        self.solver = SolverMuJoCo(model, **options)
        self._state_in = state_in
        self._state_snapshot = state_snapshot

        self._solver_snapshot: dict[str, wp.array | np.ndarray] = {}
        self._step_snapshot = 0
        self._control: Control | None = None
        self._coupling_force = wp.zeros(max(int(model.body_count), 1), dtype=wp.spatial_vector, device=self.device)
        self._composed_force = wp.zeros(max(int(model.body_count), 1), dtype=wp.spatial_vector, device=self.device)
        self._allocate_solver_snapshot()

    def begin_substep(self, state_in: State, control: Control | None, dt: float) -> None:
        """Snapshot state and warm start (``DESIGN.md`` 15 begin_substep)."""
        _ = dt
        self._control = control
        self._snapshot_solver_state()
        self.restore_iteration(0)

    def restore_iteration(self, iteration: int) -> None:
        """Restore the same substep input without committing time/history."""
        _ = iteration
        _copy_body_state(self._state_in, self._state_snapshot)
        self._restore_solver_state()

    def solve_iteration(
        self,
        coupling_wrench: wp.array,
        state_out: State,
        dt: float,
    ) -> None:
        """Inject cross wrench and run one MuJoCo solve of the same interval."""
        if self.model.body_count:
            wp.copy(self._coupling_force, coupling_wrench)
            wp.launch(
                compose_mujoco_body_force_kernel,
                dim=self.model.body_count,
                inputs=[
                    self.ownership.body_owner,
                    self._state_in.body_f,
                    self._coupling_force,
                    self._composed_force,
                ],
                device=self.device,
            )
            wp.copy(self._state_in.body_f, self._composed_force)
        self.solver.step(self._state_in, state_out, self._control, None, dt)

    def evaluate_effective_mass_block(
        self,
        body_ids: wp.array,
        out_mass: wp.array,
        out_inertia: wp.array,
    ) -> None:
        """Delegate to the MuJoCo articulated effective-mass hook (DESIGN 11.1)."""
        kind = wp.zeros(body_ids.shape[0], dtype=wp.int32, device=self.device)
        local_pos = wp.zeros(body_ids.shape[0], dtype=wp.vec3, device=self.device)
        self.solver.coupling_eval_effective_mass_block(kind, body_ids, local_pos, out_mass, out_inertia)

    def wake_from_feedback(self, world_mask: wp.array) -> None:
        """Wake sleeping trees that received non-zero cross feedback (DESIGN 15.1)."""
        _ = world_mask  # Sleeping is rejected for the two-way backend at construction.

    def commit_substep(self, state_out: State) -> None:
        """Commit the selected final iteration; advance counters once (DESIGN 15)."""
        _ = state_out

    def abort_substep(self) -> None:
        """Restore snapshots after a failed iteration (``DESIGN.md`` 17.4)."""
        self._restore_solver_state()

    def _allocate_solver_snapshot(self) -> None:
        data = self.solver.mj_data if self.solver.use_mujoco_cpu else self.solver.mjw_data
        if data is not None:
            for name in _PERSISTENT_MUJOCO_FIELDS:
                value = getattr(data, name, None)
                if isinstance(value, wp.array):
                    self._solver_snapshot[name] = wp.clone(value)
                elif isinstance(value, np.ndarray):
                    self._solver_snapshot[name] = np.empty_like(value)

    def _snapshot_solver_state(self) -> None:
        data = self.solver.mj_data if self.solver.use_mujoco_cpu else self.solver.mjw_data
        if data is not None:
            for name, snapshot in self._solver_snapshot.items():
                source = getattr(data, name, None)
                if isinstance(source, wp.array) and isinstance(snapshot, wp.array):
                    wp.copy(snapshot, source)
                elif isinstance(source, np.ndarray) and isinstance(snapshot, np.ndarray):
                    snapshot[...] = source
        self._step_snapshot = int(self.solver._step)

    def _restore_solver_state(self) -> None:
        data = self.solver.mj_data if self.solver.use_mujoco_cpu else self.solver.mjw_data
        if data is not None:
            for name, snapshot in self._solver_snapshot.items():
                target = getattr(data, name, None)
                if isinstance(target, wp.array) and isinstance(snapshot, wp.array):
                    wp.copy(target, snapshot)
                elif isinstance(target, np.ndarray) and isinstance(snapshot, np.ndarray):
                    target[...] = snapshot
        self.solver._step = self._step_snapshot

    def reset(self, state: State, world_mask: wp.array | None, flags: object) -> None:
        self.solver.reset(state, world_mask=world_mask, flags=flags)

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        self.solver.notify_model_changed(flags)

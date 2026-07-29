# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Accepted and trial state ownership for MuJoCo--VBD sweeps."""

from __future__ import annotations

import warp as wp

from ...sim import State
from ..coupled.solver_coupled import SolverEntry
from .ownership_partition import MuJoCoVBDOwnershipPartition


@wp.kernel(enable_backward=False)
def _scatter_body_state(
    owned_global: wp.array[int],
    global_to_local: wp.array[int],
    local_q: wp.array[wp.transform],
    local_qd: wp.array[wp.spatial_vector],
    global_q: wp.array[wp.transform],
    global_qd: wp.array[wp.spatial_vector],
):
    index = wp.tid()
    global_body = owned_global[index]
    local_body = global_to_local[global_body]
    global_q[global_body] = local_q[local_body]
    global_qd[global_body] = local_qd[local_body]


@wp.kernel(enable_backward=False)
def _scatter_particle_state(
    owned_global: wp.array[int],
    global_to_local: wp.array[int],
    local_q: wp.array[wp.vec3],
    local_qd: wp.array[wp.vec3],
    global_q: wp.array[wp.vec3],
    global_qd: wp.array[wp.vec3],
):
    index = wp.tid()
    global_particle = owned_global[index]
    local_particle = global_to_local[global_particle]
    global_q[global_particle] = local_q[local_particle]
    global_qd[global_particle] = local_qd[local_particle]


@wp.kernel(enable_backward=False)
def _scatter_scalar_state(
    owned_global: wp.array[int],
    global_to_local: wp.array[int],
    local_values: wp.array[float],
    global_values: wp.array[float],
):
    index = wp.tid()
    global_index = owned_global[index]
    local_index = global_to_local[global_index]
    global_values[global_index] = local_values[local_index]


class MuJoCoVBDTrialWorkspace:
    """Own accepted and trial global states for one unified solver instance.

    The workspace holds only state data.  Contact multipliers, friction anchors
    and DAT metadata are owned by their respective new solver modules and are
    deliberately not committed here until a sweep has passed all acceptance
    gates.

    Args:
        partition: Static ownership partition for this solver instance.
    """

    def __init__(self, partition: MuJoCoVBDOwnershipPartition) -> None:
        self.partition = partition
        self.model = partition.model
        self.accepted = self.model.state(requires_grad=False)
        self.trial = self.model.state(requires_grad=False)

    def begin(self, state: State) -> None:
        """Copy the input state into both accepted and mutable trial storage."""
        self._copy_state(state, self.accepted)
        self._copy_state(state, self.trial)

    def rollback(self) -> None:
        """Discard trial changes and restore the last accepted state."""
        self._copy_state(self.accepted, self.trial)

    def accept(self) -> None:
        """Promote the current trial state after all acceptance gates pass."""
        self._copy_state(self.trial, self.accepted)

    def scatter_articulation(self, state: State) -> None:
        """Write articulated q/FK state into the global trial state."""
        self._scatter_entry(self.partition.articulation_entry, state)

    def scatter_vbd_owned(self, state: State) -> None:
        """Write only VBD-owned free bodies and particles into trial storage."""
        self._scatter_entry(self.partition.vbd_entry, state)

    def _scatter_entry(self, entry: SolverEntry, state: State) -> None:
        if entry.body_indices.shape[0] and state.body_q is not None:
            wp.launch(
                _scatter_body_state,
                dim=entry.body_indices.shape[0],
                inputs=[
                    entry.body_indices,
                    entry.body_global_to_local,
                    state.body_q,
                    state.body_qd,
                    self.trial.body_q,
                    self.trial.body_qd,
                ],
                device=self.model.device,
            )
        if entry.particle_indices.shape[0] and state.particle_q is not None:
            wp.launch(
                _scatter_particle_state,
                dim=entry.particle_indices.shape[0],
                inputs=[
                    entry.particle_indices,
                    entry.particle_global_to_local,
                    state.particle_q,
                    state.particle_qd,
                    self.trial.particle_q,
                    self.trial.particle_qd,
                ],
                device=self.model.device,
            )
        if entry.joint_q_indices.shape[0] and state.joint_q is not None:
            wp.launch(
                _scatter_scalar_state,
                dim=entry.joint_q_indices.shape[0],
                inputs=[
                    entry.joint_q_indices,
                    entry.joint_coord_global_to_local,
                    state.joint_q,
                    self.trial.joint_q,
                ],
                device=self.model.device,
            )
        if entry.joint_qd_indices.shape[0] and state.joint_qd is not None:
            wp.launch(
                _scatter_scalar_state,
                dim=entry.joint_qd_indices.shape[0],
                inputs=[
                    entry.joint_qd_indices,
                    entry.joint_dof_global_to_local,
                    state.joint_qd,
                    self.trial.joint_qd,
                ],
                device=self.model.device,
            )

    @staticmethod
    def _copy_state(source: State, destination: State) -> None:
        if source.body_q is not None and destination.body_q is not None:
            wp.copy(destination.body_q, source.body_q)
            wp.copy(destination.body_qd, source.body_qd)
        if source.particle_q is not None and destination.particle_q is not None:
            wp.copy(destination.particle_q, source.particle_q)
            wp.copy(destination.particle_qd, source.particle_qd)
        if source.joint_q is not None and destination.joint_q is not None:
            wp.copy(destination.joint_q, source.joint_q)
            wp.copy(destination.joint_qd, source.joint_qd)

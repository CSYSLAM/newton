# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed runtime buffers and public-state snapshot for :class:`SolverMuJoCoVBD`.

See ``DESIGN.md`` section 9.1. Every array here is allocated once at construction
time; nothing is resized after CUDA Graph capture.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from ...sim import Model, State
from .config import MuJoCoVBDCouplingOptions
from .model_overlay import MuJoCoVBDModelOverlays
from .ownership import MuJoCoVBDOwnership

__all__ = ["MuJoCoVBDRuntime", "allocate_runtime"]


@wp.kernel
def _clear_masked_coupling_body_arrays(
    world_count: int,
    world_mask: wp.array[wp.bool],
    body_world: wp.array[wp.int32],
    wrench_raw: wp.array[wp.spatial_vector],
    wrench_relaxed: wp.array[wp.spatial_vector],
    wrench_previous: wp.array[wp.spatial_vector],
    residual_current: wp.array[wp.spatial_vector],
    residual_previous: wp.array[wp.spatial_vector],
):
    body = wp.tid()
    world = body_world[body]
    selected = False
    if world >= 0:
        selected = world_mask[world]
    elif world_mask.shape[0] > world_count:
        selected = world_mask[world_count]
    if selected:
        wrench_raw[body] = wp.spatial_vector()
        wrench_relaxed[body] = wp.spatial_vector()
        wrench_previous[body] = wp.spatial_vector()
        residual_current[body] = wp.spatial_vector()
        residual_previous[body] = wp.spatial_vector()


@dataclass
class MuJoCoVBDRuntime:
    """Fixed per-substep working buffers (``DESIGN.md`` 9.1)."""

    mujoco_state_in: State
    mujoco_state_out: State
    vbd_state_in: State
    vbd_state_out: State

    substep_state_snapshot: State
    final_mujoco_state: State
    final_vbd_state: State

    proxy_qd_before: wp.array  # spatial_vector[body_count]
    proxy_mass: wp.array  # float[n_proxy]
    proxy_inertia: wp.array  # mat33[n_proxy]

    wrench_raw: wp.array  # spatial_vector[body_count]
    wrench_relaxed: wp.array  # spatial_vector[body_count]
    wrench_previous: wp.array  # spatial_vector[body_count]
    residual_current: wp.array  # spatial_vector[body_count]
    residual_previous: wp.array  # spatial_vector[body_count]
    wrench_raw_snapshot: wp.array  # spatial_vector[body_count]
    wrench_relaxed_snapshot: wp.array  # spatial_vector[body_count]
    wrench_previous_snapshot: wp.array  # spatial_vector[body_count]
    residual_current_snapshot: wp.array  # spatial_vector[body_count]
    residual_previous_snapshot: wp.array  # spatial_vector[body_count]

    aitken_omega: wp.array  # float[world_count]
    aitken_has_previous: wp.array  # int32[world_count]
    converged: wp.array  # bool[world_count]
    nonfinite_flag: wp.array  # int32[world_count]

    def snapshot_public_state(self, state_in: State) -> None:
        """Copy the public input into the abort/restore snapshot (``DESIGN.md`` 17.1)."""
        _copy_state(self.substep_state_snapshot, state_in)

    def snapshot_coupling_history(self) -> None:
        """Save warm-start state so an aborted substep has no hidden side effects."""
        for snapshot, live in (
            (self.wrench_raw_snapshot, self.wrench_raw),
            (self.wrench_relaxed_snapshot, self.wrench_relaxed),
            (self.wrench_previous_snapshot, self.wrench_previous),
            (self.residual_current_snapshot, self.residual_current),
            (self.residual_previous_snapshot, self.residual_previous),
        ):
            wp.copy(snapshot, live)

    def restore_coupling_history(self) -> None:
        """Restore the pre-substep warm-start state after transaction failure."""
        for live, snapshot in (
            (self.wrench_raw, self.wrench_raw_snapshot),
            (self.wrench_relaxed, self.wrench_relaxed_snapshot),
            (self.wrench_previous, self.wrench_previous_snapshot),
            (self.residual_current, self.residual_current_snapshot),
            (self.residual_previous, self.residual_previous_snapshot),
        ):
            wp.copy(live, snapshot)

    def restore_public_snapshot(self, state_out: State | None = None) -> None:
        """Restore the public snapshot after a failed substep (``DESIGN.md`` 17.4)."""
        if state_out is not None:
            _copy_state(state_out, self.substep_state_snapshot)

    def clear_wrench_warm_start(self) -> None:
        self.wrench_raw.zero_()
        self.wrench_relaxed.zero_()
        self.wrench_previous.zero_()
        self.residual_current.zero_()
        self.residual_previous.zero_()

    def clear_wrench_warm_start_masked(self, model: Model, world_mask: wp.array | None) -> None:
        if world_mask is None:
            self.clear_wrench_warm_start()
            return
        if model.body_count:
            wp.launch(
                _clear_masked_coupling_body_arrays,
                dim=model.body_count,
                inputs=[
                    model.world_count,
                    world_mask,
                    model.body_world,
                    self.wrench_raw,
                    self.wrench_relaxed,
                    self.wrench_previous,
                    self.residual_current,
                    self.residual_previous,
                ],
                device=model.device,
            )


def _copy_state(dst: State, src: State) -> None:
    dst.assign(src)


def allocate_runtime(
    model: Model,
    overlays: MuJoCoVBDModelOverlays,
    ownership: MuJoCoVBDOwnership,
    options: MuJoCoVBDCouplingOptions,
) -> MuJoCoVBDRuntime:
    """Allocate all fixed runtime buffers (``DESIGN.md`` 9.1)."""
    device = model.device
    body_count = max(int(model.body_count), 1)
    world_count = max(int(model.world_count), 1)
    n_proxy = max(int(ownership.proxy_body_ids.shape[0]), 1)

    runtime = MuJoCoVBDRuntime(
        mujoco_state_in=overlays.mujoco.state(),
        mujoco_state_out=overlays.mujoco.state(),
        vbd_state_in=overlays.vbd.state(),
        vbd_state_out=overlays.vbd.state(),
        substep_state_snapshot=model.state(),
        final_mujoco_state=overlays.mujoco.state(),
        final_vbd_state=overlays.vbd.state(),
        proxy_qd_before=wp.zeros(body_count, dtype=wp.spatial_vector, device=device),
        proxy_mass=wp.zeros(n_proxy, dtype=float, device=device),
        proxy_inertia=wp.zeros(n_proxy, dtype=wp.mat33, device=device),
        wrench_raw=wp.zeros(body_count, dtype=wp.spatial_vector, device=device),
        wrench_relaxed=wp.zeros(body_count, dtype=wp.spatial_vector, device=device),
        wrench_previous=wp.zeros(body_count, dtype=wp.spatial_vector, device=device),
        residual_current=wp.zeros(body_count, dtype=wp.spatial_vector, device=device),
        residual_previous=wp.zeros(body_count, dtype=wp.spatial_vector, device=device),
        wrench_raw_snapshot=wp.zeros(body_count, dtype=wp.spatial_vector, device=device),
        wrench_relaxed_snapshot=wp.zeros(body_count, dtype=wp.spatial_vector, device=device),
        wrench_previous_snapshot=wp.zeros(body_count, dtype=wp.spatial_vector, device=device),
        residual_current_snapshot=wp.zeros(body_count, dtype=wp.spatial_vector, device=device),
        residual_previous_snapshot=wp.zeros(body_count, dtype=wp.spatial_vector, device=device),
        aitken_omega=wp.zeros(world_count, dtype=float, device=device),
        aitken_has_previous=wp.zeros(world_count, dtype=wp.int32, device=device),
        converged=wp.zeros(world_count, dtype=wp.bool, device=device),
        nonfinite_flag=wp.zeros(world_count, dtype=wp.int32, device=device),
    )
    runtime.aitken_omega.fill_(float(options.relaxation_initial))
    return runtime

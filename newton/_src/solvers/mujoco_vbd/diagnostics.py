# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read-only coupling diagnostics for :class:`SolverMuJoCoVBD`.

See ``DESIGN.md`` sections 5.3 and 21. All fields are device-resident so that
CUDA Graph capture never forces a host synchronize; tests and explicit polling
read them outside the graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import warp as wp

from ...sim import Model

__all__ = ["MuJoCoVBDDiagnostics", "allocate_diagnostics", "record_vbd_contact_overflow"]


@wp.kernel
def _record_vbd_contact_overflow_kernel(
    rigid_overflow_max: wp.array[wp.int32],
    rigid_capacity: int,
    particle_overflow_max: wp.array[wp.int32],
    particle_capacity: int,
    rigid_overflow: wp.array[wp.int32],
    particle_overflow: wp.array[wp.int32],
):
    """Latch both VBD adjacency overflows with one fixed-topology launch."""
    if rigid_overflow_max[0] > rigid_capacity:
        rigid_overflow[0] = 1
    if particle_overflow_max[0] > particle_capacity:
        particle_overflow[0] = 1


def record_vbd_contact_overflow(
    rigid_overflow_max: wp.array,
    rigid_capacity: int,
    particle_overflow_max: wp.array,
    particle_capacity: int,
    rigid_overflow: wp.array,
    particle_overflow: wp.array,
) -> None:
    """Record both VBD adjacency overflow classes without a host sync."""
    wp.launch(
        _record_vbd_contact_overflow_kernel,
        dim=1,
        inputs=[
            rigid_overflow_max,
            rigid_capacity,
            particle_overflow_max,
            particle_capacity,
            rigid_overflow,
            particle_overflow,
        ],
        device=rigid_overflow.device,
    )


@dataclass
class MuJoCoVBDDiagnostics:
    """Device arrays exposing residual, convergence, and overflow state.

    Two-way-only fields are ``None`` when the selected backend has no feedback
    module or buffer (``DESIGN.md`` 5.3); ``None`` means "not allocated", never
    "allocated but currently zero".
    """

    residual_force_l2: wp.array | None  # float[world_count], two-way only
    residual_force_relative: wp.array | None  # float[world_count], two-way only
    residual_velocity_max: wp.array | None  # float[world_count], two-way only
    converged: wp.array | None  # bool[world_count], two-way only
    nonfinite_flag: wp.array | None  # int32[world_count]
    diverged_flag: wp.array | None  # int32[world_count], two-way only
    rigid_contact_overflow: wp.array | None  # int32[1]
    soft_contact_overflow: wp.array | None  # int32[1]
    body_particle_overflow: wp.array | None  # int32[1]
    feedback_wrench_raw: wp.array | None  # spatial_vector[body_count], two-way only
    feedback_wrench_relaxed: wp.array | None  # spatial_vector[body_count], two-way only
    backend: object = None

    def clear(self) -> None:
        """Zero transient per-substep diagnostics but keep the last results readable."""
        for field in (
            self.nonfinite_flag,
            self.diverged_flag,
            self.rigid_contact_overflow,
            self.soft_contact_overflow,
            self.body_particle_overflow,
        ):
            if field is not None:
                field.zero_()


def allocate_diagnostics(
    model: Model,
    *,
    backend: object = None,
    feedback_enabled: bool = True,
) -> MuJoCoVBDDiagnostics:
    """Allocate diagnostic arrays for the selected backend (``DESIGN.md`` 5.3, 9.1)."""
    device = model.device
    world_count = max(int(model.world_count), 1)
    body_count = max(int(model.body_count), 1)

    def _two_way(alloc):
        return alloc() if feedback_enabled else None

    return MuJoCoVBDDiagnostics(
        residual_force_l2=_two_way(lambda: wp.zeros(world_count, dtype=float, device=device)),
        residual_force_relative=_two_way(lambda: wp.zeros(world_count, dtype=float, device=device)),
        residual_velocity_max=_two_way(lambda: wp.zeros(world_count, dtype=float, device=device)),
        converged=_two_way(lambda: wp.zeros(world_count, dtype=wp.bool, device=device)),
        nonfinite_flag=wp.zeros(world_count, dtype=wp.int32, device=device),
        diverged_flag=_two_way(lambda: wp.zeros(world_count, dtype=wp.int32, device=device)),
        rigid_contact_overflow=wp.zeros(1, dtype=wp.int32, device=device),
        soft_contact_overflow=wp.zeros(1, dtype=wp.int32, device=device),
        body_particle_overflow=wp.zeros(1, dtype=wp.int32, device=device),
        feedback_wrench_raw=_two_way(lambda: wp.zeros(body_count, dtype=wp.spatial_vector, device=device)),
        feedback_wrench_relaxed=_two_way(lambda: wp.zeros(body_count, dtype=wp.spatial_vector, device=device)),
        backend=backend,
    )

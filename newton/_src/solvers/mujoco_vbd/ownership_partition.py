# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static ownership partitioning for the MuJoCo--VBD joint solver."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import warp as wp

from ...sim import Model
from ..coupled.model_view import ModelView
from ..coupled.solver_coupled import SolverCoupled
from ..solver import SolverBase


class MuJoCoVBDOwnershipPartition(SolverCoupled):
    """Build strict local solver models for one MuJoCo--VBD scene.

    This is construction-time infrastructure only.  It borrows the existing
    ``ModelView`` compaction rules because those rules already remap every
    finalized model attribute and all state/control domains.  The joint solver
    never calls :meth:`SolverCoupled.step`: its own outer VBD sweep consumes
    these immutable views and local/global maps directly.

    An articulation entry must contain a complete enabled articulation.  A
    partial view, or a compaction fallback to a full model, is an error: either
    case could place VBD-owned free-body DOFs in the reduced q-block.

    Args:
        model: Shared scene model.
        articulation_bodies: Global bodies belonging to the articulated
            reduced-coordinate system.
        articulation_joints: Global joints belonging to complete
            articulations.
        vbd_bodies: Global free rigid bodies owned by VBD.
        vbd_particles: Global particles owned by VBD.
        articulation_solver: Factory receiving the compact articulation view.
        vbd_solver: Factory receiving the compact VBD view.
    """

    def __init__(
        self,
        model: Model,
        *,
        articulation_bodies: Sequence[int],
        articulation_joints: Sequence[int],
        vbd_bodies: Sequence[int],
        vbd_particles: Sequence[int],
        articulation_solver: Callable[[ModelView], SolverBase],
        vbd_solver: Callable[[ModelView], SolverBase],
    ) -> None:
        self._articulation_body_ids = {int(body) for body in articulation_bodies}
        self._strict_compaction_entry: str | None = None
        super().__init__(
            model,
            entries=(
                self.Entry(
                    name="articulation",
                    solver=articulation_solver,
                    bodies=articulation_bodies,
                    joints=articulation_joints,
                ),
                self.Entry(
                    name="vbd",
                    solver=vbd_solver,
                    bodies=vbd_bodies,
                    particles=vbd_particles,
                ),
            ),
        )
        articulation = self._entries["articulation"]
        if articulation.view.joint_dof_count >= model.joint_dof_count and vbd_bodies:
            raise ValueError("articulation q-block was not reduced away from VBD-owned free-body DOFs")

    @property
    def articulation_entry(self):
        """Return the compact articulated solver entry and its index maps."""
        return self._entries["articulation"]

    @property
    def vbd_entry(self):
        """Return the compact VBD solver entry and its index maps."""
        return self._entries["vbd"]

    def _compact_entry_view_if_needed(self, view, cfg, proxy_body_keep, proxy_particle_keep, proxy_joint_keep):
        self._strict_compaction_entry = cfg.name
        compact = super()._compact_entry_view_if_needed(
            view,
            cfg,
            proxy_body_keep,
            proxy_particle_keep,
            proxy_joint_keep,
        )
        if compact is None:
            raise ValueError(
                f"MuJoCo-VBD cannot construct the strict {cfg.name!r} ownership view; "
                "select complete, world-homogeneous ownership domains."
            )
        if cfg.name == "vbd" and self._articulation_body_ids:
            body_projection = compact.projections[self.model.AttributeFrequency.BODY]
            read_only_local = [
                body_projection.global_to_local[body]
                for body in self._articulation_body_ids
                if body_projection.global_to_local[body] >= 0
            ]
            if len(read_only_local) != len(self._articulation_body_ids):
                raise ValueError("VBD ownership view lost an articulated collision boundary")
            view.disable_body_dynamics(wp.array(read_only_local, dtype=int, device=self.model.device))
        return compact

    def _entry_proxy_body_keep_indices(self, name: str) -> set[int]:
        """Keep articulated links as read-only collision boundaries in VBD."""
        if name == "vbd":
            return self._articulation_body_ids
        return set()

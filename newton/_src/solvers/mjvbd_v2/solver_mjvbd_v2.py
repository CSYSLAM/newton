# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint-only MuJoCo coupled to full VBD/AVBD dynamics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import warp as wp

from ...sim import BodyFlags, CollisionPipeline, Model, ModelBuilder, ModelFlags
from ..coupled.solver_coupled_proxy import SolverCoupledProxy
from .collision_pipeline import MJVBDV2SoftContactPipeline
from .mujoco.solver_mujoco import SolverMuJoCo
from .ownership import MJVBDV2Ownership, resolve_ownership
from .vbd.solver_vbd import SolverVBD, _get_pneumatic_counts
from .vbd_soft.solver_vbd import SolverVBD as SolverVBDSoft

__all__ = ["SolverMJVBDV2"]


class _OneWayCoupledProxy(SolverCoupledProxy):
    """Proxy composition whose source never receives destination feedback."""

    def _apply_proxy_body_effective_masses(self) -> None:
        for mapping in self._proxy_mappings:
            if mapping.proxy_ids_local is None or mapping.proxy_ids_local.shape[0] == 0:
                continue
            destination = self._entries[mapping.dst_name]
            destination.view.disable_body_dynamics(mapping.proxy_ids_local)
            destination.solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

    def _blend_proxy_feedback(self, proxy) -> None:
        proxy.coupling_forces.zero_()
        if proxy.coupling_forces_previous is not None:
            proxy.coupling_forces_previous.zero_()


class SolverMJVBDV2(_OneWayCoupledProxy):
    """One-way MuJoCo-joint to VBD, with full coupling among VBD objects.

    MuJoCo owns only the selected articulation bodies and joints. Those link
    bodies are synchronized into VBD as zero-inverse-mass moving colliders.
    Every remaining rigid body and every particle are owned by VBD.
    """

    def __init__(
        self,
        model: Model,
        *,
        mujoco_articulations: Sequence[int] | None = None,
        mujoco_joints: Sequence[int] | None = None,
        joint_mode: Literal["dynamic", "kinematic"] = "dynamic",
        contact_mode: Literal["auto", "soft", "full"] = "auto",
        vbd_options: Mapping[str, object] | None = None,
        mujoco_options: Mapping[str, object] | None = None,
        collision_options: Mapping[str, object] | None = None,
    ) -> None:
        if joint_mode not in ("dynamic", "kinematic"):
            raise ValueError("joint_mode must be 'dynamic' or 'kinematic'")
        if contact_mode not in ("auto", "soft", "full"):
            raise ValueError("contact_mode must be 'auto', 'soft', or 'full'")

        ownership = resolve_ownership(
            model,
            mujoco_articulations=mujoco_articulations,
            mujoco_joints=mujoco_joints,
        )
        self.ownership: MJVBDV2Ownership = ownership
        self.joint_mode = joint_mode
        self.contact_mode = (
            ("full" if ownership.has_vbd_dynamic_bodies else "soft") if contact_mode == "auto" else contact_mode
        )

        mujoco_kwargs = dict(mujoco_options or {})
        requested_disable_contacts = mujoco_kwargs.pop("disable_contacts", True)
        if requested_disable_contacts is not True:
            raise ValueError("MJVBDV2 requires mujoco_options['disable_contacts']=True")
        requested_mujoco_contacts = mujoco_kwargs.pop("use_mujoco_contacts", True)
        if requested_mujoco_contacts is not True:
            raise ValueError("MJVBDV2 currently requires mujoco_options['use_mujoco_contacts']=True")
        mujoco_kwargs["disable_contacts"] = True
        mujoco_kwargs["use_mujoco_contacts"] = True

        vbd_kwargs = dict(vbd_options or {})
        external_rigid = not ownership.has_vbd_dynamic_bodies
        requested_external = vbd_kwargs.pop("integrate_with_external_rigid_solver", external_rigid)
        if bool(requested_external) != external_rigid:
            required = "True" if external_rigid else "False"
            raise ValueError(
                "MJVBDV2 selects the VBD rigid integration mode from entity ownership; "
                f"integrate_with_external_rigid_solver must be {required} for this model"
            )
        vbd_kwargs["integrate_with_external_rigid_solver"] = external_rigid
        vbd_kwargs["external_rigid_state_from_input"] = external_rigid
        vbd_kwargs["one_way_proxy_bodies"] = True
        pneumatic_cavity_count, _ = _get_pneumatic_counts(model)
        vbd_solver_type = SolverVBDSoft if external_rigid and pneumatic_cavity_count == 0 else SolverVBD

        collision_kwargs = dict(collision_options or {})
        soft_contact_margin = float(collision_kwargs.get("soft_contact_margin", 0.0))
        if soft_contact_margin < 0.0:
            raise ValueError("collision_options['soft_contact_margin'] must be non-negative")
        if self.contact_mode == "soft" and collision_kwargs.get("enable_rigid_soft_full_surface_contact", False):
            raise ValueError("contact_mode='soft' does not support full-surface rigid-soft contacts")

        def configure_mujoco_view(view) -> None:
            if joint_mode != "kinematic" or view.body_count == 0:
                return
            flags = np.asarray(view.body_flags.numpy(), dtype=np.int32).copy()
            flags |= int(BodyFlags.KINEMATIC)
            view.body_flags = wp.array(flags, dtype=wp.int32, device=view.device)

        def make_collision_pipeline(view):
            if self.contact_mode == "soft":
                return MJVBDV2SoftContactPipeline(view, margin=soft_contact_margin)
            options = dict(collision_kwargs)
            options.setdefault("broad_phase", "nxn")
            options.setdefault("include_static_kinematic_pairs", False)
            return CollisionPipeline(view, **options)

        entries = [
            SolverCoupledProxy.Entry(
                name="mujoco",
                solver=lambda view: SolverMuJoCo(view, **mujoco_kwargs),
                bodies=ownership.mujoco_bodies,
                joints=ownership.mujoco_joints,
                configure_view=configure_mujoco_view,
            ),
            SolverCoupledProxy.Entry(
                name="vbd",
                solver=lambda view: vbd_solver_type(view, **vbd_kwargs),
                bodies=ownership.vbd_bodies,
                particles=ownership.vbd_particles,
            ),
        ]
        coupling = SolverCoupledProxy.Config(
            proxies=[
                SolverCoupledProxy.Proxy(
                    source="mujoco",
                    destination="vbd",
                    bodies=ownership.mujoco_bodies,
                    mode="staggered",
                    proxy_relaxation=0.0,
                    collision_pipeline=make_collision_pipeline,
                    collide_interval=1,
                )
            ],
            iterations=1,
        )
        super().__init__(model=model, entries=entries, coupling=coupling)

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register attributes required by the private MuJoCo and VBD copies."""
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
        SolverMuJoCo.register_custom_attributes(builder)

    @property
    def mujoco_solver(self) -> SolverMuJoCo:
        """Private MuJoCo joint solver."""
        return self._entries["mujoco"].solver

    @property
    def vbd_solver(self) -> SolverVBD | SolverVBDSoft:
        """Private VBD object solver."""
        return self._entries["vbd"].solver

    @property
    def contacts(self):
        """Current V2-owned post-MuJoCo contact buffer."""
        return self.get_proxy_contacts("mujoco", "vbd")

    def rebuild_bvh(self, state) -> None:
        """Rebuild the private VBD particle self-contact BVH."""
        vbd_entry = self._entries["vbd"]
        if vbd_entry.state_0 is None:
            return
        rebuild = getattr(vbd_entry.solver, "rebuild_bvh", None)
        if callable(rebuild):
            rebuild(vbd_entry.state_0)

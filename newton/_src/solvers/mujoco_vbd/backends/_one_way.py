# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

"""Shared construction and sequencing for the four one-way backends.

See ``DESIGN.md`` 4.1 and 7.2. One-way coupling drives a VBD solve from a source
pose (kinematic ``state_in`` or a dynamic MuJoCo step) without ever returning
force or pose to the source. No feedback iteration, effective mass, or Aitken
state is constructed.

.. warning::
    Interface-complete construction and step sequencing. Full numerical
    equivalence to the MJVBD_V2 one-way proxy contract (``DESIGN.md`` requirement
    4) still needs GPU validation.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ....sim import BodyFlags, Contacts, Control, JointType, ModelFlags, State, StateFlags
from ..config import PROXY_RESPONSE_DIRICHLET, MuJoCoVBDResolvedOptions
from ..contact_routing import build_contact_routing
from ..diagnostics import allocate_diagnostics
from ..kernels import reconcile_owned_body_state_kernel, sync_and_rewind_proxy_bodies_kernel
from ..model_overlay import build_model_overlays
from ..ownership import MuJoCoVBDOwnership
from .base import MuJoCoVBDBackendBase

__all__ = ["OneWayBackend"]


class OneWayBackend(MuJoCoVBDBackendBase):
    """Base for kinematic/dynamic one-way coupling (``DESIGN.md`` 4.1)."""

    _source_is_dynamic: bool = False
    _vbd_core: str = "full"

    def __init__(self, model, ownership: MuJoCoVBDOwnership, options: MuJoCoVBDResolvedOptions) -> None:
        super().__init__(model, ownership, options)

        routing = build_contact_routing(
            model,
            ownership,
            collision_options=options.collision_options,
            static_contact_owner=options.coupling.static_contact_owner,
        )
        self.routing = routing
        self.overlays = build_model_overlays(model, ownership, routing, options.coupling)
        # This is the exact MJVBD_V2 kinematic soft-contact contract: the
        # caller supplies the prescribed rigid pose in state_out, contacts are
        # generated from state_in particles against that pose, and VBD only
        # advances particles.  A proxy overlay changes force accounting and is
        # both slower and numerically different for this case.
        self._kinematic_soft_path = not self._source_is_dynamic and self._vbd_core == "soft"
        if self._kinematic_soft_path:
            self.overlays.vbd = model
        self._external_rigid = not ownership.has_vbd_dynamic_bodies
        if not self._kinematic_soft_path:
            self._configure_vbd_overlay()

        # Construct the VBD core on the VBD overlay so source joints are disabled.
        # Pneumatic cavities require the full core even when the pipeline is soft.
        from ..vbd.solver_vbd import _get_pneumatic_counts

        cavity, _ = _get_pneumatic_counts(model)
        use_full_core = self._vbd_core == "full" or cavity > 0
        self._effective_vbd_core = "full" if use_full_core else "soft"
        if use_full_core:
            from ..vbd.solver_vbd import SolverVBD
        else:
            from ..vbd_soft.solver_vbd import SolverVBD

        external_rigid = self._external_rigid
        vbd_options = dict(options.vbd_options)
        requested_external = bool(vbd_options.pop("integrate_with_external_rigid_solver", external_rigid))
        if requested_external != external_rigid:
            raise ValueError(
                "integrate_with_external_rigid_solver is selected from ownership and proxy response; "
                f"expected {external_rigid}, got {requested_external}."
            )
        if self._kinematic_soft_path:
            if "external_rigid_state_from_input" in vbd_options:
                raise ValueError("external_rigid_state_from_input is not used by the kinematic soft-contact backend")
            if "one_way_proxy_bodies" in vbd_options:
                raise ValueError("one_way_proxy_bodies is not used by the kinematic soft-contact backend")
            self.vbd = SolverVBD(
                self.overlays.vbd,
                integrate_with_external_rigid_solver=True,
                **vbd_options,
            )
        else:
            requested_input = bool(vbd_options.pop("external_rigid_state_from_input", external_rigid))
            if requested_input != external_rigid:
                raise ValueError(
                    "external_rigid_state_from_input must match integrate_with_external_rigid_solver "
                    f"({external_rigid})."
                )
            requested_one_way = bool(vbd_options.pop("one_way_proxy_bodies", True))
            if not requested_one_way:
                raise ValueError("one-way backends require one_way_proxy_bodies=True")
            self.vbd = SolverVBD(
                self.overlays.vbd,
                integrate_with_external_rigid_solver=external_rigid,
                external_rigid_state_from_input=external_rigid,
                one_way_proxy_bodies=True,
                **vbd_options,
            )

        collision_options = dict(options.collision_options)
        if self._vbd_core == "soft":
            margin = float(collision_options.pop("soft_contact_margin", 0.0))
            if collision_options:
                raise ValueError(
                    "soft one-way contact accepts only collision_options['soft_contact_margin']; "
                    f"unsupported keys are {sorted(collision_options)}"
                )
            if self._kinematic_soft_path:
                from ..soft_contact_pipeline import MJVBDSoftContactPipeline

                self.pipeline = MJVBDSoftContactPipeline(self.overlays.vbd, margin=margin)
                self._contacts = self.pipeline.make_contacts()
            else:
                from ..collision_pipeline import MJVBDV2SoftContactPipeline

                self.pipeline = MJVBDV2SoftContactPipeline(self.overlays.vbd, margin=margin)
                self._contacts = self.pipeline.contacts()
        else:
            from ..full_contact_pipeline import MJVBDV2CollisionPipeline

            collision_options.setdefault("broad_phase", "nxn")
            collision_options.setdefault("include_static_kinematic_pairs", False)
            self.pipeline = MJVBDV2CollisionPipeline(self.overlays.vbd, **collision_options)
            self._contacts = self.pipeline.contacts()

        # Dynamic source additionally constructs the private MuJoCo core.
        self.mujoco = None
        self._mujoco_state_out = None
        if self._source_is_dynamic:
            from ..mujoco.solver_mujoco import SolverMuJoCo

            self.mujoco = SolverMuJoCo(self.overlays.mujoco, **dict(options.mujoco_options))
            self._mujoco_state_out = self.overlays.mujoco.state()

        self._vbd_state_in = self.overlays.vbd.state()
        self._vbd_state_out = self.overlays.vbd.state()
        proxy_inv_mass = np.asarray(self.overlays.vbd.body_inv_mass.numpy(), dtype=np.float32)[
            list(ownership.proxy_bodies)
        ]
        self._proxy_inv_mass = wp.array(proxy_inv_mass, dtype=float, device=self.device)
        self._proxy_inv_inertia = wp.zeros(max(len(proxy_inv_mass), 1), dtype=wp.mat33, device=self.device)
        self._proxy_qd_before = wp.zeros(max(int(model.body_count), 1), dtype=wp.spatial_vector, device=self.device)
        self._zero_wrench = wp.zeros(max(int(model.body_count), 1), dtype=wp.spatial_vector, device=self.device)
        self._zero_gravity = wp.zeros(max(int(model.body_count), 1), dtype=wp.vec3, device=self.device)
        self._diagnostics = allocate_diagnostics(model, backend=None, feedback_enabled=False)

    @property
    def mujoco_solver(self):
        return self.mujoco

    @property
    def vbd_solver(self):
        return self.vbd

    @property
    def contacts(self) -> Contacts | None:
        return self._contacts

    def _configure_vbd_overlay(self) -> None:
        """Disable source joints and configure proxy solve-time inertia."""
        model = self.model
        overlay = self.overlays.vbd
        body_indices = np.asarray(self.ownership.proxy_bodies, dtype=np.int32)
        joint_indices = np.asarray(self.ownership.mujoco_joints, dtype=np.int32)

        if body_indices.size:
            body_flags = np.asarray(overlay.body_flags.numpy(), dtype=np.int32).copy()
            # Fixed one-way orchestration never calls the generic coupling
            # interface, so these are ordinary external/kinematic VBD bodies,
            # not CouplingInterface proxy bodies. PROXY changes VBD's force
            # accounting and breaks MJVBD_V2 kinematic parity.
            body_flags[body_indices] &= ~int(BodyFlags.PROXY)
            body_flags[body_indices] |= int(BodyFlags.KINEMATIC)
            body_inv_mass = np.asarray(overlay.body_inv_mass.numpy(), dtype=np.float32).copy()
            body_inv_mass[body_indices] = 0.0
            overlay.body_inv_mass = wp.array(body_inv_mass, dtype=float, device=model.device)
            body_inv_inertia = np.asarray(overlay.body_inv_inertia.numpy(), dtype=np.float32).copy()
            body_inv_inertia[body_indices] = 0.0
            overlay.body_inv_inertia = wp.array(body_inv_inertia, dtype=wp.mat33, device=model.device)
            overlay.body_flags = wp.array(body_flags, dtype=wp.int32, device=model.device)

        if joint_indices.size:
            joint_enabled = np.asarray(overlay.joint_enabled.numpy(), dtype=np.bool_).copy()
            joint_enabled[joint_indices] = False
            overlay.joint_enabled = wp.array(joint_enabled, dtype=wp.bool, device=model.device)

            joint_type = np.asarray(overlay.joint_type.numpy(), dtype=np.int32).copy()
            cable = joint_type[joint_indices] == int(JointType.CABLE)
            joint_type[joint_indices[cable]] = int(JointType.D6)
            overlay.joint_type = wp.array(joint_type, dtype=wp.int32, device=model.device)

    def _sync_proxy_from_source(self, source_state: State, dt: float) -> None:
        n_proxy = int(self.ownership.proxy_body_ids.shape[0])
        if n_proxy == 0:
            return
        wp.launch(
            sync_and_rewind_proxy_bodies_kernel,
            dim=n_proxy,
            inputs=[
                dt,
                self.ownership.proxy_body_ids,
                source_state.body_q,
                source_state.body_qd,
                self._zero_gravity,  # gravity channel unused for verbatim copy
                self._zero_wrench,
                self._proxy_inv_mass,
                self._proxy_inv_inertia,
                PROXY_RESPONSE_DIRICHLET,
                self._vbd_state_in.body_q,
                self._vbd_state_in.body_qd,
                self._proxy_qd_before,
            ],
            device=self.device,
        )

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        if self._kinematic_soft_path:
            selected_contacts = self._contacts if contacts is None else contacts
            self.pipeline.generate(state_in, state_out, selected_contacts)
            self.vbd.step(state_in, state_out, control, selected_contacts, dt)
            return

        # Seed the VBD input from the caller's public state.
        self._copy_public_state(self._vbd_state_in, state_in)

        if self._source_is_dynamic and self.mujoco is not None:
            self.mujoco.step(state_in, self._mujoco_state_out, control, None, dt)
            source_state = self._mujoco_state_out
        else:
            source_state = state_in

        self._sync_proxy_from_source(source_state, dt)
        selected_contacts = self._contacts if contacts is None else contacts
        self.pipeline.collide(self._vbd_state_in, selected_contacts)
        self.vbd.step(self._vbd_state_in, self._vbd_state_out, control, selected_contacts, dt)

        # Reconcile: source bodies from the source solve, VBD bodies/particles from VBD.
        if self.model.body_count:
            wp.launch(
                reconcile_owned_body_state_kernel,
                dim=self.model.body_count,
                inputs=[
                    self.ownership.body_owner,
                    source_state.body_q,
                    source_state.body_qd,
                    self._vbd_state_out.body_q,
                    self._vbd_state_out.body_qd,
                    state_out.body_q,
                    state_out.body_qd,
                ],
                device=self.device,
            )
        if self.model.particle_count and state_out.particle_q is not None:
            wp.copy(state_out.particle_q, self._vbd_state_out.particle_q)
            wp.copy(state_out.particle_qd, self._vbd_state_out.particle_qd)
        if state_out.joint_q is not None and source_state.joint_q is not None:
            wp.copy(state_out.joint_q, source_state.joint_q)
        if state_out.joint_qd is not None and source_state.joint_qd is not None:
            wp.copy(state_out.joint_qd, source_state.joint_qd)

    def reset(self, state, world_mask: wp.array | None = None, flags: StateFlags | int | None = None) -> None:
        self.vbd.reset(state, world_mask=world_mask, flags=flags)
        if self.mujoco is not None:
            self.mujoco.reset(state, world_mask=world_mask, flags=flags)
        self._contacts.clear()

    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        self.vbd.notify_model_changed(flags)
        if self.mujoco is not None:
            self.mujoco.notify_model_changed(flags)
        if int(flags) & (int(ModelFlags.SHAPE_PROPERTIES) | int(ModelFlags.MODEL_PROPERTIES)):
            rebuild = getattr(self.pipeline, "rebuild", None)
            if callable(rebuild):
                rebuild()

    def rebuild_bvh(self, state: State) -> None:
        rebuild = getattr(self.vbd, "rebuild_bvh", None)
        if callable(rebuild):
            rebuild(state)

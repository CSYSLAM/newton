# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene-specialized dispatch for :class:`SolverMJVBDV2`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import (
    BodyFlags,
    CollisionPipeline,
    Contacts,
    Control,
    JointType,
    Model,
    ModelBuilder,
    ModelFlags,
    State,
    StateFlags,
)
from ..solver import SolverBase
from .collision_pipeline import MJVBDV2SoftContactPipeline
from .ownership import MJVBDV2Ownership, resolve_ownership
from .soft_contact_pipeline import MJVBDSoftContactPipeline
from .solver_mjvbd_v2 import SolverMJVBDV2 as _SolverMJVBDV2Coupled
from .vbd.solver_vbd import SolverVBD
from .vbd_soft.solver_vbd import SolverVBD as SolverVBDSoft

__all__ = ["SolverMJVBDV2"]


class _PureVBDBackend(SolverBase):
    """Full VBD backend used when no joints are assigned to MuJoCo."""

    def __init__(
        self,
        model: Model,
        ownership: MJVBDV2Ownership,
        *,
        contact_mode: Literal["auto", "soft", "full"],
        vbd_options: Mapping[str, object] | None,
        collision_options: Mapping[str, object] | None,
    ) -> None:
        super().__init__(model)
        self.contact_mode = (
            ("full" if ownership.has_vbd_dynamic_bodies else "soft") if contact_mode == "auto" else contact_mode
        )

        vbd_kwargs = dict(vbd_options or {})
        external_rigid = not ownership.has_vbd_dynamic_bodies
        requested_external = vbd_kwargs.pop("integrate_with_external_rigid_solver", external_rigid)
        if bool(requested_external) != external_rigid:
            required = "True" if external_rigid else "False"
            raise ValueError(
                "MJVBDV2 pure-VBD mode derives rigid integration from the model; "
                f"integrate_with_external_rigid_solver must be {required}"
            )
        self.vbd_solver = SolverVBD(
            model,
            integrate_with_external_rigid_solver=external_rigid,
            **vbd_kwargs,
        )

        options = dict(collision_options or {})
        if self.contact_mode == "soft":
            margin = float(options.pop("soft_contact_margin", 0.0))
            if options:
                raise ValueError(
                    "MJVBDV2 contact_mode='soft' accepts only collision_options['soft_contact_margin']; "
                    f"unsupported keys are {sorted(options)}"
                )
            self.pipeline = MJVBDV2SoftContactPipeline(model, margin=margin)
        else:
            options.setdefault("broad_phase", "nxn")
            options.setdefault("include_static_kinematic_pairs", False)
            self.pipeline = CollisionPipeline(model, **options)
        self.contacts = self.pipeline.contacts()

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        selected = self.contacts if contacts is None else contacts
        self.pipeline.collide(state_in, selected)
        self.vbd_solver.step(state_in, state_out, control, selected, dt)

    @override
    def reset(
        self,
        state: State,
        world_mask=None,
        flags: StateFlags | int | None = None,
    ) -> None:
        self.vbd_solver.reset(state, world_mask=world_mask, flags=flags)
        self.contacts.clear()

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        self.vbd_solver.notify_model_changed(flags)

    def rebuild_bvh(self, state: State) -> None:
        self.vbd_solver.rebuild_bvh(state)


class _KinematicSoftBackend(SolverBase):
    """Optimized MJVBD path for prescribed joints interacting with particles."""

    def __init__(
        self,
        model: Model,
        *,
        vbd_options: Mapping[str, object] | None,
        collision_options: Mapping[str, object] | None,
    ) -> None:
        super().__init__(model)
        options = dict(collision_options or {})
        margin = float(options.pop("soft_contact_margin", 0.0))
        if options:
            raise ValueError(
                "The kinematic soft-only MJVBDV2 backend accepts only "
                "collision_options['soft_contact_margin']; "
                f"unsupported keys are {sorted(options)}"
            )
        vbd_kwargs = dict(vbd_options or {})
        requested_external = vbd_kwargs.pop("integrate_with_external_rigid_solver", True)
        if requested_external is not True:
            raise ValueError("The kinematic soft-only backend requires external rigid integration")
        self.vbd_solver = SolverVBDSoft(
            model,
            integrate_with_external_rigid_solver=True,
            **vbd_kwargs,
        )
        self.pipeline = MJVBDSoftContactPipeline(model, margin=margin)
        self.contacts = self.pipeline.make_contacts()

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("MJVBDV2 timestep dt must be positive")
        selected = self.contacts if contacts is None else contacts
        self.pipeline.validate_contacts(selected)
        self.pipeline.generate(state_in, state_out, selected)
        self.vbd_solver.step(state_in, state_out, control, selected, dt)

    @override
    def reset(
        self,
        state: State,
        world_mask=None,
        flags: StateFlags | int | None = None,
    ) -> None:
        self.vbd_solver.reset(state, world_mask=world_mask, flags=flags)
        self.contacts.clear()

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        self.vbd_solver.notify_model_changed(flags)
        if flags & (ModelFlags.SHAPE_PROPERTIES | ModelFlags.MODEL_PROPERTIES):
            self.pipeline.rebuild()
            self.contacts = self.pipeline.make_contacts()

    def rebuild_bvh(self, state: State) -> None:
        self.vbd_solver.rebuild_bvh(state)


class _KinematicFullVBDBackend(SolverBase):
    """Full VBD solve with selected articulation links as moving colliders."""

    def __init__(
        self,
        model: Model,
        ownership: MJVBDV2Ownership,
        *,
        vbd_options: Mapping[str, object] | None,
        collision_options: Mapping[str, object] | None,
    ) -> None:
        super().__init__(model)
        # A shallow Model overlay keeps large topology arrays shared while
        # avoiding ModelView attribute-proxy overhead inside every VBD launch.
        view = copy(model)
        body_indices = np.asarray(ownership.mujoco_bodies, dtype=np.int32)
        joint_indices = np.asarray(ownership.mujoco_joints, dtype=np.int32)

        body_inv_mass = model.body_inv_mass.numpy().copy()
        body_inv_mass[body_indices] = 0.0
        view.body_inv_mass = wp.array(body_inv_mass, dtype=float, device=model.device)
        body_inv_inertia = model.body_inv_inertia.numpy().copy()
        body_inv_inertia[body_indices] = 0.0
        view.body_inv_inertia = wp.array(body_inv_inertia, dtype=wp.mat33, device=model.device)
        body_flags = np.asarray(model.body_flags.numpy(), dtype=np.int32).copy()
        body_flags[body_indices] |= int(BodyFlags.KINEMATIC)
        view.body_flags = wp.array(body_flags, dtype=wp.int32, device=model.device)

        joint_enabled = model.joint_enabled.numpy().copy()
        joint_enabled[joint_indices] = False
        view.joint_enabled = wp.array(joint_enabled, dtype=wp.bool, device=model.device)
        joint_type = np.asarray(model.joint_type.numpy(), dtype=np.int32).copy()
        cable_mask = joint_type[joint_indices] == int(JointType.CABLE)
        joint_type[joint_indices[cable_mask]] = int(JointType.D6)
        view.joint_type = wp.array(joint_type, dtype=wp.int32, device=model.device)
        self.view = view

        vbd_kwargs = dict(vbd_options or {})
        requested_external = vbd_kwargs.pop("integrate_with_external_rigid_solver", False)
        if requested_external is not False:
            raise ValueError("The kinematic full-VBD backend requires internal VBD rigid integration")
        self.vbd_solver = SolverVBD(
            view,
            integrate_with_external_rigid_solver=False,
            **vbd_kwargs,
        )
        options = dict(collision_options or {})
        options.setdefault("broad_phase", "nxn")
        options.setdefault("include_static_kinematic_pairs", False)
        self.pipeline = CollisionPipeline(view, **options)
        self.contacts = self.pipeline.contacts()

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        # Full VBD writes particle and body outputs itself. Preserve only the
        # prescribed generalized coordinates that VBD does not own.
        wp.copy(state_out.joint_q, state_in.joint_q)
        wp.copy(state_out.joint_qd, state_in.joint_qd)
        selected = self.contacts if contacts is None else contacts
        self.pipeline.collide(state_in, selected)
        self.vbd_solver.step(state_in, state_out, control, selected, dt)

    @override
    def reset(
        self,
        state: State,
        world_mask=None,
        flags: StateFlags | int | None = None,
    ) -> None:
        self.vbd_solver.reset(state, world_mask=world_mask, flags=flags)
        self.contacts.clear()

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        self.vbd_solver.notify_model_changed(flags)

    def rebuild_bvh(self, state: State) -> None:
        self.vbd_solver.rebuild_bvh(state)


class SolverMJVBDV2(SolverBase):
    """Dispatch MuJoCo-joint/VBD-object coupling to the cheapest valid backend."""

    @dataclass(frozen=True)
    class Features:
        """Scene features used to select and audit the V2 backend."""

        backend: Literal["pure_vbd", "mjvbd_kinematic_soft", "vbd_kinematic_full", "coupled"]
        mujoco_joint_count: int
        vbd_body_count: int
        vbd_dynamic_body_count: int
        particle_count: int
        triangle_count: int
        edge_count: int
        tetrahedron_count: int
        spring_count: int
        rigid_solve_enabled: bool
        particle_solve_enabled: bool
        triangle_solve_enabled: bool
        bending_solve_enabled: bool
        tetrahedron_solve_enabled: bool
        spring_solve_enabled: bool

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
        super().__init__(model)
        if joint_mode not in ("dynamic", "kinematic"):
            raise ValueError("joint_mode must be 'dynamic' or 'kinematic'")
        if contact_mode not in ("auto", "soft", "full"):
            raise ValueError("contact_mode must be 'auto', 'soft', or 'full'")

        self.ownership = resolve_ownership(
            model,
            mujoco_articulations=mujoco_articulations,
            mujoco_joints=mujoco_joints,
        )
        if self.ownership.has_vbd_dynamic_bodies and contact_mode == "soft":
            raise ValueError("contact_mode='soft' cannot solve VBD-owned dynamic rigid-body contacts")
        inv_mass = np.asarray(model.body_inv_mass.numpy(), dtype=np.float64)
        body_flags = np.asarray(model.body_flags.numpy(), dtype=np.int32)
        dynamic_body_count = sum(
            inv_mass[index] > 0.0 and (int(body_flags[index]) & int(BodyFlags.KINEMATIC)) == 0
            for index in self.ownership.vbd_bodies
        )

        if not self.ownership.mujoco_joints:
            if mujoco_options:
                raise ValueError("mujoco_options are not used when no joints are assigned to MuJoCo")
            backend_name = "pure_vbd"
            backend = _PureVBDBackend(
                model,
                self.ownership,
                contact_mode=contact_mode,
                vbd_options=vbd_options,
                collision_options=collision_options,
            )
        elif (
            joint_mode == "kinematic" and not self.ownership.has_vbd_dynamic_bodies and contact_mode in ("auto", "soft")
        ):
            if mujoco_options:
                raise ValueError("mujoco_options are not used by the kinematic soft-only MJVBD backend")
            backend_name = "mjvbd_kinematic_soft"
            backend = _KinematicSoftBackend(
                model,
                vbd_options=vbd_options,
                collision_options=collision_options,
            )
        elif joint_mode == "kinematic":
            if mujoco_options:
                raise ValueError("mujoco_options are not used by the kinematic full-VBD backend")
            backend_name = "vbd_kinematic_full"
            backend = _KinematicFullVBDBackend(
                model,
                self.ownership,
                vbd_options=vbd_options,
                collision_options=collision_options,
            )
        else:
            backend_name = "coupled"
            backend = _SolverMJVBDV2Coupled(
                model,
                mujoco_articulations=mujoco_articulations,
                mujoco_joints=mujoco_joints,
                joint_mode=joint_mode,
                contact_mode=contact_mode,
                vbd_options=vbd_options,
                mujoco_options=mujoco_options,
                collision_options=collision_options,
            )

        self.backend = backend
        self.features = self.Features(
            backend=backend_name,
            mujoco_joint_count=len(self.ownership.mujoco_joints),
            vbd_body_count=len(self.ownership.vbd_bodies),
            vbd_dynamic_body_count=int(dynamic_body_count),
            particle_count=int(model.particle_count),
            triangle_count=int(model.tri_count),
            edge_count=int(model.edge_count),
            tetrahedron_count=int(model.tet_count),
            spring_count=int(model.spring_count),
            rigid_solve_enabled=dynamic_body_count > 0,
            particle_solve_enabled=model.particle_count > 0,
            triangle_solve_enabled=model.tri_count > 0,
            bending_solve_enabled=model.edge_count > 0,
            tetrahedron_solve_enabled=model.tet_count > 0,
            spring_solve_enabled=model.spring_count > 0,
        )

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register attributes needed by every possible V2 backend."""
        _SolverMJVBDV2Coupled.register_custom_attributes(builder)

    @property
    def contacts(self) -> Contacts:
        """Contact buffer owned by the selected backend."""
        return self.backend.contacts

    @property
    def vbd_solver(self):
        """VBD implementation owned by the selected backend."""
        return self.backend.vbd_solver

    @property
    def mujoco_solver(self):
        """MuJoCo solver, or ``None`` when the selected backend skips it."""
        return getattr(self.backend, "mujoco_solver", None)

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        self.backend.step(state_in, state_out, control, contacts, dt)

    @override
    def reset(
        self,
        state: State,
        world_mask=None,
        flags: StateFlags | int | None = None,
    ) -> None:
        self.backend.reset(state, world_mask=world_mask, flags=flags)

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        self.backend.notify_model_changed(flags)

    def rebuild_bvh(self, state: State) -> None:
        self.backend.rebuild_bvh(state)

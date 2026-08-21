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
    Contacts,
    Control,
    JointType,
    Model,
    ModelBuilder,
    ModelFlags,
    State,
    StateFlags,
)
from ..coupled.solver_coupled import SolverCoupled
from ..solver import SolverBase
from .collision_pipeline import MJVBDV2SoftContactPipeline
from .full_contact_pipeline import MJVBDV2CollisionPipeline
from .mujoco.solver_mujoco import SolverMuJoCo
from .ownership import MJVBDV2Ownership, resolve_ownership
from .soft_contact_pipeline import MJVBDSoftContactPipeline
from .solver_mjvbd_v2 import SolverMJVBDV2 as _SolverMJVBDV2Coupled
from .solver_mjvbd_v2 import _SolverMJVBDV2Pneumatic
from .vbd.solver_vbd import SolverVBD, _get_pneumatic_counts
from .vbd_soft.solver_vbd import SolverVBD as SolverVBDSoft

__all__ = ["SolverMJVBDV2"]


class _PureMuJoCoBackend(SolverBase):
    """Compact MuJoCo-only backend for dynamic articulation-only scenes."""

    def __init__(
        self,
        model: Model,
        ownership: MJVBDV2Ownership,
        *,
        mujoco_options: Mapping[str, object] | None,
    ) -> None:
        super().__init__(model)
        mujoco_kwargs = dict(mujoco_options or {})
        requested_disable_contacts = mujoco_kwargs.pop("disable_contacts", False)
        requested_mujoco_contacts = mujoco_kwargs.pop("use_mujoco_contacts", True)
        if requested_mujoco_contacts is not True:
            raise ValueError("The pure-MuJoCo MJVBDV2 backend requires use_mujoco_contacts=True")
        mujoco_kwargs["disable_contacts"] = requested_disable_contacts
        mujoco_kwargs["use_mujoco_contacts"] = True

        selected_shapes = tuple(
            [*model.body_shapes.get(-1, ())]
            + [shape for body in ownership.mujoco_bodies for shape in model.body_shapes.get(body, ())]
        )
        self.coupled_solver = SolverCoupled(
            model=model,
            entries=[
                SolverCoupled.Entry(
                    name="mujoco",
                    solver=lambda view: SolverMuJoCo(view, **mujoco_kwargs),
                    bodies=ownership.mujoco_bodies,
                    joints=ownership.mujoco_joints,
                    shapes=selected_shapes,
                )
            ],
        )
        self.contacts = None
        self.vbd_solver = None

    @property
    def mujoco_solver(self) -> SolverMuJoCo:
        """Private MuJoCo joint solver."""
        return self.coupled_solver.solver("mujoco")

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        del contacts
        self.coupled_solver.step(state_in, state_out, control, None, dt)

    @override
    def reset(
        self,
        state: State,
        world_mask=None,
        flags: StateFlags | int | None = None,
    ) -> None:
        self.coupled_solver.reset(state, world_mask=world_mask, flags=flags)

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        self.coupled_solver.notify_model_changed(flags)

    def rebuild_bvh(self, state: State) -> None:
        del state


class _KinematicPassthroughBackend(SolverBase):
    """No-op backend for externally prescribed articulation-only scenes."""

    def __init__(self, model: Model) -> None:
        super().__init__(model)
        self.contacts = None
        self.mujoco_solver = None
        self.vbd_solver = None

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        del state_in, state_out, control, contacts
        if dt <= 0.0:
            raise ValueError("MJVBDV2 timestep dt must be positive")

    def rebuild_bvh(self, state: State) -> None:
        del state


class _PureVBDBackend(SolverBase):
    """Scene-specialized VBD backend used when MuJoCo owns no joints."""

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
        pneumatic_cavity_count, _ = _get_pneumatic_counts(model)
        use_full_vbd = ownership.has_vbd_dynamic_bodies or pneumatic_cavity_count > 0 or model.particle_count == 0
        vbd_solver_type = SolverVBD if use_full_vbd else SolverVBDSoft
        self.vbd_solver = vbd_solver_type(
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
            self.pipeline = MJVBDV2CollisionPipeline(model, **options)
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
        use_full_vbd: bool,
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
        vbd_solver_type = SolverVBD if use_full_vbd else SolverVBDSoft
        self.vbd_solver = vbd_solver_type(
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
        self.pipeline = MJVBDV2CollisionPipeline(view, **options)
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
    """Couple selected articulations one-way to VBD/AVBD objects.

    .. experimental::
        SolverMJVBDV2's public API and behavior may change without prior notice.

    MuJoCo owns the selected articulation joints and link bodies. Every
    unselected free rigid body and every particle are owned by VBD, so cloth,
    tetrahedral soft bodies, springs, pneumatic shells, and VBD rigid bodies
    interact in one VBD/AVBD solve. In dynamic coupled scenes, MuJoCo link poses
    become zero-inverse-mass moving colliders for VBD. Contact impulses from VBD
    do not feed back into MuJoCo.

    The solver inspects the resolved ownership and model topology at
    construction and selects one of six specialized backends. Joint-free scenes
    run only VBD; articulation-only scenes run only MuJoCo or a kinematic
    passthrough; kinematic particle scenes skip MuJoCo; and only dynamic mixed
    scenes construct the coupled backend. The selected path is available through
    :attr:`features`.

    Call :meth:`register_custom_attributes` on the builder before finalizing the
    model. The solver owns the collision pipeline required by the selected
    backend, so passing ``contacts=None`` to :meth:`step` is supported.

    Args:
        model: Model to simulate.
        mujoco_articulations: Articulation IDs assigned to MuJoCo. If neither
            this argument nor ``mujoco_joints`` is provided, selects
            articulations that contain at least one non-free, non-fixed joint.
        mujoco_joints: Closed joint trees assigned to MuJoCo. Mutually exclusive
            with ``mujoco_articulations``.
        joint_mode: Use ``"dynamic"`` to integrate selected joints in MuJoCo or
            ``"kinematic"`` to consume externally prescribed joint and link
            states.
        contact_mode: Use ``"soft"`` for sparse particle-shape contacts,
            ``"full"`` for the complete collision pipeline, or ``"auto"`` to
            choose full contact only when VBD owns dynamic rigid bodies.
        vbd_options: Keyword arguments forwarded to the selected private VBD
            implementation.
        mujoco_options: Keyword arguments forwarded to the private MuJoCo
            implementation when the selected backend uses MuJoCo.
        collision_options: Keyword arguments forwarded to the selected contact
            pipeline. Soft-only paths accept ``soft_contact_margin``.

    Attributes:
        features: Frozen snapshot of the selected backend and active solve
            branches.
        ownership: Resolved, disjoint MuJoCo/VBD entity partition.

    Note:
        MuJoCo sleeping is supported only by the articulation-only
        ``pure_mujoco`` backend. It is rejected by the dynamic coupled backend
        because one-way VBD contacts cannot wake a MuJoCo articulation.
    """

    @dataclass(frozen=True)
    class Features:
        """Scene features used to select and audit the V2 backend.

        Count fields describe the resolved MuJoCo/VBD ownership and present
        particle constraint topology. The ``*_solve_enabled`` fields report
        which solver modules the selected backend will execute.
        """

        backend: Literal[
            "pure_mujoco",
            "kinematic_passthrough",
            "pure_vbd",
            "mjvbd_kinematic_soft",
            "vbd_kinematic_full",
            "coupled",
        ]
        mujoco_joint_count: int
        vbd_body_count: int
        vbd_dynamic_body_count: int
        particle_count: int
        triangle_count: int
        edge_count: int
        tetrahedron_count: int
        spring_count: int
        pneumatic_cavity_count: int
        pneumatic_face_count: int
        mujoco_solve_enabled: bool
        vbd_solve_enabled: bool
        rigid_solve_enabled: bool
        particle_solve_enabled: bool
        triangle_solve_enabled: bool
        bending_solve_enabled: bool
        tetrahedron_solve_enabled: bool
        spring_solve_enabled: bool
        pneumatic_solve_enabled: bool

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
        pneumatic_cavity_count, pneumatic_face_count = _get_pneumatic_counts(model)

        has_vbd_dynamics = self.ownership.has_vbd_dynamic_bodies or model.particle_count > 0

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
        elif not has_vbd_dynamics and joint_mode == "kinematic":
            if mujoco_options:
                raise ValueError("mujoco_options are not used by the kinematic passthrough MJVBDV2 backend")
            backend_name = "kinematic_passthrough"
            backend = _KinematicPassthroughBackend(model)
        elif not has_vbd_dynamics:
            backend_name = "pure_mujoco"
            backend = _PureMuJoCoBackend(
                model,
                self.ownership,
                mujoco_options=mujoco_options,
            )
        elif (
            joint_mode == "kinematic" and not self.ownership.has_vbd_dynamic_bodies and contact_mode in ("auto", "soft")
        ):
            if mujoco_options:
                raise ValueError("mujoco_options are not used by the kinematic soft-only MJVBD backend")
            backend_name = "mjvbd_kinematic_soft"
            backend = _KinematicSoftBackend(
                model,
                use_full_vbd=pneumatic_cavity_count > 0,
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
            coupled_backend_type = _SolverMJVBDV2Pneumatic if pneumatic_cavity_count > 0 else _SolverMJVBDV2Coupled
            backend = coupled_backend_type(
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
            pneumatic_cavity_count=pneumatic_cavity_count,
            pneumatic_face_count=pneumatic_face_count,
            mujoco_solve_enabled=backend_name in ("pure_mujoco", "coupled"),
            vbd_solve_enabled=backend_name in ("pure_vbd", "mjvbd_kinematic_soft", "vbd_kinematic_full", "coupled"),
            rigid_solve_enabled=dynamic_body_count > 0,
            particle_solve_enabled=model.particle_count > 0,
            triangle_solve_enabled=model.tri_count > 0,
            bending_solve_enabled=model.edge_count > 0,
            tetrahedron_solve_enabled=model.tet_count > 0,
            spring_solve_enabled=model.spring_count > 0,
            pneumatic_solve_enabled=pneumatic_cavity_count > 0,
        )

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register attributes needed by every possible V2 backend.

        Args:
            builder: Builder on which to register the solver attributes.
        """
        _SolverMJVBDV2Coupled.register_custom_attributes(builder)

    @property
    def contacts(self) -> Contacts | None:
        """Contact buffer owned by the selected backend."""
        return self.backend.contacts

    @property
    def vbd_solver(self):
        """VBD implementation owned by the selected backend."""
        return getattr(self.backend, "vbd_solver", None)

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
        """Rebuild the selected VBD backend's particle self-contact BVH.

        Args:
            state: State whose particle positions define the rebuilt hierarchy.
        """
        self.backend.rebuild_bvh(state)

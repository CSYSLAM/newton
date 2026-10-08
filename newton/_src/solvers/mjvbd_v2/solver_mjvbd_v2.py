# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint-only MuJoCo coupled to full VBD/AVBD dynamics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import warp as wp

from ...sim import BodyFlags, Model, ModelBuilder, ModelFlags, State, StateFlags
from ..coupled.solver_coupled_proxy import SolverCoupledProxy
from .collision_pipeline import MJVBDV2SoftContactPipeline
from .full_contact_pipeline import MJVBDV2CollisionPipeline
from .mujoco.solver_mujoco import SolverMuJoCo
from .ownership import MJVBDV2Ownership, resolve_ownership
from .vbd.solver_vbd import SolverVBD, _get_pneumatic_counts
from .vbd_soft.solver_vbd import SolverVBD as SolverVBDSoft

__all__ = ["SolverMJVBDV2"]

_PNEUMATIC_STATE_FIELDS = ("volume", "absolute_pressure", "volume_rate", "clamp_flags")

_SURFACE_FAST_VBD_OPTIONS: dict[str, object] = {
    "iterations": 8,
    "particle_displacement_threshold": 5e-6,
    "particle_chebyshev_spectral_radius": 0.8,
    "particle_enable_batched_jacobi": True,
    "particle_jacobi_batch_count": 2,
    "particle_jacobi_relaxation": 1.0,
    "particle_enable_multilevel_correction": True,
    "particle_multilevel_checkpoints": (4,),
    "particle_multilevel_min_residual_reduction": 1.0e-4,
    "particle_multilevel_max_clamp_fraction": 0.5,
    "particle_multilevel_selective_polish_iterations": 0,
    "particle_multilevel_selective_polish_threshold_fraction": 0.001,
    "particle_multilevel_selective_polish_rings": 0,
    "particle_multilevel_selective_polish_max_radius_fraction": 0.001,
    "particle_multilevel_fallback_iterations": 20,
    "particle_enable_surface_cache": True,
    "particle_enable_truncation_cache": True,
    "particle_collision_detection_interval": -1,
}


def _resolve_vbd_options(
    model: Model,
    preset: Literal["surface-fast"] | None,
    overrides: Mapping[str, object] | None,
    *,
    use_external_rigid_surface_path: bool = True,
) -> dict[str, object]:
    """Resolve a high-level VBD policy before applying expert overrides."""
    if preset not in (None, "surface-fast"):
        raise ValueError("vbd_preset must be None or 'surface-fast'")

    options: dict[str, object] = {}
    if preset == "surface-fast":
        requested_deterministic = (overrides or {}).get("deterministic")
        effective_deterministic = (
            wp.config.deterministic if requested_deterministic is None else requested_deterministic
        )
        surface_particle_count = (
            np.unique(np.asarray(model.tri_indices.numpy(), dtype=np.int32)).size if model.tri_count else 0
        )
        supports_fast_surface_path = (
            model.device.is_cuda
            and not model.requires_grad
            and use_external_rigid_surface_path
            and surface_particle_count == model.particle_count
            and model.tet_count == 0
            and model.spring_count == 0
            and _get_pneumatic_counts(model)[0] == 0
            and effective_deterministic == wp.DeterministicMode.NOT_GUARANTEED
        )
        if supports_fast_surface_path:
            options.update(_SURFACE_FAST_VBD_OPTIONS)
        else:
            # Keep the preset semantically safe on CPU, differentiable,
            # deterministic, and volumetric models where its CUDA surface
            # accelerators are unavailable or have not met the accuracy gate.
            options["iterations"] = 20

    options.update(overrides or {})
    return options


@wp.kernel
def _copy_pneumatic_state_kernel(
    local_to_global: wp.array[wp.int32],
    source_volume: wp.array[float],
    source_absolute_pressure: wp.array[float],
    source_volume_rate: wp.array[float],
    source_clamp_flags: wp.array[wp.int32],
    destination_volume: wp.array[float],
    destination_absolute_pressure: wp.array[float],
    destination_volume_rate: wp.array[float],
    destination_clamp_flags: wp.array[wp.int32],
    scatter: bool,
):
    """Gather or scatter all persistent fields for one cavity row."""
    local = wp.tid()
    global_index = local_to_global[local]
    source_index = local
    destination_index = global_index
    if not scatter:
        source_index = global_index
        destination_index = local
    destination_volume[destination_index] = source_volume[source_index]
    destination_absolute_pressure[destination_index] = source_absolute_pressure[source_index]
    destination_volume_rate[destination_index] = source_volume_rate[source_index]
    destination_clamp_flags[destination_index] = source_clamp_flags[source_index]


_TWO_WAY_COUPLING_DEFAULTS: dict[str, object] = {
    "mass_scale": 1.0,
    "mode": "staggered",
    "proxy_relaxation": 0.5,
    "proxy_relaxation_mode": "fixed",
    "proxy_relaxation_min": 0.1,
    "proxy_relaxation_max": 1.0,
    "iterations": 1,
}


def _resolve_coupling_options(
    coupling: Literal["one_way", "two_way"],
    overrides: Mapping[str, object] | None,
) -> dict[str, object]:
    """Validate the MuJoCo/VBD feedback policy and fill two-way defaults."""
    if coupling not in ("one_way", "two_way"):
        raise ValueError("coupling must be 'one_way' or 'two_way'")
    if coupling == "one_way":
        if overrides:
            raise ValueError("coupling_options are only used with coupling='two_way'")
        return {}
    options = dict(_TWO_WAY_COUPLING_DEFAULTS)
    unknown = sorted(set(overrides or {}) - set(options))
    if unknown:
        raise ValueError(f"Unsupported coupling_options keys {unknown}; expected a subset of {sorted(options)}")
    options.update(overrides or {})
    return options


class _OneWayCoupledProxy(SolverCoupledProxy):
    """Proxy composition whose source optionally never receives destination feedback.

    Subclasses set ``_two_way`` before construction. When it is false, MuJoCo
    links become zero-inverse-mass VBD colliders and all feedback is zeroed.
    When it is true, the shared proxy path installs MuJoCo effective inertia on
    the VBD proxies and returns harvested contact wrenches to MuJoCo.
    """

    _two_way: bool = False

    def _proxy_feedback_enabled(self) -> bool:
        return self._two_way

    def _apply_proxy_body_effective_masses(self) -> None:
        if self._two_way:
            super()._apply_proxy_body_effective_masses()
            return
        for mapping in self._proxy_mappings:
            if mapping.proxy_ids_local is None or mapping.proxy_ids_local.shape[0] == 0:
                continue
            destination = self._entries[mapping.dst_name]
            destination.view.disable_body_dynamics(mapping.proxy_ids_local)
            destination.solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

    def _blend_proxy_feedback(self, proxy) -> None:
        if self._two_way:
            super()._blend_proxy_feedback(proxy)
            return
        proxy.coupling_forces.zero_()
        if proxy.coupling_forces_previous is not None:
            proxy.coupling_forces_previous.zero_()


class SolverMJVBDV2(_OneWayCoupledProxy):
    """MuJoCo-joint to VBD coupling, with full coupling among VBD objects.

    MuJoCo owns only the selected articulation bodies and joints. Every
    remaining rigid body and every particle are owned by VBD. With
    ``coupling="one_way"`` the link bodies are synchronized into VBD as
    zero-inverse-mass moving colliders. With ``coupling="two_way"`` they become
    VBD proxies with MuJoCo effective inertia, VBD contact wrenches are returned
    to MuJoCo, and MuJoCo resolves link contacts against static world shapes.
    """

    def __init__(
        self,
        model: Model,
        *,
        mujoco_articulations: Sequence[int] | None = None,
        mujoco_joints: Sequence[int] | None = None,
        joint_mode: Literal["dynamic", "kinematic"] = "dynamic",
        contact_mode: Literal["auto", "soft", "full"] = "auto",
        coupling: Literal["one_way", "two_way"] = "one_way",
        coupling_options: Mapping[str, object] | None = None,
        vbd_options: Mapping[str, object] | None = None,
        mujoco_options: Mapping[str, object] | None = None,
        collision_options: Mapping[str, object] | None = None,
    ) -> None:
        """Create the specialized MuJoCo/VBD solver.

        Args:
            model: Simulation model.
            mujoco_articulations: Articulations owned by MuJoCo.
            mujoco_joints: Joints owned by MuJoCo.
            joint_mode: Whether MuJoCo joints are dynamic or kinematic.
            contact_mode: Particle/rigid contact pipeline selection.
            coupling: ``"one_way"`` or ``"two_way"`` MuJoCo/VBD feedback.
            coupling_options: Two-way proxy overrides, see :data:`_TWO_WAY_COUPLING_DEFAULTS`.
            vbd_options: Options forwarded by the public MJVBDV2 dispatcher. The full VBD backend accepts
                experimental ``enable_cuda_fast_path=True`` for instance-local CUDA scheduling. The matching
                collision option enables cooperative SDF queries; neither option changes physical parameters.
            mujoco_options: Options forwarded to the private MuJoCo solver.
            collision_options: Options forwarded to the contact pipeline.
        """
        if joint_mode not in ("dynamic", "kinematic"):
            raise ValueError("joint_mode must be 'dynamic' or 'kinematic'")
        if contact_mode not in ("auto", "soft", "full"):
            raise ValueError("contact_mode must be 'auto', 'soft', or 'full'")
        resolved_coupling = _resolve_coupling_options(coupling, coupling_options)
        two_way = coupling == "two_way"
        if two_way and joint_mode != "dynamic":
            raise ValueError(
                "coupling='two_way' requires joint_mode='dynamic' so MuJoCo can respond to VBD contact "
                "wrenches; drive the robot through control.joint_target_q instead of prescribing joint_q"
            )
        # Read by the feedback hooks during SolverCoupledProxy construction.
        self._two_way = two_way
        self.coupling = coupling

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
        requested_sleeping = mujoco_kwargs.get("enable_sleeping")
        if requested_sleeping is None:
            mujoco_namespace = getattr(model, "mujoco", None)
            sleeping_attribute = (
                None if mujoco_namespace is None else getattr(mujoco_namespace, "enable_sleeping", None)
            )
            requested_sleeping = False if sleeping_attribute is None else bool(sleeping_attribute.numpy()[0])
        if requested_sleeping:
            raise ValueError(
                "enable_sleeping=True is unsupported by the coupled MJVBDV2 backend because "
                "VBD contacts cannot wake MuJoCo bodies"
            )
        mujoco_kwargs["enable_sleeping"] = False
        # One-way coupling leaves MuJoCo contact-free so it cannot duplicate
        # VBD contacts. Two-way coupling keeps MuJoCo contacts by default: the
        # MuJoCo view holds only the selected links and static world shapes,
        # so it resolves link-vs-world and self contacts that VBD never
        # returns as feedback.
        requested_disable_contacts = mujoco_kwargs.pop("disable_contacts", not two_way)
        if not two_way and requested_disable_contacts is not True:
            raise ValueError("One-way MJVBDV2 requires mujoco_options['disable_contacts']=True")
        requested_mujoco_contacts = mujoco_kwargs.pop("use_mujoco_contacts", True)
        if requested_mujoco_contacts is not True:
            raise ValueError("MJVBDV2 currently requires mujoco_options['use_mujoco_contacts']=True")
        mujoco_kwargs["disable_contacts"] = bool(requested_disable_contacts)
        mujoco_kwargs["use_mujoco_contacts"] = True

        vbd_kwargs = dict(vbd_options or {})
        # Two-way proxies need VBD-integrated bodies: the proxy is displaced by
        # contact inside the VBD solve, and the contact-force harvest uses the
        # AVBD rigid history.
        external_rigid = not ownership.has_vbd_dynamic_bodies and not two_way
        requested_external = vbd_kwargs.pop("integrate_with_external_rigid_solver", external_rigid)
        if bool(requested_external) != external_rigid:
            required = "True" if external_rigid else "False"
            raise ValueError(
                "MJVBDV2 selects the VBD rigid integration mode from entity ownership and coupling; "
                f"integrate_with_external_rigid_solver must be {required} for this model"
            )
        vbd_kwargs["integrate_with_external_rigid_solver"] = external_rigid
        vbd_kwargs["external_rigid_state_from_input"] = external_rigid
        vbd_kwargs["one_way_proxy_bodies"] = not two_way
        if two_way:
            # Penalty contacts are separated by the end of the VBD solve, so
            # re-evaluating them there reports almost no force on light links
            # such as gripper fingers. The proxy momentum change does not.
            vbd_kwargs.setdefault("proxy_body_feedback", "momentum")
        pneumatic_cavity_count, _ = _get_pneumatic_counts(model)
        vbd_solver_type = SolverVBDSoft if external_rigid and pneumatic_cavity_count == 0 else SolverVBD
        if vbd_solver_type is not SolverVBDSoft and vbd_kwargs.pop("particle_displacement_threshold", 0.0) != 0.0:
            raise ValueError(
                "particle_displacement_threshold requires the particle solver with external rigid colliders"
            )

        collision_kwargs = dict(collision_options or {})
        soft_contact_margin = float(collision_kwargs.get("soft_contact_margin", 0.0))
        if soft_contact_margin < 0.0:
            raise ValueError("collision_options['soft_contact_margin'] must be non-negative")
        if self.contact_mode == "soft" and collision_kwargs.get("enable_rigid_soft_full_surface_contact", False):
            raise ValueError("contact_mode='soft' does not support full-surface rigid-soft contacts")
        if two_way and collision_kwargs.get("enable_rigid_soft_full_surface_contact", False):
            raise ValueError(
                "coupling='two_way' does not support full-surface rigid-soft contacts because edge/face "
                "contact forces are not harvested onto MuJoCo proxies"
            )

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
            return MJVBDV2CollisionPipeline(view, **options)

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
        if two_way:
            proxy_kwargs = {key: value for key, value in resolved_coupling.items() if key != "iterations"}
            iterations = int(resolved_coupling["iterations"])
        else:
            proxy_kwargs = {"mode": "staggered", "proxy_relaxation": 0.0}
            iterations = 1
        proxy_config = SolverCoupledProxy.Config(
            proxies=[
                SolverCoupledProxy.Proxy(
                    source="mujoco",
                    destination="vbd",
                    bodies=ownership.mujoco_bodies,
                    collision_pipeline=make_collision_pipeline,
                    collide_interval=1,
                    **proxy_kwargs,
                )
            ],
            iterations=iterations,
        )
        super().__init__(model=model, entries=entries, coupling=proxy_config)

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


class _SolverMJVBDV2Pneumatic(SolverMJVBDV2):
    """Coupled backend that additionally owns pneumatic state transfer."""

    def _copy_pneumatic_state(
        self,
        source: State,
        destination: State,
        *,
        scatter: bool,
    ) -> None:
        """Copy VBD-owned cavity state through the compact entry mapping."""
        entry = self._entries["vbd"]
        local_to_global = entry.attribute_local_to_global["pneumatic:cavity"]
        source_namespace = getattr(source, "pneumatic", None)
        destination_namespace = getattr(destination, "pneumatic", None)
        if source_namespace is None or destination_namespace is None:
            raise ValueError("Pneumatic MJVBDV2 state must be created with model.state().")
        wp.launch(
            kernel=_copy_pneumatic_state_kernel,
            dim=local_to_global.shape[0],
            inputs=[
                local_to_global,
                *(getattr(source_namespace, field) for field in _PNEUMATIC_STATE_FIELDS),
                *(getattr(destination_namespace, field) for field in _PNEUMATIC_STATE_FIELDS),
                scatter,
            ],
            device=self.model.device,
        )

    def _distribute_state(
        self,
        state_in: State,
        *,
        dt: float = 0.0,
        iteration_restart: bool = False,
    ) -> None:
        """Distribute core state and VBD-owned pneumatic history."""
        super()._distribute_state(state_in, dt=dt, iteration_restart=iteration_restart)
        self._copy_pneumatic_state(state_in, self._entries["vbd"].state_0, scatter=False)

    def _distribute_reset_state(self, state_in: State, world_mask: wp.array[wp.bool]) -> None:
        """Seed VBD cavity history before the shared masked-reset path."""
        super()._distribute_reset_state(state_in, world_mask)
        self._copy_pneumatic_state(state_in, self._entries["vbd"].state_0, scatter=False)

    def _sync_entry_reset_state(
        self,
        entry,
        world_mask: wp.array[wp.bool],
        flags: StateFlags | int | None,
    ) -> None:
        """Keep both VBD ping-pong states consistent after cavity reset."""
        super()._sync_entry_reset_state(entry, world_mask, flags)
        if entry.name != "vbd" or entry.state_1 is None or entry.state_1 is entry.state_0:
            return
        source_namespace = entry.state_0.pneumatic
        destination_namespace = entry.state_1.pneumatic
        for field in _PNEUMATIC_STATE_FIELDS:
            wp.copy(dest=getattr(destination_namespace, field), src=getattr(source_namespace, field))

    def _reconcile_reset_state(
        self,
        state_out: State,
        world_mask: wp.array[wp.bool],
        flags: StateFlags | int | None,
    ) -> None:
        """Scatter reset VBD cavity history back to the parent state."""
        super()._reconcile_reset_state(state_out, world_mask, flags)
        self._copy_pneumatic_state(self._entries["vbd"].state_0, state_out, scatter=True)

    def _reconcile_state(self, state_out: State) -> None:
        """Reconcile core state and VBD-owned pneumatic observables."""
        super()._reconcile_state(state_out)
        entry_output = self._entries["vbd"].state_1
        if entry_output is not None:
            self._copy_pneumatic_state(entry_output, state_out, scatter=True)

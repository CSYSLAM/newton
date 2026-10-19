# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private MuJoCo articulation and MPM environment coupling."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import warp as wp

from newton._src.sim import Model, ModelBuilder, ModelFlags
from newton._src.sim.articulation import eval_ik
from newton._src.solvers.mjmpm.base import SolverBase
from newton._src.solvers.mjmpm.coupling.solver_coupled_proxy import SolverCoupledProxy
from newton._src.solvers.mjmpm.environment import Environment
from newton._src.solvers.mjmpm.mpm.solver_implicit_mpm import SolverImplicitMPM
from newton._src.solvers.mjmpm.ownership import resolve_ownership


@dataclass(frozen=True)
class _Features:
    backend: str
    mujoco_body_count: int
    mpm_body_count: int
    particle_count: int


class SolverMJMPM(SolverBase):
    """Experimental independent MuJoCo/MPM solver with explicit proxy feedback.

    MuJoCo owns complete articulation trees. Free rigid bodies and material
    particles are integrated by the private MPM environment. Two-way feedback
    currently arrives on the next step. The rigid contact solver is a small-scene
    reference implementation; full articulated contact response is not yet enabled.

    Args:
        model: Newton model with registered solver attributes.
        mujoco_joints: Complete joint trees assigned to MuJoCo; inferred if omitted.
        coupling: Whether articulation contacts receive environment feedback.
        mpm_options: Private implicit MPM configuration fields. The environment
            defaults to particle-based contact (``collider_basis="pic"``).
            Graph capture requires an explicit per-cell particle capacity,
            for example ``collider_basis="pic27"``, in addition to fixed grid capacity.
        mujoco_options: Private MuJoCo backend configuration fields.
        rigid_iterations: Projected rigid contact sweeps per step.
        rigid_contact_hard: Use hard rigid contact constraints. If False, use
            implicit spring/damper contacts from the shapes' ``ke`` and ``kd``.
        mass_scale: Virtual articulation proxy inertia multiplier.
        relaxation: Feedback mixing factor in [0, 1].
    """

    class Coupling(Enum):
        """Feedback policy for articulation/environment contacts."""

        ONE_WAY = "one_way"
        TWO_WAY = "two_way"

    def __init__(
        self,
        model: Model,
        *,
        mujoco_joints: Sequence[int] | None = None,
        coupling: Coupling = Coupling.TWO_WAY,
        mpm_options: Mapping[str, object] | None = None,
        mujoco_options: Mapping[str, object] | None = None,
        rigid_iterations: int = 20,
        rigid_contact_hard: bool = True,
        mass_scale: float = 1.0,
        relaxation: float = 1.0,
    ):
        super().__init__(model)
        if not isinstance(coupling, self.Coupling):
            raise TypeError("coupling must be a SolverMJMPM.Coupling value")
        if isinstance(rigid_iterations, bool) or not isinstance(rigid_iterations, int) or rigid_iterations < 1:
            raise ValueError("rigid_iterations must be a positive integer")
        if not math.isfinite(mass_scale) or mass_scale <= 0.0:
            raise ValueError("mass_scale must be finite and positive")
        if not math.isfinite(relaxation) or not 0.0 <= relaxation <= 1.0:
            raise ValueError("relaxation must be finite and in [0, 1]")
        self.ownership = resolve_ownership(model, mujoco_joints=mujoco_joints)
        self.coupling = coupling
        owned = self.ownership
        if model.joint_count:
            owned_joints = set(owned.mujoco_joints)
            articulation_ids = model.joint_articulation.numpy()
            environment_articulations = sorted(
                {
                    int(articulation_ids[j])
                    for j in range(model.joint_count)
                    if j not in owned_joints and articulation_ids[j] >= 0
                }
            )
        else:
            environment_articulations = []
        self._environment_articulations = wp.array(environment_articulations, dtype=int, device=model.device)

        def environment(view):
            return Environment(
                view,
                mpm_options=mpm_options,
                rigid_iterations=rigid_iterations,
                rigid_contact_hard=rigid_contact_hard,
            )

        if not owned.mujoco_joints:
            if mujoco_options:
                raise ValueError("mujoco_options require an articulation")
            self.backend = environment(model)
            self.environment = self.backend
            backend_name = "pure_mpm"
        else:
            from newton._src.solvers.mjmpm.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415

            options = dict(mujoco_options or {})
            if options.get("enable_sleeping", False):
                raise ValueError("Coupled MJMPM does not support MuJoCo sleeping")
            options["enable_sleeping"] = False
            options.setdefault("use_mujoco_contacts", True)
            self.backend = SolverCoupledProxy(
                model=model,
                entries=[
                    SolverCoupledProxy.Entry(
                        name="mujoco",
                        solver=lambda v: SolverMuJoCo(v, **options),
                        bodies=owned.mujoco_bodies,
                        joints=owned.mujoco_joints,
                    ),
                    SolverCoupledProxy.Entry(
                        name="mpm", solver=environment, bodies=owned.mpm_bodies, particles=owned.mpm_particles
                    ),
                ],
                coupling=SolverCoupledProxy.Config(
                    proxies=[
                        SolverCoupledProxy.Proxy(
                            source="mujoco",
                            destination="mpm",
                            bodies=owned.mujoco_bodies,
                            mode="lagged",
                            mass_scale=mass_scale,
                            proxy_relaxation=relaxation if coupling == self.Coupling.TWO_WAY else 0.0,
                            collision_pipeline=lambda view: None,
                        )
                    ],
                    iterations=1,
                ),
            )
            self.environment = self.backend.solver("mpm")
            if coupling == self.Coupling.ONE_WAY:
                entry = self.backend._entries["mpm"]
                proxy_ids = self.backend._proxy_mappings[0].proxy_ids_local
                entry.view.disable_body_dynamics(proxy_ids)
                entry.solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
            backend_name = "coupled"
        self.features = _Features(backend_name, len(owned.mujoco_bodies), len(owned.mpm_bodies), model.particle_count)

    @classmethod
    def register_custom_attributes(cls, builder: ModelBuilder) -> None:
        """Register material state and MuJoCo attributes before model finalization."""
        SolverImplicitMPM.register_custom_attributes(builder)
        from newton._src.solvers.mjmpm.mujoco.solver_mujoco import SolverMuJoCo  # noqa: PLC0415

        SolverMuJoCo.register_custom_attributes(builder)

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        """Advance one physical step of duration ``dt`` [s]."""
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if contacts is not None:
            raise ValueError("MJMPM owns collision detection; pass contacts=None")
        self.validate_observables(observables, contacts)
        if self.model.device.is_capturing and not self.environment.capture_ready:
            raise ValueError(
                "MPM graph capture requires grid_type='fixed', positive max_active_cell_count, "
                "and bounded particle bases (for example collider_basis='pic27')"
            )
        self.backend.step(state_in, state_out, control, None, dt)
        if self._environment_articulations.shape[0]:
            eval_ik(
                self.model, state_out, state_out.joint_q, state_out.joint_qd, indices=self._environment_articulations
            )

    def reset(self, state, world_mask=None, flags=None):
        """Reset selected worlds and their coupling/material history."""
        self.backend.reset(state, world_mask=world_mask, flags=flags)

    def notify_model_changed(self, flags):
        """Refresh private backend model properties."""
        self.backend.notify_model_changed(flags)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private MuJoCo articulation and MPM environment coupling."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
import warp as wp

from newton._src.sim import Model, ModelBuilder, ModelFlags
from newton._src.sim.articulation import eval_ik
from newton._src.solvers.mjmpm.base import SolverBase
from newton._src.solvers.mjmpm.coupling.articulated import LinkMobility
from newton._src.solvers.mjmpm.coupling.same_step import SameStepIteration
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
    """Experimental independent MuJoCo/MPM solver with proxy contact feedback.

    MuJoCo owns complete articulation trees. Free rigid bodies and material
    particles are integrated by the private MPM environment. With one coupling
    pass and proxy inertia, two-way feedback arrives on the next step.
    ``coupling_iterations`` converges the feedback within the step, and
    ``articulated_response`` lets MPM-side contacts move the links through the
    MuJoCo link mobility so that one pass already acts within the step.

    Args:
        model: Newton model with registered solver attributes.
        mujoco_joints: Complete joint trees assigned to MuJoCo; inferred if omitted.
        coupling: Whether articulation contacts receive environment feedback.
        mpm_options: Private implicit MPM configuration fields. The environment
            defaults to particle-based contact (``collider_basis="pic"``),
            whose per-cell capacity is sized at construction and grown by
            uncaptured steps. Graph capture
            requires a positive ``max_active_cell_count`` with a fixed or
            rebuildable sparse grid; see :meth:`check_capacity_status`.
        mujoco_options: Private MuJoCo backend configuration fields.
        rigid_iterations: Projected rigid contact sweeps per step.
        rigid_contact_hard: Use hard rigid contact constraints. If False, use
            implicit spring/damper contacts from the shapes' ``ke`` and ``kd``.
        mass_scale: Virtual articulation proxy inertia multiplier. With
            several coupling iterations it only changes convergence speed.
        relaxation: Feedback mixing factor in [0, 1]. With
            ``relaxation_mode="aitken"`` it is the initial factor of each step.
        relaxation_mode: ``"fixed"`` mixes feedback with ``relaxation``;
            ``"aitken"`` adapts the factor from consecutive residuals within
            a step.
        coupling_iterations: MuJoCo/MPM passes per step. One pass applies
            contact feedback on the next step. More passes repeat the step
            from its start state with the latest feedback so both domains
            agree on the link motion within the step.
        coupling_tolerance: Interface velocity residual [m/s, rad/s] below
            which uncaptured steps stop iterating early. Zero runs every pass.
        articulated_response: Let MPM-side contacts move articulation links
            through the MuJoCo link mobility ``J H^-1 J^T`` instead of
            independent proxy inertia, so an impulse on one link also moves
            the links it is jointed to. MuJoCo then re-solves the step with
            the harvested feedback, so contact feedback acts in the same step
            even with one coupling pass. Requires two-way coupling and the
            MuJoCo Warp backend.
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
        relaxation_mode: str = "fixed",
        coupling_iterations: int = 1,
        coupling_tolerance: float = 0.0,
        articulated_response: bool = False,
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
        if relaxation_mode not in ("fixed", "aitken"):
            raise ValueError("relaxation_mode must be 'fixed' or 'aitken'")
        if isinstance(coupling_iterations, bool) or not isinstance(coupling_iterations, int) or coupling_iterations < 1:
            raise ValueError("coupling_iterations must be a positive integer")
        if not math.isfinite(coupling_tolerance) or coupling_tolerance < 0.0:
            raise ValueError("coupling_tolerance must be finite and nonnegative")
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

        self._iteration = None
        two_way_settings = (
            coupling_iterations != 1 or coupling_tolerance != 0.0 or relaxation_mode != "fixed" or articulated_response
        )
        if two_way_settings and (not owned.mujoco_joints or coupling != self.Coupling.TWO_WAY):
            raise ValueError(
                "coupling_iterations, coupling_tolerance, relaxation_mode, and articulated_response require "
                "two-way coupling with an articulation"
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
            if (coupling_iterations > 1 or articulated_response) and options.get("update_data_interval", 1) != 1:
                raise ValueError(
                    "coupling_iterations > 1 and articulated_response require mujoco_options update_data_interval=1"
                )
            if articulated_response and options.get("use_mujoco_cpu", False):
                raise ValueError("articulated_response requires the MuJoCo Warp backend (use_mujoco_cpu=False)")
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
                            proxy_relaxation_mode=relaxation_mode,
                            collision_pipeline=lambda view: None,
                        )
                    ],
                    iterations=coupling_iterations,
                ),
            )
            self.environment = self.backend.solver("mpm")
            if coupling == self.Coupling.TWO_WAY:
                mujoco = self.backend.solver("mujoco")
                if articulated_response:
                    mobility = LinkMobility(mujoco)
                    proxy = self.backend._proxy_mappings[0]
                    source_to_proxy = proxy.source_local_to_proxy_local.numpy()
                    link_body = mobility.link_body.numpy()
                    self.environment.enable_articulated_response(
                        mobility, np.where(link_body >= 0, source_to_proxy[np.maximum(link_body, 0)], -1)
                    )
                self._iteration = SameStepIteration(
                    model.device,
                    iterations=coupling_iterations,
                    tolerance=coupling_tolerance,
                    solvers=(mujoco, self.environment),
                    resolve_sources=coupling_iterations > 1 or articulated_response,
                )
                self.backend.iteration_observer = self._iteration
            if coupling == self.Coupling.ONE_WAY:
                entry = self.backend._entries["mpm"]
                proxy_ids = self.backend._proxy_mappings[0].proxy_ids_local
                entry.view.disable_body_dynamics(proxy_ids)
                entry.solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
            backend_name = "coupled"
        self.features = _Features(backend_name, len(owned.mujoco_bodies), len(owned.mpm_bodies), model.particle_count)

    @property
    def coupling_residuals(self) -> wp.array | None:
        """Interface velocity residuals of the latest step, or ``None`` without two-way coupling.

        Shape ``(coupling_iterations, 2)``: per pass, the largest linear
        [m/s] and angular [rad/s] mismatch between the MPM proxy and MuJoCo
        link end velocities implied by that pass's feedback update. Rows of
        passes skipped by an early exit are zero.
        """
        return None if self._iteration is None else self._iteration.residuals

    @property
    def coupling_passes(self) -> int:
        """Passes run in the latest uncaptured step (every pass under graph capture)."""
        return 0 if self._iteration is None else self._iteration.passes

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
                "MPM graph capture requires a positive max_active_cell_count with grid_type='fixed' "
                "or a rebuildable sparse grid (grid_padding=0)"
            )
        self.backend.step(state_in, state_out, control, None, dt)
        if self._environment_articulations.shape[0]:
            eval_ik(
                self.model, state_out, state_out.joint_q, state_out.joint_qd, indices=self._environment_articulations
            )

    def check_capacity_status(self):
        """Raise if graph replay exceeded a reserved MPM capacity.

        Captured steps cannot grow the sparse-grid or per-cell particle
        capacities sized during uncaptured warm-up steps. Call this outside
        capture after replay; it synchronizes the device.

        Raises:
            RuntimeError: If a sparse-grid rebuild failed or a grid cell held
                more particles than the particle basis capacity.
        """
        self.environment.check_capacity_status()

    def reset(self, state, world_mask=None, flags=None):
        """Reset selected worlds and their coupling/material history."""
        self.backend.reset(state, world_mask=world_mask, flags=flags)

    def notify_model_changed(self, flags):
        """Refresh private backend model properties."""
        self.backend.notify_model_changed(flags)

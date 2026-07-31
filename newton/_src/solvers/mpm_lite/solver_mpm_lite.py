# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Newton adapter for the MPM Lite reference solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import warp as wp

import newton

from ...core.types import override
from ...sim import ModelFlags, StateFlags
from ..solver import SolverBase
from ._vendor.engine.solver3d import MPMSolver
from ._vendor.engine.types import Material

__all__ = ["SolverMPMLite"]


@wp.kernel
def _particle_volumes_from_mass(
    mass: wp.array[float],
    density: float,
    volume: wp.array[float],
):
    volume[wp.tid()] = mass[wp.tid()] / density


class SolverMPMLite(SolverBase):
    """Implicit or explicit MPM Lite simulation through the Newton solver API.

    The initial integration supports one Newton world and one uniform material.
    Newton owns particle positions, velocities, and MPM Lite history through
    ``State.mpm_lite``; the sparse grid remains private to the solver.
    """

    @dataclass
    class Config:
        """Configuration for :class:`SolverMPMLite`."""

        grid_size: tuple[int, int, int] = (128, 128, 128)
        """Number of grid nodes in each axis."""
        voxel_size: float = 0.01
        """Grid-node spacing [m]."""
        solver_type: Literal["lite_explicit", "lite_implicit"] = "lite_implicit"
        """MPM Lite stepping method."""
        max_iterations: int = 50
        """Maximum implicit iterations per step."""
        tolerance: float = 1.0e-4
        """Implicit velocity-update tolerance [m/s]."""
        flip_ratio: float = 0.9
        """FLIP contribution in the particle transfer."""
        enable_apic: bool = True
        """Enable APIC affine particle transfer."""
        density: float = 1000.0
        """Uniform material density [kg/m^3]."""
        young_modulus: float = 5.0e6
        """Uniform Young's modulus [Pa]."""
        poisson_ratio: float = 0.3
        """Uniform Poisson ratio."""
        yield_stress: float = 1.0e4
        """Uniform Von Mises yield stress [Pa]."""

    @classmethod
    def register_custom_attributes(cls, builder: newton.ModelBuilder) -> None:
        """Register MPM Lite per-particle history on Newton states.

        This method must be called before particles are added to *builder*.
        """
        identity = wp.mat33(np.eye(3, dtype=np.float32))
        for name, dtype, default in (
            ("particle_F", wp.mat33, identity),
            ("particle_G", wp.mat33, wp.mat33(0.0)),
            ("particle_dlogJ", wp.float32, 0.0),
        ):
            builder.add_custom_attribute(
                newton.ModelBuilder.CustomAttribute(
                    name=name,
                    frequency=newton.Model.AttributeFrequency.PARTICLE,
                    assignment=newton.Model.AttributeAssignment.STATE,
                    dtype=dtype,
                    default=default,
                    namespace="mpm_lite",
                )
            )

    def __init__(self, model: newton.Model, config: Config | None = None):
        """Create an MPM Lite solver for *model*.

        Args:
            model: Newton model containing MPM particles.
            config: MPM Lite configuration. Defaults to :class:`Config`.

        Raises:
            ValueError: If the model is empty, multi-world, lacks MPM Lite
                state history, or the grid configuration is invalid.
        """
        super().__init__(model)
        self.config = config or self.Config()
        if model.particle_count == 0:
            raise ValueError("SolverMPMLite requires at least one particle.")
        if model.world_count != 1:
            raise ValueError("SolverMPMLite currently supports exactly one Newton world.")
        if len(self.config.grid_size) != 3 or any(size < 2 for size in self.config.grid_size):
            raise ValueError("Config.grid_size must contain three integers greater than one.")
        if self.config.voxel_size <= 0.0:
            raise ValueError("Config.voxel_size must be positive.")
        if self.config.density <= 0.0:
            raise ValueError("Config.density must be positive.")

        initial_state = model.state()
        if not hasattr(initial_state, "mpm_lite"):
            raise ValueError(
                "MPM Lite state attributes are missing. Call "
                "SolverMPMLite.register_custom_attributes(builder) before finalizing the model."
            )

        gravity = float(model.gravity.numpy()[0][2])
        self._solver = MPMSolver(
            grid_size=self.config.grid_size,
            dx=self.config.voxel_size,
            device=str(model.device),
            gravity=gravity,
            solver_type=self.config.solver_type,
            enable_apic=self.config.enable_apic,
            flip_ratio=self.config.flip_ratio,
            n_psi=1,
        )
        self._particle_volume = wp.empty(model.particle_count, dtype=float, device=model.device)
        wp.launch(
            _particle_volumes_from_mass,
            dim=model.particle_count,
            inputs=[model.particle_mass, self.config.density, self._particle_volume],
            device=model.device,
        )
        self._solver.n_ptc = model.particle_count
        self._solver.n_ptc_wp.fill_(model.particle_count)
        self._solver.ptc_m = model.particle_mass
        self._solver.ptc_vol0 = self._particle_volume
        self._solver.ptc_k = wp.zeros(model.particle_count, dtype=int, device=model.device)
        self._solver.add_material(
            material_k=0,
            material=Material.vonmises,
            E=self.config.young_modulus,
            nu=self.config.poisson_ratio,
            yield_stress=self.config.yield_stress,
        )
        self._particle_count = model.particle_count
        self._bind_state(initial_state)

    @property
    def voxel_size(self) -> float:
        """Grid-node spacing [m]."""
        return self.config.voxel_size

    def paint_boundary(
        self,
        node_indices: np.ndarray,
        boundary_type: np.ndarray,
        normals: np.ndarray | None = None,
        velocities: np.ndarray | None = None,
    ) -> None:
        """Apply grid-node boundary conditions.

        Args:
            node_indices: Grid node coordinates, shape ``[count, 3]``.
            boundary_type: Boundary modes: 1 for sticky or 2 for slippery.
            normals: Unit normals for slippery nodes, shape ``[count, 3]``.
            velocities: Prescribed node velocities [m/s], shape ``[count, 3]``.
        """
        self._solver.paint_boundary(node_indices, boundary_type, normals, velocities)

    def paint_halfspace_boundary(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        velocities: np.ndarray,
        boundary_type: np.ndarray,
    ) -> None:
        """Apply moving half-space boundary conditions.

        Args:
            points: Points on the half-space planes [m], shape ``[count, 3]``.
            normals: Half-space outward normals, shape ``[count, 3]``.
            velocities: Prescribed plane velocities [m/s], shape ``[count, 3]``.
            boundary_type: Boundary modes: 1 for sticky or 2 for slippery.
        """
        self._solver.paint_hf_boundary(points, normals, velocities, boundary_type)

    def _bind_state(self, state: newton.State) -> None:
        if state.particle_count != self._particle_count or not hasattr(state, "mpm_lite"):
            raise ValueError("State is incompatible with this SolverMPMLite instance.")
        self._solver.ptc_x = state.particle_q
        self._solver.ptc_v = state.particle_qd
        self._solver.ptc_F = state.mpm_lite.particle_F
        self._solver.ptc_G = state.mpm_lite.particle_G
        self._solver.ptc_dlogJ = state.mpm_lite.particle_dlogJ

    @override
    def step(
        self,
        state_in: newton.State,
        state_out: newton.State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        dt: float,
    ) -> None:
        """Advance MPM Lite by one time step.

        Args:
            state_in: Input particle state.
            state_out: Output particle state.
            control: Unused.
            contacts: Unused; use :meth:`paint_boundary` or
                :meth:`paint_halfspace_boundary` for the initial integration.
            dt: Time step duration [s].
        """
        del control, contacts
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        for destination, source in (
            (state_out.particle_q, state_in.particle_q),
            (state_out.particle_qd, state_in.particle_qd),
            (state_out.mpm_lite.particle_F, state_in.mpm_lite.particle_F),
            (state_out.mpm_lite.particle_G, state_in.mpm_lite.particle_G),
            (state_out.mpm_lite.particle_dlogJ, state_in.mpm_lite.particle_dlogJ),
        ):
            if destination.ptr != source.ptr:
                wp.copy(destination, source)
        self._bind_state(state_out)
        self._solver.set_dt(dt)
        self._solver.step(
            max_iters=self.config.max_iterations, v_tol=self.config.tolerance, print_every=0, verbose=False
        )

    @override
    def reset(
        self,
        state: newton.State,
        world_mask: wp.array | None = None,
        flags: StateFlags | int | None = None,
    ) -> None:
        """Reset MPM Lite particle history for the single supported world."""
        del flags
        if world_mask is not None:
            raise NotImplementedError("SolverMPMLite does not yet support selective world resets.")
        self._bind_state(state)
        self._solver.ptc_F.fill_(wp.mat33(np.eye(3, dtype=np.float32)))
        self._solver.ptc_G.zero_()
        self._solver.ptc_dlogJ.zero_()
        self._solver.sim_steps = 0
        self._solver.sim_time = 0.0

    @override
    def notify_model_changed(self, flags: ModelFlags | int) -> None:
        """Refresh model-derived solver parameters after Newton model changes."""
        if flags & ModelFlags.MODEL_PROPERTIES:
            self._solver.gravity = float(self.model.gravity.numpy()[0][2])

    @override
    def update_contacts(self, contacts: newton.Contacts, state: newton.State | None = None) -> None:
        """Leave Newton contact buffers unchanged because MPM Lite owns grid boundaries."""
        del contacts, state

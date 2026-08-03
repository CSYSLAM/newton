# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
D3Q27 lattice-Boltzmann fluid solver coupled to Newton rigid bodies.

``SolverLBM`` is an independent fluid backend that advances a D3Q27 lattice per
world and applies hydrodynamic wrenches to the model's rigid bodies. Rigid-body
dynamics are delegated to another solver (typically :class:`SolverMuJoCo`):
``step`` extracts body poses from ``state_in.body_q``, updates the immersed-solid
meshes, runs one stream/collide + boundary-condition + moment-swap advance, and
writes the lattice forces/torques into ``state_in.body_f`` in physical units
(world-frame COM wrench) so the rigid solver consumes them.

The solver is a port of the D3Q27 backend from the Open HOME-LBM project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import warp as wp

import newton
from newton._src.geometry.types import GeoType
from newton._src.sim.model import Model
from newton._src.sim.state import State
from newton._src.solvers.solver import SolverBase

from .lbm_core import HomeFlow
from .lbm_coupling_kernels import (
    convert_and_update_solid_batch_3d,
    extract_body_states_from_body_q,
    extract_forces_torques_physical_3d,
    fill_body_f_kernel,
)
from .lbm_flow import (
    set_boundary_velocity_3d_kernel,
    set_uniform_flow_3d_kernel,
)
from .lbm_geometry import generate_shape_mesh
from .lbm_kernels import (
    InitBoundary3D,
    InitFlow3D,
    Swap_Mom_3D,
    apply_bc_3d,
    get_u_projection_front_3d,
    get_u_projection_side_3d,
    get_u_projection_topdown_3d,
    get_vorticity_projection_front_3d,
    get_vorticity_projection_side_3d,
    get_vorticity_projection_topdown_3d,
    init_force_3d_batch,
    stream_and_collide_3d,
)


def _quat_rot_mat(q: np.ndarray) -> np.ndarray:
    """Build a (3, 3) rotation matrix from a (qx, qy, qz, qw) quaternion."""
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_points(tf7: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a warp transform stored as (px, py, pz, qx, qy, qz, qw) to points."""
    tf7 = np.asarray(tf7, dtype=np.float64)
    p = tf7[:3]
    q = tf7[3:7]
    return (points.astype(np.float64) @ _quat_rot_mat(q).T + p).astype(np.float32)


@wp.kernel
def vorticity_points_kernel(
    flow: HomeFlow,
    positions: wp.array3d(dtype=wp.vec3),
    colors: wp.array3d(dtype=wp.vec3),
    radii: wp.array3d(dtype=wp.float32),
    stride_x: int,
    stride_y: int,
    stride_z: int,
):
    """Fill a downsampled signed-vorticity point cloud for ``viewer.log_points``."""
    i, j, k = wp.tid()
    x = wp.min(i * stride_x, flow.nx - 1)
    y = wp.min(j * stride_y, flow.ny - 1)
    z = wp.min(k * stride_z, flow.nz - 1)
    u = flow.u
    inv = 1.0 / flow.grid_length
    dvdx = (u[wp.min(x + 1, flow.nx - 1), y, z][1] - u[wp.max(x - 1, 0), y, z][1]) * inv * 0.5
    dudy = (u[x, wp.min(y + 1, flow.ny - 1), z][0] - u[x, wp.max(y - 1, 0), z][0]) * inv * 0.5
    vort_z = dvdx - dudy
    c = wp.clamp(0.5 + 0.5 * vort_z / wp.max(wp.abs(vort_z) + 1.0e-6, 1.0e-4), 0.0, 1.0)
    positions[i, j, k] = wp.vec3(float(x), float(y), float(z))
    colors[i, j, k] = wp.vec3(c, 0.5, 1.0 - c)
    radii[i, j, k] = 0.6


class SolverLBM(SolverBase):
    """Immersed-boundary D3Q27 lattice-Boltzmann fluid solver.

    The solver owns a D3Q27 fluid lattice per world and a set of immersed solid
    meshes resolved against the model's rigid bodies. ``step`` reads body poses
    from ``state_in.body_q``, advances the fluid one lattice step, and writes the
    resulting hydrodynamic forces/torques into ``state_in.body_f`` in physical
    units. Rigid-body advance is delegated to a separate solver.

    Physics conversion follows the Open HOME-LBM convention:

    .. math::

        F_{phys} = F_{lbm} * rho * dx^4 / dt^2, \\qquad
        \\tau_{phys} = \\tau_{lbm} * rho * dx^5 / dt^2

    where ``dx = 1 / (lbm_scale * nx)`` is the physical grid spacing and ``dt``
    the rigid-body timestep passed to :meth:`step`.
    """

    @dataclass
    class Config:
        """Configuration for :class:`SolverLBM`."""

        nx: int = 128
        """Lattice resolution along x."""
        ny: int = 128
        """Lattice resolution along y."""
        nz: int = 64
        """Lattice resolution along z."""
        lbm_scale: float = 0.1
        """Geometry scale relative to ``nx`` (dx = 1 / (lbm_scale * nx))."""
        fluid_density: float = 1000.0
        """Physical fluid density [kg/m^3]."""
        viscosity: float = 0.1
        """Lattice kinematic viscosity."""
        initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Uniform initial lattice velocity."""
        bc_type: tuple[int, int, int, int, int, int] = (1, 1, 1, 1, 1, 1)
        """Boundary condition type per face (0 = velocity, 1 = outflow)."""
        bc_value: tuple[tuple[float, float, float], ...] = (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
        """Boundary velocity per face (left, right, top, bottom, front, back)."""

    @dataclass
    class SolidConfig:
        """An immersed solid tracked by :class:`SolverLBM`.

        Args:
            body_index: Index of the rigid body this solid resolves against.
            lbm_position: Initial center in lattice coordinates. Defaults to the
                domain center.
            is_static: If ``True`` the solid blocks fluid but receives no force
                feedback.
            mesh: Optional ``(vertices, faces)`` pair in the body-local frame.
                Defaults to a mesh generated from the body's shapes.
        """

        body_index: int
        lbm_position: tuple[float, float, float] | None = None
        is_static: bool = False
        mesh: tuple[np.ndarray, np.ndarray] | None = None

    def __init__(self, model: Model, config: Config | None = None) -> None:
        super().__init__(model)
        self.config = config if config is not None else self.Config()

        if self.config.nx <= 0 or self.config.ny <= 0 or self.config.nz <= 0:
            raise ValueError("LBM grid dimensions must be positive")

        self._device = model.device

        self.solids: list[SolverLBM.SolidConfig] = []
        self._finalized = False

        # Populated by finalize()
        self.flows: list[HomeFlow] = []
        self.flows_wp: wp.array[Any] | None = None
        self._mesh_keep_alive: list[wp.Mesh] = []

        self._solid_ids_wp: wp.array[wp.int32] | None = None
        self._body_ids_wp: wp.array[wp.int32] | None = None
        self._solid_ids_dyn_wp: wp.array[wp.int32] | None = None
        self._body_ids_dyn_wp: wp.array[wp.int32] | None = None
        self._mujoco_origins_wp: wp.array[wp.vec3] | None = None
        self._lbm_origins_wp: wp.array[wp.vec3] | None = None
        self._scales_wp: wp.array[wp.float32] | None = None

        self._positions_buffer: wp.array[Any] | None = None
        self._quaternions_buffer: wp.array[Any] | None = None
        self._forces_buffer: wp.array[Any] | None = None
        self._torques_buffer: wp.array[Any] | None = None

        self._force_conversion = 1.0
        self._torque_conversion = 1.0

    # ====================== Configuration ======================

    def add_solid(
        self,
        body_index: int,
        lbm_position: tuple[float, float, float] | None = None,
        is_static: bool = False,
        mesh: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> int:
        """Register an immersed solid resolved against a rigid body.

        Args:
            body_index: Index of the rigid body.
            lbm_position: Initial lattice position. Defaults to the domain center.
            is_static: If ``True``, blocks fluid but receives no forces.
            mesh: Optional ``(vertices, faces)`` in the body-local frame.

        Returns:
            The solid id assigned to this solid.
        """
        if self._finalized:
            raise RuntimeError("Cannot add solids after finalize()")
        solid = self.SolidConfig(
            body_index=body_index,
            lbm_position=lbm_position,
            is_static=is_static,
            mesh=mesh,
        )
        self.solids.append(solid)
        return len(self.solids) - 1

    def set_viscosity(self, viscosity: float) -> None:
        """Set the lattice kinematic viscosity and invalidate any captured graph.

        Args:
            viscosity: Positive kinematic viscosity in lattice units.

        Raises:
            ValueError: If ``viscosity`` is not finite and positive.
        """
        value = float(viscosity)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("viscosity must be finite and positive")
        if not self._finalized:
            self.config.viscosity = value
            return
        wp.synchronize()
        for flow in self.flows:
            flow.vis_shear = value
        self._rebuild_flows_wp()

    def set_boundary_velocity(
        self, boundary_index: int, velocity: tuple[float, float, float]
    ) -> None:
        """Update the velocity boundary condition for one domain face.

        Args:
            boundary_index: Face index in ``[0, 6)`` (left, right, top, bottom,
                front, back).
            velocity: New boundary velocity in lattice units.
        """
        if not self._finalized:
            raise RuntimeError("finalize() must be called before set_boundary_velocity()")
        wp.launch(
            set_boundary_velocity_3d_kernel,
            dim=(self.model.world_count,),
            inputs=[self.flows_wp, boundary_index, *map(float, velocity)],
            device=self._device,
        )

    # ====================== Build / finalize ======================

    def _build_body_mesh(self, body_index: int) -> tuple[np.ndarray, np.ndarray] | None:
        """Build a combined body-local triangle mesh from the body's shapes."""
        model = self.model
        shape_types = model.shape_type.numpy()
        shape_scales = model.shape_scale.numpy()
        shape_bodies = model.shape_body.numpy()
        shape_tfs = model.shape_transform.numpy()

        verts_all: list[np.ndarray] = []
        faces_all: list[np.ndarray] = []
        offset = 0
        for s in range(model.shape_count):
            if int(shape_bodies[s]) != body_index:
                continue
            mesh = generate_shape_mesh(
                int(shape_types[s]), shape_scales[s], model.shape_source[s]
            )
            if mesh is None:
                continue
            verts, faces = mesh
            if verts.shape[0] == 0:
                continue
            verts = _transform_points(shape_tfs[s], verts)
            verts_all.append(verts)
            faces_all.append(faces.astype(np.int32) + offset)
            offset += verts.shape[0]

        if not verts_all:
            return None
        return np.concatenate(verts_all, axis=0), np.concatenate(faces_all, axis=0)

    def _initial_body_com(self, body_index: int) -> np.ndarray:
        """World-frame COM of a body from the model's initial state (world 0)."""
        model = self.model
        tf7 = model.body_q.numpy()[body_index]
        com = model.body_com.numpy()[body_index].astype(np.float64)
        return (_transform_points(tf7, com[None, :])[0]).astype(np.float32)

    def finalize(self) -> None:
        """Allocate the lattice grids, immersed-solid meshes, and mappings.

        Must be called after all :meth:`add_solid` calls and before :meth:`step`.
        """
        if self._finalized:
            return
        model = self.model
        cfg = self.config
        nworld = model.world_count
        n_solids = len(self.solids)
        if n_solids == 0:
            raise RuntimeError("SolverLBM requires at least one solid")

        # ----- Build one HomeFlow per world -----
        self.flows = [HomeFlow() for _ in range(nworld)]
        for flow in self.flows:
            flow.Initialize(cfg.nx, cfg.ny, cfg.nz, n_objects=n_solids)
            flow.vis_shear = cfg.viscosity
            flow.bc_type = wp.types.vector(length=6, dtype=wp.int32)(*cfg.bc_type)
            flow.bc_value = wp.array(
                tuple(wp.vec3(*v) for v in cfg.bc_value), dtype=wp.vec3, device=self._device
            )
        self._rebuild_flows_wp()
        wp.launch(
            InitBoundary3D,
            dim=(nworld, cfg.nx, cfg.ny, cfg.nz),
            inputs=[self.flows_wp],
            device=self._device,
        )
        wp.launch(
            InitFlow3D,
            dim=(nworld, cfg.nx, cfg.ny, cfg.nz),
            inputs=[self.flows_wp],
            device=self._device,
        )
        if any(cfg.initial_velocity):
            wp.launch(
                set_uniform_flow_3d_kernel,
                dim=(nworld, cfg.nx, cfg.ny, cfg.nz),
                inputs=[self.flows_wp, *map(float, cfg.initial_velocity)],
                device=self._device,
            )

        # ----- Build immersed-solid meshes and register into every flow -----
        scale_actual = cfg.lbm_scale * cfg.nx
        mujoco_origins: list[np.ndarray] = []
        lbm_origins: list[np.ndarray] = []
        scales: list[float] = []

        for solid_id, solid in enumerate(self.solids):
            if solid.mesh is not None:
                verts, faces = solid.mesh
                verts = np.asarray(verts, dtype=np.float32)
                faces = np.asarray(faces, dtype=np.int32)
            else:
                built = self._build_body_mesh(solid.body_index)
                if built is None:
                    raise ValueError(
                        f"No volumetric geometry found for body {solid.body_index}"
                    )
                verts, faces = built

            scaled = verts * scale_actual
            if scaled.shape[0] == 0 or faces.shape[0] == 0:
                raise ValueError(f"Empty mesh for body {solid.body_index}")
            bound_radius = float(np.linalg.norm(scaled, axis=1).max())

            with wp.ScopedDevice(self._device):
                mesh_wp = wp.Mesh(
                    points=wp.array(scaled, dtype=wp.vec3),
                    indices=wp.array(faces.reshape(-1), dtype=wp.int32),
                )
            self._mesh_keep_alive.append(mesh_wp)

            lbm_pos = (
                np.asarray(solid.lbm_position, dtype=np.float32)
                if solid.lbm_position is not None
                else np.array([cfg.nx * 0.5, cfg.ny * 0.6, cfg.nz * 0.5], dtype=np.float32)
            )
            mujoco_origin = self._initial_body_com(solid.body_index)

            for flow in self.flows:
                self._set_flow_solid(flow, solid_id, mesh_wp.id, lbm_pos, bound_radius)

            mujoco_origins.append(mujoco_origin)
            lbm_origins.append(lbm_pos)
            scales.append(scale_actual)

        self._rebuild_flows_wp()

        # ----- Coordinate-mapping arrays -----
        self._mujoco_origins_wp = wp.array(mujoco_origins, dtype=wp.vec3, device=self._device)
        self._lbm_origins_wp = wp.array(lbm_origins, dtype=wp.vec3, device=self._device)
        self._scales_wp = wp.array(scales, dtype=wp.float32, device=self._device)

        solid_ids_all = list(range(n_solids))
        body_ids_all = [s.body_index for s in self.solids]
        solid_ids_dyn = [i for i, s in enumerate(self.solids) if not s.is_static]
        body_ids_dyn = [s.body_index for s in self.solids if not s.is_static]
        self._solid_ids_wp = wp.array(solid_ids_all, dtype=wp.int32, device=self._device)
        self._body_ids_wp = wp.array(body_ids_all, dtype=wp.int32, device=self._device)
        self._solid_ids_dyn_wp = wp.array(solid_ids_dyn, dtype=wp.int32, device=self._device)
        self._body_ids_dyn_wp = wp.array(body_ids_dyn, dtype=wp.int32, device=self._device)
        self._n_dynamic = len(body_ids_dyn)

        # ----- Coupling buffers -----
        n_dyn = max(self._n_dynamic, 1)
        self._positions_buffer = wp.zeros((nworld, n_solids, 3), dtype=wp.float32, device=self._device)
        self._quaternions_buffer = wp.zeros((nworld, n_solids, 4), dtype=wp.float32, device=self._device)
        self._forces_buffer = wp.zeros((nworld, n_dyn, 3), dtype=wp.float32, device=self._device)
        self._torques_buffer = wp.zeros((nworld, n_dyn, 3), dtype=wp.float32, device=self._device)

        self._finalized = True

    def _set_flow_solid(
        self, flow: HomeFlow, solid_id: int, mesh_id: int, lbm_pos: np.ndarray, bound_radius: float
    ) -> None:
        """Register one solid's mesh + initial pose into a flow (host side)."""
        mesh_ids = flow.mesh_ids.numpy()
        mesh_ids[solid_id] = mesh_id
        flow.mesh_ids = wp.array(mesh_ids, dtype=wp.uint64)

        scales = flow.mesh_scale_sizes.numpy()
        scales[solid_id] = (1.0, 1.0, 1.0)
        flow.mesh_scale_sizes = wp.array(scales, dtype=wp.vec3)

        pos = flow.solid_position.numpy()
        pos[solid_id] = lbm_pos
        flow.solid_position = wp.array(pos, dtype=wp.vec3)

        quat = flow.solid_quaternion.numpy()
        quat[solid_id] = (1.0, 0.0, 0.0, 0.0)
        flow.solid_quaternion = wp.array(quat, dtype=wp.vec4)

        radius = flow.solid_bound_radius.numpy()
        radius[solid_id] = bound_radius
        flow.solid_bound_radius = wp.array(radius, dtype=wp.float32)

    def _rebuild_flows_wp(self) -> None:
        """Re-export the batched flow array after mutating flow structs."""
        self.flows_wp = wp.array(self.flows, dtype=HomeFlow, device=self._device)

    # ====================== SolverBase overrides ======================

    def step(
        self,
        state_in: State,
        state_out: State,
        control: newton.Control | None,
        contacts: newton.Contacts | None,
        dt: float,
    ) -> None:
        """Advance the fluid one lattice step and couple to the rigid bodies.

        Body poses are read from ``state_in.body_q``, the immersed-solid meshes
        are updated, the lattice is advanced one step, and the resulting
        hydrodynamic wrenches are written into ``state_in.body_f`` (physical
        units, world-frame COM reference) for the rigid solver to consume.

        Args:
            state_in: Input state (current rigid-body poses).
            state_out: Output state. ``body_f`` on the input state is populated
                with the hydrodynamic wrenches; the output state is not modified.
            control: Ignored by the LBM solver.
            contacts: Ignored by the LBM solver.
            dt: Rigid-body timestep used for the physical unit conversion.
        """
        if not self._finalized:
            raise RuntimeError("SolverLBM.finalize() must be called before step()")

        # Physical unit conversion (rho * dx^4 / dt^2 etc.)
        dx = 1.0 / (self.config.lbm_scale * self.config.nx)
        self._force_conversion = self.config.fluid_density * dx**4 / (dt * dt)
        self._torque_conversion = self.config.fluid_density * dx**5 / (dt * dt)

        nworld = self.model.world_count
        n_all = len(self.solids)

        # 1. Extract rigid-body poses and update immersed-solid transforms.
        wp.launch(
            extract_body_states_from_body_q,
            dim=(nworld, n_all),
            inputs=[
                state_in.body_q,
                self.model.body_com,
                self._body_ids_wp,
                self._positions_buffer,
                self._quaternions_buffer,
            ],
            device=self._device,
        )
        wp.launch(
            convert_and_update_solid_batch_3d,
            dim=(nworld, n_all),
            inputs=[
                self.flows_wp,
                self._solid_ids_wp,
                self._positions_buffer,
                self._quaternions_buffer,
                self._mujoco_origins_wp,
                self._lbm_origins_wp,
                self._scales_wp,
            ],
            device=self._device,
        )

        # 2. Advance the fluid one lattice step.
        wp.launch(
            init_force_3d_batch,
            dim=(nworld,),
            inputs=[self.flows_wp],
            device=self._device,
        )
        wp.launch(
            stream_and_collide_3d,
            dim=(nworld, self.config.nx, self.config.ny, self.config.nz),
            inputs=[self.flows_wp],
            device=self._device,
        )
        wp.launch(
            apply_bc_3d,
            dim=(nworld, self.config.nx, self.config.ny, self.config.nz),
            inputs=[self.flows_wp],
            device=self._device,
        )
        wp.launch(
            Swap_Mom_3D,
            dim=(nworld,),
            inputs=[self.flows_wp],
            device=self._device,
        )

        # 3. Write hydrodynamic wrenches to the rigid bodies.
        if self._n_dynamic > 0 and state_in.body_f is not None:
            wp.launch(
                extract_forces_torques_physical_3d,
                dim=(nworld, self._n_dynamic),
                inputs=[
                    self.flows_wp,
                    self._solid_ids_dyn_wp,
                    self._force_conversion,
                    self._torque_conversion,
                    self._forces_buffer,
                    self._torques_buffer,
                ],
                device=self._device,
            )
            wp.launch(
                fill_body_f_kernel,
                dim=(nworld, self._n_dynamic),
                inputs=[
                    state_in.body_f,
                    self._body_ids_dyn_wp,
                    self._forces_buffer,
                    self._torques_buffer,
                ],
                device=self._device,
            )

    def reset(
        self,
        state: State,
        world_mask: wp.array | None = None,
        flags: int | None = None,
    ) -> None:
        """Re-initialize the fluid fields (and solid mesh-transform history).

        Args:
            state: The simulation state (unused by the LBM grid, kept for API parity).
            world_mask: Optional boolean mask of shape ``(world_count + 1,)``.
            flags: Unused; kept for API parity.
        """
        super().reset(state, world_mask, flags)
        if not self._finalized:
            return
        nworld = self.model.world_count
        from .lbm_kernels import ResetSingleWorldFlow3D, ResetSingleWorldSolidTransform3D

        if world_mask is None:
            mask = wp.full(nworld + 1, 1, dtype=wp.int32, device=self._device)
        else:
            bool_mask = self._normalize_reset_world_mask(world_mask)
            mask = wp.array(bool_mask.numpy().astype(np.int32), dtype=wp.int32, device=self._device)

        wp.launch(
            ResetSingleWorldFlow3D,
            dim=(nworld, self.config.nx, self.config.ny, self.config.nz),
            inputs=[self.flows_wp, mask],
            device=self._device,
        )
        wp.launch(
            ResetSingleWorldSolidTransform3D,
            dim=(nworld, len(self.solids), 1),
            inputs=[self.flows_wp, mask],
            device=self._device,
        )
        if any(self.config.initial_velocity):
            wp.launch(
                set_uniform_flow_3d_kernel,
                dim=(nworld, self.config.nx, self.config.ny, self.config.nz),
                inputs=[self.flows_wp, *map(float, self.config.initial_velocity)],
                device=self._device,
            )

    def notify_model_changed(self, flags: int) -> None:
        """Rebuild immersed-solid meshes when shape geometry changes.

        Args:
            flags: Bit-mask of :class:`newton.ModelFlags`.
        """
        super().notify_model_changed(flags)
        if not self._finalized:
            return
        if flags & int(newton.ModelFlags.SHAPE_PROPERTIES):
            # Rebuild meshes from the (possibly updated) model geometry.
            self._mesh_keep_alive.clear()
            scale_actual = self.config.lbm_scale * self.config.nx
            for solid_id, solid in enumerate(self.solids):
                built = self._build_body_mesh(solid.body_index)
                if built is None:
                    continue
                verts, faces = built
                scaled = verts * scale_actual
                with wp.ScopedDevice(self._device):
                    mesh_wp = wp.Mesh(
                        points=wp.array(scaled, dtype=wp.vec3),
                        indices=wp.array(faces.reshape(-1), dtype=wp.int32),
                    )
                self._mesh_keep_alive.append(mesh_wp)
                for flow in self.flows:
                    mesh_ids = flow.mesh_ids.numpy()
                    mesh_ids[solid_id] = mesh_wp.id
                    flow.mesh_ids = wp.array(mesh_ids, dtype=wp.uint64)
            self._rebuild_flows_wp()

    # ====================== Rendering helpers ======================

    def vorticity_projection(self, view: str) -> wp.array2d:
        """Project the vorticity field onto a 2D plane (device buffer).

        Args:
            view: One of ``"topdown"``, ``"side"``, ``"front"``.

        Returns:
            The 2D float32 buffer holding the projected signed vorticity.
        """
        if not self._finalized:
            raise RuntimeError("finalize() must be called before vorticity_projection()")
        flow = self.flows[0]
        if view == "topdown":
            kernel = get_vorticity_projection_topdown_3d
            dim = (self.config.nx, self.config.ny)
            buf = flow.u_img_xy
        elif view == "side":
            kernel = get_vorticity_projection_side_3d
            dim = (self.config.ny, self.config.nz)
            buf = flow.u_img_xz
        elif view == "front":
            kernel = get_vorticity_projection_front_3d
            dim = (self.config.nx, self.config.nz)
            buf = flow.u_img_xz_front
        else:
            raise ValueError(f"Unknown view '{view}'; expected topdown, side, or front")
        wp.launch(kernel, dim=dim, inputs=[flow], device=self._device)
        return buf

    def velocity_projection(self, view: str) -> wp.array2d:
        """Project the velocity magnitude field onto a 2D plane (device buffer)."""
        flow = self.flows[0]
        if view == "topdown":
            kernel = get_u_projection_topdown_3d
            dim = (self.config.nx, self.config.ny)
            buf = flow.u_img_xy
        elif view == "side":
            kernel = get_u_projection_side_3d
            dim = (self.config.ny, self.config.nz)
            buf = flow.u_img_xz
        elif view == "front":
            kernel = get_u_projection_front_3d
            dim = (self.config.nx, self.config.nz)
            buf = flow.u_img_xz_front
        else:
            raise ValueError(f"Unknown view '{view}'; expected topdown, side, or front")
        wp.launch(kernel, dim=dim, inputs=[flow], device=self._device)
        return buf

    def vorticity_points(
        self, stride: int = 4
    ) -> tuple[wp.array, wp.array, wp.array]:
        """Downsampled signed-vorticity point cloud for ``viewer.log_points``.

        Args:
            stride: Cell stride along each lattice axis.

        Returns:
            ``(positions, colors, radii)`` device arrays where ``positions`` are
            lattice coordinates, ``colors`` are RGB in ``[0, 1]`` from the
            vorticity sign/magnitude, and ``radii`` are per-point radii.
        """
        flow = self.flows[0]
        nx, ny, nz = self.config.nx, self.config.ny, self.config.nz
        sx = max(1, nx // stride)
        sy = max(1, ny // stride)
        sz = max(1, nz // stride)
        stride_x = nx // sx
        stride_y = ny // sy
        stride_z = nz // sz

        positions = wp.empty((sx, sy, sz), dtype=wp.vec3, device=self._device)
        colors = wp.empty((sx, sy, sz), dtype=wp.vec3, device=self._device)
        radii = wp.empty((sx, sy, sz), dtype=wp.float32, device=self._device)

        wp.launch(
            vorticity_points_kernel,
            dim=(sx, sy, sz),
            inputs=[flow, positions, colors, radii, stride_x, stride_y, stride_z],
            device=self._device,
        )
        return positions, colors, radii

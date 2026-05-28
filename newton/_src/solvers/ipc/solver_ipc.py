# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from ...core import Axis
from ...geometry import GeoType, Mesh, ShapeFlags, TetMesh
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase


def _quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = quat / norm
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw)],
            [2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)],
            [2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _transform_to_matrix(transform: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_xyzw_to_matrix(transform[3:7])
    matrix[:3, 3] = np.asarray(transform[:3], dtype=np.float64)
    return matrix


def _transform_points(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return vertices @ matrix[:3, :3].T + matrix[:3, 3]


def _compute_tet_total_volume(vertices: np.ndarray, tet_indices: np.ndarray) -> float:
    tet_vertices = vertices[tet_indices]
    a = tet_vertices[:, 0]
    b = tet_vertices[:, 1]
    c = tet_vertices[:, 2]
    d = tet_vertices[:, 3]
    volumes = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0
    return float(volumes.sum())


class SolverIPC(SolverBase):
    """Experimental `uipc`-backed implicit solver for tetrahedral soft bodies.

    `SolverIPC` delegates tetrahedral soft-body integration and barrier contact
    handling to the optional third-party `uipc` package. Rigid bodies are not
    integrated by this solver. Any colliding shapes in the model are treated as
    static or kinematic obstacles driven directly from `state_in.body_q`.

    Args:
        model: Newton model to simulate. The current implementation supports
            exactly one world and at least one tetrahedral soft body.
        backend: Optional `uipc` engine backend. Defaults to `"cuda"` when the
            Newton model lives on a CUDA device, otherwise `"cpu"`.
        workspace: Optional directory for `uipc` scratch files.
        friction: Contact friction coefficient used for soft-soft and
            soft-obstacle pairs. Defaults to the mean Newton shape friction when
            obstacle shapes exist, otherwise `0.5`.
        contact_resistance: Contact resistance passed to `uipc` pair
            registration.
        contact_d_hat: Optional `uipc` contact barrier distance override [m].
    """

    def __init__(
        self,
        model: Model,
        *,
        backend: str | None = None,
        workspace: str | Path | None = None,
        friction: float | None = None,
        contact_resistance: float = 1.0e9,
        contact_d_hat: float | None = None,
    ) -> None:
        super().__init__(model)
        self._uipc = self._import_uipc()
        self._uipc_core = importlib.import_module("uipc.core")
        self._uipc_constitution = importlib.import_module("uipc.constitution")

        self.backend = backend or ("cuda" if self.device.is_cuda else "cpu")
        self.friction = self._resolve_contact_friction(friction)
        self.contact_resistance = float(contact_resistance)
        self.contact_d_hat = contact_d_hat

        self._workspace = Path(workspace) if workspace is not None else None
        self._owns_workspace = workspace is None
        self._dt: float | None = None
        self._scene_dirty = True

        self._engine: Any = None
        self._world: Any = None
        self._scene: Any = None
        self._soft_contact: Any = None
        self._obstacle_contact: Any = None
        self._tet_slot: Any = None
        self._abd_constitution: Any = None
        self._workspace_path: Path | None = None
        self._obstacle_slots: list[Any] = []
        self._obstacle_shape_indices: list[int] = []

        self._shape_type_np = model.shape_type.numpy() if model.shape_type is not None else np.empty(0, dtype=np.int32)
        self._shape_body_np = model.shape_body.numpy() if model.shape_body is not None else np.empty(0, dtype=np.int32)
        self._shape_transform_np = (
            model.shape_transform.numpy() if model.shape_transform is not None else np.empty((0, 7), dtype=np.float32)
        )
        self._shape_scale_np = model.shape_scale.numpy() if model.shape_scale is not None else np.empty((0, 3), dtype=np.float32)
        self._shape_flags_np = model.shape_flags.numpy() if model.shape_flags is not None else np.empty(0, dtype=np.int32)
        self._tet_indices_np = model.tet_indices.numpy().reshape(-1, 4) if model.tet_indices is not None else np.empty((0, 4), dtype=np.int32)

        self._validate_model()

    @staticmethod
    def _import_uipc() -> Any:
        try:
            return importlib.import_module("uipc")
        except ImportError as exc:  # pragma: no cover - exercised in environments without uipc
            raise ImportError(
                "SolverIPC requires the optional `uipc` package. Install the matching "
                "`uipc` Python bindings before constructing `newton.solvers.SolverIPC`."
            ) from exc

    def _resolve_contact_friction(self, friction: float | None) -> float:
        if friction is not None:
            return float(friction)
        if self.model.shape_count and self.model.shape_friction is not None:
            shape_friction = self.model.shape_friction.numpy()
            if shape_friction.size:
                return float(np.mean(shape_friction))
        return 0.5

    def _validate_model(self) -> None:
        if self.model.world_count != 1:
            raise ValueError("SolverIPC currently supports exactly one world.")
        if self.model.tet_count == 0:
            raise ValueError("SolverIPC currently requires at least one tetrahedral soft body.")

        boundary_tris = TetMesh.compute_surface_triangles(self._tet_indices_np.reshape(-1))
        expected_tri_count = len(boundary_tris) // 3
        if self.model.tri_count != expected_tri_count:
            raise ValueError(
                "SolverIPC currently supports tet soft bodies only. Cloth elements or extra triangle-only "
                "surface elements are not yet supported."
            )

        for shape_index, shape_type_value in enumerate(self._shape_type_np):
            shape_flags = int(self._shape_flags_np[shape_index]) if shape_index < len(self._shape_flags_np) else 0
            if (shape_flags & (ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES)) == 0:
                continue
            shape_type = GeoType(int(shape_type_value))
            if shape_type in (GeoType.GAUSSIAN, GeoType.HFIELD):
                raise NotImplementedError(f"SolverIPC does not support obstacle shape type {shape_type.name}.")
            if shape_type == GeoType.PLANE:
                scale = np.asarray(self._shape_scale_np[shape_index], dtype=np.float64)
                if not np.isclose(scale[0], 0.0) or not np.isclose(scale[1], 0.0):
                    raise NotImplementedError("SolverIPC currently supports infinite planes only.")
                if int(self._shape_body_np[shape_index]) >= 0:
                    raise NotImplementedError("SolverIPC does not support body-attached plane obstacles yet.")

    def _create_scene_config(self, dt: float) -> dict[str, Any]:
        config = self._uipc_core.Scene.default_config()
        config["dt"] = float(dt)
        gravity = np.asarray(self.model.gravity.numpy()[0], dtype=np.float64)
        config["gravity"] = [[float(component)] for component in gravity]
        config["contact"]["enable"] = 1
        if self.contact_d_hat is not None:
            config["contact"]["d_hat"] = float(self.contact_d_hat)
        return config

    def _destroy_world(self) -> None:
        self._tet_slot = None
        getattr(self, "_obstacle_slots", []).clear()
        getattr(self, "_obstacle_shape_indices", []).clear()
        self._soft_contact = None
        self._obstacle_contact = None
        self._abd_constitution = None
        self._scene = None
        self._world = None
        self._engine = None
        workspace_path = getattr(self, "_workspace_path", None)
        if getattr(self, "_owns_workspace", False) and workspace_path is not None:
            shutil.rmtree(workspace_path, ignore_errors=True)
        self._workspace_path = None

    def _ensure_world(self, state_in: State, dt: float) -> None:
        if self._world is not None and not self._scene_dirty:
            if self._dt is not None and not np.isclose(dt, self._dt):
                raise ValueError(
                    f"SolverIPC does not support changing dt after initialization "
                    f"(existing dt={self._dt}, requested dt={dt})."
                )
            return

        self._destroy_world()
        self._dt = float(dt)
        self._scene_dirty = False

        if self._workspace is None:
            self._workspace_path = Path(tempfile.mkdtemp(prefix="newton_ipc_"))
        else:
            self._workspace_path = self._workspace
            self._workspace_path.mkdir(parents=True, exist_ok=True)

        self._engine = self._uipc_core.Engine(self.backend, str(self._workspace_path))
        self._world = self._uipc_core.World(self._engine)
        self._scene = self._uipc_core.Scene(self._create_scene_config(dt))

        self._build_softbody(state_in)
        self._build_obstacles(state_in)
        self._register_contacts()

        self._world.init(self._scene)
        self._world.dump()

    def _build_softbody(self, state_in: State) -> None:
        positions = np.asarray(state_in.particle_q.numpy(), dtype=np.float64)
        total_volume = _compute_tet_total_volume(positions, self._tet_indices_np)
        if total_volume <= 0.0:
            raise ValueError("SolverIPC requires a positive total rest volume for tetrahedral soft bodies.")

        particle_mass = self.model.particle_mass.numpy()
        total_mass = float(np.sum(particle_mass))
        if total_mass <= 0.0:
            raise ValueError("SolverIPC requires a positive total particle mass for tetrahedral soft bodies.")

        if self.model.tet_materials is None:
            raise ValueError("SolverIPC requires `Model.tet_materials`.")
        tet_materials = np.asarray(self.model.tet_materials.numpy(), dtype=np.float64)
        mu = float(np.mean(tet_materials[:, 0]))
        lam = float(np.mean(tet_materials[:, 1]))
        if mu <= 0.0 or lam + mu <= 0.0:
            raise ValueError("SolverIPC requires positive Lame parameters.")

        youngs_modulus = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
        poissons_ratio = float(np.clip(lam / (2.0 * (lam + mu)), -0.49, 0.49))
        density = total_mass / total_volume

        tet_geom = self._uipc.geometry.tetmesh(positions, self._tet_indices_np.astype(np.int32, copy=False))
        self._uipc.geometry.label_surface(tet_geom)

        constitutions = self._scene.constitution_tabular()
        stable_neo_hookean = self._uipc_constitution.StableNeoHookean()
        constitutions.insert(stable_neo_hookean)
        moduli = self._uipc_constitution.ElasticModuli.youngs_poisson(youngs_modulus, poissons_ratio)
        stable_neo_hookean.apply_to(tet_geom, moduli, mass_density=float(density))

        soft_object = self._scene.objects().create("tet_softbody")
        self._tet_slot, _ = soft_object.geometries().create(tet_geom)

    def _build_obstacles(self, state_in: State) -> None:
        for shape_index, shape_type_value in enumerate(self._shape_type_np):
            shape_flags = int(self._shape_flags_np[shape_index]) if shape_index < len(self._shape_flags_np) else 0
            if (shape_flags & (ShapeFlags.COLLIDE_SHAPES | ShapeFlags.COLLIDE_PARTICLES)) == 0:
                continue

            shape_type = GeoType(int(shape_type_value))
            if shape_type == GeoType.PLANE:
                plane_geom = self._build_plane_geometry(shape_index)
                obstacle_object = self._scene.objects().create(f"obstacle_plane_{shape_index}")
                slot, _ = obstacle_object.geometries().create(plane_geom)
            else:
                vertices, indices = self._shape_vertices_and_indices(shape_index)
                obstacle_geom = self._uipc.geometry.trimesh(vertices, indices)
                self._uipc.geometry.label_surface(obstacle_geom)
                self._uipc.view(obstacle_geom.transforms())[0] = self._shape_world_matrix(shape_index, state_in)
                slot = None
                if hasattr(self._uipc_constitution, "AffineBodyConstitution"):
                    if self._abd_constitution is None:
                        self._abd_constitution = self._uipc_constitution.AffineBodyConstitution()
                        self._scene.constitution_tabular().insert(self._abd_constitution)
                    kwargs = {"mass_density": 1.0}
                    if hasattr(self._uipc, "unit") and hasattr(self._uipc.unit, "MPa"):
                        kwargs["kappa"] = 1.0 * self._uipc.unit.MPa
                    self._abd_constitution.apply_to(obstacle_geom, **kwargs)
                    external_kinetic_attr = obstacle_geom.instances().find(self._uipc.builtin.external_kinetic)
                    if external_kinetic_attr is not None:
                        self._uipc.view(external_kinetic_attr)[:] = 1
                    is_fixed_attr = obstacle_geom.instances().find(self._uipc.builtin.is_fixed)
                    if is_fixed_attr is not None:
                        self._uipc.view(is_fixed_attr)[:] = int(self._shape_body_np[shape_index] < 0)

                obstacle_object = self._scene.objects().create(f"obstacle_mesh_{shape_index}")
                slot, _ = obstacle_object.geometries().create(obstacle_geom)

            self._obstacle_slots.append(slot)
            self._obstacle_shape_indices.append(shape_index)

    def _register_contacts(self) -> None:
        contacts = self._scene.contact_tabular()
        self._soft_contact = contacts.create("soft_contact")
        self._soft_contact.apply_to(self._tet_slot.geometry())

        contacts.insert(self._soft_contact, self._soft_contact, self.friction, self.contact_resistance, True)
        if self._obstacle_slots:
            self._obstacle_contact = contacts.create("obstacle_contact")
            for slot in self._obstacle_slots:
                self._obstacle_contact.apply_to(slot.geometry())
            contacts.insert(self._soft_contact, self._obstacle_contact, self.friction, self.contact_resistance, True)
            contacts.insert(self._obstacle_contact, self._obstacle_contact, 0.0, 0.0, False)

    def _shape_vertices_and_indices(self, shape_index: int) -> tuple[np.ndarray, np.ndarray]:
        shape_type = GeoType(int(self._shape_type_np[shape_index]))
        scale = np.asarray(self._shape_scale_np[shape_index], dtype=np.float64)

        if shape_type in (GeoType.MESH, GeoType.CONVEX_MESH):
            shape_source = self.model.shape_source[shape_index]
            if shape_source is None:
                raise ValueError(f"SolverIPC requires mesh source data for shape {shape_index}.")
            vertices = np.asarray(shape_source.vertices, dtype=np.float64) * scale[None, :]
            indices = np.asarray(shape_source.indices, dtype=np.int32).reshape(-1, 3)
            return vertices, indices

        common_kwargs = {"compute_normals": False, "compute_uvs": False, "compute_inertia": False}
        if shape_type == GeoType.BOX:
            primitive_mesh = Mesh.create_box(scale[0], scale[1], scale[2], duplicate_vertices=False, **common_kwargs)
        elif shape_type == GeoType.SPHERE:
            primitive_mesh = Mesh.create_sphere(scale[0], **common_kwargs)
        elif shape_type == GeoType.CAPSULE:
            primitive_mesh = Mesh.create_capsule(scale[0], scale[1], up_axis=Axis.Z, **common_kwargs)
        elif shape_type == GeoType.CYLINDER:
            primitive_mesh = Mesh.create_cylinder(scale[0], scale[1], up_axis=Axis.Z, **common_kwargs)
        elif shape_type == GeoType.CONE:
            primitive_mesh = Mesh.create_cone(scale[0], scale[1], up_axis=Axis.Z, **common_kwargs)
        elif shape_type == GeoType.ELLIPSOID:
            primitive_mesh = Mesh.create_ellipsoid(scale[0], scale[1], scale[2], **common_kwargs)
        else:
            raise NotImplementedError(f"SolverIPC does not support obstacle shape type {shape_type.name}.")

        vertices = np.asarray(primitive_mesh.vertices, dtype=np.float64)
        indices = np.asarray(primitive_mesh.indices, dtype=np.int32).reshape(-1, 3)
        return vertices, indices

    def _build_plane_geometry(self, shape_index: int) -> Any:
        world_matrix = _transform_to_matrix(self._shape_transform_np[shape_index])
        normal = world_matrix[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        normal /= np.linalg.norm(normal)
        height = float(np.dot(world_matrix[:3, 3], normal))
        return self._uipc.geometry.ground(height, normal)

    def _shape_world_matrix(self, shape_index: int, state: State) -> np.ndarray:
        shape_matrix = _transform_to_matrix(self._shape_transform_np[shape_index])
        body_index = int(self._shape_body_np[shape_index])
        if body_index < 0:
            return shape_matrix
        body_q = np.asarray(state.body_q.numpy()[body_index], dtype=np.float64)
        return _transform_to_matrix(body_q) @ shape_matrix

    def _sync_softbody_state(self, state_in: State) -> np.ndarray:
        positions = np.asarray(state_in.particle_q.numpy(), dtype=np.float64)
        tet_geom = self._tet_slot.geometry()
        tet_geom.positions().view()[:] = positions
        return positions.astype(np.float32, copy=False)

    def _sync_obstacles(self, state_in: State) -> None:
        for slot, shape_index in zip(self._obstacle_slots, self._obstacle_shape_indices, strict=True):
            shape_type = GeoType(int(self._shape_type_np[shape_index]))
            if shape_type == GeoType.PLANE:
                continue
            self._uipc.view(slot.geometry().transforms())[0] = self._shape_world_matrix(shape_index, state_in)

    def _read_softbody_positions(self) -> np.ndarray:
        tet_geom = self._tet_slot.geometry()
        transformed_geom, *_ = self._uipc.geometry.apply_transform(tet_geom)
        return np.asarray(transformed_geom.positions().view(), dtype=np.float32).reshape(-1, 3)

    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control | None,
        contacts: Contacts | None,
        dt: float,
    ) -> None:
        del control
        self._ensure_world(state_in, dt)

        if contacts is not None:
            contacts.clear()

        state_out.assign(state_in)
        positions_in = self._sync_softbody_state(state_in)
        self._sync_obstacles(state_in)

        self._world.advance()
        self._world.retrieve()

        positions_out = self._read_softbody_positions()
        state_out.particle_q.assign(positions_out)
        state_out.particle_qd.assign((positions_out - positions_in) / float(dt))

    def notify_model_changed(self, flags: int) -> None:
        del flags
        self._scene_dirty = True

    def update_contacts(self, contacts: Contacts, state: State | None = None) -> None:
        del state
        contacts.clear()

    def __del__(self) -> None:
        self._destroy_world()

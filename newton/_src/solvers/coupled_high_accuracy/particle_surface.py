# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Closed render meshes reconstructed from CUDA particle positions."""

from typing import NamedTuple

import numpy as np
import warp as wp


class ParticleSurfaceMesh(NamedTuple):
    """Triangle mesh with optional interpolated initial particle coordinates for material mapping.

    Indices are a flat int32 array with three entries per face. Buffers may be reused
    by the next reconstruction; copy them before retaining a mesh across rebuilds.
    """

    vertices: wp.array
    indices: wp.array
    rest_positions: wp.array | None = None


def connected_components(count, bonds):
    """Label connected particles by hooking larger roots to smaller roots and compressing paths."""
    parent = np.arange(count, dtype=np.int32)
    while len(bonds):
        roots = parent[bonds]
        updated = parent.copy()
        np.minimum.at(updated, roots.max(axis=1), roots.min(axis=1))
        while not np.array_equal(updated, updated[updated]):
            updated = updated[updated]
        if np.array_equal(parent, updated):
            break
        parent = updated
    return parent


@wp.kernel
def _surface_field(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    labels: wp.array[int],
    label: int,
    lower: wp.vec3,
    voxel_size: float,
    support: float,
    radius: float,
    clearance: float,
    field: wp.array3d[float],
):
    i, j, k = wp.tid()
    point = lower + voxel_size * wp.vec3(float(i), float(j), float(k))
    center = wp.vec3(0.0)
    weight = float(0.0)
    own_distance = support
    foreign_distance = support
    for p in wp.hash_grid_query(grid, point, support):
        offset = point - positions[p]
        distance = wp.length(offset)
        if labels[p] == label:
            own_distance = wp.min(own_distance, distance)
            w = wp.max(1.0 - wp.dot(offset, offset) / (support * support), 0.0)
            w = w * w * w
            center += w * positions[p]
            weight += w
        else:
            foreign_distance = wp.min(foreign_distance, distance)
    value = support
    if weight > 1.0e-6:
        # Zhu-Bridson's weighted center smooths the lattice while retaining isolated grains.
        value = wp.length(point - center / weight) - radius
        # Sparse kernels can average into an empty cut; require material density too.
        value = wp.max(value, radius * (0.5 - weight))
        if foreign_distance < support:
            # Independently fractured pieces remain disjoint when their reconstructed radii overlap on recontact.
            value = wp.max(value, 0.5 * (own_distance - foreign_distance) + clearance)
    field[i, j, k] = value


@wp.kernel
def _surface_rest_positions(
    grid: wp.uint64,
    positions: wp.array[wp.vec3],
    rest_positions: wp.array[wp.vec3],
    labels: wp.array[int],
    label: int,
    vertices: wp.array[wp.vec3],
    support: float,
    output: wp.array[wp.vec3],
):
    i = wp.tid()
    point = vertices[i]
    rest = wp.vec3(0.0)
    weight = float(0.0)
    for p in wp.hash_grid_query(grid, point, support):
        if labels[p] == label:
            offset = point - positions[p]
            w = wp.max(1.0 - wp.dot(offset, offset) / (support * support), 0.0)
            w = w * w * w
            rest += w * rest_positions[p]
            weight += w
    if weight > 1.0e-6:
        rest /= weight
    output[i] = rest


@wp.kernel
def _copy_indices(source: wp.array[int], vertex_offset: int, index_offset: int, output: wp.array[int]):
    i = wp.tid()
    output[index_offset + i] = source[i] + vertex_offset


class ParticleSurface:
    """Reconstruct closed particle surfaces with a weighted-center field and marching cubes.

    ``initial_positions`` and every ``rebuild`` input must contain the same selected
    particles in the same order, excluding solver auxiliaries and boundary samples.
    CUDA is required. Reconstruction synchronizes and allocates dynamically, so call
    it outside a simulation CUDA graph. Defaults use a voxel size of spacing / 3,
    support radius of 2 * spacing, and surface radius of 0.6 * spacing.

    Optional bonds (vec2i indices into the selection) and broken flags (int32, nonzero
    for broken) reconstruct connected material pieces separately, including after
    recontact. Without bonds, nearby fluid particles can merge naturally. Enable
    ``interpolate_rest_positions`` to map persistent material onto newly exposed faces.
    """

    def __init__(
        self,
        spacing: float,
        initial_positions: wp.array,
        *,
        voxel_size: float | None = None,
        support_radius: float | None = None,
        surface_radius: float | None = None,
        bonds: wp.array | None = None,
        broken: wp.array | None = None,
        interpolate_rest_positions: bool = False,
    ):
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("spacing must be positive and finite")
        self.voxel_size = spacing / 3.0 if voxel_size is None else voxel_size
        self.support = 2.0 * spacing if support_radius is None else support_radius
        self.radius = 0.6 * spacing if surface_radius is None else surface_radius
        for name, value in (
            ("voxel_size", self.voxel_size),
            ("support_radius", self.support),
            ("surface_radius", self.radius),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.radius >= self.support:
            raise ValueError("surface_radius must be smaller than support_radius")
        if initial_positions.dtype != wp.vec3 or initial_positions.ndim != 1:
            raise ValueError("initial_positions must be a one-dimensional Warp vec3 array")
        self.device = initial_positions.device
        if not self.device.is_cuda:
            raise ValueError("Particle surface reconstruction requires a CUDA device")
        self.particle_count = len(initial_positions)
        self.rest_positions = wp.clone(initial_positions) if interpolate_rest_positions else None
        if (bonds is None) != (broken is None):
            raise ValueError("Surface fracture tracking requires both bonds and broken flags")
        self.bonds = None
        if bonds is not None:
            if bonds.dtype != wp.vec2i or bonds.ndim != 1:
                raise ValueError("bonds must be a one-dimensional Warp vec2i array")
            if broken.dtype != wp.int32 or broken.ndim != 1 or len(broken) != len(bonds):
                raise ValueError("broken must be a one-dimensional int32 array with one flag per bond")
            self.bonds = bonds.numpy()
            if np.any(self.bonds < 0) or np.any(self.bonds >= self.particle_count):
                raise ValueError("bond indices must be within the selected particles")
        # Half a voxel on each side accommodates interpolation at oblique cut faces.
        self.clearance = 0.5 * self.voxel_size
        self.broken = broken
        self._broken_flags = None
        self.component_labels = np.zeros(self.particle_count, dtype=np.int32)
        self.labels = wp.zeros(self.particle_count, dtype=int, device=self.device)
        self.grid = wp.HashGrid(64, 32, 32, device=self.device)
        self.fields = {}

    def _update_components(self):
        if self.broken is not None:
            broken = self.broken.numpy().astype(bool)
            if not np.array_equal(broken, self._broken_flags):
                self.component_labels = connected_components(self.particle_count, self.bonds[~broken])
                self.labels.assign(self.component_labels)
                self._broken_flags = broken

    def rebuild(self, positions: wp.array) -> ParticleSurfaceMesh:
        """Build a mesh following current positions, with flat triangle indices and optional rest coordinates."""
        if positions.dtype != wp.vec3 or positions.ndim != 1 or len(positions) != self.particle_count:
            raise ValueError("positions must be a Warp vec3 array matching the initial particle selection")
        if positions.device != self.device:
            raise ValueError("positions must use the surface reconstruction device")
        points = positions.numpy()
        if not np.isfinite(points).all():
            raise ValueError("Cannot reconstruct a surface from non-finite particle positions")
        self._update_components()
        active_labels = np.unique(self.component_labels)
        self.fields = {label: cached for label, cached in self.fields.items() if label in active_labels}
        parts = []
        if len(points):
            self.grid.build(positions, self.support)
            for label in active_labels:
                part = self._rebuild_component(positions, points[self.component_labels == label], int(label))
                if len(part.vertices):
                    parts.append(part)
        if len(parts) == 1:
            return parts[0]
        vertices = wp.empty(sum(len(part.vertices) for part in parts), dtype=wp.vec3, device=self.device)
        indices = wp.empty(sum(len(part.indices) for part in parts), dtype=int, device=self.device)
        rest = wp.empty(len(vertices), dtype=wp.vec3, device=self.device) if self.rest_positions is not None else None
        vertex_offset, index_offset = 0, 0
        for part in parts:
            wp.copy(vertices, part.vertices, dest_offset=vertex_offset)
            if rest is not None:
                wp.copy(rest, part.rest_positions, dest_offset=vertex_offset)
            wp.launch(
                _copy_indices,
                dim=len(part.indices),
                inputs=[part.indices, vertex_offset, index_offset],
                outputs=[indices],
                device=self.device,
            )
            vertex_offset += len(part.vertices)
            index_offset += len(part.indices)
        return ParticleSurfaceMesh(vertices, indices, rest)

    def _rebuild_component(self, positions, points, label):
        # Follow moving pieces and keep a full exterior layer beyond the compact kernel's support.
        padding = self.support + self.voxel_size
        lower = np.floor((points.min(axis=0) - padding) / self.voxel_size) * self.voxel_size
        upper = np.ceil((points.max(axis=0) + padding) / self.voxel_size) * self.voxel_size
        shape = tuple(np.rint((upper - lower) / self.voxel_size).astype(int) + 1)
        cached = self.fields.get(label)
        if cached is None or cached[0].shape != shape:
            field = wp.empty(shape, dtype=float, device=self.device)
            marching_cubes = wp.MarchingCubes(*shape)
            self.fields[label] = field, marching_cubes
        else:
            field, marching_cubes = cached
        marching_cubes.domain_bounds_lower_corner = wp.vec3(*lower)
        marching_cubes.domain_bounds_upper_corner = wp.vec3(*upper)
        wp.launch(
            _surface_field,
            dim=shape,
            inputs=[
                self.grid.id,
                positions,
                self.labels,
                label,
                wp.vec3(*lower),
                self.voxel_size,
                self.support,
                self.radius,
                self.clearance,
            ],
            outputs=[field],
            device=self.device,
        )
        marching_cubes.surface(field, 0.0)
        vertices, indices = marching_cubes.verts, marching_cubes.indices
        rest = None
        if self.rest_positions is not None:
            rest = wp.empty(len(vertices), dtype=wp.vec3, device=self.device)
            wp.launch(
                _surface_rest_positions,
                dim=len(vertices),
                inputs=[self.grid.id, positions, self.rest_positions, self.labels, label, vertices, self.support],
                outputs=[rest],
                device=self.device,
            )
        return ParticleSurfaceMesh(vertices, indices, rest)


def log_surface_mesh(
    viewer,
    name: str,
    mesh: ParticleSurfaceMesh,
    *,
    uvs=None,
    texture=None,
    color=None,
    roughness=None,
    metallic=None,
    texture_lookup: bool = False,
):
    """Log changing mesh topology to Newton viewers; lookup textures disable spatial mipmap filtering."""
    # GL caches indices and fixed-size buffers, so replace only this render object.
    if hasattr(viewer, "objects"):
        old_mesh = viewer.objects.pop(viewer._qualify(name), None)
        if old_mesh is not None:
            old_mesh.destroy()
    viewer.log_mesh(
        name, mesh.vertices, mesh.indices, uvs=uvs, texture=texture, color=color, roughness=roughness, metallic=metallic
    )
    if texture is not None and hasattr(viewer, "objects"):
        # GL uploads textures but does not enable sampling in the material automatically.
        render_mesh = viewer.objects[viewer._qualify(name)]
        r, m, checker, _ = render_mesh.material
        render_mesh.material = (r, m, checker, 1.0)
        if texture_lookup:
            # Attribute UVs index a color table; spatial mipmaps would mix unrelated material values.
            gl = viewer.renderer.gl
            gl.glBindTexture(gl.GL_TEXTURE_2D, render_mesh.texture_id)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    if hasattr(viewer, "stage") and hasattr(viewer, "_meshes"):
        # USD authors topology and UVs only once; reconstruction changes both every frame.
        from pxr import UsdGeom

        usd_mesh = viewer._meshes[viewer._qualify(name)]
        frame = viewer._frame_index
        usd_mesh.GetFaceVertexCountsAttr().Set([3] * (len(mesh.indices) // 3), frame)
        usd_mesh.GetFaceVertexIndicesAttr().Set(mesh.indices.numpy(), frame)
        if uvs is not None:
            UsdGeom.PrimvarsAPI(usd_mesh).GetPrimvar("st").Set(uvs.numpy(), frame)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host side of self-collision (native ``build_collision_bvh``, ``precompute_self_particle_pairs`` and the per-step
``begin/accumulate/multiply_self_particle_pairs``).

Three LBVHs (collision triangles, edges and particles) are built once over the rest positions of world 0; every
world shares their topology. Each step refits them to the current positions (level by level, see
:func:`.kernels.self_collision.refit_level`), queries at most :data:`MAX_SLOTS` candidates per primitive and
accumulates penalty forces and Hessian blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from .kernels import self_collision as kernels

# Candidate slots per query primitive, as natively.
MAX_SLOTS = 31


@dataclass
class CollisionBvh:
    count: int = 0
    sorted: wp.array | None = None
    parents: wp.array | None = None
    children: wp.array | None = None
    escape: wp.array | None = None
    primitive_leaf: wp.array | None = None
    level_nodes: wp.array | None = None
    levels: list[tuple[int, int]] = field(default_factory=list)
    leaf_lower: wp.array | None = None
    leaf_upper: wp.array | None = None
    node_lower: wp.array | None = None
    node_upper: wp.array | None = None


@dataclass
class Candidates:
    local: int = 0
    count: int = 0
    indices: wp.array | None = None
    counts: wp.array | None = None
    slots: wp.array | None = None


class SelfCollisionHost:
    """Mixin of :class:`~.solver.Solver` with the self-collision host logic."""

    def _primitive_boxes(
        self,
        kind: str,
        x: wp.array,
        indices: wp.array,
        count: int,
        local: int,
        margin: float,
        lower: wp.array,
        upper: wp.array,
        key: str | None,
    ) -> None:
        model = self.model
        if kind == "particles":
            radii = self._view(model.particle_radii, wp.float32, self.per_world)
            args = [x, indices, radii, lower, upper, local, self.per_world]
            kernel = kernels.build_particle_aabbs
        elif kind == "edges":
            args = [x, indices, lower, upper, margin, local, self.per_world]
            kernel = kernels.build_edge_aabbs
        else:
            args = [x, indices, lower, upper, margin, local, self.per_world]
            kernel = kernels.build_triangle_aabbs
        if key is None:
            wp.launch(kernel, dim=count, inputs=args, device=self.device)
        else:
            self._launch(key, kernel, count, args)

    def _build_collision_bvh(self, kind: str, count: int, indices: wp.array) -> CollisionBvh:
        """Karras LBVH over world 0's rest-position boxes, with height levels for the bottom-up refit."""
        bvh = CollisionBvh()
        if not count:
            return bvh
        if count > 0x40000000:
            raise OverflowError("LBVH node indices exceed the supported signed 32-bit range")
        self._require(self.model.particle_rest_positions, self.count, "particle_rest_positions")
        rest = self._view(self.model.particle_rest_positions, wp.vec3, self.count)
        copies = self.count // self.per_world
        nodes = 2 * count - 1
        bvh.count = count
        bvh.leaf_lower = self._zeros(copies * count, wp.vec3)
        bvh.leaf_upper = self._zeros(copies * count, wp.vec3)
        bvh.node_lower = self._zeros(copies * nodes, wp.vec3)
        bvh.node_upper = self._zeros(copies * nodes, wp.vec3)
        lower = self._zeros(count, wp.vec3)
        upper = self._zeros(count, wp.vec3)
        self._primitive_boxes(kind, rest, indices, count, count, 0.0, lower, upper, None)
        parents = np.full(nodes, -1, dtype=np.int32)
        children = np.zeros((max(count - 1, 1), 2), dtype=np.int32)
        order = np.arange(count, dtype=np.int32)
        if count > 1:
            low, high = lower.numpy(), upper.numpy()
            scene_lower = wp.vec3(*np.min(low, axis=0))
            scene_upper = wp.vec3(*np.max(high, axis=0))
            codes = wp.zeros(2 * count, dtype=wp.int32, device=self.device)
            wp.launch(
                kernels.generate_morton_codes,
                dim=count,
                inputs=[codes, lower, upper, scene_lower, scene_upper],
                device=self.device,
            )
            sorted_ids = wp.array(np.concatenate([order, order]), dtype=wp.int32, device=self.device)
            # A stable LSD radix sort, like the native CUB sort, so equal codes keep their primitive order.
            wp.utils.radix_sort_pairs(codes, sorted_ids, count)
            parent_array = wp.array(parents, dtype=wp.int32, device=self.device)
            children_array = wp.zeros(count - 1, dtype=wp.vec2i, device=self.device)
            wp.launch(
                kernels.build_radix_tree,
                dim=count - 1,
                inputs=[parent_array, children_array, codes, count],
                device=self.device,
            )
            parents = parent_array.numpy()
            children = children_array.numpy()
            order = sorted_ids.numpy()[:count].astype(np.int32)
        bvh.sorted = self._upload(order, wp.int32)
        bvh.parents = self._upload(parents, wp.int32)
        bvh.children = self._upload(children, wp.vec2i)
        bvh.escape = self._zeros(nodes, wp.int32)
        wp.launch(
            kernels.build_escape_nodes, dim=nodes, inputs=[bvh.escape, bvh.parents, bvh.children], device=self.device
        )
        bvh.primitive_leaf = self._zeros(count, wp.int32)
        wp.launch(
            kernels.build_primitive_leaf_nodes,
            dim=count,
            inputs=[bvh.primitive_leaf, bvh.sorted, count],
            device=self.device,
        )
        # Height of every internal node (leaves are height 0); refit processes heights in increasing order.
        internal = count - 1
        height = np.zeros(nodes, dtype=np.int64)
        if internal:
            pending = [0]
            post_order: list[int] = []
            while pending:
                node = pending.pop()
                post_order.append(node)
                for child in children[node]:
                    if child < internal:
                        pending.append(int(child))
            for node in reversed(post_order):
                height[node] = 1 + max(height[children[node][0]], height[children[node][1]])
        level_nodes: list[int] = []
        for level in range(1, int(height.max(initial=0)) + 1):
            members = np.flatnonzero(height[:internal] == level)
            bvh.levels.append((len(level_nodes), len(members)))
            level_nodes.extend(members.tolist())
        bvh.level_nodes = self._upload(np.asarray(level_nodes, dtype=np.int32), wp.int32)
        return bvh

    def _refit(self, bvh: CollisionBvh, kind: str, x: wp.array, indices: wp.array, margin: float, key: str) -> None:
        copies = self.count // self.per_world
        total = bvh.count * copies
        self._primitive_boxes(kind, x, indices, total, bvh.count, margin, bvh.leaf_lower, bvh.leaf_upper, key)
        self._launch(
            "self_refit",
            kernels.copy_leaf_boxes,
            total,
            [
                bvh.leaf_lower,
                bvh.leaf_upper,
                bvh.sorted,
                bvh.node_lower,
                bvh.node_upper,
                bvh.count,
            ],
        )
        for start, size in bvh.levels:
            self._launch(
                "self_refit",
                kernels.refit_level,
                size * copies,
                [
                    bvh.level_nodes,
                    start,
                    size,
                    bvh.children,
                    bvh.node_lower,
                    bvh.node_upper,
                    bvh.count,
                ],
            )

    def _candidates(self, local: int, indices: wp.array) -> Candidates:
        copies = self.count // self.per_world
        c = Candidates(local=local, count=local * copies, indices=indices)
        c.counts = self._zeros(c.count, wp.int32)
        c.slots = self._zeros(c.count * MAX_SLOTS, wp.int32)
        self.self_capacity += c.count * MAX_SLOTS
        return c

    def _precompute_self_particle_pairs(self) -> None:
        model = self.model
        per_world = self.per_world
        copies = self.count // per_world
        self.self_capacity = 0
        self.self_available = False
        groups = np.zeros(per_world, dtype=np.int32)
        for indices_span, n, group, field_name in (
            (model.fluid_particle_indices, model.fluid_particle_count, 1, "fluid_particle_indices"),
            (model.granular_particle_indices, model.granular_particle_count, 2, "granular_particle_indices"),
        ):
            if indices_span.size < n:
                raise ValueError(f"{field_name} is smaller than its collision count")
            for index in self._read(indices_span, wp.int32, n):
                index = int(index)  # noqa: PLW2901
                if index < 0 or index >= per_world:
                    raise ValueError(
                        "fluid particle index is out of range"
                        if group == 1
                        else "granular particle index is out of range"
                    )
                if group == 2 and groups[index] == 1:
                    raise ValueError("fluid and granular particle selections cannot overlap")
                groups[index] = group
        self.self_groups = self._upload(groups, wp.int32)
        triangle_count, edge_count = model.collision_triangle_count, model.collision_edge_count
        particle_count = model.collision_particle_count
        triangles = self._read(model.collision_triangle_indices, wp.vec3i, triangle_count)
        edges = self._read(model.collision_edge_indices, wp.vec4i, edge_count)
        vertices: set[int] = set()
        adjacent: dict[tuple[int, int], set[int]] = {}
        for triangle in triangles.tolist():
            vertices.update(triangle)
            for i in range(3):
                a, b = triangle[(i + 1) % 3], triangle[(i + 2) % 3]
                adjacent.setdefault((min(a, b), max(a, b)), set()).add(triangle[i])
        offsets = [0]
        opposite: list[int] = []
        for edge in edges.tolist():
            vertices.add(edge[2])
            vertices.add(edge[3])
            neighbors = adjacent.setdefault((min(edge[2], edge[3]), max(edge[2], edge[3])), set())
            if edge[0] >= 0:
                neighbors.add(edge[0])
            if edge[1] >= 0:
                neighbors.add(edge[1])
            opposite.extend(sorted(neighbors))
            offsets.append(len(opposite))
        self.self_available = bool((triangle_count and vertices) or edge_count > 1 or particle_count > 1)
        self.self_opposite_offsets = self._upload(np.asarray(offsets, dtype=np.int32), wp.int32)
        self.self_opposite_vertices = self._upload(np.asarray(opposite, dtype=np.int32), wp.int32)
        empty = Candidates()
        self.self_vf_mesh = self.self_vf_free = self.self_ee = self.self_ef = empty
        self.self_pp_local = particle_count
        self.self_pp_count = particle_count * copies if particle_count > 1 else 0
        self.self_capacity = self.self_pp_count * MAX_SLOTS
        if triangle_count:
            mesh_vertices = self._upload(np.asarray(sorted(vertices), dtype=np.int32), wp.int32)
            self.self_vf_mesh = self._candidates(len(vertices), mesh_vertices)
        particle_indices = self._view(model.collision_particle_indices, wp.int32, particle_count)
        if particle_count and triangle_count:
            self.self_vf_free = self._candidates(particle_count, particle_indices)
        self.self_pp_counts = self._zeros(copies * particle_count, wp.int32)
        self.self_pp_slots = self._zeros(copies * particle_count * MAX_SLOTS, wp.int32)
        edge_indices = self._view(model.collision_edge_indices, wp.vec4i, edge_count)
        if edge_count > 1:
            self.self_ee = self._candidates(edge_count, edge_indices)
        if triangle_count and edge_count:
            self.self_ef = self._candidates(edge_count, edge_indices)
        self.self_diagonal = self._zeros(self.count, wp.mat33)
        self.self_counter = self._zeros(1, wp.int32)
        self.self_contacts = wp.zeros(0, dtype=kernels.SelfContact, device=self.device)
        self.triangle_bvh = self._build_collision_bvh(
            "triangles", triangle_count, self._view(model.collision_triangle_indices, wp.vec3i, triangle_count)
        )
        self.edge_bvh = self._build_collision_bvh("edges", edge_count, edge_indices)
        self.particle_bvh = self._build_collision_bvh("particles", particle_count, particle_indices)

    def _begin_self(self, x: wp.array) -> None:
        """Refit the trees to the current positions and query the candidates (before the prediction)."""
        model = self.model
        radius = self._config.self_collision_radius
        triangles = self._view(model.collision_triangle_indices, wp.vec3i, model.collision_triangle_count)
        edges = self._view(model.collision_edge_indices, wp.vec4i, model.collision_edge_count)
        particles = self._view(model.collision_particle_indices, wp.int32, model.collision_particle_count)
        if self.triangle_bvh.count:
            self._refit(self.triangle_bvh, "triangles", x, triangles, radius, "bvh_triangles")
        if self.edge_bvh.count:
            self._refit(self.edge_bvh, "edges", x, edges, radius, "bvh_edges")
        if self.self_pp_local:
            self._refit(self.particle_bvh, "particles", x, particles, 0.0, "bvh_particles")
        worlds = self._view(model.particle_world, wp.int32, self.per_world)
        tri = self.triangle_bvh
        radii = self._view(model.particle_radii, wp.float32, self.per_world)
        for candidate, vertex_radii, incident in (
            (self.self_vf_mesh, self._empty(wp.float32), 1),
            (self.self_vf_free, radii, 0),
        ):
            if candidate.count:
                self._launch(
                    "self_query_vf",
                    kernels.query_vertex_face,
                    candidate.count,
                    [
                        x,
                        candidate.indices,
                        vertex_radii,
                        triangles,
                        worlds,
                        tri.node_lower,
                        tri.node_upper,
                        tri.parents,
                        tri.children,
                        tri.sorted,
                        radius,
                        radius,
                        incident,
                        candidate.counts,
                        candidate.slots,
                        candidate.local,
                        tri.count,
                        self.per_world,
                        MAX_SLOTS,
                    ],
                )
        if self.self_pp_count:
            pb = self.particle_bvh
            self._launch(
                "self_query_pp",
                kernels.query_particle_pairs,
                self.self_pp_count,
                [
                    x,
                    particles,
                    radii,
                    worlds,
                    self.self_groups,
                    pb.node_lower,
                    pb.node_upper,
                    pb.parents,
                    pb.children,
                    pb.sorted,
                    pb.escape,
                    pb.primitive_leaf,
                    self.self_pp_counts,
                    self.self_pp_slots,
                    self.self_pp_local,
                    self.per_world,
                    MAX_SLOTS,
                ],
            )
        if self.self_ee.count:
            eb = self.edge_bvh
            self._launch(
                "self_query_ee",
                kernels.query_edge_edge,
                self.self_ee.count,
                [
                    x,
                    edges,
                    worlds,
                    eb.node_lower,
                    eb.node_upper,
                    eb.parents,
                    eb.children,
                    eb.sorted,
                    radius,
                    self.self_ee.counts,
                    self.self_ee.slots,
                    self.self_ee.local,
                    self.per_world,
                    MAX_SLOTS,
                ],
            )
        if self.self_ef.count:
            self._launch(
                "self_query_ef",
                kernels.query_edge_face,
                self.self_ef.count,
                [
                    x,
                    edges,
                    triangles,
                    worlds,
                    tri.node_lower,
                    tri.node_upper,
                    tri.parents,
                    tri.children,
                    tri.sorted,
                    radius,
                    self.self_ef.counts,
                    self.self_ef.slots,
                    self.self_ef.local,
                    tri.count,
                    self.per_world,
                    MAX_SLOTS,
                ],
            )

    def _accumulate_self(self, x: wp.array) -> None:
        model, config = self.model, self._config
        full = config.enable_collision_hessian_non_diagonal
        if full and self.self_contacts.shape[0] < self.self_capacity:
            self.self_contacts = wp.zeros(self.self_capacity, dtype=kernels.SelfContact, device=self.device)
        self.self_diagonal.zero_()
        if full:
            self.self_counter.zero_()
        contacts = self.self_contacts if full else wp.zeros(0, dtype=kernels.SelfContact, device=self.device)
        radius = config.self_collision_radius
        thickness = radius + radius
        stiffness_scale = self.collision_stiffness
        triangles = self._view(model.collision_triangle_indices, wp.vec3i, model.collision_triangle_count)
        edges = self._view(model.collision_edge_indices, wp.vec4i, model.collision_edge_count)
        particles = self._view(model.collision_particle_indices, wp.int32, model.collision_particle_count)
        radii = self._view(model.particle_radii, wp.float32, self.per_world)
        for candidate, vertex_radii, incident in (
            (self.self_vf_mesh, self._empty(wp.float32), 1),
            (self.self_vf_free, radii, 0),
        ):
            if candidate.count and config.self_collision_vertex_face_stiffness > 0.0:
                self._launch(
                    "self_accumulate_vf",
                    kernels.accumulate_vertex_face,
                    candidate.count,
                    [
                        x,
                        candidate.indices,
                        vertex_radii,
                        triangles,
                        candidate.counts,
                        candidate.slots,
                        stiffness_scale,
                        radius,
                        radius,
                        incident,
                        config.self_collision_vertex_face_stiffness,
                        self.rhs,
                        self.self_diagonal,
                        contacts,
                        self.self_counter,
                        candidate.local,
                        self.per_world,
                        MAX_SLOTS,
                    ],
                )
        if self.self_pp_count and config.self_collision_particle_particle_stiffness > 0.0:
            self._launch(
                "self_accumulate_pp",
                kernels.accumulate_particle_pairs,
                self.self_pp_count,
                [
                    x,
                    particles,
                    radii,
                    self.self_pp_counts,
                    self.self_pp_slots,
                    stiffness_scale,
                    config.self_collision_particle_particle_stiffness,
                    self.rhs,
                    self.self_diagonal,
                    contacts,
                    self.self_counter,
                    self.self_pp_local,
                    self.per_world,
                    MAX_SLOTS,
                ],
            )
        if self.self_ee.count and config.self_collision_edge_edge_stiffness > 0.0:
            self._launch(
                "self_accumulate_ee",
                kernels.accumulate_edge_edge,
                self.self_ee.count,
                [
                    x,
                    edges,
                    self.self_opposite_offsets,
                    self.self_opposite_vertices,
                    self.self_ee.counts,
                    self.self_ee.slots,
                    stiffness_scale,
                    thickness,
                    config.self_collision_edge_edge_stiffness,
                    self.rhs,
                    self.self_diagonal,
                    contacts,
                    self.self_counter,
                    self.self_ee.local,
                    self.per_world,
                    MAX_SLOTS,
                ],
            )
        if self.self_ef.count and config.self_collision_edge_face_stiffness > 0.0:
            self._launch(
                "self_accumulate_ef",
                kernels.accumulate_edge_face,
                self.self_ef.count,
                [
                    x,
                    edges,
                    self.self_opposite_offsets,
                    self.self_opposite_vertices,
                    triangles,
                    self.self_ef.counts,
                    self.self_ef.slots,
                    stiffness_scale,
                    thickness,
                    config.self_collision_edge_face_stiffness,
                    self.rhs,
                    self.self_diagonal,
                    contacts,
                    self.self_counter,
                    self.self_ef.local,
                    self.per_world,
                    MAX_SLOTS,
                ],
            )

    def _multiply_self(self) -> None:
        if self._config.enable_collision_hessian_non_diagonal:
            self._launch(
                "self_full_tangent",
                kernels.multiply_contact_tangents,
                min(self.self_capacity, 256 * 256),
                [
                    self.self_contacts,
                    self.self_counter,
                    self.direction,
                    self.contact_product,
                    min(self.self_capacity, 256 * 256),
                ],
            )
        else:
            self._launch(
                "self_diagonal_tangent",
                kernels.multiply_hessian_diagonal,
                self.count,
                [
                    self.self_diagonal,
                    self.direction,
                    self.contact_product,
                ],
            )

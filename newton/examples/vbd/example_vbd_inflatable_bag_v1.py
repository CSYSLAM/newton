# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop a rigid cube onto a closed pneumatic bag.

Run with ``python -m newton.examples vbd_inflatable_bag``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples


@dataclass(frozen=True)
class _SealedBagMesh:
    """Geometry and layer correspondence for a perimeter-sealed bag."""

    vertices: list[list[float]]
    indices: list[int]
    seal_pairs: list[tuple[int, int]]
    bottom_vertices: list[int]
    top_vertices: list[int]
    bottom_indices: list[int]
    top_indices: list[int]
    bottom_grid_edges: list[tuple[int, int]]
    top_grid_edges: list[tuple[int, int]]


@wp.kernel
def _gather_grid_edges(
    positions: wp.array[wp.vec3],
    edge_indices: wp.array[int],
    lift: float,
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    edge = wp.tid()
    offset = wp.vec3(0.0, 0.0, lift)
    starts[edge] = positions[edge_indices[2 * edge]] + offset
    ends[edge] = positions[edge_indices[2 * edge + 1]] + offset


def _pillow_surface(
    width: float,
    depth: float,
    height: float,
    bulge: float,
    resolution: int,
) -> tuple[list[list[float]], list[int]]:
    """Return a closed, two-layer pillow mesh centered at the origin.

    The top and bottom faces have independent grid interiors and share their
    perimeter through narrow side panels. This gives the shell enough degrees
    of freedom to form a local contact dimple instead of moving as a box.
    """
    if resolution < 2:
        raise ValueError("resolution must be at least 2.")

    node_count = resolution + 1
    layer_size = node_count * node_count
    vertices: list[list[float]] = []
    for layer in range(2):
        for j in range(node_count):
            y = depth * (j / resolution - 0.5)
            for i in range(node_count):
                x = width * (i / resolution - 0.5)
                profile = np.sin(np.pi * i / resolution) * np.sin(np.pi * j / resolution)
                z = (height * 0.5 + bulge * profile) * (-1.0 if layer == 0 else 1.0)
                vertices.append([x, y, z])

    def vertex(layer: int, i: int, j: int) -> int:
        return layer * layer_size + j * node_count + i

    faces: list[int] = []
    for j in range(resolution):
        for i in range(resolution):
            bottom_a = vertex(0, i, j)
            bottom_b = vertex(0, i + 1, j)
            bottom_c = vertex(0, i, j + 1)
            bottom_d = vertex(0, i + 1, j + 1)
            top_a = vertex(1, i, j)
            top_b = vertex(1, i + 1, j)
            top_c = vertex(1, i, j + 1)
            top_d = vertex(1, i + 1, j + 1)

            # Bottom faces point down; top faces point up.
            faces.extend((bottom_a, bottom_c, bottom_b, bottom_b, bottom_c, bottom_d))
            faces.extend((top_a, top_b, top_c, top_b, top_d, top_c))

    perimeter = (
        [(i, 0) for i in range(resolution)]
        + [(resolution, j) for j in range(resolution)]
        + [(i, resolution) for i in range(resolution, 0, -1)]
        + [(0, j) for j in range(resolution, 0, -1)]
    )
    for (i0, j0), (i1, j1) in zip(perimeter, perimeter[1:] + perimeter[:1], strict=True):
        top_a = vertex(1, i0, j0)
        top_b = vertex(1, i1, j1)
        bottom_a = vertex(0, i0, j0)
        bottom_b = vertex(0, i1, j1)
        faces.extend((top_a, bottom_a, top_b, top_b, bottom_a, bottom_b))

    return vertices, faces


def _sealed_bag_surface(
    width: float,
    depth: float,
    bulge: float,
    seal_width: float,
    resolution: int,
) -> _SealedBagMesh:
    """Return a closed two-film bag with a flat perimeter seal.

    The films share one continuous outer boundary instead of being connected
    by vertical side panels. The two seal surfaces use matching diagonals, so
    their coincident triangles have the same discretization. Corresponding
    seal vertices are returned in ``seal_pairs`` so the caller can weld the
    layers.
    """
    if resolution < 4:
        raise ValueError("resolution must be at least 4.")
    if width <= 0.0 or depth <= 0.0:
        raise ValueError("width and depth must be positive.")
    if bulge <= 0.0:
        raise ValueError("bulge must be positive.")
    if not 0.0 < seal_width < 0.5 * min(width, depth):
        raise ValueError("seal_width must be positive and smaller than half the bag dimensions.")

    node_count = resolution + 1
    inner_half_width = 0.5 * width - seal_width
    inner_half_depth = 0.5 * depth - seal_width
    vertices: list[list[float]] = []
    bottom_vertices = [[-1] * node_count for _ in range(node_count)]
    top_vertices = [[-1] * node_count for _ in range(node_count)]
    seal_pairs: list[tuple[int, int]] = []

    for j in range(node_count):
        y = depth * (j / resolution - 0.5)
        for i in range(node_count):
            x = width * (i / resolution - 0.5)
            on_outer_boundary = i in (0, resolution) or j in (0, resolution)
            if on_outer_boundary:
                boundary_vertex = len(vertices)
                vertices.append([x, y, 0.0])
                bottom_vertices[j][i] = boundary_vertex
                top_vertices[j][i] = boundary_vertex
                continue

            inside_cavity = abs(x) < inner_half_width and abs(y) < inner_half_depth
            profile = 0.0
            if inside_cavity:
                x_phase = 0.5 * (x / inner_half_width + 1.0)
                y_phase = 0.5 * (y / inner_half_depth + 1.0)
                profile = float(np.sin(np.pi * x_phase) * np.sin(np.pi * y_phase))

            bottom_vertex = len(vertices)
            vertices.append([x, y, -bulge * profile])
            top_vertex = len(vertices)
            vertices.append([x, y, bulge * profile])
            bottom_vertices[j][i] = bottom_vertex
            top_vertices[j][i] = top_vertex
            if not inside_cavity:
                seal_pairs.append((bottom_vertex, top_vertex))

    faces: list[int] = []
    bottom_faces: list[int] = []
    top_faces: list[int] = []
    for j in range(resolution):
        for i in range(resolution):
            bottom_a = bottom_vertices[j][i]
            bottom_b = bottom_vertices[j][i + 1]
            bottom_c = bottom_vertices[j + 1][i]
            bottom_d = bottom_vertices[j + 1][i + 1]
            top_a = top_vertices[j][i]
            top_b = top_vertices[j][i + 1]
            top_c = top_vertices[j + 1][i]
            top_d = top_vertices[j + 1][i + 1]

            # Keep both films' visible diagonals aligned. At two corner cells,
            # flip the diagonal so it cannot connect two shared boundary nodes.
            flip_corner = (i == resolution - 1 and j == 0) or (i == 0 and j == resolution - 1)
            if flip_corner:
                bottom_triangles = ((bottom_a, bottom_c, bottom_b), (bottom_b, bottom_c, bottom_d))
                top_triangles = ((top_a, top_b, top_c), (top_b, top_d, top_c))
            else:
                bottom_triangles = ((bottom_a, bottom_d, bottom_b), (bottom_a, bottom_c, bottom_d))
                top_triangles = ((top_a, top_b, top_d), (top_a, top_d, top_c))

            for triangle in bottom_triangles:
                faces.extend(triangle)
                bottom_faces.extend(triangle)
            for triangle in top_triangles:
                faces.extend(triangle)
                top_faces.extend(triangle)

    bottom_grid_edges: list[tuple[int, int]] = []
    top_grid_edges: list[tuple[int, int]] = []
    for j in range(node_count):
        for i in range(resolution):
            bottom_grid_edges.append((bottom_vertices[j][i], bottom_vertices[j][i + 1]))
            top_grid_edges.append((top_vertices[j][i], top_vertices[j][i + 1]))
    for j in range(resolution):
        for i in range(node_count):
            bottom_grid_edges.append((bottom_vertices[j][i], bottom_vertices[j + 1][i]))
            top_grid_edges.append((top_vertices[j][i], top_vertices[j + 1][i]))

    return _SealedBagMesh(
        vertices=vertices,
        indices=faces,
        seal_pairs=seal_pairs,
        bottom_vertices=[vertex for row in bottom_vertices for vertex in row],
        top_vertices=[vertex for row in top_vertices for vertex in row],
        bottom_indices=bottom_faces,
        top_indices=top_faces,
        bottom_grid_edges=bottom_grid_edges,
        top_grid_edges=top_grid_edges,
    )


def _add_seal_welds(
    builder: newton.ModelBuilder,
    particle_start: int,
    seal_pairs: list[tuple[int, int]],
    stiffness: float,
    damping: float,
) -> np.ndarray:
    """Weld corresponding film vertices across the flat seal band."""
    global_pairs = np.asarray(seal_pairs, dtype=np.int32) + particle_start
    for bottom_vertex, top_vertex in global_pairs:
        builder.add_spring(
            int(bottom_vertex),
            int(top_vertex),
            ke=stiffness,
            kd=damping,
            control=0.0,
        )

    return global_pairs


PARAMS = {
    "fps": 60,
    "sim_substeps": 4,
    "solver_iterations": 20,
    "gravity": (0.0, 0.0, -9.81),
    "bag_width": 0.34,
    "bag_depth": 0.24,
    "bag_bulge": 0.035,
    "bag_seal_width": 0.035,
    "bag_seal_ke": 2.0e5,
    "bag_seal_kd": 5.0,
    "bag_resolution": 16,
    "bag_center_z": 0.095,
    "bag_density": 0.12,
    "bag_tri_ke": 8.0e3,
    "bag_tri_ka": 8.0e3,
    "bag_tri_kd": 20.0,
    "bag_edge_ke": 0.5,
    "bag_edge_kd": 0.05,
    "bag_reference_absolute_pressure": 101_400.0,
    "particle_radius": 0.004,
    "soft_contact_margin": 0.004,
    "seal_support_half_height": 0.010,
    "block_half_extent": 0.012,
    "block_initial_z": 0.20,
    "block_density": 2.0e5,
}


class Example:
    """Demonstrate a closed gas cavity deforming under a falling rigid block."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / PARAMS["fps"]
        self.sim_dt = self.frame_dt / PARAMS["sim_substeps"]
        self.sim_time = 0.0

        builder = newton.ModelBuilder(gravity=PARAMS["gravity"])
        bag_mesh = _sealed_bag_surface(
            PARAMS["bag_width"],
            PARAMS["bag_depth"],
            PARAMS["bag_bulge"],
            PARAMS["bag_seal_width"],
            PARAMS["bag_resolution"],
        )
        particle_start = builder.particle_count
        self.cavity = newton.solvers.add_inflatable_mesh(
            builder,
            pos=wp.vec3(0.0, 0.0, PARAMS["bag_center_z"]),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=bag_mesh.vertices,
            indices=bag_mesh.indices,
            density=PARAMS["bag_density"],
            tri_ke=PARAMS["bag_tri_ke"],
            tri_ka=PARAMS["bag_tri_ka"],
            tri_kd=PARAMS["bag_tri_kd"],
            edge_ke=PARAMS["bag_edge_ke"],
            edge_kd=PARAMS["bag_edge_kd"],
            particle_radius=PARAMS["particle_radius"],
            config=newton.solvers.PneumaticConfig(
                mode=newton.solvers.PneumaticMode.ISOTHERMAL,
                reference_absolute_pressure=PARAMS["bag_reference_absolute_pressure"],
                ambient_pressure=101_325.0,
                bulk_damping=50.0,
                max_absolute_pressure=200_000.0,
            ),
        )
        self.seal_pairs = _add_seal_welds(
            builder,
            particle_start,
            bag_mesh.seal_pairs,
            PARAMS["bag_seal_ke"],
            PARAMS["bag_seal_kd"],
        )
        self.bottom_vertices = np.asarray(bag_mesh.bottom_vertices, dtype=np.int32) + particle_start
        self.top_vertices = np.asarray(bag_mesh.top_vertices, dtype=np.int32) + particle_start
        bag_bottom_triangle_indices = np.asarray(bag_mesh.bottom_indices, dtype=np.int32) + particle_start
        bag_top_triangle_indices = np.asarray(bag_mesh.top_indices, dtype=np.int32) + particle_start
        bag_bottom_grid_edges = np.asarray(bag_mesh.bottom_grid_edges, dtype=np.int32) + particle_start
        bag_top_grid_edges = np.asarray(bag_mesh.top_grid_edges, dtype=np.int32) + particle_start

        contact_cfg = newton.ModelBuilder.ShapeConfig(density=PARAMS["block_density"], ke=4.0e4, kd=100.0, mu=0.7)
        self.block_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, PARAMS["block_initial_z"]), wp.quat_identity()),
            label="falling_block",
        )
        builder.add_shape_box(
            self.block_body,
            hx=PARAMS["block_half_extent"],
            hy=PARAMS["block_half_extent"],
            hz=PARAMS["block_half_extent"],
            cfg=contact_cfg,
            color=(0.8, 0.3, 0.2),
        )
        support_half_height = PARAMS["seal_support_half_height"]
        support_top = PARAMS["bag_center_z"] - PARAMS["particle_radius"] - PARAMS["soft_contact_margin"]
        support_z = support_top - support_half_height
        support_half_width = 0.5 * PARAMS["bag_seal_width"]
        support_x = 0.5 * PARAMS["bag_width"] - support_half_width
        support_y = 0.5 * PARAMS["bag_depth"] - support_half_width
        inner_half_width = 0.5 * PARAMS["bag_width"] - PARAMS["bag_seal_width"]
        support_specs = (
            (-support_x, 0.0, support_half_width, 0.5 * PARAMS["bag_depth"]),
            (support_x, 0.0, support_half_width, 0.5 * PARAMS["bag_depth"]),
            (0.0, -support_y, inner_half_width, support_half_width),
            (0.0, support_y, inner_half_width, support_half_width),
        )
        for x, y, hx, hy in support_specs:
            support_shape = builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(x, y, support_z), wp.quat_identity()),
                hx=hx,
                hy=hy,
                hz=support_half_height,
                cfg=contact_cfg,
                label="hidden_seal_support",
            )
            builder.shape_flags[support_shape] &= ~int(newton.ShapeFlags.VISIBLE)

        builder.color()
        self.model = builder.finalize()
        self.bag_bottom_triangle_indices = wp.array(bag_bottom_triangle_indices, dtype=int, device=self.model.device)
        self.bag_top_triangle_indices = wp.array(bag_top_triangle_indices, dtype=int, device=self.model.device)
        self.bag_bottom_grid_edges = wp.array(bag_bottom_grid_edges.reshape(-1), dtype=int, device=self.model.device)
        self.bag_top_grid_edges = wp.array(bag_top_grid_edges.reshape(-1), dtype=int, device=self.model.device)
        self.bag_bottom_grid_starts = wp.empty(len(bag_bottom_grid_edges), dtype=wp.vec3, device=self.model.device)
        self.bag_bottom_grid_ends = wp.empty(len(bag_bottom_grid_edges), dtype=wp.vec3, device=self.model.device)
        self.bag_grid_starts = wp.empty(len(bag_top_grid_edges), dtype=wp.vec3, device=self.model.device)
        self.bag_grid_ends = wp.empty(len(bag_top_grid_edges), dtype=wp.vec3, device=self.model.device)
        self.model.soft_contact_ke = 4.0e4
        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_mu = 0.7
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=PARAMS["solver_iterations"],
            rigid_body_particle_contact_buffer_size=4096,
        )
        self.pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=PARAMS["soft_contact_margin"],
            enable_rigid_soft_full_surface_contact=True,
        )
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.initial_local_profile = self._local_dimple_depth(self.state_0.particle_q.numpy())
        self.max_profile_change = 0.0

        self.viewer.set_model(self.model)
        # Draw each film separately so the coincident seal layers cannot z-fight.
        self.viewer.show_triangles = False
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = False
            self.viewer.renderer.draw_edges = False
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.0, -0.38, 0.48), -52.0, 90.0)

    def step(self):
        """Advance the falling cube, contact solve, and deformable bag."""
        for _ in range(PARAMS["sim_substeps"]):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            if hasattr(self.viewer, "apply_forces"):
                self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def render(self):
        """Render the current pneumatic state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/bag/bottom_surface",
            self.state_0.particle_q,
            self.bag_bottom_triangle_indices,
            backface_culling=True,
            color=(0.72, 0.52, 0.24),
        )
        self.viewer.log_mesh(
            "/bag/top_surface",
            self.state_0.particle_q,
            self.bag_top_triangle_indices,
            backface_culling=True,
            color=(0.86, 0.68, 0.34),
        )
        wp.launch(
            _gather_grid_edges,
            dim=len(self.bag_grid_starts),
            inputs=[self.state_0.particle_q, self.bag_top_grid_edges, 1.0e-4],
            outputs=[self.bag_grid_starts, self.bag_grid_ends],
            device=self.model.device,
        )
        wp.launch(
            _gather_grid_edges,
            dim=len(self.bag_bottom_grid_starts),
            inputs=[self.state_0.particle_q, self.bag_bottom_grid_edges, -1.0e-4],
            outputs=[self.bag_bottom_grid_starts, self.bag_bottom_grid_ends],
            device=self.model.device,
        )
        self.viewer.log_lines(
            "/bag/top_grid",
            self.bag_grid_starts,
            self.bag_grid_ends,
            (0.08, 0.06, 0.02),
        )
        self.viewer.log_lines(
            "/bag/bottom_grid",
            self.bag_bottom_grid_starts,
            self.bag_bottom_grid_ends,
            (0.08, 0.06, 0.02),
        )
        self.viewer.end_frame()

    def _local_dimple_depth(self, positions: np.ndarray) -> float:
        """Return the center indentation relative to its second-ring neighbors."""
        node_count = PARAMS["bag_resolution"] + 1
        center = node_count // 2
        ring_offset = 2
        top = positions[self.top_vertices].reshape(node_count, node_count, 3)
        surrounding_height = np.mean(
            (
                top[center - ring_offset, center, 2],
                top[center + ring_offset, center, 2],
                top[center, center - ring_offset, 2],
                top[center, center + ring_offset, 2],
            )
        )
        return float(surrounding_height - top[center, center, 2])

    def test_post_step(self):
        """Track how far contact flattens or indents the central profile."""
        local_profile = self._local_dimple_depth(self.state_0.particle_q.numpy())
        self.max_profile_change = max(self.max_profile_change, local_profile - self.initial_local_profile)

    def test_final(self):
        """Verify the cube rests in a persistent dimple without opening the bag."""
        volume = self.state_0.pneumatic.volume.numpy()[self.cavity.cavity_index]
        positions = self.state_0.particle_q.numpy()
        block_z = self.state_0.body_q.numpy()[self.block_body, 2]
        block_speed = np.linalg.norm(self.state_0.body_qd.numpy()[self.block_body, :3])
        local_dimple = self._local_dimple_depth(positions)
        node_count = PARAMS["bag_resolution"] + 1
        top = positions[self.top_vertices].reshape(node_count, node_count, 3)
        bottom = positions[self.bottom_vertices].reshape(node_count, node_count, 3)
        rim_height = np.mean(np.concatenate((top[0, :, 2], top[-1, :, 2], top[1:-1, 0, 2], top[1:-1, -1, 2])))
        top_bulge = np.max(top[:, :, 2]) - rim_height
        bottom_bulge = rim_height - np.min(bottom[:, :, 2])
        seal_gap = np.linalg.norm(
            positions[self.seal_pairs[:, 0]] - positions[self.seal_pairs[:, 1]],
            axis=1,
        )
        center_index = (node_count // 2) * node_count + node_count // 2
        center_thickness = np.linalg.norm(
            positions[self.top_vertices[center_index]] - positions[self.bottom_vertices[center_index]]
        )
        assert np.isfinite(positions).all()
        assert block_z < PARAMS["block_initial_z"]
        assert block_z > 2.0 * PARAMS["block_half_extent"], f"block fell through the bag to z={block_z:.6f} m"
        assert block_speed < 0.02, f"block did not settle; final speed was {block_speed:.6f} m/s"
        assert local_dimple > 0.003, f"settled dimple was only {local_dimple:.6f} m deep"
        assert top_bulge > 0.01, f"top film bulge was only {top_bulge:.6f} m"
        assert bottom_bulge > 0.02, f"bottom film bulge was only {bottom_bulge:.6f} m"
        assert volume > self.cavity.rest_volume * 0.2
        assert np.max(seal_gap) < 0.002, f"seal opened by {np.max(seal_gap):.6f} m"
        assert center_thickness > 0.02, f"central film separation collapsed to {center_thickness:.6f} m"
        assert self.max_profile_change > 0.005, f"maximum local profile change was only {self.max_profile_change:.6f} m"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)

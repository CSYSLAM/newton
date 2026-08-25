# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Drop a nonwoven OBJ bag a short distance onto a table.

The Z-up ``nonwoven_5k.obj`` garment starts slightly above a visible static
table and settles under gravity. Formed-shape elasticity models the nonwoven
material's structural memory. This isolated scene contains no hands, rod, or
rigid payload.

Run from the repository root::

    uv run --extra examples -m newton.examples \
        mjvbd_v2_nonwoven_bag_table_drop --viewer gl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets"
BAG_OBJ_PATH = ASSET_ROOT / "style3d_probe" / "bag" / "nonwoven_5k" / "nonwoven_5k.obj"

FPS = 60
SIM_SUBSTEPS = 10
VBD_ITERATIONS = 30

BAG_SCALE = 1.0
BAG_TABLE_CLEARANCE = 0.012
BAG_AREAL_DENSITY = 0.01
BAG_PARTICLE_RADIUS = 0.0015
BAG_TRI_KE = 1.0e10
BAG_TRI_KA = 1.0e10
BAG_TRI_KD = 10.0
BAG_EDGE_KE = 1.0e5
BAG_EDGE_KD = 20.0
BAG_COLOR = (0.86, 0.50, 0.16)
BAG_OPACITY = 1.0
AIR_DRAG_RATE = 6.0
REST_SHAPE_STIFFNESS = 1.0e4
REST_SHAPE_DAMPING = 200.0
REST_SHAPE_TARGET_CLEARANCE = 0.003

TABLE_TOP_Z = 0.45
TABLE_COLOR = (0.34, 0.20, 0.10)
TABLE_CONTACT_MARGIN = 0.003
TABLE_CONTACT_KE = 4.0e8
TABLE_CONTACT_KD = 100.0
TABLE_FRICTION = 0.8

SOFT_CONTACT_KE = 2.0e6
SOFT_CONTACT_KD = 200.0
SOFT_CONTACT_FRICTION = 0.35
SOFT_CONTACT_MARGIN = 0.008
SOFT_CONTACT_MAX = 65536
RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = 4096
SELF_CONTACT_MARGIN = 0.003


def _prepare_bag_mesh(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load continuous simulation topology and authored render normals."""
    points = []
    normals = []
    face_corners = []
    with path.open(encoding="utf-8", errors="strict") as obj_file:
        for line_number, line in enumerate(obj_file, start=1):
            if line.startswith("v "):
                values = line.split()
                if len(values) < 4:
                    raise ValueError(f"Invalid OBJ vertex at {path}:{line_number}")
                points.append((float(values[1]), float(values[2]), float(values[3])))
            elif line.startswith("vn "):
                values = line.split()
                if len(values) < 4:
                    raise ValueError(f"Invalid OBJ normal at {path}:{line_number}")
                normals.append((float(values[1]), float(values[2]), float(values[3])))
            elif line.startswith("f "):
                corners = line.split()[1:]
                if len(corners) != 3:
                    raise ValueError(f"OBJ face must be triangular at {path}:{line_number}")
                triangle_corners = []
                for corner in corners:
                    corner_indices = corner.split("/")
                    vertex_index = int(corner_indices[0])
                    if vertex_index == 0:
                        raise ValueError(f"OBJ vertex indices are one-based at {path}:{line_number}")
                    vertex_index = vertex_index - 1 if vertex_index > 0 else len(points) + vertex_index
                    if len(corner_indices) < 3 or not corner_indices[2]:
                        raise ValueError(f"OBJ face is missing a normal index at {path}:{line_number}")
                    normal_index = int(corner_indices[2])
                    normal_index = normal_index - 1 if normal_index > 0 else len(normals) + normal_index
                    triangle_corners.append((vertex_index, normal_index))
                face_corners.append(triangle_corners)

    points = np.asarray(points, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    simulation_indices = np.asarray(
        [[vertex_index for vertex_index, _ in triangle] for triangle in face_corners], dtype=np.int32
    ).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError(f"OBJ mesh points have an invalid shape in {path}: {points.shape}")
    if simulation_indices.ndim != 1 or simulation_indices.size % 3 != 0:
        raise ValueError(f"OBJ mesh must be fully triangulated: {path}")
    if normals.ndim != 2 or normals.shape[1] != 3 or len(normals) == 0:
        raise ValueError(f"OBJ normals have an invalid shape in {path}: {normals.shape}")

    points[:, :2] -= 0.5 * (points[:, :2].min(axis=0) + points[:, :2].max(axis=0))
    points[:, 2] -= points[:, 2].min()
    points *= scale

    render_vertex_lookup: dict[tuple[int, int], int] = {}
    render_particle_indices = []
    render_normals = []
    render_indices = []
    for triangle in face_corners:
        for corner in triangle:
            render_vertex = render_vertex_lookup.get(corner)
            if render_vertex is None:
                render_vertex = len(render_particle_indices)
                render_vertex_lookup[corner] = render_vertex
                render_particle_indices.append(corner[0])
                render_normals.append(normals[corner[1]])
            render_indices.append(render_vertex)

    return (
        points,
        simulation_indices,
        np.asarray(render_particle_indices, dtype=np.int32),
        np.asarray(render_indices, dtype=np.int32),
        np.asarray(render_normals, dtype=np.float32),
    )


@wp.kernel
def _apply_particle_drag(
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    drag_rate: float,
    particle_f: wp.array[wp.vec3],
):
    """Apply mass-proportional air drag to all cloth particles."""
    particle = wp.tid()
    particle_f[particle] = particle_f[particle] - drag_rate * particle_mass[particle] * particle_qd[particle]


@wp.kernel
def _apply_rest_shape_elasticity(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    target_q: wp.array[wp.vec3],
    stiffness: float,
    damping: float,
    particle_f: wp.array[wp.vec3],
):
    """Apply mass-proportional elasticity toward the bag's formed shape."""
    particle = wp.tid()
    restoring_acceleration = stiffness * (target_q[particle] - particle_q[particle])
    damping_acceleration = damping * particle_qd[particle]
    particle_f[particle] = particle_f[particle] + particle_mass[particle] * (
        restoring_acceleration - damping_acceleration
    )


@wp.kernel
def _gather_render_positions(
    particle_q: wp.array[wp.vec3],
    render_particle_indices: wp.array[int],
    render_q: wp.array[wp.vec3],
):
    """Expand simulation positions across authored OBJ normal seams."""
    render_vertex = wp.tid()
    render_q[render_vertex] = particle_q[render_particle_indices[render_vertex]]


class Example:
    """Drop the nonwoven bag from a small height onto a static table."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / SIM_SUBSTEPS
        self.sim_time = 0.0
        self.bag_initial_bottom_z = TABLE_TOP_Z + BAG_TABLE_CLEARANCE

        bag_obj = Path(args.bag_obj).expanduser().resolve()
        if not bag_obj.is_file():
            raise FileNotFoundError(f"Nonwoven bag OBJ not found: {bag_obj}")
        bag_vertices, bag_indices, render_particle_indices, render_indices, render_normals = _prepare_bag_mesh(
            bag_obj, args.bag_scale
        )
        self.initial_particle_positions = bag_vertices + np.array(
            (0.0, 0.0, self.bag_initial_bottom_z), dtype=np.float32
        )
        self.rest_shape_target_positions = bag_vertices + np.array(
            (0.0, 0.0, TABLE_TOP_Z + REST_SHAPE_TARGET_CLEARANCE), dtype=np.float32
        )
        self.initial_bag_height = float(np.ptp(bag_vertices[:, 2]))

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        newton.solvers.SolverMJVBDV2.register_custom_attributes(builder)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, self.bag_initial_bottom_z),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=bag_vertices,
            indices=bag_indices,
            density=BAG_AREAL_DENSITY,
            tri_ke=BAG_TRI_KE,
            tri_ka=BAG_TRI_KA,
            tri_kd=BAG_TRI_KD,
            edge_ke=BAG_EDGE_KE,
            edge_kd=BAG_EDGE_KD,
            particle_radius=BAG_PARTICLE_RADIUS,
        )

        self.bag_triangle_indices = bag_indices.reshape(-1, 3)
        self.bag_edges = np.unique(
            np.sort(
                np.concatenate(
                    (
                        self.bag_triangle_indices[:, (0, 1)],
                        self.bag_triangle_indices[:, (1, 2)],
                        self.bag_triangle_indices[:, (2, 0)],
                    )
                ),
                axis=1,
            ),
            axis=0,
        )
        self.rest_edge_lengths = np.linalg.norm(
            bag_vertices[self.bag_edges[:, 1]] - bag_vertices[self.bag_edges[:, 0]], axis=1
        )

        table_cfg = newton.ModelBuilder.ShapeConfig(
            ke=TABLE_CONTACT_KE,
            kd=TABLE_CONTACT_KD,
            mu=TABLE_FRICTION,
            margin=TABLE_CONTACT_MARGIN,
        )
        self.table_shape_index = builder.add_ground_plane(
            height=TABLE_TOP_Z,
            cfg=table_cfg,
            color=TABLE_COLOR,
            label="tabletop",
        )
        builder.color(include_bending=True)

        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = SOFT_CONTACT_KE
        self.model.soft_contact_kd = SOFT_CONTACT_KD
        self.model.soft_contact_mu = SOFT_CONTACT_FRICTION
        self.rest_shape_targets = wp.array(
            self.rest_shape_target_positions,
            dtype=wp.vec3,
            device=self.model.device,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.solver = newton.solvers.SolverMJVBDV2(
            self.model,
            contact_mode="full",
            vbd_options={
                "iterations": VBD_ITERATIONS,
                "friction_epsilon": 1.0e-4,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": BAG_PARTICLE_RADIUS,
                "particle_self_contact_margin": SELF_CONTACT_MARGIN,
                "particle_vertex_contact_buffer_size": 48,
                "particle_edge_contact_buffer_size": 96,
                "particle_collision_detection_interval": -1,
                "particle_topological_contact_filter_threshold": 2,
                "particle_rest_shape_contact_exclusion_radius": SELF_CONTACT_MARGIN,
                "rigid_body_particle_contact_buffer_size": RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
            },
            collision_options={
                "broad_phase": "nxn",
                "soft_contact_margin": SOFT_CONTACT_MARGIN,
                "soft_contact_max": SOFT_CONTACT_MAX,
                "include_static_kinematic_pairs": False,
            },
        )
        if self.solver.features.backend != "pure_vbd":
            raise RuntimeError(f"This scene requires pure_vbd, got {self.solver.features.backend}")

        self.render_particle_indices = wp.array(
            render_particle_indices,
            dtype=wp.int32,
            device=self.model.device,
        )
        self.render_positions = wp.zeros(len(render_particle_indices), dtype=wp.vec3, device=self.model.device)
        self.render_indices = wp.array(render_indices, dtype=wp.int32, device=self.model.device)
        self.render_normals = wp.array(render_normals, dtype=wp.vec3, device=self.model.device)
        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = False
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(0.72, -1.08, 0.72), pitch=-10.0, yaw=124.0)

        self.use_graph = bool(args.graph_capture) and self.model.device.is_cuda
        self.capture()

    def capture(self) -> None:
        """Capture one complete simulation frame on CUDA."""
        self.graph = None
        if not self.use_graph:
            return

        state_0_backup = self.model.state()
        state_1_backup = self.model.state()
        state_0_backup.assign(self.state_0)
        state_1_backup.assign(self.state_1)
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph
        self.state_0.assign(state_0_backup)
        self.state_1.assign(state_1_backup)

    def simulate(self) -> None:
        """Advance the falling bag by one rendered frame."""
        for _ in range(SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                _apply_particle_drag,
                dim=self.model.particle_count,
                inputs=[self.state_0.particle_qd, self.model.particle_mass, AIR_DRAG_RATE],
                outputs=[self.state_0.particle_f],
                device=self.model.device,
            )
            wp.launch(
                _apply_rest_shape_elasticity,
                dim=self.model.particle_count,
                inputs=[
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self.model.particle_mass,
                    self.rest_shape_targets,
                    REST_SHAPE_STIFFNESS,
                    REST_SHAPE_DAMPING,
                ],
                outputs=[self.state_0.particle_f],
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        """Advance the table-drop scene by one frame."""
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        """Render the nonwoven bag and table."""
        wp.launch(
            _gather_render_positions,
            dim=len(self.render_particle_indices),
            inputs=[self.state_0.particle_q, self.render_particle_indices],
            outputs=[self.render_positions],
            device=self.model.device,
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/nonwoven_bag",
            self.render_positions,
            self.render_indices,
            normals=self.render_normals,
            backface_culling=False,
            color=BAG_COLOR,
            roughness=0.78,
            metallic=0.0,
            opacity=BAG_OPACITY,
        )
        self.viewer.end_frame()

    def test_final(self) -> None:
        """Verify that the bag lands on the table without excessive stretch."""
        if self.solver.features.backend != "pure_vbd":
            raise ValueError(f"Unexpected solver backend: {self.solver.features.backend}")

        soft_contact_count = int(self.solver.contacts.soft_contact_count.numpy()[0])
        if soft_contact_count >= SOFT_CONTACT_MAX:
            raise ValueError(f"Soft-contact capacity exhausted: {soft_contact_count} >= {SOFT_CONTACT_MAX}")

        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()
        if not np.all(np.isfinite(particle_q)) or not np.all(np.isfinite(particle_qd)):
            raise ValueError("Nonwoven table-drop state is not finite")
        minimum_z = float(particle_q[:, 2].min())
        if minimum_z < TABLE_TOP_Z - 0.01:
            raise ValueError(f"The nonwoven bag penetrated the table: z={minimum_z:.6g} m")
        if minimum_z > self.bag_initial_bottom_z - 0.005:
            raise ValueError("The nonwoven bag did not fall onto the table")

        edge_lengths = np.linalg.norm(particle_q[self.bag_edges[:, 1]] - particle_q[self.bag_edges[:, 0]], axis=1)
        edge_stretch_95 = float(np.quantile(edge_lengths / self.rest_edge_lengths, 0.95))
        if edge_stretch_95 > 1.15:
            raise ValueError(f"Nonwoven bag 95th-percentile edge stretch reached {edge_stretch_95:.6g}")

        final_height_ratio = float(np.ptp(particle_q[:, 2])) / self.initial_bag_height
        if final_height_ratio < 0.75:
            raise ValueError(f"Nonwoven bag retained only {final_height_ratio:.2%} of its initial height")

        rest_centered = self.initial_particle_positions - self.initial_particle_positions.mean(axis=0)
        final_centered = particle_q - particle_q.mean(axis=0)
        covariance = rest_centered.T @ final_centered
        left_vectors, _, right_vectors = np.linalg.svd(covariance)
        rotation = right_vectors.T @ left_vectors.T
        if np.linalg.det(rotation) < 0.0:
            right_vectors[-1] *= -1.0
            rotation = right_vectors.T @ left_vectors.T
        shape_rms = float(np.sqrt(np.mean(np.sum((rest_centered @ rotation - final_centered) ** 2, axis=1))))
        if shape_rms > 0.06:
            raise ValueError(f"Nonwoven bag shape RMS deviation reached {shape_rms:.6g} m")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--bag-obj", type=Path, default=BAG_OBJ_PATH)
    parser.add_argument("--bag-scale", type=float, default=BAG_SCALE)
    parser.add_argument(
        "--graph-capture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture one complete MJVBDV2 display frame on CUDA.",
    )
    parser.set_defaults(num_frames=180)

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

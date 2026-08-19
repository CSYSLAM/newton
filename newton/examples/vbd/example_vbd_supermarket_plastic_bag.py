# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Hang a supermarket bag containing rigid balls from a rod using VBD.

The example loads the provided triangulated OBJ as a single thin shell. A
fixed horizontal rod passes through both handle holes, and contact supports
the otherwise fully dynamic bag under gravity. Particle self-contact keeps
the front and back films from passing through one another. Two independently
moving rigid balls settle on the bag bottom through two-way rigid-cloth contact.

Run from the repository root::

    uv run --extra examples -m newton.examples vbd_supermarket_plastic_bag
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

BAG_MESH_PATH = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "supermarket_plastic_bag_open_carry_v5_asset"
    / "supermarket_plastic_bag_open_carry_v5_tri.obj"
)

FPS = 60
SIM_SUBSTEPS = 6
VBD_ITERATIONS = 12

BAG_POSITION = np.array((0.0, 0.0, 0.30), dtype=np.float32)
BAG_AREAL_DENSITY = 0.02
BAG_PARTICLE_RADIUS = 0.0015
BAG_TRI_KE = 3.0e6
BAG_TRI_KA = 3.0e6
BAG_TRI_KD = 0.5
BAG_EDGE_KE = 100.0
BAG_EDGE_KD = 3.0
AIR_DRAG_RATE = 1.0  # [1/s]

HANDLE_HOLE_CENTER_Z = 0.519
ROD_RADIUS = 0.0035
ROD_HALF_LENGTH = 0.22
ROD_CONTACT_KE = 4.0e8
ROD_CONTACT_KD = 100.0
ROD_COLOR = (0.55, 0.58, 0.62)
BALL_RADIUS = 0.040
BALL_DENSITY = 30000000.0
BALL_INITIAL_DOWNWARD_SPEED = 2.0
BALL_CONTACT_MARGIN = 0.005
BALL_CONTACT_KE = 1.0e7
BALL_CONTACT_KD = 500.0
BALL_FRICTION = 0.3
SOFT_CONTACT_KE = 1.0e7
SOFT_CONTACT_KD = 500.0
SOFT_CONTACT_FRICTION = 0.06
BALL_BOTTOM_CLEARANCE = 0.025
BALL_BOTTOM_CONTACT_FRACTION = 0.75
BALL_PENETRATION_TOLERANCE = 0.002
MAX_BODY_EDGE_STRETCH_RATIO_99 = 1.12
BALL_LOCAL_POSITIONS = (
    (-0.080, 0.0, 0.22),
    (0.080, 0.0, 0.22),
)
BALL_COLORS = (
    (0.92, 0.18, 0.15),
    (0.10, 0.48, 0.92),
)
SELF_CONTACT_MARGIN = 0.003
SOFT_CONTACT_MARGIN = 0.010
BAG_COLOR = (0.86, 0.93, 0.97)
BAG_OPACITY = 0.60


def _point_segment_squared_distances(point: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Compute squared distances from one point to line segments."""
    edges = ends - starts
    edge_length_squared = np.einsum("ij,ij->i", edges, edges)
    interpolation = np.zeros_like(edge_length_squared)
    np.divide(
        np.einsum("ij,ij->i", point - starts, edges),
        edge_length_squared,
        out=interpolation,
        where=edge_length_squared > 1.0e-20,
    )
    np.clip(interpolation, 0.0, 1.0, out=interpolation)
    closest = starts + interpolation[:, None] * edges
    offsets = closest - point
    return np.einsum("ij,ij->i", offsets, offsets)


def _point_triangle_distances(point: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Compute distances from one point to triangles."""
    vertex_a = triangles[:, 0]
    vertex_b = triangles[:, 1]
    vertex_c = triangles[:, 2]
    edge_ab = vertex_b - vertex_a
    edge_ac = vertex_c - vertex_a
    offset_ap = point - vertex_a

    dot_ab_ab = np.einsum("ij,ij->i", edge_ab, edge_ab)
    dot_ab_ac = np.einsum("ij,ij->i", edge_ab, edge_ac)
    dot_ac_ac = np.einsum("ij,ij->i", edge_ac, edge_ac)
    dot_ap_ab = np.einsum("ij,ij->i", offset_ap, edge_ab)
    dot_ap_ac = np.einsum("ij,ij->i", offset_ap, edge_ac)
    denominator = dot_ab_ab * dot_ac_ac - dot_ab_ac * dot_ab_ac

    barycentric_b = np.zeros_like(denominator)
    barycentric_c = np.zeros_like(denominator)
    valid_triangle = denominator > 1.0e-20
    np.divide(
        dot_ac_ac * dot_ap_ab - dot_ab_ac * dot_ap_ac,
        denominator,
        out=barycentric_b,
        where=valid_triangle,
    )
    np.divide(
        dot_ab_ab * dot_ap_ac - dot_ab_ac * dot_ap_ab,
        denominator,
        out=barycentric_c,
        where=valid_triangle,
    )
    projection_inside = (
        valid_triangle & (barycentric_b >= 0.0) & (barycentric_c >= 0.0) & (barycentric_b + barycentric_c <= 1.0)
    )

    normal = np.cross(edge_ab, edge_ac)
    normal_length_squared = np.einsum("ij,ij->i", normal, normal)
    signed_plane_numerator = np.einsum("ij,ij->i", offset_ap, normal)
    plane_distance_squared = np.zeros_like(normal_length_squared)
    np.divide(
        signed_plane_numerator * signed_plane_numerator,
        normal_length_squared,
        out=plane_distance_squared,
        where=normal_length_squared > 1.0e-20,
    )

    edge_distance_squared = np.minimum.reduce(
        (
            _point_segment_squared_distances(point, vertex_a, vertex_b),
            _point_segment_squared_distances(point, vertex_b, vertex_c),
            _point_segment_squared_distances(point, vertex_c, vertex_a),
        )
    )
    return np.sqrt(np.where(projection_inside, plane_distance_squared, edge_distance_squared))


@wp.kernel
def _apply_particle_drag(
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    drag_rate: float,
    particle_f: wp.array[wp.vec3],
):
    particle = wp.tid()
    particle_f[particle] = particle_f[particle] - drag_rate * particle_mass[particle] * particle_qd[particle]


class Example:
    """Simulate a self-colliding VBD bag carrying rigid balls."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / SIM_SUBSTEPS
        self.sim_time = 0.0

        bag_mesh_path = Path(getattr(args, "bag_mesh", BAG_MESH_PATH)).expanduser().resolve()
        if not bag_mesh_path.is_file():
            raise FileNotFoundError(f"Plastic bag mesh not found: {bag_mesh_path}")

        bag_mesh = newton.Mesh.create_from_file(str(bag_mesh_path), compute_inertia=False, is_solid=False)
        vertices = np.asarray(bag_mesh.vertices, dtype=np.float32)
        indices = np.asarray(bag_mesh.indices, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Expected bag vertices with shape (n, 3), got {vertices.shape}")
        if indices.ndim != 1 or indices.size % 3 != 0:
            raise ValueError(f"Expected a flat triangle index buffer, got {indices.shape}")
        self.bag_triangle_indices = indices.reshape(-1, 3)

        max_abs_x = float(np.abs(vertices[:, 0]).max())
        max_z = float(vertices[:, 2].max())
        handle_local = np.flatnonzero((np.abs(vertices[:, 0]) > 0.7 * max_abs_x) & (vertices[:, 2] > 0.65 * max_z))
        left_handle_local = handle_local[vertices[handle_local, 0] < 0.0]
        right_handle_local = handle_local[vertices[handle_local, 0] > 0.0]
        if left_handle_local.size == 0 or right_handle_local.size == 0:
            raise ValueError("The bag mesh must contain both left and right handles")

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        bag_particle_start = builder.particle_count
        builder.add_cloth_mesh(
            pos=wp.vec3(*BAG_POSITION),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            density=BAG_AREAL_DENSITY,
            tri_ke=BAG_TRI_KE,
            tri_ka=BAG_TRI_KA,
            tri_kd=BAG_TRI_KD,
            edge_ke=BAG_EDGE_KE,
            edge_kd=BAG_EDGE_KD,
            particle_radius=BAG_PARTICLE_RADIUS,
        )

        triangle_vertices = indices.reshape(-1, 3)
        mesh_edges = np.concatenate(
            (
                triangle_vertices[:, (0, 1)],
                triangle_vertices[:, (1, 2)],
                triangle_vertices[:, (2, 0)],
            )
        )
        mesh_edges = np.unique(np.sort(mesh_edges, axis=1), axis=0)
        is_handle = np.zeros(vertices.shape[0], dtype=bool)
        is_handle[handle_local] = True
        body_edge_mask = ~(is_handle[mesh_edges[:, 0]] | is_handle[mesh_edges[:, 1]])
        self.bag_body_edges = mesh_edges[body_edge_mask]
        self.rest_body_edge_lengths = np.linalg.norm(
            vertices[self.bag_body_edges[:, 1]] - vertices[self.bag_body_edges[:, 0]], axis=1
        )
        self.left_handle_indices = bag_particle_start + left_handle_local
        self.right_handle_indices = bag_particle_start + right_handle_local
        self.rod_position = BAG_POSITION + np.array((0.0, 0.0, HANDLE_HOLE_CENTER_Z), dtype=np.float32)

        rod_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * np.pi)
        rod_cfg = newton.ModelBuilder.ShapeConfig(ke=ROD_CONTACT_KE, kd=ROD_CONTACT_KD, mu=0.8)
        builder.add_shape_capsule(
            -1,
            xform=wp.transform(wp.vec3(*self.rod_position), rod_rotation),
            radius=ROD_RADIUS,
            half_height=ROD_HALF_LENGTH,
            cfg=rod_cfg,
            color=ROD_COLOR,
            label="handle support rod",
        )

        ball_cfg = newton.ModelBuilder.ShapeConfig(
            density=BALL_DENSITY,
            ke=BALL_CONTACT_KE,
            kd=BALL_CONTACT_KD,
            mu=BALL_FRICTION,
            margin=BALL_CONTACT_MARGIN,
        )
        self.ball_body_indices = []
        for ball_index, (local_position, color) in enumerate(zip(BALL_LOCAL_POSITIONS, BALL_COLORS, strict=True)):
            position = BAG_POSITION + np.asarray(local_position, dtype=np.float32)
            body = builder.add_body(
                xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
                label=f"bag ball {ball_index}",
            )
            # Enter mid-fall so the balls do not share the bag's initial downward transient.
            builder.body_qd[body] = wp.spatial_vector(0.0, 0.0, -BALL_INITIAL_DOWNWARD_SPEED, 0.0, 0.0, 0.0)
            builder.add_shape_sphere(
                body,
                radius=BALL_RADIUS,
                cfg=ball_cfg,
                color=color,
                label=f"bag ball {ball_index} shape",
            )
            self.ball_body_indices.append(body)
        self.ball_body_indices = np.asarray(self.ball_body_indices, dtype=np.int32)

        builder.add_ground_plane()
        builder.color(include_bending=True)

        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = SOFT_CONTACT_KE
        self.model.soft_contact_kd = SOFT_CONTACT_KD
        self.model.soft_contact_mu = SOFT_CONTACT_FRICTION

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=SOFT_CONTACT_MARGIN,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=VBD_ITERATIONS,
            friction_epsilon=1.0e-4,
            particle_enable_self_contact=True,
            particle_self_contact_radius=BAG_PARTICLE_RADIUS,
            particle_self_contact_margin=SELF_CONTACT_MARGIN,
            particle_vertex_contact_buffer_size=48,
            particle_edge_contact_buffer_size=96,
            particle_collision_detection_interval=-1,
            particle_topological_contact_filter_threshold=2,
            particle_rest_shape_contact_exclusion_radius=SELF_CONTACT_MARGIN,
            rigid_body_particle_contact_buffer_size=1024,
        )

        self.render_indices = self.model.tri_indices.flatten()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = False
        self.viewer.set_camera(pos=wp.vec3(0.66, -1.0, 0.52), pitch=-2.0, yaw=124.0)

        self.capture()

    def capture(self):
        """Capture one simulation frame when running on CUDA."""
        if not self.model.device.is_cuda:
            self.graph = None
            return

        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        """Advance the VBD state by one rendered frame."""
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
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance the simulation by one frame."""
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the complete bag as a smooth double-layer surface."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/plastic_bag",
            self.state_0.particle_q,
            self.render_indices,
            backface_culling=False,
            color=BAG_COLOR,
            roughness=0.65,
            metallic=0.0,
            opacity=BAG_OPACITY,
        )
        self.viewer.end_frame()

    def test_final(self):
        """Verify that the bag remains finite, bounded, and on the rod."""
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(velocities)):
            raise ValueError("Plastic bag state is not finite")
        if np.max(np.abs(positions)) > 2.0:
            raise ValueError("Plastic bag left the expected simulation volume")
        if float(positions[:, 2].min()) < -0.01:
            raise ValueError("Plastic bag penetrated the ground")
        body_edge_lengths = np.linalg.norm(
            positions[self.bag_body_edges[:, 1]] - positions[self.bag_body_edges[:, 0]], axis=1
        )
        body_edge_stretch_99 = float(np.quantile(body_edge_lengths / self.rest_body_edge_lengths, 0.99))
        if body_edge_stretch_99 > MAX_BODY_EDGE_STRETCH_RATIO_99:
            raise ValueError(f"Plastic bag 99th-percentile body edge stretch reached {body_edge_stretch_99:.6g}")

        body_q = self.state_0.body_q.numpy()
        ball_positions = body_q[self.ball_body_indices, :3]
        if not np.all(np.isfinite(ball_positions)):
            raise ValueError("Rigid ball positions are not finite")
        if np.max(np.abs(ball_positions[:, 0])) > 0.2 or np.max(np.abs(ball_positions[:, 1])) > 0.08:
            raise ValueError("A rigid ball escaped through a side of the bag")
        if float(ball_positions[:, 2].min()) < float(positions[:, 2].min()) - BALL_RADIUS:
            raise ValueError("A rigid ball fell through the bottom of the bag")
        if float(ball_positions[:, 2].max()) > float(self.rod_position[2]):
            raise ValueError("A rigid ball escaped through the bag opening")
        bottom_row_limit = float(BAG_POSITION[2] + BALL_RADIUS + BALL_BOTTOM_CLEARANCE)
        if float(ball_positions[:, 2].max()) > bottom_row_limit:
            raise ValueError("A rigid ball remained pinched above the bag bottom")

        bag_triangles = positions[self.bag_triangle_indices]
        for ball_index, ball_position in enumerate(ball_positions):
            surface_distance = float(_point_triangle_distances(ball_position, bag_triangles).min())
            if surface_distance < BALL_RADIUS - BALL_PENETRATION_TOLERANCE:
                penetration = BALL_RADIUS - surface_distance
                raise ValueError(f"Rigid ball {ball_index} penetrated the plastic bag by {penetration:.6g} m")

            vertex_offsets = positions - ball_position
            contact_distance = np.linalg.norm(vertex_offsets, axis=1)
            contact_shell = contact_distance < BALL_RADIUS + BAG_PARTICLE_RADIUS + SOFT_CONTACT_MARGIN
            if not np.any(contact_shell) or float(vertex_offsets[contact_shell, 2].min()) > (
                -BALL_BOTTOM_CONTACT_FRACTION * BALL_RADIUS
            ):
                raise ValueError(f"Rigid ball {ball_index} did not reach the plastic bag bottom")

        rod_yz = self.rod_position[1:]
        left_distance = np.linalg.norm(positions[self.left_handle_indices, 1:] - rod_yz, axis=1).min()
        right_distance = np.linalg.norm(positions[self.right_handle_indices, 1:] - rod_yz, axis=1).min()
        contact_limit = ROD_RADIUS + BAG_PARTICLE_RADIUS + SOFT_CONTACT_MARGIN
        if left_distance > contact_limit or right_distance > contact_limit:
            raise ValueError(
                "Plastic bag slipped off the rod: "
                f"handle distances are {left_distance:.6g} m and {right_distance:.6g} m"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--bag-mesh",
        type=Path,
        default=BAG_MESH_PATH,
        help="Path to the triangulated plastic bag OBJ.",
    )
    parser.set_defaults(num_frames=240)

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

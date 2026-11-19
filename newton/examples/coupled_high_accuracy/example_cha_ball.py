# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop a volumetric elastic ball onto the ground with Style3D Sim."""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import coupled_high_accuracy as s3d

BALL_RADIUS = 0.5
BALL_DROP_HEIGHT = 2.5
BALL_DENSITY = 600.0
BALL_K_MU = 2.0e4
BALL_K_LAMBDA = 5.0e4
BALL_K_DAMP = 8.0
PARTICLE_RADIUS = 0.02

CONTACT_STIFFNESS = 1.0e5
CONTACT_DAMPING = 20.0
CONTACT_FRICTION = 0.4


def create_tetrahedral_ball(
    radius: float = BALL_RADIUS,
    subdivisions: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a geodesic sphere filled by tetrahedra meeting at its center."""
    if radius <= 0.0:
        raise ValueError("Ball radius must be positive.")
    if subdivisions < 0:
        raise ValueError("Subdivision count must be non-negative.")

    golden_ratio = 0.5 * (1.0 + math.sqrt(5.0))
    vertices = [
        (-1.0, golden_ratio, 0.0),
        (1.0, golden_ratio, 0.0),
        (-1.0, -golden_ratio, 0.0),
        (1.0, -golden_ratio, 0.0),
        (0.0, -1.0, golden_ratio),
        (0.0, 1.0, golden_ratio),
        (0.0, -1.0, -golden_ratio),
        (0.0, 1.0, -golden_ratio),
        (golden_ratio, 0.0, -1.0),
        (golden_ratio, 0.0, 1.0),
        (-golden_ratio, 0.0, -1.0),
        (-golden_ratio, 0.0, 1.0),
    ]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    vertices_array = np.asarray(vertices, dtype=np.float64)
    vertices_array *= radius / np.linalg.norm(vertices_array, axis=1, keepdims=True)
    vertices = [vertex for vertex in vertices_array]

    for _ in range(subdivisions):
        midpoint_cache: dict[tuple[int, int], int] = {}

        def midpoint(first: int, second: int) -> int:
            edge = (first, second) if first < second else (second, first)
            cached = midpoint_cache.get(edge)
            if cached is not None:
                return cached

            point = 0.5 * (vertices[first] + vertices[second])
            point *= radius / np.linalg.norm(point)
            index = len(vertices)
            vertices.append(point)
            midpoint_cache[edge] = index
            return index

        refined_faces: list[tuple[int, int, int]] = []
        for first, second, third in faces:
            first_second = midpoint(first, second)
            second_third = midpoint(second, third)
            third_first = midpoint(third, first)
            refined_faces.extend(
                (
                    (first, first_second, third_first),
                    (second, second_third, first_second),
                    (third, third_first, second_third),
                    (first_second, second_third, third_first),
                )
            )
        faces = refined_faces

    surface_vertices = np.asarray(vertices, dtype=np.float64)
    all_vertices = np.vstack((np.zeros((1, 3), dtype=np.float64), surface_vertices))
    surface_faces = np.asarray(faces, dtype=np.int32) + 1
    tetrahedra = np.column_stack((np.zeros(len(surface_faces), dtype=np.int32), surface_faces))

    v0 = all_vertices[tetrahedra[:, 0]]
    v1 = all_vertices[tetrahedra[:, 1]]
    v2 = all_vertices[tetrahedra[:, 2]]
    v3 = all_vertices[tetrahedra[:, 3]]
    signed_six_volumes = np.einsum(
        "ij,ij->i",
        v1 - v0,
        np.cross(v2 - v0, v3 - v0),
    )

    inverted = signed_six_volumes < 0.0
    swapped = tetrahedra[inverted, 2].copy()
    tetrahedra[inverted, 2] = tetrahedra[inverted, 3]
    tetrahedra[inverted, 3] = swapped

    minimum_six_volume = np.finfo(np.float32).eps * radius**3
    if np.any(np.abs(signed_six_volumes) <= minimum_six_volume):
        raise ValueError("Generated ball contains a degenerate tetrahedron.")

    return all_vertices.astype(np.float32), tetrahedra.astype(np.int32)


class Example:
    def __init__(self, viewer, args):
        self.args = args
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)

        ground_cfg = builder.default_shape_cfg.copy()
        ground_cfg.ke = CONTACT_STIFFNESS
        ground_cfg.kd = CONTACT_DAMPING
        ground_cfg.kf = 1.0e3
        ground_cfg.mu = CONTACT_FRICTION
        builder.add_ground_plane(cfg=ground_cfg)

        vertices, tetrahedra = create_tetrahedral_ball()
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, BALL_DROP_HEIGHT),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=tetrahedra.flatten(),
            density=BALL_DENSITY,
            k_mu=BALL_K_MU,
            k_lambda=BALL_K_LAMBDA,
            k_damp=BALL_K_DAMP,
            particle_radius=PARTICLE_RADIUS,
            validate_mesh=True,
            label="elastic_ball",
        )

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.model.soft_contact_ke = CONTACT_STIFFNESS
        self.model.soft_contact_kd = CONTACT_DAMPING
        self.model.soft_contact_kf = 1.0e3
        self.model.soft_contact_mu = CONTACT_FRICTION

        self.solver = s3d.SolverCoupledHighAccuracy(self.model)
        self.solver.config.nonlinear_iterations = 15
        self.solver.config.linear_iterations = 8

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        if args.initial_stretch != 1.0:
            positions = self.state_0.particle_q.numpy()
            positions[:, 0] *= args.initial_stretch
            self.state_0.particle_q.assign(positions)
        if args.initial_vx != 0.0:
            # A horizontal launch velocity makes the ground contact slide, exercising the friction limit.
            velocities = self.state_0.particle_qd.numpy()
            velocities[:, 0] += args.initial_vx
            self.state_0.particle_qd.assign(velocities)

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = True
        self.viewer.set_camera(
            pos=wp.vec3(4.0, -6.0, 2.8),
            pitch=-10.0,
            yaw=125.0,
        )

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda and not self.args.no_capture:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        lower = wp.vec3(-3.0, -3.0, -0.1)
        upper = wp.vec3(3.0, 3.0, 4.0)
        newton.examples.test_particle_state(
            self.state_0,
            "elastic ball remains in a finite volume above the ground",
            lambda q, _qd: newton.math.vec_inside_limits(q, lower, upper),
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--no-capture", action="store_true")
        parser.add_argument(
            "--initial-stretch",
            type=float,
            default=1.0,
            help="Stretch initial x coordinates to exercise tetrahedral elasticity.",
        )
        parser.add_argument(
            "--initial-vx", type=float, default=0.0, help="Initial horizontal velocity [m/s] so ground contact slides."
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

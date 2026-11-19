# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop a block of self-colliding particles onto a four-corner cloth.

Command: uv run python -m newton.examples cha_cloth_particles
Headless: uv run python -m newton.examples cha_cloth_particles --viewer null --num-frames 100 --test
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import coupled_high_accuracy as s3d
from newton.solvers import style3d

CLOTH_HEIGHT = 1.25
CLOTH_DIM_X = 24
CLOTH_DIM_Y = 24
CLOTH_CELL_SIZE = 0.1
CLOTH_YAW = math.radians(22.5)
CLOTH_PARTICLE_MASS = 0.02
CLOTH_PARTICLE_RADIUS = 0.012
CLOTH_TRI_STIFFNESS = wp.vec3(2.5e5, 2.5e5, 7.5e4)
CLOTH_EDGE_STIFFNESS = wp.vec3(4.0e-4, 4.0e-4, 4.0e-4)

PARTICLE_DIMENSIONS = (20, 20, 20)
PARTICLE_DROP_HEIGHT = 1.9
PARTICLE_SPACING = 0.05
PARTICLE_RADIUS = 0.02
PARTICLE_MASS = 0.01

SELF_COLLISION_RADIUS = CLOTH_PARTICLE_RADIUS

GROUND_CONTACT_STIFFNESS = 1.0e5
GROUND_CONTACT_DAMPING = 50.0
GROUND_CONTACT_FRICTION = 0.6


def _centered_cloth_origin() -> wp.vec3:
    """Return the origin of a yawed cloth whose center is on the Z axis."""
    half_x = 0.5 * CLOTH_DIM_X * CLOTH_CELL_SIZE
    half_y = 0.5 * CLOTH_DIM_Y * CLOTH_CELL_SIZE
    cosine = math.cos(CLOTH_YAW)
    sine = math.sin(CLOTH_YAW)
    rotated_center_x = cosine * half_x - sine * half_y
    rotated_center_y = sine * half_x + cosine * half_y
    return wp.vec3(-rotated_center_x, -rotated_center_y, CLOTH_HEIGHT)


def _centered_particle_origin() -> wp.vec3:
    width, depth, _ = PARTICLE_DIMENSIONS
    return wp.vec3(
        -0.5 * (width - 1) * PARTICLE_SPACING,
        -0.5 * (depth - 1) * PARTICLE_SPACING,
        PARTICLE_DROP_HEIGHT,
    )


class Example:
    def __init__(self, viewer, args):
        del args
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)

        ground_cfg = builder.default_shape_cfg.copy()
        ground_cfg.ke = GROUND_CONTACT_STIFFNESS
        ground_cfg.kd = GROUND_CONTACT_DAMPING
        ground_cfg.kf = 1.0e3
        ground_cfg.mu = GROUND_CONTACT_FRICTION
        builder.add_ground_plane(cfg=ground_cfg, label="ground_below_cloth")

        self.cloth_start = len(builder.particle_q)
        style3d.add_cloth_grid(
            builder,
            pos=_centered_cloth_origin(),
            rot=wp.quat_from_axis_angle(
                wp.vec3(0.0, 0.0, 1.0),
                CLOTH_YAW,
            ),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=CLOTH_DIM_X,
            dim_y=CLOTH_DIM_Y,
            cell_x=CLOTH_CELL_SIZE,
            cell_y=CLOTH_CELL_SIZE,
            mass=CLOTH_PARTICLE_MASS,
            tri_aniso_ke=CLOTH_TRI_STIFFNESS,
            edge_aniso_ke=CLOTH_EDGE_STIFFNESS,
            tri_kd=50.0,
            edge_kd=5.0,
            particle_radius=CLOTH_PARTICLE_RADIUS,
            label="four_corner_cloth",
        )
        self.cloth_stop = len(builder.particle_q)

        row_width = CLOTH_DIM_X + 1
        self.corner_indices = np.asarray(
            (
                self.cloth_start,
                self.cloth_start + CLOTH_DIM_X,
                self.cloth_start + CLOTH_DIM_Y * row_width,
                self.cloth_start + CLOTH_DIM_Y * row_width + CLOTH_DIM_X,
            ),
            dtype=np.int32,
        )
        self.cloth_center_index = self.cloth_start + (CLOTH_DIM_Y // 2) * row_width + CLOTH_DIM_X // 2
        for particle_index in self.corner_indices:
            builder.particle_flags[particle_index] &= ~newton.ParticleFlags.ACTIVE
            builder.particle_mass[particle_index] = 0.0

        self.free_particle_start = len(builder.particle_q)
        builder.add_particle_grid(
            pos=_centered_particle_origin(),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=PARTICLE_DIMENSIONS[0],
            dim_y=PARTICLE_DIMENSIONS[1],
            dim_z=PARTICLE_DIMENSIONS[2],
            cell_x=PARTICLE_SPACING,
            cell_y=PARTICLE_SPACING,
            cell_z=PARTICLE_SPACING,
            mass=PARTICLE_MASS,
            jitter=0.008,
            radius_mean=PARTICLE_RADIUS,
        )
        self.free_particle_stop = len(builder.particle_q)

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.model.soft_contact_ke = GROUND_CONTACT_STIFFNESS
        self.model.soft_contact_kd = GROUND_CONTACT_DAMPING
        self.model.soft_contact_kf = 1.0e3
        self.model.soft_contact_mu = GROUND_CONTACT_FRICTION
        initial_positions = self.model.particle_q.numpy()
        self.initial_corner_positions = initial_positions[self.corner_indices].copy()
        self.initial_cloth_center_height = float(initial_positions[self.cloth_center_index, 2])

        self.solver = s3d.SolverCoupledHighAccuracy(self.model)
        self.solver.config.nonlinear_iterations = 20
        self.solver.config.linear_iterations = 12
        self.solver.config.enable_self_collision = True
        self.solver.config.self_collision_radius = SELF_COLLISION_RADIUS
        self.solver.config.self_collision_vertex_face_stiffness = 0.5
        self.solver.config.self_collision_edge_edge_stiffness = 0.1
        self.solver.config.self_collision_edge_face_stiffness = 1.0
        self.solver.config.self_collision_particle_particle_stiffness = 0.25

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.stream = wp.get_stream(self.model.device)

        self.particles_reached_cloth = False
        self.minimum_cloth_center_height = self.initial_cloth_center_height
        self.maximum_particles_near_cloth = 0

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        self.viewer.show_triangles = True
        self.viewer.set_camera(
            pos=wp.vec3(4.0, -5.5, 3.3),
            pitch=-14.0,
            yaw=126.0,
        )

        self.capture()

    def capture(self):
        if self.model.device.is_cuda:
            with wp.ScopedCapture(stream=self.stream) as capture:
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
            wp.capture_launch(self.graph, stream=self.stream)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def test_post_step(self):
        positions = self.state_0.particle_q.numpy()
        free_positions = positions[self.free_particle_start : self.free_particle_stop]
        cloth_center_height = float(positions[self.cloth_center_index, 2])
        near_cloth = int(
            np.count_nonzero(free_positions[:, 2] <= self.initial_cloth_center_height + 2.0 * PARTICLE_RADIUS)
        )

        self.minimum_cloth_center_height = min(
            self.minimum_cloth_center_height,
            cloth_center_height,
        )
        self.maximum_particles_near_cloth = max(
            self.maximum_particles_near_cloth,
            near_cloth,
        )
        self.particles_reached_cloth |= near_cloth > 0

    def test_final(self):
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("Cloth-particle simulation produced a non-finite state")

        np.testing.assert_allclose(
            positions[self.corner_indices],
            self.initial_corner_positions,
            rtol=0.0,
            atol=1.0e-6,
            err_msg="The four fixed cloth corners moved",
        )

        cloth_flags = self.model.particle_flags.numpy()[self.cloth_start : self.cloth_stop]
        cloth_masses = self.model.particle_mass.numpy()[self.cloth_start : self.cloth_stop]
        inactive = np.flatnonzero((cloth_flags & int(newton.ParticleFlags.ACTIVE)) == 0)
        zero_mass = np.flatnonzero(cloth_masses == 0.0)
        expected_inactive = self.corner_indices - self.cloth_start
        if not np.array_equal(inactive, expected_inactive) or not np.array_equal(
            zero_mass,
            expected_inactive,
        ):
            raise ValueError("Exactly the four cloth corners must be fixed")

        if not self.particles_reached_cloth:
            raise ValueError("The free particles never reached the cloth")
        if self.maximum_particles_near_cloth < PARTICLE_DIMENSIONS[0]:
            raise ValueError("Too few free particles reached the cloth surface")
        if self.minimum_cloth_center_height >= self.initial_cloth_center_height - 0.08:
            raise ValueError("The suspended cloth did not deform under the particles")

        free_positions = positions[self.free_particle_start : self.free_particle_stop]
        pair_offsets = free_positions[:, None, :] - free_positions[None, :, :]
        pair_distances = np.linalg.norm(pair_offsets, axis=2)
        np.fill_diagonal(pair_distances, np.inf)
        minimum_pair_distance = float(pair_distances.min())
        if minimum_pair_distance <= 1.05 * PARTICLE_RADIUS:
            raise ValueError(
                "Free-particle self-collision allowed excessive overlap: "
                f"minimum center distance is {minimum_pair_distance:.5f} m"
            )

        cloth_positions = positions[self.cloth_start : self.cloth_stop]
        cloth_lower = np.array((-3.0, -3.0, -1.5), dtype=np.float32)
        cloth_upper = np.array((3.0, 3.0, 1.5), dtype=np.float32)
        if np.any(cloth_positions < cloth_lower) or np.any(cloth_positions > cloth_upper):
            raise ValueError("The cloth left its expected simulation volume")

        particle_lower = np.array((-6.0, -6.0, -0.1), dtype=np.float32)
        particle_upper = np.array((6.0, 6.0, 6.0), dtype=np.float32)
        if np.any(free_positions < particle_lower) or np.any(free_positions > particle_upper):
            raise ValueError("The free particles left their expected simulation volume")

        maximum_speed = float(np.linalg.norm(velocities, axis=1).max())
        if maximum_speed >= 20.0:
            raise ValueError(f"The cloth-particle simulation reached an excessive speed of {maximum_speed:.3f} m/s")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=100)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

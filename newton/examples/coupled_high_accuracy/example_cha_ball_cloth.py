# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop the volumetric elastic ball onto a four-corner-supported cloth."""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.coupled_high_accuracy.example_cha_ball import (
    BALL_DENSITY,
    BALL_K_DAMP,
    BALL_K_LAMBDA,
    BALL_K_MU,
    BALL_RADIUS,
    PARTICLE_RADIUS,
    create_tetrahedral_ball,
)
from newton.solvers import coupled_high_accuracy as s3d
from newton.solvers import style3d

BALL_DROP_HEIGHT = 2.7

CLOTH_HEIGHT = 1.3
CLOTH_DIM_X = 24
CLOTH_DIM_Y = 24
CLOTH_CELL_SIZE = 0.1
CLOTH_YAW = math.radians(25.0)
CLOTH_PARTICLE_MASS = 0.025
CLOTH_PARTICLE_RADIUS = 0.015
CLOTH_TRI_STIFFNESS = wp.vec3(3.0e5, 3.0e5, 8.0e4)
CLOTH_EDGE_STIFFNESS = wp.vec3(4.0e-4, 4.0e-4, 4.0e-4)

SELF_COLLISION_RADIUS = 0.01


def _cloth_origin() -> wp.vec3:
    """Return the translation that keeps the yaw-rotated cloth centered."""
    half_x = 0.5 * CLOTH_DIM_X * CLOTH_CELL_SIZE
    half_y = 0.5 * CLOTH_DIM_Y * CLOTH_CELL_SIZE
    cosine = math.cos(CLOTH_YAW)
    sine = math.sin(CLOTH_YAW)
    rotated_center_x = cosine * half_x - sine * half_y
    rotated_center_y = sine * half_x + cosine * half_y
    return wp.vec3(-rotated_center_x, -rotated_center_y, CLOTH_HEIGHT)


class Example:
    @staticmethod
    def _signed_six_volumes(tetrahedra: np.ndarray) -> np.ndarray:
        return np.einsum(
            "ij,ij->i",
            tetrahedra[:, 1] - tetrahedra[:, 0],
            np.cross(
                tetrahedra[:, 2] - tetrahedra[:, 0],
                tetrahedra[:, 3] - tetrahedra[:, 0],
            ),
        )

    def __init__(self, viewer, args):
        del args
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)

        self.ball_start = len(builder.particle_q)
        ball_vertices, ball_tetrahedra = create_tetrahedral_ball()
        self.ball_tetrahedra = ball_tetrahedra
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, BALL_DROP_HEIGHT),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=ball_vertices,
            indices=ball_tetrahedra.flatten(),
            density=BALL_DENSITY,
            k_mu=BALL_K_MU,
            k_lambda=BALL_K_LAMBDA,
            k_damp=BALL_K_DAMP,
            particle_radius=PARTICLE_RADIUS,
            validate_mesh=True,
            label="elastic_ball",
        )
        self.ball_stop = len(builder.particle_q)

        self.cloth_start = len(builder.particle_q)
        style3d.add_cloth_grid(
            builder,
            pos=_cloth_origin(),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), CLOTH_YAW),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=CLOTH_DIM_X,
            dim_y=CLOTH_DIM_Y,
            cell_x=CLOTH_CELL_SIZE,
            cell_y=CLOTH_CELL_SIZE,
            mass=CLOTH_PARTICLE_MASS,
            tri_aniso_ke=CLOTH_TRI_STIFFNESS,
            edge_aniso_ke=CLOTH_EDGE_STIFFNESS,
            tri_kd=8.0,
            edge_kd=1.0,
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

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        initial_positions = self.model.particle_q.numpy()
        self.initial_corner_positions = initial_positions[self.corner_indices].copy()
        ball_masses = self.model.particle_mass.numpy()[self.ball_start : self.ball_stop]
        self.ball_mass_weights = ball_masses / np.sum(ball_masses)
        initial_ball_positions = initial_positions[self.ball_start : self.ball_stop]
        self.initial_ball_signed_six_volumes = self._signed_six_volumes(initial_ball_positions[self.ball_tetrahedra])

        self.solver = s3d.SolverCoupledHighAccuracy(self.model)
        self.solver.config.nonlinear_iterations = 20
        self.solver.config.linear_iterations = 10
        self.solver.config.enable_self_collision = True
        self.solver.config.self_collision_radius = SELF_COLLISION_RADIUS
        self.solver.config.self_collision_vertex_face_stiffness = 0.5
        self.solver.config.self_collision_edge_edge_stiffness = 0.1
        self.solver.config.self_collision_edge_face_stiffness = 1.0

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.ball_reached_cloth = False
        self.ball_rebounded = False
        self.maximum_rebound_velocity = 0.0
        self.minimum_cloth_center_height = CLOTH_HEIGHT
        self.minimum_post_contact_ball_height = math.inf
        self.minimum_post_contact_center_gap = math.inf

        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = True
        self.viewer.set_camera(
            pos=wp.vec3(4.2, -5.8, 3.2),
            pitch=-14.0,
            yaw=126.0,
        )

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
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

    def test_post_step(self):
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        ball_positions = positions[self.ball_start : self.ball_stop]
        ball_velocities = velocities[self.ball_start : self.ball_stop]
        ball_center = np.sum(ball_positions * self.ball_mass_weights[:, None], axis=0)
        ball_velocity = np.sum(
            ball_velocities * self.ball_mass_weights[:, None],
            axis=0,
        )
        cloth_center = positions[self.cloth_center_index]
        center_gap = float(ball_center[2] - cloth_center[2])
        horizontal_gap = float(np.linalg.norm(ball_center[:2] - cloth_center[:2]))

        contact_distance = BALL_RADIUS + 5.0 * SELF_COLLISION_RADIUS
        if (
            not self.ball_reached_cloth
            and ball_velocity[2] < -0.25
            and 0.0 < center_gap <= contact_distance
            and horizontal_gap <= BALL_RADIUS
        ):
            self.ball_reached_cloth = True

        if not self.ball_reached_cloth:
            return

        center_height = float(ball_center[2])
        center_velocity = float(ball_velocity[2])
        self.minimum_cloth_center_height = min(
            self.minimum_cloth_center_height,
            float(cloth_center[2]),
        )
        self.minimum_post_contact_ball_height = min(
            self.minimum_post_contact_ball_height,
            center_height,
        )
        self.minimum_post_contact_center_gap = min(
            self.minimum_post_contact_center_gap,
            center_gap,
        )
        if center_velocity > 0.25 and center_height >= self.minimum_post_contact_ball_height + 0.15:
            self.ball_rebounded = True
            self.maximum_rebound_velocity = max(
                self.maximum_rebound_velocity,
                center_velocity,
            )

    def test_final(self):
        positions = self.state_0.particle_q.numpy()
        velocities = self.state_0.particle_qd.numpy()
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise ValueError("Ball-cloth simulation produced a non-finite particle state")

        np.testing.assert_allclose(
            positions[self.corner_indices],
            self.initial_corner_positions,
            rtol=0.0,
            atol=1.0e-6,
            err_msg="The four fixed cloth corners moved",
        )

        cloth_flags = self.model.particle_flags.numpy()[self.cloth_start : self.cloth_stop]
        inactive = np.flatnonzero((cloth_flags & int(newton.ParticleFlags.ACTIVE)) == 0)
        cloth_masses = self.model.particle_mass.numpy()[self.cloth_start : self.cloth_stop]
        zero_mass = np.flatnonzero(cloth_masses == 0.0)
        expected_inactive = self.corner_indices - self.cloth_start
        if not np.array_equal(inactive, expected_inactive) or not np.array_equal(
            zero_mass,
            expected_inactive,
        ):
            raise ValueError("Exactly the four cloth corners must be fixed")

        if not self.ball_reached_cloth:
            raise ValueError("The falling ball never reached the cloth during the test")
        if self.minimum_cloth_center_height >= CLOTH_HEIGHT - 0.2:
            raise ValueError("The suspended cloth center did not deform under the ball")
        if self.minimum_post_contact_center_gap <= 0.05:
            raise ValueError("The elastic ball center crossed through the cloth center")
        if not self.ball_rebounded:
            raise ValueError("The elastic ball did not rebound after descending onto the cloth")

        ball_positions = positions[self.ball_start : self.ball_stop]
        cloth_positions = positions[self.cloth_start : self.cloth_stop]
        cloth_lower = np.array((-3.0, -3.0, -1.0), dtype=np.float32)
        cloth_upper = np.array((3.0, 3.0, 2.0), dtype=np.float32)
        if np.any(cloth_positions < cloth_lower) or np.any(cloth_positions > cloth_upper):
            raise ValueError("The cloth left its expected simulation volume")

        ball_lower = np.array((-3.0, -3.0, -0.5), dtype=np.float32)
        ball_upper = np.array((3.0, 3.0, 4.5), dtype=np.float32)
        if np.any(ball_positions < ball_lower) or np.any(ball_positions > ball_upper):
            raise ValueError("The elastic ball left its expected simulation volume")

        maximum_speed = float(np.linalg.norm(velocities, axis=1).max())
        if maximum_speed >= 25.0:
            raise ValueError(f"The ball-cloth simulation reached an excessive speed of {maximum_speed:.3f} m/s")

        tetrahedra = ball_positions[self.ball_tetrahedra]
        signed_six_volumes = self._signed_six_volumes(tetrahedra)
        volume_ratios = signed_six_volumes / self.initial_ball_signed_six_volumes
        if np.any(volume_ratios < 0.25) or np.any(volume_ratios > 4.0):
            raise ValueError("The elastic ball contains an inverted or excessively distorted tetrahedron")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        return newton.examples.create_parser()


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

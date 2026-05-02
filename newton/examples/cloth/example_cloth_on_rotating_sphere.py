# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Sim Cloth on Rotating Sphere
#
# This simulation demonstrates a cloth falling onto a rotating sphere,
# similar to PhysX's SnippetPBDCloth example. The cloth drapes over
# the sphere and is affected by its rotation.
#
# Command: python -m newton.examples cloth_on_rotating_sphere
#
###########################################################################

import warp as wp

import newton
import newton.examples


@wp.kernel
def set_rotating_sphere_state(
    joint_q_start: int,
    joint_qd_start: int,
    sim_time: wp.array[wp.float32],
    angular_speed: float,
    # outputs
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
):
    """Set prescribed state for the sphere's revolute root joint."""
    angle = angular_speed * sim_time[0]
    joint_q[joint_q_start] = angle
    joint_qd[joint_qd_start] = angular_speed


@wp.kernel
def advance_time(sim_time: wp.array[wp.float32], dt: float):
    sim_time[0] = sim_time[0] + dt


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 10
        self.sim_time = 0.0

        self.viewer = viewer

        # Sphere parameters
        self.sphere_radius = 1.5
        self.sphere_height = 2.5
        self.sphere_angular_speed = 2.0  # rad/s

        builder = newton.ModelBuilder()

        # Ground plane
        ground_cfg = newton.ModelBuilder.ShapeConfig()
        ground_cfg.ke = 1.0e5
        ground_cfg.kd = 1.0e0
        ground_cfg.mu = 0.5
        builder.add_ground_plane(cfg=ground_cfg)

        # Rotating sphere (kinematic body)
        sphere_cfg = newton.ModelBuilder.ShapeConfig()
        sphere_cfg.density = 0.0
        sphere_cfg.has_particle_collision = True
        sphere_cfg.ke = 1.0e4
        sphere_cfg.kd = 1.0e2
        sphere_cfg.mu = 0.5

        self.sphere_body = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(0.0, 0.0, self.sphere_height),
                q=wp.quat_identity(),
            ),
            mass=0.0,
            is_kinematic=True,
            label="rotating_sphere",
        )
        builder.add_shape_sphere(
            self.sphere_body,
            radius=self.sphere_radius,
            cfg=sphere_cfg,
            label="rotating_sphere_shape",
        )
        # Add a revolute joint for rotation around Z axis
        self.sphere_joint = builder.add_joint_revolute(
            parent=-1,
            child=self.sphere_body,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, self.sphere_height), q=wp.quat_identity()),
            label="sphere_joint",
        )
        builder.add_articulation([self.sphere_joint], label="rotating_sphere")

        # Cloth grid
        # PhysX: 250x250 particles with 0.05 spacing = 12.5m x 12.5m cloth
        # Total mass = 10kg
        num_points_x = 64
        num_points_z = 64
        particle_spacing = 0.1
        total_cloth_mass = 10.0
        cloth_size_x = num_points_x * particle_spacing
        cloth_size_z = num_points_z * particle_spacing
        cloth_height = 5.0  # Height of the cloth

        builder.add_cloth_grid(
            pos=wp.vec3(-cloth_size_x / 2, -cloth_size_z / 2, cloth_height),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=num_points_x,
            dim_y=num_points_z,
            cell_x=particle_spacing,
            cell_y=particle_spacing,
            mass=total_cloth_mass / ((num_points_x + 1) * (num_points_z + 1)),
            fix_left=False,
            fix_right=False,
            fix_top=False,
            fix_bottom=False,
            tri_ke=5.0e2,
            tri_ka=5.0e2,
            tri_kd=2.0e-1,
            edge_ke=1.0e-1,
            edge_kd=1.0e-1,
            particle_radius=particle_spacing * 0.3,
        )

        # Falling boxes
        box_cfg = newton.ModelBuilder.ShapeConfig()
        box_cfg.ke = 1.0e4
        box_cfg.kd = 1.0e3
        box_cfg.mu = 0.5
        box_cfg.has_particle_collision = True

        box_height = cloth_height + 2.0  # Boxes above the cloth
        for i in range(5):
            box_body = builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(i - 2.0, 2.0, box_height),
                    q=wp.quat_identity(),
                ),
                mass=5.0,
                label=f"box_{i}",
            )
            builder.add_shape_box(
                box_body,
                hx=0.5,
                hy=0.5,
                hz=0.5,
                cfg=box_cfg,
            )
            builder.add_articulation([builder.add_joint_free(box_body)], label=f"box_{i}")

        builder.color(include_bending=True)
        self.model = builder.finalize()

        # Contact parameters
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e1
        self.model.soft_contact_mu = 0.5

        # Use VBD solver with self-contact
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=particle_spacing * 0.25,
            particle_self_contact_margin=particle_spacing * 0.3,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Use CollisionPipeline for particle-rigid contact
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=0.02,
        )
        self.contacts = self.collision_pipeline.contacts()

        # Initialize body state from model joint buffers
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # Get joint coordinate starts for the sphere
        q_starts = self.model.joint_q_start.numpy()
        qd_starts = self.model.joint_qd_start.numpy()
        self.sphere_joint_q_start = int(q_starts[self.sphere_joint])
        self.sphere_joint_qd_start = int(qd_starts[self.sphere_joint])

        # Warp array for sim_time (needed for CUDA graph)
        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(20.0, -15.0, 12.0), -20.0, 140.0)
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

            # Update kinematic sphere rotation
            wp.launch(
                set_rotating_sphere_state,
                dim=1,
                inputs=[
                    self.sphere_joint_q_start,
                    self.sphere_joint_qd_start,
                    self.sim_time_wp,
                    self.sphere_angular_speed,
                ],
                outputs=[self.state_0.joint_q, self.state_0.joint_qd],
                device=self.model.device,
            )

            # Update maximal coordinates of kinematic bodies
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0, self.state_1, self.control, self.contacts, self.sim_dt
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

            wp.launch(advance_time, dim=1, inputs=[self.sim_time_wp, self.sim_dt], device=self.model.device)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "particles are above the ground",
            lambda q, qd: q[2] > 0.0,
        )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)

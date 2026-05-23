# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example: XPBD + VBD One-Way Coupling (Rigid Body + Soft Body)
#
# Demonstrates one-way coupling between a rigid body (XPBD) and a soft
# body (VBD). A rigid capsule rests on the ground while a soft cube drops
# onto it. The rigid body affects the soft body (collision response), but
# the soft body does NOT affect the rigid body.
#
# A capsule shape is used instead of a box because particle-shape
# collision cannot fully resolve sharp box corners — soft body particles
# tend to slip through corners due to discontinuous collision normals.
# Capsules provide smooth, continuous normals that prevent this artifact.
#
# Compare with:
#   - xpbd_vbd_rigid_cloth: Rigid body + cloth
#   - xpbd_vbd_joint_cloth: Articulated body + cloth
#   - mujoco_vbd_joint_cloth_soft: MuJoCo articulated body + cloth & soft body
#
# Command: uv run -m newton.examples xpbd_vbd_rigid_soft
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.multiphysics.one_way_coupling import step_one_way_coupling
from newton.solvers import SolverXPBD, SolverVBD


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()
        builder.default_shape_cfg.density = 100.0

        builder.add_ground_plane()

        # Rigid capsule (XPBD) — rests on the ground
        rigid_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        )
        builder.add_shape_capsule(body=rigid_body, radius=0.15, half_height=0.15)

        # Soft cube (VBD) — above the rigid capsule
        builder.add_soft_grid(
            pos=wp.vec3(-0.3, -0.3, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=6,
            dim_y=6,
            dim_z=6,
            cell_x=0.1,
            cell_y=0.1,
            cell_z=0.1,
            density=100.0,
            k_mu=1.0e5,
            k_lambda=1.0e5,
            k_damp=1e-4,
            particle_radius=0.02,
        )
        builder.color()

        self.model = builder.finalize()
        self.real_particle_count = self.model.particle_count

        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1e-3
        self.model.soft_contact_mu = 0.5

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=0.05,
        )
        self.contacts_xpbd = self.collision_pipeline.contacts()
        self.contacts_vbd = self.collision_pipeline.contacts()

        self.xpbd_solver = SolverXPBD(self.model, iterations=15)
        self.vbd_solver = SolverVBD(
            self.model,
            iterations=20,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=False,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(4.0, -4.0, 3.0),
            pitch=-25.0,
            yaw=45.0,
        )

        # No CUDA graph capture — we modify model counts between collide calls
        self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.viewer.apply_forces(self.state_0)

            step_one_way_coupling(
                rigid_solver=self.xpbd_solver,
                vbd_solver=self.vbd_solver,
                collision_pipeline=self.collision_pipeline,
                model=self.model,
                state_0=self.state_0,
                state_1=self.state_1,
                control=self.control,
                dt=self.sim_dt,
                real_particle_count=self.real_particle_count,
                contacts_rigid=self.contacts_xpbd,
                contacts_soft=self.contacts_vbd,
            )

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        particle_z = self.state_0.particle_q.numpy()[:, 2]
        mean_z = np.mean(particle_z)
        assert mean_z < 1.8, f"Soft cube didn't fall enough, mean z={mean_z:.3f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts_vbd, self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        return newton.examples.create_parser()


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
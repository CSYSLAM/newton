# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example: XPBD + VBD with NO coupling
#
# Demonstrates what happens when two solvers run independently without
# exchanging interaction data. The rigid cube (XPBD) and soft cube (VBD)
# ignore each other — the soft body passes straight through the rigid body.
#
# Both solvers still interact with the ground independently:
#   - XPBD gets ground-rigid shape contacts (rigid cube stays on ground)
#   - VBD gets ground-particle soft contacts (soft cube rests on ground)
#
# Compare with xpbd_vbd_rigid_soft which has collision coupling enabled.
#
# Command: uv run -m newton.examples xpbd_vbd_no_coupling
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
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

        # Rigid cube (XPBD)
        rigid_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        )
        builder.add_shape_box(body=rigid_body, hx=0.5, hy=0.5, hz=0.5)

        # Soft cube (VBD) — directly above the rigid cube
        builder.add_soft_grid(
            pos=wp.vec3(-0.3, -0.3, 2.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=6, dim_y=6, dim_z=6,
            cell_x=0.1, cell_y=0.1, cell_z=0.1,
            density=100.0,
            k_mu=1.0e5,
            k_lambda=1.0e5,
            k_damp=1e-4,
            particle_radius=0.02,
        )
        builder.color()

        self.model = builder.finalize()
        self.real_particle_count = self.model.particle_count
        self.real_shape_count = self.model.shape_count

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Collision pipeline for ground contacts
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=0.05,
        )

        # Two separate contacts objects:
        #   contacts_xpbd: only ground-rigid shape contacts (no particles)
        #   contacts_vbd: only ground-particle soft contacts (no rigid shapes)
        self.contacts_xpbd = self.collision_pipeline.contacts()
        self.contacts_vbd = self.collision_pipeline.contacts()

        # Contact parameters for VBD ground interaction
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1e-3
        self.model.soft_contact_mu = 0.5

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

            # Collision for XPBD: hide particles so only ground-rigid
            # shape contacts are generated
            self.model.particle_count = 0
            self.collision_pipeline.collide(self.state_0, self.contacts_xpbd)

            # XPBD: rigid body with ground contacts only
            self.xpbd_solver.step(self.state_0, self.state_1, self.control, self.contacts_xpbd, self.sim_dt)

            # Collision for VBD: restore particles, hide rigid shapes
            # (shape_count=1 means only ground plane) so only ground-particle
            # soft contacts are generated — no rigid-soft interaction
            self.model.particle_count = self.real_particle_count
            self.model.shape_count = 1  # ground plane only
            self.state_0.particle_f.zero_()
            self.collision_pipeline.collide(self.state_0, self.contacts_vbd)
            self.model.shape_count = self.real_shape_count

            # VBD: soft body with ground contacts only (no rigid body coupling)
            self.vbd_solver.step(self.state_0, self.state_1, self.control, self.contacts_vbd, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        particle_z = self.state_0.particle_q.numpy()[:, 2]
        mean_z = np.mean(particle_z)
        # Without coupling, the soft cube falls through the rigid cube
        # and lands on the ground (z ~ 0.3)
        assert mean_z < 1.0, f"Soft cube didn't fall through, mean z={mean_z:.3f}"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        return newton.examples.create_parser()


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)

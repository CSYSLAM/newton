# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example: MuJoCo + VBD One-Way Coupling (Articulated Body + Cloth & Soft Body)
#
# Demonstrates one-way coupling between an articulated rigid body (MuJoCo)
# and both cloth and soft body (VBD). A pendulum bar (revolute joint)
# swings back and forth while a cloth sheet and a soft cube drop onto it.
# The articulated body affects the soft bodies (collision response), but
# the soft bodies do NOT affect the articulated body.
#
# Capsule shapes are used for the bar because particle-shape collision
# cannot fully resolve sharp box corners — particles tend to slip through
# corners due to discontinuous collision normals.
#
# Compare with:
#   - xpbd_vbd_rigid_cloth: XPBD free rigid body + cloth
#   - xpbd_vbd_rigid_soft: XPBD rigid body + soft body
#   - xpbd_vbd_joint_cloth: XPBD articulated body + cloth
#
# Command: uv run -m newton.examples mujoco_vbd_joint_cloth_soft
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.multiphysics.one_way_coupling import step_one_way_coupling
from newton.solvers import SolverMuJoCo, SolverVBD


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        builder = newton.ModelBuilder()

        builder.add_ground_plane()

        # Pendulum bar (MuJoCo) — pivot at top, bar swings down
        bar_length = 0.5
        bar_radius = 0.06
        pivot_z = 0.8

        # Anchor link (fixed to world)
        anchor = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, pivot_z), q=wp.quat_identity()),
        )
        builder.add_shape_box(anchor, hx=0.05, hy=0.05, hz=0.05)

        j_fixed = builder.add_joint_fixed(
            parent=-1,
            child=anchor,
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, pivot_z), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.0), q=wp.quat_identity()),
        )

        # Bar link (swings around Z axis with initial angle)
        bar = builder.add_link(
            xform=wp.transform(
                p=wp.vec3(bar_length * 0.3, 0.0, pivot_z - bar_length * 0.5),
                q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.5),
            ),
            label="pendulum_bar",
        )
        bar_cfg = newton.ModelBuilder.ShapeConfig()
        bar_cfg.density = 5.0
        bar_cfg.has_particle_collision = True
        builder.add_shape_capsule(bar, radius=bar_radius, half_height=bar_length / 2, cfg=bar_cfg)

        # Revolute joint: bar pivots around Z at the anchor point
        j_rev = builder.add_joint_revolute(
            parent=anchor,
            child=bar,
            axis=wp.vec3(0.0, 0.0, 1.0),
            parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.025), q=wp.quat_identity()),
            child_xform=wp.transform(p=wp.vec3(0.0, 0.0, bar_length / 2 + 0.025), q=wp.quat_identity()),
        )

        builder.add_articulation([j_fixed, j_rev])

        # Cloth sheet (VBD) — drops from above the pivot
        cloth_size = 0.6
        dim = 20
        cell = cloth_size / dim
        builder.add_cloth_grid(
            pos=wp.vec3(-cloth_size / 2, -cloth_size / 2, pivot_z + 0.3),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim,
            dim_y=dim,
            cell_x=cell,
            cell_y=cell,
            mass=0.001,
            tri_ke=8.0e1,
            tri_ka=8.0e1,
            tri_kd=5.0e-1,
            edge_ke=5.0e-2,
            edge_kd=5.0e-1,
            particle_radius=0.015,
        )
        builder.color(include_bending=True)

        # Soft cube (VBD) — drops next to the pendulum
        builder.add_soft_grid(
            pos=wp.vec3(0.3, -0.15, pivot_z + 0.2),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=4,
            dim_y=4,
            dim_z=4,
            cell_x=0.05,
            cell_y=0.05,
            cell_z=0.05,
            density=100.0,
            k_mu=1.0e5,
            k_lambda=1.0e5,
            k_damp=1e-4,
            particle_radius=0.01,
        )

        self.model = builder.finalize()
        self.real_particle_count = self.model.particle_count

        self.model.soft_contact_ke = 5.0e3
        self.model.soft_contact_kd = 1.0e2
        self.model.soft_contact_mu = 0.8

        # Initialize body state from joint coordinates
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=0.02,
        )
        self.contacts_mujoco = self.collision_pipeline.contacts()
        self.contacts_vbd = self.collision_pipeline.contacts()

        self.mujoco_solver = SolverMuJoCo(self.model)
        self.vbd_solver = SolverVBD(
            self.model,
            iterations=20,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=False,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(2.0, -2.0, 1.5),
            pitch=-20.0,
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
                rigid_solver=self.mujoco_solver,
                vbd_solver=self.vbd_solver,
                collision_pipeline=self.collision_pipeline,
                model=self.model,
                state_0=self.state_0,
                state_1=self.state_1,
                control=self.control,
                dt=self.sim_dt,
                real_particle_count=self.real_particle_count,
                contacts_rigid=self.contacts_mujoco,
                contacts_soft=self.contacts_vbd,
            )

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def test_final(self):
        particle_z = self.state_0.particle_q.numpy()[:, 2]
        min_z = np.min(particle_z)
        assert min_z > -0.5, f"Particles exploded, min z={min_z:.3f}"

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
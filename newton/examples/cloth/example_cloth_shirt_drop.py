# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Shirt Drop
#
# This simulation places two square cloth sheets on a table and drops a shirt
# cloth mesh from above so it settles onto them.
#
# Command: python -m newton.examples cloth_shirt_drop
#
###########################################################################

import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps

        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 8

        self.viewer = viewer

        builder = newton.ModelBuilder(gravity=-9.81)

        shirt_tri_ke = 1.0e4
        shirt_tri_ka = 1.0e4
        shirt_tri_kd = 1.5e-6
        shirt_edge_ke = 5.0
        shirt_edge_kd = 1.0e-2
        shirt_density = 0.02
        shirt_particle_radius = 0.008

        soft_contact_ke = 1.0e4
        soft_contact_kd = 1.0e-2
        soft_contact_mu = 0.25

        shape_contact_ke = 5.0e4
        shape_contact_kd = 1.0e-3
        shape_contact_mu = 1.5
        shape_cfg = newton.ModelBuilder.ShapeConfig(
            ke=shape_contact_ke,
            kd=shape_contact_kd,
            mu=shape_contact_mu,
        )

        table_hx = 0.45
        table_hy = 0.45
        table_hz = 0.10
        builder.add_shape_box(
            -1,
            wp.transform((0.0, 0.0, table_hz), wp.quat_identity()),
            hx=table_hx,
            hy=table_hy,
            hz=table_hz,
            cfg=shape_cfg,
        )
        builder.add_ground_plane(cfg=shape_cfg)

        base_cloth_size = 0.32
        base_cloth_dim = 18
        base_cloth_z = 0.235
        base_cloth_centers = (-0.18, 0.18)
        self.base_cloth_indices = []

        for center_x in base_cloth_centers:
            cloth_start = builder.particle_count
            builder.add_cloth_grid(
                pos=wp.vec3(center_x - base_cloth_size * 0.5, -base_cloth_size * 0.5, base_cloth_z),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=base_cloth_dim,
                dim_y=base_cloth_dim,
                cell_x=base_cloth_size / base_cloth_dim,
                cell_y=base_cloth_size / base_cloth_dim,
                mass=0.03,
                fix_left=False,
                fix_right=False,
                fix_top=False,
                fix_bottom=False,
                tri_ke=1.0e3,
                tri_ka=1.0e3,
                tri_kd=5.0e-2,
                edge_ke=5.0e-1,
                edge_kd=5.0e-2,
                particle_radius=0.01,
            )
            self.base_cloth_indices.extend(range(cloth_start, builder.particle_count))

        usd_stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/shirt")

        shirt_mesh = newton.usd.get_mesh(usd_prim)
        shirt_vertices = [wp.vec3(v) for v in shirt_mesh.vertices]
        shirt_indices = shirt_mesh.indices

        shirt_start = builder.particle_count
        builder.add_cloth_mesh(
            vertices=shirt_vertices,
            indices=shirt_indices,
            pos=wp.vec3(0.0, 1.20, 0.42),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi),
            scale=0.01,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=shirt_density,
            tri_ke=shirt_tri_ke,
            tri_ka=shirt_tri_ka,
            tri_kd=shirt_tri_kd,
            edge_ke=shirt_edge_ke,
            edge_kd=shirt_edge_kd,
            particle_radius=shirt_particle_radius,
        )
        self.shirt_particle_indices = list(range(shirt_start, builder.particle_count))

        builder.color(include_bending=True)

        self.model = builder.finalize()
        self.model.soft_contact_ke = soft_contact_ke
        self.model.soft_contact_kd = soft_contact_kd
        self.model.soft_contact_mu = soft_contact_mu

        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.iterations,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.002,
            particle_self_contact_margin=0.002,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.0,
        )

        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=0.008,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.95, -0.90, 0.70),
            pitch=-18.0,
            yaw=132.0,
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
        self.solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.viewer.apply_forces(self.state_0)

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        p_lower = wp.vec3(-0.8, -0.8, -0.05)
        p_upper = wp.vec3(0.8, 0.8, 1.2)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 5.0,
        )

        particle_positions = self.state_0.particle_q.numpy()
        base_heights = particle_positions[self.base_cloth_indices, 2]
        shirt_heights = particle_positions[self.shirt_particle_indices, 2]

        assert float(base_heights.mean()) < 0.30, "base cloths did not stay near the table"
        assert float(shirt_heights.mean()) < 0.42, "shirt did not settle onto the base cloths"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=300)
    viewer, args = newton.examples.init(parser)

    newton.examples.run(Example(viewer, args), args)
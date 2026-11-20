# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot Cloth Trampoline
#
# Drops a Unitree G1 robot onto a deformable trampoline. The trampoline's
# outer ring is fixed while the remaining vertices are simulated as cloth.
#
# Command: python -m newton.examples cha_robot_cloth_trampoline
#
###########################################################################

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode, ParticleFlags
from newton.examples.coupled_high_accuracy.trampoline.mesh import (
    DEFAULT_OBJ,
    boundary_vertex_indices,
    load_trampoline_mesh,
    triangle_mesh_area,
)
from newton.solvers import coupled_high_accuracy as cha
from newton.solvers import style3d


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 6
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)

        contact_stiffness = 1.0e5
        contact_damping = 1.0e3

        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3,
            limit_kd=1.0e1,
            friction=1.0e-5,
        )
        builder.default_shape_cfg.ke = contact_stiffness
        builder.default_shape_cfg.kd = contact_damping
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("unitree_g1")
        builder.add_usd(
            str(asset_path / "usd_structured" / "g1_29dof_with_hand_rev_1_0.usda"),
            # The structured asset has its own root height; this offset releases the feet
            # approximately 0.3 m above the trampoline at z=1.0.
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.3)),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )

        for dof in range(6, builder.joint_dof_count):
            builder.joint_target_ke[dof] = 500.0
            builder.joint_target_kd[dof] = 10.0
            builder.joint_target_mode[dof] = int(JointTargetMode.POSITION)

        builder.approximate_meshes("bounding_box")

        # USD materials may override the builder defaults. Soft contacts use
        # the smaller of the particle and shape stiffness, so set every G1
        # collision shape explicitly to avoid silently limiting the response.
        for shape_index in range(len(builder.shape_body)):
            builder.shape_material_ke[shape_index] = contact_stiffness
            builder.shape_material_kd[shape_index] = contact_damping

        mesh_path = getattr(args, "trampoline_mesh", None)
        trampoline_vertices, trampoline_indices = load_trampoline_mesh(mesh_path)
        source = mesh_path or (DEFAULT_OBJ if DEFAULT_OBJ.is_file() else "generated circular mesh (radius 1 m)")
        print(f"[INFO] Trampoline: {source}, {len(trampoline_vertices)} vertices")
        trampoline_area = triangle_mesh_area(trampoline_vertices, trampoline_indices)
        trampoline_start = len(builder.particle_q)
        trampoline_mass = 5.0

        style3d.add_cloth_mesh(
            builder,
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            vertices=trampoline_vertices,
            indices=trampoline_indices,
            density=trampoline_mass / trampoline_area,
            tri_aniso_ke=wp.vec3(2.0e5, 2.0e5, 2e5),
            edge_aniso_ke=wp.vec3(2.0e-4, 2.0e-4, 2.0e-4),
            tri_kd=5.0,
            edge_kd=1.0,
            particle_radius=0.025,
            label="trampoline",
        )

        self.trampoline_particle_count = len(trampoline_vertices)
        self.fixed_rim = boundary_vertex_indices(trampoline_indices)
        for vertex_index in self.fixed_rim:
            particle_index = trampoline_start + vertex_index
            builder.particle_flags[particle_index] &= ~ParticleFlags.ACTIVE
            builder.particle_mass[particle_index] = 0.0

        builder.add_ground_plane()

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.model.soft_contact_ke = contact_stiffness
        self.model.soft_contact_kd = contact_damping
        self.model.soft_contact_mu = 0.6

        # Synreal builds affine proxies from model.body_q, so initialize the
        # authored articulation pose before constructing the solver.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)

        self.solver = cha.SolverCoupledHighAccuracy(self.model)
        self.solver.config.affine_body_rigidity_model = cha.AffineBodyRigidityModel.SIX_SPRINGS
        self.solver.config.joint_constraint_mode = cha.JointConstraintMode.AUGMENTED_LAGRANGIAN
        self.solver.config.nonlinear_solver = cha.NonlinearSolver.NEWTON
        self.solver.config.inertia_mode = cha.InertiaMode.MIDPOINT_CAYLEY
        self.solver.config.joint_constraint_primal_iterations = 2
        self.solver.config.nonlinear_iterations = 20
        self.solver.config.linear_iterations = 5
        self.solver.config.enable_collision_hessian_non_diagonal = True

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        # Foot edges and the interiors of cloth triangles must collide too; vertex-only
        # contact can let a foot pass through a coarse trampoline cell.
        self.collision_pipeline = newton.CollisionPipeline(self.model, enable_rigid_soft_full_surface_contact=True)
        self.contacts = self.collision_pipeline.contacts()

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self._fixed_rim_positions = self.state_0.particle_q.numpy()[self.fixed_rim].copy()

        self.viewer.set_model(self.model)
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
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        bodies, body_velocity = self.state_0.body_q.numpy(), self.state_0.body_qd.numpy()
        cloth = self.state_0.particle_q.numpy()[: self.trampoline_particle_count]
        cloth_velocity = self.state_0.particle_qd.numpy()[: self.trampoline_particle_count]
        if not all(np.isfinite(value).all() for value in (bodies, body_velocity, cloth, cloth_velocity)):
            raise ValueError("Trampoline simulation contains non-finite state")
        if np.linalg.norm(body_velocity[:, :3], axis=1).max() > 30:
            raise ValueError("Trampoline robot speed exceeds 30 m/s")
        if np.linalg.norm(body_velocity[:, 3:], axis=1).max() > 300:
            raise ValueError("Trampoline robot angular speed exceeds 300 rad/s")
        if np.linalg.norm(cloth_velocity, axis=1).max() > 100:
            raise ValueError("Trampoline cloth speed exceeds 100 m/s")
        if not np.allclose(cloth[self.fixed_rim], self._fixed_rim_positions, atol=1e-6, rtol=0):
            raise ValueError("The fixed trampoline rim moved")

    def test_final(self):
        self.test_post_step()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--trampoline-mesh",
            type=Path,
            default=None,
            help="Optional OBJ mesh; otherwise use the historical OBJ if present or generate a circular mesh.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    try:
        newton.examples.run(Example(viewer, args), args)
    except BaseException:
        viewer.close()
        raise

# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A freely moving H1 in a jacket, coupled through the SynReal solver.

Run: uv run python -m newton.examples cha_cloth_h1_free
Joint springs weakly restore the initial posture; the floating base has no pose controller.
"""

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.usd
import newton.utils
from newton.solvers import coupled_high_accuracy as cha
from newton.solvers import style3d


class Example:
    def __init__(self, viewer, args):
        for name in ("joint_stiffness", "joint_damping", "joint_effort_limit"):
            value = getattr(args, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.graph = None

        h1 = newton.ModelBuilder()
        h1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=10.0, friction=1.0e-5)
        h1.add_mjcf(
            newton.utils.download_asset("unitree_h1") / "mjcf/h1_with_hand.xml",
            floating=True,
            enable_self_collisions=False,
        )
        # Ground contact must support the robot's weight; cloth still uses the softer soft-contact material.
        for shape in range(h1.shape_count):
            h1.shape_material_ke[shape] = 1.0e5
            h1.shape_material_kd[shape] = 1.0e3
            h1.shape_material_mu[shape] = 0.6
        h1.default_shape_cfg.ke = 1.0e5
        h1.default_shape_cfg.kd = 1.0e3
        h1.default_shape_cfg.mu = 0.6
        h1.add_ground_plane()

        # Style3D mesh utilities only author the anisotropic cloth attributes consumed by SynReal.
        garment_usd_name = "h1_jacket"
        # garment_usd_name = "h1_cake_skirt"
        cloth_builder = newton.ModelBuilder()
        newton.solvers.SolverStyle3D.register_custom_attributes(cloth_builder)
        asset_path = newton.utils.download_asset("style3d")
        usd_stage = Usd.Stage.Open(f"{asset_path}/garments/{garment_usd_name}.usd")
        usd_prim_garment = usd_stage.GetPrimAtPath(f"/Root/{garment_usd_name}/Root_Garment")

        garment_mesh = newton.usd.get_mesh(usd_prim_garment, load_uvs=False)
        self.garment_mesh_indices = garment_mesh.indices
        self.garment_mesh_points = garment_mesh.vertices[:, [2, 0, 1]]  # y-up to z-up

        # Load raw UV values and indices directly from the primvar
        # (get_mesh expands indexed UVs, but style3d needs raw values + indices)
        uv_primvar = UsdGeom.PrimvarsAPI(usd_prim_garment).GetPrimvar("st")
        self.garment_mesh_uv = np.array(uv_primvar.Get()) * 1e-3
        self.garment_mesh_uv_indices = np.array(uv_primvar.GetIndices())

        style3d.add_cloth_mesh(
            cloth_builder,
            pos=wp.vec3(0, 0, 0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            panel_verts=self.garment_mesh_uv.tolist(),
            panel_indices=self.garment_mesh_uv_indices.tolist(),
            vertices=self.garment_mesh_points.tolist(),
            indices=self.garment_mesh_indices.tolist(),
            density=0.5,
            scale=1.0,
            particle_radius=3.0e-3,
            tri_aniso_ke=wp.vec3(1.0e2, 1.0e2, 1.0e2) * 10.0,
            edge_aniso_ke=wp.vec3(1.0e-6, 1.0e-6, 1.0e-6) * 40.0,
        )
        # Keep rigid bodies, cloth particles, and SynReal's affine proxies in one Newton world block.
        h1.add_builder(cloth_builder)
        self.cloth_particle_count = len(self.garment_mesh_points)
        self.model = h1.finalize()
        self.model.soft_contact_ke = 5.0e3
        self.model.soft_contact_mu = 0.3
        self._set_posture_targets(args)

        # Seed affine proxies from the authored pose, then allocate states after SynReal appends the proxies.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.solver = cha.SolverCoupledHighAccuracy(self.model)
        self.solver.config.affine_body_rigidity_model = cha.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
        self.solver.config.joint_constraint_mode = cha.JointConstraintMode.AUGMENTED_LAGRANGIAN
        self.solver.config.inertia_mode = cha.InertiaMode.MIDPOINT_CAYLEY
        self.solver.config.nonlinear_solver = cha.NonlinearSolver.NEWTON
        self.solver.config.joint_constraint_primal_iterations = 4
        self.solver.config.nonlinear_iterations = 10
        self.solver.config.linear_iterations = 20
        self.solver.config.enable_self_collision = True
        self.solver.config.self_collision_radius = 3.5e-3

        self.state = self.model.state()
        self.state1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(3.0, 0.0, 1.4), 0.0, 180.0)

    def _set_posture_targets(self, args):
        model = self.model
        joint_types = model.joint_type.numpy()
        q_starts = model.joint_q_start.numpy()
        dof_starts = model.joint_qd_start.numpy()
        target_starts = model.joint_target_q_start.numpy()
        initial_q = model.joint_q.numpy()
        targets = model.joint_target_q.numpy()
        stiffness = np.zeros(model.joint_dof_count, dtype=np.float32)
        damping = np.zeros_like(stiffness)
        modes = np.full(model.joint_dof_count, int(newton.JointTargetMode.NONE), dtype=np.int32)
        effort_limits = model.joint_effort_limit.numpy()
        self.posture_dofs = []
        for joint, kind in enumerate(joint_types):
            if kind == int(newton.JointType.REVOLUTE):
                dof = int(dof_starts[joint])
                targets[target_starts[joint]] = initial_q[q_starts[joint]]
                stiffness[dof] = args.joint_stiffness
                damping[dof] = args.joint_damping
                modes[dof] = int(newton.JointTargetMode.POSITION)
                effort_limits[dof] = min(effort_limits[dof], args.joint_effort_limit)
                self.posture_dofs.append(dof)
        # FREE coordinates retain zero gains and NONE mode, so the pelvis can translate and rotate freely.
        model.joint_target_q.assign(targets)
        model.joint_target_qd.zero_()
        model.joint_target_ke.assign(stiffness)
        model.joint_target_kd.assign(damping)
        model.joint_target_mode.assign(modes)
        model.joint_effort_limit.assign(effort_limits)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state.clear_forces()
            self.viewer.apply_forces(self.state)
            self.collision_pipeline.collide(self.state, self.contacts)
            self.solver.step(self.state, self.state1, self.control, self.contacts, self.sim_dt)
            self.state, self.state1 = self.state1, self.state

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            # Eight substeps restore the state-buffer order after capture and each replay.
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph
        self.sim_time += self.frame_dt

    def test_final(self):
        cloth = self.state.particle_q.numpy()[: self.cloth_particle_count]
        for values in (cloth, self.state.particle_qd.numpy(), self.state.body_q.numpy(), self.state.body_qd.numpy()):
            if not np.isfinite(values).all():
                raise AssertionError("H1 or jacket state contains nonfinite values")
        # A free robot may fall over, so the jacket is checked against the floor rather than a standing-pose box.
        if cloth[:, 2].min() < -0.03:
            raise AssertionError(f"Jacket penetrated the ground: minimum z={cloth[:, 2].min():.6f}")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=601)
        parser.add_argument(
            "--joint-stiffness", type=float, default=20.0, help="Initial-angle restoring stiffness (Nm/rad)."
        )
        parser.add_argument("--joint-damping", type=float, default=2.0, help="Joint restoring damping (Nm s/rad).")
        parser.add_argument(
            "--joint-effort-limit", type=float, default=20.0, help="Maximum restoring torque per joint (Nm)."
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

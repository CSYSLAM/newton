# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Independent MuJoCo-MPM: a gravity-compensated press compresses an elastic block.
# The press holds still before a commanded compression, load hold, and retraction.
# MuJoCo owns the articulation; the private MPM environment owns the material.
# Command: python -m newton.examples mjmpm_press
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMJMPM


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.time = 0.0
        self.phase = "Hold above material"
        self.target = 0.0
        self.frame_dt = 1.0 / 60.0
        self.substeps = 10
        builder = newton.ModelBuilder()
        SolverMJMPM.register_custom_attributes(builder)
        pose = wp.transform(wp.vec3(0.0, 0.0, 0.55), wp.quat_identity())
        press = builder.add_link(
            xform=pose,
            mass=2.0,
            inertia=wp.mat33(np.eye(3) * 0.02),
            custom_attributes={"mujoco:gravcomp": 1.0},
            label="Gravity-compensated press",
        )
        builder.add_shape_box(
            press, hx=0.16, hy=0.16, hz=0.025, cfg=newton.ModelBuilder.ShapeConfig(density=0.0), color=(0.2, 0.45, 0.8)
        )
        joint = builder.add_joint_prismatic(
            parent=-1,
            child=press,
            parent_xform=pose,
            axis=(0.0, 0.0, 1.0),
            target_ke=150.0,
            target_kd=20.0,
            damping=2.0,
            effort_limit=80.0,
            armature=0.1,
        )
        builder.add_articulation([joint])
        builder.add_ground_plane()
        builder.add_particle_grid(
            pos=wp.vec3(-0.125, -0.125, 0.035),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=6,
            dim_y=6,
            dim_z=4,
            cell_x=0.05,
            cell_y=0.05,
            cell_z=0.05,
            mass=0.05,
            jitter=0.0,
            radius_mean=0.025,
            custom_attributes={
                "mpm:young_modulus": 1.0e4,
                "mpm:poisson_ratio": 0.2,
                "mpm:yield_stress": 1.0e6,
                "mpm:tensile_yield_ratio": 1.0,
            },
        )
        self.model = builder.finalize()
        mode = SolverMJMPM.Coupling(args.coupling)
        self.solver = SolverMJMPM(
            self.model,
            coupling=mode,
            relaxation=0.5,
            mpm_options={
                "grid_type": "fixed",
                "voxel_size": 0.075,
                "grid_padding": 6,
                "max_active_cell_count": 256,
                "collider_basis": "pic27",
                "max_iterations": 20,
                "solver": "jacobi",
            },
        )
        self.state = self.model.state()
        self.initial_material_height = float(np.ptp(self.state.particle_q.numpy()[:, 2]))
        self.particle_radii = self.model.particle_radius.numpy()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.8, -0.95, 0.65), pitch=-20.0, yaw=130.0)
        self.viewer.show_particles = True
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.render_ui, position="side")

    @staticmethod
    def _press_target(time):
        if time < 1.0:
            return "Hold above material", 0.0
        if time < 4.0:
            progress = (time - 1.0) / 3.0
            return "Commanded compression", -0.48 * progress**2 * (3.0 - 2.0 * progress)
        if time < 5.0:
            return "Hold under load", -0.48
        if time < 8.0:
            progress = (time - 5.0) / 3.0
            return "Commanded retraction", -0.48 * (1.0 - progress**2 * (3.0 - 2.0 * progress))
        return "Finished", 0.0

    def step(self):
        dt = self.frame_dt / self.substeps
        for _ in range(self.substeps):
            self.phase, self.target = self._press_target(self.time)
            self.control.joint_target_q.fill_(self.target)
            self.state.clear_forces()
            self.viewer.apply_forces(self.state)
            self.solver.step(self.state, self.state, self.control, None, dt)
            self.time += dt

    def render(self):
        self.viewer.begin_frame(self.time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def render_ui(self, imgui):
        imgui.text(self.phase)
        imgui.text("Press weight is compensated; gravity remains on the material.")
        imgui.text(f"Commanded travel: {self.target:.3f} m")
        imgui.text(f"Measured travel: {float(self.state.joint_q.numpy()[0]):.3f} m")
        imgui.text(f"Surface gap: {self.surface_gap():.4f} m")

    def surface_gap(self):
        """Vertical clearance between the plate bottom and the particle surfaces."""
        plate_bottom = float(self.state.body_q.numpy()[0, 2]) - 0.025
        material_top = float(np.max(self.state.particle_q.numpy()[:, 2] + self.particle_radii))
        return plate_bottom - material_top

    def test_post_step(self):
        if self.surface_gap() > 0.002:
            assert float(np.linalg.norm(self.solver.environment.material_impulses.numpy()[0])) < 1.0e-8, (
                "Press received a material impulse across an empty gap"
            )
        if self.time <= 1.0 + 1.0e-9:
            height = float(np.ptp(self.state.particle_q.numpy()[:, 2]))
            assert height > 0.85 * self.initial_material_height, "Material collapsed before press contact"
            assert abs(float(self.state.joint_q.numpy()[0])) < 1.0e-4
            assert float(np.linalg.norm(self.solver.environment.material_impulses.numpy()[0])) < 1.0e-8

    def test_final(self):
        assert np.isfinite(self.state.body_q.numpy()).all()
        assert np.isfinite(self.state.particle_q.numpy()).all()
        assert self.solver.features.mujoco_body_count == 1
        assert self.solver.features.mpm_body_count == 0
        assert float(self.state.particle_q.numpy()[:, 2].min()) > -0.075


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--coupling", choices=("one_way", "two_way"), default="two_way")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

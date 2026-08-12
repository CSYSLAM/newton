# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One-way kinematic robot grasp of an inflatable closed bag.

Run with ``python -m newton.examples vbd_inflatable_robot_grasp``.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd.example_vbd_inflatable_bag import _pillow_surface

PARAMS = {
    "fps": 60,
    "sim_substeps": 4,
    "solver_iterations": 20,
    "gravity": (0.0, 0.0, -9.81),
    "bag_width": 0.34,
    "bag_depth": 0.24,
    "bag_height": 0.020,
    "bag_bulge": 0.070,
    "bag_resolution": 12,
    "bag_center_z": 0.115,
    "bag_density": 0.12,
    "particle_radius": 0.004,
    "finger_half_x": 0.11,
    "finger_half_y": 0.012,
    "finger_half_z": 0.10,
    "open_half_gap": 0.16,
    "closed_half_gap": 0.060,
    "grab_z": 0.115,
    "lift_height": 0.20,
    "close_duration": 0.8,
    "lift_duration": 1.2,
}


def _smoothstep(value: float) -> float:
    """Interpolate a clamped scalar with zero endpoint velocity."""
    value = min(max(value, 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)


class Example:
    """Drive a kinematic parallel gripper without feeding bag forces back."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / PARAMS["fps"]
        self.sim_dt = self.frame_dt / PARAMS["sim_substeps"]
        self.sim_time = 0.0

        builder = newton.ModelBuilder(gravity=PARAMS["gravity"])
        vertices, indices = _pillow_surface(
            PARAMS["bag_width"],
            PARAMS["bag_depth"],
            PARAMS["bag_height"],
            PARAMS["bag_bulge"],
            PARAMS["bag_resolution"],
        )
        self.cavity = newton.solvers.add_inflatable_mesh(
            builder,
            pos=wp.vec3(0.0, 0.0, PARAMS["bag_center_z"]),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=vertices,
            indices=indices,
            density=PARAMS["bag_density"],
            tri_ke=6.0e4,
            tri_ka=6.0e4,
            tri_kd=80.0,
            edge_ke=20.0,
            edge_kd=0.5,
            particle_radius=PARAMS["particle_radius"],
            config=newton.solvers.PneumaticConfig(
                mode=newton.solvers.PneumaticMode.ISOTHERMAL,
                reference_absolute_pressure=101_400.0,
                ambient_pressure=101_325.0,
                bulk_damping=50.0,
                max_absolute_pressure=200_000.0,
            ),
        )

        finger_cfg = newton.ModelBuilder.ShapeConfig(density=1.0, ke=5.0e4, kd=150.0, mu=1.0)
        self.left_finger = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, -PARAMS["open_half_gap"], PARAMS["grab_z"]), wp.quat_identity()),
            is_kinematic=True,
            label="robot_left_finger",
        )
        self.right_finger = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, PARAMS["open_half_gap"], PARAMS["grab_z"]), wp.quat_identity()),
            is_kinematic=True,
            label="robot_right_finger",
        )
        for finger, color in ((self.left_finger, (0.85, 0.25, 0.2)), (self.right_finger, (0.2, 0.35, 0.85))):
            builder.add_shape_box(
                finger,
                hx=PARAMS["finger_half_x"],
                hy=PARAMS["finger_half_y"],
                hz=PARAMS["finger_half_z"],
                cfg=finger_cfg,
                color=color,
            )
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.025), wp.quat_identity()),
            hx=0.7,
            hy=0.7,
            hz=0.025,
            cfg=finger_cfg,
            color=(0.35, 0.35, 0.35),
        )

        builder.color()
        self.model = builder.finalize()
        self.model.soft_contact_ke = 5.0e4
        self.model.soft_contact_kd = 150.0
        self.model.soft_contact_mu = 1.0
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=PARAMS["solver_iterations"],
            integrate_with_external_rigid_solver=True,
            rigid_body_particle_contact_buffer_size=2048,
        )
        self.pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=0.01,
            enable_rigid_soft_full_surface_contact=True,
        )
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.initial_mean_bag_height = float(self.state_0.particle_q.numpy()[:, 2].mean())

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = False
            self.viewer.renderer.draw_edges = True
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.0, -0.52, 0.36), -28.0, 90.0)

    def _gripper_pose(self, time: float) -> tuple[float, float]:
        """Return half jaw spacing and lift height for the robot trajectory."""
        close = _smoothstep(time / PARAMS["close_duration"])
        lift = _smoothstep((time - PARAMS["close_duration"]) / PARAMS["lift_duration"])
        gap = PARAMS["open_half_gap"] + (PARAMS["closed_half_gap"] - PARAMS["open_half_gap"]) * close
        z = PARAMS["grab_z"] + PARAMS["lift_height"] * lift
        return gap, z

    def _advance_robot_kinematics(self, next_time: float) -> None:
        """Write externally prescribed robot state without applying bag reactions."""
        current_q = self.state_0.body_q.numpy()
        next_q = current_q.copy()
        gap, z = self._gripper_pose(next_time)
        next_q[self.left_finger, 1] = -gap
        next_q[self.right_finger, 1] = gap
        next_q[self.left_finger, 2] = z
        next_q[self.right_finger, 2] = z
        self.state_1.body_q.assign(next_q)

        body_qd = self.state_1.body_qd.numpy()
        body_qd[:] = 0.0
        body_qd[:, :3] = (next_q[:, :3] - current_q[:, :3]) / self.sim_dt
        self.state_1.body_qd.assign(body_qd)

    def step(self):
        """Advance one-way robot kinematics followed by VBD bag deformation."""
        for _ in range(PARAMS["sim_substeps"]):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            if hasattr(self.viewer, "apply_forces"):
                self.viewer.apply_forces(self.state_0)
            self._advance_robot_kinematics(self.sim_time + self.sim_dt)
            wp.copy(self.state_1.particle_q, self.state_0.particle_q)
            self.pipeline.collide(self.state_1, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def render(self):
        """Render the externally driven robot and deformable bag."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify the one-way robot lifts a finite, non-collapsed bag."""
        volume = self.state_0.pneumatic.volume.numpy()[self.cavity.cavity_index]
        positions = self.state_0.particle_q.numpy()
        assert np.isfinite(positions).all()
        assert volume > self.cavity.rest_volume * 0.2
        assert positions[:, 2].mean() > self.initial_mean_bag_height + PARAMS["lift_height"] * 0.5


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Affine Joints
#
# Displays free, fixed, ball, hinge, cylindrical, and slider joints in one
# collision-free scene. NewtonSolver assembles each joint from affine-body
# point-point and point-line constraints.
#
# Command: uv run python examples/joints/example_joints.py
#
###########################################################################

import warp as wp

import newton
import newton.examples
from newton.solvers import coupled_high_accuracy as s3d

_X_AXIS = wp.vec3(1.0, 0.0, 0.0)
_Z_AXIS = wp.vec3(0.0, 0.0, 1.0)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 20
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        builder = s3d.ModelBuilder(
            up_axis=newton.Axis.Z,
            gravity=0.0,
            affine_body_particle_radius=0.0,
        )
        shape_cfg = builder.default_shape_cfg.copy()
        shape_cfg.density = 200.0
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False

        joint_specs = (
            ("fixed", wp.vec3(-3.0, 0.0, 2.7), wp.vec3(0.82, 0.28, 0.18)),
            ("ball", wp.vec3(0.0, 0.0, 2.7), wp.vec3(0.20, 0.62, 0.34)),
            ("hinge", wp.vec3(3.0, 0.0, 2.7), wp.vec3(0.18, 0.42, 0.82)),
            ("slider", wp.vec3(-3.0, 0.0, 0.9), wp.vec3(0.86, 0.62, 0.12)),
            ("cylindrical", wp.vec3(0.0, 0.0, 0.9), wp.vec3(0.66, 0.30, 0.72)),
            ("free", wp.vec3(3.0, 0.0, 0.9), wp.vec3(0.12, 0.66, 0.70)),
        )
        velocities = {
            "fixed": wp.spatial_vector(),
            "ball": wp.spatial_vector(0.0, 0.0, 0.0, 0.5, 0.7, 0.4),
            "hinge": wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            "slider": wp.spatial_vector(0.15, 0.0, 0.0, 0.0, 0.0, 0.0),
            "cylindrical": wp.spatial_vector(0.12, 0.0, 0.0, 1.0, 0.0, 0.0),
            "free": wp.spatial_vector(0.08, 0.03, 0.0, 0.2, 0.5, 0.3),
        }

        self.bodies = {}
        guide_starts = []
        guide_ends = []
        for joint_kind, position, color in joint_specs:
            body = builder.add_link(
                xform=wp.transform(position, wp.quat_identity()),
                label=f"{joint_kind}_joint_body",
            )
            builder.add_shape_box(
                body,
                hx=0.45,
                hy=0.12,
                hz=0.08,
                cfg=shape_cfg,
                color=color,
            )
            self.bodies[joint_kind] = body

            world_anchor = wp.transform(position, wp.quat_identity())
            identity = wp.transform_identity()
            if joint_kind == "free":
                joint = builder.add_joint_free(child=body)
            elif joint_kind == "fixed":
                joint = builder.add_joint_fixed(
                    parent=-1,
                    child=body,
                    parent_xform=world_anchor,
                    child_xform=identity,
                )
            elif joint_kind == "ball":
                joint = builder.add_joint_ball(
                    parent=-1,
                    child=body,
                    parent_xform=world_anchor,
                    child_xform=identity,
                )
            elif joint_kind == "hinge":
                joint = builder.add_joint_revolute(
                    parent=-1,
                    child=body,
                    axis=_Z_AXIS,
                    parent_xform=world_anchor,
                    child_xform=identity,
                )
                guide_starts.append(position - 0.65 * _Z_AXIS)
                guide_ends.append(position + 0.65 * _Z_AXIS)
            elif joint_kind == "slider":
                joint = builder.add_joint_prismatic(
                    parent=-1,
                    child=body,
                    axis=_X_AXIS,
                    parent_xform=world_anchor,
                    child_xform=identity,
                )
                guide_starts.append(position - 1.2 * _X_AXIS)
                guide_ends.append(position + 1.2 * _X_AXIS)
            else:
                axis = builder.JointDofConfig.create_unlimited(newton.Axis.X)
                joint = builder.add_joint_d6(
                    parent=-1,
                    child=body,
                    linear_axes=[axis],
                    angular_axes=[axis],
                    parent_xform=world_anchor,
                    child_xform=identity,
                )
                guide_starts.append(position - 1.2 * _X_AXIS)
                guide_ends.append(position + 1.2 * _X_AXIS)

            builder.add_articulation([joint])
            builder.body_qd[body] = velocities[joint_kind]

        self.model = builder.finalize()
        self.solver = s3d.SolverCoupledHighAccuracy(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.stream = wp.get_stream(self.model.device)
        self.guide_starts = wp.array(guide_starts, dtype=wp.vec3, device=self.model.device)
        self.guide_ends = wp.array(guide_ends, dtype=wp.vec3, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -9.0, 2.0), pitch=-2.0, yaw=90.0)
        self.capture()

    def capture(self):
        if self.model.device.is_cuda:
            with wp.ScopedCapture(stream=self.stream) as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph, stream=self.stream)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines(
            "/joints/axes",
            self.guide_starts,
            self.guide_ends,
            (0.95, 0.74, 0.12),
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

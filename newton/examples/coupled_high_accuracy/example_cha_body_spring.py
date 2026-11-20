# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Body Spring
#
# Demonstrates ten free rigid boxes connected end-to-end by point springs.
# Each zero-rest-length spring acts as a stiff ball joint between neighboring
# box endpoints while the native solver advances the affine-body proxies.
#
# Command: uv run python examples/python/example_body_spring.py
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import coupled_high_accuracy as cha

_BODY_COUNT = 10
_BOX_HALF_LENGTH = 0.5


@wp.kernel
def update_spring_lines(
    body_q: wp.array[wp.transform],
    body_pairs: wp.array[wp.vec2i],
    point0: wp.vec3,
    point1: wp.vec3,
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    """Update rendered lines from all pairs of body-local attachment points."""
    spring = wp.tid()
    bodies = body_pairs[spring]
    starts[spring] = wp.transform_point(body_q[bodies[0]], point0)
    ends[spring] = wp.transform_point(body_q[bodies[1]], point1)


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        builder = cha.ModelBuilder(
            up_axis=newton.Axis.Z,
            gravity=-0.98,
            affine_body_particle_radius=0.04,
        )
        shape_cfg = builder.default_shape_cfg.copy()
        shape_cfg.density = 1.0e2
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False

        self.bodies = []
        spacing = 2.0 * _BOX_HALF_LENGTH
        for index in range(_BODY_COUNT):
            body = builder.add_body(
                xform=wp.transform(
                    wp.vec3(index * spacing, 0.0, 1.5),
                    wp.quat_identity(),
                ),
                label=f"spring_body_{index}",
            )
            color = wp.vec3(0.82, 0.24, 0.16) if index % 2 == 0 else wp.vec3(0.12, 0.42, 0.82)
            builder.add_shape_box(body, hx=_BOX_HALF_LENGTH, hy=0.15, hz=0.15, cfg=shape_cfg, color=color)
            self.bodies.append(body)

        self.point0 = wp.vec3(_BOX_HALF_LENGTH, 0.0, 0.0)
        self.point1 = wp.vec3(-_BOX_HALF_LENGTH, 0.0, 0.0)
        self.body_pairs = list(zip(self.bodies[:-1], self.bodies[1:]))
        for body0, body1 in self.body_pairs:
            builder.add_body_spring(body0, body1, self.point0, self.point1, stiffness=1.0e7, rest_length=0.0)

        self.model = builder.finalize()
        self.solver = cha.SolverCoupledHighAccuracy(self.model)
        self.solver.config.nonlinear_iterations = 10
        self.solver.config.linear_iterations = 2

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.stream = wp.get_stream(self.model.device)

        self.body_pairs_device = wp.array(self.body_pairs, dtype=wp.vec2i, device=self.model.device)
        self.spring_starts = wp.empty(len(self.body_pairs), dtype=wp.vec3, device=self.model.device)
        self.spring_ends = wp.empty(len(self.body_pairs), dtype=wp.vec3, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.5 * (_BODY_COUNT - 1) * spacing, -7.0, 2.5), pitch=-10.0, yaw=90.0)
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

    def test_final(self):
        poses = self.state_0.body_q.numpy()
        velocities = self.state_0.body_qd.numpy()
        if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
            raise ValueError("Body spring simulation produced a non-finite rigid-body state")
        self.update_spring_lines()
        joint_gaps = np.linalg.norm(self.spring_ends.numpy() - self.spring_starts.numpy(), axis=1)
        maximum_joint_gap = float(joint_gaps.max())
        if maximum_joint_gap >= 1.0e-3:
            raise ValueError(
                f"A zero-rest-length body spring allowed neighboring box endpoints to separate by {maximum_joint_gap:.6f} m"
            )
        maximum_speed = float(np.linalg.norm(velocities[self.bodies], axis=1).max())
        if maximum_speed >= 1.0e-2:
            raise ValueError(f"Body spring chain developed a spurious speed of {maximum_speed:.6f} m/s")

    def update_spring_lines(self):
        wp.launch(
            update_spring_lines,
            dim=len(self.body_pairs),
            inputs=[self.state_0.body_q, self.body_pairs_device, self.point0, self.point1],
            outputs=[self.spring_starts, self.spring_ends],
            device=self.model.device,
            stream=self.stream,
        )

    def render(self):
        self.update_spring_lines()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines("/body_spring", self.spring_starts, self.spring_ends, (0.95, 0.75, 0.10))
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

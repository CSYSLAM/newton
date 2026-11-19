# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Body Point-Line Constraint
#
# A red slider uses one constrained point, while a green ball uses two points
# along its local x axis. All points rest on a segment attached to the blue guide
# body. Beyond either endpoint, projection clamps and the constraint acts as a
# zero-rest-length point spring to that endpoint.
#
# Command: uv run python examples/python/example_body_point_line.py
#
###########################################################################

import warp as wp

import newton
import newton.examples
from newton.solvers import coupled_high_accuracy as s3d

_LINE_HALF_LENGTH = 1.5
_CONSTRAINT_STIFFNESS = 1.0e5
_CONSTRAINT_DAMPING = 1.0e10


@wp.kernel
def update_constraint_lines(
    body_q: wp.array[wp.transform],
    body_pairs: wp.array[wp.vec2i],
    local_points: wp.array[wp.vec3],
    local_endpoints0: wp.array[wp.vec3],
    local_endpoints1: wp.array[wp.vec3],
    segment_starts: wp.array[wp.vec3],
    segment_ends: wp.array[wp.vec3],
    error_starts: wp.array[wp.vec3],
    error_ends: wp.array[wp.vec3],
):
    constraint = wp.tid()
    bodies = body_pairs[constraint]
    point = wp.transform_point(body_q[bodies[0]], local_points[constraint])
    endpoint0 = wp.transform_point(body_q[bodies[1]], local_endpoints0[constraint])
    endpoint1 = wp.transform_point(body_q[bodies[1]], local_endpoints1[constraint])
    segment = endpoint1 - endpoint0
    segment_length_squared = wp.dot(segment, segment)
    projection = float(0.0)
    if segment_length_squared > 1.0e-12:
        projection = wp.clamp(
            wp.dot(point - endpoint0, segment) / segment_length_squared,
            0.0,
            1.0,
        )
    closest_point = endpoint0 + projection * segment

    segment_starts[constraint] = endpoint0
    segment_ends[constraint] = endpoint1
    error_starts[constraint] = point
    error_ends[constraint] = closest_point


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        builder = s3d.ModelBuilder(
            up_axis=newton.Axis.Z,
            gravity=0.0,
            affine_body_particle_radius=0.0,
        )

        guide_cfg = builder.default_shape_cfg.copy()
        guide_cfg.density = 2.0e3
        guide_cfg.has_shape_collision = False
        guide_cfg.has_particle_collision = False
        self.guide_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            label="point_line_guide",
        )
        builder.add_shape_box(
            self.guide_body,
            hx=_LINE_HALF_LENGTH,
            hy=0.06,
            hz=0.06,
            cfg=guide_cfg,
            color=wp.vec3(0.12, 0.38, 0.78),
        )

        slider_cfg = builder.default_shape_cfg.copy()
        slider_cfg.density = 2.0e1
        slider_cfg.has_shape_collision = False
        slider_cfg.has_particle_collision = False
        self.slider_body = builder.add_body(
            xform=wp.transform(wp.vec3(-0.8, 0.0, 1.0), wp.quat_identity()),
            label="point_line_slider",
        )
        builder.add_shape_sphere(
            self.slider_body,
            radius=0.16,
            cfg=slider_cfg,
            color=wp.vec3(0.85, 0.20, 0.12),
        )

        two_point_ball_cfg = builder.default_shape_cfg.copy()
        two_point_ball_cfg.density = 2.0e2
        two_point_ball_cfg.has_shape_collision = False
        two_point_ball_cfg.has_particle_collision = False
        self.two_point_ball_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.65, 0.0, 1.0), wp.quat_identity()),
            label="point_line_two_point_ball",
        )
        builder.add_shape_sphere(
            self.two_point_ball_body,
            radius=0.14,
            cfg=two_point_ball_cfg,
            color=wp.vec3(0.16, 0.70, 0.32),
        )

        self.local_endpoint0 = wp.vec3(-_LINE_HALF_LENGTH, 0.0, 0.0)
        self.local_endpoint1 = wp.vec3(_LINE_HALF_LENGTH, 0.0, 0.0)
        self.constraint_body_pairs = [
            (self.slider_body, self.guide_body),
            (self.two_point_ball_body, self.guide_body),
            (self.two_point_ball_body, self.guide_body),
        ]
        self.constraint_local_points = [
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(-0.06, 0.0, 0.0),
            wp.vec3(0.06, 0.0, 0.0),
        ]
        for (point_body, line_body), local_point in zip(self.constraint_body_pairs, self.constraint_local_points):
            builder.add_body_point_line(
                point_body,
                line_body,
                local_point,
                self.local_endpoint0,
                self.local_endpoint1,
                stiffness=_CONSTRAINT_STIFFNESS,
                damping=_CONSTRAINT_DAMPING,
            )

        self.model = builder.finalize()
        self.solver = s3d.SolverCoupledHighAccuracy(self.model)
        self.solver.config.nonlinear_iterations = 10
        self.solver.config.linear_iterations = 5

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.stream = wp.get_stream(self.model.device)

        self.constraint_count = len(self.constraint_body_pairs)
        self.constraint_body_pairs_device = wp.array(
            self.constraint_body_pairs, dtype=wp.vec2i, device=self.model.device
        )
        self.constraint_local_points_device = wp.array(
            self.constraint_local_points, dtype=wp.vec3, device=self.model.device
        )
        self.constraint_local_endpoints0_device = wp.array(
            [self.local_endpoint0] * self.constraint_count,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.constraint_local_endpoints1_device = wp.array(
            [self.local_endpoint1] * self.constraint_count,
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.segment_starts = wp.empty(self.constraint_count, dtype=wp.vec3, device=self.model.device)
        self.segment_ends = wp.empty(self.constraint_count, dtype=wp.vec3, device=self.model.device)
        self.error_starts = wp.empty(self.constraint_count, dtype=wp.vec3, device=self.model.device)
        self.error_ends = wp.empty(self.constraint_count, dtype=wp.vec3, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -5.5, 2.1), pitch=-8.0, yaw=90.0)
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

    def update_constraint_lines(self):
        wp.launch(
            update_constraint_lines,
            dim=self.constraint_count,
            inputs=[
                self.state_0.body_q,
                self.constraint_body_pairs_device,
                self.constraint_local_points_device,
                self.constraint_local_endpoints0_device,
                self.constraint_local_endpoints1_device,
            ],
            outputs=[
                self.segment_starts,
                self.segment_ends,
                self.error_starts,
                self.error_ends,
            ],
            device=self.model.device,
            stream=self.stream,
        )

    def render(self):
        self.update_constraint_lines()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines(
            "/point_line/segment",
            self.segment_starts,
            self.segment_ends,
            (0.95, 0.76, 0.10),
        )
        self.viewer.log_lines(
            "/point_line/error",
            self.error_starts,
            self.error_ends,
            (0.20, 0.85, 0.32),
        )
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Distance Constraint
#
# Compares a stiff Hooke spring with an augmented-Lagrangian hard distance
# constraint. Identical bodies are pulled away from fixed anchors by the same
# load. The spring settles with visible compliance, while the hard constraint
# preserves its rest length.
#
# Command: uv run python examples/python/example_distance_constraint.py
# Headless: uv run python examples/python/example_distance_constraint.py --viewer null --test
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import coupled_high_accuracy as cha

_REST_LENGTH = 1.0
_SPRING_STIFFNESS = 1.0e4
_LOAD = 2.0e3
_DAMPING = 5.0e2


@wp.kernel
def apply_comparison_loads(
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    bodies: wp.array[wp.int32],
    load: float,
    damping: float,
):
    comparison = wp.tid()
    body = bodies[comparison]
    velocity = body_qd[body]
    body_f[body] = body_f[body] + wp.spatial_vector(load - damping * velocity[0], 0.0, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def update_line_endpoint(
    body_q: wp.array[wp.transform],
    body: int,
    endpoint: wp.array[wp.vec3],
):
    endpoint[0] = wp.transform_get_translation(body_q[body])


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
            gravity=0.0,
            affine_body_particle_radius=0.04,
        )
        shape_cfg = builder.default_shape_cfg.copy()
        shape_cfg.density = 1.0e2
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False

        self.spring_anchor = wp.vec3(0.0, -0.65, 1.0)
        self.constraint_anchor = wp.vec3(0.0, 0.65, 1.0)
        self.spring_body = builder.add_body(
            xform=wp.transform(
                self.spring_anchor + wp.vec3(_REST_LENGTH, 0.0, 0.0),
                wp.quat_identity(),
            ),
            label="stiff_spring",
        )
        self.constraint_body = builder.add_body(
            xform=wp.transform(
                self.constraint_anchor + wp.vec3(_REST_LENGTH, 0.0, 0.0),
                wp.quat_identity(),
            ),
            label="al_distance_constraint",
        )
        builder.add_shape_sphere(
            -1,
            xform=wp.transform(self.spring_anchor, wp.quat_identity()),
            radius=0.08,
            cfg=shape_cfg,
            color=wp.vec3(0.95, 0.65, 0.08),
        )
        builder.add_shape_sphere(
            -1,
            xform=wp.transform(self.constraint_anchor, wp.quat_identity()),
            radius=0.08,
            cfg=shape_cfg,
            color=wp.vec3(0.05, 0.72, 0.82),
        )
        builder.add_shape_box(
            self.spring_body,
            hx=0.18,
            hy=0.25,
            hz=0.25,
            cfg=shape_cfg,
            color=wp.vec3(0.95, 0.65, 0.08),
        )
        builder.add_shape_box(
            self.constraint_body,
            hx=0.18,
            hy=0.25,
            hz=0.25,
            cfg=shape_cfg,
            color=wp.vec3(0.05, 0.72, 0.82),
        )
        builder.add_body_spring(
            self.spring_body,
            -1,
            point1=self.spring_anchor,
            stiffness=_SPRING_STIFFNESS,
            rest_length=_REST_LENGTH,
        )
        builder.add_body_distance_constraint(
            self.constraint_body,
            -1,
            point1=self.constraint_anchor,
            rest_length=_REST_LENGTH,
        )

        self.model = builder.finalize()
        self.solver = cha.SolverCoupledHighAccuracy(self.model)
        self.solver.config.nonlinear_iterations = 10
        self.solver.config.linear_iterations = 4

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.stream = wp.get_stream(self.model.device)
        self.loaded_bodies = wp.array(
            (self.spring_body, self.constraint_body),
            dtype=wp.int32,
            device=self.model.device,
        )
        self.spring_line_start = wp.array((self.spring_anchor,), dtype=wp.vec3, device=self.model.device)
        self.constraint_line_start = wp.array((self.constraint_anchor,), dtype=wp.vec3, device=self.model.device)
        self.spring_line_end = wp.empty(1, dtype=wp.vec3, device=self.model.device)
        self.constraint_line_end = wp.empty(1, dtype=wp.vec3, device=self.model.device)
        self.spring_error = 0.0
        self.constraint_error = 0.0
        self.max_spring_error = 0.0
        self.max_constraint_error = 0.0

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.0, -5.0, 2.2), pitch=-8.0, yaw=90.0)
        if hasattr(self.viewer, "renderer"):
            self.viewer.register_ui_callback(self.render_ui, position="side")
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
            wp.launch(
                apply_comparison_loads,
                dim=2,
                inputs=[
                    self.state_0.body_qd,
                    self.state_0.body_f,
                    self.loaded_bodies,
                    _LOAD,
                    _DAMPING,
                ],
                device=self.model.device,
                stream=self.stream,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def update_errors(self):
        positions = self.state_0.body_q.numpy()[:, :3]
        self.spring_error = abs(
            np.linalg.norm(positions[self.spring_body] - np.asarray(self.spring_anchor)) - _REST_LENGTH
        )
        self.constraint_error = abs(
            np.linalg.norm(positions[self.constraint_body] - np.asarray(self.constraint_anchor)) - _REST_LENGTH
        )
        self.max_spring_error = max(self.max_spring_error, self.spring_error)
        self.max_constraint_error = max(self.max_constraint_error, self.constraint_error)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph, stream=self.stream)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.update_errors()

    def update_lines(self):
        wp.launch(
            update_line_endpoint,
            dim=1,
            inputs=[self.state_0.body_q, self.spring_body],
            outputs=[self.spring_line_end],
            device=self.model.device,
            stream=self.stream,
        )
        wp.launch(
            update_line_endpoint,
            dim=1,
            inputs=[self.state_0.body_q, self.constraint_body],
            outputs=[self.constraint_line_end],
            device=self.model.device,
            stream=self.stream,
        )

    def render_ui(self, imgui):
        imgui.text(f"Same outward load: {_LOAD:.0f} N")
        imgui.separator()
        imgui.text(f"Stiff spring error: {self.spring_error:.5f} m")
        imgui.text(f"AL constraint error: {self.constraint_error:.5f} m")
        imgui.text(f"Spring stiffness: {_SPRING_STIFFNESS:.0f} N/m")

    def render(self):
        self.update_lines()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines(
            "/stiff_spring",
            self.spring_line_start,
            self.spring_line_end,
            (0.95, 0.65, 0.08),
        )
        self.viewer.log_lines(
            "/al_distance_constraint",
            self.constraint_line_start,
            self.constraint_line_end,
            (0.05, 0.72, 0.82),
        )
        self.viewer.end_frame()

    def test_final(self):
        self.update_errors()
        ratio = self.max_spring_error / max(self.max_constraint_error, 1.0e-12)
        print(
            "Distance constraint comparison:\n"
            f"  stiff spring maximum error: {self.max_spring_error:.6f} m\n"
            f"  AL constraint maximum error: {self.max_constraint_error:.6f} m\n"
            f"  error ratio: {ratio:.1f}x"
        )
        if self.max_spring_error < 5.0e-2:
            raise ValueError("The stiff spring did not show visible compliance")
        if self.max_constraint_error >= 2.0e-3:
            raise ValueError("The AL distance constraint exceeded its error tolerance")
        if ratio < 50.0:
            raise ValueError("The AL distance constraint was not clearly harder than the spring")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

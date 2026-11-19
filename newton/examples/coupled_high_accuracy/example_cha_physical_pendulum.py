# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Physical Pendulum
#
# Reproduces the setup of M-ABD Fig. 10: a uniform rigid cylinder is fixed
# at one end by a revolute joint and released horizontally from rest.
#
# Command: uv run python -m newton.examples cha_physical_pendulum
# Headless: uv run python -m newton.examples cha_physical_pendulum --viewer null --num-frames 3000
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import coupled_high_accuracy as s3d

_DEFAULT_LENGTH = 3.0
_DEFAULT_MASS = 5000.0
_DEFAULT_RADIUS = 0.06
_REFERENCE_TIME_STEP = 1.0 / 60
_GRAVITY = -9.81


def _validate_parameters(
    length: float,
    mass: float,
    radius: float,
) -> None:
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("Pendulum length must be finite and positive.")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("Pendulum mass must be finite and positive.")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Pendulum radius must be finite and positive.")
    if radius >= 0.5 * length:
        raise ValueError("Pendulum radius must be less than half its length.")


@wp.kernel
def update_pendulum_diagnostics(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    proxy_indices: wp.vec4i,
    pivot_weights: wp.vec4,
    tip_weights: wp.vec4,
    com_weights: wp.vec4,
    mass: float,
    time_step: float,
    gravity: float,
    previous_com_velocity: wp.array[wp.vec3],
    diagnostics: wp.array[wp.vec3],
):
    p0 = particle_q[proxy_indices[0]]
    p1 = particle_q[proxy_indices[1]]
    p2 = particle_q[proxy_indices[2]]
    p3 = particle_q[proxy_indices[3]]

    diagnostics[0] = pivot_weights[0] * p0 + pivot_weights[1] * p1 + pivot_weights[2] * p2 + pivot_weights[3] * p3
    diagnostics[1] = tip_weights[0] * p0 + tip_weights[1] * p1 + tip_weights[2] * p2 + tip_weights[3] * p3

    v0 = particle_qd[proxy_indices[0]]
    v1 = particle_qd[proxy_indices[1]]
    v2 = particle_qd[proxy_indices[2]]
    v3 = particle_qd[proxy_indices[3]]
    com_velocity = com_weights[0] * v0 + com_weights[1] * v1 + com_weights[2] * v2 + com_weights[3] * v3
    com_acceleration = (com_velocity - previous_com_velocity[0]) / time_step
    diagnostics[2] = mass * (com_acceleration - wp.vec3(0.0, 0.0, gravity))
    previous_com_velocity[0] = com_velocity


class Example:
    def __init__(self, viewer, args):
        self.length = float(getattr(args, "length", _DEFAULT_LENGTH))
        self.mass = float(getattr(args, "mass", _DEFAULT_MASS))
        self.radius = float(getattr(args, "radius", _DEFAULT_RADIUS))
        _validate_parameters(self.length, self.mass, self.radius)

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        self.pivot = np.array((0.0, 0.0, 0.55 * self.length), dtype=np.float64)
        initial_direction = np.array((1.0, 0.0, 0.0), dtype=np.float64)
        center = self.pivot + 0.5 * self.length * initial_direction
        body_rotation = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*initial_direction))

        transverse_inertia = self.mass * (3.0 * self.radius**2 + self.length**2) / 12.0
        axial_inertia = 0.5 * self.mass * self.radius**2

        builder = s3d.ModelBuilder(
            up_axis=newton.Axis.Z,
            gravity=_GRAVITY,
            affine_body_particle_radius=0.0,
        )
        self.pendulum = builder.add_link(
            xform=wp.transform(wp.vec3(*center), body_rotation),
            mass=self.mass,
            inertia=wp.mat33(
                transverse_inertia,
                0.0,
                0.0,
                0.0,
                transverse_inertia,
                0.0,
                0.0,
                0.0,
                axial_inertia,
            ),
            lock_inertia=True,
            label="physical_pendulum",
        )

        shape_cfg = builder.default_shape_cfg.copy()
        shape_cfg.has_shape_collision = False
        shape_cfg.has_particle_collision = False
        builder.add_shape_cylinder(
            self.pendulum,
            radius=self.radius,
            half_height=0.5 * self.length,
            cfg=shape_cfg,
            color=wp.vec3(0.78, 0.52, 0.16),
        )

        bob_cfg = builder.default_shape_cfg.copy()
        bob_cfg.has_shape_collision = False
        bob_cfg.has_particle_collision = False
        builder.add_shape_sphere(
            self.pendulum,
            xform=wp.transform(
                wp.vec3(0.0, 0.0, 0.5 * self.length),
                wp.quat_identity(),
            ),
            radius=2.0 * self.radius,
            cfg=bob_cfg,
            color=wp.vec3(0.28, 0.31, 0.36),
        )

        joint = builder.add_joint_revolute(
            parent=-1,
            child=self.pendulum,
            parent_xform=wp.transform(wp.vec3(*self.pivot), body_rotation),
            child_xform=wp.transform(
                wp.vec3(0.0, 0.0, -0.5 * self.length),
                wp.quat_identity(),
            ),
            axis=wp.vec3(0.0, 1.0, 0.0),
            label="fixed_pendulum_pivot",
        )
        builder.add_articulation([joint], label="physical_pendulum")

        self.model = builder.finalize()
        self.solver = s3d.SolverCoupledHighAccuracy(self.model)
        self.solver.config.affine_body_rigidity_model = s3d.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
        self.solver.config.joint_constraint_mode = s3d.JointConstraintMode.AUGMENTED_LAGRANGIAN
        self.solver.config.nonlinear_solver = s3d.NonlinearSolver.NEWTON
        self.solver.config.inertia_mode = s3d.InertiaMode.MIDPOINT_CAYLEY
        self.solver.config.joint_constraint_primal_iterations = 2
        self.solver.config.nonlinear_iterations = 20
        self.solver.config.linear_iterations = 10

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.stream = wp.get_stream(self.model.device)

        affine_body_ids = self.model.synreal_affine_body_ids.numpy()
        affine_body = int(np.flatnonzero(affine_body_ids == self.pendulum).item())
        indices = self.model.synreal_affine_body_indices.numpy()[affine_body]
        self.proxy_indices = wp.vec4i(*(int(value) for value in indices))
        body_com = self.model.synreal_affine_body_com.numpy()[affine_body]

        def point_weights(local_point: np.ndarray) -> wp.vec4:
            coordinate = local_point - body_com
            weights = np.concatenate(((1.0,), coordinate))
            return wp.vec4(*(float(value) for value in weights))

        self.pivot_weights = point_weights(np.array((0.0, 0.0, -0.5 * self.length), dtype=np.float64))
        self.tip_weights = point_weights(np.array((0.0, 0.0, 0.5 * self.length), dtype=np.float64))
        self.com_weights = point_weights(np.asarray(body_com, dtype=np.float64))

        self.previous_com_velocity = wp.zeros(1, dtype=wp.vec3, device=self.model.device)
        initial_tip = self.pivot + self.length * initial_direction
        self.diagnostics = wp.array(
            (
                wp.vec3(*self.pivot),
                wp.vec3(*initial_tip),
                wp.vec3(0.0, 0.0, 0.0),
            ),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.pivot_position = self.diagnostics[0:1]
        self.tip_position = self.diagnostics[1:2]

        self.reference_phi = 0.5 * math.pi
        self.reference_angular_velocity = 0.0
        pivot_inertia = transverse_inertia + self.mass * (0.5 * self.length) ** 2
        self.reference_frequency_squared = self.mass * abs(_GRAVITY) * 0.5 * self.length / pivot_inertia
        self.maximum_pivot_drift = 0.0
        self.maximum_length_ratio = 1.0

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.5 * self.length, -2.0 * self.length, 0.15 * self.length),
            pitch=-2.0,
            yaw=90.0,
        )
        self.capture()

    def capture(self):
        self.graph = None
        self.use_graph = False
        if self.model.device.is_cuda:
            self.use_graph = True
            with wp.ScopedCapture(stream=self.stream) as capture:
                self.simulate()
            self.graph = capture.graph

    def _update_device_diagnostics(self):
        wp.launch(
            update_pendulum_diagnostics,
            dim=1,
            inputs=[
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self.proxy_indices,
                self.pivot_weights,
                self.tip_weights,
                self.com_weights,
                self.mass,
                self.sim_dt,
                _GRAVITY,
            ],
            outputs=[
                self.previous_com_velocity,
                self.diagnostics,
            ],
            device=self.model.device,
            stream=self.stream,
        )

    def simulate(self):
        copy_final_state = self.use_graph and self.sim_substeps % 2 == 1

        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            if copy_final_state and substep == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0
            self._update_device_diagnostics()

    def _advance_reference(self):

        steps = max(1, int(math.ceil(self.frame_dt / _REFERENCE_TIME_STEP)))
        dt = self.frame_dt / steps

        def derivative(phi: float, angular_velocity: float) -> tuple[float, float]:
            return (
                angular_velocity,
                -self.reference_frequency_squared * math.sin(phi),
            )

        phi = self.reference_phi
        angular_velocity = self.reference_angular_velocity
        for _ in range(steps):
            k1_phi, k1_omega = derivative(phi, angular_velocity)
            k2_phi, k2_omega = derivative(
                phi + 0.5 * dt * k1_phi,
                angular_velocity + 0.5 * dt * k1_omega,
            )
            k3_phi, k3_omega = derivative(
                phi + 0.5 * dt * k2_phi,
                angular_velocity + 0.5 * dt * k2_omega,
            )
            k4_phi, k4_omega = derivative(
                phi + dt * k3_phi,
                angular_velocity + dt * k3_omega,
            )
            phi += dt * (k1_phi + 2.0 * k2_phi + 2.0 * k3_phi + k4_phi) / 6.0
            angular_velocity += dt * (k1_omega + 2.0 * k2_omega + 2.0 * k3_omega + k4_omega) / 6.0

        self.reference_phi = phi
        self.reference_angular_velocity = angular_velocity

    def _reference_joint_force(self) -> float:
        phi = self.reference_phi
        angular_velocity = self.reference_angular_velocity
        radius = 0.5 * self.length
        radial = np.array((math.sin(phi), 0.0, -math.cos(phi)))
        tangent = np.array((math.cos(phi), 0.0, math.sin(phi)))
        angular_acceleration = -self.reference_frequency_squared * math.sin(phi)
        acceleration = -radius * angular_velocity**2 * radial + radius * angular_acceleration * tangent
        reaction = self.mass * (acceleration - np.array((0.0, 0.0, _GRAVITY)))
        return float(np.linalg.norm(reaction))

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph, stream=self.stream)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._advance_reference()

        diagnostics = self.diagnostics.numpy()
        pivot = diagnostics[0]
        tip = diagnostics[1]
        direction = tip - pivot
        current_length = float(np.linalg.norm(direction))
        length_ratio = current_length / self.length
        rotation_angle = math.acos(float(np.clip(direction[0] / current_length, -1.0, 1.0)))
        pivot_drift = float(np.linalg.norm(pivot - self.pivot))
        joint_force = float(np.linalg.norm(diagnostics[2]))
        reference_angle = 0.5 * math.pi - self.reference_phi
        reference_force = self._reference_joint_force()

        self.maximum_pivot_drift = max(self.maximum_pivot_drift, pivot_drift)
        self.maximum_length_ratio = max(self.maximum_length_ratio, length_ratio)
        print(
            f"time={self.sim_time:7.3f} s  "
            f"theta={rotation_angle:9.6f} rad  "
            f"theta_ref={reference_angle:9.6f} rad  "
            f"F_joint={joint_force / 1000.0:9.6f} kN  "
            f"F_ref={reference_force / 1000.0:9.6f} kN  "
            f"pivot_drift={pivot_drift:10.6e} m  "
            f"L/L0={length_ratio:9.6f}"
        )

    def test_final(self):
        poses = self.state_0.body_q.numpy()
        velocities = self.state_0.body_qd.numpy()
        particles = self.state_0.particle_q.numpy()
        if not np.isfinite(poses).all() or not np.isfinite(velocities).all() or not np.isfinite(particles).all():
            raise ValueError("Physical-pendulum simulation produced a non-finite state")
        if self.maximum_pivot_drift >= 0.02 * self.length:
            raise ValueError("Physical-pendulum pivot drift exceeded 2% of rod length")
        if self.maximum_length_ratio >= 1.10:
            raise ValueError("Physical-pendulum rod stretched beyond 1.10 times its length")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines(
            "/physical_pendulum/axis",
            self.pivot_position,
            self.tip_position,
            (0.95, 0.72, 0.12),
        )
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=3000)
        parser.add_argument(
            "--length",
            type=float,
            default=_DEFAULT_LENGTH,
            metavar="L",
            help="Physical-pendulum rod length in meters.",
        )
        parser.add_argument(
            "--mass",
            type=float,
            default=_DEFAULT_MASS,
            metavar="M",
            help="Physical-pendulum rod mass in kilograms.",
        )
        parser.add_argument(
            "--radius",
            type=float,
            default=_DEFAULT_RADIUS,
            metavar="R",
            help="Physical-pendulum rod radius in meters.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

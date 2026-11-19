# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Chain With Heavy Ball
#
# Reproduces the setup of M-ABD Fig. 11: a chain is released from a coiled
# configuration with its first link fixed and a heavy ball at the other end.
# The reference coil runs radially outward, descends through four turns, and
# then runs radially inward. It is rescaled to the requested chain length.
#
# Command: uv run python -m newton.examples cha_chain_heavy_ball
# Headless: uv run python -m newton.examples cha_chain_heavy_ball --viewer null --num-frames 300
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import coupled_high_accuracy as s3d

_DEFAULT_CHAIN_LENGTH = 32.583153
_DEFAULT_LINK_COUNT = 163
_DEFAULT_END_MASS = 10.0
_DEFAULT_BALL_RADIUS = 1

_REFERENCE_COIL_RADIUS = 1.20
_REFERENCE_COIL_TURNS = 4
_REFERENCE_PITCH_PER_TURN = -0.30
_LINK_RADIUS_TO_LENGTH = 0.10
_LINK_DENSITY = 100.0
_CHAIN_AFFINE_SPRING_ALPHA = 100_000.0
_CHAIN_JOINT_CONSTRAINT_ALPHA = 30_000.0
_CHAIN_LINEAR_ITERATIONS = 80


def _coiled_chain_nodes(total_length: float, link_count: int) -> np.ndarray:
    """Sample the Fig. 11 reference path and make its chord sum total_length."""
    radius = _REFERENCE_COIL_RADIUS
    pitch = _REFERENCE_PITCH_PER_TURN
    theta_end = 2.0 * np.pi * _REFERENCE_COIL_TURNS
    pitch_per_radian = pitch / (2.0 * np.pi)
    helix_length_per_radian = np.sqrt(radius**2 + pitch_per_radian**2)
    helix_length = theta_end * helix_length_per_radian
    reference_length = 2.0 * radius + helix_length
    final_height = pitch * _REFERENCE_COIL_TURNS

    nodes = np.empty((link_count + 1, 3), dtype=np.float64)
    for index, arc_length in enumerate(np.linspace(0.0, reference_length, link_count + 1)):
        if arc_length <= radius:
            nodes[index] = (arc_length, 0.0, 0.0)
        elif arc_length <= radius + helix_length:
            theta = (arc_length - radius) / helix_length_per_radian
            nodes[index] = (
                radius * np.cos(theta),
                radius * np.sin(theta),
                pitch_per_radian * theta,
            )
        else:
            inward_fraction = (arc_length - radius - helix_length) / radius
            nodes[index] = (
                radius * (1.0 - inward_fraction),
                0.0,
                final_height,
            )

    chord_length = float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum())
    if not np.isfinite(chord_length) or chord_length <= 0.0:
        raise ValueError("The coiled chain path has an invalid length.")
    return nodes * (total_length / chord_length)


def _validate_parameters(
    total_length: float,
    link_count: int,
    end_mass: float,
    ball_radius: float,
) -> None:
    if not np.isfinite(total_length) or total_length <= 0.0:
        raise ValueError("Chain length must be finite and positive.")
    if link_count < 8:
        raise ValueError("The coiled chain requires at least eight links.")
    if not np.isfinite(end_mass) or end_mass <= 0.0:
        raise ValueError("End mass must be finite and positive.")
    if not np.isfinite(ball_radius) or ball_radius <= 0.0:
        raise ValueError("Ball radius must be finite and positive.")


@wp.kernel
def interpolate_affine_endpoint(
    particle_q: wp.array[wp.vec3],
    proxy_indices: wp.vec4i,
    weights: wp.vec4,
    endpoint: wp.array[wp.vec3],
):
    endpoint[0] = (
        weights[0] * particle_q[proxy_indices[0]]
        + weights[1] * particle_q[proxy_indices[1]]
        + weights[2] * particle_q[proxy_indices[2]]
        + weights[3] * particle_q[proxy_indices[3]]
    )


class Example:
    def __init__(self, viewer, args):
        self.total_length = float(getattr(args, "length", _DEFAULT_CHAIN_LENGTH))
        self.link_count = int(getattr(args, "segments", _DEFAULT_LINK_COUNT))
        self.end_mass = float(getattr(args, "mass", _DEFAULT_END_MASS))
        self.ball_radius = float(getattr(args, "radius", _DEFAULT_BALL_RADIUS))
        _validate_parameters(
            self.total_length,
            self.link_count,
            self.end_mass,
            self.ball_radius,
        )

        # Fig. 11 uses h=1e-3 s. Ten substeps at 100 output frames per second
        # preserve that integration step exactly.
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer

        nodes = _coiled_chain_nodes(self.total_length, self.link_count)
        segment_vectors = np.diff(nodes, axis=0)
        self.segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        segment_directions = segment_vectors / self.segment_lengths[:, None]
        link_radius = _LINK_RADIUS_TO_LENGTH * self.total_length / self.link_count

        # Place the fixed end high enough to keep the fully extended chain in
        # a useful world-coordinate range. The scene intentionally has no floor.
        self.upper_endpoint = np.array((0.0, 0.0, 0.55 * self.total_length), dtype=np.float64)
        nodes = nodes + self.upper_endpoint

        builder = s3d.ModelBuilder(
            up_axis=newton.Axis.Z,
            gravity=-9.81,
            affine_body_particle_radius=0.0,
        )
        link_cfg = builder.default_shape_cfg.copy()
        link_cfg.density = _LINK_DENSITY
        link_cfg.has_shape_collision = False
        link_cfg.has_particle_collision = False

        self.links = []
        link_rotations = []
        for index, (start, end, direction, segment_length) in enumerate(
            zip(
                nodes[:-1],
                nodes[1:],
                segment_directions,
                self.segment_lengths,
            )
        ):
            center = 0.5 * (start + end)
            rotation = wp.quat_between_vectors(wp.vec3(0.0, 0.0, 1.0), wp.vec3(*direction))
            body = builder.add_link(
                xform=wp.transform(wp.vec3(*center), rotation),
                label=f"chain_link_{index:03d}",
            )
            color = wp.vec3(0.70, 0.72, 0.76) if index % 2 == 0 else wp.vec3(0.32, 0.36, 0.42)
            builder.add_shape_cylinder(
                body,
                radius=link_radius,
                half_height=0.5 * float(segment_length),
                cfg=link_cfg,
                color=color,
            )
            self.links.append(body)
            link_rotations.append(rotation)

        payload_inertia = 0.4 * self.end_mass * self.ball_radius**2
        self.payload = builder.add_link(
            xform=wp.transform(wp.vec3(*nodes[-1]), wp.quat_identity()),
            mass=self.end_mass,
            inertia=wp.mat33(
                payload_inertia,
                0.0,
                0.0,
                0.0,
                payload_inertia,
                0.0,
                0.0,
                0.0,
                payload_inertia,
            ),
            lock_inertia=True,
            label="heavy_end_ball",
        )
        payload_cfg = builder.default_shape_cfg.copy()
        payload_cfg.has_shape_collision = False
        payload_cfg.has_particle_collision = False
        builder.add_shape_sphere(
            self.payload,
            radius=self.ball_radius,
            cfg=payload_cfg,
            color=wp.vec3(0.78, 0.18, 0.10),
        )

        anchor_cfg = builder.default_shape_cfg.copy()
        anchor_cfg.has_shape_collision = False
        anchor_cfg.has_particle_collision = False
        builder.add_shape_sphere(
            -1,
            xform=wp.transform(wp.vec3(*self.upper_endpoint), wp.quat_identity()),
            radius=2.5 * link_radius,
            cfg=anchor_cfg,
            color=wp.vec3(0.92, 0.68, 0.10),
        )

        identity = wp.quat_identity()
        joints = [
            builder.add_joint_fixed(
                parent=-1,
                child=self.links[0],
                parent_xform=wp.transform(
                    wp.vec3(*(0.5 * (nodes[0] + nodes[1]))),
                    link_rotations[0],
                ),
                child_xform=wp.transform_identity(),
                label="fixed_chain_root",
            )
        ]
        for index in range(self.link_count - 1):
            joints.append(
                builder.add_joint_ball(
                    parent=self.links[index],
                    child=self.links[index + 1],
                    parent_xform=wp.transform(
                        wp.vec3(0.0, 0.0, 0.5 * self.segment_lengths[index]),
                        identity,
                    ),
                    child_xform=wp.transform(
                        wp.vec3(
                            0.0,
                            0.0,
                            -0.5 * self.segment_lengths[index + 1],
                        ),
                        identity,
                    ),
                    label=f"chain_ball_joint_{index:03d}",
                )
            )
        joints.append(
            builder.add_joint_ball(
                parent=self.links[-1],
                child=self.payload,
                parent_xform=wp.transform(
                    wp.vec3(0.0, 0.0, 0.5 * self.segment_lengths[-1]),
                    identity,
                ),
                child_xform=wp.transform_identity(),
                label="payload_ball_joint",
            )
        )
        builder.add_articulation(joints, label="coiled_chain")
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.solver = s3d.SolverCoupledHighAccuracy(self.model)
        self.solver.config.affine_body_rigidity_model = s3d.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
        self.solver.config.joint_constraint_mode = s3d.JointConstraintMode.AUGMENTED_LAGRANGIAN
        self.solver.config.nonlinear_solver = s3d.NonlinearSolver.NEWTON
        self.solver.config.inertia_mode = s3d.InertiaMode.MIDPOINT_CAYLEY
        self.solver.config.joint_constraint_primal_iterations = 2
        self.solver.config.nonlinear_iterations = 30
        self.solver.config.linear_iterations = 20

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.stream = wp.get_stream(self.model.device)

        affine_body_ids = self.model.synreal_affine_body_ids.numpy()
        last_affine_body = int(np.flatnonzero(affine_body_ids == self.links[-1]).item())
        proxy_indices = self.model.synreal_affine_body_indices.numpy()[last_affine_body]
        body_com = self.model.synreal_affine_body_com.numpy()[last_affine_body]
        local_endpoint = np.array((0.0, 0.0, 0.5 * self.segment_lengths[-1]), dtype=np.float64)
        affine_coordinate = local_endpoint - body_com
        endpoint_weights = np.concatenate(((1.0,), affine_coordinate))
        self.last_proxy_indices = wp.vec4i(*(int(value) for value in proxy_indices))
        self.last_endpoint_weights = wp.vec4(*(float(value) for value in endpoint_weights))
        self.upper_endpoint_device = wp.array(
            (wp.vec3(*self.upper_endpoint),),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.lower_endpoint_device = wp.array(
            (wp.vec3(*nodes[-1]),),
            dtype=wp.vec3,
            device=self.model.device,
        )
        self.end_to_end_length = float(np.linalg.norm(nodes[-1] - nodes[0]))
        self.end_to_end_ratio = self.end_to_end_length / self.total_length
        self.maximum_end_to_end_ratio = self.end_to_end_ratio

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(
                0.0,
                -0.75 * self.total_length,
                0.05 * self.total_length,
            ),
            pitch=-2.0,
            yaw=90.0,
        )
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

    def update_end_to_end_length(self):
        wp.launch(
            interpolate_affine_endpoint,
            dim=1,
            inputs=[
                self.state_0.particle_q,
                self.last_proxy_indices,
                self.last_endpoint_weights,
            ],
            outputs=[self.lower_endpoint_device],
            device=self.model.device,
            stream=self.stream,
        )
        lower_endpoint = self.lower_endpoint_device.numpy()[0]
        self.end_to_end_length = float(np.linalg.norm(lower_endpoint - self.upper_endpoint))
        self.end_to_end_ratio = self.end_to_end_length / self.total_length
        self.maximum_end_to_end_ratio = max(self.maximum_end_to_end_ratio, self.end_to_end_ratio)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph, stream=self.stream)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.update_end_to_end_length()
        print(f"time={self.sim_time:7.3f} s  L={self.end_to_end_length:10.6f} m  L/L0={self.end_to_end_ratio:9.6f}")

    def test_final(self):
        poses = self.state_0.body_q.numpy()
        velocities = self.state_0.body_qd.numpy()
        particles = self.state_0.particle_q.numpy()
        if not np.isfinite(poses).all() or not np.isfinite(velocities).all() or not np.isfinite(particles).all():
            raise ValueError("Heavy-chain simulation produced a non-finite state")
        if not np.isfinite(self.end_to_end_ratio):
            raise ValueError("Heavy-chain end-to-end ratio is non-finite")
        if self.maximum_end_to_end_ratio >= 1.25:
            raise ValueError("Heavy-chain endpoints separated beyond 1.25 times the nominal length")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_lines(
            "/heavy_chain/end_to_end",
            self.upper_endpoint_device,
            self.lower_endpoint_device,
            (0.95, 0.72, 0.12),
        )
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=300)
        parser.add_argument(
            "--length",
            type=float,
            default=_DEFAULT_CHAIN_LENGTH,
            metavar="L",
            help="Nominal total chain length in meters.",
        )
        parser.add_argument(
            "--segments",
            "--links",
            dest="segments",
            type=int,
            default=_DEFAULT_LINK_COUNT,
            metavar="N",
            help="Number of rigid cylindrical links.",
        )
        parser.add_argument(
            "--mass",
            "--end-mass",
            dest="mass",
            type=float,
            default=_DEFAULT_END_MASS,
            metavar="M",
            help="Mass of the ball attached to the lower end in kilograms.",
        )
        parser.add_argument(
            "--radius",
            "--ball-radius",
            dest="radius",
            type=float,
            default=_DEFAULT_BALL_RADIUS,
            metavar="R",
            help="Radius of the lower-end ball in meters.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

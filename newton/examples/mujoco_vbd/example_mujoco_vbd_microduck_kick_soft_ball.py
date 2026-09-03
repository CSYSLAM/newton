# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kick a tetrahedral soft ball with a dynamically driven Microduck leg.

Microduck's fixed base prevents balance control from obscuring the contact
experiment, while all fourteen servo joints remain dynamic.  The right-leg
position drive performs a backswing and strike.  Its visible sole mesh is also
the only robot shape allowed to contact VBD particles; no hidden collision
proxy or scripted force acts on the ball.  ``SolverMuJoCoVBD`` returns the
deformable contact wrench to the finite-mass MuJoCo articulation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMuJoCoVBD

ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets" / "microduck"
MICRODUCK_MJCF = ASSET_ROOT / "robot_groundcontact.xml"

BALL_CENTER = np.array((0.100, -0.058, 0.046), dtype=np.float32)
BALL_RADIUS = 0.045
BALL_PARTICLE_RADIUS = 0.002

HOME_POSE = np.array(
    (
        0.0,
        -0.0872664626,
        -0.457924,
        -0.004940,
        0.452984,
        0.3490658504,
        0.3490658504,
        0.0,
        0.0,
        0.0,
        0.0872664626,
        0.457924,
        0.004940,
        -0.452984,
    ),
    dtype=np.float32,
)


def _make_pose(*, right_hip_pitch: float, right_knee: float, right_ankle: float) -> np.ndarray:
    pose = HOME_POSE.copy()
    pose[11] = right_hip_pitch
    pose[12] = right_knee
    pose[13] = right_ankle
    return pose


KICK_KEYFRAMES = (
    (0.00, HOME_POSE),
    (0.45, HOME_POSE),
    (0.78, _make_pose(right_hip_pitch=-0.15, right_knee=-0.45, right_ankle=0.45)),
    (0.90, _make_pose(right_hip_pitch=-0.15, right_knee=-0.45, right_ankle=0.45)),
    (1.08, _make_pose(right_hip_pitch=1.20, right_knee=-0.45, right_ankle=-0.30)),
    (1.32, _make_pose(right_hip_pitch=1.45, right_knee=-0.15, right_ankle=-0.90)),
    (2.05, HOME_POSE),
    (5.00, HOME_POSE),
)


def _normalized(vector: tuple[float, float, float]) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    return value / np.linalg.norm(value)


def _midpoint_index(
    vertices: list[np.ndarray],
    cache: dict[tuple[int, int], int],
    first: int,
    second: int,
) -> int:
    edge = (min(first, second), max(first, second))
    cached = cache.get(edge)
    if cached is not None:
        return cached
    value = vertices[first] + vertices[second]
    index = len(vertices)
    vertices.append(value / np.linalg.norm(value))
    cache[edge] = index
    return index


def _tetrahedral_icosphere(subdivisions: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return a unit icosphere shell tetrahedralized to one interior node."""
    phi = (1.0 + math.sqrt(5.0)) * 0.5
    vertices = [
        _normalized(value)
        for value in (
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        )
    ]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]

    for _ in range(subdivisions):
        midpoint_cache: dict[tuple[int, int], int] = {}
        refined = []
        for first, second, third in faces:
            ab = _midpoint_index(vertices, midpoint_cache, first, second)
            bc = _midpoint_index(vertices, midpoint_cache, second, third)
            ca = _midpoint_index(vertices, midpoint_cache, third, first)
            refined.extend(((first, ab, ca), (second, bc, ab), (third, ca, bc), (ab, bc, ca)))
        faces = refined

    shell = np.asarray(vertices, dtype=np.float32)
    all_vertices = np.concatenate((np.zeros((1, 3), dtype=np.float32), shell), axis=0)
    tetrahedra = []
    for face in faces:
        first, second, third = (index + 1 for index in face)
        signed_six_volume = float(np.dot(all_vertices[first], np.cross(all_vertices[second], all_vertices[third])))
        if signed_six_volume < 0.0:
            second, third = third, second
        tetrahedra.append((0, first, second, third))
    return all_vertices, np.asarray(tetrahedra, dtype=np.int32)


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


class Example:
    """Drive one finite-mass Microduck foot through a VBD soft ball."""

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = int(args.substeps)
        if self.sim_substeps < 1:
            raise ValueError("--substeps must be at least 1")
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.track_metrics = bool(args.test)
        self.right_foot_contact_observed = False
        self.max_feedback_force = 0.0
        self.max_ball_deformation = 0.0
        self.minimum_ball_height = float("inf")

        if not MICRODUCK_MJCF.is_file():
            raise FileNotFoundError(f"Microduck MJCF not found: {MICRODUCK_MJCF}")

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        SolverMuJoCoVBD.register_custom_attributes(builder)
        builder.default_shape_cfg.ke = 1.0e5
        builder.default_shape_cfg.kd = 100.0
        builder.default_shape_cfg.mu = 0.7
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.001

        articulation_start = builder.articulation_count
        robot_shape_start = builder.shape_count
        builder.add_mjcf(
            str(MICRODUCK_MJCF),
            floating=False,
            enable_self_collisions=False,
            parse_visuals=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.robot_articulations = tuple(range(articulation_start, builder.articulation_count))
        if len(self.robot_articulations) != 1:
            raise RuntimeError(f"Expected one Microduck articulation, got {self.robot_articulations}")
        robot_shape_end = builder.shape_count

        if len(builder.joint_q) != HOME_POSE.size:
            raise RuntimeError(f"Expected {HOME_POSE.size} Microduck coordinates, got {len(builder.joint_q)}")
        builder.joint_q[:] = HOME_POSE.tolist()
        self.right_foot_shape = self._configure_robot_particle_contact(builder, robot_shape_start, robot_shape_end)
        self.right_foot_body = int(builder.shape_body[self.right_foot_shape])

        self.ball_particle_start = builder.particle_count
        ball_vertices, ball_tetrahedra = _tetrahedral_icosphere(subdivisions=int(args.ball_subdivisions))
        builder.add_soft_mesh(
            pos=wp.vec3(*BALL_CENTER),
            rot=wp.quat_identity(),
            scale=BALL_RADIUS,
            vel=wp.vec3(0.0),
            vertices=ball_vertices,
            indices=ball_tetrahedra.reshape(-1),
            density=180.0,
            k_mu=2.0e4,
            k_lambda=3.5e4,
            k_damp=120.0,
            tri_ke=0.0,
            tri_ka=0.0,
            tri_kd=0.0,
            edge_ke=0.0,
            edge_kd=0.0,
            particle_radius=BALL_PARTICLE_RADIUS,
            validate_mesh=True,
            label="soft_kick_ball",
        )
        self.ball_particle_end = builder.particle_count

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=1.0e5,
            kd=100.0,
            mu=0.65,
            margin=0.0,
            gap=0.001,
        )
        builder.add_ground_plane(cfg=ground_cfg, color=(0.12, 0.15, 0.19), label="kick_ground")
        builder.color(balance_colors=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = 1.5e5
        self.model.soft_contact_kd = 120.0
        self.model.soft_contact_mu = 0.65

        self.solver = SolverMuJoCoVBD(
            self.model,
            mujoco_articulations=self.robot_articulations,
            joint_mode="dynamic",
            coupling_mode="two_way",
            contact_mode="soft",
            coupling_options={
                "iterations": int(args.coupling_iterations),
                "relaxation": "fixed",
                "relaxation_initial": 0.65,
                "soft_contact_speculative_distance": 0.003,
                "soft_contact_augmented_lagrangian": True,
            },
            mujoco_options={"njmax": 256},
            vbd_options={
                "iterations": int(args.vbd_iterations),
                "friction_epsilon": 0.005,
                "particle_enable_self_contact": False,
                "particle_self_contact_radius": BALL_PARTICLE_RADIUS,
                "particle_self_contact_margin": 0.0,
            },
            collision_options={"soft_contact_margin": 0.0015},
        )
        if self.solver.backend_kind.value != "two_way":
            raise RuntimeError(f"Expected two_way, got {self.solver.backend_kind.value}")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.control.joint_target_q.assign(HOME_POSE)
        self.control.joint_target_qd.zero_()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.contacts = self.solver.contacts

        initial_ball = self.state_0.particle_q.numpy()[self.ball_particle_start : self.ball_particle_end]
        self.initial_ball_center = np.mean(initial_ball, axis=0)
        self.initial_ball_radii = np.linalg.norm(initial_ball - self.initial_ball_center, axis=1)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.42, -0.55, 0.27), pitch=-13.0, yaw=128.0)

        self.graph = None
        if bool(args.graph_capture) and self.model.device.is_cuda:
            with wp.ScopedDevice(self.model.device), wp.ScopedCapture() as capture:
                self._simulate_frame()
            self.graph = capture.graph

    @staticmethod
    def _configure_robot_particle_contact(
        builder: newton.ModelBuilder,
        shape_start: int,
        shape_end: int,
    ) -> int:
        """Use the visible right sole's matching mesh as the sole particle collider."""
        particle_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        right_foot_shape = -1
        for shape in range(shape_start, shape_end):
            if builder.shape_label[shape].endswith("/right_foot_collision"):
                right_foot_shape = shape
                builder.shape_flags[shape] |= particle_flag
            else:
                builder.shape_flags[shape] &= ~particle_flag
        if right_foot_shape < 0:
            raise RuntimeError("Microduck MJCF did not provide right_foot_collision")
        return right_foot_shape

    def _target_pose(self, time_s: float) -> np.ndarray:
        for index in range(len(KICK_KEYFRAMES) - 1):
            time_a, pose_a = KICK_KEYFRAMES[index]
            time_b, pose_b = KICK_KEYFRAMES[index + 1]
            if time_s <= time_b:
                alpha = _smoothstep((time_s - time_a) / (time_b - time_a))
                return (1.0 - alpha) * pose_a + alpha * pose_b
        return HOME_POSE

    def _simulate_frame(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Update finite-stiffness servo targets and advance one display frame."""
        self.control.joint_target_q.assign(self._target_pose(self.sim_time))
        if self.graph is None:
            self._simulate_frame()
        else:
            wp.capture_launch(self.graph)

        if self.track_metrics:
            wp.synchronize()
            count = int(self.contacts.soft_contact_count.numpy()[0])
            shape_indices = self.contacts.soft_contact_shape.numpy()
            active = min(count, shape_indices.size)
            self.right_foot_contact_observed |= bool(np.any(shape_indices[:active] == self.right_foot_shape))
            feedback = self.solver.diagnostics.feedback_wrench_raw.numpy()[self.right_foot_body, :3]
            self.max_feedback_force = max(self.max_feedback_force, float(np.linalg.norm(feedback)))

            ball = self.state_0.particle_q.numpy()[self.ball_particle_start : self.ball_particle_end]
            center = np.mean(ball, axis=0)
            radii = np.linalg.norm(ball - center, axis=1)
            self.max_ball_deformation = max(
                self.max_ball_deformation,
                float(np.max(np.abs(radii - self.initial_ball_radii))),
            )
            self.minimum_ball_height = min(self.minimum_ball_height, float(np.min(ball[:, 2])))

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Require real foot contact, two-way reaction, deformation, and a forward kick."""
        body_q = self.state_0.body_q.numpy()
        ball = self.state_0.particle_q.numpy()[self.ball_particle_start : self.ball_particle_end]
        final_center = np.mean(ball, axis=0)
        forward_displacement = float(final_center[0] - self.initial_ball_center[0])
        assert np.all(np.isfinite(body_q))
        assert np.all(np.isfinite(ball))
        assert self.right_foot_contact_observed, "The right sole never contacted the soft ball"
        assert self.max_feedback_force > 1.0e-3, "No soft-ball reaction was returned to Microduck"
        assert self.max_feedback_force < 100.0, f"Soft-ball impact spiked to {self.max_feedback_force:.1f} N"
        assert self.max_ball_deformation > 5.0e-4, "The tetrahedral ball did not visibly deform"
        assert self.max_ball_deformation < 0.012, (
            f"Soft ball collapsed by {self.max_ball_deformation:.4f} m; increase temporal resolution"
        )
        assert forward_displacement > 0.03, f"Soft ball moved only {forward_displacement:.4f} m forward"
        assert self.minimum_ball_height > -0.006, f"Soft ball penetrated the ground to {self.minimum_ball_height:.4f} m"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=300)
        parser.add_argument("--substeps", type=int, default=4, help="Simulation substeps per 60 Hz frame.")
        parser.add_argument(
            "--coupling-iterations",
            type=int,
            default=2,
            help="Two-way fixed-point iterations per substep.",
        )
        parser.add_argument("--vbd-iterations", type=int, default=8, help="VBD nonlinear iterations per substep.")
        parser.add_argument(
            "--ball-subdivisions",
            type=int,
            choices=(1, 2),
            default=2,
            help="Icosphere refinement level; 2 creates 163 particles and 320 tetrahedra.",
        )
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture all substeps as one CUDA graph.",
        )
        return parser


def main():
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

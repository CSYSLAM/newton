# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Crush a tetrahedral Armadillo with the standalone MuJoCo/VBD solver.

This reproduces the DexSim gear-crusher scene with its original VTK asset,
procedural 16-tooth rollers, dimensions, material parameters, and prescribed
roller motion. The scene has no joints, so the solver selects its private
pure-VBD branch for the tetrahedral body, self-contact, and particle contact
against the two externally prescribed kinematic rollers.

Run, from the repository root::

    uv run --extra examples -m newton.examples mujoco_vbd_gear_crusher
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

FPS = 60
DEFAULT_NUM_FRAMES = 360
DEFAULT_SUBSTEPS = 10
DEFAULT_VBD_ITERATIONS = 10

ROLLER_INNER_RADIUS = 0.36
ROLLER_OUTER_RADIUS = 0.40
ROLLER_LENGTH = 1.60
ROLLER_TEETH = 16
ROLLER_GAP = 0.08
ROLLER_SEPARATION = 2.0 * ROLLER_OUTER_RADIUS + ROLLER_GAP

ARMADILLO_DENSITY = 1000.0
K_MU = 1.0e5
K_LAMBDA = 1.0e6
CONTACT_KE = 1.0e6
CONTACT_KD = 1.0e-7
CONTACT_MU = 0.2
CONTACT_RADIUS = 0.005
SOFT_CONTACT_MARGIN = 0.01

GEAR_COLOR = (0.50, 0.53, 0.58)
ARMADILLO_COLOR = (0.20, 0.78, 0.42)
GROUND_COLOR = (0.18, 0.20, 0.23)

ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets" / "gear_crusher_assets"
DEFAULT_ARMADILLO_ASSET = ASSET_ROOT / "Armadilo_15K.1.vtk"


@wp.kernel
def prescribe_crusher_bodies(
    left_body: int,
    right_body: int,
    sim_time: wp.array[float],
    angular_speed: float,
    body_q_in: wp.array[wp.transform],
    body_qd_in: wp.array[wp.spatial_vector],
    body_q_out: wp.array[wp.transform],
    body_qd_out: wp.array[wp.spatial_vector],
):
    """Set counter-rotating roller poses in both state buffers."""
    t = sim_time[0]
    axis = wp.vec3(1.0, 0.0, 0.0)
    zero = wp.vec3(0.0, 0.0, 0.0)

    left_angle = -angular_speed * t
    right_angle = angular_speed * t
    left_q = wp.transform(
        wp.vec3(0.0, -0.5 * ROLLER_SEPARATION, 0.0),
        wp.quat_from_axis_angle(axis, left_angle),
    )
    right_q = wp.transform(
        wp.vec3(0.0, 0.5 * ROLLER_SEPARATION, 0.0),
        wp.quat_from_axis_angle(axis, right_angle),
    )
    left_qd = wp.spatial_vector(zero, axis * -angular_speed)
    right_qd = wp.spatial_vector(zero, axis * angular_speed)

    body_q_in[left_body] = left_q
    body_q_in[right_body] = right_q
    body_qd_in[left_body] = left_qd
    body_qd_in[right_body] = right_qd
    body_q_out[left_body] = left_q
    body_q_out[right_body] = right_q
    body_qd_out[left_body] = left_qd
    body_qd_out[right_body] = right_qd


@wp.kernel
def advance_crusher_time(sim_time: wp.array[float], dt: float):
    """Advance the device-resident prescribed-motion clock."""
    sim_time[0] = sim_time[0] + dt


@wp.kernel
def mark_gear_contact(
    contact_count: wp.array[int],
    contact_shape: wp.array[int],
    left_shape: int,
    right_shape: int,
    observed: wp.array[int],
):
    """Record whether a test run generated any roller contact."""
    contact = wp.tid()
    if contact >= contact_count[0]:
        return
    shape = contact_shape[contact]
    if shape == left_shape or shape == right_shape:
        wp.atomic_max(observed, 0, 1)


def create_gear_cylinder_mesh() -> newton.Mesh:
    """Create the original extruded 16-tooth roller along the X axis."""
    profile: list[tuple[float, float]] = []
    tooth_angle = 2.0 * math.pi / ROLLER_TEETH
    for tooth in range(ROLLER_TEETH):
        base = tooth * tooth_angle
        profile.extend(
            (
                (base, ROLLER_INNER_RADIUS),
                (base + tooth_angle * 0.08, ROLLER_INNER_RADIUS),
                (base + tooth_angle * 0.15, ROLLER_OUTER_RADIUS),
                (base + tooth_angle * 0.85, ROLLER_OUTER_RADIUS),
                (base + tooth_angle * 0.92, ROLLER_INNER_RADIUS),
            )
        )

    half_length = 0.5 * ROLLER_LENGTH
    vertices = []
    for x in (-half_length, half_length):
        vertices.extend((x, radius * math.cos(angle), radius * math.sin(angle)) for angle, radius in profile)

    profile_count = len(profile)
    triangles = []
    for index in range(profile_count):
        next_index = (index + 1) % profile_count
        triangles.append((index, next_index, profile_count + next_index))
        triangles.append((index, profile_count + next_index, profile_count + index))

    left_center = len(vertices)
    vertices.append((-half_length, 0.0, 0.0))
    right_center = len(vertices)
    vertices.append((half_length, 0.0, 0.0))
    for index in range(profile_count):
        next_index = (index + 1) % profile_count
        triangles.append((left_center, next_index, index))
        triangles.append((right_center, profile_count + index, profile_count + next_index))

    mesh = newton.Mesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        indices=np.asarray(triangles, dtype=np.int32).reshape(-1),
        compute_inertia=False,
    )
    mesh.color = GEAR_COLOR
    mesh.roughness = 0.38
    mesh.metallic = 0.92
    return mesh


def load_vtk_tet_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the ASCII legacy-VTK Armadillo tetrahedral mesh."""
    if not path.is_file():
        raise FileNotFoundError(f"Gear-crusher asset is missing: {path}")

    tokens = path.read_text(encoding="utf-8").split()
    try:
        points_at = tokens.index("POINTS")
        point_count = int(tokens[points_at + 1])
        points_start = points_at + 3
        points_end = points_start + 3 * point_count
        vertices = np.asarray(tokens[points_start:points_end], dtype=np.float32).reshape(-1, 3)

        cells_at = tokens.index("CELLS", points_end)
        cell_count = int(tokens[cells_at + 1])
        cursor = cells_at + 3
        tetrahedra: list[list[int]] = []
        for _ in range(cell_count):
            arity = int(tokens[cursor])
            cursor += 1
            cell = [int(index) for index in tokens[cursor : cursor + arity]]
            cursor += arity
            if arity == 4:
                tetrahedra.append(cell)
    except (ValueError, IndexError) as error:
        raise ValueError(f"Invalid legacy VTK tetrahedral mesh: {path}") from error

    if len(vertices) != point_count or not tetrahedra:
        raise ValueError(f"VTK mesh has incomplete points or no tetrahedra: {path}")
    return vertices, np.asarray(tetrahedra, dtype=np.int32)


class Example:
    """Run the gear crusher through the private pure-VBD specialization."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / FPS
        self.sim_substeps = int(args.substeps)
        self.vbd_iterations = int(args.vbd_iterations)
        if self.sim_substeps < 1:
            raise ValueError("--substeps must be at least 1")
        if self.vbd_iterations < 1:
            raise ValueError("--vbd-iterations must be at least 1")
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.rotation_speed = float(args.rotation_speed)

        asset_path = DEFAULT_ARMADILLO_ASSET if args.asset_path is None else args.asset_path.expanduser()
        vertices, tetrahedra = load_vtk_tet_mesh(asset_path)

        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.81))
        builder.default_particle_radius = CONTACT_RADIUS
        SolverMuJoCoVBD.register_custom_attributes(builder)

        gear_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=CONTACT_MU,
            restitution=0.0,
        )
        gear_mesh = create_gear_cylinder_mesh()
        gear_mass = 25.0
        i_axis = 0.5 * gear_mass * ROLLER_OUTER_RADIUS**2
        i_transverse = gear_mass * (3.0 * ROLLER_OUTER_RADIUS**2 + ROLLER_LENGTH**2) / 12.0
        gear_inertia = wp.mat33(i_axis, 0.0, 0.0, 0.0, i_transverse, 0.0, 0.0, 0.0, i_transverse)

        self.left_body = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, -0.5 * ROLLER_SEPARATION, 0.0), wp.quat_identity()),
            mass=gear_mass,
            inertia=gear_inertia,
            is_kinematic=True,
            label="left_crusher_gear",
        )
        self.right_body = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.5 * ROLLER_SEPARATION, 0.0), wp.quat_identity()),
            mass=gear_mass,
            inertia=gear_inertia,
            is_kinematic=True,
            label="right_crusher_gear",
        )
        self.left_shape = builder.add_shape_mesh(
            self.left_body,
            mesh=gear_mesh,
            cfg=gear_cfg,
            label="left_crusher_gear_mesh",
        )
        self.right_shape = builder.add_shape_mesh(
            self.right_body,
            mesh=gear_mesh,
            cfg=gear_cfg,
            label="right_crusher_gear_mesh",
        )
        builder.add_shape_collision_filter_pair(self.left_shape, self.right_shape)

        ground_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=CONTACT_MU,
        )
        builder.add_ground_plane(
            height=-1.5,
            cfg=ground_cfg,
            color=GROUND_COLOR,
            label="crusher_ground",
        )

        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=tetrahedra.reshape(-1).tolist(),
            density=ARMADILLO_DENSITY,
            k_mu=K_MU,
            k_lambda=K_LAMBDA,
            k_damp=CONTACT_KD,
            particle_radius=CONTACT_RADIUS,
            validate_mesh=True,
            label="paper_armadillo",
        )

        builder.color(balance_colors=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = CONTACT_KE
        self.model.soft_contact_kd = CONTACT_KD
        self.model.soft_contact_mu = CONTACT_MU

        self.solver = SolverMuJoCoVBD(
            self.model,
            joint_mode="dynamic",
            coupling_mode="auto",
            contact_mode="soft",
            collision_options={"soft_contact_margin": SOFT_CONTACT_MARGIN},
            vbd_options={
                "iterations": self.vbd_iterations,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": CONTACT_RADIUS,
                "particle_self_contact_margin": 0.0075,
                "particle_conservative_bound_relaxation": 0.85,
                "particle_collision_detection_interval": 5,
                "particle_topological_contact_filter_threshold": 1,
                "particle_rest_shape_contact_exclusion_radius": 0.02,
                "particle_enable_tile_solve": True,
                "particle_vertex_contact_buffer_size": 32,
                "particle_edge_contact_buffer_size": 64,
                "rigid_body_particle_contact_buffer_size": 16384,
            },
        )
        if self.solver.features.backend.value != "pure_vbd_soft":
            raise RuntimeError(f"Gear crusher requires pure_vbd_soft, got {self.solver.features.backend.value}")
        if self.solver.features.mujoco_solve_enabled:
            raise RuntimeError("Gear crusher unexpectedly enabled the MuJoCo solve")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.solver.contacts
        self.sim_time_wp = wp.zeros(1, dtype=float, device=self.model.device)
        self.initial_particle_q = self.state_0.particle_q.numpy().copy()
        self.surface_indices = self.model.tri_indices.flatten()
        self.track_gear_contacts = bool(args.test)
        self.gear_contact_observed = wp.zeros(1, dtype=int, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(3.2, -6.4, 2.35), pitch=-5.0, yaw=116.0)
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.fov = 43.0

        self.use_graph = bool(args.graph_capture) and self.model.device.is_cuda
        self.graph = None
        self.capture()

    def capture(self):
        """Capture one complete display-frame update."""
        if not self.use_graph:
            return
        with wp.ScopedDevice(self.model.device), wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        """Advance all pure-VBD crusher substeps."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.copy(self.state_1.particle_q, self.state_0.particle_q)
            wp.copy(self.state_1.particle_qd, self.state_0.particle_qd)
            wp.launch(
                prescribe_crusher_bodies,
                dim=1,
                inputs=[
                    self.left_body,
                    self.right_body,
                    self.sim_time_wp,
                    self.rotation_speed,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.state_1.body_q,
                    self.state_1.body_qd,
                ],
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            if self.track_gear_contacts:
                wp.launch(
                    mark_gear_contact,
                    dim=self.contacts.soft_contact_max,
                    inputs=[
                        self.contacts.soft_contact_count,
                        self.contacts.soft_contact_shape,
                        self.left_shape,
                        self.right_shape,
                        self.gear_contact_observed,
                    ],
                    device=self.model.device,
                )
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(
                advance_crusher_time,
                dim=1,
                inputs=[self.sim_time_wp, self.sim_dt],
                device=self.model.device,
            )

    def step(self):
        """Advance one displayed frame."""
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the rollers and tetrahedral surface."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/gear_crusher/armadillo",
            self.state_0.particle_q,
            self.surface_indices,
            hidden=not getattr(self.viewer, "show_triangles", True),
            backface_culling=False,
            color=ARMADILLO_COLOR,
        )
        self.viewer.end_frame()

    def test_final(self):
        """Verify finite pure-VBD state and prescribed roller motion."""
        particle_q = self.state_0.particle_q.numpy()
        if not np.all(np.isfinite(particle_q)):
            raise ValueError("Gear-crusher Armadillo state is not finite")
        if self.solver.features.backend.value != "pure_vbd_soft" or self.solver.features.mujoco_solve_enabled:
            raise ValueError("Gear crusher did not retain the pure-VBD backend")
        if not self.solver.features.tetrahedron_solve_enabled:
            raise ValueError("Gear crusher did not enable tetrahedral VBD")
        if self.sim_time >= 0.5:
            if int(self.gear_contact_observed.numpy()[0]) == 0:
                raise ValueError("Armadillo never contacted either crusher roller")
            displacement = np.linalg.norm(particle_q - self.initial_particle_q, axis=1)
            if float(np.max(displacement)) < 0.01:
                raise ValueError("Armadillo did not respond to gravity and the crusher rollers")

    @staticmethod
    def create_parser():
        """Create command-line arguments for the gear-crusher scene."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=DEFAULT_NUM_FRAMES)
        parser.add_argument(
            "--asset-path",
            type=Path,
            default=None,
            help="Optional Armadillo legacy-VTK path; defaults to the repository asset.",
        )
        parser.add_argument(
            "--rotation-speed",
            type=float,
            default=1.0,
            help="Magnitude of each roller's counter-rotation speed [rad/s].",
        )
        parser.add_argument(
            "--substeps",
            type=int,
            default=DEFAULT_SUBSTEPS,
            help="Pure-VBD substeps per displayed frame.",
        )
        parser.add_argument(
            "--vbd-iterations",
            type=int,
            default=DEFAULT_VBD_ITERATIONS,
            help="VBD iterations per substep.",
        )
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture a complete displayed frame on CUDA.",
        )
        return parser


def main():
    """Run the gear-crusher example."""
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

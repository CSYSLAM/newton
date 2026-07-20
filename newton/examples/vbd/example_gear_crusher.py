# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Gear Crusher
#
# Reproduces the Gear Crusher experiment from Figure 8 of "Divide and
# Truncate: A Penetration and Inversion Free Framework for Coupled
# Multi-physics Systems". A tetrahedral Stanford Armadillo is pulled through
# two animated, counter-rotating crusher drums using SolverVBD's Planar-DAT
# contact handling.
#
# Command: python -m newton.examples gear_crusher
###########################################################################

from __future__ import annotations

import math
import os

import numpy as np
import warp as wp

import newton
import newton.examples

# Parameters reported for Gear Crusher in Table 1 of the DAT paper.
FPS = 60
DEFAULT_NUM_FRAMES = 360
PAPER_SIM_SUBSTEPS = 10
PAPER_SOLVER_ITERATIONS = 10
QUALITY_PRESETS = {
    "interactive": (3, 4),
    "balanced": (5, 6),
    "paper": (PAPER_SIM_SUBSTEPS, PAPER_SOLVER_ITERATIONS),
}
K_LAMBDA = 1.0e6
K_MU = 1.0e5
CONTACT_KE = 1.0e6
CONTACT_MU = 0.2
CONTACT_RADIUS = 0.005
CONTACT_QUERY_RADIUS = 1.5 * CONTACT_RADIUS

# Figure 8 uses 16-tooth counter-rotating drums. Their angular speed is not
# tabulated; pi/2 rad/s matches the repeated tooth phase in the 1-second
# snapshots included with the paper source.
GEAR_TEETH = 16
GEAR_ROOT_RADIUS = 0.60
GEAR_TIP_RADIUS = 0.75
GEAR_HALF_WIDTH = 0.65
GEAR_CENTER_X = 0.80
GEAR_ANGULAR_SPEED = 0.5 * math.pi
GEAR_PHASE = math.pi / GEAR_TEETH

# Armadillo15K contains 14,779 vertices and 54,855 tetrahedra, closely
# matching the paper's reported 15K vertices and 60K tetrahedra.
ARMADILLO_SCALE = 0.0075
ARMADILLO_CLEARANCE = 0.03
ARMADILLO_DENSITY = 1000.0


@wp.kernel
def set_gear_joint_state(
    left_q_start: int,
    left_qd_start: int,
    right_q_start: int,
    right_qd_start: int,
    sim_time: wp.array[float],
    angular_speed: float,
    phase: float,
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
):
    """Prescribe inward counter-rotation for both crusher drums."""
    t = sim_time[0]
    joint_q[left_q_start] = phase + angular_speed * t
    joint_qd[left_qd_start] = angular_speed
    joint_q[right_q_start] = -phase - angular_speed * t
    joint_qd[right_qd_start] = -angular_speed


@wp.kernel
def advance_time(sim_time: wp.array[float], dt: float):
    sim_time[0] = sim_time[0] + dt


def create_crusher_gear() -> newton.Mesh:
    """Create the closed 16-tooth drum used in the paper scene."""
    pitch = 2.0 * math.pi / GEAR_TEETH
    half_tip_angle = 0.32 * pitch
    profile = []

    # Duplicate the angular coordinate at each tooth wall so the longitudinal
    # teeth have radial sides instead of the sloped sides of a conventional cog.
    for tooth in range(GEAR_TEETH):
        center = tooth * pitch
        a0 = center - half_tip_angle
        a1 = center + half_tip_angle
        profile.extend(
            (
                (GEAR_ROOT_RADIUS, a0),
                (GEAR_TIP_RADIUS, a0),
                (GEAR_TIP_RADIUS, a1),
                (GEAR_ROOT_RADIUS, a1),
            )
        )

    ring = np.asarray(
        [[radius * math.cos(angle), radius * math.sin(angle)] for radius, angle in profile],
        dtype=np.float32,
    )
    count = len(ring)
    front = np.column_stack((ring[:, 0], np.full(count, GEAR_HALF_WIDTH), ring[:, 1]))
    back = np.column_stack((ring[:, 0], np.full(count, -GEAR_HALF_WIDTH), ring[:, 1]))
    vertices = np.vstack((front, back, [[0.0, GEAR_HALF_WIDTH, 0.0], [0.0, -GEAR_HALF_WIDTH, 0.0]])).astype(np.float32)
    front_center = 2 * count
    back_center = front_center + 1
    triangles = []

    for i in range(count):
        j = (i + 1) % count
        fi, fj = i, j
        bi, bj = count + i, count + j

        # The duplicated radial profile points form tooth walls and have no
        # cap area, so omit only those degenerate fan triangles.
        if not np.allclose(ring[i], ring[j]):
            triangles.append((front_center, fj, fi))
            triangles.append((back_center, bi, bj))

        triangles.append((fi, fj, bj))
        triangles.append((fi, bj, bi))

    mesh = newton.Mesh(
        vertices=vertices,
        indices=np.asarray(triangles, dtype=np.int32).reshape(-1),
        compute_inertia=False,
    )
    mesh.color = (0.45, 0.48, 0.55)
    mesh.roughness = 0.38
    mesh.metallic = 0.92
    return mesh


def cylinder_inertia(mass: float) -> wp.mat33:
    """Return a conservative solid-cylinder inertia approximation."""
    length = 2.0 * GEAR_HALF_WIDTH
    i_axis = 0.5 * mass * GEAR_TIP_RADIUS**2
    i_transverse = mass * (3.0 * GEAR_TIP_RADIUS**2 + length**2) / 12.0
    return wp.mat33(i_transverse, 0.0, 0.0, 0.0, i_axis, 0.0, 0.0, 0.0, i_transverse)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / FPS
        self.sim_substeps, self.solver_iterations = QUALITY_PRESETS[args.quality]
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.gear_speed = float(args.gear_speed)

        asset_path = os.path.join(newton.examples.get_asset_directory(), "armadillo15k.npz")
        armadillo = newton.TetMesh.create_from_file(asset_path)
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.81))
        builder.default_particle_radius = CONTACT_RADIUS

        gear_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=CONTACT_KE,
            kd=0.0,
            mu=CONTACT_MU,
            margin=0.0,
        )
        gear_cfg.has_particle_collision = True
        gear_mesh = create_crusher_gear()
        self.enable_full_surface_contact = wp.get_device().is_cuda
        if self.enable_full_surface_contact:
            cache_root = os.environ.get(
                "NEWTON_CACHE_PATH",
                os.path.join(os.path.expanduser("~"), ".cache", "newton"),
            )
            gear_mesh.build_sdf(
                device=wp.get_device(),
                narrow_band_range=(-0.02, 0.02),
                max_resolution=256,
                margin=0.02,
                cache_dir=os.path.join(cache_root, "sdf", "gear_crusher"),
            )
        gear_mass = 25.0
        gear_inertia = cylinder_inertia(gear_mass)
        self.left_gear_body = builder.add_link(
            mass=gear_mass,
            inertia=gear_inertia,
            is_kinematic=True,
            label="left_crusher_gear",
        )
        self.right_gear_body = builder.add_link(
            mass=gear_mass,
            inertia=gear_inertia,
            is_kinematic=True,
            label="right_crusher_gear",
        )
        left_shape = builder.add_shape_mesh(
            self.left_gear_body,
            mesh=gear_mesh,
            cfg=gear_cfg,
            label="left_crusher_gear_mesh",
        )
        right_shape = builder.add_shape_mesh(
            self.right_gear_body,
            mesh=gear_mesh,
            cfg=gear_cfg,
            label="right_crusher_gear_mesh",
        )
        builder.add_shape_collision_filter_pair(left_shape, right_shape)

        self.left_gear_joint = builder.add_joint_revolute(
            parent=-1,
            child=self.left_gear_body,
            axis=newton.Axis.Y,
            parent_xform=wp.transform(p=wp.vec3(-GEAR_CENTER_X, 0.0, 0.0), q=wp.quat_identity()),
            label="left_crusher_gear_joint",
        )
        self.right_gear_joint = builder.add_joint_revolute(
            parent=-1,
            child=self.right_gear_body,
            axis=newton.Axis.Y,
            parent_xform=wp.transform(p=wp.vec3(GEAR_CENTER_X, 0.0, 0.0), q=wp.quat_identity()),
            label="right_crusher_gear_joint",
        )
        builder.add_articulation([self.left_gear_joint], label="left_crusher_gear")
        builder.add_articulation([self.right_gear_joint], label="right_crusher_gear")

        # The source mesh is Y-up. Rotate it to Newton's Z-up convention, center
        # it over the drums, and place its feet just above the tooth tips.
        bounds_min = np.min(armadillo.vertices, axis=0)
        bounds_max = np.max(armadillo.vertices, axis=0)
        armadillo_pos = wp.vec3(
            -0.5 * ARMADILLO_SCALE * (bounds_min[0] + bounds_max[0]),
            0.5 * ARMADILLO_SCALE * (bounds_min[2] + bounds_max[2]),
            GEAR_TIP_RADIUS + ARMADILLO_CLEARANCE - ARMADILLO_SCALE * bounds_min[1],
        )
        armadillo_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi)
        builder.add_soft_mesh(
            pos=armadillo_pos,
            rot=armadillo_rot,
            scale=ARMADILLO_SCALE,
            vel=wp.vec3(0.0, 0.0, 0.0),
            mesh=armadillo,
            density=ARMADILLO_DENSITY,
            k_mu=K_MU,
            k_lambda=K_LAMBDA,
            k_damp=0.0,
            particle_radius=CONTACT_RADIUS,
            validate_mesh=True,
            label="armadillo15k",
        )

        builder.color()
        self.model = builder.finalize()
        self.model.soft_contact_ke = CONTACT_KE
        self.model.soft_contact_kd = 0.0
        self.model.soft_contact_mu = CONTACT_MU
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=self.solver_iterations,
            friction_epsilon=1.0e-3,
            particle_enable_self_contact=True,
            particle_self_contact_radius=CONTACT_RADIUS,
            particle_self_contact_margin=CONTACT_QUERY_RADIUS,
            particle_conservative_bound_relaxation=0.85,
            particle_vertex_contact_buffer_size=96,
            particle_edge_contact_buffer_size=192,
            # Refresh midway through each solve because crusher-scale deformation
            # can invalidate the conservative candidate set within one substep.
            particle_collision_detection_interval=max(1, self.solver_iterations // 2),
            particle_topological_contact_filter_threshold=2,
            particle_enable_tile_solve=True,
            rigid_contact_hard=False,
            rigid_body_particle_contact_buffer_size=16384,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=CONTACT_QUERY_RADIUS - CONTACT_RADIUS,
            enable_rigid_soft_full_surface_contact=self.enable_full_surface_contact,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.collision_pipeline.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        q_starts = self.model.joint_q_start.numpy()
        qd_starts = self.model.joint_qd_start.numpy()
        self.left_q_start = int(q_starts[self.left_gear_joint])
        self.right_q_start = int(q_starts[self.right_gear_joint])
        self.left_qd_start = int(qd_starts[self.left_gear_joint])
        self.right_qd_start = int(qd_starts[self.right_gear_joint])
        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(3.2, -6.4, 2.35), pitch=-5.0, yaw=116.0)
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.fov = 43.0

        self.capture()

    def capture(self):
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            wp.launch(
                set_gear_joint_state,
                dim=1,
                inputs=[
                    self.left_q_start,
                    self.left_qd_start,
                    self.right_q_start,
                    self.right_qd_start,
                    self.sim_time_wp,
                    self.gear_speed,
                    GEAR_PHASE,
                ],
                outputs=[self.state_0.joint_q, self.state_0.joint_qd],
                device=self.model.device,
            )
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(advance_time, dim=1, inputs=[self.sim_time_wp, self.sim_dt], device=self.model.device)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        positions = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(positions)), "Armadillo state contains non-finite values"

        extent = np.ptp(positions, axis=0)
        assert np.max(extent) < 4.0, f"Armadillo state diverged: extent={extent}"

        if self.sim_time >= DEFAULT_NUM_FRAMES / FPS:
            max_height = float(np.max(positions[:, 2]))
            assert max_height < -GEAR_TIP_RADIUS, f"Armadillo did not clear the crusher gears: z_max={max_height}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--gear-speed",
            type=float,
            default=GEAR_ANGULAR_SPEED,
            help="Magnitude of each crusher drum's angular velocity [rad/s].",
        )
        parser.add_argument(
            "--quality",
            choices=tuple(QUALITY_PRESETS),
            default="balanced",
            help="Simulation quality: interactive=3x4, balanced=5x6, paper=10x10 substeps x iterations.",
        )
        parser.set_defaults(num_frames=DEFAULT_NUM_FRAMES)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

# SPDX-License-Identifier: Apache-2.0
"""Gear-crusher soft-body demo for Newton's VBD + Planar-DAT solver.

This scene reproduces the setup of Fig. 8 in:
  Chen et al., "Divide and Truncate: A Penetration and Inversion Free
  Framework for Coupled Multi-physics Systems", 2026.

The paper uses a tetrahedral Armadillo between two animated counter-rotating
gear crushers. This standalone demo uses an original, procedurally generated
armadillo-like tetrahedral proxy so the package is redistributable.

Run from this directory:
  python example_gear_crusher.py --viewer gl --device cuda:0 --num-frames 360

It targets Newton v1.3.0 / current main (July 2026).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

ASSET_DIR = Path(__file__).resolve().parent / "assets"

# Paper parameters for Gear Crusher, Table 1.
FPS = 60
SIM_SUBSTEPS = 10               # dt = 1 / 600 s
VBD_ITERATIONS = 10
K_LAMBDA = 1.0e6
K_MU = 1.0e5
CONTACT_KE = 1.0e6
CONTACT_MU = 0.2
PAPER_CONTACT_RADIUS = 0.005    # 5 mm

# Scene scale/tuning for the supplied 4.5 cm voxel tet proxy.
SOFT_SCALE = 0.55
SOFT_DENSITY = 1000.0
SOFT_DAMPING = 2.0e-3
SELF_CONTACT_RADIUS = 0.014
SELF_CONTACT_MARGIN = 0.022
GEAR_CENTER_X = 0.72
GEAR_CENTER_Z = 0.15
GEAR_ANGULAR_SPEED = 1.4        # rad/s; signs are assigned per gear
GEAR_PHASE = math.pi / 16.0     # half-tooth offset for intermeshing


@wp.kernel
def set_gear_joint_state(
    left_q_start: int,
    left_qd_start: int,
    right_q_start: int,
    right_qd_start: int,
    sim_time: wp.array(dtype=wp.float32),
    angular_speed: float,
    phase: float,
    # outputs
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    """Prescribe the two animated gear angles and angular velocities."""
    t = sim_time[0]
    # Looking along -Y, the inner surfaces of both gears move downward.
    joint_q[left_q_start] = phase + angular_speed * t
    joint_qd[left_qd_start] = angular_speed
    joint_q[right_q_start] = -phase - angular_speed * t
    joint_qd[right_qd_start] = -angular_speed


@wp.kernel
def advance_time(sim_time: wp.array(dtype=wp.float32), dt: float):
    sim_time[0] = sim_time[0] + dt


def load_mesh_npz(path: Path) -> newton.Mesh:
    data = np.load(path)
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    triangles = np.asarray(data["triangles"], dtype=np.int32).reshape(-1)
    mesh = newton.Mesh(vertices=vertices, indices=triangles, compute_inertia=False)
    mesh.color = (0.22, 0.25, 0.30)
    mesh.roughness = 0.42
    mesh.metallic = 0.88
    return mesh


def load_soft_tet_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    tets = np.asarray(data["tets"], dtype=np.int32)
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"Expected tetrahedra with shape (N, 4), got {tets.shape}")
    return vertices, tets


def cylinder_inertia(mass: float, radius: float, half_width: float) -> wp.mat33:
    """Approximate gear inertia by a solid cylinder whose axis is Y."""
    height = 2.0 * half_width
    i_axis = 0.5 * mass * radius * radius
    i_transverse = mass * (3.0 * radius * radius + height * height) / 12.0
    return wp.mat33(
        i_transverse, 0.0, 0.0,
        0.0, i_axis, 0.0,
        0.0, 0.0, i_transverse,
    )


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / FPS
        self.sim_substeps = SIM_SUBSTEPS
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.gear_speed = float(args.gear_speed)

        gear_mesh = load_mesh_npz(ASSET_DIR / "crusher_gear.npz")
        soft_vertices, soft_tets = load_soft_tet_npz(ASSET_DIR / "armadillo_proxy_tet.npz")

        builder = newton.ModelBuilder()
        builder.default_particle_radius = SELF_CONTACT_RADIUS
        builder.particle_max_velocity = 35.0

        # Rigid/particle contact material. The paper reports kc=1e6 and muf=0.2.
        gear_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=CONTACT_KE,
            kd=1.0e-4,
            kf=1.0e3,
            mu=CONTACT_MU,
            collision_group=11,
        )
        guide_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=CONTACT_KE,
            kd=1.0e-4,
            kf=1.0e3,
            mu=0.25,
            collision_group=12,
        )

        ground_shape = builder.add_ground_plane(cfg=guide_cfg)

        # Front/back transparent-style guide plates keep the soft body in the
        # crusher's extrusion width. They are collision geometry, not animated.
        guide_shapes = []
        for y in (-0.47, 0.47):
            guide_shapes.append(
                builder.add_shape_box(
                    body=-1,
                    xform=wp.transform(p=wp.vec3(0.0, y, 0.75), q=wp.quat_identity()),
                    hx=1.65,
                    hy=0.025,
                    hz=2.15,
                    cfg=guide_cfg,
                    label=f"guide_plate_{'front' if y < 0 else 'back'}",
                )
            )

        gear_mass = 25.0
        gear_inertia = cylinder_inertia(gear_mass, radius=0.78, half_width=0.34)

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
        self.left_gear_shape = builder.add_shape_mesh(
            self.left_gear_body,
            mesh=gear_mesh,
            cfg=gear_cfg,
            label="left_crusher_gear_mesh",
        )
        self.right_gear_shape = builder.add_shape_mesh(
            self.right_gear_body,
            mesh=gear_mesh,
            cfg=gear_cfg,
            label="right_crusher_gear_mesh",
        )

        self.left_gear_joint = builder.add_joint_revolute(
            parent=-1,
            child=self.left_gear_body,
            axis=newton.Axis.Y,
            parent_xform=wp.transform(
                p=wp.vec3(-GEAR_CENTER_X, 0.0, GEAR_CENTER_Z),
                q=wp.quat_identity(),
            ),
            label="left_crusher_gear_joint",
        )
        self.right_gear_joint = builder.add_joint_revolute(
            parent=-1,
            child=self.right_gear_body,
            axis=newton.Axis.Y,
            parent_xform=wp.transform(
                p=wp.vec3(GEAR_CENTER_X, 0.0, GEAR_CENTER_Z),
                q=wp.quat_identity(),
            ),
            label="right_crusher_gear_joint",
        )
        builder.add_articulation([self.left_gear_joint], label="left_crusher_gear_articulation")
        builder.add_articulation([self.right_gear_joint], label="right_crusher_gear_articulation")

        # The gears intentionally intermesh and are both prescribed kinematic
        # bodies, so exclude gear-gear and gear-environment rigid collisions.
        builder.add_shape_collision_filter_pair(self.left_gear_shape, self.right_gear_shape)
        for gear_shape in (self.left_gear_shape, self.right_gear_shape):
            builder.add_shape_collision_filter_pair(gear_shape, ground_shape)
            for guide_shape in guide_shapes:
                builder.add_shape_collision_filter_pair(gear_shape, guide_shape)

        # Original volumetric proxy, approximately matching the paper's
        # 15k-vertex / 60k-tet Armadillo resolution after scaling.
        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 1.88),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.08),
            scale=SOFT_SCALE,
            vel=wp.vec3(0.0, 0.0, -0.20),
            vertices=soft_vertices,
            indices=soft_tets.reshape(-1).tolist(),
            density=SOFT_DENSITY,
            k_mu=K_MU,
            k_lambda=K_LAMBDA,
            k_damp=SOFT_DAMPING,
        )

        # VBD requires graph coloring for the particle and rigid systems.
        builder.color()
        self.model = builder.finalize()

        self.model.soft_contact_ke = CONTACT_KE
        self.model.soft_contact_kd = 1.0e-4
        self.model.soft_contact_kf = 1.0e3
        self.model.soft_contact_mu = CONTACT_MU

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=VBD_ITERATIONS,
            friction_epsilon=1.0e-3,
            particle_enable_self_contact=True,
            particle_self_contact_radius=SELF_CONTACT_RADIUS,
            particle_self_contact_margin=SELF_CONTACT_MARGIN,
            particle_conservative_bound_relaxation=0.85,
            particle_vertex_contact_buffer_size=96,
            particle_edge_contact_buffer_size=192,
            # The DAT paper refreshed collision queries once every 5 VBD iterations.
            particle_collision_detection_interval=5,
            particle_topological_contact_filter_threshold=2,
            particle_enable_tile_solve=False,
            rigid_contact_hard=False,
            rigid_body_particle_contact_buffer_size=16384,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        # Initialize body poses from the revolute joints.
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        q_starts = self.model.joint_q_start.numpy()
        qd_starts = self.model.joint_qd_start.numpy()
        self.left_q_start = int(q_starts[self.left_gear_joint])
        self.right_q_start = int(q_starts[self.right_gear_joint])
        self.left_qd_start = int(qd_starts[self.left_gear_joint])
        self.right_qd_start = int(qd_starts[self.right_gear_joint])

        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(3.3, -6.8, 2.25),
            pitch=-4.0,
            yaw=116.0,
        )
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.fov = 48.0

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
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

            self.model.collide(self.state_0, self.contacts)
            self.solver.step(
                self.state_0,
                self.state_1,
                self.control,
                self.contacts,
                self.sim_dt,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0

            wp.launch(
                advance_time,
                dim=1,
                inputs=[self.sim_time_wp, self.sim_dt],
                device=self.model.device,
            )

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(particle_q)), "Soft-body state contains non-finite values"
        bbox = np.max(particle_q, axis=0) - np.min(particle_q, axis=0)
        assert np.linalg.norm(bbox) < 8.0, f"Soft body exploded: bbox={bbox}"
        assert np.min(particle_q[:, 2]) > -2.0, "Soft body fell through the receiving ground"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--gear-speed",
            type=float,
            default=GEAR_ANGULAR_SPEED,
            help="Magnitude of each crusher gear's angular velocity [rad/s].",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

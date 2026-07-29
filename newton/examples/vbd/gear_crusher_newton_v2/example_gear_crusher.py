# SPDX-License-Identifier: Apache-2.0
"""Newton reproduction of the DAT paper's Gear Crusher scene (Fig. 8).

Scene:
  * Stanford Armadillo tetrahedral soft body (~15.8k vertices / 60.2k tets)
  * two 18-tooth, counter-rotating kinematic crusher drums
  * Newton SolverVBD particle self-contact with Planar-DAT truncation

Run:
    pip install "newton[examples]==1.3.0"
    python example_gear_crusher.py --viewer gl --device cuda:0 --num-frames 300
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"

# Parameters reported for Gear Crusher in Table 1 of the DAT paper.
FPS = 60
SIM_SUBSTEPS = 10  # 60 fps * 10 substeps -> dt = 1/600 s
VBD_ITERATIONS = 10
K_LAMBDA = 1.0e6
K_MU = 1.0e5
CONTACT_KE = 1.0e6
CONTACT_MU = 0.2
CONTACT_RADIUS = 0.005  # 5 mm

# Scene geometry reconstructed from Fig. 8.
GEAR_CENTER_X = 0.59
GEAR_CENTER_Z = 0.72
GEAR_TIP_RADIUS = 0.62
GEAR_HALF_WIDTH = 0.38
GEAR_TOOTH_COUNT = 18
GEAR_PHASE = math.pi / GEAR_TOOTH_COUNT
GEAR_ANGULAR_SPEED = 1.55  # rad/s
ARMADILLO_START_Z = 1.20

SOFT_DENSITY = 1000.0
SOFT_DAMPING = 2.0e-3


@wp.kernel
def prescribe_crusher_joints(
    left_q_start: int,
    left_qd_start: int,
    right_q_start: int,
    right_qd_start: int,
    sim_time: wp.array(dtype=wp.float32),
    angular_speed: float,
    phase: float,
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
):
    """Drive both kinematic gears so their inner surfaces move downward."""
    t = sim_time[0]
    joint_q[left_q_start] = phase + angular_speed * t
    joint_qd[left_qd_start] = angular_speed
    joint_q[right_q_start] = -phase - angular_speed * t
    joint_qd[right_qd_start] = -angular_speed


@wp.kernel
def advance_time(sim_time: wp.array(dtype=wp.float32), dt: float):
    sim_time[0] += dt


def load_crusher_mesh(path: Path) -> newton.Mesh:
    data = np.load(path)
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    triangles = np.asarray(data["triangles"], dtype=np.int32).reshape(-1)
    return newton.Mesh(
        vertices=vertices,
        indices=triangles,
        compute_inertia=False,
        is_solid=True,
        color=(0.58, 0.61, 0.67),
        roughness=0.38,
        metallic=0.75,
    )


def load_armadillo_tets(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    tets = np.asarray(data["tets"], dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Invalid Armadillo vertices: {vertices.shape}")
    if tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"Invalid Armadillo tets: {tets.shape}")
    return vertices, tets


def cylinder_inertia(mass: float, radius: float, half_width: float) -> wp.mat33:
    height = 2.0 * half_width
    i_axis = 0.5 * mass * radius * radius
    i_transverse = mass * (3.0 * radius * radius + height * height) / 12.0
    return wp.mat33(
        i_transverse,
        0.0,
        0.0,
        0.0,
        i_axis,
        0.0,
        0.0,
        0.0,
        i_transverse,
    )


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / FPS
        self.sim_substeps = int(args.substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.gear_speed = float(args.gear_speed)

        gear_mesh = load_crusher_mesh(ASSET_DIR / "crusher_drum_18t.npz")
        soft_vertices, soft_tets = load_armadillo_tets(ASSET_DIR / "armadillo_stanford_tet.npz")

        builder = newton.ModelBuilder()
        builder.default_particle_radius = CONTACT_RADIUS
        builder.particle_max_velocity = 40.0

        gear_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=CONTACT_KE,
            kd=2.0e-4,
            kf=1.0e3,
            mu=CONTACT_MU,
            collision_group=11,
        )
        ground_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=CONTACT_KE,
            kd=2.0e-4,
            kf=1.0e3,
            mu=CONTACT_MU,
            collision_group=12,
        )
        ground_shape = builder.add_ground_plane(cfg=ground_cfg)

        gear_mass = 30.0
        inertia = cylinder_inertia(gear_mass, GEAR_TIP_RADIUS, GEAR_HALF_WIDTH)
        self.left_body = builder.add_link(
            mass=gear_mass,
            inertia=inertia,
            is_kinematic=True,
            label="left_crusher_drum",
        )
        self.right_body = builder.add_link(
            mass=gear_mass,
            inertia=inertia,
            is_kinematic=True,
            label="right_crusher_drum",
        )
        left_shape = builder.add_shape_mesh(
            self.left_body,
            mesh=gear_mesh,
            cfg=gear_cfg,
            label="left_crusher_drum_mesh",
        )
        right_shape = builder.add_shape_mesh(
            self.right_body,
            mesh=gear_mesh,
            cfg=gear_cfg,
            label="right_crusher_drum_mesh",
        )

        self.left_joint = builder.add_joint_revolute(
            parent=-1,
            child=self.left_body,
            axis=newton.Axis.Y,
            parent_xform=wp.transform(
                p=wp.vec3(-GEAR_CENTER_X, 0.0, GEAR_CENTER_Z),
                q=wp.quat_identity(),
            ),
            label="left_crusher_joint",
        )
        self.right_joint = builder.add_joint_revolute(
            parent=-1,
            child=self.right_body,
            axis=newton.Axis.Y,
            parent_xform=wp.transform(
                p=wp.vec3(GEAR_CENTER_X, 0.0, GEAR_CENTER_Z),
                q=wp.quat_identity(),
            ),
            label="right_crusher_joint",
        )
        builder.add_articulation([self.left_joint], label="left_crusher_articulation")
        builder.add_articulation([self.right_joint], label="right_crusher_articulation")

        # The two crusher drums are prescribed kinematic meshes and intentionally
        # overlap slightly at tooth tips.  They contact particles, not one another.
        builder.add_shape_collision_filter_pair(left_shape, right_shape)
        builder.add_shape_collision_filter_pair(left_shape, ground_shape)
        builder.add_shape_collision_filter_pair(right_shape, ground_shape)

        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, ARMADILLO_START_Z),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, -0.08),
            vertices=soft_vertices,
            indices=soft_tets.reshape(-1).tolist(),
            density=SOFT_DENSITY,
            k_mu=K_MU,
            k_lambda=K_LAMBDA,
            k_damp=SOFT_DAMPING,
            particle_radius=CONTACT_RADIUS,
            validate_mesh=False,
            label="stanford_armadillo_soft_body",
        )

        # VBD is a colored Gauss-Seidel solver.  This creates tet/particle color
        # groups once during setup; all frame simulation remains on the GPU.
        builder.color(balance_colors=True)
        self.model = builder.finalize()
        self.model.soft_contact_ke = CONTACT_KE
        self.model.soft_contact_kd = 2.0e-4
        self.model.soft_contact_kf = 1.0e3
        self.model.soft_contact_mu = CONTACT_MU

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=VBD_ITERATIONS,
            friction_epsilon=1.0e-3,
            particle_enable_self_contact=True,
            particle_self_contact_radius=CONTACT_RADIUS,
            particle_self_contact_margin=1.5 * CONTACT_RADIUS,
            particle_conservative_bound_relaxation=0.85,
            particle_vertex_contact_buffer_size=128,
            particle_edge_contact_buffer_size=256,
            particle_collision_detection_interval=5,
            particle_topological_contact_filter_threshold=2,
            particle_enable_tile_solve=False,
            rigid_contact_hard=False,
            rigid_body_particle_contact_buffer_size=32768,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        self.left_q_start = int(q_start[self.left_joint])
        self.right_q_start = int(q_start[self.right_joint])
        self.left_qd_start = int(qd_start[self.left_joint])
        self.right_qd_start = int(qd_start[self.right_joint])
        self.sim_time_wp = wp.zeros(1, dtype=wp.float32, device=self.model.device)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.7, -6.2, 2.15), pitch=-4.0, yaw=112.0)
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.fov = 46.0

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
                prescribe_crusher_joints,
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
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(advance_time, dim=1, inputs=[self.sim_time_wp, self.sim_dt], device=self.model.device)

    def step(self):
        if self.graph is not None:
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
        assert np.all(np.isfinite(particle_q)), "Soft-body state contains non-finite positions"
        extent = np.ptp(particle_q, axis=0)
        assert float(np.linalg.norm(extent)) < 8.0, f"Soft body exploded: extent={extent}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--gear-speed",
            type=float,
            default=GEAR_ANGULAR_SPEED,
            help="Magnitude of both counter-rotating gear velocities [rad/s].",
        )
        parser.add_argument(
            "--substeps",
            type=int,
            default=SIM_SUBSTEPS,
            help="Simulation substeps per 60 Hz frame; 10 reproduces dt=1/600 s.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MuJoCo-VBD Cloth Franka MJCF Teleop
#
# Recreates the Genesis IPC Franka cloth teleoperation scene using Newton's
# MuJoCo + VBD proxy coupling path. The Franka arm is imported directly from
# the Genesis MJCF asset, two square cloth sheets are generated procedurally,
# and a 4x4 grid of static cubes supports the cloth for grasping.
#
# Controls:
#   Arrow keys: translate target in XY
#   J / K: move target down / up
#   N / M: yaw left / right
#   U / O: pitch up / down
#   Y / H: roll left / right
#   Space: close gripper while held
#   R: reset scene
#
# Command: python -m newton.examples mujoco_vbd_cloth_franka_mjcf_teleop
#
###########################################################################

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverMuJoCo, SolverVBD
from newton.solvers.experimental.coupled import SolverCoupledProxy


DEFAULT_FRANKA_MJCF_PATH = Path(
    r"C:\csy_work\CG\Engine\genesis-world\genesis\assets\xml\franka_emika_panda\panda_non_overlap.xml"
)

FRANKA_HOME_Q = np.array(
    [2.2116, -1.5328, -0.7347, -1.7235, -1.3377, 0.7519, -1.4410, 0.04, 0.04],
    dtype=np.float32,
)

DEBUG_CUBE_SIZE = 0.01
DEBUG_CUBE_COLOR = wp.vec3(1.0, 0.2, 0.2)

DELTA_POS = 0.003
DELTA_ROT = 0.02

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.002
GRIPPER_CONTACT_MU = 10.0

DEFAULT_PICK_STIFFNESS = 150.0
DEFAULT_PICK_DAMPING = 25.0
DEFAULT_PICK_MAX_ACCELERATION = 30.0


@wp.kernel
def broadcast_ik_solution_kernel(
    ik_solution: wp.array2d[wp.float32],
    joint_targets: wp.array[wp.float32],
    gripper_value: float,
):
    joint_targets[0] = ik_solution[0, 0]
    joint_targets[1] = ik_solution[0, 1]
    joint_targets[2] = ik_solution[0, 2]
    joint_targets[3] = ik_solution[0, 3]
    joint_targets[4] = ik_solution[0, 4]
    joint_targets[5] = ik_solution[0, 5]
    joint_targets[6] = ik_solution[0, 6]
    joint_targets[7] = gripper_value
    joint_targets[8] = gripper_value


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1.0e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    axis = axis / axis_norm
    half = 0.5 * angle
    sin_half = np.sin(half)
    return np.array([axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, np.cos(half)], dtype=np.float32)


def quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    x0, y0, z0, w0 = lhs
    x1, y1, z1, w1 = rhs
    return np.array(
        [
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        ],
        dtype=np.float32,
    )


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm < 1.0e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quat / norm


def build_square_cloth_mesh(size_x: float, size_y: float, dim_x: int, dim_y: int) -> tuple[list[wp.vec3], list[int]]:
    vertices: list[wp.vec3] = []
    indices: list[int] = []

    for iy in range(dim_y):
        v = iy / float(dim_y - 1)
        y = (v - 0.5) * size_y
        for ix in range(dim_x):
            u = ix / float(dim_x - 1)
            x = (u - 0.5) * size_x
            vertices.append(wp.vec3(x, y, 0.0))

    for iy in range(dim_y - 1):
        for ix in range(dim_x - 1):
            i0 = iy * dim_x + ix
            i1 = i0 + 1
            i2 = i0 + dim_x
            i3 = i2 + 1
            indices.extend((i0, i2, i1, i1, i2, i3))

    return vertices, indices


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = int(args.substeps)
        self.sim_dt = self.frame_dt / float(self.sim_substeps)
        self.sim_time = 0.0

        self.vbd_iterations = int(args.vbd_iterations)
        self.proxy_iterations = int(args.proxy_iterations)

        self.cloth_particle_radius = 0.008
        self.cloth_body_contact_margin = 0.012
        self.particle_self_contact_radius = 0.002
        self.particle_self_contact_margin = 0.002

        self.soft_contact_ke = 1.0e4
        self.soft_contact_kd = 1.0e-2
        self.soft_contact_mu = 0.25

        self.robot_contact_ke = 5.0e4
        self.robot_contact_kd = 1.0e-3
        self.robot_contact_mu = 1.5
        self.static_contact_ke = 1.0e4
        self.static_contact_kd = 1.0e-2
        self.static_contact_mu = 0.5
        self.gripper_contact_mu = GRIPPER_CONTACT_MU

        builder = newton.ModelBuilder(gravity=-9.81)
        builder.add_ground_plane()

        self.robot_body_start = builder.body_count
        self._add_franka(builder)
        self.robot_body_end = builder.body_count
        self.robot_body_indices = list(range(self.robot_body_start, self.robot_body_end))
        self.proxy_body_indices = [self.ee_body, self.left_finger_body, self.right_finger_body]

        self._add_support_cubes(builder)

        self.cloth_particle_start = builder.particle_count
        self._add_cloth_sheets(builder)
        self.cloth_particle_end = builder.particle_count
        self.cloth_particle_indices = list(range(self.cloth_particle_start, self.cloth_particle_end))

        builder.color(include_bending=True)

        self.model = builder.finalize(requires_grad=False)
        self.model.edge_rest_angle.zero_()
        self._configure_materials()

        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="robot",
                    solver=lambda model_view: SolverMuJoCo(
                        model=model_view,
                        use_mujoco_contacts=False,
                        njmax=512,
                        nconmax=512,
                    ),
                    bodies=self.robot_body_indices,
                    joints=list(range(self.model.joint_count)),
                ),
                SolverCoupledProxy.Entry(
                    name="vbd",
                    solver=lambda model_view: SolverVBD(
                        model=model_view,
                        iterations=self.vbd_iterations,
                        friction_epsilon=1.0e-3,
                        particle_enable_self_contact=True,
                        particle_self_contact_radius=self.particle_self_contact_radius,
                        particle_self_contact_margin=self.particle_self_contact_margin,
                        particle_topological_contact_filter_threshold=1,
                        particle_rest_shape_contact_exclusion_radius=0.005,
                        particle_vertex_contact_buffer_size=24,
                        particle_edge_contact_buffer_size=24,
                        particle_collision_detection_interval=-1,
                    ),
                    particles=self.cloth_particle_indices,
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="robot",
                        destination="vbd",
                        bodies=self.proxy_body_indices,
                        mass_scale=args.mass_scale,
                        mode=args.coupling_mode,
                        collision_pipeline=lambda model: newton.examples.create_collision_pipeline(
                            model,
                            args,
                            soft_contact_margin=self.cloth_body_contact_margin,
                        ),
                        collide_interval=1,
                    )
                ],
                iterations=self.proxy_iterations,
            ),
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        wp.copy(self.control.joint_target_pos[:9], self.model.joint_q[:9])

        self._setup_ik()
        self._capture_initial_state()
        self._configure_picking()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(1.9, -0.9, 1.35), 148.0, -24.0)

    def _find_label_index(self, labels: list[str], name: str) -> int:
        for index, label in enumerate(labels):
            short = label.rsplit("/", 1)[-1] if "/" in label else label
            if short == name:
                return index
        raise ValueError(f"Could not find body label {name!r}")

    def _resolve_mjcf_path(self) -> Path:
        mjcf_path = Path(self.args.mjcf_path)
        if mjcf_path.exists():
            return mjcf_path
        raise FileNotFoundError(
            f"Franka MJCF not found at {mjcf_path}. Pass --mjcf-path to a valid panda_non_overlap.xml asset."
        )

    def _add_franka(self, builder: newton.ModelBuilder) -> None:
        builder.add_mjcf(
            self._resolve_mjcf_path(),
            xform=wp.transform((0.0, 0.0, 0.005), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
            parse_sites=False,
        )

        builder.joint_q[:9] = FRANKA_HOME_Q.tolist()
        builder.joint_target_pos[:9] = FRANKA_HOME_Q.tolist()
        builder.joint_target_ke[:7] = [4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0]
        builder.joint_target_ke[7:9] = [2500.0, 2500.0]
        builder.joint_target_kd[:7] = [450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0]
        builder.joint_target_kd[7:9] = [100.0, 100.0]

        self.ee_body = self._find_label_index(builder.body_label, "hand")
        self.left_finger_body = self._find_label_index(builder.body_label, "left_finger")
        self.right_finger_body = self._find_label_index(builder.body_label, "right_finger")
        self.ee_offset = wp.vec3(0.0, 0.0, 0.0)

    def _add_support_cubes(self, builder: newton.ModelBuilder) -> None:
        cube_half = 0.025
        cube_height = 0.02501
        grid_spacing = 0.15
        for i in range(4):
            for j in range(4):
                x = (i + 1.7) * grid_spacing
                y = (j - 1.5) * grid_spacing
                builder.add_shape_box(
                    -1,
                    wp.transform((x, y, cube_height), wp.quat_identity()),
                    hx=cube_half,
                    hy=cube_half,
                    hz=cube_half,
                )

    def _add_cloth_sheet(
        self,
        builder: newton.ModelBuilder,
        center: tuple[float, float, float],
        size: float,
        resolution: int,
        density: float,
        edge_ke: float,
    ) -> None:
        vertices, indices = build_square_cloth_mesh(size, size, resolution, resolution)
        builder.add_cloth_mesh(
            pos=wp.vec3(*center),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=vertices,
            indices=indices,
            density=density,
            tri_ke=6.0e3,
            tri_ka=6.0e3,
            tri_kd=1.5e-6,
            edge_ke=edge_ke,
            edge_kd=1.0e-2,
            particle_radius=self.cloth_particle_radius,
        )

    def _add_cloth_sheets(self, builder: newton.ModelBuilder) -> None:
        self._add_cloth_sheet(builder, center=(0.5, 0.0, 0.1), size=0.5, resolution=21, density=0.02, edge_ke=2.5)
        self._add_cloth_sheet(builder, center=(0.5, 0.0, 0.14), size=0.3, resolution=17, density=0.025, edge_ke=8.0)

    def _configure_picking(self) -> None:
        if not hasattr(self.viewer, "picking"):
            return

        pick_state = self.viewer.picking.pick_state.numpy()
        pick_state[0]["pick_stiffness"] = DEFAULT_PICK_STIFFNESS
        pick_state[0]["pick_damping"] = DEFAULT_PICK_DAMPING
        pick_state[0]["pick_max_acceleration"] = DEFAULT_PICK_MAX_ACCELERATION
        self.viewer.picking.pick_state.assign(pick_state)

    def _configure_materials(self) -> None:
        self.model.soft_contact_ke = self.soft_contact_ke
        self.model.soft_contact_kd = self.soft_contact_kd
        self.model.soft_contact_mu = self.soft_contact_mu

        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu = self.model.shape_material_mu.numpy()
        shape_body = self.model.shape_body.numpy()

        shape_ke[...] = self.static_contact_ke
        shape_kd[...] = self.static_contact_kd
        shape_mu[...] = self.static_contact_mu

        for shape_index, body_index in enumerate(shape_body):
            if body_index in self.robot_body_indices:
                shape_ke[shape_index] = self.robot_contact_ke
                shape_kd[shape_index] = self.robot_contact_kd
                shape_mu[shape_index] = self.robot_contact_mu
            if body_index in (self.left_finger_body, self.right_finger_body):
                shape_mu[shape_index] = self.gripper_contact_mu

        self.model.shape_material_ke = wp.array(
            shape_ke,
            dtype=self.model.shape_material_ke.dtype,
            device=self.model.device,
        )
        self.model.shape_material_kd = wp.array(
            shape_kd,
            dtype=self.model.shape_material_kd.dtype,
            device=self.model.device,
        )
        self.model.shape_material_mu = wp.array(
            shape_mu,
            dtype=self.model.shape_material_mu.dtype,
            device=self.model.device,
        )

    def _setup_ik(self) -> None:
        body_q = self.state_0.body_q.numpy()
        ee_tf = body_q[self.ee_body]

        self.target_position = np.array(ee_tf[:3], dtype=np.float32)
        self.target_rotation = quat_normalize(np.array(ee_tf[3:7], dtype=np.float32))
        self.target_position_home = self.target_position.copy()
        self.target_rotation_home = self.target_rotation.copy()

        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
        self.pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_body,
            link_offset=self.ee_offset,
            target_positions=wp.array([wp.vec3(*self.target_position.tolist())], dtype=wp.vec3),
        )
        self.rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*self.target_rotation.tolist())], dtype=wp.vec4),
        )
        self.joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.rot_obj, self.joint_limits_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24

        self.gripper_target = GRIPPER_OPEN
        self.reset_key_prev = False

    def _capture_initial_state(self) -> None:
        self.initial_model_joint_q = wp.clone(self.model.joint_q)
        self.initial_model_joint_qd = wp.clone(self.model.joint_qd)
        self.initial_control_joint_target_pos = wp.clone(self.control.joint_target_pos)
        self.initial_ik_joint_q = wp.clone(self.ik_joint_q)
        self.initial_body_q = wp.clone(self.state_0.body_q)
        self.initial_body_qd = wp.clone(self.state_0.body_qd)
        self.initial_particle_q = wp.clone(self.state_0.particle_q)
        self.initial_particle_qd = wp.clone(self.state_0.particle_qd)

    def _apply_keyboard_input(self) -> None:
        if not hasattr(self.viewer, "is_key_down"):
            return

        if self.viewer.is_key_down("up"):
            self.target_position[0] -= DELTA_POS
        if self.viewer.is_key_down("down"):
            self.target_position[0] += DELTA_POS
        if self.viewer.is_key_down("left"):
            self.target_position[1] -= DELTA_POS
        if self.viewer.is_key_down("right"):
            self.target_position[1] += DELTA_POS
        if self.viewer.is_key_down("j"):
            self.target_position[2] -= DELTA_POS
        if self.viewer.is_key_down("k"):
            self.target_position[2] += DELTA_POS

        if self.viewer.is_key_down("n"):
            self.target_rotation = quat_normalize(
                quat_multiply(self.target_rotation, quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float32), DELTA_ROT))
            )
        if self.viewer.is_key_down("m"):
            self.target_rotation = quat_normalize(
                quat_multiply(self.target_rotation, quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float32), -DELTA_ROT))
            )
        if self.viewer.is_key_down("u"):
            self.target_rotation = quat_normalize(
                quat_multiply(self.target_rotation, quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float32), DELTA_ROT))
            )
        if self.viewer.is_key_down("o"):
            self.target_rotation = quat_normalize(
                quat_multiply(self.target_rotation, quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float32), -DELTA_ROT))
            )
        if self.viewer.is_key_down("y"):
            self.target_rotation = quat_normalize(
                quat_multiply(self.target_rotation, quat_from_axis_angle(np.array([1.0, 0.0, 0.0], dtype=np.float32), DELTA_ROT))
            )
        if self.viewer.is_key_down("h"):
            self.target_rotation = quat_normalize(
                quat_multiply(self.target_rotation, quat_from_axis_angle(np.array([1.0, 0.0, 0.0], dtype=np.float32), -DELTA_ROT))
            )

        self.gripper_target = GRIPPER_CLOSED if self.viewer.is_key_down("space") else GRIPPER_OPEN

        reset_down = bool(self.viewer.is_key_down("r"))
        if reset_down and not self.reset_key_prev:
            self.reset_scene()
        self.reset_key_prev = reset_down

    def reset_scene(self) -> None:
        wp.copy(self.model.joint_q, self.initial_model_joint_q)
        wp.copy(self.model.joint_qd, self.initial_model_joint_qd)
        wp.copy(self.control.joint_target_pos, self.initial_control_joint_target_pos)
        wp.copy(self.ik_joint_q, self.initial_ik_joint_q)

        wp.copy(self.state_0.body_q, self.initial_body_q)
        wp.copy(self.state_0.body_qd, self.initial_body_qd)
        wp.copy(self.state_1.body_q, self.initial_body_q)
        wp.copy(self.state_1.body_qd, self.initial_body_qd)

        wp.copy(self.state_0.particle_q, self.initial_particle_q)
        wp.copy(self.state_0.particle_qd, self.initial_particle_qd)
        wp.copy(self.state_1.particle_q, self.initial_particle_q)
        wp.copy(self.state_1.particle_qd, self.initial_particle_qd)

        self.target_position[:] = self.target_position_home
        self.target_rotation[:] = self.target_rotation_home
        self.gripper_target = GRIPPER_OPEN
        self.sim_time = 0.0

    def set_joint_targets(self) -> None:
        self.pos_obj.set_target_position(0, wp.vec3(*self.target_position.tolist()))
        self.rot_obj.set_target_rotation(0, wp.vec4(*self.target_rotation.tolist()))
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            broadcast_ik_solution_kernel,
            dim=1,
            inputs=[self.ik_joint_q, self.control.joint_target_pos, self.gripper_target],
            device=self.model.device,
        )

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        self._apply_keyboard_input()
        self.set_joint_targets()
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        proxy_contacts = self.solver.get_proxy_contacts("robot", "vbd")
        if proxy_contacts is not None:
            self.viewer.log_contacts(proxy_contacts, self.state_0)

        self.viewer.log_shapes(
            "/debug_cube",
            newton.GeoType.BOX,
            (DEBUG_CUBE_SIZE, DEBUG_CUBE_SIZE, DEBUG_CUBE_SIZE),
            wp.array(
                [
                    wp.transform(
                        wp.vec3(*self.target_position.tolist()),
                        wp.quat(*self.target_rotation.tolist()),
                    )
                ],
                dtype=wp.transform,
            ),
            wp.array([DEBUG_CUBE_COLOR], dtype=wp.vec3),
        )
        self.viewer.end_frame()

    def test_final(self) -> None:
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        assert np.isfinite(particle_q).all(), "Particle positions contain NaN or inf values"
        assert np.isfinite(body_q).all(), "Body transforms contain NaN or inf values"

        cloth_q = particle_q[self.cloth_particle_start : self.cloth_particle_end]
        min_pos = np.min(cloth_q, axis=0)
        max_pos = np.max(cloth_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)
        assert bbox_size < 2.5, f"Cloth bounding box exploded: size={bbox_size:.3f}"
        assert min_pos[2] > -0.05, f"Cloth penetrated too far below ground: z_min={min_pos[2]:.4f}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--mjcf-path",
            type=str,
            default=str(DEFAULT_FRANKA_MJCF_PATH),
            help="Path to Genesis panda_non_overlap.xml",
        )
        parser.add_argument(
            "--coupling-mode",
            help="Proxy body state transfer mode",
            type=str,
            choices=["lagged", "staggered"],
            default="lagged",
        )
        parser.add_argument(
            "--proxy-iterations",
            help="Number of proxy relaxation passes per substep",
            type=int,
            default=2,
        )
        parser.add_argument(
            "--vbd-iterations",
            help="Number of VBD iterations per substep",
            type=int,
            default=8,
        )
        parser.add_argument(
            "--substeps",
            help="Simulation substeps per rendered frame",
            type=int,
            default=10,
        )
        parser.add_argument(
            "--mass-scale",
            help="Scale factor for the hand proxy mass/inertia on the VBD side",
            type=float,
            default=0.75,
        )
        parser.set_defaults(num_frames=240)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
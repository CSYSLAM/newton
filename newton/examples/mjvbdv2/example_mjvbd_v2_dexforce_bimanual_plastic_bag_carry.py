# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Support a loaded supermarket bag with two kinematic Dexforce W1 hands.

The example starts both standalone hands from a pose saved by
``example_mjvbd_v2_dexforce_bimanual_plastic_bag_pose_recorder.py``. There is
no support rod and no pinned bag particle. Physical hand-to-cloth contact
supports the original bag mesh while four rigid balls load its bottom.

Run from the repository root::

    uv run --extra examples -m newton.examples \
        mjvbd_v2_dexforce_bimanual_plastic_bag_carry --viewer gl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets"
BAG_MESH_PATH = (
    ASSET_ROOT / "supermarket_plastic_bag_open_carry_v5_asset" / "supermarket_plastic_bag_open_carry_v5_tri.obj"
)
LEFT_HAND_URDF = ASSET_ROOT / "W1_left_hand" / "DexforceW1_left_hand.urdf"
RIGHT_HAND_URDF = ASSET_ROOT / "W1_right_hand" / "DexforceW1_right_hand.urdf"
HAND_POSE_PATH = ASSET_ROOT / "vbd_mjvbd_v2" / "vbd_w1_bimanual_plastic_bag_pose.json"

FPS = 60
SIM_SUBSTEPS = 6
VBD_ITERATIONS = 12

BAG_POSITION = np.array((0.0, 0.0, 0.30), dtype=np.float32)
BAG_AREAL_DENSITY = 0.02
BAG_PARTICLE_RADIUS = 0.0015
BAG_TRI_KE = 3.0e6
BAG_TRI_KA = 3.0e6
BAG_TRI_KD = 0.5
BAG_EDGE_KE = 100.0
BAG_EDGE_KD = 3.0
AIR_DRAG_RATE = 1.0  # [1/s]

HAND_CONTACT_MARGIN = 0.006
HAND_CONTACT_KE = 1.0e7
HAND_CONTACT_KD = 500.0
HAND_CONTACT_MU = 2.0

BALL_RADIUS = 0.040
BALL_DENSITY = 30000000.0
BALL_INITIAL_DOWNWARD_SPEED = 2.0
BALL_CONTACT_MARGIN = 0.005
BALL_CONTACT_KE = 1.0e7
BALL_CONTACT_KD = 500.0
BALL_FRICTION = 0.3
BALL_LOCAL_POSITIONS = (
    (-0.080, 0.0, 0.22),
    (0.080, 0.0, 0.22),
    (-0.040, 0.0, 0.32),
    (0.040, 0.0, 0.32),
)
BALL_COLORS = (
    (0.92, 0.18, 0.15),
    (0.10, 0.48, 0.92),
    (0.20, 0.75, 0.30),
    (0.95, 0.65, 0.10),
)

SOFT_CONTACT_KE = 1.0e7
SOFT_CONTACT_KD = 500.0
SOFT_CONTACT_FRICTION = 0.06
SOFT_CONTACT_MARGIN = 0.010
SOFT_CONTACT_MAX = 32768
SELF_CONTACT_MARGIN = 0.003
BAG_COLOR = (0.88, 0.035, 0.025)
BAG_OPACITY = 0.48


def _load_hand_pose(path: Path):
    """Load and validate the two saved standalone-hand poses."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    hands = payload.get("hands")
    if not isinstance(hands, dict):
        raise ValueError(f"Hand pose file does not contain a hands mapping: {path}")

    result = {}
    for side in ("LEFT", "RIGHT"):
        hand = hands.get(side)
        if not isinstance(hand, dict):
            raise ValueError(f"Hand pose file does not contain {side}: {path}")
        root = hand.get("target_root_world")
        joints = hand.get("finger_joints_degrees")
        if not isinstance(root, dict) or not isinstance(joints, dict):
            raise ValueError(f"Saved {side} pose is incomplete: {path}")
        position = np.asarray(root.get("position_m"), dtype=np.float32)
        rotation = np.asarray(root.get("quaternion_xyzw"), dtype=np.float32)
        if position.shape != (3,) or rotation.shape != (4,):
            raise ValueError(f"Saved {side} root pose has an invalid shape: {path}")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
            raise ValueError(f"Saved {side} root pose is not finite: {path}")
        norm = float(np.linalg.norm(rotation))
        if norm < 1.0e-8:
            raise ValueError(f"Saved {side} root quaternion has zero length: {path}")
        result[side] = {
            "transform": wp.transform(wp.vec3(*position), wp.quat(*(rotation / norm))),
            "joint_degrees": {str(name): float(value) for name, value in joints.items()},
        }
    return result


@wp.kernel
def _apply_particle_drag(
    particle_qd: wp.array[wp.vec3],
    particle_mass: wp.array[float],
    drag_rate: float,
    particle_f: wp.array[wp.vec3],
):
    """Apply mass-proportional air drag to all particles."""
    particle = wp.tid()
    particle_f[particle] = particle_f[particle] - drag_rate * particle_mass[particle] * particle_qd[particle]


class Example:
    """Carry the original bag mesh with two fixed-pose physical hands."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / SIM_SUBSTEPS
        self.sim_time = 0.0

        bag_mesh_path = Path(args.bag_mesh).expanduser().resolve()
        left_hand_urdf = Path(args.left_hand_urdf).expanduser().resolve()
        right_hand_urdf = Path(args.right_hand_urdf).expanduser().resolve()
        hand_pose_path = Path(args.hand_pose).expanduser().resolve()
        for description, path in (
            ("Plastic bag mesh", bag_mesh_path),
            ("Left-hand URDF", left_hand_urdf),
            ("Right-hand URDF", right_hand_urdf),
            ("Bimanual hand pose", hand_pose_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{description} not found: {path}")
        hand_poses = _load_hand_pose(hand_pose_path)

        bag_mesh = newton.Mesh.create_from_file(str(bag_mesh_path), compute_inertia=False, is_solid=False)
        vertices = np.asarray(bag_mesh.vertices, dtype=np.float32)
        indices = np.asarray(bag_mesh.indices, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Expected bag vertices with shape (n, 3), got {vertices.shape}")
        if indices.ndim != 1 or indices.size % 3 != 0:
            raise ValueError(f"Expected a flat triangle index buffer, got {indices.shape}")

        max_abs_x = float(np.abs(vertices[:, 0]).max())
        max_z = float(vertices[:, 2].max())
        handle_local = np.flatnonzero((np.abs(vertices[:, 0]) > 0.7 * max_abs_x) & (vertices[:, 2] > 0.65 * max_z))
        self.left_handle_indices = handle_local[vertices[handle_local, 0] < 0.0]
        self.right_handle_indices = handle_local[vertices[handle_local, 0] > 0.0]
        if self.left_handle_indices.size == 0 or self.right_handle_indices.size == 0:
            raise ValueError("The bag mesh must contain both left and right handles")

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        newton.solvers.SolverMJVBDV2.register_custom_attributes(builder)
        builder.add_cloth_mesh(
            pos=wp.vec3(*BAG_POSITION),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=vertices,
            indices=indices,
            density=BAG_AREAL_DENSITY,
            tri_ke=BAG_TRI_KE,
            tri_ka=BAG_TRI_KA,
            tri_kd=BAG_TRI_KD,
            edge_ke=BAG_EDGE_KE,
            edge_kd=BAG_EDGE_KD,
            particle_radius=BAG_PARTICLE_RADIUS,
        )

        builder.default_shape_cfg.ke = HAND_CONTACT_KE
        builder.default_shape_cfg.kd = HAND_CONTACT_KD
        builder.default_shape_cfg.mu = HAND_CONTACT_MU
        builder.default_shape_cfg.margin = HAND_CONTACT_MARGIN
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        self.hand_articulations: list[int] = []
        self.hand_particle_shapes: list[int] = []
        for side, hand_path in (("LEFT", left_hand_urdf), ("RIGHT", right_hand_urdf)):
            articulation_start = builder.articulation_count
            body_start = builder.body_count
            shape_start = builder.shape_count
            builder.add_urdf(
                str(hand_path),
                xform=hand_poses[side]["transform"],
                floating=True,
                enable_self_collisions=False,
                collapse_fixed_joints=True,
                parse_visuals_as_colliders=False,
                force_show_colliders=False,
            )
            self.hand_articulations.extend(range(articulation_start, builder.articulation_count))
            for body in range(body_start, builder.body_count):
                builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
            for shape in range(shape_start, builder.shape_count):
                if builder.shape_flags[shape] & collide_particles:
                    builder.shape_flags[shape] &= ~collide_shapes
                    self.hand_particle_shapes.append(shape)
        if len(self.hand_articulations) != 2:
            raise RuntimeError("Expected one articulation from each standalone hand URDF")
        if not self.hand_particle_shapes:
            raise RuntimeError("The hand URDFs did not produce particle-collision shapes")

        ball_cfg = newton.ModelBuilder.ShapeConfig(
            density=BALL_DENSITY,
            ke=BALL_CONTACT_KE,
            kd=BALL_CONTACT_KD,
            mu=BALL_FRICTION,
            margin=BALL_CONTACT_MARGIN,
        )
        self.ball_body_indices = []
        for ball_index, (local_position, color) in enumerate(zip(BALL_LOCAL_POSITIONS, BALL_COLORS, strict=True)):
            position = BAG_POSITION + np.asarray(local_position, dtype=np.float32)
            body = builder.add_body(
                xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
                label=f"bag ball {ball_index}",
            )
            builder.body_qd[body] = wp.spatial_vector(0.0, 0.0, -BALL_INITIAL_DOWNWARD_SPEED, 0.0, 0.0, 0.0)
            builder.add_shape_sphere(
                body,
                radius=BALL_RADIUS,
                cfg=ball_cfg,
                color=color,
                label=f"bag ball {ball_index} shape",
            )
            self.ball_body_indices.append(body)
        self.ball_body_indices = np.asarray(self.ball_body_indices, dtype=np.int32)

        builder.add_ground_plane()
        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = SOFT_CONTACT_KE
        self.model.soft_contact_kd = SOFT_CONTACT_KD
        self.model.soft_contact_mu = SOFT_CONTACT_FRICTION

        joint_q = self.model.joint_q.numpy()
        joint_types = self.model.joint_type.numpy()
        joint_parents = self.model.joint_parent.numpy()
        joint_children = self.model.joint_child.numpy()
        joint_q_starts = self.model.joint_q_start.numpy()
        for side in ("LEFT", "RIGHT"):
            base_suffix = f"/{side.lower()}_hand_base"
            root_joint = next(
                joint
                for joint, (joint_type, parent, child) in enumerate(
                    zip(joint_types, joint_parents, joint_children, strict=True)
                )
                if int(joint_type) == int(newton.JointType.FREE)
                and int(parent) == -1
                and self.model.body_label[int(child)].endswith(base_suffix)
            )
            root = int(joint_q_starts[root_joint])
            transform = hand_poses[side]["transform"]
            position = wp.transform_get_translation(transform)
            rotation = wp.transform_get_rotation(transform)
            joint_q[root : root + 7] = (*position, *rotation)
            for suffix, degrees in hand_poses[side]["joint_degrees"].items():
                label = f"{side}_{suffix}"
                joint = next(index for index, name in enumerate(self.model.joint_label) if name.endswith("/" + label))
                joint_q[int(joint_q_starts[joint])] = np.radians(degrees)
        self.model.joint_q.assign(joint_q)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.grip_body_indices = np.asarray(
            (
                next(index for index, label in enumerate(self.model.body_label) if label.endswith("/left_middle_dist")),
                next(
                    index for index, label in enumerate(self.model.body_label) if label.endswith("/right_middle_dist")
                ),
            ),
            dtype=np.int32,
        )
        self.solver = newton.solvers.SolverMJVBDV2(
            self.model,
            mujoco_articulations=tuple(self.hand_articulations),
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": VBD_ITERATIONS,
                "friction_epsilon": 1.0e-4,
                "rigid_body_contact_buffer_size": 2048,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": BAG_PARTICLE_RADIUS,
                "particle_self_contact_margin": SELF_CONTACT_MARGIN,
                "particle_vertex_contact_buffer_size": 48,
                "particle_edge_contact_buffer_size": 96,
                "particle_collision_detection_interval": -1,
                "particle_topological_contact_filter_threshold": 2,
                "particle_rest_shape_contact_exclusion_radius": SELF_CONTACT_MARGIN,
                "rigid_body_particle_contact_buffer_size": 4096,
            },
            collision_options={
                "broad_phase": "nxn",
                "soft_contact_margin": SOFT_CONTACT_MARGIN,
                "soft_contact_max": SOFT_CONTACT_MAX,
                "enable_rigid_soft_full_surface_contact": True,
                "include_static_kinematic_pairs": False,
            },
        )
        if self.solver.features.backend != "vbd_kinematic_full":
            raise RuntimeError(
                f"The bimanual carry scene requires vbd_kinematic_full, got {self.solver.features.backend}"
            )

        self.render_indices = self.model.tri_indices.flatten()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = False
        self.viewer.set_camera(pos=wp.vec3(0.72, -1.05, 0.66), pitch=-7.0, yaw=124.0)

        self.use_graph = bool(args.graph_capture) and self.model.device.is_cuda
        self.capture()

    def capture(self):
        """Capture one complete fixed-hand simulation frame on CUDA."""
        self.graph = None
        if not self.use_graph:
            return
        state_0_backup = self.model.state()
        state_1_backup = self.model.state()
        state_0_backup.assign(self.state_0)
        state_1_backup.assign(self.state_1)
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph
        self.state_0.assign(state_0_backup)
        self.state_1.assign(state_1_backup)

    def simulate(self):
        """Advance the loaded bag while both hand poses remain fixed."""
        for _ in range(SIM_SUBSTEPS):
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                _apply_particle_drag,
                dim=self.model.particle_count,
                inputs=[self.state_0.particle_qd, self.model.particle_mass, AIR_DRAG_RATE],
                outputs=[self.state_0.particle_f],
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance the simulation by one rendered frame."""
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the original bag mesh as a transparent red film."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/plastic_bag",
            self.state_0.particle_q,
            self.render_indices,
            backface_culling=False,
            color=BAG_COLOR,
            roughness=0.65,
            metallic=0.0,
            opacity=BAG_OPACITY,
        )
        self.viewer.end_frame()

    def test_final(self):
        """Verify finite supported-bag state and active hand contact shapes."""
        assert self.solver.features.backend == "vbd_kinematic_full"
        assert not any("rod" in label.lower() for label in self.model.shape_label)
        shape_flags = self.model.shape_flags.numpy()[self.hand_particle_shapes]
        assert np.all((shape_flags & int(newton.ShapeFlags.COLLIDE_PARTICLES)) != 0)

        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        joint_q = self.state_0.joint_q.numpy()
        if not np.all(np.isfinite(particle_q)) or not np.all(np.isfinite(body_q)) or not np.all(np.isfinite(joint_q)):
            raise ValueError("Bimanual carry state is not finite")
        if float(particle_q[:, 2].min()) < -0.01:
            raise ValueError("Plastic bag penetrated the ground")

        grip_positions = body_q[self.grip_body_indices, :3]
        left_distance = np.linalg.norm(particle_q[self.left_handle_indices] - grip_positions[0], axis=1).min()
        right_distance = np.linalg.norm(particle_q[self.right_handle_indices] - grip_positions[1], axis=1).min()
        if left_distance > 0.12 or right_distance > 0.12:
            raise ValueError(
                "Plastic bag slipped away from a hand: "
                f"handle distances are {left_distance:.6g} m and {right_distance:.6g} m"
            )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--bag-mesh", type=Path, default=BAG_MESH_PATH)
    parser.add_argument("--left-hand-urdf", type=Path, default=LEFT_HAND_URDF)
    parser.add_argument("--right-hand-urdf", type=Path, default=RIGHT_HAND_URDF)
    parser.add_argument("--hand-pose", type=Path, default=HAND_POSE_PATH)
    parser.add_argument(
        "--graph-capture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture one complete MJVBDV2 display frame on CUDA.",
    )
    parser.set_defaults(num_frames=240)

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

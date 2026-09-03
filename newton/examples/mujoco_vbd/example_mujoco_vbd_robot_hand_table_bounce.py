# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounce the dynamic right hand of a full Dexforce W1 off a VBD table.

The gravity-compensated, fixed-base W1 starts in a crouched pose with its right
arm extended over the table.  One shoulder degree of freedom receives an
initial downward joint velocity and is then left passive while finite-
stiffness drives hold the rest of the robot.  No target trajectory or scripted
velocity reversal is applied: the hand can move upward only after the VBD
table contact returns a wrench to the MuJoCo articulation through two-way
coupling.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import ClassVar

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMuJoCoVBD

ASSET_ROOT = Path(__file__).resolve().parents[3] / "assets"
DEFAULT_ROBOT_URDF = ASSET_ROOT / "DexforceW1V021" / "DexforceW1V021.urdf"

TABLE_TOP = 0.98
INITIAL_RIGHT_J2_SPEED = 0.35
MAX_CONTACT_PENETRATION = 0.006
HAND_VHACD_MAX_HULLS = 2
HAND_VHACD_RESOLUTION = 50_000
HAND_VHACD_VOLUME_ERROR = 4.0
HAND_VHACD_MAX_RECURSION = 6
HAND_VHACD_MAX_VERTICES = 32


@wp.kernel
def _accumulate_hand_table_distance(
    contact_count: wp.array[wp.int32],
    shape0: wp.array[wp.int32],
    shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    distance: wp.array[float],
    right_hand_body_mask: wp.array[wp.int32],
    table_body: int,
    minimum_distance: wp.array[float],
):
    """Reduce the signed W1-hand/table distance over one collision pass."""
    contact = wp.tid()
    if contact >= contact_count[0]:
        return
    body0 = shape_body[shape0[contact]]
    body1 = shape_body[shape1[contact]]
    hand_table = (body0 == table_body and body1 >= 0 and right_hand_body_mask[body1] != 0) or (
        body1 == table_body and body0 >= 0 and right_hand_body_mask[body0] != 0
    )
    if hand_table:
        wp.atomic_min(minimum_distance, 0, distance[contact])


@wp.kernel
def _accumulate_hand_dynamics(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    feedback_wrench: wp.array[wp.spatial_vector],
    right_hand_bodies: wp.array[wp.int32],
    wrist_body: int,
    minimum_wrist_height: wp.array[float],
    maximum_rebound_speed: wp.array[float],
    maximum_feedback_force: wp.array[float],
):
    """Reduce right-hand motion and feedback over one coupled substep."""
    index = wp.tid()
    body = right_hand_bodies[index]
    force = wp.spatial_top(feedback_wrench[body])
    wp.atomic_max(maximum_feedback_force, 0, wp.length(force))
    if body == wrist_body:
        height = wp.transform_get_translation(body_q[body])[2]
        vertical_speed = wp.spatial_top(body_qd[body])[2]
        wp.atomic_min(minimum_wrist_height, 0, height)
        wp.atomic_max(maximum_rebound_speed, 0, vertical_speed)


class Example:
    """Impact a VBD-owned table with one passive dynamic DOF of a full W1."""

    RIGHT_ARM = ("RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7")
    HAND_BODY_KEYWORDS = ("j7", "thumb", "index", "middle", "ring", "pinky")
    CROUCH_JOINTS: ClassVar[dict[str, float]] = {
        "ANKLE": math.radians(55.0),
        "KNEE": math.radians(-110.0),
        "BUTTOCK": math.radians(70.0),
    }
    OPEN_HAND_JOINTS: ClassVar[dict[str, float]] = {
        "HAND_THUMB2": 0.0,
        "HAND_THUMB1": 0.0,
        "HAND_INDEX": 0.0,
        "INDEX_PIP": 0.0,
        "HAND_MIDDLE": 0.0,
        "MIDDLE_PIP": 0.0,
        "HAND_RING": 0.0,
        "RING_PIP": 0.0,
        "HAND_PINKY": 0.0,
        "PINKY_PIP": 0.0,
    }

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
        self.contact_observed = False
        self.max_feedback_force = 0.0
        self.minimum_wrist_height = float("inf")
        self.minimum_contact_distance = float("inf")
        self.maximum_rebound_speed = -float("inf")

        self.robot_urdf = Path(args.robot_urdf).expanduser().resolve()
        if not self.robot_urdf.is_file():
            raise FileNotFoundError(f"Dexforce W1 URDF not found: {self.robot_urdf}")

        # A gravity-compensated arm isolates impact response from sustained
        # shoulder loading after the bounce.
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.default_joint_cfg.armature = 0.02
        builder.default_shape_cfg.ke = 1.0e6
        builder.default_shape_cfg.kd = 500.0
        builder.default_shape_cfg.mu = 0.8
        builder.default_shape_cfg.margin = 0.0
        builder.default_shape_cfg.gap = 0.001
        builder.rigid_gap = 0.001
        SolverMuJoCoVBD.register_custom_attributes(builder)

        robot_articulation_start = builder.articulation_count
        builder.add_urdf(
            str(self.robot_urdf),
            xform=wp.transform_identity(),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        robot_articulations = tuple(range(robot_articulation_start, builder.articulation_count))
        if len(robot_articulations) != 1:
            raise RuntimeError(f"Expected one W1 articulation, got {robot_articulations}")
        self.robot_shape_end = builder.shape_count
        self._set_builder_posture(builder)
        self._decompose_right_hand_collision_meshes(builder)
        self.robot_shape_end = builder.shape_count
        self._configure_robot_collision_flags(builder)

        table_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=1.0e6,
            kd=500.0,
            mu=0.8,
            margin=0.001,
            gap=0.001,
        )
        self.table_body = builder.add_link(
            xform=wp.transform(wp.vec3(0.10, -0.78, TABLE_TOP - 0.055), wp.quat_identity()),
            mass=40.0,
            inertia=wp.mat33(np.diag([4.0, 4.0, 6.0])),
            label="vbd_table",
        )
        self._add_table_shapes(builder, table_cfg)
        table_joint = builder.add_joint_fixed(
            parent=-1,
            child=self.table_body,
            parent_xform=wp.transform(wp.vec3(0.10, -0.78, TABLE_TOP - 0.055), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            label="table_world_fixed",
        )
        builder.add_articulation([table_joint], label="vbd_table_support")

        builder.add_ground_plane(
            height=0.0,
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, ke=2.0e5, kd=100.0, mu=0.8),
            color=(0.16, 0.18, 0.22),
            label="impact_ground",
        )
        builder.color(balance_colors=True)
        self.model = builder.finalize(requires_grad=False)
        self._configure_dynamic_joint_drives()

        self.right_wrist_body = self._body_index("right_j7")
        self.right_hand_bodies = np.asarray(
            [
                body
                for body, label in enumerate(self.model.body_label)
                if "right_" in label.lower() and any(word in label.lower() for word in self.HAND_BODY_KEYWORDS)
            ],
            dtype=np.int32,
        )
        self.right_j2_dof = self._joint_dof_index("RIGHT_J2")
        self.solver = SolverMuJoCoVBD(
            self.model,
            mujoco_articulations=robot_articulations,
            joint_mode="dynamic",
            coupling_mode="two_way",
            contact_mode="full",
            coupling_options={
                "iterations": 8,
                "relaxation": "fixed",
                "relaxation_initial": 0.5,
            },
            mujoco_options={"njmax": 256},
            vbd_options={
                "iterations": 12,
                "rigid_contact_hard": False,
                "rigid_avbd_contact_alpha": 0.0,
                "rigid_contact_history": True,
                "rigid_body_contact_buffer_size": 1024,
            },
            collision_options={
                "broad_phase": "nxn",
                "rigid_contact_max": 2048,
                "contact_matching": "latest",
            },
        )
        if self.solver.backend_kind.value != "two_way":
            raise RuntimeError(f"Expected two_way, got {self.solver.backend_kind.value}")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        wp.copy(self.control.joint_target_q, self.model.joint_q)
        self.control.joint_target_qd.zero_()
        initial_qd = self.model.joint_qd.numpy()
        initial_qd[self.right_j2_dof] = INITIAL_RIGHT_J2_SPEED
        self.state_0.joint_qd.assign(initial_qd)
        newton.eval_fk(self.model, self.model.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.initial_table_pose = self.state_0.body_q.numpy()[self.table_body].copy()

        self.contacts = self.solver.contacts
        if self.track_metrics:
            self.contact_distance = wp.empty(
                self.contacts.rigid_contact_max,
                dtype=wp.float32,
                device=self.model.device,
            )
            right_hand_body_mask = np.zeros(self.model.body_count, dtype=np.int32)
            right_hand_body_mask[self.right_hand_bodies] = 1
            self.right_hand_body_mask = wp.array(
                right_hand_body_mask,
                dtype=wp.int32,
                device=self.model.device,
            )
            self.right_hand_bodies_device = wp.array(
                self.right_hand_bodies,
                dtype=wp.int32,
                device=self.model.device,
            )
            self.metric_minimum_distance = wp.full(1, 1.0e6, dtype=wp.float32, device=self.model.device)
            self.metric_minimum_wrist_height = wp.full(1, 1.0e6, dtype=wp.float32, device=self.model.device)
            self.metric_maximum_rebound_speed = wp.full(1, -1.0e6, dtype=wp.float32, device=self.model.device)
            self.metric_maximum_feedback_force = wp.zeros(1, dtype=wp.float32, device=self.model.device)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.35, -1.90, 1.30), pitch=-9.0, yaw=128.0)

        self.graph = None
        if bool(args.graph_capture) and self.model.device.is_cuda:
            with wp.ScopedDevice(self.model.device), wp.ScopedCapture() as capture:
                self._simulate_frame()
            self.graph = capture.graph

    def _set_builder_posture(self, builder: newton.ModelBuilder) -> None:
        """Author a crouched W1 with open hands and straight arms."""
        for joint_name, value in self.CROUCH_JOINTS.items():
            self._set_builder_joint_coordinate(builder, joint_name, value)
        for side in ("LEFT", "RIGHT"):
            for suffix, value in self.OPEN_HAND_JOINTS.items():
                self._set_builder_joint_coordinate(builder, f"{side}_{suffix}", value)

    @staticmethod
    def _set_builder_joint_coordinate(builder: newton.ModelBuilder, short_name: str, value: float) -> None:
        """Set one scalar W1 coordinate by its unprefixed URDF name."""
        joint = next(
            (index for index, label in enumerate(builder.joint_label) if label.endswith("/" + short_name)),
            None,
        )
        if joint is None:
            raise ValueError(f"W1 joint is missing: {short_name}")
        builder.joint_q[joint] = float(value)

    def _configure_robot_collision_flags(self, builder: newton.ModelBuilder) -> None:
        """Allow only visible right-hand links to collide with the table."""
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        collision_mask = collide_shapes | collide_particles
        self.robot_contact_shapes = []
        for shape in range(self.robot_shape_end):
            body = int(builder.shape_body[shape])
            label = "" if body < 0 else builder.body_label[body].lower()
            is_right_hand = "right_" in label and any(word in label for word in self.HAND_BODY_KEYWORDS)
            is_collider = bool(builder.shape_flags[shape] & collision_mask)
            if is_right_hand and is_collider:
                builder.shape_flags[shape] |= collide_shapes
                builder.shape_flags[shape] &= ~collide_particles
                self.robot_contact_shapes.append(shape)
            else:
                builder.shape_flags[shape] &= ~collision_mask
        if not self.robot_contact_shapes:
            raise RuntimeError("The W1 asset did not produce active right-hand collision shapes")

    def _decompose_right_hand_collision_meshes(self, builder: newton.ModelBuilder) -> None:
        """Convert right-hand collision meshes to bounded convex GJK/MPR parts."""
        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        source_shapes = []
        for shape in range(self.robot_shape_end):
            body = int(builder.shape_body[shape])
            if body < 0 or not (builder.shape_flags[shape] & collide_shapes):
                continue
            label = builder.body_label[body].lower()
            is_right_hand = "right_" in label and any(word in label for word in self.HAND_BODY_KEYWORDS)
            if is_right_hand and builder.shape_type[shape] == newton.GeoType.MESH:
                source_shapes.append(shape)
        if not source_shapes:
            raise RuntimeError("The W1 asset did not provide right-hand collision meshes")

        remeshed = builder.approximate_meshes(
            method="vhacd",
            shape_indices=source_shapes,
            raise_on_failure=True,
            keep_visual_shapes=False,
            maxConvexHulls=HAND_VHACD_MAX_HULLS,
            resolution=HAND_VHACD_RESOLUTION,
            minimumVolumePercentErrorAllowed=HAND_VHACD_VOLUME_ERROR,
            maxRecursionDepth=HAND_VHACD_MAX_RECURSION,
            maxNumVerticesPerCH=HAND_VHACD_MAX_VERTICES,
            asyncACD=False,
        )
        if len(remeshed) != len(source_shapes):
            raise RuntimeError(f"V-HACD converted {len(remeshed)} of {len(source_shapes)} hand meshes")

    @staticmethod
    def _add_table_shapes(builder: newton.ModelBuilder, cfg: newton.ModelBuilder.ShapeConfig) -> None:
        """Create the tabletop and its four visible supports."""
        table = builder.body_count - 1
        wood = (0.42, 0.22, 0.08)
        builder.add_shape_box(table, hx=0.42, hy=0.42, hz=0.055, cfg=cfg, color=wood, label="tabletop")
        for index, (x, y) in enumerate(((-0.34, -0.34), (-0.34, 0.34), (0.34, -0.34), (0.34, 0.34))):
            builder.add_shape_box(
                table,
                xform=wp.transform(wp.vec3(x, y, -0.49), wp.quat_identity()),
                hx=0.035,
                hy=0.035,
                hz=0.49,
                cfg=cfg,
                color=wood,
                label=f"table_leg_{index}",
            )

    def _configure_dynamic_joint_drives(self) -> None:
        """Hold the W1 posture while leaving the impacting shoulder DOF passive."""
        mode = self.model.joint_target_mode.numpy()
        stiffness = self.model.joint_target_ke.numpy()
        damping = self.model.joint_target_kd.numpy()
        effort_limit = self.model.joint_effort_limit.numpy()
        qd_start = self.model.joint_qd_start.numpy()

        right_arm_names = set(self.RIGHT_ARM)
        for joint, label in enumerate(self.model.joint_label):
            begin = int(qd_start[joint])
            end = int(qd_start[joint + 1])
            if begin == end:
                continue
            short_name = label.rsplit("/", maxsplit=1)[-1]
            if short_name == "RIGHT_J2":
                mode[begin:end] = int(newton.JointTargetMode.NONE)
                stiffness[begin:end] = 0.0
                damping[begin:end] = 0.0
            elif short_name in right_arm_names:
                mode[begin:end] = int(newton.JointTargetMode.POSITION_VELOCITY)
                stiffness[begin:end] = 1200.0
                damping[begin:end] = 60.0
                effort_limit[begin:end] = np.maximum(effort_limit[begin:end], 150.0)
            elif "_HAND_" in short_name or short_name.endswith("_PIP"):
                mode[begin:end] = int(newton.JointTargetMode.POSITION_VELOCITY)
                stiffness[begin:end] = 80.0
                damping[begin:end] = 8.0
                effort_limit[begin:end] = np.maximum(effort_limit[begin:end], 20.0)
            else:
                mode[begin:end] = int(newton.JointTargetMode.POSITION_VELOCITY)
                stiffness[begin:end] = 2500.0
                damping[begin:end] = 120.0
                effort_limit[begin:end] = np.maximum(effort_limit[begin:end], 250.0)

        self.model.joint_target_mode.assign(mode)
        self.model.joint_target_ke.assign(stiffness)
        self.model.joint_target_kd.assign(damping)
        self.model.joint_effort_limit.assign(effort_limit)

    def _body_index(self, short_name: str) -> int:
        """Return one W1 body index by its unprefixed name."""
        return next(index for index, label in enumerate(self.model.body_label) if label.endswith("/" + short_name))

    def _joint_dof_index(self, short_name: str) -> int:
        """Return the scalar DOF index for one W1 joint."""
        joint = next(index for index, label in enumerate(self.model.joint_label) if label.endswith("/" + short_name))
        return int(self.model.joint_qd_start.numpy()[joint])

    def _simulate_frame(self) -> None:
        """Enqueue all coupled substeps for one display frame."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            if self.track_metrics:
                newton.eval_rigid_contact_kinematics(
                    self.model,
                    self.state_0,
                    self.contacts,
                    out_distance=self.contact_distance,
                )
                wp.launch(
                    _accumulate_hand_table_distance,
                    dim=self.contacts.rigid_contact_max,
                    inputs=[
                        self.contacts.rigid_contact_count,
                        self.contacts.rigid_contact_shape0,
                        self.contacts.rigid_contact_shape1,
                        self.model.shape_body,
                        self.contact_distance,
                        self.right_hand_body_mask,
                        self.table_body,
                        self.metric_minimum_distance,
                    ],
                    device=self.model.device,
                )
                wp.launch(
                    _accumulate_hand_dynamics,
                    dim=len(self.right_hand_bodies),
                    inputs=[
                        self.state_0.body_q,
                        self.state_0.body_qd,
                        self.solver.diagnostics.feedback_wrench_raw,
                        self.right_hand_bodies_device,
                        self.right_wrist_body,
                        self.metric_minimum_wrist_height,
                        self.metric_maximum_rebound_speed,
                        self.metric_maximum_feedback_force,
                    ],
                    device=self.model.device,
                )

    def step(self) -> None:
        """Advance the impact and collect contact-driven rebound metrics."""
        if self.graph is None:
            self._simulate_frame()
        else:
            wp.capture_launch(self.graph)

        if self.track_metrics:
            self.minimum_contact_distance = float(self.metric_minimum_distance.numpy()[0])
            self.minimum_wrist_height = float(self.metric_minimum_wrist_height.numpy()[0])
            self.maximum_rebound_speed = float(self.metric_maximum_rebound_speed.numpy()[0])
            self.max_feedback_force = float(self.metric_maximum_feedback_force.numpy()[0])
            self.contact_observed = self.minimum_contact_distance < 1.0e5 and self.max_feedback_force > 1.0

        self.sim_time += self.frame_dt

    def render(self) -> None:
        """Render the full W1, VBD table, ground, and generated contacts."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts is not None:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        """Require finite W1 state, two-way reaction, no tunneling, and rebound."""
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        assert np.all(np.isfinite(body_q))
        assert np.all(np.isfinite(body_qd))
        assert self.contact_observed, "The W1 right hand never contacted the VBD table"
        assert self.max_feedback_force > 10.0, "The table reaction was not returned to the W1"
        assert np.isfinite(self.minimum_contact_distance), "No W1 hand-table contact distance was measured"
        assert self.minimum_contact_distance > -MAX_CONTACT_PENETRATION, (
            f"The W1 right hand penetrated the tabletop by {-self.minimum_contact_distance:.6f} m"
        )
        assert self.minimum_wrist_height > TABLE_TOP, "The W1 right hand passed through the tabletop"
        assert self.maximum_rebound_speed > 0.05, "The W1 right hand did not rebound after contact"
        table_error = float(np.max(np.abs(body_q[self.table_body] - self.initial_table_pose)))
        assert table_error < 1.0e-3, f"The fixed table drifted by {table_error:.6f} m"

    @staticmethod
    def create_parser():
        """Create command-line options for the full-W1 impact scene."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=180)
        parser.add_argument("--robot-urdf", default=str(DEFAULT_ROBOT_URDF), help="Dexforce W1 URDF path.")
        parser.add_argument("--substeps", type=int, default=24, help="Coupled substeps per 60 Hz frame.")
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture all impact substeps as one CUDA graph.",
        )
        return parser


def main():
    """Run the full-W1 hand/table bounce example."""
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

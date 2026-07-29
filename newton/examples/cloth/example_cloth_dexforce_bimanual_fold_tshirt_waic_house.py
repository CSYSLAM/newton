# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Dexforce W1 bimanual, two-pass T-shirt folding in the WAIC house.

The fixed robot motion is solved once in an *IK-only* model and replayed as
cached kinematic joint targets. ``SolverMuJoCoVBD`` solves those reduced
coordinates together with the VBD-owned shirt, so cloth contact reactions use
the direct q-block path. The optional house USD is rendering-only: it never
adds a collider or a particle-contact candidate.

Run, from the repository root::

    uv run --extra examples -m newton.examples cloth_dexforce_bimanual_fold_tshirt_waic_house
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.usd
from newton.solvers import SolverMuJoCoVBD

# The captured trajectory was authored at this table, then rigidly transformed
# into the aligned WAIC dining-table coordinate system below.
OLD_TABLE_POS = wp.vec3(0.55, 0.0, 1.15)
TABLE_HALF_EXTENTS = (0.26, 0.62, 0.025)
TABLE_COLOR = (0.35, 0.42, 0.48)
WAIC_ROBOT_BASE_POS = wp.vec3(-0.34931439, -3.24669516, -0.00377202)
WAIC_ROBOT_BASE_QUAT = wp.quat(0.0, 0.0, 0.70710677, 0.70710677)
CAMERA_POS = wp.vec3(2.0240853, -5.6667042, 2.0068865)
CAMERA_PITCH, CAMERA_YAW = -20.6224, 126.2538
DEFAULT_HOUSE_USD = (
    "/home/oem/code/engine/newton/newton/examples/cloth/assets/house_background/"
    "House5_Simple2_visual_table01_table02_box_top_aligned_table02_w1_edge_translated.usd"
)

SHIRT_POS = wp.vec3(float(OLD_TABLE_POS[0]), 0.0, float(OLD_TABLE_POS[2]) + TABLE_HALF_EXTENTS[2] + 0.014)
SHIRT_ROT = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.5 * np.pi)
SHIRT_SCALE = 0.0064
SHIRT_DENSITY = 0.02
SHIRT_COLOR = (0.70, 0.70, 0.70)
TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)

RADIUS = 0.008
SOFT_MARGIN = 0.008
SELF_RADIUS = 0.002
SELF_MARGIN = 0.002
VBD_ITERATIONS = 20
IK_ITERATIONS = 24
LEGACY_SOFT_CONTACT_KD = 5.0e-2
LEGACY_SHAPE_CONTACT_KD = 1.0e-6
FULL_PROCESS_CLOTH_CONTACT_MU = 1.2
FULL_PROCESS_TABLE_CONTACT_MU = 1.2

# Captured wrist TCP poses, grouped into the first and second fold passes.
P = wp.vec3
Q = wp.quat
LEFT_APPROACH = wp.transform(P(0.4117, 0.5145, 1.3642), Q(0.1099, 0.6989, -0.6980, -0.1107))
RIGHT_APPROACH = wp.transform(P(0.4053, -0.5598, 1.3642), Q(-0.0658, 0.7043, 0.7037, -0.0668))
LEFT_GRASP = wp.transform(P(0.3565, 0.1627, 1.2406), Q(0.1103, 0.6987, -0.6982, -0.1105))
RIGHT_GRASP = wp.transform(P(0.3519, -0.1623, 1.2405), Q(-0.0733, 0.7031, 0.7037, -0.0717))
LEFT_LIFT = wp.transform(P(0.3709, 0.1679, 1.3359), Q(0.1105, 0.6985, -0.6981, -0.1119))
RIGHT_LIFT = wp.transform(P(0.3764, -0.1638, 1.3433), Q(-0.0738, 0.7030, 0.7037, -0.0721))
LEFT_TRAVEL = wp.transform(P(0.5542, 0.1679, 1.3376), Q(0.1093, 0.6986, -0.6984, -0.1107))
RIGHT_TRAVEL = wp.transform(P(0.6038, -0.1635, 1.3444), Q(-0.0738, 0.7030, 0.7036, -0.0719))
LEFT_PLACE = wp.transform(P(0.6871, 0.1563, 1.2571), Q(0.0195, 0.7068, -0.7067, -0.0247))
RIGHT_PLACE = wp.transform(P(0.6965, -0.1642, 1.2722), Q(0.0237, 0.6998, 0.7126, -0.0439))
LEFT_RELEASE = wp.transform(P(0.6973, 0.1596, 1.4336), Q(-0.0203, -0.7068, 0.7068, 0.0239))
RIGHT_RELEASE = wp.transform(P(0.6974, -0.1641, 1.4101), Q(0.0238, 0.6998, 0.7126, -0.0441))
LEFT_2GRASP = wp.transform(P(0.6225, 0.2642, 1.2460), Q(0.0216, 0.7067, -0.7067, -0.0245))
RIGHT_2GRASP = wp.transform(P(0.6223, -0.2669, 1.2432), Q(0.0207, 0.7001, 0.7123, -0.0457))
LEFT_2LIFT = wp.transform(P(0.6428, 0.1735, 1.3145), Q(0.0215, 0.7067, -0.7067, -0.0246))
RIGHT_2LIFT = wp.transform(P(0.6232, -0.2284, 1.3094), Q(0.0209, 0.7001, 0.7123, -0.0461))
LEFT_2TRAVEL = wp.transform(P(0.6436, 0.0678, 1.3139), Q(-0.0212, -0.7067, 0.7067, 0.0250))
RIGHT_2TRAVEL = wp.transform(P(0.6224, -0.0861, 1.3101), Q(0.0206, 0.7001, 0.7123, -0.0457))
LEFT_2PLACE = wp.transform(P(0.6438, 0.0677, 1.2762), Q(-0.0212, -0.7067, 0.7067, 0.0251))
RIGHT_2PLACE = wp.transform(P(0.6227, -0.0860, 1.2694), Q(0.0207, 0.7001, 0.7123, -0.0458))
LEFT_2RELEASE = wp.transform(P(0.6438, 0.0677, 1.4336), Q(-0.0212, -0.7067, 0.7067, 0.0251))
RIGHT_2RELEASE = wp.transform(P(0.6227, -0.0860, 1.4101), Q(0.0207, 0.7001, 0.7123, -0.0458))


@wp.kernel
def _interpolate_q(q0: wp.array[float], q1: wp.array[float], alpha: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = q0[i] * (1.0 - alpha) + q1[i] * alpha


@wp.kernel
def _joint_velocity(q0: wp.array[float], q1: wp.array[float], inv_dt: float, out: wp.array[float]):
    i = wp.tid()
    out[i] = (q1[i] - q0[i]) * inv_dt


@wp.kernel
def _lock_q(q: wp.array2d[float], indices: wp.array[int], values: wp.array[float]):
    i = wp.tid()
    q[0, indices[i]] = values[i]


@wp.kernel
def _copy_joint_q(src: wp.array[float], dst: wp.array[float]):
    i = wp.tid()
    dst[i] = src[i]


@wp.kernel
def _load_cached_joint_q(
    cached_q: wp.array2d[float], frame_counter: wp.array[wp.int32], max_frame_index: int, dst: wp.array[float]
):
    i = wp.tid()
    dst[i] = cached_q[wp.min(frame_counter[0], max_frame_index), i]


@wp.kernel
def _advance_frame_counter(frame_counter: wp.array[wp.int32], max_frame_index: int):
    if wp.tid() == 0:
        frame_counter[0] = wp.min(frame_counter[0] + 1, max_frame_index)


@wp.kernel
def _set_shape_friction(shape_mu: wp.array[float], robot_shape_end: int, robot_mu: float, table_mu: float):
    i = wp.tid()
    shape_mu[i] = robot_mu if i < robot_shape_end else table_mu


class Example:
    LEFT_ARM = ("LEFT_J1", "LEFT_J2", "LEFT_J3", "LEFT_J4", "LEFT_J5", "LEFT_J6", "LEFT_J7")
    RIGHT_ARM = ("RIGHT_J1", "RIGHT_J2", "RIGHT_J3", "RIGHT_J4", "RIGHT_J5", "RIGHT_J6", "RIGHT_J7")
    HAND_SUFFIXES = (
        "HAND_THUMB2",
        "HAND_THUMB1",
        "HAND_INDEX",
        "INDEX_PIP",
        "HAND_MIDDLE",
        "MIDDLE_PIP",
        "HAND_RING",
        "RING_PIP",
        "HAND_PINKY",
        "PINKY_PIP",
    )

    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        self.base_pos = wp.vec3(args.waic_robot_base_x, args.waic_robot_base_y, args.waic_robot_base_z)
        self.base_rot = self._normal_quat(
            wp.quat(args.waic_robot_base_qx, args.waic_robot_base_qy, args.waic_robot_base_qz, args.waic_robot_base_qw)
        )
        self.house_visual_usd = args.house_visual_usd

        self._build_scene()
        self.device = self.model.device
        self.control = self.model.control()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=SOFT_MARGIN,
        )
        self.contacts = self.pipeline.contacts()
        self.solver = SolverMuJoCoVBD(
            self.model,
            articulation_bodies=list(range(self.robot_body_end)),
            articulation_joints=list(range(self.robot_joint_end)),
            vbd_bodies=[],
            vbd_particles=list(range(self.shirt_particle_start, self.shirt_particle_end)),
            articulation_mode="kinematic",
            vbd_options={
                "iterations": VBD_ITERATIONS,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": SELF_RADIUS,
                "particle_self_contact_margin": SELF_MARGIN,
                "particle_topological_contact_filter_threshold": 1,
                "particle_rest_shape_contact_exclusion_radius": 0.03,
                "particle_vertex_contact_buffer_size": 16,
                "particle_edge_contact_buffer_size": 20,
                "rigid_body_particle_contact_buffer_size": 256,
                "particle_collision_detection_interval": -1,
            },
            collide=self.pipeline.collide,
        )

        self.left_body = self._body_index(self.model.body_label, "left_j7")
        self.right_body = self._body_index(self.model.body_label, "right_j7")
        self._build_ik()
        self.left_home = self._tcp(self.state_0, self.left_body)
        self.right_home = self._tcp(self.state_0, self.right_body)
        self.segments = self._segments()
        self.ik_q = wp.clone(self.model.joint_q[: self.ik_model.joint_coord_count]).reshape((1, -1))
        self.frame_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_q_end = wp.zeros_like(self.model.joint_q)
        self.lock_indices, self.lock_values = self._locked_q()
        self.hand_indices, self.hand_open, self.hand_grasp = self._hand_q()
        self._build_joint_target_cache()
        self.graph_frame_index = wp.array([1], dtype=wp.int32, device=self.device)

        self._attach_house_usd()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(CAMERA_POS, CAMERA_PITCH, CAMERA_YAW)
        # DAT acceptance performs a device-to-host feasibility read. It cannot
        # participate in CUDA graph capture until that gate is graph-safe.
        self.use_graph = False
        self.graph = None
        self.graphs = {}
        self.capture()

    # --- scene ----------------------------------------------------------
    def _robot_urdf(self) -> Path:
        if self.args.robot_urdf:
            path = Path(self.args.robot_urdf).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"--robot-urdf does not exist: {path}")
            return path
        path = (
            Path(__file__).parents[1]
            / "multiphysics"
            / "newton_cloth_dexforce_place_tablecloth"
            / "DexforceW1V021"
            / "DexforceW1V021.urdf"
        )
        if path.is_file():
            return path
        raise FileNotFoundError("Dexforce W1 URDF is unavailable; pass --robot-urdf PATH.")

    def _build_scene(self):
        self.urdf_path = self._robot_urdf()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.default_joint_cfg.armature = 0.02
        builder.default_shape_cfg.ke = 5.0e5
        builder.default_shape_cfg.kd = 1.0e-6
        builder.default_shape_cfg.mu = 2.0
        builder.add_urdf(
            str(self.urdf_path),
            xform=wp.transform(self.base_pos, self.base_rot),
            floating=False,
            enable_self_collisions=bool(self.args.enable_self_collisions),
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.robot_body_end = builder.body_count
        self.robot_joint_end = builder.joint_count
        self.robot_shape_end = builder.shape_count
        for body in range(self.robot_body_end):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        self._add_table(builder)
        self._add_shirt(builder)
        self._configure_flags(builder)
        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        if self.model.edge_rest_angle is not None:
            self.model.edge_rest_angle.zero_()
        self.model.soft_contact_ke = 3.0e5
        self.model.soft_contact_kd = LEGACY_SOFT_CONTACT_KD
        self._set_robot_contact_stiffness(3.0e5)
        self._set_legacy_contact_damping()
        self._set_materials(0.50, 0.25, 0.50)

    def _add_table(self, builder):
        old_table = self._world_vec(OLD_TABLE_POS)
        cfg = newton.ModelBuilder.ShapeConfig(
            ke=5.0e5, kd=1.0e-6, mu=1.2, is_visible=bool(self.args.show_physics_table)
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(old_table, self.base_rot),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=cfg,
            color=TABLE_COLOR,
            label="waic_physics_table",
        )
        builder.add_ground_plane(height=float(self.base_pos[2]), label="waic_ground")

    def _add_shirt(self, builder):
        stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        if stage is None:
            raise RuntimeError("Unable to load examples asset unisex_shirt.usd")
        mesh = newton.usd.get_mesh(stage.GetPrimAtPath("/root/shirt"))
        vertices = np.asarray(mesh.vertices, dtype=np.float32).copy()
        vertices[:, :2] -= 0.5 * (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0))
        vertices[:, 2] -= vertices[:, 2].min()
        self.shirt_particle_start = builder.particle_count
        builder.add_cloth_mesh(
            vertices=[wp.vec3(*v) for v in vertices * SHIRT_SCALE],
            indices=mesh.indices,
            pos=self._world_vec(SHIRT_POS),
            rot=self._quat_mul(self.base_rot, SHIRT_ROT),
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=SHIRT_DENSITY,
            scale=1.0,
            tri_ke=1.5e3,
            tri_ka=1.5e3,
            tri_kd=1.0e-5,
            edge_ke=1.2,
            edge_kd=0.1,
            particle_radius=RADIUS,
            label="fold_tshirt",
        )
        self.shirt_particle_end = builder.particle_count

    def _configure_flags(self, builder):
        cp, cs = int(newton.ShapeFlags.COLLIDE_PARTICLES), int(newton.ShapeFlags.COLLIDE_SHAPES)
        # Match the captured fold scene: every robot collider can touch cloth,
        # but the rigid-only collision path remains disabled in external-FK mode.
        for i in range(self.robot_shape_end):
            builder.shape_flags[i] &= ~cs
            builder.shape_flags[i] |= cp
        for i in range(self.robot_shape_end, builder.shape_count):
            builder.shape_flags[i] |= cp

    # --- independent IK model -----------------------------------------
    def _build_ik(self):
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        b.add_urdf(
            str(self.urdf_path),
            xform=wp.transform(self.base_pos, self.base_rot),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.ik_model = b.finalize(device=self.model.device)
        left = self._body_index(self.ik_model.body_label, "left_j7")
        right = self._body_index(self.ik_model.body_label, "right_j7")
        initial = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, initial)
        self.left_obj = ik.IKObjectivePosition(
            left,
            TCP_OFFSET,
            wp.array(
                [wp.transform_get_translation(self._tcp(initial, self.left_body))], dtype=wp.vec3, device=self.device
            ),
        )
        self.left_rot = ik.IKObjectiveRotation(
            left,
            wp.quat_identity(),
            wp.array(
                [self._v4(wp.transform_get_rotation(self._tcp(initial, self.left_body)))],
                dtype=wp.vec4,
                device=self.device,
            ),
        )
        self.right_obj = ik.IKObjectivePosition(
            right,
            TCP_OFFSET,
            wp.array(
                [wp.transform_get_translation(self._tcp(initial, self.right_body))], dtype=wp.vec3, device=self.device
            ),
        )
        self.right_rot = ik.IKObjectiveRotation(
            right,
            wp.quat_identity(),
            wp.array(
                [self._v4(wp.transform_get_rotation(self._tcp(initial, self.right_body)))],
                dtype=wp.vec4,
                device=self.device,
            ),
        )
        lower, upper = self._joint_limits()
        limits = ik.IKObjectiveJointLimit(
            wp.array(lower, dtype=wp.float32, device=self.device),
            wp.array(upper, dtype=wp.float32, device=self.device),
            weight=25.0,
        )
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            n_problems=1,
            objectives=[self.left_obj, self.left_rot, self.right_obj, self.right_rot, limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    # --- motion / simulation ------------------------------------------
    def _segments(self):
        w = self._world_tf
        return (
            (0.8, self.left_home, self.left_home, self.right_home, self.right_home, 0.0, 0.0),
            (2.0, self.left_home, w(LEFT_APPROACH), self.right_home, w(RIGHT_APPROACH), 0.0, 0.0),
            (1.2, w(LEFT_APPROACH), w(LEFT_GRASP), w(RIGHT_APPROACH), w(RIGHT_GRASP), 0.0, 0.0),
            (1.4, w(LEFT_GRASP), w(LEFT_GRASP), w(RIGHT_GRASP), w(RIGHT_GRASP), 0.0, 1.0),
            (1.4, w(LEFT_GRASP), w(LEFT_LIFT), w(RIGHT_GRASP), w(RIGHT_LIFT), 1.0, 1.0),
            (2.4, w(LEFT_LIFT), w(LEFT_TRAVEL), w(RIGHT_LIFT), w(RIGHT_TRAVEL), 1.0, 1.0),
            (1.4, w(LEFT_TRAVEL), w(LEFT_PLACE), w(RIGHT_TRAVEL), w(RIGHT_PLACE), 1.0, 1.0),
            (1.0, w(LEFT_PLACE), w(LEFT_PLACE), w(RIGHT_PLACE), w(RIGHT_PLACE), 1.0, 0.75),
            (1.2, w(LEFT_PLACE), w(LEFT_RELEASE), w(RIGHT_PLACE), w(RIGHT_RELEASE), 0.75, 0.0),
            (1.0, w(LEFT_RELEASE), w(LEFT_RELEASE), w(RIGHT_RELEASE), w(RIGHT_RELEASE), 0.0, 0.0),
            (2.0, w(LEFT_RELEASE), w(LEFT_2GRASP), w(RIGHT_RELEASE), w(RIGHT_2GRASP), 0.0, 0.0),
            (1.2, w(LEFT_2GRASP), w(LEFT_2GRASP), w(RIGHT_2GRASP), w(RIGHT_2GRASP), 0.0, 1.0),
            (1.2, w(LEFT_2GRASP), w(LEFT_2LIFT), w(RIGHT_2GRASP), w(RIGHT_2LIFT), 1.0, 1.0),
            (1.8, w(LEFT_2LIFT), w(LEFT_2TRAVEL), w(RIGHT_2LIFT), w(RIGHT_2TRAVEL), 1.0, 1.0),
            (1.2, w(LEFT_2TRAVEL), w(LEFT_2PLACE), w(RIGHT_2TRAVEL), w(RIGHT_2PLACE), 1.0, 1.0),
            (0.8, w(LEFT_2PLACE), w(LEFT_2PLACE), w(RIGHT_2PLACE), w(RIGHT_2PLACE), 1.0, 0.75),
            (1.2, w(LEFT_2PLACE), w(LEFT_2RELEASE), w(RIGHT_2PLACE), w(RIGHT_2RELEASE), 0.75, 0.0),
            (2.0, w(LEFT_2RELEASE), w(LEFT_APPROACH), w(RIGHT_2RELEASE), w(RIGHT_APPROACH), 0.0, 0.0),
            (1.0, w(LEFT_APPROACH), w(LEFT_APPROACH), w(RIGHT_APPROACH), w(RIGHT_APPROACH), 0.0, 0.0),
        )

    def _materials_for_script_time(self, script_time, grip):
        release = self._is_release_time(script_time)
        if release:
            # Keep the folded shirt anchored on the table while allowing the
            # fingers to release it cleanly.
            return FULL_PROCESS_CLOTH_CONTACT_MU, 0.0, FULL_PROCESS_TABLE_CONTACT_MU
        if grip > 0.15:
            return FULL_PROCESS_CLOTH_CONTACT_MU, 1.2, FULL_PROCESS_TABLE_CONTACT_MU
        return FULL_PROCESS_CLOTH_CONTACT_MU, 0.25, FULL_PROCESS_TABLE_CONTACT_MU

    def _build_joint_target_cache(self):
        """Solve the fixed scripted motion once, before simulation starts."""
        script_duration = sum(segment[0] for segment in self.segments)
        script_frames = int(np.ceil(script_duration / (self.frame_dt * self.args.trajectory_time_scale)))
        self.cached_joint_target_frame_count = max(int(self.args.num_frames), script_frames)

        initial_q = np.asarray(self.model.joint_q.numpy(), dtype=np.float32)
        cache = np.repeat(initial_q[None, :], self.cached_joint_target_frame_count + 1, axis=0)
        hand_indices = self.hand_indices.numpy()
        hand_open = self.hand_open.numpy()
        hand_grasp = self.hand_grasp.numpy()
        self.cached_materials = [self._materials_for_script_time(0.0, 0.0)]

        for frame_index in range(1, self.cached_joint_target_frame_count + 1):
            script_time = frame_index * self.frame_dt * self.args.trajectory_time_scale
            left, right, grip = self._sample(script_time)
            self.left_obj.set_target_position(0, wp.transform_get_translation(left))
            self.left_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(left)))
            self.right_obj.set_target_position(0, wp.transform_get_translation(right))
            self.right_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(right)))
            self.ik_solver.step(self.ik_q, self.ik_q, iterations=IK_ITERATIONS)
            wp.launch(
                _lock_q,
                self.lock_indices.shape[0],
                [self.ik_q, self.lock_indices, self.lock_values],
                device=self.device,
            )
            cache[frame_index, : self.ik_model.joint_coord_count] = self.ik_q.numpy()[0]
            cache[frame_index, hand_indices] = hand_open * (1.0 - grip) + hand_grasp * grip
            self.cached_materials.append(self._materials_for_script_time(script_time, grip))

        self.cached_joint_targets = wp.array(cache, dtype=wp.float32, device=self.device)
        self.cached_materials = tuple(self.cached_materials)
        self.ik_q = wp.array(initial_q, dtype=wp.float32, device=self.device).reshape((1, -1))

    def _prepare_cached_frame(self):
        """Load one baked target for the uncaptured CPU execution path."""
        cache_index = min(self.frame_index + 1, self.cached_joint_target_frame_count)
        wp.copy(self.frame_q_start, self.state_0.joint_q)
        wp.copy(self.frame_q_end, self.cached_joint_targets[cache_index])
        self._set_materials(*self.cached_materials[cache_index])

    def _is_release_time(self, time):
        """Return true only during the two deliberate friction-release cracks."""
        first_start = sum(segment[0] for segment in self.segments[:7])
        first_end = sum(segment[0] for segment in self.segments[:9])
        second_start = sum(segment[0] for segment in self.segments[:15])
        second_end = sum(segment[0] for segment in self.segments[:17])
        return first_start <= time <= first_end or second_start <= time <= second_end

    def _simulate_substeps(self):
        for step in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (step + 1) / self.sim_substeps
            wp.launch(
                _interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _joint_velocity,
                self.model.joint_dof_count,
                [self.frame_q_start, self.frame_q_end, 1.0 / self.frame_dt, self.state_0.joint_qd],
                device=self.device,
            )
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def simulate(self):
        self._prepare_cached_frame()
        self._simulate_substeps()
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def step(self):
        if not self.use_graph:
            self.simulate()
            return

        cache_index = min(self.frame_index + 1, self.cached_joint_target_frame_count)
        self.graph = self.graphs[self.cached_materials[cache_index]]
        wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def _capture_graph(self, materials):
        """Capture one material variant of a complete scripted display frame."""
        self.model.soft_contact_mu = materials[0]
        state_0_backup = self.model.state()
        state_1_backup = self.model.state()
        state_0_backup.assign(self.state_0)
        state_1_backup.assign(self.state_1)

        with wp.ScopedCapture() as capture:
            wp.launch(
                _copy_joint_q,
                self.model.joint_coord_count,
                [self.state_0.joint_q, self.frame_q_start],
                device=self.device,
            )
            wp.launch(
                _load_cached_joint_q,
                self.model.joint_coord_count,
                [
                    self.cached_joint_targets,
                    self.graph_frame_index,
                    self.cached_joint_target_frame_count,
                    self.frame_q_end,
                ],
                device=self.device,
            )
            wp.launch(
                _set_shape_friction,
                self.model.shape_count,
                [self.model.shape_material_mu, self.robot_shape_end, materials[1], materials[2]],
                device=self.device,
            )
            self._simulate_substeps()
            wp.launch(
                _advance_frame_counter,
                1,
                [self.graph_frame_index, self.cached_joint_target_frame_count],
                device=self.device,
            )

        self.state_0.assign(state_0_backup)
        self.state_1.assign(state_1_backup)
        return capture.graph

    def capture(self):
        """Capture every fixed-trajectory material variant on CUDA."""
        self.graph = None
        self.graphs = {}
        if not self.use_graph:
            return

        for materials in dict.fromkeys(self.cached_materials):
            self.graphs[materials] = self._capture_graph(materials)
        self.graph_frame_index.fill_(1)
        self.graph = self.graphs[self.cached_materials[1]]

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/fold_tshirt",
            self.state_0.particle_q,
            self.model.tri_indices.flatten(),
            hidden=not getattr(self.viewer, "show_triangles", True),
            backface_culling=False,
            color=SHIRT_COLOR,
        )
        self.viewer.end_frame()

    def test_final(self):
        if not np.all(np.isfinite(self.state_0.particle_q.numpy())):
            raise ValueError("T-shirt particle positions are not finite")

    # --- helpers --------------------------------------------------------
    def _set_materials(self, cloth_mu, robot_mu, table_mu):
        self.model.soft_contact_mu = cloth_mu
        mu = self.model.shape_material_mu.numpy()
        mu[: self.robot_shape_end] = robot_mu
        mu[self.robot_shape_end :] = table_mu
        self.model.shape_material_mu.assign(mu)

    def _set_robot_contact_stiffness(self, robot_ke):
        ke = self.model.shape_material_ke.numpy()
        ke[: self.robot_shape_end] = robot_ke
        self.model.shape_material_ke.assign(ke)

    def _set_legacy_contact_damping(self):
        """Translate the reference demo's proportional damping into current VBD units."""
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        legacy_mixed_kd = 0.5 * (LEGACY_SOFT_CONTACT_KD + LEGACY_SHAPE_CONTACT_KD)
        mixed_kd = legacy_mixed_kd * 0.5 * (self.model.soft_contact_ke + shape_ke)
        shape_kd[:] = 2.0 * mixed_kd - self.model.soft_contact_kd
        self.model.shape_material_kd.assign(shape_kd)

    def _joint_limits(self):
        lo, hi = self.model.joint_limit_lower.numpy().copy(), self.model.joint_limit_upper.numpy().copy()
        q = self.model.joint_q.numpy()
        qs = self.model.joint_q_start.numpy()
        qds = self.model.joint_qd_start.numpy()
        controlled = {f"DexforceW1V021/{n}" for n in (*self.LEFT_ARM, *self.RIGHT_ARM)}
        for j, label in enumerate(self.model.joint_label):
            if label not in controlled:
                lo[int(qds[j])] = q[int(qs[j])] - 1.0e-4
                hi[int(qds[j])] = q[int(qs[j])] + 1.0e-4
        return lo, hi

    def _locked_q(self):
        q, qs = self.model.joint_q.numpy(), self.model.joint_q_start.numpy()
        controlled = {f"DexforceW1V021/{n}" for n in (*self.LEFT_ARM, *self.RIGHT_ARM)}
        ids = [int(qs[j]) for j, label in enumerate(self.model.joint_label) if label not in controlled]
        return wp.array(ids, dtype=wp.int32, device=self.device), wp.array(
            [q[i] for i in ids], dtype=wp.float32, device=self.device
        )

    def _hand_q(self):
        q, qs = self.model.joint_q.numpy(), self.model.joint_q_start.numpy()
        ids = []
        open_q = []
        grasp = []
        targets = {"HAND_THUMB2": 0.84, "HAND_THUMB1": 0.46, "HAND_INDEX": 0.70, "INDEX_PIP": 0.90}
        for side in ("LEFT", "RIGHT"):
            for suffix in self.HAND_SUFFIXES:
                j = self._joint_index(f"{side}_{suffix}")
                i = int(qs[j])
                ids.append(i)
                open_q.append(q[i])
                grasp.append(targets.get(suffix, q[i]))
        return (
            wp.array(ids, dtype=wp.int32, device=self.device),
            wp.array(open_q, dtype=wp.float32, device=self.device),
            wp.array(grasp, dtype=wp.float32, device=self.device),
        )

    def _joint_index(self, name):
        return next(i for i, label in enumerate(self.model.joint_label) if label.endswith("/" + name))

    @staticmethod
    def _body_index(labels, name):
        return next(i for i, label in enumerate(labels) if label.endswith("/" + name))

    def _tcp(self, state, body):
        tf = wp.transform(*state.body_q.numpy()[body])
        return wp.transform(
            wp.transform_get_translation(tf) + wp.quat_rotate(wp.transform_get_rotation(tf), TCP_OFFSET),
            wp.transform_get_rotation(tf),
        )

    def _sample(self, time):
        t = time
        for d, la, lb, ra, rb, ga, gb in self.segments:
            if t <= d:
                a = float(np.clip(t / d, 0.0, 1.0))
                return self._lerp_tf(la, lb, a), self._lerp_tf(ra, rb, a), ga * (1 - a) + gb * a
            t -= d
        _, _, lb, _, rb, _, g = self.segments[-1]
        return lb, rb, g

    def _world_vec(self, v):
        r = wp.quat_rotate(self.base_rot, v)
        return wp.vec3(
            float(r[0]) + float(self.base_pos[0]),
            float(r[1]) + float(self.base_pos[1]),
            float(r[2]) + float(self.base_pos[2]),
        )

    def _world_tf(self, tf):
        return wp.transform(
            self._world_vec(wp.transform_get_translation(tf)),
            self._quat_mul(self.base_rot, wp.transform_get_rotation(tf)),
        )

    @staticmethod
    def _normal_quat(q):
        a = np.array([float(q[0]), float(q[1]), float(q[2]), float(q[3])])
        a /= max(np.linalg.norm(a), 1.0e-8)
        return wp.quat(*a)

    @staticmethod
    def _quat_mul(a, b):
        ax, ay, az, aw = map(float, a)
        bx, by, bz, bw = map(float, b)
        return wp.quat(
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    @staticmethod
    def _v4(q):
        return wp.vec4(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    @staticmethod
    def _lerp_tf(a, b, t):
        pa = np.array(wp.transform_get_translation(a))
        pb = np.array(wp.transform_get_translation(b))
        qa = np.array(wp.transform_get_rotation(a))
        qb = np.array(wp.transform_get_rotation(b))
        if np.dot(qa, qb) < 0:
            qb = -qb
        qa /= np.linalg.norm(qa)
        qb /= np.linalg.norm(qb)
        dot = float(np.clip(np.dot(qa, qb), -1.0, 1.0))
        if dot > 0.9995:
            q = qa * (1.0 - t) + qb * t
            q /= np.linalg.norm(q)
        else:
            theta = np.arccos(dot)
            sin_theta = np.sin(theta)
            q = qa * (np.sin((1.0 - t) * theta) / sin_theta) + qb * (np.sin(t * theta) / sin_theta)
        p = pa * (1 - t) + pb * t
        return wp.transform(wp.vec3(*p), wp.quat(*q))

    def _attach_house_usd(self):
        if not self.house_visual_usd or not hasattr(self.viewer, "stage"):
            return
        if not os.path.isfile(self.house_visual_usd):
            print(f"WAIC house USD not found; continuing without it: {self.house_visual_usd}")
            return
        prim = self.viewer.stage.DefinePrim("/root/waic_house_background", "Xform")
        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(os.path.abspath(self.house_visual_usd))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=900)
        parser.add_argument(
            "--robot-urdf", default=None, help="Optional Dexforce W1 URDF; defaults to the ignored tablecloth asset."
        )
        parser.add_argument(
            "--house-visual-usd", default=DEFAULT_HOUSE_USD, help="Optional WAIC USD reference; it is visual-only."
        )
        parser.add_argument("--show-physics-table", action="store_true")
        parser.add_argument("--enable-self-collisions", action="store_true")
        parser.add_argument("--trajectory-time-scale", type=float, default=4.0)
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture fixed-target replay and physics as CUDA graphs.",
        )
        parser.add_argument("--waic-robot-base-x", type=float, default=float(WAIC_ROBOT_BASE_POS[0]))
        parser.add_argument("--waic-robot-base-y", type=float, default=float(WAIC_ROBOT_BASE_POS[1]))
        parser.add_argument("--waic-robot-base-z", type=float, default=float(WAIC_ROBOT_BASE_POS[2]))
        parser.add_argument("--waic-robot-base-qx", type=float, default=float(WAIC_ROBOT_BASE_QUAT[0]))
        parser.add_argument("--waic-robot-base-qy", type=float, default=float(WAIC_ROBOT_BASE_QUAT[1]))
        parser.add_argument("--waic-robot-base-qz", type=float, default=float(WAIC_ROBOT_BASE_QUAT[2]))
        parser.add_argument("--waic-robot-base-qw", type=float, default=float(WAIC_ROBOT_BASE_QUAT[3]))
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

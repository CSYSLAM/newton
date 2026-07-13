# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MPM W1 Scoop
#
# A Dexforce W1 robot (single right arm, full dexterous hand URDF) scoops
# granular material (rice/sand) off the table with a spoon-shaped
# end-effector and pours it into a bowl. The robot is driven kinematically
# from an IK-solved wrist-TCP keyframe trajectory (approach -> scoop -> lift
# -> travel -> pour), exactly as in the bimanual fold-cloth demos. It acts
# as a one-way kinematic boundary for the MPM particles: particles are pushed
# by the spoon/arm/table/bowl but do not push the robot back.
#
# Command: python -m newton.examples mpm_w1_scoop
#
###########################################################################

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverImplicitMPM


@wp.kernel
def lock_joint_q_kernel(
    joint_q: wp.array2d[wp.float32],
    locked_q_indices: wp.array[wp.int32],
    locked_q_values: wp.array[wp.float32],
):
    i = wp.tid()
    joint_q[0, locked_q_indices[i]] = locked_q_values[i]


@wp.kernel
def copy_ik_to_joint_q_kernel(
    ik_joint_q: wp.array2d[wp.float32],
    joint_q_out: wp.array[wp.float32],
):
    i = wp.tid()
    joint_q_out[i] = ik_joint_q[0, i]


@wp.kernel
def interpolate_joint_positions_kernel(
    joint_q_start: wp.array[wp.float32],
    joint_q_end: wp.array[wp.float32],
    alpha: float,
    joint_q_out: wp.array[wp.float32],
):
    i = wp.tid()
    joint_q_out[i] = joint_q_start[i] * (1.0 - alpha) + joint_q_end[i] * alpha


@wp.kernel
def update_joint_velocity_kernel(
    joint_q_prev: wp.array[wp.float32],
    joint_q_next: wp.array[wp.float32],
    inv_dt: float,
    joint_qd: wp.array[wp.float32],
):
    i = wp.tid()
    joint_qd[i] = (joint_q_next[i] - joint_q_prev[i]) * inv_dt


# --------------------------------------------------------------------------
# Scene constants
# --------------------------------------------------------------------------

# Robot + table layout is identical to the bimanual fold-cloth demos
# (TABLE_POS=(0.55,0,1.15), robot base at origin) so the right arm's natural
# workspace lines up with the table.
TABLE_POS = wp.vec3(0.55, 0.0, 1.15)
TABLE_HALF_EXTENTS = (0.26, 0.62, 0.025)
TABLE_TOP_Z = float(TABLE_POS[2]) + TABLE_HALF_EXTENTS[2]  # 1.175

# Granular pile on the table, on the robot's right (-y) side where the right
# arm reaches in to scoop.
PILE_CENTER = wp.vec3(0.40, -0.50, TABLE_TOP_Z)
PILE_HALF = (0.07, 0.07, 0.03)  # half extents of the initial particle block

# Bowl: a small tray of four thin walls + floor, on the table. Boxes (not a
# mesh) keep the collider rasterization robust.
BOWL_CENTER = wp.vec3(0.50, -0.15, TABLE_TOP_Z)
BOWL_INNER_HALF = (0.055, 0.055, 0.025)
WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.045

# Spoon: a shallow box attached to the right wrist (right_j7). The TCP offset
# matches the fold-cloth demos' wrist offset along the tool axis.
SPOON_HALF = (0.05, 0.035, 0.004)  # length(x), width(y), half-thickness(z)
TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)

VOXEL_SIZE = 0.012  # ~1.2 cm cells; spoon is several voxels across
PARTICLES_PER_CELL = 3.0
SAND_DENSITY = 1500.0

# Right-arm seed pose: a folded "ready" stance over the table so the IK solver
# starts inside its workspace instead of the URDF T-pose.
RIGHT_ARM_HOME_Q = (0.0, 0.6, 0.0, 1.0, 0.0, 0.4, 0.0)

# Script timing (seconds). Each phase interpolates the wrist TCP between two
# transforms; the pour phase also tilts the spoon.
HOME_HOLD_TIME = 0.6
APPROACH_TIME = 1.6
SCOOP_TIME = 1.8
LIFT_TIME = 1.2
TRAVEL_TIME = 1.6
POUR_TIME = 1.8
RETURN_TIME = 1.4
END_HOLD_TIME = 0.8
SCRIPT_END_TIME = (
    HOME_HOLD_TIME
    + APPROACH_TIME
    + SCOOP_TIME
    + LIFT_TIME
    + TRAVEL_TIME
    + POUR_TIME
    + RETURN_TIME
    + END_HOLD_TIME
)

CAMERA_POS = wp.vec3(1.35, -1.55, 1.55)
CAMERA_PITCH = -18.0
CAMERA_YAW = 55.0


class Example:
    """W1 single-arm scoop-and-pour over an MPM granular pile."""

    # Right arm joints are IK-controlled; everything else is locked at home.
    RIGHT_ARM_JOINTS = (
        "RIGHT_J1",
        "RIGHT_J2",
        "RIGHT_J3",
        "RIGHT_J4",
        "RIGHT_J5",
        "RIGHT_J6",
        "RIGHT_J7",
    )

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.trajectory_time_scale = float(args.trajectory_time_scale)
        self.device = wp.get_device()

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.default_joint_cfg.armature = 0.02
        builder.default_shape_cfg.ke = 5.0e5
        builder.default_shape_cfg.kd = 1.0e-6
        builder.default_shape_cfg.mu = 2.0

        # --- Robot (full dexterous-hand W1, same URDF as the fold demos) ---
        urdf_path = Path(__file__).resolve().parents[1] / "cloth" / "DexforceW1V021" / "DexforceW1V021.urdf"
        builder.add_urdf(
            urdf_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        # Seed the right arm with a folded "ready" pose over the table so the
        # IK solver starts inside its workspace (the URDF T-pose reaches far
        # past the table and IK cannot converge from there).
        for j, value in enumerate(RIGHT_ARM_HOME_Q, start=1):
            joint_idx = self._builder_joint_index(builder, f"RIGHT_J{j}")
            builder.joint_q[builder.joint_q_start[joint_idx]] = float(value)

        self.robot_shape_end = builder.shape_count
        self._configure_robot(builder)

        # --- Spoon (a shallow cup) attached to the right wrist (right_j7) ---
        self.right_ee_index = self._builder_body_index(builder, "right_j7")
        self._add_spoon(builder)

        # --- Table ---
        table_cfg = newton.ModelBuilder.ShapeConfig()
        table_cfg.ke = 5.0e5
        table_cfg.kd = 1.0e-6
        table_cfg.mu = 0.5
        table_cfg.has_particle_collision = True
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(TABLE_POS, wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
        )

        # --- Bowl (floor + four walls) ---
        self._add_bowl(builder)

        # --- Granular particles ---
        SolverImplicitMPM.register_custom_attributes(builder)
        self._emit_particles(builder)

        builder.add_ground_plane()

        self.model = builder.finalize(requires_grad=False)
        self.model.set_gravity((0.0, 0.0, -9.81))

        # Only the spoon + table + bowl collide with particles; robot body off.
        self._configure_particle_contacts()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # --- IK for the right wrist TCP ---
        self.right_tf = self._current_tcp_transform(self.right_ee_index)
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count), device=self.device)
        self.frame_joint_q_start = wp.zeros_like(self.model.joint_q)
        self.frame_joint_q_end = wp.zeros_like(self.model.joint_q)
        self.locked_q_indices, self.locked_q_values = self._build_locked_joint_arrays()
        self.setup_ik()

        # Bake the "ready" keyframe into the initial state so the arm starts
        # exactly on trajectory (otherwise frame 0 demands a large IK jump the
        # arm cannot follow in one step).
        self._initialize_to_ready_pose()

        # --- MPM solver ---
        # The robot arm is driven kinematically (joint angles set directly from
        # the IK trajectory and recomputed via forward kinematics each substep),
        # so no rigid-body solver is needed. The robot/table/bowl act as
        # kinematic colliders for the particles (one-way coupling).
        mpm_options = SolverImplicitMPM.Config()
        mpm_options.voxel_size = VOXEL_SIZE
        mpm_options.tolerance = 1.0e-6
        mpm_options.transfer_scheme = "pic"
        mpm_options.grid_type = "sparse"
        mpm_options.strain_basis = "P0"
        mpm_options.max_iterations = 50
        mpm_options.critical_fraction = 0.0
        mpm_options.air_drag = 1.0
        mpm_options.collider_velocity_mode = "backward"
        self.mpm_solver = SolverImplicitMPM(self.model, mpm_options)

        # One-way coupling: robot/table/bowl are kinematic boundaries for MPM.
        self.mpm_solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
            body_q=self.state_0.body_q,
        )

        # Per-particle material: dry granular (rice/sand) — Drucker-Prager.
        self._init_materials()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(CAMERA_POS, CAMERA_PITCH, CAMERA_YAW)
        self.viewer.show_particles = True

        self.particle_colors = wp.full(
            self.model.particle_count, value=wp.vec3(0.85, 0.78, 0.55), dtype=wp.vec3, device=self.device
        )

        self.capture()

    # ------------------------------------------------------------------
    # Scene construction helpers
    # ------------------------------------------------------------------

    def _add_spoon(self, builder: newton.ModelBuilder) -> None:
        """Attach a shallow cup-shaped spoon to the right wrist.

        Built from a thin floor plus four low walls in the wrist local frame.
        The cup opens along the wrist's tool axis (local -x, the same
        direction as TCP_OFFSET) so that as the arm reaches toward a target
        the cup faces the direction of motion and can gather material, then
        hold it when lifted. Centered on TCP_OFFSET.
        """
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.ke = 5.0e5
        cfg.kd = 1.0e-3
        cfg.mu = 0.6
        cfg.has_particle_collision = True

        hx, hy = SPOON_HALF[0], SPOON_HALF[1]  # 0.05, 0.035
        hz = SPOON_HALF[2]  # 0.004
        t = 0.004  # wall/floor thickness
        depth = 0.025  # cup interior depth
        cx, cy, cz = float(TCP_OFFSET[0]), float(TCP_OFFSET[1]), float(TCP_OFFSET[2])

        # Floor of the cup, perpendicular to local x (the tool axis). In local
        # frame the cup extends from the floor toward -x (the opening).
        builder.add_shape_box(
            body=self.right_ee_index,
            xform=wp.transform(wp.vec3(cx + t * 0.5, cy, cz), wp.quat_identity()),
            hx=t * 0.5,
            hy=hy,
            hz=hz,
            cfg=cfg,
            color=(0.8, 0.8, 0.85),
        )
        # Four walls around the opening (local y/z plane), height = depth.
        wall_h = depth * 0.5
        wall_x = cx - wall_h  # walls sit toward the opening (-x)
        walls = [
            (wall_x, cy + hy, cz, wall_h, t * 0.5, hz),
            (wall_x, cy - hy, cz, wall_h, t * 0.5, hz),
            (wall_x, cy, cz + hz, wall_h, hy, t * 0.5),
            (wall_x, cy, cz - hz, wall_h, hy, t * 0.5),
        ]
        for wx, wy, wz, whx, why, whz in walls:
            builder.add_shape_box(
                body=self.right_ee_index,
                xform=wp.transform(wp.vec3(wx, wy, wz), wp.quat_identity()),
                hx=whx,
                hy=why,
                hz=whz,
                cfg=cfg,
                color=(0.8, 0.8, 0.85),
            )

    def _add_bowl(self, builder: newton.ModelBuilder) -> None:
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.ke = 5.0e5
        cfg.kd = 1.0e-3
        cfg.mu = 0.4
        cfg.has_particle_collision = True
        cx, cy, cz = float(BOWL_CENTER[0]), float(BOWL_CENTER[1]), float(BOWL_CENTER[2])
        hx, hy, hz = BOWL_INNER_HALF
        t = WALL_THICKNESS
        # Floor of the tray
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(wp.vec3(cx, cy, cz - hz + t * 0.5), wp.quat_identity()),
            hx=hx + t,
            hy=hy + t,
            hz=t * 0.5,
            cfg=cfg,
            color=(0.7, 0.55, 0.4),
        )
        walls = [
            (cx, cy + hy + t * 0.5, cz + WALL_HEIGHT * 0.5, hx + t, t * 0.5, WALL_HEIGHT * 0.5),
            (cx, cy - hy - t * 0.5, cz + WALL_HEIGHT * 0.5, hx + t, t * 0.5, WALL_HEIGHT * 0.5),
            (cx + hx + t * 0.5, cy, cz + WALL_HEIGHT * 0.5, t * 0.5, hy, WALL_HEIGHT * 0.5),
            (cx - hx - t * 0.5, cy, cz + WALL_HEIGHT * 0.5, t * 0.5, hy, WALL_HEIGHT * 0.5),
        ]
        for wx, wy, wz, whx, why, whz in walls:
            builder.add_shape_box(
                body=-1,
                xform=wp.transform(wp.vec3(wx, wy, wz), wp.quat_identity()),
                hx=whx,
                hy=why,
                hz=whz,
                cfg=cfg,
                color=(0.7, 0.55, 0.4),
            )

    def _emit_particles(self, builder: newton.ModelBuilder) -> None:
        # Start the pile a little above the table top so no particle is
        # initialized inside the table box (embedded particles cause the MPM
        # collider to project them violently and destabilize the solve).
        pile_lo_z = float(PILE_CENTER[2]) + 0.5 * VOXEL_SIZE
        lo = np.array(
            [
                float(PILE_CENTER[0]) - PILE_HALF[0],
                float(PILE_CENTER[1]) - PILE_HALF[1],
                pile_lo_z,
            ]
        )
        hi = np.array(
            [
                float(PILE_CENTER[0]) + PILE_HALF[0],
                float(PILE_CENTER[1]) + PILE_HALF[1],
                pile_lo_z + 2.0 * PILE_HALF[2],
            ]
        )
        res = np.array(np.ceil(PARTICLES_PER_CELL * (hi - lo) / VOXEL_SIZE), dtype=int)
        cell_size = (hi - lo) / res
        cell_volume = float(np.prod(cell_size))
        radius = float(np.max(cell_size) * 0.5)
        mass = cell_volume * SAND_DENSITY

        builder.add_particle_grid(
            pos=wp.vec3(lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=int(res[0]) + 1,
            dim_y=int(res[1]) + 1,
            dim_z=int(res[2]) + 1,
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
            custom_attributes={"mpm:friction": 0.5},
        )

    def _configure_robot(self, builder: newton.ModelBuilder) -> None:
        # PD gains are present for completeness; the arm is driven kinematically
        # (joint_q set directly from the IK trajectory), so these are not used
        # to track the trajectory.
        for i in range(builder.joint_dof_count):
            builder.joint_target_pos[i] = builder.joint_q[i]
            builder.joint_target_ke[i] = 650.0
            builder.joint_target_kd[i] = 65.0
            builder.joint_armature[i] = 0.02

    def _configure_particle_contacts(self) -> None:
        flags = self.model.shape_flags.numpy()
        particle_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        # Robot body shapes do NOT collide with particles; only the spoon
        # (added after robot_shape_end), table and bowl do.
        flags[: self.robot_shape_end] &= ~particle_flag
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.device)

    # ------------------------------------------------------------------
    # IK setup
    # ------------------------------------------------------------------

    def setup_ik(self) -> None:
        tcp = self._current_tcp_transform(self.right_ee_index)
        self.right_pos_obj = ik.IKObjectivePosition(
            link_index=self.right_ee_index,
            link_offset=TCP_OFFSET,
            target_positions=wp.array([wp.transform_get_translation(tcp)], dtype=wp.vec3, device=self.device),
        )
        # Rotation target: keep the cup opening (wrist local -x) roughly up
        # (world +z) so the spoon can hold material when lifted. Low weight so
        # position tracking is prioritized (the arm has limited reach to the
        # table while keeping the cup perfectly upright).
        self.right_rot_obj = ik.IKObjectiveRotation(
            link_index=self.right_ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([self._quat_to_vec4(self._cup_up_quat())], dtype=wp.vec4, device=self.device),
            weight=0.3,
        )
        lower, upper = self._joint_limits_with_locked_dofs()
        self.joint_limits_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=wp.array(lower, dtype=wp.float32, device=self.device),
            joint_limit_upper=wp.array(upper, dtype=wp.float32, device=self.device),
            weight=5.0,
        )
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.right_pos_obj, self.right_rot_obj, self.joint_limits_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 32

    @staticmethod
    def _cup_up_quat() -> wp.quat:
        """Wrist orientation that points the cup opening (local -x) up (world +z).

        Maps local -x -> world +z, and local +y -> world +y (arbitrary roll).
        """
        # local -x -> world +z  =>  local +x -> world -z
        # Build basis: local x -> (0,0,-1), local y -> (0,1,0), local z -> (1,0,0)
        return Example._basis_quat(
            local_x=wp.vec3(0.0, 0.0, -1.0),
            local_y=wp.vec3(0.0, 1.0, 0.0),
            local_z=wp.vec3(1.0, 0.0, 0.0),
        )

    def _initialize_to_ready_pose(self) -> None:
        """Solve the first (ready) keyframe and bake it into the initial state."""
        ready_tf = self._build_motion_segments()[0][1]
        self.right_pos_obj.set_target_position(0, wp.transform_get_translation(ready_tf))
        self.right_rot_obj.set_target_rotation(0, self._quat_to_vec4(self._cup_up_quat()))
        for _ in range(12):
            self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            lock_joint_q_kernel,
            dim=self.locked_q_indices.shape[0],
            inputs=[self.ik_joint_q, self.locked_q_indices, self.locked_q_values],
            device=self.device,
        )
        wp.launch(
            copy_ik_to_joint_q_kernel,
            dim=self.model.joint_coord_count,
            inputs=[self.ik_joint_q],
            outputs=[self.state_0.joint_q],
            device=self.device,
        )
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def _joint_limits_with_locked_dofs(self) -> tuple[np.ndarray, np.ndarray]:
        lower = self.model.joint_limit_lower.numpy().copy()
        upper = self.model.joint_limit_upper.numpy().copy()
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        controlled = self._controlled_joint_labels()

        for joint_idx, label in enumerate(self.model.joint_label):
            if label in controlled:
                continue
            q_idx = int(q_start[joint_idx])
            dof_idx = int(qd_start[joint_idx])
            lower[dof_idx] = q_home[q_idx] - 1.0e-4
            upper[dof_idx] = q_home[q_idx] + 1.0e-4
        return lower, upper

    def _build_locked_joint_arrays(self) -> tuple[wp.array, wp.array]:
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        controlled = self._controlled_joint_labels()
        idx, val = [], []
        for joint_idx, label in enumerate(self.model.joint_label):
            if label in controlled:
                continue
            q_idx = int(q_start[joint_idx])
            idx.append(q_idx)
            val.append(float(q_home[q_idx]))
        return (
            wp.array(idx, dtype=wp.int32, device=self.device),
            wp.array(val, dtype=wp.float32, device=self.device),
        )

    def _controlled_joint_labels(self) -> set[str]:
        return {f"DexforceW1V021/{name}" for name in self.RIGHT_ARM_JOINTS}

    # ------------------------------------------------------------------
    # Trajectory script
    # ------------------------------------------------------------------

    def _build_motion_segments(self):
        """Right-hand wrist TCP keyframes (world space).

        Each segment: (duration, tf_start, tf_end). All targets lie on the
        robot's right (-y) side within the right-arm workspace. The spoon
        orientation is authored in world space: "bowl up" keeps the concave
        side facing +z with the tip toward -y; "pour" rotates about the spoon
        long axis to tip toward the bowl.
        """
        # Bowl-up orientation: link local x (tool / spoon tip) -> world -y,
        # local y -> world -x, local z (bowl normal) -> world +z.
        q_bowl_up = self._basis_quat(
            local_x=wp.vec3(0.0, -1.0, 0.0),
            local_y=wp.vec3(-1.0, 0.0, 0.0),
            local_z=wp.vec3(0.0, 0.0, 1.0),
        )
        # Pour: tip ~110 deg about the spoon long axis (world -y here) so the
        # bowl faces +x (toward the bowl at +x).
        q_pour = wp.quat_from_axis_angle(wp.vec3(0.0, -1.0, 0.0), float(np.radians(110.0))) * q_bowl_up

        px, py = float(PILE_CENTER[0]), float(PILE_CENTER[1])
        bx, by = float(BOWL_CENTER[0]), float(BOWL_CENTER[1])

        ready = wp.transform(wp.vec3(0.45, -0.35, TABLE_TOP_Z + 0.22), q_bowl_up)
        approach = wp.transform(wp.vec3(px, py + 0.08, TABLE_TOP_Z + 0.18), q_bowl_up)
        # Scoop: plunge the cup into the top of the pile (cup stays roughly
        # upright) and shift it forward (-y) to gather material.
        scoop = wp.transform(wp.vec3(px, py + 0.02, TABLE_TOP_Z + 0.06), q_bowl_up)
        lift = wp.transform(wp.vec3(px, py - 0.02, TABLE_TOP_Z + 0.22), q_bowl_up)
        travel = wp.transform(wp.vec3(bx, by, TABLE_TOP_Z + 0.24), q_bowl_up)
        pour = wp.transform(wp.vec3(bx, by, TABLE_TOP_Z + 0.14), q_pour)
        ret = wp.transform(wp.vec3(0.45, -0.35, TABLE_TOP_Z + 0.26), q_bowl_up)

        return (
            (HOME_HOLD_TIME, ready, ready),
            (APPROACH_TIME, ready, approach),
            (SCOOP_TIME, approach, scoop),
            (LIFT_TIME, scoop, lift),
            (TRAVEL_TIME, lift, travel),
            (POUR_TIME, travel, pour),
            (RETURN_TIME, pour, ret),
            (END_HOLD_TIME, ret, ret),
        )

    def _sample_script(self, query_time: float) -> wp.transform:
        remaining = query_time
        for duration, tf_start, tf_end in self._build_motion_segments():
            if remaining <= duration:
                alpha = float(np.clip(remaining / max(duration, 1.0e-6), 0.0, 1.0))
                return self._interpolate_transform(tf_start, tf_end, alpha)
            remaining -= duration
        return self._build_motion_segments()[-1][2]

    # ------------------------------------------------------------------
    # Per-frame simulation
    # ------------------------------------------------------------------

    def _prepare_frame_targets(self) -> None:
        wp.copy(self.frame_joint_q_start, self.state_0.joint_q)
        script_time = (self.sim_time + self.frame_dt) * self.trajectory_time_scale
        self.right_tf = self._sample_script(script_time)

        self.right_pos_obj.set_target_position(0, wp.transform_get_translation(self.right_tf))
        self.right_rot_obj.set_target_rotation(0, self._quat_to_vec4(self._cup_up_quat()))
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            lock_joint_q_kernel,
            dim=self.locked_q_indices.shape[0],
            inputs=[self.ik_joint_q, self.locked_q_indices, self.locked_q_values],
            device=self.device,
        )
        wp.launch(
            copy_ik_to_joint_q_kernel,
            dim=self.model.joint_coord_count,
            inputs=[self.ik_joint_q],
            outputs=[self.frame_joint_q_end],
            device=self.device,
        )

    def simulate(self) -> None:
        self._prepare_frame_targets()

        # The robot is driven kinematically: each substep we interpolate the
        # IK-solved joint trajectory and recompute body transforms via forward
        # kinematics. Tracking is exact; the arm serves purely as a moving
        # collider boundary for the MPM particles (one-way coupling).
        for substep in range(self.sim_substeps):
            substep_alpha = float((substep + 1) / self.sim_substeps)
            wp.launch(
                interpolate_joint_positions_kernel,
                dim=self.model.joint_coord_count,
                inputs=[self.frame_joint_q_start, self.frame_joint_q_end, substep_alpha],
                outputs=[self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                update_joint_velocity_kernel,
                dim=self.model.joint_dof_count,
                inputs=[self.frame_joint_q_start, self.state_0.joint_q, 1.0 / self.sim_dt],
                outputs=[self.state_0.joint_qd],
                device=self.device,
            )
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # MPM step (in-place). Robot's current body_q is the kinematic collider.
        self.mpm_solver.step(self.state_0, self.state_0, contacts=None, control=None, dt=self.frame_dt)

    def capture(self):
        # Graph capture is disabled: the MPM sparse grid is reallocated each
        # step, which cannot run inside a captured stream.
        self.graph = None

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------

    def _init_materials(self) -> None:
        m = self.model.mpm
        # Dry granular: stiff, frictional, finite compressive yield.
        m.young_modulus.fill_(1.0e7)
        m.poisson_ratio.fill_(0.3)
        m.friction.fill_(0.5)
        m.damping.fill_(0.0)
        m.yield_pressure.fill_(5.0e4)
        m.tensile_yield_ratio.fill_(0.0)  # no tensile cohesion -> free-flowing
        m.yield_stress.fill_(0.0)
        m.hardening.fill_(0.0)
        m.dilatancy.fill_(0.0)
        m.viscosity.fill_(0.0)
        self.state_0.mpm.particle_Jp.fill_(1.0)

    # ------------------------------------------------------------------
    # Rendering / tests
    # ------------------------------------------------------------------

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "/mpm_scoop/grains",
            points=self.state_0.particle_q,
            radii=self.model.particle_radius,
            colors=self.particle_colors,
            hidden=not self.viewer.show_particles,
        )
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_particle_state(
            self.state_0,
            "all particles have finite positions",
            lambda q, qd: wp.length(q) < 1.0e6,
        )
        newton.examples.test_particle_state(
            self.state_0,
            "all particles are above the ground",
            lambda q, qd: q[2] > -VOXEL_SIZE,
        )

    # ------------------------------------------------------------------
    # Small transform helpers
    # ------------------------------------------------------------------

    def _current_tcp_transform(self, body_index: int) -> wp.transform:
        body_q_np = self.state_0.body_q.numpy()
        body_tf = wp.transform(*body_q_np[body_index])
        pos = wp.transform_get_translation(body_tf)
        rot = wp.transform_get_rotation(body_tf)
        tcp_pos = pos + wp.quat_rotate(rot, TCP_OFFSET)
        return wp.transform(tcp_pos, rot)

    def _interpolate_transform(self, tf_a: wp.transform, tf_b: wp.transform, alpha: float) -> wp.transform:
        pos_a = self._vec3_to_np(wp.transform_get_translation(tf_a))
        pos_b = self._vec3_to_np(wp.transform_get_translation(tf_b))
        quat_a = self._quat_to_np(wp.transform_get_rotation(tf_a))
        quat_b = self._quat_to_np(wp.transform_get_rotation(tf_b))
        pos = pos_a * (1.0 - alpha) + pos_b * alpha
        quat = self._slerp_quat_xyzw(quat_a, quat_b, alpha)
        return wp.transform(wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])), wp.quat(*quat.tolist()))

    def _slerp_quat_xyzw(self, qa: np.ndarray, qb: np.ndarray, alpha: float) -> np.ndarray:
        qa = self._normalize_quat(qa)
        qb = self._normalize_quat(qb)
        dot = float(np.dot(qa, qb))
        if dot < 0.0:
            qb = -qb
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            return self._normalize_quat(qa * (1.0 - alpha) + qb * alpha)
        theta_0 = np.arccos(dot)
        sin_t0 = np.sin(theta_0)
        theta = theta_0 * alpha
        return qa * (np.sin(theta_0 - theta) / sin_t0) + qb * (np.sin(theta) / sin_t0)

    def _normalize_quat(self, q: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(q))
        if n == 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return q / n

    @staticmethod
    def _basis_quat(local_x: wp.vec3, local_y: wp.vec3, local_z: wp.vec3) -> wp.quat:
        """Build a quaternion mapping the link local frame to the given world axes."""
        R = np.array(
            [
                [float(local_x[0]), float(local_y[0]), float(local_z[0])],
                [float(local_x[1]), float(local_y[1]), float(local_z[1])],
                [float(local_x[2]), float(local_y[2]), float(local_z[2])],
            ],
            dtype=np.float64,
        )
        t = float(R[0, 0] + R[1, 1] + R[2, 2])
        if t > 0.0:
            s = np.sqrt(t + 1.0) * 2.0
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
        q = np.array([qx, qy, qz, qw], dtype=np.float64)
        q /= np.linalg.norm(q)
        return wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def _quat_to_vec4(self, q: wp.quat) -> wp.vec4:
        return wp.vec4(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def _quat_to_np(self, q: wp.quat) -> np.ndarray:
        return np.array([float(q[0]), float(q[1]), float(q[2]), float(q[3])], dtype=np.float64)

    def _vec3_to_np(self, v: wp.vec3) -> np.ndarray:
        return np.array([float(v[0]), float(v[1]), float(v[2])], dtype=np.float64)

    def _builder_body_index(self, builder: newton.ModelBuilder, name: str) -> int:
        suffix = f"/{name}"
        return next(i for i, label in enumerate(builder.body_label) if label.endswith(suffix))

    def _builder_joint_index(self, builder: newton.ModelBuilder, name: str) -> int:
        suffix = f"/{name}"
        return next(i for i, label in enumerate(builder.joint_label) if label.endswith(suffix))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--trajectory-time-scale",
            type=float,
            default=1.0,
            help="Multiplier on the trajectory time (>1 replays faster).",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

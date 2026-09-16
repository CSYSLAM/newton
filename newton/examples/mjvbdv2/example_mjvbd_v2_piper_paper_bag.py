# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pack a deformable paper shopping bag and lift it with two PiPER arms.

uv run -m newton.examples mjvbd_v2_piper_paper_bag
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
from newton.examples.mjvbdv2.support.paper_bag_asset import (
    ASSET_DIR,
    add_paper_bag,
    deform_render_mesh,
    make_paper_bag,
    measure_paper_shape,
    parcel_is_contained,
)
from newton.solvers import SolverMJVBDV2

TABLE = 0.555
BAG_POSITION = np.array((0.0, 0.22, TABLE + 0.003), dtype=np.float32)
PACK_START = 2.5
PARCEL_DURATION = 8.0
HANDLE_START = PACK_START + PARCEL_DURATION
LIFT_START = HANDLE_START + 5.0
# The jaw gaps must accommodate both the material and its collision envelope.
# The right silicone pads project 2 mm into the gap from each side.
SUPPORT_OPENING = 0.0033
HANDLE_OPENINGS = (0.0045, 0.0060)


def smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.kernel
def prescribe(
    start: wp.array[float],
    end: wp.array[float],
    qs: wp.array[int],
    qds: wp.array[int],
    alpha: float,
    q: wp.array[float],
    qd: wp.array[float],
):
    j = wp.tid()
    if qs[j + 1] > qs[j]:
        i = qs[j]
        q[i] = wp.lerp(start[i], end[i], alpha)
        qd[qds[j]] = 60.0 * (end[i] - start[i])


class Example:
    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        self.frame, self.sim_time = 0, 0.0
        if args.substeps < 1 or args.iterations < 1:
            raise ValueError("Substeps and iterations must be positive")
        self.dt = 1.0 / (60 * args.substeps)
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.004
        source = Path(__file__).resolve().parents[3] / "assets/piper_bag/piper/piper.xml"
        self.arms = []
        for side in (-1, 1):
            body_start, coord_start = builder.body_count, len(builder.joint_q)
            builder.add_mjcf(
                str(source),
                xform=wp.transform(
                    wp.vec3(side * 0.36, 0.20, TABLE),
                    wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.0 if side < 0 else math.pi),
                ),
                floating=False,
                enable_self_collisions=False,
                collapse_fixed_joints=False,
            )
            self.arms.append({"bodies": range(body_start, builder.body_count), "start": coord_start})
        self.robot_joints = builder.joint_count
        self.robot_coords = len(builder.joint_q)
        for i in range(builder.body_count):
            builder.body_flags[i] = int(newton.BodyFlags.KINEMATIC)
        full_surface_shapes = []
        for i in range(builder.shape_count):
            builder.shape_material_mu[i] = 1.0
            builder.shape_material_ke[i] = 2.0e4
            builder.shape_material_kd[i] = 30.0
            if builder.body_label[builder.shape_body[i]].endswith(("/link7", "/link8")):
                builder.shape_material_mu[i] = 1.1
                if builder.shape_body[i] in self.arms[1]["bodies"]:
                    builder.shape_material_ke[i] = 2.0e5
                builder.shape_margin[i] = 0.002
                if builder.shape_flags[i] & int(newton.ShapeFlags.COLLIDE_PARTICLES):
                    builder.shape_source[i].build_sdf(target_voxel_size=0.001)
                    full_surface_shapes.append(i)
        # Match the supermarket gripper's visible silicone fingertip pads.
        # These boxes supply both the rendered surface and physical contact.
        for body in self.arms[1]["bodies"]:
            if builder.body_label[body].endswith(("/link7", "/link8")):
                builder.add_shape_box(
                    body,
                    xform=wp.transform(wp.vec3(0, -0.018, 0), wp.quat_identity()),
                    hx=0.013,
                    hy=0.016,
                    hz=0.002,
                    cfg=newton.ModelBuilder.ShapeConfig(ke=2e5, kd=30, mu=1.1, margin=0.001),
                    color=(0.20, 0.22, 0.23),
                    label="Soft parcel fingertip pad",
                )
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0, 0.12, TABLE - 0.025), wp.quat_identity()),
            hx=0.62,
            hy=0.43,
            hz=0.025,
            cfg=newton.ModelBuilder.ShapeConfig(ke=6e4, kd=100, mu=0.5),
            color=(0.68, 0.7, 0.72),
        )
        for x in (-0.53, 0.53):
            for y in (-0.2, 0.46):
                builder.add_shape_box(
                    -1,
                    xform=wp.transform(wp.vec3(x, y, 0.265), wp.quat_identity()),
                    hx=0.025,
                    hy=0.025,
                    hz=0.24,
                    cfg=newton.ModelBuilder.ShapeConfig(has_particle_collision=False),
                    color=(0.18, 0.2, 0.22),
                )
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(has_particle_collision=False), color=(0.3, 0.32, 0.35)
        )
        self.bag = make_paper_bag()
        add_paper_bag(
            builder,
            self.bag,
            BAG_POSITION,
            membrane_stiffness=args.paper_stiffness,
            bending_stiffness=args.paper_bending,
        )
        center = np.array((0.19, -0.09, TABLE + 0.025), dtype=np.float32)
        begin, tri_begin = builder.particle_count, len(builder.tri_indices)
        builder.add_soft_grid(
            pos=wp.vec3(*(center - 0.022)),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=5,
            dim_y=5,
            dim_z=5,
            cell_x=0.0088,
            cell_y=0.0088,
            cell_z=0.0088,
            density=250,
            k_mu=3e4,
            k_lambda=8e4,
            k_damp=0.5,
            particle_radius=0.002,
            color=(0.75, 0.25, 0.14),
            label="Soft parcel",
        )
        self.parcel_particles = slice(begin, builder.particle_count)
        self.parcel_pick = center.copy()
        parcel_triangles = np.asarray(builder.tri_indices[tri_begin:], dtype=np.int32)
        full_surface_shapes.extend(i for i, kind in enumerate(builder.shape_type) if kind == newton.GeoType.BOX)
        builder.color(include_bending=True)
        self.model = builder.finalize()
        # Resolve the gripper contact against the volumetric parcel stiffness.
        self.model.soft_contact_ke, self.model.soft_contact_kd, self.model.soft_contact_mu = 2e5, 10.0, 0.4
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        self.frame_start, self.frame_end = wp.clone(self.model.joint_q), wp.clone(self.model.joint_q)
        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=(0, 1),
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": args.iterations,
                "rigid_soft_enable_dat": True,
                "friction_epsilon": 0.001,
                "particle_enable_multilevel_correction": True,
                "particle_multilevel_operator": "galerkin",
                "particle_multilevel_cluster_size": 8,
                "particle_multilevel_coarse_iterations": 32,
                "particle_multilevel_relaxation": 0.8,
                "particle_multilevel_max_radius_fraction": 2.0,
                "particle_multilevel_checkpoints": tuple(
                    sorted(
                        {max(1, args.iterations // 3), max(1, 2 * args.iterations // 3), max(1, args.iterations - 3)}
                    )
                ),
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": 0.0012,
                "particle_self_contact_margin": 0.003,
                "particle_rest_shape_contact_exclusion_radius": 0.004,
                "particle_topological_contact_filter_threshold": 2,
                "particle_chebyshev_spectral_radius": 0.9,
                "particle_chebyshev_warmup_iterations": 2,
                "particle_chebyshev_polish_iterations": 3,
                "particle_chebyshev_contact_rings": 1,
                "particle_chebyshev_max_radius_fraction": 1.0,
                "rigid_contact_hard": False,
                "rigid_contact_history": False,
                "rigid_body_contact_buffer_size": 128,
                "rigid_body_particle_contact_buffer_size": 8192,
            },
            collision_options={
                "enable_rigid_soft_full_surface_contact": True,
                "rigid_soft_full_surface_shape_indices": full_surface_shapes,
                "soft_contact_margin": 0.004,
                "soft_contact_max": 131072,
                "broad_phase": "nxn",
                "include_static_kinematic_pairs": False,
            },
        )
        self._build_ik()
        self._prepare_left_approach()
        self.parcel_triangles = wp.array(parcel_triangles.ravel(), dtype=int)
        render = np.load(ASSET_DIR / "render_mesh.npz")
        binding = np.load(ASSET_DIR / "render_binding.npz")
        self.render_points = wp.empty(len(render["vertices"]), dtype=wp.vec3)
        self.render_binding = [
            wp.array(binding["indices"], dtype=wp.vec3i),
            wp.array(binding["weights"], dtype=wp.vec3),
            wp.array(binding["offset"], dtype=wp.vec3),
        ]
        parts = json.loads((ASSET_DIR / "render_parts.json").read_text())
        uv = np.column_stack(((render["vertices"][:, 1] + 0.1) / 0.2, render["vertices"][:, 2] / 0.26)).astype(
            np.float32
        )
        self.paper_uv = wp.array(uv, dtype=wp.vec2)
        self.paper_texture = str(ASSET_DIR / "kraft_albedo.png")
        self.render_transform = wp.array([wp.transform_identity()], dtype=wp.transform)
        self.render_scale = wp.array([wp.vec3(1.0)], dtype=wp.vec3)
        self.render_color = wp.array([wp.vec3(1.0)], dtype=wp.vec3)
        self.render_material = wp.array([wp.vec4(0.9, 0.0, 0.0, 1.0)], dtype=wp.vec4)
        self.paper_render_parts = []
        for kind, color in [("paper", (0.43, 0.285, 0.12)), ("cord", (0.27, 0.185, 0.07))]:
            indices = np.concatenate(
                [render["faces"][part["start"] : part["end"]].ravel() for part in parts if part["material"] == kind]
            )
            self.paper_render_parts.append((wp.array(indices, dtype=int), color, kind))
        self.graph = None
        self.viewer.set_model(self.model)
        self.viewer.show_particles, self.viewer.show_triangles = False, False
        self.viewer.set_camera(pos=wp.vec3(1.05, -1.28, 1.30), pitch=-25, yaw=132)
        self.peak_height = 0.0
        self.max_ik_error = 0.0
        self.loaded = False
        self.packed = False
        self.lift_targets = None
        self.support_shape_max = 0.0
        self.support_speed_sq = 0.0
        self.support_speed_samples = 0
        self.lift_height = 0.0
        self.handle_tops = [ids[np.argsort(self.bag.vertices[ids, 2])[-12:]] for ids in self.bag.handles]
        self.handle_grasped = [False, False]
        self.static_reference = None
        self.static_max_motion = 0.0
        self.static_shape = {}
        print("[PaperBag]", self.model.particle_count, "particles;", self.model.tri_count, "triangles", flush=True)

    def _build_ik(self):
        body_q = self.state_0.body_q.numpy()
        objectives = []
        for arm in self.arms:
            ee = next(i for i in arm["bodies"] if self.model.body_label[i].endswith("/gripper_base_left"))
            # Pinch at the distal jaw pads; the whole-finger bounding-box center
            # would put the fingertips almost 4 cm through the tabletop.
            arm["tcp"] = wp.vec3(0.0, 0.0, 0.120 if arm is self.arms[0] else 0.1158)
            tcp_world = np.asarray(wp.transform_point(wp.transform(*body_q[ee]), arm["tcp"]))
            arm["ee"], arm["home"] = ee, tcp_world
            arm["objective"] = ik.IKObjectivePosition(ee, arm["tcp"], wp.array([wp.vec3(*tcp_world)], dtype=wp.vec3))
            objectives.append(arm["objective"])
            arm["home_rotation"] = wp.quat(*body_q[ee, 3:])
            arm["rotation"] = ik.IKObjectiveRotation(
                ee, wp.quat_identity(), wp.array([wp.vec4(*body_q[ee, 3:])], dtype=wp.vec4), weight=0.3
            )
            objectives.append(arm["rotation"])
        objectives.append(
            ik.IKObjectiveJointLimit(self.model.joint_limit_lower, self.model.joint_limit_upper, weight=10.0)
        )
        self.ik_q = wp.clone(self.model.joint_q).reshape((1, -1))
        self.ik_solver = ik.IKSolver(
            self.model, 1, objectives, jacobian_mode=ik.IKJacobianType.MIXED, lambda_initial=0.04
        )
        self.ik_graph = None
        self.lower, self.upper = self.model.joint_limit_lower.numpy(), self.model.joint_limit_upper.numpy()

    def _prepare_left_approach(self):
        """Plan the initial arm approach in joint space around the wrist singularity."""
        self.command_state = self.model.state()
        self.approach_home = self.model.joint_q.numpy()
        self.approach_hold = self.approach_home.copy()
        if self.args.settle_only:
            return
        arm = self.arms[0]
        target = BAG_POSITION + np.array((-0.12, -0.1, 0.32))
        tilt = math.radians(float((target[2] - TABLE - 0.06) / 0.004))
        arm["objective"].set_target_position(0, wp.vec3(*target))
        arm["rotation"].set_target_rotation(0, wp.vec4(*wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi - tilt)))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=250)
        self.command_state.joint_q.assign(self.ik_q.numpy()[0])
        newton.eval_fk(self.model, self.command_state.joint_q, self.command_state.joint_qd, self.command_state)
        pose = self.command_state.body_q.numpy()[arm["ee"]]
        error = np.linalg.norm(np.asarray(wp.transform_point(wp.transform(*pose), arm["tcp"])) - target)
        if error > 0.001:
            raise ValueError(f"Left support posture is unreachable: {error:.6f} m")
        self.approach_hold[:8] = self.ik_q.numpy()[0, :8]
        self.ik_q.assign(self.approach_home.reshape((1, -1)))

    def plan(self, t):
        if self.args.settle_only:
            return [a["home"] for a in self.arms], [0.035, 0.035]
        hold = BAG_POSITION + np.array((0.0, -0.1, 0.258))
        prehold = BAG_POSITION + np.array((-0.12, -0.1, 0.32))
        above_hold = BAG_POSITION + np.array((0.0, -0.1, 0.32))
        if t < 1.0:
            left = (1 - smooth(t)) * self.arms[0]["home"] + smooth(t) * prehold
        elif t < 1.6:
            a = smooth((t - 1.0) / 0.6)
            left = (1 - a) * prehold + a * above_hold
        else:
            a = smooth((t - 1.6) / 0.4)
            left = (1 - a) * above_hold + a * hold
        right, right_open = self.arms[1]["home"].copy(), 0.035
        left_open = 0.035 - (0.035 - SUPPORT_OPENING) * smooth((t - 2) / 0.5)
        if PACK_START <= t < HANDLE_START:
            phase = t - PACK_START
            # Freeze the measured grasp point before descending: following a
            # parcel already compressed by the fingers drives the jaws down.
            if phase <= 0.9:
                points = self.state_0.particle_q.numpy()[self.parcel_particles]
                self.parcel_pick = points.mean(0) + np.array((0.0, 0.0, 0.004))
            pick = self.parcel_pick
            # Clear both cord loops before crossing the rim. The parcel then
            # falls through the opening while the jaws remain above the cords.
            drop = BAG_POSITION + np.array((0.0, -0.03, 0.41))
            above = pick + np.array((0, 0, 0.15))
            transfer = np.array((pick[0], pick[1], drop[2]))
            points = [
                self.arms[1]["home"],
                above,
                pick,
                pick,
                pick + np.array((0, 0, 0.12)),
                transfer,
                drop,
                drop,
                np.array((*self.arms[1]["home"][:2], drop[2])),
                self.arms[1]["home"],
            ]
            times = [0, 0.9, 1.8, 2.6, 3.4, 4.5, 5.8, 6.8, 7.4, PARCEL_DURATION]
            j = min(max(np.searchsorted(times, phase, side="right") - 1, 0), len(times) - 2)
            a = smooth((phase - times[j]) / (times[j + 1] - times[j]))
            right = (1 - a) * points[j] + a * points[j + 1]
            closed = 0.018
            right_open = 0.035 - (0.035 - closed) * smooth((phase - 1.8) / 0.8)
            if phase >= 5.8:
                right_open += (0.035 - right_open) * smooth((phase - 5.8) / 0.4)
        if t >= HANDLE_START:
            phase = t - HANDLE_START
            if self.lift_targets is None or (phase >= 2.1 and not self.lift_targets_frozen):
                q = self.state_0.particle_q.numpy()
                self.lift_targets = []
                for ids in self.bag.handles:
                    top = ids[np.argsort(q[ids, 2])[-12:]]
                    self.lift_targets.append(q[top].mean(0))
                # Seat the right cord inside the flat pads, away from the
                # tapered fingertip edge that can push it out during closure.
                self.lift_targets[1] += np.array((-0.008, 0.0, -0.0014))
                self.lift_targets_frozen = phase >= 2.1
            # Release the rim, clear it vertically, then enter each cord loop
            # from the outside. Freeze targets before contacting the cords.
            left_clear = np.array((hold[0], hold[1], self.lift_targets[0][2] + 0.02))
            right_clear = np.array((*self.arms[1]["home"][:2], self.lift_targets[1][2] + 0.02))
            outside = [
                target + np.array((side * 0.10, 0, 0)) for target, side in zip(self.lift_targets, (-1, 1), strict=True)
            ]
            points = (
                [hold, hold, left_clear, outside[0], self.lift_targets[0]],
                [self.arms[1]["home"], self.arms[1]["home"], right_clear, outside[1], self.lift_targets[1]],
            )
            times = [0, 0.5, 1.3, 2.1, 3.5]
            j = min(max(np.searchsorted(times, phase, side="right") - 1, 0), len(times) - 2)
            a = smooth((phase - times[j]) / (times[j + 1] - times[j]))
            left, right = [(1 - a) * p[j] + a * p[j + 1] for p in points]
            left_open = SUPPORT_OPENING + (0.035 - SUPPORT_OPENING) * smooth(phase / 0.5)
            right_open = 0.035
            closure = smooth((phase - 3.5) / 1.0)
            left_open -= (left_open - HANDLE_OPENINGS[0]) * closure
            right_open -= (right_open - HANDLE_OPENINGS[1]) * closure
            lift = 0.14 * smooth((t - LIFT_START) / 3)
            left = left + np.array((0, 0, lift))
            right = right + np.array((0, 0, lift))
        return [left, right], [left_open, right_open]

    def simulate(self):
        for substep in range(self.args.substeps):
            wp.launch(
                prescribe,
                self.robot_joints,
                [
                    self.frame_start,
                    self.frame_end,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    (substep + 1) / self.args.substeps,
                    self.state_0.joint_q,
                    self.state_0.joint_qd,
                ],
            )
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _set_commands(self):
        targets, openings = self.plan(self.sim_time + 1 / 60)
        approach = not self.args.settle_only and self.sim_time < 1.0
        approach_pose = None
        if approach:
            alpha = smooth(self.sim_time + 1 / 60)
            planned_q = self.approach_home + alpha * (self.approach_hold - self.approach_home)
            self.command_state.joint_q.assign(planned_q)
            newton.eval_fk(self.model, self.command_state.joint_q, self.command_state.joint_qd, self.command_state)
            arm = self.arms[0]
            approach_pose = self.command_state.body_q.numpy()[arm["ee"]]
            targets[0] = np.asarray(wp.transform_point(wp.transform(*approach_pose), arm["tcp"]))
        for arm, target in zip(self.arms, targets, strict=True):
            arm["objective"].set_target_position(0, wp.vec3(*target))
            side = -1.0 if arm is self.arms[0] else 1.0
            tilt = math.radians(float(np.clip((target[2] - TABLE - 0.06) / 0.004, 0.0, 80.0)))
            radial = np.asarray(target[:2]) - np.array((side * 0.36, 0.20))
            radial /= np.linalg.norm(radial)
            if side < 0:
                radial = np.array((1.0, 0.0))
            closing = np.array((-radial[1], radial[0], 0.0))
            forward = np.array((radial[0] * math.sin(tilt), radial[1] * math.sin(tilt), -math.cos(tilt)))
            rotation = wp.quat_from_matrix(
                wp.mat33(*np.column_stack((np.cross(closing, forward), closing, forward)).ravel())
            )
            if self.sim_time >= HANDLE_START:
                # Close across the cord cross-section, not along its tangent.
                angle = math.radians(80)
                handle_forward = np.array((-side * math.sin(angle), 0, -math.cos(angle)))
                handle_closing = np.array((-side * math.cos(angle), 0, math.sin(angle)))
                handle_rotation = wp.quat_from_matrix(
                    wp.mat33(
                        *np.column_stack(
                            (np.cross(handle_closing, handle_forward), handle_closing, handle_forward)
                        ).ravel()
                    )
                )
                rotation = wp.quat_slerp(
                    rotation, handle_rotation, float(smooth((self.sim_time - HANDLE_START - 1.3) / 0.8))
                )
            rotation = wp.quat_slerp(
                arm["home_rotation"],
                rotation,
                float(1.0 if side < 0 else smooth((self.sim_time + 1 / 60) / 2.0)),
            )
            if self.args.settle_only:
                rotation = arm["home_rotation"]
            elif approach and side < 0:
                rotation = wp.quat(*approach_pose[3:])
            arm["rotation"].set_target_rotation(0, wp.vec4(*rotation))
        self._solve_runtime_ik()
        end = self.state_0.joint_q.numpy()
        end[: self.robot_coords] = self.ik_q.numpy()[0, : self.robot_coords]
        if approach:
            end[:8] = planned_q[:8]
        end[: self.robot_coords] = np.clip(
            end[: self.robot_coords], self.lower[: self.robot_coords], self.upper[: self.robot_coords]
        )
        for arm, opening in zip(self.arms, openings, strict=True):
            end[arm["start"] + 6 : arm["start"] + 8] = (opening, -opening)
        self.ik_q.assign(end.reshape((1, -1)))
        self.frame_start.assign(self.state_0.joint_q)
        self.frame_end.assign(end)
        return targets

    def _solve_runtime_ik(self):
        """Replay fixed IK work while reading the current target and joint buffers."""
        if not self.model.device.is_cuda or getattr(self.args, "no_cuda_graph", False):
            self.ik_solver.step(self.ik_q, self.ik_q, iterations=24)
            return
        if self.ik_graph is None:
            # Warm up before capture (also needed for --settle-only), preserving
            # the seed so the first frame still performs exactly one IK solve.
            seed = wp.clone(self.ik_q)
            self.ik_solver.step(self.ik_q, self.ik_q, iterations=24)
            self.ik_q.assign(seed)
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.ik_solver.step(self.ik_q, self.ik_q, iterations=24)
            self.ik_q.assign(seed)
            self.ik_graph = capture.graph
        wp.capture_launch(self.ik_graph)

    def step(self):
        targets = self._set_commands()
        if self.graph is None:
            self.simulate()
            if (
                self.model.device.is_cuda
                and not getattr(self.args, "no_cuda_graph", False)
                and self.args.substeps % 2 == 0
            ):
                saved_a, saved_b = self.model.state(), self.model.state()
                saved_a.assign(self.state_0)
                saved_b.assign(self.state_1)
                with wp.ScopedCapture(device=self.model.device) as capture:
                    self.simulate()
                self.state_0.assign(saved_a)
                self.state_1.assign(saved_b)
                self.graph = capture.graph
        else:
            wp.capture_launch(self.graph)
        self.frame += 1
        self.sim_time = self.frame / 60
        q, bodies = self.state_0.particle_q.numpy(), self.state_0.body_q.numpy()
        errors = [
            np.linalg.norm(np.asarray(wp.transform_point(wp.transform(*bodies[a["ee"]]), a["tcp"])) - target)
            for a, target in zip(self.arms, targets, strict=True)
        ]
        self.max_ik_error = float(max(self.max_ik_error, *errors))
        points = q[self.parcel_particles]
        self.peak_height = max(self.peak_height, float(points.mean(0)[2]))
        self.loaded = parcel_is_contained(self.bag, q, points)
        if PACK_START + 6.8 <= self.sim_time < HANDLE_START:
            self.packed |= self.loaded
        joint_q = self.state_0.joint_q.numpy()
        for i, (arm, ids) in enumerate(zip(self.arms, self.handle_tops, strict=True)):
            local = np.asarray(
                wp.transform_point(wp.transform_inverse(wp.transform(*bodies[arm["ee"]])), wp.vec3(*q[ids].mean(0)))
            )
            opening = abs(float(joint_q[arm["start"] + 6]))
            # Check the original apex segment remains between the jaws. A
            # lifted bag alone can pass even when only one arm has a handle.
            self.handle_grasped[i] = bool(
                abs(local[0]) < 0.020 and abs(local[1]) < opening + 0.006 and 0.075 < local[2] < 0.142
            )
        if not self.args.settle_only and 2.5 <= self.sim_time <= 4.0:
            self.support_shape_max = max(self.support_shape_max, measure_paper_shape(self.bag, q)["shape_max"])
        if not self.args.settle_only and 3.0 <= self.sim_time <= PACK_START + 5.5:
            # Both support position and aperture are stationary in this window;
            # sustained paper motion is contact chatter, not commanded motion.
            paper_end = int(self.bag.handles[0].min())
            velocity = self.state_0.particle_qd.numpy()[:paper_end]
            speed_sq = float(np.mean(np.sum(velocity * velocity, axis=1)))
            self.support_speed_samples += 1
            self.support_speed_sq += (speed_sq - self.support_speed_sq) / self.support_speed_samples
        self.lift_height = float(q[self.bag.bottom, 2].min() - TABLE)
        if self.args.settle_only:
            paper = q[: int(self.bag.handles[0].min())]
            self.static_shape = measure_paper_shape(self.bag, q)
            if self.frame == 600:
                self.static_reference = paper.copy()
            if self.static_reference is not None:
                self.static_max_motion = max(
                    self.static_max_motion, float(np.linalg.norm(paper - self.static_reference, axis=1).max())
                )
            if self.frame % 60 == 0:
                print(
                    f"[PaperBox static] shape_rms={self.static_shape['shape_rms'] * 1000:.4f} mm "
                    f"shape_max={self.static_shape['shape_max'] * 1000:.4f} mm "
                    f"motion_since_10s={self.static_max_motion * 1000:.4f} mm",
                    flush=True,
                )
        if self.frame % 60 == 0:
            print(
                f"[PaperBag] t={self.sim_time:.1f} IK={max(errors):.4f} loaded={self.loaded} bottom={self.lift_height:.4f}",
                flush=True,
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        wp.launch(
            deform_render_mesh,
            len(self.render_points),
            [self.state_0.particle_q, *self.render_binding, self.render_points],
        )
        for indices, color, name in self.paper_render_parts:
            self.viewer.log_mesh(
                "/bag/" + name,
                self.render_points,
                indices,
                color=(1.0, 1.0, 1.0) if name == "paper" else color,
                roughness=0.9,
                backface_culling=False,
                uvs=self.paper_uv if name == "paper" else None,
                texture=self.paper_texture if name == "paper" else None,
                hidden=name == "paper",
            )
            if name == "paper":
                self.viewer.log_instances(
                    "/bag/paper/material",
                    "/bag/paper",
                    self.render_transform,
                    self.render_scale,
                    self.render_color,
                    self.render_material,
                )
        self.viewer.log_mesh(
            "/soft",
            self.state_0.particle_q,
            self.parcel_triangles,
            color=(0.75, 0.25, 0.14),
            roughness=0.85,
            backface_culling=False,
        )
        self.viewer.end_frame()

    def test_post_step(self):
        if not np.isfinite(self.state_0.particle_q.numpy()).all():
            raise AssertionError("Nonfinite deformable state")
        if self.max_ik_error > 0.025:
            raise AssertionError(f"IK position error {self.max_ik_error:.4f} m")
        if not self.args.settle_only:
            if self.support_shape_max > 0.005:
                raise AssertionError(f"Support approach deformed the paper by {self.support_shape_max:.4f} m")
            if self.sim_time >= PACK_START + 5.5 and self.support_speed_sq > 0.01**2:
                raise AssertionError(
                    f"Paper vibrates under stationary support: {math.sqrt(self.support_speed_sq):.4f} m/s RMS"
                )
            if self.sim_time >= HANDLE_START and not self.packed:
                raise AssertionError("Parcel did not clear the rim after release")

    def test_final(self):
        self.test_post_step()
        if self.args.settle_only:
            if self.sim_time < 30:
                raise AssertionError("Static stability validation requires 30 seconds (1800 frames)")
            if self.static_shape["shape_rms"] > 0.0005 or self.static_shape["shape_max"] > 0.002:
                raise AssertionError(f"Paper box lost its unloaded shape: {self.static_shape}")
            if self.static_max_motion > 0.003:
                raise AssertionError(f"Paper box continues moving after settling: {self.static_max_motion:.6f} m")
            return
        if self.sim_time < LIFT_START + 4:
            raise AssertionError(f"Packing and cooperative lift require at least {LIFT_START + 4:g} seconds")
        if not self.loaded:
            raise AssertionError("Parcel not retained in bag")
        if self.peak_height < TABLE + 0.14:
            raise AssertionError("Parcel must be physically lifted from the table")
        if self.lift_height < 0.08:
            raise AssertionError(f"Bag bottom not lifted: {self.lift_height:.4f} m")
        if not all(self.handle_grasped):
            raise AssertionError(f"Both arms must retain their handle: {self.handle_grasped}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1200)
        parser.add_argument(
            "--settle-only",
            action="store_true",
            help="Keep both arms at rest; use --num-frames 1800 for the 30-second stability test.",
        )
        parser.add_argument("--substeps", type=int, default=10)
        parser.add_argument("--iterations", type=int, default=15)
        parser.add_argument("--paper-stiffness", type=float, default=5e4)
        parser.add_argument("--paper-bending", type=float, default=100.0)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

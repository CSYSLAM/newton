# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pick a steel sphere and release it into the WAIC bag with MJVBDV2.

The source desk, rack and PiPER meshes use the reference FBD_03 bag. The arm is
kinematically commanded; the sphere and bag are dynamically simulated.
No attachment or prescribed sphere trajectory is used.

    uv run --extra examples -m newton.examples mjvbd_v2_piper_ball_into_bag
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.ik as ik
import newton.usd
from newton.solvers import SolverMJVBDV2

ASSETS = Path(__file__).resolve().parents[3] / "assets/piper_bag"
TABLE_TOP = 0.555


def _mesh(name):
    stage = Usd.Stage.Open(str(ASSETS / f"{name}.usd"))
    prim = next(p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh))
    mesh = newton.usd.get_mesh(prim, load_visual_materials=False)
    v = np.asarray(mesh.vertices).copy()
    v[:, 1], v[:, 2] = -v[:, 2].copy(), v[:, 1].copy()
    return v, np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)


def _smooth(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _support_height(vertices: np.ndarray, faces: np.ndarray, point: np.ndarray) -> float:
    """Return the highest triangle below a vertical ray through the ball center."""
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    e0, e1, p = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2], point[:2] - a[:, :2]
    denominator = e0[:, 0] * e1[:, 1] - e0[:, 1] * e1[:, 0]
    valid = np.abs(denominator) > 1.0e-16
    u = np.zeros(len(faces))
    v = np.zeros(len(faces))
    np.divide(p[:, 0] * e1[:, 1] - p[:, 1] * e1[:, 0], denominator, out=u, where=valid)
    np.divide(e0[:, 0] * p[:, 1] - e0[:, 1] * p[:, 0], denominator, out=v, where=valid)
    height = a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2])
    inside = valid & (u >= -1.0e-6) & (v >= -1.0e-6) & (u + v <= 1.0 + 1.0e-6) & (height < point[2])
    return float(np.max(height[inside])) if np.any(inside) else float("-inf")


def _inclined_support_distance(vertices: np.ndarray, faces: np.ndarray, point: np.ndarray) -> float:
    """Measure normal distance to a lower triangle with an interior projection.

    A sphere resting on an inclined face need not touch the vertical ray below
    its center. Vertical faces and projections outside a face supply no support
    for this conservative geometric check.
    """
    a, b, c = (vertices[faces[:, i]].astype(np.float64) for i in range(3))
    ab, ac, ap = b - a, c - a, point - a
    aa = np.sum(ab * ab, axis=1)
    cc = np.sum(ac * ac, axis=1)
    cross = np.sum(ab * ac, axis=1)
    denominator = aa * cc - cross * cross
    valid = denominator > 1.0e-24
    u, v = np.zeros(len(faces)), np.zeros(len(faces))
    pa, pc = np.sum(ap * ab, axis=1), np.sum(ap * ac, axis=1)
    np.divide(cc * pa - cross * pc, denominator, out=u, where=valid)
    np.divide(aa * pc - cross * pa, denominator, out=v, where=valid)
    projection = a + u[:, None] * ab + v[:, None] * ac
    delta = point - projection
    distance = np.linalg.norm(delta, axis=1)
    supported = valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (delta[:, 2] > 1.0e-6)
    return float(np.min(distance[supported])) if np.any(supported) else float("inf")


def _sphere_surface_penetration(vertices: np.ndarray, faces: np.ndarray, center: np.ndarray, radius: float) -> float:
    """Bound sphere/triangle penetration, including interiors with no inside vertices."""
    a, b, c = (vertices[faces[:, i]].astype(np.float64) for i in range(3))
    ab, ac, ap = b - a, c - a, center - a
    aa, cc, cross = np.sum(ab * ab, axis=1), np.sum(ac * ac, axis=1), np.sum(ab * ac, axis=1)
    denominator = aa * cc - cross * cross
    valid = denominator > 1.0e-24
    u, v = np.zeros(len(faces)), np.zeros(len(faces))
    pa, pc = np.sum(ap * ab, axis=1), np.sum(ap * ac, axis=1)
    np.divide(cc * pa - cross * pc, denominator, out=u, where=valid)
    np.divide(aa * pc - cross * pa, denominator, out=v, where=valid)
    projection = a + u[:, None] * ab + v[:, None] * ac
    inside = valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
    minimum = float(np.min(np.linalg.norm(projection[inside] - center, axis=1))) if np.any(inside) else float("inf")
    for start, end in ((a, b), (b, c), (c, a)):
        edge = end - start
        length_sq = np.sum(edge * edge, axis=1)
        t = np.zeros(len(faces))
        np.divide(np.sum((center - start) * edge, axis=1), length_sq, out=t, where=length_sq > 0.0)
        closest = start + np.clip(t, 0.0, 1.0)[:, None] * edge
        minimum = min(minimum, float(np.min(np.linalg.norm(closest - center, axis=1))))
    return max(0.0, radius - minimum)


def _surface_winding(vertices: np.ndarray, faces: np.ndarray, point: np.ndarray) -> float:
    """Return oriented solid-angle coverage, approximately one inside the bag."""
    a, b, c = (vertices[faces[:, i]].astype(np.float64) - point for i in range(3))
    la, lb, lc = (np.linalg.norm(x, axis=1) for x in (a, b, c))
    numerator = np.sum(a * np.cross(b, c), axis=1)
    denominator = la * lb * lc + np.sum(a * b, axis=1) * lc + np.sum(b * c, axis=1) * la + np.sum(c * a, axis=1) * lb
    return float(np.sum(2.0 * np.arctan2(numerator, denominator)) / (4.0 * np.pi))


@wp.kernel
def _prescribe(
    start: wp.array[float],
    end: wp.array[float],
    q_start: wp.array[int],
    qd_start: wp.array[int],
    alpha: float,
    q: wp.array[float],
    qd: wp.array[float],
):
    j = wp.tid()
    if q_start[j + 1] > q_start[j]:
        i = q_start[j]
        q[i] = wp.lerp(start[i], end[i], alpha)
        qd[qd_start[j]] = (end[i] - start[i]) * 60.0


class Example:
    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        if args.substeps < 1 or args.iterations < 1:
            raise ValueError("Substeps and iterations must be positive")
        if not np.isfinite(args.bending_stiffness) or args.bending_stiffness < 0.0:
            raise ValueError("Bending stiffness must be finite and nonnegative")
        if not 0.0 < args.grip_half_width < 0.035 or args.ball_mass <= 0.0:
            raise ValueError("Require 0 < grip half-width < 0.035 m and positive ball mass")
        self.sim_dt = self.frame_dt / args.substeps
        self.frame = 0
        self.peak_ball_z = 0.0
        self.peak_ik_error = 0.0
        self.entered_bag = False
        builder = newton.ModelBuilder()
        # Keep speculative mesh contacts near the actual contact surface.
        builder.rigid_gap = 0.005
        builder.add_mjcf(
            str(ASSETS / "piper/piper.xml"),
            xform=wp.transform(wp.vec3(-0.3, 0.0, TABLE_TOP), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
        )
        self.robot_joints = builder.joint_count
        self.robot_coords = len(builder.joint_q)
        self.robot_bodies = builder.body_count
        for i in range(self.robot_bodies):
            builder.body_flags[i] = int(newton.BodyFlags.KINEMATIC)
        for i in range(builder.shape_count):
            builder.shape_material_mu[i] = 1.0
            builder.shape_material_ke[i] = args.contact_stiffness
            builder.shape_material_kd[i] = args.contact_damping
            # The reference finger MeshCollider uses a 5 mm collision gap.
            if builder.body_label[builder.shape_body[i]].endswith(("/link7", "/link8")):
                builder.shape_margin[i] = 0.005
        for name in ("desk", "rack"):
            v, f = _mesh(name)
            mesh = newton.Mesh(v, f.reshape(-1))
            if name == "rack":
                # Coarse cloth edges can cross the thin rods between vertices.
                # Full-surface contact needs an SDF of the original rack mesh.
                mesh.build_sdf(target_voxel_size=0.001)
            station_shape = builder.add_shape_mesh(
                body=-1,
                mesh=mesh,
                label=name,
                cfg=newton.ModelBuilder.ShapeConfig(
                    ke=2.0e6 if name == "rack" else 2.0e5,
                    kd=100.0,
                    mu=0.5,
                    has_shape_collision=name != "desk",
                    has_particle_collision=name != "desk",
                ),
                color=(0.55, 0.58, 0.62) if name == "desk" else (0.18, 0.22, 0.28),
            )
            if name == "rack":
                self.rack_shape = station_shape
        # Match the reference native solver's ground_height, independently
        # of its frontend floor visualization.
        builder.add_ground_plane(
            height=TABLE_TOP,
            cfg=newton.ModelBuilder.ShapeConfig(is_visible=False, ke=6.0e4, kd=100.0, mu=1.0),
        )
        ball_v, ball_f = _mesh("wangqiu")
        ball_center = 0.5 * (ball_v.min(0) + ball_v.max(0))
        ball_v -= ball_center
        self.ball_radius = float(np.max(np.linalg.norm(ball_v, axis=1)))
        self.ball_body = builder.add_body(
            xform=wp.transform(wp.vec3(*(ball_center + np.array((0, 0, TABLE_TOP + 0.4)))), wp.quat_identity()),
            label="steel_ball",
        )
        ball_mesh = newton.Mesh(ball_v, ball_f.reshape(-1))
        ball_mesh.build_sdf(target_voxel_size=0.001)
        self.ball_shape = builder.add_shape_mesh(
            self.ball_body,
            mesh=ball_mesh,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=1000.0, ke=args.contact_stiffness, kd=args.contact_damping, mu=1.0, is_visible=False
            ),
            label="steel_ball",
            color=(0.65, 0.69, 0.74),
        )
        mass_scale = args.ball_mass / builder.body_mass[self.ball_body]
        builder.body_mass[self.ball_body] *= mass_scale
        builder.body_inertia[self.ball_body] *= mass_scale
        v, f = _mesh("FBD_03")
        self.bag_rest = v.copy()
        self.bag_faces = f
        self.bag_edges = np.unique(np.sort(np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]])), axis=1), axis=0)
        self.rest_edge_lengths = np.linalg.norm(v[self.bag_edges[:, 0]] - v[self.bag_edges[:, 1]], axis=1)
        self.bag_center = 0.5 * (v.min(0) + v.max(0))
        self.mouth_height = float(v[:, 2].max())
        top = v[:, 2] > self.mouth_height - 0.01
        self.handle_top_indices = (
            np.flatnonzero(top & (v[:, 0] < self.bag_center[0])),
            np.flatnonzero(top & (v[:, 0] >= self.bag_center[0])),
        )
        builder.add_cloth_mesh(
            pos=wp.vec3(),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=v.tolist(),
            indices=f.reshape(-1).tolist(),
            density=0.3,
            tri_ke=args.membrane_stiffness,
            tri_ka=args.membrane_stiffness,
            tri_kd=0.8,
            edge_ke=args.bending_stiffness,
            edge_kd=1.0e-3,
            particle_radius=0.001,
        )
        builder.add_ground_plane(
            height=0.0,
            cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False),
            label="ground_display",
        )
        builder.color(include_bending=True)
        self.model = builder.finalize()
        self.model.soft_contact_ke = 100.0
        self.model.soft_contact_kd = 0.1
        self.model.soft_contact_mu = 0.5
        self.robot_lower = self.model.joint_limit_lower.numpy()[: self.robot_coords]
        self.robot_upper = self.model.joint_limit_upper.numpy()[: self.robot_coords]
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        self.frame_start, self.frame_end = wp.clone(self.model.joint_q), wp.clone(self.model.joint_q)
        acceleration_options = {}
        if args.cloth_acceleration == "chebyshev":
            acceleration_options.update(
                particle_chebyshev_spectral_radius=0.95,
                particle_chebyshev_max_radius_fraction=1.0,
                particle_chebyshev_warmup_iterations=2,
                particle_chebyshev_polish_iterations=3,
                particle_chebyshev_contact_rings=1,
            )
        iterations = args.iterations
        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=(0,),
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": iterations,
                "rigid_soft_enable_dat": args.rigid_soft_dat,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": 0.001,
                "particle_self_contact_margin": 0.002,
                "particle_topological_contact_filter_threshold": 2,
                "particle_rest_shape_contact_exclusion_radius": 0.004,
                "rigid_body_particle_contact_buffer_size": 65536,
                "rigid_body_contact_buffer_size": 4096,
                "friction_epsilon": 1.0e-4,
                "rigid_contact_history": False,
                "rigid_contact_hard": not args.soft_rigid_contact,
                "rigid_contact_k_start": 2.0e5,
                **acceleration_options,
            },
            collision_options={
                "enable_cuda_fast_path": False,
                "enable_rigid_soft_full_surface_contact": True,
                "rigid_soft_full_surface_shape_indices": [self.rack_shape, self.ball_shape],
                "broad_phase": "nxn",
                "soft_contact_max": 262144,
                "soft_contact_margin": 0.004,
                "include_static_kinematic_pairs": False,
            },
        )
        self.contacts = self.solver.contacts
        self._build_ik()
        self.triangles = wp.array(f.reshape(-1), dtype=int, device=self.model.device)
        self.ball_color = wp.array([wp.vec3(0.68, 0.72, 0.78)], dtype=wp.vec3, device=self.model.device)
        self.ball_material = wp.array([wp.vec4(0.22, 1.0, 0.0, 0.0)], dtype=wp.vec4, device=self.model.device)
        self.graph = None
        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = False
        self.viewer.set_camera(pos=wp.vec3(1.0, -1.25, 1.45), pitch=-23.0, yaw=133.0)
        print(
            f"[PiperBag] particles={self.model.particle_count}, ball_mass={self.model.body_mass.numpy()[self.ball_body]:.4f} kg, "
            f"substeps={args.substeps}, local_sweeps={iterations}"
        )

    def _build_ik(self):
        self.ee = next(i for i, label in enumerate(self.model.body_label) if label.endswith("/gripper_base_left"))
        # Use the source's collision-mesh bounding-box centers as the TCP.
        body_q = self.state_0.body_q.numpy()
        transforms = self.model.shape_transform.numpy()
        scales = self.model.shape_scale.numpy()
        flags = self.model.shape_flags.numpy()
        centers = []
        for body, label in enumerate(self.model.body_label):
            if not label.endswith(("/link7", "/link8")):
                continue
            points = []
            for shape in self.model.body_shapes[body]:
                mesh = self.model.shape_source[shape]
                if not flags[shape] & int(newton.ShapeFlags.COLLIDE_SHAPES) or not isinstance(mesh, newton.Mesh):
                    continue
                vertices = np.asarray(mesh.vertices)
                local = transforms[shape]
                local_rotation = np.asarray(wp.quat_to_matrix(wp.quat(*local[3:]))).reshape(3, 3)
                vertices = (vertices @ local_rotation.T + local[:3]) * scales[shape]
                world = body_q[body]
                world_rotation = np.asarray(wp.quat_to_matrix(wp.quat(*world[3:]))).reshape(3, 3)
                points.extend(vertices @ world_rotation.T + world[:3])
            vertices = np.asarray(points)
            centers.append(0.5 * (vertices.min(0) + vertices.max(0)))
        center = np.mean(centers, axis=0)
        self.tcp = wp.transform_point(wp.transform_inverse(wp.transform(*body_q[self.ee])), wp.vec3(*center))
        self.ik_q = wp.clone(self.model.joint_q).reshape((1, -1))
        self.ik_state = self.model.state()
        self.initial_tcp = np.array(wp.transform_point(wp.transform(*self.state_0.body_q.numpy()[self.ee]), self.tcp))
        self.pos_obj = ik.IKObjectivePosition(self.ee, self.tcp, wp.array([wp.vec3(*self.initial_tcp)], dtype=wp.vec3))
        self.ik_solver = ik.IKSolver(
            self.model,
            1,
            [
                self.pos_obj,
                ik.IKObjectiveJointLimit(self.model.joint_limit_lower, self.model.joint_limit_upper, weight=10.0),
            ],
            jacobian_mode=ik.IKJacobianType.MIXED,
            lambda_initial=0.08,
        )
        self.pick = np.array((0.0, 0.0, TABLE_TOP + self.ball_radius))

    def _plan(self, t):
        # Pickup tracking must not translate the release point away from the bag.
        drop = np.array((self.bag_center[0], self.bag_center[1], TABLE_TOP + self.ball_radius + 0.40))
        points = [
            self.initial_tcp,
            self.initial_tcp,
            self.pick + np.array((0, 0, 0.20)),
            self.pick,
            self.pick,
            self.pick + np.array((0, 0, 0.48)),
            drop,
            drop + np.array((0.0, 0.0, -0.05)),
            self.initial_tcp,
        ]
        times = [0.0, 0.4, 1.4, 2.0, 2.35, 3.3, 4.4, 5.2, 6.0]
        i = max(0, min(int(np.searchsorted(times, t, side="right")) - 1, len(times) - 2))
        a = _smooth((t - times[i]) / (times[i + 1] - times[i]))
        target = (1 - a) * points[i] + a * points[i + 1]
        opening = 0.035 - (0.035 - self.args.grip_half_width) * _smooth((t - 2.0) / 0.35)
        if t >= 4.4:
            opening = 0.035
        return target, float(opening)

    def _simulate(self):
        for substep in range(self.args.substeps):
            wp.launch(
                _prescribe,
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
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.sim_time <= 2.35:
            self.pick = self.state_0.body_q.numpy()[self.ball_body, :3].copy()
        target, opening = self._plan(self.sim_time + self.frame_dt)
        self.pos_obj.set_target_position(0, wp.vec3(*target))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=18)
        end = self.state_0.joint_q.numpy()
        end[: self.robot_coords] = self.ik_q.numpy()[0, : self.robot_coords]
        end[: self.robot_coords] = np.clip(
            end[: self.robot_coords],
            self.robot_lower,
            self.robot_upper,
        )
        end[6:8] = (opening, -opening)
        self.ik_q.assign(end.reshape(1, -1))
        self.frame_start.assign(self.state_0.joint_q)
        self.frame_end.assign(end)
        if self.graph is None:
            self._simulate()
            if self.model.device.is_cuda and not self.args.no_cuda_graph and self.args.substeps % 2 == 0:
                saved_0, saved_1 = self.model.state(), self.model.state()
                saved_0.assign(self.state_0)
                saved_1.assign(self.state_1)
                with wp.ScopedCapture(device=self.model.device) as capture:
                    self._simulate()
                self.state_0.assign(saved_0)
                self.state_1.assign(saved_1)
                self.graph = capture.graph
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt
        self.frame += 1
        body = self.state_0.body_q.numpy()
        ball = body[self.ball_body, :3]
        if 2.35 < self.sim_time < 4.4:
            self.peak_ball_z = max(self.peak_ball_z, float(ball[2]))
        tcp = np.array(wp.transform_point(wp.transform(*body[self.ee]), self.tcp))
        self.peak_ik_error = max(self.peak_ik_error, float(np.linalg.norm(tcp - target)))
        if (
            self.sim_time > 4.4
            and np.linalg.norm(ball[:2] - self.bag_center[:2]) < 0.07
            and ball[2] < self.mouth_height
        ):
            self.entered_bag = True
        if self.frame % 60 == 0:
            print(
                f"[PiperBag] t={self.sim_time:.2f} ball={ball.round(4)} tcp={tcp.round(4)} IK={np.linalg.norm(tcp - target):.4f} entered={self.entered_bag}",
                flush=True,
            )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_shapes(
            "/steel_ball",
            newton.GeoType.MESH,
            (1.0, 1.0, 1.0),
            self.state_0.body_q[self.ball_body : self.ball_body + 1],
            colors=self.ball_color,
            materials=self.ball_material,
            geo_src=self.model.shape_source[self.ball_shape],
        )
        self.viewer.log_mesh(
            "/bag", self.state_0.particle_q, self.triangles, color=(0.75, 0.82, 0.9), backface_culling=False
        )
        self.viewer.end_frame()

    def _edge_stretch(self):
        positions = self.state_0.particle_q.numpy()
        lengths = np.linalg.norm(positions[self.bag_edges[:, 0]] - positions[self.bag_edges[:, 1]], axis=1)
        return lengths / self.rest_edge_lengths

    def test_post_step(self):
        """Reject nonfinite states and escaped cloth or rigid bodies."""
        for array in (self.state_0.particle_q, self.state_0.particle_qd, self.state_0.body_q, self.state_0.body_qd):
            if not np.isfinite(array.numpy()).all():
                raise AssertionError("Nonfinite simulation state")
        positions = self.state_0.particle_q.numpy()
        ball = self.state_0.body_q.numpy()[self.ball_body, :3]
        penetration = _sphere_surface_penetration(positions, self.bag_faces, ball, self.ball_radius)
        if penetration > 0.002:
            raise AssertionError(f"Ball penetrated the bag by {penetration * 1000.0:.3f} mm")
        if float(positions[:, 2].min()) < TABLE_TOP - 0.002:
            raise AssertionError("Bag penetrated the tabletop by more than 2 mm")
        if self.frame >= 120 and float(positions[:, 2].min()) < TABLE_TOP + 0.003:
            raise AssertionError("Settled bag must remain at least 3 mm above the tabletop")
        for handle in self.handle_top_indices:
            if float(positions[handle, 2].max()) < self.mouth_height - 0.025:
                raise AssertionError("A bag handle slipped off the rack")
        if np.max(np.abs(positions)) > 3.0:
            raise AssertionError("Bag escaped scene")

    def test_final(self):
        """Require a physical lift and a released ball retained inside the bag."""
        self.test_post_step()
        ball = self.state_0.body_q.numpy()[self.ball_body, :3]
        stretch = self._edge_stretch()
        if np.quantile(stretch, 0.99) > 1.10 or float(stretch.max()) > 1.5:
            raise AssertionError(f"Excessive bag stretching: p99={np.quantile(stretch, 0.99)}, max={stretch.max()}")
        if self.frame < 600:
            raise AssertionError("Run at least 600 frames to validate the full task")
        if self.peak_ball_z < TABLE_TOP + 0.30:
            raise AssertionError(f"Ball was not lifted: peak z={self.peak_ball_z}")
        if not self.entered_bag or np.linalg.norm(ball[:2] - self.bag_center[:2]) > 0.09:
            raise AssertionError(f"Ball did not remain in bag: {ball}")
        winding = abs(_surface_winding(self.state_0.particle_q.numpy(), self.bag_faces, ball))
        # At the final resting height the open mouth subtends a small solid angle.
        # Nearby exterior folds can support a ball without enclosing its center.
        if winding < 0.75:
            raise AssertionError(f"Ball center is not enclosed by the bag: winding={winding}, center={ball}")
        if not TABLE_TOP + self.ball_radius * 0.8 < ball[2] < self.mouth_height:
            raise AssertionError(f"Ball escaped bag vertically: {ball}")
        support = _support_height(self.state_0.particle_q.numpy(), self.bag_faces, ball)
        if abs(float(ball[2]) - support - self.ball_radius) > 0.01:
            distance = _inclined_support_distance(self.state_0.particle_q.numpy(), self.bag_faces, ball)
            if abs(distance - self.ball_radius) > 0.002:
                raise AssertionError(
                    f"Ball is not supported by the bag: center={ball}, support={support}, distance={distance}"
                )
        if self.model.body_flags.numpy()[self.ball_body] & int(newton.BodyFlags.KINEMATIC):
            raise AssertionError("Steel ball must remain dynamic")
        if self.peak_ik_error > 0.02:
            raise AssertionError(f"IK error exceeded 2 cm: {self.peak_ik_error}")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=600)
        parser.add_argument("--no-cuda-graph", action="store_true")
        parser.add_argument("--substeps", type=int, default=10)
        parser.add_argument("--iterations", type=int, default=15, help="Local VBD sweeps per substep")
        parser.add_argument("--rigid-soft-dat", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--cloth-acceleration", choices=("none", "chebyshev"), default="chebyshev")
        parser.add_argument("--contact-stiffness", type=float, default=1.0e4)
        parser.add_argument("--contact-damping", type=float, default=20.0)
        parser.add_argument("--soft-rigid-contact", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--membrane-stiffness", type=float, default=3.0e5)
        parser.add_argument(
            "--bending-stiffness", type=float, default=5.0e-5, help="Resistance to bending away from the rest shape"
        )
        parser.add_argument("--ball-mass", type=float, default=0.058)
        parser.add_argument("--grip-half-width", type=float, default=0.025)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

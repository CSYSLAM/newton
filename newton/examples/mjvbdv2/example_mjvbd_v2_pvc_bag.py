# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspect an elastic clear PVC bag and stiff handles in a normal Newton window.

uv run --extra examples -m newton.examples mjvbd_v2_pvc_bag

Pinch the upper handle of a bag lying on the table, lift it under gravity,
then deliver two free blocks from a retracting tray. Use Space to pause,
the sidebar to reset, or --no-task for manual mouse inspection.
This standalone scene needs only the clear_plastic_bag asset. It uses a reinforced
experimental PVC profile, without robots or WebXR.
"""

import argparse
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMJVBDV2

from .support.w1_pvc_bag import FILM_COLOR, HANDLE_COLOR, add_pvc_material, load_pvc_bag


@wp.kernel
def _move_bodies(
    bodies: wp.array[int],
    start: wp.array[wp.transform],
    end: wp.array[wp.transform],
    alpha: float,
    frame_dt: float,
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    a, b = start[i], end[i]
    p0, p1 = wp.transform_get_translation(a), wp.transform_get_translation(b)
    r0, r1 = wp.transform_get_rotation(a), wp.transform_get_rotation(b)
    axis, angle = wp.quat_to_axis_angle(r1 * wp.quat_inverse(r0))
    q[bodies[i]] = wp.transform(wp.lerp(p0, p1, alpha), wp.quat_slerp(r0, r1, alpha))
    qd[bodies[i]] = wp.spatial_vector((p1 - p0) / frame_dt, axis * (angle / frame_dt))


def smooth_step(t):
    """Interpolate a task phase with zero velocity at both ends."""
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


class Example:
    """Lift a lying PVC bag through jaw contact, then drop a free load inside."""

    reset_in_place = True
    table_height = 0.65

    def __init__(self, viewer, args):
        if args.substeps < 1 or args.iterations < 1:
            raise ValueError("Substeps and iterations must be positive")
        if not np.isfinite(args.handle_stiffness) or args.handle_stiffness <= 0:
            raise ValueError("Handle stiffness must be finite and positive")
        if not np.isfinite(args.lift_height) or args.lift_height <= 0:
            raise ValueError("Lift height must be finite and positive")
        if not np.isfinite(args.opacity) or not 0 <= args.opacity <= 1:
            raise ValueError("Opacity must be between 0 and 1")
        self.viewer, self.args = viewer, args
        self.frame_dt, self.sim_dt = 1 / 60, 1 / (60 * args.substeps)
        self.sim_time, self.frame, self.graph = 0.0, 0, None
        self.opacity = args.opacity
        self.bag = load_pvc_bag(args.pvc_assets)
        vertices, faces = self.bag["vertices"], self.bag["faces"]
        # Cap only the mouth for load-containment checks, never for collisions.
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
        boundary = edges[counts[inverse] == 1]
        cap = np.column_stack((boundary[:, 1], boundary[:, 0], np.full(len(boundary), len(vertices))))
        self.containment_faces = np.concatenate((faces, cap))
        rotation = wp.quat_identity()
        if args.pose == "lying":
            # Same broad-side-down orientation and -Y mouth direction as W1 rope teleop.
            rotation = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2) * wp.quat_from_axis_angle(
                wp.vec3(0, 1, 0), -np.pi / 2
            )
        matrix = np.asarray(wp.quat_to_matrix(rotation)).reshape(3, 3)
        self.bag_rotation = rotation
        points = vertices @ matrix.T
        position = -(points.min(0) + points.max(0)) / 2
        position[2] = self.table_height + 0.002 - points[:, 2].min()

        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.002
        builder.add_cloth_mesh(
            pos=wp.vec3(*position),
            rot=rotation,
            scale=1.0,
            vel=wp.vec3(),
            vertices=vertices.tolist(),
            indices=faces.ravel().tolist(),
            density=0.336,
            tri_ke=2e4,
            tri_ka=2e4,
            tri_kd=0.04,
            edge_ke=0.08,
            edge_kd=0.002,
            particle_radius=0.0012,
            color=FILM_COLOR,
        )
        add_pvc_material(builder, self.bag, handle_stiffness=args.handle_stiffness, reinforced=True)
        # A prescribed support uses the same full-contact VBD backend as packing.
        table = builder.add_body(
            xform=wp.transform((0, 0, self.table_height - 0.022), wp.quat_identity()),
            mass=1.0,
            is_kinematic=True,
            label="table",
        )
        table_shape = builder.add_shape_box(
            table,
            hx=0.42,
            hy=0.45,
            hz=0.022,
            cfg=builder.ShapeConfig(ke=4e4, kd=80, mu=0.65),
            color=(0.28, 0.32, 0.34),
        )
        for x in (-0.34, 0.34):
            for y in (-0.37, 0.37):
                height = (self.table_height - 0.044) / 2
                builder.add_shape_box(
                    -1,
                    xform=wp.transform((x, y, height), wp.quat_identity()),
                    hx=0.018,
                    hy=0.018,
                    hz=height,
                    color=(0.19, 0.21, 0.23),
                    cfg=builder.ShapeConfig(has_particle_collision=False),
                )
        builder.add_ground_plane(color=(0.12, 0.14, 0.16))
        full_shapes = [table_shape]
        self._add_task(builder, full_shapes)
        builder.color(include_bending=True)
        self.model = builder.finalize()
        self.model.soft_contact_ke, self.model.soft_contact_kd, self.model.soft_contact_mu = 2e6, 10.0, 0.6
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.initial_state, self.control = self.model.state(), self.model.control()
        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=tuple(range(1 + len(self.driven_bodies))),
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": args.iterations,
                # Penalty contacts recover existing overlap on the moving tray;
                # cold-started hard contacts can accumulate penetration here.
                "rigid_contact_hard": False,
                "rigid_soft_enable_dat": True,
                "particle_enable_multilevel_correction": True,
                "particle_multilevel_operator": "galerkin",
                "particle_multilevel_cluster_size": 8,
                "particle_multilevel_coarse_iterations": 32,
                "particle_multilevel_checkpoints": tuple(
                    sorted(
                        {
                            max(1, args.iterations // 3),
                            max(1, 2 * args.iterations // 3),
                            max(1, args.iterations - 2),
                        }
                    )
                ),
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": 0.0012,
                "particle_self_contact_margin": 0.003,
                "particle_topological_contact_filter_threshold": 2,
                "particle_rest_shape_contact_exclusion_radius": 0.0015,
                "particle_collision_detection_interval": 0,
                "particle_enable_surface_cache": True,
                "particle_enable_truncation_cache": True,
                "particle_chebyshev_spectral_radius": 0.9,
                "particle_chebyshev_warmup_iterations": 2,
                "particle_chebyshev_polish_iterations": 3,
                "particle_chebyshev_contact_rings": 1,
                "rigid_body_particle_contact_buffer_size": 8192,
                "friction_epsilon": 0.001,
            },
            collision_options={
                "include_static_kinematic_pairs": False,
                "enable_rigid_soft_full_surface_contact": True,
                "rigid_soft_full_surface_shape_indices": full_shapes,
                "soft_contact_margin": 0.004,
                "soft_contact_max": 131072,
            },
        )
        self.driven_body_ids = wp.array(self.driven_bodies, dtype=wp.int32, device=self.model.device)
        self.driven_start = wp.array(self.initial_driven_poses, dtype=wp.transform, device=self.model.device)
        self.driven_end = wp.clone(self.driven_start)
        self._reset_task()
        self.render_indices = [
            wp.array(part.ravel(), dtype=wp.int32, device=self.model.device)
            for part in (faces[: self.bag["shell_faces"]], faces[self.bag["shell_faces"] :])
        ]
        viewer.set_model(self.model)
        viewer.show_particles, viewer.show_triangles = False, False
        viewer.set_camera(pos=wp.vec3(0.90, -1.15, 1.50), pitch=-18, yaw=128)
        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(self.gui, position="side")

    def _add_task(self, builder, full_shapes):
        """Add two independent payloads and a pair of prescribed physical jaws."""
        self.jaws, self.payloads, self.initial_driven_poses = [], [], []
        self.driven_bodies = []
        if not self.args.task:
            return
        points = np.asarray(builder.particle_q)
        self.handle_index = int(np.argmax([points[tube, 2].mean() for tube in self.bag["handles"]]))
        self.grip_axis = np.asarray(wp.quat_rotate(self.bag_rotation, wp.vec3(0, 0, 1)))
        self.approach_axis = np.array((0.0, 0.0, 1.0)) if self.args.pose == "lying" else np.array((1.0, 0.0, 0.0))
        self._track_handle(points)
        self.initial_grip_axis = self.grip_axis.copy()
        self.initial_approach_axis = self.approach_axis.copy()
        self.initial_grip_rotation = self.grip_rotation.copy()
        self.initial_grip = self.grip_center.copy()
        for sign, color in ((-1, (0.9, 0.36, 0.08)), (1, (0.12, 0.5, 0.86))):
            position = self.initial_grip + self.approach_axis * 0.09 + self.grip_axis * (sign * 0.0455)
            pose = wp.transform(wp.vec3(*position), wp.quat(*self.grip_rotation))
            body = builder.add_body(xform=pose, mass=0.1, is_kinematic=True, label=f"jaw_{sign}")
            shape = builder.add_shape_box(
                body,
                hx=0.018,
                hy=0.022,
                hz=0.022,
                cfg=builder.ShapeConfig(ke=2e6, kd=100, mu=1.4, margin=0.0005),
                color=color,
            )
            self.jaws.append(body)
            self.initial_driven_poses.append(np.asarray(pose))
            full_shapes.append(shape)
        # Both payloads remain dynamic, even during delivery. Friction carries
        # them on this physical tray; a fast downward/sideways withdrawal releases them.
        self.tray_home = np.array((0.29, 0.05, self.table_height + 0.006))
        tray_pose = wp.transform(wp.vec3(*self.tray_home), wp.quat_identity())
        self.tray = builder.add_body(xform=tray_pose, mass=0.1, is_kinematic=True, label="delivery_tray")
        shape = builder.add_shape_box(
            self.tray,
            hx=0.09,
            hy=0.045,
            hz=0.004,
            cfg=builder.ShapeConfig(ke=4e4, kd=80, mu=0.8),
            color=(0.44, 0.48, 0.50),
        )
        full_shapes.append(shape)
        self.initial_driven_poses.append(np.asarray(tray_pose))
        self.driven_bodies = [*self.jaws, self.tray]
        for index, (offset, mass, color) in enumerate(
            ((-0.038, 0.020, (0.94, 0.64, 0.09)), (0.038, 0.025, (0.25, 0.68, 0.28)))
        ):
            position = self.tray_home + np.array((offset, 0, 0.026))
            body = builder.add_body(
                xform=wp.transform(wp.vec3(*position), wp.quat_identity()), label=f"payload_{index}"
            )
            half = (0.018, 0.024, 0.020)
            shape = builder.add_shape_box(
                body,
                hx=half[0],
                hy=half[1],
                hz=half[2],
                cfg=builder.ShapeConfig(density=float(mass / (8 * np.prod(half))), ke=4e4, kd=80, mu=0.5),
                color=color,
            )
            self.payloads.append(body)
            full_shapes.append(shape)

    def _track_handle(self, points):
        """Pinch the upper crown from inside/outside its arch, approaching from above."""
        centers = points[self.bag["handles"][self.handle_index]].mean(axis=1)
        apex = (len(centers) - 1) // 2
        self.grip_center = centers[apex].copy()
        _, _, axes = np.linalg.svd(centers - centers.mean(0))
        normal = axes[-1]
        tangent = centers[apex + 1] - centers[apex - 1]
        tangent -= normal * (tangent @ normal)
        tangent /= np.linalg.norm(tangent)
        axis = np.cross(normal, tangent)
        if axis @ self.grip_axis < 0:
            axis = -axis
        if normal @ self.approach_axis < 0:
            normal = -normal
        self.approach_axis = normal
        self.grip_axis = axis
        matrix = np.column_stack((axis, tangent, np.cross(axis, tangent)))
        self.grip_rotation = np.asarray(wp.quat_from_matrix(wp.mat33(*matrix.ravel())))

    def _reset_task(self):
        self.phase = "Approach the upper handle" if self.args.task else "Manual inspection"
        self.grip_center = None
        if self.args.task:
            self.grip_axis = self.initial_grip_axis.copy()
            self.approach_axis = self.initial_approach_axis.copy()
            self.grip_rotation = self.initial_grip_rotation.copy()
        self.last_driven_poses = np.asarray(self.initial_driven_poses, dtype=np.float32).reshape(-1, 7)
        self.driven_start.assign(self.last_driven_poses)
        self.driven_end.assign(self.last_driven_poses)
        self.lift_baseline, self.delivery_target = None, None

    def _prepare_task(self):
        """Grasp and lift vertically; let gravity orient the bag before delivery."""
        if not self.args.task:
            return
        t = self.sim_time + self.frame_dt
        if self.grip_center is None or t < 0.52:
            self._track_handle(self.state_0.particle_q.numpy())
        jaw_rotation = wp.quat(*self.grip_rotation)
        if t < 0.15:
            self.phase = "Approach the upper handle"
            target = self.grip_center + self.approach_axis * (0.09 * (1 - smooth_step(t / 0.15)))
            gap = 0.055
        elif t < 0.6:
            self.phase = "Close the two jaws"
            target = self.grip_center
            gap = 0.055 + (0.0055 - 0.055) * smooth_step((t - 0.15) / 0.45)
        else:
            gap = 0.0055
            if self.lift_baseline is None:
                self.lift_baseline = self.state_0.particle_q.numpy()[self.bag["bottom"], 2].mean()
            height = self.args.lift_height * smooth_step((t - 0.8) / 3.0)
            target = self.grip_center + np.array((0, 0, height))
            self.phase = "Hold the grip" if t < 0.8 else "Lift; let the bag hang under gravity"
        axis = np.asarray(wp.quat_rotate(jaw_rotation, wp.vec3(1, 0, 0)))
        jaw_rotation = np.asarray(jaw_rotation)
        poses = [[*(target + sign * (0.018 + gap / 2) * axis), *jaw_rotation] for sign in (-1, 1)]

        # Raise beside the bag first, then translate above the opening. Use the
        # actual mouth position after lifting, not the undeformed asset's rim.
        tray = self.tray_home.copy()
        delivery_height = self.initial_grip[2] + self.args.lift_height + 0.14
        tray[2] += (delivery_height - tray[2]) * smooth_step((t - 0.8) / 2.0)
        if t >= 3.8:
            self.phase = "Deliver two blocks above the mouth"
            mouth = self.state_0.particle_q.numpy()[self.bag["rim"]].mean(0)
            if self.delivery_target is None:
                self.delivery_target = mouth.copy()
                self.delivery_target[2] = delivery_height
            elif t < 5.4:
                # Follow the swinging opening gently enough for tray friction
                # to carry the blocks. Freeze the tray target once release starts.
                delta = (mouth[:2] - self.delivery_target[:2]) * (1 - np.exp(-self.frame_dt / 0.1))
                self.delivery_target[:2] += np.clip(delta, -0.06 * self.frame_dt, 0.06 * self.frame_dt)
            alpha = smooth_step((t - 3.8) / 1.6)
            tray = (1 - alpha) * tray + alpha * self.delivery_target
        if t >= 5.4:
            self.phase = "Withdraw tray; drop the load" if t < 5.65 else "Hold the loaded bag"
            tray += np.array((0, 0.22, -0.20)) * smooth_step((t - 5.4) / 0.25)
        poses.append([*tray, 0, 0, 0, 1])
        poses = np.asarray(poses, dtype=np.float32)
        self.driven_start.assign(self.last_driven_poses)
        self.driven_end.assign(poses)
        self.last_driven_poses = poses

    def _simulate(self):
        for substep in range(self.args.substeps):
            if self.jaws:
                wp.launch(
                    _move_bodies,
                    dim=len(self.driven_bodies),
                    inputs=[
                        self.driven_body_ids,
                        self.driven_start,
                        self.driven_end,
                        (substep + 1) / self.args.substeps,
                        self.frame_dt,
                        self.state_0.body_q,
                        self.state_0.body_qd,
                    ],
                    device=self.model.device,
                )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Step gravity, self-contact and mouse interaction, with optional CUDA capture."""
        self._prepare_task()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate()
            if (
                self.frame == 2
                and self.model.device.is_cuda
                and not self.args.no_cuda_graph
                and self.args.substeps % 2 == 0
            ):
                saved_0, saved_1 = self.model.state(), self.model.state()
                saved_0.assign(self.state_0)
                saved_1.assign(self.state_1)
                with wp.ScopedCapture() as capture:
                    self._simulate()
                self.state_0.assign(saved_0)
                self.state_1.assign(saved_1)
                self.graph = capture.graph
        self.frame += 1
        self.sim_time += self.frame_dt

    def reset_physics(self, *, source="viewer"):
        """Restore the initial pose and material rest state without rebuilding assets."""
        self.state_0.assign(self.initial_state)
        self.state_1.assign(self.initial_state)
        self.solver.reset(self.state_0, flags=0)
        self.solver.reset(self.state_1, flags=0)
        self.frame, self.sim_time = 0, 0.0
        self._reset_task()

    def gui(self, ui):
        """Expose simple inspection controls without teleoperation setup."""
        ui.text("PVC film / stiff molded handles")
        ui.text(self.phase)
        ui.text("Right-drag to pull; Space to pause")
        _, self.opacity = ui.slider_float("Film opacity", self.opacity, 0.05, 1.0)
        if ui.button("Reset bag"):
            self.reset_physics()

    def render(self):
        """Render translucent film and smoother reinforced handles."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        for name, indices, color, opacity, roughness in zip(
            ("film", "handles"),
            self.render_indices,
            (FILM_COLOR, HANDLE_COLOR),
            (self.opacity, 0.72),
            (0.22, 0.14),
            strict=True,
        ):
            self.viewer.log_mesh(
                f"/pvc/{name}",
                self.state_0.particle_q,
                indices,
                color=color,
                opacity=opacity,
                roughness=roughness,
                metallic=0.0,
                backface_culling=False,
            )
        self.viewer.end_frame()

    def test_final(self):
        """Check free dynamics, jaw contact, and completed load-bearing lifts."""
        q, qd = self.state_0.particle_q.numpy(), self.state_0.particle_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise AssertionError("PVC bag state is nonfinite")
        if q[:, 2].min() < -0.02 or np.linalg.norm(qd, axis=1).max() > 20:
            raise AssertionError("PVC bag escaped contact or became unstable")
        if np.any(self.model.particle_inv_mass.numpy() <= 0):
            raise AssertionError("PVC film and handles must remain unpinned")
        bodies = self.state_0.body_q.numpy()
        if not np.isfinite(bodies).all() or not np.isfinite(self.state_0.body_qd.numpy()).all():
            raise AssertionError("Rigid body state is nonfinite")
        if not self.args.task:
            return
        handle = q[self.bag["handles"][self.handle_index]].reshape(-1, 3)
        for pose in bodies[self.jaws]:
            rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)
            local = (handle - pose[:3]) @ rotation
            penetration = np.min(np.array((0.018, 0.022, 0.022)) - np.abs(local), axis=1)
            if penetration.max() > 0.004:
                raise AssertionError("The stiff handle penetrated a jaw by more than 4 mm")
        # The sheet must fold, while each hollow handle retains its U curve.
        rest = self.initial_state.particle_q.numpy()
        groups = []
        for index, tube in enumerate(self.bag["handles"]):
            groups.append((f"handle {index}", tube[[0, (len(tube) - 1) // 2, -1], 0]))
        for name, indices in groups:
            current = q[indices]
            initial = rest[indices]
            distances = np.linalg.norm(initial[:, None] - initial[None, :], axis=-1)
            lengths = np.linalg.norm(current[:, None] - current[None, :], axis=-1)
            valid = distances > 0.02
            if np.max(np.abs(lengths[valid] / distances[valid] - 1)) > 0.12:
                raise AssertionError(f"The molded PVC {name} changed its span by more than 12%")
        if self.sim_time >= 6.5 and self.args.lift_height >= 0.5:
            if q[self.bag["bottom"], 2].min() < self.table_height + 0.015:
                raise AssertionError("The bag bottom did not lift clear of the table")
            if bodies[self.payloads, 2].min() < self.table_height + 0.04:
                raise AssertionError("The rigid load did not lift with the bag")
            center = bodies[self.jaws, :3].mean(0)
            if np.linalg.norm(handle - center, axis=1).min() > 0.03:
                raise AssertionError("The handle slipped out of the jaws")
            vertices = np.vstack((q, q[self.bag["rim"]].mean(0))).astype(np.float64)
            for body in self.payloads:
                triangles = vertices[self.containment_faces] - bodies[body, :3]
                a, b, c = np.moveaxis(triangles, 1, 0)
                lengths = np.linalg.norm(triangles, axis=2)
                numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
                denominator = lengths.prod(1)
                denominator += np.einsum("ij,ij->i", a, b) * lengths[:, 2]
                denominator += np.einsum("ij,ij->i", b, c) * lengths[:, 0]
                denominator += np.einsum("ij,ij->i", c, a) * lengths[:, 1]
                winding = np.sum(2 * np.arctan2(numerator, denominator)) / (4 * np.pi)
                if abs(winding) < 0.95:
                    raise AssertionError("A payload was not delivered inside the lifted bag")

    @classmethod
    def create_parser(cls):
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=100000)
        parser.add_argument("--pvc-assets", type=Path, default=Path.home() / "下载/clear_plastic_bag")
        parser.add_argument("--task", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--pose", choices=("upright", "lying"), default="lying")
        parser.add_argument("--opacity", type=float, default=0.32)
        parser.add_argument("--lift-height", type=float, default=0.5, help="Upward jaw travel after gripping [m].")
        parser.add_argument("--handle-stiffness", type=float, default=1.0)
        parser.add_argument("--no-cuda-graph", action="store_true")
        parser.add_argument("--substeps", type=int, default=6)
        parser.add_argument("--iterations", type=int, default=24)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)

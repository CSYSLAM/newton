# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""W1 grasps an open Wuliangye bottle and pours MPM liquid into a stemmed glass.

    uv run --extra examples -m newton.examples mjmpm_wuliangye_pour

Experimental: a reference-inspired bottle asset and illustrative liquid
parameters, not a calibrated model of the commercial beverage. The bottle
is a free body held by finger contact; liquid is neither emitted nor deleted.
"""

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import warp as wp
import warp.fem as fem

import newton
import newton.examples
from newton.examples.mjmpm.support.w1_two_way import (
    FINGER_OPEN,
    FINGER_SHUT,
    HORIZONTAL_GRIPPER,
    TABLE_Z,
    W1MJMPMScene,
    smoothstep,
)
from newton.viewer import ViewerGL

ASSETS = Path(__file__).resolve().parent / "assets/wuliangye_pour"
BOTTLE_CENTER = np.array((0.40, -0.25, TABLE_Z + 0.081))
BOTTLE_ROTATION = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(np.deg2rad(60.0)))
POUR_DIRECTION = np.array((0.5, np.sqrt(0.75), 0.0))
GLASS_CENTER = np.array((0.44, -0.25, TABLE_Z)) + 0.112 * POUR_DIRECTION
GRIP = np.array((0.0, 0.0, 0.068))
# Coarser sampling halves the count while preserving approximately the same
# liquid volume; particle mass and simulation radius follow this spacing.
SPACING = 0.0025 * 2.0 ** (1.0 / 3.0)
FILL_HEIGHT = 0.165
END_FRAME = 1320


def liquid_points():
    """Sample a measured charge above the bottle floor with wall clearance."""
    x, y, z = np.meshgrid(
        np.arange(-0.034, 0.035, SPACING),
        np.arange(-0.034, 0.035, SPACING),
        np.arange(0.019, FILL_HEIGHT, SPACING),
        indexing="ij",
    )
    radius = (
        0.8 * np.interp(z, [0.015, 0.070, 0.099, 0.125, 0.140, 0.180], [0.040, 0.042, 0.038, 0.021, 0.014, 0.014])
        - 0.001
    )
    inside = x * x + y * y < radius**2
    return np.column_stack((x[inside], y[inside], z[inside]))


def tilt_rotation(angle):
    return wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(angle))


@wp.kernel
def limit_arm_targets(
    coordinates: wp.array[int],
    previous: wp.array[float],
    targets: wp.array[float],
    ik_q: wp.array2d[float],
    max_delta: float,
):
    i = coordinates[wp.tid()]
    value = wp.clamp(targets[i], previous[i] - max_delta, previous[i] + max_delta)
    targets[i] = value
    ik_q[0, i] = value


class Example(W1MJMPMScene):
    gripper_rotation = wp.vec4(*wp.mul(BOTTLE_ROTATION, wp.quat(*HORIZONTAL_GRIPPER)))
    home_offset = -0.13 * POUR_DIRECTION + np.array((0.0, 0.0, 0.10))
    camera_pos = wp.vec3(0.83, -0.73, 1.36)
    camera_pitch = -24.0
    camera_yaw = 119.0

    def initial_targets(self):
        return np.array(((0.40, 0.25, TABLE_Z + 0.15), BOTTLE_CENTER + GRIP))

    def build_task(self, builder):
        # A dark tabletop gives clear glass edges contrast without increasing
        # wall opacity and hiding the liquid.
        for shape, label in enumerate(builder.shape_label):
            if label == "worktop":
                builder.shape_color[shape] = (0.28, 0.23, 0.18)
        # The bottle wall separates the fingers from the liquid. Avoid
        # sampling the nearby finger surfaces through that thin wall.
        for shape, body in enumerate(builder.shape_body):
            if 0 <= body < self.robot_bodies:
                builder.shape_flags[shape] &= ~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        self.grasp = BOTTLE_CENTER + GRIP
        self.add_box(
            builder,
            "bottle_rest",
            (0.40, -0.25, TABLE_Z + 0.04),
            (0.045, 0.045, 0.04),
            (0.30, 0.18, 0.10),
            cfg=replace(self.surface_cfg, is_visible=False),
        )
        self.bottle = builder.add_body(
            xform=wp.transform(wp.vec3(*BOTTLE_CENTER), BOTTLE_ROTATION), label="Wuliangye_bottle"
        )
        with np.load(ASSETS / "props.npz") as assets:
            self.glass_profile = assets["glass_inner_profile"].copy()
            for name in assets["names"].tolist():
                vertices, faces = assets[name + "_vertices"].copy(), assets[name + "_faces"]
                normals = assets[name + "_normals"].astype(np.float32) if name + "_normals" in assets else None
                group = str(assets[name + "_group"])
                if group in ("bottle", "cap"):
                    vertices[:, :2] *= 0.8
                    if normals is not None:
                        normals[:, :2] /= 0.8
                        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
                on_bottle = group == "bottle"
                position = (
                    (0.0, 0.0, 0.0) if on_bottle else GLASS_CENTER if group == "glass" else (0.40, -0.37, TABLE_Z)
                )
                if group == "rest":
                    position = (0.40, -0.25, TABLE_Z)
                collider = name in ("bottle", "glass")
                shape = builder.add_shape_mesh(
                    self.bottle if on_bottle else -1,
                    xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
                    mesh=newton.Mesh(
                        vertices,
                        faces.ravel(),
                        normals=normals,
                        uvs=assets[name + "_uvs"] if name + "_uvs" in assets else None,
                        texture=str(ASSETS / "label.jpg") if name == "label" else None,
                        roughness=float(assets[name + "_roughness"]),
                        metallic=float(assets[name + "_metallic"]),
                        compute_inertia=name == "bottle",
                        is_solid=collider,
                    ),
                    cfg=newton.ModelBuilder.ShapeConfig(
                        density=650.0 if name == "bottle" else 0.0,
                        ke=2.0e4,
                        kd=100.0,
                        mu=0.05,
                        has_shape_collision=collider,
                        has_particle_collision=collider,
                        is_visible=not collider,
                    ),
                    color=tuple(assets[name + "_color"]),
                    opacity=float(assets[name + "_opacity"]),
                    label=name,
                )
                if name in ("bottle", "glass"):
                    # Fingers contact the exterior grip patches rather than
                    # both sides of the hollow bottle wall. The static glass
                    # receives liquid through its full MPM collision mesh.
                    for other in range(shape):
                        if builder.shape_body[other] >= 0 and (
                            name == "glass" or builder.shape_body[other] < self.robot_bodies
                        ):
                            builder.add_shape_collision_filter_pair(shape, other)
        # Exterior patches approximate the faceted bottle sides near its
        # center of mass, where the gripper has enough torque to tilt it.
        collar_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0, ke=2.0e4, kd=100.0, mu=6.0, has_particle_collision=False, is_visible=False
        )
        for z in (0.056, 0.080):
            for y in (-0.0355, 0.0355):
                builder.add_shape_box(
                    self.bottle,
                    xform=wp.transform(wp.vec3(0.0, y, z), wp.quat_identity()),
                    hx=0.016,
                    hy=0.002,
                    hz=0.010,
                    cfg=collar_cfg,
                    color=(0.78, 0.53, 0.13),
                    label="bottle_grip_patch",
                )
        self.liquid_begin = builder.particle_count
        points = liquid_points()
        for point in points:
            builder.add_particle(
                wp.vec3(*(BOTTLE_CENTER + np.asarray(wp.quat_rotate(BOTTLE_ROTATION, wp.vec3(*point))))),
                wp.vec3(0.0),
                920.0 * SPACING**3,
                0.5 * SPACING,
                custom_attributes={
                    "mpm:friction": 0.0,
                    "mpm:tensile_yield_ratio": 1.0,
                    "mpm:viscosity": 0.002,
                    "mpm:hardening_rate": 0.0,
                    "mpm:softening_rate": 0.0,
                },
            )
        self.liquid_end = builder.particle_count
        print(f"[WuliangyePour] {len(points)} liquid particles, {len(points) * SPACING**3 * 1e6:.1f} ml", flush=True)

    def mpm_option_overrides(self):
        return {
            "grid_type": "sparse",
            "grid_padding": 0,
            "max_active_cell_count": 4096,
            "collider_basis": "pic1024",
            "air_drag": 1.0e-3,
            "transfer_scheme": "pic",
            "integration_scheme": "pic",
            "tolerance": 1.0e-4,
            "critical_fraction": 1.0,
            "density_correction": 0.1,
            "warmstart_mode": "none",
        }

    def _sample(self, time):
        lifted = self.grasp + np.array((0.0, 0.0, 0.12))
        # Leave room for the outgoing jet's horizontal travel before it reaches
        # the rim. A higher initial charge starts flowing earlier in the tilt.
        upright = np.array((0.44, -0.25, TABLE_Z + 0.27)) - 0.018 * POUR_DIRECTION
        inclined = upright
        pouring = upright
        target, angle, finger = smoothstep(
            [
                (0.0, self.home[1], 0.0, FINGER_OPEN),
                (0.6, self.home[1], 0.0, FINGER_OPEN),
                (1.6, self.grasp - 0.10 * POUR_DIRECTION, 0.0, FINGER_OPEN),
                (2.6, self.grasp, 0.0, FINGER_OPEN),
                (3.5, self.grasp, 0.0, FINGER_SHUT),
                (4.0, self.grasp, 0.0, FINGER_SHUT),
                (5.5, lifted, 0.0, FINGER_SHUT),
                (6.5, upright, 0.0, FINGER_SHUT),
                (8.5, inclined, np.deg2rad(70.0), FINGER_SHUT),
                (11.0, pouring, np.deg2rad(120.0), FINGER_SHUT),
                (14.0, pouring, np.deg2rad(120.0), FINGER_SHUT),
                (16.0, inclined, np.deg2rad(70.0), FINGER_SHUT),
                (17.5, upright, 0.0, FINGER_SHUT),
                (18.5, lifted, 0.0, FINGER_SHUT),
                (20.0, self.grasp, 0.0, FINGER_SHUT),
                (21.0, self.grasp, 0.0, FINGER_OPEN),
                (22.0, self.home[1], 0.0, FINGER_OPEN),
            ],
            time,
        )
        return np.asarray(target), float(angle), float(finger)

    def plan(self, time):
        target, _, finger = self._sample(time)
        return np.array((self.home[0], target)), np.array((FINGER_OPEN, finger))

    def gripper_rotations(self, time):
        _, angle, _ = self._sample(time)
        return self.gripper_rotation, wp.mul(
            BOTTLE_ROTATION, wp.mul(tilt_rotation(angle), wp.quat(*HORIZONTAL_GRIPPER))
        )

    def tracks_freely(self, time):
        return time < 2.6

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        # Reuse FEM scratch allocations in a scene-owned pool without changing
        # Warp's global store or other examples.
        self.solver.environment.material.temporary_store = fem.TemporaryStore()
        if isinstance(viewer, ViewerGL):
            viewer.set_transparent_reflections(True)
        self.arm_coordinates = wp.array(
            [self.coords[f"{side}_J{j}"] for side in ("LEFT", "RIGHT") for j in range(1, 8)],
            dtype=int,
            device=self.model.device,
        )
        self.count = self.liquid_end - self.liquid_begin
        self.radii = wp.full(self.count, 0.66 * SPACING, dtype=float, device=self.model.device)
        # Pale blue-white keeps a colorless spirit visible without surface reconstruction.
        self.colors = wp.full(self.count, wp.vec3(0.55, 0.80, 0.94), dtype=wp.vec3, device=self.model.device)
        self.viewer.show_particles = False
        self.in_glass = 0
        self.peak_in_glass = 0
        self.peak_lift = 0.0
        self.max_grasp_error = 0.0
        self.settled_fill_height = None
        self._fps_start = None
        self._fps_reported = False
        self._screenshots_saved = set()

    def _solve_frame_ik(self):
        super()._solve_frame_ik()
        # Redundant IK can switch branches while leaving the TCP almost
        # unchanged. Bound motor target speed instead of jerking a held bottle.
        wp.launch(
            limit_arm_targets,
            dim=14,
            inputs=[self.arm_coordinates, self.frame_start, self.frame_end, self.ik_q, 1.2 * self.frame_dt],
            device=self.model.device,
        )

    def step(self):
        if self.frame < END_FRAME:
            super().step()

    def glass_mask(self):
        points = self.state_0.particle_q.numpy()[self.liquid_begin : self.liquid_end] - GLASS_CENTER
        radius = np.interp(points[:, 2], self.glass_profile[:, 1], self.glass_profile[:, 0])
        return (
            (np.linalg.norm(points[:, :2], axis=1) < radius + 0.002) & (points[:, 2] > 0.093) & (points[:, 2] < 0.197)
        )

    def glass_volume_estimate(self):
        """Compare reference volume with a settled, level-surface cavity estimate.

        This diagnostic assumes a filled, axisymmetric bowl below the surface;
        it is not a reconstruction of arbitrary splashes or interior voids.
        """
        points = self.state_0.particle_q.numpy()[self.liquid_begin : self.liquid_end] - GLASS_CENTER
        points = points[self.glass_mask()]
        reference = len(points) * SPACING**3
        if len(points) == 0:
            return reference, 0.0
        height = float(np.quantile(points[:, 2], 0.99)) + 0.5 * SPACING
        profile = self.glass_profile
        height = np.clip(height, profile[0, 1], profile[-1, 1])
        below = profile[:, 1] < height
        z = np.append(profile[below, 1], height)
        radius = np.append(profile[below, 0], np.interp(height, profile[:, 1], profile[:, 0]))
        volume = np.pi / 3.0 * np.sum(np.diff(z) * (radius[:-1] ** 2 + radius[:-1] * radius[1:] + radius[1:] ** 2))
        return reference, float(volume)

    def after_step(self):
        if self.frame in (120, 240):
            pose = self.state_0.body_q.numpy()[self.bottle]
            points = self.state_0.particle_q.numpy()[self.liquid_begin : self.liquid_end]
            local = (points - pose[:3]) @ np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)
            self.settled_fill_height = float(np.quantile(local[:, 2], 0.99))
            print(f"[WuliangyePour] settled liquid height: {self.settled_fill_height:.3f} m", flush=True)
            if self.args.test and self.settled_fill_height < 0.090:
                raise AssertionError("Settled liquid must reach above half the 18 cm bottle height")
        if self.frame % 30 == 0:
            self.in_glass = int(self.glass_mask().sum())
            self.peak_in_glass = max(self.peak_in_glass, self.in_glass)
            if self.args.test and self.peak_in_glass > 100 and self.in_glass < 0.9 * self.peak_in_glass:
                raise AssertionError(f"Liquid left the glass: {self.in_glass}/{self.peak_in_glass}")
        if self.args.test and self.frame % 10 == 0:
            poses = self.state_0.body_q.numpy()
            self.peak_lift = max(self.peak_lift, float(poses[self.bottle, 2] - BOTTLE_CENTER[2]))
            if 4.5 < self.sim_time < 20.0:
                grip = np.asarray(wp.transform_point(wp.transform(*poses[self.bottle]), wp.vec3(*GRIP)))
                self.max_grasp_error = max(
                    self.max_grasp_error, float(np.linalg.norm(grip - self.tcp_positions(poses)[1]))
                )

    def test_post_step(self):
        """Check every simulated state for nonfinite values."""
        self.check_finite()

    def test_final(self):
        """Verify delivery, grasp retention, and an upright returned bottle."""
        self.check_finite()
        self.solver.check_capacity_status()
        if self.frame < END_FRAME:
            raise AssertionError("Run the complete 22-second grasp, pour, return, and settle sequence")
        fraction = float(self.glass_mask().mean())
        pose = self.state_0.body_q.numpy()[self.bottle]
        up = np.asarray(wp.quat_rotate(wp.quat(*pose[3:]), wp.vec3(0, 0, 1)))
        if fraction < 0.95:
            raise AssertionError(f"Only {fraction:.1%} of liquid reached the glass")
        if self.peak_lift < 0.065 or self.max_grasp_error > 0.025:
            raise AssertionError(f"Bottle grasp failed: lift={self.peak_lift:.3f}, slip={self.max_grasp_error:.3f}m")
        if np.linalg.norm(pose[:3] - BOTTLE_CENTER) > 0.03 or up[2] < 0.97:
            raise AssertionError(f"Bottle was not returned upright: {pose}")
        points = self.state_0.particle_q.numpy()[self.liquid_begin : self.liquid_end]
        local = (points - pose[:3]) @ np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)
        radius = 0.8 * np.interp(
            local[:, 2],
            [0.015, 0.070, 0.099, 0.125, 0.140, 0.180],
            [0.040, 0.042, 0.038, 0.021, 0.014, 0.014],
        )
        in_bottle = (
            (np.linalg.norm(local[:, :2], axis=1) < radius + 0.002) & (local[:, 2] > 0.012) & (local[:, 2] < 0.183)
        )
        escaped = int(np.count_nonzero(~(in_bottle | self.glass_mask())))
        print(f"[WuliangyePour] delivery={fraction:.1%}, escaped={escaped}/{self.count}", flush=True)
        if escaped > 0.005 * self.count:
            raise AssertionError(f"{escaped} liquid particles escaped both vessels")
        flags = self.model.particle_flags.numpy()[self.liquid_begin : self.liquid_end]
        if not np.all(flags & int(newton.ParticleFlags.ACTIVE)):
            raise AssertionError("Liquid particles must stay active throughout the demonstration")
        reference, occupied = self.glass_volume_estimate()
        print(f"[WuliangyePour] reference={reference * 1e6:.1f} ml, level estimate={occupied * 1e6:.1f} ml", flush=True)
        if not 0.90 * reference <= occupied <= 1.10 * reference:
            raise AssertionError("Settled liquid volume estimate differs from reference by more than 10%")
        print(f"[WuliangyePour] PASS: {fraction:.1%} in glass, grasp error {self.max_grasp_error:.4f}m", flush=True)

    def render_ui(self, imgui):
        super().render_ui(imgui)
        imgui.text(f"Wuliangye in glass: {self.in_glass}/{self.count} particles")
        imgui.text(f"Reference liquid volume: {self.in_glass * SPACING**3 * 1e6:.1f} ml")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "/liquor", self.state_0.particle_q[self.liquid_begin : self.liquid_end], self.radii, self.colors
        )
        self.viewer.end_frame()
        if self.args.report_fps:
            if self.frame == 60:
                self._fps_start = time.perf_counter()
            elif self.frame == min(self.args.num_frames, END_FRAME) and self._fps_start and not self._fps_reported:
                print(f"[WuliangyePour] viewer: {(self.frame - 60) / (time.perf_counter() - self._fps_start):.1f} FPS")
                self._fps_reported = True
        if (
            self.args.debug_screenshots
            and self.frame in (1, 240, 360, 480, 540, 600, 630, 660, 690, 720, 810, 960, 1080, 1200, 1320)
            and self.frame not in self._screenshots_saved
        ):
            from PIL import Image  # noqa: PLC0415

            output = Path(self.args.debug_screenshots)
            output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(self.viewer.get_frame().numpy()).save(output / f"frame_{self.frame:04d}.jpg", quality=95)
            self._screenshots_saved.add(self.frame)

    @staticmethod
    def create_parser():
        parser = W1MJMPMScene.add_arguments(newton.examples.create_parser(), num_frames=END_FRAME)
        parser.add_argument("--debug-screenshots", type=str, default=None)
        parser.add_argument("--report-fps", action="store_true")
        parser.set_defaults(
            # Graph replay is validated for this reduced-particle PIC preset.
            # The previous high-charge APIC preset required eager execution.
            no_cuda_graph=False,
            substeps=4,
            ik_iterations=8,
            rigid_iterations=60,
            finger_kp=2000.0,
            finger_kd=60.0,
            finger_armature=0.5,
            coupling_iterations=1,
            # Keep about two particles per voxel axis with the coarser fill.
            voxel_size=0.006,
            strain_basis="P0",
            mpm_solver="gs",
            iterations=256,
            stall_tolerance=0.0,
            rigid_material_coupling="constrained",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

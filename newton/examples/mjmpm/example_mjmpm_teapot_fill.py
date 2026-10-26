# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""W1 grasps the Utah Teapot and pours MPM water into a bowl.

    uv run --extra examples -m newton.examples mjmpm_teapot_fill

The W1 arm and fingers are MuJoCo bodies. The teapot is a free rigid body held
by finger contact, and water is implicit MPM material. MJMPM feeds the teapot
and water reactions back through the grasp to the robot. Water is drawn as particles.
"""

import time
from pathlib import Path

import numpy as np
import warp as wp

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

ASSETS = Path(__file__).resolve().parent / "assets/teapot_fill"
TEAPOT_CENTER = np.array((0.43, -0.25, TABLE_Z), dtype=np.float32)
BOWL_CENTER = np.array((0.63, -0.25, TABLE_Z), dtype=np.float32)
HANDLE_CENTER = np.array((-0.091, 0.0, 0.053), dtype=np.float32)
HANDLE_BOXES = (((-0.091, 0.0, 0.053), (0.012, 0.011, 0.022)),)
WATER_SPACING = 0.004
WATER_DENSITY = 1000.0
POUR_ANGLE = np.deg2rad(80.0)
END_FRAME = 900


@wp.kernel
def _retire_out_of_scene_water(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    model_flags: wp.array[int],
    mpm_flags: wp.array[int],
    material_flags: wp.array[int],
    first: int,
    active_flag: int,
    min_z: float,
):
    i = first + wp.tid()
    if (mpm_flags[i] & active_flag) == 0:
        return
    p = positions[i]
    v = velocities[i]
    invalid = (
        not wp.isfinite(p[0])
        or not wp.isfinite(p[1])
        or not wp.isfinite(p[2])
        or not wp.isfinite(v[0])
        or not wp.isfinite(v[1])
        or not wp.isfinite(v[2])
    )
    if invalid or p[2] < min_z or p[0] < -0.5 or p[0] > 1.5 or p[1] < -1.5 or p[1] > 1.5:
        # Keep the state finite even after deactivation; renderers and test
        # diagnostics may still read positions of inactive particles.
        if invalid:
            positions[i] = wp.vec3(0.0, 0.0, min_z - 1.0)
            velocities[i] = wp.vec3(0.0, 0.0, 0.0)
        model_flags[i] = model_flags[i] & ~active_flag
        mpm_flags[i] = mpm_flags[i] & ~active_flag
        material_flags[i] = material_flags[i] & ~active_flag


def bowl_mesh(segments=64):
    """Return the closed, open-top catch bowl collider in local coordinates."""
    angles = np.arange(segments) * (2.0 * np.pi / segments)
    # Extend the catch area along the pouring direction without moving the
    # bowl into the teapot's starting pose.
    unit = np.stack((1.4 * np.cos(angles), np.sin(angles)), axis=1)
    rings = [(0.0254, 0.0049), (0.0508, 0.04357), (0.0526, 0.04357), (0.0301, 0.0)]
    vertices = [np.column_stack((radius * unit, np.full(segments, z))) for radius, z in rings]
    vertices.extend((np.array([[0.0, 0.0, 0.0049]]), np.array([[0.0, 0.0, 0.0]])))
    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        for a, b, c, d in (
            (i, i + segments, j, j + segments),
            (i + 3 * segments, j + 3 * segments, i + 2 * segments, j + 2 * segments),
            (i + segments, i + 2 * segments, j + segments, j + 2 * segments),
        ):
            faces.extend(((a, b, c), (c, b, d)))
        faces.extend(((4 * segments, i, j), (4 * segments + 1, j + 3 * segments, i + 3 * segments)))
    return np.concatenate(vertices).astype(np.float32), np.asarray(faces, dtype=np.int32)


def water_points(spacing=WATER_SPACING):
    """Sample the teapot's hollow center, inset from its ceramic shell."""
    points = []
    for z in np.arange(0.011, 0.061, spacing):
        radius = 0.033 if z < 0.02 else 0.037 if z < 0.045 else 0.035
        for x in np.arange(-radius, radius + 0.5 * spacing, spacing):
            for y in np.arange(-radius, radius + 0.5 * spacing, spacing):
                if x * x + y * y < (radius - 0.5 * spacing) ** 2:
                    points.append((x, y, z))
    return np.asarray(points, dtype=np.float32)


def _tilt(angle):
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(angle))
    return wp.mul(rotation, wp.quat(*HORIZONTAL_GRIPPER))


class Example(W1MJMPMScene):
    camera_pos = wp.vec3(0.95, -0.83, 1.38)
    camera_pitch = -25.0
    camera_yaw = 120.0

    def initial_targets(self):
        grasp = TEAPOT_CENTER + HANDLE_CENTER
        return np.array(((0.43, 0.25, TABLE_Z + 0.12), grasp))

    def build_task(self, builder):
        self.grasp = self.initial_targets()[1]
        teapot_cfg = newton.ModelBuilder.ShapeConfig(density=400.0, ke=2.0e4, kd=100.0, mu=0.8, is_visible=False)
        self.teapot = builder.add_body(
            xform=wp.transform(wp.vec3(*TEAPOT_CENTER), wp.quat_identity()), label="Utah_Teapot"
        )
        with np.load(ASSETS / "utah_teapot_collision_12k.npz") as asset:
            vertices, faces = asset["vertices"], asset["faces"]
        teapot_max = TEAPOT_CENTER[:2] + vertices[:, :2].max(axis=0)
        teapot_min = TEAPOT_CENTER[:2] + vertices[:, :2].min(axis=0)
        builder.add_shape_mesh(
            self.teapot,
            mesh=newton.Mesh(vertices, faces.ravel()),
            cfg=teapot_cfg,
            label="Utah_Teapot_Collider",
        )
        with np.load(ASSETS / "utah_teapot_collision.npz") as asset:
            vertices, faces = asset["vertices"], asset["faces"]
        builder.add_shape_mesh(
            self.teapot,
            mesh=newton.Mesh(vertices, faces.ravel(), compute_inertia=False, is_solid=False),
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False),
            color=(0.70, 0.35, 0.16),
            label="Modified_Utah_Teapot",
        )
        handle_cfg = newton.ModelBuilder.ShapeConfig(density=400.0, ke=1.0e5, kd=100.0, mu=6.0, is_visible=False)
        for center, half in HANDLE_BOXES:
            builder.add_shape_box(
                self.teapot,
                xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                hx=half[0],
                hy=half[1],
                hz=half[2],
                cfg=handle_cfg,
                label="teapot_grip",
            )

        vertices, faces = bowl_mesh()
        bowl_min = BOWL_CENTER[:2] + vertices[:, :2].min(axis=0)
        bowl_max = BOWL_CENTER[:2] + vertices[:, :2].max(axis=0)
        clearance = np.maximum(bowl_min - teapot_max, teapot_min - bowl_max)
        if clearance.max() < 0.01:
            raise ValueError("The teapot and bowl need at least 10 mm of clearance at startup")
        # Settled water sits below the near rim; translucency keeps it visible.
        bowl = builder.add_shape_mesh(
            -1,
            xform=wp.transform(wp.vec3(*BOWL_CENTER), wp.quat_identity()),
            mesh=newton.Mesh(vertices, faces.ravel(), compute_inertia=False, is_solid=False),
            cfg=newton.ModelBuilder.ShapeConfig(ke=2.0e4, kd=100.0, mu=0.2),
            color=(0.93, 0.94, 0.91),
            opacity=0.55,
            label="catch_bowl",
        )
        # The bowl receives MPM water but must not interfere with the nearby
        # rigid teapot or gripper during pickup.
        for shape in range(bowl):
            if builder.shape_body[shape] >= 0:
                builder.add_shape_collision_filter_pair(bowl, shape)
        self.water_begin = builder.particle_count
        points = water_points()
        for point in points:
            builder.add_particle(
                wp.vec3(*(TEAPOT_CENTER + point)),
                wp.vec3(0.0),
                WATER_DENSITY * WATER_SPACING**3,
                0.5 * WATER_SPACING,
                custom_attributes={"mpm:friction": 0.0, "mpm:tensile_yield_ratio": 1.0, "mpm:viscosity": 1.0e-3},
            )
        self.water_end = builder.particle_count
        print(f"[W1TeapotFill] {len(points)} MPM water particles", flush=True)

    def mpm_option_overrides(self):
        return {
            "grid_type": "sparse",
            "grid_padding": 0,
            "max_active_cell_count": 16384,
            "collider_basis": "pic512",
            "air_drag": 1.0e-3,
            "transfer_scheme": "apic",
            "integration_scheme": "pic",
            "tolerance": 1.0e-4,
        }

    def _keyframes(self):
        g = self.grasp
        behind = g + np.array((-0.13, 0.0, 0.0))
        lifted = g + np.array((0.0, 0.0, 0.20))
        over_bowl = lifted + np.array((0.16, 0.0, 0.0))
        pouring = lifted + np.array((0.21, 0.0, 0.0))
        return [
            (0.0, self.home[1], 0.0, FINGER_OPEN),
            (0.6, self.home[1], 0.0, FINGER_OPEN),
            (1.6, behind, 0.0, FINGER_OPEN),
            (2.6, g, 0.0, FINGER_OPEN),
            (3.5, g, 0.0, FINGER_SHUT),
            (4.0, g, 0.0, FINGER_SHUT),
            (5.3, lifted, 0.0, FINGER_SHUT),
            (6.8, over_bowl, 0.0, FINGER_SHUT),
            (8.2, pouring, POUR_ANGLE, FINGER_SHUT),
            (12.0, pouring, POUR_ANGLE, FINGER_SHUT),
        ]

    def _sample(self, time):
        target, tilt, finger = smoothstep(self._keyframes(), time)
        return np.asarray(target), float(tilt), float(finger)

    def plan(self, time):
        target, _tilt_angle, finger = self._sample(time)
        return np.array((self.home[0], target)), np.array((FINGER_OPEN, finger))

    def gripper_rotations(self, time):
        _target, tilt, _finger = self._sample(time)
        return self.gripper_rotation, _tilt(tilt)

    def tracks_freely(self, time):
        return time <= 2.6

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        count = self.water_end - self.water_begin
        self.water_radii = wp.full(count, 0.003, dtype=float, device=self.model.device)
        self.water_colors = wp.full(count, wp.vec3(0.12, 0.42, 0.9), dtype=wp.vec3, device=self.model.device)
        self.viewer.show_particles = False
        self.peak_lift = 0.0
        self.peak_tilt = 0.0
        self.bowl_particle_count = 0
        self.peak_bowl_particle_count = 0
        self.retired_particle_count = 0
        self.pour_end_particle_count = 0
        mpm_model = self.solver.environment.material._mpm_model
        self.mpm_particle_flags = mpm_model.particle_flags
        self.mpm_material_flags = mpm_model.material_particle_flags
        self._fps_start = None
        self._fps_reported = False
        self._screenshots_saved = set()
        print(f"[W1TeapotFill] teapot mass {float(self.model.body_mass.numpy()[self.teapot]):.3f} kg", flush=True)

    def step(self):
        if self.frame < END_FRAME:
            super().step()

    def before_substep(self):
        # Particles that have left the finite work area no longer contribute to
        # its sparse MPM grid. Water inside the teapot or bowl remains active.
        wp.launch(
            _retire_out_of_scene_water,
            dim=self.water_end - self.water_begin,
            inputs=[
                self.state_0.particle_q,
                self.state_0.particle_qd,
                self.model.particle_flags,
                self.mpm_particle_flags,
                self.mpm_material_flags,
                self.water_begin,
                int(newton.ParticleFlags.ACTIVE),
                TABLE_Z - 0.08,
            ],
        )

    def _retire_invalid_after_step(self):
        # A particle can become invalid on the last substep of a frame.
        # Clean it before rendering or checking the completed frame.
        self.before_substep()

    def after_step(self):
        self._retire_invalid_after_step()
        if self.frame % 30 == 0:
            self.bowl_particle_count = self.water_in_bowl_count()
            if self.args.test and self.peak_bowl_particle_count > 100:
                if self.bowl_particle_count < 0.5 * self.peak_bowl_particle_count:
                    raise AssertionError(
                        f"Water disappeared from the bowl at frame {self.frame}: "
                        f"{self.bowl_particle_count} remain after a peak of {self.peak_bowl_particle_count}"
                    )
            self.peak_bowl_particle_count = max(self.peak_bowl_particle_count, self.bowl_particle_count)
            flags = self.mpm_particle_flags.numpy()[self.water_begin : self.water_end]
            self.retired_particle_count = int(np.count_nonzero((flags & int(newton.ParticleFlags.ACTIVE)) == 0))
            if self.frame == 720:
                self.pour_end_particle_count = self.bowl_particle_count
        if not self.args.test or self.frame % 10:
            return
        pose = self.state_0.body_q.numpy()[self.teapot]
        self.peak_lift = max(self.peak_lift, float(pose[2] - TABLE_Z))
        up = np.asarray(wp.quat_rotate(wp.quat(*pose[3:]), wp.vec3(0.0, 0.0, 1.0)))
        self.peak_tilt = max(self.peak_tilt, float(np.arccos(np.clip(up[2], -1.0, 1.0))))

    def water_in_bowl_count(self):
        points = self.state_0.particle_q.numpy()[self.water_begin : self.water_end]
        in_bowl = (
            (((points[:, 0] - BOWL_CENTER[0]) / 0.075) ** 2 + ((points[:, 1] - BOWL_CENTER[1]) / 0.055) ** 2 < 1.0)
            & (points[:, 2] > TABLE_Z - 0.01)
            & (points[:, 2] < TABLE_Z + 0.055)
        )
        return int(in_bowl.sum())

    def water_in_bowl(self):
        return self.water_in_bowl_count() / (self.water_end - self.water_begin)

    def render_ui(self, imgui):
        super().render_ui(imgui)
        imgui.text(f"Water in bowl: {self.bowl_particle_count}/{self.water_end - self.water_begin} particles")
        imgui.text(f"Outside scene: {self.retired_particle_count} particles")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "/water",
            self.state_0.particle_q[self.water_begin : self.water_end],
            self.water_radii,
            self.water_colors,
        )
        self.viewer.end_frame()
        if self.args.report_fps:
            if self.frame == 60:
                self._fps_start = time.perf_counter()
            elif (
                self.frame == min(self.args.num_frames, END_FRAME)
                and self._fps_start is not None
                and not self._fps_reported
            ):
                elapsed = time.perf_counter() - self._fps_start
                print(f"[W1TeapotFill] viewer: {(self.frame - 60) / elapsed:.1f} FPS", flush=True)
                self._fps_reported = True
        if (
            self.args.debug_screenshots
            and self.frame in (1, 180, 300, 420, 540, 600, 660, 720, 780, 840, 900)
            and self.frame not in self._screenshots_saved
        ):
            from PIL import Image  # noqa: PLC0415

            path = Path(self.args.debug_screenshots)
            path.mkdir(parents=True, exist_ok=True)
            Image.fromarray(self.viewer.get_frame().numpy()).save(path / f"frame_{self.frame:04d}.jpg", quality=90)
            self._screenshots_saved.add(self.frame)

    def test_post_step(self):
        try:
            self.check_finite()
        except AssertionError as exc:
            arrays = {
                "body_q": self.state_0.body_q,
                "body_qd": self.state_0.body_qd,
                "joint_q": self.state_0.joint_q,
                "particle_q": self.state_0.particle_q,
                "particle_qd": self.state_0.particle_qd,
                "elastic_strain": self.state_0.mpm.particle_elastic_strain,
                "stress": self.state_0.mpm.particle_stress,
                "plastic_volume": self.state_0.mpm.particle_Jp,
            }
            failed = [
                name for name, array in arrays.items() if array is not None and not np.isfinite(array.numpy()).all()
            ]
            raise AssertionError(f"Nonfinite {', '.join(failed)} at t={self.sim_time:.3f}s") from exc

    def test_final(self):
        self.test_post_step()
        self.solver.check_capacity_status()
        if self.sim_time < 14.9:
            raise AssertionError("Run through grasp, pour, and three seconds of settling")
        body_q = self.state_0.body_q.numpy()
        pose = body_q[self.teapot]
        up = np.asarray(wp.quat_rotate(wp.quat(*pose[3:]), wp.vec3(0.0, 0.0, 1.0)))
        final_tilt = float(np.arccos(np.clip(up[2], -1.0, 1.0)))
        if pose[2] < TABLE_Z + 0.13 or not np.deg2rad(60.0) < final_tilt < np.deg2rad(100.0):
            raise AssertionError(
                f"Teapot lost its pouring pose: height {pose[2]:.3f}m, tilt {np.rad2deg(final_tilt):.0f}deg"
            )
        handle = np.asarray(wp.transform_point(wp.transform(*pose), wp.vec3(*HANDLE_CENTER)))
        grasp_error = float(np.linalg.norm(handle - self.tcp_positions(body_q)[1]))
        if grasp_error > 0.10:
            raise AssertionError(f"Teapot slipped from the gripper: handle-TCP distance {grasp_error:.3f}m")
        fraction = self.water_in_bowl()
        if fraction < 0.85:
            points = self.state_0.particle_q.numpy()[self.water_begin : self.water_end]
            low = points[(points[:, 2] > TABLE_Z - 0.01) & (points[:, 2] < TABLE_Z + 0.055)]
            center = np.median(low[:, :2], axis=0) if len(low) else np.array((np.nan, np.nan))
            raise AssertionError(f"Only {fraction:.0%} of water reached the bowl; low water center {center}")
        if self.bowl_particle_count < 0.9 * self.pour_end_particle_count:
            raise AssertionError("Water left the bowl after pouring stopped")
        if self.retired_particle_count > 0.05 * (self.water_end - self.water_begin):
            raise AssertionError("Too much water escaped the simulated work area")
        print(
            f"[W1TeapotFill] PASS: lifted {self.peak_lift:.3f}m, tilted {np.rad2deg(self.peak_tilt):.0f}deg, "
            f"{round(fraction * (self.water_end - self.water_begin))}/{self.water_end - self.water_begin} "
            f"water particles in bowl ({fraction:.0%}), {self.retired_particle_count} retired outside scene",
            flush=True,
        )

    @staticmethod
    def create_parser():
        parser = W1MJMPMScene.add_arguments(newton.examples.create_parser(), num_frames=END_FRAME)
        parser.add_argument("--debug-screenshots", type=str, default=None)
        parser.add_argument(
            "--report-fps", action="store_true", help="Report measured viewer-loop FPS after 60 warmup frames"
        )
        parser.set_defaults(
            substeps=4,
            rigid_iterations=60,
            finger_kp=6000.0,
            voxel_size=0.003,
            strain_basis="P0",
            mpm_solver="gs",
            iterations=15,
            stall_tolerance=0.0,
            rigid_material_coupling="constrained",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

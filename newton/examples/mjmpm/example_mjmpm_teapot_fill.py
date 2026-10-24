# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pour MPM water from the Utah Teapot into a bowl with MuJoCo-driven motion.

    uv run --extra examples -m newton.examples mjmpm_teapot_fill

The scene follows Isaac Lab 3.0's ``teapot-fill`` demo: a 70%-height water
fill, a 65-degree tilt, then a gentle lift above a shallow receiving bowl.
MuJoCo drives the two teapot joints and receives the MPM water reaction through
MJMPM coupling. The water is rendered directly from its MPM particles.
"""

import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMJMPM

ASSETS = Path(__file__).resolve().parent / "assets/teapot_fill"
TABLE_Z = 0.90041
BOWL_CENTER = np.array((0.106, 0.0, TABLE_Z), dtype=np.float32)
# Keep the spout above the receiver during the tilt. The reference's scripted
# kinematic teapot starts 5 cm farther left; this MuJoCo-driven version needs
# the shorter throw to retain the same receiving-bowl action.
TEAPOT_CENTER = np.array((-0.055, 0.0, TABLE_Z + 0.04357 + 0.115), dtype=np.float32)
HOLD_TIME = 0.55
TILT_TIME = 2.0
LIFT_TIME = 3.0
LIFT_HEIGHT = 0.24
POUR_ANGLE = np.deg2rad(65.0)
WATER_DENSITY = 1000.0


def _smoothstep(value):
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def teapot_targets(time):
    """Return lift [m] and tilt [rad] for the scripted pour."""
    tilt = POUR_ANGLE * _smoothstep((time - HOLD_TIME) / TILT_TIME)
    lift = LIFT_HEIGHT * np.clip((time - HOLD_TIME - TILT_TIME) / LIFT_TIME, 0.0, 1.0)
    return np.array((lift, tilt), dtype=np.float32)


def bowl_mesh(segments=64):
    """Build a closed, thin, open-topped catch bowl with a flat inner floor."""
    angles = np.arange(segments) * (2.0 * np.pi / segments)
    unit = np.stack((np.cos(angles), np.sin(angles)), axis=1)
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


def water_points(spacing):
    """Sample the known central air cavity, inset from the double-walled shell."""
    # The simplified teapot remains closed at the lid and spout. A conservative
    # radial envelope avoids seeding any particle in its thin ceramic walls.
    points = []
    for z in np.arange(0.011, 0.055, spacing):
        radius = 0.033 if z < 0.02 else 0.037 if z < 0.045 else 0.035
        for x in np.arange(-radius, radius + 0.5 * spacing, spacing):
            for y in np.arange(-radius, radius + 0.5 * spacing, spacing):
                if x * x + y * y < (radius - 0.5 * spacing) ** 2:
                    points.append((x, y, z))
    return np.asarray(points, dtype=np.float32)


class Example:
    def __init__(self, viewer, args):
        if args.substeps < 2 or args.substeps % 2:
            raise ValueError("substeps must be an even number >= 2 for CUDA graph capture")
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / args.fps
        self.sim_dt = self.frame_dt / args.substeps
        self.sim_time = 0.0
        self.frame = 0
        self.graph = None
        self._last_frame_end = None
        self._frame_times = []

        builder = newton.ModelBuilder()
        SolverMJMPM.register_custom_attributes(builder)
        builder.add_ground_plane(height=TABLE_Z - 0.018, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, TABLE_Z - 0.009), wp.quat_identity()),
            hx=0.31565,
            hy=0.62045,
            hz=0.009,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.5),
            color=(0.48, 0.38, 0.26),
            label="table",
        )
        bowl_vertices, bowl_faces = bowl_mesh()
        builder.add_shape_mesh(
            -1,
            xform=wp.transform(wp.vec3(*BOWL_CENTER), wp.quat_identity()),
            mesh=newton.Mesh(bowl_vertices, bowl_faces.ravel(), compute_inertia=False, is_solid=False),
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.2),
            color=(0.93, 0.94, 0.91),
            label="catch_bowl",
        )

        base = wp.transform(wp.vec3(*TEAPOT_CENTER), wp.quat_identity())
        carriage = builder.add_link(
            xform=base,
            mass=0.5,
            inertia=wp.mat33(np.eye(3) * 0.001),
            custom_attributes={"mujoco:gravcomp": 1.0},
            label="teapot_lift",
        )
        lift = builder.add_joint_prismatic(
            parent=-1,
            child=carriage,
            parent_xform=base,
            axis=(0.0, 0.0, 1.0),
            target_ke=2000.0,
            target_kd=100.0,
            effort_limit=100.0,
        )
        pot = builder.add_link(
            xform=base,
            mass=0.35,
            inertia=wp.mat33(np.eye(3) * 0.001),
            custom_attributes={"mujoco:gravcomp": 1.0},
            label="Utah_Teapot",
        )
        with np.load(ASSETS / "utah_teapot_collision.npz") as asset:
            vertices = asset["vertices"]
            faces = asset["faces"]
        builder.add_shape_mesh(
            pot,
            mesh=newton.Mesh(vertices, faces.ravel(), compute_inertia=False, is_solid=False),
            cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=2.0e-5),
            color=(0.70, 0.35, 0.16),
            label="Modified_Utah_Teapot",
        )
        hinge = builder.add_joint_revolute(
            parent=carriage,
            child=pot,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform_identity(),
            child_xform=wp.transform_identity(),
            target_ke=80.0,
            target_kd=5.0,
            effort_limit=30.0,
        )
        builder.add_articulation([lift, hinge])

        self.water_begin = builder.particle_count
        points = water_points(args.fill_spacing)
        for point in points:
            builder.add_particle(
                wp.vec3(*(TEAPOT_CENTER + point)),
                wp.vec3(0.0),
                WATER_DENSITY * args.fill_spacing**3,
                0.5 * args.fill_spacing,
                custom_attributes={"mpm:friction": 0.0, "mpm:tensile_yield_ratio": 1.0, "mpm:viscosity": 1.0e-3},
            )
        self.water_end = builder.particle_count
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.solver = SolverMJMPM(
            self.model,
            coupling=SolverMJMPM.Coupling.TWO_WAY,
            articulated_response=True,
            mpm_options={
                "grid_type": "sparse",
                "voxel_size": args.voxel_size,
                "grid_padding": 0,
                "max_active_cell_count": args.max_active_cells,
                "max_iterations": args.iterations,
                "solver": args.mpm_solver,
                "tolerance": 1.0e-4,
                "strain_basis": "P0",
                "collider_basis": args.collider_basis,
                "transfer_scheme": "apic",
                "integration_scheme": "pic",
                "air_drag": 1.0e-3,
            },
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)

        self.water_radii = wp.full(len(points), 0.003, dtype=float, device=self.model.device)
        self.water_colors = wp.full(len(points), wp.vec3(0.12, 0.42, 0.9), dtype=wp.vec3, device=self.model.device)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -0.72, 1.18), pitch=-14.0, yaw=90.0)
        self.viewer.show_particles = False
        print(
            f"[TeapotFill] {len(points)} MPM water particles, {len(points) * args.fill_spacing**3 * 1000:.3f} L",
            flush=True,
        )

    def _simulate(self):
        for _ in range(self.args.substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.control.joint_target_q.assign(teapot_targets(self.sim_time + self.frame_dt))
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self._simulate()
            if self.model.device.is_cuda and not self.args.no_cuda_graph:
                saved = (self.model.state(), self.model.state())
                saved[0].assign(self.state_0)
                saved[1].assign(self.state_1)
                with wp.ScopedCapture() as capture:
                    self._simulate()
                self.state_0.assign(saved[0])
                self.state_1.assign(saved[1])
                self.graph = capture.graph
        self.frame += 1
        self.sim_time = self.frame * self.frame_dt
        if self.args.test and self.frame % 25 == 0:
            positions = self.state_0.particle_q.numpy()[self.water_begin : self.water_end]
            q = self.state_0.joint_q.numpy()
            bowl = (np.linalg.norm(positions[:, :2] - BOWL_CENTER[:2], axis=1) < 0.055) & (
                positions[:, 2] < TABLE_Z + 0.055
            )
            print(
                f"[TeapotFill] t={self.sim_time:.2f}, joints={q[:2]}, bowl={bowl.mean():.1%}, "
                f"bounds={positions.min(axis=0)}..{positions.max(axis=0)}",
                flush=True,
            )

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
        if self.args.benchmark is not False:
            now = time.perf_counter()
            if self._last_frame_end is not None and self.frame > 3:
                self._frame_times.append((self.sim_time, now - self._last_frame_end))
            self._last_frame_end = now
            if self.frame == self.args.num_frames:
                for name, start, end in (("tilt", 0.55, 2.55), ("pour", 2.55, 3.55), ("lift", 3.55, 5.55)):
                    times = [duration for t, duration in self._frame_times if start <= t < end]
                    if times:
                        print(f"[TeapotFill] {name}: {len(times) / sum(times):.1f} FPS", flush=True)
        if self.args.screenshot_path and self.frame == self.args.screenshot_frame:
            from PIL import Image  # noqa: PLC0415

            image = Image.fromarray(self.viewer.get_frame().numpy())
            side = min(image.size)
            left = (image.width - side) // 2
            top = (image.height - side) // 2
            image = image.crop((left, top, left + side, top + side))
            path = Path(self.args.screenshot_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            image.resize((320, 320), Image.Resampling.LANCZOS).save(path, quality=92)

    def test_post_step(self):
        for array in (self.state_0.joint_q, self.state_0.particle_q, self.state_0.particle_qd):
            if not np.isfinite(array.numpy()).all():
                raise AssertionError("Nonfinite teapot or water state")
        positions = self.state_0.particle_q.numpy()[self.water_begin : self.water_end]
        if np.min(positions[:, 2]) < TABLE_Z - 0.03:
            raise AssertionError("Water passed through the table or bowl")

    def test_final(self):
        self.test_post_step()
        self.solver.check_capacity_status()
        if self.sim_time < HOLD_TIME + TILT_TIME + 0.5:
            raise AssertionError("Run long enough to cover the tilt and pouring interval")
        positions = self.state_0.particle_q.numpy()[self.water_begin : self.water_end]
        in_bowl = (
            (np.linalg.norm(positions[:, :2] - BOWL_CENTER[:2], axis=1) < 0.055)
            & (positions[:, 2] < TABLE_Z + 0.055)
            & (positions[:, 2] > TABLE_Z - 0.01)
        )
        fraction = float(np.mean(in_bowl))
        if fraction < 0.4:
            raise AssertionError(f"Only {fraction:.0%} of the water reached the bowl")
        print(f"[TeapotFill] PASS: {fraction:.0%} of water in the bowl", flush=True)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(fps=50.0, num_frames=300)
        parser.add_argument("--substeps", type=int, default=6)
        parser.add_argument("--voxel-size", type=float, default=0.003)
        parser.add_argument("--fill-spacing", type=float, default=0.004)
        parser.add_argument("--iterations", type=int, default=30)
        parser.add_argument("--mpm-solver", choices=("auto", "gs", "gs-soa", "jacobi"), default="auto")
        parser.add_argument("--collider-basis", choices=("S2", "pic128", "pic512"), default="pic512")
        parser.add_argument("--max-active-cells", type=int, default=65536)
        parser.add_argument("--screenshot-path", type=str, default=None)
        parser.add_argument("--screenshot-frame", type=int, default=150)
        parser.add_argument("--no-cuda-graph", action="store_true")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

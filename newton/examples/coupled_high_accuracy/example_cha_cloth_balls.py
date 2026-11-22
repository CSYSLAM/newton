# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A single-layer plastic bag with two handle holes carrying six balls, with two more on the floor.

Run from the repository root:
    uv run python -m newton.examples cha_cloth_balls
"""

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import coupled_high_accuracy as cha
from newton.solvers import style3d

BAG_CELLS = (20, 10, 25)
CELL_SIZE = 0.02
BAG_BOTTOM = 0.30
HANDLE_CELLS = (8, 3)
HANDLE_TOP_CELLS = 2
PARTICLE_RADIUS = 0.002
BALL_RADIUS = 0.07
BALL_MASS = 0.08
# Dimensionless rolling resistance; the contact torque limit is mu_r * normal_load * contact_radius.
GROUND_ROLLING_FRICTION = 0.03
BALL_COLORS = (
    (0.90, 0.24, 0.12),
    (0.12, 0.54, 0.80),
    (0.94, 0.66, 0.10),
    (0.25, 0.68, 0.37),
    (0.70, 0.28, 0.57),
    (0.18, 0.72, 0.70),
)


def create_bag_mesh():
    """Return welded 3D vertices, faces, independent 2D panels, panel faces, and four rim pins."""
    nx, ny, nz = BAG_CELLS
    # Panel axes give consistent outward normals, including a downward-facing bottom.
    panels = (
        ((0, ny, 0), (1, 0, 0), (0, -1, 0), nx, ny),
        ((0, 0, 0), (1, 0, 0), (0, 0, 1), nx, nz),
        ((nx, 0, 0), (0, 1, 0), (0, 0, 1), ny, nz),
        ((nx, ny, 0), (-1, 0, 0), (0, 0, 1), nx, nz),
        ((0, ny, 0), (0, -1, 0), (0, 0, 1), ny, nz),
    )
    vertices, faces, panel_vertices, panel_faces = [], [], [], []
    vertex_ids = {}
    offset = np.array((-nx * CELL_SIZE * 0.5, -ny * CELL_SIZE * 0.5, BAG_BOTTOM))
    for panel, (origin, axis_u, axis_v, nu, nv) in enumerate(panels):
        grid = np.empty((nv + 1, nu + 1), dtype=np.int32)
        panel_start = len(panel_vertices)
        for v in range(nv + 1):
            for u in range(nu + 1):
                key = tuple(origin[k] + u * axis_u[k] + v * axis_v[k] for k in range(3))
                # Weld physical seams, but keep separate metric rest coordinates for each flat panel.
                if key not in vertex_ids:
                    vertex_ids[key] = len(vertices)
                    vertices.append(np.asarray(key) * CELL_SIZE + offset)
                grid[v, u] = vertex_ids[key]
                panel_vertices.append((u * CELL_SIZE, v * CELL_SIZE))
        for v in range(nv):
            for u in range(nu):
                a = v * (nu + 1) + u
                b, c, d = a + 1, a + nu + 1, a + nu + 2
                # Alternate diagonals to avoid a preferred fold direction across the whole bag.
                triangles = ((a, b, d), (a, d, c)) if (u + v) % 2 == 0 else ((a, b, c), (b, d, c))
                if panel in (1, 3):
                    left = (nu - HANDLE_CELLS[0]) // 2
                    right = left + HANDLE_CELLS[0]
                    top = nv - HANDLE_TOP_CELLS
                    bottom = top - HANDLE_CELLS[1]
                    if left <= u < right and bottom <= v < top:
                        # Leave one outer triangle at each corner to chamfer the grip opening.
                        # The two intact rows above the hole connect the handle to both side seams.
                        corners = {
                            (left, bottom): ((a, b, c),),
                            (right - 1, bottom): ((a, b, d),),
                            (right - 1, top - 1): ((b, d, c),),
                            (left, top - 1): ((a, d, c),),
                        }
                        triangles = corners.get((u, v), ())
                for triangle in triangles:
                    faces.append([int(grid.flat[i]) for i in triangle])
                    panel_faces.append([panel_start + i for i in triangle])
    pins = [vertex_ids[(x, y, nz)] for x in (0, nx) for y in (0, ny)]
    # Cutouts must not leave isolated, massless particles or unused rest-panel vertices in the model.
    used_vertices, compact_faces = np.unique(np.asarray(faces, dtype=np.int32), return_inverse=True)
    used_panel_vertices, compact_panel_faces = np.unique(np.asarray(panel_faces, dtype=np.int32), return_inverse=True)
    return (
        np.asarray(vertices, dtype=np.float32)[used_vertices],
        compact_faces.reshape(-1, 3).astype(np.int32),
        np.asarray(panel_vertices, dtype=np.float32)[used_panel_vertices],
        compact_panel_faces.reshape(-1, 3).astype(np.int32),
        np.searchsorted(used_vertices, pins).astype(np.int32),
    )


def ball_positions():
    """Place two separated layers inside the bag and two balls directly on the ground."""
    lower_z = BAG_BOTTOM + BALL_RADIUS + 0.025
    inside = [(x, y, lower_z) for y in (-0.085, 0.085) for x in (-0.085, 0.085)]
    inside.extend((x, 0.0, lower_z + 0.155) for x in (-0.085, 0.085))
    ground = [(-0.46, -0.31, BALL_RADIUS), (0.45, -0.26, BALL_RADIUS)]
    return np.asarray(inside, dtype=np.float32), np.asarray(ground, dtype=np.float32)


class Example:
    def __init__(self, viewer, args):
        if args.substeps < 2 or args.substeps % 2:
            raise ValueError("--substeps must be a positive even number for ping-pong CUDA graph replay")
        if args.iterations < 1 or args.linear_iterations < 1:
            raise ValueError("Solver iteration counts must be positive")
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.graph = None

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        points, indices, panel_points, panel_indices, self.fixed_points = create_bag_mesh()
        self.bag_particle_count = len(points)
        self.initial_fixed_positions = points[self.fixed_points].copy()
        style3d.add_cloth_mesh(
            builder,
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            vertices=points.tolist(),
            indices=indices.ravel().tolist(),
            panel_verts=panel_points.tolist(),
            panel_indices=panel_indices.ravel().tolist(),
            density=0.25,
            particle_radius=PARTICLE_RADIUS,
            tri_aniso_ke=wp.vec3(1.0e3, 1.0e3, 1.0e2),
            tri_ka=100.0,
            tri_kd=1.0e-4,
            edge_aniso_ke=wp.vec3(2.0e-5),
            edge_kd=1.0e-5,
            validate_mesh=True,
            label="plastic_bag",
        )
        for particle in self.fixed_points:
            builder.particle_flags[particle] &= ~int(newton.ParticleFlags.ACTIVE)
            builder.particle_mass[particle] = 0.0

        builder.default_shape_cfg.ke = 2.0e3
        builder.default_shape_cfg.kd = 1.0
        builder.default_shape_cfg.mu = 0.3
        ball_cfg = builder.default_shape_cfg.copy()
        ball_cfg.density = BALL_MASS / (4.0 * np.pi * BALL_RADIUS**3 / 3.0)
        inside, ground = ball_positions()
        self.initial_ground_positions = ground.copy()
        self.bag_bodies = []
        self.ground_bodies = []
        for group, positions, bodies in (("bag", inside, self.bag_bodies), ("ground", ground, self.ground_bodies)):
            for i, position in enumerate(positions):
                label = f"{group}_ball_{i}"
                body = builder.add_body(xform=wp.transform(wp.vec3(*position), wp.quat_identity()), label=label)
                builder.add_shape_sphere(
                    body,
                    radius=BALL_RADIUS,
                    cfg=ball_cfg,
                    color=BALL_COLORS[i % len(BALL_COLORS)],
                    label=label,
                )
                bodies.append(body)
        ground_cfg = builder.default_shape_cfg.copy()
        ground_cfg.mu_rolling = GROUND_ROLLING_FRICTION
        builder.add_ground_plane(cfg=ground_cfg)
        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.model.soft_contact_ke = 2.0e3
        self.model.soft_contact_kd = 1.0e-2
        self.model.soft_contact_mu = 0.2

        self.solver = cha.SolverCoupledHighAccuracy(self.model)
        self.solver.config.affine_body_rigidity_model = cha.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
        self.solver.config.nonlinear_solver = cha.NonlinearSolver.NEWTON
        self.solver.config.inertia_mode = cha.InertiaMode.MIDPOINT_CAYLEY
        self.solver.config.nonlinear_iterations = args.iterations
        self.solver.config.linear_iterations = args.linear_iterations
        # Keep cloth and rigid contact blocks coupled when the bag supports the stacked load.
        self.solver.config.enable_collision_hessian_non_diagonal = True
        self.solver.config.self_collision_radius = PARTICLE_RADIUS
        self.solver.config.enable_self_collision = False

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        # Face and edge contacts close the gaps between particles in this single-layer shell.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_gap=0.006,
            enable_rigid_soft_full_surface_contact=True,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.show_triangles = True
        self.viewer.log_mesh(
            "/model/triangles",
            self.state_0.particle_q,
            self.model.tri_indices.flatten(),
            color=(0.75, 0.85, 0.92),
            roughness=0.3,
            backface_culling=False,
        )
        self.viewer.set_camera(pos=wp.vec3(0.32, -0.42, 2.5), pitch=-75.0, yaw=126.0)
        self.capture()

    def capture(self):
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        for name in ("particle_q", "particle_qd", "body_q", "body_qd"):
            if not np.isfinite(getattr(self.state_0, name).numpy()).all():
                raise AssertionError(f"Bag-and-balls {name} became non-finite")
        cloth = self.state_0.particle_q.numpy()[: self.bag_particle_count]
        np.testing.assert_allclose(cloth[self.fixed_points], self.initial_fixed_positions, rtol=0.0, atol=1.0e-6)
        centers = self.state_0.body_q.numpy()[:, :3]
        inside = centers[self.bag_bodies]
        half_widths = np.asarray(BAG_CELLS[:2]) * CELL_SIZE * 0.5
        if np.any(np.abs(inside[:, :2]) > half_widths + 0.02) or np.any(inside[:, 2] < cloth[:, 2].min()):
            raise AssertionError("A carried ball escaped the bag")
        if cloth[:, 2].min() < 0.05 or np.any(centers[:, 2] < BALL_RADIUS - 0.005):
            raise AssertionError("The bag or a ball fell through the floor")
        if self.sim_time >= 0.5:
            np.testing.assert_allclose(centers[self.ground_bodies, 2], BALL_RADIUS, rtol=0.0, atol=0.005)
            np.testing.assert_allclose(
                centers[self.ground_bodies, :2], self.initial_ground_positions[:, :2], rtol=0.0, atol=0.005
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--substeps", type=int, default=2, help="Even number of physics substeps per frame.")
        parser.add_argument("--iterations", type=int, default=5)
        parser.add_argument("--linear-iterations", type=int, default=50)
        return parser


if __name__ == "__main__":
    wp.config.kernel_cache_dir = str(Path(__file__).resolve().parent / "build" / "warp-cloth-balls")
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

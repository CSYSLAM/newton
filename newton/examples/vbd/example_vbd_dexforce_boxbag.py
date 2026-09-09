# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""DexForce W1 picks a soft cube, then a rigid cube, into a cloth bag using VBD.

The final00 recording from FAST_MJVBDV2 supplies joint position targets only.
All robot links, objects, and cloth are integrated by SolverVBD. The open bag
is suspended by its rim. Grasping uses the original URDF collision meshes and
stage-dependent contact materials; there are no object attachments.

Command: python -m newton.examples vbd_dexforce_boxbag
"""

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

# Durations and contact phases of the original final00 trajectory, in seconds.
SEGMENTS = (
    (0.5, "soft_wait"),
    (2.0, "soft_prepare"),
    (1.8, "soft_grasp"),
    (9.2, "soft_carry"),
    (2.15, "soft_release"),
    (1.5, "rigid_move"),
    (2.5, "rigid_prepare"),
    (0.45, "rigid_grasp"),
    (7.75, "rigid_carry"),
    (3.3, "rigid_release"),
)
BASE_POS = wp.vec3(-0.34931439, -3.24669516, -0.00377202)
BASE_ROT = wp.quat(0.0, 0.0, 0.70710677, 0.70710677)
CUBE_HALF = (0.027, 0.012, 0.027)
BAG_POS = wp.vec3(0.448, -0.59, 0.935)
BAG_SIZE = (0.20, 0.16, 0.24)


def _world(position):
    return BASE_POS + wp.quat_rotate(BASE_ROT, wp.vec3(*position))


def _generate_box_bag(half_x: float, half_y: float, height: float, resolution: int):
    """Generate a merged five-face box mesh with an open top."""
    cell_x = 2.0 * half_x / resolution
    cell_y = 2.0 * half_y / resolution
    cell_z = height / resolution
    vertex_map = {}
    vertices = []
    indices = []

    def vertex(x, y, z):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append((x, y, z))
        return vertex_map[key]

    def quad(v00, v10, v01, v11):
        indices.extend((v00, v10, v01, v10, v11, v01))

    for i in range(resolution):
        for j in range(resolution):
            x0 = -half_x + i * cell_x
            x1 = x0 + cell_x
            y0 = -half_y + j * cell_y
            y1 = y0 + cell_y
            quad(vertex(x0, y0, 0.0), vertex(x1, y0, 0.0), vertex(x0, y1, 0.0), vertex(x1, y1, 0.0))

    for i in range(resolution):
        for j in range(resolution):
            x0 = -half_x + i * cell_x
            x1 = x0 + cell_x
            y0 = -half_y + i * cell_y
            y1 = y0 + cell_y
            z0 = j * cell_z
            z1 = z0 + cell_z
            quad(vertex(x0, -half_y, z0), vertex(x1, -half_y, z0), vertex(x0, -half_y, z1), vertex(x1, -half_y, z1))
            quad(vertex(x1, half_y, z0), vertex(x0, half_y, z0), vertex(x1, half_y, z1), vertex(x0, half_y, z1))
            quad(vertex(-half_x, y1, z0), vertex(-half_x, y0, z0), vertex(-half_x, y1, z1), vertex(-half_x, y0, z1))
            quad(vertex(half_x, y0, z0), vertex(half_x, y1, z0), vertex(half_x, y0, z1), vertex(half_x, y1, z1))

    return np.asarray(vertices, dtype=np.float32), indices


@wp.kernel
def _drive_targets(
    trajectory: wp.array2d[float],
    frame: wp.array[int],
    substep_alpha: float,
    speed: float,
    target_q: wp.array[float],
    target_qd: wp.array[float],
):
    i = wp.tid()
    f = wp.min(frame[0], trajectory.shape[0] - 1)
    next_f = wp.min(f + 1, trajectory.shape[0] - 1)
    q0 = trajectory[f, i]
    q1 = trajectory[next_f, i]
    target_q[i] = q0 + substep_alpha * (q1 - q0)
    target_qd[i] = (q1 - q0) * 60.0 * speed


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame = 0
        self.sim_time = 0.0
        self.sim_substeps = args.substeps
        if self.sim_substeps < 2 or self.sim_substeps % 2:
            raise ValueError("--substeps must be a positive even number")
        if args.speed <= 0:
            raise ValueError("--speed must be positive")
        self.frame_dt = 1.0 / (60.0 * args.speed)
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.speed = args.speed
        self.bag_opacity = args.bag_opacity
        if not 0.0 <= self.bag_opacity <= 1.0:
            raise ValueError("--bag-opacity must be between 0 and 1")
        if args.bag_stretch_stiffness <= 0 or args.bag_bend_stiffness < 0:
            raise ValueError("Bag stretch stiffness must be positive and bending stiffness nonnegative")
        asset_root = Path(newton.examples.get_asset_directory()) / "dexforce_boxbag"
        with np.load(asset_root / "robot_targets.npz") as recording:
            trajectory = recording["joint_q"]
            joint_names = recording["joint_names"].tolist()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(ke=2.0e5, kd=1.0e-4, mu=1.0)
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        builder.add_urdf(
            str(asset_root / "DexforceW1V021/DexforceW1V021.urdf"),
            xform=wp.transform(BASE_POS, BASE_ROT),
            floating=False,
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
        )
        imported_names = [
            label.rsplit("/", 1)[-1]
            for joint, label in enumerate(builder.joint_label)
            if builder.joint_type[joint] != newton.JointType.FIXED
        ]
        if imported_names != joint_names:
            raise ValueError("DexForce URDF joint order does not match the recorded targets")
        self.robot_dofs = builder.joint_dof_count
        self.robot_body_count = builder.body_count
        # Recorded PIP joints were edited independently. Preserve those targets,
        # rather than imposing the URDF's default mimic relation on the recording.
        builder.joint_mimic_joint[:] = [-1] * builder.joint_count
        builder.joint_friction[:] = [0.0] * builder.joint_dof_count
        builder.joint_q[:] = trajectory[0].tolist()
        builder.joint_target_q[:] = builder.joint_q
        builder.joint_target_ke[:] = [100.0 if "HAND" in n or "PIP" in n else 1.0e5 for n in joint_names]
        builder.joint_target_kd[:] = [1.0 if "HAND" in n or "PIP" in n else 1000.0 for n in joint_names]
        self.hand_shapes = []
        collision_mask = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape, body in enumerate(builder.shape_body):
            label = builder.body_label[body].lower() if body >= 0 else ""
            if (
                builder.shape_flags[shape] & collision_mask
                and "right" in label
                and any(part in label for part in ("thumb", "index", "middle", "ring", "pinky", "hand"))
            ):
                self.hand_shapes.append(shape)
            else:
                builder.shape_flags[shape] &= ~collision_mask
        builder.add_shape_box(
            -1,
            xform=wp.transform(_world((0.55, 0.0, 1.15)), BASE_ROT),
            hx=0.32,
            hy=0.45,
            hz=0.025,
            cfg=newton.ModelBuilder.ShapeConfig(ke=3.0e5, kd=1.0e-4, mu=0.9),
            color=(0.35, 0.42, 0.5),
            label="pick_table",
        )
        builder.add_ground_plane(height=float(BASE_POS[2]))
        vertices, indices = _generate_box_bag(BAG_SIZE[0] / 2, BAG_SIZE[1] / 2, BAG_SIZE[2], 20)
        builder.add_cloth_mesh(
            pos=_world(BAG_POS),
            rot=BASE_ROT,
            scale=1.0,
            vel=wp.vec3(),
            vertices=vertices.tolist(),
            indices=indices,
            density=0.08,
            # Low bending resistance lets the walls fold instead of retaining
            # the initial box shape. Moderate membrane stiffness carries the cubes.
            tri_ke=args.bag_stretch_stiffness,
            tri_ka=args.bag_stretch_stiffness,
            tri_kd=1.0,
            edge_ke=args.bag_bend_stiffness,
            edge_kd=0.001,
            particle_radius=0.003,
            label="box_bag",
        )
        self.bag_count = builder.particle_count
        self.rim_indices = np.flatnonzero(np.abs(vertices[:, 2] - 0.24) < 1.0e-5)
        for i in self.rim_indices:
            builder.particle_mass[i] = 0.0
        cube_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2)
        soft_origin = wp.vec3(0.48, -0.20, 1.203) - wp.quat_rotate(cube_rot, wp.vec3(*CUBE_HALF))
        builder.add_soft_grid(
            pos=_world(soft_origin),
            rot=BASE_ROT * cube_rot,
            vel=wp.vec3(),
            dim_x=6,
            dim_y=4,
            dim_z=6,
            cell_x=0.054 / 6,
            cell_y=0.024 / 4,
            cell_z=0.054 / 6,
            density=300.0,
            k_mu=3.0e5,
            k_lambda=1.0e6,
            k_damp=15.0,
            particle_radius=0.0025,
            label="soft_cube",
        )
        self.cube_body = builder.add_body(
            xform=wp.transform(_world((0.48, -0.09, 1.203)), BASE_ROT), label="rigid_cube"
        )
        self.cube_shape = builder.add_shape_box(
            self.cube_body,
            hx=CUBE_HALF[0],
            hy=CUBE_HALF[1],
            hz=CUBE_HALF[2],
            xform=wp.transform(wp.vec3(), cube_rot),
            cfg=newton.ModelBuilder.ShapeConfig(density=1500.0, ke=3000.0, kd=1.0, mu=3000.0, margin=0.0015),
            color=(1.0, 0.4, 0.08),
            label="rigid_cube",
        )
        builder.color(include_bending=True)
        self.model = builder.finalize()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_max=8192,
            rigid_contact_max=4096,
            soft_contact_gap=0.01,
            contact_matching="latest",
        )
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=args.iterations,
            rigid_compliant_alm=True,
            rigid_joint_linear_ke=1.0e8,
            rigid_joint_angular_ke=1.0e8,
            rigid_articulation_solve=args.rigid_articulation_solve,
            rigid_contact_history=True,
            rigid_body_contact_buffer_size=512,
            rigid_body_particle_contact_buffer_size=4096,
            particle_enable_self_contact=True,
            particle_self_contact_margin=0.003,
            particle_self_contact_gap=0.003,
            particle_topological_contact_filter_threshold=3,
            particle_rest_shape_contact_exclusion_radius=0.03,
            collision_pipeline=self.pipeline,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.trajectory = wp.array(trajectory, dtype=float, device=self.model.device)
        self.frame_index = wp.zeros(1, dtype=int, device=self.model.device)
        self.phase_ends = np.cumsum([duration for duration, _ in SEGMENTS])
        self.phase = None
        self.graphs = {}
        self.use_graph = self.model.device.is_cuda and not args.disable_graph
        self.initial_particles = self.model.particle_q.numpy()
        self.max_heights = np.array(
            [self.initial_particles[self.bag_count :, 2].mean(), self.model.body_q.numpy()[self.cube_body, 2]]
        )
        self.initial_heights = self.max_heights.copy()
        self.checked_transport = np.zeros(2, dtype=bool)
        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        self.viewer.set_camera(pos=wp.vec3(1.85, -4.7, 2.15), pitch=-24.0, yaw=137.0)
        triangles = self.model.tri_indices.numpy()
        self.bag_triangles = wp.array(triangles[np.all(triangles < self.bag_count, axis=1)].flatten(), dtype=int)
        self.cube_triangles = wp.array(triangles[np.all(triangles >= self.bag_count, axis=1)].flatten(), dtype=int)
        self._set_phase("soft_wait")

    def _set_phase(self, phase):
        """Reproduce final00's contact schedule; graph keys include scalar materials."""
        self.phase = phase
        flags = self.model.shape_flags.numpy()
        particle_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        rigid_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
        flags[self.hand_shapes] &= ~(particle_flag | rigid_flag)
        if phase != "rigid_move":
            flags[self.hand_shapes] |= rigid_flag
        if phase in {"soft_prepare", "soft_grasp", "soft_carry", "soft_release"}:
            flags[self.hand_shapes] |= particle_flag
        self.model.shape_flags.assign(flags)
        soft_grasp = phase in {"soft_prepare", "soft_grasp", "soft_carry"}
        rigid_grasp = phase in {"rigid_prepare", "rigid_grasp", "rigid_carry"}
        soft_material = (15000.0, 0.2, 6.0) if soft_grasp else (5000.0, 0.05, 0.25)
        if phase == "rigid_release":
            soft_material = (5000.0, 0.0, 0.0)
        self.model.soft_contact_ke, self.model.soft_contact_kd, self.model.soft_contact_mu = soft_material
        hand_material = (15000.0, 0.2, 40.0) if soft_grasp else (5000.0, 0.0, 0.0)
        shapes = self.hand_shapes
        if rigid_grasp or phase == "rigid_release":
            shapes = [*shapes, self.cube_shape]
            hand_material = (3000.0, 1.0, 3000.0) if rigid_grasp else (5000.0, 0.0, 0.0)
            margins = self.model.shape_margin.numpy()
            margins[shapes] = 0.0015
            self.model.shape_margin.assign(margins)
        for array, value in zip(
            (self.model.shape_material_ke, self.model.shape_material_kd, self.model.shape_material_mu),
            hand_material,
            strict=True,
        ):
            values = array.numpy()
            values[shapes] = value
            array.assign(values)
        print(f"Frame {self.frame}: {phase}")

    def simulate(self):
        for substep in range(self.sim_substeps):
            wp.launch(
                _drive_targets,
                dim=self.robot_dofs,
                inputs=[self.trajectory, self.frame_index, (substep + 1) / self.sim_substeps, self.speed],
                outputs=[self.control.joint_target_q, self.control.joint_target_qd],
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)

    def step(self):
        phase_index = min(int(np.searchsorted(self.phase_ends, self.frame / 60.0, side="right")), len(SEGMENTS) - 1)
        phase = SEGMENTS[phase_index][1]
        if phase != self.phase:
            self._set_phase(phase)
        self.frame_index.fill_(self.frame)
        if self.use_graph:
            key = (self.model.soft_contact_ke, self.model.soft_contact_kd, self.model.soft_contact_mu)
            if key not in self.graphs:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graphs[key] = capture.graph
            wp.capture_launch(self.graphs[key])
        else:
            self.simulate()
        self.frame += 1
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/box_bag",
            self.state_0.particle_q,
            self.bag_triangles,
            backface_culling=False,
            color=(0.85, 0.88, 0.93),
            opacity=self.bag_opacity,
        )
        self.viewer.log_mesh(
            "/soft_cube", self.state_0.particle_q, self.cube_triangles, backface_culling=False, color=(0.2, 0.8, 0.2)
        )
        self.viewer.end_frame()

    def test_post_step(self):
        """Check finite dynamics, pinned rim, and sustained transport of each cube."""
        particles = self.state_0.particle_q.numpy()
        bodies = self.state_0.body_q.numpy()
        assert np.isfinite(particles).all() and np.isfinite(bodies).all(), "Non-finite simulation state"
        np.testing.assert_allclose(particles[self.rim_indices], self.initial_particles[self.rim_indices], atol=1.0e-6)
        heights = [particles[self.bag_count :, 2].mean(), bodies[self.cube_body, 2]]
        self.max_heights = np.maximum(self.max_heights, heights)
        for index, window in enumerate(((450, 720), (1350, 1560))):
            if window[0] <= self.frame <= window[1]:
                assert heights[index] > self.initial_heights[index] + 0.03, (
                    f"Cube {index} slipped during transport at frame {self.frame}: height={heights[index]}"
                )
                self.checked_transport[index] = True

    def test_final(self):
        """Require both dynamic cubes to finish inside the suspended bag."""
        assert self.frame >= 1900, "Run at least 1900 frames to check both placements"
        assert np.all(self.checked_transport), "Both sustained transport phases must be checked"
        assert np.all(self.max_heights > self.initial_heights + 0.045), "Both cubes must be physically lifted"
        particles = self.state_0.particle_q.numpy()
        centres = [particles[self.bag_count :].mean(axis=0), self.state_0.body_q.numpy()[self.cube_body, :3]]
        for name, centre in zip(("soft", "rigid"), centres, strict=True):
            local = wp.quat_rotate(wp.quat_inverse(BASE_ROT), wp.vec3(*centre) - _world(BAG_POS))
            assert abs(local[0]) < 0.10 and abs(local[1]) < 0.08, f"{name} cube missed the bag: {local}"
            assert -0.12 < local[2] < 0.20, f"{name} cube was not retained below the rim: {local}"
        assert not np.any(self.model.body_flags.numpy() & int(newton.BodyFlags.KINEMATIC))

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1900)
        parser.add_argument("--substeps", type=int, default=12)
        parser.add_argument("--iterations", type=int, default=24)
        parser.add_argument(
            "--speed", type=float, default=1.0, help="Trajectory speed; below 1 gives more settling time"
        )
        parser.add_argument("--disable-graph", action="store_true")
        parser.add_argument(
            "--bag-opacity", type=float, default=0.55, help="Bag display opacity; 1 shows an opaque bag"
        )
        parser.add_argument("--bag-stretch-stiffness", type=float, default=1000.0, help="Bag membrane stiffness")
        parser.add_argument("--bag-bend-stiffness", type=float, default=0.02, help="Bag bending stiffness")
        parser.add_argument(
            "--rigid-articulation-solve", choices=("local", "block_sparse_joints"), default="block_sparse_joints"
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)

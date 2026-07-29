# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Dexforce W1 physically places rigid objects in a VBD cloth bag through ADMM.

The robot articulation and its position-controlled hand belong to MuJoCo. The
five objects and the suspended open bag belong to VBD. ``SolverCoupledADMM``
resolves every robot--VBD contact, while object--bag and object--table contact
remain inside the VBD entry. Unlike the single-solver scripted predecessor,
this example never overwrites a held object's pose: the hand closes around
each object and carries it through contact before opening over the bag.

Run from the repository root::

    uv run --extra examples -m newton.examples dexforce_rigid_into_bag_admm
"""

from __future__ import annotations

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledADMM

import newton
import newton.examples
from newton.examples.vbd import example_vbd_dexforce_throw_rigid_into_bag as bag
from newton.solvers import SolverMuJoCo, SolverVBD

ADMM_ITERATIONS = 8
MUJOCO_ITERATIONS = 16
MUJOCO_LINE_SEARCH_ITERATIONS = 32
VBD_ITERATIONS = 20
HAND_CONTACT_KE = 3.0e5
HAND_CONTACT_KD = 1.0e-3
HAND_CONTACT_MU = 2.5
OBJECT_CONTACT_MU = 1.2
TABLE_CONTACT_MU = 0.9


@wp.kernel(enable_backward=False)
def _scatter_cached_joint_targets(
    cached_q: wp.array2d[wp.float32],
    frame_index: int,
    q_indices: wp.array[wp.int32],
    target_indices: wp.array[wp.int32],
    joint_target_q: wp.array[wp.float32],
):
    i = wp.tid()
    joint_target_q[target_indices[i]] = cached_q[frame_index, q_indices[i]]


@wp.kernel(enable_backward=False)
def _hold_bag_rim(
    particle_ids: wp.array[wp.int32],
    rest_q: wp.array[wp.vec3],
    particle_q_in: wp.array[wp.vec3],
    particle_q_out: wp.array[wp.vec3],
):
    i = wp.tid()
    particle = particle_ids[i]
    particle_q_in[particle] = rest_q[i]
    particle_q_out[particle] = rest_q[i]


class Example(bag.Example):
    """ADMM-coupled physical-grasp variant of the rigid-into-bag task."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / bag.FPS
        self.sim_substeps = max(1, int(args.substeps))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame_index = 0
        if args.impact_height <= 0.0:
            raise ValueError(f"--impact-height must be > 0, got {args.impact_height}")
        self.base_pos = wp.vec3(args.waic_robot_base_x, args.waic_robot_base_y, args.waic_robot_base_z)
        self.base_rot = self._normal_quat(
            wp.quat(args.waic_robot_base_qx, args.waic_robot_base_qy, args.waic_robot_base_qz, args.waic_robot_base_qw)
        )
        self.house_visual_usd = args.house_visual_usd

        self._build_scene()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.control = self.model.control()

        flags = self.model.particle_flags.numpy()
        flags[self.bag_top_indices] &= ~int(newton.ParticleFlags.ACTIVE)
        self.model.particle_flags.assign(flags)
        bag_q = self.state_0.particle_q.numpy()[self.bag_top_indices].copy()
        self.bag_pinned_indices = wp.array(self.bag_top_indices, dtype=wp.int32, device=self.device)
        self.bag_pinned_original = wp.array(bag_q, dtype=wp.vec3, device=self.device)

        self.solver = SolverCoupledADMM(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name="mjc",
                    solver=lambda view: SolverMuJoCo(
                        model=view,
                        solver="newton",
                        integrator="implicitfast",
                        iterations=int(args.mujoco_iterations),
                        ls_iterations=int(args.mujoco_ls_iterations),
                        use_mujoco_contacts=False,
                        njmax=1024,
                        nconmax=512,
                    ),
                    bodies=self.robot_bodies,
                    joints=self.robot_joints,
                ),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda view: SolverVBD(
                        model=view,
                        iterations=int(args.vbd_iterations),
                        rigid_contact_history=False,
                        rigid_body_contact_buffer_size=4096,
                        rigid_body_particle_contact_buffer_size=bag.RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                        particle_enable_self_contact=False,
                        particle_topological_contact_filter_threshold=3,
                    ),
                    bodies=self.object_bodies,
                    particles=list(range(self.bag_particle_start, self.bag_particle_end)),
                ),
            ],
            coupling=SolverCoupledADMM.Config(
                iterations=int(args.admm_iterations),
                rho=float(args.rho),
                gamma=float(args.gamma),
                baumgarte=float(args.baumgarte),
                rigid_contact_matching="latest",
                contact_pairs=[SolverCoupledADMM.ContactPair(source="mjc", destination="vbd")],
            ),
        )
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=bag.SOFT_CONTACT_MARGIN,
            contact_matching="latest",
        )
        self.contacts = self.pipeline.contacts()
        self.solver.prepare_contacts(self.contacts)

        self.left_body = self._body_index(self.model.body_label, "left_j7")
        self.right_body = self._body_index(self.model.body_label, "right_j7")
        self.left_home = self._tcp(self.state_0, self.left_body)
        self.right_home = self._tcp(self.state_0, self.right_body)
        self._build_ik()
        self.segments = self._segments()
        self.ik_q = wp.clone(self.model.joint_q[: self.ik_model.joint_coord_count]).reshape((1, -1))
        self.lock_indices, self.lock_values = self._locked_q()
        self.hand_indices, self.hand_open, self.hand_grasp = self._right_hand_q()
        self._build_joint_target_cache()
        self.target_q_indices, self.target_control_indices = self._control_target_indices()

        count = len(self.object_bodies)
        self.pick_commanded = np.zeros(count, dtype=bool)
        self.release_commanded = np.zeros(count, dtype=bool)
        self.pick_min_tcp_distance = np.full(count, np.inf, dtype=np.float64)
        self.object_peak_z = np.asarray(self.state_0.body_q.numpy()[self.object_bodies, 2], dtype=np.float64)
        self.max_admm_contacts = 0

        self._attach_house_usd()
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = True
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(bag.CAMERA_POS, bag.CAMERA_PITCH, bag.CAMERA_YAW)

    def _build_scene(self):
        """Build the MuJoCo-owned robot and the VBD-owned bag/object domain."""
        self.urdf_path = self._robot_urdf()
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        SolverMuJoCo.register_custom_attributes(builder)
        SolverVBD.register_custom_attributes(builder, dahl_defaults_enabled=False)
        builder.default_shape_cfg.ke = HAND_CONTACT_KE
        builder.default_shape_cfg.kd = HAND_CONTACT_KD
        builder.default_shape_cfg.mu = HAND_CONTACT_MU

        builder.add_urdf(
            str(self.urdf_path),
            xform=wp.transform(self.base_pos, self.base_rot),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.robot_body_end = builder.body_count
        self.robot_joint_end = builder.joint_count
        self.robot_shape_end = builder.shape_count
        self.robot_bodies = list(range(self.robot_body_end))
        self.robot_joints = list(range(self.robot_joint_end))
        self._configure_robot_pd(builder)

        table_cfg = newton.ModelBuilder.ShapeConfig(
            ke=3.0e5,
            kd=1.0e-3,
            mu=TABLE_CONTACT_MU,
            is_visible=bool(self.args.show_physics_table),
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(self._world_vec(bag.TABLE_POS), self.base_rot),
            hx=bag.TABLE_HALF_EXTENTS[0],
            hy=bag.TABLE_HALF_EXTENTS[1],
            hz=bag.TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=bag.TABLE_COLOR,
            label="pick_table",
        )
        builder.add_ground_plane(height=float(self.base_pos[2]), label="pick_ground")

        bag_vertices, bag_indices = bag._generate_box_bag(
            0.5 * bag.BAG_WIDTH,
            0.5 * bag.BAG_DEPTH,
            bag.BAG_HEIGHT,
            bag.BAG_RESOLUTION,
        )
        self.bag_particle_start = builder.particle_count
        builder.add_cloth_mesh(
            pos=self._world_vec(bag.BAG_POS),
            rot=self.base_rot,
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=bag_vertices.tolist(),
            indices=bag_indices,
            density=bag.BAG_DENSITY,
            tri_ke=bag.BAG_TRI_KE,
            tri_ka=bag.BAG_TRI_KA,
            tri_kd=bag.BAG_TRI_KD,
            edge_ke=bag.BAG_EDGE_KE,
            edge_kd=bag.BAG_EDGE_KD,
            particle_radius=bag.BAG_PARTICLE_RADIUS,
            label="suspended_soft_box_bag",
        )
        self.bag_particle_end = builder.particle_count
        top = np.flatnonzero(np.abs(bag_vertices[:, 2] - bag.BAG_HEIGHT) < 1.0e-5)
        self.bag_top_indices = top.astype(np.int32) + self.bag_particle_start

        object_cfg = newton.ModelBuilder.ShapeConfig(
            density=bag.SHAPE_DENSITY,
            ke=bag.SOFT_CONTACT_KE,
            kd=bag.SOFT_CONTACT_KD,
            mu=OBJECT_CONTACT_MU,
            margin=bag.SHAPE_MARGIN,
        )
        object_cfg.has_particle_collision = True
        bear_points, bear_indices = bag._load_bear_mesh(bag.SHAPE_SIZE)
        bear_mesh = newton.Mesh(bear_points, bear_indices)
        self.object_bodies = []
        self.object_shapes = []
        for name, position, color in zip(bag.SHAPE_NAMES, bag.SHAPE_POSITIONS, bag.SHAPE_COLORS, strict=True):
            body = builder.add_body(
                xform=wp.transform(self._world_vec(position), self.base_rot),
                label=f"pick_{name}",
            )
            shape = builder.shape_count
            if name == "mesh":
                builder.add_shape_mesh(body, mesh=bear_mesh, cfg=object_cfg, color=color, label=f"pick_{name}_shape")
            elif name == "cone":
                builder.add_shape_cone(
                    body,
                    radius=bag.SHAPE_SIZE,
                    half_height=bag.SHAPE_SIZE,
                    cfg=object_cfg,
                    color=color,
                    label=f"pick_{name}_shape",
                )
            elif name == "sphere":
                builder.add_shape_sphere(
                    body, radius=bag.SHAPE_SIZE, cfg=object_cfg, color=color, label=f"pick_{name}_shape"
                )
            elif name == "box":
                builder.add_shape_box(
                    body,
                    hx=bag.SHAPE_SIZE,
                    hy=bag.SHAPE_SIZE,
                    hz=bag.SHAPE_SIZE,
                    cfg=object_cfg,
                    color=color,
                    label=f"pick_{name}_shape",
                )
            elif name == "cylinder":
                builder.add_shape_cylinder(
                    body,
                    radius=bag.SHAPE_SIZE,
                    half_height=0.5 * bag.SHAPE_SIZE,
                    cfg=object_cfg,
                    color=color,
                    label=f"pick_{name}_shape",
                )
            self.object_bodies.append(body)
            self.object_shapes.append(shape)

        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape in range(self.robot_shape_end):
            body = int(builder.shape_body[shape])
            label = builder.body_label[body].lower() if body >= 0 else ""
            is_right_hand = "right" in label and any(word in label for word in self.HAND_CONTACT_KEYWORDS)
            if is_right_hand:
                builder.shape_flags[shape] |= collide_shapes
                builder.shape_flags[shape] &= ~collide_particles
            else:
                builder.shape_flags[shape] &= ~(collide_shapes | collide_particles)
        for shape in range(self.robot_shape_end, builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes | collide_particles

        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = bag.SOFT_CONTACT_KE
        self.model.soft_contact_kd = bag.SOFT_CONTACT_KD
        self.model.soft_contact_mu = bag.SOFT_CONTACT_MU

    def _configure_robot_pd(self, builder: newton.ModelBuilder):
        """Give every robot joint a stable MuJoCo position target."""
        qd_start = builder.joint_qd_start
        for joint in range(self.robot_joint_end):
            dof_start = int(qd_start[joint])
            dof_end = int(qd_start[joint + 1]) if joint + 1 < self.robot_joint_end else builder.joint_dof_count
            label = builder.joint_label[joint].lower()
            is_hand = any(word in label for word in self.HAND_CONTACT_KEYWORDS)
            for dof in range(dof_start, dof_end):
                builder.joint_target_ke[dof] = 1100.0 if is_hand else 700.0
                builder.joint_target_kd[dof] = 90.0 if is_hand else 70.0
                builder.joint_effort_limit[dof] = 100.0 if is_hand else 220.0
                builder.joint_armature[dof] = 0.005 if is_hand else 0.03

    def _control_target_indices(self):
        q_start = self.model.joint_q_start.numpy()
        target_start = self.model.joint_target_q_start.numpy()
        q_indices: list[int] = []
        target_indices: list[int] = []
        for joint in self.robot_joints:
            q_begin = int(q_start[joint])
            q_end = int(q_start[joint + 1]) if joint + 1 < self.robot_joint_end else self.model.joint_coord_count
            target_begin = int(target_start[joint])
            target_end = (
                int(target_start[joint + 1]) if joint + 1 < self.robot_joint_end else self.model.joint_target_q.shape[0]
            )
            for local in range(min(q_end - q_begin, target_end - target_begin)):
                q_indices.append(q_begin + local)
                target_indices.append(target_begin + local)
        return (
            wp.array(q_indices, dtype=wp.int32, device=self.device),
            wp.array(target_indices, dtype=wp.int32, device=self.device),
        )

    def _prepare_frame_targets(self):
        frame = min(self.frame_index + 1, self.cached_frame_count)
        wp.launch(
            _scatter_cached_joint_targets,
            dim=self.target_q_indices.shape[0],
            inputs=[
                self.cached_joint_targets,
                frame,
                self.target_q_indices,
                self.target_control_indices,
                self.control.joint_target_q,
            ],
            device=self.device,
        )
        object_index = int(self.cached_objects[frame])
        grip = float(self.cached_grips[frame])
        if object_index >= 0:
            if grip >= 0.99:
                self.pick_commanded[object_index] = True
                object_q = self.state_0.body_q.numpy()[self.object_bodies[object_index]]
                object_position = np.asarray(object_q[:3], dtype=np.float64)
                tcp_position = np.asarray(wp.transform_get_translation(self._tcp(self.state_0, self.right_body)))
                self.pick_min_tcp_distance[object_index] = min(
                    self.pick_min_tcp_distance[object_index], float(np.linalg.norm(object_position - tcp_position))
                )
            elif self.pick_commanded[object_index]:
                self.release_commanded[object_index] = True

    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.launch(
                _hold_bag_rim,
                dim=self.bag_pinned_indices.shape[0],
                inputs=[
                    self.bag_pinned_indices,
                    self.bag_pinned_original,
                    self.state_0.particle_q,
                    self.state_1.particle_q,
                ],
                device=self.device,
            )
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.model.collide(self.state_0, self.contacts, collision_pipeline=self.pipeline)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            newton.eval_ik(self.model, self.state_1, self.state_1.joint_q, self.state_1.joint_qd)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self._prepare_frame_targets()
        self.simulate()
        self.sim_time += self.frame_dt
        self.frame_index += 1
        object_q = self.state_0.body_q.numpy()[self.object_bodies]
        self.object_peak_z = np.maximum(self.object_peak_z, object_q[:, 2])
        self.max_admm_contacts = max(self.max_admm_contacts, int(self.solver.collision_contact_count_max))

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        if hasattr(self.viewer, "log_coupled_view"):
            newton.examples.log_coupled_view(self, self.contacts)
        else:
            self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        body_q = self.state_0.body_q.numpy()[self.object_bodies]
        assert np.isfinite(particle_q).all(), "Bag particle positions contain non-finite values"
        assert np.isfinite(body_q).all(), "Rigid object states contain non-finite values"
        script_frames = int(
            np.ceil(sum(segment[0] for segment in self.segments) / (self.frame_dt * self.args.trajectory_time_scale))
        )
        if self.frame_index < script_frames:
            return

        assert self.max_admm_contacts > 0, "ADMM did not create any robot--VBD contact candidates"
        assert np.all(self.pick_commanded), (
            f"Not every object reached a closed-hand phase: {self.pick_commanded.tolist()}"
        )
        assert np.all(self.release_commanded), (
            f"Not every object reached a release phase: {self.release_commanded.tolist()}"
        )
        assert np.all(self.pick_min_tcp_distance < 0.07), (
            "A grasp command was issued before the hand reached its object: "
            f"min TCP gaps={self.pick_min_tcp_distance.tolist()}"
        )
        assert np.all(self.object_peak_z > bag.TABLE_TOP_Z + 0.10), (
            f"At least one object never lifted from the table: peak z={self.object_peak_z.tolist()}"
        )

        bag_scene_q = np.asarray([self._scene_vec(wp.vec3(*position)) for position in particle_q])
        bag_min_z = float(bag_scene_q[:, 2].min())
        inside = 0
        for transform in body_q:
            position = self._scene_vec(wp.vec3(*transform[:3]))
            if (
                abs(float(position[0]) - float(bag.BAG_POS[0])) < 0.5 * bag.BAG_WIDTH + bag.SHAPE_SIZE
                and abs(float(position[1]) - float(bag.BAG_POS[1])) < 0.5 * bag.BAG_DEPTH + bag.SHAPE_SIZE
                and bag_min_z - bag.SHAPE_SIZE < float(position[2]) < bag.TABLE_TOP_Z + 0.08
            ):
                inside += 1
        assert inside == len(self.object_bodies), (
            f"Only {inside}/{len(self.object_bodies)} rigid objects settled in the bag"
        )

    @staticmethod
    def create_parser():
        parser = bag.Example.create_parser()
        # The predecessor drives the robot kinematically at 2x script speed.
        # MuJoCo must physically track the same targets and establish frictional
        # finger contacts, so use real-time script playback by default.
        parser.set_defaults(num_frames=1500, trajectory_time_scale=1.0)
        parser.add_argument("--substeps", type=int, default=8, help="ADMM physics substeps per rendered frame.")
        parser.add_argument("--admm-iterations", type=int, default=ADMM_ITERATIONS, help="ADMM iterations per substep.")
        parser.add_argument("--rho", type=float, default=200.0, help="ADMM penalty parameter.")
        parser.add_argument("--gamma", type=float, default=0.001, help="ADMM proximal mass scaling.")
        parser.add_argument("--baumgarte", type=float, default=0.5, help="ADMM position-error correction ratio.")
        parser.add_argument(
            "--mujoco-iterations", type=int, default=MUJOCO_ITERATIONS, help="MuJoCo iterations per ADMM pass."
        )
        parser.add_argument(
            "--mujoco-ls-iterations",
            type=int,
            default=MUJOCO_LINE_SEARCH_ITERATIONS,
            help="MuJoCo line-search iterations per ADMM pass.",
        )
        parser.add_argument("--vbd-iterations", type=int, default=VBD_ITERATIONS, help="VBD iterations per ADMM pass.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared scenes for the MJVBDV2 coupling-matrix examples."""

from __future__ import annotations

import math
from collections.abc import Collection

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMJVBDV2

FPS = 60
DEFAULT_SUBSTEPS = 4
DEFAULT_VBD_ITERATIONS = 8
ROBOT_ANGLE = 0.85
ROBOT_SWEEP_TIME = 1.5
PARTICLE_RADIUS = 0.009

VALID_OBJECTS = frozenset({"rigid", "soft", "cloth"})


@wp.kernel
def _accumulate_contact_counts(
    rigid_count: wp.array[int],
    soft_count: wp.array[int],
    max_rigid_count: wp.array[int],
    max_soft_count: wp.array[int],
):
    if wp.tid() == 0:
        max_rigid_count[0] = wp.max(max_rigid_count[0], rigid_count[0])
        max_soft_count[0] = wp.max(max_soft_count[0], soft_count[0])


def _drive_sample(time: float) -> tuple[float, float]:
    phase = math.pi * min(time / ROBOT_SWEEP_TIME, 2.0)
    return -ROBOT_ANGLE * math.cos(phase), ROBOT_ANGLE * (math.pi / ROBOT_SWEEP_TIME) * math.sin(phase)


def _add_robot(builder: newton.ModelBuilder, *, kinematic: bool) -> tuple[int, int, int]:
    inertia = wp.mat33(
        0.015,
        0.0,
        0.0,
        0.0,
        0.08,
        0.0,
        0.0,
        0.0,
        0.08,
    )
    link = builder.add_link(
        mass=2.0,
        inertia=inertia,
        is_kinematic=kinematic,
        label="mjvbd_v2_sweep_link",
    )
    robot_cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=2.0e4,
        kd=2.0e-2,
        mu=0.7,
        margin=0.004,
    )
    builder.add_shape_box(
        link,
        hx=0.23,
        hy=0.028,
        hz=0.04,
        cfg=robot_cfg,
        color=(0.92, 0.36, 0.18),
        label="mjvbd_v2_sweep_paddle",
    )
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.065), q=wp.quat_identity()),
        child_xform=wp.transform(p=wp.vec3(-0.23, 0.0, 0.0), q=wp.quat_identity()),
        label="mjvbd_v2_sweep_joint",
    )
    q_start = builder.joint_q_start[joint]
    qd_start = builder.joint_qd_start[joint]
    builder.joint_q[q_start] = -ROBOT_ANGLE
    builder.joint_target_q[qd_start] = -ROBOT_ANGLE
    builder.joint_target_ke[qd_start] = 120.0
    builder.joint_target_kd[qd_start] = 12.0
    builder.joint_target_mode[qd_start] = int(newton.JointTargetMode.POSITION)
    articulation = builder.articulation_count
    builder.add_articulation([joint], label="mjvbd_v2_sweep_robot")
    return link, joint, articulation


def _add_rigid(builder: newton.ModelBuilder) -> int:
    body = builder.add_body(
        xform=wp.transform(p=wp.vec3(0.32, -0.15, 0.052), q=wp.quat_identity()),
        mass=0.45,
        label="mjvbd_v2_rigid_object",
    )
    cfg = newton.ModelBuilder.ShapeConfig(
        density=0.0,
        ke=2.0e4,
        kd=2.0e-2,
        mu=0.55,
        margin=0.004,
    )
    builder.add_shape_sphere(
        body,
        radius=0.05,
        cfg=cfg,
        color=(0.22, 0.58, 0.94),
        label="mjvbd_v2_rigid_sphere",
    )
    return body


def _add_soft(builder: newton.ModelBuilder, *, pos: wp.vec3 | None = None) -> range:
    start = builder.particle_count
    builder.add_soft_grid(
        pos=pos or wp.vec3(0.27, -0.045, 0.012),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=2,
        dim_y=2,
        dim_z=2,
        cell_x=0.03,
        cell_y=0.03,
        cell_z=0.03,
        density=350.0,
        k_mu=2.0e3,
        k_lambda=2.0e3,
        k_damp=1.0e-2,
        particle_radius=PARTICLE_RADIUS,
        label="mjvbd_v2_tet_soft_body",
    )
    return range(start, builder.particle_count)


def _add_cloth(
    builder: newton.ModelBuilder,
    *,
    pos: wp.vec3 | None = None,
    fixed_edges: bool = False,
) -> range:
    start = builder.particle_count
    builder.add_cloth_grid(
        pos=pos or wp.vec3(0.25, 0.10, 0.014),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=5,
        dim_y=5,
        cell_x=0.02,
        cell_y=0.02,
        mass=0.002,
        fix_left=fixed_edges,
        fix_right=fixed_edges,
        tri_ke=4.0e2,
        tri_ka=4.0e2,
        tri_kd=2.0e-2,
        edge_ke=0.2,
        edge_kd=1.0e-3,
        particle_radius=PARTICLE_RADIUS,
        label="mjvbd_v2_cloth",
    )
    return range(start, builder.particle_count)


class RobotCouplingExample:
    """Programmatic robot coupled to a selectable set of VBD object types."""

    OBJECTS: frozenset[str] = frozenset()

    def __init__(self, viewer, args):
        if not self.OBJECTS or not self.OBJECTS <= VALID_OBJECTS:
            raise ValueError(f"OBJECTS must be a non-empty subset of {sorted(VALID_OBJECTS)}")
        self.viewer = viewer
        self.args = args
        self.joint_mode = args.joint_mode
        self.frame_dt = 1.0 / FPS
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg.ke = 2.0e4
        builder.default_shape_cfg.kd = 2.0e-2
        builder.default_shape_cfg.mu = 0.6
        SolverMJVBDV2.register_custom_attributes(builder)
        self.robot_body, self.robot_joint, self.robot_articulation = _add_robot(
            builder,
            kinematic=self.joint_mode == "kinematic",
        )
        builder.add_ground_plane(label="mjvbd_v2_ground")

        self.rigid_body = _add_rigid(builder) if "rigid" in self.OBJECTS else None
        self.soft_particles = _add_soft(builder) if "soft" in self.OBJECTS else range(0)
        self.cloth_particles = _add_cloth(builder) if "cloth" in self.OBJECTS else range(0)

        builder.color(include_bending=True, balance_colors=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = 8.0e3
        self.model.soft_contact_kd = 5.0e-2
        self.model.soft_contact_mu = 0.4
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        wp.copy(self.control.joint_target_q, self.model.joint_target_q)

        has_rigid = self.rigid_body is not None
        collision_options = {"soft_contact_margin": 0.006}
        if has_rigid:
            collision_options.update(
                broad_phase="nxn",
                enable_rigid_soft_full_surface_contact=True,
            )
        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=[self.robot_articulation],
            joint_mode=self.joint_mode,
            contact_mode="auto",
            vbd_options={
                "iterations": args.vbd_iterations,
                "rigid_body_contact_buffer_size": 512,
                "rigid_body_particle_contact_buffer_size": 1024,
                "particle_enable_self_contact": len(self.OBJECTS & {"soft", "cloth"}) > 1,
                "particle_self_contact_radius": PARTICLE_RADIUS,
                "particle_self_contact_margin": 2.0 * PARTICLE_RADIUS,
                "particle_topological_contact_filter_threshold": 2,
            },
            mujoco_options={"use_mujoco_cpu": self.model.device.is_cpu} if self.joint_mode == "dynamic" else None,
            collision_options=collision_options,
        )
        self.contacts = self.solver.contacts
        self.max_rigid_contacts = wp.zeros(1, dtype=wp.int32, device=self.model.device)
        self.max_soft_contacts = wp.zeros(1, dtype=wp.int32, device=self.model.device)

        self.initial_joint_q = self.state_0.joint_q.numpy().copy()
        self.initial_body_q = self.state_0.body_q.numpy().copy()
        self.initial_particle_q = (
            self.state_0.particle_q.numpy().copy()
            if self.state_0.particle_q is not None
            else np.empty((0, 3), dtype=np.float32)
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.72, -0.85, 0.48), pitch=-22.0, yaw=120.0)

    def _step_kinematic(self, target_q: float, target_qd: float) -> None:
        if self.rigid_body is None:
            wp.copy(self.state_1.particle_q, self.state_0.particle_q)
            wp.copy(self.state_1.particle_qd, self.state_0.particle_qd)
            self.state_1.joint_q.fill_(target_q)
            self.state_1.joint_qd.fill_(target_qd)
            newton.eval_fk(
                self.model,
                self.state_1.joint_q,
                self.state_1.joint_qd,
                self.state_1,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )
        else:
            self.state_0.joint_q.fill_(target_q)
            self.state_0.joint_qd.fill_(target_qd)
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            target_q, target_qd = _drive_sample(self.sim_time)
            if self.joint_mode == "dynamic":
                self.control.joint_target_q.fill_(target_q)
                self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            else:
                self._step_kinematic(target_q, target_qd)
            wp.launch(
                _accumulate_contact_counts,
                dim=1,
                inputs=[
                    self.contacts.rigid_contact_count,
                    self.contacts.soft_contact_count,
                    self.max_rigid_contacts,
                    self.max_soft_contacts,
                ],
                device=self.model.device,
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def step(self):
        self.simulate()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        joint_q = self.state_0.joint_q.numpy()
        body_q = self.state_0.body_q.numpy()
        particle_q = (
            self.state_0.particle_q.numpy()
            if self.state_0.particle_q is not None
            else np.empty((0, 3), dtype=np.float32)
        )
        assert np.all(np.isfinite(joint_q)), "Robot joint state contains non-finite values"
        assert np.all(np.isfinite(body_q)), "Rigid state contains non-finite values"
        assert np.all(np.isfinite(particle_q)), "Particle state contains non-finite values"

        if self.joint_mode == "dynamic":
            expected_backend = "coupled"
        elif self.rigid_body is not None:
            expected_backend = "vbd_kinematic_full"
        else:
            expected_backend = "mjvbd_kinematic_soft"
        assert self.solver.features.backend == expected_backend, (
            f"Expected backend {expected_backend}, got {self.solver.features.backend}"
        )

        if self.sim_time < 1.5:
            return
        assert abs(float(joint_q[0] - self.initial_joint_q[0])) > 0.2, "Robot joint did not sweep through the scene"
        if self.rigid_body is not None:
            displacement = np.linalg.norm(body_q[self.rigid_body, :3] - self.initial_body_q[self.rigid_body, :3])
            assert displacement > 0.01, f"Rigid object did not respond to the scene: displacement={displacement}"
            assert int(self.max_rigid_contacts.numpy()[0]) > 0, "No rigid contacts were generated"
        if particle_q.size:
            displacement = np.linalg.norm(particle_q - self.initial_particle_q, axis=1)
            assert float(np.max(displacement)) > 0.01, "Particle objects did not deform or move"
            assert int(self.max_soft_contacts.numpy()[0]) > 0, "No robot/particle contacts were generated"

    @classmethod
    def create_parser(cls):
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=180)
        parser.add_argument(
            "--joint-mode",
            choices=("dynamic", "kinematic"),
            default="kinematic",
            help="Use MuJoCo joint dynamics or a prescribed kinematic joint trajectory.",
        )
        parser.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS)
        parser.add_argument("--vbd-iterations", type=int, default=DEFAULT_VBD_ITERATIONS)
        return parser


def run_robot_example(example_class: type[RobotCouplingExample]) -> None:
    parser = example_class.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(example_class(viewer, args), args)


class VBDMixExample:
    """Rigid + tetrahedral soft body + cloth, selectable between V2 and native VBD."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.frame_dt = 1.0 / FPS
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        SolverMJVBDV2.register_custom_attributes(builder)
        builder.add_ground_plane(label="mjvbd_v2_mix_ground")
        self.cloth_particles = _add_cloth(
            builder,
            pos=wp.vec3(-0.18, -0.18, 0.24),
            fixed_edges=True,
        )
        self.soft_particles = _add_soft(builder, pos=wp.vec3(-0.045, -0.045, 0.48))
        self.rigid_body = builder.add_body(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, 0.72), q=wp.quat_identity()),
            mass=0.5,
            label="mjvbd_v2_mix_rigid",
        )
        rigid_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=2.0e4, kd=2.0e-2, mu=0.5, margin=0.004)
        builder.add_shape_sphere(
            self.rigid_body,
            radius=0.055,
            cfg=rigid_cfg,
            color=(0.22, 0.58, 0.94),
            label="mjvbd_v2_mix_sphere",
        )
        builder.color(include_bending=True, balance_colors=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = 8.0e3
        self.model.soft_contact_kd = 5.0e-2
        self.model.soft_contact_mu = 0.4
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        vbd_options = {
            "iterations": args.vbd_iterations,
            "rigid_body_contact_buffer_size": 512,
            "rigid_body_particle_contact_buffer_size": 1024,
            "particle_enable_self_contact": True,
            "particle_self_contact_radius": PARTICLE_RADIUS,
            "particle_self_contact_margin": 2.0 * PARTICLE_RADIUS,
            "particle_topological_contact_filter_threshold": 2,
        }
        collision_options = {
            "broad_phase": "nxn",
            "soft_contact_margin": 0.006,
            "enable_rigid_soft_full_surface_contact": True,
            "include_static_kinematic_pairs": False,
        }
        self.pipeline = None
        if args.solver == "mjvbd-v2":
            self.solver = SolverMJVBDV2(
                self.model,
                vbd_options=vbd_options,
                collision_options=collision_options,
            )
            self.contacts = self.solver.contacts
            assert self.solver.features.backend == "pure_vbd"
        else:
            self.pipeline = newton.CollisionPipeline(self.model, **collision_options)
            self.contacts = self.pipeline.contacts()
            self.solver = newton.solvers.SolverVBD(self.model, **vbd_options)

        self.initial_rigid_z = float(self.state_0.body_q.numpy()[self.rigid_body, 2])
        self.initial_particle_q = self.state_0.particle_q.numpy().copy()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.65, -0.8, 0.65), pitch=-28.0, yaw=129.0)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.pipeline is not None:
                self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def step(self):
        self.simulate()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        body_q = self.state_0.body_q.numpy()
        assert np.all(np.isfinite(particle_q)), "Mixed particle state contains non-finite values"
        assert np.all(np.isfinite(body_q)), "Mixed rigid state contains non-finite values"
        if self.sim_time < 1.5:
            return
        assert float(body_q[self.rigid_body, 2]) < self.initial_rigid_z - 0.05, "Rigid sphere did not fall"
        max_particle_motion = float(np.max(np.linalg.norm(particle_q - self.initial_particle_q, axis=1)))
        assert max_particle_motion > 0.05, "Soft/cloth objects did not deform"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=180)
        parser.add_argument("--solver", choices=("mjvbd-v2", "vbd"), default="mjvbd-v2")
        parser.add_argument("--substeps", type=int, default=DEFAULT_SUBSTEPS)
        parser.add_argument("--vbd-iterations", type=int, default=DEFAULT_VBD_ITERATIONS)
        return parser


def run_vbd_mix_example() -> None:
    parser = VBDMixExample.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(VBDMixExample(viewer, args), args)


def object_title(objects: Collection[str]) -> str:
    return " + ".join(name.capitalize() for name in sorted(objects))

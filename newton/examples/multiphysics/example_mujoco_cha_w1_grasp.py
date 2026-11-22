# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Grasp two deformable blocks with the W1 robot using MuJoCo and CHA.

MuJoCo integrates the fixed-base W1 articulation and its two parallel grippers.
SolverCoupledHighAccuracy (CHA) integrates two free tetrahedral blocks. The
experimental SolverCoupledProxy transfers the four finger bodies into CHA and
returns their contact reactions to MuJoCo with lagged virtual-inertia coupling.
The blocks have no attachments or prescribed particles. Flat rubber pads are
attached to the original finger links; all gripping forces come from contact.

Success criteria:
- Both W1 hands approach, close around separate blocks, and lift them off the table.
- Both blocks visibly compress and remain held during the final stationary interval.
- The robot and deformables remain finite and bounded, without explosive motion.

Runs:
- `python -m newton.examples mujoco_cha_w1_grasp`
- `python -m newton.examples mujoco_cha_w1_grasp --viewer null --test`
- `python -m newton.examples mujoco_cha_w1_grasp --no-graph-capture`
"""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupledProxy

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverCoupledHighAccuracy, SolverMuJoCo
from newton.solvers import coupled_high_accuracy as cha

TABLE_Z = 0.92
BLOCK_SIZE = np.array((0.072, 0.048, 0.08))
BLOCK_CENTERS = np.array(((0.49, 0.28, TABLE_Z + 0.045), (0.49, -0.28, TABLE_Z + 0.045)))
TCP = wp.vec3(0.0, 0.0, 0.108)
DURATION = 8.0


def _smooth(value):
    t = np.clip(value, 0.0, 1.0)
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def _targets(time):
    approach = _smooth(time / 1.5)
    lift = _smooth((time - 3.0) / 2.0)
    centers = BLOCK_CENTERS + np.array((-0.04 * (1.0 - approach), 0.0, 0.12 * (1.0 - approach) + 0.18 * lift))
    opening = 0.042 + (0.026 - 0.042) * _smooth((time - 1.5) / 1.0)
    return centers, opening


@wp.kernel
def _drive(
    clock: wp.array[int],
    plan: wp.array2d[float],
    target_ids: wp.array[int],
    substeps: int,
    target_q: wp.array[float],
    target_qd: wp.array[float],
):
    row = wp.min(clock[0] // substeps, plan.shape[0] - 2)
    blend = wp.min(float(clock[0] % substeps) / float(substeps), 1.0)
    for j in range(target_ids.shape[0]):
        target = target_ids[j]
        target_q[target] = wp.lerp(plan[row, j], plan[row + 1, j], blend)
        target_qd[target] = (plan[row + 1, j] - plan[row, j]) * 60.0
    clock[0] += 1


def _add_robot(builder):
    asset = Path(__file__).resolve().parents[1] / "coupled_high_accuracy/assets/w1/robot/w1-030/robot.urdf"
    tree = ET.parse(asset)
    # Fix unused stance/head joints in the imported model while retaining both dynamic arms.
    for joint in tree.getroot().findall("joint"):
        if joint.get("name") in ("ANKLE", "KNEE", "BUTTOCK", "WAIST", "NECK1", "NECK2"):
            joint.set("type", "fixed")
    for mesh in tree.getroot().iter("mesh"):
        mesh.set("filename", str((asset.parent / mesh.get("filename")).resolve()))
    with tempfile.TemporaryDirectory(prefix="newton_w1_") as directory:
        path = Path(directory) / "robot.urdf"
        tree.write(path, encoding="utf-8", xml_declaration=True)
        builder.add_urdf(str(path), floating=False, collapse_fixed_joints=False, enable_self_collisions=False)

    finger_bodies = [
        i
        for i, label in enumerate(builder.body_label)
        if label.rsplit("/", 1)[-1] in ("left_finger1", "left_finger2", "right_finger1", "right_finger2")
    ]
    robot_shapes = list(range(builder.shape_count))
    pad = newton.ModelBuilder.ShapeConfig(density=0.0, ke=2.0e3, kd=1.0, kf=1.0e3, mu=2.5)
    for body in finger_bodies:
        for shape in robot_shapes:
            source = builder.shape_source[shape]
            if (
                builder.shape_body[shape] == body
                and source is not None
                and builder.shape_flags[shape] & newton.ShapeFlags.COLLIDE_PARTICLES
            ):
                # The pad is the deformable contact surface, avoiding a second overlapping metal contact layer.
                builder.shape_flags[shape] &= ~newton.ShapeFlags.COLLIDE_PARTICLES
        shape = builder.add_shape_box(
            body,
            xform=wp.transform(wp.vec3(0.0, -0.0285, -0.0015), wp.quat_identity()),
            hx=0.014,
            hy=0.0315,
            hz=0.0015,
            cfg=pad,
            color=(0.08, 0.09, 0.12),
            label=builder.body_label[body] + "_pad",
        )
        for other in robot_shapes:
            builder.add_shape_collision_filter_pair(other, shape)
        robot_shapes.append(shape)
    for joint, label in enumerate(builder.joint_label):
        if builder.joint_type[joint] == newton.JointType.FIXED:
            continue
        dof = builder.joint_qd_start[joint]
        builder.joint_target_mode[dof] = int(newton.JointTargetMode.POSITION_VELOCITY)
        builder.joint_target_ke[dof] = 8000.0 if "FINGER" in label else 12000.0
        builder.joint_target_kd[dof] = 60.0 if "FINGER" in label else 200.0
    return finger_bodies


def _plan(model):
    types = model.joint_type.numpy()
    starts = model.joint_q_start.numpy()
    dofs = model.joint_qd_start.numpy()
    joints = [j for j in range(model.joint_count) if int(types[j]) != newton.JointType.FIXED]
    coords = starts[joints]
    names = {model.joint_label[j].rsplit("/", 1)[-1]: int(starts[j]) for j in joints}
    ee = [
        next(i for i, name in enumerate(model.body_label) if name.endswith("/" + side + "_gripper_base"))
        for side in ("left", "right")
    ]
    positions = [
        ik.IKObjectivePosition(body, TCP, wp.array([wp.vec3(*p)], dtype=wp.vec3, device=model.device))
        for body, p in zip(ee, _targets(0.0)[0], strict=True)
    ]
    rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), wp.pi / 2.0)
    rotations = [
        ik.IKObjectiveRotation(
            body, wp.quat_identity(), wp.array([wp.vec4(*rotation)], dtype=wp.vec4, device=model.device)
        )
        for body in ee
    ]
    mask = np.zeros(model.joint_dof_count, dtype=bool)
    q0 = model.joint_q.numpy()
    for side, sign in (("LEFT", -1), ("RIGHT", 1)):
        q0[names[f"{side}_J2"]] = sign * 0.9
        q0[names[f"{side}_J4"]] = sign * 1.0
        for joint in joints:
            if model.joint_label[joint].rsplit("/", 1)[-1] in {f"{side}_J{j}" for j in range(1, 8)}:
                mask[dofs[joint]] = True
    solver = ik.IKSolver(
        model,
        1,
        positions
        + rotations
        + [ik.IKObjectiveJointLimit(model.joint_limit_lower, model.joint_limit_upper, weight=10.0)],
        joint_dof_mask=wp.array(mask, dtype=wp.bool, device=model.device),
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
        lambda_initial=0.03,
    )
    q = wp.array(q0.reshape(1, -1), dtype=float, device=model.device)
    state = model.state()
    max_error = 0.0
    plan = []
    for frame in range(round(DURATION * 60) + 1):
        centers, opening = _targets(frame / 60.0)
        for objective, center in zip(positions, centers, strict=True):
            objective.set_target_position(0, wp.vec3(*center))
        solver.step(q, q, iterations=250 if frame == 0 else 12)
        values = q.numpy()[0]
        for side in ("LEFT", "RIGHT"):
            for finger in (1, 2):
                values[names[f"{side}_FINGER{finger}_JOINT"]] = opening
        q.assign(values.reshape(1, -1))
        plan.append(values[coords].copy())
        if frame % 60 == 0:
            newton.eval_fk(model, q.flatten(), model.joint_qd, state)
            poses = state.body_q.numpy()
            actual = np.asarray([wp.transform_point(wp.transform(*poses[b]), TCP) for b in ee])
            max_error = max(max_error, float(np.max(np.linalg.norm(actual - centers, axis=1))))
    if max_error > 0.006:
        raise RuntimeError(f"W1 IK position error exceeds 6 mm: {max_error:.6f} m")
    return np.asarray(plan, np.float32), joints, coords


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.substeps = args.substeps
        if self.substeps <= 0 or self.substeps % 2:
            raise ValueError("--substeps must be a positive even number")
        self.dt = self.frame_dt / self.substeps
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverMuJoCo.register_custom_attributes(builder)
        fingers = _add_robot(builder)
        table = newton.ModelBuilder.ShapeConfig(mu=0.6, ke=2.0e3, kd=1.0, kf=1.0e3)
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.54, 0.0, TABLE_Z - 0.025), wp.quat_identity()),
            hx=0.34,
            hy=0.63,
            hz=0.025,
            cfg=table,
            color=(0.33, 0.36, 0.40),
            label="worktop",
        )
        builder.add_ground_plane()
        self.groups = []
        for i, center in enumerate(BLOCK_CENTERS):
            start = builder.particle_count
            first_triangle = builder.tri_count
            builder.add_soft_grid(
                pos=wp.vec3(*(center - BLOCK_SIZE / 2)),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=3,
                dim_y=3,
                dim_z=4,
                cell_x=BLOCK_SIZE[0] / 3,
                cell_y=BLOCK_SIZE[1] / 3,
                cell_z=BLOCK_SIZE[2] / 4,
                density=250.0,
                k_mu=2000.0,
                k_lambda=4000.0,
                k_damp=0.05,
                particle_radius=0.004,
                label=("left" if i == 0 else "right") + "_soft_block",
            )
            self.groups.append(slice(start, builder.particle_count))
            color = (0.12, 0.65, 0.72) if i == 0 else (0.95, 0.48, 0.16)
            for triangle in range(first_triangle, builder.tri_count):
                builder.tri_color[triangle] = wp.vec3(*color)
        self.model = builder.finalize()
        rest = self.model.particle_q.numpy()
        # Measure the pinched middle layer; the ungripped top/bottom can bulge sideways.
        self.grip_vertices = [
            np.flatnonzero((np.abs(rest[group, 0] - center[0]) < 0.02) & (np.abs(rest[group, 2] - center[2]) < 0.006))
            + group.start
            for group, center in zip(self.groups, BLOCK_CENTERS, strict=True)
        ]
        plan, joints, coords = _plan(self.model)
        initial = self.model.joint_q.numpy()
        initial[coords] = plan[0]
        self.model.joint_q.assign(initial)
        targets = self.model.joint_target_q.numpy()
        target_ids = self.model.joint_target_q_start.numpy()[joints]
        targets[target_ids] = plan[0]
        self.model.joint_target_q.assign(targets)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.model)
        self.model.soft_contact_ke = 2.0e3
        self.model.soft_contact_kd = 1.0
        self.model.soft_contact_kf = 1.0e3
        self.model.soft_contact_mu = 2.5
        self.plan = wp.array(plan, dtype=float, device=self.model.device)
        self.target_ids = wp.array(target_ids, dtype=int, device=self.model.device)
        self.clock = wp.zeros(1, dtype=int, device=self.model.device)

        def make_cha(view):
            solver = SolverCoupledHighAccuracy(view)
            solver.config.affine_body_rigidity_model = cha.AffineBodyRigidityModel.SO3_AUGMENTED_LAGRANGIAN
            # Predict proxy wrenches using the articulated effective inertia supplied by MuJoCo.
            solver.config.inertia_mode = cha.InertiaMode.IMPLICIT_EULER
            solver.config.constraint_alpha = 500.0
            solver.config.affine_body_block_preconditioner = True
            solver.config.nonlinear_iterations = args.cha_iterations
            solver.config.linear_iterations = args.linear_iterations
            solver.config.enable_collision_hessian_non_diagonal = True
            self.cha = solver
            return solver

        self.solver = SolverCoupledProxy(
            self.model,
            entries=[
                SolverCoupledProxy.Entry(
                    name="mujoco",
                    solver=lambda v: SolverMuJoCo(v, disable_contacts=True, use_mujoco_contacts=False, iterations=30),
                    bodies=list(range(self.model.body_count)),
                    joints=list(range(self.model.joint_count)),
                ),
                SolverCoupledProxy.Entry(name="cha", solver=make_cha, particles=list(range(self.model.particle_count))),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source="mujoco",
                        destination="cha",
                        bodies=fingers,
                        proxy_relaxation=args.proxy_relaxation,
                        collision_pipeline=lambda v: newton.CollisionPipeline(
                            v,
                            enable_rigid_soft_full_surface_contact=True,
                            full_surface_force_contacts_only=True,
                            soft_contact_max=8192,
                        ),
                        collide_interval=1,
                    )
                ],
            ),
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        for state in (self.state_0, self.state_1):
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, state)
        # MuJoCo requires a buffer even with contacts disabled; CHA detects contacts on its own view.
        self.outer_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.outer_pipeline.contacts()
        self.viewer.set_model(self.model)
        self.viewer.show_particles = False
        self.viewer.set_camera(pos=wp.vec3(2.25, -2.1, 1.75), pitch=-15.0, yaw=136.0)
        self.minimum_width = np.full(2, BLOCK_SIZE[1])
        self.late_centers = []
        self.graph = None
        if args.graph_capture:
            with wp.ScopedCapture(device=self.model.device) as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.substeps):
            wp.launch(
                _drive,
                dim=1,
                inputs=[
                    self.clock,
                    self.plan,
                    self.target_ids,
                    self.substeps,
                    self.control.joint_target_q,
                    self.control.joint_target_qd,
                ],
                device=self.model.device,
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def test_post_step(self):
        """Check finite, bounded motion and measure compression at the finger pads."""
        particles = self.state_0.particle_q.numpy()
        assert np.isfinite(particles).all(), "Non-finite block positions"
        assert np.isfinite(self.state_0.body_q.numpy()).all(), "Non-finite MuJoCo state"
        for i, group in enumerate(self.groups):
            points = particles[group]
            assert np.linalg.norm(np.ptp(points, axis=0)) < 0.3, "Block expanded excessively"
            self.minimum_width[i] = min(self.minimum_width[i], np.ptp(particles[self.grip_vertices[i], 1]))
        if self.sim_time >= 6.0:
            self.late_centers.append(np.asarray([particles[group].mean(axis=0) for group in self.groups]))

    def test_final(self):
        """Verify both free blocks compress and remain lifted through the final hold."""
        particles = self.state_0.particle_q.numpy()
        for i, group in enumerate(self.groups):
            center = particles[group].mean(axis=0)
            assert center[2] > BLOCK_CENTERS[i, 2] + 0.12, f"Hand {i} did not retain its lifted block: {center}"
            assert np.linalg.norm(center[:2] - BLOCK_CENTERS[i, :2]) < 0.06, f"Block {i} slipped sideways"
            assert self.minimum_width[i] < BLOCK_SIZE[1] - 0.001, f"Block {i} did not compress"
        assert np.all(self.model.particle_mass.numpy() > 0.0), "Blocks must have no fixed particles"
        assert len(self.late_centers) >= 60, "Run at least 7 seconds to validate the stationary hold"
        assert np.max(np.ptp(self.late_centers, axis=0)) < 0.003, "Blocks drifted during the stationary hold"
        feedback = self.solver.entry_state("mujoco", phase="input").body_f.numpy()
        assert np.linalg.norm(feedback) > 1.0e-3, "CHA contact reactions did not reach MuJoCo"

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=round(DURATION * 60))
        parser.add_argument("--substeps", type=int, default=12)
        parser.add_argument("--cha-iterations", type=int, default=4)
        parser.add_argument("--linear-iterations", type=int, default=40)
        parser.add_argument("--proxy-relaxation", type=float, default=0.5)
        parser.add_argument("--graph-capture", action=argparse.BooleanOptionalAction, default=True)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)

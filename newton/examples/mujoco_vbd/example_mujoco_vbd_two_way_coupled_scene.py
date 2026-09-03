# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the multiphysics proxy-coupling scene with SolverMuJoCoVBD.

Three free rigid boxes and a three-link pendulum are integrated by the private
MuJoCo core.  A pinned cloth sheet and three tetrahedral soft bodies are
integrated by the private VBD core.  Their contacts use the standalone
solver's finite-mass two-way coupling path; this example has no dependency on
``SolverCoupledProxy``.
"""

from __future__ import annotations

import argparse

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMuJoCoVBD


class Example:
    """Dynamic rigid bodies and a pendulum interacting with deformables."""

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.track_metrics = bool(args.test)
        self.max_soft_contact_count = 0
        self.max_feedback_force = 0.0

        builder = newton.ModelBuilder()
        SolverMuJoCoVBD.register_custom_attributes(builder)
        builder.default_shape_cfg.ke = 2.0e4
        builder.add_ground_plane(label="coupled_scene_ground")

        mujoco_joints: list[int] = []
        self.rigid_bodies = self._emit_rigid_bodies(builder, mujoco_joints)
        self.pendulum_bodies = self._emit_articulated_chain(builder, mujoco_joints)
        self._emit_cloth(builder)
        self._emit_soft_bodies(builder)
        builder.color()

        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_mu = 0.5

        self.solver = SolverMuJoCoVBD(
            self.model,
            mujoco_joints=mujoco_joints,
            joint_mode="dynamic",
            coupling_mode="two_way",
            contact_mode="soft",
            coupling_options={
                "iterations": 3,
                "relaxation": "fixed",
                "soft_contact_speculative_distance": 0.01,
                "soft_contact_augmented_lagrangian": True,
            },
            mujoco_options={"njmax": 200},
            vbd_options={
                "iterations": 10,
                "friction_epsilon": 0.01,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": 0.01,
                "particle_self_contact_margin": 0.01,
            },
            collision_options={"soft_contact_margin": 0.01},
        )
        if self.solver.backend_kind.value != "two_way":
            raise RuntimeError(f"Expected two_way, got {self.solver.backend_kind.value}")

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.solver.contacts
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(2.6, -3.5, 2.25), pitch=-8.0, yaw=126.0)

        self.graph = None
        if bool(args.graph_capture) and self.model.device.is_cuda:
            with wp.ScopedDevice(self.model.device), wp.ScopedCapture() as capture:
                self._simulate_frame()
            self.graph = capture.graph

    def _simulate_frame(self):
        """Enqueue one complete display-frame simulation."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance one display frame and collect two-way diagnostics."""
        if self.graph is None:
            self._simulate_frame()
        else:
            wp.capture_launch(self.graph)

        if self.track_metrics:
            wp.synchronize()
            if self.contacts is not None:
                count = int(self.contacts.soft_contact_count.numpy()[0])
                self.max_soft_contact_count = max(self.max_soft_contact_count, count)
            feedback = self.solver.diagnostics.feedback_wrench_raw.numpy()
            if feedback.size:
                self.max_feedback_force = max(
                    self.max_feedback_force,
                    float(np.max(np.linalg.norm(feedback[:, :3], axis=1))),
                )

        self.sim_time += self.frame_dt

    def render(self):
        """Render the combined MuJoCo/VBD state and generated contacts."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts is not None:
            self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Reject explosions, deep ground penetration, and inactive coupling."""
        body_q = self.state_0.body_q.numpy()
        particle_q = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(body_q))
        assert np.all(np.isfinite(particle_q))
        assert float(np.min(body_q[:, 2])) > -0.2
        assert float(np.min(particle_q[:, 2])) > -0.5
        assert float(np.linalg.norm(np.ptp(particle_q, axis=0))) < 20.0
        assert self.max_soft_contact_count > 0, "No rigid-deformable contact was generated"
        assert self.max_feedback_force > 1.0e-3, "No deformable reaction was returned to MuJoCo"

    @staticmethod
    def _emit_rigid_bodies(builder: newton.ModelBuilder, mujoco_joints: list[int]) -> list[int]:
        """Add the original scene's three free MuJoCo boxes."""
        bodies: list[int] = []
        boxes = [
            (wp.vec3(0.0, 0.0, 2.0), (0.15, 0.15, 0.15), 10.0),
            (wp.vec3(0.3, 0.1, 2.5), (0.10, 0.20, 0.10), 5.0),
            (wp.vec3(-0.2, -0.1, 3.0), (0.12, 0.12, 0.12), 8.0),
        ]
        for index, (position, (hx, hy, hz), mass) in enumerate(boxes):
            body = builder.add_body(
                xform=wp.transform(position, wp.quat_identity()),
                mass=mass,
                label=f"falling_box_{index}",
            )
            mujoco_joints.append(builder.joint_count - 1)
            builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, color=(0.9, 0.35, 0.08))
            bodies.append(body)
        return bodies

    @staticmethod
    def _emit_cloth(builder: newton.ModelBuilder) -> None:
        """Add the original 30 by 30 pinned cloth sheet."""
        stiffness = 1.0e5
        builder.add_cloth_grid(
            pos=wp.vec3(-0.5, -0.5, 1.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            fix_left=True,
            fix_right=True,
            dim_x=30,
            dim_y=30,
            cell_x=1.0 / 30.0,
            cell_y=1.0 / 30.0,
            mass=0.1,
            tri_ke=stiffness,
            tri_ka=stiffness,
            tri_kd=1.0e-2 * stiffness,
            edge_ke=0.01,
            edge_kd=1.0e-4,
            particle_radius=0.01,
            label="pinned_cloth",
        )

    @staticmethod
    def _emit_articulated_chain(builder: newton.ModelBuilder, mujoco_joints: list[int]) -> list[int]:
        """Add the original three-link MuJoCo pendulum."""
        hx, hy, hz = 0.21, 0.05, 0.05
        anchor = wp.vec3(0.6, 0.0, 2.25)
        bodies: list[int] = []
        joints: list[int] = []
        parent = -1
        for index in range(3):
            body = builder.add_link(label=f"pendulum_link_{index}")
            builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, color=(0.12, 0.4, 0.9))
            parent_xform = (
                wp.transform(anchor, wp.quat_identity())
                if parent < 0
                else wp.transform(wp.vec3(hx, 0.0, 0.0), wp.quat_identity())
            )
            joint = builder.add_joint_revolute(
                parent=parent,
                child=body,
                axis=wp.vec3(0.0, 1.0, 0.0),
                target_kd=5.0,
                parent_xform=parent_xform,
                child_xform=wp.transform(wp.vec3(-hx, 0.0, 0.0), wp.quat_identity()),
                label=f"pendulum_joint_{index}",
            )
            bodies.append(body)
            joints.append(joint)
            parent = body
        builder.add_articulation(joints, label="pendulum")
        mujoco_joints.extend(joints)
        return bodies

    @staticmethod
    def _emit_soft_bodies(builder: newton.ModelBuilder) -> None:
        """Add the original three volumetric VBD bodies."""
        grids = [
            (wp.vec3(-0.15, -0.15, 1.3), (3, 3, 3), 0.07),
            (wp.vec3(0.25, 0.20, 1.5), (2, 2, 4), 0.07),
            (wp.vec3(-0.30, 0.25, 1.8), (2, 4, 2), 0.07),
        ]
        for index, (position, (dim_x, dim_y, dim_z), cell) in enumerate(grids):
            builder.add_soft_grid(
                pos=position,
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=dim_x,
                dim_y=dim_y,
                dim_z=dim_z,
                cell_x=cell,
                cell_y=cell,
                cell_z=cell,
                density=1.0e3,
                k_mu=1.0e6,
                k_lambda=1.0e6,
                k_damp=1.0e3,
                particle_radius=0.025,
                label=f"soft_body_{index}",
            )

    @staticmethod
    def create_parser():
        """Create the standard example command-line parser."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=300)
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture all eight substeps as one CUDA graph.",
        )
        return parser


def main():
    """Run the standalone two-way coupled-scene reproduction."""
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

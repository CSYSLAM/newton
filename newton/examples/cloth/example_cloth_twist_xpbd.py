# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Twist the square cloth with global XPBD FEM and Planar-DAT.

Run: uv run --extra examples -m newton.examples cloth_twist_xpbd
Use --fem-solver tgs for the separate PhysX-equation baseline.
Use --no-self-contact to inspect elasticity without mesh contact.
The mesh, opposing boundary rotations, camera and duration match cloth_twist.
"""

import argparse
import math
import os

import numpy as np
import warp as wp
import warp.examples
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton import ParticleFlags


@wp.kernel
def initialize_rotation(
    ids: wp.array[int], pos: wp.array[wp.vec3], roots: wp.array[wp.vec3], offsets: wp.array[wp.vec3]
):
    i = wp.tid()
    p = pos[ids[i]]
    roots[i] = wp.vec3(0.0, p[1], 0.0)
    offsets[i] = p - roots[i]


@wp.kernel
def set_rotation_targets(
    ids: wp.array[int],
    axes: wp.array[wp.vec3],
    roots: wp.array[wp.vec3],
    offsets: wp.array[wp.vec3],
    time: wp.array[float],
    dt: float,
    targets: wp.array[wp.vec3],
):
    i = wp.tid()
    # Preserve the reference demo's 1/600 s endpoint lag when changing the
    # outer detection cadence; frame-end prescribed positions remain equal.
    target_time = wp.clamp(time[0] + dt - 1.0 / 600.0, 0.0, 10.0)
    rotation = wp.quat_from_axis_angle(axes[i], target_time * (wp.pi / 3.0))
    targets[ids[i]] = roots[i] + wp.quat_rotate(rotation, offsets[i])


@wp.kernel
def advance_clock(dt: float, time: wp.array[float]):
    # Separate launch: every anchor reads the same time, without a read/write race.
    time[0] += dt


class Example:
    """Drive fixed boundary targets through the same DAT transaction as the cloth."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        if self.sim_substeps <= 0:
            raise ValueError("substeps must be positive")
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frames = 0
        self.verify_every = args.verify_every
        global_fem = args.fem_solver == "global"
        iterations = args.iterations if args.iterations is not None else (2 if global_fem else 12)
        damping = args.damping if args.damping is not None else (0.0002 if global_fem else 600.0)
        bending_damping = (
            args.bending_damping if args.bending_damping is not None else (0.01 if global_fem else damping)
        )
        stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
        mesh = newton.usd.get_mesh(stage.GetPrimAtPath("/root/cloth/cloth"))
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), math.pi / 2),
            scale=0.01,
            vertices=[wp.vec3(v) for v in mesh.vertices],
            indices=mesh.indices,
            vel=wp.vec3(0.0),
            density=0.2,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=damping,
            edge_ke=1.0e-3,
            edge_kd=bending_damping,
        )
        self.model = builder.finalize()
        # VBD's example multiplies edge_ke by rest length during its solve.
        # PhysX-style FEM expects the cooked hinge energy coefficient itself.
        bending = self.model.edge_bending_properties.numpy()
        bending[:, 0] *= self.model.edge_rest_length.numpy()
        self.model.edge_bending_properties.assign(bending)
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_mu = 0.2
        self.model.soft_contact_kd = 0.1
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        # Select the two opposite boundaries by geometry, without assuming vertex ordering.
        rest = self.model.particle_q.numpy()
        axis = 1
        left = np.flatnonzero(np.isclose(rest[:, axis], rest[:, axis].max(), atol=1.0e-6))
        right = np.flatnonzero(np.isclose(rest[:, axis], rest[:, axis].min(), atol=1.0e-6))
        self.anchor_ids = np.concatenate((left, right))
        flags = self.model.particle_flags.numpy()
        flags[self.anchor_ids] &= ~int(ParticleFlags.ACTIVE)
        self.model.particle_flags.assign(flags)
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            particle_fem=True,
            particle_fem_solver=args.fem_solver,
            particle_fem_linear_iterations=(
                args.linear_iterations if args.linear_iterations is not None else (5 if global_fem else 4)
            ),
            iterations=iterations,
            particle_self_contact_radius=0.002 if args.self_contact else 0.0,
            particle_self_contact_margin=args.contact_margin,
        )
        self.rot_ids = wp.array(self.anchor_ids, dtype=int, device=self.model.device)
        # Match the original example's rotation axis, perpendicular to each clamped edge.
        rotation_axis = wp.vec3(0.0, 1.0, 0.0)
        axes = [rotation_axis] * len(left) + [-rotation_axis] * len(right)
        self.axes = wp.array(axes, dtype=wp.vec3, device=self.model.device)
        self.roots = wp.zeros(len(self.anchor_ids), dtype=wp.vec3, device=self.model.device)
        self.offsets = wp.zeros_like(self.roots)
        self.t = wp.zeros(1, dtype=float, device=self.model.device)
        wp.launch(
            initialize_rotation,
            dim=len(self.anchor_ids),
            inputs=[
                self.rot_ids,
                self.state_0.particle_q,
            ],
            outputs=[self.roots, self.offsets],
            device=self.model.device,
        )
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(2.25, 0.0, 0.0), 0.0, -180.0)
        self.graph = None
        if self.model.device.is_cuda and not args.no_graph:
            # The captured frame must restore ping-pong ownership.
            if self.sim_substeps % 2:
                raise ValueError("CUDA frame capture requires an even --substeps value")
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        self.solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(
                set_rotation_targets,
                dim=len(self.anchor_ids),
                inputs=[
                    self.rot_ids,
                    self.axes,
                    self.roots,
                    self.offsets,
                    self.t,
                    self.sim_dt,
                ],
                outputs=[self.solver.particle_kinematic_targets],
                device=self.model.device,
            )
            wp.launch(advance_clock, dim=1, inputs=[self.sim_dt, self.t], device=self.model.device)
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.frames += 1
        self.sim_time += self.frame_dt
        if self.verify_every > 0 and self.frames % self.verify_every == 0:
            self.test_final()

    def test_final(self):
        """Check finite states and independently audit segment-triangle crossings."""
        self.solver.validate_particle_contacts()
        pos = self.state_0.particle_q.numpy()
        velocity = self.state_0.particle_qd.numpy()
        if not np.isfinite(pos).all() or not np.isfinite(velocity).all():
            raise AssertionError("Nonfinite cloth state")
        if np.max(np.abs(pos)) > 2.0:
            raise AssertionError("Cloth escaped the expected scene bounds")
        targets = self.solver.particle_kinematic_targets.numpy()
        if np.linalg.norm(pos[self.anchor_ids] - targets[self.anchor_ids], axis=1).max() > 2.0e-5:
            raise AssertionError("DAT blocked the driven boundary; the intended twist was not completed")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=600)
        parser.add_argument("--substeps", type=int, default=10)
        parser.add_argument("--contact-margin", type=float, default=0.0035)
        parser.add_argument("--fem-solver", choices=("global", "tgs"), default="global")
        parser.add_argument("--iterations", type=int, help="Nonlinear solves (global: 2) or temporal slices (tgs: 12).")
        parser.add_argument(
            "--linear-iterations", type=int, help="PCG iterations per global nonlinear solve (default: 5)."
        )
        parser.add_argument("--damping", type=float, help="Membrane damping: time [s] in global; rate [1/s] in tgs.")
        parser.add_argument("--bending-damping", type=float, help="Hinge damping; same units as --damping.")
        parser.add_argument("--self-contact", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument(
            "--verify-every",
            type=int,
            default=60,
            help="Frames between independent geometry audits; 0 disables periodic audits.",
        )
        parser.add_argument("--no-graph", action="store_true")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Softbody IPC Twist
#
# This simulation twists a tetrahedral soft beam by prescribing opposite
# rotations on its left and right end faces while solving the interior
# deformation with the experimental uipc-backed SolverIPC.
#
# Command: python -m newton.examples softbody_ipc_twist
#
###########################################################################

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton import ParticleFlags


@wp.kernel
def set_end_state(
    particle_indices: wp.array[int],
    centers: wp.array[wp.vec3],
    axes: wp.array[wp.vec3],
    offsets: wp.array[wp.vec3],
    angle: float,
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
):
    tid = wp.tid()
    particle_index = particle_indices[tid]
    center = centers[tid]
    axis = axes[tid]
    offset = offsets[tid]

    theta = angle
    ux = axis[0]
    uy = axis[1]
    uz = axis[2]

    rot = wp.mat33(
        wp.cos(theta) + ux * ux * (1.0 - wp.cos(theta)),
        ux * uy * (1.0 - wp.cos(theta)) - uz * wp.sin(theta),
        ux * uz * (1.0 - wp.cos(theta)) + uy * wp.sin(theta),
        uy * ux * (1.0 - wp.cos(theta)) + uz * wp.sin(theta),
        wp.cos(theta) + uy * uy * (1.0 - wp.cos(theta)),
        uy * uz * (1.0 - wp.cos(theta)) - ux * wp.sin(theta),
        uz * ux * (1.0 - wp.cos(theta)) - uy * wp.sin(theta),
        uz * uy * (1.0 - wp.cos(theta)) + ux * wp.sin(theta),
        wp.cos(theta) + uz * uz * (1.0 - wp.cos(theta)),
    )

    rotated = center + rot * offset
    particle_q[particle_index] = rotated
    particle_qd[particle_index] = wp.vec3(0.0, 0.0, 0.0)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.twist_angular_velocity = math.pi / 3.0
        self.twist_end_time = 6.0

        dim_x = 20
        dim_y = 4
        dim_z = 4
        cell_size = 0.05

        beam_length = dim_x * cell_size
        beam_width = dim_y * cell_size
        beam_height = dim_z * cell_size

        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        builder.add_soft_grid(
            pos=wp.vec3(-0.5 * beam_length, -0.5 * beam_width, 0.9),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=dim_x,
            dim_y=dim_y,
            dim_z=dim_z,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=7.5e2,
            k_mu=8.0e4,
            k_lambda=1.2e5,
            k_damp=0.0,
            particle_radius=0.01,
        )
        builder.color()

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverIPC(self.model, backend=args.ipc_backend)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        left_indices, right_indices = self._build_end_face_indices(dim_x, dim_y, dim_z)
        twist_indices = left_indices + right_indices

        flags = self.model.particle_flags.numpy()
        for particle_index in twist_indices:
            flags[particle_index] = int(flags[particle_index]) & ~int(ParticleFlags.ACTIVE)
        self.model.particle_flags = wp.array(flags, dtype=wp.int32, device=self.model.device)

        particle_positions = self.state_0.particle_q.numpy()
        left_center = particle_positions[left_indices].mean(axis=0)
        right_center = particle_positions[right_indices].mean(axis=0)
        centers = np.vstack(
            [
                np.repeat(left_center[None, :], len(left_indices), axis=0),
                np.repeat(right_center[None, :], len(right_indices), axis=0),
            ]
        ).astype(np.float32, copy=False)
        offsets = (particle_positions[twist_indices] - centers).astype(np.float32, copy=False)
        axes = np.vstack(
            [
                np.repeat(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), len(left_indices), axis=0),
                np.repeat(np.array([[-1.0, 0.0, 0.0]], dtype=np.float32), len(right_indices), axis=0),
            ]
        )

        self.twist_particle_indices = wp.array(np.array(twist_indices, dtype=np.int32), dtype=wp.int32)
        self.twist_centers = wp.array(centers, dtype=wp.vec3)
        self.twist_axes = wp.array(axes, dtype=wp.vec3)
        self.twist_offsets = wp.array(offsets, dtype=wp.vec3)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(2.0, 0.0, 1.0), 0.0, -25.0)

    @staticmethod
    def _build_end_face_indices(dim_x: int, dim_y: int, dim_z: int) -> tuple[list[int], list[int]]:
        stride_x = 1
        stride_y = dim_x + 1
        stride_z = (dim_x + 1) * (dim_y + 1)

        def grid_index(x: int, y: int, z: int) -> int:
            return z * stride_z + y * stride_y + x * stride_x

        left = []
        right = []
        for z in range(dim_z + 1):
            for y in range(dim_y + 1):
                left.append(grid_index(0, y, z))
                right.append(grid_index(dim_x, y, z))
        return left, right

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            angle = min(self.sim_time, self.twist_end_time) * self.twist_angular_velocity

            wp.launch(
                set_end_state,
                dim=self.twist_particle_indices.shape[0],
                inputs=[
                    self.twist_particle_indices,
                    self.twist_centers,
                    self.twist_axes,
                    self.twist_offsets,
                    angle,
                ],
                outputs=[self.state_0.particle_q, self.state_0.particle_qd],
            )
            wp.launch(
                set_end_state,
                dim=self.twist_particle_indices.shape[0],
                inputs=[
                    self.twist_particle_indices,
                    self.twist_centers,
                    self.twist_axes,
                    self.twist_offsets,
                    angle,
                ],
                outputs=[self.state_1.particle_q, self.state_1.particle_qd],
            )

            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            wp.launch(
                set_end_state,
                dim=self.twist_particle_indices.shape[0],
                inputs=[
                    self.twist_particle_indices,
                    self.twist_centers,
                    self.twist_axes,
                    self.twist_offsets,
                    angle,
                ],
                outputs=[self.state_1.particle_q, self.state_1.particle_qd],
            )
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def step(self):
        self.simulate()

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        p_lower = wp.vec3(-1.5, -1.0, -0.1)
        p_upper = wp.vec3(1.5, 1.0, 2.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles remain within a reasonable volume",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities remain bounded",
            lambda _q, qd: max(abs(qd)) < 5.0,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--ipc-backend",
            type=str,
            default=None,
            help="Override the uipc engine backend passed to SolverIPC (for example: cpu or cuda).",
        )
        parser.set_defaults(num_frames=360)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Twist a self-contacting FEM cloth using the standalone MuJoCo/VBD solver.

This intentionally matches ``cloth_twist`` so the two examples can be
compared directly. Run with::

    uv run --extra examples -m newton.examples mujoco_vbd_cloth_twist
"""

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
from newton.solvers import SolverMuJoCoVBD


@wp.kernel
def initialize_rotation(
    vertex_indices_to_rot: wp.array[wp.int32],
    pos: wp.array[wp.vec3],
    rot_centers: wp.array[wp.vec3],
    rot_axes: wp.array[wp.vec3],
    t: wp.array[float],
    roots: wp.array[wp.vec3],
    roots_to_ps: wp.array[wp.vec3],
):
    """Initialize the rotation decomposition for each driven cloth vertex."""
    tid = wp.tid()
    vertex = vertex_indices_to_rot[tid]
    offset = pos[vertex] - rot_centers[tid]
    root = wp.dot(offset, rot_axes[tid]) * rot_axes[tid]
    roots[tid] = root
    roots_to_ps[tid] = pos[vertex] - root
    if tid == 0:
        t[0] = 0.0


@wp.kernel
def apply_rotation(
    vertex_indices_to_rot: wp.array[wp.int32],
    rot_axes: wp.array[wp.vec3],
    roots: wp.array[wp.vec3],
    roots_to_ps: wp.array[wp.vec3],
    t: wp.array[float],
    angular_velocity: float,
    dt: float,
    end_time: float,
    pos_0: wp.array[wp.vec3],
    pos_1: wp.array[wp.vec3],
):
    """Apply the prescribed boundary rotation to one cloth vertex."""
    current_time = t[0]
    if current_time > end_time:
        return

    tid = wp.tid()
    vertex = vertex_indices_to_rot[tid]
    axis = rot_axes[tid]
    ux = axis[0]
    uy = axis[1]
    uz = axis[2]
    theta = current_time * angular_velocity
    cosine = wp.cos(theta)
    sine = wp.sin(theta)
    rotation = wp.mat33(
        cosine + ux * ux * (1.0 - cosine),
        ux * uy * (1.0 - cosine) - uz * sine,
        ux * uz * (1.0 - cosine) + uy * sine,
        uy * ux * (1.0 - cosine) + uz * sine,
        cosine + uy * uy * (1.0 - cosine),
        uy * uz * (1.0 - cosine) - ux * sine,
        uz * ux * (1.0 - cosine) - uy * sine,
        uz * uy * (1.0 - cosine) + ux * sine,
        cosine + uz * uz * (1.0 - cosine),
    )
    rotated = roots[tid] + rotation * roots_to_ps[tid]
    pos_0[vertex] = rotated
    pos_1[vertex] = rotated
    if tid == 0:
        t[0] = current_time + dt


class Example:
    """Twist a self-contacting cloth with :class:`SolverMuJoCoVBD`."""

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.iterations = 4
        self.bvh_rebuild_frames = 10
        self.rot_angular_velocity = math.pi / 3
        self.rot_end_time = 10
        self.viewer = viewer

        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "square_cloth.usd"))
        usd_prim = usd_stage.GetPrimAtPath("/root/cloth/cloth")
        cloth_mesh = newton.usd.get_mesh(usd_prim)
        mesh_points = cloth_mesh.vertices
        mesh_indices = cloth_mesh.indices

        vertices = [wp.vec3(vertex) for vertex in mesh_points]
        self.faces = mesh_indices.reshape(-1, 3)

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        SolverMuJoCoVBD.register_custom_attributes(builder)
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 2),
            scale=0.01,
            vertices=vertices,
            indices=mesh_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=0.2,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=2.0e-4,
            edge_ke=1.0e-3,
            edge_kd=1.0e-2,
        )
        builder.color()
        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e-1
        self.model.soft_contact_mu = 0.2

        cloth_size = 50
        left_side = [cloth_size - 1 + index * cloth_size for index in range(cloth_size)]
        right_side = [index * cloth_size for index in range(cloth_size)]
        rot_point_indices = left_side + right_side

        flags = self.model.particle_flags.numpy()
        for fixed_vertex_id in rot_point_indices:
            flags[fixed_vertex_id] &= ~ParticleFlags.ACTIVE
        self.model.particle_flags = wp.array(flags)

        self.solver = SolverMuJoCoVBD(
            self.model,
            joint_mode="dynamic",
            coupling_mode="auto",
            vbd_options={
                "iterations": self.iterations,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": 0.002,
                "particle_self_contact_margin": 0.0035,
            },
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.solver.contacts

        rot_axes = [[0, 1, 0]] * len(right_side) + [[0, -1, 0]] * len(left_side)
        self.rot_point_indices = wp.array(rot_point_indices, dtype=int)
        self.t = wp.zeros(1, dtype=float)
        self.rot_centers = wp.zeros(len(rot_point_indices), dtype=wp.vec3)
        self.rot_axes = wp.array(rot_axes, dtype=wp.vec3)
        self.roots = wp.zeros_like(self.rot_centers)
        self.roots_to_ps = wp.zeros_like(self.rot_centers)

        wp.launch(
            kernel=initialize_rotation,
            dim=self.rot_point_indices.shape[0],
            inputs=[
                self.rot_point_indices,
                self.state_0.particle_q,
                self.rot_centers,
                self.rot_axes,
                self.t,
            ],
            outputs=[self.roots, self.roots_to_ps],
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(2.25, 0.0, 0.0), 0.0, -180.0)
        self.capture()

    def capture(self):
        """Capture one complete display-frame simulation update."""
        self.graph = None
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        """Advance all cloth substeps."""
        self.solver.rebuild_bvh(self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            wp.launch(
                kernel=apply_rotation,
                dim=self.rot_point_indices.shape[0],
                inputs=[
                    self.rot_point_indices,
                    self.rot_axes,
                    self.roots,
                    self.roots_to_ps,
                    self.t,
                    self.rot_angular_velocity,
                    self.sim_dt,
                    self.rot_end_time,
                ],
                outputs=[self.state_0.particle_q, self.state_1.particle_q],
            )

            # The private solver owns its sparse particle-shape pipeline. This scene has no
            # shapes, so only cloth self-contact is active.
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        """Advance one display frame."""
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        """Render the current cloth state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify a finite pure-cloth MuJoCo/VBD solve."""
        assert self.solver.features.backend.value == "pure_vbd_soft"
        assert not self.solver.features.mujoco_solve_enabled
        assert not self.solver.features.rigid_solve_enabled
        assert not self.solver.features.tetrahedron_solve_enabled

        p_lower = wp.vec3(-0.6, -0.9, -0.6)
        p_upper = wp.vec3(0.6, 0.9, 0.6)
        newton.examples.test_particle_state(
            self.state_0,
            "particles are within a reasonable volume",
            lambda q, qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )
        newton.examples.test_particle_state(
            self.state_0,
            "particle velocities are within a reasonable range",
            lambda q, qd: max(abs(qd)) < 1.5,
        )

    @staticmethod
    def create_parser():
        """Create command-line options for the cloth-twist scene."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=300)
        return parser


def main():
    """Run the cloth-twist example."""
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

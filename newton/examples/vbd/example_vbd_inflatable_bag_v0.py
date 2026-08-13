# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop a rigid cube onto a Blender-authored closed pneumatic chip bag.

Run with ``python -m newton.examples vbd_inflatable_bag_v0``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import warp as wp

import newton
import newton.examples


@dataclass(frozen=True)
class _ChipBagMesh:
    """Simulation triangles and rendering edges for the authored bag."""

    vertices: list[list[float]]
    indices: list[int]
    edges: list[tuple[int, int]]


@wp.kernel
def _gather_edges(
    positions: wp.array[wp.vec3],
    edge_indices: wp.array[int],
    lift: float,
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    edge = wp.tid()
    offset = wp.vec3(0.0, 0.0, lift)
    starts[edge] = positions[edge_indices[2 * edge]] + offset
    ends[edge] = positions[edge_indices[2 * edge + 1]] + offset


def _load_chip_bag_mesh() -> _ChipBagMesh:
    """Load the closed Blender-authored bag mesh."""
    path = newton.examples.get_asset("newton_chip_bag_sealed_cylinder.obj")
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, encoding="utf-8") as obj_file:
        for line in obj_file:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                vertices.append([float(value) for value in fields[1:4]])
            elif fields[0] == "f":
                faces.append([int(value.split("/", maxsplit=1)[0]) - 1 for value in fields[1:]])

    if not vertices or not faces:
        raise ValueError(f"{path} does not contain a mesh.")

    triangles = [
        (face[0], face[vertex_index], face[vertex_index + 1])
        for face in faces
        for vertex_index in range(1, len(face) - 1)
    ]
    if not triangles:
        raise ValueError(f"{path} does not contain any triangle faces.")

    edge_counts = Counter(
        tuple(sorted((vertex0, vertex1)))
        for triangle in triangles
        for vertex0, vertex1 in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0]))
    )
    if any(count != 2 for count in edge_counts.values()):
        raise ValueError(f"{path} must be a closed two-manifold surface after triangulation.")

    return _ChipBagMesh(
        vertices=vertices,
        indices=[vertex for triangle in triangles for vertex in triangle],
        edges=sorted(edge_counts),
    )


# The source asset is an upright cylindrical shell with compressed axial ends.
# Rotate its broad membrane faces horizontal for the falling-block contact case.
_BAG_ROTATION = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.pi * 0.5)

PARAMS = {
    "fps": 60,
    "sim_substeps": 4,
    "solver_iterations": 20,
    "gravity": (0.0, 0.0, -9.81),
    "bag_center_z": 0.065,
    "bag_density": 0.12,
    "bag_reference_absolute_pressure": 108_000.0,
    "particle_radius": 0.004,
    "block_half_extent": 0.015,
    "block_initial_z": 0.48,
    "block_density": 1.25e6,
}


class Example:
    """Demonstrate a sealed cylindrical bag deforming under a falling cube."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / PARAMS["fps"]
        self.sim_dt = self.frame_dt / PARAMS["sim_substeps"]
        self.sim_time = 0.0

        builder = newton.ModelBuilder(gravity=PARAMS["gravity"])
        bag_mesh = _load_chip_bag_mesh()
        particle_start = builder.particle_count
        self.cavity = newton.solvers.add_inflatable_mesh(
            builder,
            pos=wp.vec3(0.0, 0.0, PARAMS["bag_center_z"]),
            rot=_BAG_ROTATION,
            scale=1.0,
            vel=wp.vec3(),
            vertices=bag_mesh.vertices,
            indices=bag_mesh.indices,
            density=PARAMS["bag_density"],
            tri_ke=6.0e4,
            tri_ka=6.0e4,
            tri_kd=80.0,
            edge_ke=20.0,
            edge_kd=0.5,
            particle_radius=PARAMS["particle_radius"],
            validate_mesh=True,
            label="sealed cylindrical chip bag",
            config=newton.solvers.PneumaticConfig(
                mode=newton.solvers.PneumaticMode.ISOTHERMAL,
                reference_absolute_pressure=PARAMS["bag_reference_absolute_pressure"],
                ambient_pressure=101_325.0,
                bulk_damping=50.0,
                max_absolute_pressure=200_000.0,
            ),
        )
        bag_triangle_indices = np.asarray(bag_mesh.indices, dtype=np.int32) + particle_start
        bag_edges = np.asarray(bag_mesh.edges, dtype=np.int32) + particle_start

        contact_cfg = newton.ModelBuilder.ShapeConfig(density=PARAMS["block_density"], ke=4.0e4, kd=100.0, mu=0.7)
        self.block_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, PARAMS["block_initial_z"]), wp.quat_identity()),
            label="falling_block",
        )
        builder.add_shape_box(
            self.block_body,
            hx=PARAMS["block_half_extent"],
            hy=PARAMS["block_half_extent"],
            hz=PARAMS["block_half_extent"],
            cfg=contact_cfg,
            color=(0.8, 0.3, 0.2),
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.025), wp.quat_identity()),
            hx=0.7,
            hy=0.7,
            hz=0.025,
            cfg=contact_cfg,
            color=(0.35, 0.35, 0.35),
        )

        builder.color()
        self.model = builder.finalize()
        self.bag_triangle_indices = wp.array(bag_triangle_indices, dtype=int, device=self.model.device)
        self.bag_edges = wp.array(bag_edges.reshape(-1), dtype=int, device=self.model.device)
        self.bag_edge_starts = wp.empty(len(bag_edges), dtype=wp.vec3, device=self.model.device)
        self.bag_edge_ends = wp.empty(len(bag_edges), dtype=wp.vec3, device=self.model.device)
        self.model.soft_contact_ke = 4.0e4
        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_mu = 0.7
        self.solver = newton.solvers.SolverMJVBDV2(
            self.model,
            vbd_options={
                "iterations": PARAMS["solver_iterations"],
                "rigid_body_particle_contact_buffer_size": 8192,
            },
            collision_options={
                "soft_contact_margin": 0.01,
                "enable_rigid_soft_full_surface_contact": True,
            },
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = False
            self.viewer.renderer.draw_edges = False
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.0, -0.42, 0.40), -37.5, 90.0)

    def step(self):
        """Advance the falling cube, contact solve, and deformable bag."""
        for _ in range(PARAMS["sim_substeps"]):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            if hasattr(self.viewer, "apply_forces"):
                self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def render(self):
        """Render the current pneumatic state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/bag/surface",
            self.state_0.particle_q,
            self.bag_triangle_indices,
            backface_culling=True,
            color=(0.86, 0.68, 0.34),
        )
        wp.launch(
            _gather_edges,
            dim=len(self.bag_edge_starts),
            inputs=[self.state_0.particle_q, self.bag_edges, 1.0e-4],
            outputs=[self.bag_edge_starts, self.bag_edge_ends],
            device=self.model.device,
        )
        self.viewer.log_lines(
            "/bag/grid",
            self.bag_edge_starts,
            self.bag_edge_ends,
            (0.08, 0.06, 0.02),
        )
        self.viewer.end_frame()

    def test_final(self):
        """Verify the cube deforms a finite sealed bag."""
        volume = self.state_0.pneumatic.volume.numpy()[self.cavity.cavity_index]
        positions = self.state_0.particle_q.numpy()
        block_z = self.state_0.body_q.numpy()[self.block_body, 2]
        assert np.isfinite(positions).all()
        assert block_z < PARAMS["block_initial_z"]
        assert volume > self.cavity.rest_volume * 0.2


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)

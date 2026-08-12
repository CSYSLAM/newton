# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop a rigid cube onto a closed pneumatic bag.

Run with ``python -m newton.examples vbd_inflatable_bag``.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples


def _pillow_surface(
    width: float,
    depth: float,
    height: float,
    bulge: float,
    resolution: int,
) -> tuple[list[list[float]], list[int]]:
    """Return a closed, two-layer pillow mesh centered at the origin.

    The top and bottom faces have independent grid interiors and share their
    perimeter through narrow side panels. This gives the shell enough degrees
    of freedom to form a local contact dimple instead of moving as a box.
    """
    if resolution < 2:
        raise ValueError("resolution must be at least 2.")

    node_count = resolution + 1
    layer_size = node_count * node_count
    vertices: list[list[float]] = []
    for layer in range(2):
        for j in range(node_count):
            y = depth * (j / resolution - 0.5)
            for i in range(node_count):
                x = width * (i / resolution - 0.5)
                profile = np.sin(np.pi * i / resolution) * np.sin(np.pi * j / resolution)
                z = (height * 0.5 + bulge * profile) * (-1.0 if layer == 0 else 1.0)
                vertices.append([x, y, z])

    def vertex(layer: int, i: int, j: int) -> int:
        return layer * layer_size + j * node_count + i

    faces: list[int] = []
    for j in range(resolution):
        for i in range(resolution):
            bottom_a = vertex(0, i, j)
            bottom_b = vertex(0, i + 1, j)
            bottom_c = vertex(0, i, j + 1)
            bottom_d = vertex(0, i + 1, j + 1)
            top_a = vertex(1, i, j)
            top_b = vertex(1, i + 1, j)
            top_c = vertex(1, i, j + 1)
            top_d = vertex(1, i + 1, j + 1)

            # Bottom faces point down; top faces point up.
            faces.extend((bottom_a, bottom_c, bottom_b, bottom_b, bottom_c, bottom_d))
            faces.extend((top_a, top_b, top_c, top_b, top_d, top_c))

    perimeter = (
        [(i, 0) for i in range(resolution)]
        + [(resolution, j) for j in range(resolution)]
        + [(i, resolution) for i in range(resolution, 0, -1)]
        + [(0, j) for j in range(resolution, 0, -1)]
    )
    for (i0, j0), (i1, j1) in zip(perimeter, perimeter[1:] + perimeter[:1], strict=True):
        top_a = vertex(1, i0, j0)
        top_b = vertex(1, i1, j1)
        bottom_a = vertex(0, i0, j0)
        bottom_b = vertex(0, i1, j1)
        faces.extend((top_a, bottom_a, top_b, top_b, bottom_a, bottom_b))

    return vertices, faces


PARAMS = {
    "fps": 60,
    "sim_substeps": 4,
    "solver_iterations": 20,
    "gravity": (0.0, 0.0, -9.81),
    "bag_width": 0.34,
    "bag_depth": 0.24,
    "bag_height": 0.020,
    "bag_bulge": 0.070,
    "bag_resolution": 12,
    "bag_center_z": 0.095,
    "bag_density": 0.12,
    "bag_reference_absolute_pressure": 101_400.0,
    "particle_radius": 0.004,
    "block_half_extent": 0.015,
    "block_initial_z": 0.48,
    "block_density": 1.0e6,
}


class Example:
    """Demonstrate a closed gas cavity deforming under a falling rigid block."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / PARAMS["fps"]
        self.sim_dt = self.frame_dt / PARAMS["sim_substeps"]
        self.sim_time = 0.0

        builder = newton.ModelBuilder(gravity=PARAMS["gravity"])
        vertices, indices = _pillow_surface(
            PARAMS["bag_width"],
            PARAMS["bag_depth"],
            PARAMS["bag_height"],
            PARAMS["bag_bulge"],
            PARAMS["bag_resolution"],
        )
        self.cavity = newton.solvers.add_inflatable_mesh(
            builder,
            pos=wp.vec3(0.0, 0.0, PARAMS["bag_center_z"]),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=vertices,
            indices=indices,
            density=PARAMS["bag_density"],
            tri_ke=6.0e4,
            tri_ka=6.0e4,
            tri_kd=80.0,
            edge_ke=20.0,
            edge_kd=0.5,
            particle_radius=PARAMS["particle_radius"],
            config=newton.solvers.PneumaticConfig(
                mode=newton.solvers.PneumaticMode.ISOTHERMAL,
                reference_absolute_pressure=PARAMS["bag_reference_absolute_pressure"],
                ambient_pressure=101_325.0,
                bulk_damping=50.0,
                max_absolute_pressure=200_000.0,
            ),
        )

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
        self.model.soft_contact_ke = 4.0e4
        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_mu = 0.7
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=PARAMS["solver_iterations"],
            rigid_body_particle_contact_buffer_size=2048,
        )
        self.pipeline = newton.CollisionPipeline(
            self.model,
            soft_contact_margin=0.01,
            enable_rigid_soft_full_surface_contact=True,
        )
        self.contacts = self.pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "renderer"):
            self.viewer.renderer.draw_wireframe = False
            self.viewer.renderer.draw_edges = True
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(0.0, -0.42, 0.40), -37.5, 90.0)

    def step(self):
        """Advance the falling cube, contact solve, and deformable bag."""
        for _ in range(PARAMS["sim_substeps"]):
            self.state_0.clear_forces()
            self.state_1.clear_forces()
            if hasattr(self.viewer, "apply_forces"):
                self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def render(self):
        """Render the current pneumatic state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify the block makes a local dimple in a finite closed shell."""
        volume = self.state_0.pneumatic.volume.numpy()[self.cavity.cavity_index]
        positions = self.state_0.particle_q.numpy()
        block_z = self.state_0.body_q.numpy()[self.block_body, 2]
        node_count = PARAMS["bag_resolution"] + 1
        top_start = node_count * node_count
        top = positions[top_start : 2 * top_start].reshape(node_count, node_count, 3)
        center = top[node_count // 2, node_count // 2, 2]
        ring_offset = 3
        surrounding_height = np.mean(
            (
                top[node_count // 2 - ring_offset, node_count // 2, 2],
                top[node_count // 2 + ring_offset, node_count // 2, 2],
                top[node_count // 2, node_count // 2 - ring_offset, 2],
                top[node_count // 2, node_count // 2 + ring_offset, 2],
            )
        )
        assert np.isfinite(positions).all()
        assert block_z < PARAMS["block_initial_z"]
        assert volume > self.cavity.rest_volume * 0.2
        assert surrounding_height - center > 0.002


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)

"""Reconstruction of Fig. 8: an Armadillo fed through two gear crushers.

The Armadillo, procedural gear geometry, dimensions, and physical parameters
come from the first author's public Newton fork.  Newton 1.3 applies public
Planar-DAT to the soft body's triangle-surface self contact; gear/soft contact
still uses Newton's particle-body contact, and tet inversion is monitored as a
diagnostic because the paper's private inversion DAT was not merged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp

from paper_scenes import (
    create_gear_cylinder_mesh,
    fetch_armadillo_asset,
    load_vtk_tet_mesh,
    signed_tetrahedron_volumes,
)


@wp.kernel
def drive_counter_rotating_gears(
    time: wp.array[float],
    dt: float,
    angular_speed: float,
    separation: float,
    left_body: int,
    right_body: int,
    body_q_previous: wp.array[wp.transform],
    body_q_current: wp.array[wp.transform],
):
    """Write previous/current kinematic poses for external VBD integration."""

    t0 = time[0]
    t1 = t0 + dt
    axis = wp.vec3(1.0, 0.0, 0.0)
    left_position = wp.vec3(0.0, -0.5 * separation, 0.0)
    right_position = wp.vec3(0.0, 0.5 * separation, 0.0)

    body_q_previous[left_body] = wp.transform(
        left_position,
        wp.quat_from_axis_angle(axis, -angular_speed * t0),
    )
    body_q_current[left_body] = wp.transform(
        left_position,
        wp.quat_from_axis_angle(axis, -angular_speed * t1),
    )
    body_q_previous[right_body] = wp.transform(
        right_position,
        wp.quat_from_axis_angle(axis, angular_speed * t0),
    )
    body_q_current[right_body] = wp.transform(
        right_position,
        wp.quat_from_axis_angle(axis, angular_speed * t1),
    )
    time[0] = t1


class Example:
    """Author-scene port for Newton 1.3's VBD solver and viewer API."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.sim_time = 0.0
        self.frame_count = 0
        self._reported_overflow = False
        self._reported_inversion = False

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps  # paper: 1/600 s
        self.iterations = 10

        self.roller_inner_radius = 0.36
        self.roller_outer_radius = 0.40
        self.roller_length = 1.60
        self.roller_teeth = 16
        self.roller_gap = 0.08
        self.roller_separation = 2.0 * self.roller_outer_radius + self.roller_gap

        asset_path = Path(args.asset_path) if args.asset_path else fetch_armadillo_asset()
        armadillo_vertices, armadillo_tets = load_vtk_tet_mesh(asset_path)
        print(
            f"Gear crusher Armadillo: {len(armadillo_vertices):,} vertices, "
            f"{len(armadillo_tets):,} tetrahedra; asset={asset_path}"
        )
        self.tetrahedra = armadillo_tets

        builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.81))
        builder.add_ground_plane(height=-1.5)

        gear_vertices, gear_faces = create_gear_cylinder_mesh(
            inner_radius=self.roller_inner_radius,
            outer_radius=self.roller_outer_radius,
            length=self.roller_length,
            num_teeth=self.roller_teeth,
        )
        gear_mesh = newton.Mesh(gear_vertices, gear_faces.reshape(-1))
        gear_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            ke=1.0e6,
            kd=1.0e-7,
            mu=0.2,
        )
        self.left_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, -0.5 * self.roller_separation, 0.0), wp.quat_identity()),
            label="left_gear",
        )
        builder.add_shape_mesh(
            self.left_body,
            mesh=gear_mesh,
            cfg=gear_cfg,
            color=(0.50, 0.53, 0.58),
            label="left_gear_mesh",
        )
        self.right_body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.5 * self.roller_separation, 0.0), wp.quat_identity()),
            label="right_gear",
        )
        builder.add_shape_mesh(
            self.right_body,
            mesh=gear_mesh,
            cfg=gear_cfg,
            color=(0.50, 0.53, 0.58),
            label="right_gear_mesh",
        )

        builder.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 1.0),
            rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=armadillo_vertices,
            indices=armadillo_tets.reshape(-1),
            density=1000.0,
            k_mu=1.0e5,
            k_lambda=1.0e6,
            k_damp=1.0e-7,
            particle_radius=0.005,
            label="paper_armadillo",
        )
        builder.color(include_bending=False)
        self.model = builder.finalize()
        self.model.soft_contact_ke = 1.0e6
        self.model.soft_contact_kd = 1.0e-7
        self.model.soft_contact_mu = 0.2

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=self.iterations,
            integrate_with_external_rigid_solver=True,
            particle_enable_self_contact=True,
            particle_self_contact_radius=0.005,
            particle_self_contact_margin=0.0075,
            particle_conservative_bound_relaxation=0.85,
            particle_collision_detection_interval=5,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.02,
            particle_enable_tile_solve=True,
            particle_vertex_contact_buffer_size=args.vertex_buffer,
            particle_edge_contact_buffer_size=args.edge_buffer,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=0.01,
            soft_contact_max=64 * 1024,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.sim_time_device = wp.zeros(1, dtype=float, device=self.model.device)

        initial_positions = self.state_0.particle_q.numpy()
        initial_volumes = signed_tetrahedron_volumes(initial_positions, self.tetrahedra)
        if np.any(initial_volumes <= 0.0):
            raise RuntimeError("the Armadillo asset contains non-positive initial tetrahedra")
        self.initial_min_volume = float(np.min(initial_volumes))

        self.viewer.set_model(self.model)
        # Draw the deformable surface separately to match the paper's green material.
        self.viewer.show_triangles = False
        self.surface_indices = self.model.tri_indices.flatten()
        self.viewer.set_camera(pos=wp.vec3(2.45, -2.20, 1.20), pitch=-16.0, yaw=138.0)
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 46.0

        self.capture()

    def capture(self):
        if self.args.cuda_graph and self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.launch(
                drive_counter_rotating_gears,
                dim=1,
                inputs=[
                    self.sim_time_device,
                    self.sim_dt,
                    self.args.rotation_speed,
                    self.roller_separation,
                    self.left_body,
                    self.right_body,
                    self.state_0.body_q,
                    self.state_1.body_q,
                ],
                device=self.model.device,
            )
            self.solver.rebuild_bvh(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _oriented_min_volume(self) -> float:
        positions = self.state_0.particle_q.numpy()
        return float(np.min(signed_tetrahedron_volumes(positions, self.tetrahedra)))

    def _collision_overflow(self) -> np.ndarray:
        return self.solver.trimesh_collision_detector.resize_flags.numpy().copy()

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame_count += 1

        if self.frame_count == 1 or self.frame_count % 60 == 0:
            flags = self._collision_overflow()
            if np.any(flags) and not self._reported_overflow:
                self._reported_overflow = True
                print(
                    "WARNING: Armadillo surface self-contact buffers overflowed "
                    f"(VT={flags[0]}, EE={flags[2]})."
                )
            minimum_volume = self._oriented_min_volume()
            if minimum_volume <= 0.0 and not self._reported_inversion:
                self._reported_inversion = True
                print(
                    "WARNING: an Armadillo tetrahedron inverted. Public Newton 1.3 lacks the "
                    "paper's tetrahedral inversion-DAT kernel."
                )

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/paper/armadillo",
            self.state_0.particle_q,
            self.surface_indices,
            backface_culling=False,
            color=(0.20, 0.78, 0.42),
            roughness=0.72,
        )
        self.viewer.end_frame()

    def test_final(self):
        positions = self.state_0.particle_q.numpy()
        assert np.all(np.isfinite(positions)), "Armadillo contains non-finite positions"
        assert np.linalg.norm(np.ptp(positions, axis=0)) < 8.0, "Armadillo bounding box exploded"
        if self.args.require_no_inversion:
            assert self._oriented_min_volume() > 0.0, "an Armadillo tetrahedron inverted"
        overflow = self._collision_overflow()
        assert not np.any(overflow), f"self-contact buffer overflow flags: {overflow.tolist()}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--asset-path",
            type=Path,
            help=(
                "Use an existing Armadillo VTK; otherwise fetch the authors' "
                "checksum-pinned research asset."
            ),
        )
        parser.add_argument(
            "--rotation-speed",
            type=float,
            default=1.0,
            help="Counter-rotation speed in rad/s.",
        )
        parser.add_argument("--vertex-buffer", type=int, default=32, help="VT slots per surface vertex.")
        parser.add_argument("--edge-buffer", type=int, default=64, help="EE slots per surface edge.")
        parser.add_argument(
            "--require-no-inversion",
            action="store_true",
            help=(
                "Make --test fail if any tet inverts. This is expected to fail in the full crusher "
                "run because public Newton 1.3 lacks the paper's inversion-DAT kernel."
            ),
        )
        parser.add_argument(
            "--cuda-graph",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture one rendered frame as a CUDA graph.",
        )
        return parser


def main() -> None:
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

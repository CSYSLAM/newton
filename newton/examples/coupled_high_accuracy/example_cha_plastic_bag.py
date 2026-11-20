"""Simulate the supplied plastic bag as an indexed Style3D cloth mesh.

Run from the repository root:
    uv run python -m newton.examples cha_plastic_bag
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.usd
from newton import ParticleFlags
from newton.solvers import coupled_high_accuracy as cha
from newton.solvers import style3d

ASSET = Path(__file__).resolve().parent / "assets" / "plastic_bag.usd"
PRIM_PATH = "/Root/plastic_bag/Root_Garment"


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.frame_dt = 1.0 / 60.0
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.graph = None

        if not ASSET.is_file():
            raise FileNotFoundError(
                f"Plastic bag asset is missing: {ASSET}. "
                "To create a local replacement from the bundled bag-mesh generator, run: "
                ".venv/bin/python scripts/generate_plastic_bag_asset.py"
            )
        stage = Usd.Stage.Open(str(ASSET))
        prim = stage.GetPrimAtPath(PRIM_PATH)
        if not prim or not prim.IsA(UsdGeom.Mesh):
            raise ValueError(f"Missing mesh prim {PRIM_PATH} in {ASSET}")

        # Keep the authored mesh vertices and face indices. get_mesh expands face-varying UVs,
        # so read the raw primvar below to preserve sewn panel seams for Style3D.
        mesh = newton.usd.get_mesh(prim, load_uvs=False)
        points = np.asarray(mesh.vertices, dtype=np.float64)
        indices = np.asarray(mesh.indices, dtype=np.int32).reshape(-1, 3)
        uv_primvar = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        if not uv_primvar or not uv_primvar.HasValue() or not uv_primvar.IsIndexed():
            raise ValueError("The plastic bag must contain indexed face-varying st UVs")
        panel_verts = np.asarray(uv_primvar.Get(), dtype=np.float64)
        panel_indices = np.asarray(uv_primvar.GetIndices(), dtype=np.int32).reshape(-1, 3)
        if panel_indices.shape != indices.shape:
            raise ValueError("The st index count must match the mesh face corners")

        # The USD texture coordinates are clockwise. Reflect V so Style3D receives positive panel areas.
        panel_verts[:, 1] *= -1.0
        corners = panel_verts[panel_indices]
        panel_edge_a = corners[:, 1] - corners[:, 0]
        panel_edge_b = corners[:, 2] - corners[:, 0]
        panel_area = (panel_edge_a[:, 0] * panel_edge_b[:, 1] - panel_edge_a[:, 1] * panel_edge_b[:, 0]) * 0.5
        if np.any(panel_area <= 0.0):
            raise ValueError("The plastic bag st panels contain degenerate or inverted triangles")

        # Center the bag over the default ground and lift its lowest point above the floor.
        bounds = np.stack((points.min(axis=0), points.max(axis=0)))
        points[:, :2] -= (bounds[0, :2] + bounds[1, :2]) * 0.5
        points[:, 2] -= bounds[0, 2] - 0.25
        self.points = points.astype(np.float32)
        self.indices = indices
        self.panel_verts = panel_verts.astype(np.float32)
        self.panel_indices = panel_indices

        # Match total panel area to the authored 3D surface because the USD has no UV unit metadata.
        surface_edges = points[indices[:, 1:]] - points[indices[:, :1]]
        surface_area = np.linalg.norm(np.cross(surface_edges[:, 0], surface_edges[:, 1]), axis=1).sum() * 0.5
        uv_scale = float(np.sqrt(surface_area / panel_area.sum()))
        self.panel_verts *= uv_scale

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverStyle3D.register_custom_attributes(builder)
        style3d.add_cloth_mesh(
            builder,
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            vertices=self.points.tolist(),
            indices=self.indices.ravel().tolist(),
            panel_verts=self.panel_verts.tolist(),
            panel_indices=self.panel_indices.ravel().tolist(),
            density=args.density,
            particle_radius=3.0e-3,
            tri_aniso_ke=wp.vec3(args.stretch_ke, args.stretch_ke, args.shear_ke),
            tri_ka=args.area_ke,
            tri_kd=args.tri_kd,
            edge_aniso_ke=wp.vec3(args.bend_ke),
            edge_kd=args.edge_kd,
        )
        builder.add_ground_plane()
        self.model = builder.finalize()

        # Pin the highest vertices of the negative-Y handle to make the standalone bag hang in place.
        held_side = self.points[:, 1] < 0.0
        crown = np.flatnonzero(held_side & (self.points[:, 2] > self.points[held_side, 2].max() - 0.01))
        flags = self.model.particle_flags.numpy()
        flags[crown] &= ~int(ParticleFlags.ACTIVE)
        self.model.particle_flags = wp.array(flags, dtype=wp.int32, device=self.model.device)
        self.fixed_points = crown

        self.model.soft_contact_ke = args.contact_ke
        self.model.soft_contact_kd = args.contact_kd
        self.model.soft_contact_mu = 0.25
        self.model.set_gravity((0.0, 0.0, -9.81))
        if args.solver == "synreal":
            self.solver = cha.SolverCoupledHighAccuracy(self.model)
            self.solver.config.nonlinear_iterations = args.iterations
            self.solver.config.linear_iterations = args.linear_iterations
            self.solver.config.enable_self_collision = True
        else:
            self.solver = newton.solvers.SolverStyle3D(self.model, iterations=args.iterations)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.viewer.set_model(self.model)
        self.viewer.show_static = False
        self.viewer.set_camera(wp.vec3(0.9, -1.2, 0.85), -12.0, 125.0)
        self.capture()

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        positions = self.state_0.particle_q.numpy()
        if not np.isfinite(positions).all():
            raise AssertionError("plastic bag particles became non-finite")


def create_parser():
    parser = newton.examples.create_parser()
    parser.add_argument("--solver", choices=("synreal", "style3d"), default="synreal")
    parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per rendered frame.")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--linear-iterations", type=int, default=80)
    parser.add_argument("--density", type=float, default=0.25, help="Bag surface density in kg/m^2.")
    parser.add_argument("--stretch-ke", type=float, default=1e4)
    parser.add_argument("--shear-ke", type=float, default=1e3)
    parser.add_argument("--area-ke", type=float, default=100.0)
    parser.add_argument("--bend-ke", type=float, default=1.0e-8)
    parser.add_argument("--tri-kd", type=float, default=1.0e-4)
    parser.add_argument("--edge-kd", type=float, default=1.0e-5)
    parser.add_argument("--contact-ke", type=float, default=2.0e3)
    parser.add_argument("--contact-kd", type=float, default=1.0e-2)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    wp.config.kernel_cache_dir = str(Path(__file__).resolve().parent / "build" / "warp-plastic-bag")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

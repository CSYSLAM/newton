# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Slice an MPM Lite burger patty in the WAIC W1 kitchen scene.

The scene layout and knife trajectory follow
``example_mpm_w1_burger_slice_waic_kitchen``.  MPM Lite provides grid-node
boundaries rather than triangle-mesh colliders, so the pan and the finite knife
blade are represented by sticky node bands.  The W1 and kitchen remain visual
scene geometry and are deliberately not passed to a Newton MPM solver.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp
from pxr import Gf, Usd, UsdGeom

import newton
import newton.examples
import newton.ik as ik
from newton.solvers import SolverMPMLite

WAIC_ROBOT_BASE_POS = wp.vec3(5.156154155731201, 0.6696404814720154, -0.0037720240652561188)
WAIC_WORKTOP_Z = 0.9000003337860107
PAN_CENTER = np.asarray((5.827057838439941, 0.331002801656723, 0.8562719821929932), dtype=np.float32)
PAN_RADIUS = 0.1125
PAN_DISK_HALF_HEIGHT = 0.0125
PAN_TOP_Z = float(PAN_CENTER[2]) + PAN_DISK_HALF_HEIGHT

MEAT_LENGTH = 0.14
MEAT_WIDTH = 0.065
MEAT_HEIGHT = 0.045
BLADE_HX = 0.003
BLADE_HY = 0.10
BLADE_HZ = 0.06
KNIFE_REST_CENTER = np.asarray((5.63, 0.58, WAIC_WORKTOP_Z + BLADE_HZ + 0.005), dtype=np.float32)
RIGHT_WRIST_QUAT = wp.quat(-0.0711, 0.7045, 0.7017, -0.0789)
TCP_OFFSET = wp.vec3(-0.18, 0.0, 0.0)


@wp.kernel
def _set_body_transform(body_q: wp.array[wp.transform], body_index: int, position: wp.vec3):
    body_q[body_index] = wp.transform(position, wp.quat_identity())


@wp.kernel
def _copy_ik_to_joint_q(ik_joint_q: wp.array2d[wp.float32], joint_q: wp.array[wp.float32]):
    joint_q[wp.tid()] = ik_joint_q[0, wp.tid()]


class Example:
    """Use MPM Lite grid boundaries to slice a burger patty in WAIC kitchen."""

    def __init__(self, viewer, options):
        self.viewer = viewer
        self.options = options
        self.sim_dt = options.dt
        self.sim_substeps = options.substeps
        self.sim_time = 0.0
        self.domain_extent = np.asarray(options.grid_size, dtype=np.float32) * options.voxel_size
        self.meat_lo = PAN_CENTER - np.asarray((0.5 * options.meat_length, 0.5 * options.meat_width, 0.0))
        self.meat_lo[2] = PAN_TOP_Z + 0.012
        self.meat_hi = self.meat_lo + np.asarray((options.meat_length, options.meat_width, options.meat_height))
        self.knife_center = KNIFE_REST_CENTER.copy()
        self._has_robot_visual = False

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, options.gravity))
        SolverMPMLite.register_custom_attributes(builder)
        self._add_robot(builder, options)
        self._add_kitchen_visual(builder, options)
        self._add_scene_geometry(builder)
        self._add_meat(builder, options)
        builder.color()

        self.model = builder.finalize(requires_grad=False)
        self.state_0 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self._setup_robot_ik()
        self._set_knife_visual(self.knife_center)
        self.solver = SolverMPMLite(
            self.model,
            SolverMPMLite.Config(
                grid_size=tuple(options.grid_size),
                voxel_size=options.voxel_size,
                solver_type="lite_implicit",
                max_iterations=options.max_iterations,
                density=options.density,
                young_modulus=options.young_modulus,
                poisson_ratio=options.poisson_ratio,
                yield_stress=options.yield_stress,
            ),
        )
        self._table_nodes = self._make_table_nodes(options)
        self._paint_boundaries(self.knife_center, np.zeros(3, dtype=np.float32))

        self.viewer.set_model(self.model)
        self.viewer.show_particles = True
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(pos=wp.vec3(4.10, -1.25, 1.85), pitch=-17.0, yaw=62.0)
        print(
            f"[newton] MPM Lite W1 burger slice: particles={self.model.particle_count}, "
            f"grid={tuple(options.grid_size)}, dx={options.voxel_size}"
        )

    @staticmethod
    def _asset_root() -> Path:
        return Path(__file__).with_name("assets") / "mpm_lite_waic"

    def _add_robot(self, builder: newton.ModelBuilder, options) -> None:
        if options.hide_robot:
            return
        urdf_path = self._asset_root() / "hands-only" / "DexforceW1V021_visual_collision_hands_only.urdf"
        if not urdf_path.exists():
            print(f"[newton] W1 visual asset not found, skipping: {urdf_path}")
            return
        builder.add_urdf(
            str(urdf_path),
            xform=wp.transform(WAIC_ROBOT_BASE_POS, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        self._has_robot_visual = True

    def _add_kitchen_visual(self, builder: newton.ModelBuilder, options) -> None:
        if options.hide_waic_visual:
            return
        usd_path = self._asset_root() / "waic_kitchen_counter_table_pan.usd"
        if not usd_path.exists():
            print(f"[newton] WAIC kitchen visual asset not found, skipping: {usd_path}")
            return
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            print(f"[newton] Failed to open WAIC kitchen visual: {usd_path}")
            return

        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.density = 0.0
        cfg.has_shape_collision = False
        cfg.has_particle_collision = False
        cfg.is_visible = True
        cache = UsdGeom.XformCache()
        mesh_count = 0
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            vertices, indices = self._usd_mesh_arrays(UsdGeom.Mesh(prim), cache)
            if len(vertices) == 0 or len(indices) == 0:
                continue
            mesh = newton.Mesh(vertices=vertices, indices=indices, compute_inertia=False, is_solid=False)
            builder.add_shape_mesh(
                body=-1,
                mesh=mesh,
                xform=wp.transform_identity(),
                cfg=cfg,
                color=self._kitchen_color(str(prim.GetPath())),
                label=f"waic_kitchen_visual_{mesh_count:03d}",
            )
            mesh_count += 1

    def _add_scene_geometry(self, builder: newton.ModelBuilder) -> None:
        visual_cfg = newton.ModelBuilder.ShapeConfig()
        visual_cfg.density = 0.0
        visual_cfg.has_shape_collision = False
        visual_cfg.has_particle_collision = False
        visual_cfg.is_visible = True
        builder.add_shape_cylinder(
            body=-1,
            xform=wp.transform(wp.vec3(*PAN_CENTER), wp.quat_identity()),
            radius=PAN_RADIUS,
            half_height=PAN_DISK_HALF_HEIGHT,
            cfg=visual_cfg,
            color=(0.08, 0.09, 0.10),
            label="mpm_lite_pan",
        )
        self.knife_body = builder.add_body(xform=wp.transform(wp.vec3(*KNIFE_REST_CENTER), wp.quat_identity()))
        builder.add_shape_box(
            body=self.knife_body,
            xform=wp.transform_identity(),
            hx=BLADE_HX,
            hy=BLADE_HY,
            hz=BLADE_HZ,
            cfg=visual_cfg,
            color=(0.80, 0.80, 0.85),
            label="mpm_lite_knife",
        )

    def _add_meat(self, builder: newton.ModelBuilder, options) -> None:
        extent = self.meat_hi - self.meat_lo
        resolution = np.ceil(options.particles_per_cell * extent / options.voxel_size).astype(int)
        cell_size = extent / resolution
        builder.add_particle_grid(
            pos=wp.vec3(*self.meat_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=int(resolution[0]) + 1,
            dim_y=int(resolution[1]) + 1,
            dim_z=int(resolution[2]) + 1,
            cell_x=float(cell_size[0]),
            cell_y=float(cell_size[1]),
            cell_z=float(cell_size[2]),
            mass=float(np.prod(cell_size) * options.density),
            jitter=0.15 * float(np.max(cell_size) * 0.5),
            radius_mean=float(np.max(cell_size) * 0.5),
        )

    def _make_table_nodes(self, options) -> np.ndarray:
        dx = options.voxel_size
        x = np.arange(np.floor((PAN_CENTER[0] - PAN_RADIUS) / dx), np.ceil((PAN_CENTER[0] + PAN_RADIUS) / dx) + 1)
        y = np.arange(np.floor((PAN_CENTER[1] - PAN_RADIUS) / dx), np.ceil((PAN_CENTER[1] + PAN_RADIUS) / dx) + 1)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        disk = (xx * dx - PAN_CENTER[0]) ** 2 + (yy * dx - PAN_CENTER[1]) ** 2 <= PAN_RADIUS**2
        z = np.full(np.count_nonzero(disk), int(round(PAN_TOP_Z / dx)), dtype=np.int32)
        return np.column_stack((xx[disk], yy[disk], z)).astype(np.int32)

    def _knife_nodes(self, center: np.ndarray, options) -> np.ndarray:
        dx = options.voxel_size
        half_extents = np.asarray((BLADE_HX, BLADE_HY, BLADE_HZ), dtype=np.float32)
        low = np.floor((center - half_extents) / dx).astype(np.int32)
        high = np.ceil((center + half_extents) / dx).astype(np.int32)
        nodes = np.stack(np.meshgrid(*(np.arange(low[i], high[i] + 1) for i in range(3)), indexing="ij"), axis=-1)
        return nodes.reshape(-1, 3).astype(np.int32)

    def _paint_boundaries(self, center: np.ndarray, velocity: np.ndarray) -> None:
        knife_nodes = self._knife_nodes(center, self.options)
        nodes = np.concatenate((self._table_nodes, knife_nodes))
        velocities = np.zeros((len(nodes), 3), dtype=np.float32)
        velocities[len(self._table_nodes) :] = velocity
        grid_max = np.asarray(self.options.grid_size, dtype=np.int32) - 1
        valid = np.all((nodes >= 0) & (nodes <= grid_max), axis=1)
        self.solver.paint_boundary(
            nodes[valid], np.ones(np.count_nonzero(valid), dtype=np.int32), velocities=velocities[valid]
        )

    def _knife_target(self, time: float) -> np.ndarray:
        cut_y_lo = PAN_CENTER[1] - min(0.018, 0.35 * self.options.meat_width)
        cut_y_hi = PAN_CENTER[1] + min(0.018, 0.35 * self.options.meat_width)
        lift = KNIFE_REST_CENTER + np.asarray((0.0, 0.0, 0.22), dtype=np.float32)
        above = np.asarray((PAN_CENTER[0], cut_y_lo, PAN_TOP_Z + 0.28 - BLADE_HZ), dtype=np.float32)
        descend = np.asarray((PAN_CENTER[0], cut_y_lo, PAN_TOP_Z + 0.006 + BLADE_HZ), dtype=np.float32)
        cut_end = np.asarray((PAN_CENTER[0], cut_y_hi, PAN_TOP_Z + 0.006 + BLADE_HZ), dtype=np.float32)
        segments = (
            (1.2, KNIFE_REST_CENTER, KNIFE_REST_CENTER),
            (0.8, KNIFE_REST_CENTER, lift),
            (1.2, lift, above),
            (0.8, above, descend),
            (3.2, descend, cut_end),
            (0.8, cut_end, above),
            (0.8, above, lift),
            (0.8, lift, KNIFE_REST_CENTER),
        )
        remaining = time
        for duration, start, end in segments:
            if remaining <= duration:
                return start + (end - start) * np.float32(np.clip(remaining / duration, 0.0, 1.0))
            remaining -= duration
        return KNIFE_REST_CENTER.copy()

    def _set_knife_visual(self, center: np.ndarray) -> None:
        wp.launch(
            _set_body_transform,
            dim=1,
            inputs=[self.state_0.body_q, self.knife_body, wp.vec3(*center)],
            device=self.model.device,
        )

    def _setup_robot_ik(self) -> None:
        if not self._has_robot_visual:
            self.ik_solver = None
            return
        try:
            right_ee_index = next(
                index for index, label in enumerate(self.model.body_label) if label.endswith("/right_j7")
            )
        except StopIteration:
            print("[newton] W1 right wrist was not found, leaving the visual robot static.")
            self.ik_solver = None
            return
        right_ee_transform = wp.transform(*self.state_0.body_q.numpy()[right_ee_index])
        self.right_pos_obj = ik.IKObjectivePosition(
            link_index=right_ee_index,
            link_offset=TCP_OFFSET,
            target_positions=wp.array([wp.transform_get_translation(right_ee_transform)], dtype=wp.vec3),
        )
        self.right_rot_obj = ik.IKObjectiveRotation(
            link_index=right_ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*RIGHT_WRIST_QUAT)], dtype=wp.vec4),
        )
        self.ik_solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.right_pos_obj, self.right_rot_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))

    def _drive_robot(self, knife_center: np.ndarray) -> None:
        if self.ik_solver is None:
            return
        tcp_position = wp.vec3(float(knife_center[0]), float(knife_center[1]), float(knife_center[2] + BLADE_HZ))
        self.right_pos_obj.set_target_position(0, tcp_position)
        self.right_rot_obj.set_target_rotation(0, wp.vec4(*RIGHT_WRIST_QUAT))
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=24)
        wp.launch(
            _copy_ik_to_joint_q,
            dim=self.model.joint_coord_count,
            inputs=[self.ik_joint_q],
            outputs=[self.state_0.joint_q],
            device=self.model.device,
        )
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            next_center = self._knife_target(self.sim_time + self.sim_dt)
            velocity = (next_center - self.knife_center) / self.sim_dt
            self._paint_boundaries(next_center, velocity)
            self._drive_robot(next_center)
            self._set_knife_visual(next_center)
            self.solver.step(self.state_0, self.state_0, None, None, self.sim_dt)
            self.knife_center = next_center
            self.sim_time += self.sim_dt

    def step(self) -> None:
        self.simulate()

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self) -> None:
        """Keep burger particles finite throughout the knife trajectory."""
        if not np.isfinite(self.state_0.particle_q.numpy()).all():
            raise ValueError("MPM Lite burger particle positions are not finite.")

    def test_final(self) -> None:
        """Keep burger particles inside the MPM Lite grid."""
        positions = self.state_0.particle_q.numpy()
        if not np.isfinite(positions).all() or np.any(positions < 0.0) or np.any(positions > self.domain_extent):
            raise ValueError("MPM Lite burger particles left the simulation grid.")

    @staticmethod
    def _usd_mesh_arrays(mesh: UsdGeom.Mesh, cache: UsdGeom.XformCache) -> tuple[np.ndarray, np.ndarray]:
        points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not face_counts or not face_indices:
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int32)
        transform = cache.GetLocalToWorldTransform(mesh.GetPrim())
        vertices = np.asarray(
            [transform.Transform(Gf.Vec3d(point[0], point[1], point[2])) for point in points], dtype=np.float32
        )
        triangles: list[int] = []
        offset = 0
        for count in face_counts:
            face = face_indices[offset : offset + int(count)]
            for index in range(1, len(face) - 1):
                triangles.extend((int(face[0]), int(face[index]), int(face[index + 1])))
            offset += int(count)
        return vertices, np.asarray(triangles, dtype=np.int32)

    @staticmethod
    def _kitchen_color(path: str) -> tuple[float, float, float]:
        name = path.lower()
        if "pan" in name:
            return (0.08, 0.09, 0.10)
        if "table" in name:
            return (0.52, 0.37, 0.23)
        return (0.70, 0.68, 0.62)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=480)
        parser.add_argument("--substeps", type=int, default=1)
        parser.add_argument("--grid-size", type=int, nargs=3, default=(448, 128, 128))
        parser.add_argument("--voxel-size", "-dx", type=float, default=0.016)
        parser.add_argument("--dt", type=float, default=1.0 / 120.0)
        parser.add_argument("--max-iterations", type=int, default=100)
        parser.add_argument("--density", type=float, default=1000.0)
        parser.add_argument("--young-modulus", type=float, default=1.0e6)
        parser.add_argument("--poisson-ratio", type=float, default=0.45)
        parser.add_argument("--yield-stress", type=float, default=1.0e5)
        parser.add_argument("--gravity", type=float, default=-9.81)
        parser.add_argument("--particles-per-cell", type=int, default=2)
        parser.add_argument("--meat-length", type=float, default=MEAT_LENGTH)
        parser.add_argument("--meat-width", type=float, default=MEAT_WIDTH)
        parser.add_argument("--meat-height", type=float, default=MEAT_HEIGHT)
        parser.add_argument("--hide-robot", action="store_true")
        parser.add_argument("--hide-waic-visual", action="store_true")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

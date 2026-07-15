# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Dexforce Three Workcell Director
#
# A three-workcell room-level director for Dexforce W1 tasks. During navigation
# a real W1 URDF is displayed in the room and moved directly by setting its
# link poses. No robot/workcell simulation is stepped until W1 reaches a
# workcell trigger and the corresponding heavyweight task example is loaded.
#
# Command: python -m newton.examples cloth_dexforce_three_workcell_director
#
###########################################################################

from __future__ import annotations

import gc
import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton.examples.cloth.example_cloth_dexforce_bimanual_ik_tshirt import (
    SHIRT_ASSET,
    SHIRT_DENSITY,
    SHIRT_EDGE_KD,
    SHIRT_EDGE_KE,
    SHIRT_POS,
    SHIRT_PRIM_PATH,
    SHIRT_ROT,
    SHIRT_SCALE,
    SHIRT_TRI_KA,
    SHIRT_TRI_KD,
    SHIRT_TRI_KE,
)


DEMO_CAMERA_POS = wp.vec3(1.15, -2.10, 1.65)
DEMO_CAMERA_PITCH = -16.0
DEMO_CAMERA_YAW = 112.0


def _lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def _smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _lerp_vec3(a: wp.vec3, b: wp.vec3, u: float) -> wp.vec3:
    return wp.vec3(
        _lerp(float(a[0]), float(b[0]), u),
        _lerp(float(a[1]), float(b[1]), u),
        _lerp(float(a[2]), float(b[2]), u),
    )


def _center_shirt_vertices(vertices: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32).copy()
    bounds_min = verts.min(axis=0)
    bounds_max = verts.max(axis=0)
    center_xy = 0.5 * (bounds_min[:2] + bounds_max[:2])
    verts[:, 0] -= center_xy[0]
    verts[:, 1] -= center_xy[1]
    verts[:, 2] -= bounds_min[2]
    return verts * SHIRT_SCALE


def _camera_angles_for_target(pos: wp.vec3, target: wp.vec3) -> tuple[float, float]:
    direction = np.array(
        [
            float(target[0]) - float(pos[0]),
            float(target[1]) - float(pos[1]),
            float(target[2]) - float(pos[2]),
        ],
        dtype=np.float64,
    )
    length = np.linalg.norm(direction)
    if length < 1.0e-6:
        return 0.0, -180.0

    direction /= length
    pitch = np.rad2deg(np.arcsin(np.clip(direction[2], -1.0, 1.0)))
    yaw = np.rad2deg(np.arctan2(direction[1], direction[0]))
    return float(pitch), float((yaw + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class WorkcellSpec:
    name: str
    module_path: str
    room_pos: wp.vec3
    travel_time: float
    task_time: float
    trigger_radius: float = 0.70

    @property
    def table_center(self) -> wp.vec3:
        return self.room_pos + wp.vec3(0.58, 0.0, 1.15)


WORKCELLS = [
    WorkcellSpec(
        name="fold_tshirt_v2",
        module_path="newton.examples.cloth.example_cloth_dexforce_bimanual_fold_tshirt_v2",
        room_pos=wp.vec3(-3.6, 1.4, 0.0),
        travel_time=4.5,
        task_time=10.0,
    ),
    WorkcellSpec(
        name="grasp_cloth_v3",
        module_path="newton.examples.cloth.example_cloth_dexforce_bimanual_grasp_cloth_v3",
        room_pos=wp.vec3(0.0, -1.6, 0.0),
        travel_time=4.5,
        task_time=8.0,
    ),
    WorkcellSpec(
        name="grasp_cube",
        module_path="newton.examples.cloth.example_cloth_dexforce_bimanual_grasp_cube",
        room_pos=wp.vec3(3.6, 1.4, 0.0),
        travel_time=4.5,
        task_time=8.0,
    ),
]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args

        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.loop = args.loop
        self.dry_run_tasks = args.dry_run_tasks
        self.cache_tasks = args.cache_tasks

        self.home_pos = wp.vec3(-5.2, -2.4, 0.0)
        self.w1_pos = self.home_pos
        self.active_workcell_index: int | None = None
        self.active_task = None
        self.task_cache = {}
        self.events: list[dict] = []

        self.timeline = self._build_timeline()
        self.cycle_time = self.timeline[-1]["end"]

        self._build_room_scene()
        self._show_room_scene()
        self._set_w1_nav_pose(self.home_pos)
        self._update_navigation_camera(self.home_pos)

    def _build_timeline(self):
        timeline = []
        t = 0.0
        prev = self.home_pos
        for i, workcell in enumerate(WORKCELLS):
            travel_start = t
            travel_end = travel_start + workcell.travel_time
            timeline.append(
                {
                    "type": "travel",
                    "workcell": i,
                    "start": travel_start,
                    "end": travel_end,
                    "from": prev,
                    "to": workcell.room_pos,
                }
            )

            task_start = travel_end
            task_end = task_start + workcell.task_time
            timeline.append(
                {
                    "type": "task",
                    "workcell": i,
                    "start": task_start,
                    "end": task_end,
                    "from": workcell.room_pos,
                    "to": workcell.room_pos,
                }
            )
            t = task_end
            prev = workcell.room_pos

        return_start = t
        return_end = return_start + WORKCELLS[0].travel_time
        timeline.append(
            {
                "type": "travel",
                "workcell": 0,
                "start": return_start,
                "end": return_end,
                "from": prev,
                "to": WORKCELLS[0].room_pos,
            }
        )

        return timeline

    def _build_room_scene(self):
        builder = newton.ModelBuilder()
        floor_cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, density=0.0)
        table_cfg = newton.ModelBuilder.ShapeConfig(mu=1.1, density=0.0)

        builder.add_shape_box(
            body=-1,
            cfg=floor_cfg,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.03), wp.quat_identity()),
            hx=6.8,
            hy=3.8,
            hz=0.03,
            color=(0.22, 0.24, 0.26),
        )

        self._add_fold_tshirt_preview(builder, table_cfg, WORKCELLS[0].room_pos)
        self._add_grasp_cloth_preview(builder, table_cfg, WORKCELLS[1].room_pos)
        self._add_grasp_cube_preview(builder, table_cfg, WORKCELLS[2].room_pos)

        w1_body_start = builder.body_count
        urdf_path = Path("E:/csy_work/CG/assets/DexforceW1V021") / "DexforceW1V021.urdf"
        builder.add_urdf(
            urdf_path,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            force_show_colliders=False,
        )
        self.w1_body_indices = np.arange(w1_body_start, builder.body_count, dtype=np.int32)

        self.room_model = builder.finalize()
        self.room_state = self.room_model.state()
        self.w1_root_joint = 0
        self.w1_root_q_start = int(self.room_model.joint_q_start.numpy()[self.w1_root_joint])
        self.w1_root_q0 = self.room_model.joint_q.numpy()[self.w1_root_q_start : self.w1_root_q_start + 7].copy()
        newton.eval_fk(self.room_model, self.room_model.joint_q, self.room_model.joint_qd, self.room_state)

    def _add_demo_table(
        self,
        builder: newton.ModelBuilder,
        cfg: newton.ModelBuilder.ShapeConfig,
        origin: wp.vec3,
        table_pos: wp.vec3,
        half_extents: tuple[float, float, float],
        color: tuple[float, float, float] = (0.35, 0.42, 0.48),
    ):
        builder.add_shape_box(
            body=-1,
            cfg=cfg,
            xform=wp.transform(origin + table_pos, wp.quat_identity()),
            hx=half_extents[0],
            hy=half_extents[1],
            hz=half_extents[2],
            color=color,
        )

    def _add_fold_tshirt_preview(
        self,
        builder: newton.ModelBuilder,
        table_cfg: newton.ModelBuilder.ShapeConfig,
        origin: wp.vec3,
    ):
        table_pos = wp.vec3(0.55, 0.0, 1.15)
        table_half = (0.26, 0.62, 0.025)
        self._add_demo_table(builder, table_cfg, origin, table_pos, table_half)

        usd_stage = Usd.Stage.Open(newton.examples.get_asset(SHIRT_ASSET))
        usd_prim = usd_stage.GetPrimAtPath(SHIRT_PRIM_PATH)
        shirt_mesh = newton.usd.get_mesh(usd_prim)
        vertices = _center_shirt_vertices(shirt_mesh.vertices)
        builder.add_cloth_mesh(
            vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in vertices],
            indices=shirt_mesh.indices,
            rot=SHIRT_ROT,
            pos=origin + SHIRT_POS,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=SHIRT_DENSITY,
            scale=1.0,
            tri_ke=SHIRT_TRI_KE,
            tri_ka=SHIRT_TRI_KA,
            tri_kd=SHIRT_TRI_KD,
            edge_ke=SHIRT_EDGE_KE,
            edge_kd=SHIRT_EDGE_KD,
            particle_radius=0.0,
            label="fold_tshirt_preview",
        )

    def _add_grasp_cloth_preview(
        self,
        builder: newton.ModelBuilder,
        table_cfg: newton.ModelBuilder.ShapeConfig,
        origin: wp.vec3,
    ):
        table_pos = wp.vec3(0.60, 0.0, 1.15)
        table_half = (0.32, 0.78, 0.025)
        table_top_z = float(table_pos[2]) + table_half[2]
        self._add_demo_table(builder, table_cfg, origin, table_pos, table_half)

        cloth_dim_x = 24
        cloth_dim_y = 36
        cloth_cell_x = 0.022
        cloth_cell_y = 0.025
        cloth_cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False)
        builder.add_shape_box(
            body=-1,
            cfg=cloth_cfg,
            xform=wp.transform(origin + wp.vec3(0.60, 0.0, table_top_z + 0.010), wp.quat_identity()),
            hx=0.5 * cloth_dim_x * cloth_cell_x,
            hy=0.5 * cloth_dim_y * cloth_cell_y,
            hz=0.004,
            color=(0.78, 0.12, 0.10),
        )

    def _add_grasp_cube_preview(
        self,
        builder: newton.ModelBuilder,
        table_cfg: newton.ModelBuilder.ShapeConfig,
        origin: wp.vec3,
    ):
        table_pos = wp.vec3(0.60, 0.0, 1.15)
        table_half = (0.32, 0.78, 0.025)
        table_top_z = float(table_pos[2]) + table_half[2]
        cube_half = 0.025
        cube_z = table_top_z + cube_half
        self._add_demo_table(builder, table_cfg, origin, table_pos, table_half)

        cube_cfg = newton.ModelBuilder.ShapeConfig(has_shape_collision=False, has_particle_collision=False)
        builder.add_shape_box(
            body=-1,
            cfg=cube_cfg,
            xform=wp.transform(origin + wp.vec3(0.52, 0.53, cube_z), wp.quat_identity()),
            hx=cube_half,
            hy=cube_half,
            hz=cube_half,
            color=(0.95, 0.38, 0.16),
        )
        builder.add_shape_box(
            body=-1,
            cfg=cube_cfg,
            xform=wp.transform(
                origin + wp.vec3(0.52, -0.53, cube_z),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.35),
            ),
            hx=cube_half,
            hy=cube_half,
            hz=cube_half,
            color=(0.16, 0.55, 0.78),
        )

    def _show_room_scene(self):
        self.viewer.set_model(self.room_model)

    def _timeline_segment(self):
        local_time = self.sim_time
        if self.loop:
            local_time %= self.cycle_time
        else:
            local_time = min(local_time, self.cycle_time - 1.0e-6)

        for segment in self.timeline:
            if segment["start"] <= local_time < segment["end"]:
                return segment, local_time
        return self.timeline[-1], local_time

    def _pose_for_segment(self, segment, local_time: float) -> wp.vec3:
        if segment["type"] == "task":
            return segment["to"]

        u = (local_time - segment["start"]) / max(segment["end"] - segment["start"], 1.0e-6)
        return _lerp_vec3(segment["from"], segment["to"], _smoothstep(u))

    def _set_w1_nav_pose(self, pos: wp.vec3, yaw: float = 0.0):
        self.w1_pos = pos
        root_q = self.room_model.joint_q.numpy()
        q_start = self.w1_root_q_start
        root_q[q_start : q_start + 3] = np.array(
            [
                float(self.w1_root_q0[0]) + float(pos[0]),
                float(self.w1_root_q0[1]) + float(pos[1]),
                float(self.w1_root_q0[2]) + float(pos[2]),
            ],
            dtype=root_q.dtype,
        )
        root_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw) * wp.quat(
            float(self.w1_root_q0[3]),
            float(self.w1_root_q0[4]),
            float(self.w1_root_q0[5]),
            float(self.w1_root_q0[6]),
        )
        root_q[q_start + 3 : q_start + 7] = np.array(
            [float(root_rot[0]), float(root_rot[1]), float(root_rot[2]), float(root_rot[3])],
            dtype=root_q.dtype,
        )
        self.room_model.joint_q.assign(root_q)
        newton.eval_fk(self.room_model, self.room_model.joint_q, self.room_model.joint_qd, self.room_state)

    def _workcell_facing_yaw(self, workcell_index: int) -> float:
        workcell = WORKCELLS[workcell_index]
        delta = np.array(
            [
                float(workcell.table_center[0]) - float(workcell.room_pos[0]),
                float(workcell.table_center[1]) - float(workcell.room_pos[1]),
            ],
            dtype=np.float64,
        )
        if np.linalg.norm(delta) < 1.0e-6:
            return 0.0
        return float(np.arctan2(delta[1], delta[0]))

    def _update_navigation_camera(self, pos: wp.vec3):
        camera_pos = pos + wp.vec3(-1.4, -2.1, 1.75)
        target = pos + wp.vec3(0.35, 0.0, 0.75)
        pitch, yaw = _camera_angles_for_target(camera_pos, target)
        self.viewer.set_camera(camera_pos, pitch, yaw)

    def _update_room_workcell_camera(self, workcell: WorkcellSpec):
        self.viewer.set_camera(workcell.room_pos + DEMO_CAMERA_POS, DEMO_CAMERA_PITCH, DEMO_CAMERA_YAW)

    def _update_task_camera(self):
        self.viewer.set_camera(DEMO_CAMERA_POS, DEMO_CAMERA_PITCH, DEMO_CAMERA_YAW)

    def _task_reached(self, workcell: WorkcellSpec) -> bool:
        delta = np.array(
            [
                float(self.w1_pos[0]) - float(workcell.room_pos[0]),
                float(self.w1_pos[1]) - float(workcell.room_pos[1]),
                float(self.w1_pos[2]) - float(workcell.room_pos[2]),
            ],
            dtype=np.float64,
        )
        return float(np.linalg.norm(delta)) <= workcell.trigger_radius

    def _default_task_args(self, workcell: WorkcellSpec):
        mod = importlib.import_module(workcell.module_path)
        parser_factory = getattr(mod.Example, "create_parser", None)
        parser = parser_factory() if parser_factory is not None else newton.examples.create_parser()
        task_args = newton.examples.default_args(parser)
        task_args.viewer = self.args.viewer
        task_args.device = self.args.device
        task_args.quiet = self.args.quiet
        task_args.test = False

        if hasattr(task_args, "trajectory_time_scale"):
            task_args.trajectory_time_scale = self.args.fold_trajectory_time_scale
        if hasattr(task_args, "print_interval"):
            task_args.print_interval = self.args.print_interval
        if hasattr(task_args, "enable_self_collisions"):
            task_args.enable_self_collisions = self.args.enable_self_collisions

        return mod, task_args

    def _activate_task(self, index: int):
        if self.active_workcell_index == index:
            return

        workcell = WORKCELLS[index]
        if not self._task_reached(workcell):
            return

        self._deactivate_task()
        self.active_workcell_index = index
        self.events.append({"time": self.sim_time, "event": "task_start", "workcell": workcell.name})

        if self.dry_run_tasks:
            self._show_room_scene()
            self._update_room_workcell_camera(workcell)
            return

        if self.cache_tasks and index in self.task_cache:
            self.active_task = self.task_cache[index]
            self.viewer.set_model(self.active_task.model)
            self._update_task_camera()
            return

        if hasattr(self.viewer, "clear_model"):
            self.viewer.clear_model()

        mod, task_args = self._default_task_args(workcell)
        self.active_task = mod.Example(self.viewer, task_args)
        self._update_task_camera()

    def _deactivate_task(self):
        if self.active_workcell_index is None:
            return

        workcell = WORKCELLS[self.active_workcell_index]
        self.events.append({"time": self.sim_time, "event": "task_end", "workcell": workcell.name})

        if self.active_task is not None and self.cache_tasks:
            self.task_cache[self.active_workcell_index] = self.active_task
        else:
            self.active_task = None
            gc.collect()

        self.active_workcell_index = None
        if hasattr(self.viewer, "clear_model"):
            self.viewer.clear_model()
        self._show_room_scene()

    def _step_task(self):
        if self.active_task is not None:
            self.active_task.step()

    def _render_task(self):
        task = self.active_task
        if task is None:
            return

        self.viewer.log_state(task.state_0)
        contacts = getattr(task, "contacts", None)
        if contacts is not None:
            self.viewer.log_contacts(contacts, task.state_0)

    def step(self):
        segment, local_time = self._timeline_segment()
        pos = self._pose_for_segment(segment, local_time)
        yaw = self._workcell_facing_yaw(segment["workcell"])

        if segment["type"] == "task":
            self._set_w1_nav_pose(pos, yaw)
            self._activate_task(segment["workcell"])
            self._step_task()
        else:
            self._deactivate_task()
            self._set_w1_nav_pose(pos, yaw)
            self._update_navigation_camera(pos)

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        if self.active_task is None or self.dry_run_tasks:
            self.viewer.log_state(self.room_state)
        else:
            self._render_task()
        self.viewer.end_frame()

    def test_final(self):
        if not self.dry_run_tasks and not self.events:
            raise ValueError("No workcell task events were recorded.")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(loop=True)
        parser.add_argument("--loop", action="store_true", help="Loop the three-workcell timeline.")
        parser.add_argument("--no-loop", action="store_false", dest="loop", help="Stop after returning to the first workcell.")
        parser.add_argument(
            "--dry-run-tasks",
            action="store_true",
            default=False,
            help="Move the navigation W1 through all workcells without loading heavy task examples.",
        )
        parser.add_argument(
            "--cache-tasks",
            action="store_true",
            default=False,
            help="Keep task examples alive after leaving a workcell. Faster revisits, higher memory use.",
        )
        parser.add_argument(
            "--fold-trajectory-time-scale",
            type=float,
            default=4.0,
            help="Trajectory time scale passed to fold_tshirt_v2.",
        )
        parser.add_argument(
            "--print-interval",
            type=float,
            default=9999.0,
            help="TCP report interval passed to grasp tasks.",
        )
        parser.add_argument(
            "--enable-self-collisions",
            action="store_true",
            default=False,
            help="Enable Dexforce URDF self-collisions in grasp workcells.",
        )
        parser.set_defaults(num_frames=2700)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

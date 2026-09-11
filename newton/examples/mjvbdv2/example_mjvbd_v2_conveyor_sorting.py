# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sort rigid, soft, cloth, and pneumatic parcels with a full Dexforce W1.

Run ``uv run --extra examples -m newton.examples mjvbd_v2_conveyor_sorting``.
The indexed conveyor stops when the next parcel reaches the picking station.
Realtime IK drives the right arm and its articulated fingers. Parcels move
through conveyor friction and hand contact, with no attachment constraints.
The robot is kinematic; all four parcels remain dynamically simulated by
MJVBDV2, including tetrahedral elasticity, cloth bending, and sealed gas pressure.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp
from PIL import Image, ImageDraw, ImageFont

import newton
import newton.examples
import newton.ik as ik
from newton.examples.mjvbdv2.support import example_vbd_mjvbd_v2_right_hand_inflatable_bag_recorder as bag_reference
from newton.examples.mjvbdv2.support.conveyor_belt import ConveyorBeltVisual
from newton.examples.mjvbdv2.support.conveyor_clearance import RobotClearanceAudit
from newton.examples.mjvbdv2.support.conveyor_motion import (
    JointMotionRetimer,
    interpolate_waypoints,
    read_motion_limits,
)
from newton.solvers import SolverMJVBDV2, add_inflatable_mesh

ROBOT_URDF = Path(__file__).resolve().parents[3] / "assets/DexforceW1V021/DexforceW1V021.urdf"
FPS = 60
BELT_TOP = 0.76
BELT_X = 0.60
PICK_Y = -0.12
COLORS = ((0.95, 0.48, 0.12), (0.26, 0.70, 0.40), (0.23, 0.57, 0.95), (0.78, 0.38, 0.83))
KINDS = ("rigid", "soft", "cloth", "inflatable")
# Give the draped fabric lateral clearance; the compact soft block needs
# less space. Keep the rigid and pneumatic drop points within the arm's reach.
TRAYS = ((0.65, -0.60), (0.28, -0.34), (0.04, -0.48), (0.34, -0.60))
TRAY_HALF_EXTENTS = ((0.125, 0.125), (0.095, 0.125), (0.125, 0.20), (0.125, 0.125))
TCP_OFFSET = wp.vec3(-0.066, 0.0, 0.0)
HOME = np.array((0.43, -0.30, 1.05), dtype=np.float32)
# World-space TCP correction at 80% pinch closure, preserving the grasp center [m].
CLOTH_OPEN_CORRECTION = np.array((-0.0053969, -0.0048804, 0.0027935))
CLOTH_THUMB_OPPOSITION = 0.74
CLOTH_CELLS = 40
CLOTH_GRASP_NODE = 36


@dataclass
class _Parcel:
    kind: str
    destination: np.ndarray
    tray_half_extent: tuple[float, float] = (0.125, 0.125)
    body: int = -1
    begin: int = 0
    end: int = 0
    triangles: wp.array | None = None
    uvs: wp.array | None = None
    texture: np.ndarray | None = None
    picked: bool = False
    released: bool = False
    sorted: bool = False
    lift: float = 0.0


@wp.kernel
def _prescribe_robot(
    start: wp.array[float],
    end: wp.array[float],
    q_start: wp.array[int],
    qd_start: wp.array[int],
    fraction: float,
    inv_dt: float,
    q: wp.array[float],
    qd: wp.array[float],
):
    joint = wp.tid()
    # The fixed-base W1 has only scalar movable joints.
    if q_start[joint + 1] > q_start[joint]:
        i = q_start[joint]
        q[i] = wp.lerp(start[i], end[i], fraction)
        qd[qd_start[joint]] = (end[i] - start[i]) * inv_dt


@wp.kernel
def _lock_coordinates(indices: wp.array[int], values: wp.array[float], q: wp.array2d[float]):
    i = wp.tid()
    q[0, indices[i]] = values[i]


@wp.kernel
def _discard_cloth_microsteps(
    begin: int,
    tray: wp.vec3,
    threshold: float,
    dt: float,
    previous: wp.array[wp.vec3],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
):
    """Discard frame-to-frame noise near the tray after all contact substeps."""
    local = wp.tid()
    i = begin + local
    p = q[i]
    # Include the rim and a small fabric overhang, so a fold crossing the rim
    # is not split into filtered and persistently vibrating particles.
    if wp.abs(p[0] - tray[0]) < 0.25 and wp.abs(p[1] - tray[1]) < 0.18 and 0.684 < p[2] < 0.82:
        if wp.length(p - previous[local]) < threshold:
            q[i] = previous[local]
            # Reconstruct from the accepted position, just as the solver does.
            qd[i] = (q[i] - previous[local]) / dt


@wp.kernel
def _move_belt(
    bodies: wp.array[int],
    shapes: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    shape_transform: wp.array[wp.transform],
    motion: wp.array[float],
    substep_dt: float,
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    speed = motion[1]
    offset = motion[0] + speed * substep_dt
    y = -0.24 + wp.mod(float(i) * 0.08 - offset + 20.0, 2.0)
    lower = wp.max(-0.24, y - 0.08)
    upper = wp.min(1.76, y + 0.08)
    shape_scale[shapes[i]] = wp.vec3(0.175, 0.5 * (upper - lower), 0.03)
    shape_transform[shapes[i]] = wp.transform(wp.vec3(0.0, 0.5 * (upper + lower) - y, 0.0), wp.quat_identity())
    q[bodies[i]] = wp.transform(wp.vec3(0.60, y, 0.73), wp.quat_identity())
    qd[bodies[i]] = wp.spatial_vector(0.0, -speed, 0.0, 0.0, 0.0, 0.0)


@wp.kernel
def _move_rollers(
    bodies: wp.array[int],
    motion: wp.array[float],
    substep_dt: float,
    q: wp.array[wp.transform],
    qd: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    radius = 0.0521
    angle = (motion[0] + motion[1] * substep_dt) / radius
    y = -0.24 + 2.0 * float(i)
    q[bodies[i]] = wp.transform(wp.vec3(0.6, y, 0.708), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angle))
    qd[bodies[i]] = wp.spatial_vector(0.0, 0.0, 0.0, motion[1] / radius, 0.0, 0.0)


class Example:
    """Run a contact-driven, indexed sorting station for four material types."""

    def __init__(self, viewer, args):
        if args.substeps < 1 or args.vbd_iterations < 1 or args.ik_iterations < 1:
            raise ValueError("Substep and iteration counts must be positive")
        if not 0.02 <= args.belt_speed <= 0.20:
            raise ValueError("--belt-speed must be between 0.02 and 0.20 m/s")
        if not math.isfinite(args.cloth_displacement_threshold) or args.cloth_displacement_threshold < 0.0:
            raise ValueError("--cloth-displacement-threshold must be finite and nonnegative")
        self.viewer, self.args = viewer, args
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / args.substeps
        self.sim_time = self.phase_time = self.belt_offset = 0.0
        self.phase = "feed"
        self.active = 0
        self.completed = False
        self.robot_clearance = None
        self.motion_peak_speed_ratio = 0.0
        self.motion_min_arm_margin = math.inf
        self.robot_min_separation = math.inf
        self.gripping = False
        self.thumb_angle = math.pi / 2.0
        self.target = HOME.copy()
        self.motion_start = HOME.copy()
        self.place_target = HOME.copy()
        self.release_target = HOME.copy()
        self.release_rotation = wp.quat_identity()
        self.parcels = []
        self.cloth_speed_samples = deque(maxlen=FPS)
        self._build_scene(Path(args.robot_urdf).expanduser())
        # A free-fall frame must remain larger than the deadband. Work at frame
        # boundaries so the threshold does not depend on the substep count.
        gravity = float(np.linalg.norm(self.model.gravity.numpy()[0]))
        self.cloth_displacement_threshold = min(args.cloth_displacement_threshold, 0.25 * gravity * self.frame_dt**2)
        self._build_materials()
        self.belt_visual = ConveyorBeltVisual(ROBOT_URDF.parents[1] / "conveyor_station", self.model.device)
        self._load_grasps()
        self.motion_lower, self.motion_upper, self.motion_speed = read_motion_limits(
            self.model,
            Path(args.robot_urdf).expanduser(),
            self.robot_coord_count,
            finger_speed=math.radians(args.finger_speed),
        )
        self.arm_indices = np.asarray(
            [
                self.model.joint_q_start.numpy()[j]
                for j, name in enumerate(self.model.joint_label)
                if name.rsplit("/", 1)[-1] in {f"{side}_J{i}" for side in ("LEFT", "RIGHT") for i in range(1, 8)}
            ]
        )
        self.motion_lower[self.arm_indices] += math.radians(3.0)
        self.motion_upper[self.arm_indices] -= math.radians(3.0)
        self.motion_retimer = JointMotionRetimer(self.motion_speed, self.frame_dt)
        self._build_ik()
        self.feed_rotation = self.grasp_rotations[0]
        self.rotation = self.grasp_rotations[0]
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        cloth = self.parcels[2]
        self.cloth_previous = wp.empty(cloth.end - cloth.begin, dtype=wp.vec3, device=self.model.device)
        self.frame_start, self.frame_end = wp.clone(self.model.joint_q), wp.clone(self.model.joint_q)
        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=(0,),
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": args.vbd_iterations,
                "friction_epsilon": 1.0e-4,
                "rigid_avbd_contact_alpha": 0.95,
                "rigid_contact_history": False,
                "rigid_body_contact_buffer_size": 4096,
                "rigid_body_particle_contact_buffer_size": 4096,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": 0.0015,
                "particle_self_contact_margin": 0.003,
                "particle_topological_contact_filter_threshold": 1,
                "particle_rest_shape_contact_exclusion_radius": 0.03,
            },
            collision_options={
                "broad_phase": "nxn",
                "contact_matching": "latest",
                "soft_contact_max": 32768,
                "soft_contact_margin": 0.008,
                "include_static_kinematic_pairs": False,
                "enable_rigid_soft_full_surface_contact": True,
                "rigid_soft_full_surface_shape_indices": self.hand_shapes,
            },
        )
        self._configure_viewer()
        self.graph = None
        self.belt_motion = wp.zeros(2, dtype=float, device=self.model.device)
        self.rest_volume = float(self.state_0.pneumatic.volume.numpy()[0])
        self.use_graph = self.model.device.is_cuda and args.substeps % 2 == 0

    @classmethod
    def create_render_scene(cls, viewer, args):
        """Build matching render geometry without constructing IK or physics solvers."""
        scene = cls.__new__(cls)
        scene.viewer, scene.args = viewer, args
        scene.parcels = []
        scene.cloth_speed_samples = deque(maxlen=FPS)
        scene.sim_time = scene.belt_offset = 0.0
        scene.phase, scene.active, scene.completed = "feed", 0, False
        scene._build_scene(Path(args.robot_urdf).expanduser())
        scene._build_materials()
        scene.belt_visual = ConveyorBeltVisual(ROBOT_URDF.parents[1] / "conveyor_station", scene.model.device)
        scene.state_0 = scene.model.state()
        scene._configure_viewer()
        return scene

    def _configure_viewer(self):
        """Share lighting, materials, camera, and status between simulation and replay."""
        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._render_ui)
        self.viewer.show_particles = False
        self.viewer.show_triangles = False
        self.viewer.set_camera(wp.vec3(2.2, -2.3, 1.9), -24.0, 126.0)
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            renderer = self.viewer.renderer
            renderer.sky_upper = (0.19, 0.23, 0.28)
            renderer.sky_lower = (0.38, 0.41, 0.44)
            renderer.ambient_sky = (0.68, 0.72, 0.78)
            renderer.ambient_ground = (0.25, 0.26, 0.28)
            renderer.exposure = 1.3
            renderer.shadow_extents = 3.5

    def _build_scene(self, robot_path: Path):
        if not robot_path.is_file():
            raise FileNotFoundError(f"W1 URDF not found: {robot_path}")
        builder = newton.ModelBuilder()
        SolverMJVBDV2.register_custom_attributes(builder)
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 1.0
        builder.default_shape_cfg.mu = 0.8
        builder.default_shape_cfg.margin = 0.0015
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        builder.add_urdf(
            str(robot_path),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
        )
        # Use neutral shell and joint finishes for the otherwise unmodified CAD meshes.
        for shape, color in enumerate(builder.shape_color):
            r, g, b = color
            if g > 1.15 * r and g > 1.10 * b:
                builder.shape_color[shape] = (0.74, 0.78, 0.81)
            elif r > 1.15 * g and r > 1.05 * b:
                builder.shape_color[shape] = (0.19, 0.22, 0.26)
        self.robot_joint_count = builder.joint_count
        for body in range(builder.body_count):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        self.hand_shapes = []
        collision_mask = int(newton.ShapeFlags.COLLIDE_SHAPES | newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape in range(builder.shape_count):
            body = builder.shape_body[shape]
            label = builder.body_label[body].lower() if body >= 0 else ""
            if "right" in label and any(
                word in label for word in ("j7", "hand", "thumb", "index", "middle", "ring", "pinky")
            ):
                if builder.shape_flags[shape] & collision_mask:
                    self.hand_shapes.append(shape)
        # Closed convex parts retain the detailed hand silhouette while avoiding
        # concave collision pockets around the finger joints and screw recesses.
        thumb_shapes = [
            shape for shape in self.hand_shapes if "thumb" in builder.body_label[builder.shape_body[shape]].lower()
        ]
        builder.approximate_meshes(
            method="convex_hull", shape_indices=thumb_shapes, raise_on_failure=True, keep_visual_shapes=False
        )
        builder.approximate_meshes(
            method="vhacd",
            shape_indices=[shape for shape in self.hand_shapes if shape not in thumb_shapes],
            raise_on_failure=True,
            keep_visual_shapes=False,
            maxConvexHulls=8,
            resolution=200000,
            minimumVolumePercentErrorAllowed=1.0,
            maxRecursionDepth=8,
            maxNumVerticesPerCH=64,
            asyncACD=False,
        )
        self.hand_shapes = [
            shape
            for shape in range(builder.shape_count)
            if builder.shape_body[shape] >= 0
            and "right" in builder.body_label[builder.shape_body[shape]].lower()
            and any(
                word in builder.body_label[builder.shape_body[shape]].lower()
                for word in ("j7", "hand", "thumb", "index", "middle", "ring", "pinky")
            )
            and builder.shape_flags[shape] & collision_mask
        ]
        self.hand_body = next(i for i, name in enumerate(builder.body_label) if name.endswith("/right_j7"))
        posture = {"ANKLE": 55.0, "KNEE": -110.0, "BUTTOCK": 70.0}
        idle_left = {
            "LEFT_J1": -12.0,
            "LEFT_J2": -78.0,
            "LEFT_J3": -10.0,
            "LEFT_J4": -35.0,
            "LEFT_J5": 0.0,
            "LEFT_J6": 0.0,
            "LEFT_J7": 0.0,
        }
        self.idle_left_indices, self.idle_left_values = [], []
        self.finger_indices = []
        self.finger_names = []
        for j, name in enumerate(builder.joint_label):
            suffix = name.rsplit("/", 1)[-1]
            if suffix in posture:
                builder.joint_q[builder.joint_q_start[j]] = math.radians(posture[suffix])
            if suffix in idle_left:
                self.idle_left_indices.append(builder.joint_q_start[j])
                self.idle_left_values.append(math.radians(idle_left[suffix]))
            if suffix.startswith("LEFT_") and ("HAND_" in suffix or suffix.endswith("_PIP")):
                self.idle_left_indices.append(builder.joint_q_start[j])
                self.idle_left_values.append(
                    0.35 if suffix.endswith("THUMB2") else (0.18 if suffix.endswith("_PIP") else 0.12)
                )
            if suffix.endswith("HAND_THUMB2"):
                builder.joint_q[builder.joint_q_start[j]] = math.pi / 2.0
                if suffix.startswith("RIGHT_"):
                    self.thumb_opposition_index = builder.joint_q_start[j]
            if (
                suffix.startswith("RIGHT_")
                and any(s in suffix for s in ("HAND_", "_PIP"))
                and not suffix.endswith("THUMB2")
            ):
                self.finger_indices.append(builder.joint_q_start[j])
                self.finger_names.append(suffix)
        cfg = newton.ModelBuilder.ShapeConfig(ke=3.0e4, kd=10.0, mu=0.9, density=1500.0)
        self.ik_model = builder.finalize()
        self.ik_model.joint_label = list(self.ik_model.joint_label)
        self.ik_model.body_label = list(self.ik_model.body_label)
        self.robot_coord_count = builder.joint_coord_count
        ground = builder.add_ground_plane(color=(0.16, 0.19, 0.23))
        builder.shape_flags[ground] &= ~int(newton.ShapeFlags.VISIBLE)

        def box(position, half_size, color, *, body=-1, label=""):
            shape = builder.add_shape_box(
                body,
                xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
                hx=half_size[0],
                hy=half_size[1],
                hz=half_size[2],
                cfg=cfg,
                color=color,
                label=label,
            )
            if body == -1:
                builder.shape_flags[shape] &= ~int(newton.ShapeFlags.VISIBLE)
            return shape

        belt_bodies = []
        belt_shapes = []
        belt_collision = cfg.copy()
        belt_collision.is_visible = False
        visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        # Keep the section shape numbering used by the calibrated contact grasp.
        # A separate closed mesh renders the entire belt, including its return run.
        visual.is_visible = False
        yy, xx = np.indices((128, 256))
        rubber = 35 + 3 * ((xx + yy) % 3) + 5 * (yy % 32 < 2)
        belt_texture = np.uint8(np.stack((rubber * 0.85, rubber, rubber * 1.08), axis=-1))
        belt_surface = newton.Mesh(
            vertices=[(-0.15, -0.04, 0.0301), (0.15, -0.04, 0.0301), (0.15, 0.04, 0.0301), (-0.15, 0.04, 0.0301)],
            indices=[0, 1, 2, 0, 2, 3],
            uvs=[(0, 0), (1, 0), (1, 1), (0, 1)],
            compute_inertia=False,
            texture=belt_texture,
            roughness=0.92,
            metallic=0.0,
        )
        for i in range(25):
            body = builder.add_body(
                xform=wp.transform(wp.vec3(BELT_X, -0.24 + i * 0.08, 0.73), wp.quat_identity()),
                is_kinematic=True,
                label=f"belt_slat_{i}",
            )
            belt_bodies.append(body)
            shape = box((0, 0, 0), (0.15, 0.04, 0.03), (0.12, 0.15, 0.18) if i % 4 else (0.26, 0.31, 0.34), body=body)
            builder.shape_flags[shape] = 0
            builder.add_shape_mesh(body, mesh=belt_surface, cfg=visual, color=(1.0, 1.0, 1.0))
            # Overlapping collision sections form a continuous belt surface.
            # Butt-jointed boxes can wedge cloth between their vertical faces.
            belt_shapes.append(builder.add_shape_box(body, hx=0.175, hy=0.08, hz=0.03, cfg=belt_collision))
        for x in (0.425, 0.775):
            box((x, 0.76, 0.70), (0.025, 1.04, 0.08), (0.55, 0.61, 0.66))
            for y in (-0.14, 1.66):
                box((x, y, 0.33), (0.025, 0.035, 0.33), (0.34, 0.40, 0.47))
        for i, (x, y) in enumerate(TRAYS):
            tray_hx, tray_hy = TRAY_HALF_EXTENTS[i]
            box((x, y, 0.66), (tray_hx - 0.01, tray_hy - 0.01, 0.025), (0.28, 0.31, 0.34), label=f"{KINDS[i]}_tray")
            for dx, dy, hx, hy in (
                (0.01 - tray_hx, 0, 0.01, tray_hy),
                (tray_hx - 0.01, 0, 0.01, tray_hy),
                (0, 0.01 - tray_hy, tray_hx - 0.02, 0.01),
                (0, tray_hy - 0.01, tray_hx - 0.02, 0.01),
            ):
                box((x + dx, y + dy, 0.705), (hx, hy, 0.025), tuple(0.65 * c for c in COLORS[i]))
        # A notched workbench supports the trays and leaves a cloth pickup opening.
        box((0.16, -0.52, 0.62), (0.29, 0.32, 0.015), (0.48, 0.52, 0.55))
        box((0.625, -0.62, 0.62), (0.175, 0.22, 0.015), (0.48, 0.52, 0.55))
        for x in (-0.045, 0.715):
            for y in (-0.785, -0.255 if x < 0.45 else -0.45):
                box((x, y, 0.31), (0.022, 0.022, 0.295), (0.29, 0.34, 0.39))

        for i, kind in enumerate(KINDS):
            y = 0.12 + i * 0.34
            parcel = _Parcel(
                kind,
                np.array((*TRAYS[i], 0.81 if kind in ("soft", "inflatable") else 0.76), dtype=np.float32),
                tray_half_extent=TRAY_HALF_EXTENTS[i],
            )
            parcel.begin = builder.particle_count
            tri_start = len(builder.tri_indices)
            if kind == "rigid":
                parcel.body = builder.add_body(
                    xform=wp.transform(wp.vec3(BELT_X, y, 0.792), wp.quat_identity()), label="rigid_parcel"
                )
                box((0, 0, 0), (0.012, 0.027, 0.027), COLORS[i], body=parcel.body)
            elif kind == "soft":
                builder.add_soft_grid(
                    pos=wp.vec3(BELT_X - 0.012, y - 0.027, 0.768),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(),
                    dim_x=6,
                    dim_y=10,
                    dim_z=10,
                    cell_x=0.004,
                    cell_y=0.0054,
                    cell_z=0.0054,
                    density=300.0,
                    k_mu=1.0e6,
                    k_lambda=3.0e6,
                    k_damp=20.0,
                    particle_radius=0.004,
                    color=COLORS[i],
                )
            elif kind == "cloth":
                builder.add_cloth_grid(
                    pos=wp.vec3(BELT_X - 0.10, y - 0.10, 0.773),
                    rot=wp.quat_identity(),
                    vel=wp.vec3(),
                    dim_x=CLOTH_CELLS,
                    dim_y=CLOTH_CELLS,
                    cell_x=0.20 / CLOTH_CELLS,
                    cell_y=0.20 / CLOTH_CELLS,
                    mass=0.000033 * 441 / (CLOTH_CELLS + 1) ** 2,
                    tri_ke=500.0,
                    tri_ka=500.0,
                    tri_kd=0.002,
                    edge_ke=0.005,
                    edge_kd=0.001,
                    particle_radius=0.0015,
                    color=COLORS[i],
                )
            else:
                bag_mesh = bag_reference._load_chip_bag_mesh()
                volume = bag_reference._scaled_mesh_volume(bag_mesh, bag_reference.BAG_SCALE)
                add_inflatable_mesh(
                    builder,
                    pos=wp.vec3(BELT_X, y, BELT_TOP + 0.028),
                    rot=wp.quat_from_axis_angle(wp.vec3(1, 0, 0), math.pi * 0.5),
                    scale=bag_reference.BAG_SCALE,
                    vel=wp.vec3(),
                    vertices=bag_mesh.vertices,
                    indices=bag_mesh.indices,
                    density=bag_reference.BAG_DENSITY,
                    tri_ke=bag_reference.BAG_TRI_KE,
                    tri_ka=bag_reference.BAG_TRI_KA,
                    tri_kd=bag_reference.BAG_TRI_KD,
                    edge_ke=bag_reference.BAG_EDGE_KE,
                    edge_kd=bag_reference.BAG_EDGE_KD,
                    particle_radius=bag_reference.BAG_PARTICLE_RADIUS,
                    color=COLORS[i],
                    config=bag_reference._make_pneumatic_config("isothermal", volume),
                )
            parcel.end = builder.particle_count
            parcel.triangles = np.asarray(builder.tri_indices[tri_start:], dtype=np.int32).reshape(-1)
            self.parcels.append(parcel)
        self._add_station_details(builder)
        rollers = []
        for y in (-0.24, 1.76):
            body = builder.add_body(
                xform=wp.transform(wp.vec3(BELT_X, y, 0.708), wp.quat_identity()),
                is_kinematic=True,
                label="belt_roller",
            )
            rollers.append(body)
            builder.add_shape_cylinder(
                body,
                xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.pi / 2)),
                radius=0.0521,
                half_height=0.175,
                cfg=belt_collision,
            )
        builder.color()
        self.model = builder.finalize()
        self.belt_bodies = wp.array(belt_bodies, dtype=int, device=self.model.device)
        self.belt_collision_shapes = wp.array(belt_shapes, dtype=int, device=self.model.device)
        self.belt_rollers = wp.array(rollers, dtype=int, device=self.model.device)
        for parcel in self.parcels:
            parcel.triangles = wp.array(parcel.triangles, dtype=int, device=self.model.device)
        self.model.soft_contact_ke = 2.0e5
        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_mu = 0.6

    def _build_materials(self):
        """Give the fabric a woven pattern and the sealed bag a printed wrapper."""
        positions = self.model.particle_q.numpy()
        yy, xx = np.indices((256, 256))
        self.render_transform = wp.array([wp.transform_identity()], dtype=wp.transform, device=self.model.device)
        self.render_scale = wp.array([wp.vec3(1.0)], dtype=wp.vec3, device=self.model.device)
        self.render_color = wp.array([wp.vec3(1.0)], dtype=wp.vec3, device=self.model.device)
        self.render_materials = {
            "cloth": wp.array([wp.vec4(0.9, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=self.model.device),
            "inflatable": wp.array([wp.vec4(0.35, 0.12, 0.0, 1.0)], dtype=wp.vec4, device=self.model.device),
            "station": wp.array([wp.vec4(0.85, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=self.model.device),
        }
        for index in (2, 3):
            parcel = self.parcels[index]
            local = positions[parcel.begin : parcel.end, :2]
            uv = np.zeros((self.model.particle_count, 2), dtype=np.float32)
            uv[parcel.begin : parcel.end] = (local - local.min(axis=0)) / np.ptp(local, axis=0)
            parcel.uvs = wp.array(uv, dtype=wp.vec2, device=self.model.device)
            if parcel.kind == "cloth":
                weave = 0.82 + 0.12 * ((xx + yy) % 2) + 0.06 * (xx % 3 == 0)
                pattern = np.ones((256, 256, 3)) * np.array(COLORS[index])
                pattern[(xx % 64 < 5) | (yy % 64 < 5)] = (0.80, 0.88, 0.94)
                hem = (xx < 5) | (xx > 250) | (yy < 5) | (yy > 250)
                pattern[hem] *= 0.65
                stitches = (((xx == 3) | (xx == 252)) & (yy % 6 < 3)) | (((yy == 3) | (yy == 252)) & (xx % 6 < 3))
                pattern[stitches] = (0.72, 0.80, 0.88)
                parcel.texture = np.uint8(255 * pattern * weave[..., None])
            else:
                pattern = np.ones((256, 256, 3)) * np.array(COLORS[index])
                pattern[55:200, 45:210] = (0.92, 0.89, 0.84)
                pattern[70:85, 65:188] = COLORS[index]
                pattern[94:102, 65:168] = COLORS[index]
                pattern[115:120, 65:145] = (0.28, 0.26, 0.30)
                for column in range(66, 184, 5):
                    pattern[152:184, column : column + 2 + column % 3] = (0.12, 0.11, 0.14)
                wrapper = Image.fromarray(np.uint8(255 * pattern))
                draw = ImageDraw.Draw(wrapper)
                draw.text((65, 128), "SEALED / 04", font=ImageFont.load_default(size=12), fill=(45, 40, 48))
                parcel.texture = np.asarray(wrapper)

        self.decals = []

        def decal(name, points, texture, roughness=0.85, repeats=1.0):
            self.decals.append(
                (
                    name,
                    wp.array(points, dtype=wp.vec3, device=self.model.device),
                    wp.array([0, 1, 2, 0, 2, 3], dtype=int, device=self.model.device),
                    wp.array(
                        [(0, 0), (repeats, 0), (repeats, repeats), (0, repeats)],
                        dtype=wp.vec2,
                        device=self.model.device,
                    ),
                    texture,
                    roughness,
                )
            )

        # Deterministic fine aggregate and expansion joints, at world scale.
        rng = np.random.default_rng(42)
        yy, xx = np.indices((1024, 1024))
        grain = rng.normal(0.0, 1.5, (1024, 1024))
        concrete = np.clip(116 + grain, 0, 255)
        concrete[(xx % 128 < 1) | (yy % 128 < 1)] *= 0.78
        floor = np.uint8(np.stack((concrete * 0.96, concrete, concrete * 1.03), axis=-1))
        decal("floor", [(-64, -64, 0.0), (64, -64, 0.0), (64, 64, 0.0), (-64, 64, 0.0)], floor, repeats=8.0)
        for i, (x, y) in enumerate(TRAYS):
            panel = Image.new("RGB", (512, 128), (26, 33, 40))
            draw = ImageDraw.Draw(panel)
            draw.rectangle((0, 0, 14, 127), fill=tuple(int(255 * c) for c in COLORS[i]))
            draw.text(
                (32, 14), f"0{i + 1}  {KINDS[i].upper()}", font=ImageFont.load_default(size=38), fill=(229, 234, 238)
            )
            draw.text((34, 76), "MATERIAL SORTING", font=ImageFont.load_default(size=22), fill=(152, 170, 182))
            decal(
                f"tray_{i}_label",
                [
                    (x - 0.095, y - 0.1282, 0.681),
                    (x + 0.095, y - 0.1282, 0.681),
                    (x + 0.095, y - 0.1282, 0.728),
                    (x - 0.095, y - 0.1282, 0.728),
                ],
                np.asarray(panel),
            )
        panel = Image.new("RGB", (768, 128), (28, 37, 45))
        draw = ImageDraw.Draw(panel)
        draw.text((28, 16), "W1 / MATERIAL SORTING", font=ImageFont.load_default(size=43), fill=(226, 232, 236))
        draw.text(
            (30, 80),
            "CONTACT-DRIVEN HANDLING    /    CELL 01",
            font=ImageFont.load_default(size=24),
            fill=(147, 175, 187),
        )
        decal(
            "station_label",
            [(0.804, 0.28, 0.68), (0.804, 0.88, 0.68), (0.804, 0.88, 0.78), (0.804, 0.28, 0.78)],
            np.asarray(panel),
        )

    @staticmethod
    def _add_station_details(builder):
        """Load Blender-authored surfaces over the validated collision geometry."""
        path = ROBOT_URDF.parents[1] / "conveyor_station/station.npz"
        visual = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            has_shape_collision=False,
            has_particle_collision=False,
        )
        with np.load(path, allow_pickle=False) as asset:
            for i, (r, g, b, roughness, metallic) in enumerate(asset["materials"]):
                mesh = newton.Mesh(
                    vertices=asset[f"vertices_{i}"],
                    indices=asset[f"indices_{i}"],
                    normals=asset[f"normals_{i}"],
                    compute_inertia=False,
                    roughness=float(roughness),
                    metallic=float(metallic),
                )
                builder.add_shape_mesh(
                    -1,
                    mesh=mesh,
                    cfg=visual,
                    color=(float(r), float(g), float(b)),
                    label=f"station_material_{i}",
                )

    def _load_grasps(self):
        yaw = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), -0.5 * math.pi)
        hand_frame = wp.quat(0.5, -0.5, 0.5, 0.5)
        self.grasp_offsets, self.grasp_rotations, self.grasp_fingers = [], [], []
        for kind in KINDS:
            suffix = {"rigid": "rigid_cube_", "soft": "", "cloth": "", "inflatable": "inflatable_bag_"}[kind]
            path = ROBOT_URDF.parents[1] / "vbd_mjvbd_v2" / f"vbd_w1_right_hand_{suffix}last_keyframe.json"
            key = json.loads(path.read_text())["keyframe"]
            root = key["target_root_pose"]
            reference_center = np.array((-0.14931439, -2.76669516, 1.20422798))
            position = np.array(root["position_m"])
            if kind == "inflatable":
                position[2] -= 0.033
                reference_center = np.asarray(bag_reference.BAG_CENTER, dtype=np.float64)
            self.grasp_offsets.append(np.asarray(wp.quat_rotate(yaw, wp.vec3(*(position - reference_center)))))
            self.grasp_rotations.append(yaw * wp.quat(*root["quaternion_xyzw"]) * wp.quat_inverse(hand_frame))
            fingers = key["target_finger_joints_degrees"]
            self.grasp_fingers.append(
                1.07 * np.radians([fingers[name] for name in self.finger_names]).astype(np.float32)
            )
        self.grasp_fingers[3] *= 1.10
        self.grasp_offsets[3][2] += 0.006
        # Reorient the recorded pinch to approach the hanging cloth below the roller.
        cloth_rotation = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), math.radians(10.0)) * wp.quat(
            0.0245, 0.6878, 0.7139, -0.1294
        )
        self.grasp_rotations[2] = cloth_rotation
        pinch = {"RIGHT_HAND_THUMB1": 0.61, "RIGHT_HAND_INDEX": 1.02, "RIGHT_INDEX_PIP": 0.85}
        self.grasp_fingers[2] = np.array([pinch.get(name, 0.0) for name in self.finger_names], dtype=np.float32)
        # Calibrated thumb/index contact midpoint in the wrist frame [m].
        pinch_local = wp.vec3(-0.18888462, -0.04892516, 0.03359285)
        self.cloth_tcp_from_pinch = np.asarray(wp.quat_rotate(cloth_rotation, TCP_OFFSET - pinch_local))

    def _build_ik(self):
        q = self.ik_model.joint_q.numpy()
        lower, upper = self.ik_model.joint_limit_lower.numpy(), self.ik_model.joint_limit_upper.numpy()
        starts, dofs = self.ik_model.joint_q_start.numpy(), self.ik_model.joint_qd_start.numpy()
        locked = []
        for joint, label in enumerate(self.ik_model.joint_label):
            if label.rsplit("/", 1)[-1] in {f"{side}_J{i}" for side in ("LEFT", "RIGHT") for i in range(1, 8)}:
                lower[dofs[joint]] = self.motion_lower[starts[joint]]
                upper[dofs[joint]] = self.motion_upper[starts[joint]]
                continue
            if starts[joint + 1] > starts[joint]:
                idx = starts[joint]
                locked.append(idx)
                lower[dofs[joint]], upper[dofs[joint]] = q[idx] - 1.0e-5, q[idx] + 1.0e-5
        self.lock_indices = wp.array(locked, dtype=int, device=self.ik_model.device)
        self.lock_values = wp.array(q[locked], dtype=float, device=self.ik_model.device)
        waist = next(j for j, name in enumerate(self.ik_model.joint_label) if name.endswith("/WAIST"))
        self.waist_index = int(starts[waist])
        self.waist_dof = int(dofs[waist])
        self.waist_lock = locked.index(self.waist_index)
        self.waist_angle = float(q[self.waist_index])
        self.waist_velocity = 0.0
        self.ik_lower, self.ik_upper = lower, upper
        self.lock_targets = q[locked].copy()
        stance_joints = [
            next(j for j, name in enumerate(self.ik_model.joint_label) if name.endswith("/" + part))
            for part in ("ANKLE", "KNEE", "BUTTOCK")
        ]
        self.stance_indices = starts[stance_joints]
        self.stance_dofs = dofs[stance_joints]
        self.stance_locks = np.asarray([locked.index(i) for i in self.stance_indices])
        self.stance_reference = q[self.stance_indices].copy()
        self.stance_angle = self.stance_velocity = 0.0
        self.position_objective = ik.IKObjectivePosition(
            self.hand_body, TCP_OFFSET, wp.array([HOME], dtype=wp.vec3, device=self.ik_model.device)
        )
        self.rotation_objective = ik.IKObjectiveRotation(
            self.hand_body,
            wp.quat_identity(),
            wp.array([wp.vec4(*self.grasp_rotations[0])], dtype=wp.vec4, device=self.ik_model.device),
            weight=1.0,
        )
        # Preserve the recorded grasp's bimanual IK reference. The idle left
        # arm is prescribed separately so its posture cannot perturb the right
        # arm's redundant IK solution and the contact-only cloth pinch.
        left_body = next(i for i, name in enumerate(self.ik_model.body_label) if name.endswith("/left_j7"))
        left_position = ik.IKObjectivePosition(
            left_body,
            wp.vec3(-0.18, 0, 0),
            wp.array([wp.vec3(0.35, 0.30, 0.95)], dtype=wp.vec3, device=self.model.device),
        )
        left_rotation = ik.IKObjectiveRotation(
            left_body,
            wp.quat_identity(),
            wp.array([wp.vec4(0.60942, 0.30651, 0.72839, -0.06413)], dtype=wp.vec4, device=self.model.device),
        )
        limits = ik.IKObjectiveJointLimit(
            wp.array(lower, device=self.ik_model.device), wp.array(upper, device=self.ik_model.device), weight=30.0
        )
        self.ik_limits = limits
        self.ik_solver = ik.IKSolver(
            self.ik_model,
            n_problems=1,
            objectives=[self.position_objective, self.rotation_objective, left_position, left_rotation, limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_q = wp.clone(self.ik_model.joint_q).reshape((1, -1))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=300)
        wp.launch(_lock_coordinates, len(locked), [self.lock_indices, self.lock_values, self.ik_q])
        self.ik_reference_q = wp.clone(self.ik_q)
        self.ik_home_seed = wp.clone(self.ik_q)
        wp.copy(self.model.joint_q, self.ik_q[0], count=self.robot_coord_count)
        initial = self.model.joint_q.numpy()
        initial[self.idle_left_indices] = self.idle_left_values
        initial[: self.robot_coord_count] = np.clip(
            initial[: self.robot_coord_count], self.motion_lower, self.motion_upper
        )
        self.model.joint_q.assign(initial)
        neck_joints = [
            next(j for j, name in enumerate(self.model.joint_label) if name.endswith("/" + neck))
            for neck in ("NECK1", "NECK2")
        ]
        self.neck_indices = np.asarray([starts[j] for j in neck_joints])
        self.neck_lower = self.model.joint_limit_lower.numpy()[[dofs[j] for j in neck_joints]]
        self.neck_upper = self.model.joint_limit_upper.numpy()[[dofs[j] for j in neck_joints]]
        self.neck_angles = self.model.joint_q.numpy()[self.neck_indices].copy()
        self.neck_velocity = np.zeros(2)
        self.neck_parent = int(self.model.joint_parent.numpy()[neck_joints[0]])
        self.neck_origin = wp.transform(*self.model.joint_X_p.numpy()[neck_joints[0]])

    def _turn_waist(self):
        """Turn toward the receiving trays while preserving the world-space grasp."""
        placing = self.phase in ("transfer", "lower", "release", "drop_wait", "clear")
        desired = math.radians(-30.0 if self.active == 2 else -15.0) if placing and self.active in (2, 3) else 0.0
        velocity = float(np.clip((desired - self.waist_angle) / 0.35, -0.45, 0.45))
        self.waist_velocity += float(np.clip(velocity - self.waist_velocity, -0.8 * self.frame_dt, 0.8 * self.frame_dt))
        self.waist_angle += self.waist_velocity * self.frame_dt

    def _lower_body(self):
        """Lower the shoulders for the hanging cloth while preserving torso pitch."""
        desired = math.radians(10.0) if self.active == 2 else 0.0
        velocity = float(np.clip((desired - self.stance_angle) / 0.35, -0.2, 0.2))
        self.stance_velocity += float(
            np.clip(velocity - self.stance_velocity, -0.4 * self.frame_dt, 0.4 * self.frame_dt)
        )
        self.stance_angle += self.stance_velocity * self.frame_dt

    def _set_waist_target(self, angle):
        self.lock_targets[self.stance_locks] = self.stance_reference + self.stance_angle * np.array((1, -2, 1))
        self.ik_lower[self.stance_dofs] = self.lock_targets[self.stance_locks] - 1.0e-5
        self.ik_upper[self.stance_dofs] = self.lock_targets[self.stance_locks] + 1.0e-5
        self.lock_targets[self.waist_lock] = angle
        self.lock_values.assign(self.lock_targets)
        self.ik_lower[self.waist_dof] = angle - 1.0e-5
        self.ik_upper[self.waist_dof] = angle + 1.0e-5
        self.ik_limits.joint_limit_lower.assign(self.ik_lower)
        self.ik_limits.joint_limit_upper.assign(self.ik_upper)

    def _track_parcel(self, end):
        """Aim the head at the parcel with bounded neck speed and acceleration."""
        parent = wp.transform(*self.state_0.body_q.numpy()[self.neck_parent])
        neck = parent * self.neck_origin
        center = self._center(self.parcels[min(self.active, len(self.parcels) - 1)])
        local = np.asarray(wp.transform_point(wp.transform_inverse(neck), wp.vec3(*center)))
        # NECK2 sits 81 mm above the yaw pivot. Its positive angle looks up.
        local[2] -= 0.081
        desired = np.array((math.atan2(local[1], local[0]), math.atan2(local[2], math.hypot(local[0], local[1]))))
        desired = np.clip(desired, self.neck_lower, self.neck_upper)
        velocity = np.clip((desired - self.neck_angles) / 0.20, -0.70, 0.70)
        self.neck_velocity += np.clip(velocity - self.neck_velocity, -2.0 * self.frame_dt, 2.0 * self.frame_dt)
        self.neck_angles = np.clip(
            self.neck_angles + self.neck_velocity * self.frame_dt, self.neck_lower, self.neck_upper
        )
        end[self.neck_indices] = self.neck_angles

    def _center(self, parcel):
        if parcel.body >= 0:
            return self.state_0.body_q.numpy()[parcel.body, :3]
        return self.state_0.particle_q.numpy()[parcel.begin : parcel.end].mean(axis=0)

    def _pickup_y(self):
        return -0.26 if self.active == 2 else PICK_Y

    def _enter(self, phase):
        self.phase, self.phase_time = phase, 0.0
        self.motion_start = self.target.copy()
        if phase == "feed":
            self.feed_rotation = self.rotation
            # Recover a known empty-hand branch after the cloth's rotated
            # grasp. Only the IK seed changes; executed joints remain retimed.
            if self.active == 3:
                wp.copy(self.ik_reference_q, self.ik_home_seed)

    def _grasp(self, parcel):
        self.gripping = parcel.picked = True
        self.pick_height = float(self._center(parcel)[2])
        hand = wp.transform(*self.state_0.body_q.numpy()[self.hand_body])
        tcp = np.asarray(wp.transform_point(hand, TCP_OFFSET))
        self.grip_offset = tcp - self._center(parcel)
        self._set_place_target(parcel)

    def _set_place_target(self, parcel):
        self.place_target = parcel.destination + self.grip_offset
        tilt = self._release_tilt(1.0)
        self.release_target = parcel.destination + np.asarray(wp.quat_rotate(tilt, wp.vec3(*self.grip_offset)))
        self.release_rotation = tilt * self.grasp_rotations[self.active]

    @staticmethod
    def _release_tilt(fraction):
        return wp.quat_from_axis_angle(wp.vec3(0, 1, 0), math.radians(40.0) * fraction)

    def _controller(self):
        if self.completed:
            return 0.0, 0.0
        parcel = self.parcels[self.active]
        self.rotation = self.grasp_rotations[self.active]
        self.phase_time += self.frame_dt
        if self.phase == "feed":
            fraction = min(self.phase_time / 2.5, 1.0)
            blend = fraction * fraction * (3.0 - 2.0 * fraction)
            self.rotation = wp.quat_slerp(self.feed_rotation, self.grasp_rotations[self.active], blend)
            center = self._center(parcel)
            if center[1] > self._pickup_y():
                if self.phase_time > 40.0:
                    raise RuntimeError(f"{parcel.kind} did not reach the pickup station")
                return self.args.belt_speed, 0.0
            if fraction < 1.0:
                # The next parcel can arrive before the empty hand finishes
                # reorienting. Stop the belt, but complete that motion first.
                return 0.0, 0.0
            self._enter("settle")
        if self.phase == "settle":
            if self.phase_time >= 0.35:
                center = self._center(parcel).copy()
                self.pick_target = center + self.grasp_offsets[self.active]
                if parcel.kind == "cloth":
                    # Target a cloth node: this contact model resolves particle/shape contacts.
                    corner = self.state_0.particle_q.numpy()[parcel.begin + CLOTH_GRASP_NODE]
                    self.pick_target = corner + self.cloth_tcp_from_pinch
                self._enter("approach")
            return 0.0, 0.0
        durations = {
            "approach": 1.4,
            "close": 1.2,
            "lift": 2.0 if parcel.kind == "cloth" else 1.2,
            "transfer": 3.0 if parcel.kind == "cloth" else 1.8,
            "lower": 1.4 if parcel.kind == "cloth" else 1.0,
            "release": 1.2,
            "drop_wait": 1.0,
            "clear": 1.6,
            "retreat": 1.2,
        }
        duration = durations[self.phase]
        fraction = min(self.phase_time / duration, 1.0)
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        destinations = {
            "approach": self.pick_target,
            "close": self.pick_target,
            "lift": self.pick_target + np.array((0, 0, 0.45 if parcel.kind == "cloth" else 0.24)),
            "transfer": self.place_target + np.array((0, 0, 0.22)),
            "lower": self.place_target,
            "release": self.release_target,
            "drop_wait": self.release_target,
            "clear": self.release_target + np.array((-0.12, 0, 0.18)),
            "retreat": HOME,
        }
        self.target = (1.0 - smooth) * self.motion_start + smooth * destinations[self.phase]
        if parcel.kind == "cloth" and self.phase == "approach":
            opened = self.pick_target + CLOTH_OPEN_CORRECTION
            below = opened + np.array((0, 0, -0.035))
            front = below + np.array((0, -0.04, 0))
            high = front.copy()
            high[2] = max(self.motion_start[2], self.pick_target[2] + 0.16)
            self.target = interpolate_waypoints((self.motion_start, high, front, below, opened), fraction)
        elif parcel.kind == "cloth" and self.phase == "lift":
            peel = self.pick_target + np.array((0, -0.04, 0.0))
            raised = self.pick_target + np.array((0, -0.04, 0.45))
            self.target = interpolate_waypoints((self.motion_start, peel, raised), fraction)
        closure = (
            smooth if self.phase == "close" else (1.0 - smooth if self.phase == "release" else float(self.gripping))
        )
        if parcel.kind == "cloth" and self.phase in ("approach", "close"):
            closure = 0.8 * min(1.0, 3.0 * smooth) if self.phase == "approach" else 0.8 + 0.2 * smooth
            if self.phase == "close":
                self.target = self.pick_target + (1.0 - smooth) * CLOTH_OPEN_CORRECTION
        elif parcel.kind == "cloth" and self.gripping:
            closure = 1.0
        if self.phase == "release":
            closure = max(0.0, 1.0 - 3.0 * fraction)
            tilt = self._release_tilt(smooth)
            self.rotation = tilt * self.grasp_rotations[self.active]
            self.target = parcel.destination + np.asarray(wp.quat_rotate(tilt, wp.vec3(*self.grip_offset)))
        elif self.phase in ("drop_wait", "clear"):
            self.rotation = self.release_rotation
        elif self.phase == "retreat":
            self.rotation = self.release_rotation
        if self.gripping:
            parcel.lift = max(parcel.lift, float(self._center(parcel)[2]) - self.pick_height)
        if fraction >= 1.0:
            if self.phase == "close":
                self._grasp(parcel)
            if self.phase == "lift" and float(self._center(parcel)[2]) - self.pick_height < 0.10:
                raise RuntimeError(f"{parcel.kind} grasp failed: parcel did not leave the belt")
            if self.phase == "lift" and parcel.kind == "cloth":
                hand = wp.transform(*self.state_0.body_q.numpy()[self.hand_body])
                self.grip_offset = np.asarray(wp.transform_point(hand, TCP_OFFSET)) - self._center(parcel)
                self._set_place_target(parcel)
            if self.phase == "lift" and parcel.kind == "inflatable":
                # Recenter the deformed bag while preserving the calibrated
                # wrist height needed for opening clearance.
                hand = wp.transform(*self.state_0.body_q.numpy()[self.hand_body])
                self.grip_offset[:2] = np.asarray(wp.transform_point(hand, TCP_OFFSET))[:2] - self._center(parcel)[:2]
                self._set_place_target(parcel)
            if self.phase == "lower":
                self.gripping = False
            if self.phase == "release":
                parcel.released = True
            if self.phase == "retreat":
                self._check_placement(parcel)
                parcel.sorted = True
                self.active += 1
                if self.active == len(self.parcels):
                    self.completed = True
                    self.phase = "complete"
                    print("Sorted all four parcels: rigid, soft, cloth, inflatable.")
                else:
                    self._enter("feed")
            else:
                phases = tuple(durations)
                self._enter(phases[phases.index(self.phase) + 1])
        return 0.0, closure

    def _plan_motion(self):
        speed, closure = self._controller()
        self._turn_waist()
        self._lower_body()
        self.position_objective.set_target_position(0, wp.vec3(*self.target))
        self.rotation_objective.set_target_rotation(0, wp.vec4(*self.rotation))
        # Keep the original grasp branch independent of the temporary torso turn.
        self._set_waist_target(0.0)
        self.ik_solver.step(self.ik_reference_q, self.ik_reference_q, iterations=self.args.ik_iterations)
        wp.launch(_lock_coordinates, len(self.lock_indices), [self.lock_indices, self.lock_values, self.ik_reference_q])
        if self.phase in ("feed", "settle"):
            # Recover the grasp branch gradually while the hand is empty.
            self.ik_q.assign(0.95 * self.ik_q.numpy() + 0.05 * self.ik_reference_q.numpy())
        self._set_waist_target(self.waist_angle)
        wp.launch(_lock_coordinates, len(self.lock_indices), [self.lock_indices, self.lock_values, self.ik_q])
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=self.args.ik_iterations)
        wp.launch(_lock_coordinates, len(self.lock_indices), [self.lock_indices, self.lock_values, self.ik_q])
        end = self.state_0.joint_q.numpy()
        end[: self.robot_coord_count] = self.ik_q.numpy().reshape(-1)
        end[self.finger_indices] = closure * self.grasp_fingers[min(self.active, 3)]
        thumb_target = CLOTH_THUMB_OPPOSITION if self.active == 2 else math.pi / 2.0
        # Open cloth and bag grasps without sweeping the thumb opposition
        # joint through the parcel. Reorient it after the hand has retreated.
        if self.active not in (2, 3) and self.phase in ("release", "drop_wait", "clear", "retreat", "complete"):
            thumb_target *= closure
        self.thumb_angle += float(np.clip(thumb_target - self.thumb_angle, -math.pi / 120.0, math.pi / 120.0))
        end[self.thumb_opposition_index] = self.thumb_angle
        end[self.idle_left_indices] = self.idle_left_values
        start = self.state_0.joint_q.numpy()[: self.robot_coord_count]
        end[self.neck_indices] = start[self.neck_indices]
        goal = np.clip(end[: self.robot_coord_count], self.motion_lower, self.motion_upper)
        self.motion_retimer.begin(start, goal)
        self.motion_belt_speed = speed

    def step(self):
        if self.motion_retimer.remaining == 0:
            self._plan_motion()
        wp.copy(self.frame_start, self.state_0.joint_q)
        end = self.state_0.joint_q.numpy()
        end[: self.robot_coord_count] = self.motion_retimer.advance()
        self._track_parcel(end)
        self.frame_end.assign(end)
        previous = self.frame_start.numpy()[: self.robot_coord_count]
        ratio = np.abs(end[: self.robot_coord_count] - previous) / (self.motion_speed * self.frame_dt)
        self.motion_peak_speed_ratio = max(self.motion_peak_speed_ratio, float(np.max(ratio)))
        arm = end[self.arm_indices]
        margin = np.minimum(arm - self.motion_lower[self.arm_indices], self.motion_upper[self.arm_indices] - arm)
        self.motion_min_arm_margin = min(self.motion_min_arm_margin, float(np.min(margin)) + math.radians(3.0))
        speed = self.motion_belt_speed
        if self.phase == "feed":
            distance = float(self._center(self.parcels[self.active])[1]) - self._pickup_y()
            speed = min(speed, max(0.0, distance / self.frame_dt))
        self.belt_motion.assign(np.array((self.belt_offset, speed), dtype=np.float32))
        cloth = self.parcels[2]
        settle_cloth = cloth.sorted and self.cloth_displacement_threshold > 0.0
        if settle_cloth:
            wp.copy(self.cloth_previous, self.state_0.particle_q, src_offset=cloth.begin, count=cloth.end - cloth.begin)
        if self.graph is None:
            self._simulate()
            if self.use_graph:
                backup_0, backup_1 = self.model.state(), self.model.state()
                backup_0.assign(self.state_0)
                backup_1.assign(self.state_1)
                with wp.ScopedCapture() as capture:
                    self._simulate()
                self.state_0.assign(backup_0)
                self.state_1.assign(backup_1)
                self.graph = capture.graph
        else:
            wp.capture_launch(self.graph)
        if settle_cloth:
            wp.launch(
                _discard_cloth_microsteps,
                cloth.end - cloth.begin,
                [
                    cloth.begin,
                    wp.vec3(*cloth.destination),
                    self.cloth_displacement_threshold,
                    self.frame_dt,
                    self.cloth_previous,
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                ],
            )
        self.belt_offset += speed * self.frame_dt
        self.sim_time += self.frame_dt
        cloth = self.parcels[2]
        if cloth.released:
            velocities = self.state_0.particle_qd.numpy()[cloth.begin : cloth.end]
            self.cloth_speed_samples.append(float(np.sqrt(np.mean(np.sum(velocities**2, axis=1)))))

    def _simulate(self):
        for substep in range(self.args.substeps):
            wp.launch(
                _prescribe_robot,
                self.robot_joint_count,
                [
                    self.frame_start,
                    self.frame_end,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    (substep + 1) / self.args.substeps,
                    float(FPS),
                    self.state_0.joint_q,
                    self.state_0.joint_qd,
                ],
            )
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )
            wp.launch(
                _move_belt,
                len(self.belt_bodies),
                [
                    self.belt_bodies,
                    self.belt_collision_shapes,
                    self.model.shape_scale,
                    self.model.shape_transform,
                    self.belt_motion,
                    (substep + 1) * self.sim_dt,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                ],
            )
            wp.launch(
                _move_rollers,
                2,
                [
                    self.belt_rollers,
                    self.belt_motion,
                    (substep + 1) * self.sim_dt,
                    self.state_0.body_q,
                    self.state_0.body_qd,
                ],
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.belt_visual.render(self.viewer, self.belt_offset)
        for name, points, indices, uvs, texture, roughness in self.decals:
            self.viewer.log_mesh(
                f"/station/{name}",
                points,
                indices,
                uvs=uvs,
                texture=texture,
                hidden=True,
                roughness=roughness,
                metallic=0.0,
                backface_culling=False,
            )
            self.viewer.log_instances(
                f"/station/{name}/material",
                f"/station/{name}",
                self.render_transform,
                self.render_scale,
                self.render_color,
                self.render_materials["station"],
            )
        for i, parcel in enumerate(self.parcels):
            if parcel.body < 0:
                self.viewer.log_mesh(
                    f"/{parcel.kind}/surface",
                    self.state_0.particle_q,
                    parcel.triangles,
                    color=COLORS[i],
                    uvs=parcel.uvs,
                    texture=parcel.texture,
                    hidden=parcel.texture is not None,
                    roughness=0.35 if parcel.kind == "inflatable" else 0.9,
                    metallic=0.12 if parcel.kind == "inflatable" else 0.0,
                    backface_culling=False,
                )
                if parcel.texture is not None:
                    self.viewer.log_instances(
                        f"/{parcel.kind}/material",
                        f"/{parcel.kind}/surface",
                        self.render_transform,
                        self.render_scale,
                        self.render_color,
                        self.render_materials[parcel.kind],
                    )
        self.viewer.end_frame()

    def _render_ui(self, imgui):
        imgui.text("W1 conveyor sorting")
        imgui.text(f"Station: {self.phase.replace('_', ' ').title()}")
        imgui.text(f"Sorted: {sum(parcel.sorted for parcel in self.parcels)} / 4")
        if self.cloth_speed_samples:
            imgui.text(f"Cloth motion: {1000 * np.mean(self.cloth_speed_samples):.2f} mm/s")
        imgui.separator()
        for parcel in self.parcels:
            status = (
                "Placed"
                if parcel.sorted
                else ("Releasing" if parcel.released else ("In hand" if parcel.picked else "On belt"))
            )
            imgui.text(f"{parcel.kind.title()}: {status}")

    def _check_placement(self, parcel):
        center = self._center(parcel)
        allowance = np.asarray(parcel.tray_half_extent) - 0.02
        if np.any(np.abs(center[:2] - parcel.destination[:2]) > allowance) or not 0.68 < center[2] < 0.82:
            raise AssertionError(f"{parcel.kind} did not settle in its tray: {center}")
        if parcel.body < 0:
            positions = self.state_0.particle_q.numpy()[parcel.begin : parcel.end]
            inside = np.all(np.abs(positions[:, :2] - parcel.destination[:2]) <= parcel.tray_half_extent, axis=1)
            if np.mean(inside) < 0.90:
                raise AssertionError(f"{parcel.kind} extends too far outside its tray")

    def test_post_step(self):
        """Check finite dynamics and keep every parcel above the floor."""
        for values in (self.state_0.body_q.numpy(), self.state_0.particle_q.numpy(), self.state_0.particle_qd.numpy()):
            if not np.isfinite(values).all():
                raise AssertionError("Sorting produced a non-finite simulation state")
        if self.motion_peak_speed_ratio > 1.0001:
            raise AssertionError(f"Robot exceeds its configured joint speeds: {self.motion_peak_speed_ratio:.4f}")
        if self.motion_min_arm_margin < math.radians(3.0) - 1e-5:
            raise AssertionError("Robot arm violates its 3-degree joint-limit margin")
        if self.robot_clearance is None:
            self.robot_clearance = RobotClearanceAudit(self.model, self.ik_model.body_count)
        separation = self.robot_clearance.minimum_separation(self.state_0)
        self.robot_min_separation = min(self.robot_min_separation, separation)
        if separation < -0.001:
            contact = self.robot_clearance.inspect(self.state_0)[0]
            raise AssertionError(
                f"Robot collision in {self.phase}: {self.model.shape_label[contact[1]]} / "
                f"{self.model.shape_label[contact[2]]}, separation {contact[0]:.4f} m"
            )
        for parcel in self.parcels:
            if self._center(parcel)[2] < 0.55:
                raise AssertionError(f"{parcel.kind} fell off the conveyor or missed its tray")

    def test_final(self):
        """Verify all four released parcels settle into their assigned trays."""
        self.test_post_step()
        if self.solver.features.backend != "vbd_kinematic_full":
            raise AssertionError("Sorting requires the full kinematic MJVBDV2 backend")
        if not self.completed:
            raise AssertionError("Sorting is incomplete; increase --num-frames (9000 at the default belt speed)")
        for parcel in self.parcels:
            self._check_placement(parcel)
            if not parcel.sorted or not parcel.released or parcel.lift < 0.12:
                raise AssertionError(f"{parcel.kind} was not lifted and released")
        volume = float(self.state_0.pneumatic.volume.numpy()[0])
        if not 0.4 < volume / self.rest_volume < 1.6:
            raise AssertionError("The pneumatic parcel lost its inflated volume")
        speed = float(np.mean(self.cloth_speed_samples)) if self.cloth_speed_samples else math.inf
        if len(self.cloth_speed_samples) < FPS or not math.isfinite(speed) or speed > 0.002:
            raise AssertionError("The released cloth has not settled below 2 mm/s RMS over the last second")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=9000)
        parser.add_argument("--robot-urdf", default=str(ROBOT_URDF))
        parser.add_argument("--belt-speed", type=float, default=0.12, help="Indexed conveyor speed [m/s], 0.02-0.20.")
        parser.add_argument("--substeps", type=int, default=8)
        parser.add_argument("--vbd-iterations", type=int, default=16)
        parser.add_argument("--ik-iterations", type=int, default=24)
        parser.add_argument(
            "--finger-speed",
            type=float,
            default=90.0,
            help="Demo finger speed [deg/s]; URDF finger velocities are unspecified (zero).",
        )
        parser.add_argument(
            "--cloth-displacement-threshold",
            type=float,
            default=5.0e-4,
            help="Cloth displacement threshold per 60 Hz simulation frame near its tray [m]; 0 disables it.",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)

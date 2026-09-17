# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pack deformable groceries with a Piper and its parallel gripper.

The robot is a prescribed moving boundary in MJVBD V2. Groceries are never
attached to it. Slotted handles rest on the visible hanging rack through
contact; all shell vertices, the tetrahedral grocery and sealed pouch are free.
Material parameters are illustrative, not calibrated supermarket products.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from itertools import pairwise
from pathlib import Path

import mujoco
import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import PneumaticConfig, PneumaticMode, SolverMJVBDV2, add_inflatable_mesh

ASSETS = Path(__file__).resolve().parent / "assets" / "supermarket"
TABLE_Z = 0.72
BAG_POSITION = np.array([0.43, 0.25, 0.82])
GOODS = (np.array([0.40, -0.22, TABLE_Z + 0.034]), np.array([0.50, -0.20, TABLE_Z + 0.024]))
TCP_HOME = np.array([0.42, -0.15, 1.14])
OPEN = 0.035
ROBOT_BASE = np.array([0.16, 0.0, 1.07])
ROBOT_INITIAL = np.array([0.0, 1.0, -1.1, 0.0, 0.7, 0.0, OPEN, -OPEN])
CYCLE = 12.0
AIR_DENSITY = 1.2  # kg/m^3, still room air (illustrative, not fitted to a trajectory).
FILM_NORMAL_DRAG = 1.2
FILM_TANGENTIAL_DRAG = 0.02


def robot_xml():
    """Adapt the WAIC Piper asset without changing its joint limits."""
    root = ET.parse(ASSETS / "piper" / "piper_with_texture.xml").getroot()
    root.find("compiler").set("meshdir", str(ASSETS / "piper"))
    root.find("worldbody/body").set("pos", " ".join(map(str, ROBOT_BASE)))
    # The upstream scene material references a texture not bundled with the arm.
    asset = root.find("asset")
    asset.remove(asset.find("material[@name='groundplane']"))
    hand = root.find(".//body[@name='gripper_base_left']")
    ET.SubElement(hand, "site", name="packing_tcp", pos="0 0 0.1158", quat="0.38268343 0 0 0.92387953")
    for body in root.findall(".//body"):
        for index, geom in enumerate(body.findall("geom")):
            geom.set("name", f"{body.get('name')}_{geom.get('class')}_{index}")
    # Visible silicone pads cover the native fingertips' flat inner faces.
    # The render and collision boxes have identical dimensions and transforms.
    for name in ("link7", "link8"):
        jaw = root.find(f".//body[@name='{name}']")
        for kind in ("collision", "visual"):
            ET.SubElement(
                jaw,
                "geom",
                name=f"{name}_rubber_pad_{kind}",
                type="box",
                size="0.013 0.016 0.002",
                pos="0 -0.018 0",
                rgba="0.20 0.22 0.23 1",
                friction="1.1 0.005 0.0001",
                **{"class": kind},
            )
    return ET.tostring(root, encoding="unicode")


def pouch_mesh():
    """Build a closed pillow-shaped packaging membrane with sealed edges."""
    vertices, faces = [], []
    nx, ny = 8, 10
    for side in (-1, 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                u, v = i / nx, j / ny
                z = side * (0.003 + 0.014 * math.sin(math.pi * u) * math.sin(math.pi * v))
                vertices.append((0.066 * (u - 0.5), 0.092 * (v - 0.5), z))
    layer = (nx + 1) * (ny + 1)
    for side in range(2):
        for j in range(ny):
            for i in range(nx):
                a = side * layer + j * (nx + 1) + i
                ts = [(a, a + 1, a + nx + 2), (a, a + nx + 2, a + nx + 1)]
                faces.extend([t if side else tuple(reversed(t)) for t in ts])
    perimeter = list(range(nx + 1)) + [j * (nx + 1) + nx for j in range(1, ny + 1)]
    perimeter += [ny * (nx + 1) + i for i in range(nx - 1, -1, -1)]
    perimeter += [j * (nx + 1) for j in range(ny - 1, 0, -1)]
    for a, b in zip(perimeter, perimeter[1:] + perimeter[:1], strict=True):
        faces.extend([(a, b, b + layer), (a, b + layer, a + layer)])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def bread_mesh():
    """Build a domed loaf with tetrahedra in its actual reference shape."""
    grid = newton.ModelBuilder()
    grid.add_soft_grid(
        pos=wp.vec3(-0.0275, -0.035, -0.030),
        rot=wp.quat_identity(),
        vel=wp.vec3(),
        dim_x=5,
        dim_y=6,
        dim_z=5,
        cell_x=0.011,
        cell_y=0.070 / 6,
        cell_z=0.012,
        density=250,
        k_mu=1.5e4,
        k_lambda=3.0e4,
        k_damp=1.0,
    )
    vertices = np.asarray(grid.particle_q, dtype=np.float32).copy()
    u, v = vertices[:, 0] / 0.0275, vertices[:, 1] / 0.035
    height = (vertices[:, 2] + 0.030) / 0.060
    vertices[:, 0] *= 1 - 0.12 * v * v
    vertices[:, 1] *= 1 - 0.08 * u * u
    vertices[:, 2] = -0.030 + height * 0.060 * (1 - 0.18 * u * u - 0.12 * v * v)
    # The final builder recomputes rest matrices from these shaped vertices.
    return vertices, np.asarray(grid.tet_indices, dtype=np.int32)


def target(time):
    """Prescribe robot motion only, with slow closure and a clear transfer arc."""
    if time < 1:
        return TCP_HOME.copy(), OPEN, -1
    item = min(int((time - 1) / CYCLE), 1)
    t = time - 1 - item * CYCLE
    pick = GOODS[item].copy()
    # Pads grip above the table rather than straddling its collision surface.
    pick[2] = TABLE_Z + (0.032 if item == 0 else 0.022)
    hover = pick.copy()
    hover[2] = 1.16
    # Open above the central neck: opening inside the film can drag the bag
    # sideways during release and withdrawal.
    drop = BAG_POSITION + np.array(((-0.035 if item == 0 else 0.035), 0, 0.32))
    carry = drop.copy()
    carry[2] = 1.16
    closed = 0.023 if item == 0 else 0.010
    points = [
        (0, TCP_HOME, OPEN),
        (1.5, hover, OPEN),
        (3, pick, OPEN),
        (4.2, pick, closed),
        (5.7, hover, closed),
        (8, carry, closed),
        (9, drop, closed),
        (10, drop, OPEN),
        (11.5, carry, OPEN),
        (12, TCP_HOME, OPEN),
    ]
    for (a, pa, qa), (b, pb, qb) in pairwise(points):
        if t <= b:
            u = np.clip((t - a) / (b - a), 0, 1)
            u = u * u * (3 - 2 * u)
            return (1 - u) * pa + u * pb, (1 - u) * qa + u * qb, item
    return TCP_HOME.copy(), OPEN, item


@wp.kernel
def accumulate_film_drag(
    triangles: wp.array2d[wp.int32],
    q: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    drag: wp.array[wp.mat33],
    rho: float,
    normal_cd: float,
    tangent_cd: float,
):
    face = wp.tid()
    a, b, c = triangles[face, 0], triangles[face, 1], triangles[face, 2]
    cross = wp.cross(q[b] - q[a], q[c] - q[a])
    double_area = wp.length(cross)
    if double_area > 1.0e-12:
        n = cross / double_area
        nn = wp.outer(n, n)
        tangent = wp.identity(n=3, dtype=float) - nn
        for corner in range(3):
            i = triangles[face, corner]
            v = velocity[i]
            vn = wp.dot(v, n)
            vt = v - vn * n
            # Lump one third of the real triangle area to each vertex.
            d = (rho * double_area / 12.0) * (normal_cd * wp.abs(vn) * nn + tangent_cd * wp.length(vt) * tangent)
            wp.atomic_add(drag, i, d)


@wp.kernel
def apply_film_drag(
    velocity: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    drag: wp.array[wp.mat33],
    dt: float,
    forces: wp.array[wp.vec3],
):
    i = wp.tid()
    if inv_mass[i] > 0.0:
        d = drag[i]
        # Backward Euler with frozen quadratic coefficients. D is PSD, so
        # drag alone cannot add kinetic energy or reverse a velocity mode.
        v_air = wp.inverse(wp.identity(n=3, dtype=float) + dt * inv_mass[i] * d) * velocity[i]
        forces[i] += -d * v_air


@wp.kernel
def prescribe_joints(
    start: wp.array[float], end: wp.array[float], fraction: float, q: wp.array[float], qd: wp.array[float]
):
    i = wp.tid()
    q[i] = (1.0 - fraction) * start[i] + fraction * end[i]
    qd[i] = (end[i] - start[i]) * 60.0


class Example:
    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        if args.substeps < 2 or args.substeps % 2 or args.iterations < 1:
            raise ValueError("Use a positive even substep count and at least one iteration")
        self.sim_time = 0.0
        self.dt = 1 / (60 * args.substeps)
        self.max_lift = np.zeros(2)
        self.grasp_reference = [None, None]
        self.max_grasp_slip = np.zeros(2)
        self.graph = None
        xml = robot_xml()
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.site = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "packing_tcp")
        self.mj_data.qpos[:] = ROBOT_INITIAL
        self.jac_p = np.zeros((3, self.mj_model.nv))
        self.jac_r = np.zeros_like(self.jac_p)
        initial = self._ik(TCP_HOME, OPEN)
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        builder.add_mjcf(xml, floating=False, enable_self_collisions=False, parse_sites=False)
        self.joint_count = builder.joint_coord_count
        builder.joint_q[:] = initial.tolist()
        for body in range(builder.body_count):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
        for shape in range(builder.shape_count):
            builder.shape_material_ke[shape] = 2.0e5
            builder.shape_material_kd[shape] = 30.0
            if "rubber_pad" in builder.shape_label[shape] or builder.shape_label[shape].startswith(
                ("link7_", "link8_")
            ):
                builder.shape_material_mu[shape] = 1.1
            else:
                builder.shape_material_mu[shape] = 0.9
        self._scenery(builder)
        bag = json.loads((ASSETS / "shopping_bag.json").read_text())
        self.bag_start = builder.particle_count
        builder.add_cloth_mesh(
            pos=wp.vec3(*BAG_POSITION),
            rot=wp.quat_identity(),
            scale=1,
            vel=wp.vec3(),
            vertices=bag["vertices"],
            indices=np.asarray(bag["triangles"]).flatten().tolist(),
            **bag["simulation_material"],
            color=(0.91, 0.94, 0.88),
            label="Blender shopping bag",
        )
        self.bag_end = builder.particle_count
        self.goods_ranges = []
        start = builder.particle_count
        bread_vertices, bread_tets = bread_mesh()
        builder.add_soft_mesh(
            pos=wp.vec3(*GOODS[0]),
            rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), math.pi / 4),
            scale=1.0,
            vel=wp.vec3(),
            vertices=bread_vertices.tolist(),
            indices=bread_tets.flatten().tolist(),
            density=250,
            k_mu=1.5e4,
            k_lambda=3.0e4,
            k_damp=1.0,
            particle_radius=0.002,
            color=(0.65, 0.32, 0.105),
            label="Soft bread loaf",
        )
        self.goods_ranges.append((start, builder.particle_count))
        start = builder.particle_count
        vertices, triangles = pouch_mesh()
        self.cavity = add_inflatable_mesh(
            builder,
            pos=wp.vec3(*GOODS[1]),
            rot=wp.quat_from_axis_angle(wp.vec3(0, 0, 1), math.pi / 4),
            scale=1,
            vel=wp.vec3(),
            vertices=vertices.tolist(),
            indices=triangles.flatten().tolist(),
            density=0.18,
            tri_ke=2.0e4,
            tri_ka=2.0e4,
            tri_kd=1,
            edge_ke=0.015,
            edge_kd=0.002,
            particle_radius=0.0015,
            label="Deformable milk pouch (pneumatic surrogate)",
            validate_mesh=True,
            config=PneumaticConfig(
                mode=PneumaticMode.ISOTHERMAL,
                reference_absolute_pressure=101800,
                ambient_pressure=101325,
                bulk_damping=10,
                max_absolute_pressure=180000,
            ),
        )
        self.goods_ranges.append((start, builder.particle_count))
        builder.color(include_bending=True)
        self.model = builder.finalize()
        # Film contact is not rubber-pad contact. Shape materials retain the
        # gripper's own friction; the existing solver combines material pairs.
        self.model.soft_contact_ke, self.model.soft_contact_kd, self.model.soft_contact_mu = 2e5, 10.0, 0.4
        self.bag_faces = wp.array(
            np.asarray(bag["triangles"], dtype=np.int32) + self.bag_start, dtype=wp.int32, device=self.model.device
        )
        self.film_drag = wp.zeros(self.model.particle_count, dtype=wp.mat33, device=self.model.device)
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        self.begin = wp.clone(self.model.joint_q)
        self.end = wp.clone(self.model.joint_q)
        self.last_q = initial
        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=(0,),
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": args.iterations,
                # Resolve slow grip slip rather than smoothing it over 1 cm/s.
                "friction_epsilon": 1.0e-3,
                "enable_cuda_fast_path": True,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": 0.002,
                "particle_self_contact_margin": 0.004,
                "particle_topological_contact_filter_threshold": 2,
                "particle_rest_shape_contact_exclusion_radius": 0.004,
                "particle_vertex_contact_buffer_size": 64,
                "particle_edge_contact_buffer_size": 64,
                "rigid_body_particle_contact_buffer_size": 8192,
                "particle_enable_truncation_cache": True,
            },
            collision_options={
                "soft_contact_max": 65536,
                "rigid_contact_max": 4096,
                "soft_contact_margin": 0.004,
                "include_static_kinematic_pairs": False,
                "enable_rigid_soft_full_surface_contact": True,
                "enable_cuda_fast_path": True,
            },
        )
        self.rest = self.state_0.particle_q.numpy().copy()
        self.tets = self.model.tet_indices.numpy()
        self.rest_tet_volume = self._tet_volumes(self.rest)
        self.min_tet_volume_ratio = 1.0
        self.min_bag_height = float("inf")
        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        self.viewer.show_particles = False
        self._prepare_deformable_rendering()
        self.viewer.set_camera(pos=wp.vec3(2.0, -2.2, 1.95), pitch=-24, yaw=130)
        print(
            json.dumps(
                {
                    "particles": self.model.particle_count,
                    "tetrahedra": self.model.tet_count,
                    "robot": "WAIC Piper + native parallel gripper",
                    "pinned_bag_vertices": len(bag["handle_support_vertices"]),
                }
            ),
            flush=True,
        )

    def _ik(self, position, opening):
        desired = np.diag([1.0, -1.0, -1.0])
        for _ in range(200):
            mujoco.mj_forward(self.mj_model, self.mj_data)
            current = self.mj_data.site_xmat[self.site].reshape(3, 3)
            delta_quat = np.empty(4)
            error_r = np.empty(3)
            mujoco.mju_mat2Quat(delta_quat, (desired @ current.T).flatten())
            mujoco.mju_quat2Vel(error_r, delta_quat, 1.0)
            error_p = position - self.mj_data.site_xpos[self.site]
            if np.linalg.norm(error_p) < 0.0002 and np.linalg.norm(error_r) < 0.001:
                break
            mujoco.mj_jacSite(self.mj_model, self.mj_data, self.jac_p, self.jac_r, self.site)
            jac = np.vstack((self.jac_p[:, :6], self.jac_r[:, :6] * 0.25))
            error = np.concatenate((error_p, error_r * 0.25))
            update = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * 1e-5, error)
            self.mj_data.qpos[:6] += np.clip(update, -0.15, 0.15)
            self.mj_data.qpos[:6] = np.clip(
                self.mj_data.qpos[:6], self.mj_model.jnt_range[:6, 0], self.mj_model.jnt_range[:6, 1]
            )
        if np.linalg.norm(error_p) > 0.004 or np.linalg.norm(error_r) > 0.01:
            raise RuntimeError(f"Piper target unreachable: {position}, error={np.linalg.norm(error_p):.4f} m")
        self.mj_data.qpos[6:] = (opening, -opening)
        return self.mj_data.qpos.copy()

    @staticmethod
    def _box(builder, pos, size, color, label, *, collision=True):
        cfg = newton.ModelBuilder.ShapeConfig(
            ke=2e5, kd=30, mu=0.6, margin=0.001, has_shape_collision=collision, has_particle_collision=collision
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(*pos), wp.quat_identity()),
            hx=size[0],
            hy=size[1],
            hz=size[2],
            cfg=cfg,
            color=color,
            label=label,
        )

    def _scenery(self, builder):
        self._box(builder, (0.40, -0.20, TABLE_Z - 0.03), (0.65, 0.29, 0.03), (0.70, 0.73, 0.72), "Checkout worktop")
        self._box(
            builder,
            (ROBOT_BASE[0], ROBOT_BASE[1], (ROBOT_BASE[2] + TABLE_Z) / 2),
            (0.075, 0.075, (ROBOT_BASE[2] - TABLE_Z) / 2),
            (0.20, 0.22, 0.24),
            "Piper mounting pedestal",
        )

        # Contact-only rack: rails thread the front and back handle slots.
        # No bag particle is pinned and no shelf supports its bottom.
        def tube(a, b, radius, label):
            a, b = np.asarray(a), np.asarray(b)
            direction = b - a
            length = float(np.linalg.norm(direction))
            direction /= length
            axis = np.cross((0.0, 0.0, 1.0), direction)
            rotation = wp.quat_identity()
            if np.linalg.norm(axis) > 1e-8:
                rotation = wp.quat_from_axis_angle(wp.vec3(*axis / np.linalg.norm(axis)), math.acos(direction[2]))
            elif direction[2] < 0:
                rotation = wp.quat_from_axis_angle(wp.vec3(1, 0, 0), math.pi)
            cfg = newton.ModelBuilder.ShapeConfig(ke=2e5, kd=30, mu=0.35, margin=0.001)
            builder.add_shape_capsule(
                -1,
                xform=wp.transform(wp.vec3(*((a + b) / 2)), rotation),
                radius=radius,
                half_height=length / 2,
                cfg=cfg,
                color=(0.62, 0.65, 0.67),
                label=label,
            )

        bag_asset = json.loads((ASSETS / "shopping_bag.json").read_text())
        for local_center in bag_asset["rack_rail_centers_local"]:
            x, y, z = BAG_POSITION + local_center
            tube((x, y + 0.15, 0.04), (x, y + 0.15, z), 0.008, "Round rack upright")
            tube((x, y - 0.13, z), (x, y + 0.15, z), 0.005, "Handle support rail")
            tube((x, y - 0.13, z), (x, y - 0.13, z + 0.025), 0.005, "Upturned rail end")
        self._box(
            builder,
            (BAG_POSITION[0], BAG_POSITION[1] + 0.15, 0.02),
            (0.23, 0.08, 0.02),
            (0.18, 0.20, 0.21),
            "Hanging rack foot",
        )
        decor = json.loads((ASSETS / "checkout_decor.json").read_text())
        self.decor_meshes = []
        for name, group in decor.items():
            self.decor_meshes.append(
                (
                    name,
                    wp.array(group["vertices"], dtype=wp.vec3),
                    wp.array(group["indices"], dtype=wp.int32),
                    wp.array(group["normals"], dtype=wp.vec3),
                    group["color"],
                    group.get("roughness", 0.58),
                    group.get("metallic", 0.0),
                )
            )
        self.pantry_meshes = []
        self.floor_points = wp.array(
            [(-0.75, -0.75, 0), (0.75, -0.75, 0), (0.75, 0.75, 0), (-0.75, 0.75, 0)], dtype=wp.vec3
        )
        self.floor_indices = wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32)
        self.floor_uv = wp.array([(0, 0), (1, 0), (1, 1), (0, 1)], dtype=wp.vec2)
        floor_poses = [
            wp.transform(wp.vec3(0.4 + i * 1.5, 0.1 + j * 1.5, 0.002), wp.quat_identity())
            for i in range(-2, 3)
            for j in range(-2, 3)
        ]
        self.floor_poses = wp.array(floor_poses, dtype=wp.transform)
        self.floor_colors = wp.array([wp.vec3(1.0)] * len(floor_poses), dtype=wp.vec3)
        self.floor_materials = wp.array([wp.vec4(0.65, 0, 0, 1)] * len(floor_poses), dtype=wp.vec4)
        pantry = json.loads((ASSETS / "pantry_meshes.json").read_text())
        for name, group in pantry.items():
            poses = []
            if name.endswith("milk"):
                for i in range(6):
                    for y in (0.97, 1.095):
                        poses.append(wp.transform(wp.vec3(-0.37 + i * 0.193, y, 1.0925), wp.quat_identity()))
            elif name.endswith("sardines"):
                for i in (8, 9):
                    for layer in range(4):
                        poses.append(
                            wp.transform(wp.vec3(-0.37 + i * 0.193, 0.99, 0.2325 + layer * 0.028), wp.quat_identity())
                        )
            else:
                parity = 0 if name.endswith("tomatoes") else 1
                for i in range(parity, 8, 2):
                    for y in (0.97, 1.105):
                        poses.append(wp.transform(wp.vec3(-0.37 + i * 0.193, y, 0.2325), wp.quat_identity()))
            self.pantry_meshes.append(
                (
                    name,
                    wp.array(group["vertices"], dtype=wp.vec3),
                    wp.array(group["indices"], dtype=wp.int32),
                    wp.array(group["normals"], dtype=wp.vec3),
                    wp.array(group["uvs"], dtype=wp.vec2),
                    group["texture"],
                    wp.array(poses, dtype=wp.transform),
                    wp.array([wp.vec3(1.0)] * len(poses), dtype=wp.vec3),
                    wp.array([wp.vec4(0.48, 0.0, 0.0, 1.0)] * len(poses), dtype=wp.vec4),
                )
            )
        builder.add_ground_plane()

    def _simulate(self):
        for i in range(self.args.substeps):
            wp.launch(
                prescribe_joints,
                self.joint_count,
                [self.begin, self.end, (i + 1) / self.args.substeps, self.state_0.joint_q, self.state_0.joint_qd],
            )
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )
            self.state_0.clear_forces()
            self.film_drag.zero_()
            wp.launch(
                accumulate_film_drag,
                self.bag_faces.shape[0],
                [
                    self.bag_faces,
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self.film_drag,
                    AIR_DENSITY,
                    FILM_NORMAL_DRAG,
                    FILM_TANGENTIAL_DRAG,
                ],
                device=self.model.device,
            )
            wp.launch(
                apply_film_drag,
                self.model.particle_count,
                [
                    self.state_0.particle_qd,
                    self.model.particle_inv_mass,
                    self.film_drag,
                    self.dt,
                    self.state_0.particle_f,
                ],
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        pos, opening, _ = target(self.sim_time + 1 / 60)
        q = self._ik(pos, opening)
        self.begin.assign(self.last_q.astype(np.float32))
        self.end.assign(q.astype(np.float32))
        self.last_q = q
        if self.graph is None:
            self._simulate()
            if self.model.device.is_cuda:
                with wp.ScopedCapture() as capture:
                    self._simulate()
                self.graph = capture.graph
        else:
            wp.capture_launch(self.graph)
        self.sim_time += 1 / 60
        if round(self.sim_time * 60) % 15 == 0:
            self._measure()

    def _prepare_deformable_rendering(self):
        """Apply print colors to the simulated surface, not a rigid overlay."""
        triangles = self.model.tri_indices.numpy()
        is_bag = np.all((triangles >= self.bag_start) & (triangles < self.bag_end), axis=1)
        a, b = self.goods_ranges[1]
        is_pouch = np.all((triangles >= a) & (triangles < b), axis=1)
        local = self.rest - BAG_POSITION
        uv = np.column_stack(((local[:, 0] + 0.16) / 0.32, local[:, 2] / 0.40))
        self.bag_uv = wp.array(uv.astype(np.float32), dtype=wp.vec2, device=self.model.device)
        self.surface_render_transform = wp.array(
            [wp.transform_identity()], dtype=wp.transform, device=self.model.device
        )
        self.surface_render_color = wp.array([wp.vec3(1.0)], dtype=wp.vec3, device=self.model.device)
        self.bag_render_opacity = wp.array([0.90], dtype=float, device=self.model.device)
        self.surface_textures = {"bag_film": (self.bag_uv, "bag_print.png")}
        for name, center, size, filename in (
            ("bread", GOODS[0], (0.055, 0.070), "bread_crust.png"),
            ("milk", GOODS[1], (0.066, 0.092), "milk_pouch.png"),
        ):
            angle = math.pi / 4
            rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            uv = ((self.rest[:, :2] - center[:2]) @ rotation) / np.array(size) + 0.5
            self.surface_textures[name] = (
                wp.array(uv.astype(np.float32), dtype=wp.vec2, device=self.model.device),
                filename,
            )
        self.surface_materials = {
            name: wp.array([wp.vec4(roughness, 0, 0, 1)], dtype=wp.vec4, device=self.model.device)
            for name, roughness in (("bag_film", 0.27), ("bread", 0.82), ("milk", 0.38))
        }
        self.solid_render_opacity = wp.array([1.0], dtype=float, device=self.model.device)
        self.render_parts = []
        for name, mask, color, roughness, opacity in (
            ("bag_film", is_bag, (1.0, 1.0, 1.0), 0.27, 0.90),
            ("milk", is_pouch, (1.0, 1.0, 1.0), 0.38, 1.0),
            ("bread", ~is_bag & ~is_pouch, (1.0, 1.0, 1.0), 0.82, 1.0),
        ):
            indices = wp.array(triangles[mask].flatten(), dtype=wp.int32, device=self.model.device)
            self.render_parts.append((name, indices, color, roughness, opacity))

    def _tet_volumes(self, q):
        a, b, c, d = (q[self.tets[:, i]] for i in range(4))
        return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6

    def _measure(self):
        q = self.state_0.particle_q.numpy()
        if not np.isfinite(q).all():
            raise AssertionError("Nonfinite grocery or bag state")
        self.min_bag_height = min(self.min_bag_height, float(q[self.bag_start : self.bag_end, 2].min()))
        if self.min_bag_height < 0.05:
            raise AssertionError("The suspended bag reached the floor")
        ratio = self._tet_volumes(q) / self.rest_tet_volume
        self.min_tet_volume_ratio = min(self.min_tet_volume_ratio, float(ratio.min()))
        if self.min_tet_volume_ratio <= 0:
            raise AssertionError("The tetrahedral grocery inverted during manipulation")
        centers = np.array([q[a:b].mean(axis=0) for a, b in self.goods_ranges])
        self.max_lift = np.maximum(self.max_lift, centers[:, 2] - np.array(GOODS)[:, 2])
        position, _, item = target(self.sim_time)
        phase = self.sim_time - 1 - item * CYCLE
        if item >= 0 and 4.2 <= phase <= 9.0 + 1.0e-6:
            offset = centers[item] - position
            if self.grasp_reference[item] is None:
                self.grasp_reference[item] = offset.copy()
            slip = float(np.linalg.norm(offset - self.grasp_reference[item]))
            self.max_grasp_slip[item] = max(self.max_grasp_slip[item], slip)
            if slip > 0.018:
                raise AssertionError(
                    f"Grocery {item} slipped during transport at {self.sim_time:.3f}s: {slip * 1000:.1f} mm"
                )
        over_counter = (q[:, 0] > -0.25) & (q[:, 0] < 1.05) & (q[:, 1] > -0.49) & (q[:, 1] < 0.09)
        if np.any(q[over_counter, 2] < TABLE_Z - 0.01) or np.min(q[:, 2]) < 0.015:
            raise AssertionError(
                f"Object fell below the work area at {self.sim_time:.3f}s; "
                f"centers={centers.tolist()}, minimum_z={float(q[:, 2].min()):.4f}"
            )
        if round(self.sim_time * 60) % 120 == 0:
            print(
                json.dumps({"time": self.sim_time, "centers": centers.tolist(), "max_lift": self.max_lift.tolist()}),
                flush=True,
            )
        return centers

    def test_final(self):
        centers = self._measure()
        report = {
            "time": self.sim_time,
            "centers": centers.tolist(),
            "max_lift": self.max_lift.tolist(),
            "max_grasp_slip_m": self.max_grasp_slip.tolist(),
            "min_tet_volume_ratio": self.min_tet_volume_ratio,
            "minimum_bag_height_m": self.min_bag_height,
        }
        print(json.dumps(report), flush=True)
        if self.sim_time >= 25 - 1.0e-6:
            delta = centers - BAG_POSITION
            if np.any(self.max_lift < 0.10) or np.any(np.abs(delta[:, 0]) > 0.14) or np.any(np.abs(delta[:, 1]) > 0.10):
                raise AssertionError("Both free groceries must be lifted and deposited inside the bag")
            bag_q = self.state_0.particle_q.numpy()[self.bag_start : self.bag_end]
            if np.any(centers[:, 2] < bag_q[:, 2].min() - 0.005) or np.any(delta[:, 2] > 0.27):
                raise AssertionError("Groceries must settle below the open bag mouth")

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/store_floor",
            self.floor_points,
            self.floor_indices,
            uvs=self.floor_uv,
            texture=str(ASSETS / "polyhaven/floor_tiles_08_diff_1k.jpg"),
            hidden=True,
        )
        self.viewer.log_instances(
            "/store_floor_instances", "/store_floor", self.floor_poses, None, self.floor_colors, self.floor_materials
        )
        for name, points, indices, normals, color, roughness, metallic in self.decor_meshes:
            self.viewer.log_mesh(
                "/checkout/" + name,
                points,
                indices,
                normals=normals,
                color=color,
                roughness=roughness,
                metallic=metallic,
            )
        for name, points, indices, normals, uv, texture, poses, colors, materials in self.pantry_meshes:
            path = "/pantry/" + name
            self.viewer.log_mesh(
                path,
                points,
                indices,
                normals=normals,
                uvs=uv,
                texture=str(ASSETS / texture),
                hidden=True,
            )
            self.viewer.log_instances(path + "_instances", path, poses, None, colors, materials)
        for name, indices, color, roughness, opacity in self.render_parts:
            self.viewer.log_mesh(
                "/groceries/" + name,
                self.state_0.particle_q,
                indices,
                color=color,
                roughness=roughness,
                metallic=0.0,
                opacity=opacity,
                backface_culling=False,
                uvs=self.surface_textures[name][0],
                texture=str(ASSETS / self.surface_textures[name][1]),
                hidden=True,
            )
            # Texture coordinates and ink deform with the actual surface.
            self.viewer.log_instances(
                "/groceries/" + name + "_instance",
                "/groceries/" + name,
                self.surface_render_transform,
                None,
                self.surface_render_color,
                self.surface_materials[name],
                opacities=self.bag_render_opacity if name == "bag_film" else self.solid_render_opacity,
            )
        self.viewer.end_frame()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1800)
        parser.add_argument("--substeps", type=int, default=8)
        parser.add_argument("--iterations", type=int, default=24)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

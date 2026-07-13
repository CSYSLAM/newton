# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Dexforce Bimanual IK T-Shirt
#
# Loads DexforceW1V021 in its URDF pose, places the unisex T-shirt cloth mesh
# on the same table used by the bimanual IK cloth scene, and exposes an IK
# gizmo for each wrist TCP so shirt grasp trajectories can be debugged.
#
# Command: python -m newton.examples cloth_dexforce_bimanual_ik_tshirt
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
from newton.examples.cloth.example_cloth_dexforce_bimanual_grasp_cloth import (
    set_indexed_joint_q_kernel,
)
from newton.examples.cloth.example_cloth_dexforce_bimanual_ik_cloth import (
    Example as DexforceBimanualIKClothExample,
)


TABLE_POS = wp.vec3(0.55, 0.0, 1.15)
TABLE_HALF_EXTENTS = (0.26, 0.62, 0.025)
TABLE_TOP_Z = float(TABLE_POS[2]) + TABLE_HALF_EXTENTS[2]
TABLE_COLOR = (0.35, 0.42, 0.48)

SHIRT_ASSET = "unisex_shirt.usd"
SHIRT_PRIM_PATH = "/root/shirt"
SHIRT_SCALE = 0.0080 * 0.8
SHIRT_DENSITY = 0.02
SHIRT_POS = wp.vec3(float(TABLE_POS[0]), 0.0, TABLE_TOP_Z + 0.014)
SHIRT_ROT = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.5 * np.pi)
SHIRT_COLOR = (0.7, 0.7, 0.7)

SHIRT_COLLISION_RADIUS = 0.008
SHIRT_SOFT_CONTACT_MARGIN = 0.008
SHIRT_SELF_CONTACT_RADIUS = 0.002
SHIRT_SELF_CONTACT_MARGIN = 0.002

SHIRT_TRI_KE = 1.0e3
SHIRT_TRI_KA = 1.0e3
SHIRT_TRI_KD = 1.0e-5
SHIRT_EDGE_KE = 1.0
SHIRT_EDGE_KD = 0.1


class Example(DexforceBimanualIKClothExample):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self.hand_q_indices, self.hand_open, self.hand_grasp = self._build_hand_targets()

    def _prepare_frame_targets(self) -> None:
        super()._prepare_frame_targets()

        wp.launch(
            set_indexed_joint_q_kernel,
            dim=self.hand_q_indices.shape[0],
            inputs=[self.hand_q_indices, self.hand_open, self.hand_grasp, 1.0],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )

    def _add_table_scene(self, builder: newton.ModelBuilder) -> None:
        table_cfg = newton.ModelBuilder.ShapeConfig()
        table_cfg.ke = 5.0e5
        table_cfg.kd = 1.0e-6
        table_cfg.mu = 1.2

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(TABLE_POS, wp.quat_identity()),
            hx=TABLE_HALF_EXTENTS[0],
            hy=TABLE_HALF_EXTENTS[1],
            hz=TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=TABLE_COLOR,
        )
        builder.add_ground_plane()

    def _add_cloth(self, builder: newton.ModelBuilder) -> None:
        self.particle_radius = SHIRT_COLLISION_RADIUS
        self.soft_contact_margin = SHIRT_SOFT_CONTACT_MARGIN
        self.particle_self_contact_radius = SHIRT_SELF_CONTACT_RADIUS
        self.particle_self_contact_margin = SHIRT_SELF_CONTACT_MARGIN

        usd_stage = Usd.Stage.Open(newton.examples.get_asset(SHIRT_ASSET))
        usd_prim = usd_stage.GetPrimAtPath(SHIRT_PRIM_PATH)
        shirt_mesh = newton.usd.get_mesh(usd_prim)
        vertices = self._center_shirt_vertices(shirt_mesh.vertices)

        builder.add_cloth_mesh(
            vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in vertices],
            indices=shirt_mesh.indices,
            rot=SHIRT_ROT,
            pos=SHIRT_POS,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=SHIRT_DENSITY,
            scale=1.0,
            tri_ke=SHIRT_TRI_KE,
            tri_ka=SHIRT_TRI_KA,
            tri_kd=SHIRT_TRI_KD,
            edge_ke=SHIRT_EDGE_KE,
            edge_kd=SHIRT_EDGE_KD,
            particle_radius=self.particle_radius,
        )

    def _configure_particle_contacts(self) -> None:
        flags = self.model.shape_flags.numpy()
        flags |= int(newton.ShapeFlags.COLLIDE_PARTICLES)
        flags[: self.robot_shape_end] &= ~(
            int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
        )
        self.model.shape_flags = wp.array(flags, dtype=self.model.shape_flags.dtype, device=self.model.device)

    def _build_hand_targets(self) -> tuple[wp.array, wp.array, wp.array]:
        q_home = self.model.joint_q.numpy()
        q_start = self.model.joint_q_start.numpy()
        q_indices = []
        open_values = []
        grasp_values = []
        targets = {
            "HAND_THUMB2": 0.84,
            "HAND_THUMB1": 0.46,
            "HAND_INDEX": 0.70,
            "INDEX_PIP": 0.90,
        }

        for side in ("LEFT", "RIGHT"):
            for suffix in self.HAND_JOINT_SUFFIXES:
                joint_idx = self._joint_index(f"{side}_{suffix}")
                q_idx = int(q_start[joint_idx])
                open_value = float(q_home[q_idx])
                q_indices.append(q_idx)
                open_values.append(open_value)
                grasp_values.append(targets.get(suffix, open_value))

        return (
            wp.array(q_indices, dtype=wp.int32, device=self.model.device),
            wp.array(open_values, dtype=wp.float32, device=self.model.device),
            wp.array(grasp_values, dtype=wp.float32, device=self.model.device),
        )

    def _joint_index(self, joint_name: str) -> int:
        suffix = f"/{joint_name}"
        return next(i for i, label in enumerate(self.model.joint_label) if label.endswith(suffix))

    def _controlled_joint_labels(self) -> set[str]:
        return {
            f"DexforceW1V021/{name}"
            for name in (*self.BODY_JOINTS, *self.LEFT_ARM_JOINTS, *self.RIGHT_ARM_JOINTS)
        }

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        if hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo(
                "dexforce_left_tcp_target",
                self.left_tf,
                snap_to=self._current_tcp_transform(self.left_ee_index, self.left_ee_offset),
            )
            self.viewer.log_gizmo(
                "dexforce_right_tcp_target",
                self.right_tf,
                snap_to=self._current_tcp_transform(self.right_ee_index, self.right_ee_offset),
            )
        show_triangles = getattr(self.viewer, "show_triangles", True)
        if hasattr(self.viewer, "show_triangles"):
            self.viewer.show_triangles = False
        self.viewer.log_state(self.state_0)
        if hasattr(self.viewer, "show_triangles"):
            self.viewer.show_triangles = show_triangles
        self.viewer.log_mesh(
            "/debug_tshirt",
            self.state_0.particle_q,
            self.model.tri_indices.flatten(),
            hidden=not show_triangles,
            backface_culling=False,
            color=SHIRT_COLOR,
        )
        self.viewer.end_frame()

    @staticmethod
    def _center_shirt_vertices(vertices: np.ndarray) -> np.ndarray:
        verts = np.asarray(vertices, dtype=np.float32).copy()
        bounds_min = verts.min(axis=0)
        bounds_max = verts.max(axis=0)
        center_xy = 0.5 * (bounds_min[:2] + bounds_max[:2])
        verts[:, 0] -= center_xy[0]
        verts[:, 1] -= center_xy[1]
        verts[:, 2] -= bounds_min[2]
        return verts * SHIRT_SCALE

    @staticmethod
    def create_parser():
        parser = DexforceBimanualIKClothExample.create_parser()
        parser.set_defaults(num_frames=900)
        return parser

    HAND_JOINT_SUFFIXES = (
        "HAND_THUMB2",
        "HAND_THUMB1",
        "HAND_INDEX",
        "INDEX_PIP",
        "HAND_MIDDLE",
        "MIDDLE_PIP",
        "HAND_RING",
        "RING_PIP",
        "HAND_PINKY",
        "PINKY_PIP",
    )
    BODY_JOINTS = (
        # "ANKLE",
        # "KNEE",
        # "BUTTOCK",
        # "WAIST",
    )


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

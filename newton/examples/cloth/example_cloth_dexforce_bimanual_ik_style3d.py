# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Cloth Dexforce Bimanual IK Style3D
#
# Loads DexforceW1V021 in its URDF pose, places the Style3D sweatshirt mesh
# from the Franka Style3D IK example on the same table used by the bimanual IK
# cloth scene, and exposes an IK gizmo for each wrist TCP.
#
# Command: python -m newton.examples cloth_dexforce_bimanual_ik_style3d
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from pxr import Usd

import newton
import newton.examples
import newton.usd
import newton.utils
from newton.examples.cloth.example_cloth_dexforce_bimanual_ik_cloth import (
    TABLE_POS,
    TABLE_TOP_Z,
    Example as DexforceBimanualIKClothExample,
)


GARMENT_USD_NAME = "Women_Sweatshirt"
GARMENT_SCALE = 0.75
GARMENT_DENSITY = 0.02
GARMENT_POS = wp.vec3(float(TABLE_POS[0]), 0.0, TABLE_TOP_Z + 0.006)
GARMENT_ROT = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), -0.5 * np.pi)
GARMENT_COLOR = (0.88, 0.03, 0.025)

GARMENT_COLLISION_RADIUS = 0.003
GARMENT_SOFT_CONTACT_MARGIN = 0.006
GARMENT_SELF_CONTACT_RADIUS = 0.0025
GARMENT_SELF_CONTACT_MARGIN = 0.003


class Example(DexforceBimanualIKClothExample):
    def _add_cloth(self, builder: newton.ModelBuilder) -> None:
        self.particle_radius = GARMENT_COLLISION_RADIUS
        self.soft_contact_margin = GARMENT_SOFT_CONTACT_MARGIN
        self.particle_self_contact_radius = GARMENT_SELF_CONTACT_RADIUS
        self.particle_self_contact_margin = GARMENT_SELF_CONTACT_MARGIN

        asset_path = newton.utils.download_asset("style3d")
        usd_stage = Usd.Stage.Open(str(asset_path / "garments" / f"{GARMENT_USD_NAME}.usd"))
        usd_prim = usd_stage.GetPrimAtPath(f"/Root/{GARMENT_USD_NAME}/Root_Garment")
        garment_mesh = newton.usd.get_mesh(usd_prim)
        vertices = self._place_garment_vertices_on_table(garment_mesh.vertices)

        builder.add_cloth_mesh(
            vertices=[wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in vertices],
            indices=garment_mesh.indices,
            rot=GARMENT_ROT,
            pos=GARMENT_POS,
            vel=wp.vec3(0.0, 0.0, 0.0),
            density=GARMENT_DENSITY,
            scale=1.0,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e-5,
            edge_ke=1.0,
            edge_kd=0.05,
            particle_radius=self.particle_radius,
        )

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
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/debug_style3d_garment",
            self.state_0.particle_q,
            self.model.tri_indices.flatten(),
            hidden=not getattr(self.viewer, "show_triangles", True),
            backface_culling=False,
            color=GARMENT_COLOR,
        )
        self.viewer.end_frame()

    @staticmethod
    def _place_garment_vertices_on_table(vertices: np.ndarray) -> np.ndarray:
        verts = np.asarray(vertices, dtype=np.float32).copy()
        bounds_min = verts.min(axis=0)
        bounds_max = verts.max(axis=0)
        center_xy = 0.5 * (bounds_min[:2] + bounds_max[:2])
        verts[:, 0] -= center_xy[0]
        verts[:, 1] -= center_xy[1]
        verts[:, 2] -= bounds_min[2]
        return verts * GARMENT_SCALE

    @staticmethod
    def create_parser():
        parser = DexforceBimanualIKClothExample.create_parser()
        parser.set_defaults(num_frames=900)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

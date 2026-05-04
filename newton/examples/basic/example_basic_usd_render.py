# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""USD asset parsing and rendering example.

Command: python -m newton.examples basic_usd_render
"""

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.time = 0.0
        self.rotation_speed = args.rotation_speed

        # Load USD and create meshes
        self.meshes = []
        self.transforms = []
        stage = Usd.Stage.Open(args.usd_path)

        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh):
                usd_mesh = UsdGeom.Mesh(prim)
                vertices = np.array(usd_mesh.GetPointsAttr().Get(), dtype=np.float32)
                face_indices = np.array(usd_mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
                face_counts = np.array(usd_mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)

                indices = self.triangulate(face_indices, face_counts)
                normals = self.compute_normals(vertices, indices)

                mesh = newton.Mesh(vertices, indices, normals=normals)
                mesh.finalize()
                self.meshes.append(mesh)
                self.transforms.append(UsdGeom.Xform(prim).GetLocalTransformation())

        self.color = wp.array([wp.vec3(0.8, 0.85, 0.9)], dtype=wp.vec3)
        self.material = wp.array([wp.vec4(0.3, 0.2, 0.0, 0.0)], dtype=wp.vec4)

    def triangulate(self, face_indices, face_counts):
        triangles = []
        offset = 0
        for count in face_counts:
            verts = face_indices[offset : offset + count]
            offset += count
            for i in range(1, count - 1):
                triangles.append([verts[0], verts[i], verts[i + 1]])
        return np.array(triangles, dtype=np.int32).flatten()

    def compute_normals(self, vertices, indices):
        normals = np.zeros((len(vertices), 3), dtype=np.float32)
        for i in range(0, len(indices), 3):
            i0, i1, i2 = indices[i], indices[i + 1], indices[i + 2]
            v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
            n = np.cross(v1 - v0, v2 - v0)
            normals[i0] += n
            normals[i1] += n
            normals[i2] += n
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        lengths[lengths == 0] = 1.0
        return normals / lengths

    def step(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.time)

        angle = self.rotation_speed * self.time
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angle)

        for i, mesh in enumerate(self.meshes):
            tf = self.transforms[i]
            pos = wp.vec3(tf[0][3], tf[1][3], tf[2][3])
            xform = wp.array([wp.transform(pos, quat)], dtype=wp.transform)
            self.viewer.log_shapes(
                f"/mesh_{i}", newton.GeoType.MESH, (1.0, 1.0, 1.0),
                xform, self.color, self.material, geo_src=mesh
            )

        self.viewer.log_shapes(
            "/ground", newton.GeoType.PLANE, (1.0, 1.0),
            wp.array([wp.transform_identity()], dtype=wp.transform),
            wp.array([wp.vec3(0.3, 0.3, 0.35)], dtype=wp.vec3),
            wp.array([wp.vec4(0.5, 0.5, 1.0, 0.0)], dtype=wp.vec4),
        )

        self.viewer.end_frame()
        self.time += 1.0 / 60.0

    def test_final(self):
        pass

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--usd-path", type=str, default="newton/_src/usd/cup_new.usdc")
        parser.add_argument("--rotation-speed", type=float, default=0.5)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)

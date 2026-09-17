# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate the W1 finger visuals with ordinary back-face culling enabled."""

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "examples/mjvbdv2/assets/w1_v030/w1-030/visual"
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


class TestW1GripperAssets(unittest.TestCase):
    def test_finger_shells_face_outward(self):
        """Each closed component must remain visible from outside the gripper."""
        import trimesh

        for side in ("left_hand", "right_hand"):
            for finger in (7, 8):
                with self.subTest(side=side, finger=finger):
                    mesh = trimesh.load(ASSETS / side / f"link{finger}.dae", force="mesh")
                    mesh.merge_vertices(merge_tex=True, merge_norm=True)
                    self.assertTrue(mesh.is_watertight)
                    self.assertTrue(mesh.is_winding_consistent)
                    for component in mesh.split():
                        self.assertGreater(component.volume, 0)

    def test_authored_normals_agree_with_faces(self):
        """Do not leave inverted lighting normals behind after repairing winding."""
        for side in ("left_hand", "right_hand"):
            for finger in (7, 8):
                with self.subTest(side=side, finger=finger):
                    root = ET.parse(ASSETS / side / f"link{finger}.dae").getroot()
                    for mesh in root.findall(".//c:geometry/c:mesh", NS):
                        arrays = {
                            source.attrib["id"]: np.fromstring(source.find("c:float_array", NS).text, sep=" ").reshape(
                                -1, int(source.find("c:technique_common/c:accessor", NS).attrib["stride"])
                            )
                            for source in mesh.findall("c:source", NS)
                        }
                        vertices = arrays[mesh.find("c:vertices/c:input", NS).attrib["source"][1:]]
                        for group in mesh.findall("c:triangles", NS):
                            inputs = {item.attrib["semantic"]: item.attrib for item in group.findall("c:input", NS)}
                            stride = max(int(item["offset"]) for item in inputs.values()) + 1
                            corners = np.fromstring(group.find("c:p", NS).text, sep=" ", dtype=int).reshape(
                                -1, 3, stride
                            )
                            points = vertices[corners[:, :, int(inputs["VERTEX"]["offset"])]]
                            face_normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
                            face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-30)
                            normal_input = inputs["NORMAL"]
                            normals = arrays[normal_input["source"][1:]][corners[:, :, int(normal_input["offset"])]]
                            self.assertGreaterEqual(float(np.einsum("fci,fi->fc", normals, face_normals).min()), -1e-4)


if __name__ == "__main__":
    unittest.main()

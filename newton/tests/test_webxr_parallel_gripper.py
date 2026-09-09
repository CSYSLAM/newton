# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Test optical gripper mapping without importing Newton/Warp or initializing CUDA.

Run with ``uv run --no-sync python newton/tests/test_webxr_parallel_gripper.py``.
"""

import importlib.util
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "parallel_gripper", _ROOT / "newton/examples/mjvbdv2/_webxr_parallel_gripper.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
ParallelGripperRetargeter = _MODULE.ParallelGripperRetargeter


def skeleton(span):
    points = np.zeros((25, 3))
    points[:, 1] = np.linspace(0, 0.15, 25)
    points[11] = [0, 0.08, 0]
    points[4] = [-span / 2, 0.12, 0]
    points[9] = [span / 2, 0.12, 0]
    return points


class TestParallelGripper(unittest.TestCase):
    def setUp(self):
        """Provide a minimal symmetric prismatic model independent of private assets."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.urdf = Path(self.directory.name) / "gripper.urdf"
        robot = ET.Element("robot", name="test")
        for side in ("LEFT", "RIGHT"):
            for index, axis in ((1, "0 1 0"), (2, "0 -1 0")):
                joint = ET.SubElement(robot, "joint", name=f"{side}_FINGER{index}_JOINT", type="prismatic")
                ET.SubElement(joint, "axis", xyz=axis)
                ET.SubElement(joint, "limit", lower="0", upper="0.05")
                if index == 2:
                    ET.SubElement(joint, "mimic", joint=f"{side}_FINGER1_JOINT")
        ET.ElementTree(robot).write(self.urdf)

    def test_trigger_and_pinch_share_open_closed_direction(self):
        """Keep trigger and optical pinch endpoints identical with symmetric jaws."""
        for side in ("left", "right"):
            mapper = ParallelGripperRetargeter(self.urdf, side=side)
            np.testing.assert_allclose(mapper.coordinates(0), [0.05, 0.05])
            np.testing.assert_allclose(mapper.coordinates(1), [0, 0])
            np.testing.assert_allclose(mapper.solve(skeleton(0.10)), mapper.coordinates(0))
            np.testing.assert_allclose(mapper.solve(skeleton(0.015)), mapper.coordinates(1), atol=1.0e-8)
            np.testing.assert_allclose(mapper.solve(skeleton(0.0575)), mapper.coordinates(0.5))
            self.assertAlmostEqual(mapper.closure(mapper.coordinates(0.5)), 0.5)

    def test_opening_is_monotonic_bounded_and_pose_invariant(self):
        """Open continuously without exceeding travel or depending on world placement."""
        mapper = ParallelGripperRetargeter(self.urdf)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        previous = -1.0
        for span in np.linspace(0, 0.18, 31):
            points = skeleton(span)
            q = mapper.solve(points)
            np.testing.assert_allclose(mapper.solve(points @ rotation.T + [2, -3, 1]), q, atol=1.0e-7)
            self.assertGreaterEqual(float(q[0]), previous)
            self.assertGreaterEqual(float(q[0]), -1.0e-8)
            self.assertLessEqual(float(q[0]), 0.050001)
            self.assertEqual(q[0], q[1])
            previous = float(q[0])

    def test_invalid_tracking_and_calibration_are_rejected(self):
        """Reject invalid skeletons and calibration before issuing jaw targets."""
        mapper = ParallelGripperRetargeter(self.urdf)
        for points in (np.zeros((25, 3)), np.zeros((24, 3)), np.full((25, 3), np.nan)):
            with self.assertRaises(ValueError):
                mapper.solve(points)
        for closed, opened in ((0.1, 0.1), (-0.1, 0.1), (0.2, 0.1), (0.0, np.inf)):
            with self.assertRaises(ValueError):
                ParallelGripperRetargeter(self.urdf, closed_span=closed, open_span=opened)

    @unittest.skipUnless(_MODULE.DEFAULT_GRIPPER_URDF.is_file(), "Download the private MR model to verify assets")
    def test_downloaded_model_has_tcp_mimic_and_all_meshes(self):
        """Verify the real MR model resolves every mesh and retains its authored TCP."""
        urdf = _MODULE.DEFAULT_GRIPPER_URDF
        mapper = ParallelGripperRetargeter(urdf)
        np.testing.assert_allclose(mapper.upper, [0.05, 0.05])
        robot = ET.parse(urdf)
        tcp = next(j for j in robot.iter("joint") if j.get("name") == "RIGHT_EE_TO_RIGHT_GRIPPER_TCP")
        np.testing.assert_allclose(np.fromstring(tcp.find("origin").get("xyz"), sep=" "), [0, 0, 0.14])
        for mesh in robot.iter("mesh"):
            self.assertTrue((urdf.parent / mesh.attrib["filename"]).is_file(), mesh.attrib["filename"])


if __name__ == "__main__":
    unittest.main()

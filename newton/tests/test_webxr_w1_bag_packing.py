# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise independent W1 V030 controller and optical gripper input."""

import unittest
from dataclasses import replace

import numpy as np

from newton.examples.mjvbdv2._webxr_gripper_input import GripperInput
from newton.examples.mjvbdv2._webxr_parallel_gripper import ParallelGripperRetargeter
from newton.examples.mjvbdv2._webxr_teleop import ControllerState, HandState, Pose, XRFrame
from newton.examples.mjvbdv2.example_mjvbd_v2_w1_pick_place import ASSET
from newton.tests.test_webxr_parallel_gripper import skeleton


def frame(sequence=0, *, mode="controllers", position=(0, 0, 0), activation=1, enabled=True, span=0.10):
    """Build a real protocol frame with independent left and right inputs."""
    pose = Pose(np.array(position, dtype=np.float32), np.array((0, 0, 0, 1), dtype=np.float32))
    controller = ControllerState("left", pose, True, False, (), (), (), (0, 0), 1.0)
    return XRFrame(
        "test",
        sequence,
        sequence * 16,
        0,
        "local-floor",
        "newton-world",
        "visible",
        {"left": controller},
        input_mode=mode,
        hands={"left": HandState(pose, skeleton(span), enabled, activation)},
    )


class TestPackingInput(unittest.TestCase):
    def setUp(self):
        """Seed both V030 grippers at their measured TCPs and jaw openings."""
        self.pose = Pose(np.array((0.3, 0.25, 1.2)), np.array((0, 0, 0, 1)))
        self.jaws = np.array((0.03, 0.03))
        self.inputs = {
            side: GripperInput(side, ParallelGripperRetargeter(ASSET, side=side), self.pose, self.jaws)
            for side in ("left", "right")
        }

    def update(self, value):
        """Consume one sample with measured robot state independent of requested motion."""
        for control in self.inputs.values():
            control.update(value, self.pose, self.jaws)

    def test_controller_clutch_rotation_and_independent_jaws(self):
        """Move only the clutched hand and map its trigger to symmetric closed jaws."""
        self.update(frame())
        value = frame(1, position=(0.1, 0, 0))
        rotated = replace(value.controllers["left"].pose, orientation=np.array((0, 0, 0.2, np.sqrt(0.96))))
        value = replace(value, controllers={"left": replace(value.controllers["left"], pose=rotated)})
        self.update(value)
        np.testing.assert_allclose(self.inputs["left"].position, self.pose.position + np.array((0.1, 0, 0)))
        np.testing.assert_allclose(self.inputs["left"].orientation, rotated.orientation, atol=1e-6)
        np.testing.assert_allclose(self.inputs["left"].jaws, (0, 0))
        np.testing.assert_allclose(self.inputs["right"].jaws, self.jaws)
        np.testing.assert_allclose(self.inputs["right"].position, self.pose.position)

    def test_optical_pinch_and_loss_require_new_activation(self):
        """Hold measured poses and jaw openings through loss until a fresh wrist baseline arrives."""
        self.update(frame(mode="hands", span=0.015))
        np.testing.assert_allclose(self.inputs["left"].jaws, (0, 0), atol=1e-7)
        self.update(frame(1, mode="hands", position=(0.08, 0, 0), span=0.10))
        np.testing.assert_allclose(self.inputs["left"].jaws, (0.05, 0.05))
        self.update(None)
        np.testing.assert_allclose(self.inputs["left"].jaws, self.jaws)
        self.update(frame(2, mode="hands", position=(1, 0, 0)))
        self.assertEqual(self.inputs["left"].status, "paused")
        np.testing.assert_allclose(self.inputs["left"].position, self.pose.position)
        self.update(frame(3, mode="hands", position=(1, 0, 0), activation=2))
        self.assertEqual(self.inputs["left"].status, "tracking")
        np.testing.assert_allclose(self.inputs["left"].position, self.pose.position)

    def test_visibility_mode_switch_and_invalid_skeleton_hold(self):
        """Disarm hidden sessions and reject skeleton jumps before they move the robot."""
        self.update(frame(mode="hands"))
        self.update(frame(1, mode="hands", position=(0.5, 0, 0)))
        self.assertEqual(self.inputs["left"].status, "invalid-tracking")
        np.testing.assert_allclose(self.inputs["left"].position, self.pose.position)
        self.update(replace(frame(2), visibility_state="hidden"))
        np.testing.assert_allclose(self.inputs["left"].jaws, self.jaws)
        self.update(frame(3, position=(0.8, 0, 0)))
        np.testing.assert_allclose(self.inputs["left"].position, self.pose.position)
        value = frame(4, mode="hands", activation=2)
        bad = replace(value.hands["left"], joints=np.zeros((25, 3)))
        self.update(replace(value, hands={"left": bad}))
        self.assertEqual(self.inputs["left"].status, "invalid-tracking")

    def test_duplicate_frame_and_released_clutch_do_not_advance(self):
        """Avoid reapplying duplicate input and hold immediately when the grip is released."""
        value = frame(mode="hands")
        self.update(value)
        self.update(replace(value, hands={"left": replace(value.hands["left"], joints=skeleton(0.015))}))
        np.testing.assert_allclose(self.inputs["left"].jaws, (0.05, 0.05))
        value = frame(1)
        self.update(replace(value, controllers={"left": replace(value.controllers["left"], clutch=False)}))
        self.assertFalse(self.inputs["left"].retargeter.active)
        np.testing.assert_allclose(self.inputs["left"].position, self.pose.position)

    def test_new_controller_stream_reanchors(self):
        """Re-anchor a reconnected controller instead of applying its old world offset."""
        self.update(frame())
        self.update(frame(1, position=(0.1, 0, 0)))
        self.update(replace(frame(2, position=(2, 0, 0)), stream_id="reconnected"))
        np.testing.assert_allclose(self.inputs["left"].position, self.pose.position)


if __name__ == "__main__":
    unittest.main()

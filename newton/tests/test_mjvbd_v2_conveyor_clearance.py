# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check independent robot collision queries."""

import unittest

import warp as wp

import newton
from newton.examples.mjvbdv2.support.conveyor_clearance import RobotClearanceAudit


class TestConveyorClearance(unittest.TestCase):
    def test_kinematic_overlap(self):
        """Detect overlapping kinematic arm and torso geometry through explicit pairs."""
        builder = newton.ModelBuilder()
        arm = builder.add_body(label="robot/right_j3")
        torso = builder.add_body(xform=wp.transform(wp.vec3(1.9, 0, 0), wp.quat_identity()), label="robot/waist")
        builder.body_flags[arm] = int(newton.BodyFlags.KINEMATIC)
        builder.body_flags[torso] = int(newton.BodyFlags.KINEMATIC)
        a = builder.add_shape_sphere(arm, radius=1.0)
        b = builder.add_shape_sphere(torso, radius=1.0)
        builder.add_shape_collision_filter_pair(a, b)
        model = builder.finalize(device="cpu")
        audit = RobotClearanceAudit(model, model.body_count)
        self.assertLess(audit.minimum_separation(model.state()), -0.09)
        state = model.state()
        poses = state.body_q.numpy()
        poses[torso, 0] = 2.1
        state.body_q.assign(poses)
        self.assertGreater(audit.minimum_separation(state), 0.0)


if __name__ == "__main__":
    unittest.main()

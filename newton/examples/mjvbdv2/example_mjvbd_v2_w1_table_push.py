# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Drive both W1 grippers toward targets below the worktop with two-way coupling.

    uv run --extra examples -m newton.examples mjvbd_v2_w1_table_push
    uv run --extra examples -m newton.examples mjvbd_v2_w1_table_push --coupling one_way

The worktop is raised to 1 m so that targets 8 cm below its surface stay
inside the arms' reach. The grippers point down with their fingers shut,
and the IK targets dive below the surface twice. The left gripper
presses on the bare worktop, which MuJoCo resolves as a link-vs-static
contact. The right gripper presses a dense block that VBD owns; the block's
contact wrench is returned to MuJoCo. Both arms stall on the surface while
their drives push down, then follow the targets back up. With
``--coupling one_way`` the arms ignore both obstacles and sweep through them.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mjvbdv2.support.w1_two_way import DOWNWARD_GRIPPER, FINGER_SHUT, W1TwoWayScene, smoothstep

TABLE_Z = 1.0
PRESS_X = 0.38
BLOCK_HALF = np.array((0.07, 0.07, 0.03))
HOVER = 0.12
DIVE = -0.08


class Example(W1TwoWayScene):
    table_z = TABLE_Z
    gripper_rotation = DOWNWARD_GRIPPER

    def initial_targets(self):
        return np.array(((PRESS_X, 0.25, TABLE_Z + HOVER), (PRESS_X, -0.25, TABLE_Z + HOVER)))

    def build_task(self, builder):
        self.press = self.initial_targets()
        self.block_center = np.array((PRESS_X, -0.25, TABLE_Z + BLOCK_HALF[2]))
        self.block = builder.add_body(
            xform=wp.transform(wp.vec3(*self.block_center), wp.quat_identity()), label="pressed_block"
        )
        self.add_box(
            builder,
            "pressed_block",
            None,
            BLOCK_HALF,
            (0.18, 0.55, 0.78),
            body=self.block,
            cfg=newton.ModelBuilder.ShapeConfig(density=2000.0, ke=1.0e5, kd=100.0, mu=1.0),
        )
        self.lowest_tcp = np.full(2, np.inf)
        self.lowest_block_bottom = np.inf
        self.max_press_error = 0.0

    def _keyframes(self):
        hover = self.press
        dive = self.press + np.array((0.0, 0.0, DIVE - HOVER))
        return [
            (0.0, self.home),
            (0.5, self.home),
            (1.5, hover),
            (3.0, dive),
            (4.5, dive),
            (5.5, hover),
            (6.5, hover),
            (7.5, dive),
            (9.0, dive),
            (10.0, hover),
        ]

    def plan(self, time):
        (targets,) = smoothstep(self._keyframes(), time)
        return targets, FINGER_SHUT

    def pressing(self, time):
        return 3.2 <= time <= 4.5 or 7.7 <= time <= 9.0

    def tracks_freely(self, time):
        return time <= 2.0 or time >= 10.8

    def after_step(self):
        body_q = self.state_0.body_q.numpy()
        tcp = self.tcp_positions(body_q)
        self.lowest_tcp = np.minimum(self.lowest_tcp, tcp[:, 2])
        self.lowest_block_bottom = min(self.lowest_block_bottom, float(body_q[self.block, 2]) - BLOCK_HALF[2])
        if self.pressing(self.sim_time):
            self.max_press_error = max(self.max_press_error, float(np.max(tcp[:, 2] - self.targets[:, 2])))
        if self.frame % 60 == 0:
            print(
                f"[W1TablePush] t={self.sim_time:4.1f} tcp_x={np.round(tcp[:, 0], 4).tolist()} tcp_z_above_table={np.round(tcp[:, 2] - TABLE_Z, 4).tolist()} "
                f"target_z_above_table={np.round(self.targets[:, 2] - TABLE_Z, 4).tolist()} "
                f"block_bottom={float(body_q[self.block, 2]) - BLOCK_HALF[2] - TABLE_Z:+.4f}",
                flush=True,
            )

    def test_post_step(self):
        self.check_finite()
        if self.args.coupling != "two_way":
            return
        # The shut fingertips reach about 2 cm past the TCP.
        if self.lowest_tcp[0] < TABLE_Z + 0.01:
            raise AssertionError(f"The left gripper pushed into the worktop: {self.lowest_tcp[0] - TABLE_Z}")
        if self.lowest_tcp[1] < TABLE_Z + 2.0 * BLOCK_HALF[2] + 0.01:
            raise AssertionError(f"The right gripper pushed into the block: {self.lowest_tcp[1] - TABLE_Z}")
        if self.lowest_block_bottom < TABLE_Z - 0.004:
            raise AssertionError(f"The block was driven into the worktop: {self.lowest_block_bottom - TABLE_Z}")

    def test_final(self):
        self.test_post_step()
        if self.args.coupling != "two_way":
            return
        if self.frame < 720:
            raise AssertionError("Run at least 720 frames to cover both dives and the return")
        if self.max_press_error < 0.06:
            raise AssertionError(f"The drives never pushed below the obstacles: {self.max_press_error}")
        if self.peak_free_tcp_error > 0.01:
            raise AssertionError(f"TCP error in free motion exceeded 1 cm: {self.peak_free_tcp_error}")
        # Stalled drives do not push straight down, so the block may creep,
        # but it must stay under the right gripper.
        block = self.state_0.body_q.numpy()[self.block]
        if np.any(np.abs(block[:2] - self.press[1, :2]) > BLOCK_HALF[:2] - 0.01):
            raise AssertionError(f"The pressed block left the right gripper: {block[:3]}")
        print("[W1TablePush] PASS: both grippers stalled on their obstacles and recovered.", flush=True)

    @staticmethod
    def create_parser():
        return W1TwoWayScene.add_arguments(newton.examples.create_parser(), num_frames=720)


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

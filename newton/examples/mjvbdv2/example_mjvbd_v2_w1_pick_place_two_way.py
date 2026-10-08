# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run the W1 V030 two-block pick and place on two-way MJVBDV2 coupling.

    uv run --extra examples -m newton.examples mjvbd_v2_w1_pick_place_two_way

This is the ``mjvbd_v2_w1_pick_place`` task with a dynamic robot. Instead of
prescribing the fingers 1.5 mm inside the blocks, the finger targets go fully
closed and VBD contact feedback stops them on the block faces, so only the
10 N finger effort and friction hold each block. MuJoCo also keeps the wrists
from passing through the worktop, so the grasp point is raised 1.5 cm above
the block centre and moved back to center the finger pads, which sit beyond
the TCP, on the block.

A friction grip does not release a tall block perfectly symmetrically, so the
original 12 cm drop over the bin rim topples it unpredictably. The bins are
therefore trays that are open toward the robot: the horizontal gripper sets
each block on the tray floor, opens, and backs out.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mjvbdv2.example_mjvbd_v2_w1_pick_place import HALF_SIZE
from newton.examples.mjvbdv2.support.w1_two_way import (
    FINGER_OPEN,
    FINGER_SHUT,
    SIDES,
    TABLE_Z,
    W1TwoWayScene,
    smoothstep,
)

GRASP_RAISE = 0.015
TRAY_FLOOR = 0.018
COLORS = ((0.18, 0.55, 0.78), (0.88, 0.39, 0.15))


class Example(W1TwoWayScene):
    def initial_targets(self):
        return self.block_start + self.grasp_offset

    @property
    def block_start(self):
        return np.array(((0.43, 0.25, TABLE_Z + HALF_SIZE[2]), (0.43, -0.25, TABLE_Z + HALF_SIZE[2])))

    @property
    def grasp_offset(self):
        return np.array((-self.args.grasp_x_offset, 0.0, GRASP_RAISE))

    def build_task(self, builder):
        self.pick = self.block_start + self.grasp_offset
        self.bins = np.array(((0.34, 0.50, TABLE_Z), (0.34, -0.50, TABLE_Z)))
        for side, center, color in zip(SIDES, self.bins, COLORS, strict=True):
            self.add_box(builder, f"{side}_bin_floor", center + np.array((0, 0, 0.009)), (0.11, 0.13, 0.009), color)
            self.add_box(builder, f"{side}_bin_end", center + np.array((0.105, 0, 0.065)), (0.005, 0.13, 0.065), color)
            for sign in (-1, 1):
                self.add_box(
                    builder,
                    f"{side}_bin_side",
                    center + np.array((0, sign * 0.125, 0.065)),
                    (0.10, 0.005, 0.065),
                    color,
                )
        self.objects = []
        for side, position, color in zip(SIDES, self.block_start, COLORS, strict=True):
            body = builder.add_body(xform=wp.transform(wp.vec3(*position), wp.quat_identity()), label=f"{side}_block")
            self.add_box(
                builder,
                f"{side}_block",
                None,
                HALF_SIZE,
                color,
                body=body,
                cfg=newton.ModelBuilder.ShapeConfig(density=450.0, mu=1.0, ke=2.0e4, kd=100.0),
            )
            self.objects.append(body)
        self.peak_z = self.block_start[:, 2].copy()
        self.carry_fingers = []
        self.peak_carry_tilt = np.zeros(2)

    def _keyframes(self):
        lift = self.pick + np.array((0, 0, 0.25))
        carry = self.bins + self.grasp_offset + np.array((0, 0, HALF_SIZE[2] + 0.25))
        # Set the block 4 mm above the tray floor, then back out of the open side.
        place = self.bins + self.grasp_offset + np.array((0, 0, TRAY_FLOOR + HALF_SIZE[2] + 0.004))
        approach = self.pick + np.array((-0.13, 0, 0))
        backed_out = place + np.array((-0.13, 0, 0))
        retreat = backed_out + np.array((0, 0, 0.20))
        return [
            (0.0, self.home, FINGER_OPEN),
            (0.5, self.home, FINGER_OPEN),
            (1.5, approach, FINGER_OPEN),
            (2.8, self.pick, FINGER_OPEN),
            (3.8, self.pick, FINGER_SHUT),
            (5.5, lift, FINGER_SHUT),
            (7.5, carry, FINGER_SHUT),
            (8.8, place, FINGER_SHUT),
            (9.6, place, FINGER_OPEN),
            (10.6, backed_out, FINGER_OPEN),
            (11.8, retreat, FINGER_OPEN),
        ]

    def plan(self, time):
        targets, finger = smoothstep(self._keyframes(), time)
        return targets, float(finger)

    def tracks_freely(self, time):
        return time <= 2.8 or time >= 10.6

    def block_tilt(self, body_q):
        """Angle [deg] between each block's up axis and world up."""
        return np.array(
            [
                np.degrees(
                    np.arccos(np.clip(wp.quat_rotate(wp.quat(*body_q[o, 3:]), wp.vec3(0.0, 0.0, 1.0))[2], -1, 1))
                )
                for o in self.objects
            ]
        )

    def after_step(self):
        body_q = self.state_0.body_q.numpy()
        self.peak_z = np.maximum(self.peak_z, body_q[self.objects, 2])
        if 4.3 <= self.sim_time <= 8.5:
            self.carry_fingers.append(self.finger_q())
            self.peak_carry_tilt = np.maximum(self.peak_carry_tilt, self.block_tilt(body_q))
        if self.frame % 60 == 0:
            print(
                f"[W1PickPlaceTwoWay] t={self.sim_time:4.1f} blocks={body_q[self.objects, :3].round(4).tolist()} "
                f"finger_q[mm]={np.round(self.finger_q() * 1e3, 2).tolist()} "
                f"tilt[deg]={np.round(self.block_tilt(body_q), 2).tolist()}",
                flush=True,
            )

    def test_post_step(self):
        self.check_finite()
        if np.min(self.state_0.body_q.numpy()[self.objects, 2]) < TABLE_Z - 0.01:
            raise AssertionError("A block fell through or off the table")

    def test_final(self):
        """Require finger stops on the blocks, both lifts, and both blocks settled in their bins."""
        self.test_post_step()
        if self.frame < 840:
            raise AssertionError("Run at least 840 frames to validate both placements")
        fingers = np.asarray(self.carry_fingers)
        if fingers.min() < HALF_SIZE[1] - 0.002:
            raise AssertionError(f"Shut fingers closed into a block: {fingers.min()}")
        if np.any(self.peak_carry_tilt > 2.0):
            raise AssertionError(f"A block pivoted in the grasp: {self.peak_carry_tilt} deg")
        if np.any(self.peak_z < self.block_start[:, 2] + 0.18):
            raise AssertionError(f"Both blocks must be lifted at least 18 cm: {self.peak_z}")
        if self.peak_free_tcp_error > 0.01:
            raise AssertionError(f"TCP error in free motion exceeded 1 cm: {self.peak_free_tcp_error}")
        poses = self.state_0.body_q.numpy()[self.objects]
        for pose, center in zip(poses, self.bins, strict=True):
            rotation = np.asarray(wp.quat_to_matrix(wp.quat(*pose[3:]))).reshape(3, 3)
            extent = np.abs(rotation) @ HALF_SIZE
            if np.any(np.abs(pose[:2] - center[:2]) + extent[:2] > (0.10, 0.12)):
                raise AssertionError(f"Block not fully inside its bin: {pose}")
            if abs(pose[2] - extent[2] - TABLE_Z - TRAY_FLOOR) > 0.006:
                raise AssertionError(f"Block not resting on bin floor: {pose}")
        if np.max(np.linalg.norm(self.state_0.body_qd.numpy()[self.objects], axis=1)) > 0.05:
            raise AssertionError("Released blocks have not settled")
        if np.any(np.abs(self.finger_q() - FINGER_OPEN) > 1.0e-3):
            raise AssertionError("Both grippers must be open after release")
        print(
            f"[W1PickPlaceTwoWay] PASS: fingers stopped at {fingers.min() * 1e3:.1f} mm, "
            f"peak carry tilt {self.peak_carry_tilt.max():.2f} deg, both blocks placed.",
            flush=True,
        )

    @staticmethod
    def create_parser():
        parser = W1TwoWayScene.add_arguments(newton.examples.create_parser(), num_frames=900)
        parser.add_argument(
            "--grasp-x-offset",
            type=float,
            default=0.02,
            help="Distance [m] the TCP stops short of the block centre so the finger pads center on it.",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

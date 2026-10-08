# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Command both W1 grippers fully shut on a rigid block and a soft cube.

    uv run --extra examples -m newton.examples mjvbd_v2_w1_squeeze_grasp
    uv run --extra examples -m newton.examples mjvbd_v2_w1_squeeze_grasp --coupling one_way

Both finger targets go to fully closed, as with a MuJoCo position actuator.
With two-way coupling, the VBD contact wrenches stop the left fingers on the
rigid block faces and let the right fingers compress the tetrahedral cube only
until its elastic force balances the 10 N finger effort limit. The robot then
lifts, holds, lowers, and releases both objects. With ``--coupling one_way``
the fingers close through both objects.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mjvbdv2.support.w1_two_way import (
    FINGER_OPEN,
    FINGER_SHUT,
    TABLE_Z,
    W1TwoWayScene,
    smoothstep,
)

BLOCK_HALF = np.array((0.024, 0.024, 0.045))
SOFT_SIZE = np.array((0.06, 0.06, 0.08))
SOFT_CELLS = (5, 5, 6)
GRASP_HEIGHT = 0.06
LIFT = 0.15
HOLD = (4.4, 7.4)


class Example(W1TwoWayScene):
    def initial_targets(self):
        return np.array(((0.43, 0.25, TABLE_Z + GRASP_HEIGHT), (0.43, -0.25, TABLE_Z + GRASP_HEIGHT)))

    def build_task(self, builder):
        self.grasp = self.initial_targets()
        self.block = builder.add_body(
            xform=wp.transform(wp.vec3(0.43, 0.25, TABLE_Z + BLOCK_HALF[2]), wp.quat_identity()), label="rigid_block"
        )
        self.add_box(
            builder,
            "rigid_block",
            None,
            BLOCK_HALF,
            (0.18, 0.55, 0.78),
            body=self.block,
            cfg=newton.ModelBuilder.ShapeConfig(density=450.0, ke=2.0e4, kd=100.0, mu=1.0),
        )
        self.soft_begin = builder.particle_count
        builder.add_soft_grid(
            pos=wp.vec3(0.43 - SOFT_SIZE[0] / 2, -0.25 - SOFT_SIZE[1] / 2, TABLE_Z + 0.002),
            rot=wp.quat_identity(),
            vel=wp.vec3(),
            dim_x=SOFT_CELLS[0],
            dim_y=SOFT_CELLS[1],
            dim_z=SOFT_CELLS[2],
            cell_x=SOFT_SIZE[0] / SOFT_CELLS[0],
            cell_y=SOFT_SIZE[1] / SOFT_CELLS[1],
            cell_z=SOFT_SIZE[2] / SOFT_CELLS[2],
            density=250.0,
            k_mu=6.0e3,
            k_lambda=1.5e4,
            k_damp=1.0e-3,
            particle_radius=0.003,
        )
        self.soft_end = builder.particle_count
        self.hold_fingers = []
        self.peak_lift = np.zeros(2)
        self.lowest_particle = np.inf

    def configure_model(self, model):
        # Keep the particle contact stiffer than the cube so the squeeze
        # deforms the elastic body instead of the contact layer.
        model.soft_contact_ke = 2.0e4
        model.soft_contact_mu = 1.0

    def collision_options(self):
        return {"soft_contact_margin": 0.003}

    def _keyframes(self):
        behind = self.grasp + np.array((-0.13, 0.0, 0.0))
        lifted = self.grasp + np.array((0.0, 0.0, LIFT))
        return [
            (0.0, self.home, FINGER_OPEN),
            (0.5, self.home, FINGER_OPEN),
            (1.5, behind, FINGER_OPEN),
            (2.8, self.grasp, FINGER_OPEN),
            (3.8, self.grasp, FINGER_SHUT),
            (4.3, self.grasp, FINGER_SHUT),
            (5.8, lifted, FINGER_SHUT),
            (7.3, lifted, FINGER_SHUT),
            (8.8, self.grasp, FINGER_SHUT),
            (9.6, self.grasp, FINGER_OPEN),
            (10.6, behind, FINGER_OPEN),
        ]

    def plan(self, time):
        targets, finger = smoothstep(self._keyframes(), time)
        return targets, float(finger)

    def tracks_freely(self, time):
        return time <= 2.8 or time >= 10.6

    def object_heights(self):
        body_q = self.state_0.body_q.numpy()
        particles = self.state_0.particle_q.numpy()[self.soft_begin : self.soft_end]
        return np.array((float(body_q[self.block, 2]) - BLOCK_HALF[2], float(particles[:, 2].min()))), particles

    def after_step(self):
        bottoms, particles = self.object_heights()
        self.peak_lift = np.maximum(self.peak_lift, bottoms - TABLE_Z)
        self.lowest_particle = min(self.lowest_particle, float(particles[:, 2].min()))
        fingers = self.finger_q()
        if HOLD[0] <= self.sim_time <= HOLD[1]:
            self.hold_fingers.append(fingers)
        if self.frame % 60 == 0:
            width = float(np.ptp(particles[:, 1]))
            print(
                f"[W1SqueezeGrasp] t={self.sim_time:4.1f} finger_q[mm]={np.round(fingers * 1e3, 2).tolist()} "
                f"lift[cm]={np.round((bottoms - TABLE_Z) * 100, 2).tolist()} soft_width[mm]={width * 1e3:.1f}",
                flush=True,
            )

    def test_post_step(self):
        self.check_finite()
        if self.args.coupling != "two_way":
            return
        if self.lowest_particle < TABLE_Z - 0.004:
            raise AssertionError(f"The soft cube sank into the worktop: {self.lowest_particle - TABLE_Z}")

    def test_final(self):
        self.test_post_step()
        if self.args.coupling != "two_way":
            return
        if self.frame < 720:
            raise AssertionError("Run at least 720 frames to cover the grasp, hold, and release")
        hold = np.asarray(self.hold_fingers)
        rigid, soft = hold[:, 0], hold[:, 1]
        # Shut fingers must stop on the rigid faces instead of closing to zero.
        if rigid.min() < BLOCK_HALF[1] - 0.002:
            raise AssertionError(f"The left fingers crushed the rigid block: {rigid.min()}")
        # The soft cube is squeezed, but its elastic force stops the fingers.
        if soft.mean() > SOFT_SIZE[1] / 2 - 0.002:
            raise AssertionError(f"The right fingers did not indent the soft cube: {soft.mean()}")
        if soft.min() < 0.01:
            raise AssertionError(f"The right fingers crushed the soft cube: {soft.min()}")
        if np.any(self.peak_lift < LIFT - 0.03):
            raise AssertionError(f"Both objects must be lifted: {self.peak_lift}")
        if self.peak_free_tcp_error > 0.01:
            raise AssertionError(f"TCP error in free motion exceeded 1 cm: {self.peak_free_tcp_error}")
        bottoms, particles = self.object_heights()
        if np.any(np.abs(bottoms - TABLE_Z) > 0.01):
            raise AssertionError(f"Released objects are not back on the worktop: {bottoms - TABLE_Z}")
        if np.ptp(particles[:, 1]) < SOFT_SIZE[1] - 0.006:
            raise AssertionError("The soft cube did not recover its width after release")
        print(
            f"[W1SqueezeGrasp] PASS: rigid fingers held at {rigid.min() * 1e3:.1f} mm, "
            f"soft cube indented {(SOFT_SIZE[1] / 2 - soft.mean()) * 1e3:.1f} mm per side; "
            "both objects lifted and released.",
            flush=True,
        )

    @staticmethod
    def create_parser():
        return W1TwoWayScene.add_arguments(newton.examples.create_parser(), num_frames=720)


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

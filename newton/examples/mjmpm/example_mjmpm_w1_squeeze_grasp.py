# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Command both W1 grippers fully shut on a rigid block and a soft cube.

    uv run --extra examples -m newton.examples mjmpm_w1_squeeze_grasp
    uv run --extra examples -m newton.examples mjmpm_w1_squeeze_grasp --coupling one_way

Both finger targets go to fully closed, as with a MuJoCo position actuator.
With two-way coupling, the MPM contact wrenches stop the left fingers on the
rigid block faces and let the right fingers compress the MPM elastic cube only
until its elastic force balances the 10 N finger effort limit. The robot then
lifts, holds, lowers, and releases both objects. With ``--coupling one_way``
the fingers close through both objects.
"""

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mjmpm.support.w1_two_way import (
    FINGER_OPEN,
    FINGER_SHUT,
    TABLE_Z,
    W1MJMPMScene,
    smoothstep,
)

BLOCK_HALF = np.array((0.024, 0.024, 0.045))
SOFT_SIZE = np.array((0.06, 0.06, 0.08))
SOFT_CELLS = (6, 6, 8)
PARTICLE_RADIUS = 0.005
GRASP_HEIGHT = 0.06
LIFT = 0.15
HOLD = (4.4, 7.4)
# Lifted and stationary: the grip should not visibly pulse.
STEADY = (6.0, 7.3)
STEADY_P2P = 0.0005
# Both finger drives saturate while squeezing, so nothing centers the held
# object between them; it must not slide noticeably along the finger axis.
MAX_SLIDE = 0.003


class Example(W1MJMPMScene):
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
        # Match the reference cube dimensions, density and small-strain Lame
        # parameters, using material particles instead of tetrahedra.
        spacing = SOFT_SIZE / np.array(SOFT_CELLS)
        builder.add_particle_grid(
            pos=wp.vec3(
                0.43 - SOFT_SIZE[0] / 2 + PARTICLE_RADIUS,
                -0.25 - SOFT_SIZE[1] / 2 + PARTICLE_RADIUS,
                TABLE_Z + 0.002 + PARTICLE_RADIUS,
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=SOFT_CELLS[0],
            dim_y=SOFT_CELLS[1],
            dim_z=SOFT_CELLS[2],
            cell_x=spacing[0],
            cell_y=spacing[1],
            cell_z=spacing[2],
            mass=250.0 * float(np.prod(spacing)),
            jitter=0.0,
            radius_mean=PARTICLE_RADIUS,
            custom_attributes={
                "mpm:young_modulus": 6.0e3 * (3 * 1.5e4 + 2 * 6.0e3) / (1.5e4 + 6.0e3),
                "mpm:poisson_ratio": 1.5e4 / (2 * (1.5e4 + 6.0e3)),
                "mpm:yield_stress": 1.0e6,
                "mpm:tensile_yield_ratio": 1.0,
            },
        )
        self.soft_end = builder.particle_count
        self.hold_fingers = []
        self.hold_slides = []
        self.steady_fingers = []
        self.steady_rigid_poses = []
        self.peak_lift = np.zeros(2)
        self.lowest_particle = np.inf

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
        orientation = np.asarray(wp.quat_to_matrix(wp.quat(*body_q[self.block, 3:]))).reshape(3, 3)
        rigid_half_height = float(np.abs(orientation[2]) @ BLOCK_HALF)
        return np.array(
            (float(body_q[self.block, 2]) - rigid_half_height, float(particles[:, 2].min()) - PARTICLE_RADIUS)
        ), particles

    def after_step(self):
        bottoms, particles = self.object_heights()
        self.peak_lift = np.maximum(self.peak_lift, bottoms - TABLE_Z)
        self.lowest_particle = min(self.lowest_particle, float(particles[:, 2].min()) - PARTICLE_RADIUS)
        fingers = self.finger_q()
        pairs = self.finger_pairs()
        if HOLD[0] <= self.sim_time <= HOLD[1]:
            # Half the finger gap, and the object's offset from the gripper center.
            self.hold_fingers.append(pairs.mean(axis=1))
            self.hold_slides.append(0.5 * (pairs[:, 0] - pairs[:, 1]))
        if STEADY[0] <= self.sim_time <= STEADY[1]:
            self.steady_fingers.append(fingers)
            self.steady_rigid_poses.append(self.state_0.body_q.numpy()[self.block].copy())
        if self.frame % 60 == 0:
            width = float(np.ptp(particles[:, 1])) + 2 * PARTICLE_RADIUS
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
        if self.sim_time <= 2.8:
            _, particles = self.object_heights()
            if np.ptp(particles[:, 2]) + 2 * PARTICLE_RADIUS < 0.9 * SOFT_SIZE[2]:
                raise AssertionError("The material collapsed before grasp contact")
        if STEADY[0] <= self.sim_time <= STEADY[1] and self.finger_pairs()[0].mean() < BLOCK_HALF[1] - 0.002:
            raise AssertionError("The rigid block escaped the gripper during the hold")

    def test_final(self):
        self.test_post_step()
        # Captured frames cannot grow the MPM capacities sized before capture.
        self.solver.check_capacity_status()
        if self.solver.ownership.mujoco_bodies != tuple(range(self.robot_bodies)):
            raise AssertionError("MuJoCo must own the complete dynamic W1 articulation")
        if self.solver.ownership.mpm_bodies != (self.block,):
            raise AssertionError("The MPM environment must own the free rigid block")
        robot_flags = self.model.body_flags.numpy()[: self.robot_bodies]
        if np.any(robot_flags & int(newton.BodyFlags.KINEMATIC)):
            raise AssertionError("The robot must be dynamic")
        if self.args.coupling != "two_way":
            return
        if self.frame < 720:
            raise AssertionError("Run at least 720 frames to cover the grasp, hold, and release")
        hold = np.asarray(self.hold_fingers)
        rigid, soft = hold[:, 0], hold[:, 1]
        # Shut fingers must stop on the rigid faces instead of closing to zero.
        if rigid.min() < BLOCK_HALF[1] - 0.002:
            raise AssertionError(f"The left fingers crushed the rigid block: half gap {rigid.min()}")
        slide = np.max(np.abs(np.asarray(self.hold_slides)), axis=0)
        if np.any(slide > MAX_SLIDE):
            raise AssertionError(f"The held objects slid between the fingers: {slide * 1e3} mm")
        # The soft cube is squeezed, but its elastic force stops the fingers.
        if soft.mean() > SOFT_SIZE[1] / 2 - 0.002:
            raise AssertionError(f"The right fingers did not indent the soft cube: {soft.mean()}")
        if soft.min() < 0.01:
            raise AssertionError(f"The right fingers crushed the soft cube: {soft.min()}")
        steady = np.ptp(np.asarray(self.steady_fingers), axis=0)
        if np.any(steady > STEADY_P2P):
            raise AssertionError(f"The held grip pulses: finger peak-to-peak {steady * 1e3} mm")
        poses = np.asarray(self.steady_rigid_poses)
        translation_jitter = float(np.max(np.linalg.norm(poses[:, :3] - poses[0, :3], axis=1)))
        quaternion_dot = np.clip(np.abs(poses[:, 3:] @ poses[0, 3:]), 0.0, 1.0)
        rotation_jitter = float(np.max(2.0 * np.arccos(quaternion_dot)))
        if translation_jitter > 0.005 or rotation_jitter > np.deg2rad(2.0):
            raise AssertionError(
                "The rigid block drifted while held: "
                f"translation {translation_jitter * 1e3:.2f} mm, rotation {np.rad2deg(rotation_jitter):.2f} deg"
            )
        if np.any(self.peak_lift < LIFT - 0.03):
            raise AssertionError(f"Both objects must be lifted: {self.peak_lift}")
        if self.peak_free_tcp_error > 0.01:
            raise AssertionError(f"TCP error in free motion exceeded 1 cm: {self.peak_free_tcp_error}")
        bottoms, particles = self.object_heights()
        if np.any(np.abs(bottoms - TABLE_Z) > 0.01):
            raise AssertionError(f"Released objects are not back on the worktop: {bottoms - TABLE_Z}")
        if np.ptp(particles[:, 1]) + 2 * PARTICLE_RADIUS < SOFT_SIZE[1] - 0.006:
            raise AssertionError("The soft cube did not recover its width after release")
        if np.ptp(particles[:, 1]) + 2 * PARTICLE_RADIUS > SOFT_SIZE[1] * 1.12:
            raise AssertionError("The released elastic material retained excessive lateral deformation")
        if np.ptp(particles[:, 2]) + 2 * PARTICLE_RADIUS > SOFT_SIZE[2] * 1.16:
            raise AssertionError("The released elastic material retained excessive vertical deformation")
        print(
            f"[W1SqueezeGrasp] PASS: rigid fingers held at {rigid.min() * 1e3:.1f} mm, "
            f"peak slide {np.round(slide * 1e3, 2).tolist()} mm, "
            f"held-grip jitter {np.round(steady * 1e3, 2).tolist()} mm, "
            f"rigid-block drift {translation_jitter * 1e3:.2f} mm / {np.rad2deg(rotation_jitter):.2f} deg, "
            f"soft cube indented {(SOFT_SIZE[1] / 2 - soft.mean()) * 1e3:.1f} mm per side; "
            "both objects lifted and released.",
            flush=True,
        )

    @staticmethod
    def create_parser():
        return W1MJMPMScene.add_arguments(newton.examples.create_parser(), num_frames=720)


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

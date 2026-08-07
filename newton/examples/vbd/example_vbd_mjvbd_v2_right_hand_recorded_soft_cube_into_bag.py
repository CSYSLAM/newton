# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Place a soft cube into a soft bag with two recorded right-hand poses.

This example loads only the floating W1 right-hand URDF. It uses the recorded
target-point hand pose for the initial approach and the latest recorder
keyframe for the five-finger grasp. The cube is held and released only through
MJVBD-v2 rigid-soft contact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd import example_vbd_mjvbd_v2_right_hand_soft_cube_recorder as recorder
from newton.examples.vbd import example_vbd_mjvbd_v2_dexforce_grasp_rigid_into_bag_soft0 as full_reference


DEFAULT_GRASP_KEYFRAME = Path("vbd_w1_right_hand_last_keyframe.json")

# Previously recorded pose at the target pick point, before final closure.
APPROACH_ROOT = wp.transform(
    wp.vec3(-0.16214203834533691, -2.838686943054199, 1.3409454822540283),
    wp.quat(0.09465623646974564, 0.9546480774879456, -0.2820824682712555, 0.010803722776472569),
)
APPROACH_JOINTS = {
    "RIGHT_HAND_THUMB1": 6.0,
    "RIGHT_HAND_THUMB2": 90.0,
    "RIGHT_HAND_INDEX": 41.0,
    "RIGHT_INDEX_PIP": 24.0,
    "RIGHT_HAND_MIDDLE": 57.0,
    "RIGHT_MIDDLE_PIP": 0.0,
    "RIGHT_HAND_RING": 48.0,
    "RIGHT_RING_PIP": 15.0,
    "RIGHT_HAND_PINKY": 24.0,
    "RIGHT_PINKY_PIP": 26.0,
}
OPEN_JOINTS = {name: 0.0 for name in recorder.HAND_JOINTS}

# Keep material, contact, and solver settings identical to the full-robot
# reference; only the articulated robot is replaced with the isolated hand.
recorder.CUBE_DENSITY = 100.0
recorder.CUBE_DIMS = full_reference.SOFT_CUBE_DIMS
recorder.CUBE_K_MU = full_reference.SOFT_CUBE_K_MU
recorder.CUBE_K_LAMBDA = full_reference.SOFT_CUBE_K_LAMBDA
recorder.CUBE_K_DAMP = full_reference.SOFT_CUBE_K_DAMP
recorder.CUBE_PARTICLE_RADIUS = full_reference.SOFT_CUBE_PARTICLE_RADIUS
recorder.CUBE_SELF_CONTACT_RADIUS = 0.003
recorder.CUBE_SELF_CONTACT_MARGIN = 0.006
recorder.CONTACT_KE = full_reference.GRASP_CONTACT_KE
recorder.CONTACT_KD = full_reference.GRASP_CONTACT_KD
recorder.CONTACT_MU = full_reference.GRASP_FRICTION
recorder.CONTACT_MARGIN = full_reference.SOFT_CONTACT_MARGIN
recorder.RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = full_reference.RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE

# Match the stable suspended-bag material from example_vbd_soft_rigid_mix_contact.
recorder.BAG_DENSITY = 0.08
recorder.BAG_TRI_KE = 1.0e5
recorder.BAG_TRI_KA = 1.0e5
recorder.BAG_TRI_KD = 1.0e2
recorder.BAG_EDGE_KE = 50.0
recorder.BAG_EDGE_KD = 0.5


class Example(recorder.Example):
    """Run the recorded right-hand grasp and physical bag placement."""

    def __init__(self, viewer, args):
        self.include_bag = True
        # Unlike the rigid bodies in example_vbd_soft_rigid_mix_contact, both
        # the cube and bag are particle soft bodies here. This must remain on
        # for cube-to-bag particle contacts at the bag floor and walls.
        self.particle_self_contact_enabled = True
        self.grasp_root, self.grasp_joints = self._load_grasp_keyframe(args.grasp_keyframe)
        self.release_friction_applied = False
        self.hand_soft_contact_enabled = True
        super().__init__(viewer, args)
        self._set_hand_target(APPROACH_ROOT, APPROACH_JOINTS)
        self._set_initial_hand_pose()
        self._set_hand_soft_contact(False)
        self.segments = self._build_segments()
        self.script_duration = sum(segment[0] for segment in self.segments)

    @staticmethod
    def _load_grasp_keyframe(path_value: str):
        """Load the most recently recorded physical grasp keyframe."""

        path = Path(path_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Recorded grasp keyframe not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        keyframe = payload["keyframe"]
        root = keyframe["target_root_pose"]
        position = root["position_m"]
        rotation = root["quaternion_xyzw"]
        joints = keyframe["target_finger_joints_degrees"]
        if len(position) != 3 or len(rotation) != 4:
            raise ValueError(f"Invalid root pose in recorded grasp keyframe: {path}")
        return wp.transform(wp.vec3(*position), wp.quat(*rotation)), {name: float(value) for name, value in joints.items()}

    def _set_hand_target(self, root: wp.transform, joints: dict[str, float]):
        """Set the next kinematic root and five-finger target without moving particles."""

        target_q = self.manual_target_q.numpy()
        position = wp.transform_get_translation(root)
        rotation = wp.transform_get_rotation(root)
        target_q[self.root_q_start : self.root_q_start + 7] = [*position, *rotation]
        for name, index in self.hand_joint_indices.items():
            target_q[index] = np.radians(joints[name])
        self.manual_target_q.assign(target_q)

    def _build_segments(self):
        """Build approach, closure, transport, release, and retreat phases."""

        grasp_position = wp.transform_get_translation(APPROACH_ROOT)
        cube_position = recorder.CUBE_CENTRE
        root_cube_offset = grasp_position - cube_position
        cube_release_height = float(recorder.BAG_POS[2]) + recorder.BAG_HEIGHT + 0.06
        bag_hover = wp.transform(
            wp.vec3(
                float(recorder.BAG_POS[0]) + float(root_cube_offset[0]),
                float(recorder.BAG_POS[1]) + float(root_cube_offset[1]),
                cube_release_height + float(root_cube_offset[2]),
            ),
            wp.transform_get_rotation(APPROACH_ROOT),
        )
        lift = wp.transform(
            grasp_position + wp.vec3(0.0, 0.0, 0.07),
            wp.transform_get_rotation(APPROACH_ROOT),
        )
        retreat = wp.transform(
            wp.transform_get_translation(bag_hover) + wp.vec3(0.0, 0.0, 0.10),
            wp.transform_get_rotation(APPROACH_ROOT),
        )
        return (
            (0.50, APPROACH_ROOT, APPROACH_ROOT, APPROACH_JOINTS, APPROACH_JOINTS, False),
            (1.80, APPROACH_ROOT, APPROACH_ROOT, APPROACH_JOINTS, self.grasp_joints, False),
            (0.60, APPROACH_ROOT, APPROACH_ROOT, self.grasp_joints, self.grasp_joints, False),
            (1.20, APPROACH_ROOT, lift, self.grasp_joints, self.grasp_joints, False),
            (7.00, lift, bag_hover, self.grasp_joints, self.grasp_joints, False),
            (0.40, bag_hover, bag_hover, self.grasp_joints, self.grasp_joints, False),
            (0.25, bag_hover, bag_hover, self.grasp_joints, OPEN_JOINTS, True),
            (0.90, bag_hover, bag_hover, OPEN_JOINTS, OPEN_JOINTS, True),
            (1.00, bag_hover, retreat, OPEN_JOINTS, OPEN_JOINTS, True),
        )

    def _sample(self, time_s: float):
        """Interpolate the recorded root and joint targets at a script time."""

        for duration, root_a, root_b, joints_a, joints_b, release in self.segments:
            if time_s <= duration:
                alpha = float(np.clip(time_s / duration, 0.0, 1.0))
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                root = self._lerp_transform(root_a, root_b, alpha)
                joints = {name: joints_a[name] * (1.0 - alpha) + joints_b[name] * alpha for name in recorder.HAND_JOINTS}
                return root, joints, release
            time_s -= duration
        _, _, root, _, joints, release = self.segments[-1]
        return root, joints, release

    @staticmethod
    def _lerp_transform(a: wp.transform, b: wp.transform, alpha: float):
        position_a = np.asarray(wp.transform_get_translation(a), dtype=np.float32)
        position_b = np.asarray(wp.transform_get_translation(b), dtype=np.float32)
        rotation_a = np.asarray(wp.transform_get_rotation(a), dtype=np.float32)
        rotation_b = np.asarray(wp.transform_get_rotation(b), dtype=np.float32)
        if np.dot(rotation_a, rotation_b) < 0.0:
            rotation_b = -rotation_b
        rotation = rotation_a * (1.0 - alpha) + rotation_b * alpha
        rotation /= max(np.linalg.norm(rotation), 1.0e-8)
        return wp.transform(wp.vec3(*(position_a * (1.0 - alpha) + position_b * alpha)), wp.quat(*rotation))

    def _apply_release_friction(self):
        """Remove hand friction after opening so the cube releases physically."""

        if self.release_friction_applied:
            return
        friction = self.model.shape_material_mu.numpy()
        friction[: self.hand_shape_end] = 0.0
        self.model.shape_material_mu.assign(friction)
        self.model.soft_contact_ke = full_reference.SOFT_CONTACT_KE
        self.model.soft_contact_kd = full_reference.SOFT_CONTACT_KD
        self.model.soft_contact_mu = full_reference.SOFT_CONTACT_MU
        self.release_friction_applied = True

    def _set_hand_soft_contact(self, enabled: bool):
        """Match the reference contact material before and during the pinch."""

        if enabled == self.hand_soft_contact_enabled:
            return
        flags = self.model.shape_flags.numpy()
        particle_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        if enabled:
            flags[: self.hand_shape_end] |= particle_flag
            self.model.soft_contact_ke = full_reference.GRASP_CONTACT_KE
            self.model.soft_contact_kd = full_reference.GRASP_CONTACT_KD
            self.model.soft_contact_mu = full_reference.GRASP_SOFT_CONTACT_MU
        else:
            flags[: self.hand_shape_end] &= ~particle_flag
            self.model.soft_contact_ke = full_reference.SOFT_CONTACT_KE
            self.model.soft_contact_kd = full_reference.SOFT_CONTACT_KD
            self.model.soft_contact_mu = full_reference.SOFT_CONTACT_MU
        self.model.shape_flags.assign(flags)
        self.hand_soft_contact_enabled = enabled

    def step(self):
        """Advance the autonomous recorded-pose trajectory by one physical frame."""

        root, joints, release = self._sample(self.sim_time)
        self._set_hand_target(root, joints)
        if self.sim_time >= 0.50:
            self._set_hand_soft_contact(True)
        if release:
            self._apply_release_friction()
        self.step_once()

    def render(self):
        """Render the physical hand, cube, and bag without the tuning gizmo."""

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify finite hand, soft-cube, and bag particle states."""

        assert np.all(np.isfinite(self.state_0.joint_q.numpy()))
        assert np.all(np.isfinite(self.state_0.particle_q.numpy()))
        if self.frame_index >= 30:
            bag_q = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
            bag_height = float(bag_q[:, 2].max() - bag_q[:, 2].min())
            assert bag_height < 0.40, f"Bag stretched before release: height={bag_height:.3f} m"
        if self.sim_time >= self.script_duration:
            cube_q = self.state_0.particle_q.numpy()[self.cube_particle_start : self.cube_particle_end]
            cube_centre = cube_q.mean(axis=0)
            inside = (
                abs(float(cube_centre[0]) - float(recorder.BAG_POS[0])) < 0.5 * recorder.BAG_WIDTH
                and abs(float(cube_centre[1]) - float(recorder.BAG_POS[1])) < 0.5 * recorder.BAG_DEPTH
                and float(recorder.BAG_POS[2]) - 0.02
                < float(cube_centre[2])
                < float(recorder.BAG_POS[2]) + recorder.BAG_HEIGHT
            )
            assert inside, f"Soft cube missed the bag: centre={tuple(float(value) for value in cube_centre)}"

    @staticmethod
    def create_parser():
        """Create parser options for the right-hand recorded grasp demo."""

        parser = recorder.Example.create_parser()
        parser.set_defaults(num_frames=800, paused=False)
        parser.add_argument(
            "--grasp-keyframe",
            default=str(DEFAULT_GRASP_KEYFRAME),
            help="Latest grasp keyframe JSON generated by the right-hand recorder.",
        )
        return parser


def main():
    """Run the right-hand-only physical soft-cube bag-placement example."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

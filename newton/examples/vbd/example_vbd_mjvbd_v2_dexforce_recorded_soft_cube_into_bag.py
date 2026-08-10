# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the recorded right-hand soft-cube trajectory on the full W1.

The full robot starts at the right-hand example's recorded target point. From
that initial pose onward, it uses the same fixed wrist/root pose, five-finger
closure, lift, bag transport, and release timing as the isolated-hand example.
The soft cube is transported and released solely through MJVBD-v2 rigid-soft
contact; particle positions are never attached to the hand.

Run from the repository root::

    uv run --extra examples newton/examples/vbd/example_vbd_mjvbd_v2_dexforce_recorded_soft_cube_into_bag.py --viewer gl
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd import example_vbd_mjvbd_v2_dexforce_grasp_rigid_into_bag_soft0 as soft0

# Match the isolated-hand demo's soft cube exactly.
soft0.SOFT_CUBE_DENSITY = 100.0
soft0.SOFT_CUBE_DIMS = (6, 4, 6)

DEFAULT_GRASP_KEYFRAME = Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2" / "vbd_w1_right_hand_last_keyframe.json"
INITIAL_IK_ITERATIONS = 240

# Fixed mount from the full W1 URDF:
# right_j7 -> right_ee (x=-0.066, Ry=-90 deg) -> right_hand_base (Rx=90 deg).
RIGHT_J7_TO_HAND_BASE_OFFSET = wp.vec3(-0.066, 0.0, 0.0)
RIGHT_J7_TO_HAND_BASE_ROTATION = wp.quat(0.5, -0.5, 0.5, 0.5)

# Frame recorded at the target pick point before the final five-finger closure.
APPROACH_ROOT_POSITION = (-0.16214203834533691, -2.838686943054199, 1.3409454822540283)
APPROACH_ROOT_ROTATION = (0.09465623646974564, 0.9546480774879456, -0.2820824682712555, 0.010803722776472569)
APPROACH_JOINTS_DEGREES = {
    "HAND_THUMB2": 90.0,
    "HAND_THUMB1": 6.0,
    "HAND_INDEX": 41.0,
    "INDEX_PIP": 24.0,
    "HAND_MIDDLE": 57.0,
    "MIDDLE_PIP": 0.0,
    "HAND_RING": 48.0,
    "RING_PIP": 15.0,
    "HAND_PINKY": 24.0,
    "PINKY_PIP": 26.0,
}
START_HOLD_DURATION = 0.50
START_TO_APPROACH_DURATION = 1.50
APPROACH_HOLD_DURATION = 0.50


class Example(soft0.Example):
    """Use recorded approach and grasp poses in the physical bag-placement demo."""

    def __init__(self, viewer, args):
        self.approach_root_world = wp.transform(wp.vec3(*APPROACH_ROOT_POSITION), wp.quat(*APPROACH_ROOT_ROTATION))
        self.approach_hand_q = dict(APPROACH_JOINTS_DEGREES)
        self.grasp_root_world, self.grasp_hand_q = self._load_grasp_keyframe(args.recorded_grasp_keyframe)
        super().__init__(viewer, args)

    @staticmethod
    def _load_grasp_keyframe(path_value: str):
        """Load the current physical grasp pose written by the hand recorder."""

        path = Path(path_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Recorded grasp keyframe not found: {path}. Run the right-hand recorder and click Record keyframe."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        keyframe = payload["keyframe"]
        root = keyframe["target_root_pose"]
        position = root["position_m"]
        rotation = root["quaternion_xyzw"]
        joints = keyframe["target_finger_joints_degrees"]
        if len(position) != 3 or len(rotation) != 4:
            raise ValueError(f"Invalid root pose in recorded grasp keyframe: {path}")
        grasp_joints = {name.removeprefix("RIGHT_"): float(value) for name, value in joints.items()}
        return wp.transform(wp.vec3(*position), wp.quat(*rotation)), grasp_joints

    def _load_recorded_hand_pose(self, path_value):
        """Disable the legacy one-pose loader from the base example."""

        return None

    def _root_to_tcp(self, root_transform):
        """Convert an isolated ``right_hand_base`` pose to the full-W1 IK target."""

        hand_position = wp.transform_get_translation(root_transform)
        hand_rotation = wp.transform_get_rotation(root_transform)
        wrist_rotation = self._quat_mul(
            hand_rotation,
            wp.quat_inverse(RIGHT_J7_TO_HAND_BASE_ROTATION),
        )
        target_offset = soft0.TCP_OFFSET - RIGHT_J7_TO_HAND_BASE_OFFSET
        target_position = hand_position + wp.quat_rotate(wrist_rotation, target_offset)
        return wp.transform(target_position, wrist_rotation)

    def _right_hand_q(self):
        """Return the recorded target-point and closure finger configurations."""

        q_start = self.model.joint_q_start.numpy()
        indices = []
        start_q = []
        approach_q = []
        grasp_q = []
        for suffix in self.HAND_SUFFIXES:
            joint = self._joint_index(f"RIGHT_{suffix}")
            index = int(q_start[joint])
            indices.append(index)
            start_q.append(np.radians(90.0 if suffix == "HAND_THUMB2" else 0.0))
            approach_q.append(np.radians(self.approach_hand_q[suffix]))
            grasp_q.append(np.radians(self.grasp_hand_q[suffix]))
        self.hand_start = wp.array(start_q, dtype=wp.float32, device=self.device)
        return (
            wp.array(indices, dtype=wp.int32, device=self.device),
            wp.array(approach_q, dtype=wp.float32, device=self.device),
            wp.array(grasp_q, dtype=wp.float32, device=self.device),
        )

    def _set_hand_particle_collision(self, enabled: bool):
        """Match the isolated hand's URDF-only soft-contact geometry."""

        if enabled == self.hand_particle_collision_enabled:
            return
        flags = self.model.shape_flags.numpy()
        particle_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        pad_shapes = set(getattr(self, "finger_pad_shapes", ()))
        urdf_hand_shapes = [shape for shape in self.right_hand_shapes if shape not in pad_shapes]
        if enabled:
            flags[urdf_hand_shapes] |= particle_flag
            self.model.soft_contact_ke = soft0.GRASP_CONTACT_KE
            self.model.soft_contact_kd = soft0.GRASP_CONTACT_KD
            self.model.soft_contact_mu = soft0.GRASP_SOFT_CONTACT_MU
        else:
            flags[urdf_hand_shapes] &= ~particle_flag
            self.model.soft_contact_ke = soft0.SOFT_CONTACT_KE
            self.model.soft_contact_kd = soft0.SOFT_CONTACT_KD
            self.model.soft_contact_mu = soft0.SOFT_CONTACT_MU
        if pad_shapes:
            flags[list(pad_shapes)] &= ~particle_flag
        self.model.shape_flags.assign(flags)
        self.hand_particle_collision_enabled = enabled

    def _build_joint_target_cache(self):
        """Initialize W1 at the recorded hand pose before caching the script."""

        approach = self._root_to_tcp(self.approach_root_world)
        self.left_obj.set_target_position(0, wp.transform_get_translation(self.left_home))
        self.left_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(self.left_home)))
        self.right_obj.set_target_position(0, wp.transform_get_translation(approach))
        self.right_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(approach)))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=INITIAL_IK_ITERATIONS)
        wp.launch(
            soft0._lock_q,
            self.lock_indices.shape[0],
            [self.ik_q, self.lock_indices, self.lock_values],
            device=self.model.device,
        )

        initial_q = self.model.joint_q.numpy()
        initial_q[: self.ik_model.joint_coord_count] = self.ik_q.numpy()[0]
        initial_q[self.hand_indices.numpy()] = self.hand_start.numpy()
        self.model.joint_q.assign(initial_q)
        self.state_0.joint_q.assign(initial_q)
        self.state_1.joint_q.assign(initial_q)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
        super()._build_joint_target_cache()

        cache = self.cached_joint_targets.numpy()
        hand_indices = self.hand_indices.numpy()
        start_q = self.hand_start.numpy()
        approach_q = self.hand_open.numpy()
        transition_end = START_HOLD_DURATION + START_TO_APPROACH_DURATION
        for frame in range(self.cached_frame_count + 1):
            script_time = frame * self.frame_dt * self.args.trajectory_time_scale
            if script_time <= START_HOLD_DURATION:
                cache[frame, hand_indices] = start_q
            elif script_time <= transition_end:
                alpha = (script_time - START_HOLD_DURATION) / START_TO_APPROACH_DURATION
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                cache[frame, hand_indices] = start_q * (1.0 - alpha) + approach_q * alpha
            else:
                break
        self.cached_joint_targets.assign(cache)

    def _segments(self):
        """Build the isolated-hand trajectory from its recorded initial pose."""

        approach = self._root_to_tcp(self.approach_root_world)
        approach_root_position = wp.transform_get_translation(self.approach_root_world)
        cube_position = self._world_vec(soft0.CUBE_POSITIONS[0])
        root_cube_offset = approach_root_position - cube_position
        bag_position = self._world_vec(soft0.BAG_POS)
        bag_hover_root = wp.transform(
            wp.vec3(
                float(bag_position[0]) + float(root_cube_offset[0]),
                float(bag_position[1]) + float(root_cube_offset[1]),
                float(bag_position[2]) + soft0.BAG_HEIGHT + 0.06 + float(root_cube_offset[2]),
            ),
            wp.transform_get_rotation(self.approach_root_world),
        )
        bag_hover = self._root_to_tcp(bag_hover_root)
        lift = wp.transform(
            wp.transform_get_translation(approach) + wp.vec3(0.0, 0.0, 0.07),
            wp.transform_get_rotation(approach),
        )
        retreat = wp.transform(
            wp.transform_get_translation(bag_hover) + wp.vec3(0.0, 0.0, 0.10),
            wp.transform_get_rotation(bag_hover),
        )

        self.hand_collision_enable_time = 0.50
        return (
            (START_HOLD_DURATION, self.left_home, self.left_home, approach, approach, 0.0, 0.0, 0),
            (START_TO_APPROACH_DURATION, self.left_home, self.left_home, approach, approach, 0.0, 0.0, 0),
            (APPROACH_HOLD_DURATION, self.left_home, self.left_home, approach, approach, 0.0, 0.0, 0),
            (1.80, self.left_home, self.left_home, approach, approach, 0.0, 1.0, 0),
            (0.60, self.left_home, self.left_home, approach, approach, 1.0, 1.0, 0),
            (1.20, self.left_home, self.left_home, approach, lift, 1.0, 1.0, 0),
            (7.00, self.left_home, self.left_home, lift, bag_hover, 1.0, 1.0, 0),
            (0.40, self.left_home, self.left_home, bag_hover, bag_hover, 1.0, 1.0, 0),
            (soft0.SOFT_CUBE_RELEASE_OPEN_DURATION, self.left_home, self.left_home, bag_hover, bag_hover, 1.0, 0.0, 0),
            (
                soft0.SOFT_CUBE_RELEASE_SETTLE_DURATION,
                self.left_home,
                self.left_home,
                bag_hover,
                bag_hover,
                0.0,
                0.0,
                0,
            ),
            (1.00, self.left_home, self.left_home, bag_hover, retreat, 0.0, 0.0, -1),
        )

    @staticmethod
    def create_parser():
        """Create parser options for the recorded two-pose grasp demo."""

        parser = soft0.Example.create_parser()
        parser.add_argument(
            "--recorded-grasp-keyframe",
            default=str(DEFAULT_GRASP_KEYFRAME),
            help="Latest keyframe JSON from example_vbd_mjvbd_v2_right_hand_soft_cube_recorder.py.",
        )
        return parser


def main():
    """Run the recorded two-pose soft-cube bag-placement demo."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Grasp, lift, and release an inflatable bag with a recorded W1 hand pose.

The recorded ``position_offset_mm[2]`` is ignored because it represents a lift
performed after the grasp. The hand first approaches the corresponding
pre-lift root pose, closes at the recorder's contact-aware speed, lifts by the
scripted distance, holds, and opens to release the bag physically.

CUDA devices capture the warmed physics frame by default. Pass
``--no-graph-capture`` to use direct kernel launches instead.

Run from the repository root::

    uv run --extra examples -m newton.examples \
        vbd_mjvbd_v2_right_hand_recorded_inflatable_bag_pick_release --viewer gl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mjvbdv2 import example_vbd_mjvbd_v2_right_hand_inflatable_bag_recorder as recorder

DEFAULT_GRASP_KEYFRAME = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "vbd_mjvbd_v2"
    / "vbd_w1_right_hand_inflatable_bag_last_keyframe.json"
)

HAND_JOINTS = tuple(recorder.INITIAL_HAND_JOINTS)
OPEN_JOINTS = dict(recorder.INITIAL_HAND_JOINTS)

HAND_TRANSLATION_SPEED = 0.040
HAND_ANGULAR_SPEED_DEG_S = 90.0
INITIAL_POSE_HOLD_DURATION = 0.1
GRASP_SETTLE_DURATION = 0.75
SCRIPTED_LIFT_HEIGHT = 0.120
LIFTED_HOLD_DURATION = 0.1
DROP_SETTLE_DURATION = 2.0
FINGER_PHASE_PADDING = 0.25
MINIMUM_ROOT_PHASE_DURATION = 0.25
DEMO_MAX_ABSOLUTE_PRESSURE = 500_000.0
RELEASE_FRICTION = 1.0

_SOFT_MATERIAL_GRASP = 0
_SOFT_MATERIAL_RELEASE = 1


@dataclass(frozen=True)
class _RecordedGrasp:
    """Store a validated pre-lift hand target and scripted lift height."""

    root: wp.transform
    joints_degrees: dict[str, float]
    lift_height: float


@dataclass(frozen=True)
class _Phase:
    """Describe one autonomous hand-motion phase."""

    name: str
    duration: float
    root_start: wp.transform
    root_end: wp.transform
    finger_target_degrees: dict[str, float]
    release: bool = False


class Example(recorder.Example):
    """Replay a recorded physical grasp, lift, and release sequence."""

    def __init__(self, viewer, args):
        self.recorded_grasp = self._load_recorded_grasp(args.grasp_keyframe)
        super().__init__(viewer, args)

        self._validate_initial_pose()
        self._raise_pressure_limit()
        self.graph = None
        self.use_graph = bool(args.graph_capture) and self.device.is_cuda
        self.script_target_q = self.manual_target_q.numpy()
        self.soft_contact_materials = None
        self.soft_contact_material_index = None
        if self.use_graph:
            self.soft_contact_materials = wp.array(
                np.asarray(
                    (
                        (recorder.CONTACT_KE, recorder.CONTACT_KD, recorder.CONTACT_MU),
                        (recorder.CONTACT_KE, recorder.CONTACT_KD, RELEASE_FRICTION),
                    ),
                    dtype=np.float32,
                ),
                dtype=wp.vec3,
                device=self.device,
            )
            self.soft_contact_material_index = wp.full(
                1,
                _SOFT_MATERIAL_GRASP,
                dtype=wp.int32,
                device=self.device,
            )
            self.solver.vbd_solver.set_soft_contact_material_source(
                self.soft_contact_materials,
                self.soft_contact_material_index,
            )
        self.release_material_applied = False
        self.initial_bag_center_z = self._bag_center_z()
        self.lifted_bag_center_z: float | None = None
        self.minimum_volume_ratio = 1.0
        self.maximum_pressure = recorder.BAG_REFERENCE_ABSOLUTE_PRESSURE
        self.phases = self._build_phases()
        self.script_duration = sum(phase.duration for phase in self.phases)
        self.active_phase_index = -1
        self.active_phase_name = "initialization"

    @staticmethod
    def _validated_vector(value, length: int, label: str, path: Path) -> np.ndarray:
        """Return a finite vector with the requested length."""

        if not isinstance(value, list) or len(value) != length:
            raise ValueError(f"Invalid {label} in recorded grasp: {path}")
        vector = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"Non-finite {label} in recorded grasp: {path}")
        return vector

    @classmethod
    def _load_recorded_grasp(cls, path_value: str) -> _RecordedGrasp:
        """Load the grasp while ignoring its recorded post-grasp Z offset."""

        path = Path(path_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Inflatable-bag grasp keyframe not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        keyframe = payload.get("keyframe")
        if not isinstance(keyframe, dict):
            raise ValueError(f"Missing keyframe object in recorded grasp: {path}")

        gizmo = keyframe.get("gizmo_world")
        if not isinstance(gizmo, dict):
            raise ValueError(f"Missing gizmo_world in recorded grasp: {path}")
        gizmo_position = cls._validated_vector(gizmo.get("position_m"), 3, "gizmo position", path)
        gizmo_rotation = cls._validated_vector(gizmo.get("quaternion_xyzw"), 4, "gizmo rotation", path)
        rotation_norm = float(np.linalg.norm(gizmo_rotation))
        if rotation_norm < 1.0e-8:
            raise ValueError(f"Zero-length gizmo quaternion in recorded grasp: {path}")
        gizmo_rotation /= rotation_norm

        position_offset = cls._validated_vector(
            keyframe.get("position_offset_mm"),
            3,
            "position offset",
            path,
        )
        rotation_offset = cls._validated_vector(
            keyframe.get("rotation_offset_deg"),
            3,
            "rotation offset",
            path,
        )
        grasp_position = gizmo_position + position_offset * np.asarray((1.0e-3, 1.0e-3, 0.0))
        rx, ry, rz = np.radians(rotation_offset)
        offset_rotation = recorder.Example._quat_mul(
            recorder.Example._quat_mul(
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(rx)),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(ry)),
            ),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rz)),
        )
        grasp_rotation = recorder.Example._quat_mul(offset_rotation, wp.quat(*gizmo_rotation))

        recorded_target = keyframe.get("target_root_pose")
        if not isinstance(recorded_target, dict):
            raise ValueError(f"Missing target_root_pose in recorded grasp: {path}")
        recorded_position = cls._validated_vector(
            recorded_target.get("position_m"),
            3,
            "target root position",
            path,
        )
        recorded_rotation = cls._validated_vector(
            recorded_target.get("quaternion_xyzw"),
            4,
            "target root rotation",
            path,
        )
        recorded_rotation /= max(float(np.linalg.norm(recorded_rotation)), 1.0e-8)
        if not np.allclose(recorded_position[:2], grasp_position[:2], atol=2.0e-5):
            raise ValueError(f"Recorded target root XY is inconsistent with gizmo_world and position offsets: {path}")
        if abs(float(np.dot(recorded_rotation, grasp_rotation))) < 1.0 - 1.0e-5:
            raise ValueError(f"Recorded target rotation is inconsistent with rotation offsets: {path}")

        joints = keyframe.get("finger_joints_degrees")
        if not isinstance(joints, dict):
            joints = keyframe.get("target_finger_joints_degrees")
        if not isinstance(joints, dict):
            raise ValueError(f"Missing finger joints in recorded grasp: {path}")
        missing_joints = set(HAND_JOINTS) - joints.keys()
        if missing_joints:
            raise ValueError(f"Missing hand joints in recorded grasp {path}: {sorted(missing_joints)}")
        joint_targets = {name: float(joints[name]) for name in HAND_JOINTS}
        if not np.all(np.isfinite(tuple(joint_targets.values()))):
            raise ValueError(f"Non-finite finger joints in recorded grasp: {path}")

        return _RecordedGrasp(
            root=wp.transform(wp.vec3(*grasp_position), grasp_rotation),
            joints_degrees=joint_targets,
            lift_height=SCRIPTED_LIFT_HEIGHT,
        )

    def _validate_initial_pose(self):
        """Verify initial and recorded targets are finite and within limits."""

        initial_q = self.state_0.joint_q.numpy()
        expected_position = np.asarray(wp.transform_get_translation(recorder.INITIAL_HAND_ROOT), dtype=np.float64)
        expected_rotation = np.asarray(wp.transform_get_rotation(recorder.INITIAL_HAND_ROOT), dtype=np.float64)
        actual_root = initial_q[self.root_q_start : self.root_q_start + 7]
        if not np.allclose(actual_root[:3], expected_position, atol=1.0e-6):
            raise ValueError("Initial hand root position does not match the recorder pose.")
        if abs(float(np.dot(actual_root[3:], expected_rotation))) < 1.0 - 1.0e-5:
            raise ValueError("Initial hand root rotation does not match the recorder pose.")

        for name, q_index in self.hand_joint_indices.items():
            lower, upper = self.joint_limits[name]
            initial_degrees = float(np.degrees(initial_q[q_index]))
            expected_degrees = float(OPEN_JOINTS[name])
            if abs(initial_degrees - expected_degrees) > 1.0e-4:
                raise ValueError(f"Initial joint {name} is {initial_degrees:.6f}°, expected {expected_degrees:.6f}°.")
            grasp_degrees = self.recorded_grasp.joints_degrees[name]
            if not lower - 1.0e-4 <= grasp_degrees <= upper + 1.0e-4:
                raise ValueError(f"Recorded joint {name}={grasp_degrees:.6f}° is outside [{lower:.6f}, {upper:.6f}]°.")

        cavity_index = self.cavity.cavity_index
        initial_volume = float(self.state_0.pneumatic.volume.numpy()[cavity_index])
        if abs(initial_volume / self.cavity.rest_volume - 1.0) > 1.0e-4:
            raise ValueError("Initial inflatable-bag volume does not match its authored rest volume.")

    def _raise_pressure_limit(self):
        """Avoid losing target-volume response when the grasp pressure rises."""

        pressure_limit = self.model.pneumatic.max_absolute_pressure.numpy()
        pressure_limit[self.cavity.cavity_index] = DEMO_MAX_ABSOLUTE_PRESSURE
        self.model.pneumatic.max_absolute_pressure.assign(pressure_limit)

    @staticmethod
    def _copy_transform(transform: wp.transform) -> wp.transform:
        """Copy one Warp transform."""

        return wp.transform(
            wp.vec3(*wp.transform_get_translation(transform)),
            wp.quat(*wp.transform_get_rotation(transform)),
        )

    @staticmethod
    def _transform_duration(start: wp.transform, end: wp.transform) -> float:
        """Compute a duration bounded by linear and angular hand speeds."""

        start_position = np.asarray(wp.transform_get_translation(start), dtype=np.float64)
        end_position = np.asarray(wp.transform_get_translation(end), dtype=np.float64)
        distance = float(np.linalg.norm(end_position - start_position))

        start_rotation = np.asarray(wp.transform_get_rotation(start), dtype=np.float64)
        end_rotation = np.asarray(wp.transform_get_rotation(end), dtype=np.float64)
        cosine = float(np.clip(abs(np.dot(start_rotation, end_rotation)), 0.0, 1.0))
        angle = 2.0 * float(np.arccos(cosine))
        angular_speed = float(np.radians(HAND_ANGULAR_SPEED_DEG_S))
        return max(
            MINIMUM_ROOT_PHASE_DURATION,
            distance / HAND_TRANSLATION_SPEED,
            angle / angular_speed,
        )

    def _build_phases(self) -> tuple[_Phase, ...]:
        """Build validation, grasp, lift, hold, and physical-release phases."""

        initial_root = self._copy_transform(recorder.INITIAL_HAND_ROOT)
        grasp_root = self._copy_transform(self.recorded_grasp.root)
        grasp_position = wp.transform_get_translation(grasp_root)
        grasp_rotation = wp.transform_get_rotation(grasp_root)
        lifted_root = wp.transform(
            grasp_position + wp.vec3(0.0, 0.0, self.recorded_grasp.lift_height),
            grasp_rotation,
        )
        maximum_finger_delta = max(
            abs(self.recorded_grasp.joints_degrees[name] - OPEN_JOINTS[name]) for name in HAND_JOINTS
        )
        finger_duration = maximum_finger_delta / recorder.MAX_FINGER_CONTACT_SPEED_DEG_S + FINGER_PHASE_PADDING
        return (
            _Phase("validate_initial", INITIAL_POSE_HOLD_DURATION, initial_root, initial_root, OPEN_JOINTS),
            _Phase(
                "approach",
                self._transform_duration(initial_root, grasp_root),
                initial_root,
                grasp_root,
                OPEN_JOINTS,
            ),
            _Phase("close", finger_duration, grasp_root, grasp_root, self.recorded_grasp.joints_degrees),
            _Phase(
                "grasp_settle",
                GRASP_SETTLE_DURATION,
                grasp_root,
                grasp_root,
                self.recorded_grasp.joints_degrees,
            ),
            _Phase(
                "lift",
                self._transform_duration(grasp_root, lifted_root),
                grasp_root,
                lifted_root,
                self.recorded_grasp.joints_degrees,
            ),
            _Phase(
                "lifted_hold",
                LIFTED_HOLD_DURATION,
                lifted_root,
                lifted_root,
                self.recorded_grasp.joints_degrees,
            ),
            _Phase("release", finger_duration, lifted_root, lifted_root, OPEN_JOINTS, release=True),
            _Phase("drop_settle", DROP_SETTLE_DURATION, lifted_root, lifted_root, OPEN_JOINTS, release=True),
        )

    @staticmethod
    def _interpolate_transform(start: wp.transform, end: wp.transform, alpha: float) -> wp.transform:
        """Interpolate translation and take the shortest normalized quaternion path."""

        start_position = np.asarray(wp.transform_get_translation(start), dtype=np.float64)
        end_position = np.asarray(wp.transform_get_translation(end), dtype=np.float64)
        start_rotation = np.asarray(wp.transform_get_rotation(start), dtype=np.float64)
        end_rotation = np.asarray(wp.transform_get_rotation(end), dtype=np.float64)
        if np.dot(start_rotation, end_rotation) < 0.0:
            end_rotation = -end_rotation
        rotation = start_rotation * (1.0 - alpha) + end_rotation * alpha
        rotation /= max(float(np.linalg.norm(rotation)), 1.0e-8)
        position = start_position * (1.0 - alpha) + end_position * alpha
        return wp.transform(wp.vec3(*position), wp.quat(*rotation))

    def _sample(self, time_s: float) -> tuple[wp.transform, dict[str, float], int]:
        """Sample the current scripted root and rate-limited finger target."""

        for phase_index, phase in enumerate(self.phases):
            if time_s <= phase.duration:
                alpha = float(np.clip(time_s / phase.duration, 0.0, 1.0))
                root = self._interpolate_transform(phase.root_start, phase.root_end, alpha)
                return root, phase.finger_target_degrees, phase_index
            time_s -= phase.duration
        final_index = len(self.phases) - 1
        final_phase = self.phases[final_index]
        return self._copy_transform(final_phase.root_end), final_phase.finger_target_degrees, final_index

    def _set_hand_target(self, root: wp.transform, joints_degrees: dict[str, float]):
        """Set the next root and finger targets without teleporting bag particles."""

        target_q = self.script_target_q
        position = wp.transform_get_translation(root)
        rotation = wp.transform_get_rotation(root)
        target_q[self.root_q_start : self.root_q_start + 7] = [*position, *rotation]
        for name, q_index in self.hand_joint_indices.items():
            target_q[q_index] = np.radians(joints_degrees[name])
        self.manual_target_q.assign(target_q)

    def _bag_center_z(self) -> float:
        """Return the current inflatable-bag center height [m]."""

        positions = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        return float(np.mean(positions[:, 2]))

    def _apply_release_material(self):
        """Remove hand friction once finger opening begins."""

        if self.release_material_applied:
            return
        friction = self.model.shape_material_mu.numpy()
        friction[: self.hand_shape_end] = RELEASE_FRICTION
        self.model.shape_material_mu.assign(friction)
        self.model.soft_contact_mu = RELEASE_FRICTION
        if self.soft_contact_material_index is not None:
            self.soft_contact_material_index.fill_(_SOFT_MATERIAL_RELEASE)
        self.release_material_applied = True

    def _enter_phase(self, phase_index: int):
        """Record lift diagnostics and apply one-time phase changes."""

        if phase_index == self.active_phase_index:
            return
        self.active_phase_index = phase_index
        phase = self.phases[phase_index]
        self.active_phase_name = phase.name
        if phase.name == "release":
            self.lifted_bag_center_z = self._bag_center_z()
        if phase.release:
            self._apply_release_material()

    def _capture_simulation_graph(self):
        """Capture the warmed physics frame as one reusable CUDA graph."""

        state_0_backup = self.model.state()
        state_1_backup = self.model.state()
        state_0_backup.assign(self.state_0)
        state_1_backup.assign(self.state_1)

        with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
            self._advance_physics_frame()

        self.state_0.assign(state_0_backup)
        self.state_1.assign(state_1_backup)
        self.graph = capture.graph
        if self.graph is None:
            raise RuntimeError(f"CUDA graph capture failed on device {self.device}")

    def step(self):
        """Advance one frame, replaying a warmed CUDA graph when available."""

        root, joints, phase_index = self._sample(self.sim_time)
        self._enter_phase(phase_index)
        self._set_hand_target(root, joints)
        if self.graph is None:
            self._advance_physics_frame()
            if self.use_graph:
                self._capture_simulation_graph()
        else:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def render(self):
        """Render the physical hand and pneumatic bag without recorder controls."""

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_mesh(
            "/inflatable_bag/surface",
            self.state_0.particle_q,
            self.bag_triangle_indices,
            backface_culling=True,
            color=(0.86, 0.68, 0.34),
        )
        wp.launch(
            recorder._gather_edges,
            dim=len(self.bag_edge_starts),
            inputs=[self.state_0.particle_q, self.bag_edges, 1.0e-4],
            outputs=[self.bag_edge_starts, self.bag_edge_ends],
            device=self.model.device,
        )
        self.viewer.log_lines(
            "/inflatable_bag/grid",
            self.bag_edge_starts,
            self.bag_edge_ends,
            (0.08, 0.06, 0.02),
        )
        self.viewer.end_frame()

    def test_post_step(self):
        """Verify every phase keeps the hand and pneumatic state finite."""

        positions = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        joint_q = self.state_0.joint_q.numpy()
        cavity_index = self.cavity.cavity_index
        volume = float(self.state_0.pneumatic.volume.numpy()[cavity_index])
        pressure = float(self.state_0.pneumatic.absolute_pressure.numpy()[cavity_index])
        volume_ratio = volume / self.cavity.rest_volume
        assert np.all(np.isfinite(positions))
        assert np.all(np.isfinite(joint_q))
        assert np.isfinite(volume_ratio) and volume_ratio > 0.70
        assert np.isfinite(pressure) and 0.0 < pressure <= DEMO_MAX_ABSOLUTE_PRESSURE + 1.0
        self.minimum_volume_ratio = min(self.minimum_volume_ratio, volume_ratio)
        self.maximum_pressure = max(self.maximum_pressure, pressure)
        if self.active_phase_name == "validate_initial":
            hand_contacts, _ = self._contact_counts()
            assert hand_contacts == 0, f"Initial hand pose unexpectedly has {hand_contacts} bag contacts."

    def test_final(self):
        """Verify the hand lifts and physically releases the inflatable bag."""

        super().test_final()
        if self.use_graph:
            assert self.graph is not None, "The warmed CUDA physics frame was not captured."
        if self.sim_time + self.frame_dt < self.script_duration:
            return

        assert self.lifted_bag_center_z is not None, "The scripted lift phase did not complete."
        assert self.lifted_bag_center_z > self.initial_bag_center_z + 0.010, (
            f"The bag was not lifted: initial z={self.initial_bag_center_z:.6f}, "
            f"lifted z={self.lifted_bag_center_z:.6f}."
        )
        final_bag_center_z = self._bag_center_z()
        assert final_bag_center_z < self.lifted_bag_center_z - 0.008, (
            f"The released bag did not fall: lifted z={self.lifted_bag_center_z:.6f}, final z={final_bag_center_z:.6f}."
        )
        assert self.minimum_volume_ratio > 0.85, (
            f"The grasp compressed the target-volume bag excessively: minimum ratio={self.minimum_volume_ratio:.6f}."
        )

        joint_q = self.state_0.joint_q.numpy()
        maximum_open_error = max(
            abs(float(np.degrees(joint_q[q_index])) - OPEN_JOINTS[name])
            for name, q_index in self.hand_joint_indices.items()
        )
        assert maximum_open_error < 2.0, f"The hand did not reopen fully: error={maximum_open_error:.3f}°."

    @staticmethod
    def create_parser():
        """Create command-line arguments for the recorded pick-and-release demo."""

        parser = recorder.Example.create_parser()
        parser.set_defaults(num_frames=720, paused=False, pneumatic_mode="target-volume")
        parser.add_argument(
            "--graph-capture",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Capture the warmed physics frame as one CUDA graph.",
        )
        parser.add_argument(
            "--grasp-keyframe",
            default=str(DEFAULT_GRASP_KEYFRAME),
            help="Inflatable-bag grasp keyframe JSON generated by the recorder.",
        )
        return parser


def main():
    """Run the recorded inflatable-bag pick-and-release demo."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

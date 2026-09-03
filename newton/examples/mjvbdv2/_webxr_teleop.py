# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example-local WebXR transport, pose retargeting, and trajectory writing.

The module deliberately stores only the newest controller and render frames.
Network delays therefore reduce update rate instead of replaying stale motion
into the realtime simulation.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import struct
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_VERSION = 1
DEFAULT_XR_TO_ROBOT = np.array(
    (
        (0.0, 0.0, -1.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ),
    dtype=np.float32,
)

_VALID_HANDS = frozenset(("left", "right"))
_VALID_REFERENCE_SPACES = frozenset(("local-floor", "bounded-floor", "local"))
_VALID_CONTROLLER_SPACES = frozenset(("webxr-reference", "newton-world"))
_VALID_VIEW_MODES = frozenset(("observer", "robot-first-person"))


class ProtocolError(ValueError):
    """Raised when a browser message does not match the WebXR protocol."""


def pack_scene_geometry(
    meshes: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    shapes: Sequence[Mapping[str, Any]],
) -> bytes:
    """Pack static meshes and instances into the Quest client's binary format."""
    mesh_headers: list[dict[str, int]] = []
    chunks: list[bytes] = []
    byte_offset = 0
    for mesh_index, (vertices_value, normals_value, indices_value) in enumerate(meshes):
        vertices = np.asarray(vertices_value, dtype=np.float32).reshape(-1, 3)
        normals = np.asarray(normals_value, dtype=np.float32).reshape(-1, 3)
        if vertices.shape != normals.shape:
            raise ValueError(f"mesh {mesh_index} vertices and normals must have the same shape")
        if not np.all(np.isfinite(vertices)) or not np.all(np.isfinite(normals)):
            raise ValueError(f"mesh {mesh_index} vertices and normals must be finite")

        indices_source = np.asarray(indices_value)
        try:
            indices_int64 = np.asarray(indices_value, dtype=np.int64).reshape(-1)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"mesh {mesh_index} indices must be integers") from exc
        if not np.array_equal(indices_source.reshape(-1), indices_int64):
            raise ValueError(f"mesh {mesh_index} indices must be integers")
        if len(indices_int64) % 3 != 0:
            raise ValueError(f"mesh {mesh_index} index count must be divisible by three")
        if len(indices_int64) and (int(indices_int64.min()) < 0 or int(indices_int64.max()) >= len(vertices)):
            raise ValueError(f"mesh {mesh_index} indices are outside the vertex range")

        interleaved = np.empty((len(vertices), 6), dtype="<f4")
        interleaved[:, :3] = vertices
        interleaved[:, 3:] = normals
        vertex_bytes = interleaved.tobytes()
        index_bytes = np.asarray(indices_int64, dtype="<u4").tobytes()
        mesh_headers.append(
            {
                "vertexByteOffset": byte_offset,
                "vertexCount": len(vertices),
                "indexByteOffset": byte_offset + len(vertex_bytes),
                "indexCount": len(indices_int64),
            }
        )
        chunks.extend((vertex_bytes, index_bytes))
        byte_offset += len(vertex_bytes) + len(index_bytes)

    header = {
        "version": 1,
        "vertexStrideFloats": 6,
        "meshes": mesh_headers,
        "shapes": [dict(shape) for shape in shapes],
    }
    header_bytes = json.dumps(header, separators=(",", ":"), allow_nan=False).encode("utf-8")
    padding = b"\0" * (-len(header_bytes) % 4)
    return struct.pack("<4sI", b"NXR1", len(header_bytes)) + header_bytes + padding + b"".join(chunks)


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{name} must contain {length} numbers") from exc
    if vector.shape != (length,):
        raise ProtocolError(f"{name} must contain {length} numbers")
    if not np.all(np.isfinite(vector)):
        raise ProtocolError(f"{name} must be finite")
    return vector


def _normalized_quaternion(value: Any, name: str) -> np.ndarray:
    quaternion = _finite_vector(value, 4, name)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-7:
        raise ProtocolError(f"{name} must be non-zero")
    return quaternion / norm


def _rotation_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(component) for component in quaternion)
    return np.array(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )


def _quaternion_from_rotation(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ),
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            quaternion = np.array(
                (
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ),
                dtype=np.float64,
            )
        elif axis == 1:
            scale = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            quaternion = np.array(
                (
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ),
                dtype=np.float64,
            )
        else:
            scale = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            quaternion = np.array(
                (
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ),
                dtype=np.float64,
            )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("rotation produced a zero quaternion")
    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion *= -1.0
    return quaternion.astype(np.float32)


def _nearest_rotation(value: np.ndarray) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(rotation)):
        raise ValueError("XR-to-robot rotation must be finite")
    left, _, right = np.linalg.svd(rotation)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation.astype(np.float32)


@dataclass(frozen=True)
class Pose:
    """A position in metres and an ``xyzw`` quaternion."""

    position: np.ndarray
    orientation: np.ndarray

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], name: str) -> Pose:
        if not isinstance(value, Mapping):
            raise ProtocolError(f"{name} must be an object")
        return cls(
            position=_finite_vector(value.get("position"), 3, f"{name}.position"),
            orientation=_normalized_quaternion(value.get("orientation"), f"{name}.orientation"),
        )


@dataclass(frozen=True)
class ControllerState:
    """One tracked WebXR controller sample."""

    handedness: str
    pose: Pose
    clutch: bool
    selecting: bool
    button_values: tuple[float, ...]
    button_pressed: tuple[bool, ...]
    axes: tuple[float, ...]
    thumbstick: tuple[float, float]
    trigger_value: float

    @classmethod
    def from_mapping(cls, handedness: str, value: Mapping[str, Any]) -> ControllerState:
        if handedness not in _VALID_HANDS:
            raise ProtocolError(f"unsupported handedness: {handedness!r}")
        if not isinstance(value, Mapping):
            raise ProtocolError(f"controllers.{handedness} must be an object")

        buttons_value = value.get("buttons", ())
        if not isinstance(buttons_value, list):
            raise ProtocolError(f"controllers.{handedness}.buttons must be an array")
        button_values: list[float] = []
        button_pressed: list[bool] = []
        for index, button in enumerate(buttons_value):
            if not isinstance(button, Mapping):
                raise ProtocolError(f"controllers.{handedness}.buttons[{index}] must be an object")
            try:
                button_value = float(button.get("value", 0.0))
            except (TypeError, ValueError) as exc:
                raise ProtocolError(f"controllers.{handedness}.buttons[{index}].value is invalid") from exc
            if not math.isfinite(button_value):
                raise ProtocolError(f"controllers.{handedness}.buttons[{index}].value must be finite")
            button_values.append(float(np.clip(button_value, 0.0, 1.0)))
            button_pressed.append(bool(button.get("pressed", False)))

        axes_value = value.get("axes", ())
        if not isinstance(axes_value, list):
            raise ProtocolError(f"controllers.{handedness}.axes must be an array")
        axes_array = _finite_vector(axes_value, len(axes_value), f"controllers.{handedness}.axes")
        thumbstick_value = value.get("thumbstick")
        if thumbstick_value is None:
            thumbstick_array = axes_array[-2:] if len(axes_array) >= 2 else np.zeros(2, dtype=np.float32)
        else:
            thumbstick_array = _finite_vector(
                thumbstick_value,
                2,
                f"controllers.{handedness}.thumbstick",
            )
        thumbstick_array = np.clip(thumbstick_array, -1.0, 1.0)

        trigger_value = value.get("triggerValue", button_values[0] if button_values else 0.0)
        try:
            trigger = float(trigger_value)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"controllers.{handedness}.triggerValue is invalid") from exc
        if not math.isfinite(trigger):
            raise ProtocolError(f"controllers.{handedness}.triggerValue must be finite")

        return cls(
            handedness=handedness,
            pose=Pose.from_mapping(value.get("pose"), f"controllers.{handedness}.pose"),
            clutch=bool(value.get("clutch", False)),
            selecting=bool(value.get("selecting", False)),
            button_values=tuple(button_values),
            button_pressed=tuple(button_pressed),
            axes=tuple(float(component) for component in axes_array),
            thumbstick=(float(thumbstick_array[0]), float(thumbstick_array[1])),
            trigger_value=float(np.clip(trigger, 0.0, 1.0)),
        )

    def is_button_pressed(self, index: int) -> bool:
        """Return whether a gamepad button exists and is pressed."""
        return 0 <= index < len(self.button_pressed) and self.button_pressed[index]


@dataclass(frozen=True)
class XRFrame:
    """A validated, latest-value WebXR head and controller frame."""

    stream_id: str
    sequence: int
    client_time_ms: float
    received_monotonic: float
    reference_space: str
    controller_space: str
    visibility_state: str
    controllers: Mapping[str, ControllerState]
    view_mode: str = "observer"
    head_pose: Pose | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        received_monotonic: float | None = None,
    ) -> XRFrame:
        if not isinstance(value, Mapping):
            raise ProtocolError("WebXR frame must be an object")
        if value.get("type") != "xr-frame":
            raise ProtocolError("unsupported message type")
        try:
            version = int(value.get("version", -1))
        except (TypeError, ValueError) as exc:
            raise ProtocolError("protocol version is invalid") from exc
        if version != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported protocol version: {version}")
        try:
            stream_id = str(value["streamId"])
            sequence = int(value["sequence"])
            client_time_ms = float(value["timeMs"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError("streamId, sequence and timeMs are required") from exc
        if not stream_id or len(stream_id) > 128:
            raise ProtocolError("streamId is invalid")
        if sequence < 0 or not math.isfinite(client_time_ms):
            raise ProtocolError("sequence and timeMs must be finite and non-negative")

        reference_space = str(value.get("referenceSpace", ""))
        if reference_space not in _VALID_REFERENCE_SPACES:
            raise ProtocolError(f"unsupported reference space: {reference_space!r}")
        controller_space = str(value.get("controllerSpace", "webxr-reference"))
        if controller_space not in _VALID_CONTROLLER_SPACES:
            raise ProtocolError(f"unsupported controller space: {controller_space!r}")
        view_mode = str(value.get("viewMode", "observer"))
        if view_mode not in _VALID_VIEW_MODES:
            raise ProtocolError(f"unsupported view mode: {view_mode!r}")
        head_pose_value = value.get("headPose")
        head_pose = None if head_pose_value is None else Pose.from_mapping(head_pose_value, "headPose")
        controllers_value = value.get("controllers", {})
        if not isinstance(controllers_value, Mapping):
            raise ProtocolError("controllers must be an object")
        controllers = {
            handedness: ControllerState.from_mapping(handedness, controller)
            for handedness, controller in controllers_value.items()
            if handedness in _VALID_HANDS
        }
        return cls(
            stream_id=stream_id,
            sequence=sequence,
            client_time_ms=client_time_ms,
            received_monotonic=time.monotonic() if received_monotonic is None else float(received_monotonic),
            reference_space=reference_space,
            controller_space=controller_space,
            visibility_state=str(value.get("visibilityState", "unknown")),
            controllers=controllers,
            view_mode=view_mode,
            head_pose=head_pose,
        )

    @classmethod
    def from_json(cls, payload: str, *, received_monotonic: float | None = None) -> XRFrame:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ProtocolError(f"message is not valid JSON: {exc.msg}") from exc
        return cls.from_mapping(value, received_monotonic=received_monotonic)


@dataclass(frozen=True)
class XRStreamStatus:
    """Thread-safe stream counters exposed by the health endpoint."""

    clients: int
    messages: int
    rejected_messages: int
    reset_requests: int
    reset_pending: bool
    shutdown_requests: int
    shutdown_pending: bool
    teleoperation_mode_requests: int
    teleoperation_mode_pending: bool
    teleoperation_active: bool
    simulation_active: bool
    operation_mode: str
    last_sequence: int | None
    controller_space: str | None
    age_seconds: float | None


class LatestXRFrame:
    """Keep only the newest WebXR frame so lag cannot queue robot motion."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: XRFrame | None = None
        self._clients = 0
        self._messages = 0
        self._rejected_messages = 0
        self._reset_requests = 0
        self._consumed_reset_request = 0
        self._shutdown_requests = 0
        self._consumed_shutdown_request = 0
        self._teleoperation_mode_requests = 0
        self._consumed_teleoperation_mode_request = 0
        self._requested_teleoperation_active = True
        self._requested_simulation_active = True
        self._teleoperation_active = True
        self._simulation_active = True

    def update(self, frame: XRFrame) -> bool:
        """Store ``frame`` unless its sequence is stale within the same stream."""
        with self._lock:
            if (
                self._frame is not None
                and self._frame.stream_id == frame.stream_id
                and frame.sequence <= self._frame.sequence
            ):
                return False
            self._frame = frame
            self._messages += 1
            return True

    def reject(self) -> None:
        with self._lock:
            self._rejected_messages += 1

    def request_reset(self) -> int:
        """Request an in-place scene reset and return its monotonic identifier."""
        with self._lock:
            self._reset_requests += 1
            return self._reset_requests

    def consume_reset(self) -> int | None:
        """Consume all pending reset requests as one latest-value operation."""
        with self._lock:
            if self._consumed_reset_request == self._reset_requests:
                return None
            self._consumed_reset_request = self._reset_requests
            return self._consumed_reset_request

    def request_shutdown(self) -> int:
        """Request a cooperative main-loop shutdown and return its identifier."""
        with self._lock:
            self._shutdown_requests += 1
            return self._shutdown_requests

    def consume_shutdown(self) -> int | None:
        """Consume all pending cooperative shutdown requests once."""
        with self._lock:
            if self._consumed_shutdown_request == self._shutdown_requests:
                return None
            self._consumed_shutdown_request = self._shutdown_requests
            return self._consumed_shutdown_request

    def request_teleoperation_mode(self, active: bool, *, simulation_active: bool | None = None) -> int:
        """Request active controls and independently select simulation activity."""
        requested_teleoperation_active = bool(active)
        requested_simulation_active = (
            requested_teleoperation_active if simulation_active is None else bool(simulation_active)
        )
        if requested_teleoperation_active and not requested_simulation_active:
            raise ValueError("active teleoperation requires an active simulation")
        with self._lock:
            self._teleoperation_mode_requests += 1
            self._requested_teleoperation_active = requested_teleoperation_active
            self._requested_simulation_active = requested_simulation_active
            return self._teleoperation_mode_requests

    def request_standby(self) -> int | None:
        """Disarm a running simulation without restarting one that is parked."""
        with self._lock:
            if not self._requested_simulation_active:
                return None
            self._teleoperation_mode_requests += 1
            self._requested_teleoperation_active = False
            self._requested_simulation_active = True
            return self._teleoperation_mode_requests

    def consume_teleoperation_mode(self) -> tuple[int, bool, bool] | None:
        """Consume the newest requested teleoperation mode once."""
        with self._lock:
            if self._consumed_teleoperation_mode_request == self._teleoperation_mode_requests:
                return None
            self._consumed_teleoperation_mode_request = self._teleoperation_mode_requests
            self._teleoperation_active = self._requested_teleoperation_active
            self._simulation_active = self._requested_simulation_active
            return (
                self._consumed_teleoperation_mode_request,
                self._teleoperation_active,
                self._simulation_active,
            )

    def client_connected(self) -> None:
        with self._lock:
            self._clients += 1

    def client_disconnected(self) -> None:
        with self._lock:
            self._clients = max(0, self._clients - 1)

    def snapshot(
        self,
        *,
        max_age_seconds: float | None = None,
        now: float | None = None,
    ) -> XRFrame | None:
        with self._lock:
            frame = self._frame
        if frame is None or max_age_seconds is None:
            return frame
        current_time = time.monotonic() if now is None else float(now)
        if current_time - frame.received_monotonic > max_age_seconds:
            return None
        return frame

    def status(self, *, now: float | None = None) -> XRStreamStatus:
        current_time = time.monotonic() if now is None else float(now)
        with self._lock:
            frame = self._frame
            if self._teleoperation_active:
                operation_mode = "active"
            elif self._simulation_active:
                operation_mode = "standby"
            else:
                operation_mode = "parked"
            return XRStreamStatus(
                clients=self._clients,
                messages=self._messages,
                rejected_messages=self._rejected_messages,
                reset_requests=self._reset_requests,
                reset_pending=self._consumed_reset_request != self._reset_requests,
                shutdown_requests=self._shutdown_requests,
                shutdown_pending=self._consumed_shutdown_request != self._shutdown_requests,
                teleoperation_mode_requests=self._teleoperation_mode_requests,
                teleoperation_mode_pending=(
                    self._consumed_teleoperation_mode_request != self._teleoperation_mode_requests
                ),
                teleoperation_active=self._teleoperation_active,
                simulation_active=self._simulation_active,
                operation_mode=operation_mode,
                last_sequence=None if frame is None else frame.sequence,
                controller_space=None if frame is None else frame.controller_space,
                age_seconds=None if frame is None else max(0.0, current_time - frame.received_monotonic),
            )


@dataclass(frozen=True)
class TargetPose:
    """A retargeted robot pose in the Newton world frame."""

    position: np.ndarray
    orientation: np.ndarray


class RelativePoseRetargeter:
    """Retarget clutched controller motion relative to the engagement pose."""

    def __init__(
        self,
        *,
        xr_to_robot_rotation: np.ndarray = DEFAULT_XR_TO_ROBOT,
        translation_scale: float = 1.0,
        max_translation: float = 0.60,
    ) -> None:
        if not math.isfinite(translation_scale) or translation_scale <= 0.0:
            raise ValueError("translation_scale must be finite and positive")
        if not math.isfinite(max_translation) or max_translation <= 0.0:
            raise ValueError("max_translation must be finite and positive")
        self.xr_to_robot_rotation = _nearest_rotation(xr_to_robot_rotation)
        self.translation_scale = float(translation_scale)
        self.max_translation = float(max_translation)
        self._controller_anchor: Pose | None = None
        self._robot_anchor: TargetPose | None = None
        self._anchor_basis: np.ndarray | None = None

    @property
    def active(self) -> bool:
        """Whether a clutch anchor is currently active."""
        return self._controller_anchor is not None

    def reset(self) -> None:
        """Release the current clutch anchor."""
        self._controller_anchor = None
        self._robot_anchor = None
        self._anchor_basis = None

    def update(
        self,
        controller_pose: Pose,
        *,
        clutch: bool,
        robot_position: np.ndarray,
        robot_orientation: np.ndarray,
        source_to_robot_rotation: np.ndarray | None = None,
    ) -> TargetPose | None:
        """Return a target while clutched, otherwise clear the anchor."""
        if not clutch:
            self.reset()
            return None

        current_position = _finite_vector(robot_position, 3, "robot_position")
        current_orientation = _normalized_quaternion(robot_orientation, "robot_orientation")
        basis = (
            self.xr_to_robot_rotation
            if source_to_robot_rotation is None
            else _nearest_rotation(source_to_robot_rotation)
        )
        if self._anchor_basis is not None and not np.allclose(basis, self._anchor_basis, atol=1.0e-6):
            self.reset()
        if self._controller_anchor is None:
            self._controller_anchor = Pose(
                position=np.asarray(controller_pose.position, dtype=np.float32).copy(),
                orientation=_normalized_quaternion(controller_pose.orientation, "controller_pose.orientation"),
            )
            self._robot_anchor = TargetPose(current_position.copy(), current_orientation.copy())
            self._anchor_basis = basis.copy()

        assert self._controller_anchor is not None
        assert self._robot_anchor is not None
        assert self._anchor_basis is not None
        controller_anchor = self._controller_anchor
        robot_anchor = self._robot_anchor
        basis = self._anchor_basis

        delta_xr = np.asarray(controller_pose.position, dtype=np.float32) - controller_anchor.position
        delta_robot = basis @ delta_xr * self.translation_scale
        distance = float(np.linalg.norm(delta_robot))
        if distance > self.max_translation:
            delta_robot *= self.max_translation / distance

        controller_rotation = _rotation_from_quaternion(
            _normalized_quaternion(controller_pose.orientation, "controller_pose.orientation")
        )
        anchor_rotation = _rotation_from_quaternion(controller_anchor.orientation)
        delta_rotation_xr = controller_rotation @ anchor_rotation.T
        delta_rotation_robot = basis @ delta_rotation_xr @ basis.T
        robot_rotation = delta_rotation_robot @ _rotation_from_quaternion(robot_anchor.orientation)

        return TargetPose(
            position=(robot_anchor.position + delta_robot).astype(np.float32),
            orientation=_quaternion_from_rotation(robot_rotation),
        )


class JsonlTrajectoryRecorder:
    """Write replayable teleoperation samples incrementally as JSON Lines."""

    def __init__(self, path: Path, metadata: Mapping[str, Any], *, flush_every: int = 60) -> None:
        if flush_every < 1:
            raise ValueError("flush_every must be at least one")
        self.path = Path(path).expanduser()
        self.metadata = dict(metadata)
        self.flush_every = int(flush_every)
        self.recording = False
        self.sample_count = 0
        self._file = None

    def start(self) -> None:
        """Open the output lazily and begin accepting samples."""
        if self._file is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w", encoding="utf-8")
            header = {"type": "metadata", "format": "newton_webxr_trajectory_v1", **self.metadata}
            self._file.write(json.dumps(header, separators=(",", ":")) + "\n")
            self._file.flush()
        self.recording = True

    def pause(self) -> None:
        """Pause sample writes without closing the trajectory."""
        self.recording = False
        if self._file is not None:
            self._file.flush()

    def toggle(self) -> bool:
        """Toggle recording and return the new state."""
        if self.recording:
            self.pause()
        else:
            self.start()
        return self.recording

    def append(self, sample: Mapping[str, Any]) -> bool:
        """Append one sample when recording is active."""
        if not self.recording:
            return False
        assert self._file is not None
        self._file.write(json.dumps({"type": "frame", **sample}, separators=(",", ":")) + "\n")
        self.sample_count += 1
        if self.sample_count % self.flush_every == 0:
            self._file.flush()
        return True

    def append_event(self, event: Mapping[str, Any]) -> bool:
        """Append and flush an episode event once the trajectory file exists."""
        if self._file is None:
            return False
        self._file.write(json.dumps({"type": "event", **event}, separators=(",", ":")) + "\n")
        self._file.flush()
        return True

    def close(self) -> None:
        """Flush and close the output file."""
        self.recording = False
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


class WebXRServer:
    """Serve the Quest WebXR client and exchange latest-value state frames."""

    def __init__(
        self,
        state: LatestXRFrame | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        assets_dir: Path | None = None,
        geometry_payload: bytes | None = None,
        require_simulation_ready: bool = False,
    ) -> None:
        if not 0 < int(port) < 65536:
            raise ValueError("port must be between 1 and 65535")
        self.state = state if state is not None else LatestXRFrame()
        self.host = str(host)
        self.port = int(port)
        self.assets_dir = (
            Path(assets_dir)
            if assets_dir is not None
            else Path(__file__).resolve().parents[1] / "assets" / "webxr_teleop"
        )
        self.geometry_payload = None if geometry_payload is None else bytes(geometry_payload)
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._scene_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._simulation_ready = threading.Event()
        if not require_simulation_ready:
            self._simulation_ready.set()
        self._startup_error: BaseException | None = None
        self._sockets: set[Any] = set()
        self._scene_lock = threading.Lock()
        self._scene_payload: dict[str, Any] | None = None

    @staticmethod
    def _aiohttp():
        try:
            from aiohttp import WSMsgType, web  # noqa: PLC0415
        except ModuleNotFoundError as exc:
            raise RuntimeError("WebXR teleoperation requires `uv sync --extra examples`") from exc
        return web, WSMsgType

    @property
    def running(self) -> bool:
        """Whether the server worker is alive."""
        return self._thread is not None and self._thread.is_alive()

    @property
    def simulation_ready(self) -> bool:
        """Whether the simulation completed its startup warm-up."""
        return self._simulation_ready.is_set()

    def mark_simulation_ready(self) -> None:
        """Allow health checks to succeed after simulation warm-up."""
        self._simulation_ready.set()

    def start(self, timeout: float = 10.0) -> None:
        """Start the HTTP/WebSocket worker and wait until its port is bound."""
        if self.running:
            return
        if not self.assets_dir.is_dir():
            raise FileNotFoundError(f"WebXR client assets not found: {self.assets_dir}")
        self._ready.clear()
        self._startup_error = None
        self._thread = threading.Thread(target=self._thread_main, name="newton-webxr", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise TimeoutError("timed out while starting the WebXR server")
        if self._startup_error is not None:
            raise RuntimeError("failed to start the WebXR server") from self._startup_error

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the worker and close all browser sockets."""
        loop = self._loop
        stop_event = self._stop_event
        thread = self._thread
        if loop is not None and not loop.is_closed() and stop_event is not None:
            loop.call_soon_threadsafe(stop_event.set)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        self._thread = None
        self._loop = None
        self._stop_event = None
        self._scene_event = None

    def publish_scene(self, payload: Mapping[str, Any]) -> None:
        """Replace the pending render frame without queueing intermediate frames."""
        with self._scene_lock:
            self._scene_payload = dict(payload)
        loop = self._loop
        event = self._scene_event
        if loop is not None and not loop.is_closed() and event is not None:
            loop.call_soon_threadsafe(event.set)

    def scene_snapshot(self) -> dict[str, Any] | None:
        """Return a shallow copy of the newest headset render state."""
        with self._scene_lock:
            return None if self._scene_payload is None else self._scene_payload.copy()

    def health_snapshot(self) -> tuple[int, dict[str, Any]]:
        """Return the HTTP status and payload for the health endpoint."""
        status = self.state.status()
        simulation_ready = self.simulation_ready
        return (
            200 if simulation_ready else 503,
            {
                "ok": simulation_ready,
                "protocolVersion": PROTOCOL_VERSION,
                "simulationReady": simulation_ready,
                "clients": status.clients,
                "messages": status.messages,
                "rejectedMessages": status.rejected_messages,
                "resetRequests": status.reset_requests,
                "resetPending": status.reset_pending,
                "shutdownRequests": status.shutdown_requests,
                "shutdownPending": status.shutdown_pending,
                "teleoperationModeRequests": status.teleoperation_mode_requests,
                "teleoperationModePending": status.teleoperation_mode_pending,
                "teleoperationActive": status.teleoperation_active,
                "simulationActive": status.simulation_active,
                "operationMode": status.operation_mode,
                "lastSequence": status.last_sequence,
                "controllerSpace": status.controller_space,
                "ageSeconds": status.age_seconds,
                "sceneAvailable": self.scene_snapshot() is not None,
                "geometryAvailable": self.geometry_payload is not None,
                "geometryBytes": 0 if self.geometry_payload is None else len(self.geometry_payload),
            },
        )

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()

    async def _run(self) -> None:
        web, _ = self._aiohttp()
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        self._scene_event = asyncio.Event()

        app = web.Application(client_max_size=64 * 1024)
        app.router.add_get("/", self._index)
        app.router.add_get("/app.js", self._app_js)
        app.router.add_get("/style.css", self._style_css)
        app.router.add_get("/scene.bin", self._geometry)
        app.router.add_get("/ws", self._websocket)
        app.router.add_get("/healthz", self._health)
        app.router.add_post("/control/exit-immersive", self._exit_immersive)
        app.router.add_post("/control/standby", self._standby)
        app.router.add_post("/control/park", self._park)
        app.router.add_post("/control/resume", self._resume)
        app.router.add_post("/control/shutdown", self._shutdown)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        sender = asyncio.create_task(self._scene_sender())
        self._ready.set()
        try:
            await self._stop_event.wait()
        finally:
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender
            await runner.cleanup()

    def _file_response(self, name: str):
        web, _ = self._aiohttp()
        return web.FileResponse(self.assets_dir / name, headers={"Cache-Control": "no-store"})

    async def _index(self, _request):
        return self._file_response("index.html")

    async def _app_js(self, _request):
        return self._file_response("app.js")

    async def _style_css(self, _request):
        return self._file_response("style.css")

    async def _geometry(self, _request):
        web, _ = self._aiohttp()
        if self.geometry_payload is None:
            raise web.HTTPNotFound(text="scene geometry is unavailable")
        return web.Response(
            body=self.geometry_payload,
            content_type="application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    async def _health(self, _request):
        web, _ = self._aiohttp()
        http_status, payload = self.health_snapshot()
        return web.json_response(payload, status=http_status)

    async def _exit_immersive(self, _request):
        """Ask every connected browser to leave its immersive XR session."""
        web, _ = self._aiohttp()
        sockets = tuple(socket for socket in self._sockets if not socket.closed)
        message = json.dumps({"type": "exit-immersive", "version": PROTOCOL_VERSION}, separators=(",", ":"))
        if sockets:
            await asyncio.gather(*(socket.send_str(message) for socket in sockets), return_exceptions=True)
        return web.json_response({"ok": True, "notifiedClients": len(sockets)})

    async def _shutdown(self, _request):
        """Ask the Newton main loop to close at its next frame boundary."""
        web, _ = self._aiohttp()
        request_id = self.state.request_shutdown()
        return web.json_response({"ok": True, "shutdownRequest": request_id})

    async def _park(self, _request):
        """Stop submitting simulation work without destroying CUDA state."""
        web, _ = self._aiohttp()
        request_id = self.state.request_teleoperation_mode(False, simulation_active=False)
        return web.json_response(
            {
                "ok": True,
                "teleoperationActive": False,
                "simulationActive": False,
                "operationMode": "parked",
                "modeRequest": request_id,
            }
        )

    async def _standby(self, _request):
        """Disarm controls while maintaining CUDA simulation submissions."""
        web, _ = self._aiohttp()
        request_id = self.state.request_standby()
        simulation_active = request_id is not None
        return web.json_response(
            {
                "ok": True,
                "teleoperationActive": False,
                "simulationActive": simulation_active,
                "operationMode": "standby" if simulation_active else "parked",
                "modeRequest": request_id,
            }
        )

    async def _resume(self, _request):
        """Resume a parked simulation in the existing CUDA process."""
        web, _ = self._aiohttp()
        request_id = self.state.request_teleoperation_mode(True, simulation_active=True)
        return web.json_response(
            {
                "ok": True,
                "teleoperationActive": True,
                "simulationActive": True,
                "operationMode": "active",
                "modeRequest": request_id,
            }
        )

    async def _websocket(self, request):
        web, ws_message_type = self._aiohttp()
        socket = web.WebSocketResponse(heartbeat=10.0, max_msg_size=64 * 1024)
        await socket.prepare(request)
        self._sockets.add(socket)
        self.state.client_connected()
        await socket.send_json({"type": "server-hello", "protocolVersion": PROTOCOL_VERSION})
        scene = self.scene_snapshot()
        if scene is not None:
            await socket.send_str(json.dumps(scene, separators=(",", ":")))
        try:
            async for message in socket:
                if message.type != ws_message_type.TEXT:
                    continue
                try:
                    try:
                        payload = json.loads(message.data)
                    except json.JSONDecodeError as exc:
                        raise ProtocolError(f"message is not valid JSON: {exc.msg}") from exc
                    if not isinstance(payload, Mapping):
                        raise ProtocolError("browser message must be an object")
                    if payload.get("type") == "reset-scene":
                        try:
                            version = int(payload.get("version", -1))
                        except (TypeError, ValueError) as exc:
                            raise ProtocolError("protocol version is invalid") from exc
                        if version != PROTOCOL_VERSION:
                            raise ProtocolError(f"unsupported protocol version: {version}")
                        reset_request = self.state.request_reset()
                        await socket.send_json(
                            {
                                "type": "reset-accepted",
                                "requestId": payload.get("requestId"),
                                "resetRequest": reset_request,
                            }
                        )
                    else:
                        self.state.update(XRFrame.from_mapping(payload))
                except ProtocolError as exc:
                    self.state.reject()
                    await socket.send_json({"type": "protocol-error", "message": str(exc)})
        finally:
            self._sockets.discard(socket)
            self.state.client_disconnected()
        return socket

    async def _scene_sender(self) -> None:
        assert self._scene_event is not None
        while True:
            await self._scene_event.wait()
            self._scene_event.clear()
            scene = self.scene_snapshot()
            sockets = tuple(socket for socket in self._sockets if not socket.closed)
            if scene is None or not sockets:
                continue
            encoded = json.dumps(scene, separators=(",", ":"))
            await asyncio.gather(*(socket.send_str(encoded) for socket in sockets), return_exceptions=True)


__all__ = [
    "DEFAULT_XR_TO_ROBOT",
    "PROTOCOL_VERSION",
    "ControllerState",
    "JsonlTrajectoryRecorder",
    "LatestXRFrame",
    "Pose",
    "ProtocolError",
    "RelativePoseRetargeter",
    "TargetPose",
    "WebXRServer",
    "XRFrame",
    "XRStreamStatus",
    "pack_scene_geometry",
]

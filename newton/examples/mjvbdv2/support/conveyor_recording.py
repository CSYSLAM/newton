# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Streaming visual-state cache for W1 conveyor recording and replay."""

import hashlib
import json
from pathlib import Path

import numpy as np

DEFAULT_RECORDING = Path("newton/tests/outputs/conveyor_recording")
SCENE_OPTIONS = (
    "robot_urdf",
    "belt_speed",
    "substeps",
    "vbd_iterations",
    "ik_iterations",
    "finger_speed",
    "cloth_displacement_threshold",
)


def scene_signature(scene):
    """Identify the body ordering and deformable topology expected by the cache."""
    digest = hashlib.sha256()
    digest.update("\n".join(scene.model.body_label).encode())
    for array in (
        scene.model.tri_indices,
        scene.model.shape_body,
        scene.model.shape_transform,
        scene.model.shape_scale,
    ):
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


class RecordingWriter:
    """Stream frames to memory-mapped arrays without retaining the movie in RAM."""

    def __init__(self, path, scene, capacity):
        if capacity < 1:
            raise ValueError("Recording capacity must be positive")
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.count = 0
        self.capacity = capacity
        self.arrays = {}
        samples = self._sample(scene)
        for name, sample in samples.items():
            self.arrays[name] = np.lib.format.open_memmap(
                self.path / f"{name}.npy", mode="w+", dtype=sample.dtype, shape=(capacity, *sample.shape)
            )
        self.metadata = {
            "format": "newton-conveyor-visual-v1",
            "fps": 60,
            "frame_count": 0,
            "scene_signature": scene_signature(scene),
            "scene_options": {name: getattr(scene.args, name) for name in SCENE_OPTIONS},
            "complete": False,
        }

    @staticmethod
    def _sample(scene):
        return {
            "body_q": scene.state_0.body_q.numpy(),
            "particle_q": scene.state_0.particle_q.numpy(),
            "timing": np.array([scene.sim_time, scene.belt_offset], dtype=np.float64),
            "phase": np.asarray(scene.phase, dtype="U16"),
            "active": np.asarray(scene.active, dtype=np.int32),
            "flags": np.asarray([(p.picked, p.released, p.sorted) for p in scene.parcels], dtype=np.uint8),
        }

    def append(self, scene):
        """Capture every physical frame, including belt travel and parcel status."""
        if self.count >= self.capacity:
            raise ValueError("Recording capacity exceeded")
        for name, sample in self._sample(scene).items():
            self.arrays[name][self.count] = sample
        self.count += 1

    def close(self, *, complete):
        """Publish only the written prefix, including an interrupted recording."""
        for array in self.arrays.values():
            array.flush()
        self.metadata.update(frame_count=self.count, complete=bool(complete))
        temporary = self.path / "metadata.tmp"
        temporary.write_text(json.dumps(self.metadata, indent=2) + "\n")
        temporary.replace(self.path / "metadata.json")


class RecordingReader:
    """Read individual cached frames without decompressing the whole recording."""

    def __init__(self, path):
        self.path = Path(path)
        self.metadata = json.loads((self.path / "metadata.json").read_text())
        if self.metadata.get("format") != "newton-conveyor-visual-v1" or self.metadata.get("fps") != 60:
            raise ValueError("Unsupported conveyor recording format or frame rate")
        self.count = int(self.metadata["frame_count"])
        if self.count < 1:
            raise ValueError("Recording contains no frames")
        self.arrays = {
            name: np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("body_q", "particle_q", "timing", "phase", "active", "flags")
        }
        if any(len(array) < self.count for array in self.arrays.values()):
            raise ValueError("Recording is truncated")

    def validate_scene(self, scene):
        """Reject geometry changes rather than map recorded states onto the wrong model."""
        if scene_signature(scene) != self.metadata["scene_signature"]:
            raise ValueError("Recording geometry differs from this demo; make a new recording")
        for name, sample in RecordingWriter._sample(scene).items():
            array = self.arrays[name]
            if array.shape[1:] != sample.shape or array.dtype != sample.dtype:
                raise ValueError(f"Recording field {name} has an incompatible shape or dtype")

    def restore(self, scene, frame):
        """Restore render state only; never execute FK, IK, collision, or dynamics."""
        if not 0 <= frame < self.count:
            raise IndexError(frame)
        for name in ("body_q", "particle_q"):
            values = self.arrays[name][frame]
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite {name} in frame {frame}")
            getattr(scene.state_0, name).assign(values)
        scene.sim_time, scene.belt_offset = map(float, self.arrays["timing"][frame])
        scene.phase = str(self.arrays["phase"][frame])
        scene.active = int(self.arrays["active"][frame])
        for parcel, flags in zip(scene.parcels, self.arrays["flags"][frame], strict=True):
            parcel.picked, parcel.released, parcel.sorted = map(bool, flags)
        scene.completed = all(p.sorted for p in scene.parcels)

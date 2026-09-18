# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stream W1 bag render states to disk and replay them without simulation."""

import hashlib
import json
import time
from pathlib import Path
from typing import ClassVar

import numpy as np

DEFAULT_RECORDING = Path(__file__).resolve().parents[3] / "tests/outputs/w1_bag_packing"
SCENE_OPTIONS = ("snacks", "robot_setback", "substeps", "iterations")
FORMAT = "newton-w1-bag-visual-v1"


def scene_signature(scene):
    """Identify the body ordering, geometry, and bag topology used by the recording."""
    digest = hashlib.sha256("\n".join(scene.model.body_label).encode())
    for name in ("tri_indices", "shape_body", "shape_transform", "shape_scale", "shape_type"):
        digest.update(getattr(scene.model, name).numpy().tobytes())
    digest.update(scene.rest.tobytes())
    for source in scene.model.shape_source:
        if source is not None and hasattr(source, "vertices"):
            digest.update(np.asarray(source.vertices).tobytes())
            digest.update(np.asarray(source.indices).tobytes())
    return digest.hexdigest()


def _sample(scene):
    return {
        "body_q": scene.state_0.body_q.numpy(),
        "particle_q": scene.state_0.particle_q.numpy(),
        "sim_time": np.asarray(scene.sim_time, dtype=np.float64),
    }


class RecordingWriter:
    """Store every 60 Hz frame in memory-mapped arrays with bounded RAM usage."""

    def __init__(self, path, scene, capacity):
        if capacity < 1:
            raise ValueError("Recording capacity must be positive")
        self.path = Path(path).expanduser()
        self.path.mkdir(parents=True, exist_ok=False)
        self.count, self.capacity = 0, capacity
        self.arrays = {
            name: np.lib.format.open_memmap(
                self.path / f"{name}.npy", mode="w+", dtype=sample.dtype, shape=(capacity, *sample.shape)
            )
            for name, sample in _sample(scene).items()
        }
        self.metadata = {
            "format": FORMAT,
            "fps": 60,
            "frame_count": 0,
            "complete": False,
            "scene_options": {name: getattr(scene.args, name) for name in SCENE_OPTIONS},
            "scene_signature": scene_signature(scene),
        }
        self.close(complete=False)

    def append(self, scene):
        """Record exact robot, snack, and paper poses, including the initial frame."""
        if self.count >= self.capacity:
            raise ValueError("Recording capacity exceeded")
        for name, sample in _sample(scene).items():
            self.arrays[name][self.count] = sample
        self.count += 1
        if self.count % 60 == 0:
            self.close(complete=False)

    def close(self, *, complete):
        """Publish the flushed prefix atomically so interrupted recordings remain usable."""
        for array in self.arrays.values():
            array.flush()
        self.metadata.update(frame_count=self.count, complete=bool(complete))
        temporary = self.path / "metadata.tmp"
        temporary.write_text(json.dumps(self.metadata, indent=2) + "\n")
        temporary.replace(self.path / "metadata.json")


class RecordingReader:
    """Load individual recorded frames without reading the whole sequence into RAM."""

    def __init__(self, path):
        self.path = Path(path).expanduser()
        metadata = self.path / "metadata.json"
        if not metadata.is_file():
            raise FileNotFoundError(f"No recording at {self.path}; run this scene with --record first")
        self.metadata = json.loads(metadata.read_text())
        if self.metadata.get("format") != FORMAT or self.metadata.get("fps") != 60:
            raise ValueError("Unsupported W1 bag recording format or frame rate")
        self.count = int(self.metadata["frame_count"])
        if self.count < 1:
            raise ValueError("Recording contains no frames")
        self.arrays = {
            name: np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("body_q", "particle_q", "sim_time")
        }
        if any(array.ndim < 1 or len(array) < self.count for array in self.arrays.values()):
            raise ValueError("Recording is truncated")

    def validate_scene(self, scene):
        """Reject incompatible assets or state layouts before restoring any frame."""
        if scene_signature(scene) != self.metadata["scene_signature"]:
            raise ValueError("Recording geometry differs from this scene; make a new recording")
        for name, sample in _sample(scene).items():
            array = self.arrays[name]
            if array.shape[1:] != sample.shape or array.dtype != sample.dtype:
                raise ValueError(f"Recording field {name} has an incompatible shape or dtype")

    def restore(self, scene, frame):
        """Restore render state directly without FK, IK, collision detection, or dynamics."""
        if not 0 <= frame < self.count:
            raise IndexError(frame)
        values = {name: array[frame] for name, array in self.arrays.items()}
        for name, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite {name} in frame {frame}")
        for name in ("body_q", "particle_q"):
            getattr(scene.state_0, name).assign(values[name])
        scene.sim_time, scene.frame = float(values["sim_time"]), frame


class TeleopRecordingReader:
    """Index full-state JSONL recordings and load one frame at a time for replay."""

    _fields: ClassVar[dict[str, str]] = {
        "body_q": "bodyPoses",
        "body_qd": "bodyVelocities",
        "particle_q": "bagParticleQ",
        "particle_qd": "bagParticleQd",
        "joint_q": "jointQ",
        "joint_qd": "jointQd",
    }

    def __init__(self, path):
        self.path = Path(path).expanduser()
        self.offsets = []
        header, complete = None, True
        with self.path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if not line.endswith(b"\n"):
                        complete = False
                        break
                    raise ValueError(f"Malformed trajectory record at byte {offset}") from None
                if record.get("type") == "metadata":
                    if header is not None or self.offsets:
                        raise ValueError("Duplicate or misplaced trajectory metadata")
                    header = record
                elif record.get("type") == "frame":
                    self.offsets.append(offset)
        if (
            header is None
            or header.get("format") != "newton_webxr_trajectory_v1"
            or header.get("scene") != "w1-bag-packing"
            or header.get("recordingKind") != "full-state"
        ):
            raise ValueError("Use a full-state recording made by the updated W1 teleoperation demo")
        if not np.isclose(header["frameDtSeconds"], 1 / 60):
            raise ValueError("Unsupported trajectory frame rate; expected 60 Hz")
        self.count = len(self.offsets)
        if not self.count:
            raise ValueError("Recording contains no frames")
        self.metadata = {
            "scene_signature": header["sceneSignature"],
            "scene_options": header["sceneOptions"],
            "complete": complete,
        }

    def _values(self, frame):
        if not 0 <= frame < self.count:
            raise IndexError(frame)
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[frame])
            record = json.loads(stream.readline())
        values = {name: np.asarray(record[key], dtype=np.float32) for name, key in self._fields.items()}
        values["sim_time"] = np.asarray(record["simulationTimeSeconds"], dtype=np.float64)
        for name, value in values.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Nonfinite {name} in frame {frame}")
        return values

    def validate_scene(self, scene):
        """Reject mismatched assets before assigning any recorded state."""
        if self.metadata["scene_signature"] != scene_signature(scene):
            raise ValueError("Recording geometry differs from this scene; use the matching assets")
        self._validate_shapes(scene, self._values(0))

    def _validate_shapes(self, scene, values):
        for name in self._fields:
            if values[name].shape != getattr(scene.state_0, name).numpy().shape:
                raise ValueError(f"Recording field {name} has an incompatible shape")
        if values["sim_time"].shape != ():
            raise ValueError("Recording timestamp must be a scalar")

    def restore(self, scene, frame):
        """Restore positions, deformations, joints and velocities without running physics."""
        values = self._values(frame)
        self._validate_shapes(scene, values)
        for name in self._fields:
            getattr(scene.state_0, name).assign(values[name])
        scene.sim_time, scene.frame = float(values["sim_time"]), frame


def run_recording(scene_type, viewer, args):
    """Bake full physics once, retaining a playable prefix on interruption."""
    writer, complete = None, False
    try:
        if args.num_frames < 1:
            raise ValueError("Recording frame count must be positive")
        path = Path(args.record).expanduser()
        if path.exists():
            raise FileExistsError(f"Recording already exists: {path}; choose a new --record directory")
        scene = scene_type(viewer, args)
        writer = RecordingWriter(path, scene, args.num_frames + 1)
        writer.append(scene)
        started = time.perf_counter()
        if hasattr(viewer, "hide_loading_splash"):
            viewer.hide_loading_splash()
        while scene.frame < args.num_frames and viewer.is_running():
            if viewer.should_step():
                scene.step()
                writer.append(scene)
                if args.test:
                    scene.test_post_step()
                if scene.frame % 300 == 0:
                    print(
                        f"[W1Bag record] {scene.frame}/{args.num_frames}, wall {time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
            if args.viewer != "null":
                scene.render()
                if viewer.is_paused():
                    time.sleep(1 / 60)
        # Short clips are useful for inspection, but are not full packing validations.
        if scene.sim_time >= 24 + args.snacks * 12 + 1:
            scene.test_final()
            complete = True
    finally:
        if writer is not None:
            writer.close(complete=complete)
            print(f"[W1Bag record] Saved {writer.count} frames to {writer.path}; validated={complete}", flush=True)
        viewer.close()


class Playback:
    """Track replay position and expose a seek slider in the GL sidebar."""

    def __init__(self, recording, *, start_frame=0, loop=False):
        if not 0 <= start_frame < recording.count:
            raise ValueError("Start frame is outside the recording")
        self.recording, self.frame, self.loop = recording, start_frame, loop
        self.paused = False

    def gui(self, ui):
        """Allow seeking, restarting, pausing, and looping without rerunning physics."""
        ui.text("W1 bag recording (60 FPS)")
        _, self.paused = ui.checkbox("Pause replay", self.paused)
        _, self.loop = ui.checkbox("Loop replay", self.loop)
        changed, frame = ui.slider_int("Recorded frame", self.frame, 0, self.recording.count - 1)
        if changed:
            self.frame, self.paused = frame, True
        if ui.button("Restart replay"):
            self.frame, self.paused = 0, False
        ui.text(f"{self.frame / 60:.2f} / {(self.recording.count - 1) / 60:.2f} s")

    def advance(self):
        """Advance one saved frame, optionally wrapping to the initial state."""
        if self.frame + 1 < self.recording.count:
            self.frame += 1
            return True
        if self.loop:
            self.frame = 0
            return True
        return False


def run_replay(scene_type, viewer, args):
    """Render saved frames at 60 FPS, preserving camera controls and all materials."""
    try:
        path = Path(args.replay).expanduser()
        if not path.exists():
            raise FileNotFoundError(
                f"Recording not found: {path}. Check the running teleoperation process's --trajectory-output; "
                "resuming an existing process does not change its recording filename."
            )
        recording = TeleopRecordingReader(path) if path.is_file() else RecordingReader(path)
        playback = Playback(recording, start_frame=args.start_frame, loop=args.loop)
        for name in SCENE_OPTIONS:
            setattr(args, name, recording.metadata["scene_options"][name])
        scene = scene_type.create_render_scene(viewer, args)
        recording.validate_scene(scene)
        if hasattr(viewer, "register_ui_callback"):
            viewer.register_ui_callback(playback.gui, position="side")
        if hasattr(viewer, "hide_loading_splash"):
            viewer.hide_loading_splash()
        print(f"[W1Bag replay] {recording.count} frames from {recording.path}; physics disabled", flush=True)
        if not recording.metadata["complete"]:
            print("[W1Bag replay] Playing a partial or unvalidated recording.", flush=True)
        rendered, started = 0, time.perf_counter()
        while viewer.is_running() and rendered < args.num_frames:
            tick = time.perf_counter()
            advance = viewer.should_step()
            displayed_frame = playback.frame
            recording.restore(scene, displayed_frame)
            scene.render()
            rendered += 1
            if advance and not playback.paused and playback.frame == displayed_frame and not playback.advance():
                if args.viewer == "gl" and not args.headless:
                    playback.paused = True
                else:
                    break
            if not args.unthrottled:
                time.sleep(max(0.0, 1 / 60 - (time.perf_counter() - tick)))
        elapsed = time.perf_counter() - started
        print(f"[W1Bag replay] {rendered} frames in {elapsed:.3f}s ({rendered / max(elapsed, 1e-9):.1f} FPS)")
    finally:
        viewer.close()

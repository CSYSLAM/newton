# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Record the W1 inflatable-bag pick-and-release simulation for replay.

The simulation runs headlessly and stores every completed frame in a compressed
NumPy cache under the repository's ``assets`` directory by default. The
matching replay example restores the cached state without advancing physics.

Run from the repository root::

    uv run --extra examples -m newton.examples \
        vbd_mjvbd_v2_right_hand_recorded_inflatable_bag_pick_release_record
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton.examples
from newton.examples.mjvbdv2 import (
    example_vbd_mjvbd_v2_right_hand_recorded_inflatable_bag_pick_release as simulation,
)

ASSET_DIRECTORY = Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2"
DEFAULT_RECORDING = ASSET_DIRECTORY / "vbd_w1_right_hand_inflatable_bag_pick_release.npz"
CACHE_FORMAT = "newton_w1_right_hand_inflatable_bag_pick_release_v1"
CACHE_STATE_PATHS = {
    "joint_q": ("joint_q",),
    "joint_qd": ("joint_qd",),
    "body_q": ("body_q",),
    "body_qd": ("body_qd",),
    "particle_q": ("particle_q",),
    "particle_qd": ("particle_qd",),
    "pneumatic_volume": ("pneumatic", "volume"),
    "pneumatic_absolute_pressure": ("pneumatic", "absolute_pressure"),
    "pneumatic_volume_rate": ("pneumatic", "volume_rate"),
    "pneumatic_clamp_flags": ("pneumatic", "clamp_flags"),
}


class Example(simulation.Example):
    """Run the autonomous grasp simulation for offline state recording."""

    @staticmethod
    def create_parser():
        """Create parser options for offline state recording."""

        parser = simulation.Example.create_parser()
        parser.set_defaults(viewer="null", paused=False, num_frames=720)
        parser.add_argument(
            "--recording",
            default=str(DEFAULT_RECORDING),
            help="Output NumPy state cache; defaults to the repository assets directory.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace an existing recording file.",
        )
        parser.add_argument(
            "--progress-interval",
            type=int,
            default=60,
            help="Print recording progress every N frames; use 0 to disable.",
        )
        return parser


def _state_array(state, path: tuple[str, ...]) -> wp.array[Any]:
    """Return one required state array addressed by an attribute path."""

    value = state
    for attribute in path:
        value = getattr(value, attribute, None)
        if value is None:
            raise ValueError(f"Simulation state does not provide {'.'.join(path)}")
    if not isinstance(value, wp.array):
        raise TypeError(f"Simulation state field {'.'.join(path)} is not a Warp array")
    return value


def _allocate_state_cache(example: Example, frame_count: int) -> dict[str, np.ndarray]:
    """Allocate host arrays matching all replay-relevant state arrays."""

    cache = {}
    for name, path in CACHE_STATE_PATHS.items():
        frame = _state_array(example.state_0, path).numpy()
        cache[name] = np.empty((frame_count, *frame.shape), dtype=frame.dtype)
    return cache


def _store_state(cache: dict[str, np.ndarray], example: Example, frame: int) -> None:
    """Copy one completed simulation frame into host memory."""

    for name, path in CACHE_STATE_PATHS.items():
        cache[name][frame] = _state_array(example.state_0, path).numpy()


def _save_cache(path: Path, args, example: Example, cache: dict[str, np.ndarray]) -> None:
    """Atomically save state arrays and scene-reconstruction metadata."""

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    metadata = {
        "format": np.asarray(CACHE_FORMAT),
        "fps": np.asarray(1.0 / example.frame_dt, dtype=np.float32),
        "frame_count": np.asarray(args.num_frames, dtype=np.int32),
        "grasp_keyframe": np.asarray(str(Path(args.grasp_keyframe).expanduser().resolve())),
        "pneumatic_mode": np.asarray(str(args.pneumatic_mode)),
    }
    try:
        with temporary_path.open("wb") as output:
            np.savez_compressed(output, **metadata, **cache)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def main():
    """Run the physical simulation headlessly and save completed states."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    recording_path = Path(args.recording).expanduser()
    if args.num_frames <= 0:
        parser.error("--num-frames must be greater than zero")
    if args.progress_interval < 0:
        parser.error("--progress-interval must not be negative")
    if recording_path.suffix.lower() != ".npz":
        parser.error("--recording must use the .npz extension")
    if recording_path.exists() and not args.overwrite:
        raise FileExistsError(f"Recording already exists: {recording_path}; pass --overwrite to replace it")
    recording_path.parent.mkdir(parents=True, exist_ok=True)

    example = Example(viewer, args)
    state_cache = _allocate_state_cache(example, args.num_frames)
    start_time = time.perf_counter()
    try:
        for frame in range(args.num_frames):
            example.step()
            if args.test:
                example.test_post_step()
            _store_state(state_cache, example, frame)
            completed = frame + 1
            if args.progress_interval > 0 and (completed % args.progress_interval == 0 or completed == args.num_frames):
                elapsed = time.perf_counter() - start_time
                simulation_fps = completed / max(elapsed, 1.0e-9)
                print(f"Recorded {completed}/{args.num_frames} frames ({simulation_fps:.2f} simulation FPS)")
        if args.test:
            example.test_final()
        _save_cache(recording_path, args, example, state_cache)
    finally:
        viewer.close()

    elapsed = time.perf_counter() - start_time
    size_mib = recording_path.stat().st_size / (1024.0 * 1024.0)
    print(f"Saved {args.num_frames} frames to {recording_path} ({size_mib:.1f} MiB) in {elapsed:.1f} s")


if __name__ == "__main__":
    main()

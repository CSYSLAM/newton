# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Record the full-W1 soft-then-rigid simulation for fast state replay.

The expensive IK, collision detection, and MJVBD solve run only while this
script creates the cache. The matching replay example restores the cached
joint, rigid-body, and particle states without advancing physics.

Run from the repository root::

    uv run --extra examples -m newton.examples vbd_mjvbd_v2_dexforce_recorded_soft_then_rigid_cube_into_bag_record
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import newton.examples
from newton.examples.mjvbdv2 import example_vbd_mjvbd_v2_dexforce_recorded_soft_then_rigid_cube_into_bag as simulation

DEFAULT_RECORDING = (
    Path(__file__).resolve().parents[3]
    / "assets"
    / "vbd_mjvbd_v2"
    / "vbd_mjvbd_v2_dexforce_soft_then_rigid_cube_into_bag.npz"
)
CACHE_FORMAT = "newton_dexforce_soft_then_rigid_replay_v1"
CACHE_STATE_FIELDS = (
    "joint_q",
    "joint_qd",
    "body_q",
    "body_qd",
    "particle_q",
    "particle_qd",
)


class Example(simulation.Example):
    """Run the canonical full-W1 simulation for offline state recording."""

    @staticmethod
    def create_parser():
        """Create parser options for offline state recording."""

        parser = simulation.Example.create_parser()
        parser.set_defaults(viewer="null", paused=False, num_frames=1900)
        parser.add_argument(
            "--recording",
            default=str(DEFAULT_RECORDING),
            help="Output NumPy state cache.",
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


def _allocate_state_cache(example: Example, frame_count: int) -> dict[str, np.ndarray]:
    """Allocate host arrays matching all replay-relevant state arrays."""

    cache = {}
    for field in CACHE_STATE_FIELDS:
        state_array = getattr(example.state_0, field)
        if state_array is None:
            raise ValueError(f"Simulation state does not provide {field}")
        frame = state_array.numpy()
        cache[field] = np.empty((frame_count, *frame.shape), dtype=frame.dtype)
    return cache


def _store_state(cache: dict[str, np.ndarray], example: Example, frame: int) -> None:
    """Copy one completed simulation frame into host memory."""

    for field in CACHE_STATE_FIELDS:
        cache[field][frame] = getattr(example.state_0, field).numpy()


def _save_cache(path: Path, args, cache: dict[str, np.ndarray]) -> None:
    """Atomically save state arrays and reconstruction metadata."""

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    metadata = {
        "format": np.asarray(CACHE_FORMAT),
        "fps": np.asarray(simulation.soft0.FPS, dtype=np.float32),
        "frame_count": np.asarray(args.num_frames, dtype=np.int32),
        "recorded_grasp_keyframe": np.asarray(str(Path(args.recorded_grasp_keyframe).expanduser())),
        "rigid_grasp_keyframe": np.asarray(str(Path(args.rigid_grasp_keyframe).expanduser())),
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
            _store_state(state_cache, example, frame)
            completed = frame + 1
            if args.progress_interval > 0 and (completed % args.progress_interval == 0 or completed == args.num_frames):
                elapsed = time.perf_counter() - start_time
                fps = completed / max(elapsed, 1.0e-9)
                print(f"Recorded {completed}/{args.num_frames} frames ({fps:.2f} simulation FPS)")
        if args.test:
            example.test_final()
        _save_cache(recording_path, args, state_cache)
    finally:
        viewer.close()

    elapsed = time.perf_counter() - start_time
    size_mib = recording_path.stat().st_size / (1024.0 * 1024.0)
    print(f"Saved {args.num_frames} frames to {recording_path} ({size_mib:.1f} MiB) in {elapsed:.1f} s")


if __name__ == "__main__":
    main()

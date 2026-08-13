# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Replay cached W1 inflatable-bag pick-and-release states.

Run the matching record example once, then play the generated asset::

    uv run --extra examples -m newton.examples \
        vbd_mjvbd_v2_right_hand_recorded_inflatable_bag_pick_release_replay \
        --viewer gl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import newton.examples
from newton.examples.mjvbdv2 import (
    example_vbd_mjvbd_v2_right_hand_recorded_inflatable_bag_pick_release as simulation,
)
from newton.examples.mjvbdv2 import (
    example_vbd_mjvbd_v2_right_hand_recorded_inflatable_bag_pick_release_record as recorder,
)


class Example:
    """Restore cached states into the matching hand-and-bag model."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.recording_path = Path(args.recording).expanduser()
        self.frames, metadata = self._load_cache(self.recording_path)
        self.total_frames = int(metadata["frame_count"])
        self.recorded_fps = float(metadata["fps"])
        if not 0 <= args.start_frame < self.total_frames:
            raise ValueError(f"--start-frame must be in [0, {self.total_frames}), got {args.start_frame}")

        args.grasp_keyframe = self._resolve_keyframe(str(metadata["grasp_keyframe"]), args.grasp_keyframe)
        args.pneumatic_mode = str(metadata["pneumatic_mode"])
        self.source = simulation.Example(viewer, args)
        self.model = self.source.model
        self.state_0 = self.source.state_0
        self._validate_state_shapes()

        self.frame_index = args.start_frame
        self.displayed_frame = args.start_frame
        self.sim_time = (self.displayed_frame + 1) / self.recorded_fps
        self._restore_state(self.frame_index)

    @staticmethod
    def _resolve_keyframe(stored: str, fallback: str) -> str:
        """Resolve a recorded keyframe, preferring assets on this machine."""

        stored_path = Path(stored).expanduser()
        if stored_path.is_file():
            return str(stored_path)
        asset_path = recorder.ASSET_DIRECTORY / stored_path.name
        if asset_path.is_file():
            return str(asset_path)
        fallback_path = Path(fallback).expanduser()
        if fallback_path.is_file():
            return str(fallback_path)
        raise FileNotFoundError(
            f"Grasp keyframe not found at recorded, asset, or fallback paths: "
            f"{stored_path}, {asset_path}, {fallback_path}"
        )

    @staticmethod
    def _load_cache(path: Path) -> tuple[dict[str, np.ndarray], dict[str, int | float | str]]:
        """Load and validate a recording before reconstructing the model."""

        if not path.is_file():
            raise FileNotFoundError(f"Recording not found: {path}. Run the matching record example first.")
        try:
            with np.load(path, allow_pickle=False) as archive:
                cache_format = str(archive["format"].item())
                if cache_format != recorder.CACHE_FORMAT:
                    raise ValueError(f"Unsupported recording format {cache_format!r}")
                metadata: dict[str, int | float | str] = {
                    "fps": float(archive["fps"].item()),
                    "frame_count": int(archive["frame_count"].item()),
                    "grasp_keyframe": str(archive["grasp_keyframe"].item()),
                    "pneumatic_mode": str(archive["pneumatic_mode"].item()),
                }
                frames = {name: np.asarray(archive[name]) for name in recorder.CACHE_STATE_PATHS}
        except (KeyError, OSError, ValueError) as error:
            raise ValueError(f"Invalid recording {path}: {error}") from error

        fps = float(metadata["fps"])
        frame_count = int(metadata["frame_count"])
        if not np.isfinite(fps) or fps <= 0.0 or frame_count <= 0:
            raise ValueError(f"Invalid recording timing metadata: {metadata}")
        if metadata["pneumatic_mode"] not in simulation.recorder.PNEUMATIC_MODES:
            raise ValueError(f"Invalid recorded pneumatic mode: {metadata['pneumatic_mode']!r}")
        for name, values in frames.items():
            if values.shape[0] != frame_count:
                raise ValueError(f"Recording field {name} has {values.shape[0]} frames, expected {frame_count}")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Recording field {name} contains non-finite values")
        return frames, metadata

    def _validate_state_shapes(self) -> None:
        """Require the reconstructed model to match every cached state array."""

        for name, path in recorder.CACHE_STATE_PATHS.items():
            state_shape = recorder._state_array(self.state_0, path).numpy().shape
            recorded_shape = self.frames[name].shape[1:]
            if recorded_shape != state_shape:
                raise ValueError(
                    f"Recording field {name} has per-frame shape {recorded_shape}, "
                    f"but the reconstructed model expects {state_shape}"
                )

    def _restore_state(self, frame: int) -> None:
        """Copy one cached frame to the render state."""

        for name, path in recorder.CACHE_STATE_PATHS.items():
            recorder._state_array(self.state_0, path).assign(self.frames[name][frame])

    def step(self):
        """Load the next cached frame without executing simulation work."""

        self._restore_state(self.frame_index)
        self.displayed_frame = self.frame_index
        self.sim_time = (self.displayed_frame + 1) / self.recorded_fps
        self.source.sim_time = self.sim_time
        next_frame = self.frame_index + self.args.frame_step
        if next_frame >= self.total_frames:
            self.frame_index = self.args.start_frame if self.args.loop else self.total_frames - 1
        else:
            self.frame_index = next_frame

    def render(self):
        """Render the cached hand and explicit inflatable-bag surface."""

        self.source.sim_time = self.sim_time
        self.source.render()

    def test_final(self):
        """Verify that every restored replay field remains finite."""

        for name, path in recorder.CACHE_STATE_PATHS.items():
            values = recorder._state_array(self.state_0, path).numpy()
            assert np.all(np.isfinite(values)), f"Replay state field {name} contains non-finite values."

    @staticmethod
    def create_parser():
        """Create parser options for cached-state playback."""

        parser = simulation.Example.create_parser()
        parser.set_defaults(viewer="gl", paused=False, num_frames=720, render_fps=60.0)
        parser.add_argument(
            "--recording",
            default=str(recorder.DEFAULT_RECORDING),
            help="NumPy state cache produced by the matching record example.",
        )
        parser.add_argument(
            "--start-frame",
            type=int,
            default=0,
            help="First frame to display and loop back to.",
        )
        parser.add_argument(
            "--frame-step",
            type=int,
            default=1,
            help="Number of cached frames advanced per rendered frame.",
        )
        parser.add_argument(
            "--loop",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Loop to --start-frame after reaching the recording end.",
        )
        return parser


def main():
    """Load and replay the cached hand-and-bag trajectory."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    if args.frame_step <= 0:
        parser.error("--frame-step must be greater than zero")
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

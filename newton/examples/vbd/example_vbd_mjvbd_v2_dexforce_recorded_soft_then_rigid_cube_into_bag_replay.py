# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Replay cached full-W1 soft-then-rigid states without advancing physics.

Run the matching record example once, then play the resulting cache::

    uv run --extra examples -m newton.examples vbd_mjvbd_v2_dexforce_recorded_soft_then_rigid_cube_into_bag_replay --viewer gl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import newton.examples
from newton.examples.vbd import example_vbd_mjvbd_v2_dexforce_recorded_soft_then_rigid_cube_into_bag as simulation
from newton.examples.vbd import example_vbd_mjvbd_v2_dexforce_recorded_soft_then_rigid_cube_into_bag_record as recorder


class Example:
    """Restore cached states into a matching full-W1 model for rendering."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.recording_path = Path(args.recording).expanduser()
        self.frames, metadata = self._load_cache(self.recording_path)
        self.total_frames = int(metadata["frame_count"])
        self.recorded_fps = float(metadata["fps"])
        if not 0 <= args.start_frame < self.total_frames:
            raise ValueError(f"--start-frame must be in [0, {self.total_frames}), got {args.start_frame}")

        # Reconstruct the same model once. Physics, collision detection, and IK
        # are never called by this replay object's step().
        args.recorded_grasp_keyframe = metadata["recorded_grasp_keyframe"]
        args.rigid_grasp_keyframe = metadata["rigid_grasp_keyframe"]
        self.source = simulation.Example(viewer, args)
        self.model = self.source.model
        self.state_0 = self.source.state_0
        self._validate_state_shapes()

        self.frame_index = args.start_frame
        self.displayed_frame = args.start_frame
        self.sim_time = self.displayed_frame / self.recorded_fps
        self._restore_state(self.frame_index)

    @staticmethod
    def _load_cache(path: Path) -> tuple[dict[str, np.ndarray], dict[str, int | float | str]]:
        """Load and validate the recording before reconstructing the model."""

        if not path.is_file():
            raise FileNotFoundError(f"Recording not found: {path}. Run the matching record example first.")
        try:
            with np.load(path, allow_pickle=False) as archive:
                cache_format = str(archive["format"].item())
                if cache_format != recorder.CACHE_FORMAT:
                    raise ValueError(f"Unsupported recording format {cache_format!r}")
                metadata = {
                    "fps": float(archive["fps"].item()),
                    "frame_count": int(archive["frame_count"].item()),
                    "recorded_grasp_keyframe": str(archive["recorded_grasp_keyframe"].item()),
                    "rigid_grasp_keyframe": str(archive["rigid_grasp_keyframe"].item()),
                }
                frames = {field: np.asarray(archive[field]) for field in recorder.CACHE_STATE_FIELDS}
        except (KeyError, OSError, ValueError) as error:
            raise ValueError(f"Invalid recording {path}: {error}") from error

        if metadata["fps"] <= 0.0 or metadata["frame_count"] <= 0:
            raise ValueError(f"Invalid recording timing metadata: {metadata}")
        for field, values in frames.items():
            if values.shape[0] != metadata["frame_count"]:
                raise ValueError(
                    f"Recording field {field} has {values.shape[0]} frames, expected {metadata['frame_count']}"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Recording field {field} contains non-finite values")
        return frames, metadata

    def _validate_state_shapes(self) -> None:
        """Require the reconstructed model to match every cached state array."""

        for field, values in self.frames.items():
            state_array = getattr(self.state_0, field)
            if state_array is None:
                raise ValueError(f"Reconstructed simulation state does not provide {field}")
            state_shape = state_array.numpy().shape
            if values.shape[1:] != state_shape:
                raise ValueError(
                    f"Recording field {field} has per-frame shape {values.shape[1:]}, "
                    f"but the reconstructed model expects {state_shape}"
                )

    def _restore_state(self, frame: int) -> None:
        """Copy one cached frame to the render state."""

        for field, values in self.frames.items():
            getattr(self.state_0, field).assign(values[frame])

    def step(self):
        """Load the next cached frame without executing simulation work."""

        self._restore_state(self.frame_index)
        self.displayed_frame = self.frame_index
        self.sim_time = self.displayed_frame / self.recorded_fps
        next_frame = self.frame_index + self.args.frame_step
        if next_frame >= self.total_frames:
            self.frame_index = self.args.start_frame if self.args.loop else self.total_frames - 1
        else:
            self.frame_index = next_frame

    def render(self):
        """Render the currently loaded cached state."""

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify that the restored replay state remains finite."""

        assert np.all(np.isfinite(self.state_0.joint_q.numpy()))
        assert np.all(np.isfinite(self.state_0.body_q.numpy()))
        assert np.all(np.isfinite(self.state_0.particle_q.numpy()))

    @staticmethod
    def create_parser():
        """Create parser options for cached-state playback."""

        parser = simulation.Example.create_parser()
        parser.set_defaults(viewer="gl", paused=False, render_fps=60.0)
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
    """Load and replay the cached full-W1 trajectory."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    if args.frame_step <= 0:
        parser.error("--frame-step must be greater than zero")
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Replay the recorded W1 handle-free bag-packing scene without physics or IK.

Run from the repository root::

    uv run --extra examples -m newton.examples mjvbd_v2_w1_bag_packing_replay

This defaults to ``recordings/bag_no_handles_05.jsonl`` with looping enabled, matching
``mjvbd_v2_w1_bag_packing --replay recordings/bag_no_handles_05.jsonl --loop``.
The recording must be supplied locally. Use ``--replay PATH`` for another
full-state JSONL or state directory, ``--no-loop`` to play once, and the sidebar
to pause or seek. Geometry, materials, camera, and state restoration are shared
with the original example; scene options come from the recording metadata.
"""

import argparse
from pathlib import Path

import numpy as np

import newton.examples
from newton.examples.mjvbdv2.example_mjvbd_v2_w1_bag_packing import Example as BagScene
from newton.examples.mjvbdv2.support.w1_bag_recording import load_replay_scene

DEFAULT_RECORDING = Path(__file__).resolve().parents[3] / "recordings/bag_no_handles_05.jsonl"


class Example:
    """Display exact saved scene states using the standard example viewer."""

    def __init__(self, viewer, args):
        self.viewer, self.args = viewer, args
        self.scene, self.playback = load_replay_scene(BagScene, viewer, args)
        self.model, self.state_0 = self.scene.model, self.scene.state_0
        self.frame_dt = self.scene.frame_dt
        self._advance = False
        self._rendered = 0
        self.exit_requested = False
        self.playback.recording.restore(self.scene, self.playback.frame)
        self.sim_time = self.scene.sim_time
        print(
            f"[W1Bag replay] {self.playback.recording.count} frames from "
            f"{self.playback.recording.path}; physics disabled",
            flush=True,
        )
        if not self.playback.recording.metadata["complete"]:
            print("[W1Bag replay] Playing a partial or unvalidated recording.", flush=True)

    def step(self):
        """Request one saved-frame advance after rendering the current frame."""
        self._advance = True

    def render(self):
        """Render recorded states, including seeks while playback is paused."""
        displayed_frame = self.playback.frame
        self.playback.recording.restore(self.scene, displayed_frame)
        self.sim_time = self.scene.sim_time
        self.scene.render()
        self._rendered += 1
        if self._advance and not self.playback.paused and self.playback.frame == displayed_frame:
            if not self.playback.advance():
                if self.args.viewer == "gl" and not self.args.headless:
                    self.playback.paused = True
                else:
                    self.exit_requested = True
        self._advance = False
        if self._rendered >= self.args.num_frames:
            self.exit_requested = True

    def gui(self, ui):
        """Expose the original replay pause, seek, restart, and loop controls."""
        self.playback.gui(ui)

    def test_final(self):
        """Verify replayed body and particle positions remain finite."""
        for name in ("body_q", "particle_q"):
            if not np.isfinite(getattr(self.state_0, name).numpy()).all():
                raise ValueError(f"Nonfinite {name} in W1 bag replay")

    @staticmethod
    def create_parser():
        """Create playback options with the reference recording and loop defaults."""
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=2**31 - 1, render_fps=60.0)
        parser.add_argument("--replay", default=str(DEFAULT_RECORDING), metavar="PATH", help="Full-state recording.")
        parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True, help="Loop the recording.")
        parser.add_argument("--start-frame", type=int, default=0, help="First saved frame (zero-based).")
        parser.add_argument("--unthrottled", action="store_true", help="Replay without the render FPS limit.")
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    if args.unthrottled:
        args.render_fps = None
    try:
        example = Example(viewer, args)
    except Exception:
        viewer.close()
        raise
    newton.examples.run(example, args)

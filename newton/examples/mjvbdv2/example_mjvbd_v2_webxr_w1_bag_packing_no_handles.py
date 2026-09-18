# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate W1 packing with a separate paper-bag asset without handles.

Start and record from the repository root:

./scripts/start_quest_webxr_w1_bag_packing_no_handles_teleop.sh --trajectory-output recordings/bag_no_handles_01.jsonl

Press right-controller A or the hand-tracking panel's record button to start.
Press again to pause; the dedicated stop script closes and flushes the recording.
To discard a failed take, pause, reset physics, then press record again. The last
step replaces all previous frames in the same file. Pause/resume without reset appends.
Replay automatically selects the recorded bag asset:

uv run --extra examples -m newton.examples mjvbd_v2_w1_bag_packing --replay recordings/bag_no_handles_01.jsonl --loop
"""

from .example_mjvbd_v2_webxr_w1_bag_packing import Example as PackingExample
from .example_mjvbd_v2_webxr_w1_bag_packing import main


class Example(PackingExample):
    """Reuse packing physics and both input modes with a handle-free paper shell."""

    recording_prefix = "webxr_w1_bag_packing_no_handles"
    scene_title = "W1 无提手纸袋装零食遥操作"

    @staticmethod
    def create_parser():
        parser = PackingExample.create_parser()
        parser.set_defaults(bag_variant="no-handles", webxr_port=8774)
        return parser


if __name__ == "__main__":
    main(Example)

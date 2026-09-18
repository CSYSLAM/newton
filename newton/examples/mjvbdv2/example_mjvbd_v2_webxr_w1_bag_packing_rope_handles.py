# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate W1 packing with thick rope handles pointing away from the mouth.

./scripts/start_quest_webxr_w1_bag_packing_rope_handles_teleop.sh --trajectory-output recordings/bag_rope_01.jsonl

Press right-controller A or the optical-hand record button to start or pause.
Pause, reset physics, then record again to replace a failed take in the same file.
Replay automatically selects the rope-handle asset:

uv run --extra examples -m newton.examples mjvbd_v2_w1_bag_packing --replay recordings/bag_rope_01.jsonl --loop
"""

from .example_mjvbd_v2_webxr_w1_bag_packing import Example as PackingExample
from .example_mjvbd_v2_webxr_w1_bag_packing import main


class Example(PackingExample):
    """Keep the shared scene and controls with separate downward rope handles."""

    recording_prefix = "webxr_w1_bag_packing_rope_handles"
    scene_title = "W1 麻绳提手纸袋装零食遥操作"
    handle_color = (0.24, 0.16, 0.085)
    _paper_stiffness_scale = 1.5
    _initial_bag_offset = (0.0, 0.28, 0.0)

    @staticmethod
    def create_parser():
        parser = PackingExample.create_parser()
        parser.set_defaults(bag_variant="rope-handles", webxr_port=8775)
        return parser


if __name__ == "__main__":
    main(Example)

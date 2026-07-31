# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Replay the baked complete-W1 shuffle and deck-centering continuation.

Generate the cache from the Genesis repository root::

    uv run python examples/IPC_Solver/card_shuffle_w1_hands/w1_full_robot_square_bake.py

Replay it from the Newton repository root::

    uv run --extra examples -m newton.examples genesis_w1_card_shuffle_square_replay

This reuses the established Genesis W1 cache renderer.  Robot links and
deforming card vertices are interpolated from the bake; Newton does not
re-solve IPC or arm IK during playback.
"""

from __future__ import annotations

from pathlib import Path

import newton
import newton.examples

from newton.examples.cloth.example_genesis_w1_card_shuffle_replay import (
    Example as BaseExample,
)


DEFAULT_CACHE = (
    Path(__file__).resolve().parent
    / "assets"
    / "genesis_w1_card_shuffle"
    / "w1_card_shuffle_square_bake.npz"
)


class Example(BaseExample):
    """Render the baked complete W1 and recorded deck-centering motion."""

    @staticmethod
    def create_parser():
        parser = BaseExample.create_parser()
        parser.set_defaults(
            cache=str(DEFAULT_CACHE),
            loop=False,
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    if args.playback_fps is not None and args.playback_fps <= 0.0:
        parser.error("--playback-fps must be positive.")
    if args.speed <= 0.0:
        parser.error("--speed must be positive.")
    newton.examples.run(Example(viewer, args), args)

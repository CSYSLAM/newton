# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure the warm default popcorn control loop without rendering."""

import argparse
import json
import time

import warp as wp

from newton.examples.mjvbdv2.example_mjvbd_v2_popcorn import Example
from newton.viewer import ViewerNull


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--validate", action="store_true", help="Run the scene's unchanged final acceptance")
    args = parser.parse_args()
    if args.warmup < 1 or args.frames < 1:
        parser.error("Frame counts must be positive")
    options = Example.create_parser().parse_args([])
    example = Example(ViewerNull(), options)
    for _ in range(args.warmup):
        example.step()
    wp.synchronize_device(example.model.device)
    start = time.perf_counter()
    for _ in range(args.frames):
        example.step()
    wp.synchronize_device(example.model.device)
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            {
                "device": str(example.model.device),
                "frames": args.frames,
                "substeps": options.substeps,
                "iterations": options.vbd_iterations,
                "grains": options.popcorn_count,
                "wall_ms_per_frame": 1000 * elapsed / args.frames,
                "fps": args.frames / elapsed,
                "rendering": False,
            }
        ),
        flush=True,
    )
    if args.validate:
        example.test_final()


if __name__ == "__main__":
    main()

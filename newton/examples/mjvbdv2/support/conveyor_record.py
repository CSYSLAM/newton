# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record the complete conveyor simulation for solver-free 60 FPS playback."""

import time

import numpy as np

import newton.examples
from newton.examples.mjvbdv2.example_mjvbd_v2_conveyor_sorting import Example
from newton.examples.mjvbdv2.support.conveyor_recording import DEFAULT_RECORDING, RecordingWriter


def main():
    """Run full physics offline and stream each 60 Hz state to a new directory."""
    parser = Example.create_parser()
    parser.set_defaults(viewer="null")
    parser.add_argument(
        "--recording",
        default=str(DEFAULT_RECORDING),
        help="New output directory; existing paths are never overwritten.",
    )
    parser.add_argument(
        "--settle-frames", type=int, default=120, help="Additional physical frames after all four placements."
    )
    viewer, args = newton.examples.init(parser)
    writer = None
    validated = False
    try:
        if args.num_frames < 1 or args.settle_frames < 0:
            raise ValueError("Frame count must be positive and settle frames nonnegative")
        scene = Example(viewer, args)
        writer = RecordingWriter(args.recording, scene, args.num_frames + 1)
        writer.append(scene)
        finished = 0
        started = time.perf_counter()
        for frame in range(args.num_frames):
            scene.step()
            if args.test:
                scene.test_post_step()
            writer.append(scene)
            if args.viewer != "null":
                scene.render()
                if not viewer.is_running():
                    break
            if frame % 300 == 0:
                print(
                    f"Recorded {frame + 1} frames, {scene.sim_time:.1f} s, {scene.phase}, wall {time.perf_counter() - started:.1f} s",
                    flush=True,
                )
            if scene.completed:
                finished += 1
                if finished > args.settle_frames and float(np.mean(scene.cloth_speed_samples)) <= 0.002:
                    scene.test_final()
                    validated = True
                    break
        if scene.completed and not validated:
            scene.test_final()
            validated = True
        print(f"Saved {writer.count} frames to {args.recording}; complete={validated}", flush=True)
    finally:
        if writer is not None:
            writer.close(complete=validated)
        viewer.close()


if __name__ == "__main__":
    main()

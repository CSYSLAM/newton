# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Play cached conveyor states at 60 FPS without running the solvers."""

import time

import newton.examples
from newton.examples.mjvbdv2.example_mjvbd_v2_conveyor_sorting import Example
from newton.examples.mjvbdv2.support.conveyor_recording import DEFAULT_RECORDING, RecordingReader


def main():
    """Restore one recorded frame per display tick, with pause and optional looping."""
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=9000, render_fps=60)
    parser.add_argument("--recording", default=str(DEFAULT_RECORDING))
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--unthrottled", action="store_true", help="Measure rendering throughput without the 60 FPS cap."
    )
    viewer, args = newton.examples.init(parser)
    try:
        recording = RecordingReader(args.recording)
        print(f"Loaded {recording.count} frames ({(recording.count - 1) / 60:.2f} s) from {args.recording}", flush=True)
        if not recording.metadata["complete"]:
            print("Recording ended before final validation passed; playing its saved frames.", flush=True)
        if not 0 <= args.start_frame < recording.count:
            raise ValueError("Start frame is outside the recording")
        scene_args = newton.examples.default_args(Example.create_parser())
        for name, value in recording.metadata["scene_options"].items():
            setattr(scene_args, name, value)
        scene = Example.create_render_scene(viewer, scene_args)
        recording.validate_scene(scene)
        frame = args.start_frame
        recording.restore(scene, frame)
        if hasattr(viewer, "hide_loading_splash"):
            viewer.hide_loading_splash()
        rendered = 0
        started = time.perf_counter()
        deadline = started
        while viewer.is_running() and rendered < args.num_frames:
            advance = viewer.should_step()
            recording.restore(scene, frame)
            scene.render()
            rendered += 1
            if advance:
                frame += 1
                if frame == recording.count:
                    if args.loop:
                        frame = 0
                    else:
                        break
            if not args.unthrottled:
                deadline += 1 / 60
                now = time.perf_counter()
                if deadline > now:
                    time.sleep(deadline - now)
                else:
                    # Do not skip recorded frames or burst to catch up after a stall.
                    deadline = now
        elapsed = time.perf_counter() - started
        print(f"Replay: {rendered} frames in {elapsed:.3f} s ({rendered / max(elapsed, 1e-9):.2f} FPS)")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()

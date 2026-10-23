# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Record the physical bread-cutting trajectory, stills and measurements."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import warp as wp

from newton.examples.mjmpm.example_mjmpm_w1_bread_cut import Example
from newton.examples.mjmpm.example_mjmpm_w1_bread_three_pieces import Example as ThreePieceExample
from newton.viewer import ViewerGL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--scene", choices=("single", "three-pieces"), default="single")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    options, extra = parser.parse_known_args()
    from PIL import Image

    options.output_dir.mkdir(parents=True, exist_ok=True)
    wp.set_device("cuda:0")
    example_class = ThreePieceExample if options.scene == "three-pieces" else Example
    args = example_class.create_parser().parse_args(["--viewer", "null", *extra])
    required_frames = 1200 if options.scene == "three-pieces" else 660
    if options.frames is None:
        options.frames = required_frames
    stills = (
        (0, 179, 389, 599, 659, 809, 929, 1025, 1199)
        if options.scene == "three-pieces"
        else (0, 179, 269, 389, 449, 599, 659)
    )
    viewer = ViewerGL(width=options.width, height=options.height, headless=True, num_frames=options.frames)
    example = example_class(viewer, args)
    frame_buffer = wp.empty((options.height, options.width, 3), dtype=wp.uint8, device=viewer.device)
    encoder = None
    log = None
    if options.video:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to export video")
        log = (options.output_dir / "encoder.log").open("w")
        encoder = subprocess.Popen(
            [
                ffmpeg,
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{options.width}x{options.height}",
                "-framerate",
                "60",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-threads",
                "4",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(options.output_dir / "bread_cut.mp4"),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    snapshots = {}
    try:
        for frame in range(options.frames):
            example.step()
            example.test_post_step()
            if encoder is not None or frame in stills:
                example.render()
                viewer.renderer.gl.glFinish()
                picture = viewer.get_frame(output=frame_buffer).numpy()
                if encoder is not None:
                    encoder.stdin.write(picture.tobytes())
                if frame in stills:
                    Image.fromarray(picture).save(options.output_dir / f"frame_{frame:04d}.png")
                    snapshots[f"q_{frame}"] = example.state_0.particle_q.numpy()
                    snapshots[f"stress_{frame}"] = example.state_0.mpm.particle_stress.numpy()
        example.solver.check_capacity_status()
        if options.frames >= required_frames:
            example.test_final()
        snapshots["connection_pairs"] = example.cohesion.pairs.numpy()
        snapshots["connection_active"] = example.cohesion.active.numpy()
        snapshots["connection_rest_lengths"] = example.cohesion.lengths.numpy()
        snapshots["particle_damage"] = example.damage.numpy()
        np.savez_compressed(options.output_dir / "states.npz", rest=example.rest, **snapshots)
        (options.output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "columns": getattr(
                        example,
                        "metric_columns",
                        [
                            "time_s",
                            "height_m",
                            "opening_m",
                            "knife_z_m",
                            "cut_surface_z_m",
                            "elastic_gradient_norm",
                        ],
                    ),
                    "samples": example.metrics,
                    "validation": getattr(example, "validation", None),
                },
                indent=2,
            )
        )
    finally:
        if encoder is not None:
            encoder.stdin.close()
            result = encoder.wait(timeout=60)
            log.close()
            if result:
                raise RuntimeError(f"Video encoding failed ({result})")
        viewer.close()


if __name__ == "__main__":
    main()

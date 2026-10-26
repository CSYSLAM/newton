# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Export full MJMPM GL simulations with measured frame-rate overlays.

Examples:
    uv run --extra examples scripts/record_mjmpm_fps_video.py --scene bread --output bread.mp4
    uv run --extra examples scripts/record_mjmpm_fps_video.py --scene water --output water.mp4
"""

import argparse
import json
import shutil
import subprocess
import time
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np
import warp as wp

from newton.examples.mjmpm.example_mjmpm_teapot_fill import Example as WaterExample
from newton.examples.mjmpm.example_mjmpm_w1_bread_three_pieces import Example as BreadExample
from newton.viewer import ViewerGL

SCENES = {
    "bread": (BreadExample, 1200, "W1 bread | two cuts, three pieces"),
    "water": (WaterExample, 900, "W1 teapot | MPM water pour"),
}
SIM_FPS = 60
WARMUP_FRAMES = 60


@lru_cache(maxsize=3)
def _font(size):
    from PIL import ImageFont

    font_path = Path("C:/Windows/Fonts/consolab.ttf")
    return ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default()


def _overlay(picture, title, frame, durations, width, height):
    from PIL import Image, ImageDraw

    image = Image.fromarray(picture)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 18, min(width - 18, 705), 143), radius=12, fill=(10, 16, 26))
    draw.text((32, 27), title, font=_font(27), fill=(255, 255, 255))
    if durations:
        fps = len(durations) / sum(durations)
        color = (92, 225, 135) if fps >= 60 else (255, 207, 82) if fps >= 25 else (255, 105, 95)
        label = f"SIM + GL: {fps:5.1f} FPS   ({fps / SIM_FPS:.2f}x realtime)"
    else:
        label = "SIM + GL: warming up"
        color = (180, 205, 230)
    draw.text((32, 65), label, font=_font(25), fill=color)
    draw.text(
        (32, 106),
        f"t = {(frame + 1) / SIM_FPS:5.2f}s  |  frame {frame + 1}  |  60 Hz playback",
        font=_font(19),
        fill=(222, 230, 240),
    )
    draw.rounded_rectangle((18, height - 47, min(width - 18, 775), height - 12), radius=8, fill=(10, 16, 26))
    draw.text(
        (30, height - 42),
        "FPS measures simulation + GL; video encoding excluded",
        font=_font(18),
        fill=(222, 230, 240),
    )
    return image.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    options = parser.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to export MP4 video")
    options.output.parent.mkdir(parents=True, exist_ok=True)
    example_class, frame_count, title = SCENES[options.scene]
    wp.set_device("cuda:0")
    args = example_class.create_parser().parse_args(["--viewer", "null"])
    args.test = False
    viewer = ViewerGL(width=options.width, height=options.height, headless=True, num_frames=frame_count)
    example = example_class(viewer, args)
    frame_buffer = wp.empty((options.height, options.width, 3), dtype=wp.uint8, device=viewer.device)
    log_path = options.output.with_suffix(".encoder.log")
    durations = deque(maxlen=60)
    measured = []
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
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
                str(SIM_FPS),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-threads",
                "4",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(options.output),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            for frame in range(frame_count):
                start = time.perf_counter()
                example.step()
                example.render()
                viewer.renderer.gl.glFinish()
                elapsed = time.perf_counter() - start
                if frame >= WARMUP_FRAMES:
                    durations.append(elapsed)
                    measured.append(elapsed)
                picture = viewer.get_frame(output=frame_buffer).numpy()
                encoder.stdin.write(_overlay(picture, title, frame, durations, options.width, options.height))
                if (frame + 1) % 60 == 0:
                    fps = len(durations) / sum(durations) if durations else 0.0
                    print(f"[{options.scene}] {frame + 1}/{frame_count}, SIM+GL {fps:.1f} FPS", flush=True)
            encoder.stdin.close()
            if encoder.wait(timeout=120):
                raise RuntimeError(f"ffmpeg failed; inspect {log_path}")
        finally:
            if encoder.poll() is None:
                encoder.kill()
                encoder.wait()
            viewer.close()

    if options.scene == "bread":
        _, sizes = example.cohesion.components()
        validation = {"piece_sizes": sorted(int(s) for s in sizes)}
        if len(sizes) != 3:
            raise AssertionError(f"Expected three bread pieces, got {validation}")
    else:
        bowl_count = example.water_in_bowl_count()
        particle_count = example.water_end - example.water_begin
        validation = {"water_in_bowl": bowl_count, "water_total": particle_count}
        if bowl_count < 0.85 * particle_count:
            raise AssertionError(f"Too little water reached the bowl: {validation}")

    report = {
        "scene": options.scene,
        "frame_count": frame_count,
        "simulation_seconds": frame_count / SIM_FPS,
        "video_playback_fps": SIM_FPS,
        "mean_sim_gl_fps_after_warmup": len(measured) / sum(measured),
        "median_sim_gl_fps_after_warmup": float(np.median(1.0 / np.asarray(measured))),
        "recording_wall_seconds": time.perf_counter() - started,
        "validation": validation,
    }
    options.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

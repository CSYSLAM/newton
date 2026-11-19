"""Render both independently validated pairs and concatenate the pharmacy comparison."""

import argparse
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--encode-only", action="store_true")
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--pair", choices=("all", "1", "2"), default="all")
    args = p.parse_args()
    folder = args.output.resolve()
    pairs = (1, 2) if args.pair == "all" else (int(args.pair),)
    if args.pair == "all":
        subprocess.run([sys.executable, str(HERE / "validate_scene.py"), str(folder)], check=True)
    videos = []
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for pair in pairs:
        target = folder / f"pair_{pair}"
        if args.pair != "all":
            subprocess.run([sys.executable, str(HERE / "validate_scene.py"), str(target)], check=True)
        subprocess.run([sys.executable, str(HERE.parent / "w1_shopping_bag/robot_quality.py"), str(target)], check=True)
        with np.load(target / "trajectory.npz") as d:
            count = len(d["particles"]) - 1
        if not args.encode_only:
            priority = [f for f in (90, 183, 330) if f < count]
            frames = priority + [f for f in range(count) if f not in priority]
            subprocess.run(
                [
                    "D:/software/blender/blender.exe",
                    "--background",
                    "--python-exit-code",
                    "1",
                    "--python",
                    str(HERE / "render_scene.py"),
                    "--",
                    "--output",
                    str(target),
                    "--samples",
                    str(args.samples),
                    "--frames",
                    ",".join(map(str, frames)),
                ],
                check=True,
            )
        video = target / f"pharmacy_pair_{pair}.mp4"
        videos.append(video)
        missing = [frame for frame in range(count) if not (target / "frames" / f"frame_{frame:04d}.png").is_file()]
        if missing:
            raise RuntimeError(f"Pair {pair}: missing {len(missing)} rendered frames, first={missing[0]}")
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-framerate",
                "30",
                "-i",
                str(target / "frames/frame_%04d.png"),
                "-frames:v",
                str(count),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video),
            ],
            check=True,
        )
    listing = folder / "video_sequence.txt"
    listing.write_text("".join("file '" + v.as_posix() + "'\n" for v in videos), encoding="utf-8")
    output = folder / "w1_pharmacy_carton_variants.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()

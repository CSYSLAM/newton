"""Validate a recorded W1 snack-basket run, render every frame in Blender Cycles and encode an H.264 video."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "build/w1-snack-basket")
    parser.add_argument("--blender", default=str(ROOT / "build/third_party/blender-5.2.0-linux-x64/blender"))
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--resolution", type=int, nargs=2, default=(1920, 1080))
    parser.add_argument("--encode-only", action="store_true")
    args = parser.parse_args()
    folder = args.output.resolve()
    subprocess.run([sys.executable, str(HERE / "validate.py"), str(folder)], check=True)
    if not args.encode_only:
        subprocess.run(
            [
                args.blender,
                "--background",
                "--python-exit-code",
                "1",
                "--python",
                str(HERE / "render.py"),
                "--",
                "--output",
                str(folder),
                "--frames",
                "all",
                "--samples",
                str(args.samples),
                "--resolution",
                *map(str, args.resolution),
            ],
            check=True,
        )
    frames = len(list((folder / "frames").glob("frame_*.png")))
    video = folder / "w1_snack_basket.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-framerate",
            "30",
            "-start_number",
            "0",
            "-i",
            str(folder / "frames/frame_%04d.png"),
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    count, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    if count != frames:
        raise SystemExit(f"Encoded {count} of {frames} frames")
    (folder / "video_verification.json").write_text(json.dumps(dict(frames=count, seconds=seconds), indent=2))
    print(video, flush=True)


if __name__ == "__main__":
    main()

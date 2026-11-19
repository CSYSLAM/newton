"""Render and verify the 15-second simultaneous W1 mailer demonstration."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE.parents[1] / "build/w1-courier")
    parser.add_argument("--blender", default="D:/software/blender/blender.exe")
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--hd", action="store_true", help="Native 1920x1080 render with 64 samples.")
    args = parser.parse_args()
    folder = args.output.resolve()
    subprocess.run([sys.executable, str(HERE / "validate_w1.py"), str(folder)], check=True)
    if args.hd:
        subprocess.run([sys.executable, str(HERE / "check_settling.py"), str(folder)], check=True)
    if not args.encode_only:
        subprocess.run(
            [
                args.blender,
                "--background",
                "--python-exit-code",
                "1",
                "--python",
                str(HERE / "render_w1.py"),
                "--",
                "--output",
                str(folder),
                "--frames",
                ",".join(map(str, range(450))),
            ]
            + (["--hd"] if args.hd else []),
            check=True,
        )
    video = folder / ("w1_dual_courier_transport_1080p.mp4" if args.hd else "w1_dual_courier_transport.mp4")
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
            str(folder / "reference_frames/frame_%04d.png"),
            "-frames:v",
            "450",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "16" if args.hd else "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    subprocess.run([ffmpeg, "-v", "error", "-i", str(video), "-f", "null", "-"], check=True)
    count, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    assert count == 450 and abs(seconds - 15) < 0.05
    reader = imageio_ffmpeg.read_frames(str(video))
    metadata = next(reader)
    reader.close()
    if args.hd:
        assert metadata["size"] == (1920, 1080)
    (folder / "video_verification.json").write_text(
        json.dumps(dict(frames=count, seconds=seconds, resolution=metadata["size"], full_decode_passed=True), indent=2)
    )
    print(video, flush=True)


if __name__ == "__main__":
    main()

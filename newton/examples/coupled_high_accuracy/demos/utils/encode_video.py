"""Encode the preserved Cycles PNG sequence and verify the resulting H.264 video.

Paths are relative to --output-dir (default: ./outputs), unless absolute. Example:
    python demos/utils/encode_video.py --output-dir demos/my_scene/outputs --output video.mp4 --end 240
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import struct
import subprocess
import sys
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_provenance import sha256, verify_frame_provenance


def local_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def find_ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        executable = shutil.which("ffmpeg")
        if executable:
            return executable
    raise RuntimeError("Install imageio-ffmpeg (uv sync --extra examples) or put ffmpeg on PATH.")


def inspect_sequence(pattern: Path, start: int, end: int | None) -> tuple[list[Path], tuple[int, int]]:
    match = re.search(r"%0?(\d*)d", pattern.name)
    if match is None or len(re.findall(r"%0?\d*d", pattern.name)) != 1:
        raise ValueError("--input must contain one frame number placeholder, such as frames/frame_%04d.png.")
    width = int(match.group(1) or 0)
    digits = rf"(\d{{{width},}})" if width else r"(\d+)"
    expression = re.compile(re.escape(pattern.name[: match.start()]) + digits + re.escape(pattern.name[match.end() :]))
    indexed = {}
    for candidate in pattern.parent.glob(pattern.name[: match.start()] + "*" + pattern.name[match.end() :]):
        candidate_match = expression.fullmatch(candidate.name)
        if candidate_match:
            number = int(candidate_match.group(1))
            if candidate.name == pattern.name % number:
                indexed[number] = candidate
    if not indexed:
        raise FileNotFoundError(f"No PNG frames match {pattern}")
    last = max(indexed) if end is None else end
    if last < start:
        raise ValueError(f"The final frame ({last}) is before --start ({start}).")
    missing = [number for number in range(start, last + 1) if number not in indexed]
    if missing:
        raise ValueError(f"The PNG sequence has missing frames: {missing[:20]}")
    paths = [indexed[number] for number in range(start, last + 1)]
    dimensions = set()
    for path in paths:
        with path.open("rb") as source:
            header = source.read(24)
        if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            raise ValueError(f"Invalid PNG header: {path}")
        dimensions.add(struct.unpack(">II", header[16:24]))
    if len(dimensions) != 1:
        raise ValueError(f"Frame dimensions are inconsistent: {sorted(dimensions)}")
    size = dimensions.pop()
    if any(dimension <= 0 or dimension % 2 for dimension in size):
        raise ValueError(f"H.264 yuv420p requires positive, even dimensions; found {size}.")
    return paths, size


def make_contact_sheet(paths: list[Path], output: Path, start: int) -> None:
    from PIL import Image, ImageDraw

    thumb_width, thumb_height = 640, 360
    label_height = 30
    sheet = Image.new("RGB", (3 * thumb_width, 2 * (thumb_height + label_height)), "#10151e")
    draw = ImageDraw.Draw(sheet)
    for slot in range(6):
        index = round(slot * (len(paths) - 1) / 5)
        with Image.open(paths[index]) as source:
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = (slot % 3) * thumb_width
        y = (slot // 3) * (thumb_height + label_height)
        sheet.paste(thumbnail, (x + (thumb_width - thumbnail.width) // 2, y))
        draw.text((x + 12, y + thumb_height + 8), f"Frame {start + index:04d}", fill="#e7edf5")
    sheet.save(output)


def verify_video(ffmpeg: str, path: Path, size: tuple[int, int], fps: float, frames: int) -> dict:
    # A full decode detects truncated packets even when the container metadata looks valid.
    subprocess.run([ffmpeg, "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"], check=True)
    probe = shutil.which("ffprobe")
    if probe is None:
        sibling = Path(ffmpeg).with_name("ffprobe.exe" if Path(ffmpeg).suffix == ".exe" else "ffprobe")
        probe = str(sibling) if sibling.is_file() else None
    if probe is None:
        return {"full_decode": "passed", "ffprobe": "unavailable"}
    result = subprocess.run(
        [probe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    metadata = json.loads(result.stdout)
    stream = metadata["streams"][0]
    if (stream["width"], stream["height"]) != size:
        raise RuntimeError("Encoded dimensions differ from the PNG sequence.")
    if stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p":
        raise RuntimeError("Encoded video is not H.264 yuv420p.")
    if abs(float(Fraction(stream["avg_frame_rate"])) - fps) > 1e-3:
        raise RuntimeError("Encoded frame rate differs from the requested frame rate.")
    if int(stream.get("nb_frames", frames)) != frames:
        raise RuntimeError("Encoded frame count differs from the PNG sequence.")
    return {"full_decode": "passed", "ffprobe": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Base directory for relative input/output paths",
    )
    parser.add_argument("--input", default="frames/frame_%04d.png", help="PNG pattern, relative to --output-dir")
    parser.add_argument("--output", required=True, help="MP4 path, relative to --output-dir")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback frame rate")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--end", type=int, help="Inclusive final frame; otherwise include the entire contiguous sequence."
    )
    parser.add_argument("--crf", type=int, default=17, help="H.264 quality, 0 (lossless) to 51 (lowest).")
    parser.add_argument("--preset", default="slow", choices=("medium", "slow", "slower", "veryslow"))
    parser.add_argument(
        "--contact-sheet", action="store_true", help="Save six sampled frames beside the video (requires Pillow)."
    )
    args = parser.parse_args()
    if args.fps <= 0 or not 0 <= args.crf <= 51 or args.start < 0:
        parser.error("--fps must be positive, --start nonnegative, and --crf between 0 and 51.")
    base = args.output_dir.resolve()
    pattern, output = local_path(args.input, base), local_path(args.output, base)
    if output.suffix.lower() != ".mp4":
        parser.error("--output must use the .mp4 extension.")
    paths, size = inspect_sequence(pattern, args.start, args.end)
    provenance = verify_frame_provenance(paths, base, args.start, args.fps)
    ffmpeg = find_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".encoding.mp4")
    command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-framerate",
        str(args.fps),
        "-start_number",
        str(args.start),
        "-i",
        str(pattern),
        "-frames:v",
        str(len(paths)),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        args.preset,
        "-crf",
        str(args.crf),
        "-vf",
        "scale=in_range=pc:out_range=tv:out_color_matrix=bt709",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    print(f"Encoding {len(paths)} frames at {size[0]}x{size[1]}, {args.fps:g} fps into {output}", flush=True)
    subprocess.run(command, check=True)
    verification = verify_video(ffmpeg, temporary, size, args.fps, len(paths))
    # Keep a previous successful deliverable intact until the new movie has decoded successfully.
    temporary.replace(output)
    if isinstance(verification["ffprobe"], dict):
        verification["ffprobe"]["format"]["filename"] = args.output
    if args.contact_sheet:
        make_contact_sheet(paths, output.with_name("contact_sheet.jpg"), args.start)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "output_dir": str(base),
        "input": args.input,
        "output": args.output,
        "first_frame": args.start,
        "last_frame": args.start + len(paths) - 1,
        "frame_count": len(paths),
        "width": size[0],
        "height": size[1],
        "fps": args.fps,
        "duration_seconds": len(paths) / args.fps,
        "codec": "h264",
        "pixel_format": "yuv420p",
        "color_space": "bt709",
        "crf": args.crf,
        "preset": args.preset,
        "verification": verification,
        **provenance,
        "output_sha256": sha256(output),
    }
    output.with_suffix(".video.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Video verified: {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)", flush=True)


if __name__ == "__main__":
    main()

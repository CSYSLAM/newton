"""Export one Piper bag transport simulation and produce three videos with verified shared provenance."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
UTILS = DIRECTORY.parent / "utils"
ALL_STEPS = ("export", "validate", "telemetry", "build", "check", "render", "encode", "verify")


def find_blender(explicit):
    if explicit:
        return str(explicit)
    executable = shutil.which("blender")
    if executable:
        return executable
    for base in (os.environ.get("ProgramFiles", "C:/Program Files"), "D:/Program Files"):
        for candidate in sorted((Path(base) / "Blender Foundation").glob("Blender */blender.exe"), reverse=True):
            if candidate.is_file():
                return str(candidate)
    raise FileNotFoundError("Blender not found. Pass --blender with the Blender executable path.")


def commands(args):
    config = args.config.resolve()
    cfg = json.loads(config.read_text(encoding="utf-8"))
    usd = (config.parent / cfg["usd"]).resolve()
    project = (config.parent / cfg["project"]).resolve()
    if args.output_dir:
        usd = args.output_dir.resolve() / usd.name
        project = args.output_dir.resolve() / project.name
    if usd.parent != project.parent:
        raise ValueError("Keep the USD and Blender project in the same output directory for a portable bundle")
    root = usd.parent
    metadata = usd.with_name(usd.stem + "_metadata.json")
    steps = ALL_STEPS if args.steps == ["all"] else tuple(args.steps)
    if "all" in steps:
        raise ValueError("Use --steps all on its own")
    if "export" in steps:
        last = args.num_frames
    elif metadata.is_file():
        last = json.loads(metadata.read_text(encoding="utf-8"))["simulation"]["frames"]
    else:
        last = args.num_frames
    start = cfg.get("frame_start", 0)
    end = last if cfg.get("frame_end") is None else cfg["frame_end"]
    if start < 0 or (end is not None and start > end) or (last is not None and end > last):
        raise ValueError("Configured rendering range is outside the exported simulation")
    if last is None and set(steps) & {"preview", "encode"}:
        raise ValueError("Export metadata is required to determine the preview and encoding frame range")
    blender = find_blender(args.blender) if set(steps) & {"build", "check", "preview", "render"} else None
    python = [sys.executable, "-u"]

    def blend(script, *options, load=True):
        return [
            blender,
            "--background",
            *([str(project)] if load else []),
            "--python-exit-code",
            "1",
            "--python",
            str(script),
            "--",
            *map(str, options),
        ]

    render_options = ["--animation"]
    if args.draft:
        render_options += ["--preview", "--config", str(config), "--output-dir", str(root / "draft_frames")]
    # A newly exported/rebuilt scene must not silently reuse images from a previous simulation or material setup.
    if args.overwrite or "export" in steps or "build" in steps:
        render_options.append("--overwrite")
    if args.device:
        render_options += ["--device", args.device]
    preview_frames = sorted({round(start + (end - start) * i / 5) for i in range(6)}) if end is not None else []
    available = {
        "export": python
        + [
            str(DIRECTORY / "export_simulation.py"),
            "--output",
            str(usd),
            *(["--num-frames", str(args.num_frames)] if args.num_frames is not None else []),
            *args.physics_args,
        ],
        "validate": python + [str(DIRECTORY / "validate_simulation.py"), "--metadata", str(metadata)],
        "telemetry": python
        + [
            str(DIRECTORY / "plot_motor_telemetry.py"),
            "--csv",
            str(root / "motor_telemetry.csv"),
            "--output-dir",
            str(root),
            "--fps",
            str(cfg["video_fps"]),
        ],
        "build": blend(
            DIRECTORY / "build_render_scene.py",
            "--config",
            config,
            "--usd",
            usd,
            "--project",
            project,
            *(["--device", args.device] if args.device else []),
            load=False,
        ),
        "check": blend(UTILS / "validate_import.py", "--usd", usd),
        "preview": blend(
            UTILS / "render_frames.py",
            "--frames",
            ",".join(map(str, preview_frames)),
            "--preview",
            "--config",
            config,
            "--overwrite",
            *(["--device", args.device] if args.device else []),
        ),
        "render": blend(UTILS / "render_frames.py", *render_options),
        "encode": python
        + [
            str(UTILS / "encode_video.py"),
            "--output-dir",
            str(root),
            "--input",
            "draft_frames/frame_%04d.png" if args.draft else "frames/frame_%04d.png",
            "--output",
            "piper_bag_transport_preview.mp4" if args.draft else "piper_bag_transport.mp4",
            "--start",
            str(start),
            "--end",
            str(end),
            "--fps",
            str(cfg["video_fps"]),
            "--crf",
            str(cfg.get("video_crf", 16)),
            "--preset",
            cfg.get("video_preset", "slow"),
            "--contact-sheet",
        ],
        "verify": python
        + [
            str(DIRECTORY / "verify_run.py"),
            "--output-dir",
            str(root),
            "--scene-video",
            "piper_bag_transport_preview.mp4" if args.draft else "piper_bag_transport.mp4",
        ],
    }
    return [(step, available[step]) for step in steps]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", choices=("all", *ALL_STEPS, "preview"), default=["all"])
    parser.add_argument("--blender", type=Path)
    parser.add_argument("--config", type=Path, default=DIRECTORY / "render_config.json")
    parser.add_argument("--output-dir", type=Path, help="Override the generated artifact directory")
    parser.add_argument(
        "--num-frames", type=int, help="Physics frames to export; defaults to the complete demo sequence"
    )
    parser.add_argument(
        "--draft", action="store_true", help="Render the full sequence with preview quality in draft_frames/"
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing requested PNGs instead of resuming")
    parser.add_argument("--device", choices=("CPU", "OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "physics_args",
        nargs=argparse.REMAINDER,
        help="Demo options after --; the validated sequence retains four nonlinear iterations",
    )
    args = parser.parse_args()
    if args.num_frames is not None and args.num_frames < 1:
        parser.error("--num-frames must be positive")
    if args.physics_args[:1] == ["--"]:
        args.physics_args = args.physics_args[1:]
    steps = ALL_STEPS if args.steps == ["all"] else tuple(args.steps)
    if "all" in steps:
        parser.error("Use --steps all on its own")
    overwrite = args.overwrite or "export" in steps or "build" in steps
    for index, step in enumerate(steps):
        # Resolve later ranges after export writes its actual duration, without importing the GPU demo here.
        step_args = argparse.Namespace(**(vars(args) | {"steps": [step], "overwrite": overwrite}))
        for _, command in commands(step_args):
            print(f"[{step}] {subprocess.list2cmdline(command)}", flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
        if args.dry_run and step == "export" and args.num_frames is None and index + 1 < len(steps):
            remaining = ", ".join(steps[index + 1 :])
            print(f"After export, read its metadata frame range for: {remaining}", flush=True)
            break


if __name__ == "__main__":
    main()

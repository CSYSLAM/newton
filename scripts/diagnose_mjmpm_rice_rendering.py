# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Record the live rice pour or compare display costs without changing physics.

Examples:
    uv run --no-sync python scripts/diagnose_mjmpm_rice_rendering.py video --output-dir output/rice
    uv run --no-sync python scripts/diagnose_mjmpm_rice_rendering.py none --output-dir output/rice
    uv run --no-sync python scripts/diagnose_mjmpm_rice_rendering.py full --output-dir output/rice

Use separate processes and the same resolution/frame count for each comparison.
Video encoding and framebuffer readback are excluded from benchmark cases.
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton.examples.mjmpm.example_mjmpm_w1_rice_cooker import Example
from newton.viewer import ViewerGL, ViewerNull


def main():
    """Run one complete trajectory with synchronized step and display timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case", choices=("video", "none", "simple", "full", "untextured", "opaque", "no-shadows", "validation")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=990)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--timing", choices=("synchronized", "native"), default="synchronized")
    parser.add_argument(
        "--visible", action="store_true", help="Measure the actual desktop window instead of offscreen rendering"
    )
    opts = parser.parse_args()
    if opts.frames <= opts.warmup:
        parser.error("--frames must exceed --warmup")
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    wp.set_device("cuda:0")
    test = opts.case in ("video", "validation")
    style = "collision" if opts.case == "simple" else "realistic"
    args = Example.create_parser().parse_args(["--viewer", "null", "--visual-style", style])
    args.test = test
    viewer = (
        ViewerNull(num_frames=opts.frames)
        if opts.case == "none"
        else ViewerGL(width=opts.width, height=opts.height, headless=not opts.visible, num_frames=opts.frames)
    )
    removed = []
    if opts.case == "untextured":
        original_set_model = viewer.set_model

        def set_untextured_model(model):
            for source in {id(s): s for s in model.shape_source if s is not None}.values():
                if getattr(source, "texture", None) is not None:
                    removed.append(type(source).__name__)
                    source.texture = None
            original_set_model(model)

        viewer.set_model = set_untextured_model

    example = Example(viewer, args)
    source_texture_count = sum(
        getattr(s, "texture", None) is not None
        for s in {id(s): s for s in example.model.shape_source if s is not None}.values()
    )
    if opts.case == "opaque":
        example.model.shape_opacity.fill_(1.0)
        for _name, _parent, opacity, _color, _material in example.visuals.parts:
            opacity.fill_(1.0)
    if opts.case == "no-shadows":
        viewer.renderer.draw_shadows = False

    encoder = None
    encoder_log = None
    video_path = opts.output_dir / "rice_pour_newton_1080p60.mp4"
    if opts.case == "video":
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("Video export needs ffmpeg on PATH")
        encoder_log = (opts.output_dir / "video_encoder.log").open("w", encoding="utf-8")
        encoder = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{opts.width}x{opts.height}",
                "-framerate",
                f"{1 / example.frame_dt:g}",
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
                str(video_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=encoder_log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    samples = []
    framebuffer_history = []
    started = None
    texture_inventory = []
    try:
        for frame in range(opts.frames):
            if opts.timing == "synchronized" or frame == opts.warmup:
                wp.synchronize_device(example.model.device)
            if frame == opts.warmup:
                started = time.perf_counter()
            begin = time.perf_counter()
            example.step()
            if opts.timing == "synchronized":
                wp.synchronize_device(example.model.device)
            stepped = time.perf_counter()
            if test:
                example.test_post_step()
            checked = time.perf_counter()
            if opts.case != "none":
                example.render()
                if opts.timing == "synchronized":
                    viewer.renderer.gl.glFinish()
                    wp.synchronize_device(example.model.device)
                resolution = list(viewer.renderer.window.get_framebuffer_size())
                if not framebuffer_history or framebuffer_history[-1]["resolution"] != resolution:
                    framebuffer_history.append({"frame": frame, "resolution": resolution})
            rendered = time.perf_counter()
            if frame == 0 and opts.case != "none":
                texture_inventory = [
                    name for name, obj in viewer.objects.items() if getattr(obj, "texture_id", None) is not None
                ]
            if encoder is not None:
                picture = viewer.get_frame().numpy()
                encoder.stdin.write(picture.tobytes())
                if frame in (0, 360, 510, 989):
                    from PIL import Image

                    Image.fromarray(picture).save(opts.output_dir / f"video_frame_{frame:04d}.png")
            finished = time.perf_counter()
            if frame >= opts.warmup:
                samples.append((stepped - begin, checked - stepped, rendered - checked, finished - rendered))
            if frame % 120 == 0:
                print(f"[{opts.case}] {frame + 1}/{opts.frames}, simulation {example.sim_time:.2f} s", flush=True)
        if opts.case != "none":
            viewer.renderer.gl.glFinish()
        wp.synchronize_device(example.model.device)
        elapsed = time.perf_counter() - started
        example.solver.check_capacity_status()
        example.check_finite()
        if test:
            example.test_final()
        values = np.asarray(samples) * 1000
        phase_metrics = {}
        sampled_frames = np.arange(opts.warmup, opts.frames)
        for name, first, last in (
            ("grasp_to_pour", 180, 600),
            ("lift", 258, 348),
            ("carry", 348, 420),
            ("pour", 420, 600),
        ):
            selected = values[(sampled_frames >= first) & (sampled_frames < last)]
            if selected.size:
                total = selected.sum(axis=1)
                phase_metrics[name] = {
                    "fps": float(1000 / np.mean(total)),
                    "step_ms": float(np.mean(selected[:, 0])),
                    "render_ms": float(np.mean(selected[:, 2])),
                    "frame_p50_ms": float(np.median(total)),
                    "frame_p90_ms": float(np.percentile(total, 90)),
                }
        inventory = []
        if opts.case != "none":
            for name, obj in viewer.objects.items():
                mesh = getattr(obj, "mesh", None)
                count = getattr(obj, "active_instances", 0)
                if mesh is not None and count and not obj.hidden:
                    inventory.append(
                        {"name": name, "instances": int(count), "triangles": int(mesh.num_indices // 3) * int(count)}
                    )
        report = {
            "phases": phase_metrics,
            "frame_timings_ms": values.tolist(),
            "case": opts.case,
            "frames": opts.frames,
            "warmup": opts.warmup,
            "resolution": [opts.width, opts.height],
            "framebuffer_resolution": list(viewer.renderer.window.get_framebuffer_size())
            if opts.case != "none"
            else None,
            "framebuffer_history": framebuffer_history,
            "timing": opts.timing,
            "visible": opts.visible,
            "fps": len(samples) / elapsed,
            "step_ms": float(values[:, 0].mean()),
            "validation_ms": float(values[:, 1].mean()),
            "render_ms": float(values[:, 2].mean()),
            "readback_encode_ms": float(values[:, 3].mean()),
            "frame_ms_p50": float(np.median(values.sum(axis=1))),
            "frame_ms_p90": float(np.percentile(values.sum(axis=1), 90)),
            "source_textured_meshes": source_texture_count,
            "removed_textures": len(removed),
            "uploaded_textures": texture_inventory,
            "visible_triangle_instances": inventory,
            "rice_count": example.rice_end - example.rice_begin,
            "substeps": args.substeps,
            "rigid_iterations": args.rigid_iterations,
            "mpm_iterations": args.iterations,
            "test_checks": test,
            "cuda_graph": example.graph is not None,
            "rice_locations": example.rice_locations(),
            "video": str(video_path) if encoder is not None else None,
        }
        (opts.output_dir / f"{opts.case}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "frame_timings_ms"}), flush=True)
    finally:
        if encoder is not None:
            encoder.stdin.close()
            returncode = encoder.wait()
            encoder_log.close()
            if returncode:
                raise RuntimeError(f"ffmpeg failed with status {returncode}; see video_encoder.log")
        viewer.close()


if __name__ == "__main__":
    main()

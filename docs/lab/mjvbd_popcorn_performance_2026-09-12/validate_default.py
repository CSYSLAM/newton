# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Time the ordinary popcorn defaults, including 1080p OpenGL rendering."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton.examples.mjvbdv2.example_mjvbd_v2_popcorn import Example
from newton.viewer import ViewerGL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=2850)
    parser.add_argument("--screenshots", type=Path)
    parser.add_argument("--local-iterations", type=int)
    parser.add_argument("--linear-iterations", type=int)
    parser.add_argument("--max-radius-fraction", type=float)
    args = parser.parse_args()
    viewer = ViewerGL(width=1920, height=1080, headless=True, vsync=False)
    try:
        overrides = [] if args.local_iterations is None else ["--vbd-iterations", str(args.local_iterations)]
        example = Example(viewer, Example.create_parser().parse_args(overrides))
        if args.linear_iterations is not None:
            example.solver.vbd_solver.particle_multilevel.coarse_iterations = args.linear_iterations
        if args.max_radius_fraction is not None:
            example.solver.vbd_solver.particle_multilevel.max_radius_fraction = args.max_radius_fraction
        for _ in range(30):
            try:
                example.step()
            except (AssertionError, RuntimeError):
                print(
                    json.dumps(
                        {
                            "failed_at": example.sim_time,
                            "forces": example.grasp_force_filtered.tolist(),
                            "offsets_deg": np.degrees(example.grasp_joint_offset).tolist(),
                        }
                    ),
                    flush=True,
                )
                raise
            example.render()
        from pyglet import gl

        gl.glFinish()
        # Same state, same camera: audit against the uncached public renderer.
        fast_pixels = viewer.get_frame().numpy()
        example._persistent_render = False
        viewer.cache_static_appearance = False
        for upload in viewer._shared_mesh_uploads.values():
            upload.bind_buffers(False)
        example.render()
        reference_pixels = viewer.get_frame().numpy()
        difference = np.abs(fast_pixels.astype(np.int16) - reference_pixels.astype(np.int16))
        print(
            json.dumps(
                {
                    "render_pixel_max_difference": int(difference.max()),
                    "render_changed_channels": int(np.count_nonzero(difference)),
                }
            ),
            flush=True,
        )
        if difference.max() > 1 or np.count_nonzero(difference) > max(1, difference.size // 100000):
            raise AssertionError("Cached renderer differs from the same-state uncached reference")
        example._persistent_render = True
        viewer.cache_static_appearance = True
        example.render()
        gl.glFinish()
        start = time.perf_counter()
        for frame in range(args.frames):
            example.step()
            example.render()
            if args.screenshots and int(round(example.sim_time * 60)) in (900, 1260, 2700):
                from PIL import Image

                args.screenshots.mkdir(parents=True, exist_ok=True)
                Image.fromarray(viewer.get_frame().numpy()).save(args.screenshots / f"frame-{example.sim_time:.1f}.png")
            if (frame + 1) % 600 == 0:
                print(
                    json.dumps(
                        {
                            "time": example.sim_time,
                            "wall_ms": 1000 * (time.perf_counter() - start) / (frame + 1),
                            "cup_lift": example.current_cup_lift,
                            "delivered": example.delivered_inside,
                        }
                    ),
                    flush=True,
                )
        wp.synchronize_device(example.model.device)
        gl.glFinish()
        ms = 1000 * (time.perf_counter() - start) / args.frames
        print(json.dumps({"frames": args.frames, "wall_ms_per_frame": ms, "fps": 1000 / ms}), flush=True)
        example.test_final()
    except (AssertionError, RuntimeError):
        if "example" in locals():
            print(
                json.dumps(
                    {
                        "failed_at": example.sim_time,
                        "forces": example.grasp_force_filtered.tolist(),
                        "offsets_deg": np.degrees(example.grasp_joint_offset).tolist(),
                    }
                ),
                flush=True,
            )
        raise
    finally:
        viewer.close()


if __name__ == "__main__":
    main()

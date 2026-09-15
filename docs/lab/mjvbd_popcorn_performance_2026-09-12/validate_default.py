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
    parser.add_argument("--substeps", type=int)
    parser.add_argument("--popcorn-count", type=int)
    parser.add_argument("--linear-iterations", type=int)
    parser.add_argument("--max-radius-fraction", type=float)
    parser.add_argument(
        "--cup-offset-mm", type=float, default=0.0, help="Perturb initial cup x position for grasp tests"
    )
    args = parser.parse_args()
    viewer = ViewerGL(width=1920, height=1080, headless=True, vsync=False)
    try:
        overrides = [] if args.local_iterations is None else ["--vbd-iterations", str(args.local_iterations)]
        if args.substeps is not None:
            overrides.extend(["--substeps", str(args.substeps)])
        if args.popcorn_count is not None:
            overrides.extend(["--popcorn-count", str(args.popcorn_count)])
        example = Example(viewer, Example.create_parser().parse_args(overrides))
        if not np.isfinite(args.cup_offset_mm):
            raise ValueError("Cup placement perturbation must be finite")
        if args.cup_offset_mm:
            positions = example.state_0.particle_q.numpy().copy()
            positions[:, 0] += np.float32(args.cup_offset_mm * 0.001)
            example.state_0.particle_q.assign(positions)
            example.state_1.particle_q.assign(positions)
        masses = example.model.body_mass.numpy()[example.popcorn_bodies]
        if len(masses) != example.args.popcorn_count or not np.all(masses > 0):
            raise AssertionError("Every authored popcorn grain must remain a dynamic massive body")
        print(
            json.dumps(
                {
                    "dynamic_grains": len(masses),
                    "substeps": example.args.substeps,
                    "local_iterations": example.args.vbd_iterations,
                    "cup_offset_mm": args.cup_offset_mm,
                }
            ),
            flush=True,
        )
        if args.linear_iterations is not None:
            if args.linear_iterations < 1:
                raise ValueError("Linear iterations must be positive")
            correction = example.solver.vbd_solver.particle_multilevel
            correction.coarse_iterations = args.linear_iterations
            # Diagnostics store one residual per iteration. Resize before any
            # simulation graph captures this buffer, not just the loop budget.
            correction.runtime_metrics = wp.zeros(5 + args.linear_iterations, device=example.model.device)
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
        from pyglet import gl  # noqa: PLC0415 -- initialize after the viewer creates its GL context

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
                from PIL import Image  # noqa: PLC0415 -- optional screenshot dependency

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
                            "grains_in_machine": example.grains_in_machine,
                            "grain_max_speed_m_s": example.max_grain_speed,
                            "cup_slip_mm": 1000 * getattr(example, "cup_grip_slip", 0.0),
                            "cup_deformation_mm": 1000 * example.max_cup_deformation,
                            "rim_min_radius_mm": 1000 * example.rim_min_radius,
                            "surface_fallback": example.solver.vbd_solver._cuda_surface.fallback.numpy().tolist(),
                            "max_surface_adjacency": int(example.solver.vbd_solver._cuda_surface.counts.numpy().max()),
                        }
                    ),
                    flush=True,
                )
        wp.synchronize_device(example.model.device)
        gl.glFinish()
        ms = 1000 * (time.perf_counter() - start) / args.frames
        print(json.dumps({"frames": args.frames, "wall_ms_per_frame": ms, "fps": 1000 / ms}), flush=True)
        example.test_final()
        print(json.dumps({"validation_passed": True}), flush=True)
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

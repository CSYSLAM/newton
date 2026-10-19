# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run and record the dynamic W1 MJMPM squeeze/grasp acceptance scene."""

import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.mjmpm.example_mjmpm_w1_squeeze_grasp import PARTICLE_RADIUS, TABLE_Z, Example
from newton.viewer import ViewerGL, ViewerNull


def main():
    parser = Example.create_parser()
    parser.add_argument("--output", type=Path, default=Path("docs/lab/mjmpm/w1_grasp.json"))
    parser.add_argument("--screenshots", action="store_true")
    args = parser.parse_args()
    wp.config.log_level = wp.LOG_WARNING
    wp.set_device(args.device or wp.get_preferred_device())
    viewer = ViewerGL(width=960, height=720, headless=True, num_frames=1) if args.screenshots else ViewerNull()
    start = time.perf_counter()
    example = Example(viewer, args)
    setup_seconds = time.perf_counter() - start
    report = {
        "coupling": args.coupling,
        "substeps": args.substeps,
        "iterations": args.iterations,
        "mpm_solver": args.mpm_solver,
        "strain_basis": args.strain_basis,
        "proxy_mass_scale": args.proxy_mass_scale,
        "voxel_size": args.voxel_size,
        "device": str(example.model.device),
        "cuda_graph": not args.no_cuda_graph and example.model.device.is_cuda,
        "setup_seconds": setup_seconds,
        "mujoco_bodies": len(example.solver.ownership.mujoco_bodies),
        "mpm_bodies": len(example.solver.ownership.mpm_bodies),
        "mpm_particles": example.model.particle_count,
        "robot_kinematic_body_count": int(
            np.count_nonzero(example.model.body_flags.numpy()[: example.robot_bodies] & int(newton.BodyFlags.KINEMATIC))
        ),
        "rigid_contact_hard": example.solver.environment.rigid_contact_hard,
        "samples": [],
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_start = time.perf_counter()
    try:
        for frame in range(args.num_frames):
            example.step()
            example.test_post_step()
            if frame == 0 or example.frame % 30 == 0:
                bottoms, particles = example.object_heights()
                cells = np.floor(particles / args.voxel_size).astype(np.int32)
                _, occupancy = np.unique(cells, axis=0, return_counts=True)
                row = {
                    "time": example.sim_time,
                    "finger_q": example.finger_q().tolist(),
                    "rigid_pose": example.state_0.body_q.numpy()[example.block].tolist(),
                    "rigid_velocity": example.state_0.body_qd.numpy()[example.block].tolist(),
                    "object_lift": (bottoms - TABLE_Z).tolist(),
                    "soft_width": float(np.ptp(particles[:, 1])) + 2 * PARTICLE_RADIUS,
                    "soft_height": float(np.ptp(particles[:, 2])) + 2 * PARTICLE_RADIUS,
                    "soft_center": particles.mean(axis=0).tolist(),
                    "max_particles_per_cell": int(occupancy.max()),
                    "max_particle_speed": float(np.linalg.norm(example.state_0.particle_qd.numpy(), axis=1).max()),
                }
                report["samples"].append(row)
                print(json.dumps(row), flush=True)
            if args.screenshots and example.frame in (1, 168, 240, 390, 720):
                from PIL import Image, ImageOps

                example.render()
                screenshot = Image.fromarray(viewer.get_frame().numpy())
                screenshot.save(args.output.parent / f"w1_grasp_{example.frame:04d}.jpg", quality=92)
                if example.frame == 390:
                    ImageOps.fit(screenshot, (320, 320), method=Image.Resampling.LANCZOS).save(
                        "docs/images/examples/example_mjmpm_w1_squeeze_grasp.jpg", quality=92
                    )
        example.test_final()
        report["passed"] = True
    except Exception as error:
        report["failure"] = str(error)
        raise
    finally:
        report["frames"] = example.frame
        report["simulation_seconds"] = example.sim_time
        report["run_seconds"] = time.perf_counter() - run_start
        report["peak_lift"] = example.peak_lift.tolist()
        report["peak_free_tcp_error"] = example.peak_free_tcp_error
        report["steady_finger_peak_to_peak"] = (
            np.ptp(np.asarray(example.steady_fingers), axis=0).tolist() if example.steady_fingers else []
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        viewer.close()
        print(args.output, flush=True)


if __name__ == "__main__":
    main()

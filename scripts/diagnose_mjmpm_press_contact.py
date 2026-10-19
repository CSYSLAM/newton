# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure visible surface clearance and contact impulses throughout the MJMPM press cycle."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from newton.examples.mpm.example_mjmpm_press import Example
from newton.viewer import ViewerGL, ViewerNull


def sample(example):
    positions = example.state.particle_q.numpy()
    elastic_strain = example.state.mpm.particle_elastic_strain.numpy()
    plastic_volume = example.state.mpm.particle_Jp.numpy()
    return {
        "time": example.time,
        "phase": example.phase,
        "target": example.target,
        "joint_q": float(example.state.joint_q.numpy()[0]),
        "press_z": float(example.state.body_q.numpy()[0, 2]),
        "surface_gap": example.surface_gap(),
        "material_height": float(np.ptp(positions[:, 2])),
        "material_z_range": [float(positions[:, 2].min()), float(positions[:, 2].max())],
        "mean_elastic_diagonal": np.diagonal(elastic_strain, axis1=1, axis2=2).mean(axis=0).tolist(),
        "Jp_range": [float(plastic_volume.min()), float(plastic_volume.max())],
        "press_impulse": example.solver.environment.material_impulses.numpy()[0].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=540)
    parser.add_argument("--coupling", choices=("one_way", "two_way"), default="two_way")
    parser.add_argument("--screenshots", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("docs/lab/mjmpm/contact_validation.json"))
    args = parser.parse_args()
    wp.config.log_level = wp.LOG_WARNING
    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    viewer = ViewerGL(width=640, height=640, headless=True, num_frames=1) if args.screenshots else ViewerNull()
    with wp.ScopedDevice(args.device):
        example = Example(viewer, args)
    samples = []
    first_contact = None
    max_impulse_with_gap = 0.0
    max_contact_gap = -float("inf")
    max_particles_per_cell = 0
    screenshots = {
        59: "material_rest_fixed.jpg",
        149: "material_approach_fixed.jpg",
        299: "material_pressed_fixed.jpg",
        539: "material_recovered_fixed.jpg",
    }
    try:
        for frame in range(args.num_frames):
            example.step()
            example.test_post_step()
            gap = example.surface_gap()
            impulse = float(np.linalg.norm(example.solver.environment.material_impulses.numpy()[0]))
            cells = np.floor(example.state.particle_q.numpy() / 0.075).astype(np.int32)
            _, counts = np.unique(cells, axis=0, return_counts=True)
            max_particles_per_cell = max(max_particles_per_cell, int(counts.max()))
            if gap > 0.002:
                max_impulse_with_gap = max(max_impulse_with_gap, impulse)
            if impulse > 1.0e-6:
                max_contact_gap = max(max_contact_gap, gap)
                if first_contact is None:
                    first_contact = frame
            if frame in (5, 59, 149, 179, 209, 239, 299, 479, 539) or frame == first_contact:
                row = sample(example)
                samples.append(row)
                print(json.dumps(row), flush=True)
            if args.screenshots and (frame in screenshots or frame == first_contact):
                from PIL import Image

                example.render()
                screenshot = Image.fromarray(viewer.get_frame().numpy())
                name = screenshots.get(frame, "material_first_contact_fixed.jpg")
                screenshot.save(output_dir / name, quality=92)
                if frame == 299:
                    screenshot.resize((320, 320), Image.Resampling.LANCZOS).save(
                        "docs/images/examples/example_mjmpm_press.jpg", quality=92
                    )
        example.test_final()
        assert max_particles_per_cell <= 27, "Particle contact capacity exceeded"
        if args.num_frames >= 300:
            assert first_contact is not None, "Press never contacted material"
        (output_dir / "press_cycle.json").write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
        args.output.write_text(
            json.dumps(
                {
                    "device": args.device,
                    "coupling": args.coupling,
                    "collider_basis": example.solver.environment.material.collider_basis,
                    "first_contact_frame": first_contact,
                    "max_impulse_with_gap_over_2mm": max_impulse_with_gap,
                    "max_surface_gap_with_impulse_over_1e-6": max_contact_gap if first_contact is not None else None,
                    "max_particles_per_cell": max_particles_per_cell,
                    "frames": args.num_frames,
                    "seconds": example.time,
                    "samples": samples,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(args.output, flush=True)
    finally:
        viewer.close()


if __name__ == "__main__":
    main()

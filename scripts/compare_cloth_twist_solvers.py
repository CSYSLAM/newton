# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare the unmodified VBD twist demo with XPBD; save metrics and real frames."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton import ParticleFlags
from newton.examples.cloth.example_cloth_twist import Example as VBDExample
from newton.examples.cloth.example_cloth_twist_xpbd import Example as XPBDExample
from newton.viewer import ViewerGL, ViewerNull


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=("vbd", "xpbd"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--contact-margin", type=float, default=0.0035)
    parser.add_argument("--stiffness-scale", type=float, default=1.0)
    parser.add_argument("--area-scale", type=float, default=1.0)
    parser.add_argument("--damping", type=float, default=0.0)
    parser.add_argument("--match-vbd-hinge", action="store_true")
    parser.add_argument("--bending-damping", type=float, default=0.0)
    parser.add_argument("--contact-stiffness", type=float)
    parser.add_argument("--global-xpbd", action="store_true")
    parser.add_argument("--cg-iterations", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("outputs/xpbd_physx_alignment"))
    args = parser.parse_args()
    if args.frames <= 10:
        parser.error("--frames must exceed the 10-frame timing warmup")
    if args.solver == "vbd" and (
        args.iterations != 4
        or args.substeps != 10
        or args.contact_margin != 0.0035
        or args.stiffness_scale != 1.0
        or args.area_scale != 1.0
        or args.damping
        or args.match_vbd_hinge
        or args.bending_damping
        or args.contact_stiffness is not None
        or args.global_xpbd
    ):
        parser.error("The VBD baseline is unmodified; material/iteration overrides are XPBD-only")
    options = XPBDExample.create_parser().parse_args([])
    options.iterations = args.iterations
    options.substeps = args.substeps
    options.contact_margin = args.contact_margin
    options.fem_solver = "global" if args.global_xpbd else "tgs"
    options.linear_iterations = args.cg_iterations if args.global_xpbd else 4
    options.verify_every = 0
    options.damping = 0.0
    example = (VBDExample if args.solver == "vbd" else XPBDExample)(ViewerNull(), options)
    args.output.mkdir(parents=True, exist_ok=True)
    material = example.model.tri_materials.numpy().copy()
    material[:, :2] *= args.stiffness_scale
    material[:, 1] *= args.area_scale
    if args.damping:
        material[:, 2] = args.damping
    example.model.tri_materials.assign(material)
    bending = example.model.edge_bending_properties.numpy().copy()
    if args.solver == "xpbd":
        # Comparison overrides are explicit, independent of evolving demo defaults.
        bending[:, 0] = 1.0e-3
    if args.match_vbd_hinge:
        bending[:, 0] *= example.model.edge_rest_length.numpy()
    if args.solver == "xpbd":
        bending[:, 1] = args.bending_damping
    example.model.edge_bending_properties.assign(bending)
    if args.contact_stiffness is not None:
        example.model.soft_contact_ke = args.contact_stiffness
        # Scalar material values are captured as kernel arguments; recapture.
        if args.solver == "xpbd":
            with wp.ScopedCapture() as capture:
                example.simulate()
            example.graph = capture.graph
    rest = example.model.particle_q.numpy().copy()
    faces = example.model.tri_indices.numpy()
    rest_pose = example.model.tri_poses.numpy()
    start, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
    gpu, wall, snapshots, velocities, records = [], [], [], [], []
    dynamic = (example.model.particle_flags.numpy() & int(ParticleFlags.ACTIVE)) != 0
    for frame in range(1, args.frames + 1):
        begin = time.perf_counter()
        wp.record_event(start)
        example.step()
        wp.record_event(end)
        elapsed = wp.get_event_elapsed_time(start, end)
        if frame > 10:
            gpu.append(elapsed)
            wall.append(1000 * (time.perf_counter() - begin))
        if args.solver == "xpbd":
            try:
                example.test_final()
            except (RuntimeError, AssertionError) as error:
                fem = example.solver._fem
                count = min(int(fem.count.numpy()[0]), len(fem.pairs))
                np.savez_compressed(
                    args.output / f"{args.label}_failed.npz",
                    frame=frame,
                    positions=example.state_0.particle_q.numpy(),
                    base=fem.base.numpy(),
                    pairs=fem.pairs.numpy()[:count],
                    kinds=fem.kinds.numpy()[:count],
                    gaps=fem.gaps.numpy()[:count],
                    status=fem.status.numpy(),
                    rest=example.model.particle_q.numpy(),
                    faces=faces,
                )
                print(json.dumps({"failed_frame": frame, "error": str(error)}), flush=True)
                raise
        if frame % 100 == 0 or frame == args.frames:
            pos = example.state_0.particle_q.numpy()
            vel = example.state_0.particle_qd.numpy()
            tri = pos[faces]
            f = np.stack((tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=2) @ rest_pose
            stretch = np.linalg.svd(f, compute_uv=False)
            area = np.prod(stretch, axis=1)
            mass = example.model.particle_mass.numpy()
            record = {
                "frame": frame,
                "area_min": float(area.min()),
                "area_max": float(area.max()),
                "area_rms": float(np.sqrt(np.mean((area - 1) ** 2))),
                "stretch_rms": float(np.sqrt(np.mean((stretch - 1) ** 2))),
                "stretch_max": float(stretch.max()),
                "speed_max": float(np.linalg.norm(vel, axis=1).max()),
                "kinetic": float(np.sum(mass * np.sum(vel**2, axis=1)) / 2),
                "dynamic_speed_max": float(np.linalg.norm(vel[dynamic], axis=1).max()),
                "dynamic_speed_p95": float(np.quantile(np.linalg.norm(vel[dynamic], axis=1), 0.95)),
                "dynamic_kinetic": float(np.sum(mass[dynamic] * np.sum(vel[dynamic] ** 2, axis=1)) / 2),
                "gpu_ms": float(np.mean(gpu[-100:])),
                "wall_ms": float(np.mean(wall[-100:])),
            }
            snapshots.append(pos)
            velocities.append(vel)
            records.append(record)
            print(json.dumps(record), flush=True)
    np.savez_compressed(
        args.output / f"{args.label}.npz",
        rest=rest,
        faces=faces,
        positions=np.array(snapshots),
        velocities=np.array(velocities),
        dynamic=dynamic,
    )
    summary = {
        "solver": args.solver,
        "global_xpbd": args.global_xpbd,
        "cg_iterations": args.cg_iterations,
        "iterations": args.iterations,
        "substeps": args.substeps,
        "contact_margin": args.contact_margin,
        "stiffness_scale": args.stiffness_scale,
        "area_scale": args.area_scale,
        "damping": args.damping,
        "match_vbd_hinge": args.match_vbd_hinge,
        "bending_damping": args.bending_damping,
        "contact_stiffness": example.model.soft_contact_ke,
        "contact_damping": example.model.soft_contact_kd,
        "contact_damping_enabled": args.solver == "vbd" or args.global_xpbd,
        "contact_response": "official_vbd"
        if args.solver == "vbd"
        else "global_barrier"
        if args.global_xpbd
        else "quadratic_distance_jacobi",
        "gpu_ms": float(np.mean(gpu)),
        "wall_ms": float(np.mean(wall)),
        "records": records,
    }
    (args.output / f"{args.label}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    viewer = ViewerGL(width=640, height=640, headless=True)
    viewer.set_model(example.model)
    viewer.set_camera(wp.vec3(2.25, 0.0, 0.0), 0.0, -180.0)
    from PIL import Image

    for record, pos in zip(records, snapshots, strict=True):
        example.state_0.particle_q.assign(pos)
        viewer.begin_frame(record["frame"] / 60)
        viewer.log_state(example.state_0)
        viewer.end_frame()
        Image.fromarray(viewer.get_frame().numpy()).convert("RGB").save(
            args.output / f"{args.label}_{record['frame']:04d}.jpg", quality=95
        )
    viewer.close()
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()

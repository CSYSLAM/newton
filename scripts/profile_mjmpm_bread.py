# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Profile the real bread trajectory without counting acceptance checks as FPS.

Use separate runs with --events for coarse CUDA phases and --render for
synchronized display cost. Nested/overlapping CUDA phase timings do not add.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples.mjmpm.example_mjmpm_w1_bread_cut as scene
from newton._src.solvers.mjmpm.mpm import solve_rheology as rheology
from newton.examples.mjmpm.example_mjmpm_w1_bread_cut import Example
from newton.viewer import ViewerGL, ViewerNull
from scripts.profile_mjmpm_frame import Events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--frames", type=int, default=660)
    parser.add_argument("--mpm-solver", choices=("gs", "gs-soa"), default="gs")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--cg-loop", choices=("fixed", "conditional"), default="fixed")
    parser.add_argument("--connection-product", choices=("gather", "scatter"), default="gather")
    parser.add_argument("--rigidity-feedback", choices=("auto", "reference"), default="auto")
    parser.add_argument(
        "--gs-blocks", type=int, default=0, help="Experimental GS block count; zero retains the production launch"
    )
    parser.add_argument("--empty-collision", action="store_true")
    parser.add_argument("--visual-batching", choices=("reference", "batched"), default="batched")
    options = parser.parse_args()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    if options.rigidity_feedback == "reference":
        rheology._FUSE_RIGIDITY_FEEDBACK = False
    if options.gs_blocks:
        original_init = rheology._GaussSeidelSolver.__init__

        def sized_init(self, *a, **kw):
            original_init(self, *a, **kw)
            dimension = options.gs_blocks * 64
            for command in (self.apply_stress_launch, self.solve_local_launch):
                command.set_dim(dimension)
                command.set_param_at_index(1, dimension)

        rheology._GaussSeidelSolver.__init__ = sized_init
    wp.set_device("cuda:0")
    args = Example.create_parser().parse_args(["--viewer", "null"])
    args.test = False
    args.mpm_solver = options.mpm_solver
    viewer = (
        ViewerGL(width=1920, height=1080, headless=not options.visible, num_frames=options.frames)
        if options.render
        else ViewerNull(num_frames=options.frames)
    )
    if options.visual_batching == "reference":
        visual_class = scene.BreadCutVisuals

        def unbatched(*a, **kw):
            visuals = visual_class(*a, **kw, batch_opaque=False)
            visuals.reuse_properties = False
            return visuals

        scene.BreadCutVisuals = unbatched
    example = Example(viewer, args)
    if options.connection_product == "scatter":
        example.cohesion.cg._A._matvec = example.cohesion._matvec_scatter
    if options.cg_loop == "conditional":
        example.cohesion.cg._use_cuda_graph = True
    device = example.model.device
    env = example.solver.environment
    if options.empty_collision:
        if np.any((env.model.body_flags.numpy() & int(newton.BodyFlags.PROXY)) == 0):
            raise ValueError("The empty-contact experiment requires all environment bodies to be proxies")

        def empty_collide(*a, **kw):
            env.contacts.rigid_contact_count.zero_()

        env.detector.collide = empty_collide
    events = Events(device)
    structure = {}
    if options.events:
        original_loop = rheology._run_solver_loop

        def iteration_loop(rheology_solver, contact_solver, *a, **kw):
            if not structure:
                structure.update(
                    {
                        "solver": type(rheology_solver).__name__,
                        "colors": getattr(rheology_solver, "color_count", None),
                        "strain_rows": len(rheology_solver.rheology.stress),
                        "contact_points": len(contact_solver.collision.collider_impulse),
                    }
                )
                operator = contact_solver.collision.rigidity_operator
                if operator is not None:
                    j, inverse = operator[:2]
                    structure["rigidity"] = {
                        "rows": j.nrow,
                        "columns": j.ncol,
                        "nnz_capacity": j.nnz,
                        "block_size": j.block_size,
                        "shape": j.shape,
                        "dynamic_rows": j.row_counts is not None or inverse.row_counts is not None,
                        "independent_bodies": operator[3].body_count if len(operator) > 3 else None,
                    }
            return original_loop(rheology_solver, contact_solver, *a, **kw)

        rheology._run_solver_loop = iteration_loop
        events.wrap(rheology, "_run_solver_loop", "mpm_iteration_loop")
        for obj, method, name in (
            (example, "_simulate_frame", "frame_gpu"),
            (example, "_solve_frame_ik", "ik"),
            (example.cohesion, "cut", "cut_connections"),
            (example.cohesion, "apply", "cohesion_total"),
            (example.cohesion.cg, "_run", "cohesion_cg"),
            (example.solver, "step", "coupled_step"),
            (env, "step", "environment"),
            (env.detector, "collide", "environment_rigid_collision"),
            (env, "_solve_rigid_contacts", "environment_rigid_solve"),
            (env, "_prepare_constrained_material", "prepare_response"),
            (env, "_finish_constrained_material", "finish_response"),
            (env.material, "prepare_step", "mpm_prepare_overlapped"),
            (env.material, "step", "mpm_main"),
            (env.material, "_solve_rheology", "mpm_rheology"),
            (env._link_mobility, "update", "articulated_mobility"),
            (env._link_mobility._solver, "step", "mujoco"),
        ):
            events.wrap(obj, method, name)

    # Separate diagnostic computation from the GPU wait its first numpy()
    # would otherwise hide. The wall step timing still includes both costs.
    diagnostics = [0.0]
    original_after = example.after_step

    def after_step():
        if example.frame % 30 == 0:
            wp.synchronize_device(device)
        begin = time.perf_counter()
        original_after()
        diagnostics[0] = (time.perf_counter() - begin) * 1000

    example.after_step = after_step
    records, contact_samples = [], []
    for frame in range(options.frames):
        begin = time.perf_counter()
        example.step()
        wp.synchronize_device(device)
        stepped = time.perf_counter()
        if options.render:
            example.render()
            viewer.renderer.gl.glFinish()
            wp.synchronize_device(device)
        end = time.perf_counter()
        if options.validate:
            example.test_post_step()
        records.append(
            {
                "frame": frame,
                "step_ms": (stepped - begin) * 1000,
                "render_ms": (end - stepped) * 1000,
                "total_ms": (end - begin) * 1000,
                "diagnostics_ms": diagnostics[0],
                **(events.read() if options.events else {}),
            }
        )
        if options.cg_loop == "conditional":
            records[-1]["cg_last_substep_iterations"] = int(example.cohesion.cg._cur_iter_and_condition.numpy()[0])
        if frame in (60, 180, 300, 420, 480, 540, 650):
            contact_samples.append(
                {"frame": frame, "environment_rigid_contacts": int(env.contacts.rigid_contact_count.numpy()[0])}
            )
    example.check_finite()
    example.solver.check_capacity_status()
    if options.validate:
        example.test_final()
    phases = {}
    for name, start, stop in (
        ("idle", 60, 120),
        ("cut", 240, 420),
        ("separate", 432, 492),
        ("withdraw", 492, 576),
        ("hold", 600, 660),
        ("whole_warm", 60, 660),
    ):
        samples = [r for r in records if start <= r["frame"] < stop]
        if not samples:
            continue
        phases[name] = {
            key + "_mean": float(np.mean([s[key] for s in samples])) for key in samples[0] if key != "frame"
        }
        phases[name]["step_ms_median"] = float(np.median([s["step_ms"] for s in samples]))
        phases[name]["synchronized_fps"] = 1000 / phases[name]["total_ms_mean"]
    report = {
        "measurement": "synchronized wall time; acceptance checks excluded; diagnostics retained",
        "device": device.name,
        "events": options.events,
        "mpm_solver": options.mpm_solver,
        "cg_loop": options.cg_loop,
        "connection_product": options.connection_product,
        "rigidity_feedback": options.rigidity_feedback,
        "gs_blocks": options.gs_blocks,
        "empty_collision": options.empty_collision,
        "visual_batches": len(example.visuals.parts),
        "visual_batching": options.visual_batching,
        "mpm_structure": structure,
        "validation": getattr(example, "validation", None),
        "render": options.render,
        "visible": options.visible,
        "particles": example.model.particle_count,
        "connections": len(example.cohesion.pairs),
        "free_rigid_bodies": len(example.solver.ownership.mpm_bodies),
        "substeps": args.substeps,
        "mpm_iterations": args.iterations,
        "cg_iterations": 24,
        "max_active_cell_count": env.material.max_active_cell_count,
        "optimizations": {
            "frame_cuda_graph": example.graph is not None,
            "articulated_response": env._link_mobility is not None,
            "cooperative_rigid_buckets": env._use_cooperative_rigid_buckets,
            "contact_space_enabled": env._use_contact_space,
            "contact_space_allocated": env._contact_space is not None,
            "overlapped_material_prepare": env._material_stream is not None,
        },
        "contact_samples": contact_samples,
        "phases": phases,
        "cuda_calls_per_captured_frame": dict(events.used),
        "samples": records,
    }
    options.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps({"phases": phases, "optimizations": report["optimizations"], "contacts": contact_samples}, indent=2),
        flush=True,
    )
    viewer.close()


if __name__ == "__main__":
    main()

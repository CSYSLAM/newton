# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare frozen-substep iteration maps without changing the scene policy.

An extra ordinary GS sweep measures a fixed-point defect, not a force residual
or an energy norm. All variants share inertia, kinematic poses and collision
candidates. Sampling synchronizes the GPU and is not a performance benchmark.
"""

import argparse
import json
from unittest.mock import patch

import numpy as np
import warp as wp

import newton.examples
import newton.viewer
from newton._src.solvers.mjvbd_v2.particle_multilevel import ParticleMultilevelCorrection
from newton.examples.mjvbdv2 import (
    example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00 as scene,
)
from newton.solvers import SolverMJVBDV2


class ProbeComplete(Exception):
    """Stop the diagnostic after its single frozen substep."""


def rms(values):
    return float(np.sqrt(np.mean(np.sum(np.asarray(values, dtype=np.float64) ** 2, axis=1))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-frames", type=int, default=900)
    parser.add_argument("--coarse", action="store_true", help="Probe existing coarse operators after six fast sweeps")
    parser.add_argument(
        "--contact-coupling",
        action="store_true",
        help="Compare identical Galerkin solves with/without contact coupling",
    )
    parser.add_argument("--history", action="store_true", help="CPU prototype of bounded iterate-history acceleration")
    parser.add_argument("--history-coefficient-limit", type=float, default=10.0)
    args = parser.parse_args()
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be nonnegative")
    wp.set_device("cuda:0")
    wp.config.log_level = wp.LOG_WARNING
    original_init = SolverMJVBDV2.__init__

    def configure(self, model, **kwargs):
        options = dict(kwargs["vbd_options"])
        options.pop("particle_jacobi_polish_iterations", None)
        kwargs["vbd_options"] = options
        original_init(self, model, **kwargs)

    with patch.object(SolverMJVBDV2, "__init__", configure):
        example = scene.Example(newton.viewer.ViewerNull(), newton.examples.default_args(scene.Example.create_parser()))
    for _ in range(args.warmup_frames):
        example.step()
    backend = example.solver.vbd_solver
    if backend.particle_multilevel is not None or backend.particle_collision_detection_interval != -1:
        raise RuntimeError("This probe requires multilevel disabled and a frozen per-substep contact snapshot")
    original_iteration = backend._solve_particle_iteration
    reference_positions = {}
    triangles = example.model.tri_indices.numpy()
    edges = np.unique(
        np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1), axis=0
    )
    source = np.concatenate([edges[:, 0], edges[:, 1]])
    target = np.concatenate([edges[:, 1], edges[:, 0]])
    degree = np.maximum(np.bincount(target, minlength=example.model.particle_count), 1)[:, None]

    def smooth(values, rounds=4):
        values = values.astype(np.float64)
        for _ in range(rounds):
            neighbor_sum = np.zeros_like(values)
            np.add.at(neighbor_sum, target, values[source])
            values = 0.5 * values + 0.5 * neighbor_sum / degree
        return values

    coarse_operators = {}
    if args.coarse or args.contact_coupling:
        for name in ("uncoupled", "coupled", "coupled32") if args.contact_coupling else ("graph", "galerkin"):
            operator = "galerkin" if args.contact_coupling else name
            coarse_operators[name] = ParticleMultilevelCorrection(
                example.model,
                operator=operator,
                cluster_size=8,
                coarse_iterations=32 if name == "coupled32" else 8,
                coupling=0.5,
                relaxation=0.1,
                max_radius_fraction=0.05,
                minimum_residual_reduction=1.0e-4,
                max_clamp_fraction=0.5,
            )
            coarse_operators[name].contact_projection_enabled = name != "uncoupled"

    def probe(state_in, state_out, contacts, dt, iter_num):
        if iter_num != 0:
            raise RuntimeError("Expected the beginning of a substep")
        initial_q = wp.clone(state_in.particle_q)
        initial_delta = wp.clone(backend.particle_displacements)
        initial_q_numpy = initial_q.numpy()
        saved = {
            name: getattr(backend, name)
            for name in (
                "iterations",
                "particle_chebyshev_enabled",
                "particle_enable_batched_jacobi",
                "particle_chebyshev_weights",
                "_particle_truncation_cache",
                "_surface_cached_kernel",
                "surface_anchor_angles",
                "_particle_jacobi_color_schedules",
                "_particle_jacobi_group_schedules",
                "particle_multilevel",
                "particle_multilevel_checkpoints",
                "particle_jacobi_relaxation",
            )
        }
        q_scratch = wp.empty_like(initial_q)
        delta_scratch = wp.empty_like(initial_delta)
        truncation_scratch = wp.empty_like(backend.truncation_ts)
        exclusion_scratch = wp.empty_like(backend.particle_chebyshev_collided)

        def restore_start():
            state_in.particle_q.assign(initial_q)
            backend.particle_displacements.assign(initial_delta)
            backend.particle_chebyshev_older.assign(initial_q)
            backend.particle_chebyshev_collided.zero_()
            if backend._particle_truncation_cache is not None:
                backend._particle_truncation_cache.rebuild(backend)

        def defect():
            q_scratch.assign(state_in.particle_q)
            delta_scratch.assign(backend.particle_displacements)
            truncation_scratch.assign(backend.truncation_ts)
            exclusion_scratch.assign(backend.particle_chebyshev_collided)
            flags = backend.particle_enable_batched_jacobi, backend.particle_chebyshev_enabled
            cache = backend._particle_truncation_cache, backend._surface_cached_kernel, backend.surface_anchor_angles
            backend.particle_enable_batched_jacobi = False
            backend.particle_chebyshev_enabled = False
            backend._particle_truncation_cache = None
            backend._surface_cached_kernel = None
            backend.surface_anchor_angles = None
            original_iteration(state_in, state_out, contacts, dt, 0)
            movement = state_in.particle_q.numpy() - q_scratch.numpy()
            state_in.particle_q.assign(q_scratch)
            backend.particle_displacements.assign(delta_scratch)
            backend.truncation_ts.assign(truncation_scratch)
            backend.particle_chebyshev_collided.assign(exclusion_scratch)
            backend.particle_enable_batched_jacobi, backend.particle_chebyshev_enabled = flags
            backend._particle_truncation_cache, backend._surface_cached_kernel, backend.surface_anchor_angles = cache
            return rms(movement)

        try:
            labels = ["ordinary", "fast", "fast_final_gs"]
            if args.contact_coupling:
                labels.extend(f"{operator}_0.1" for operator in coarse_operators)
            elif args.coarse:
                labels.extend(
                    f"{operator}_{relaxation}" for operator in coarse_operators for relaxation in (0.1, 0.5, 1.0)
                )
                labels.extend(f"{operator}_1.0_smooth" for operator in coarse_operators)
            if args.history:
                labels.extend(("fixed_half", "aa_3_0.5", "aa_5_0.5", "aa_3_0.8", "aa_5_0.8", "aa_5_1.0"))
                labels.extend(("aa1batch_3_0.5", "aa1batch_5_0.5", "aa1batch_5_0.8"))
                labels.extend(("aasmooth_5_0.8", "aasmooth_5_1.0"))
                labels.extend(("aasmooth_1_0.8", "aasmooth_2_0.8", "aasmooth1batch_5_0.8"))
            for label in labels:
                for name, value in saved.items():
                    setattr(backend, name, value)
                backend.iterations = 60
                backend.particle_chebyshev_weights = backend._build_particle_chebyshev_weights(0.8, 60)
                if label == "ordinary":
                    backend.particle_enable_batched_jacobi = False
                    backend.particle_chebyshev_enabled = False
                    backend._particle_truncation_cache = None
                    backend._surface_cached_kernel = None
                    backend.surface_anchor_angles = None
                anderson = label.startswith("aa")
                single_batch = "1batch" in label
                history_mode = label == "fixed_half" or anderson
                if history_mode:
                    backend.particle_chebyshev_enabled = False
                    backend.particle_jacobi_relaxation = float(label.split("_")[2]) if anderson else 0.5
                if single_batch:
                    backend._particle_jacobi_color_schedules = (wp.zeros(example.model.particle_count, dtype=wp.int32),)
                    backend._particle_jacobi_group_schedules = (
                        (wp.array(np.arange(example.model.particle_count), dtype=wp.int32),),
                    )
                restore_start()
                previous = initial_q_numpy
                samples = []
                history_f, history_g = [], []
                iteration_budget = 60 if label in ("ordinary", "fast") else (12 if single_batch else 8)
                for iteration in range(iteration_budget + 1):
                    if iteration:
                        if label == "fast_final_gs" and iteration == 8:
                            backend.particle_enable_batched_jacobi = False
                            backend.particle_chebyshev_enabled = False
                        original_iteration(state_in, state_out, contacts, dt, 0 if history_mode else iteration - 1)
                        if anderson:
                            g = state_in.particle_q.numpy().astype(np.float64)
                            f = g - previous
                            if label.startswith("aasmooth"):
                                f = smooth(f)
                            history_f.append(f.reshape(-1))
                            history_g.append(g.reshape(-1))
                            depth = min(int(label.split("_")[1]), len(history_f) - 1)
                            if depth:
                                df = np.diff(np.array(history_f[-depth - 1 :]), axis=0).T
                                dg = np.diff(np.array(history_g[-depth - 1 :]), axis=0).T
                                theta = np.linalg.lstsq(df, history_f[-1], rcond=1.0e-6)[0]
                                if np.linalg.norm(theta) <= args.history_coefficient_limit:
                                    correction = -(dg @ theta).reshape(-1, 3)
                                    if label.startswith("aasmooth"):
                                        correction = smooth(correction, rounds=2)
                                    cap = 0.05 * example.model.particle_radius.numpy()
                                    correction *= np.minimum(
                                        1.0, cap / np.maximum(np.linalg.norm(correction, axis=1), 1e-30)
                                    )[:, None]
                                    # DAT consumes displacement from its collision anchor,
                                    # not an incremental change from the current iterate.
                                    backend.particle_displacements.assign(
                                        backend.particle_displacements.numpy() + correction.astype(np.float32)
                                    )
                                    backend._penetration_free_truncation(state_in.particle_q)
                        if label.split("_")[0] in coarse_operators and iteration == 6:
                            operator, relaxation = label.split("_")[:2]
                            correction = coarse_operators[operator]
                            correction.relaxation = float(relaxation)
                            backend.particle_multilevel = correction
                            backend.particle_multilevel_checkpoints = ()
                            truncate = backend._penetration_free_truncation
                            before_coarse = backend.particle_displacements.numpy()

                            def smooth_and_truncate(q, before_coarse=before_coarse, truncate=truncate):
                                coarse_delta = backend.particle_displacements.numpy() - before_coarse
                                coarse_delta = smooth(coarse_delta, rounds=2)
                                inactive = (
                                    example.model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)
                                ) == 0
                                coarse_delta[inactive] = 0.0
                                backend.particle_displacements.assign((before_coarse + coarse_delta).astype(np.float32))
                                truncate(q)

                            with patch.object(
                                backend,
                                "_penetration_free_truncation",
                                smooth_and_truncate if label.endswith("smooth") else truncate,
                            ):
                                backend._apply_particle_multilevel_correction(
                                    state_in,
                                    contacts,
                                    state_out.body_q,
                                    backend._external_body_q_prev,
                                    state_out.body_qd,
                                    dt,
                                )
                            # A changed iterate invalidates two-step extrapolation history.
                            backend.particle_chebyshev_enabled = False
                    current = state_in.particle_q.numpy()
                    if not np.all(np.isfinite(current)):
                        raise ValueError(f"Nonfinite {label} at iteration {iteration}")
                    if iteration in (0, 1, 2, 4, 8, 12, 20, 30, 60):
                        if label == "ordinary":
                            reference_positions[iteration] = current.copy()
                        samples.append(
                            {
                                "iteration": iteration,
                                "last_update_rms_m": rms(current - previous),
                                "ordinary_sweep_defect_m": defect(),
                                "chebyshev_excluded_particles": int(
                                    np.count_nonzero(backend.particle_chebyshev_collided.numpy())
                                ),
                                "q": current.copy(),
                                "coarse_status": (
                                    backend.particle_multilevel.runtime_status.numpy().tolist()
                                    if backend.particle_multilevel is not None
                                    else None
                                ),
                                "coarse_metrics": (
                                    backend.particle_multilevel.runtime_metrics.numpy().tolist()
                                    if backend.particle_multilevel is not None
                                    else None
                                ),
                            }
                        )
                    previous = current
                for sample in samples:
                    current = sample.pop("q")
                    sample.update(
                        case=label,
                        warmup_frames=args.warmup_frames,
                        rms_to_ordinary30_m=rms(current - reference_positions[30]),
                        rms_to_ordinary60_m=rms(current - reference_positions[60]),
                        smoothed_rms_to_ordinary30_m=rms(smooth(current - reference_positions[30])),
                    )
                    print(json.dumps(sample), flush=True)
        finally:
            for name, value in saved.items():
                setattr(backend, name, value)
            restore_start()
        raise ProbeComplete

    with patch.object(backend, "_solve_particle_iteration", probe):
        try:
            example.simulate()
        except ProbeComplete:
            pass


if __name__ == "__main__":
    main()

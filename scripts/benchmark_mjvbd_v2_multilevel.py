# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare T-shirt solver candidates against ordinary 20/30-sweep VBD.

Run from the repository root with ``uv run scripts/benchmark_mjvbd_v2_multilevel.py``.
Use ``--compare-demo`` for the exact current demo baseline and 20-sweep reference.
Timing includes every physics/IK frame, but excludes construction, graph capture,
and checkpoint readback. Independent contact trajectories are nondeterministic;
the repeated reference estimates variability, not a rigorous error bound.
"""

import argparse
import gc
import importlib
import json
import time
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.mjvbd_v2 import particle_surface_cache
from newton._src.solvers.mjvbd_v2.vbd_soft.solver_vbd import SolverVBD as SolverVBDSoft
from newton.solvers import SolverMJVBDV2

_EXAMPLE_MODULE = "newton.examples.mjvbdv2.example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00"


def _coarse_options(sweeps, operator, reference_sweeps):
    return {
        "iterations": sweeps,
        "particle_enable_multilevel_correction": operator is not None,
        "particle_multilevel_operator": operator or "graph",
        "particle_multilevel_cluster_size": 4 if operator == "galerkin" else 8,
        "particle_multilevel_relaxation": 0.6 if operator == "galerkin" else 0.1,
        "particle_multilevel_max_radius_fraction": 0.2,
        "particle_multilevel_min_residual_reduction": 1.0e-4,
        "particle_multilevel_max_clamp_fraction": 0.5,
        "particle_multilevel_fallback_iterations": reference_sweeps if operator else None,
    }


def _example_args(args):
    module = importlib.import_module(_EXAMPLE_MODULE)
    example_args = newton.examples.default_args(module.Example.create_parser())
    example_args.device = args.device
    example_args.viewer = "null"
    example_args.num_frames = args.frames
    example_args.test = False
    example_args.benchmark = False
    example_args.graph_capture = args.graph_capture
    return module, example_args


def _mesh_edges(model):
    triangles = model.tri_indices.numpy().reshape((-1, 3))
    edges = np.concatenate([triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]])
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _run_case(label, sweeps, operator, args, *, demo_mode=None):
    original_init = SolverMJVBDV2.__init__

    def configured_init(self, model, *solver_args, **kwargs):
        options = dict(kwargs.get("vbd_options") or {})
        if demo_mode is None:
            options.update(_coarse_options(sweeps, operator, args.reference_sweeps))
        kwargs["vbd_options"] = options
        return original_init(self, model, *solver_args, **kwargs)

    module, example_args = _example_args(args)
    if demo_mode is not None:
        example_args.particle_solver_mode = demo_mode
    with mock.patch.object(SolverMJVBDV2, "__init__", configured_init):
        example = module.Example(newton.viewer.ViewerNull(num_frames=args.frames), example_args)

    correction = example.solver.vbd_solver.particle_multilevel
    if operator is not None and correction is None:
        raise RuntimeError(f"{label}: requested multilevel correction was disabled on this model/device")

    edges = _mesh_edges(example.model)
    checkpoints = []
    elapsed = 0.0
    checkpoint_frame_ms = []
    checkpoint_status = []
    cleanup_checkpoint_status = []
    cleanup_checkpoint_max_fraction = []
    for start in range(0, args.frames, args.checkpoint_interval):
        stop = min(args.frames, start + args.checkpoint_interval)
        wp.synchronize_device(example.device)
        begin = time.perf_counter()
        for _frame in range(start, stop):
            example.step()
        wp.synchronize_device(example.device)
        checkpoint_elapsed = time.perf_counter() - begin
        elapsed += checkpoint_elapsed
        checkpoint_frame_ms.append(1000.0 * checkpoint_elapsed / (stop - start))
        positions = example.state_0.particle_q.numpy().copy()
        if not np.all(np.isfinite(positions)):
            raise ValueError(f"{label}: nonfinite particles at frame {stop}")
        checkpoints.append((stop, positions))
        if correction is not None:
            checkpoint_status.append(int(correction.runtime_status.numpy()[0]))
        cleanup_status = example.solver.vbd_solver.particle_chebyshev_cleanup_status
        cleanup_metrics = example.solver.vbd_solver.particle_chebyshev_cleanup_metrics
        if cleanup_status is not None:
            cleanup_checkpoint_status.append(int(cleanup_status.numpy()[0]))
            cleanup_checkpoint_max_fraction.append(float(cleanup_metrics.numpy()[0]))
    example.test_final()
    metrics = {
        "label": label,
        "sweeps": sweeps,
        "operator": operator,
        "demo_mode": demo_mode,
        "frames": args.frames,
        "mean_frame_ms": 1000.0 * elapsed / args.frames,
        "particle_count": example.model.particle_count,
        "cluster_count": correction.cluster_count if correction is not None else 0,
        "checkpoint_frame_ms": checkpoint_frame_ms,
        # These are only the last substep at each checkpoint, not fallback rates.
        "checkpoint_runtime_status": checkpoint_status,
        "cleanup_checkpoint_status": cleanup_checkpoint_status,
        "cleanup_checkpoint_max_fraction": cleanup_checkpoint_max_fraction,
        "test_final_passed": True,
    }
    del correction, example
    gc.collect()
    return metrics, checkpoints, edges


def _compare(checkpoints, reference, edges):
    errors = []
    for (frame, positions), (reference_frame, reference_positions) in zip(checkpoints, reference, strict=True):
        if frame != reference_frame or positions.shape != reference_positions.shape:
            raise ValueError("Candidate and reference checkpoints differ")
        positions64 = positions.astype(np.float64)
        reference64 = reference_positions.astype(np.float64)
        difference = positions64 - reference64
        lengths = np.linalg.norm(positions64[edges[:, 0]] - positions64[edges[:, 1]], axis=1)
        reference_lengths = np.linalg.norm(reference64[edges[:, 0]] - reference64[edges[:, 1]], axis=1)
        errors.append(
            {
                "frame": frame,
                "position_rms_mm": float(1000.0 * np.sqrt(np.mean(np.sum(difference**2, axis=1)))),
                "edge_length_mae_mm": float(1000.0 * np.mean(np.abs(lengths - reference_lengths))),
            }
        )
    return errors


def _probe_same_substep(sweeps, args):
    """Fork a correction from the low-sweep state; continue the reference unchanged.

    This is an accuracy diagnostic, not a timing benchmark. The T-shirt has
    externally driven rigid bodies, so the trial only changes particle positions
    and displacements. Both are restored before the next ordinary color solve;
    temporary coarse/DAT buffers are not reference history. Sampling is at the
    final substep of every frame, not at every substep.
    """
    original_init = SolverMJVBDV2.__init__
    original_iteration = SolverVBDSoft._solve_particle_iteration
    buffers = {}

    def configured_init(self, model, *solver_args, **kwargs):
        options = dict(kwargs.get("vbd_options") or {})
        options.update(_coarse_options(args.reference_sweeps, "galerkin", None))
        kwargs["vbd_options"] = options
        original_init(self, model, *solver_args, **kwargs)
        solver = self.vbd_solver
        if not isinstance(solver, SolverVBDSoft) or not solver.integrate_with_external_rigid_solver:
            raise RuntimeError("The same-substep diagnostic requires the externally driven soft-only T-shirt backend")
        if solver.particle_multilevel is None:
            raise RuntimeError("The same-substep diagnostic requires an enabled coarse correction")
        for name in ("positions", "displacements", "candidate"):
            buffers[name] = wp.empty_like(model.particle_q)
        buffers["status"] = wp.zeros(1, dtype=wp.int32, device=model.device)

    def solve_iteration(self, state_in, state_out, contacts, dt, iter_num):
        correction = self.particle_multilevel
        self.particle_multilevel = None
        try:
            original_iteration(self, state_in, state_out, contacts, dt, iter_num)
        finally:
            self.particle_multilevel = correction
        if iter_num + 1 != sweeps:
            return
        wp.copy(buffers["positions"], state_in.particle_q)
        wp.copy(buffers["displacements"], self.particle_displacements)
        self._apply_particle_multilevel_correction(
            state_in, contacts, state_out.body_q, self._external_body_q_prev, state_out.body_qd, dt
        )
        wp.copy(buffers["candidate"], state_in.particle_q)
        wp.copy(buffers["status"], correction.runtime_status)
        wp.copy(state_in.particle_q, buffers["positions"])
        wp.copy(self.particle_displacements, buffers["displacements"])

    module, example_args = _example_args(args)
    records = {"plain": [], "galerkin_before_fallback": []}
    rejected_samples = 0
    with (
        mock.patch.object(SolverMJVBDV2, "__init__", configured_init),
        mock.patch.object(SolverVBDSoft, "_solve_particle_iteration", solve_iteration),
    ):
        example = module.Example(newton.viewer.ViewerNull(num_frames=args.frames), example_args)
        edges = _mesh_edges(example.model)
        for frame in range(args.frames):
            example.step()
            reference = example.state_0.particle_q.numpy()
            if not np.all(np.isfinite(reference)):
                raise ValueError(f"Nonfinite reference particles at frame {frame + 1}")
            rejected_samples += int(buffers["status"].numpy()[0] != 0)
            for label, name in (("plain", "positions"), ("galerkin_before_fallback", "candidate")):
                positions = buffers[name].numpy()
                if not np.all(np.isfinite(positions)):
                    raise ValueError(f"Nonfinite {label} particles at frame {frame + 1}")
                error = _compare([(frame, positions)], [(frame, reference)], edges)[0]
                records[label].append((error["position_rms_mm"], error["edge_length_mae_mm"]))
        example.test_final()
    summaries = {}
    for label, errors in records.items():
        error_array = np.asarray(errors)
        summaries[label] = {
            "mean_position_rms_mm": float(np.mean(error_array[:, 0])),
            "p95_position_rms_mm": float(np.percentile(error_array[:, 0], 95)),
            "mean_edge_length_mae_mm": float(np.mean(error_array[:, 1])),
        }
    result = {
        "mode": "same_substep",
        "sweeps": sweeps,
        "reference_sweeps": args.reference_sweeps,
        "frame_samples": args.frames,
        "rejected_last_substep_samples": rejected_samples,
        "errors": summaries,
        "test_final_passed": True,
    }
    del example
    buffers.clear()
    gc.collect()
    return result


def _probe_demo_same_substep(args):
    """Compare native demo modes on a common ordinary 20-sweep history."""
    original_init = SolverMJVBDV2.__init__
    original_iteration = SolverVBDSoft._solve_particle_iteration
    buffers = {}

    def configured_init(self, model, *solver_args, **kwargs):
        options = dict(kwargs.get("vbd_options") or {})
        # Keep the demo's exact graph defaults (including the 5% radius cap),
        # but only apply that correction inside the isolated baseline trial.
        options.update(
            iterations=20,
            particle_chebyshev_spectral_radius=0.9,
            particle_chebyshev_warmup_iterations=2,
            particle_chebyshev_polish_iterations=2,
            particle_chebyshev_contact_rings=2,
            particle_enable_multilevel_correction=True,
            particle_multilevel_fallback_iterations=None,
            particle_enable_surface_cache=True,
            particle_enable_truncation_cache=True,
        )
        kwargs["vbd_options"] = options
        original_init(self, model, *solver_args, **kwargs)
        solver = self.vbd_solver
        if not isinstance(solver, SolverVBDSoft) or not solver.integrate_with_external_rigid_solver:
            raise RuntimeError("This diagnostic requires the externally driven T-shirt soft backend")
        if solver.rigid_linear_beta != 0.0:
            raise RuntimeError("The isolated particle trials require fixed contact penalties")
        if solver.particle_collision_detection_interval != -1:
            raise RuntimeError("The isolated trials require collision candidates to stay fixed within each substep")
        buffers["correction"] = solver.particle_multilevel
        solver.particle_multilevel = None
        for name in ("surface_anchor_angles", "_surface_cached_kernel", "_particle_truncation_cache"):
            buffers[name] = getattr(solver, name)
            setattr(solver, name, None)
        solver.particle_chebyshev_enabled = False
        for name in ("positions", "displacements", "baseline", "candidate", "cached13", "guarded"):
            buffers[name] = wp.empty_like(model.particle_q)

    def save(self, state_in):
        wp.copy(buffers["positions"], state_in.particle_q)
        wp.copy(buffers["displacements"], self.particle_displacements)

    def restore(self, state_in):
        wp.copy(state_in.particle_q, buffers["positions"])
        wp.copy(self.particle_displacements, buffers["displacements"])

    def solve_iteration(self, state_in, state_out, contacts, dt, iter_num):
        if iter_num == 0:
            # Fork before the first sweep: cached13 includes one ordinary warm-up
            # and three ordinary finish sweeps, just like the native demo mode.
            save(self, state_in)
            for name in ("surface_anchor_angles", "_surface_cached_kernel", "_particle_truncation_cache"):
                setattr(self, name, buffers[name])
            self.iterations = 13
            self.particle_surface_relaxation = 1.3
            try:
                self._particle_truncation_cache.rebuild(self)
                wp.launch(
                    particle_surface_cache._prepare_anchor_angles,
                    dim=self.model.edge_count,
                    inputs=[self.particle_q_prev, self.model.edge_indices],
                    outputs=[self.surface_anchor_angles],
                    device=self.device,
                )
                for candidate_iteration in range(13):
                    original_iteration(self, state_in, state_out, contacts, dt, candidate_iteration)
                wp.copy(buffers["cached13"], state_in.particle_q)
            finally:
                for name in ("surface_anchor_angles", "_surface_cached_kernel", "_particle_truncation_cache"):
                    setattr(self, name, None)
                self.iterations = 20
                self.particle_surface_relaxation = 1.0
                restore(self, state_in)
            self.iterations = 8
            self.particle_chebyshev_weights = self._build_particle_chebyshev_weights(0.9, 4)
            self.particle_chebyshev_enabled = True
            self.particle_chebyshev_older.assign(state_in.particle_q)
            self.particle_chebyshev_collided.zero_()
            self.particle_multilevel = buffers["correction"]
            for name in ("surface_anchor_angles", "_surface_cached_kernel", "_particle_truncation_cache"):
                setattr(self, name, buffers[name])
            try:
                self._particle_truncation_cache.rebuild(self)
                wp.launch(
                    particle_surface_cache._prepare_anchor_angles,
                    dim=self.model.edge_count,
                    inputs=[self.particle_q_prev, self.model.edge_indices],
                    outputs=[self.surface_anchor_angles],
                    device=self.device,
                )
                for candidate_iteration in range(8):
                    original_iteration(self, state_in, state_out, contacts, dt, candidate_iteration)
                wp.copy(buffers["guarded"], state_in.particle_q)
            finally:
                self.particle_multilevel = None
                self.particle_chebyshev_enabled = False
                for name in ("surface_anchor_angles", "_surface_cached_kernel", "_particle_truncation_cache"):
                    setattr(self, name, None)
                self.iterations = 20
                restore(self, state_in)
        original_iteration(self, state_in, state_out, contacts, dt, iter_num)
        if iter_num == 0:
            save(self, state_in)
            self.iterations = 12
            self.particle_surface_relaxation = 1.3
            try:
                for candidate_iteration in range(1, 12):
                    original_iteration(self, state_in, state_out, contacts, dt, candidate_iteration)
                wp.copy(buffers["candidate"], state_in.particle_q)
            finally:
                self.iterations = 20
                self.particle_surface_relaxation = 1.0
                restore(self, state_in)
        if iter_num == 11:
            save(self, state_in)
            self.particle_multilevel = buffers["correction"]
            try:
                self._apply_particle_multilevel_correction(
                    state_in, contacts, state_out.body_q, self._external_body_q_prev, state_out.body_qd, dt
                )
                wp.copy(buffers["baseline"], state_in.particle_q)
            finally:
                self.particle_multilevel = None
                restore(self, state_in)

    module, example_args = _example_args(args)
    example_args.particle_solver_mode = "baseline"
    records = {"baseline": [], "contact-free": [], "cached13": [], "guarded": []}
    rejected_samples = 0
    with (
        mock.patch.object(SolverMJVBDV2, "__init__", configured_init),
        mock.patch.object(SolverVBDSoft, "_solve_particle_iteration", solve_iteration),
    ):
        example = module.Example(newton.viewer.ViewerNull(num_frames=args.frames), example_args)
        edges = _mesh_edges(example.model)
        for frame in range(args.frames):
            example.step()
            reference = example.state_0.particle_q.numpy()
            rejected = int(buffers["correction"].runtime_status.numpy()[0]) != 0
            rejected_samples += int(rejected)
            for label, name in (
                ("baseline", "baseline"),
                ("contact-free", "candidate"),
                ("cached13", "cached13"),
                ("guarded", "guarded"),
            ):
                positions = reference if label == "baseline" and rejected else buffers[name].numpy()
                if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(reference)):
                    raise ValueError(f"Nonfinite {label} state at frame {frame + 1}")
                error = _compare([(frame, positions)], [(frame, reference)], edges)[0]
                records[label].append((error["position_rms_mm"], error["edge_length_mae_mm"]))
        example.test_final()
    summaries = {}
    for label, errors in records.items():
        error_array = np.asarray(errors)
        summaries[label] = {
            "mean_position_rms_mm": float(np.mean(error_array[:, 0])),
            "p95_position_rms_mm": float(np.percentile(error_array[:, 0], 95)),
            "mean_edge_length_mae_mm": float(np.mean(error_array[:, 1])),
        }
    del example
    buffers.clear()
    gc.collect()
    return {
        "mode": "demo_same_substep",
        "reference_sweeps": 20,
        "frame_samples": args.frames,
        "baseline_rejected_last_substep_samples": rejected_samples,
        "errors": summaries,
        "test_final_passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--reference-sweeps", type=int, default=30)
    parser.add_argument("--sweeps", type=int, nargs="+", default=[12])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--graph-capture", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compare-demo",
        action="store_true",
        help="Compare the exact 12-sweep multilevel demo and contact-free candidate against ordinary 20 sweeps",
    )
    parser.add_argument(
        "--demo-modes",
        nargs="+",
        default=None,
        help="With --compare-demo, run only the named candidate labels in addition to reference20.",
    )
    parser.add_argument(
        "--same-substep",
        action="store_true",
        help="Compare candidates from identical substep states; do not measure speed",
    )
    args = parser.parse_args()
    if args.compare_demo:
        args.reference_sweeps = 20
    if min(args.frames, args.checkpoint_interval, args.reference_sweeps, *args.sweeps) < 1:
        parser.error("Frame counts, checkpoint interval, and sweep counts must be positive")
    if max(args.sweeps) > args.reference_sweeps:
        parser.error("Candidate sweeps must not exceed the reference/fallback sweeps")
    if not wp.get_device(args.device).is_cuda:
        parser.error("This benchmark compares CUDA multilevel paths; use the unit tests for CPU fallback")
    print(
        "BENCHMARK "
        + json.dumps(
            {
                "example": _EXAMPLE_MODULE,
                "warp_version": wp.__version__,
                "device": wp.get_device(args.device).name,
                "arguments": vars(args),
            }
        ),
        flush=True,
    )
    if args.same_substep:
        if args.compare_demo:
            print("RESULT " + json.dumps(_probe_demo_same_substep(args)), flush=True)
            return
        for sweeps in args.sweeps:
            print("RESULT " + json.dumps(_probe_same_substep(sweeps, args)), flush=True)
        return
    if args.compare_demo:
        metrics, reference, edges = _run_case("reference20", 20, None, args, demo_mode="reference20")
        print("RESULT " + json.dumps(metrics), flush=True)
        configurations = (
            ("reference20_repeat", 20, None, "reference20"),
            ("demo_baseline", 12, "graph", "baseline"),
            ("contact_free", 12, None, "contact-free"),
            ("cached13", 13, None, "cached13"),
            ("chebyshev8", 8, "graph", "chebyshev8"),
            ("cached_chebyshev8", 8, "graph", "cached-chebyshev8"),
            ("guarded_cached_chebyshev8", 8, "graph", "guarded-cached-chebyshev8"),
        )
        for label, sweeps, operator, mode in configurations:
            if args.demo_modes is not None and label not in args.demo_modes:
                continue
            metrics, checkpoints, candidate_edges = _run_case(label, sweeps, operator, args, demo_mode=mode)
            if not np.array_equal(edges, candidate_edges):
                raise ValueError(f"{label}: candidate topology differs from reference")
            metrics["checkpoint_errors"] = _compare(checkpoints, reference, edges)
            print("RESULT " + json.dumps(metrics), flush=True)
        return
    metrics, reference, edges = _run_case("reference", args.reference_sweeps, None, args)
    print("RESULT " + json.dumps(metrics), flush=True)
    configurations = [("reference_repeat", args.reference_sweeps, None)]
    configurations.extend(
        (f"{operator or 'plain'}_{sweeps}", sweeps, operator)
        for sweeps in args.sweeps
        for operator in (None, "graph", "galerkin")
    )
    for label, sweeps, operator in configurations:
        metrics, checkpoints, candidate_edges = _run_case(label, sweeps, operator, args)
        if not np.array_equal(edges, candidate_edges):
            raise ValueError(f"{label}: candidate topology differs from reference")
        metrics["checkpoint_errors"] = _compare(checkpoints, reference, edges)
        print("RESULT " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()

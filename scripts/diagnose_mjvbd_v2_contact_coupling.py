# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare full histories with identical coarse settings and contact policies."""

import argparse
import json
import math
import time
from unittest.mock import patch

import numpy as np
import warp as wp

import newton.examples
import newton.viewer
from newton._src.solvers.mjvbd_v2.vbd_soft import tri_mesh_collision as soft_collision
from newton.examples.mjvbdv2 import (
    example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00 as tshirt,
)
from newton.examples.mjvbdv2 import example_mjvbd_v2_cloth_twist as twist
from newton.solvers import SolverMJVBDV2
from scripts.diagnose_mjvbd_v2_intersections import intersections


def _legacy_edge_filters(n, edge_indices, adjacency, offsets):
    """Reproduce pre-reciprocity filtering for controlled diagnostic A/B runs."""
    if n <= 1:
        return None
    rows = []
    for a, b in edge_indices[:, 2:]:
        reached = set(soft_collision.leq_n_ring_vertices(a, edge_indices, n - 1, adjacency, offsets))
        reached.update(soft_collision.leq_n_ring_vertices(b, edge_indices, n - 1, adjacency, offsets))
        excluded = set()
        for vertex in reached - {a, b}:
            entries = adjacency[offsets[vertex] : offsets[vertex + 1]]
            excluded.update(entries[::2][entries[1::2] >= 2])
        for vertex in (a, b):
            excluded.difference_update(adjacency[offsets[vertex] : offsets[vertex + 1]][::2])
        rows.append(excluded)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("tshirt", "twist"), default="tshirt")
    parser.add_argument("--mode", choices=("fast", "uncoupled", "coupled"), required=True)
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--coarse-pcg", choices=("auto", "persistent"), default="auto")
    parser.add_argument("--legacy-edge-filter", action="store_true", help="Reproduce pre-fix built-in EE filters")
    args = parser.parse_args()
    if args.frames < 92:
        parser.error("--frames must be at least 92")
    wp.set_device("cuda:0")
    wp.config.log_level = wp.LOG_WARNING
    original = SolverMJVBDV2.__init__

    def configure(self, model, **kwargs):
        options = dict(kwargs["vbd_options"])
        if args.mode != "fast":
            options.update(
                particle_enable_multilevel_correction=True,
                particle_multilevel_operator="galerkin",
                particle_multilevel_cluster_size=8,
                particle_multilevel_coarse_iterations=8,
                particle_multilevel_relaxation=0.1,
                particle_multilevel_max_radius_fraction=0.05,
                particle_multilevel_min_residual_reduction=1e-4,
                particle_multilevel_max_clamp_fraction=0.5,
                # No extra ordinary fallback sweeps or scene-dependent tuning.
                particle_multilevel_fallback_iterations=None,
                particle_multilevel_checkpoints=None,
            )
        kwargs["vbd_options"] = options
        original(self, model, **kwargs)
        coarse = self.vbd_solver.particle_multilevel
        if coarse is not None:
            if args.coarse_pcg == "persistent":
                coarse.coarse_use_split_pcg = False
            coarse.contact_projection_enabled = args.mode == "coupled"
            # One common checkpoint before the final fast sweep, using the
            # scene's existing sweep budget for both policies.
            self.vbd_solver.particle_multilevel_checkpoints = (max(1, self.vbd_solver.iterations - 1),)

    scene = tshirt if args.scene == "tshirt" else twist
    scene_args = newton.examples.default_args(tshirt.Example.create_parser()) if scene is tshirt else None
    filter_builder = (
        _legacy_edge_filters if args.legacy_edge_filter else soft_collision.build_edge_n_ring_edge_collision_filter
    )
    with (
        patch.object(SolverMJVBDV2, "__init__", configure),
        patch.object(soft_collision, "build_edge_n_ring_edge_collision_filter", filter_builder),
    ):
        example = scene.Example(newton.viewer.ViewerNull(), scene_args)
    coarse = example.solver.vbd_solver.particle_multilevel

    def validate_final():
        if scene is not twist or coarse is None:
            example.test_final()
            return
        # The original twist test asserts the *default* multilevel-off policy.
        # This diagnostic intentionally overrides that policy; preserve its
        # backend, position, velocity and time-specific corner checks instead.
        features = example.solver.features
        assert features.backend == "pure_vbd"
        assert not (features.mujoco_solve_enabled or features.rigid_solve_enabled or features.tetrahedron_solve_enabled)
        if example.use_surface_fast:
            assert example.solver.vbd_solver.particle_enable_batched_jacobi
            assert example.solver.vbd_solver.particle_jacobi_batch_count == 2
        q, v = example.state_0.particle_q.numpy(), example.state_0.particle_qd.numpy()
        assert np.isfinite(q).all() and np.isfinite(v).all()
        assert (q >= [-0.6, -0.9, -0.6]).all() and (q <= [0.6, 0.9, 0.6]).all()
        assert np.max(np.abs(v)) < 1.5
        if math.isclose(example.sim_time, 5.0, abs_tol=0.5 * example.frame_dt):
            ratios = []
            for fixed, neighbor in example.corner_neighbor_pairs:
                delta = q[neighbor] - q[fixed]
                ratios.append(float(np.hypot(delta[0], delta[2])) / max(abs(float(delta[1])), 1e-12))
            assert max(ratios) < 1.2

    faces = example.model.tri_indices.numpy()
    edges = np.unique(np.sort(np.concatenate((faces[:, :2], faces[:, 1:], faces[:, ::2])), axis=1), axis=0)
    mass = example.model.particle_mass.numpy()
    previous_q = previous_delta = None
    speed, jitter, kinetic = [], [], []
    crossing, checkpoints = {}, {}
    for frame in range(1, args.frames + 1):
        example.step()
        if frame in (390, args.frames):
            crossing[frame] = len(intersections(example.state_0.particle_q.numpy(), faces, edges))
            if coarse is not None:
                checkpoints[frame] = {
                    "status": coarse.runtime_status.numpy().tolist(),
                    "metrics": coarse.runtime_metrics.numpy().tolist(),
                }
                if coarse.contact_projection is not None:
                    checkpoints[frame]["contact_count"] = int(coarse.contact_projection.data.count.numpy()[0])
                    checkpoints[frame]["overflow"] = int(coarse.contact_projection.data.overflow.numpy()[0])
                    checkpoints[frame]["unique_cluster_pairs"] = int(
                        coarse.contact_projection.data.edge_count.numpy()[0]
                    )
        if frame > args.frames - 92:
            q, v = example.state_0.particle_q.numpy(), example.state_0.particle_qd.numpy()
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(v)):
                raise ValueError(f"Nonfinite state at frame {frame}")
            if previous_q is not None:
                delta = q - previous_q
                if previous_delta is not None:
                    speed.append(float(np.sqrt(np.mean(np.sum(v * v, axis=1)))))
                    jitter.append(float(np.sqrt(np.mean(np.sum((delta - previous_delta) ** 2, axis=1)))))
                    kinetic.append(float(0.5 * np.sum(mass * np.sum(v * v, axis=1))))
                previous_delta = delta
            previous_q = q
    validate_final()
    wp.synchronize_device(example.model.device)
    start = time.perf_counter()
    for _ in range(120):
        example.step()
    wp.synchronize_device(example.model.device)
    elapsed = time.perf_counter() - start
    validate_final()
    print(
        json.dumps(
            {
                "scene": args.scene,
                "mode": args.mode,
                "coarse_pcg": args.coarse_pcg,
                "frames": args.frames,
                "sweeps": example.solver.vbd_solver.iterations,
                "legacy_edge_filter": args.legacy_edge_filter,
                "rms_speed": np.mean(speed),
                "second_difference_rms": np.mean(jitter),
                "kinetic_energy": np.mean(kinetic),
                "intersection_pairs": crossing,
                "coarse_checkpoints": checkpoints,
                "end_frame_ms": elapsed * 1000 / 120,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

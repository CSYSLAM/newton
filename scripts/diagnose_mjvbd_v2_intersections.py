# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure strict nonincident edge/triangle intersections during folding.

This diagnostic does not count coplanar overlap or touching. Counts represent
edge/face pairs, not independent holes. CPU readback makes timings unsuitable
for benchmarking. Run with the machine's existing CUDA guard left active.
"""

import argparse
import json
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.viewer
from newton.examples.mjvbdv2 import (
    example_cloth_mjvbd_v2_dexforce_bimanual_fold_tshirt_waic_house_final00 as scene,
)


def intersections(q, faces, edges):
    """Return strict segment/interior-triangle crossings in float64."""
    q = np.asarray(q, dtype=np.float64)
    triangles = q[faces]
    lo, hi = triangles.min(axis=1), triangles.max(axis=1)
    result = []
    for start in range(0, len(edges), 128):
        batch = edges[start : start + 128]
        segments = q[batch]
        overlap = np.all(segments.max(axis=1)[:, None] >= lo, axis=2)
        overlap &= np.all(segments.min(axis=1)[:, None] <= hi, axis=2)
        ei, fi = np.nonzero(overlap)
        independent = ~np.any(batch[ei, :, None] == faces[fi, None, :], axis=(1, 2))
        ei, fi = ei[independent], fi[independent]
        origin = segments[ei, 0]
        direction = segments[ei, 1] - origin
        a, b, c = triangles[fi, 0], triangles[fi, 1], triangles[fi, 2]
        ab, ac = b - a, c - a
        p = np.cross(direction, ac)
        determinant = np.einsum("ij,ij->i", ab, p)
        scale = np.linalg.norm(direction, axis=1) * np.linalg.norm(ab, axis=1) * np.linalg.norm(ac, axis=1)
        valid = np.abs(determinant) > np.maximum(1e-30, 1e-10 * scale)
        inverse = np.zeros_like(determinant)
        inverse[valid] = 1.0 / determinant[valid]
        offset = origin - a
        u = np.einsum("ij,ij->i", offset, p) * inverse
        cross = np.cross(offset, ab)
        v = np.einsum("ij,ij->i", direction, cross) * inverse
        t = np.einsum("ij,ij->i", ac, cross) * inverse
        eps = 1e-6
        hit = valid & (u > eps) & (v > eps) & (u + v < 1 - eps) & (t > eps) & (t < 1 - eps)
        result.extend(zip((ei[hit] + start).tolist(), fi[hit].tolist(), strict=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=("fast", "half", "no_cheb", "gs8", "gs20", "gs20_uncached", "uncached", "legacy20", "legacy30"),
        required=True,
    )
    parser.add_argument("--frames", type=int, default=900)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    wp.config.log_level = wp.LOG_WARNING
    wp.set_device("cuda:0")
    original = newton.solvers.SolverMJVBDV2.__init__

    def configure(self, model, **kwargs):
        options = dict(kwargs["vbd_options"])
        options.update(particle_enable_multilevel_correction=False, particle_multilevel_checkpoints=None)
        if args.case in ("legacy20", "legacy30"):
            kwargs["vbd_preset"] = None
            options.update(
                iterations=30 if args.case == "legacy30" else 20,
                particle_vertex_contact_buffer_size=16,
                particle_edge_contact_buffer_size=20,
                rigid_body_particle_contact_buffer_size=256,
                particle_collision_detection_interval=-1,
            )
        if args.case == "half":
            options["particle_jacobi_relaxation"] = 0.5
        if args.case in ("no_cheb", "gs8", "gs20", "gs20_uncached", "uncached"):
            options["particle_chebyshev_spectral_radius"] = None
        if args.case in ("gs8", "gs20", "gs20_uncached", "uncached"):
            options["particle_enable_batched_jacobi"] = False
        if args.case in ("gs20", "gs20_uncached"):
            options["iterations"] = 20
        if args.case in ("uncached", "gs20_uncached"):
            options.update(particle_enable_surface_cache=False, particle_enable_truncation_cache=False)
        kwargs["vbd_options"] = options
        original(self, model, **kwargs)

    with patch.object(newton.solvers.SolverMJVBDV2, "__init__", configure):
        example = scene.Example(newton.viewer.ViewerNull(), newton.examples.default_args(scene.Example.create_parser()))
    faces = example.model.tri_indices.numpy()
    edges = np.unique(np.sort(np.concatenate((faces[:, :2], faces[:, 1:], faces[:, ::2])), axis=1), axis=0)
    for frame in range(args.frames + 1):
        if frame in (0, 120, 240, 390, 600, 900) or frame == args.frames:
            pairs = intersections(example.state_0.particle_q.numpy(), faces, edges)
            detector = example.solver.vbd_solver.trimesh_collision_detector
            overflow = {}
            for prefix in ("vertex_colliding_triangles", "edge_colliding_edges"):
                if frame > 0:
                    counts = getattr(detector, prefix + "_count").numpy()
                    capacities = getattr(detector, prefix + "_buffer_sizes").numpy()
                    overflow[prefix] = int(np.count_nonzero(counts > capacities))
            print(
                json.dumps(
                    {
                        "case": args.case,
                        "frame": frame,
                        "edge_face_crossings": len(pairs),
                        "first_pairs": pairs[:8],
                        "overflow_rows_at_sample": overflow,
                    }
                ),
                flush=True,
            )
        if frame < args.frames:
            example.step()


if __name__ == "__main__":
    main()

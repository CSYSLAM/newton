# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit saved twist poses with independent float64 closest-feature distances."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton


def segment_distance(p, a, b):
    direction = b - a
    t = np.clip(np.sum((p - a) * direction, axis=1) / np.maximum(np.sum(direction**2, axis=1), 1e-100), 0, 1)
    return np.sum((p - a - t[:, None] * direction) ** 2, axis=1)


def pair_distances(p, ids, kinds):
    """Evaluate all edge/face boundary features, then valid interior minima."""
    x = p[ids[kinds == 0]]
    v, a, b, c = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    n = np.cross(b - a, c - a)
    n2 = np.sum(n * n, axis=1)
    q = v - n * (np.sum(n * (v - a), axis=1) / np.maximum(n2, 1e-100))[:, None]
    inside = n2 > 0
    for u, w in ((a, b), (b, c), (c, a)):
        inside &= np.sum(n * np.cross(w - u, q - u), axis=1) >= 0
    d = np.minimum.reduce([segment_distance(v, a, b), segment_distance(v, b, c), segment_distance(v, c, a)])
    d[inside] = np.sum((v - q) ** 2, axis=1)[inside]
    x = p[ids[kinds == 1]]
    a, b, c, e = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    u, w, r = b - a, e - c, c - a
    n = np.cross(u, w)
    n2 = np.sum(n * n, axis=1)
    s = np.sum(np.cross(r, w) * n, axis=1) / np.maximum(n2, 1e-100)
    t = np.sum(np.cross(r, u) * n, axis=1) / np.maximum(n2, 1e-100)
    inside = (n2 > 0) & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    ee = np.minimum.reduce(
        [segment_distance(a, c, e), segment_distance(b, c, e), segment_distance(c, a, b), segment_distance(e, a, b)]
    )
    ee[inside] = np.sum((a + s[:, None] * u - c - t[:, None] * w) ** 2, axis=1)[inside]
    return np.sqrt(np.concatenate((d, ee)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("poses", type=Path, nargs="+")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    reference = np.load(args.reference) if args.reference else None
    for path in args.poses:
        z = np.load(path)
        p = z["positions"][-1] if z["positions"].ndim == 3 else z["positions"]
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_cloth_mesh(
            pos=wp.vec3(0),
            rot=wp.quat_identity(),
            scale=1.0,
            vertices=[wp.vec3(v) for v in z["rest"]],
            indices=z["faces"].flatten(),
            vel=wp.vec3(0),
            density=0.2,
            tri_kd=0.0,
            edge_kd=0.0,
        )
        model = builder.finalize()
        solver = newton.solvers.SolverXPBD(
            model, particle_fem=True, particle_self_contact_radius=0.002, particle_self_contact_margin=0.0035
        )
        fem = solver._fem
        fem.pos.assign(p)
        fem._detect()
        if int(fem.count.numpy()[0]) > len(fem.pairs):
            raise RuntimeError("Diagnostic candidate capacity overflow")
        count = int(fem.count.numpy()[0])
        d = pair_distances(p.astype(np.float64), fem.pairs.numpy()[:count], fem.kinds.numpy()[:count])
        report = {
            "pose": str(path),
            "pairs": count,
            "min_distance_m": float(d.min()),
            "under_1um": int(sum(d < 1e-6)),
            "under_0_1mm": int(sum(d < 1e-4)),
            "distance_quantiles_m": np.quantile(d, [0, 0.001, 0.01, 0.1, 0.5, 0.9, 1]).tolist(),
        }
        if reference is not None:
            np.testing.assert_array_equal(z["faces"], reference["faces"])
            np.testing.assert_allclose(z["rest"], reference["rest"], atol=1e-7)
            error = np.linalg.norm(p - reference["positions"][-1], axis=1)
            report.update(
                position_rmse_m=float(np.sqrt(np.mean(error**2))), position_p95_m=float(np.quantile(error, 0.95))
            )
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()

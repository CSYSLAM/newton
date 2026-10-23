# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shadow contact-space sweeps with the existing body-space sweeps.

Force/impulse comparisons use 1e-5; twists use the existing 1e-4 coupled
velocity acceptance scale. Recovery twists are diagnostic, whereas the
first pass also supplies the material solve. Record actual maxima
separately and do not interpret verification overhead as performance.
"""

import argparse
import json
from pathlib import Path

import warp as wp

from newton._src.solvers.mjmpm.environment import Environment
from newton.examples.mjmpm.example_mjmpm_w1_rice_cooker import Example
from newton.viewer import ViewerNull


@wp.kernel
def compare(a: wp.array[float], b: wp.array[float], tolerance: float, stats: wp.array[int], error: wp.array[float]):
    i = wp.tid()
    difference = wp.abs(a[i] - b[i])
    if i == 0:
        wp.atomic_add(stats, 1, 1)
    wp.atomic_max(error, 0, difference)
    if not wp.isfinite(a[i]) or not wp.isfinite(b[i]) or difference > tolerance + 1.0e-5 * wp.abs(b[i]):
        wp.atomic_add(stats, 0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("recovery", "all"), default="recovery")
    parser.add_argument("--frames", type=int, default=990)
    opts = parser.parse_args()
    wp.set_device("cuda:0")
    args = Example.create_parser().parse_args(["--viewer", "null"])
    args.test = True
    viewer = ViewerNull(num_frames=opts.frames)
    example = Example(viewer, args)
    env = example.solver.environment
    env._use_contact_space = True
    env._contact_space_recovery_only = opts.mode == "recovery"
    initial = [wp.empty_like(v) for v in (env.work.body_qd, env.contact_impulses, env.rigid_impulses)]
    actual = [wp.empty_like(v) for v in initial]
    stats = [wp.zeros(2, dtype=int) for _ in initial]
    errors = [wp.zeros(1, dtype=float) for _ in initial]
    original = Environment._solve_contacts_gauss_seidel

    def verified(self, work):
        if self is not env or (env._contact_space_recovery_only and not env._recovering_contacts):
            return original(self, work)
        outputs = (work.body_qd, env.contact_impulses, env.rigid_impulses)
        for a, b in zip(initial, outputs, strict=True):
            wp.copy(a, b)
        original(self, work)
        for a, b in zip(actual, outputs, strict=True):
            wp.copy(a, b)
        for a, b in zip(outputs, initial, strict=True):
            wp.copy(a, b)
        self._solve_contacts_gauss_seidel_sweeps(work)
        for k, (a, b) in enumerate(zip(outputs, actual, strict=True)):
            wp.launch(
                compare,
                dim=a.size * wp.types.type_size_in_bytes(a.dtype) // 4,
                inputs=[
                    a.view(float).flatten(),
                    b.view(float).flatten(),
                    1.0e-4 if k == 0 else 1.0e-5,
                    stats[k],
                    errors[k],
                ],
            )

    Environment._solve_contacts_gauss_seidel = verified
    try:
        for frame in range(opts.frames):
            example.step()
            example.test_post_step()
            if frame % 120 == 0:
                print(f"Progress {frame + 1}/{opts.frames}", flush=True)
        if opts.frames >= 990:
            example.test_final()
        report = {
            "mode": opts.mode,
            "frames": opts.frames,
            "comparisons": [v.numpy().tolist() for v in stats],
            "max_absolute_errors": [float(v.numpy()[0]) for v in errors],
            "path_counts": env._contact_space.stats.numpy().tolist(),
            "absolute_tolerances": [1.0e-4, 1.0e-5, 1.0e-5],
        }
        opts.output.parent.mkdir(parents=True, exist_ok=True)
        opts.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report), flush=True)
        assert all(row[0] == 0 and row[1] > 0 for row in report["comparisons"]), report
    finally:
        Environment._solve_contacts_gauss_seidel = original
        viewer.close()


if __name__ == "__main__":
    main()

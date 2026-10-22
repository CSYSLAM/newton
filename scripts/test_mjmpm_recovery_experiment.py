# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run existing MJMPM acceptance tests with process-local recovery warm starts.

This is an experiment runner, not an installed solver option or a mock.
"""

import json
import sys
import unittest
from pathlib import Path

import warp as wp

from newton._src.solvers.mjmpm.environment import Environment


def main():
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    original_init = Environment.__init__
    counters = {"environments": 0, "initial_recordings": 0, "recovery_recordings": 0}

    def init(env, *args, **kwargs):
        original_init(env, *args, **kwargs)
        if not env._constrained_material or env.rigid_solver != "gauss_seidel":
            return
        counters["environments"] += 1
        first_impulse = wp.zeros_like(env.contact_impulses)
        sweep_original = env._solve_contacts_gauss_seidel
        finish_original = env._finish_constrained_material
        recovering = False

        def sweep(work):
            iterations = env.rigid_iterations
            if recovering:
                wp.copy(env.contact_impulses, first_impulse)
                env.rigid_iterations = min(20, iterations)
                counters["recovery_recordings"] += 1
            else:
                counters["initial_recordings"] += 1
            try:
                sweep_original(work)
            finally:
                env.rigid_iterations = iterations
            if not recovering:
                wp.copy(first_impulse, env.contact_impulses)

        def finish(work, dt):
            nonlocal recovering
            recovering = True
            try:
                return finish_original(work, dt)
            finally:
                recovering = False

        env._solve_contacts_gauss_seidel = sweep
        env._finish_constrained_material = finish

    Environment.__init__ = init
    try:
        suite = unittest.defaultTestLoader.discover("newton/tests/mjmpm", top_level_dir=".")
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        output.write_text(
            json.dumps(
                {
                    "tests_run": result.testsRun,
                    "passed": result.wasSuccessful(),
                    "failures": [(str(test), detail) for test, detail in result.failures],
                    "errors": [(str(test), detail) for test, detail in result.errors],
                    "skipped": [(str(test), reason) for test, reason in result.skipped],
                    "instrumented": counters,
                    "notes": "Process-local experimental recovery warm start, capped at min(20, configured sweeps). All existing tests exercise real solvers. Recording counters count eager execution or graph construction, not graph replays.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0 if result.wasSuccessful() else 1
    finally:
        Environment.__init__ = original_init


if __name__ == "__main__":
    sys.exit(main())

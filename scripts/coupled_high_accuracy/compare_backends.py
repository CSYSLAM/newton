# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one scene with the native and the Warp solver core and compare the states step by step.

Each backend runs in its own process (the core backend is fixed per process by ``NEWTON_CHA_BACKEND``) and records
``particle_q``, ``particle_qd``, ``body_q`` and ``body_qd`` after every step. The report lists the largest absolute
difference per field and the first step at which it exceeds the tolerance.

    uv run scripts/coupled_high_accuracy/compare_backends.py springs --steps 20
    uv run scripts/coupled_high_accuracy/compare_backends.py --list
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIELDS = ("particle_q", "particle_qd", "body_q", "body_qd")


def record(scene_name: str, steps: int, options: dict[str, str], output: Path) -> None:
    sys.path.insert(0, str(HERE))
    import scenes  # noqa: PLC0415

    frames = {field: [] for field in FIELDS}
    for state in scenes.SCENES[scene_name](steps, options):
        for field in FIELDS:
            array = getattr(state, field, None)
            frames[field].append(np.zeros(0) if array is None else array.numpy().copy())
    np.savez(output, **{field: np.stack(values) for field, values in frames.items() if values})


def run_backend(backend: str, scene: str, steps: int, options: list[str], output: Path) -> None:
    environment = dict(os.environ, NEWTON_CHA_BACKEND=backend)
    command = [sys.executable, __file__, scene, "--steps", str(steps), "--record", str(output)]
    for option in options:
        command += ["--option", option]
    subprocess.run(command, check=True, env=environment)


def main() -> int:
    sys.path.insert(0, str(HERE))
    import scenes  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", nargs="?", choices=sorted(scenes.SCENES))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--option", action="append", default=[], help="Scene option name=value (repeatable).")
    parser.add_argument("--list", action="store_true", help="List the scenes.")
    parser.add_argument(
        "--backends",
        default="native,warp",
        help="Two comma-separated backends, e.g. native,native for run-to-run noise.",
    )
    parser.add_argument("--record", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.list or args.scene is None:
        for name in sorted(scenes.SCENES):
            print(f"{name}: {(scenes.SCENES[name].__doc__ or '').strip().splitlines()[0]}")
        return 0
    options = dict(option.split("=", 1) for option in args.option)
    if args.record:
        record(args.scene, args.steps, options, args.record)
        return 0

    with tempfile.TemporaryDirectory() as directory:
        results = []
        for index, backend in enumerate(args.backends.split(",")):
            path = Path(directory) / f"{index}-{backend}.npz"
            run_backend(backend, args.scene, args.steps, args.option, path)
            results.append(dict(np.load(path)))
    failed = False
    for field in FIELDS:
        native, warp = results[0].get(field), results[1].get(field)
        if native is None or native.size == 0:
            continue
        difference = np.abs(native.astype(np.float64) - warp.astype(np.float64))
        per_step = difference.reshape(len(difference), -1).max(axis=1)
        worst = float(per_step.max())
        over = np.flatnonzero(per_step > args.tolerance)
        first = f"first step over tolerance: {int(over[0])}" if len(over) else "within tolerance"
        scale = float(np.abs(native).max())
        print(f"{field:12s} max |diff| {worst:.3e} (max |value| {scale:.3e}); {first}")
        failed |= bool(len(over)) or not np.isfinite(warp).all()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

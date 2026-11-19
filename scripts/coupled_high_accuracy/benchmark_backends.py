# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Time solver steps of comparison scenes on the native and Warp backends.

Each backend runs in its own process. A scene generator is advanced through warm-up steps (kernel compilation and
first-touch allocation), then the remaining steps are timed with a device synchronisation at both ends.

    python scripts/coupled_high_accuracy/benchmark_backends.py cloth boxes_on_ground --steps 200
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def measure(scene_name: str, steps: int, warmup: int, options: dict[str, str]) -> None:
    sys.path.insert(0, str(Path(__file__).parent))
    import scenes  # noqa: PLC0415
    import warp as wp  # noqa: PLC0415

    states = scenes.SCENES[scene_name](warmup + steps, options)
    for _ in range(warmup):
        next(states)
    wp.synchronize()
    start = time.perf_counter()
    for _ in states:
        pass
    wp.synchronize()
    print(f"{(time.perf_counter() - start) / steps * 1.0e3:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes", nargs="*")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--option", action="append", default=[], help="scene option key=value")
    parser.add_argument("--measure", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    options = dict(item.split("=", 1) for item in args.option)
    if args.measure:
        measure(args.scenes[0], args.steps, args.warmup, options)
        return 0
    print(f"{'scene':<28}{'native ms':>12}{'warp ms':>12}{'ratio':>8}")
    for scene_name in args.scenes:
        times = []
        for backend in ("native", "warp"):
            command = [sys.executable, __file__, scene_name, "--measure", "--steps", str(args.steps)]
            command += ["--warmup", str(args.warmup)] + [f"--option={item}" for item in args.option]
            result = subprocess.run(
                command, env=dict(os.environ, NEWTON_CHA_BACKEND=backend), capture_output=True, text=True, check=True
            )
            times.append(float(result.stdout.strip().splitlines()[-1]))
        print(f"{scene_name:<28}{times[0]:>12.3f}{times[1]:>12.3f}{times[1] / times[0]:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

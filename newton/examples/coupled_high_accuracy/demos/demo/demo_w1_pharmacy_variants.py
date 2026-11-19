"""New pharmacy example: four carton constructions, two successive W1 dual-hand comparisons."""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "w1_pharmacy_variants"
sys.path.insert(0, str(HERE))

from catalog import CARTONS, DURATION, PAIRS
from common import ROOT
from variant_simulation import Simulation, live, record


def worker_command(args, pair):
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--pair",
        str(pair + 1),
        "--viewer" if args.viewer else "--no-viewer",
        "--seconds",
        str(args.seconds),
        "--substeps",
        str(args.substeps),
        "--iterations",
        str(args.iterations),
        "--linear-iterations",
        str(args.linear_iterations),
        "--force-scale",
        str(args.force_scale),
        "--output",
        str(args.output.resolve()),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair", choices=("all", "1", "2"), default="all")
    p.add_argument("--seconds", type=float, default=DURATION)
    p.add_argument("--substeps", type=int, default=8)
    p.add_argument("--iterations", type=int, default=4)
    p.add_argument("--linear-iterations", type=int, default=120)
    p.add_argument("--force-scale", type=float, default=1.0)
    p.add_argument("--output", type=Path, default=ROOT / "build/w1-pharmacy-variants")
    p.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open the interactive viewer by default; --no-viewer records headlessly.",
    )
    p.add_argument("--close-on-end", action="store_true")
    p.add_argument("--render", action="store_true", help="Record and render headlessly instead of opening the viewer.")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.render:
        args.viewer = False
    if not 0 < args.seconds <= DURATION:
        p.error(f"--seconds must be in (0,{DURATION}]")
    selected = [0, 1] if args.pair == "all" else [int(args.pair) - 1]
    if args.viewer:
        if args.worker:
            sim = Simulation([CARTONS[i] for i in PAIRS[selected[0]]], args)
            raise SystemExit(0 if live(sim, args, args.close_on_end) else 2)
        # Isolate OpenGL/CUDA contexts and captured kernel modules between the two different meshes.
        for k, pair in enumerate(selected):
            command = worker_command(args, pair)
            if k < len(selected) - 1 or args.close_on_end:
                command.append("--close-on-end")
            result = subprocess.run(command)
            if result.returncode == 2:
                return
            result.check_returncode()
        return
    if args.worker:
        pair = selected[0]
        sim = Simulation([CARTONS[i] for i in PAIRS[pair]], args)
        record(sim, args, args.output / f"pair_{pair + 1}")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for pair in selected:
        command = worker_command(args, pair)
        print("Simulating pair", pair + 1, flush=True)
        with (args.output / f"pair_{pair + 1}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if args.render:
        subprocess.run(
            [sys.executable, str(HERE / "make_video.py"), "--output", str(args.output.resolve())], check=True
        )


if __name__ == "__main__":
    main()

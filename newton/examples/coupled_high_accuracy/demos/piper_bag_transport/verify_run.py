"""Verify that the scene, motor-input, and motor-feedback movies come from one exported simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
sys.path.insert(0, str(DIRECTORY.parents[1]))

if __package__:
    from .run_artifacts import verify_bundle, write_json
else:
    from run_artifacts import verify_bundle, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DIRECTORY / "outputs")
    parser.add_argument("--scene-video", default="piper_bag_transport.mp4")
    args = parser.parse_args()
    report = verify_bundle(args.output_dir, args.scene_video)
    write_json(args.output_dir / "run_validation.json", report)
    print(f"Verified all three videos against simulation {report['run_id']}", flush=True)


if __name__ == "__main__":
    main()

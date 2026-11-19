"""Check a recorded W1 snack-basket run: the basket is held up, and the bag is carried and ends inside it."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scene
from snack_geometry import six_volumes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=scene.ROOT / "build/w1-snack-basket")
    folder = parser.parse_args(argv).output
    run = json.loads((folder / "run.json").read_text())
    with np.load(folder / "trajectory.npz") as data:
        particles, bodies = data["particles"], data["bodies"]
        (film_start, film_end), (core_start, core_end) = data["group_ranges"]
        tets, rest = data["tets"], data["rest_volumes"]
        basket = int(data["basket"])
    fps = run["fps"]

    def at(seconds):
        return min(round(seconds * fps), len(particles) - 1)

    film = particles[:, film_start:film_end]
    basket_pose = bodies[:, basket]
    rotation = Rotation.from_quat(basket_pose[:, 3:])
    tilt = np.degrees(np.arccos(np.clip(rotation.apply((0, 0, 1))[:, 2], -1, 1)))
    final = len(particles) - 1
    # Final bag position in the basket body frame (the scaled asset frame).
    local = rotation[final].inv().apply(film[final] - basket_pose[final, :3])
    min_tet = min(float((six_volumes(p[core_start:core_end], tets) / rest).min()) for p in particles[:: fps // 2])
    carry = at((scene.BAG_LIFT_END + scene.BAG_CARRY_END) / 2)
    held = at((scene.BASKET_LIFT_END + scene.BASKET_LOWER_START) / 2)
    checks = dict(
        finite=bool(np.isfinite(particles).all() and np.isfinite(bodies).all()),
        bag_carried_clear_of_table=float(film[carry, :, 2].min() - scene.TABLE_Z) > 0.15,
        basket_lifted=float(basket_pose[held, 2] - basket_pose[0, 2]) > 0.03,
        basket_hangs_nearly_level=float(tilt[at(scene.BASKET_LIFT_END) : at(scene.BASKET_LOWER_START)].max()) < 8,
        bag_inside_basket_footprint=bool(
            (np.abs(local[:, 0]) < scene.BASKET_RIM_HALF[0]).all()
            and (np.abs(local[:, 1]) < scene.BASKET_RIM_HALF[1]).all()
        ),
        bag_below_rim=bool(local[:, 2].max() < scene.BASKET_RIM_Z),
        bag_on_basket_floor=bool(local[:, 2].min() > -0.003),
        basket_back_on_table=abs(float(basket_pose[final, 2] - basket_pose[0, 2])) < 0.004 and tilt[final] < 2,
        core_tetrahedra_not_inverted=min_tet > 0.1,
    )
    checks = {name: bool(value) for name, value in checks.items()}
    report = dict(
        checks=checks,
        passed=all(checks.values()),
        basket_lift_m=float(basket_pose[held, 2] - basket_pose[0, 2]),
        max_hanging_tilt_deg=float(tilt[at(scene.BASKET_LIFT_END) : at(scene.BASKET_LOWER_START)].max()),
        final_bag_local_min_m=local.min(0).round(4).tolist(),
        final_bag_local_max_m=local.max(0).round(4).tolist(),
        min_core_tet_ratio=min_tet,
    )
    (folder / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Validation failed")


if __name__ == "__main__":
    main()

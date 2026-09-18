# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Export an independent handle-free asset, preserving every paper-shell vertex."""

import shutil
from pathlib import Path

import numpy as np


def build():
    directory = Path(__file__).resolve().parent
    source = directory.with_name("w1_paper_bag")
    with np.load(source / "bag.npz") as data:
        count, faces = int(data["paper_count"]), int(data["paper_faces"])
        values = {key: data[key].copy() for key in data.files}
    values["placement_bounds"] = np.array([values["vertices"].min(0), values["vertices"].max(0)])
    values["vertices"] = values["vertices"][:count]
    patches = []
    # Handle roots replaced four wall quads in the original asset. Close these
    # openings when removing the ribbons, retaining the surrounding vertices.
    for key in ("handle_0", "handle_1"):
        for root in (values[key][:4], values[key][-4:]):
            for face in (root[[0, 1, 2]], root[[0, 2, 3]]):
                points = values["vertices"][face]
                normal = np.cross(points[1] - points[0], points[2] - points[0])
                patches.append(face if normal[0] * points[:, 0].mean() > 0 else face[::-1])
    values["faces"] = np.concatenate([values["faces"][:faces], np.asarray(patches)])
    values["paper_faces"] = np.asarray(len(values["faces"]), dtype=values["paper_faces"].dtype)
    for key in ("handle_0", "handle_1"):
        values[key] = np.empty(0, dtype=values[key].dtype)
    np.savez_compressed(directory / "bag.npz", **values)
    for name in ("snacks.npz", "snacks.json", "kraft.png", "dimensions.json"):
        shutil.copyfile(source / name, directory / name)
    with (directory / "bag.obj").open("w") as stream:
        stream.write("# Handle-free W1 paper shell, meters\no W1_paper_bag_no_handles\n")
        for x, y, z in values["vertices"]:
            stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in values["faces"] + 1:
            stream.write(f"f {a} {b} {c}\n")
    print(f"Exported {count} paper vertices and {len(values['faces'])} triangles; no handle geometry")


if __name__ == "__main__":
    build()

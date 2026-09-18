# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build thick, downward U-shaped rope handles on an independent paper bag."""

import shutil
from collections import defaultdict, deque
from itertools import pairwise
from pathlib import Path

import numpy as np


def build():
    """Keep the original paper shell and attach two rounded, subtly twisted tubes."""
    directory = Path(__file__).resolve().parent
    source = directory.with_name("w1_paper_bag")
    with np.load(source / "bag.npz") as data:
        values = {key: data[key].copy() for key in data.files}
    paper_count, paper_faces = int(values["paper_count"]), int(values["paper_faces"])
    values["placement_bounds"] = np.array([values["vertices"].min(0), values["vertices"].max(0)])
    vertices = values["vertices"][:paper_count].tolist()
    faces = values["faces"][:paper_faces].tolist()
    segments, sides, radius = 24, 8, 0.0045

    def bridge(root, ring):
        # Align each four-corner paper hole with alternate tube vertices.
        candidates = [np.roll(root[::direction], shift) for direction in (1, -1) for shift in range(4)]
        points = np.asarray(vertices)
        root = min(candidates, key=lambda ids: np.sum((points[ids] - points[ring[::2]]) ** 2))
        for j in range(4):
            a, b = int(root[j]), int(root[(j + 1) % 4])
            c, d, e = ring[2 * j], ring[(2 * j + 1) % sides], ring[(2 * j + 2) % sides]
            faces.extend(((a, c, d), (a, d, e), (a, e, b)))

    def center(t, centers):
        sign = np.sign(centers[0][0])
        weight = (1 - np.cos(np.pi * t)) / 2
        point = (1 - weight) * centers[0] + weight * centers[1]
        point[0] += sign * 0.009 * np.sin(np.pi * t) ** 0.3
        point[2] -= 0.105 * np.sin(np.pi * t) ** 1.3
        return point

    for key in ("handle_0", "handle_1"):
        roots = (values[key][:4], values[key][-4:])
        centers = [np.asarray(vertices)[root].mean(axis=0) for root in roots]
        rows = []
        axis = np.array((0.0, 1.0, 0.0))
        for k in range(1, segments):
            t = k / segments
            tangent = center(t + 1e-5, centers) - center(t - 1e-5, centers)
            tangent /= np.linalg.norm(tangent)
            axis -= tangent * np.dot(axis, tangent)
            axis /= np.linalg.norm(axis)
            second = np.cross(tangent, axis)
            row = []
            for j in range(sides):
                angle = 2 * np.pi * j / sides
                # Three shallow helical lobes suggest twisted jute without
                # separate strands, tiny collision features, or extra particles.
                r = radius * (1 + 0.06 * np.cos(3 * angle - 12 * np.pi * t))
                point = center(t, centers) + r * (np.cos(angle) * axis + np.sin(angle) * second)
                row.append(len(vertices))
                vertices.append(point.tolist())
            rows.append(row)
        bridge(roots[0], rows[0])
        for a, b in pairwise(rows):
            for j in range(sides):
                n = (j + 1) % sides
                faces.extend(((a[j], a[n], b[n]), (a[j], b[n], b[j])))
        bridge(roots[1], rows[-1])
        values[key] = np.concatenate((roots[0], np.asarray(rows).ravel(), roots[1])).astype(np.int32)
        values[key + "_rings"] = np.asarray(rows, dtype=np.int32)

    # Orient new tube faces consistently with the unchanged paper shell.
    adjacent = defaultdict(list)
    for index, face in enumerate(faces):
        for a, b in zip(face, (*face[1:], face[0]), strict=True):
            adjacent[tuple(sorted((a, b)))].append(index)
    seen = set(range(paper_faces))
    pending = deque(range(paper_faces))
    while pending:
        index = pending.popleft()
        face = faces[index]
        for a, b in zip(face, (*face[1:], face[0]), strict=True):
            for neighbor in adjacent[tuple(sorted((a, b)))]:
                if neighbor in seen:
                    continue
                other = faces[neighbor]
                if any(c == a and d == b for c, d in zip(other, (*other[1:], other[0]), strict=True)):
                    faces[neighbor] = other[::-1]
                seen.add(neighbor)
                pending.append(neighbor)
    if len(seen) != len(faces):
        raise ValueError("Rope handles must connect to the paper shell")
    values["vertices"] = np.asarray(vertices, dtype=np.float32)
    values["faces"] = np.asarray(faces, dtype=np.int32)
    values["rope_radius"] = np.asarray(radius, dtype=np.float32)
    np.savez_compressed(directory / "bag.npz", **values)
    for name in ("snacks.npz", "snacks.json", "kraft.png", "dimensions.json"):
        shutil.copyfile(source / name, directory / name)
    with (directory / "bag.obj").open("w") as stream:
        stream.write("# W1 bag with downward rope handles, meters\no W1_rope_handle_bag\n")
        for x, y, z in vertices:
            stream.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for a, b, c in values["faces"] + 1:
            stream.write(f"f {a} {b} {c}\n")
    print(f"Exported {len(vertices)} vertices and {len(faces)} triangles with two 9 mm rope handles")


if __name__ == "__main__":
    build()

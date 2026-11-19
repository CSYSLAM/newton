"""Preserve the W1 asset's convex decomposition instead of filling the open gripper with one hull."""

from pathlib import Path

import numpy as np

import newton

ASSET = Path(__file__).resolve().parents[2] / "assets/w1/robot/w1-030/collision"
CACHE = Path(__file__).resolve().parents[2] / "build/shopping-gripper-sdf"


def obj_parts(path):
    vertices = []
    groups = []
    faces = []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            vertices.append(list(map(float, line.split()[1:4])))
        elif line.startswith("o "):
            if faces:
                groups.append(faces)
                faces = []
        elif line.startswith("f "):
            ids = [int(s.split("/")[0]) - 1 for s in line.split()[1:]]
            faces.extend([ids[0], ids[i], ids[i + 1]] for i in range(1, len(ids) - 1))
    if faces:
        groups.append(faces)
    vertices = np.asarray(vertices, np.float32)
    for faces in groups:
        ids, indices = np.unique(np.asarray(faces), return_inverse=True)
        yield vertices[ids], indices.reshape(-1, 3)


def install(builder):
    robot_shapes = [i for i, b in enumerate(builder.shape_body) if b >= 0]
    created = []
    for side, indices in [("right", (27, 29, 31)), ("left", (51, 53, 55))]:
        for shape, filename in zip(indices, ("gripper_base_v100_acd.obj", "link7_acd.obj", "link8_acd.obj")):
            body = builder.shape_body[shape]
            pose = builder.shape_transform[shape]
            scale = np.asarray(builder.shape_scale[shape])
            builder.shape_flags[shape] = 0
            for part, (vertices, faces) in enumerate(obj_parts(ASSET / f"{side}_hand" / filename)):
                mesh = newton.Mesh(vertices * scale, faces.ravel(), maxhullvert=64).compute_convex_hull()
                mesh.build_sdf(max_resolution=32, narrow_band_range=(-0.008, 0.008), cache_dir=CACHE)
                cfg = builder.default_shape_cfg.copy()
                cfg.density = 0
                cfg.mu = 0.3 if "base" in filename else 0.45
                cfg.margin = 0.0004
                cfg.is_visible = False
                new_shape = builder.add_shape_convex_hull(
                    body, mesh=mesh, xform=pose, cfg=cfg, label=f"{side}_{filename}_{part}"
                )
                created.append(new_shape)
    # Preserve source exclusions first; enable_cross_arm_collisions restores the opposite-arm pairs.
    for shape in created:
        for other in robot_shapes + created:
            if shape != other:
                builder.add_shape_collision_filter_pair(shape, other)
    print("Restored gripper convex components:", len(created), flush=True)
    return len(created)


def enable_cross_arm_collisions(builder):
    """Retain adjacent-link exclusions while restoring contact between the two arms."""

    def side(shape):
        body = builder.shape_body[shape]
        if body < 0:
            return None
        name = builder.body_label[body].rsplit("/", 1)[-1]
        if "_r_" in name or name.startswith("right_"):
            return "right"
        if "_l_" in name or name.startswith("left_"):
            return "left"
        return None

    sides = [side(i) for i in range(builder.shape_count)]
    # The generic 10 cm speculative gap creates tens of thousands of irrelevant contacts
    # between the decomposed hands. Five millimetres covers several 480 Hz integration steps.
    for shape, side_name in enumerate(sides):
        if side_name is not None:
            builder.shape_gap[shape] = 0.005
    builder.rigid_gap = 0.005

    def cross(a, b):
        return sides[a] is not None and sides[b] is not None and sides[a] != sides[b]

    pairs = builder.shape_collision_filter_pairs
    builder.shape_collision_filter_pairs = [(a, b) for a, b in pairs if not cross(a, b)]
    restored = len(pairs) - len(builder.shape_collision_filter_pairs)
    print("Restored cross-arm collision pairs:", restored, flush=True)
    return restored

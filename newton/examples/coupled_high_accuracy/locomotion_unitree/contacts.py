"""Collision-shape classification for collapsed-link Go2 assets."""

from __future__ import annotations

from dataclasses import dataclass

LEGS = ("FR", "FL", "RR", "RL")


def _basename(label: str) -> str:
    return label.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class ShapeClassification:
    shape_to_foot: tuple[int, ...]
    shape_to_undesired: tuple[int, ...]
    shape_to_base_env: tuple[int, ...]
    undesired_per_env: int


def classify_shapes(
    *,
    shape_labels: list[str],
    shape_bodies: list[int],
    body_labels: list[str],
    body_per_env: int,
    num_envs: int,
) -> ShapeClassification:
    body_to_undesired = [-1] * len(body_labels)
    undesired_per_env: int | None = None
    for env in range(num_envs):
        selected: list[int] = []
        start = env * body_per_env
        for body in range(start, start + body_per_env):
            name = _basename(body_labels[body])
            if name.startswith("Head_") or name.endswith(("_hip", "_thigh", "_calf")):
                selected.append(body)
        if undesired_per_env is None:
            undesired_per_env = len(selected)
        elif len(selected) != undesired_per_env:
            raise ValueError("replicated worlds have inconsistent undesired bodies")
        for slot, body in enumerate(selected):
            body_to_undesired[body] = env * len(selected) + slot

    shape_to_foot = [-1] * len(shape_labels)
    shape_to_undesired = [-1] * len(shape_labels)
    shape_to_base_env = [-1] * len(shape_labels)
    mapped_feet = [set() for _ in range(num_envs)]
    for shape, (label, body) in enumerate(zip(shape_labels, shape_bodies)):
        if body < 0:
            continue
        env = body // body_per_env
        foot = -1
        for slot, leg in enumerate(LEGS):
            if f"/{leg}_foot/collisions/" in label:
                foot = env * len(LEGS) + slot
                shape_to_foot[shape] = foot
                mapped_feet[env].add(slot)
                break
        body_name = _basename(body_labels[body])
        if body_name == "base":
            shape_to_base_env[shape] = env
        elif foot < 0:
            shape_to_undesired[shape] = body_to_undesired[body]

    if any(mapped != set(range(len(LEGS))) for mapped in mapped_feet):
        raise ValueError("Go2 asset is missing one or more foot collision shapes")
    return ShapeClassification(
        tuple(shape_to_foot),
        tuple(shape_to_undesired),
        tuple(shape_to_base_env),
        undesired_per_env or 0,
    )

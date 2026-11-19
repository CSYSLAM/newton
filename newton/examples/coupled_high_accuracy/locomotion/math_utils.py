"""Small batched quaternion helpers using Newton's XYZW convention."""

from __future__ import annotations

import torch


def quat_rotate_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_xyz = quaternion[..., :3]
    q_w = quaternion[..., 3:4]
    twice_cross = 2.0 * torch.linalg.cross(q_xyz, vector, dim=-1)
    return vector - q_w * twice_cross + torch.linalg.cross(q_xyz, twice_cross, dim=-1)


def projected_gravity(quaternion: torch.Tensor) -> torch.Tensor:
    gravity = torch.zeros_like(quaternion[..., :3])
    gravity[..., 2] = -1.0
    return quat_rotate_inverse(quaternion, gravity)

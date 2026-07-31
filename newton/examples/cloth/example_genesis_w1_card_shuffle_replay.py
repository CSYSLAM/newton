# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Replay a complete-W1 Genesis IPC card-shuffle bake.

Genesis supplies the authoritative rigid-link poses and deforming card
vertices. Newton interpolates and renders that cache without re-solving the
IPC scene, so the robot, card deformation, two-sided geometry, and visible
card thickness remain synchronized.

Generate the cache from the Genesis repository root::

    uv run python examples/IPC_Solver/card_shuffle_w1_hands/w1_full_robot_bake.py

Replay it from the Newton repository root::

    uv run --extra examples -m newton.examples genesis_w1_card_shuffle_replay
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

BAKE_FORMAT = "genesis_w1_card_shuffle_bake_v1"
DEFAULT_CACHE = Path(__file__).resolve().parent / "assets" / "genesis_w1_card_shuffle" / "w1_card_shuffle_bake.npz"


def _quat_conjugate(quaternion):
    result = quaternion.copy()
    result[..., :3] *= -1.0
    return result


def _quat_multiply(first, second):
    first_xyz = first[..., :3]
    second_xyz = second[..., :3]
    first_w = first[..., 3:4]
    second_w = second[..., 3:4]
    return np.concatenate(
        (
            first_w * second_xyz + second_w * first_xyz + np.cross(first_xyz, second_xyz),
            first_w * second_w - np.sum(first_xyz * second_xyz, axis=-1, keepdims=True),
        ),
        axis=-1,
    )


def _quat_rotate(quaternion, vector):
    zero_w = np.zeros((*vector.shape[:-1], 1), dtype=vector.dtype)
    pure_vector = np.concatenate((vector, zero_w), axis=-1)
    return _quat_multiply(
        _quat_multiply(quaternion, pure_vector),
        _quat_conjugate(quaternion),
    )[..., :3]


def _quat_nlerp(first, second, fraction):
    second = second.copy()
    opposite = np.sum(first * second, axis=-1, keepdims=True) < 0.0
    second = np.where(opposite, -second, second)
    result = (1.0 - fraction) * first + fraction * second
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _vertex_normals(vertices, triangles):
    normals = np.zeros_like(vertices)
    first = vertices[triangles[:, 0]]
    second = vertices[triangles[:, 1]]
    third = vertices[triangles[:, 2]]
    face_normals = np.cross(second - first, third - first)
    np.add.at(normals, triangles[:, 0], face_normals)
    np.add.at(normals, triangles[:, 1], face_normals)
    np.add.at(normals, triangles[:, 2], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1.0e-12)


class Example:
    """Render a Genesis-baked complete W1 and deforming card shuffle."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.sim_time = 0.0
        self.cache_path = Path(args.cache).expanduser().resolve()
        self._load_cache()
        self.frame_dt = self.bake_frame_dt if args.playback_fps is None else 1.0 / args.playback_fps
        self._build_robot()
        self._build_card_render_data()
        self._apply_time(0.0)

        self.viewer.set_model(self.model)
        self.viewer.show_collision = False
        self.viewer.set_camera(
            pos=wp.vec3(0.0, 1.30, 1.05),
            pitch=-29.0,
            yaw=-90.0,
        )

    def _load_cache(self):
        if not self.cache_path.is_file():
            raise FileNotFoundError(f"Genesis W1 shuffle bake not found: {self.cache_path}")

        with np.load(self.cache_path, allow_pickle=False) as cache:
            bake_format = str(cache["format"])
            if bake_format != BAKE_FORMAT:
                raise ValueError(f"Unsupported bake format {bake_format!r}; expected {BAKE_FORMAT!r}.")
            self.frame_times = cache["frame_times"].astype(
                np.float64,
                copy=True,
            )
            self.robot_urdf = Path(str(cache["robot_urdf"]))
            self.robot_link_names = [str(name) for name in cache["robot_link_names"]]
            self.robot_link_positions = cache["robot_link_positions"].astype(np.float32, copy=True)
            self.robot_link_quaternions = cache["robot_link_quaternions_wxyz"][:, :, (1, 2, 3, 0)].astype(
                np.float32, copy=True
            )
            self.card_labels = [str(label) for label in cache["card_labels"]]
            self.card_vertices = cache["card_vertices"].astype(
                np.float32,
                copy=True,
            )
            self.card_triangles = cache["card_triangles"].astype(
                np.int32,
                copy=True,
            )
            self.card_uvs_host = cache["card_uvs"].astype(
                np.float32,
                copy=True,
            )
            self.card_boundary = cache["card_boundary"].astype(
                np.int32,
                copy=True,
            )
            self.card_thickness = float(cache["card_render_thickness"])

        if len(self.frame_times) < 2:
            raise ValueError("The Genesis bake must contain at least two frames.")
        if np.any(np.diff(self.frame_times) <= 0.0):
            raise ValueError("Genesis bake frame times must be increasing.")
        self.bake_frame_dt = float(np.median(np.diff(self.frame_times)))
        if not self.robot_urdf.is_file():
            raise FileNotFoundError(f"Baked W1 URDF path does not exist: {self.robot_urdf}")
        self.duration = float(self.frame_times[-1])

    def _build_robot(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_urdf(
            str(self.robot_urdf),
            floating=False,
            enable_self_collisions=False,
            ignore_inertial_definitions=True,
        )
        self.body_labels = [label.rsplit("/", 1)[-1] for label in builder.body_label]
        self.body_parents = np.full(builder.body_count, -1, dtype=np.int32)
        for parent, child in zip(
            builder.joint_parent,
            builder.joint_child,
            strict=True,
        ):
            self.body_parents[child] = parent

        self.model = builder.finalize(requires_grad=False)
        self.state_0 = self.model.state()
        newton.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            self.state_0,
        )
        initial_body_q = self.state_0.body_q.numpy().astype(
            np.float32,
            copy=True,
        )

        cached_indices = {name: index for index, name in enumerate(self.robot_link_names)}
        self.body_cache_indices = np.full(
            self.model.body_count,
            -1,
            dtype=np.int32,
        )
        self.body_anchor_indices = np.full(
            self.model.body_count,
            -1,
            dtype=np.int32,
        )
        self.body_anchor_positions = np.zeros(
            (self.model.body_count, 3),
            dtype=np.float32,
        )
        self.body_anchor_quaternions = np.zeros(
            (self.model.body_count, 4),
            dtype=np.float32,
        )

        for body, label in enumerate(self.body_labels):
            self.body_cache_indices[body] = cached_indices.get(label, -1)

        missing_cached_links = sorted(set(self.robot_link_names) - set(self.body_labels))
        if missing_cached_links:
            raise ValueError(f"Newton W1 URDF is missing baked links: {missing_cached_links}")

        for body in range(self.model.body_count):
            if self.body_cache_indices[body] >= 0:
                continue
            ancestor = int(self.body_parents[body])
            while ancestor >= 0 and self.body_cache_indices[ancestor] < 0:
                ancestor = int(self.body_parents[ancestor])
            if ancestor < 0:
                raise ValueError(f"No baked ancestor found for W1 body {self.body_labels[body]!r}.")

            anchor_position = initial_body_q[ancestor, :3]
            anchor_quaternion = initial_body_q[ancestor, 3:]
            inverse_anchor = _quat_conjugate(anchor_quaternion)
            self.body_anchor_indices[body] = ancestor
            self.body_anchor_positions[body] = _quat_rotate(
                inverse_anchor,
                initial_body_q[body, :3] - anchor_position,
            )
            self.body_anchor_quaternions[body] = _quat_multiply(
                inverse_anchor,
                initial_body_q[body, 3:],
            )

    def _build_card_render_data(self):
        self.card_indices = wp.array(
            self.card_triangles.reshape(-1),
            dtype=wp.int32,
            device=self.model.device,
        )
        self.card_reverse_indices = wp.array(
            self.card_triangles[:, ::-1].reshape(-1),
            dtype=wp.int32,
            device=self.model.device,
        )
        self.card_uvs = wp.array(
            self.card_uvs_host,
            dtype=wp.vec2,
            device=self.model.device,
        )

        boundary_count = len(self.card_boundary)
        edge_triangles = []
        for index in range(boundary_count):
            following = (index + 1) % boundary_count
            edge_triangles.extend(
                (
                    index,
                    following,
                    boundary_count + following,
                    index,
                    boundary_count + following,
                    boundary_count + index,
                )
            )
        self.card_edge_indices = wp.array(
            edge_triangles,
            dtype=wp.int32,
            device=self.model.device,
        )
        self.card_edge_triangles = np.asarray(
            edge_triangles,
            dtype=np.int32,
        ).reshape(-1, 3)
        self.card_front_points = [
            wp.empty(
                len(self.card_uvs_host),
                dtype=wp.vec3,
                device=self.model.device,
            )
            for _ in self.card_labels
        ]
        self.card_back_points = [
            wp.empty(
                len(self.card_uvs_host),
                dtype=wp.vec3,
                device=self.model.device,
            )
            for _ in self.card_labels
        ]
        self.card_front_normals = [
            wp.empty(
                len(self.card_uvs_host),
                dtype=wp.vec3,
                device=self.model.device,
            )
            for _ in self.card_labels
        ]
        self.card_back_normals = [
            wp.empty(
                len(self.card_uvs_host),
                dtype=wp.vec3,
                device=self.model.device,
            )
            for _ in self.card_labels
        ]
        self.card_edge_points = [
            wp.empty(
                2 * boundary_count,
                dtype=wp.vec3,
                device=self.model.device,
            )
            for _ in self.card_labels
        ]
        self.card_edge_normals = [
            wp.empty(
                2 * boundary_count,
                dtype=wp.vec3,
                device=self.model.device,
            )
            for _ in self.card_labels
        ]

    def _sample_indices(self, time):
        if time >= self.duration:
            return len(self.frame_times) - 2, len(self.frame_times) - 1, 1.0
        upper = int(np.searchsorted(self.frame_times, time, side="right"))
        lower = max(0, upper - 1)
        upper = min(upper, len(self.frame_times) - 1)
        interval = self.frame_times[upper] - self.frame_times[lower]
        fraction = 0.0 if interval <= 0.0 else (time - self.frame_times[lower]) / interval
        return lower, upper, float(fraction)

    def _apply_robot_pose(self, lower, upper, fraction):
        cached_positions = (1.0 - fraction) * self.robot_link_positions[lower] + fraction * self.robot_link_positions[
            upper
        ]
        cached_quaternions = _quat_nlerp(
            self.robot_link_quaternions[lower],
            self.robot_link_quaternions[upper],
            fraction,
        )
        body_q = np.empty(
            (self.model.body_count, 7),
            dtype=np.float32,
        )

        for body, cache_index in enumerate(self.body_cache_indices):
            if cache_index < 0:
                continue
            body_q[body, :3] = cached_positions[cache_index]
            body_q[body, 3:] = cached_quaternions[cache_index]

        for body, anchor in enumerate(self.body_anchor_indices):
            if anchor < 0:
                continue
            anchor_position = body_q[anchor, :3]
            anchor_quaternion = body_q[anchor, 3:]
            body_q[body, :3] = anchor_position + _quat_rotate(
                anchor_quaternion,
                self.body_anchor_positions[body],
            )
            body_q[body, 3:] = _quat_multiply(
                anchor_quaternion,
                self.body_anchor_quaternions[body],
            )

        self.state_0.body_q.assign(body_q)

    def _apply_card_pose(self, lower, upper, fraction):
        vertices = (1.0 - fraction) * self.card_vertices[lower] + fraction * self.card_vertices[upper]
        half_thickness = 0.5 * self.card_thickness
        for card_index, card_vertices in enumerate(vertices):
            normals = _vertex_normals(
                card_vertices,
                self.card_triangles,
            )
            front = card_vertices + half_thickness * normals
            back = card_vertices - half_thickness * normals
            edge = np.concatenate(
                (
                    front[self.card_boundary],
                    back[self.card_boundary],
                ),
                axis=0,
            )
            self.card_front_points[card_index].assign(front)
            self.card_back_points[card_index].assign(back)
            self.card_front_normals[card_index].assign(normals)
            self.card_back_normals[card_index].assign(-normals)
            self.card_edge_points[card_index].assign(edge)
            self.card_edge_normals[card_index].assign(_vertex_normals(edge, self.card_edge_triangles))

    def _apply_time(self, time):
        lower, upper, fraction = self._sample_indices(time)
        self._apply_robot_pose(lower, upper, fraction)
        self._apply_card_pose(lower, upper, fraction)

    def step(self):
        """Advance playback time by one display frame."""
        next_time = self.sim_time + self.frame_dt * self.args.speed
        if self.args.loop:
            self.sim_time = next_time % self.duration
        else:
            self.sim_time = min(next_time, self.duration)
        self._apply_time(self.sim_time)

    def render(self):
        """Render the interpolated complete robot and card geometry."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        for card_index, label in enumerate(self.card_labels):
            self.viewer.log_mesh(
                f"/cards/{label}/red_back",
                self.card_front_points[card_index],
                self.card_indices,
                normals=self.card_front_normals[card_index],
                uvs=self.card_uvs,
            )
            self.viewer.log_mesh(
                f"/cards/{label}/face",
                self.card_back_points[card_index],
                self.card_reverse_indices,
                normals=self.card_back_normals[card_index],
                uvs=self.card_uvs,
            )
            self.viewer.log_mesh(
                f"/cards/{label}/edge",
                self.card_edge_points[card_index],
                self.card_edge_indices,
                normals=self.card_edge_normals[card_index],
                backface_culling=False,
            )
        self.viewer.end_frame()

    def test_final(self):
        """Verify all replayed body and card coordinates remain finite."""
        if not np.all(np.isfinite(self.state_0.body_q.numpy())):
            raise ValueError("Non-finite W1 body transform in replay state.")
        if not np.all(np.isfinite(self.card_vertices)):
            raise ValueError("Non-finite card vertex in Genesis bake.")

    @staticmethod
    def create_parser():
        """Create command-line arguments for Genesis bake playback."""
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--cache",
            default=str(DEFAULT_CACHE),
            help="Genesis W1 card-shuffle NPZ bake.",
        )
        parser.add_argument(
            "--playback-fps",
            type=float,
            default=None,
            help=(
                "Playback timeline samples per second. By default, advance "
                "one baked Genesis frame per rendered frame; use 60 for "
                "real-time playback."
            ),
        )
        parser.add_argument(
            "--speed",
            type=float,
            default=1.0,
            help="Playback speed multiplier.",
        )
        parser.add_argument(
            "--loop",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Loop when playback reaches the end of the bake.",
        )
        parser.set_defaults(num_frames=800, render_fps=60.0)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    if args.playback_fps is not None and args.playback_fps <= 0.0:
        parser.error("--playback-fps must be positive.")
    if args.speed <= 0.0:
        parser.error("--speed must be positive.")
    newton.examples.run(Example(viewer, args), args)

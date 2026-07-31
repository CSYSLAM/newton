# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Render the Genesis W1 card-shuffle bake stopped at IPC frame 1300.

The bake is produced in the Genesis repository with::

    uv run python examples/IPC_Solver/card_shuffle_w1_hands/
        w1_full_robot_square_bake_frame_1300.py

Replay it from the Newton repository root with::

    uv run --extra examples python newton/examples/example_genesis_w1_card_shuffle_square_replay.py

This is a render-only replay: Newton does not solve IPC, arm IK, or card
physics.  It uses the W1 visual meshes from the baked URDF and streams the
recorded link transforms plus card mesh vertices into a Newton state.
"""

from __future__ import annotations

import argparse
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples


# This is deliberately the Genesis bake location, rather than an assets copy in
# Newton.  Regenerating the Genesis bake therefore immediately changes replay.
GENESIS_REPOSITORY = Path(__file__).resolve().parents[3] / "genesis-world"
DEFAULT_CACHE = (
    GENESIS_REPOSITORY
    / "examples"
    / "IPC_Solver"
    / "card_shuffle_w1_hands"
    / "bakes"
    / "w1_card_shuffle_square_frame_1300_bake.npz"
)

_REQUIRED_BAKE_KEYS = {
    "robot_urdf",
    "robot_link_names",
    "robot_link_positions",
    "robot_link_quaternions_wxyz",
    "card_vertices",
    "card_triangles",
    "card_boundary",
    "card_render_thickness",
}


def _rpy_to_quat_xyzw(rpy: np.ndarray) -> np.ndarray:
    """Return the URDF intrinsic-RPY rotation as an xyzw quaternion."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float32,
    )


def _quat_multiply_xyzw(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = lhs
    x2, y2, z2, w2 = rhs
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def _quat_rotate_xyzw(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate a vector by an xyzw quaternion without creating Warp arrays."""
    xyz = quat[:3]
    return vector + 2.0 * np.cross(xyz, np.cross(xyz, vector) + quat[3] * vector)


def _fixed_link_rest_transforms(urdf_path: Path) -> dict[str, tuple[str, np.ndarray, np.ndarray]]:
    """Map each fixed-joint child link to its parent-relative rest transform."""
    root = ET.parse(urdf_path).getroot()
    transforms: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
    for joint in root.findall("joint"):
        if joint.get("type") != "fixed":
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        xyz = np.zeros(3, dtype=np.float32)
        rpy = np.zeros(3, dtype=np.float32)
        if origin is not None:
            if origin.get("xyz"):
                xyz = np.fromstring(origin.get("xyz"), sep=" ", dtype=np.float32)
            if origin.get("rpy"):
                rpy = np.fromstring(origin.get("rpy"), sep=" ", dtype=np.float32)
        transforms[child.get("link")] = (parent.get("link"), xyz, _rpy_to_quat_xyzw(rpy))
    return transforms


def _visual_only_urdf_xml(urdf_path: Path) -> str:
    """Return a self-contained visual URDF suitable for direct-pose replay.

    The production W1 URDF contains Genesis-specific connector joints that are
    not a strict Newton articulation tree.  They are immaterial here because
    every link receives a baked world transform.  Removing joints also prevents
    Newton from running FK over, and overwriting, those direct poses.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for joint in root.findall("joint"):
        root.remove(joint)
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename and not Path(filename).is_absolute():
            mesh.set("filename", str((urdf_path.parent / filename).resolve()))
    return ET.tostring(root, encoding="unicode")


def _all_link_poses(
    link_names: list[str],
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    fixed_transforms: dict[str, tuple[str, np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Return world transforms for baked links and visual-only fixed links.

    Genesis bakes the 41 simulated W1 links.  The render URDF contains a few
    additional fixed links (cameras, hand adapters, chassis pieces, ...), which
    are reconstructed from their nearest available baked parent.
    """
    poses = {
        name: np.concatenate((positions[index], quaternions_wxyz[index][[1, 2, 3, 0]])).astype(np.float32)
        for index, name in enumerate(link_names)
    }
    unresolved = dict(fixed_transforms)
    while unresolved:
        made_progress = False
        for child, (parent, local_pos, local_quat) in tuple(unresolved.items()):
            parent_pose = poses.get(parent)
            if parent_pose is None:
                continue
            parent_quat = parent_pose[3:]
            poses[child] = np.concatenate(
                (
                    parent_pose[:3] + _quat_rotate_xyzw(parent_quat, local_pos),
                    _quat_multiply_xyzw(parent_quat, local_quat),
                )
            ).astype(np.float32)
            del unresolved[child]
            made_progress = True
        if not made_progress:
            # A fixed link disconnected from the baked articulation is harmless:
            # it has no useful replay transform and will keep its URDF rest pose.
            break
    return poses


def _make_thick_card_triangles(surface_triangles: np.ndarray, boundary: np.ndarray, vertex_count: int) -> np.ndarray:
    """Build closed top, bottom, and side-wall triangles for one card."""
    top = np.asarray(surface_triangles, dtype=np.int32)
    bottom = top[:, ::-1] + vertex_count
    side: list[tuple[int, int, int]] = []
    for index, first in enumerate(boundary):
        second = int(boundary[(index + 1) % len(boundary)])
        first = int(first)
        side.extend(
            (
                (first, second, second + vertex_count),
                (first, second + vertex_count, first + vertex_count),
            )
        )
    return np.concatenate((top, bottom, np.asarray(side, dtype=np.int32)), axis=0)


def _make_thick_card_vertices(vertices: np.ndarray, triangles: np.ndarray, thickness: float) -> np.ndarray:
    """Offset a deforming card surface along smooth per-vertex normals."""
    result = np.empty((*vertices.shape[:-2], vertices.shape[-2] * 2, 3), dtype=np.float32)
    for card_index, card_vertices in enumerate(vertices):
        faces = card_vertices[triangles]
        face_normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
        normals = np.zeros_like(card_vertices)
        for corner in range(3):
            np.add.at(normals, triangles[:, corner], face_normals)
        normal_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, normal_lengths, out=np.zeros_like(normals), where=normal_lengths > 1.0e-8)
        normals[normal_lengths[:, 0] <= 1.0e-8] = (0.0, 0.0, 1.0)
        offset = normals * (0.5 * thickness)
        result[card_index, : len(card_vertices)] = card_vertices + offset
        result[card_index, len(card_vertices) :] = card_vertices - offset
    return result


def _load_centering_start_frame(cache_path: Path) -> int | None:
    """Read the recorded deck-centering boundary associated with a Genesis bake."""
    trajectory_path = cache_path.parent.parent / "w1_hand_trajectory_with_square.json"
    if not trajectory_path.is_file():
        return None
    try:
        import json

        document = json.loads(trajectory_path.read_text(encoding="utf-8"))
        return int(document["phase_start_step"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


class Example:
    """A direct render-only player for the Genesis IPC-frame-1300 cache."""

    @staticmethod
    def create_parser() -> argparse.ArgumentParser:
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--cache",
            type=Path,
            default=DEFAULT_CACHE,
            help="Genesis NPZ emitted by w1_full_robot_square_bake_frame_1300.py.",
        )
        parser.add_argument(
            "--loop",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Restart from the first cached frame after the IPC-1300 bake ends.",
        )
        parser.add_argument(
            "--playback-fps",
            type=float,
            default=None,
            help="Optional display-frame cap; equivalent to --render-fps when that is not supplied.",
        )
        parser.add_argument(
            "--shuffle-speed",
            type=float,
            default=0.01,
            help="Baked-time multiplier before the deck-centering phase.",
        )
        parser.add_argument(
            "--centering-speed",
            type=float,
            default=0.05,
            help="Baked-time multiplier from the deck-centering phase onward.",
        )
        parser.add_argument(
            "--centering-start-frame",
            type=int,
            default=None,
            help="IPC/bake frame where deck centering starts; default reads Genesis trajectory metadata.",
        )
        parser.add_argument(
            "--speed",
            type=float,
            default=None,
            help="Compatibility override: use one baked-time multiplier for both phases.",
        )
        parser.add_argument(
            "--card-thickness",
            type=float,
            default=None,
            help="Override the baked card thickness in metres.",
        )
        parser.add_argument("--start-frame", type=int, default=0, help="First cached sample to display.")
        return parser

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.cache_path = Path(args.cache).expanduser().resolve()
        if not self.cache_path.is_file():
            raise FileNotFoundError(
                f"Bake not found: {self.cache_path}. Generate it with "
                "examples/IPC_Solver/card_shuffle_w1_hands/w1_full_robot_square_bake_frame_1300.py"
            )

        with np.load(self.cache_path, allow_pickle=False) as bake:
            missing = sorted(_REQUIRED_BAKE_KEYS.difference(bake.files))
            if missing:
                raise ValueError(f"Bake is missing required arrays: {', '.join(missing)}")
            self.link_names = [str(name) for name in bake["robot_link_names"]]
            self.link_positions = np.asarray(bake["robot_link_positions"], dtype=np.float32)
            self.link_quaternions_wxyz = np.asarray(bake["robot_link_quaternions_wxyz"], dtype=np.float32)
            self.card_vertices = np.asarray(bake["card_vertices"], dtype=np.float32)
            self.card_triangles = np.asarray(bake["card_triangles"], dtype=np.int32)
            self.card_boundary = np.asarray(bake["card_boundary"], dtype=np.int32)
            baked_card_thickness = float(bake["card_render_thickness"])
            self.robot_urdf = Path(str(bake["robot_urdf"])).expanduser()
            self.bake_dt = float(bake["dt_seconds"]) if "dt_seconds" in bake.files else 1.0 / 800.0

        if not self.robot_urdf.is_file():
            raise FileNotFoundError(f"W1 URDF referenced by bake does not exist: {self.robot_urdf}")
        if self.link_positions.shape[:2] != self.link_quaternions_wxyz.shape[:2]:
            raise ValueError("robot link position and quaternion trajectories have incompatible shapes")
        if self.link_positions.shape[0] != self.card_vertices.shape[0]:
            raise ValueError("robot and card bake trajectories have different frame counts")

        self.frame_count = int(self.link_positions.shape[0])
        self.card_count = int(self.card_vertices.shape[1])
        self.vertices_per_card = int(self.card_vertices.shape[2])
        self.card_thickness = float(args.card_thickness) if args.card_thickness is not None else baked_card_thickness
        if self.card_thickness <= 0.0:
            raise ValueError("--card-thickness must be positive")
        self.render_card_triangles = _make_thick_card_triangles(
            self.card_triangles, self.card_boundary, self.vertices_per_card
        )
        self.render_vertices_per_card = self.vertices_per_card * 2
        self.frame = int(np.clip(args.start_frame, 0, self.frame_count - 1))
        self.loop = bool(args.loop)
        trajectory_centering_start = _load_centering_start_frame(self.cache_path)
        self.centering_start_frame = (
            int(args.centering_start_frame)
            if args.centering_start_frame is not None
            else (trajectory_centering_start if trajectory_centering_start is not None else self.frame_count)
        )
        self.centering_start_frame = int(np.clip(self.centering_start_frame, 0, self.frame_count))
        if args.speed is None:
            self.shuffle_speed = float(args.shuffle_speed)
            self.centering_speed = float(args.centering_speed)
        else:
            self.shuffle_speed = self.centering_speed = float(args.speed)
        if args.playback_fps is not None and args.playback_fps <= 0.0:
            raise ValueError("--playback-fps must be positive")
        if self.shuffle_speed <= 0.0 or self.centering_speed <= 0.0:
            raise ValueError("--shuffle-speed, --centering-speed, and --speed must be positive")
        self._frame_remainder = 0.0
        self._last_wall_time = time.perf_counter()
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        # The model is never stepped, so importing collision shapes is safe and
        # gives the viewer the visual meshes directly from the W1 URDF.
        builder.add_urdf(
            _visual_only_urdf_xml(self.robot_urdf),
            floating=False,
            enable_self_collisions=False,
        )
        self._fixed_transforms = _fixed_link_rest_transforms(self.robot_urdf)

        # One deformable triangle mesh per card.  Positions are overwritten from
        # the bake every frame; density zero avoids any unnecessary physical role.
        first_cards = _make_thick_card_vertices(self.card_vertices[0], self.card_triangles, self.card_thickness)
        flat_indices = self.render_card_triangles.reshape(-1).tolist()
        for card_index in range(self.card_count):
            builder.add_cloth_mesh(
                pos=wp.vec3(0.0, 0.0, 0.0),
                rot=wp.quat_identity(),
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=first_cards[card_index].tolist(),
                indices=flat_indices,
                density=0.0,
                particle_radius=0.001,
                label=f"card_{card_index:02d}",
            )

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self._body_indices = {label.rsplit("/", 1)[-1]: index for index, label in enumerate(self.model.body_label)}
        unknown_baked_links = sorted(set(self.link_names).difference(self._body_indices))
        if unknown_baked_links:
            raise ValueError(f"Newton W1 URDF is missing baked links: {unknown_baked_links[:5]}")
        self.viewer.set_model(self.model)

        # Hand-tuned W1/card-deck view.  Do not call camera.look_at() here:
        # it recomputes pitch/yaw and would override this saved composition.
        self.viewer.set_camera(
            pos=wp.vec3(-0.04, 0.42, 0.36),
            pitch=2.0,
            yaw=-79.3,
        )
        if hasattr(self.viewer, "camera"):
            self.viewer.camera.fov = 50.0

        self._apply_frame(self.frame)
        print(
            f"Loaded {self.frame_count} samples from IPC-frame-1300 bake: "
            f"{len(self.link_names)} baked W1 links, {self.card_count} thick cards, "
            f"{self.render_vertices_per_card} render vertices/card; "
            f"centering starts at frame {self.centering_start_frame}, speeds "
            f"{self.shuffle_speed:g} -> {self.centering_speed:g}"
        )

    def _apply_frame(self, frame: int) -> None:
        poses_by_link = _all_link_poses(
            self.link_names,
            self.link_positions[frame],
            self.link_quaternions_wxyz[frame],
            self._fixed_transforms,
        )
        body_q = self.state_0.body_q.numpy()
        for link_name, pose in poses_by_link.items():
            body_index = self._body_indices.get(link_name)
            if body_index is not None:
                body_q[body_index] = pose
        self.state_0.body_q.assign(body_q)
        thick_cards = _make_thick_card_vertices(self.card_vertices[frame], self.card_triangles, self.card_thickness)
        self.state_0.particle_q.assign(thick_cards.reshape(-1, 3))

    def step(self) -> None:
        current_wall_time = time.perf_counter()
        elapsed = min(current_wall_time - self._last_wall_time, 0.1)
        self._last_wall_time = current_wall_time
        phase_speed = self.centering_speed if self.frame >= self.centering_start_frame else self.shuffle_speed
        self._frame_remainder += elapsed * phase_speed / self.bake_dt
        advance = int(self._frame_remainder)
        self._frame_remainder -= advance
        if advance <= 0:
            return

        next_frame = self.frame + advance
        if next_frame >= self.frame_count:
            if not self.loop:
                next_frame = self.frame_count - 1
            else:
                next_frame %= self.frame_count
        self.frame = next_frame
        self._apply_frame(self.frame)
        self.sim_time += advance * self.bake_dt

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        pass


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    if args.playback_fps is not None and args.render_fps is None:
        args.render_fps = args.playback_fps
    newton.examples.run(Example(viewer, args), args)

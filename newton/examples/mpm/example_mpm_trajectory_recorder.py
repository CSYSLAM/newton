# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Interactive trajectory segment recorder for the WAIC kitchen W1 scene.
#
# Inherits the full example_mpm_w1_burger_slice_waic_kitchen scene (W1
# hands-only robot + knife + pan + MPM meat) and overlays an interactive
# waypoint/segment recorder on top of it:
#
#   - Left/right wrist TCP gizmos are always live; drag them to pose the arm.
#   - Keyboard keys capture waypoints (start/mid/end) into a segment, then
#     play the segment back through IK in real time (MPM frozen, so the meat
#     does not deform during preview -- press X to run the real MPM physics).
#   - Segments chain automatically: the next segment's start is the previous
#     segment's end.
#   - Every joint on the robot can be tuned from the side panel: arm joints
#     drive IK targets, finger joints (incl. the un-controlled left hand)
#     write directly to joint_q + eval_fk.
#   - Trajectories serialize to JSON (F5 save / F9 load), and --trajectory-file
#     / --start-segment drive playback from a saved file.
#
# Command:
#   python -m newton.examples mpm_trajectory_recorder
#
###########################################################################

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mpm import example_mpm_w1_burger_slice_waic_kitchen as base
from newton.examples.mpm.example_mpm_w1_burger_slice_waic_kitchen import (
    Example as WaicKitchenExample,
)
from newton.examples.mpm.example_mpm_w1_burger_slice_waic_kitchen import (
    copy_ik_to_joint_q_kernel,
    lock_joint_q_kernel,
    set_indexed_joint_q_kernel,
)

# ---------------------------------------------------------------------------
# Serialization model
# ---------------------------------------------------------------------------

# A waypoint stores both IK-driven TCP targets and a full joint_q snapshot.
# The snapshot lets un-controlled joints (e.g. the 10 left-hand fingers, which
# the parent IK does not drive) replay exactly as captured, and lets the user
# hand-tune any joint from the side panel and have that tuning recorded.
Tcp7 = tuple[float, float, float, float, float, float, float]  # px,py,pz,qx,qy,qz,qw


@dataclass
class Waypoint:
    """One recorded pose: TCP targets + full joint snapshot + hand params."""

    right_tcp: Tcp7
    left_tcp: Tcp7
    joint_q_snapshot: list[float]
    hand_alpha: float
    grip_mode: float
    knife_alpha: float
    carry_alpha: float
    pan_tf: Tcp7
    role: Literal["start", "mid", "end"] = "mid"

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "right_tcp": list(self.right_tcp),
            "left_tcp": list(self.left_tcp),
            "joint_q_snapshot": list(self.joint_q_snapshot),
            "hand_alpha": float(self.hand_alpha),
            "grip_mode": float(self.grip_mode),
            "knife_alpha": float(self.knife_alpha),
            "carry_alpha": float(self.carry_alpha),
            "pan_tf": list(self.pan_tf),
        }

    @staticmethod
    def from_dict(d: dict) -> Waypoint:
        return Waypoint(
            right_tcp=tuple(float(v) for v in d["right_tcp"]),
            left_tcp=tuple(float(v) for v in d["left_tcp"]),
            joint_q_snapshot=[float(v) for v in d["joint_q_snapshot"]],
            hand_alpha=float(d.get("hand_alpha", 0.0)),
            grip_mode=float(d.get("grip_mode", 0.0)),
            knife_alpha=float(d.get("knife_alpha", 0.0)),
            carry_alpha=float(d.get("carry_alpha", 0.0)),
            pan_tf=tuple(float(v) for v in d.get("pan_tf", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])),
            role=d.get("role", "mid"),
        )


@dataclass
class Segment:
    """A trajectory segment: start -> (optional mid points) -> end."""

    duration: float
    waypoints: list[Waypoint] = field(default_factory=list)
    label: str = ""

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "duration": float(self.duration),
            "waypoints": [w.to_dict() for w in self.waypoints],
        }

    @staticmethod
    def from_dict(d: dict) -> Segment:
        return Segment(
            duration=float(d.get("duration", 2.0)),
            waypoints=[Waypoint.from_dict(w) for w in d.get("waypoints", [])],
            label=str(d.get("label", "")),
        )


@dataclass
class Trajectory:
    """An ordered list of segments plus metadata."""

    segments: list[Segment] = field(default_factory=list)
    start_segment_index: int = 0
    robot_urdf: str = ""
    joint_labels: list[str] = field(default_factory=list)
    version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "robot_urdf": self.robot_urdf,
            "joint_labels": list(self.joint_labels),
            "start_segment_index": int(self.start_segment_index),
            "segments": [s.to_dict() for s in self.segments],
        }

    @staticmethod
    def from_dict(d: dict) -> Trajectory:
        return Trajectory(
            segments=[Segment.from_dict(s) for s in d.get("segments", [])],
            start_segment_index=int(d.get("start_segment_index", 0)),
            robot_urdf=str(d.get("robot_urdf", "")),
            joint_labels=[str(x) for x in d.get("joint_labels", [])],
            version=str(d.get("version", "1.0")),
        )


def _tcp_to_transform(tcp: Tcp7) -> wp.transform:
    return wp.transform(
        wp.vec3(float(tcp[0]), float(tcp[1]), float(tcp[2])),
        wp.quat(float(tcp[3]), float(tcp[4]), float(tcp[5]), float(tcp[6])),
    )


def _transform_to_tcp(tf: wp.transform) -> Tcp7:
    p = wp.transform_get_translation(tf)
    q = wp.transform_get_rotation(tf)
    return (
        float(p[0]),
        float(p[1]),
        float(p[2]),
        float(q[0]),
        float(q[1]),
        float(q[2]),
        float(q[3]),
    )


# ---------------------------------------------------------------------------
# Recorder state
# ---------------------------------------------------------------------------


class RecorderState:
    IDLE = "idle"  # waiting to capture a start waypoint
    READY = "ready"  # have a start (and maybe mids), waiting for end / play
    PLAYING = "playing"  # playing the current segment through IK (MPM frozen)
    MPM_RUN = "mpm_run"  # running the parent scripted MPM simulation


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class TrajectoryRecorderExample(WaicKitchenExample):
    """Interactive trajectory segment recorder layered on the WAIC kitchen scene.

    Inherits the full W1 + knife + pan + MPM meat setup from
    :class:`example_mpm_w1_burger_slice_waic_kitchen.Example` and overlays a
    waypoint/segment recorder. Left/right wrist TCP gizmos are always live;
    keyboard keys capture waypoints into a segment which is then played back
    through IK in real time with MPM frozen.
    """

    # Keys (edge-detected unless noted). Avoid viewer built-ins
    # (Space=pause, .=step, H=toggle UI, F=frame camera, Esc=quit).
    KEY_CAPTURE_START = "s"
    KEY_CAPTURE_MID = "m"
    KEY_CAPTURE_END = "e"
    KEY_PLAY = "p"
    KEY_RESET_SEGMENT = "r"
    KEY_RESTORE_HOME = "0"
    KEY_NEW_SEGMENT = "n"
    KEY_TOGGLE_MPM = "x"
    KEY_SAVE = "f5"
    KEY_LOAD = "f9"

    def __init__(self, viewer, args) -> None:
        # Defer the recorder-state init until after super().__init__ sets up the
        # model/IK/solver, but _poll_keys runs inside the overridden step(), so
        # we must prime the edge-detection state before super().__init__ runs
        # the first eval_fk / viewer setup. super().__init__ does not call our
        # step(), so this is just defensive ordering.
        self._init_recorder_state(args)
        super().__init__(viewer, args)
        self._finalize_recorder_init(args)

    # -- lifecycle -----------------------------------------------------------

    def _init_recorder_state(self, args) -> None:
        self._recorder_state: str = RecorderState.IDLE
        self._trajectory: Trajectory = Trajectory(
            robot_urdf=str(base.URDF_PATH),
            start_segment_index=int(getattr(args, "start_segment", 0)),
        )
        self._current_segment: Segment = Segment(duration=float(getattr(args, "play_duration", 2.0)))
        self._play_alpha: float = 0.0
        self._freeze_mpm: bool = bool(getattr(args, "freeze_mpm", True))
        self._play_segment_index: int = int(getattr(args, "start_segment", 0))
        self._joint_q_home: np.ndarray | None = None
        # per-key previous-down flags for edge detection
        self._prev_keys: dict[str, bool] = {}
        # host scratch for non-IK joint interpolation (rebuilt per frame)
        self._non_ik_joint_q_host: np.ndarray | None = None
        # imgui-side state
        self._arm_tcp_dirty: bool = False
        self._right_tcp_edit: list[float] = [0.0, 0.0, 0.0]
        self._left_tcp_edit: list[float] = [0.0, 0.0, 0.0]
        self._save_path: str = "trajectory.json"
        # host views of joint_q for sliders (refreshed in gui)
        self._joint_q_host: np.ndarray | None = None

    def _finalize_recorder_init(self, args) -> None:
        # Home pose snapshot (full joint_q from the finalized model).
        self._joint_q_home = self.model.joint_q.numpy().copy()
        self._non_ik_joint_q_host = self._joint_q_home.copy()
        # Cache the non-IK joint indices/labels (joints the IK does NOT control).
        controlled = self._controlled_joint_labels()
        self._non_ik_q_indices: list[int] = []
        q_start = self.model.joint_q_start.numpy()
        for joint_idx, label in enumerate(self.model.joint_label):
            if label in controlled:
                continue
            self._non_ik_q_indices.append(int(q_start[joint_idx]))
        # joint_labels stored once for save/load validation
        self._trajectory.joint_labels = list(self.model.joint_label)

        # Seed gizmo targets with the current actual TCP so dragging starts
        # from where the robot already is (matches example_ik_custom pattern).
        self.right_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        self.left_tf = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        self._right_tcp_edit = [float(v) for v in wp.transform_get_translation(self.right_tf)]
        self._left_tcp_edit = [float(v) for v in wp.transform_get_translation(self.left_tf)]

        # A dedicated free-floating window for the recorder HUD.
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self._render_recorder_hud, position="free")

        # Load a trajectory up-front if requested.
        traj_file = getattr(args, "trajectory_file", "") or ""
        if traj_file:
            self._load_trajectory(traj_file)

        self._print_help()

    # -- step / render -------------------------------------------------------

    def step(self) -> None:
        self._poll_keys()
        state = self._recorder_state
        if state == RecorderState.MPM_RUN:
            self.simulate()
            return
        if state == RecorderState.PLAYING:
            self._play_segment()
            self.sim_time += self.frame_dt
            self.frame_index += 1
            return
        # IDLE / READY: keep the scene posed (gizmo edits + slider edits),
        # no MPM step. eval_fk so any direct joint_q edits show up visually.
        self._apply_direct_joint_edits()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def render(self) -> None:
        super().render()
        # Overlay the TCP gizmos. The viewer mutates self.right_tf / self.left_tf
        # in place (pass-by-ref), so the next step() reads the dragged value.
        if hasattr(self.viewer, "log_gizmo"):
            self.viewer.log_gizmo(
                "recorder_right_tcp",
                self.right_tf,
                snap_to=self._current_tcp_transform(self.right_ee_index, self.right_ee_offset),
            )
            self.viewer.log_gizmo(
                "recorder_left_tcp",
                self.left_tf,
                snap_to=self._current_tcp_transform(self.left_ee_index, self.left_ee_offset),
            )

    def render_ui(self, imgui) -> None:
        super().render_ui(imgui)
        self._render_joint_sliders(imgui)

    # -- keyboard ------------------------------------------------------------

    def _edge(self, key: str) -> bool:
        """Return True on the rising edge of `key` being held."""
        if not hasattr(self.viewer, "is_key_down"):
            return False
        down = bool(self.viewer.is_key_down(key))
        prev = self._prev_keys.get(key, False)
        self._prev_keys[key] = down
        return down and not prev

    def _poll_keys(self) -> None:
        if self._edge(self.KEY_CAPTURE_START):
            self._capture_waypoint("start")
        if self._edge(self.KEY_CAPTURE_MID):
            self._capture_waypoint("mid")
        if self._edge(self.KEY_CAPTURE_END):
            self._capture_waypoint("end")
        if self._edge(self.KEY_PLAY):
            self._begin_play()
        if self._edge(self.KEY_RESET_SEGMENT):
            self._reset_to_segment_start()
        if self._edge(self.KEY_RESTORE_HOME):
            self._restore_default_pose()
        if self._edge(self.KEY_NEW_SEGMENT):
            self._advance_segment()
        if self._edge(self.KEY_TOGGLE_MPM):
            self._toggle_mpm_run()
        if self._edge(self.KEY_SAVE):
            self._save_trajectory(self._save_path)
        if self._edge(self.KEY_LOAD):
            self._open_load_dialog()

    # -- capture -------------------------------------------------------------

    def _snapshot_joint_q(self) -> list[float]:
        return [float(v) for v in self.state_0.joint_q.numpy()]

    def _capture_waypoint(self, role: str) -> None:
        if self._recorder_state == RecorderState.MPM_RUN:
            return
        right_tcp = _transform_to_tcp(self.right_tf)
        left_tcp = _transform_to_tcp(self.left_tf)
        # Capture the currently active hand/grip/knife/carry channels so the
        # segment replays with the same hand state as when it was recorded.
        hand_alpha = float(getattr(self, "_edit_hand_alpha", 0.0))
        grip_mode = float(getattr(self, "grip_mode", 0.0))
        knife_alpha = float(getattr(self, "knife_alpha", 0.0))
        carry_alpha = float(getattr(self, "meat_carry_alpha", 0.0))
        pan_tf = _transform_to_tcp(getattr(self, "pan_tf", self.pan_initial_tf))
        waypoint = Waypoint(
            right_tcp=right_tcp,
            left_tcp=left_tcp,
            joint_q_snapshot=self._snapshot_joint_q(),
            hand_alpha=hand_alpha,
            grip_mode=grip_mode,
            knife_alpha=knife_alpha,
            carry_alpha=carry_alpha,
            pan_tf=pan_tf,
            role=role,
        )
        if role == "start":
            self._current_segment = Segment(duration=self._current_segment.duration)
            self._current_segment.waypoints = [waypoint]
            self._recorder_state = RecorderState.READY
            print(f"[recorder] segment {len(self._trajectory.segments)} start captured")
        else:
            if not self._current_segment.waypoints:
                # No start yet -- treat this capture as the start.
                self._current_segment.waypoints = [waypoint]
                self._recorder_state = RecorderState.READY
                print(f"[recorder] (no start; treating {role} as start)")
            else:
                self._current_segment.waypoints.append(waypoint)
                print(f"[recorder] {role} captured ({len(self._current_segment.waypoints)} pts)")

    # -- playback ------------------------------------------------------------

    def _begin_play(self) -> None:
        if self._recorder_state == RecorderState.MPM_RUN:
            return
        if len(self._current_segment.waypoints) < 2:
            print("[recorder] need >= 2 waypoints (start + end) to play")
            return
        self._play_alpha = 0.0
        self._recorder_state = RecorderState.PLAYING
        # Seed IK solver from the current joint_q so playback starts in-place.
        wp.copy(self.ik_joint_q, self._joint_q_2d())
        print("[recorder] playing segment")

    def _joint_q_2d(self) -> wp.array:
        """Reshape the current joint_q into the (1, n) layout ik_solver expects."""
        arr = self.state_0.joint_q.numpy().reshape(1, -1).astype(np.float32)
        return wp.array(arr, dtype=wp.float32, device=self.model.device)

    def _interpolate_waypoints(self, waypoints: list[Waypoint], alpha: float) -> tuple:
        n = len(waypoints)
        if n == 1:
            w = waypoints[0]
            return (
                _tcp_to_transform(w.right_tcp),
                _tcp_to_transform(w.left_tcp),
                list(w.joint_q_snapshot),
                w,
            )
        seg_alpha = float(np.clip(alpha, 0.0, 1.0)) * (n - 1)
        k = min(int(seg_alpha), n - 2)
        local_a = seg_alpha - k
        wa, wb = waypoints[k], waypoints[k + 1]
        right_tf = self._interpolate_transform(
            _tcp_to_transform(wa.right_tcp), _tcp_to_transform(wb.right_tcp), local_a
        )
        left_tf = self._interpolate_transform(_tcp_to_transform(wa.left_tcp), _tcp_to_transform(wb.left_tcp), local_a)
        ja = wa.joint_q_snapshot
        jb = wb.joint_q_snapshot
        joint_q = [ja[i] * (1.0 - local_a) + jb[i] * local_a for i in range(len(ja))]
        # Build a blended Waypoint-like param bundle for the scalar channels.
        blend = Waypoint(
            right_tcp=_transform_to_tcp(right_tf),
            left_tcp=_transform_to_tcp(left_tf),
            joint_q_snapshot=joint_q,
            hand_alpha=wa.hand_alpha * (1.0 - local_a) + wb.hand_alpha * local_a,
            grip_mode=wa.grip_mode * (1.0 - local_a) + wb.grip_mode * local_a,
            knife_alpha=wa.knife_alpha * (1.0 - local_a) + wb.knife_alpha * local_a,
            carry_alpha=wa.carry_alpha * (1.0 - local_a) + wb.carry_alpha * local_a,
            pan_tf=_transform_to_tcp(
                self._interpolate_transform(_tcp_to_transform(wa.pan_tf), _tcp_to_transform(wb.pan_tf), local_a)
            ),
        )
        return right_tf, left_tf, joint_q, blend

    def _play_segment(self) -> None:
        seg = self._current_segment
        wps = seg.waypoints
        if len(wps) < 2:
            self._recorder_state = RecorderState.READY
            return
        self._play_alpha += self.frame_dt / max(seg.duration, 1.0e-6)
        if self._play_alpha >= 1.0:
            self._play_alpha = 1.0
        right_tf, left_tf, joint_q, blend = self._interpolate_waypoints(wps, self._play_alpha)

        # 1. Seed IK from current joint_q.
        wp.copy(self.frame_joint_q_start, self.state_0.joint_q)
        # 2. Push TCP targets.
        self.right_pos_obj.set_target_position(0, wp.transform_get_translation(right_tf))
        self.right_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(right_tf)))
        self.left_pos_obj.set_target_position(0, wp.transform_get_translation(left_tf))
        self.left_rot_obj.set_target_rotation(0, self._quat_to_vec4(wp.transform_get_rotation(left_tf)))
        # 3. Solve IK.
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        # 4. Lock non-IK joints -- but to the interpolated snapshot values, not
        #    the parent's fixed home, so left-hand fingers replay as recorded.
        locked_values = np.array([joint_q[i] for i in self._non_ik_q_indices], dtype=np.float32)
        locked_values_wp = wp.array(locked_values, dtype=wp.float32, device=self.model.device)
        wp.launch(
            lock_joint_q_kernel,
            dim=self.locked_q_indices.shape[0],
            inputs=[self.ik_joint_q, self.locked_q_indices, locked_values_wp],
            device=self.model.device,
        )
        # 5. Copy IK result into the per-frame end buffer.
        wp.launch(
            copy_ik_to_joint_q_kernel,
            dim=self.model.joint_coord_count,
            inputs=[self.ik_joint_q],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )
        # 6. Blend right-hand fingers between open and grasp using hand_alpha.
        wp.launch(
            set_indexed_joint_q_kernel,
            dim=self.right_hand_q_indices.shape[0],
            inputs=[
                self.right_hand_q_indices,
                self.right_hand_open,
                self.right_hand_knife_grasp,
                self.right_hand_pan_grasp,
                float(blend.hand_alpha),
                float(blend.grip_mode),
            ],
            outputs=[self.frame_joint_q_end],
            device=self.model.device,
        )
        # 7. Apply the frame end to state and FK.
        wp.copy(self.state_0.joint_q, self.frame_joint_q_end)
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        # 8. Drive knife/pan kinematic bodies to follow (MPM itself frozen).
        self.knife_alpha = float(blend.knife_alpha)
        self.grip_mode = float(blend.grip_mode)
        self.meat_carry_alpha = float(blend.carry_alpha)
        self.pan_tf = _tcp_to_transform(blend.pan_tf)
        self._update_knife_transform(float(blend.knife_alpha))
        self._update_pan_transform(_tcp_to_transform(blend.pan_tf))
        # 9. No solver.step / project_outside / carry_particles -- MPM frozen.

        if self._play_alpha >= 1.0:
            self._recorder_state = RecorderState.READY
            print("[recorder] segment playback complete")

    # -- segment management --------------------------------------------------

    def _advance_segment(self) -> None:
        if self._recorder_state == RecorderState.MPM_RUN:
            return
        if not self._current_segment.waypoints:
            print("[recorder] nothing to advance (current segment empty)")
            return
        # Commit the current segment.
        self._trajectory.segments.append(self._current_segment)
        prev_end = self._current_segment.waypoints[-1]
        # New segment chains from the previous end.
        new_seg = Segment(duration=self._current_segment.duration)
        chained = Waypoint(
            right_tcp=prev_end.right_tcp,
            left_tcp=prev_end.left_tcp,
            joint_q_snapshot=list(prev_end.joint_q_snapshot),
            hand_alpha=prev_end.hand_alpha,
            grip_mode=prev_end.grip_mode,
            knife_alpha=prev_end.knife_alpha,
            carry_alpha=prev_end.carry_alpha,
            pan_tf=prev_end.pan_tf,
            role="start",
        )
        new_seg.waypoints = [chained]
        self._current_segment = new_seg
        self._play_alpha = 0.0
        self._recorder_state = RecorderState.READY
        print(f"[recorder] advanced to segment {len(self._trajectory.segments)} (start chained)")

    def _reset_to_segment_start(self) -> None:
        if self._recorder_state == RecorderState.MPM_RUN:
            return
        wps = self._current_segment.waypoints
        if not wps:
            self._restore_default_pose()
            return
        start = wps[0]
        self._play_alpha = 0.0
        self._recorder_state = RecorderState.READY
        # Restore the full joint snapshot and TCP targets.
        host = np.array(start.joint_q_snapshot, dtype=np.float32)
        wp.copy(self.state_0.joint_q, wp.array(host, dtype=wp.float32, device=self.model.device))
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.right_tf = _tcp_to_transform(start.right_tcp)
        self.left_tf = _tcp_to_transform(start.left_tcp)
        self.pan_tf = _tcp_to_transform(start.pan_tf)
        self.knife_alpha = float(start.knife_alpha)
        self.grip_mode = float(start.grip_mode)
        self.meat_carry_alpha = float(start.carry_alpha)
        self._update_knife_transform(float(start.knife_alpha))
        self._update_pan_transform(_tcp_to_transform(start.pan_tf))
        print("[recorder] reset to segment start")

    def _restore_default_pose(self) -> None:
        if self._joint_q_home is None:
            return
        wp.copy(
            self.state_0.joint_q,
            wp.array(self._joint_q_home.astype(np.float32), dtype=wp.float32, device=self.model.device),
        )
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.right_tf = self._current_tcp_transform(self.right_ee_index, self.right_ee_offset)
        self.left_tf = self._current_tcp_transform(self.left_ee_index, self.left_ee_offset)
        self._recorder_state = RecorderState.IDLE
        self._current_segment = Segment(duration=self._current_segment.duration)
        self._play_alpha = 0.0
        print("[recorder] restored default pose")

    def _toggle_mpm_run(self) -> None:
        if self._recorder_state == RecorderState.MPM_RUN:
            self._recorder_state = RecorderState.IDLE
            print("[recorder] left MPM mode -> IDLE")
        else:
            self._recorder_state = RecorderState.MPM_RUN
            print("[recorder] entered MPM physics mode (running scripted simulate)")

    # -- direct joint edits (sliders, IDLE/READY) ----------------------------

    def _apply_direct_joint_edits(self) -> None:
        """Apply non-IK joint edits staged by the side-panel sliders.

        During IDLE/READY the user can drag finger joints (incl. the
        un-controlled left hand) directly; those writes live in
        ``_non_ik_joint_q_host``. We push them into state_0.joint_q here so
        the next eval_fk reflects them.
        """
        if self._non_ik_joint_q_host is None or not self._non_ik_q_indices:
            return
        host = self.state_0.joint_q.numpy()
        for i, q_idx in enumerate(self._non_ik_q_indices):
            host[q_idx] = float(self._non_ik_joint_q_host[i])
        # Write back; numpy view of a wp.array may not be writable -> use set.
        self.state_0.joint_q = wp.array(host.astype(np.float32), dtype=wp.float32, device=self.model.device)

    # -- serialization -------------------------------------------------------

    def _save_trajectory(self, path: str) -> None:
        # Commit the in-progress segment too, so a save captures everything.
        traj = Trajectory(
            segments=list(self._trajectory.segments),
            start_segment_index=self._trajectory.start_segment_index,
            robot_urdf=self._trajectory.robot_urdf,
            joint_labels=list(self._trajectory.joint_labels),
            version=self._trajectory.version,
        )
        if self._current_segment.waypoints:
            traj.segments.append(self._current_segment)
        path = path or "trajectory.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(traj.to_dict(), f, indent=2)
        print(f"[recorder] saved {len(traj.segments)} segments -> {path}")

    def _load_trajectory(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, json.JSONDecodeError) as err:
            print(f"[recorder] load failed: {err}")
            return
        traj = Trajectory.from_dict(d)
        if traj.joint_labels and traj.joint_labels != list(self.model.joint_label):
            print("[recorder] joint labels mismatch current model; load aborted (retargeting not yet supported).")
            return
        self._trajectory = traj
        # Pick the start segment (clamped) and load it as the current segment.
        idx = max(0, min(traj.start_segment_index, len(traj.segments) - 1)) if traj.segments else 0
        self._play_segment_index = idx
        if traj.segments:
            src = traj.segments[idx]
            self._current_segment = Segment(
                duration=src.duration,
                waypoints=[Waypoint.from_dict(w.to_dict()) for w in src.waypoints],
                label=src.label,
            )
            self._reset_to_segment_start()
        print(f"[recorder] loaded {len(traj.segments)} segments from {path} (start={idx})")

    def _open_load_dialog(self) -> None:
        ui = getattr(self.viewer, "ui", None)
        if ui is not None and hasattr(ui, "open_load_file_dialog"):
            ui.open_load_file_dialog(title="Load trajectory JSON")
            self._awaiting_load_dialog = True
        else:
            print("[recorder] viewer has no file dialog; use --trajectory-file <path>")

    # -- imgui ---------------------------------------------------------------

    def _render_joint_sliders(self, imgui) -> None:
        imgui.separator()
        if imgui.collapsing_header("Joint Tuning"):
            controlled = self._controlled_joint_labels()
            q_start = self.model.joint_q_start.numpy()
            lower = self.model.joint_limit_lower.numpy()
            upper = self.model.joint_limit_upper.numpy()
            # Keep a host copy to edit.
            if self._joint_q_host is None:
                self._joint_q_host = self.state_0.joint_q.numpy().copy()
            else:
                self._joint_q_host = self.state_0.joint_q.numpy().copy()
            for joint_idx, label in enumerate(self.model.joint_label):
                short = label.split("/")[-1]
                q_idx = int(q_start[joint_idx])
                lo = float(lower[q_idx]) if q_idx < len(lower) else -3.14
                hi = float(upper[q_idx]) if q_idx < len(upper) else 3.14
                val = float(self._joint_q_host[q_idx])
                changed, new_val = imgui.slider_float(short, val, lo, hi, "%.3f")
                if changed:
                    self._joint_q_host[q_idx] = float(new_val)
                    # Non-IK joints: stage into the non-ik host for eval_fk.
                    if label not in controlled:
                        # find position in non_ik list
                        if q_idx in self._non_ik_q_indices:
                            pos = self._non_ik_q_indices.index(q_idx)
                            self._non_ik_joint_q_host[pos] = float(new_val)
                    else:
                        # IK-controlled joint edited directly: write into state
                        # joint_q so IK seeds from it next play; for arm joints
                        # the TCP gizmo is the preferred path, but allow direct
                        # edits too.
                        host = self.state_0.joint_q.numpy()
                        host[q_idx] = float(new_val)
                        self.state_0.joint_q = wp.array(
                            host.astype(np.float32), dtype=wp.float32, device=self.model.device
                        )
                        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
            imgui.separator()
            _, self._edit_hand_alpha = imgui.slider_float(
                "Right hand close (alpha)", float(getattr(self, "_edit_hand_alpha", 0.0)), 0.0, 1.0, "%.3f"
            )

    def _render_recorder_hud(self, imgui) -> None:
        imgui.begin("Trajectory Recorder")
        imgui.text(f"state: {self._recorder_state}")
        n_segs = len(self._trajectory.segments)
        n_wp = len(self._current_segment.waypoints)
        imgui.text(f"segment: {n_segs} committed, current has {n_wp} waypoint(s)")
        if self._recorder_state == RecorderState.PLAYING:
            imgui.text(f"play alpha: {self._play_alpha:.2f}")
            # ASCII progress bar.
            n_bar = 24
            filled = int(round(self._play_alpha * n_bar))
            imgui.text("  [" + "#" * filled + "." * (n_bar - filled) + "]")
        imgui.separator()
        imgui.text("keys:")
        imgui.text(f"  {self.KEY_CAPTURE_START}/{self.KEY_CAPTURE_MID}/{self.KEY_CAPTURE_END}: capture start/mid/end")
        imgui.text(f"  {self.KEY_PLAY}: play  {self.KEY_RESET_SEGMENT}: reset to start")
        imgui.text(f"  {self.KEY_RESTORE_HOME}: home  {self.KEY_NEW_SEGMENT}: new segment")
        imgui.text(f"  {self.KEY_TOGGLE_MPM}: MPM physics mode  Space: pause")
        imgui.text(f"  {self.KEY_SAVE}/{self.KEY_LOAD}: save/load (upper-case function keys)")
        imgui.separator()
        _, self._save_path = imgui.input_text("save path", self._save_path)
        if imgui.button("Save"):
            self._save_trajectory(self._save_path)
        imgui.same_line()
        if imgui.button("Load"):
            self._open_load_dialog()
        # Consume a pending file-dialog result if any.
        ui = getattr(self.viewer, "ui", None)
        if ui is not None and hasattr(ui, "consume_file_dialog_result"):
            res = ui.consume_file_dialog_result()
            if res:
                # res may be a path string or object with .path
                path = res if isinstance(res, str) else getattr(res, "path", str(res))
                self._load_trajectory(path)
        imgui.separator()
        imgui.text("committed segments:")
        for i, seg in enumerate(self._trajectory.segments):
            label = seg.label or f"seg {i}"
            sel = i == self._play_segment_index
            clicked, _ = imgui.selectable(f"{i}: {label} ({len(seg.waypoints)} pts)##{i}", sel)
            if clicked:
                self._play_segment_index = i
                self._current_segment = Segment(
                    duration=seg.duration,
                    waypoints=[Waypoint.from_dict(w.to_dict()) for w in seg.waypoints],
                    label=seg.label,
                )
                self._reset_to_segment_start()
        imgui.end()

    # -- misc ----------------------------------------------------------------

    def _print_help(self) -> None:
        print(
            "[recorder] Trajectory recorder ready. "
            f"Drag the right/left TCP gizmos; "
            f"{self.KEY_CAPTURE_START}/{self.KEY_CAPTURE_MID}/{self.KEY_CAPTURE_END} capture, "
            f"{self.KEY_PLAY} play, {self.KEY_NEW_SEGMENT} new segment, "
            f"{self.KEY_RESET_SEGMENT} reset, {self.KEY_RESTORE_HOME} home, "
            f"{self.KEY_TOGGLE_MPM} MPM physics, {self.KEY_SAVE}/{self.KEY_LOAD} save/load."
        )

    # -- tests (Example convention) ------------------------------------------

    def test_post_step(self) -> None:
        q = self.state_0.joint_q.numpy()
        if not np.all(np.isfinite(q)):
            raise AssertionError("joint_q contains non-finite values")

    def test_final(self) -> None:
        self.test_post_step()

    # -- parser --------------------------------------------------------------

    @staticmethod
    def create_parser():
        parser = WaicKitchenExample.create_parser()
        parser.set_defaults(num_frames=100000)
        parser.add_argument(
            "--trajectory-file",
            type=str,
            default="",
            help="JSON trajectory file to load at startup.",
        )
        parser.add_argument(
            "--start-segment",
            type=int,
            default=0,
            help="Segment index to start playback from.",
        )
        parser.add_argument(
            "--play-duration",
            type=float,
            default=2.0,
            help="Default playback duration per segment [s].",
        )
        parser.add_argument(
            "--freeze-mpm",
            action="store_true",
            default=True,
            help="Freeze MPM solver during IK playback (default).",
        )
        parser.add_argument(
            "--no-freeze-mpm",
            dest="freeze_mpm",
            action="store_false",
            help="Run MPM physics during playback.",
        )
        return parser


if __name__ == "__main__":
    _parser = TrajectoryRecorderExample.create_parser()
    _viewer, _args = newton.examples.init(_parser)
    newton.examples.run(TrajectoryRecorderExample(_viewer, _args), _args)

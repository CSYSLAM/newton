# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Tune and record physical Dexforce W1 hand poses for the MJVBD-v2 grasp scene.

The recorder keeps the robot kinematic and the cubes, table, and bag physical.
Changing a slider only changes the next robot target pose.  ``Step once`` advances
the actual :class:`newton.solvers.SolverMJVBDV2` simulation, including rigid and
cloth contacts, instead of attaching a cube to the hand.

Run from the repository root::

    uv run --extra examples -m newton.examples vbd_mjvbd_v2_dexforce_grasp_hand_pose_recorder \
        --viewer gl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import warp as wp

import newton
import newton.examples

from . import example_vbd_mjvbd_v2_dexforce_grasp_rigid_into_bag as grasp_example

FPS = grasp_example.FPS
IK_ITERATIONS = grasp_example.IK_ITERATIONS
TCP_POSITION_LIMIT_MM = 300.0


@wp.kernel
def _interpolate_q(
    q0: wp.array[float],
    q1: wp.array[float],
    alpha: float,
    out: wp.array[float],
):
    """Interpolate the kinematic robot coordinates for one substep."""

    i = wp.tid()
    out[i] = q0[i] * (1.0 - alpha) + q1[i] * alpha


@wp.kernel
def _joint_velocity(
    q0: wp.array[float],
    q1: wp.array[float],
    inv_dt: float,
    out: wp.array[float],
):
    """Compute the commanded kinematic joint velocity."""

    i = wp.tid()
    out[i] = (q1[i] - q0[i]) * inv_dt


@wp.kernel
def _pin_bag_rim(
    pinned_indices: wp.array[wp.int32],
    original_positions: wp.array[wp.vec3],
    pos_0: wp.array[wp.vec3],
    pos_1: wp.array[wp.vec3],
):
    """Keep the bag's inactive top-rim particles at their authored positions."""

    i = wp.tid()
    particle = pinned_indices[i]
    position = original_positions[i]
    pos_0[particle] = position
    pos_1[particle] = position


class Example(grasp_example.Example):
    """Interactive physical hand-pose recorder for the two-cube scene."""

    HAND_UI_SUFFIXES = (
        "HAND_THUMB1",
        "HAND_THUMB2",
        "HAND_INDEX",
        "INDEX_PIP",
        "HAND_MIDDLE",
        "MIDDLE_PIP",
        "HAND_RING",
        "RING_PIP",
        "HAND_PINKY",
        "PINKY_PIP",
    )
    HAND_UI_LABELS: ClassVar[dict[str, str]] = {
        "HAND_THUMB1": "THUMB1",
        "HAND_THUMB2": "THUMB2",
        "HAND_INDEX": "INDEX",
        "INDEX_PIP": "INDEX PIP",
        "HAND_MIDDLE": "MIDDLE",
        "MIDDLE_PIP": "MIDDLE PIP",
        "HAND_RING": "RING",
        "RING_PIP": "RING PIP",
        "HAND_PINKY": "PINKY",
        "PINKY_PIP": "PINKY PIP",
    }

    def __init__(self, viewer, args):
        super().__init__(viewer, args)

        self._manual_target_q = wp.clone(self.state_0.joint_q)
        self._manual_q_start = wp.zeros_like(self.model.joint_q)
        self._manual_q_end = wp.zeros_like(self.model.joint_q)
        self._zero_joint_qd = wp.zeros_like(self.state_0.joint_qd)
        self._ui_controls: dict[str, dict[str, Any]] = {}
        self._trajectory_frames: list[dict[str, Any]] = []
        self._root = None
        self._status_var = None
        self._last_target_signature = None

        self._create_hand_controls()
        self._refresh_manual_target()
        self._apply_preview_pose()
        self._store_trajectory_frame()

    def _build_joint_target_cache(self):
        """Create a small compatibility cache without auto-driving the robot."""

        initial_q = np.asarray(self.model.joint_q.numpy(), dtype=np.float32)
        self.cached_frame_count = max(1, int(self.args.num_frames))
        self.cached_joint_targets = wp.array(
            np.repeat(initial_q[None, :], self.cached_frame_count + 1, axis=0),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.cached_grips = np.zeros(self.cached_frame_count + 1, dtype=np.float32)
        self.cached_objects = np.full(self.cached_frame_count + 1, -1, dtype=np.int32)

    @staticmethod
    def create_parser():
        """Create the command-line parser for the hand recorder."""

        parser = grasp_example.Example.create_parser()
        parser.set_defaults(num_frames=1, paused=True)
        parser.add_argument(
            "--pose-output",
            default="vbd_mjvbd_v2_hand_pose.json",
            help="Path for the JSON file written by Save pose JSON.",
        )
        parser.add_argument(
            "--trajectory-output",
            default="vbd_mjvbd_v2_hand_trajectory.json",
            help="Path for the JSON file written by Save trajectory.",
        )
        parser.add_argument(
            "--recorder-no-gui",
            action="store_true",
            help="Build the physical scene and render one frame without opening Tk.",
        )
        return parser

    def _create_hand_controls(self):
        q_start = self.model.joint_q_start.numpy()
        q_lower = self.model.joint_limit_lower.numpy()
        q_upper = self.model.joint_limit_upper.numpy()
        asset_q = self.model.joint_q.numpy()

        for side, base_transform, body in (
            ("LEFT", self.left_home, self.left_body),
            ("RIGHT", self.right_home, self.right_body),
        ):
            joint_indices = {}
            joint_degrees = {}
            joint_limits = {}
            for suffix in self.HAND_UI_SUFFIXES:
                joint = self._joint_index(f"{side}_{suffix}")
                index = int(q_start[joint])
                lower = float(np.degrees(q_lower[index]))
                upper = float(np.degrees(q_upper[index]))
                if lower > upper:
                    lower, upper = upper, lower
                # Start from the URDF/model pose; the recorder must not impose
                # a grasp posture before the user moves the fingers.
                default = float(np.degrees(asset_q[index]))
                default = float(np.clip(default, lower, upper))
                joint_indices[suffix] = index
                joint_degrees[suffix] = default
                joint_limits[suffix] = (lower, upper)

            self._ui_controls[side] = {
                "body": body,
                "base_transform": self._copy_transform(base_transform),
                "gizmo_transform": self._copy_transform(base_transform),
                "joint_indices": joint_indices,
                "joint_degrees": joint_degrees,
                "joint_limits": joint_limits,
                "position_mm": np.zeros(3, dtype=np.float32),
                "rotation_deg": np.zeros(3, dtype=np.float32),
                "target_transform": self._copy_transform(base_transform),
            }

    def _offset_transform(self, control: dict[str, Any]):
        base = control["gizmo_transform"]
        base_position = np.asarray(wp.transform_get_translation(base), dtype=np.float32)
        base_rotation = wp.transform_get_rotation(base)
        position = base_position + np.asarray(control["position_mm"], dtype=np.float32) * 1.0e-3

        rotation = np.zeros(3, dtype=np.float32)
        rotation[:] = np.radians(control["rotation_deg"])
        offset_x = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(rotation[0]))
        offset_y = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(rotation[1]))
        offset_z = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rotation[2]))
        offset_rotation = self._quat_mul(self._quat_mul(offset_x, offset_y), offset_z)
        target_rotation = self._normal_quat(self._quat_mul(offset_rotation, base_rotation))
        return wp.transform(wp.vec3(*position), target_rotation)

    def _refresh_manual_target(self):
        """Solve both arm TCP targets, then apply the ten hand joint targets."""

        target_transforms = {}
        for side, control in self._ui_controls.items():
            target = self._offset_transform(control)
            control["target_transform"] = target
            target_transforms[side] = target

        current_q = np.asarray(self._manual_target_q.numpy(), dtype=np.float32)
        self.ik_q.assign(current_q[: self.ik_model.joint_coord_count].reshape(1, -1))
        left_target = target_transforms["LEFT"]
        right_target = target_transforms["RIGHT"]
        self.left_obj.set_target_position(0, wp.transform_get_translation(left_target))
        self.left_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(left_target)))
        self.right_obj.set_target_position(0, wp.transform_get_translation(right_target))
        self.right_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(right_target)))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=IK_ITERATIONS)

        target_q = current_q
        target_q[: self.ik_model.joint_coord_count] = self.ik_q.numpy()[0]
        lock_indices = self.lock_indices.numpy()
        target_q[lock_indices] = self.lock_values.numpy()
        for control in self._ui_controls.values():
            for suffix, index in control["joint_indices"].items():
                target_q[index] = np.radians(control["joint_degrees"][suffix])
        self._manual_target_q.assign(target_q)
        self._last_target_signature = self._target_signature()

    @staticmethod
    def _copy_transform(transform):
        """Copy a transform so the Viewer gizmo can mutate its own target."""

        position = wp.transform_get_translation(transform)
        rotation = wp.transform_get_rotation(transform)
        return wp.transform(
            wp.vec3(float(position[0]), float(position[1]), float(position[2])),
            wp.quat(float(rotation[0]), float(rotation[1]), float(rotation[2]), float(rotation[3])),
        )

    def _target_signature(self):
        values = []
        for control in self._ui_controls.values():
            gizmo = control["gizmo_transform"]
            values.extend(float(value) for value in wp.transform_get_translation(gizmo))
            values.extend(float(value) for value in wp.transform_get_rotation(gizmo))
            values.extend(float(value) for value in control["position_mm"])
            values.extend(float(value) for value in control["rotation_deg"])
        return tuple(values)

    def _apply_preview_pose(self):
        """Show the requested kinematic hand pose without advancing physics."""

        self.state_0.joint_q.assign(self._manual_target_q)
        wp.copy(self.state_0.joint_qd, self._zero_joint_qd)
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
            body_flag_filter=newton.BodyFlags.KINEMATIC,
        )

    def _reset_physics(self):
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self._manual_target_q = wp.clone(self.state_0.joint_q)
        for control in self._ui_controls.values():
            control["gizmo_transform"] = self._copy_transform(control["base_transform"])
            control["position_mm"].fill(0.0)
            control["rotation_deg"].fill(0.0)
        self._refresh_manual_target()
        self._apply_preview_pose()
        self.sim_time = 0.0
        self.frame_index = 0
        self._trajectory_frames.clear()
        self._store_trajectory_frame()

    def step_once(self):
        """Advance one physical frame using the current hand and TCP target."""

        wp.copy(self._manual_q_start, self.state_0.joint_q)
        wp.copy(self._manual_q_end, self._manual_target_q)
        for substep in range(self.sim_substeps):
            wp.launch(
                _pin_bag_rim,
                self.bag_pinned_indices.shape[0],
                [
                    self.bag_pinned_indices,
                    self.bag_pinned_original,
                    self.state_0.particle_q,
                    self.state_1.particle_q,
                ],
                device=self.device,
            )
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (substep + 1) / self.sim_substeps
            wp.launch(
                _interpolate_q,
                self.ik_model.joint_coord_count,
                [self._manual_q_start, self._manual_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                _joint_velocity,
                self.ik_model.joint_dof_count,
                [self._manual_q_start, self._manual_q_end, 1.0 / self.frame_dt, self.state_0.joint_qd],
                device=self.device,
            )
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.state_0.joint_q.assign(self._manual_target_q)
        newton.eval_fk(
            self.model,
            self.state_0.joint_q,
            self.state_0.joint_qd,
            self.state_0,
            body_flag_filter=newton.BodyFlags.KINEMATIC,
        )
        self.sim_time += self.frame_dt
        self.frame_index += 1
        self._store_trajectory_frame()

    def _transform_dict(self, transform):
        position = np.asarray(wp.transform_get_translation(transform), dtype=np.float64)
        quaternion = np.asarray(wp.transform_get_rotation(transform), dtype=np.float64)
        return {
            "position_m": [float(value) for value in position],
            "quaternion_xyzw": [float(value) for value in quaternion],
            "quaternion_wxyz": [float(quaternion[3]), *[float(value) for value in quaternion[:3]]],
        }

    def _body_pose_dict(self, body: int):
        return self._transform_dict(wp.transform(*self.state_0.body_q.numpy()[body]))

    def _capture_frame(self):
        actual_q = np.asarray(self.state_0.joint_q.numpy(), dtype=np.float64)
        target_q = np.asarray(self._manual_target_q.numpy(), dtype=np.float64)
        hands = {}
        for side, control in self._ui_controls.items():
            indices = control["joint_indices"]
            target_joints = {suffix: float(target_q[index]) for suffix, index in indices.items()}
            actual_tcp = self._tcp(self.state_0, control["body"])
            hands[side] = {
                "gizmo_tcp_world": self._transform_dict(control["gizmo_transform"]),
                "target_tcp_world": self._transform_dict(control["target_transform"]),
                "actual_tcp_world": self._transform_dict(actual_tcp),
                "target_joints_radians": target_joints,
                "target_joints_degrees": {suffix: float(np.degrees(value)) for suffix, value in target_joints.items()},
                "actual_j7_world": self._body_pose_dict(control["body"]),
            }

        objects = []
        body_q = self.state_0.body_q.numpy()
        for body in self.object_bodies:
            objects.append(self._transform_dict(wp.transform(*body_q[body])))
        return {
            "frame": int(self.frame_index),
            "time_s": float(self.sim_time),
            "hands": hands,
            "objects": objects,
            "joint_q_radians": [float(value) for value in actual_q],
            "target_joint_q_radians": [float(value) for value in target_q],
        }

    def _store_trajectory_frame(self):
        frame = self._capture_frame()
        for index, old_frame in enumerate(self._trajectory_frames):
            if old_frame["frame"] == frame["frame"]:
                self._trajectory_frames[index] = frame
                return
        self._trajectory_frames.append(frame)

    @staticmethod
    def _write_json(path_value: str, payload: dict[str, Any]):
        path = Path(path_value).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def save_pose(self):
        """Save the current requested and actual hand poses."""

        path = self._write_json(
            self.args.pose_output,
            {
                "format": "newton_mjvbd_v2_hand_pose_v1",
                "frame_dt_s": self.frame_dt,
                "pose": self._capture_frame(),
            },
        )
        self._set_status(f"Saved pose: {path}")

    def save_trajectory(self):
        """Save all manually stepped physical frames."""

        self._store_trajectory_frame()
        path = self._write_json(
            self.args.trajectory_output,
            {
                "format": "newton_mjvbd_v2_hand_trajectory_v1",
                "frame_dt_s": self.frame_dt,
                "sim_substeps": self.sim_substeps,
                "physics_solver": "SolverMJVBDV2",
                "frames": self._trajectory_frames,
            },
        )
        self._set_status(f"Saved trajectory: {path}")

    def test_final(self):
        """Verify that a recorder step keeps all physical states finite."""

        assert np.all(np.isfinite(self.state_0.body_q.numpy()))
        assert np.all(np.isfinite(self.state_0.particle_q.numpy()))
        assert np.all(np.isfinite(self.state_0.joint_q.numpy()))

    def render(self):
        """Render the current physical state and service the GL event queue."""

        if self._target_signature() != self._last_target_signature:
            self._refresh_manual_target()
            self._apply_preview_pose()
        self.viewer.begin_frame(self.sim_time)
        if hasattr(self.viewer, "log_gizmo"):
            for side, control in self._ui_controls.items():
                self.viewer.log_gizmo(f"target_tcp_{side.lower()}", control["gizmo_transform"])
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def _set_status(self, message: str):
        if self._status_var is not None:
            self._status_var.set(message)

    def _on_control_changed(self, side: str, variables):
        control = self._ui_controls[side]
        for suffix, variable in variables["joints"].items():
            control["joint_degrees"][suffix] = float(variable.get())
        control["position_mm"] = np.asarray(
            [float(variable.get()) for variable in variables["position"]], dtype=np.float32
        )
        control["rotation_deg"] = np.asarray(
            [float(variable.get()) for variable in variables["rotation"]], dtype=np.float32
        )
        self._refresh_manual_target()
        self._apply_preview_pose()
        self.render()

    def _make_scale(
        self,
        parent,
        row: int,
        label: str,
        variable,
        minimum: float,
        maximum: float,
        resolution: float,
        command=None,
    ):
        import tkinter as tk  # noqa: PLC0415

        ttk = self._ttk
        ttk.Label(parent, text=label, width=20).grid(row=row, column=0, sticky="w", padx=6, pady=2)
        value_label = ttk.Label(parent, textvariable=variable._display_var, width=8, anchor="e")
        value_label.grid(row=row, column=1, sticky="e", padx=3)
        scale = tk.Scale(
            parent,
            variable=variable,
            from_=minimum,
            to=maximum,
            resolution=resolution,
            orient="horizontal",
            showvalue=False,
            length=435,
            highlightthickness=0,
            command=command,
        )
        scale.grid(row=row, column=2, sticky="ew", padx=4)

        def update_display(*_):
            variable._display_var.set(f"{float(variable.get()):.1f}")

        variable.trace_add("write", update_display)
        update_display()
        return scale

    def _build_hand_tab(self, notebook, side: str):
        import tkinter as tk  # noqa: PLC0415
        from tkinter import ttk  # noqa: PLC0415

        control = self._ui_controls[side]
        tab = ttk.Frame(notebook, padding=8)
        tab.columnconfigure(2, weight=1)
        variables = {"joints": {}, "position": [], "rotation": []}
        row = 0
        joints_box = ttk.LabelFrame(tab, text="Joint angles (degrees)", padding=5)
        joints_box.grid(row=row, column=0, columnspan=3, sticky="nsew")
        joints_box.columnconfigure(2, weight=1)
        row += 1
        for suffix in self.HAND_UI_SUFFIXES:
            variable = tk.DoubleVar(value=control["joint_degrees"][suffix])
            variable._display_var = tk.StringVar()
            variables["joints"][suffix] = variable
            minimum, maximum = control["joint_limits"][suffix]
            self._make_scale(
                joints_box,
                len(variables["joints"]) - 1,
                f"{side}_{self.HAND_UI_LABELS[suffix]}",
                variable,
                minimum,
                maximum,
                1.0,
                command=lambda _value, side=side, variables=variables: self._on_control_changed(side, variables),
            )

        tcp_box = ttk.LabelFrame(tab, text="TCP fine offset / rotation relative to gizmo", padding=5)
        tcp_box.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        tcp_box.columnconfigure(2, weight=1)
        row += 1
        for index, label in enumerate(("Position X (mm)", "Position Y (mm)", "Position Z (mm)")):
            variable = tk.DoubleVar(value=0.0)
            variable._display_var = tk.StringVar()
            variables["position"].append(variable)
            self._make_scale(
                tcp_box,
                index,
                label,
                variable,
                -TCP_POSITION_LIMIT_MM,
                TCP_POSITION_LIMIT_MM,
                1.0,
                command=lambda _value, side=side, variables=variables: self._on_control_changed(side, variables),
            )
        for index, label in enumerate(("Rotation X (deg)", "Rotation Y (deg)", "Rotation Z (deg)"), start=3):
            variable = tk.DoubleVar(value=0.0)
            variable._display_var = tk.StringVar()
            variables["rotation"].append(variable)
            self._make_scale(
                tcp_box,
                index,
                label,
                variable,
                -60.0,
                60.0,
                1.0,
                command=lambda _value, side=side, variables=variables: self._on_control_changed(side, variables),
            )
        return tab

    def run_pose_recorder(self):
        """Run the Tk recorder while the Newton viewer is pumped by ``after``."""

        if self.args.recorder_no_gui:
            self.render()
            self.viewer.close()
            return

        import tkinter as tk  # noqa: PLC0415
        from tkinter import ttk  # noqa: PLC0415

        if hasattr(self.viewer, "hide_loading_splash"):
            self.viewer.hide_loading_splash()

        self._ttk = ttk
        root = tk.Tk()
        self._root = root
        root.title("MJVBD-v2 W1 physical hand-pose recorder")
        root.geometry("700x950")
        root.minsize(650, 780)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        notebook.add(self._build_hand_tab(notebook, "LEFT"), text="LEFT")
        notebook.add(self._build_hand_tab(notebook, "RIGHT"), text="RIGHT")

        buttons = ttk.Frame(root, padding=8)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Step once", command=self._step_from_ui).pack(side="left", padx=3)
        ttk.Button(buttons, text="Reset robot pose", command=self._reset_from_ui).pack(side="left", padx=3)
        ttk.Button(buttons, text="Save pose JSON", command=self.save_pose).pack(side="left", padx=3)
        ttk.Button(buttons, text="Save trajectory", command=self.save_trajectory).pack(side="left", padx=3)
        self._status_var = tk.StringVar(value="Physics ready; adjust a hand, then Step once.")
        ttk.Label(root, textvariable=self._status_var, anchor="w").pack(fill="x", padx=12, pady=(0, 7))

        def close():
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", close)

        def pump_viewer():
            if not self.viewer.is_running():
                root.destroy()
                return
            self.render()
            root.after(max(1, int(1000.0 / FPS)), pump_viewer)

        root.after(0, pump_viewer)
        try:
            root.mainloop()
        finally:
            self.viewer.close()

    def _step_from_ui(self):
        try:
            self.step_once()
            self._set_status(f"Physics step {self.frame_index}; cube state was solved dynamically.")
            self.render()
        except Exception as error:
            self._set_status(f"Step failed: {error}")
            raise

    def _reset_from_ui(self):
        self._reset_physics()
        self._set_status("Reset physical scene and robot pose.")
        self.render()


def main():
    """Launch the interactive physical hand-pose recorder."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    if args.test:
        example.step_once()
        example.test_final()
        viewer.close()
    else:
        example.run_pose_recorder()


if __name__ == "__main__":
    main()

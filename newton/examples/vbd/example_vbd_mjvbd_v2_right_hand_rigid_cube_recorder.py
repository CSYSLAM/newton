# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Interactively tune a physical W1 right-hand rigid-cube grasp.

The scene mirrors ``example_vbd_mjvbd_v2_right_hand_soft_cube_recorder.py`` but
replaces the volumetric soft cube with one dynamic rigid box.  Only the hand's
URDF collision meshes participate in the grasp; no auxiliary fingertip pads or
kinematic attachment are used.

Run from the repository root::

    uv run --extra examples -m newton.examples vbd_mjvbd_v2_right_hand_rigid_cube_recorder --viewer gl
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd import example_vbd_mjvbd_v2_right_hand_soft_cube_recorder as soft_recorder
from newton.solvers import SolverMJVBDV2

CUBE_DENSITY = 1500.0
CUBE_MARGIN = 0.001
CONTACT_KE = 3.0e4
CONTACT_KD = 0.5
CONTACT_MU = 50.0

# Match the pre-closure approach pose in
# example_vbd_mjvbd_v2_right_hand_recorded_soft_cube_into_bag.py.
INITIAL_HAND_ROOT = wp.transform(
    wp.vec3(-0.16214203834533691, -2.838686943054199, 1.3409454822540283),
    wp.quat(0.09465623646974564, 0.9546480774879456, -0.2820824682712555, 0.010803722776472569),
)
INITIAL_HAND_JOINTS = {
    "RIGHT_HAND_THUMB1": 6.0,
    "RIGHT_HAND_THUMB2": 90.0,
    "RIGHT_HAND_INDEX": 41.0,
    "RIGHT_INDEX_PIP": 24.0,
    "RIGHT_HAND_MIDDLE": 57.0,
    "RIGHT_MIDDLE_PIP": 0.0,
    "RIGHT_HAND_RING": 48.0,
    "RIGHT_RING_PIP": 15.0,
    "RIGHT_HAND_PINKY": 24.0,
    "RIGHT_PINKY_PIP": 26.0,
}


class Example(soft_recorder.Example):
    """Tune a mesh-only physical grasp of one dynamic rigid cube."""

    RIGID_BODY_CONTACT_BUFFER_SIZE = 4096

    def __init__(self, viewer, args):
        self.particle_self_contact_enabled = False
        super().__init__(viewer, args)
        if self._initial_keyframe is None:
            self._apply_reference_initial_pose()

    def _apply_reference_initial_pose(self):
        """Set the default target to the recorded soft-cube approach pose."""

        self.gizmo_transform = self._copy_transform(INITIAL_HAND_ROOT)
        self.position_mm.fill(0.0)
        self.rotation_deg.fill(0.0)
        self.joint_degrees = dict(INITIAL_HAND_JOINTS)
        self._refresh_target()
        self._set_initial_hand_pose()

    def _reset_physics(self):
        super()._reset_physics()
        if self._initial_keyframe is None:
            self._apply_reference_initial_pose()

    def _build_scene(self):
        if not soft_recorder.RIGHT_HAND_URDF.is_file():
            raise FileNotFoundError(f"Right-hand URDF not found: {soft_recorder.RIGHT_HAND_URDF}")

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg.ke = CONTACT_KE
        builder.default_shape_cfg.kd = CONTACT_KD
        builder.default_shape_cfg.mu = CONTACT_MU
        # Preserve the reference recorder's SDF representation of the original
        # hand meshes; the rigid cube still uses the full rigid contact path.
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMJVBDV2.register_custom_attributes(builder)

        articulation_start = builder.articulation_count
        builder.add_urdf(
            str(soft_recorder.RIGHT_HAND_URDF),
            xform=soft_recorder.HAND_HOME,
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.hand_articulations = tuple(range(articulation_start, builder.articulation_count))
        self.hand_shape_end = builder.shape_count
        for body in range(builder.body_count):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)

        table_cfg = newton.ModelBuilder.ShapeConfig(ke=3.0e5, kd=1.0e-4, mu=0.9, is_visible=True)
        table_cfg.configure_sdf(force_sdf=True)
        builder.add_shape_box(
            -1,
            xform=wp.transform(soft_recorder.TABLE_POS, soft_recorder.TABLE_ROTATION),
            hx=soft_recorder.TABLE_HALF_EXTENTS[0],
            hy=soft_recorder.TABLE_HALF_EXTENTS[1],
            hz=soft_recorder.TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
            label="hand_tuning_table",
        )
        builder.add_ground_plane(height=-0.00377202, label="hand_tuning_ground")

        cube_cfg = newton.ModelBuilder.ShapeConfig(
            density=CUBE_DENSITY,
            ke=CONTACT_KE,
            kd=CONTACT_KD,
            mu=CONTACT_MU,
            margin=CUBE_MARGIN,
        )
        cube_cfg.configure_sdf(force_sdf=True)
        self.cube_body = builder.add_body(
            xform=wp.transform(soft_recorder.CUBE_CENTRE, wp.quat_identity()),
            label="tunable_rigid_cube",
        )
        self.cube_shape = builder.shape_count
        builder.add_shape_box(
            self.cube_body,
            hx=soft_recorder.CUBE_HALF_EXTENTS[0],
            hy=soft_recorder.CUBE_HALF_EXTENTS[1],
            hz=soft_recorder.CUBE_HALF_EXTENTS[2],
            cfg=cube_cfg,
            color=(0.90, 0.32, 0.18),
            label="tunable_rigid_cube_shape",
        )

        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        for shape in range(builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes

        builder.color()
        self.model = builder.finalize(requires_grad=False)

    def _contact_counts(self) -> tuple[int, int]:
        """Return current hand-cube and total rigid contact counts."""

        contacts = self.solver.contacts
        total = int(contacts.rigid_contact_count.numpy()[0])
        shape_0 = contacts.rigid_contact_shape0.numpy()
        shape_1 = contacts.rigid_contact_shape1.numpy()
        active = min(total, shape_0.shape[0])
        shape_0 = shape_0[:active]
        shape_1 = shape_1[:active]
        hand_cube = np.count_nonzero(
            ((shape_0 == self.cube_shape) & (shape_1 >= 0) & (shape_1 < self.hand_shape_end))
            | ((shape_1 == self.cube_shape) & (shape_0 >= 0) & (shape_0 < self.hand_shape_end))
        )
        return int(hand_cube), total

    def _capture_frame(self):
        current_q = self.state_0.joint_q.numpy()
        root_q = current_q[self.root_q_start : self.root_q_start + 7]
        cube_q = self.state_0.body_q.numpy()[self.cube_body]
        cube_qd = self.state_0.body_qd.numpy()[self.cube_body]
        hand_cube_contacts, total_rigid_contacts = self._contact_counts()
        return {
            "frame": self.frame_index,
            "time_s": self.sim_time,
            "gizmo_world": self._transform_dict(self.gizmo_transform),
            "position_offset_mm": self.position_mm.tolist(),
            "rotation_offset_deg": self.rotation_deg.tolist(),
            "target_root_pose": self._transform_dict(self.target_transform),
            "target_finger_joints_degrees": dict(self.joint_degrees),
            "root_pose": {
                "position_m": [float(value) for value in root_q[:3]],
                "quaternion_xyzw": [float(value) for value in root_q[3:]],
            },
            "finger_joints_radians": {name: float(current_q[index]) for name, index in self.hand_joint_indices.items()},
            "finger_joints_degrees": {
                name: float(np.degrees(current_q[index])) for name, index in self.hand_joint_indices.items()
            },
            "rigid_cube_pose": {
                "position_m": [float(value) for value in cube_q[:3]],
                "quaternion_xyzw": [float(value) for value in cube_q[3:]],
            },
            "rigid_cube_twist": [float(value) for value in cube_qd],
            "hand_cube_contact_count": hand_cube_contacts,
            "total_rigid_contact_count": total_rigid_contacts,
        }

    def run_recorder(self):
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
        root.title("MJVBD-v2 W1 right-hand rigid-cube recorder")
        root.geometry("710x650")
        root.minsize(660, 630)
        self._build_controls(root)
        buttons = ttk.Frame(root, padding=8)
        buttons.pack(fill="x")
        ttk.Button(buttons, text="Record keyframe", command=self._record_keyframe_from_ui).pack(side="left", padx=3)
        ttk.Button(buttons, text="Reset physics", command=self._reset_from_ui).pack(side="left", padx=3)
        ttk.Button(buttons, text="Save pose JSON", command=self.save_pose).pack(side="left", padx=3)
        ttk.Button(buttons, text="Save trajectory", command=self.save_trajectory).pack(side="left", padx=3)
        self._status_var = tk.StringVar(
            value="Realtime rigid physics running; adjust the hand, then record a stable grasp keyframe."
        )
        ttk.Label(root, textvariable=self._status_var, anchor="w").pack(fill="x", padx=12, pady=(0, 7))

        root.protocol("WM_DELETE_WINDOW", root.destroy)

        def pump_viewer():
            if not self.viewer.is_running():
                root.destroy()
                return
            self.step_once()
            self.render()
            root.after(max(1, int(1000.0 / soft_recorder.FPS)), pump_viewer)

        root.after(0, pump_viewer)
        try:
            root.mainloop()
        finally:
            self.viewer.close()

    def _record_keyframe_from_ui(self):
        self._store_trajectory_frame()
        keyframe_path = self._write_json(
            self.args.keyframe_output,
            {"format": "newton_w1_right_hand_keyframe_v1", "keyframe": self._trajectory_frames[-1]},
        )
        trajectory_path = self._write_json(
            self.args.trajectory_output,
            {
                "format": "newton_w1_right_hand_trajectory_v1",
                "frame_dt_s": self.frame_dt,
                "frames": self._trajectory_frames,
            },
        )
        hand_cube_contacts, total_rigid_contacts = self._contact_counts()
        self._set_status(
            f"Recorded keyframe {len(self._trajectory_frames)} at physics frame {self.frame_index}; "
            f"hand-cube contacts: {hand_cube_contacts}, total rigid contacts: {total_rigid_contacts}. "
            f"Saved: {keyframe_path}, {trajectory_path}"
        )

    def _reset_from_ui(self):
        self._reset_physics()
        self._set_status("Reset the hand and rigid cube to their initial states.")
        self.render()

    def test_final(self):
        """Verify that one physical step keeps the hand and rigid cube finite."""

        assert np.all(np.isfinite(self.state_0.joint_q.numpy()))
        assert np.all(np.isfinite(self.state_0.body_q.numpy()))
        assert np.all(np.isfinite(self.state_0.body_qd.numpy()))

    @staticmethod
    def create_parser():
        parser = soft_recorder.Example.create_parser()
        parser.set_defaults(
            pose_output="vbd_w1_right_hand_rigid_cube_pose.json",
            trajectory_output="vbd_w1_right_hand_rigid_cube_trajectory.json",
            keyframe_output="vbd_w1_right_hand_rigid_cube_last_keyframe.json",
        )
        return parser


def main():
    """Launch the interactive right-hand rigid-cube recorder."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    if args.test:
        example.step_once()
        example.test_final()
        viewer.close()
    else:
        example.run_recorder()


if __name__ == "__main__":
    main()

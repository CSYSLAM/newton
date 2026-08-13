# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run the recorded soft-then-rigid placement trajectory on the full W1.

The right arm tracks the exact hand-root and finger-joint trajectory from
``example_vbd_mjvbd_v2_right_hand_recorded_soft_then_rigid_cube_into_bag``.
Each hand-root target is converted to the W1 right-arm TCP and solved with IK.
Both cubes remain physical, and only collision meshes imported from the W1
URDF participate in either grasp; no auxiliary fingertip pads are created.

Run from the repository root::

    uv run --extra examples -m newton.examples vbd_mjvbd_v2_dexforce_recorded_soft_then_rigid_cube_into_bag --viewer gl
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mjvbdv2 import example_vbd_mjvbd_v2_dexforce_grasp_rigid_into_bag_rigid as rigid_reference
from newton.examples.mjvbdv2 import example_vbd_mjvbd_v2_dexforce_recorded_soft_cube_into_bag as recorded_soft
from newton.examples.mjvbdv2 import (
    example_vbd_mjvbd_v2_right_hand_recorded_soft_then_rigid_cube_into_bag as hand_reference,
)

soft0 = recorded_soft.soft0

# The isolated-hand rigid cube is shifted by -0.11 m in world X. With the W1
# base rotated +90 degrees about Z, that is +0.11 m in scene-local Y.
RIGID_CUBE_POSITION = wp.vec3(
    float(soft0.CUBE_POSITIONS[0][0]),
    float(soft0.CUBE_POSITIONS[0][1]) + 0.11,
    float(soft0.CUBE_POSITIONS[0][2]),
)

# Build the full-robot scene with the same integration and bag material used
# by the canonical isolated-hand trajectory.
soft0.SIM_SUBSTEPS = hand_reference.sequential_base.recorder.SIM_SUBSTEPS
soft0.VBD_ITERATIONS = hand_reference.sequential_base.recorder.VBD_ITERATIONS
soft0.BAG_RESOLUTION = hand_reference.BAG_RESOLUTION
soft0.BAG_PARTICLE_RADIUS = hand_reference.BAG_PARTICLE_RADIUS
soft0.BAG_DENSITY = hand_reference.BAG_DENSITY
soft0.BAG_TRI_KE = hand_reference.BAG_TRI_KE
soft0.BAG_TRI_KA = hand_reference.BAG_TRI_KA
soft0.BAG_TRI_KD = hand_reference.BAG_TRI_KD
soft0.BAG_EDGE_KE = hand_reference.BAG_EDGE_KE
soft0.BAG_EDGE_KD = hand_reference.BAG_EDGE_KD


class Example(recorded_soft.Example):
    """Track the canonical two-pick trajectory with the full Dexforce W1."""

    def __init__(self, viewer, args):
        rigid_root, self.rigid_grasp_joints, _ = hand_reference.rigid_demo.Example._load_grasp_keyframe(
            args.rigid_grasp_keyframe
        )
        rigid_root_position = wp.transform_get_translation(rigid_root)
        self.rigid_grasp_root_world = wp.transform(
            rigid_root_position + hand_reference.RIGID_CUBE_CENTRE - hand_reference.rigid_demo.recorder.CUBE_CENTRE,
            wp.transform_get_rotation(rigid_root),
        )
        self.contact_phase = None
        self.hand_shape_collision_enabled = True
        super().__init__(viewer, args)
        self.object_released = np.zeros(2, dtype=bool)
        self._apply_contact_phase("soft_wait")

    def _finger_pad_specs(self):
        """Disable the base example's auxiliary fingertip pads."""

        return ()

    def _add_additional_finger_pads(self, builder):
        """Keep the full-W1 scene limited to URDF collision meshes."""

    def _solver_vbd_options(self):
        """Combine the canonical particle and rigid-contact solver settings."""

        options = super()._solver_vbd_options()
        options.update(
            iterations=hand_reference.rigid_demo.recorder.VBD_ITERATIONS,
            rigid_avbd_contact_alpha=0.0,
            rigid_contact_history=True,
            rigid_contact_stick_motion_eps=5.0e-4,
            rigid_contact_stick_freeze_translation_eps=2.0e-4,
            rigid_contact_stick_freeze_angular_eps=2.0e-4,
            rigid_body_contact_buffer_size=hand_reference.rigid_demo.recorder.RIGID_BODY_CONTACT_BUFFER_SIZE,
            particle_vertex_contact_buffer_size=hand_reference.soft_demo.recorder.PARTICLE_VERTEX_CONTACT_BUFFER_SIZE,
            particle_edge_contact_buffer_size=hand_reference.soft_demo.recorder.PARTICLE_EDGE_CONTACT_BUFFER_SIZE,
            particle_collision_detection_interval=0,
            particle_topological_contact_filter_threshold=3,
            particle_rest_shape_contact_exclusion_radius=0.03,
        )
        return options

    def _solver_collision_options(self):
        """Enable contact matching required by rigid-contact history."""

        options = super()._solver_collision_options()
        options["contact_matching"] = "latest"
        return options

    def _add_additional_scene_objects(self, builder):
        """Add the second dynamic cube with the canonical rigid material."""

        cube_cfg = newton.ModelBuilder.ShapeConfig(
            density=hand_reference.RIGID_CUBE_DENSITY,
            ke=hand_reference.RIGID_GRASP_CONTACT[0],
            kd=hand_reference.RIGID_GRASP_CONTACT[1],
            mu=hand_reference.RIGID_GRASP_CONTACT[2],
            margin=hand_reference.RIGID_CUBE_MARGIN,
        )
        cube_cfg.configure_sdf(force_sdf=True)
        cube_cfg.has_particle_collision = True
        self.rigid_cube_body = builder.add_body(
            xform=wp.transform(self._world_vec(RIGID_CUBE_POSITION), self.base_rot),
            label="pick_recorded_rigid_cube",
        )
        self.rigid_cube_shape = builder.shape_count
        builder.add_shape_box(
            self.rigid_cube_body,
            hx=hand_reference.sequential_base.recorder.CUBE_HALF_EXTENTS[0],
            hy=hand_reference.sequential_base.recorder.CUBE_HALF_EXTENTS[1],
            hz=hand_reference.sequential_base.recorder.CUBE_HALF_EXTENTS[2],
            cfg=cube_cfg,
            xform=wp.transform(wp.vec3(), rigid_reference.CUBE_ROTATION),
            color=rigid_reference.CUBE_COLORS[0],
            label="pick_recorded_rigid_cube_shape",
        )

    def _segments(self):
        """Convert the canonical hand-root segments to right-arm TCP segments."""

        soft_grasp_joints = {f"RIGHT_{suffix}": float(self.grasp_hand_q[suffix]) for suffix in self.HAND_SUFFIXES}
        self.hand_trajectory_segments = hand_reference._build_recorded_trajectory(
            soft_grasp_joints,
            self.rigid_grasp_root_world,
            self.rigid_grasp_joints,
        )
        arm_segments = []
        for duration, root_a, root_b, _joints_a, _joints_b, phase in self.hand_trajectory_segments:
            right_a = self._root_to_tcp(root_a)
            right_b = self._root_to_tcp(root_b)
            if phase.startswith("soft_"):
                object_index = 0
            elif phase != "rigid_move":
                object_index = 1
            else:
                object_index = -1
            arm_segments.append(
                (
                    duration,
                    self.left_home,
                    self.left_home,
                    right_a,
                    right_b,
                    0.0,
                    0.0,
                    object_index,
                )
            )
        return tuple(arm_segments)

    def _build_joint_target_cache(self):
        """Initialize the robot at the canonical idle hand and first root pose."""

        first_root = self.hand_trajectory_segments[0][1]
        approach = self._root_to_tcp(first_root)
        self.left_obj.set_target_position(0, wp.transform_get_translation(self.left_home))
        self.left_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(self.left_home)))
        self.right_obj.set_target_position(0, wp.transform_get_translation(approach))
        self.right_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(approach)))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=recorded_soft.INITIAL_IK_ITERATIONS)
        wp.launch(
            soft0._lock_q,
            self.lock_indices.shape[0],
            [self.ik_q, self.lock_indices, self.lock_values],
            device=self.model.device,
        )

        initial_q = self.model.joint_q.numpy()
        initial_q[: self.ik_model.joint_coord_count] = self.ik_q.numpy()[0]
        initial_q[self.hand_indices.numpy()] = self.hand_start.numpy()
        self.model.joint_q.assign(initial_q)
        self.state_0.joint_q.assign(initial_q)
        self.state_1.joint_q.assign(initial_q)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

    def _set_shape_material(self, shapes, ke, kd, mu, margin=None):
        """Set effective contact material for selected model shapes."""

        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu[shapes] = mu
        shape_ke[shapes] = ke
        shape_kd[shapes] = kd
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        if margin is not None:
            shape_margin = self.model.shape_margin.numpy()
            shape_margin[shapes] = margin
            self.model.shape_margin.assign(shape_margin)

    def _set_shape_friction(self, shapes, friction):
        """Set friction for selected model shapes."""

        shape_mu = self.model.shape_material_mu.numpy()
        shape_mu[shapes] = friction
        self.model.shape_material_mu.assign(shape_mu)

    def _set_hand_shape_collision(self, enabled):
        """Gate right-hand rigid contact during the inter-object move."""

        if enabled == self.hand_shape_collision_enabled:
            return
        flags = self.model.shape_flags.numpy()
        shape_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
        if enabled:
            flags[self.right_hand_shapes] |= shape_flag
        else:
            flags[self.right_hand_shapes] &= ~shape_flag
        self.model.shape_flags.assign(flags)
        self.hand_shape_collision_enabled = enabled

    def _apply_contact_phase(self, phase):
        """Apply the canonical stage-specific contact configuration."""

        soft_active = phase in {"soft_prepare", "soft_grasp", "soft_carry"}
        rigid_active = phase in {"rigid_prepare", "rigid_grasp", "rigid_carry"}
        self._set_hand_shape_collision(phase != "rigid_move")

        if soft_active:
            self._set_hand_particle_collision(True)
            self._set_shape_material(
                self.right_hand_shapes,
                hand_reference.SOFT_GRASP_CONTACT[0],
                hand_reference.SOFT_GRASP_CONTACT[1],
                soft0.GRASP_FRICTION,
            )
            self.model.soft_contact_ke = hand_reference.SOFT_GRASP_CONTACT[0]
            self.model.soft_contact_kd = hand_reference.SOFT_GRASP_CONTACT[1]
            self.model.soft_contact_mu = hand_reference.SOFT_GRASP_CONTACT[2]
        elif phase == "soft_release":
            self._set_hand_particle_collision(True)
            self._set_shape_friction(self.right_hand_shapes, 0.0)
            self.model.soft_contact_ke = hand_reference.SOFT_FREE_CONTACT[0]
            self.model.soft_contact_kd = hand_reference.SOFT_FREE_CONTACT[1]
            self.model.soft_contact_mu = hand_reference.SOFT_FREE_CONTACT[2]
        elif rigid_active:
            self._set_hand_particle_collision(False)
            rigid_shapes = [*self.right_hand_shapes, self.rigid_cube_shape]
            self._set_shape_material(
                rigid_shapes,
                hand_reference.RIGID_GRASP_CONTACT[0],
                hand_reference.RIGID_GRASP_CONTACT[1],
                hand_reference.RIGID_GRASP_CONTACT[2],
                hand_reference.RIGID_CUBE_MARGIN,
            )
            self.model.soft_contact_ke = hand_reference.rigid_demo.SOFT_CONTACT_KE
            self.model.soft_contact_kd = hand_reference.rigid_demo.SOFT_CONTACT_KD
            self.model.soft_contact_mu = hand_reference.rigid_demo.SOFT_CONTACT_MU
        elif phase == "rigid_release":
            self._set_hand_particle_collision(False)
            rigid_shapes = [*self.right_hand_shapes, self.rigid_cube_shape]
            self._set_shape_material(
                rigid_shapes,
                hand_reference.RIGID_RELEASE_CONTACT[0],
                hand_reference.RIGID_RELEASE_CONTACT[1],
                hand_reference.RIGID_RELEASE_CONTACT[2],
                hand_reference.RIGID_CUBE_MARGIN,
            )
            self.model.soft_contact_ke = hand_reference.RIGID_RELEASE_CONTACT[0]
            self.model.soft_contact_kd = hand_reference.RIGID_RELEASE_CONTACT[1]
            self.model.soft_contact_mu = hand_reference.RIGID_RELEASE_CONTACT[2]
        else:
            self._set_hand_particle_collision(False)
            self._set_shape_friction(self.right_hand_shapes, 0.0)
            self.model.soft_contact_ke = hand_reference.SOFT_FREE_CONTACT[0]
            self.model.soft_contact_kd = hand_reference.SOFT_FREE_CONTACT[1]
            self.model.soft_contact_mu = hand_reference.SOFT_FREE_CONTACT[2]
        self.contact_phase = phase

    def _prepare_frame(self):
        """Solve arm IK and copy the canonical finger target for one frame."""

        script_time = (self.frame_index + 1) * self.frame_dt * self.args.trajectory_time_scale
        left, right, _grip, _script_object = self._sample(script_time)
        _root, finger_joints, phase = hand_reference._sample_recorded_trajectory(
            self.hand_trajectory_segments,
            script_time,
        )

        self.left_obj.set_target_position(0, wp.transform_get_translation(left))
        self.left_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(left)))
        self.right_obj.set_target_position(0, wp.transform_get_translation(right))
        self.right_rot.set_target_rotation(0, self._v4(wp.transform_get_rotation(right)))
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=soft0.IK_ITERATIONS)
        wp.launch(
            soft0._lock_q,
            self.lock_indices.shape[0],
            [self.ik_q, self.lock_indices, self.lock_values],
            device=self.model.device,
        )

        wp.copy(self.frame_q_start, self.state_0.joint_q)
        target_q = self.state_0.joint_q.numpy()
        target_q[: self.ik_model.joint_coord_count] = self.ik_q.numpy()[0]
        target_q[self.hand_indices.numpy()] = [
            np.radians(finger_joints[f"RIGHT_{suffix}"]) for suffix in self.HAND_SUFFIXES
        ]
        self.frame_q_end.assign(target_q)

        if phase != self.contact_phase:
            self._apply_contact_phase(phase)
        if phase == "soft_release":
            self.object_released[0] = True
        elif phase == "rigid_release":
            self.object_released[1] = True

    def test_final(self):
        """Verify finite objects, physical releases, placement, and no pads."""

        assert not any("physical_pad" in label for label in self.model.shape_label)
        body_flags = int(self.model.body_flags.numpy()[self.rigid_cube_body])
        assert not body_flags & int(newton.BodyFlags.KINEMATIC), (
            "The rigid cube must be dynamic when the solver is constructed"
        )
        assert np.all(np.isfinite(self.state_0.body_q.numpy()[self.rigid_cube_body])), (
            "Rigid cube state contains non-finite values"
        )

        script_frames = int(
            np.ceil(sum(segment[0] for segment in self.segments) / (self.frame_dt * self.args.trajectory_time_scale))
        )
        if self.frame_index < script_frames:
            return

        super().test_final()
        rigid_world_position = wp.vec3(*self.state_0.body_q.numpy()[self.rigid_cube_body, :3])
        rigid_position = self._scene_vec(rigid_world_position)
        bag_q = self.state_0.particle_q.numpy()[self.bag_particle_start : self.bag_particle_end]
        bag_scene_q = np.asarray([self._scene_vec(wp.vec3(*position)) for position in bag_q])
        bag_min_z = float(bag_scene_q[:, 2].min())
        half_extents = hand_reference.sequential_base.recorder.CUBE_HALF_EXTENTS
        rigid_inside = (
            abs(float(rigid_position[0]) - float(soft0.BAG_POS[0])) < 0.5 * soft0.BAG_WIDTH + half_extents[0]
            and abs(float(rigid_position[1]) - float(soft0.BAG_POS[1])) < 0.5 * soft0.BAG_DEPTH + half_extents[1]
            and bag_min_z - half_extents[2] < float(rigid_position[2]) < soft0.TABLE_TOP_Z + 0.08
        )
        assert rigid_inside, f"Rigid cube did not settle in the bag; position={tuple(rigid_position)}"

    @staticmethod
    def create_parser():
        """Create parser options for both canonical grasp keyframes."""

        parser = recorded_soft.Example.create_parser()
        parser.set_defaults(num_frames=1900, paused=False)
        parser.add_argument(
            "--rigid-grasp-keyframe",
            default=str(hand_reference.DEFAULT_RIGID_GRASP_KEYFRAME),
            help="Rigid-cube keyframe used by the canonical isolated-hand trajectory.",
        )
        return parser


def main():
    """Run the full-W1 canonical soft-then-rigid placement trajectory."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

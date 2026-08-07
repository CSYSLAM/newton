# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Use the full Dexforce W1 to place a soft cube, then a rigid cube, into a soft bag.

The right arm first reproduces the recorded physical soft-cube grasp. After
releasing that cube into the compliant box bag, it returns to the table and
uses the rigid-grasp fingertip pads, contact material, and timing to pick a
dynamic rigid cube and release it into the same bag. Object poses are never
attached to or copied from the hand.

Run from the repository root::

    uv run --extra examples newton/examples/vbd/example_vbd_mjvbd_v2_dexforce_recorded_soft_then_rigid_cube_into_bag.py --viewer gl
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd import example_vbd_mjvbd_v2_dexforce_grasp_rigid_into_bag_rigid as rigid_reference
from newton.examples.vbd import example_vbd_mjvbd_v2_dexforce_recorded_soft_cube_into_bag as recorded_soft
from newton.examples.vbd import example_vbd_mjvbd_v2_right_hand_soft_then_rigid_cube_into_bag as hand_reference

soft0 = recorded_soft.soft0
RIGID_CUBE_POSITION = wp.vec3(
    float(soft0.CUBE_POSITIONS[0][0]),
    float(soft0.CUBE_POSITIONS[0][1]) + 0.11,
    float(soft0.CUBE_POSITIONS[0][2]),
)
RIGID_GRASP_ROOT = hand_reference.RIGID_GRASP_ROOT
RIGID_GRASP_JOINTS_DEGREES = {
    suffix: hand_reference.RIGID_GRASP_JOINTS[f"RIGHT_{suffix}"] for suffix in recorded_soft.soft0.Example.HAND_SUFFIXES
}

SOFT_WAIT = 0
SOFT_GRASP = 1
SOFT_RELEASE = 2
RIGID_GRASP = 3
RIGID_RELEASE = 4


class Example(recorded_soft.Example):
    """Run the two physical pick-and-place operations with the full W1."""

    def __init__(self, viewer, args):
        self.contact_phase = None
        super().__init__(viewer, args)
        self.object_released = np.zeros(2, dtype=bool)
        self.previous_grip = 0.0
        # Both pad sets must be present when the NxN broad phase is built.
        # Runtime flags can then select the stage-specific set safely.
        self._apply_contact_phase(SOFT_WAIT)

    def _add_additional_finger_pads(self, builder):
        """Add the larger rigid-grasp pads alongside the soft-stage pads."""

        pad_cfg = newton.ModelBuilder.ShapeConfig(
            ke=rigid_reference.GRASP_CONTACT_KE,
            kd=rigid_reference.GRASP_CONTACT_KD,
            mu=rigid_reference.GRASP_FRICTION,
            is_visible=False,
        )
        self.rigid_finger_pad_shapes = []
        for body_name, half_extents, pad_xform in hand_reference.RIGID_FINGER_PADS:
            body = self._body_index(builder.body_label, body_name)
            self.rigid_finger_pad_shapes.append(
                builder.add_shape_box(
                    body,
                    hx=half_extents[0],
                    hy=half_extents[1],
                    hz=half_extents[2],
                    cfg=pad_cfg,
                    xform=pad_xform,
                    label=f"{body_name}_rigid_physical_pad",
                )
            )

    def _solver_vbd_options(self):
        """Match the isolated sequential example's VBD settings."""

        options = super()._solver_vbd_options()
        options.update(
            particle_vertex_contact_buffer_size=hand_reference.recorder.PARTICLE_VERTEX_CONTACT_BUFFER_SIZE,
            particle_edge_contact_buffer_size=hand_reference.recorder.PARTICLE_EDGE_CONTACT_BUFFER_SIZE,
            particle_collision_detection_interval=0,
            particle_topological_contact_filter_threshold=3,
            particle_rest_shape_contact_exclusion_radius=0.03,
        )
        return options

    def _build_scene(self):
        """Build both reference pad sets before the broad phase is allocated."""

        super()._build_scene()

    def _add_additional_scene_objects(self, builder):
        """Add the rigid cube while keeping it dynamic at solver construction."""

        cube_cfg = newton.ModelBuilder.ShapeConfig(
            density=rigid_reference.CUBE_DENSITY,
            ke=rigid_reference.GRASP_CONTACT_KE,
            kd=rigid_reference.GRASP_CONTACT_KD,
            mu=rigid_reference.GRASP_FRICTION,
            margin=rigid_reference.CUBE_MARGIN,
        )
        cube_cfg.configure_sdf(force_sdf=True)
        cube_cfg.has_particle_collision = True
        self.rigid_cube_body = builder.add_body(
            xform=wp.transform(self._world_vec(RIGID_CUBE_POSITION), self.base_rot),
            label="pick_rigid_cube",
        )
        self.rigid_cube_shape = builder.shape_count
        builder.add_shape_box(
            self.rigid_cube_body,
            hx=hand_reference.recorder.CUBE_HALF_EXTENTS[0],
            hy=hand_reference.recorder.CUBE_HALF_EXTENTS[1],
            hz=hand_reference.recorder.CUBE_HALF_EXTENTS[2],
            cfg=cube_cfg,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), rigid_reference.CUBE_ROTATION),
            color=rigid_reference.CUBE_COLORS[0],
            label="pick_rigid_cube_shape",
        )

    def _right_hand_q(self):
        """Cache separate recorded closures for the soft and rigid grasps."""

        hand_indices, hand_open, soft_hand_grasp = super()._right_hand_q()
        initial_q = self.model.joint_q.numpy()
        self.rigid_hand_open = wp.array(
            initial_q[hand_indices.numpy()],
            dtype=wp.float32,
            device=self.device,
        )
        rigid_hand_grasp = [np.radians(RIGID_GRASP_JOINTS_DEGREES[suffix]) for suffix in self.HAND_SUFFIXES]
        self.rigid_hand_grasp = wp.array(rigid_hand_grasp, dtype=wp.float32, device=self.device)
        return hand_indices, hand_open, soft_hand_grasp

    def _segments(self):
        """Build the recorded soft trajectory followed by the rigid trajectory."""

        soft_approach = self._root_to_tcp(self.approach_root_world)
        soft_root_position = wp.transform_get_translation(self.approach_root_world)
        soft_cube_position = self._world_vec(soft0.CUBE_POSITIONS[0])
        soft_root_cube_offset = soft_root_position - soft_cube_position
        bag_position = self._world_vec(soft0.BAG_POS)
        soft_bag_root = wp.transform(
            wp.vec3(
                float(bag_position[0]) + float(soft_root_cube_offset[0]),
                float(bag_position[1]) + float(soft_root_cube_offset[1]),
                float(bag_position[2]) + soft0.BAG_HEIGHT + 0.06 + float(soft_root_cube_offset[2]),
            ),
            wp.transform_get_rotation(self.approach_root_world),
        )
        soft_bag_hover = self._root_to_tcp(soft_bag_root)
        soft_lift = wp.transform(
            wp.transform_get_translation(soft_approach) + wp.vec3(0.0, 0.0, 0.07),
            wp.transform_get_rotation(soft_approach),
        )
        soft_retreat = wp.transform(
            wp.transform_get_translation(soft_bag_hover) + wp.vec3(0.0, 0.0, 0.10),
            wp.transform_get_rotation(soft_bag_hover),
        )

        rigid_approach = self._root_to_tcp(RIGID_GRASP_ROOT)
        rigid_pregrasp_root = wp.transform(
            wp.transform_get_translation(RIGID_GRASP_ROOT) + wp.vec3(0.0, 0.0, 0.18),
            wp.transform_get_rotation(RIGID_GRASP_ROOT),
        )
        rigid_pregrasp = self._root_to_tcp(rigid_pregrasp_root)
        rigid_cube_position = hand_reference.RIGID_CUBE_CENTRE
        rigid_root_cube_offset = wp.transform_get_translation(RIGID_GRASP_ROOT) - rigid_cube_position
        rigid_bag_root = wp.transform(
            wp.vec3(
                float(bag_position[0]) + float(rigid_root_cube_offset[0]),
                float(bag_position[1]) + float(rigid_root_cube_offset[1]),
                float(bag_position[2]) + soft0.BAG_HEIGHT + 0.06 + float(rigid_root_cube_offset[2]),
            ),
            wp.transform_get_rotation(RIGID_GRASP_ROOT),
        )
        rigid_bag_hover = self._root_to_tcp(rigid_bag_root)
        rigid_lift = wp.transform(
            wp.transform_get_translation(rigid_approach) + wp.vec3(0.0, 0.0, 0.10),
            wp.transform_get_rotation(rigid_approach),
        )
        rigid_transport = wp.transform(
            wp.transform_get_translation(rigid_bag_hover) + wp.vec3(0.0, 0.0, 0.05),
            wp.transform_get_rotation(rigid_bag_hover),
        )
        rigid_retreat = wp.transform(
            wp.transform_get_translation(rigid_bag_hover) + wp.vec3(0.0, 0.0, 0.10),
            wp.transform_get_rotation(rigid_bag_hover),
        )

        segments = [
            (0.50, self.left_home, self.left_home, soft_approach, soft_approach, 0.0, 0.0, 0),
            (1.80, self.left_home, self.left_home, soft_approach, soft_approach, 0.0, 1.0, 0),
            (0.60, self.left_home, self.left_home, soft_approach, soft_approach, 1.0, 1.0, 0),
            (1.20, self.left_home, self.left_home, soft_approach, soft_lift, 1.0, 1.0, 0),
            (7.00, self.left_home, self.left_home, soft_lift, soft_bag_hover, 1.0, 1.0, 0),
            (0.40, self.left_home, self.left_home, soft_bag_hover, soft_bag_hover, 1.0, 1.0, 0),
        ]
        self.soft_contact_start_time = segments[0][0]
        self.soft_release_start_time = sum(segment[0] for segment in segments)
        segments.extend(
            (
                (
                    soft0.SOFT_CUBE_RELEASE_OPEN_DURATION,
                    self.left_home,
                    self.left_home,
                    soft_bag_hover,
                    soft_bag_hover,
                    1.0,
                    0.0,
                    0,
                ),
                (
                    soft0.SOFT_CUBE_RELEASE_SETTLE_DURATION,
                    self.left_home,
                    self.left_home,
                    soft_bag_hover,
                    soft_bag_hover,
                    0.0,
                    0.0,
                    0,
                ),
                (1.00, self.left_home, self.left_home, soft_bag_hover, soft_retreat, 0.0, 0.0, -1),
            )
        )
        self.rigid_open_transition_start_time = sum(segment[0] for segment in segments)
        segments.append((1.20, self.left_home, self.left_home, soft_retreat, self.right_home, 0.0, 0.0, -1))
        self.rigid_grasp_start_time = sum(segment[0] for segment in segments)
        segments.extend(
            (
                (0.80, self.left_home, self.left_home, self.right_home, rigid_pregrasp, 0.0, 0.0, 1),
                (1.80, self.left_home, self.left_home, rigid_pregrasp, rigid_approach, 0.0, 1.0, 1),
                (1.00, self.left_home, self.left_home, rigid_approach, rigid_approach, 1.0, 1.0, 1),
                (2.00, self.left_home, self.left_home, rigid_approach, rigid_lift, 1.0, 1.0, 1),
                (0.60, self.left_home, self.left_home, rigid_lift, rigid_lift, 1.0, 1.0, 1),
                (5.00, self.left_home, self.left_home, rigid_lift, rigid_transport, 1.0, 1.0, 1),
                (1.50, self.left_home, self.left_home, rigid_transport, rigid_bag_hover, 1.0, 1.0, 1),
                (0.40, self.left_home, self.left_home, rigid_bag_hover, rigid_bag_hover, 1.0, 1.0, 1),
            )
        )
        self.rigid_release_start_time = sum(segment[0] for segment in segments)
        segments.extend(
            (
                (0.80, self.left_home, self.left_home, rigid_bag_hover, rigid_bag_hover, 1.0, 0.0, 1),
                (0.20, self.left_home, self.left_home, rigid_bag_hover, rigid_bag_hover, 0.0, 0.0, 1),
                (1.00, self.left_home, self.left_home, rigid_bag_hover, rigid_retreat, 0.0, 0.0, 1),
                (1.20, self.left_home, self.left_home, rigid_retreat, self.right_home, 0.0, 0.0, -1),
                (1.20, self.left_home, self.left_home, self.right_home, self.right_home, 0.0, 0.0, -1),
            )
        )
        return tuple(segments)

    def _phase_at(self, script_time):
        if script_time < self.soft_contact_start_time:
            return SOFT_WAIT
        if script_time < self.soft_release_start_time:
            return SOFT_GRASP
        if script_time < self.rigid_grasp_start_time:
            return SOFT_RELEASE
        if script_time < self.rigid_release_start_time:
            return RIGID_GRASP
        return RIGID_RELEASE

    def _build_joint_target_cache(self):
        """Initialize the arm for per-frame IK tracking of the hand reference."""

        approach = self._root_to_tcp(self.approach_root_world)
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
        initial_q[self.hand_indices.numpy()] = self.hand_open.numpy()
        self.model.joint_q.assign(initial_q)
        self.state_0.joint_q.assign(initial_q)
        self.state_1.joint_q.assign(initial_q)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

    def _finger_target(self, script_time, grip):
        """Sample the reference example's exact finger trajectory."""

        soft_hand_open = self.hand_open.numpy()
        soft_hand_grasp = self.hand_grasp.numpy()
        rigid_hand_open = self.rigid_hand_open.numpy()
        rigid_hand_grasp = self.rigid_hand_grasp.numpy()
        if script_time >= self.rigid_grasp_start_time:
            hand_open = rigid_hand_open
            hand_grasp = rigid_hand_grasp
        elif script_time >= self.rigid_open_transition_start_time:
            transition_duration = self.rigid_grasp_start_time - self.rigid_open_transition_start_time
            alpha = float(
                np.clip(
                    (script_time - self.rigid_open_transition_start_time) / transition_duration,
                    0.0,
                    1.0,
                )
            )
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            hand_open = soft_hand_open * (1.0 - alpha) + rigid_hand_open * alpha
            hand_grasp = rigid_hand_grasp
        else:
            hand_open = soft_hand_open
            hand_grasp = soft_hand_grasp
        return hand_open * (1.0 - grip) + hand_grasp * grip

    def _set_shape_material(self, shapes, ke, kd, mu):
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu[shapes] = mu
        shape_ke[shapes] = ke
        shape_kd[shapes] = kd
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)

    def _set_shape_friction(self, shapes, friction):
        shape_mu = self.model.shape_material_mu.numpy()
        shape_mu[shapes] = friction
        self.model.shape_material_mu.assign(shape_mu)

    def _set_pad_shape_collision(self, shapes, enabled):
        flags = self.model.shape_flags.numpy()
        shape_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
        particle_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        if enabled:
            flags[shapes] |= shape_flag
        else:
            flags[shapes] &= ~shape_flag
        flags[shapes] &= ~particle_flag
        self.model.shape_flags.assign(flags)

    def _apply_contact_phase(self, phase):
        soft_hand_shapes = self.right_hand_shapes
        rigid_hand_shapes = [*self.right_hand_shapes, *self.rigid_finger_pad_shapes]
        if phase == SOFT_GRASP:
            self._set_pad_shape_collision(self.finger_pad_shapes, True)
            self._set_pad_shape_collision(self.rigid_finger_pad_shapes, False)
            self._set_hand_particle_collision(True)
            self._set_shape_material(
                soft_hand_shapes,
                soft0.GRASP_CONTACT_KE,
                soft0.GRASP_CONTACT_KD,
                soft0.GRASP_FRICTION,
            )
        elif phase == SOFT_RELEASE:
            self._set_pad_shape_collision(self.finger_pad_shapes, True)
            self._set_pad_shape_collision(self.rigid_finger_pad_shapes, False)
            self._set_hand_particle_collision(True)
            self._set_shape_material(
                soft_hand_shapes,
                soft0.RELEASE_CONTACT_KE,
                soft0.RELEASE_CONTACT_KD,
                soft0.RELEASE_FRICTION,
            )
            self.model.soft_contact_ke = soft0.RELEASE_CONTACT_KE
            self.model.soft_contact_kd = soft0.RELEASE_CONTACT_KD
            self.model.soft_contact_mu = soft0.RELEASE_FRICTION
        elif phase == RIGID_GRASP:
            self._set_hand_particle_collision(False)
            self._set_pad_shape_collision(self.finger_pad_shapes, False)
            self._set_pad_shape_collision(self.rigid_finger_pad_shapes, True)
            rigid_shapes = [*rigid_hand_shapes, self.rigid_cube_shape]
            self._set_shape_material(
                rigid_shapes,
                rigid_reference.GRASP_CONTACT_KE,
                rigid_reference.GRASP_CONTACT_KD,
                rigid_reference.GRASP_FRICTION,
            )
        elif phase == RIGID_RELEASE:
            self._set_hand_particle_collision(False)
            self._set_pad_shape_collision(self.finger_pad_shapes, False)
            self._set_pad_shape_collision(self.rigid_finger_pad_shapes, True)
            rigid_shapes = [*rigid_hand_shapes, self.rigid_cube_shape]
            self._set_shape_material(
                rigid_shapes,
                rigid_reference.RELEASE_CONTACT_KE,
                rigid_reference.RELEASE_CONTACT_KD,
                rigid_reference.RELEASE_FRICTION,
            )
        else:
            self._set_pad_shape_collision(self.finger_pad_shapes, True)
            self._set_pad_shape_collision(self.rigid_finger_pad_shapes, False)
            self._set_hand_particle_collision(False)
        self.contact_phase = phase

    def _prepare_frame(self):
        script_time = (self.frame_index + 1) * self.frame_dt * self.args.trajectory_time_scale
        left, right, grip, script_object = self._sample(script_time)
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
        target_q[self.hand_indices.numpy()] = self._finger_target(script_time, grip)
        self.frame_q_end.assign(target_q)

        phase = self._phase_at(script_time)
        if phase != self.contact_phase:
            self._apply_contact_phase(phase)
        if self.previous_grip > 1.0e-4 and grip <= 1.0e-4 and script_object >= 0:
            self.object_released[script_object] = True
        self.previous_grip = grip

    def test_final(self):
        """Verify that both dynamic objects are finite, released, and inside the bag."""

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
        rigid_inside = (
            abs(float(rigid_position[0]) - float(soft0.BAG_POS[0]))
            < 0.5 * soft0.BAG_WIDTH + rigid_reference.CUBE_HALF_EXTENTS[0]
            and abs(float(rigid_position[1]) - float(soft0.BAG_POS[1]))
            < 0.5 * soft0.BAG_DEPTH + rigid_reference.CUBE_HALF_EXTENTS[1]
            and bag_min_z - rigid_reference.CUBE_HALF_EXTENTS[2] < float(rigid_position[2]) < soft0.TABLE_TOP_Z + 0.08
        )
        assert rigid_inside, f"Rigid cube did not settle in the bag; position={tuple(rigid_position)}"

    @staticmethod
    def create_parser():
        """Create parser options for the full-W1 sequential placement demo."""

        parser = recorded_soft.Example.create_parser()
        parser.set_defaults(num_frames=2100, paused=False)
        return parser


def main():
    """Run the full-W1 sequential soft-and-rigid placement demo."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

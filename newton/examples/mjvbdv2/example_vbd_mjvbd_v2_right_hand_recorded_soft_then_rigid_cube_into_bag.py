# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Place a soft cube, then a rigid cube, into one bag with a recorded hand.

The floating W1 right hand starts each pick from the same idle finger pose,
transitions to the matching pre-grasp pose, and closes to the corresponding
recorded keyframe. Both objects are transported and released through physical
contact. Only collision meshes imported from the right-hand URDF touch the
objects; this example creates no auxiliary fingertip pads.

Run from the repository root::

    uv run --extra examples -m newton.examples vbd_mjvbd_v2_right_hand_recorded_soft_then_rigid_cube_into_bag --viewer gl
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mjvbdv2 import example_vbd_mjvbd_v2_right_hand_recorded_rigid_cube_into_bag as rigid_demo
from newton.examples.mjvbdv2 import example_vbd_mjvbd_v2_right_hand_recorded_soft_cube_into_bag as soft_demo
from newton.examples.mjvbdv2 import example_vbd_mjvbd_v2_right_hand_soft_then_rigid_cube_into_bag as sequential_base
from newton.solvers import SolverMJVBDV2

DEFAULT_RIGID_GRASP_KEYFRAME = (
    Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2" / "vbd_w1_right_hand_rigid_cube_last_keyframe.json"
)

HAND_JOINTS = sequential_base.recorder.HAND_JOINTS
IDLE_JOINTS = dict.fromkeys(HAND_JOINTS, 0.0)
IDLE_JOINTS["RIGHT_HAND_THUMB2"] = 90.0
OPEN_JOINTS = dict.fromkeys(HAND_JOINTS, 0.0)

SOFT_GRASP_CONTACT = (
    soft_demo.full_reference.GRASP_CONTACT_KE,
    soft_demo.full_reference.GRASP_CONTACT_KD,
    soft_demo.full_reference.GRASP_SOFT_CONTACT_MU,
)
SOFT_FREE_CONTACT = (
    soft_demo.full_reference.SOFT_CONTACT_KE,
    soft_demo.full_reference.SOFT_CONTACT_KD,
    soft_demo.full_reference.SOFT_CONTACT_MU,
)
RIGID_GRASP_CONTACT = (
    rigid_demo.recorder.CONTACT_KE,
    rigid_demo.recorder.CONTACT_KD,
    rigid_demo.recorder.CONTACT_MU,
)
RIGID_RELEASE_CONTACT = (
    rigid_demo.RELEASE_CONTACT_KE,
    rigid_demo.RELEASE_CONTACT_KD,
    rigid_demo.RELEASE_FRICTION,
)

SOFT_CUBE_CENTRE = soft_demo.recorder.CUBE_CENTRE
SOFT_CUBE_DIMS = soft_demo.recorder.CUBE_DIMS
SOFT_CUBE_DENSITY = soft_demo.recorder.CUBE_DENSITY
SOFT_CUBE_K_MU = soft_demo.recorder.CUBE_K_MU
SOFT_CUBE_K_LAMBDA = soft_demo.recorder.CUBE_K_LAMBDA
SOFT_CUBE_K_DAMP = soft_demo.recorder.CUBE_K_DAMP
SOFT_CUBE_PARTICLE_RADIUS = soft_demo.recorder.CUBE_PARTICLE_RADIUS

RIGID_CUBE_CENTRE = sequential_base.RIGID_CUBE_CENTRE
RIGID_CUBE_DENSITY = rigid_demo.recorder.CUBE_DENSITY
RIGID_CUBE_MARGIN = rigid_demo.recorder.CONTACT_MARGIN

BAG_RESOLUTION = soft_demo.recorder.BAG_RESOLUTION
BAG_PARTICLE_RADIUS = soft_demo.recorder.BAG_PARTICLE_RADIUS
BAG_DENSITY = soft_demo.recorder.BAG_DENSITY
BAG_TRI_KE = soft_demo.recorder.BAG_TRI_KE
BAG_TRI_KA = soft_demo.recorder.BAG_TRI_KA
BAG_TRI_KD = soft_demo.recorder.BAG_TRI_KD
BAG_EDGE_KE = soft_demo.recorder.BAG_EDGE_KE
BAG_EDGE_KD = soft_demo.recorder.BAG_EDGE_KD

# Use the rigid recorder's finer integration for the mixed scene. The parent
# recorder reads these module values while initializing and stepping.
sequential_base.recorder.SIM_SUBSTEPS = max(
    sequential_base.recorder.SIM_SUBSTEPS,
    rigid_demo.recorder.SIM_SUBSTEPS,
)
sequential_base.recorder.VBD_ITERATIONS = max(
    sequential_base.recorder.VBD_ITERATIONS,
    rigid_demo.recorder.VBD_ITERATIONS,
)


class Example(sequential_base.Example):
    """Run the recorded soft-cube pick followed by the recorded rigid pick."""

    def __init__(self, viewer, args):
        rigid_root, self.rigid_grasp_joints, _ = rigid_demo.Example._load_grasp_keyframe(args.rigid_grasp_keyframe)
        rigid_root_position = wp.transform_get_translation(rigid_root)
        self.rigid_grasp_root = wp.transform(
            rigid_root_position + RIGID_CUBE_CENTRE - rigid_demo.recorder.CUBE_CENTRE,
            wp.transform_get_rotation(rigid_root),
        )
        super().__init__(viewer, args)
        self.hand_rigid_contact_enabled = True

        # The parent prototype starts directly in the soft pre-grasp pose.
        # Reset both kinematic states so this demo visibly starts from idle.
        self._set_hand_target(soft_demo.APPROACH_ROOT, IDLE_JOINTS)
        self._set_initial_hand_pose()
        self.solver.reset(self.state_0, flags=0)

    def _create_solver(self):
        """Create one solver for soft-soft, rigid-soft, and rigid-rigid contact."""

        contact_radius = max(BAG_PARTICLE_RADIUS, SOFT_CUBE_PARTICLE_RADIUS)
        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=self.hand_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": rigid_demo.recorder.VBD_ITERATIONS,
                "rigid_avbd_contact_alpha": 0.0,
                "rigid_contact_history": True,
                "rigid_contact_stick_motion_eps": 5.0e-4,
                "rigid_contact_stick_freeze_translation_eps": 2.0e-4,
                "rigid_contact_stick_freeze_angular_eps": 2.0e-4,
                "rigid_body_contact_buffer_size": rigid_demo.recorder.RIGID_BODY_CONTACT_BUFFER_SIZE,
                "rigid_body_particle_contact_buffer_size": soft_demo.full_reference.RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": True,
                "particle_self_contact_radius": contact_radius,
                "particle_self_contact_margin": 2.0 * contact_radius,
                "particle_vertex_contact_buffer_size": soft_demo.recorder.PARTICLE_VERTEX_CONTACT_BUFFER_SIZE,
                "particle_edge_contact_buffer_size": soft_demo.recorder.PARTICLE_EDGE_CONTACT_BUFFER_SIZE,
                "particle_collision_detection_interval": 0,
                "particle_topological_contact_filter_threshold": 3,
                "particle_rest_shape_contact_exclusion_radius": 0.03,
            },
            collision_options={
                "broad_phase": "nxn",
                "contact_matching": "latest",
                "soft_contact_margin": soft_demo.full_reference.SOFT_CONTACT_MARGIN,
                "enable_rigid_soft_full_surface_contact": True,
            },
        )

    def _build_scene(self):
        """Build one URDF hand, two cubes, and a pinned soft bag without pads."""

        recorder = sequential_base.recorder
        if not recorder.RIGHT_HAND_URDF.is_file():
            raise FileNotFoundError(f"Right-hand URDF not found: {recorder.RIGHT_HAND_URDF}")

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg.ke = SOFT_GRASP_CONTACT[0]
        builder.default_shape_cfg.kd = SOFT_GRASP_CONTACT[1]
        builder.default_shape_cfg.mu = soft_demo.full_reference.GRASP_FRICTION
        builder.default_shape_cfg.configure_sdf(force_sdf=True)
        SolverMJVBDV2.register_custom_attributes(builder)

        articulation_start = builder.articulation_count
        builder.add_urdf(
            str(recorder.RIGHT_HAND_URDF),
            xform=recorder.HAND_HOME,
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        self.hand_articulations = tuple(range(articulation_start, builder.articulation_count))
        self.hand_soft_shape_end = builder.shape_count
        self.hand_shape_end = builder.shape_count
        for body in range(builder.body_count):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)

        table_cfg = newton.ModelBuilder.ShapeConfig(ke=3.0e5, kd=1.0e-4, mu=0.9, is_visible=True)
        table_cfg.configure_sdf(force_sdf=True)
        builder.add_shape_box(
            -1,
            xform=wp.transform(recorder.TABLE_POS, recorder.TABLE_ROTATION),
            hx=recorder.TABLE_HALF_EXTENTS[0],
            hy=recorder.TABLE_HALF_EXTENTS[1],
            hz=recorder.TABLE_HALF_EXTENTS[2],
            cfg=table_cfg,
            color=(0.35, 0.42, 0.48),
            label="recorded_two_pick_table",
        )
        builder.add_ground_plane(height=-0.00377202, label="recorded_two_pick_ground")

        bag_vertices, bag_indices = recorder._generate_box_bag(
            0.5 * recorder.BAG_WIDTH,
            0.5 * recorder.BAG_DEPTH,
            recorder.BAG_HEIGHT,
            BAG_RESOLUTION,
        )
        self.bag_particle_start = builder.particle_count
        builder.add_cloth_mesh(
            pos=recorder.BAG_POS,
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=bag_vertices.tolist(),
            indices=bag_indices,
            density=BAG_DENSITY,
            tri_ke=BAG_TRI_KE,
            tri_ka=BAG_TRI_KA,
            tri_kd=BAG_TRI_KD,
            edge_ke=BAG_EDGE_KE,
            edge_kd=BAG_EDGE_KD,
            particle_radius=BAG_PARTICLE_RADIUS,
            label="recorded_two_pick_soft_bag",
        )
        self.bag_particle_end = builder.particle_count
        bag_top = np.flatnonzero(np.abs(bag_vertices[:, 2] - recorder.BAG_HEIGHT) < 1.0e-5)
        self.bag_top_indices = bag_top.astype(np.int32) + self.bag_particle_start

        soft_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi)
        soft_origin = SOFT_CUBE_CENTRE - wp.quat_rotate(soft_rotation, wp.vec3(*recorder.CUBE_HALF_EXTENTS))
        self.soft_cube_particle_start = builder.particle_count
        builder.add_soft_grid(
            pos=soft_origin,
            rot=soft_rotation,
            vel=wp.vec3(),
            dim_x=SOFT_CUBE_DIMS[0],
            dim_y=SOFT_CUBE_DIMS[1],
            dim_z=SOFT_CUBE_DIMS[2],
            cell_x=2.0 * recorder.CUBE_HALF_EXTENTS[0] / SOFT_CUBE_DIMS[0],
            cell_y=2.0 * recorder.CUBE_HALF_EXTENTS[1] / SOFT_CUBE_DIMS[1],
            cell_z=2.0 * recorder.CUBE_HALF_EXTENTS[2] / SOFT_CUBE_DIMS[2],
            density=SOFT_CUBE_DENSITY,
            k_mu=SOFT_CUBE_K_MU,
            k_lambda=SOFT_CUBE_K_LAMBDA,
            k_damp=SOFT_CUBE_K_DAMP,
            particle_radius=SOFT_CUBE_PARTICLE_RADIUS,
            label="first_recorded_soft_cube",
        )
        self.soft_cube_particle_end = builder.particle_count

        rigid_cfg = newton.ModelBuilder.ShapeConfig(
            density=RIGID_CUBE_DENSITY,
            ke=RIGID_GRASP_CONTACT[0],
            kd=RIGID_GRASP_CONTACT[1],
            mu=RIGID_GRASP_CONTACT[2],
            margin=RIGID_CUBE_MARGIN,
        )
        rigid_cfg.configure_sdf(force_sdf=True)
        rigid_cfg.has_particle_collision = True
        self.rigid_cube_body = builder.add_body(
            xform=wp.transform(RIGID_CUBE_CENTRE, wp.quat_identity()),
            label="second_recorded_rigid_cube",
        )
        self.rigid_cube_shape = builder.shape_count
        builder.add_shape_box(
            self.rigid_cube_body,
            hx=recorder.CUBE_HALF_EXTENTS[0],
            hy=recorder.CUBE_HALF_EXTENTS[1],
            hz=recorder.CUBE_HALF_EXTENTS[2],
            cfg=rigid_cfg,
            color=(0.90, 0.32, 0.18),
            label="second_recorded_rigid_cube_shape",
        )

        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape in range(builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes | collide_particles

        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = SOFT_FREE_CONTACT[0]
        self.model.soft_contact_kd = SOFT_FREE_CONTACT[1]
        self.model.soft_contact_mu = SOFT_FREE_CONTACT[2]

    def _set_hand_rigid_contact(self, enabled: bool):
        """Gate hand-to-rigid contact while moving between the two objects."""

        if enabled == self.hand_rigid_contact_enabled:
            return
        flags = self.model.shape_flags.numpy()
        rigid_flag = int(newton.ShapeFlags.COLLIDE_SHAPES)
        if enabled:
            flags[: self.hand_shape_end] |= rigid_flag
        else:
            flags[: self.hand_shape_end] &= ~rigid_flag
        self.model.shape_flags.assign(flags)
        self.hand_rigid_contact_enabled = enabled

    def _apply_rigid_grasp_material(self):
        """Restore the standalone rigid recorder's mesh-contact material."""

        if self.rigid_grasp_material_applied:
            return
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_margin = self.model.shape_margin.numpy()
        shape_mu[: self.hand_shape_end] = RIGID_GRASP_CONTACT[2]
        shape_mu[self.rigid_cube_shape] = RIGID_GRASP_CONTACT[2]
        shape_ke[: self.hand_shape_end] = RIGID_GRASP_CONTACT[0]
        shape_ke[self.rigid_cube_shape] = RIGID_GRASP_CONTACT[0]
        shape_kd[: self.hand_shape_end] = RIGID_GRASP_CONTACT[1]
        shape_kd[self.rigid_cube_shape] = RIGID_GRASP_CONTACT[1]
        shape_margin[: self.hand_shape_end] = RIGID_CUBE_MARGIN
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        self.model.shape_margin.assign(shape_margin)
        self.model.soft_contact_ke = rigid_demo.SOFT_CONTACT_KE
        self.model.soft_contact_kd = rigid_demo.SOFT_CONTACT_KD
        self.model.soft_contact_mu = rigid_demo.SOFT_CONTACT_MU
        self.rigid_grasp_material_applied = True

    def _apply_rigid_release_material(self):
        """Remove mesh and rigid-soft friction for the second release."""

        if self.rigid_release_applied:
            return
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu[: self.hand_shape_end] = RIGID_RELEASE_CONTACT[2]
        shape_mu[self.rigid_cube_shape] = RIGID_RELEASE_CONTACT[2]
        shape_ke[: self.hand_shape_end] = RIGID_RELEASE_CONTACT[0]
        shape_ke[self.rigid_cube_shape] = RIGID_RELEASE_CONTACT[0]
        shape_kd[: self.hand_shape_end] = RIGID_RELEASE_CONTACT[1]
        shape_kd[self.rigid_cube_shape] = RIGID_RELEASE_CONTACT[1]
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        self.model.soft_contact_ke = RIGID_RELEASE_CONTACT[0]
        self.model.soft_contact_kd = RIGID_RELEASE_CONTACT[1]
        self.model.soft_contact_mu = RIGID_RELEASE_CONTACT[2]
        self.rigid_release_applied = True

    def _build_segments(self):
        """Build two idle-to-pre-grasp-to-recorded-grasp trajectories."""

        recorder = sequential_base.recorder
        soft_approach = soft_demo.APPROACH_ROOT
        rigid_approach = self.rigid_grasp_root
        soft_root_cube_offset = wp.transform_get_translation(soft_approach) - SOFT_CUBE_CENTRE
        rigid_root_cube_offset = wp.transform_get_translation(rigid_approach) - RIGID_CUBE_CENTRE
        release_height = float(recorder.BAG_POS[2]) + recorder.BAG_HEIGHT + 0.06

        soft_bag_hover = wp.transform(
            wp.vec3(
                float(recorder.BAG_POS[0]) + float(soft_root_cube_offset[0]),
                float(recorder.BAG_POS[1]) + float(soft_root_cube_offset[1]),
                release_height + float(soft_root_cube_offset[2]),
            ),
            wp.transform_get_rotation(soft_approach),
        )
        rigid_bag_hover = wp.transform(
            wp.vec3(
                float(recorder.BAG_POS[0]) + float(rigid_root_cube_offset[0]),
                float(recorder.BAG_POS[1]) + float(rigid_root_cube_offset[1]),
                release_height + float(rigid_root_cube_offset[2]),
            ),
            wp.transform_get_rotation(rigid_approach),
        )
        soft_lift = wp.transform(
            wp.transform_get_translation(soft_approach) + wp.vec3(0.0, 0.0, 0.07),
            wp.transform_get_rotation(soft_approach),
        )
        rigid_lift = wp.transform(
            wp.transform_get_translation(rigid_approach) + wp.vec3(0.0, 0.0, 0.10),
            wp.transform_get_rotation(rigid_approach),
        )
        rigid_transport = wp.transform(
            wp.transform_get_translation(rigid_bag_hover) + wp.vec3(0.0, 0.0, 0.05),
            wp.transform_get_rotation(rigid_bag_hover),
        )
        soft_retreat = wp.transform(
            wp.transform_get_translation(soft_bag_hover) + wp.vec3(0.0, 0.0, 0.10),
            wp.transform_get_rotation(soft_bag_hover),
        )
        rigid_retreat = wp.transform(
            wp.transform_get_translation(rigid_bag_hover) + wp.vec3(0.0, 0.0, 0.12),
            wp.transform_get_rotation(rigid_bag_hover),
        )

        soft_grasp = self.grasp_joints
        rigid_grasp = self.rigid_grasp_joints
        rigid_pregrasp = rigid_demo.recorder.INITIAL_HAND_JOINTS
        return (
            (0.50, soft_approach, soft_approach, IDLE_JOINTS, IDLE_JOINTS, "soft_wait"),
            (1.50, soft_approach, soft_approach, IDLE_JOINTS, soft_demo.APPROACH_JOINTS, "soft_prepare"),
            (
                0.50,
                soft_approach,
                soft_approach,
                soft_demo.APPROACH_JOINTS,
                soft_demo.APPROACH_JOINTS,
                "soft_prepare",
            ),
            (1.80, soft_approach, soft_approach, soft_demo.APPROACH_JOINTS, soft_grasp, "soft_grasp"),
            (0.60, soft_approach, soft_approach, soft_grasp, soft_grasp, "soft_carry"),
            (1.20, soft_approach, soft_lift, soft_grasp, soft_grasp, "soft_carry"),
            (7.00, soft_lift, soft_bag_hover, soft_grasp, soft_grasp, "soft_carry"),
            (0.40, soft_bag_hover, soft_bag_hover, soft_grasp, soft_grasp, "soft_carry"),
            (0.25, soft_bag_hover, soft_bag_hover, soft_grasp, OPEN_JOINTS, "soft_release"),
            (0.90, soft_bag_hover, soft_bag_hover, OPEN_JOINTS, OPEN_JOINTS, "soft_release"),
            (1.00, soft_bag_hover, soft_retreat, OPEN_JOINTS, OPEN_JOINTS, "soft_release"),
            (1.50, soft_retreat, rigid_approach, OPEN_JOINTS, IDLE_JOINTS, "rigid_move"),
            (0.50, rigid_approach, rigid_approach, IDLE_JOINTS, IDLE_JOINTS, "rigid_prepare"),
            (1.50, rigid_approach, rigid_approach, IDLE_JOINTS, rigid_pregrasp, "rigid_prepare"),
            (0.50, rigid_approach, rigid_approach, rigid_pregrasp, rigid_pregrasp, "rigid_prepare"),
            (0.45, rigid_approach, rigid_approach, rigid_pregrasp, rigid_grasp, "rigid_grasp"),
            (0.30, rigid_approach, rigid_approach, rigid_grasp, rigid_grasp, "rigid_carry"),
            (0.75, rigid_approach, rigid_lift, rigid_grasp, rigid_grasp, "rigid_carry"),
            (5.00, rigid_lift, rigid_transport, rigid_grasp, rigid_grasp, "rigid_carry"),
            (1.20, rigid_transport, rigid_bag_hover, rigid_grasp, rigid_grasp, "rigid_carry"),
            (0.50, rigid_bag_hover, rigid_bag_hover, rigid_grasp, rigid_grasp, "rigid_carry"),
            (0.80, rigid_bag_hover, rigid_bag_hover, rigid_grasp, OPEN_JOINTS, "rigid_release"),
            (1.50, rigid_bag_hover, rigid_bag_hover, OPEN_JOINTS, OPEN_JOINTS, "rigid_release"),
            (1.00, rigid_bag_hover, rigid_retreat, OPEN_JOINTS, OPEN_JOINTS, "rigid_release"),
        )

    def _sample(self, time_s: float):
        """Sample the canonical shared hand-root and finger trajectory."""

        return _sample_recorded_trajectory(self.segments, time_s)

    def step(self):
        """Advance the two recorded physical picks by one frame."""

        root, joints, phase = self._sample(self.sim_time)
        self._set_hand_target(root, joints)
        soft_contact = phase in {"soft_prepare", "soft_grasp", "soft_carry", "soft_release"}
        self._set_hand_soft_contact(soft_contact)
        self._set_hand_rigid_contact(phase != "rigid_move")
        if phase in {"soft_prepare", "soft_grasp", "soft_carry"}:
            self._set_hand_shape_friction(soft_demo.full_reference.GRASP_FRICTION)
        if phase in {"rigid_prepare", "rigid_grasp", "rigid_carry", "rigid_release"}:
            self._apply_rigid_grasp_material()
        if phase == "soft_release":
            self._apply_soft_release_material()
        if phase == "rigid_release":
            self._apply_rigid_release_material()
        self.step_once()

    def test_final(self):
        """Verify finite mixed-body state and the absence of added fingertip pads."""

        super().test_final()
        assert not any("physical_pad" in label for label in self.model.shape_label)

    @staticmethod
    def create_parser():
        """Create parser options for both recorded grasp keyframes."""

        parser = sequential_base.Example.create_parser()
        parser.set_defaults(num_frames=1900, paused=False)
        parser.add_argument(
            "--rigid-grasp-keyframe",
            default=str(DEFAULT_RIGID_GRASP_KEYFRAME),
            help="Rigid-cube keyframe generated by the standalone right-hand recorder.",
        )
        return parser


def _build_recorded_trajectory(
    soft_grasp_joints: dict[str, float],
    rigid_grasp_root: wp.transform,
    rigid_grasp_joints: dict[str, float],
):
    """Build the canonical hand-root and finger-joint trajectory."""

    inputs = SimpleNamespace(
        grasp_joints=soft_grasp_joints,
        rigid_grasp_root=rigid_grasp_root,
        rigid_grasp_joints=rigid_grasp_joints,
    )
    return Example._build_segments(inputs)


def _sample_recorded_trajectory(segments, time_s: float):
    """Sample the canonical hand-root and finger-joint trajectory."""

    for duration, root_a, root_b, joints_a, joints_b, phase in segments:
        if time_s <= duration:
            alpha = float(np.clip(time_s / duration, 0.0, 1.0))
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            root = Example._lerp_transform(root_a, root_b, alpha)
            joints = {name: joints_a[name] * (1.0 - alpha) + joints_b[name] * alpha for name in HAND_JOINTS}
            return root, joints, phase
        time_s -= duration
    _, _, root, _, joints, phase = segments[-1]
    return root, joints, phase


def main():
    """Run the mesh-only sequential soft-then-rigid placement example."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

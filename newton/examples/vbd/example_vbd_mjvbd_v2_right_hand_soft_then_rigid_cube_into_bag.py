# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pick a soft cube, then a rigid cube, into one soft box bag with a W1 hand.

The floating right hand starts at the recorded approach pose. It physically
pinches the soft cube with the recorded five-finger closure, releases it into
the bag, returns to the table, and repeats the motion for a dynamic rigid cube.
No object pose is attached to or copied from the hand.

Run from the repository root::

    uv run --extra examples -m newton.examples vbd_mjvbd_v2_right_hand_soft_then_rigid_cube_into_bag --viewer gl
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.vbd import example_vbd_mjvbd_v2_dexforce_grasp_rigid_into_bag_rigid as rigid_reference
from newton.examples.vbd import example_vbd_mjvbd_v2_dexforce_recorded_soft_cube_into_bag as soft_reference
from newton.examples.vbd import example_vbd_mjvbd_v2_right_hand_soft_cube_recorder as recorder
from newton.solvers import SolverMJVBDV2

DEFAULT_GRASP_KEYFRAME = Path(__file__).resolve().parents[3] / "assets" / "vbd_mjvbd_v2" / "vbd_w1_right_hand_last_keyframe.json"
FREE_SOFT_CONTACT = (
    soft_reference.soft0.SOFT_CONTACT_KE,
    soft_reference.soft0.SOFT_CONTACT_KD,
    soft_reference.soft0.SOFT_CONTACT_MU,
)
GRASP_SOFT_CONTACT = (
    soft_reference.soft0.GRASP_CONTACT_KE,
    soft_reference.soft0.GRASP_CONTACT_KD,
    soft_reference.soft0.GRASP_SOFT_CONTACT_MU,
)
RIGID_GRASP_CONTACT = (
    rigid_reference.GRASP_CONTACT_KE,
    rigid_reference.GRASP_CONTACT_KD,
    rigid_reference.GRASP_FRICTION,
)
RIGID_RELEASE_CONTACT = (
    rigid_reference.RELEASE_CONTACT_KE,
    rigid_reference.RELEASE_CONTACT_KD,
    rigid_reference.RELEASE_FRICTION,
)

# Match the full-W1 recorded soft-cube and compliant-bag scene.
recorder.CUBE_DENSITY = soft_reference.soft0.SOFT_CUBE_DENSITY
recorder.CUBE_DIMS = soft_reference.soft0.SOFT_CUBE_DIMS
recorder.CUBE_K_MU = soft_reference.soft0.SOFT_CUBE_K_MU
recorder.CUBE_K_LAMBDA = soft_reference.soft0.SOFT_CUBE_K_LAMBDA
recorder.CUBE_K_DAMP = soft_reference.soft0.SOFT_CUBE_K_DAMP
recorder.CUBE_PARTICLE_RADIUS = soft_reference.soft0.SOFT_CUBE_PARTICLE_RADIUS
recorder.CUBE_SELF_CONTACT_RADIUS = max(
    soft_reference.soft0.BAG_PARTICLE_RADIUS,
    soft_reference.soft0.SOFT_CUBE_PARTICLE_RADIUS,
)
recorder.CUBE_SELF_CONTACT_MARGIN = 2.0 * recorder.CUBE_SELF_CONTACT_RADIUS
recorder.CONTACT_KE = GRASP_SOFT_CONTACT[0]
recorder.CONTACT_KD = GRASP_SOFT_CONTACT[1]
recorder.CONTACT_MU = soft_reference.soft0.GRASP_FRICTION
recorder.CONTACT_MARGIN = soft_reference.soft0.SOFT_CONTACT_MARGIN
recorder.RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE = soft_reference.soft0.RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE
recorder.BAG_RESOLUTION = soft_reference.soft0.BAG_RESOLUTION
recorder.BAG_PARTICLE_RADIUS = soft_reference.soft0.BAG_PARTICLE_RADIUS
recorder.BAG_DENSITY = soft_reference.soft0.BAG_DENSITY
recorder.BAG_TRI_KE = soft_reference.soft0.BAG_TRI_KE
recorder.BAG_TRI_KA = soft_reference.soft0.BAG_TRI_KA
recorder.BAG_TRI_KD = soft_reference.soft0.BAG_TRI_KD
recorder.BAG_EDGE_KE = soft_reference.soft0.BAG_EDGE_KE
recorder.BAG_EDGE_KD = soft_reference.soft0.BAG_EDGE_KD

APPROACH_ROOT = wp.transform(
    wp.vec3(-0.16214203834533691, -2.838686943054199, 1.3409454822540283),
    wp.quat(0.09465623646974564, 0.9546480774879456, -0.2820824682712555, 0.010803722776472569),
)
APPROACH_JOINTS = {
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
OPEN_JOINTS = dict.fromkeys(recorder.HAND_JOINTS, 0.0)
RIGID_GRASP_JOINTS = dict(recorder.RECORDED_JOINT_DEGREES)

RIGID_CUBE_OFFSET = wp.vec3(-0.11, 0.0, 0.0)
RIGID_CUBE_CENTRE = recorder.CUBE_CENTRE + RIGID_CUBE_OFFSET
# Reproduce the working full-robot reference's fingertip-pad transforms in the
# isolated-hand root frame. The soft-grasp root differs by about 6.6 degrees.
RIGID_GRASP_ROOT = wp.transform(
    wp.vec3(-0.2589742839336395, -2.834425926208496, 1.3404709100723267),
    wp.quat(0.0508713573217392, 0.9490510821342468, -0.3089625239372253, 0.03544386848807335),
)
RIGID_CUBE_DENSITY = rigid_reference.CUBE_DENSITY
RIGID_CUBE_MARGIN = rigid_reference.CUBE_MARGIN
RIGID_BODY_CONTACT_BUFFER_SIZE = 4096

RIGID_FINGER_PADS = (
    (
        "right_thumb_dist",
        (0.030, 0.006, 0.015),
        wp.transform(
            wp.vec3(-0.0548988, 0.0529312, 0.0141373),
            wp.quat(0.2773950, -0.8700770, 0.0411925, 0.4053648),
        ),
    ),
    (
        "right_index_dist",
        (0.030, 0.006, 0.015),
        wp.transform(
            wp.vec3(-0.0388362, 0.0073618, -0.0145817),
            wp.quat(0.0581093, -0.6604385, 0.7410694, 0.1061119),
        ),
    ),
    (
        "right_middle_dist",
        (0.014, 0.006, 0.012),
        wp.transform(
            wp.vec3(-0.0004232, 0.0163224, 0.0208014),
            wp.quat(0.0982183, -0.9514985, 0.2900473, 0.0295879),
        ),
    ),
    (
        "right_ring_dist",
        (0.014, 0.006, 0.012),
        wp.transform(
            wp.vec3(-0.0013989, 0.0131392, 0.0240774),
            wp.quat(0.0912622, -0.9611189, 0.2605852, -0.0040627),
        ),
    ),
    (
        "right_pinky_dist",
        (0.014, 0.006, 0.012),
        wp.transform(
            wp.vec3(-0.0025118, 0.0185027, 0.0180359),
            wp.quat(0.0746882, -0.9661100, 0.2468487, -0.0108671),
        ),
    ),
)


class Example(recorder.Example):
    """Run two physical five-finger pick-and-place operations with one hand."""

    def __init__(self, viewer, args):
        self.include_bag = True
        self.particle_self_contact_enabled = True
        self.grasp_joints = self._load_grasp_joints(args.grasp_keyframe)
        self.hand_soft_contact_enabled = True
        self.soft_release_applied = False
        self.rigid_release_applied = False
        self.rigid_grasp_material_applied = False
        super().__init__(viewer, args)
        self._set_hand_target(APPROACH_ROOT, APPROACH_JOINTS)
        self._set_initial_hand_pose()
        self._set_hand_soft_contact(False)
        self._set_hand_shape_friction(0.0)
        self._create_solver()
        self.segments = self._build_segments()
        self.script_duration = sum(segment[0] for segment in self.segments)

    @staticmethod
    def _load_grasp_joints(path_value: str) -> dict[str, float]:
        """Load the recorded five-finger closure target."""

        path = Path(path_value).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Recorded grasp keyframe not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        joints = payload["keyframe"]["target_finger_joints_degrees"]
        return {name: float(value) for name, value in joints.items()}

    def _set_hand_target(self, root: wp.transform, joints: dict[str, float]):
        """Set the floating-hand root and finger target for the next frame."""

        target_q = self.manual_target_q.numpy()
        position = wp.transform_get_translation(root)
        rotation = wp.transform_get_rotation(root)
        target_q[self.root_q_start : self.root_q_start + 7] = [*position, *rotation]
        for name, index in self.hand_joint_indices.items():
            target_q[index] = np.radians(joints[name])
        self.manual_target_q.assign(target_q)

    def _create_solver(self):
        """Allocate contact buffers sized for the hand's many rigid colliders."""

        self.solver = SolverMJVBDV2(
            self.model,
            mujoco_articulations=self.hand_articulations,
            joint_mode="kinematic",
            contact_mode="full",
            vbd_options={
                "iterations": recorder.VBD_ITERATIONS,
                "rigid_body_contact_buffer_size": RIGID_BODY_CONTACT_BUFFER_SIZE,
                "rigid_body_particle_contact_buffer_size": recorder.RIGID_BODY_PARTICLE_CONTACT_BUFFER_SIZE,
                "particle_enable_self_contact": self.particle_self_contact_enabled,
                "particle_self_contact_radius": recorder.CUBE_SELF_CONTACT_RADIUS,
                "particle_self_contact_margin": recorder.CUBE_SELF_CONTACT_MARGIN,
                "particle_vertex_contact_buffer_size": recorder.PARTICLE_VERTEX_CONTACT_BUFFER_SIZE,
                "particle_edge_contact_buffer_size": recorder.PARTICLE_EDGE_CONTACT_BUFFER_SIZE,
                "particle_collision_detection_interval": 0,
                "particle_topological_contact_filter_threshold": 3,
                "particle_rest_shape_contact_exclusion_radius": 0.03,
            },
            collision_options={
                "broad_phase": "nxn",
                "soft_contact_margin": recorder.CONTACT_MARGIN,
                "enable_rigid_soft_full_surface_contact": True,
            },
        )

    def _build_scene(self):
        """Build the recorded soft-cube scene plus one dynamic rigid cube."""

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.8))
        builder.default_shape_cfg.ke = recorder.CONTACT_KE
        builder.default_shape_cfg.kd = recorder.CONTACT_KD
        builder.default_shape_cfg.mu = recorder.CONTACT_MU
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
        finger_pad_cfg = newton.ModelBuilder.ShapeConfig(
            ke=RIGID_GRASP_CONTACT[0],
            kd=RIGID_GRASP_CONTACT[1],
            mu=RIGID_GRASP_CONTACT[2],
            is_visible=False,
        )
        self.rigid_finger_pad_start = builder.shape_count
        for body_name, half_extents, pad_xform in RIGID_FINGER_PADS:
            body = next(index for index, label in enumerate(builder.body_label) if label.endswith(body_name))
            builder.add_shape_box(
                body,
                hx=half_extents[0],
                hy=half_extents[1],
                hz=half_extents[2],
                cfg=finger_pad_cfg,
                xform=pad_xform,
                label=f"{body_name}_physical_pad",
            )
        self.rigid_finger_pad_end = builder.shape_count
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
            label="two_pick_table",
        )
        builder.add_ground_plane(height=-0.00377202, label="two_pick_ground")

        bag_vertices, bag_indices = recorder._generate_box_bag(
            0.5 * recorder.BAG_WIDTH,
            0.5 * recorder.BAG_DEPTH,
            recorder.BAG_HEIGHT,
            recorder.BAG_RESOLUTION,
        )
        self.bag_particle_start = builder.particle_count
        builder.add_cloth_mesh(
            pos=recorder.BAG_POS,
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(),
            vertices=bag_vertices.tolist(),
            indices=bag_indices,
            density=recorder.BAG_DENSITY,
            tri_ke=recorder.BAG_TRI_KE,
            tri_ka=recorder.BAG_TRI_KA,
            tri_kd=recorder.BAG_TRI_KD,
            edge_ke=recorder.BAG_EDGE_KE,
            edge_kd=recorder.BAG_EDGE_KD,
            particle_radius=recorder.BAG_PARTICLE_RADIUS,
            label="two_pick_soft_box_bag",
        )
        self.bag_particle_end = builder.particle_count
        bag_top = np.flatnonzero(np.abs(bag_vertices[:, 2] - recorder.BAG_HEIGHT) < 1.0e-5)
        self.bag_top_indices = bag_top.astype(np.int32) + self.bag_particle_start

        cube_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi)
        cube_origin = recorder.CUBE_CENTRE - wp.quat_rotate(cube_rotation, wp.vec3(*recorder.CUBE_HALF_EXTENTS))
        self.soft_cube_particle_start = builder.particle_count
        builder.add_soft_grid(
            pos=cube_origin,
            rot=cube_rotation,
            vel=wp.vec3(),
            dim_x=recorder.CUBE_DIMS[0],
            dim_y=recorder.CUBE_DIMS[1],
            dim_z=recorder.CUBE_DIMS[2],
            cell_x=2.0 * recorder.CUBE_HALF_EXTENTS[0] / recorder.CUBE_DIMS[0],
            cell_y=2.0 * recorder.CUBE_HALF_EXTENTS[1] / recorder.CUBE_DIMS[1],
            cell_z=2.0 * recorder.CUBE_HALF_EXTENTS[2] / recorder.CUBE_DIMS[2],
            density=recorder.CUBE_DENSITY,
            k_mu=recorder.CUBE_K_MU,
            k_lambda=recorder.CUBE_K_LAMBDA,
            k_damp=recorder.CUBE_K_DAMP,
            particle_radius=recorder.CUBE_PARTICLE_RADIUS,
            label="first_soft_cube",
        )
        self.soft_cube_particle_end = builder.particle_count

        rigid_cfg = newton.ModelBuilder.ShapeConfig(
            density=RIGID_CUBE_DENSITY,
            ke=recorder.CONTACT_KE,
            kd=recorder.CONTACT_KD,
            mu=recorder.CONTACT_MU,
            margin=RIGID_CUBE_MARGIN,
        )
        rigid_cfg.configure_sdf(force_sdf=True)
        rigid_cfg.has_particle_collision = True
        self.rigid_cube_body = builder.add_body(
            xform=wp.transform(RIGID_CUBE_CENTRE, cube_rotation),
            label="second_rigid_cube",
        )
        self.rigid_cube_shape = builder.shape_count
        builder.add_shape_box(
            self.rigid_cube_body,
            hx=recorder.CUBE_HALF_EXTENTS[0],
            hy=recorder.CUBE_HALF_EXTENTS[1],
            hz=recorder.CUBE_HALF_EXTENTS[2],
            cfg=rigid_cfg,
            color=(0.90, 0.32, 0.18),
            label="second_rigid_cube_shape",
        )

        collide_shapes = int(newton.ShapeFlags.COLLIDE_SHAPES)
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        for shape in range(builder.shape_count):
            builder.shape_flags[shape] |= collide_shapes | collide_particles
        # Preserve the soft-cube reference contact geometry during the first
        # pick; the pads are needed only for stable rigid-rigid fingertip contact.
        for shape in range(self.rigid_finger_pad_start, self.rigid_finger_pad_end):
            builder.shape_flags[shape] &= ~collide_particles
        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = recorder.CONTACT_KE
        self.model.soft_contact_kd = recorder.CONTACT_KD
        self.model.soft_contact_mu = recorder.CONTACT_MU

    def _set_hand_soft_contact(self, enabled: bool):
        """Toggle only hand-to-soft-particle contact for the first pick."""

        if enabled == self.hand_soft_contact_enabled:
            return
        flags = self.model.shape_flags.numpy()
        particle_flag = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        if enabled:
            flags[: self.hand_soft_shape_end] |= particle_flag
            self.model.soft_contact_ke = GRASP_SOFT_CONTACT[0]
            self.model.soft_contact_kd = GRASP_SOFT_CONTACT[1]
            self.model.soft_contact_mu = GRASP_SOFT_CONTACT[2]
        else:
            flags[: self.hand_soft_shape_end] &= ~particle_flag
            self.model.soft_contact_ke = FREE_SOFT_CONTACT[0]
            self.model.soft_contact_kd = FREE_SOFT_CONTACT[1]
            self.model.soft_contact_mu = FREE_SOFT_CONTACT[2]
        self.model.shape_flags.assign(flags)
        self.hand_soft_contact_enabled = enabled

    def _set_hand_shape_friction(self, friction: float):
        """Set rigid-contact friction for the floating hand shapes."""

        values = self.model.shape_material_mu.numpy()
        values[: self.hand_shape_end] = friction
        self.model.shape_material_mu.assign(values)

    def _apply_soft_release_material(self):
        """Match the recorded soft-cube release contact and friction."""

        if self.soft_release_applied:
            return
        self._set_hand_shape_friction(0.0)
        self.model.soft_contact_ke = FREE_SOFT_CONTACT[0]
        self.model.soft_contact_kd = FREE_SOFT_CONTACT[1]
        self.model.soft_contact_mu = FREE_SOFT_CONTACT[2]
        self.soft_release_applied = True

    def _apply_rigid_grasp_material(self):
        """Match the rigid-cube reference material during the second pick."""

        if self.rigid_grasp_material_applied:
            return
        shape_mu = self.model.shape_material_mu.numpy()
        shape_ke = self.model.shape_material_ke.numpy()
        shape_kd = self.model.shape_material_kd.numpy()
        shape_mu[: self.hand_shape_end] = RIGID_GRASP_CONTACT[2]
        shape_mu[self.rigid_cube_shape] = RIGID_GRASP_CONTACT[2]
        shape_ke[: self.hand_shape_end] = RIGID_GRASP_CONTACT[0]
        shape_ke[self.rigid_cube_shape] = RIGID_GRASP_CONTACT[0]
        shape_kd[: self.hand_shape_end] = RIGID_GRASP_CONTACT[1]
        shape_kd[self.rigid_cube_shape] = RIGID_GRASP_CONTACT[1]
        self.model.shape_material_mu.assign(shape_mu)
        self.model.shape_material_ke.assign(shape_ke)
        self.model.shape_material_kd.assign(shape_kd)
        self.rigid_grasp_material_applied = True

    def _apply_rigid_release_material(self):
        """Match the rigid reference release material after opening the hand."""

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
        self.rigid_release_applied = True

    def _build_segments(self):
        """Build two recorded five-finger pick-and-place sequences."""

        soft_approach = APPROACH_ROOT
        rigid_approach = RIGID_GRASP_ROOT
        rigid_pregrasp = wp.transform(
            wp.transform_get_translation(rigid_approach) + wp.vec3(0.0, 0.0, 0.18),
            wp.transform_get_rotation(rigid_approach),
        )
        soft_root_cube_offset = wp.transform_get_translation(soft_approach) - recorder.CUBE_CENTRE
        rigid_root_cube_offset = wp.transform_get_translation(rigid_approach) - RIGID_CUBE_CENTRE
        cube_release_height = float(recorder.BAG_POS[2]) + recorder.BAG_HEIGHT + 0.06
        soft_bag_hover = wp.transform(
            wp.vec3(
                float(recorder.BAG_POS[0]) + float(soft_root_cube_offset[0]),
                float(recorder.BAG_POS[1]) + float(soft_root_cube_offset[1]),
                cube_release_height + float(soft_root_cube_offset[2]),
            ),
            wp.transform_get_rotation(soft_approach),
        )
        rigid_bag_hover = wp.transform(
            wp.vec3(
                float(recorder.BAG_POS[0]) + float(rigid_root_cube_offset[0]),
                float(recorder.BAG_POS[1]) + float(rigid_root_cube_offset[1]),
                cube_release_height + float(rigid_root_cube_offset[2]),
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
            wp.transform_get_translation(rigid_bag_hover) + wp.vec3(0.0, 0.0, 0.10),
            wp.transform_get_rotation(rigid_bag_hover),
        )
        return (
            (0.50, soft_approach, soft_approach, APPROACH_JOINTS, APPROACH_JOINTS, "soft_wait"),
            (1.80, soft_approach, soft_approach, APPROACH_JOINTS, self.grasp_joints, "soft_grasp"),
            (0.60, soft_approach, soft_approach, self.grasp_joints, self.grasp_joints, "soft_carry"),
            (1.20, soft_approach, soft_lift, self.grasp_joints, self.grasp_joints, "soft_carry"),
            (7.00, soft_lift, soft_bag_hover, self.grasp_joints, self.grasp_joints, "soft_carry"),
            (0.40, soft_bag_hover, soft_bag_hover, self.grasp_joints, self.grasp_joints, "soft_carry"),
            (
                soft_reference.soft0.SOFT_CUBE_RELEASE_OPEN_DURATION,
                soft_bag_hover,
                soft_bag_hover,
                self.grasp_joints,
                OPEN_JOINTS,
                "soft_release",
            ),
            (
                soft_reference.soft0.SOFT_CUBE_RELEASE_SETTLE_DURATION,
                soft_bag_hover,
                soft_bag_hover,
                OPEN_JOINTS,
                OPEN_JOINTS,
                "soft_release",
            ),
            (1.00, soft_bag_hover, soft_retreat, OPEN_JOINTS, OPEN_JOINTS, "soft_release"),
            (0.80, soft_retreat, rigid_pregrasp, OPEN_JOINTS, OPEN_JOINTS, "rigid_approach"),
            (1.80, rigid_pregrasp, rigid_approach, OPEN_JOINTS, RIGID_GRASP_JOINTS, "rigid_grasp"),
            (1.00, rigid_approach, rigid_approach, RIGID_GRASP_JOINTS, RIGID_GRASP_JOINTS, "rigid_carry"),
            (2.00, rigid_approach, rigid_lift, RIGID_GRASP_JOINTS, RIGID_GRASP_JOINTS, "rigid_carry"),
            (0.60, rigid_lift, rigid_lift, RIGID_GRASP_JOINTS, RIGID_GRASP_JOINTS, "rigid_carry"),
            (5.00, rigid_lift, rigid_transport, RIGID_GRASP_JOINTS, RIGID_GRASP_JOINTS, "rigid_carry"),
            (1.50, rigid_transport, rigid_bag_hover, RIGID_GRASP_JOINTS, RIGID_GRASP_JOINTS, "rigid_carry"),
            (0.40, rigid_bag_hover, rigid_bag_hover, RIGID_GRASP_JOINTS, RIGID_GRASP_JOINTS, "rigid_carry"),
            (0.80, rigid_bag_hover, rigid_bag_hover, RIGID_GRASP_JOINTS, OPEN_JOINTS, "rigid_release"),
            (0.20, rigid_bag_hover, rigid_bag_hover, OPEN_JOINTS, OPEN_JOINTS, "rigid_release"),
            (1.00, rigid_bag_hover, rigid_retreat, OPEN_JOINTS, OPEN_JOINTS, "rigid_release"),
        )

    def _sample(self, time_s: float):
        """Interpolate one hand target from the two-pick script."""

        for duration, root_a, root_b, joints_a, joints_b, phase in self.segments:
            if time_s <= duration:
                alpha = float(np.clip(time_s / duration, 0.0, 1.0))
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                root = self._lerp_transform(root_a, root_b, alpha)
                joints = {
                    name: joints_a[name] * (1.0 - alpha) + joints_b[name] * alpha for name in recorder.HAND_JOINTS
                }
                return root, joints, phase
            time_s -= duration
        _, _, root, _, joints, phase = self.segments[-1]
        return root, joints, phase

    @staticmethod
    def _lerp_transform(a: wp.transform, b: wp.transform, alpha: float):
        """Linearly interpolate position and normalize the interpolated quaternion."""

        position_a = np.asarray(wp.transform_get_translation(a), dtype=np.float32)
        position_b = np.asarray(wp.transform_get_translation(b), dtype=np.float32)
        rotation_a = np.asarray(wp.transform_get_rotation(a), dtype=np.float32)
        rotation_b = np.asarray(wp.transform_get_rotation(b), dtype=np.float32)
        if np.dot(rotation_a, rotation_b) < 0.0:
            rotation_b = -rotation_b
        rotation = rotation_a * (1.0 - alpha) + rotation_b * alpha
        rotation /= max(np.linalg.norm(rotation), 1.0e-8)
        return wp.transform(wp.vec3(*(position_a * (1.0 - alpha) + position_b * alpha)), wp.quat(*rotation))

    def step(self):
        """Advance the two physical pick-and-place operations by one frame."""

        root, joints, phase = self._sample(self.sim_time)
        self._set_hand_target(root, joints)
        soft_grasp = phase in {"soft_grasp", "soft_carry"}
        self._set_hand_soft_contact(soft_grasp or phase == "soft_release")
        if soft_grasp:
            self._set_hand_shape_friction(recorder.CONTACT_MU)
        if phase in {"rigid_approach", "rigid_grasp", "rigid_carry", "rigid_release"}:
            self._apply_rigid_grasp_material()
        if phase == "soft_release":
            self._apply_soft_release_material()
        if phase == "rigid_release":
            self._apply_rigid_release_material()
        self.step_once()

    def render(self):
        """Render the hand, two cubes, and soft bag without the recorder controls."""

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Verify finite physical state after the sequential placements."""

        body_flags = int(self.model.body_flags.numpy()[self.rigid_cube_body])
        assert not body_flags & int(newton.BodyFlags.KINEMATIC), (
            "The rigid cube must be dynamic when the solver is constructed"
        )
        assert np.all(np.isfinite(self.state_0.joint_q.numpy()))
        assert np.all(np.isfinite(self.state_0.particle_q.numpy()))
        assert np.all(np.isfinite(self.state_0.body_q.numpy()))

    @staticmethod
    def create_parser():
        """Create parser options for the sequential soft-and-rigid pick demo."""

        parser = recorder.Example.create_parser()
        parser.set_defaults(num_frames=1900, paused=False)
        parser.add_argument(
            "--grasp-keyframe",
            default=str(DEFAULT_GRASP_KEYFRAME),
            help="Latest grasp keyframe JSON generated by the right-hand recorder.",
        )
        return parser


def main():
    """Run the right-hand sequential soft-and-rigid pick demo."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

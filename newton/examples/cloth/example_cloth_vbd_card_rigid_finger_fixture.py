# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""A simplified rigid-finger fixture holding a vertical VBD card packet.

The fixture follows ``side_view_geometry_exact_curve.svg``:

* A horizontal capsule touches the packet's upper edge as the thumb.
* A second horizontal capsule stays clear of the packet as the index finger.
* Two copies of one continuous, Blender-extruded U-profile mesh support the
  packet's lower edge as the middle and ring fingers.

All finger bodies are kinematic. Distributed compliant springs keep the active
card particles inside the authored grip patches; :class:`newton.solvers.SolverVBD`
solves the complete card packet.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverVBD

ASSET_DIR = Path(__file__).with_name("assets") / "card_rigid_finger_fixture"
SUPPORT_MESH_PATH = ASSET_DIR / "support_finger_exact_svg.obj"

CARD_COUNT = 4
CARD_WIDTH = 0.063
CARD_LENGTH = 0.090
CARD_LAYER_GAP = 2.8e-4
CARD_PARTICLE_RADIUS = 1.2e-4
CARD_DIM_WIDTH = 12
CARD_DIM_LENGTH = 18
CARD_BOTTOM_Z = 0.100
CARD_TOP_Z = CARD_BOTTOM_Z + CARD_LENGTH

# A thin contact skin lets VBD react before the rendered card reaches the
# rigid surface; it is smaller than one SDF voxel at the chosen resolution.
CONTACT_MARGIN = 2.0e-4
CONTACT_QUERY_MARGIN = 1.5e-3
CONTACT_STIFFNESS = 5.0e6
CONTACT_DAMPING = 1.5e3
CONTACT_FRICTION = 4.0
SUPPORT_FRICTION = 12.0
SUPPORT_SDF_MAX_RESOLUTION = 192
SUPPORT_SDF_PADDING = 3.0e-3
BOTTOM_GRIP_SPRING_STIFFNESS = 5.0e8
BOTTOM_GRIP_SPRING_DAMPING = 5.0e3
TOP_GRIP_SPRING_STIFFNESS = 1.0e8
TOP_GRIP_SPRING_DAMPING = 2.0e3

THUMB_RADIUS = 0.006
THUMB_HALF_HEIGHT = 0.022
THUMB_PRELOAD = 5.0e-5
THUMB_CENTER_X = 0.005
THUMB_TILT = math.radians(4.0)
INDEX_RADIUS = 0.006
INDEX_HALF_HEIGHT = 0.020
INDEX_GAP = 0.010
INDEX_Z = CARD_BOTTOM_Z + 0.58 * CARD_LENGTH
INDEX_MOTION_DELAY = 0.5
INDEX_MOTION_DURATION = 2.5
INDEX_PUSH_DISTANCE = 0.013

SUPPORT_DEPTH_OFFSET = 0.017
SUPPORT_HALF_DEPTH = 0.0045
SUPPORT_HEIGHT_TOLERANCE = 1.0e-3
FINGER_COLOR = (0.72, 0.76, 0.82)
CARD_COLOR_TOP = (0.68, 0.025, 0.025)
CARD_COLOR_BOTTOM = (0.91, 0.86, 0.73)


def _load_triangle_obj(path: Path) -> newton.Mesh:
    """Load the Blender-created support mesh without an extra dependency."""
    vertices = []
    indices = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append(tuple(float(value) for value in fields[1:4]))
        elif fields[0] == "f":
            face = [int(value.split("/", 1)[0]) - 1 for value in fields[1:]]
            if len(face) != 3:
                raise ValueError(f"Expected a triangulated OBJ face in {path}")
            indices.extend(face)
    if not vertices or not indices:
        raise ValueError(f"Support mesh is empty: {path}")
    return newton.Mesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        indices=np.asarray(indices, dtype=np.int32),
        compute_inertia=False,
        is_solid=True,
    )


@wp.kernel
def _set_kinematic_body_pose(
    body_q: wp.array[wp.transform],
    body_index: int,
    pose: wp.transform,
):
    if wp.tid() == 0:
        body_q[body_index] = pose


class Example:
    """Hold a vertical VBD card packet in a simplified rigid fixture."""

    def __init__(self, viewer, args):
        if not SUPPORT_MESH_PATH.is_file():
            raise FileNotFoundError(f"Blender support mesh not found: {SUPPORT_MESH_PATH}")

        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.card_render_indices_host: list[list[int]] = []

        self._build_model()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.initial_particle_q = self.state_0.particle_q.numpy().copy()

        self.solver = SolverVBD(
            self.model,
            iterations=40,
            friction_epsilon=1.0e-6,
            particle_enable_self_contact=True,
            particle_self_contact_radius=CARD_LAYER_GAP,
            particle_self_contact_margin=2.0 * CARD_LAYER_GAP,
            particle_collision_detection_interval=2,
            particle_vertex_contact_buffer_size=64,
            particle_edge_contact_buffer_size=128,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.0,
            rigid_body_particle_contact_buffer_size=8192,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=CONTACT_QUERY_MARGIN,
            enable_rigid_soft_full_surface_contact=True,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.initial_contact_counts = self._finger_contact_counts()
        self._validate_layout(self.initial_contact_counts, "initial")

        self.card_render_indices = [
            wp.array(indices, dtype=wp.int32, device=self.device) for indices in self.card_render_indices_host
        ]

        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        self.viewer.show_particles = False
        self.viewer.set_camera(pos=wp.vec3(0.0, -0.34, 0.15), pitch=0.0, yaw=90.0)

    def _build_model(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        capsule_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * math.pi)
        thumb_rotation = wp.quat_from_axis_angle(
            wp.vec3(0.0, 1.0, 0.0),
            0.5 * math.pi - THUMB_TILT,
        )

        capsule_cfg = newton.ModelBuilder.ShapeConfig(
            ke=CONTACT_STIFFNESS,
            kd=CONTACT_DAMPING,
            mu=CONTACT_FRICTION,
            margin=CONTACT_MARGIN,
        )
        support_cfg = capsule_cfg.copy()
        support_cfg.mu = SUPPORT_FRICTION

        leftmost_card_x = -0.5 * (CARD_COUNT - 1) * CARD_LAYER_GAP
        thumb_center_z = (
            CARD_TOP_Z
            + (THUMB_CENTER_X - leftmost_card_x) * math.tan(THUMB_TILT)
            + (CARD_PARTICLE_RADIUS + CONTACT_MARGIN + THUMB_RADIUS) / math.cos(THUMB_TILT)
            - THUMB_PRELOAD
        )
        thumb_body = builder.add_body(
            xform=wp.transform(
                wp.vec3(THUMB_CENTER_X, 0.0, thumb_center_z),
                wp.quat_identity(),
            ),
            label="thumb_capsule",
        )
        builder.body_flags[thumb_body] = int(newton.BodyFlags.KINEMATIC)
        builder.add_shape_capsule(
            thumb_body,
            xform=wp.transform(wp.vec3(0.0), thumb_rotation),
            radius=THUMB_RADIUS,
            half_height=THUMB_HALF_HEIGHT,
            cfg=capsule_cfg,
            color=FINGER_COLOR,
            label="thumb_capsule_shape",
        )

        index_center_x = 0.5 * (CARD_COUNT - 1) * CARD_LAYER_GAP + INDEX_GAP + INDEX_RADIUS + INDEX_HALF_HEIGHT
        self.index_initial_position = wp.vec3(index_center_x, 0.0, INDEX_Z)
        index_body = builder.add_body(
            xform=wp.transform(self.index_initial_position, wp.quat_identity()),
            label="index_capsule",
        )
        builder.body_flags[index_body] = int(newton.BodyFlags.KINEMATIC)
        self.index_body = index_body
        builder.add_shape_capsule(
            index_body,
            xform=wp.transform(wp.vec3(0.0), capsule_rotation),
            radius=INDEX_RADIUS,
            half_height=INDEX_HALF_HEIGHT,
            cfg=capsule_cfg,
            color=FINGER_COLOR,
            label="index_capsule_shape",
        )

        support_mesh = _load_triangle_obj(SUPPORT_MESH_PATH)
        support_mesh.build_sdf(
            max_resolution=SUPPORT_SDF_MAX_RESOLUTION,
            margin=SUPPORT_SDF_PADDING,
        )
        support_vertices = np.asarray(support_mesh.vertices)
        center_band = np.abs(support_vertices[:, 0]) < 3.0e-4
        if not np.any(center_band):
            raise ValueError("The exact U-profile has no centerline samples")
        support_inner_center_z = float(np.max(support_vertices[center_band, 2]))
        support_body_z = CARD_BOTTOM_Z - CARD_PARTICLE_RADIUS - CONTACT_MARGIN - support_inner_center_z

        for label, y in (("middle_support", -SUPPORT_DEPTH_OFFSET), ("ring_support", SUPPORT_DEPTH_OFFSET)):
            body = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, y, support_body_z), wp.quat_identity()),
                label=label,
            )
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)
            builder.add_shape_mesh(
                body,
                mesh=support_mesh,
                cfg=support_cfg,
                color=FINGER_COLOR,
                label=f"{label}_shape",
            )

        self._add_card_packet(builder)
        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = CONTACT_STIFFNESS
        self.model.soft_contact_kd = CONTACT_DAMPING
        self.model.soft_contact_mu = CONTACT_FRICTION

    def _add_card_packet(self, builder):
        row_width = CARD_DIM_WIDTH + 1
        indices = []
        attachment_specs = []
        for row in range(CARD_DIM_LENGTH):
            for column in range(CARD_DIM_WIDTH):
                a = row * row_width + column
                b = a + 1
                c = a + row_width
                d = c + 1
                if (row + column) & 1:
                    indices.extend((a, b, c, b, d, c))
                else:
                    indices.extend((a, b, d, a, d, c))

        for layer in range(CARD_COUNT):
            x = (layer - 0.5 * (CARD_COUNT - 1)) * CARD_LAYER_GAP
            vertices = []
            for row in range(CARD_DIM_LENGTH + 1):
                fraction = row / CARD_DIM_LENGTH
                z = CARD_BOTTOM_Z + CARD_LENGTH * fraction
                for column in range(CARD_DIM_WIDTH + 1):
                    y = -0.5 * CARD_WIDTH + CARD_WIDTH * column / CARD_DIM_WIDTH
                    vertices.append(wp.vec3(x, y, z))

            particle_start = builder.particle_count
            builder.add_cloth_mesh(
                vertices=vertices,
                indices=indices,
                pos=wp.vec3(0.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                density=1.0,
                scale=1.0,
                tri_ke=5.0e8,
                tri_ka=5.0e8,
                tri_kd=20.0,
                edge_ke=2.0e8,
                edge_kd=500.0,
                particle_radius=CARD_PARTICLE_RADIUS,
                label=f"vertical_card_{layer:02d}",
            )
            self.card_render_indices_host.append([particle_start + index for index in indices])
            support_columns = (2, 3, CARD_DIM_WIDTH - 3, CARD_DIM_WIDTH - 2)
            top_row_start = particle_start + CARD_DIM_LENGTH * row_width
            top_columns = range(CARD_DIM_WIDTH // 2 - 2, CARD_DIM_WIDTH // 2 + 3)
            attachment_specs.extend(
                (
                    particle_start + column,
                    BOTTOM_GRIP_SPRING_STIFFNESS,
                    BOTTOM_GRIP_SPRING_DAMPING,
                )
                for column in support_columns
            )
            attachment_specs.extend(
                (
                    top_row_start + column,
                    TOP_GRIP_SPRING_STIFFNESS,
                    TOP_GRIP_SPRING_DAMPING,
                )
                for column in top_columns
            )

        for particle, stiffness, damping in attachment_specs:
            anchor = builder.add_particle(
                pos=builder.particle_q[particle],
                vel=wp.vec3(0.0),
                mass=0.0,
                radius=0.0,
                flags=0,
            )
            builder.add_spring(
                particle,
                anchor,
                ke=stiffness,
                kd=damping,
                control=0.0,
            )

    def _finger_contact_counts(self):
        self.collision_pipeline.collide(self.state_0, self.contacts)
        count = int(self.contacts.soft_contact_count.numpy()[0])
        shape_body = self.model.shape_body.numpy()
        shapes = self.contacts.soft_contact_shape.numpy()[:count]
        result = {"thumb": 0, "index": 0, "middle": 0, "ring": 0}
        label_to_role = {
            "thumb_capsule": "thumb",
            "index_capsule": "index",
            "middle_support": "middle",
            "ring_support": "ring",
        }
        for shape in shapes:
            if shape < 0:
                continue
            body = shape_body[shape]
            if body < 0:
                continue
            label = self.model.body_label[body]
            role = label_to_role.get(label)
            if role is not None:
                result[role] += 1
        return result

    @staticmethod
    def _validate_layout(counts, stage, require_index_contact=False):
        missing = [role for role in ("thumb", "middle", "ring") if counts[role] == 0]
        if missing:
            raise ValueError(f"{stage} fixture has no card contact for: {', '.join(missing)}; counts={counts}")
        if require_index_contact and counts["index"] == 0:
            raise ValueError(f"{stage} index capsule did not reach the card packet; counts={counts}")
        if not require_index_contact and counts["index"] != 0:
            raise ValueError(f"{stage} index capsule must remain clear of the card packet; counts={counts}")

    def _update_index_pose(self, time):
        u = min(max((time - INDEX_MOTION_DELAY) / INDEX_MOTION_DURATION, 0.0), 1.0)
        smooth_u = u * u * (3.0 - 2.0 * u)
        position = wp.vec3(
            self.index_initial_position[0] - INDEX_PUSH_DISTANCE * smooth_u,
            self.index_initial_position[1],
            self.index_initial_position[2],
        )
        pose = wp.transform(position, wp.quat_identity())
        for state in (self.state_0, self.state_1):
            wp.launch(
                _set_kinematic_body_pose,
                dim=1,
                inputs=[state.body_q, self.index_body, pose],
                device=self.device,
            )

    def step(self):
        for substep in range(self.sim_substeps):
            self._update_index_pose(self.sim_time + substep * self.sim_dt)
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        for layer, indices in enumerate(self.card_render_indices):
            blend = layer / max(CARD_COUNT - 1, 1)
            color = tuple(
                CARD_COLOR_BOTTOM[channel] * (1.0 - blend) + CARD_COLOR_TOP[channel] * blend for channel in range(3)
            )
            self.viewer.log_mesh(
                f"/card_packet/card_{layer:02d}",
                self.state_0.particle_q,
                indices,
                backface_culling=False,
                color=color,
            )
        self.viewer.end_frame()

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()
        if not np.all(np.isfinite(particle_q)) or not np.all(np.isfinite(particle_qd)):
            raise ValueError("Card packet state is not finite")
        motion_finished = self.sim_time >= INDEX_MOTION_DELAY + INDEX_MOTION_DURATION - 0.5 * self.frame_dt
        packet_shift_vector = np.mean(particle_q, axis=0) - np.mean(self.initial_particle_q, axis=0)
        packet_shift = np.linalg.norm(packet_shift_vector)
        if not motion_finished and packet_shift > 0.005:
            counts = self._finger_contact_counts()
            raise ValueError(
                f"Card packet slipped by {packet_shift:.6f} m (delta={packet_shift_vector.tolist()}, contacts={counts})"
            )
        particles_per_card = (CARD_DIM_WIDTH + 1) * (CARD_DIM_LENGTH + 1)
        for layer in range(CARD_COUNT):
            card_slice = slice(layer * particles_per_card, (layer + 1) * particles_per_card)
            card_shift = np.linalg.norm(
                np.mean(particle_q[card_slice], axis=0) - np.mean(self.initial_particle_q[card_slice], axis=0)
            )
            if not motion_finished and card_shift > 0.005:
                raise ValueError(f"Card {layer} slipped by {card_shift:.6f} m")
            initial_card_q = self.initial_particle_q[card_slice]
            support_mask = (
                np.abs(np.abs(initial_card_q[:, 1]) - SUPPORT_DEPTH_OFFSET) <= SUPPORT_HALF_DEPTH + CONTACT_QUERY_MARGIN
            )
            lowest_particle_z = float(np.min(particle_q[card_slice, 2][support_mask]))
            if lowest_particle_z < CARD_BOTTOM_Z - SUPPORT_HEIGHT_TOLERANCE:
                raise ValueError(f"Card {layer} penetrated the support by {CARD_BOTTOM_Z - lowest_particle_z:.6f} m")
        if motion_finished:
            row_width = CARD_DIM_WIDTH + 1
            middle_row = CARD_DIM_LENGTH // 2
            middle_displacements = []
            for layer in range(CARD_COUNT):
                row_start = layer * particles_per_card + middle_row * row_width
                row_slice = slice(row_start, row_start + row_width)
                middle_displacements.append(
                    float(np.mean(particle_q[row_slice, 0] - self.initial_particle_q[row_slice, 0]))
                )
            if max(middle_displacements) > -1.0e-4:
                raise ValueError(f"Index did not bend every card toward -X: {middle_displacements}")
        self._validate_layout(
            self._finger_contact_counts(),
            "final",
            require_index_contact=motion_finished,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=240)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

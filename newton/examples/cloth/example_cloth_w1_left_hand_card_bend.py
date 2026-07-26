# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""A standalone Dexforce W1 left hand holding a stiff, C-bent card packet.

The thumb pad contacts the packet's upper edge, while the middle and ring
fingers support its lower edge. The index finger stays flexed but is left
unconstrained for staged pose tuning, and the little finger remains straight.
The cards use curved, high-stiffness VBD rest shapes and have no pinned
particles; hand contact and friction alone hold the packet.

Run, from the repository root::

    uv run --extra examples -m newton.examples cloth_w1_left_hand_card_bend
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverVBD

HAND_URDF = Path("E:/csy_work/CG/assets/W1-hand-obj/W1-hands-blender/urdf/W1_left_hand/DexforceW1_left_hand.urdf")
HAND_ORIGIN = wp.vec3(0.0, 0.0, 0.38)

# URDF coordinate order: thumb base/distal, then base/distal for the four fingers.
HAND_JOINT_POSE = {
    "LEFT_HAND_THUMB1": np.deg2rad(18.0),
    "LEFT_HAND_THUMB2": np.deg2rad(90.0),
    "LEFT_HAND_INDEX": np.deg2rad(66.0),
    "LEFT_INDEX_PIP": np.deg2rad(102.0),
    "LEFT_HAND_MIDDLE": np.deg2rad(7.0),
    "LEFT_MIDDLE_PIP": np.deg2rad(4.0),
    "LEFT_HAND_RING": np.deg2rad(6.0),
    "LEFT_RING_PIP": np.deg2rad(11.0),
    "LEFT_HAND_PINKY": 0.0,
    "LEFT_PINKY_PIP": 0.0,
}

CARD_COUNT = 8
CARD_WIDTH = 0.058
CARD_DIM_WIDTH = 10
CARD_DIM_LENGTH = 16
CARD_LAYER_GAP = 2.8e-4
CARD_PARTICLE_RADIUS = 1.0e-4

# Hand-local C curve. The packet stays outside the complete hand collision mesh.
CARD_CENTER_X = -0.006
CARD_Y_START = 0.125
CARD_Y_END = 0.180
CARD_Z_START = 0.0791
CARD_Z_END = 0.0080
CARD_BOW = -0.0020
CARD_END_CROSS_TILT = 8.0e-4
INITIAL_CLEARANCE = 0.0

CARD_COLOR_TOP = (0.64, 0.018, 0.025)
CARD_COLOR_BOTTOM = (0.90, 0.86, 0.76)
HAND_COLOR = (0.72, 0.74, 0.78)


class Example:
    """Hold a stiff VBD card packet between the W1 thumb and lower fingers."""

    def __init__(self, viewer, args):
        if not HAND_URDF.is_file():
            raise FileNotFoundError(f"W1 left-hand URDF not found: {HAND_URDF}")

        self.viewer = viewer
        self.args = args
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.card_render_indices: list[wp.array[int]] = []
        self._card_render_indices_host: list[list[int]] = []
        self._bent_particle_q_host: list[tuple[float, float, float]] = []

        self._build_model()
        self.device = self.model.device
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)

        bent_q = wp.array(self._bent_particle_q_host, dtype=wp.vec3, device=self.device)
        wp.copy(self.state_0.particle_q, bent_q)
        wp.copy(self.state_1.particle_q, bent_q)
        self.initial_particle_q = self.state_0.particle_q.numpy().copy()

        self.initial_clearance, self.role_clearances = self._measure_initial_clearance()
        if self.initial_clearance < INITIAL_CLEARANCE:
            raise ValueError(
                f"Initial hand/card clearance is {self.initial_clearance:.6f} m; "
                f"expected at least {INITIAL_CLEARANCE:.6f} m"
            )

        self.solver = SolverVBD(
            self.model,
            iterations=40,
            friction_epsilon=1.0e-5,
            particle_enable_self_contact=True,
            particle_self_contact_radius=3.0e-4,
            particle_self_contact_margin=4.5e-4,
            particle_topological_contact_filter_threshold=1,
            particle_rest_shape_contact_exclusion_radius=0.0,
            rigid_body_particle_contact_buffer_size=4096,
        )
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="nxn",
            soft_contact_margin=3.0e-4,
            enable_rigid_soft_full_surface_contact=True,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.initial_contact_counts = self._grip_contact_counts()
        self._validate_grip_contacts(self.initial_contact_counts, "initial")

        self.card_render_indices = [
            wp.array(indices, dtype=wp.int32, device=self.device) for indices in self._card_render_indices_host
        ]

        self.viewer.set_model(self.model)
        self.viewer.show_triangles = False
        self.viewer.set_camera(pos=wp.vec3(0.29, 0.236, 0.52), pitch=-14.0, yaw=200.0)

    def _build_model(self):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.default_shape_cfg.ke = 3.0e6
        builder.default_shape_cfg.kd = 1.2e3
        builder.default_shape_cfg.mu = 3.0
        builder.default_shape_cfg.configure_sdf(max_resolution=128, force_sdf=True)

        body_start = builder.body_count
        shape_start = builder.shape_count
        builder.add_urdf(
            str(HAND_URDF),
            xform=wp.transform(HAND_ORIGIN, wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            hide_visuals=False,
            parse_visuals_as_colliders=False,
        )
        for body in range(body_start, builder.body_count):
            builder.body_flags[body] = int(newton.BodyFlags.KINEMATIC)

        self.hand_shape_end = builder.shape_count
        self._set_hand_pose(builder)

        for shape in range(shape_start, self.hand_shape_end):
            builder.shape_color[shape] = HAND_COLOR

        self._add_card_packet(builder)
        builder.color(include_bending=True)
        self.model = builder.finalize(requires_grad=False)
        self.model.soft_contact_ke = 3.0e6
        self.model.soft_contact_kd = 1.2e3
        self.model.soft_contact_mu = 3.0

    @staticmethod
    def _set_hand_pose(builder):
        assigned = set()
        for joint, label in enumerate(builder.joint_label):
            match = re.search(r"/([^/]+)$", label)
            if match is None or match.group(1) not in HAND_JOINT_POSE:
                continue
            joint_name = match.group(1)
            coordinate = builder.joint_q_start[joint]
            value = HAND_JOINT_POSE[joint_name]
            builder.joint_q[coordinate] = value
            builder.joint_target_q[coordinate] = value
            assigned.add(joint_name)
        missing = set(HAND_JOINT_POSE) - assigned
        if missing:
            raise ValueError(f"W1 hand joints missing from URDF: {sorted(missing)}")

    def _add_card_packet(self, builder):
        indices = self._card_triangles()
        bent_rows = self._card_rows()
        vertices_per_card = (CARD_DIM_WIDTH + 1) * (CARD_DIM_LENGTH + 1)

        for layer in range(CARD_COUNT):
            layer_z = layer * CARD_LAYER_GAP
            rest_vertices = []
            bent_vertices = []
            for row in range(CARD_DIM_LENGTH + 1):
                for column in range(CARD_DIM_WIDTH + 1):
                    x = CARD_CENTER_X + (column / CARD_DIM_WIDTH - 0.5) * CARD_WIDTH
                    s = row / CARD_DIM_LENGTH
                    cross_tilt = s * CARD_END_CROSS_TILT * ((x - CARD_CENTER_X) / CARD_WIDTH)
                    bent_y, bent_z = bent_rows[row]
                    rest_vertices.append(wp.vec3(x, bent_y, bent_z + cross_tilt + layer_z) + HAND_ORIGIN)
                    point = wp.vec3(x, bent_y, bent_z + cross_tilt + layer_z) + HAND_ORIGIN
                    bent_vertices.append((float(point.x), float(point.y), float(point.z)))

            particle_start = builder.particle_count
            builder.add_cloth_mesh(
                vertices=rest_vertices,
                indices=indices,
                pos=wp.vec3(0.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                density=0.22,
                scale=1.0,
                tri_ke=5.0e6,
                tri_ka=5.0e6,
                tri_kd=20.0,
                edge_ke=5.0e5,
                edge_kd=15.0,
                particle_radius=CARD_PARTICLE_RADIUS,
                label=f"w1_left_card_{layer:02d}",
            )

            self._bent_particle_q_host.extend(bent_vertices)
            self._card_render_indices_host.append([particle_start + index for index in indices])

        expected_particles = CARD_COUNT * vertices_per_card
        if len(self._bent_particle_q_host) != expected_particles:
            raise ValueError("Unexpected card-packet particle count")

    @staticmethod
    def _card_rows():
        bent_rows = []
        for row in range(CARD_DIM_LENGTH + 1):
            s = row / CARD_DIM_LENGTH
            y = CARD_Y_START + (CARD_Y_END - CARD_Y_START) * s
            z = CARD_Z_START + (CARD_Z_END - CARD_Z_START) * s + CARD_BOW * np.sin(np.pi * s)
            bent_rows.append((float(y), float(z)))

        return bent_rows

    @staticmethod
    def _card_triangles():
        indices = []
        row_width = CARD_DIM_WIDTH + 1
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
        return indices

    @staticmethod
    def _card_surface_z(y):
        s = (y - CARD_Y_START) / (CARD_Y_END - CARD_Y_START)
        return CARD_Z_START + (CARD_Z_END - CARD_Z_START) * s + CARD_BOW * np.sin(np.pi * s)

    def _collision_vertices_local(self):
        body_q = self.state_0.body_q.numpy()
        shape_body = self.model.shape_body.numpy()
        shape_q = self.model.shape_transform.numpy()
        shape_flags = self.model.shape_flags.numpy()
        collide_particles = int(newton.ShapeFlags.COLLIDE_PARTICLES)
        origin = np.asarray((float(HAND_ORIGIN.x), float(HAND_ORIGIN.y), float(HAND_ORIGIN.z)))
        result: dict[str, list[np.ndarray]] = {}

        for shape, body in enumerate(shape_body):
            if body < 0 or not (shape_flags[shape] & collide_particles):
                continue
            label = self.model.body_label[body]
            if not label.startswith("DexforceW1_left_hand/"):
                continue
            source = self.model.shape_source[shape]
            if source is None or not hasattr(source, "vertices"):
                continue
            world_from_shape = wp.transform_multiply(
                wp.transform(*body_q[body]),
                wp.transform(*shape_q[shape]),
            )
            vertices = []
            for vertex in source.vertices:
                point = wp.transform_point(world_from_shape, wp.vec3(*vertex))
                vertices.append((float(point.x), float(point.y), float(point.z)))
            result.setdefault(label, []).append(np.asarray(vertices) - origin)
        return result

    def _measure_initial_clearance(self):
        collision_vertices = self._collision_vertices_local()

        samples = {
            "thumb": self._zone_samples(0.0, -0.025, -0.011, top=True),
            "middle": self._zone_samples(1.0, -0.015, 0.000),
            "ring": self._zone_samples(1.0, 0.008, 0.020),
        }
        role_clearances = {}
        for role, target_points in samples.items():
            label = next(name for name in collision_vertices if name.endswith(f"left_{role}_dist"))
            vertices = np.concatenate(collision_vertices[label])
            distances = np.linalg.norm(vertices[:, None, :] - target_points[None, :, :], axis=2)
            role_clearances[role] = float(np.min(distances))
        return min(role_clearances.values()) - CARD_PARTICLE_RADIUS, role_clearances

    def _grip_contact_counts(self):
        self.collision_pipeline.collide(self.state_0, self.contacts)
        count = int(self.contacts.soft_contact_count.numpy()[0])
        shape_body = self.model.shape_body.numpy()
        contact_shapes = self.contacts.soft_contact_shape.numpy()[:count]
        result = {"thumb": 0, "middle": 0, "ring": 0}
        for shape in contact_shapes:
            if shape < 0:
                continue
            label = self.model.body_label[shape_body[shape]]
            for role in result:
                if label.endswith(f"left_{role}_dist"):
                    result[role] += 1
        return result

    @staticmethod
    def _validate_grip_contacts(contact_counts, stage):
        missing = [role for role, count in contact_counts.items() if count == 0]
        if missing:
            raise ValueError(f"{stage} card grip has no contacts for: {', '.join(missing)}; counts={contact_counts}")

    @staticmethod
    def _zone_samples(s, x_start, x_end, *, top=False):
        y = CARD_Y_START + (CARD_Y_END - CARD_Y_START) * s
        x = np.linspace(x_start, x_end, 17)
        z = (
            Example._card_surface_z(y)
            + s * CARD_END_CROSS_TILT * ((x - CARD_CENTER_X) / CARD_WIDTH)
            + (CARD_COUNT - 1) * CARD_LAYER_GAP * top
        )
        return np.column_stack(
            (
                x,
                np.full(17, y),
                z,
            )
        )

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
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
                f"/w1_left_card_packet/card_{layer:02d}",
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
        packet_shift = np.linalg.norm(np.mean(particle_q, axis=0) - np.mean(self.initial_particle_q, axis=0))
        if packet_shift > 0.005:
            raise ValueError(f"The friction-held card packet slipped by {packet_shift:.6f} m")
        if self.initial_clearance < INITIAL_CLEARANCE:
            raise ValueError("Initial W1 hand/card clearance regressed")
        if max(self.role_clearances.values()) > 0.004:
            raise ValueError(f"A gripping finger is too far from its card region: {self.role_clearances}")

        row_width = CARD_DIM_WIDTH + 1
        first_card = particle_q[: (CARD_DIM_LENGTH + 1) * row_width]
        top_z = np.mean(first_card[:row_width, 2])
        middle_start = (CARD_DIM_LENGTH // 2) * row_width
        middle_z = np.mean(first_card[middle_start : middle_start + row_width, 2])
        bottom_z = np.mean(first_card[-row_width:, 2])
        chord_z = 0.5 * (top_z + bottom_z)
        if chord_z - middle_z < 0.0015:
            raise ValueError("The card packet lost its C-shaped bend")
        if np.max(np.linalg.norm(particle_qd, axis=1)) > 1.0:
            raise ValueError("The stiff card packet did not settle")
        self._validate_grip_contacts(self._grip_contact_counts(), "final")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=240)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

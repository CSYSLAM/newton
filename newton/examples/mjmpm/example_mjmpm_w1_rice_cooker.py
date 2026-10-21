# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Scoop rice into a rice cooker with the W1 gripper.

    uv run --extra examples -m newton.examples mjmpm_w1_rice_cooker
    uv run --extra examples -m newton.examples mjmpm_w1_rice_cooker --coupling one_way

The right gripper closes on the handle of a measuring cup filled with MPM rice,
lifts it, carries it over the open rice cooker, rolls the wrist to pour, turns
the empty cup upright, puts it back, and releases it. MuJoCo integrates the
robot; the MPM environment integrates the free cup and the rice. The cup hangs
from the handle by finger friction alone, so the weight of the rice reaches the
robot as a torque on the grip and the finger contact load drops once the rice
has left. With ``--coupling one_way`` the fingers close through the handle and
the cup stays on the table.
"""

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mjmpm.support.w1_two_way import (
    FINGER_OPEN,
    FINGER_SHUT,
    HORIZONTAL_GRIPPER,
    TABLE_Z,
    W1MJMPMScene,
    smoothstep,
)

# Cup body: origin at the bottom centre. Walls are one MPM cell thick so grains
# cannot tunnel through them.
CUP_ORIGIN = np.array((0.49, -0.25))
CUP_OUTER = np.array((0.07, 0.07, 0.07))
CUP_WALL = 0.015
# Handle toward the robot; the gripper closes on its 24 mm width, which the
# finger pads cover like the W1 squeeze-grasp block.
HANDLE_HALF = np.array((0.025, 0.012, 0.015))
HANDLE_CENTER = np.array((-CUP_OUTER[0] / 2 - HANDLE_HALF[0], 0.0, 0.05))
RICE_SPACING = 0.005
RICE_FILL = 0.035
RICE_DENSITY = 850.0
RICE_FRICTION = 0.5
LIFT = 0.19
CARRY = 0.16
POUR_ROLL = np.deg2rad(-135.0)
# Rice cooker modelled in Blender (see assets/rice_cooker): origin at the bottom
# centre, inner pot radius 9 cm and depth 11 cm in a 15 cm body, lid open at the back.
COOKER_ASSETS = Path(__file__).resolve().parent / "assets/rice_cooker"
COOKER_CENTER = np.array((0.49, -0.05))
COOKER_INNER_RADIUS = 0.09
COOKER_HEIGHT = 0.15
FULL_HOLD = (6.2, 7.0)
EMPTY_HOLD = (11.7, 12.4)
CARRY_PHASE = (4.3, 14.0)
# The cup hangs ahead of its handle; while the arm accelerates it settles a few
# degrees in the grip (about 4 deg at 6 substeps, 12 deg at 4).
MAX_SLIP = 0.006
MAX_TURN = np.deg2rad(8.0)


def _load_obj(path):
    """Triangle mesh from an OBJ file with ``v`` and triangular ``f`` records."""
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "v":
            vertices.append([float(v) for v in fields[1:4]])
        elif fields and fields[0] == "f":
            faces.append([int(v.split("/")[0]) - 1 for v in fields[1:4]])
    return newton.Mesh(np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32).ravel())


def _roll(angle):
    """Gripper orientation rolled by ``angle`` [rad] about the world x axis (the approach axis)."""
    roll = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(angle))
    return wp.mul(roll, wp.quat(*HORIZONTAL_GRIPPER))


class Example(W1MJMPMScene):
    def initial_targets(self):
        grasp = np.array((CUP_ORIGIN[0], CUP_ORIGIN[1], TABLE_Z)) + HANDLE_CENTER
        return np.array(((0.43, 0.25, TABLE_Z + 0.10), grasp))

    def build_task(self, builder):
        self.grasp = self.initial_targets()[1]
        cup_cfg = newton.ModelBuilder.ShapeConfig(density=400.0, ke=2.0e4, kd=100.0, mu=1.0)
        self.cup = builder.add_body(
            xform=wp.transform(wp.vec3(CUP_ORIGIN[0], CUP_ORIGIN[1], TABLE_Z), wp.quat_identity()), label="cup"
        )
        hx, hy, hz = CUP_OUTER / 2
        t = CUP_WALL
        boxes = [((0.0, 0.0, t / 2), (hx, hy, t / 2))]
        boxes += [((sx * (hx - t / 2), 0.0, hz), (t / 2, hy, hz)) for sx in (-1, 1)]
        boxes += [((0.0, sy * (hy - t / 2), hz), (hx - t, t / 2, hz)) for sy in (-1, 1)]
        # A rubberized handle: the cup hangs ahead of the grip, so its weight
        # twists the handle in the pad plane against the pads' torsional friction.
        handle_cfg = newton.ModelBuilder.ShapeConfig(
            density=400.0,
            ke=float(__import__("os").environ.get("HANDLE_KE", 1.0e5)),
            kd=100.0,
            mu=float(__import__("os").environ.get("HANDLE_MU", 2.0)),
        )
        boxes = [(center, half, cup_cfg) for center, half in boxes]
        boxes += [(tuple(HANDLE_CENTER), tuple(HANDLE_HALF), handle_cfg)]
        for center, half, cfg in boxes:
            builder.add_shape_box(
                self.cup,
                xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                hx=half[0],
                hy=half[1],
                hz=half[2],
                cfg=cfg,
                color=(0.85, 0.85, 0.80),
                label="cup",
            )
        cooker = wp.transform(wp.vec3(COOKER_CENTER[0], COOKER_CENTER[1], TABLE_Z), wp.quat_identity())
        for part, color in (("body", (0.22, 0.23, 0.25)), ("lid", (0.68, 0.70, 0.73))):
            builder.add_shape_mesh(
                -1,
                xform=cooker,
                mesh=_load_obj(COOKER_ASSETS / f"{part}.obj"),
                cfg=self.surface_cfg,
                color=color,
                label=f"rice_cooker_{part}",
            )

        inner = CUP_OUTER[:2] - 2 * CUP_WALL - RICE_SPACING
        cells = np.floor(np.array((inner[0], inner[1], RICE_FILL)) / RICE_SPACING).astype(int)
        radius = 0.5 * RICE_SPACING
        self.rice_begin = builder.particle_count
        builder.add_particle_grid(
            pos=wp.vec3(
                CUP_ORIGIN[0] - 0.5 * (cells[0] - 1) * RICE_SPACING,
                CUP_ORIGIN[1] - 0.5 * (cells[1] - 1) * RICE_SPACING,
                TABLE_Z + CUP_WALL + radius + 0.001,
            ),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=int(cells[0]),
            dim_y=int(cells[1]),
            dim_z=int(cells[2]),
            cell_x=RICE_SPACING,
            cell_y=RICE_SPACING,
            cell_z=RICE_SPACING,
            mass=RICE_DENSITY * RICE_SPACING**3,
            jitter=0.2 * RICE_SPACING,
            radius_mean=radius,
            custom_attributes={"mpm:friction": RICE_FRICTION},
        )
        self.rice_end = builder.particle_count
        self.cup_in_grip = []
        self.finger_load = {"full": [], "empty": []}
        self.peak_grain = 0.0

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        # Draw the rice in its own colour instead of the viewer's default particle colour.
        self.viewer.show_particles = False
        count = self.rice_end - self.rice_begin
        self.rice_radii = wp.full(count, 0.5 * RICE_SPACING, dtype=float, device=self.model.device)
        self.rice_colors = wp.full(count, wp.vec3(0.95, 0.93, 0.85), dtype=wp.vec3, device=self.model.device)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_points(
            "/rice", self.state_0.particle_q[self.rice_begin : self.rice_end], self.rice_radii, self.rice_colors
        )
        self.viewer.end_frame()

    def mpm_option_overrides(self):
        # Granular flow pours on a 1.5 cm grid; finer grids with one-cell walls
        # eject grains once the stream leaves the rim.
        return {"max_active_cell_count": 4096, "air_drag": 1.0}

    def configure_model(self, model):
        shape_body = model.shape_body.numpy()
        labels = [model.body_label[b].rsplit("/", 1)[-1] if b >= 0 else "" for b in shape_body]
        self.finger_bodies = sorted(
            {int(b) for b, label in zip(shape_body, labels, strict=True) if "right_finger" in label}
        )

    def _keyframes(self):
        g = self.grasp
        behind = g + np.array((-0.13, 0.0, 0.0))
        lifted = g + np.array((0.0, 0.0, LIFT))
        over_cooker = g + np.array((0.0, CARRY, LIFT))
        # (time, right TCP target, right roll, right finger)
        return [
            (0.0, self.home[1], 0.0, FINGER_OPEN),
            (0.5, self.home[1], 0.0, FINGER_OPEN),
            (1.5, behind, 0.0, FINGER_OPEN),
            (2.8, g, 0.0, FINGER_OPEN),
            (3.8, g, 0.0, FINGER_SHUT),
            (4.3, g, 0.0, FINGER_SHUT),
            (5.8, lifted, 0.0, FINGER_SHUT),
            (7.0, over_cooker, 0.0, FINGER_SHUT),
            (8.5, over_cooker, POUR_ROLL, FINGER_SHUT),
            (10.0, over_cooker, POUR_ROLL, FINGER_SHUT),
            # Turn upright slowly: the handle's torsional friction holds the
            # cup's inertia only at moderate roll rates.
            (11.5, over_cooker, 0.0, FINGER_SHUT),
            (12.5, over_cooker, 0.0, FINGER_SHUT),
            (13.5, lifted, 0.0, FINGER_SHUT),
            (14.5, g, 0.0, FINGER_SHUT),
            (15.3, g, 0.0, FINGER_OPEN),
            (16.3, behind, 0.0, FINGER_OPEN),
        ]

    def _sample(self, time):
        target, roll, finger = smoothstep(self._keyframes(), time)
        return np.asarray(target), float(roll), float(finger)

    def plan(self, time):
        target, _roll_angle, finger = self._sample(time)
        return np.array((self.home[0], target)), np.array((FINGER_OPEN, finger))

    def gripper_rotations(self, time):
        _target, roll, _finger = self._sample(time)
        return self.gripper_rotation, _roll(roll)

    def tracks_freely(self, time):
        return time <= 2.8 or time >= 16.3

    def _cup_frame(self):
        q = self.state_0.body_q.numpy()[self.cup]
        return wp.transform(wp.vec3(*q[:3]), wp.quat(*q[3:]))

    def rice_locations(self):
        """Fractions of the rice in the cup, in the pot, and elsewhere."""
        x = self.state_0.particle_q.numpy()[self.rice_begin : self.rice_end]
        frame = wp.transform_inverse(self._cup_frame())
        local = np.array([wp.transform_point(frame, wp.vec3(*p)) for p in x])
        hx, hy, _ = CUP_OUTER / 2
        in_cup = (
            (np.abs(local[:, 0]) < hx)
            & (np.abs(local[:, 1]) < hy)
            & (local[:, 2] > -0.005)
            & (local[:, 2] < CUP_OUTER[2] + 0.01)
        )
        in_pot = (np.linalg.norm(x[:, :2] - COOKER_CENTER, axis=1) < COOKER_INNER_RADIUS) & (
            x[:, 2] < TABLE_Z + COOKER_HEIGHT + 0.01
        )
        return float(in_cup.mean()), float((in_pot & ~in_cup).mean()), float((~in_cup & ~in_pot).mean())

    def after_step(self):
        if not self.args.test and self.frame % 60:
            return
        qd = self.state_0.particle_qd.numpy()[self.rice_begin : self.rice_end]
        self.peak_grain = max(self.peak_grain, float(np.abs(qd).max()))
        if CARRY_PHASE[0] <= self.sim_time <= CARRY_PHASE[1]:
            # Grasped handle point and cup orientation in the right gripper's frame.
            body_q = self.state_0.body_q.numpy()
            gripper = wp.transform(*body_q[self.scene_ee[1]])
            relative = wp.transform_multiply(wp.transform_inverse(gripper), self._cup_frame())
            handle = wp.transform_point(relative, wp.vec3(*HANDLE_CENTER))
            self.cup_in_grip.append(np.array((*handle, *wp.transform_get_rotation(relative))))
        for name, span in (("full", FULL_HOLD), ("empty", EMPTY_HOLD)):
            if span[0] <= self.sim_time <= span[1]:
                # Upward force the fingers exert on the cup (the cup pushes the
                # finger links down by the same amount).
                impulses = self.solver.environment.rigid_impulses.numpy()[self.finger_bodies]
                self.finger_load[name].append(-float(impulses[:, 2].sum()) / self.sim_dt)
        if self.frame % 60 == 0:
            cup, pot, other = self.rice_locations()
            print(
                f"[W1RiceCooker] t={self.sim_time:4.1f} rice in cup {cup:.0%}, in pot {pot:.0%}, elsewhere {other:.0%}, "
                f"finger_q[mm]={np.round(self.finger_q() * 1e3, 1).tolist()}",
                flush=True,
            )

    def test_post_step(self):
        self.check_finite()
        if self.args.coupling == "two_way" and self.peak_grain > 5.0:
            raise AssertionError(f"Rice grains were ejected at {self.peak_grain:.1f} m/s")

    def test_final(self):
        self.test_post_step()
        self.solver.check_capacity_status()
        if self.args.coupling != "two_way":
            return
        if self.frame < 990:
            raise AssertionError("Run at least 990 frames to cover the pour and release")
        poses = np.asarray(self.cup_in_grip)
        slip = float(np.max(np.linalg.norm(poses[:, :3] - poses[0, :3], axis=1)))
        turn = float(np.max(2.0 * np.arccos(np.clip(np.abs(poses[:, 3:] @ poses[0, 3:]), 0.0, 1.0))))
        if slip > MAX_SLIP or turn > MAX_TURN:
            raise AssertionError(f"The cup moved in the grip: {slip * 1e3:.1f} mm, {np.rad2deg(turn):.1f} deg")
        cup, pot, other = self.rice_locations()
        if pot < 0.8 or other > 0.15:
            raise AssertionError(
                f"The rice was not poured into the pot: cup {cup:.0%}, pot {pot:.0%}, spilled {other:.0%}"
            )
        # The fingers carry the cup, plus the rice before the pour.
        full, empty = np.mean(self.finger_load["full"]), np.mean(self.finger_load["empty"])
        rice_weight = float(self.model.particle_mass.numpy()[self.rice_begin : self.rice_end].sum()) * 9.81
        if abs((full - empty) - rice_weight) > 0.3 * rice_weight:
            raise AssertionError(
                f"The finger load did not carry the rice weight ({rice_weight:.2f} N): full {full:.2f} N, empty {empty:.2f} N"
            )
        q = self.state_0.body_q.numpy()[self.cup]
        up = np.asarray(wp.quat_rotate(wp.quat(*q[3:]), wp.vec3(0.0, 0.0, 1.0)))
        if up[2] < np.cos(np.deg2rad(10.0)) or abs(q[2] - TABLE_Z) > 0.01:
            raise AssertionError(
                f"The empty cup is not back upright on the table: up {up}, height {q[2] - TABLE_Z:.3f} m"
            )
        if self.peak_free_tcp_error > 0.01:
            raise AssertionError(f"TCP error in free motion exceeded 1 cm: {self.peak_free_tcp_error}")
        print(
            f"[W1RiceCooker] PASS: cup slip {slip * 1e3:.2f} mm / {np.rad2deg(turn):.2f} deg in the grip; "
            f"rice in pot {pot:.0%}, left in cup {cup:.0%}, spilled {other:.0%}; "
            f"fingers carry {full:.2f} N full vs {empty:.2f} N empty (rice weighs {rice_weight:.2f} N); "
            f"peak grain {self.peak_grain:.2f} m/s.",
            flush=True,
        )

    @staticmethod
    def create_parser():
        parser = W1MJMPMScene.add_arguments(newton.examples.create_parser(), num_frames=990)
        # The held cup must not move inside the rice solve, or the rice walks
        # through its bottom; its reaction reaches the cup before the next
        # contact solve instead, so the grip holds against the rice's weight.
        parser.set_defaults(
            substeps=6,
            voxel_size=0.015,
            strain_basis="P0",
            mpm_solver="gs",
            iterations=50,
            stall_tolerance=0.0,
            rigid_material_coupling="lagged",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

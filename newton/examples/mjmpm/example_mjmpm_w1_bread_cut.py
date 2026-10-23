# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""W1 holds a mounted knife and cuts an elastoplastic MPM loaf.

    uv run --no-sync -m newton.examples mjmpm_w1_bread_cut

Material setup builds on WAIC's mpm_blade_cut example, adding a wedge blade
and cuttable elastoplastic particle connections. This experimental hybrid
uses MPM for bulk response/contact and implicit connections for fragment cohesion.
The knife is rigidly mounted to the dynamic wrist, including its inertia.
Its pose is never scripted independently of the robot. Material loads feed
back to MuJoCo through the wrist's articulated mobility.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mjmpm.support.bread_cut_cohesion import CuttableCohesion, count_detached_samples
from newton.examples.mjmpm.support.bread_cut_geometry import HANDLE_GRIP_OPENING, HANDLE_HALF_WIDTH, blade_mesh
from newton.examples.mjmpm.support.bread_cut_material import apply_damage, damage_at_blade
from newton.examples.mjmpm.support.bread_cut_visuals import BreadCutVisuals
from newton.examples.mjmpm.support.w1_two_way import (
    FINGER_OPEN,
    TABLE_Z,
    W1MJMPMScene,
    smoothstep,
)

BOARD_TOP = TABLE_Z + 0.018
LOAF_CENTER = np.array((0.54, -0.24, BOARD_TOP))
CUT_Y = -0.275
HANDLE = np.array((0.39, -0.43, BOARD_TOP + 0.13))
SPACING = 0.005
HANDLE_GRIP = HANDLE_GRIP_OPENING


def loaf_contains(points):
    """Rounded loaf with a flat underside, in loaf-local metres."""
    x, y, z = np.asarray(points).T
    end = np.sqrt(np.clip(1.0 - (np.maximum(np.abs(y) - 0.060, 0.0) / 0.040) ** 2, 0.0, 1.0))
    return (z >= 0.002) & ((x / 0.050) ** 2 + ((z - 0.027) / 0.038) ** 2 < end**2)


class Example(W1MJMPMScene):
    camera_pos = wp.vec3(0.90, -0.50, 1.18)
    camera_pitch = -30.5
    camera_yaw = 144.16
    home_offset = np.zeros(3)

    def initial_targets(self):
        return np.array(((0.40, 0.24, TABLE_Z + 0.18), HANDLE))

    def build_task(self, builder):
        visible = self.args.visual_style == "particles"
        self.add_box(
            builder,
            "chopping_board",
            (0.54, -0.24, TABLE_Z + 0.009),
            (0.16, 0.16, 0.009),
            (0.45, 0.27, 0.12),
            cfg=newton.ModelBuilder.ShapeConfig(ke=2.0e4, kd=100.0, mu=0.8, is_visible=visible),
        )
        self.knife = self.scene_ee[1]
        state = self.ik_model.state()
        newton.eval_fk(self.ik_model, self.ik_model.joint_q, self.ik_model.joint_qd, state)
        poses = state.body_q.numpy()
        link = self.ik_model.body_label.index(builder.body_label[self.knife])
        rotation = wp.mul(wp.quat_inverse(wp.quat(*poses[link, 3:])), wp.quat(*poses[self.ee[1], 3:]))
        rotation = wp.mul(rotation, wp.quat_inverse(wp.quat(*self.gripper_rotation)))
        self.knife_local = wp.transform(self.scene_tcp[1], rotation)
        robot_shapes = builder.shape_count
        handle_cfg = newton.ModelBuilder.ShapeConfig(density=700.0, ke=1.0e5, kd=100.0, mu=2.0, is_visible=visible)
        builder.add_shape_box(
            self.knife,
            xform=self.knife_local,
            hx=0.048,
            hy=HANDLE_HALF_WIDTH,
            hz=0.012,
            cfg=handle_cfg,
            color=(0.08, 0.05, 0.03),
            label="knife_handle",
        )
        # A resolved 12 mm spine opens the cut; its 1 mm edge avoids the
        # broad compressive stamp made by the original WAIC box blade.
        blade_cfg = newton.ModelBuilder.ShapeConfig(density=2800.0, ke=1.0e5, kd=100.0, mu=0.03, is_visible=visible)
        vertices, faces = blade_mesh()
        builder.add_shape_mesh(
            self.knife,
            xform=self.knife_local,
            mesh=newton.Mesh(vertices, faces.ravel()),
            cfg=blade_cfg,
            color=(0.75, 0.78, 0.81),
            label="knife_blade",
        )
        self.knife_shapes = (robot_shapes, robot_shapes + 1)
        for tool in self.knife_shapes:
            for robot in range(robot_shapes):
                if builder.shape_body[robot] >= 0:
                    builder.add_shape_collision_filter_pair(tool, robot)
        local = np.stack(
            np.meshgrid(
                np.arange(-0.0475, 0.05, SPACING),
                np.arange(-0.0925, 0.095, SPACING),
                np.arange(0.0045, 0.066, SPACING),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)
        local = local[loaf_contains(local)]
        self.rest = local + LOAF_CENTER
        for point in self.rest:
            builder.add_particle(
                wp.vec3(*point),
                wp.vec3(0.0),
                280.0 * SPACING**3,
                0.5 * SPACING,
                custom_attributes={
                    "mpm:young_modulus": self.args.young_modulus,
                    "mpm:poisson_ratio": 0.25,
                    "mpm:damping": 0.015,
                    "mpm:yield_pressure": self.args.yield_pressure,
                    "mpm:tensile_yield_ratio": self.args.tensile_yield_ratio,
                    "mpm:yield_stress": self.args.yield_stress,
                    "mpm:friction": 0.4,
                    "mpm:hardening": 0.0,
                    "mpm:dilatancy": 0.0,
                },
            )
        self.heights = []
        self.metrics = []
        print(
            f"[BreadCut] {len(self.rest)} MPM particles; dynamic wrist-mounted knife; WAIC plastic-separation material.",
            flush=True,
        )

    def mpm_option_overrides(self):
        return {"max_active_cell_count": 8192, "collider_basis": "pic128", "air_drag": 0.0, "transfer_scheme": "pic"}

    def _keyframes(self):
        left, grasp = self.initial_targets()
        above = grasp.copy()
        above[1] = CUT_Y
        cut = above + np.array((0.0, 0.0, -0.073))
        separated = cut + np.array((0.0, -0.018, 0.0))
        # A small sawing stroke keeps the blade from only compressing the loaf.
        return [
            (0.0, self.home, (FINGER_OPEN, FINGER_OPEN)),
            (1.3, np.array((left, grasp)), (FINGER_OPEN, FINGER_OPEN)),
            (2.2, np.array((left, grasp)), (FINGER_OPEN, HANDLE_GRIP)),
            (2.7, np.array((left, above)), (FINGER_OPEN, HANDLE_GRIP)),
            (4.5, np.array((left, cut + np.array((0.015, 0.0, 0.035)))), (FINGER_OPEN, HANDLE_GRIP)),
            (5.7, np.array((left, cut + np.array((-0.010, 0.0, 0.015)))), (FINGER_OPEN, HANDLE_GRIP)),
            (6.8, np.array((left, cut)), (FINGER_OPEN, HANDLE_GRIP)),
            (7.2, np.array((left, cut)), (FINGER_OPEN, HANDLE_GRIP)),
            (8.2, np.array((left, separated)), (FINGER_OPEN, HANDLE_GRIP)),
            (9.6, np.array((left, separated + np.array((0.0, 0.0, 0.133)))), (FINGER_OPEN, HANDLE_GRIP)),
            (11.0, np.array((left, separated + np.array((0.0, 0.0, 0.133)))), (FINGER_OPEN, HANDLE_GRIP)),
        ]

    def plan(self, time):
        return smoothstep(self._keyframes(), time)

    def tracks_freely(self, time):
        return False

    def __init__(self, viewer, args):
        if args.young_modulus <= 0 or args.yield_pressure <= 0 or args.yield_stress < 0:
            raise ValueError("Elastic modulus and yield pressure must be positive; shear strength must be nonnegative")
        if not 0 <= args.tensile_yield_ratio <= 1:
            raise ValueError("Tensile yield ratio must be between zero and one")
        if not (0 < args.cut_plastic_yield < 1 and 0 <= args.cut_plastic_limit < 1):
            raise ValueError("Cut plastic yield must be in (0, 1), and the shortening limit in [0, 1)")
        super().__init__(viewer, args)
        self.peak_speed = 0.0
        self.initial_mass = float(self.model.particle_mass.numpy().sum())
        self.minimum_grip_clearance = np.inf
        self.finger_visual_overhang, self.handle_visual_width = self._measure_grip_geometry()
        self.viewer.show_particles = False
        colors = np.tile((0.89, 0.78, 0.55), (len(self.rest), 1))
        local = self.rest - LOAF_CENTER
        radial = np.sqrt((local[:, 0] / 0.050) ** 2 + ((local[:, 2] - 0.027) / 0.038) ** 2)
        # Resolve a warm crust with several samples so lattice rows do not
        # expose alternating pale stripes. Colour never refers to CUT_Y.
        outside = (radial > 0.72) | (np.abs(local[:, 1]) > 0.079) | (local[:, 2] < 0.012)
        colors[outside] = (0.49, 0.24, 0.07)
        colors *= np.random.default_rng(57).uniform(0.94, 1.04, (len(colors), 1))
        self.colors = wp.array(colors, dtype=wp.vec3, device=self.model.device)
        self.radii = wp.full(len(self.rest), 0.57 * SPACING, dtype=float, device=self.model.device)
        self.damage = wp.zeros(len(self.rest), dtype=float, device=self.model.device)
        tool_pose = wp.mul(wp.transform(*self.state_0.body_q.numpy()[self.knife]), self.knife_local)
        self.cohesion = CuttableCohesion(
            self.rest,
            mass=float(self.model.particle_mass.numpy()[0]),
            spacing=SPACING,
            device=self.model.device,
            tool_pose=tool_pose,
            plastic_zone=self.damage,
            plastic_yield=args.cut_plastic_yield,
            plastic_limit=args.cut_plastic_limit,
        )
        self.material_arrays = [(self.model.mpm.yield_stress, self.model.mpm.tensile_yield_ratio)]
        material = self.solver.environment.model.mpm
        if material.yield_stress.ptr != self.model.mpm.yield_stress.ptr:
            self.material_arrays.append((material.yield_stress, material.tensile_yield_ratio))
        self.visuals = None
        if args.visual_style == "realistic":
            self.visuals = BreadCutVisuals(
                viewer,
                device=self.model.device,
                rest=self.rest,
                center=LOAF_CENTER,
                cut_y=CUT_Y,
                tool_body=self.knife,
                tool_local=self.knife_local,
                render_bread=args.bread_render == "surface",
            )

    def _measure_grip_geometry(self):
        """Calibrate pad/handle clearance from the actual visual meshes."""
        poses = self.state_0.body_q.numpy()
        inv_tool = wp.transform_inverse(wp.mul(wp.transform(*poses[self.knife]), self.knife_local))
        fingers = self.finger_pairs()[1]
        overhang = 0.0
        for shape, body in enumerate(self.model.shape_body.numpy()):
            if body < 0:
                continue
            name = self.model.body_label[body].lower()
            if not (name.endswith("/right_finger1") or name.endswith("/right_finger2")):
                continue
            source = self.model.shape_source[shape]
            if source is None or not hasattr(source, "vertices"):
                continue
            transform = wp.mul(
                inv_tool, wp.mul(wp.transform(*poses[body]), wp.transform(*self.model.shape_transform.numpy()[shape]))
            )
            rotation = np.array(wp.quat_to_matrix(wp.transform_get_rotation(transform))).reshape(3, 3)
            points = np.asarray(source.vertices) * self.model.shape_scale.numpy()[shape]
            points = points @ rotation.T + np.array(wp.transform_get_translation(transform))
            grip = (np.abs(points[:, 0]) < 0.049) & (np.abs(points[:, 2]) < 0.013)
            if np.any(grip):
                side = 0 if name.endswith("finger1") else 1
                overhang = max(overhang, fingers[side] - float(np.min(np.abs(points[grip, 1]))))
        width = HANDLE_HALF_WIDTH
        archive_path = Path(__file__).resolve().parent / "assets/bread_cut/visuals.npz"
        with np.load(archive_path) as archive:
            for part in json.loads(str(archive["manifest"])):
                if part["parent"] != "knife":
                    continue
                points = archive[part["key"] + "_points"]
                grip = (points[:, 0] > -0.049) & (points[:, 0] < 0.020) & (np.abs(points[:, 2]) < 0.013)
                if np.any(grip):
                    width = max(width, float(np.abs(points[grip, 1]).max()))
        print(
            f"[BreadCut] grip geometry: pad overhang {overhang * 1000:.3f} mm, handle/rivet half-width {width * 1000:.3f} mm",
            flush=True,
        )
        return float(overhang), float(width)

    def _simulate_frame(self):
        self.cohesion.cut(self.state_0, self.knife, self.knife_local)
        if self.args.blade_damage:
            wp.launch(
                damage_at_blade,
                dim=len(self.rest),
                inputs=[
                    self.state_0.particle_q,
                    self.state_0.body_q,
                    self.knife,
                    self.knife_local,
                    0.0055,
                    self.damage,
                ],
                device=self.model.device,
            )
            for shear, tension in self.material_arrays:
                wp.launch(
                    apply_damage,
                    dim=len(self.rest),
                    inputs=[self.damage, self.args.yield_stress, self.args.tensile_yield_ratio, shear, tension],
                    device=self.model.device,
                )
        super()._simulate_frame()

    def before_substep(self):
        self.cohesion.apply(self.state_0, self.sim_dt)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.visuals is not None:
            self.visuals.render(self.state_0, self.damage)
        if self.visuals is None or self.args.bread_render == "particles":
            self.viewer.log_points("/bread/material", self.state_0.particle_q, self.radii, self.colors)
        self.viewer.end_frame()

    def after_step(self):
        if self.frame % 30:
            return
        q = self.state_0.particle_q.numpy()
        if self.frame == 180:
            self.pre_cut_positions = q.copy()
        knife = np.array(wp.mul(wp.transform(*self.state_0.body_q.numpy()[self.knife]), self.knife_local))
        height = float(np.percentile(q[:, 2], 98) - BOARD_TOP)
        self.heights.append((self.sim_time, height))
        near = np.abs(self.rest[:, 1] - CUT_Y) < 0.015
        sides = np.where(self.rest[:, 1] < CUT_Y, -1.0, 1.0)
        opening = float(np.mean((q[near, 1] - self.rest[near, 1]) * sides[near]) * 2)
        surface = float(np.mean(q[near & (self.rest[:, 2] > BOARD_TOP + 0.045), 2]) - BOARD_TOP)
        strain = self.state_0.mpm.particle_elastic_strain.numpy()
        elastic = float(np.mean(np.linalg.norm(strain - np.eye(3), axis=(1, 2))))
        self.metrics.append((self.sim_time, height, opening, float(knife[2]), surface, elastic))
        if self.sim_time >= 9.5:
            affected = (self.damage.numpy() > 0) & (self.rest[:, 2] > BOARD_TOP + 0.045)
            notch = float(np.mean(self.pre_cut_positions[affected, 2] - q[affected, 2])) if np.any(affected) else 0.0
            plastic = 1 - self.cohesion.lengths.numpy() / self.cohesion.initial_lengths.numpy()
            print(
                f"[BreadCut] retained notch {notch * 1000:.2f} mm; maximum connection plastic shortening {plastic.max() * 100:.2f}%",
                flush=True,
            )
        if self.frame % 60 == 0:
            _, sizes = self.cohesion.components()
            print(
                f"[BreadCut] connected pieces: {sorted(sizes.tolist(), reverse=True)[:12]}; total={len(sizes)}",
                flush=True,
            )
        print(
            f"[BreadCut] t={self.sim_time:.1f} height={height * 1000:.1f}mm opening={opening * 1000:.1f}mm surface={surface * 1000:.1f}mm elastic={elastic:.4f} knife_z={knife[2] - BOARD_TOP:.3f}m peak_v={np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1).max():.2f}m/s",
            flush=True,
        )

    def _check_material_and_grip(self):
        self.check_finite()
        speed = float(np.linalg.norm(self.state_0.particle_qd.numpy(), axis=1).max())
        self.peak_speed = max(self.peak_speed, speed)
        if speed > 3.0:
            raise AssertionError(f"Bread became unstable: {speed:.3f} m/s")
        clearance = float(self.finger_pairs()[1].min()) - self.finger_visual_overhang - self.handle_visual_width
        self.minimum_grip_clearance = min(self.minimum_grip_clearance, clearance)
        if clearance < -0.000001:
            raise AssertionError(f"Fingers intersect the knife handle/rivets: {clearance * 1000:.3f} mm clearance")

    def test_post_step(self):
        self._check_material_and_grip()
        if self.sim_time >= 7.0:
            labels, sizes = self.cohesion.components()
            detached = count_detached_samples(self.state_0.particle_q.numpy(), labels, SPACING)
            if len(sizes) != 2 or detached:
                raise AssertionError(f"A cut piece lost coherence: sizes={sizes.tolist()}, detached={detached}")

    def _check_final_state(self, required_frames):
        self.check_finite()
        self.solver.check_capacity_status()
        if self.frame < required_frames:
            raise AssertionError(f"Run {required_frames} frames to cover cutting and withdrawal")
        if np.min(self.state_0.particle_q.numpy()[:, 2]) < BOARD_TOP - 0.006:
            raise AssertionError("Bread penetrated the chopping board")
        if self.knife not in self.solver.ownership.mujoco_bodies:
            raise AssertionError("The dynamic MuJoCo wrist must own the tool")
        if (
            self.model.particle_count != len(self.rest)
            or float(self.model.particle_mass.numpy().sum()) != self.initial_mass
        ):
            raise AssertionError("Cutting must preserve every material sample and its mass")

    def test_final(self):
        self._check_final_state(660)
        metrics = np.array(self.metrics)
        before = metrics[(metrics[:, 0] >= 2.0) & (metrics[:, 0] <= 3.0)]
        loaded = metrics[(metrics[:, 0] >= 7.8) & (metrics[:, 0] <= 8.6)]
        after = metrics[metrics[:, 0] >= 10.0]
        compression = float(before[:, 4].mean() - loaded[:, 4].min())
        permanent = float(after[:, 2].mean())
        recoil = float(loaded[:, 2].max() - permanent)
        labels, sizes = self.cohesion.components()
        detached = count_detached_samples(self.state_0.particle_q.numpy(), labels, SPACING)
        if len(sizes) != 2 or sizes.min() < 0.1 * len(self.rest) or detached != 0:
            raise AssertionError(
                f"Cut must produce two coherent pieces without loose samples: sizes={sizes.tolist()}, detached={detached}"
            )
        if permanent < 0.002 or recoil < 0.0005:
            raise AssertionError(
                f"Insufficient cutting response: indentation={compression * 1000:.2f} mm, retained opening={permanent * 1000:.2f} mm, lateral recoil={recoil * 1000:.2f} mm"
            )
        affected = (self.damage.numpy() > 0) & (self.rest[:, 2] > BOARD_TOP + 0.045)
        self.retained_notch = (
            float(np.mean(self.pre_cut_positions[affected, 2] - self.state_0.particle_q.numpy()[affected, 2]))
            if np.any(affected)
            else 0.0
        )
        if self.args.blade_damage and self.retained_notch < 0.0015:
            raise AssertionError(f"The cut recovered completely: retained notch {self.retained_notch * 1000:.3f} mm")
        self.validation = {
            "fragment_particle_counts": sorted(sizes.tolist()),
            "detached_samples": detached,
            "particle_count": self.model.particle_count,
            "retained_opening_mm": permanent * 1000,
            "lateral_recoil_mm": recoil * 1000,
            "retained_notch_mm": self.retained_notch * 1000,
            "minimum_visual_grip_clearance_mm": float(self.minimum_grip_clearance * 1000),
            "peak_particle_speed_m_s": self.peak_speed,
        }
        print(
            f"[BreadCut] PASS: two coherent pieces {sizes.tolist()}, detached samples {detached}; retained opening {permanent * 1000:.2f} mm, lateral elastic recoil {recoil * 1000:.2f} mm, retained notch {self.retained_notch * 1000:.2f} mm; minimum visual grip clearance {self.minimum_grip_clearance * 1000:.3f} mm; {self.model.particle_count} particles conserved; peak speed {self.peak_speed:.2f} m/s.",
            flush=True,
        )

    @staticmethod
    def create_parser():
        parser = W1MJMPMScene.add_arguments(newton.examples.create_parser(), num_frames=660)
        parser.add_argument("--young-modulus", type=float, default=4.0e3)
        parser.add_argument("--yield-pressure", type=float, default=5.0e3)
        parser.add_argument("--tensile-yield-ratio", type=float, default=0.3)
        parser.add_argument("--yield-stress", type=float, default=3000.0)
        parser.add_argument(
            "--cut-plastic-yield", type=float, default=0.005, help="Elastic compression threshold in the cutting zone."
        )
        parser.add_argument(
            "--cut-plastic-limit",
            type=float,
            default=0.60,
            help="Maximum permanent connection shortening near the cut.",
        )
        parser.add_argument(
            "--blade-damage",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Weaken MPM rheology at the cut; the elastic network retains cohesion inside each piece.",
        )
        parser.add_argument("--visual-style", choices=("realistic", "particles"), default="realistic")
        parser.add_argument(
            "--bread-render",
            choices=("particles", "surface"),
            default="particles",
            help="Render physical material samples (default), or the optional predefined surface patches.",
        )
        parser.set_defaults(
            substeps=6,
            rigid_iterations=60,
            finger_kp=2500.0,
            voxel_size=0.006,
            strain_basis="P0",
            mpm_solver="gs",
            iterations=50,
            stall_tolerance=0.0,
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

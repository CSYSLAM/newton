# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Cut opposite sides of an MPM loaf in two strokes, leaving three pieces.

    uv run --no-sync -m newton.examples mjmpm_w1_bread_three_pieces

Reuse the single-cut scene's assets, material and dynamic wrist-mounted tool.
The second cut follows withdrawal and a clearance-height traverse. Connections
are severed by the actual blade sweep; the loaf starts as one connected body.
"""

import numpy as np
import warp as wp

import newton.examples
from newton.examples.mjmpm.example_mjmpm_w1_bread_cut import (
    BOARD_TOP,
    CUT_Y,
    HANDLE_GRIP,
    SPACING,
)
from newton.examples.mjmpm.example_mjmpm_w1_bread_cut import (
    Example as SingleCutExample,
)
from newton.examples.mjmpm.support.bread_cut_cohesion import count_detached_samples
from newton.examples.mjmpm.support.bread_cut_visuals import limit_particle_radii_at_blade
from newton.examples.mjmpm.support.w1_two_way import FINGER_OPEN

SECOND_CUT_Y = -0.205
TOTAL_FRAMES = 1200


class Example(SingleCutExample):
    metric_columns = ("time_s", "cut_index", "opening_m", "cut_surface_z_m", "knife_z_m")
    cut_separation = 0.010
    saw_forward = 0.008
    saw_backward = -0.005

    def initial_targets(self):
        targets = super().initial_targets()
        # Fit the loaf and the sawing stroke on the straight blade segment.
        targets[1, 0] += 0.0225
        return targets

    def configure_model(self, model):
        super().configure_model(model)
        margin = model.shape_margin.numpy()
        # A small collision skin reduces subgrid contact error at the blade.
        margin[self.knife_shapes[1]] = 0.0015
        model.shape_margin.assign(margin)

    def __init__(self, viewer, args):
        if args.bread_render != "particles":
            raise ValueError(
                "Two cuts require particle rendering; the optional single-cut skin cannot represent three pieces"
            )
        super().__init__(viewer, args)
        self.blade_mesh = self.model.shape_source[self.knife_shapes[1]].finalize(self.model.device)
        self.blade_clearances = wp.empty(len(self.rest), dtype=float, device=self.model.device)
        self.minimum_blade_clearance = np.inf
        self._blade_display_frame = -1
        self.cut_history = [[], []]
        self.observed_piece_counts = set()

    def _keyframes(self):
        # Reuse the first stroke's timings and sawing pattern.
        frames = super()._keyframes()[:-1]
        fingers = (FINGER_OPEN, HANDLE_GRIP)
        release = np.array((0.0, 0.004, 0.0))
        # Withdraw into the opened kerf instead of dragging the pushed slice.
        first_release = frames[-2][1].copy()
        first_release[1] += release + np.array((0.0, 0.0, 0.030))
        frames[-1][1][1] += release
        frames.insert(-1, (8.6, first_release, fingers))
        left, grasp = self.initial_targets()
        above = grasp.copy()
        above[1] = SECOND_CUT_Y
        cut = above + np.array((0.0, 0.0, -0.073))
        separated = cut + np.array((0.0, self.cut_separation, 0.0))
        frames.extend(
            [
                (10.3, frames[-1][1].copy(), fingers),
                (11.6, np.array((left, above)), fingers),
                (13.4, np.array((left, cut + np.array((self.saw_forward, 0.0, 0.035)))), fingers),
                (14.6, np.array((left, cut + np.array((self.saw_backward, 0.0, 0.015)))), fingers),
                (15.7, np.array((left, cut)), fingers),
                (16.1, np.array((left, cut)), fingers),
                (17.1, np.array((left, separated)), fingers),
                (17.5, np.array((left, separated - release + np.array((0.0, 0.0, 0.030)))), fingers),
                (18.5, np.array((left, separated - release + np.array((0.0, 0.0, 0.133)))), fingers),
                (20.0, np.array((left, separated - release + np.array((0.0, 0.0, 0.133)))), fingers),
            ]
        )
        return frames

    def _update_blade_display(self):
        if self._blade_display_frame == self.frame:
            return
        wp.launch(
            limit_particle_radii_at_blade,
            dim=len(self.rest),
            inputs=[
                self.state_0.particle_q,
                self.state_0.body_q,
                self.knife,
                self.knife_local,
                self.blade_mesh,
                0.57 * SPACING,
            ],
            outputs=[self.radii, self.blade_clearances],
            device=self.model.device,
        )
        self._blade_display_frame = self.frame

    def render(self):
        self._update_blade_display()
        super().render()

    def after_step(self):
        if self.frame % 30:
            return
        q = self.state_0.particle_q.numpy()
        if self.frame == 180:
            self.pre_cut_positions = q.copy()
        knife = wp.mul(wp.transform(*self.state_0.body_q.numpy()[self.knife]), self.knife_local)
        for index, cut_y in enumerate((CUT_Y, SECOND_CUT_Y)):
            near = np.abs(self.rest[:, 1] - cut_y) < 0.015
            sides = np.where(self.rest[:, 1] < cut_y, -1.0, 1.0)
            opening = float(np.mean((q[near, 1] - self.rest[near, 1]) * sides[near]) * 2)
            surface = float(np.mean(q[near & (self.rest[:, 2] > BOARD_TOP + 0.045), 2]))
            self.cut_history[index].append((self.sim_time, opening, surface))
            self.metrics.append((self.sim_time, index + 1, opening, surface, float(knife[2])))
        if self.frame % 60 == 0:
            _, sizes = self.cohesion.components()
            print(f"[BreadThreePieces] t={self.sim_time:.1f}s pieces={sorted(sizes.tolist())}", flush=True)

    def test_post_step(self):
        self._check_material_and_grip()
        self._update_blade_display()
        clearance = float(self.blade_clearances.numpy().min())
        self.minimum_blade_clearance = min(self.minimum_blade_clearance, clearance)
        if clearance < -0.0001:
            raise AssertionError(f"Bread penetrated the knife at {self.sim_time:.3f}s: {clearance * 1000:.3f} mm")
        labels, sizes = self.cohesion.components()
        self.observed_piece_counts.add(len(sizes))
        expected = (
            1 if self.sim_time < 3.5 else 2 if 7.0 <= self.sim_time < 12.0 else 3 if self.sim_time >= 16.0 else None
        )
        if expected is not None and len(sizes) != expected:
            raise AssertionError(f"Expected {expected} pieces at {self.sim_time:.3f}s, got {sizes.tolist()}")
        if len(sizes) > 3 or sizes.min() < 0.1 * len(self.rest):
            raise AssertionError(f"Unexpected fragment or loose sample: {sizes.tolist()}")
        if self.sim_time >= 7.0:
            detached = count_detached_samples(self.state_0.particle_q.numpy(), labels, SPACING)
            if detached:
                raise AssertionError(f"A bread piece shed {detached} samples")

    def test_final(self):
        self._check_final_state(TOTAL_FRAMES)
        labels, sizes = self.cohesion.components()
        q = self.state_0.particle_q.numpy()
        detached = count_detached_samples(q, labels, SPACING)
        if len(sizes) != 3 or sizes.min() < 0.1 * len(self.rest) or detached:
            raise AssertionError(f"Expected three coherent pieces: sizes={sizes.tolist()}, detached={detached}")
        if self.observed_piece_counts != {1, 2, 3}:
            raise AssertionError(f"Missing sequential cutting stages: {self.observed_piece_counts}")
        cuts = []
        damage = self.damage.numpy()
        for index, (cut_y, load_start, unload_start) in enumerate(((CUT_Y, 7.8, 10.0), (SECOND_CUT_Y, 16.7, 18.9))):
            history = np.asarray(self.cut_history[index])
            loaded = history[(history[:, 0] >= load_start) & (history[:, 0] <= load_start + 0.8), 1]
            unloaded = history[(history[:, 0] >= unload_start) & (history[:, 0] <= unload_start + 1.0), 1]
            opening = float(unloaded.mean())
            recoil = float(loaded.max() - opening)
            affected = (damage > 0) & (np.abs(self.rest[:, 1] - cut_y) < 0.020) & (self.rest[:, 2] > BOARD_TOP + 0.045)
            notch = float(np.mean(self.pre_cut_positions[affected, 2] - q[affected, 2])) if np.any(affected) else 0.0
            if opening < 0.002 or recoil < 0.0005:
                raise AssertionError(
                    f"Cut {index + 1} lost opening/recoil: {opening * 1000:.3f}/{recoil * 1000:.3f} mm"
                )
            if self.args.blade_damage and notch < 0.003:
                raise AssertionError(f"Cut {index + 1} retained only {notch * 1000:.3f} mm indentation; require 3 mm")
            cuts.append({"opening_mm": opening * 1000, "recoil_mm": recoil * 1000, "notch_mm": notch * 1000})
        self.validation = {
            "fragment_particle_counts": sorted(sizes.tolist()),
            "detached_samples": detached,
            "particle_count": self.model.particle_count,
            "observed_piece_counts": sorted(self.observed_piece_counts),
            "cuts": cuts,
            "minimum_visual_grip_clearance_mm": self.minimum_grip_clearance * 1000,
            "minimum_blade_center_clearance_mm": self.minimum_blade_clearance * 1000,
            "peak_particle_speed_m_s": self.peak_speed,
        }
        print(f"[BreadThreePieces] PASS: {self.validation}", flush=True)

    @staticmethod
    def create_parser():
        parser = SingleCutExample.create_parser()
        # The optional single-cut surface has a predefined topology and cannot
        # represent this second cut. Render the actual material particles.
        parser.set_defaults(
            num_frames=TOTAL_FRAMES, bread_render="particles", young_modulus=1500.0, cut_plastic_yield=0.002
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

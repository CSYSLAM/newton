# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate a rope-handle paper bag, water bottle and pneumatic shrimp pouch.

Run scripts/start_quest_webxr_w1_bag_packing_rope_shrimp_teleop.sh with
--trajectory-output recordings/bag_rope_shrimp_01.jsonl. This independent
variant keeps the existing three-grocery scene unchanged. The pouch retains
its authored 110 x 120 x 35 mm size, free film, welded seals and gas cavity.
Its full deformation and original render binding are recorded for later use.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from newton.solvers import SolverMJVBDV2

from .example_mjvbd_v2_webxr_w1_bag_packing import main
from .example_mjvbd_v2_webxr_w1_bag_packing_rope_handles import Example as RopeExample
from .support.w1_grocery_assets import load_grocery
from .support.w1_shrimp_softbag import add_softbag

try:
    from msgspec.json import encode as _encode_json
except ImportError:
    _encode_json = None


def _dumps(payload):
    """Use the optional BSD-3-Clause msgspec encoder for full-state JSON frames."""
    if _encode_json is not None:
        return _encode_json(payload).decode("utf-8")
    return json.dumps(payload, separators=(",", ":"))


class Example(RopeExample):
    """Keep W1's calibrated pose while adding a separately simulated snack pouch."""

    recording_prefix = "bag_rope_shrimp"
    scene_title = "W1 麻绳纸袋: 矿泉水与鲜虾片软包装"
    _snack_counts = (1,)
    _grocery_names = ("water_bottle",)
    _json_dumps = staticmethod(_dumps)
    _paper_panel_bending_scale = 4.0
    # Leave a 1.6 mm pad gap for the thin paper, independently of the pouch.
    _paper_grasp_opening = 0.0015
    # W1's inward pads leave the asset's tested 3 mm gap at this coordinate.
    # A smaller gap overstrains the seam with rigid-soft DAT enabled.
    _deformable_grasp_opening = 0.0022
    _deformable_color = (0.86, 0.14, 0.035)
    _deformable_role = "shrimp-softbag"

    def _configure_snacks(self):
        """Place a 75% water bottle and reserve the nearby table edge for the pouch."""
        if self.args.soft_cube:
            raise ValueError("This variant uses the shrimp pouch instead of the soft cube")
        if getattr(self.args, "coarse_iterations", 32) < 1:
            raise ValueError("Coarse iterations must be positive")
        if getattr(self.args, "profile_every", 60) < 0:
            raise ValueError("Profile interval must be nonnegative")
        self.kinds = self._grocery_names
        self.groceries = [load_grocery(self.args.grocery_assets, "water_bottle", scale=0.75)]
        # Show the transparent PET shell as opaque geometry in untextured teleop.
        # Recorded source USD transforms remain unchanged for textured replay.
        for part in self.groceries[0]["parts"]:
            if not part["collision"] and part["opacity"] < 0.5:
                part.update(opacity=1.0, color=(0.45, 0.70, 0.82))
        self.half = np.asarray([self.groceries[0]["half"]])
        self.pick = np.array([(0.46, -0.43, self.table_z + self.half[0, 2] + 0.002)])
        self.snack_openings = {"water_bottle": float(min(self.half[0, :2]) - 0.0025)}

    def _add_snacks(self, builder, full_shapes, *, render_only):
        """Add rigid bottle geometry followed by the free, full-size soft package."""
        super()._add_snacks(builder, full_shapes, render_only=render_only)
        SolverMJVBDV2.register_custom_attributes(builder)
        first_triangle = builder.tri_count
        # Overhang the robot-side edge by 20 mm for an above/below side pinch.
        # The centre of mass remains over the table; there are no pinned points.
        self.shrimp = add_softbag(
            builder,
            directory=self.args.shrimp_assets,
            pos=(0.395, -0.19, self.table_z + 0.020),
            mass_kg=self.args.shrimp_mass,
        )
        self.soft_cube_start, self.soft_cube_end = self.shrimp.particle_start, self.shrimp.particle_end
        self.soft_cube_faces = np.asarray(builder.tri_indices[first_triangle:], dtype=np.int32).reshape(-1, 3)
        for index in range(first_triangle, builder.tri_count):
            builder.tri_color[index] = self._deformable_color

    def _vbd_options(self):
        """Add pouch pressure and caches without replacing the paper contact law."""
        options = super()._vbd_options()
        pneumatic = self.shrimp.solver_vbd_options()
        options.update(
            pneumatic_enable_incremental_volume=pneumatic["pneumatic_enable_incremental_volume"],
            pneumatic_enable_color_coupling=pneumatic["pneumatic_enable_color_coupling"],
            particle_enable_surface_cache=True,
            particle_enable_truncation_cache=True,
            particle_multilevel_coarse_iterations=getattr(self.args, "coarse_iterations", 32),
        )
        passes = getattr(self.args, "coarse_passes", 3)
        if passes < 3:
            checkpoints = [max(1, self.args.iterations - 2)]
            if passes == 2:
                checkpoints.append(max(1, self.args.iterations // 2))
            options["particle_multilevel_checkpoints"] = tuple(sorted(set(checkpoints)))
        return options

    def _build_simulation(self, full_shapes):
        """Reuse elastic cloth stepping and report the chosen solve budget."""
        super()._build_simulation(full_shapes)
        options = self._vbd_options()
        print(
            f"[W1 solve] substeps={self.args.substeps}, iterations={self.args.iterations}, "
            f"coarse={options['particle_multilevel_coarse_iterations']} x "
            f"{len(options['particle_multilevel_checkpoints'])}/substep; "
            f"particle colors={len(self.model.particle_color_groups)}",
            flush=True,
        )

    def _limit_joint_targets(self, previous, solved):
        """Follow both arms directly and rate-limit only closing either gripper."""
        q = previous.copy()
        q[self.arm_indices] = solved[self.arm_indices]
        closing_step = self.args.gripper_speed * self.frame_dt
        for side, control in self.inputs.items():
            indices = self.finger_indices[side]
            q[indices] = np.maximum(control.jaws, previous[indices] - closing_step)
        return q

    def _profile_frame(self, timestamps):
        """Report frame stages with the existing readback as the GPU completion boundary."""
        if self.args.profile_every == 0:
            return
        if not hasattr(self, "_profile_sum") or timestamps[0] - self._profile_last > 1.0:
            self._profile_sum = np.zeros(5)
            self._profile_count = 0
            self._profile_started = timestamps[0]
        self._profile_sum += np.diff(timestamps)
        self._profile_count += 1
        self._profile_last = timestamps[-1]
        if self._profile_count >= self.args.profile_every:
            elapsed = timestamps[-1] - self._profile_started
            names = ("input", "IK-submit", "physics+GPU-wait", "record", "publish")
            stages = ", ".join(
                f"{name}={ms:.1f}ms"
                for name, ms in zip(names, self._profile_sum * 1000 / self._profile_count, strict=True)
            )
            print(f"[W1 performance] {self._profile_count / elapsed:.1f} FPS; {stages}", flush=True)
            self._profile_sum.fill(0)
            self._profile_count = 0
            self._profile_started = timestamps[-1]

    def _recording_extras(self):
        """Save render/physics provenance and the full-state pouch particle range."""
        root = self.args.shrimp_assets.expanduser().resolve()
        files = ("softbag.usdc", "physics_mesh.npz", "physics.json", "textures/basecolor.png")
        # Keep the existing additional-deformable fields for stream compatibility;
        # this metadata identifies their actual geometry as the shrimp pouch.
        return {
            **super()._recording_extras(),
            "solverTuning": {
                "coarseIterations": getattr(self.args, "coarse_iterations", 32),
                "coarsePasses": getattr(self.args, "coarse_passes", 3),
            },
            "softBagAsset": {
                "kind": "oishi-shrimp-v3",
                "usd": str(root / "softbag.usdc"),
                "files": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files},
                "particleStart": self.soft_cube_start,
                "particleCount": self.soft_cube_end - self.soft_cube_start,
                "positionField": "softCubeParticleQ",
                "velocityField": "softCubeParticleQd",
                "renderBinding": "physics_mesh.npz",
                "scale": 1.0,
                "massKg": self.shrimp.config["mass_kg"],
                "materialModel": "elastic",
            },
        }

    def test_post_step(self):
        """Reject nonfinite state or a collapsed/exploded sealed pouch."""
        super().test_post_step()
        q = self.state_0.particle_q.numpy()[self.soft_cube_start : self.soft_cube_end]
        faces = self.shrimp.mesh["triangles"][self.shrimp.mesh["cavity_triangle_indices"]]
        t = (q - q.mean(axis=0))[faces]
        volume = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6
        ratio = volume / self.shrimp.config["rest_volume_m3"]
        if not 0.35 < ratio < 1.8:
            raise AssertionError(f"Shrimp pouch volume ratio outside stable range: {ratio:.4f}")

    @classmethod
    def create_parser(cls):
        """Keep independent launch and recording defaults for the two-item scene."""
        parser = super().create_parser()
        parser.set_defaults(snacks=1, soft_cube=False, substeps=6, iterations=16, webxr_port=8776)
        parser.add_argument("--shrimp-mass", type=float, default=0.010, help="Filled pouch mass in kg (default: 10 g).")
        parser.add_argument("--coarse-iterations", type=int, default=32, help="Coarse PCG iterations per correction.")
        parser.add_argument("--coarse-passes", type=int, choices=(1, 2, 3), default=3)
        parser.add_argument(
            "--profile-every", type=int, default=60, help="Log stage timings every N frames; 0 disables."
        )
        parser.add_argument(
            "--shrimp-assets",
            type=Path,
            default=Path.home() / "下载/oishi_softbag_usd_v3_20260922/oishi_shrimp_softbag",
            help="Directory containing softbag.usdc, physics_mesh.npz and physics.json.",
        )
        return parser


if __name__ == "__main__":
    main(Example)

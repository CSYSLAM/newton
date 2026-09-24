# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate W1 packing with thick rope handles pointing away from the mouth.

./scripts/start_quest_webxr_w1_bag_packing_rope_handles_teleop.sh --trajectory-output recordings/bag_rope_01.jsonl

Press right-controller A or the optical-hand record button to start or pause.
Pause, reset physics, then record again to replace a failed take in the same file.
The tabletop matches Blender's Desk at 0.9297508 m. This scene uses the
toy, soda and biscuit USD assets at 75% scale (override --grocery-assets).
Recordings include their source-to-body transforms for later textured rendering.
The shared replay entry selects this grocery geometry from the recording metadata.
"""

import math
from pathlib import Path

import numpy as np
import warp as wp

from . import example_mjvbd_v2_w1_bag_packing as scene
from .example_mjvbd_v2_webxr_w1_bag_packing import Example as PackingExample
from .example_mjvbd_v2_webxr_w1_bag_packing import main
from .support.w1_grocery_assets import load_grocery


class Example(PackingExample):
    """Keep the shared scene and controls with separate downward rope handles."""

    recording_prefix = "webxr_w1_bag_packing_rope_handles"
    scene_title = "W1 麻绳提手纸袋装零食遥操作"
    handle_color = (0.24, 0.16, 0.085)
    _paper_stiffness_scale = 1.5
    _initial_bag_offset = (0.0, 0.28, 0.0)
    # World-space top of Blender's selected Desk, measured through MCP.
    _table_height = 0.9297508001327515
    _snack_counts = (3,)
    _gripper_open_limit = 0.05
    _grocery_names = ("toy", "soda", "biscuit")
    _grocery_scale = 0.75

    def _lower_body_angles(self):
        """Raise both equal 340 mm leg links without translating or tilting the torso in XY."""
        angle = math.degrees(math.acos(math.cos(math.radians(25)) + (self.table_z - scene.TABLE_Z) / 0.68))
        return (("ANKLE", angle), ("KNEE", -2 * angle), ("BUTTOCK", angle))

    def _configure_snacks(self):
        """Replace the two primitives and soft cube with the three authored USD props."""
        if self.args.soft_cube:
            raise ValueError("The grocery scene replaces the soft cube; use --no-soft-cube")
        self.kinds = tuple(getattr(self.args, "replay_grocery_names", self._grocery_names))
        scales = getattr(self.args, "replay_grocery_scales", (self._grocery_scale,) * len(self.kinds))
        self.groceries = [
            load_grocery(self.args.grocery_assets, name, scale=scale)
            for name, scale in zip(self.kinds, scales, strict=True)
        ]
        self.half = np.asarray([asset["half"] for asset in self.groceries])
        self.pick = np.array(
            [
                (0.46, y, self.table_z + half[2] + 0.002)
                for y, half in zip((-0.48, -0.30, -0.12), self.half, strict=True)
            ]
        )
        self.snack_openings = {
            name: float(min(half[:2]) - 0.0025) for name, half in zip(self.kinds, self.half, strict=True)
        }

    def _add_snacks(self, builder, full_shapes, *, render_only):
        """Scale USD collision hulls and visuals together while preserving density."""
        self.objects = []
        self.grocery_sources = []
        for asset, point in zip(self.groceries, self.pick, strict=True):
            body = builder.add_body(xform=wp.transform(wp.vec3(*point), wp.quat_identity()), label=asset["name"])
            volume = sum(float(part["mesh"].mass) for part in asset["parts"] if part["collision"])
            if volume <= 0:
                raise ValueError(f"Invalid collision-hull volume: {asset['name']}")
            for part in asset["parts"]:
                collision = part["collision"]
                if collision and not render_only:
                    part["mesh"].build_sdf(target_voxel_size=0.001)
                shape = builder.add_shape_mesh(
                    body,
                    mesh=part["mesh"],
                    cfg=builder.ShapeConfig(
                        density=asset["mass"] / volume if collision else 0,
                        ke=4e4,
                        kd=80,
                        mu=scene.SNACK_FRICTION,
                        has_shape_collision=collision,
                        has_particle_collision=collision,
                        # Avoid opaque clear packaging hiding the colored toy pieces
                        # in the untextured Newton and WebXR viewers.
                        is_visible=not collision and part["opacity"] >= 0.5,
                    ),
                    color=part["color"],
                    label=part["path"],
                )
                if collision:
                    full_shapes.append(shape)
            self.objects.append(body)
            self.grocery_sources.append(dict(body=body, label=asset["name"], **asset["source"]))

    def _recording_extras(self):
        """Preserve the source asset alignment alongside the full-state trajectory."""
        return {"tableTopMeters": self.table_z, "groceryAssets": self.grocery_sources}

    @classmethod
    def create_render_scene(cls, viewer, args):
        """Build matching groceries without IK, physics, recording, or WebXR resources."""

        class RenderScene(cls):
            _table_height = getattr(args, "replay_table_height", cls._table_height)

            def __init__(self):
                scene.Example.__init__(self, viewer, args, render_only=True)

            def render(self):
                scene.Example.render(self)

        return RenderScene()

    @classmethod
    def create_parser(cls):
        parser = super().create_parser()
        parser.set_defaults(bag_variant="rope-handles", webxr_port=8775, snacks=3, soft_cube=False)
        parser.add_argument(
            "--grocery-assets",
            type=Path,
            default=Path.home() / "下载/scale_aligned_usd_minimal_20260918",
            help="Root of the scale-aligned toy, soda and biscuit USD directories.",
        )
        return parser


if __name__ == "__main__":
    main(Example)

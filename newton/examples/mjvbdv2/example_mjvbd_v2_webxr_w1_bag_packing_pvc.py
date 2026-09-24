# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Teleoperate W1 with a clear bag using the rope paper-bag material, soda and glue.

./scripts/start_quest_webxr_w1_bag_packing_pvc_teleop.sh --trajectory-output recordings/bag_pvc_01.jsonl

Use the original clear_plastic_bag asset at its authored meter scale. The bag
lies on its side with its mouth toward the right-hand props, as in the rope
scene. Use a stiffer shell with the paper scene's flexible handle material,
without extra support springs.
Film and handles are fully dynamic; no vertices are pinned. Material
coefficients are an experimental approximation, not a measured PVC grade.
"""

from pathlib import Path

import numpy as np
import warp as wp

from . import example_mjvbd_v2_w1_bag_packing as packing
from .example_mjvbd_v2_webxr_w1_bag_packing import main
from .example_mjvbd_v2_webxr_w1_bag_packing_rope_handles import Example as RopeExample
from .support.w1_grocery_assets import load_grocery
from .support.w1_pvc_bag import FILM_COLOR, HANDLE_COLOR, load_pvc_bag


class Example(RopeExample):
    """Keep the calibrated W1 controls while replacing only bag and groceries."""

    recording_prefix = "bag_pvc"
    scene_title = "W1 PVC 塑料袋:汽水罐与胶水"
    _snack_counts = (2,)
    _grocery_names = ("soda", "glue")
    _paper_grasp_opening = 0.0015
    bag_color = FILM_COLOR
    handle_color = HANDLE_COLOR
    bag_opacity = 0.32
    handle_opacity = 0.72
    _pvc_shell_bending_stiffness = 1440.0

    def _configure_snacks(self):
        """Put the two 75%-scale authored props on W1's right side of the table."""
        if self.args.soft_cube:
            raise ValueError("PVC packing uses soda and glue, without a soft cube")
        if not np.isfinite(self.args.pvc_handle_stiffness) or self.args.pvc_handle_stiffness <= 0:
            raise ValueError("PVC handle stiffness must be finite and positive")
        self.kinds = self._grocery_names
        self.groceries = [load_grocery(self.args.grocery_assets, name, scale=0.75) for name in self.kinds]
        self.half = np.asarray([asset["half"] for asset in self.groceries])
        self.pick = np.asarray(
            [(0.46, y, self.table_z + half[2] + 0.002) for y, half in zip((-0.44, -0.23), self.half, strict=True)]
        )
        self.snack_openings = {
            name: float(min(half[:2]) - 0.0025) for name, half in zip(self.kinds, self.half, strict=True)
        }

    def _add_bag(self, builder):
        """Replace paper topology and constitutive parameters with the authored PVC shell."""
        self.pvc = load_pvc_bag(self.args.pvc_assets, coarse_shell=getattr(self.args, "pvc_proxy_version", 2) == 2)
        self.rest, self.faces = self.pvc["vertices"], self.pvc["faces"]
        self.paper_count, self.paper_faces = self.pvc["shell_count"], self.pvc["shell_faces"]
        self.bottom, self.rim = self.pvc["bottom"], self.pvc["rim"]
        self.bag_particle_count = len(self.rest)
        self._placement_rest = self.rest
        self.lip_chains, self.peak_lip_deviation = [], 0.0
        self.free_handle_edges = np.empty((0, 2), dtype=np.int32)
        position, rotation = self._initial_bag_transform()
        self.pvc_initial_transform = [*position.tolist(), *list(rotation)]
        builder.add_cloth_mesh(
            pos=wp.vec3(*position),
            rot=rotation,
            scale=1.0,
            vel=wp.vec3(),
            vertices=self.rest.tolist(),
            indices=self.faces.ravel().tolist(),
            density=packing.BAG_SURFACE_DENSITY,
            tri_ke=1e5,
            tri_ka=1e5,
            tri_kd=0.4,
            edge_ke=60.0,
            edge_kd=2.0,
            particle_radius=0.0012,
            color=FILM_COLOR,
        )
        self._configure_pvc_material(builder)
        self._pvc_render_indices = [
            wp.array(indices.ravel(), dtype=wp.int32)
            for indices in (self.faces[: self.paper_faces], self.faces[self.paper_faces :])
        ]

    def _configure_pvc_material(self, builder):
        """Support the mouth with shell bending while allowing the handles to droop."""
        height = self.rest[self.rim, 2].min()
        for i, edge in enumerate(builder.edge_indices):
            if max(edge) >= self.paper_count:
                builder.edge_bending_properties[i] = (30.0 * self.args.pvc_handle_stiffness, 0.1)
                continue
            # Paper gussets deliberately act as soft hinges. On this wider PVC
            # proxy they let the upper panel and both handle roots fold flat.
            # Preserve the authored rest angles, but resist folding throughout
            # the shell rather than bracing handles or anchoring any particles.
            builder.edge_bending_properties[i] = (self._pvc_shell_bending_stiffness, 4.0)
        for i, face in enumerate(self.faces):
            ke, ka, kd, drag, lift = builder.tri_materials[i]
            if i < self.paper_faces:
                if self.rest[face, 2].min() > height - 0.032:
                    ke, ka, kd = 2 * ke, 2 * ka, 2 * kd
                    for vertex in face:
                        builder.particle_mass[vertex] += packing.BAG_SURFACE_DENSITY * builder.tri_areas[i] / 3
                scale = self._paper_stiffness_scale
            else:
                scale = self.args.pvc_handle_stiffness
            builder.tri_materials[i] = (ke * scale, ka * scale, kd, drag, lift)
            builder.tri_color[i] = FILM_COLOR if i < self.paper_faces else HANDLE_COLOR

    def _initial_bag_transform(self):
        """Match the rope bag's mouth center and orientation, resting the film on the table."""
        reference = RopeExample.__new__(RopeExample)
        reference.table_z = self.table_z
        with np.load(packing.ROPE_HANDLE_ASSETS / "bag.npz") as data:
            reference._placement_rest = data["vertices"]
        position, rotation = reference._initial_bag_transform()
        matrix = np.asarray(wp.quat_to_matrix(rotation)).reshape(3, 3)
        mouth = position + matrix @ np.array((0.0, 0.0, packing.HEIGHT))
        position = mouth - matrix @ self.rest[self.rim].mean(axis=0)
        # Different authored depth: preserve XY mouth alignment, adjust only height.
        points = self.rest @ matrix.T + position
        position[2] += self.table_z + 0.002 - points[:, 2].min()
        return position, rotation

    def _vbd_options(self):
        """Keep the PVC baseline on the same local solve path on CPU and CUDA."""
        options = super()._vbd_options()
        if self.args.pvc_solver == "paper":
            options.update(particle_enable_surface_cache=True, particle_enable_truncation_cache=True)
        else:
            # The paper CUDA acceleration preset tips this PVC shell in user
            # runs. CPU checks cannot validate that preset: those paths are
            # automatically disabled there. Keep material/contact laws intact
            # and use their tested local execution path on both devices.
            options.update(
                particle_enable_multilevel_correction=False,
                particle_enable_tile_solve=False,
                particle_enable_surface_cache=False,
                particle_enable_truncation_cache=False,
                enable_cuda_fast_path=False,
            )
        return options

    def _limit_joint_targets(self, previous, solved):
        """Follow both arms directly; rate-limit both grippers when opening and closing."""
        q = previous.copy()
        q[self.arm_indices] = solved[self.arm_indices]
        step = self.args.gripper_speed * self.frame_dt
        for side, control in self.inputs.items():
            indices = self.finger_indices[side]
            q[indices] += np.clip(control.jaws - previous[indices], -step, step)
        return q

    def _recording_extras(self):
        """Identify original render assets, proxy rest positions and material tuning."""
        return {
            **super()._recording_extras(),
            "pvcBagAsset": {
                **self.pvc["source"],
                "initialTransform": self.pvc_initial_transform,
                "restVertices": self.rest.tolist(),
                "shellFaceCount": self.paper_faces,
                "handleStiffnessScale": self.args.pvc_handle_stiffness,
                "materialModel": "elastic",
                "materialProfile": "pvc-supported-shell-no-springs",
                "paperStiffnessScale": self._paper_stiffness_scale,
                "shellBendingStiffness": self._pvc_shell_bending_stiffness,
                "surfaceDensityKgM2": packing.BAG_SURFACE_DENSITY,
                "solverProfile": self.args.pvc_solver,
            },
        }

    def _render_pvc(self):
        """Show translucent film and smoother, thicker handles in the Newton viewer."""
        self.viewer.show_triangles = False
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        for indices, name, color, opacity, roughness in zip(
            self._pvc_render_indices,
            ("film", "handles"),
            (FILM_COLOR, HANDLE_COLOR),
            (self.bag_opacity, self.handle_opacity),
            (0.22, 0.14),
            strict=True,
        ):
            self.viewer.log_mesh(
                f"/pvc/{name}",
                self.state_0.particle_q,
                indices,
                color=color,
                opacity=opacity,
                roughness=roughness,
                metallic=0.0,
                backface_culling=False,
            )
        self.viewer.end_frame()

    def render(self):
        """Render the PVC material while retaining teleoperation pause controls."""
        self._consume_controls()
        self._render_pvc()

    @classmethod
    def create_render_scene(cls, viewer, args):
        """Construct matching playback geometry without physics, IK or WebXR."""

        class RenderScene(cls):
            def __init__(self):
                packing.Example.__init__(self, viewer, args, render_only=True)

            def render(self):
                self._render_pvc()

        return RenderScene()

    @classmethod
    def create_parser(cls):
        """Use an independent server and recording prefix for PVC packing."""
        parser = super().create_parser()
        parser.set_defaults(
            snacks=2,
            soft_cube=False,
            bag_variant="pvc",
            substeps=6,
            iterations=16,
            webxr_port=8777,
            gripper_speed=0.035,
        )
        parser.add_argument("--pvc-assets", type=Path, default=Path.home() / "下载/clear_plastic_bag")
        parser.add_argument(
            "--pvc-solver",
            choices=("reference", "paper"),
            default="reference",
            help="Use the CPU-checked local solve by default; paper restores the previous CUDA acceleration preset.",
        )
        parser.add_argument(
            "--pvc-handle-stiffness",
            type=float,
            default=10.0,
            help="Handle material multiplier; 1 matches the rope paper bag (default: 10).",
        )
        return parser


if __name__ == "__main__":
    main(Example)

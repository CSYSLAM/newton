# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Dexforce W1 T-shirt folding in WAIC with runtime inverse kinematics.

This is the runtime-IK counterpart to
:mod:`example_cloth_dexforce_bimanual_fold_tshirt_waic_house`. It uses the
same scene, cloth parameters, and scripted TCP targets, but solves the two-arm
IK problem before every simulation frame rather than replaying a baked target
cache. The robot remains kinematic and drives the dynamic cloth one-way.

Run, from the repository root::

    uv run --extra examples -m newton.examples cloth_dexforce_bimanual_fold_tshirt_waic_house_realtime_ik
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton.examples
from newton.examples.cloth import example_cloth_dexforce_bimanual_fold_tshirt_waic_house as fold


@wp.kernel
def _set_indexed_joint_q(
    indices: wp.array[int],
    open_q: wp.array[float],
    grasp_q: wp.array[float],
    grip: wp.array[float],
    joint_q: wp.array[float],
):
    index = wp.tid()
    joint_q[indices[index]] = open_q[index] * (1.0 - grip[0]) + grasp_q[index] * grip[0]


@wp.kernel
def _write_ik_target_poses(left: wp.transform, right: wp.transform, target_poses: wp.array[wp.transform]):
    if wp.tid() == 0:
        target_poses[0] = left
        target_poses[1] = right


@wp.kernel
def _write_ik_grip(grip: float, target_grip: wp.array[float]):
    if wp.tid() == 0:
        target_grip[0] = grip


@wp.kernel
def _unpack_ik_target_poses(
    target_poses: wp.array[wp.transform],
    left_positions: wp.array[wp.vec3],
    left_rotations: wp.array[wp.vec4],
    right_positions: wp.array[wp.vec3],
    right_rotations: wp.array[wp.vec4],
):
    if wp.tid() == 0:
        left = target_poses[0]
        left_positions[0] = wp.transform_get_translation(left)
        left_rotation = wp.transform_get_rotation(left)
        left_rotations[0] = wp.vec4(left_rotation[0], left_rotation[1], left_rotation[2], left_rotation[3])
        right = target_poses[1]
        right_positions[0] = wp.transform_get_translation(right)
        right_rotation = wp.transform_get_rotation(right)
        right_rotations[0] = wp.vec4(right_rotation[0], right_rotation[1], right_rotation[2], right_rotation[3])


class Example(fold.Example):
    """Run the WAIC T-shirt fold while solving the scripted IK targets live."""

    def _build_joint_target_cache(self):
        """Skip the baked trajectory used by the cached-IK base example."""

    def capture(self):
        """Capture fixed-iteration IK, FK, and MJVBD for every material phase."""
        self.graph = None
        self.graphs = {}
        if not getattr(self, "use_graph", False) or not hasattr(self, "ik_target_poses"):
            return

        materials = []
        duration = sum(segment[0] for segment in self.segments)
        for script_time in (0.0, *(np.linspace(0.0, duration, 128))):
            _, _, grip = self._sample(float(script_time))
            materials.append(self._materials_for_script_time(float(script_time), grip))
        self.material_variants = tuple(dict.fromkeys(materials))
        for variant in self.material_variants:
            self.graphs[variant] = self._capture_graph(variant)

    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        if args.ik_iterations < 1:
            raise ValueError("--ik-iterations must be at least 1")
        self.ik_iterations = int(args.ik_iterations)
        self.ik_target_poses = wp.array([self.left_home, self.right_home], dtype=wp.transform, device=self.device)
        self.ik_target_grip = wp.zeros(1, dtype=float, device=self.device)
        self._unpack_ik_target_poses()
        self.active_materials = None
        self.use_graph = bool(args.graph_capture) and self.device.is_cuda
        self.capture()

    def set_ik_target_poses(self, left: wp.transform, right: wp.transform):
        """Write the next left/right TCP targets into the persistent GPU input buffer."""
        wp.launch(
            _write_ik_target_poses,
            1,
            [left, right, self.ik_target_poses],
            device=self.device,
        )

    def set_ik_target_grip(self, grip: float):
        """Write the normalized hand-close target into the persistent GPU input buffer."""
        wp.launch(_write_ik_grip, 1, [grip, self.ik_target_grip], device=self.device)

    def _unpack_ik_target_poses(self):
        """Expose the persistent pose input through the existing IK objectives."""
        wp.launch(
            _unpack_ik_target_poses,
            1,
            [
                self.ik_target_poses,
                self.left_obj.target_positions,
                self.left_rot.target_rotations,
                self.right_obj.target_positions,
                self.right_rot.target_rotations,
            ],
            device=self.device,
        )

    def _update_scripted_inputs(self):
        """Use the original scripted trajectory as the current GPU input producer."""
        script_time = (self.frame_index + 1) * self.frame_dt * self.args.trajectory_time_scale
        left, right, grip = self._sample(script_time)
        self.set_ik_target_poses(left, right)
        self.set_ik_target_grip(grip)
        return script_time, grip

    def _solve_runtime_ik_frame(self):
        """Run the dynamic-target part shared by eager and captured execution."""
        wp.copy(self.frame_q_start, self.state_0.joint_q)
        self._unpack_ik_target_poses()
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=self.ik_iterations)
        wp.launch(
            fold._lock_q,
            self.lock_indices.shape[0],
            [self.ik_q, self.lock_indices, self.lock_values],
            device=self.device,
        )
        wp.launch(
            fold._copy_joint_q,
            self.model.joint_coord_count,
            [self.ik_q[0], self.frame_q_end],
            device=self.device,
        )
        wp.launch(
            _set_indexed_joint_q,
            self.hand_indices.shape[0],
            [self.hand_indices, self.hand_open, self.hand_grasp, self.ik_target_grip, self.frame_q_end],
            device=self.device,
        )

    def simulate(self):
        script_time, grip = self._update_scripted_inputs()
        materials = self._materials_for_script_time(script_time, grip)
        self._set_materials(*materials)
        self._solve_runtime_ik_frame()
        self._simulate_substeps()
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def step(self):
        if not self.use_graph:
            self.simulate()
            return

        script_time, grip = self._update_scripted_inputs()
        materials = self._materials_for_script_time(script_time, grip)
        if materials != self.active_materials:
            self._set_materials(*materials)
            self.active_materials = materials
        wp.capture_launch(self.graphs[materials])
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def _capture_graph(self, materials):
        """Capture a full frame whose IK targets are read from GPU buffers."""
        self._set_materials(*materials)
        state_0_backup = self.model.state()
        state_1_backup = self.model.state()
        state_0_backup.assign(self.state_0)
        state_1_backup.assign(self.state_1)
        ik_q_backup = wp.clone(self.ik_q)

        with wp.ScopedCapture() as capture:
            self._solve_runtime_ik_frame()
            self._simulate_substeps()

        self.state_0.assign(state_0_backup)
        self.state_1.assign(state_1_backup)
        wp.copy(self.ik_q, ik_q_backup)
        return capture.graph

    @staticmethod
    def create_parser():
        parser = fold.Example.create_parser()
        parser.set_defaults(graph_capture=True)
        parser.add_argument(
            "--ik-iterations",
            type=int,
            default=24,
            help="Fixed runtime IK iterations per frame; changing it recreates the CUDA graph.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

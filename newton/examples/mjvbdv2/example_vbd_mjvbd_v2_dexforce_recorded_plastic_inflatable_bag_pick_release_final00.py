# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Replay the plastic inflatable-bag grasp with the full Dexforce W1.

The right hand follows the exact root and finger trajectory used by the
isolated-hand v1 example. Each display frame converts that root pose to the
full-W1 right-wrist TCP and solves the arm with realtime analytic-jacobian IK.
The bag retains the source example's bending plasticity, pneumatic damping,
contact-aware finger closing, and release material.

CUDA devices capture the warmed physics substeps by default. The realtime IK
solve remains outside that graph because its target changes every frame.

Run from the repository root::

    uv run --extra examples -m newton.examples \
        vbd_mjvbd_v2_dexforce_recorded_plastic_inflatable_bag_pick_release_final00 --viewer gl
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mjvbdv2 import (
    example_vbd_mjvbd_v2_dexforce_recorded_inflatable_bag_pick_release as robot_reference,
)
from newton.examples.mjvbdv2 import (
    example_vbd_mjvbd_v2_right_hand_recorded_plastic_inflatable_bag_pick_release_v1 as plastic_reference,
)


class Example(robot_reference.Example):
    """Track the v1 plastic-bag trajectory with realtime full-W1 IK."""

    def _solver_vbd_options(self):
        """Enable the measured incremental cavity-volume path on CUDA."""
        options = super()._solver_vbd_options()
        options["pneumatic_enable_incremental_volume"] = True
        return options

    def __init__(self, viewer, args):
        self.plasticity_enabled = bool(args.plastic)
        yield_angle_deg = plastic_reference._validate_nonnegative(
            args.plastic_yield_angle_deg,
            "plastic yield angle",
        )
        max_angle_deg = plastic_reference._validate_nonnegative(
            args.plastic_max_angle_deg,
            "maximum plastic angle",
        )
        if yield_angle_deg >= 180.0:
            raise ValueError("plastic yield angle must be less than 180 degrees")
        if max_angle_deg >= 180.0:
            raise ValueError("maximum plastic angle must be less than 180 degrees")
        self.plastic_yield_angle = math.radians(yield_angle_deg)
        self.plastic_flow_rate = plastic_reference._validate_nonnegative(
            args.plastic_flow_rate,
            "plastic flow rate",
        )
        self.plastic_max_angle = math.radians(max_angle_deg)
        self.plastic_hardening = plastic_reference._validate_nonnegative(
            args.plastic_hardening,
            "plastic hardening",
        )
        self.bag_bending_stiffness_scale = plastic_reference._validate_nonnegative(
            args.bag_bending_stiffness_scale,
            "bag bending stiffness scale",
        )
        self.bag_bulk_damping = plastic_reference._validate_nonnegative(
            args.bag_bulk_damping,
            "bag bulk damping",
        )

        super().__init__(viewer, args)

        if (
            self.model.edge_rest_angle is None
            or self.model.edge_indices is None
            or self.model.edge_bending_properties is None
            or self.model.edge_count == 0
        ):
            raise RuntimeError("The inflatable bag did not create bending edges.")
        edge_bending_properties = self.model.edge_bending_properties.numpy()
        edge_bending_properties[:, 0] *= self.bag_bending_stiffness_scale
        self.model.edge_bending_properties.assign(edge_bending_properties)
        pneumatic_bulk_damping = self.model.pneumatic.bulk_damping.numpy()
        pneumatic_bulk_damping[self.cavity.cavity_index] = self.bag_bulk_damping
        self.model.pneumatic.bulk_damping.assign(pneumatic_bulk_damping)
        self.authored_edge_rest_angle = wp.clone(self.model.edge_rest_angle)
        self.authored_edge_bending_properties = wp.clone(self.model.edge_bending_properties)

    def _sample_hand_trajectory(self, time_s: float):
        """Sample the v10000 source trajectory without changing its hand pose."""

        return plastic_reference.Example._sample(self, time_s)

    def _apply_bending_plasticity(self):
        """Evolve the bag rest curvature after one completed physics substep."""

        wp.launch(
            plastic_reference._update_bending_plasticity,
            dim=self.model.edge_count,
            inputs=[
                self.state_0.particle_q,
                self.model.edge_indices,
                self.authored_edge_rest_angle,
                self.model.edge_rest_angle,
                self.authored_edge_bending_properties,
                self.model.edge_bending_properties,
                self.bag_particle_start,
                self.bag_particle_end,
                self.plastic_yield_angle,
                self.plastic_flow_rate,
                self.plastic_max_angle,
                self.plastic_hardening,
                self.sim_dt,
            ],
            device=self.device,
        )

    def _simulate_substeps(self):
        """Advance realtime-IK targets and apply plastic flow each substep."""

        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (substep + 1) / self.sim_substeps
            wp.launch(
                robot_reference._interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                robot_reference._joint_velocity,
                self.ik_model.joint_dof_count,
                [self.frame_q_start, self.frame_q_end, 1.0 / self.frame_dt, self.state_0.joint_qd],
                device=self.device,
            )
            newton.eval_fk(
                self.model,
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self.state_0,
                body_flag_filter=newton.BodyFlags.KINEMATIC,
            )
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            wp.launch(
                robot_reference._accumulate_contact_diagnostics,
                1,
                [
                    self.contacts.soft_contact_count,
                    self.solver.vbd_solver.body_particle_contact_overflow_max,
                    self.maximum_soft_contact_count,
                    self.maximum_body_particle_contact_count,
                ],
                device=self.device,
            )
            if self.sim_substeps % 2 != 0 and substep == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0
            if self.plasticity_enabled:
                self._apply_bending_plasticity()

    def _capture_simulation_graph(self):
        """Capture one frame without committing capture-time plastic flow."""

        rest_angle_backup = wp.clone(self.model.edge_rest_angle)
        bending_properties_backup = wp.clone(self.model.edge_bending_properties)
        super()._capture_simulation_graph()
        wp.copy(self.model.edge_rest_angle, rest_angle_backup)
        wp.copy(self.model.edge_bending_properties, bending_properties_backup)

    def test_final(self):
        """Verify robot tracking and bounded plastic flow."""

        super().test_final()
        plastic_offset = self.model.edge_rest_angle.numpy() - self.authored_edge_rest_angle.numpy()
        assert np.all(np.isfinite(plastic_offset))
        maximum_offset = float(np.max(np.abs(plastic_offset), initial=0.0))
        assert maximum_offset <= self.plastic_max_angle + 1.0e-5
        bending_stiffness = self.model.edge_bending_properties.numpy()[:, 0]
        authored_stiffness = self.authored_edge_bending_properties.numpy()[:, 0]
        assert np.all(np.isfinite(bending_stiffness))
        assert np.all(bending_stiffness >= authored_stiffness)
        assert np.all(bending_stiffness <= authored_stiffness * (1.0 + self.plastic_hardening) + 1.0e-5)
        if not self.plasticity_enabled:
            assert maximum_offset < 1.0e-7, "Disabled plasticity changed the authored rest angles."
            assert np.array_equal(bending_stiffness, authored_stiffness)
            return
        script_time = self.sim_time * self.args.trajectory_time_scale
        if script_time + self.frame_dt * self.args.trajectory_time_scale < self.script_duration:
            return
        assert maximum_offset > math.radians(0.1), "The completed grasp did not yield any bag edges."

    @staticmethod
    def create_parser():
        """Create full-W1 realtime-IK and plastic-bag arguments."""

        parser = robot_reference.Example.create_parser()
        parser.add_argument(
            "--plastic",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable bending plasticity for the inflatable bag.",
        )
        parser.add_argument(
            "--plastic-yield-angle-deg",
            type=float,
            default=plastic_reference.PLASTIC_YIELD_ANGLE_DEG,
            help="Dihedral-angle error that starts plastic flow [deg].",
        )
        parser.add_argument(
            "--plastic-flow-rate",
            type=float,
            default=plastic_reference.PLASTIC_FLOW_RATE,
            help="Rest-angle adoption rate while forming a crease [1/s].",
        )
        parser.add_argument(
            "--plastic-max-angle-deg",
            type=float,
            default=plastic_reference.PLASTIC_MAX_ANGLE_DEG,
            help="Maximum rest-angle offset from the authored bag [deg].",
        )
        parser.add_argument(
            "--plastic-hardening",
            type=float,
            default=plastic_reference.PLASTIC_HARDENING,
            help="Maximum added bending-stiffness multiple on yielded edges.",
        )
        parser.add_argument(
            "--bag-bending-stiffness-scale",
            type=float,
            default=plastic_reference.BAG_BENDING_STIFFNESS_SCALE,
            help="Scale applied to the elastic bag bending stiffness before yielding.",
        )
        parser.add_argument(
            "--bag-bulk-damping",
            type=float,
            default=plastic_reference.BAG_BULK_DAMPING,
            help="Pneumatic volume-rate damping [Pa s/m^3].",
        )
        return parser


def main():
    """Run the full-W1 plastic inflatable-bag pick-and-release demo."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

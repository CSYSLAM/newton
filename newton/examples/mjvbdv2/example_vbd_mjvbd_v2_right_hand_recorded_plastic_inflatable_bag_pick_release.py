# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Grasp and release a volume-constrained inflatable bag with plastic creases.

This variant keeps the recorded right-hand trajectory and pneumatic model from
the elastic example, but lets sufficiently bent bag edges adopt a new rest
angle. A small elastic bending range, fast plastic flow, and pneumatic damping
produce a thin-film response instead of a rubber-like spring response. The
plastic law runs after every physics substep for every load source, including
hand contact, internal pressure, and table impact. Reverse loading follows the
same yield law instead of being biased toward the authored bag shape.

CUDA devices capture the warmed physics frame by default. Pass
``--no-graph-capture`` to use direct kernel launches instead.

Run from the repository root::

    uv run --extra examples -m newton.examples \
        vbd_mjvbd_v2_right_hand_recorded_plastic_inflatable_bag_pick_release --viewer gl
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.mjvbdv2 import (
    example_vbd_mjvbd_v2_right_hand_inflatable_bag_recorder as bag_recorder,
)
from newton.examples.mjvbdv2 import (
    example_vbd_mjvbd_v2_right_hand_recorded_inflatable_bag_pick_release as elastic_example,
)

PLASTIC_YIELD_ANGLE_DEG = 1.0
PLASTIC_FLOW_RATE = 120.0
PLASTIC_MAX_ANGLE_DEG = 60.0
PLASTIC_HARDENING = 0.0
BAG_BENDING_STIFFNESS_SCALE = 0.25
BAG_BULK_DAMPING = 2_000_000.0


@wp.kernel
def _update_bending_plasticity(
    positions: wp.array[wp.vec3],
    edge_indices: wp.array2d[wp.int32],
    authored_rest_angles: wp.array[float],
    rest_angles: wp.array[float],
    authored_bending_properties: wp.array2d[float],
    bending_properties: wp.array2d[float],
    bag_particle_start: int,
    bag_particle_end: int,
    yield_angle: float,
    flow_rate: float,
    max_plastic_angle: float,
    hardening: float,
    dt: float,
):
    """Move yielded edge rest angles toward the current bag shape."""

    edge = wp.tid()
    opposite_0 = edge_indices[edge, 0]
    opposite_1 = edge_indices[edge, 1]
    edge_start = edge_indices[edge, 2]
    edge_end = edge_indices[edge, 3]

    if opposite_0 < bag_particle_start or opposite_0 >= bag_particle_end:
        return
    if opposite_1 < bag_particle_start or opposite_1 >= bag_particle_end:
        return
    if edge_start < bag_particle_start or edge_start >= bag_particle_end:
        return
    if edge_end < bag_particle_start or edge_end >= bag_particle_end:
        return

    x0 = positions[opposite_0]
    x1 = positions[opposite_1]
    x2 = positions[edge_start]
    x3 = positions[edge_end]

    normal_0 = wp.cross(x2 - x0, x3 - x0)
    normal_1 = wp.cross(x3 - x1, x2 - x1)
    edge_vector = x3 - x2
    normal_0_length = wp.length(normal_0)
    normal_1_length = wp.length(normal_1)
    edge_length = wp.length(edge_vector)
    if normal_0_length < 1.0e-8 or normal_1_length < 1.0e-8 or edge_length < 1.0e-8:
        return

    normal_0 /= normal_0_length
    normal_1 /= normal_1_length
    edge_direction = edge_vector / edge_length
    current_angle = wp.atan2(
        wp.dot(wp.cross(normal_0, normal_1), edge_direction),
        wp.dot(normal_0, normal_1),
    )
    current_rest_angle = rest_angles[edge]
    angle_error = wp.atan2(
        wp.sin(current_angle - current_rest_angle),
        wp.cos(current_angle - current_rest_angle),
    )
    absolute_error = wp.abs(angle_error)
    if absolute_error <= yield_angle:
        return

    direction = 1.0
    if angle_error < 0.0:
        direction = -1.0

    authored_rest_angle = authored_rest_angles[edge]
    plastic_offset = current_rest_angle - authored_rest_angle
    adoption_fraction = wp.min(flow_rate * dt, 1.0)
    plastic_step_magnitude = (absolute_error - yield_angle) * adoption_fraction
    plastic_step = direction * plastic_step_magnitude
    plastic_offset = wp.clamp(
        plastic_offset + plastic_step,
        -max_plastic_angle,
        max_plastic_angle,
    )
    rest_angles[edge] = authored_rest_angle + plastic_offset

    hardening_angle = wp.max(yield_angle, 1.0e-6)
    hardening_fraction = wp.min(wp.abs(plastic_offset) / hardening_angle, 1.0)
    bending_properties[edge, 0] = authored_bending_properties[edge, 0] * (1.0 + hardening * hardening_fraction)


def _validate_nonnegative(value: float, label: str) -> float:
    """Return a finite nonnegative plasticity parameter."""

    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative, got {value}")
    return value


class Example(elastic_example.Example):
    """Replay the recorded grasp with a plastically bending bag shell."""

    def __init__(self, viewer, args):
        self.plasticity_enabled = bool(args.plastic)
        yield_angle_deg = _validate_nonnegative(args.plastic_yield_angle_deg, "plastic yield angle")
        max_angle_deg = _validate_nonnegative(args.plastic_max_angle_deg, "maximum plastic angle")
        if yield_angle_deg >= 180.0:
            raise ValueError("plastic yield angle must be less than 180 degrees")
        if max_angle_deg >= 180.0:
            raise ValueError("maximum plastic angle must be less than 180 degrees")
        self.plastic_yield_angle = math.radians(yield_angle_deg)
        self.plastic_flow_rate = _validate_nonnegative(args.plastic_flow_rate, "plastic flow rate")
        self.plastic_max_angle = math.radians(max_angle_deg)
        self.plastic_hardening = _validate_nonnegative(args.plastic_hardening, "plastic hardening")
        self.bag_bending_stiffness_scale = _validate_nonnegative(
            args.bag_bending_stiffness_scale, "bag bending stiffness scale"
        )
        self.bag_bulk_damping = _validate_nonnegative(args.bag_bulk_damping, "bag bulk damping")

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

    def _apply_bending_plasticity(self):
        """Evolve the bag rest curvature after one completed physics substep."""

        wp.launch(
            _update_bending_plasticity,
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
        """Advance the hand and apply plastic flow after every substep."""

        for substep in range(bag_recorder.SIM_SUBSTEPS):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            alpha = (substep + 1) / bag_recorder.SIM_SUBSTEPS
            wp.launch(
                bag_recorder.hand_recorder._interpolate_q,
                self.model.joint_coord_count,
                [self.frame_q_start, self.frame_q_end, alpha, self.state_0.joint_q],
                device=self.device,
            )
            wp.launch(
                bag_recorder.hand_recorder._joint_velocity,
                self.model.joint_count,
                [
                    self.frame_q_start,
                    self.frame_q_end,
                    self.model.joint_type,
                    self.model.joint_q_start,
                    self.model.joint_qd_start,
                    1.0 / self.frame_dt,
                    self.state_0.joint_qd,
                ],
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
            if bag_recorder.SIM_SUBSTEPS % 2 != 0 and substep == bag_recorder.SIM_SUBSTEPS - 1:
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
        """Verify plastic flow stays finite, bounded, and switchable."""

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
        if self.sim_time + self.frame_dt < self.script_duration:
            return
        assert maximum_offset > math.radians(0.1), "The completed grasp did not yield any bag edges."

    @staticmethod
    def create_parser():
        """Create command-line arguments for the plastic bag demo."""

        parser = elastic_example.Example.create_parser()
        parser.add_argument(
            "--plastic",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Enable bending plasticity for the inflatable bag.",
        )
        parser.add_argument(
            "--plastic-yield-angle-deg",
            type=float,
            default=PLASTIC_YIELD_ANGLE_DEG,
            help="Dihedral-angle error that starts plastic flow [deg].",
        )
        parser.add_argument(
            "--plastic-flow-rate",
            type=float,
            default=PLASTIC_FLOW_RATE,
            help="Rest-angle adoption rate while forming a crease [1/s].",
        )
        parser.add_argument(
            "--plastic-max-angle-deg",
            type=float,
            default=PLASTIC_MAX_ANGLE_DEG,
            help="Maximum rest-angle offset from the authored bag [deg].",
        )
        parser.add_argument(
            "--plastic-hardening",
            type=float,
            default=PLASTIC_HARDENING,
            help="Maximum added bending-stiffness multiple on yielded edges.",
        )
        parser.add_argument(
            "--bag-bending-stiffness-scale",
            type=float,
            default=BAG_BENDING_STIFFNESS_SCALE,
            help="Scale applied to the elastic bag bending stiffness before yielding.",
        )
        parser.add_argument(
            "--bag-bulk-damping",
            type=float,
            default=BAG_BULK_DAMPING,
            help="Pneumatic volume-rate damping [Pa s/m^3].",
        )
        return parser


def main():
    """Run the plastic inflatable-bag pick-and-release demo."""

    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)


if __name__ == "__main__":
    main()

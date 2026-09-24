# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspect the W1 PVC packing scene with live physics and no teleoperation.

uv run --extra examples -m newton.examples mjvbd_v2_w1_bag_packing_pvc

Reuse the teleoperation scene's geometry, initial IK pose, materials, contacts
and solver settings. Hold the robot at that pose while the bag, soda and glue
respond to gravity and viewer forces. No WebXR server or recorder is created.
"""

from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

from . import example_mjvbd_v2_w1_bag_packing as packing
from .example_mjvbd_v2_webxr_w1_bag_packing_pvc import Example as PVCExample


class Example(PVCExample):
    """Run the shared PVC scene without constructing the teleoperation runtime."""

    def __init__(self, viewer, args):
        # Use the physical scene constructor directly: PVCExample.__init__ also
        # creates XR input, recording and server resources even without a client.
        packing.Example.__init__(self, viewer, args)
        self._initial_state = self.model.state()
        self._initial_state.assign(self.state_0)
        # The shared builder seeds targets from the pre-IK model. Teleoperation
        # overwrites them every frame; this idle scene must hold the solved pose.
        wp.copy(self.frame_start, self._initial_state.joint_q, count=self.frame_start.size)
        wp.copy(self.frame_end, self.frame_start)

    def step(self):
        """Advance the same physics with fixed initial robot joint targets."""
        self._advance_physics()
        self.frame += 1
        self.sim_time = self.frame * self.frame_dt

    def render(self):
        """Draw the shared translucent PVC scene without consuming XR controls."""
        self._render_pvc()

    def reset_physics(self, *, source="viewer"):
        """Restore the initial physical state without restarting any services."""
        self.state_0.assign(self._initial_state)
        self.state_1.assign(self._initial_state)
        self.solver.reset(self.state_0, flags=0)
        self.solver.reset(self.state_1, flags=0)
        wp.copy(self.frame_start, self._initial_state.joint_q, count=self.frame_start.size)
        wp.copy(self.frame_end, self.frame_start)
        self.frame = 0
        self.sim_time = 0.0

    def close(self):
        """Allow the example browser to close a scene that owns no XR resources."""

    def test_post_step(self):
        """Check finite props, fixed robot targets and no unforced bag centre-of-mass rise."""
        for values in (self.state_0.body_q, self.state_0.particle_q, self.state_0.particle_qd):
            if not np.isfinite(values.numpy()).all():
                raise AssertionError("Nonfinite PVC scene state")
        np.testing.assert_allclose(self.state_0.joint_q.numpy(), self._initial_state.joint_q.numpy(), atol=1e-6)
        if np.any(self.model.particle_inv_mass.numpy() <= 0):
            raise AssertionError("The PVC bag must remain unpinned")
        if self.model.spring_count:
            raise AssertionError("The PVC scene must not add support springs")
        mass = self.model.particle_mass.numpy()
        initial_height = np.average(self._initial_state.particle_q.numpy()[:, 2], weights=mass)
        height = np.average(self.state_0.particle_q.numpy()[:, 2], weights=mass)
        if height > initial_height + 0.01:
            raise AssertionError(f"Unforced PVC bag centre of mass rose by {height - initial_height:.4f} m")
        if np.any(self.model.body_flags.numpy()[self.objects] & int(newton.BodyFlags.KINEMATIC)):
            raise AssertionError("Soda and glue must remain dynamic")

    def test_final(self):
        """Validate the final inspection state without requiring a packing task."""
        self.test_post_step()

    @classmethod
    def create_parser(cls):
        """Expose physical scene options with defaults taken from the teleoperation entry."""
        shared = PVCExample.create_parser()
        parser = newton.examples.create_parser()
        parser.set_defaults(
            num_frames=2**31 - 1,
            **{key: shared.get_default(key) for key in ("snacks", "soft_cube", "bag_variant")},
        )
        for name, kind, help_text in (
            ("substeps", int, "Physics substeps per 60 Hz frame."),
            ("iterations", int, "VBD iterations per substep."),
            ("pvc_handle_stiffness", float, "Handle material multiplier; 1 matches the rope paper bag."),
            ("robot_setback", float, "Robot offset away from the table [m]."),
            ("pvc_assets", Path, "Directory containing the authored clear PVC bag."),
            ("grocery_assets", Path, "Directory containing the authored soda and glue USD assets."),
        ):
            parser.add_argument(
                "--" + name.replace("_", "-"), type=kind, default=shared.get_default(name), help=help_text
            )
        parser.add_argument("--no-cuda-graph", action="store_true", default=shared.get_default("no_cuda_graph"))
        parser.add_argument(
            "--pvc-solver",
            choices=("reference", "paper"),
            default=shared.get_default("pvc_solver"),
            help="Local reference solve, or the previous paper CUDA acceleration preset for comparison.",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

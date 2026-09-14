# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example MuJoCo Box Packing
#
# Uses two ALOHA arms to fold and close an elastoplastic carton. The scene,
# motion replay, feedback law, and scored-fold material model are adapted
# from the reference mujoco_warp benchmark. Model construction, actuator control,
# solver stepping, and rendering use Newton APIs; the custom fold law runs at
# the MuJoCo passive-force hook exposed by SolverMuJoCo.
#
# Command: uv run --extra examples -m newton.examples mujoco_box_packing
#
###########################################################################

from __future__ import annotations

import math

import mujoco
import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverMuJoCo

from .box_packing.creases import CreaseModel
from .box_packing.feedback import CartonFrameController
from .box_packing.model import HERE, build_newton_model, load_reference_model
from .box_packing.validation import packing_fault, verify_hold

SIM_DT = 0.001
SIM_SUBSTEPS = 40
FRAME_DT = SIM_DT * SIM_SUBSTEPS
FEEDBACK_STEPS = round(0.04 / SIM_DT)
REFERENCE_RENDER_FPS = 50.0


class Example:
    def __init__(self, viewer, args):
        """Build the Newton scene and initialize the retained packing rollout."""
        self.viewer = viewer
        self.sim_time = 0.0
        self.sim_step = 0
        self.failure: str | None = None
        self.completion_metrics: dict | None = None
        self.feedback_graph = None
        self.open_loop = bool(args.open_loop)
        self.solver_tolerance = float(args.solver_tolerance)
        if self.solver_tolerance <= 0.0:
            raise ValueError("solver-tolerance must be positive")

        with np.load(HERE / "replay.npz") as replay:
            self.replay_controls = replay["ctrl"].copy()
            initial_qpos = replay["qpos"][0].copy()
            initial_qvel = replay["qvel"][0].copy()
        self.total_steps = len(self.replay_controls)

        self.reference_model = load_reference_model()
        self.reference_data = mujoco.MjData(self.reference_model)
        self.reference_data.qpos[:] = initial_qpos
        self.reference_data.qvel[:] = initial_qvel
        self.reference_data.ctrl[:] = self.replay_controls[0]
        mujoco.mj_forward(self.reference_model, self.reference_data)

        if not wp.get_device().is_cuda:
            raise RuntimeError("The MuJoCo box-packing example requires a CUDA device.")
        self.model = build_newton_model(initial_qpos, initial_qvel)
        self.solver = SolverMuJoCo(
            self.model,
            separate_worlds=False,
            solver="newton",
            integrator="euler",
            cone="pyramidal",
            jacobian="sparse",
            iterations=80,
            tolerance=self.solver_tolerance,
            nconmax=2048,
            njmax=4096,
            use_mujoco_contacts=True,
            # No Newton-side state is authored after initialization. Keeping
            # MuJoCo Warp's data resident avoids round-tripping the free body's
            # pose and twist through Newton coordinates every millisecond.
            update_data_interval=0,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        if not hasattr(self.control, "mujoco") or self.control.mujoco.ctrl is None:
            raise RuntimeError("The imported ALOHA actuators did not expose control.mujoco.ctrl")

        self.creases = CreaseModel(self.model, self.reference_model)
        self.creases.install_mujoco_callback(self.solver)
        self.controller = None if self.open_loop else CartonFrameController(self.reference_model, HERE / "replay.npz")
        self.active_control = self.replay_controls[0].copy()
        self.hold_times: list[float] = []
        self.hold_qpos: list[np.ndarray] = []

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.3338, -0.8520, 0.6016), pitch=-24.0, yaw=112.0)

    def _sync_reference_data(self) -> None:
        """Copy the solver's native generalized state into the controller model."""
        if self.solver.use_mujoco_cpu:
            qpos = self.solver.mj_data.qpos
            qvel = self.solver.mj_data.qvel
        else:
            qpos = self.solver.mjw_data.qpos.numpy()[0]
            qvel = self.solver.mjw_data.qvel.numpy()[0]
        self.reference_data.qpos[:] = qpos
        self.reference_data.qvel[:] = qvel
        mujoco.mj_forward(self.reference_model, self.reference_data)

    def _update_control(self) -> None:
        """Apply replay or carton-frame feedback controls for the current substep."""
        if self.controller is None:
            self.active_control = self.replay_controls[self.sim_step]
        else:
            if self.sim_step % FEEDBACK_STEPS == 0:
                if self.sim_step:
                    self._sync_reference_data()
                self.active_control = self.controller.update(self.reference_data, self.sim_step * SIM_DT)
            if self.sim_step * SIM_DT < 12.0:
                self.active_control = self.replay_controls[self.sim_step] + self.controller.prefix_offset
        self.control.mujoco.ctrl.assign(self.active_control.astype(np.float32))

    def _simulate_substep(self) -> None:
        """Advance one millisecond using Newton's MuJoCo solver interface."""
        self._update_control()
        self.solver.step(self.state_0, self.state_1, self.control, None, SIM_DT)
        self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_step += 1

    def step(self):
        """Advance forty physical milliseconds, displayed at reference 2x speed."""
        if self.failure is not None or self.sim_step >= self.total_steps:
            return
        substeps = min(SIM_SUBSTEPS, self.total_steps - self.sim_step)
        try:
            if self.controller is not None and self.sim_step >= 12000 and substeps == SIM_SUBSTEPS:
                self._update_control()
                # Feedback holds controls for forty steps. Capture only the
                # physics work; observations and the controller stay outside.
                if self.feedback_graph is None:
                    with wp.ScopedCapture(device=self.model.device) as capture:
                        for _ in range(SIM_SUBSTEPS):
                            self.solver.step(self.state_0, self.state_1, self.control, None, SIM_DT)
                            self.state_0, self.state_1 = self.state_1, self.state_0
                    self.feedback_graph = capture.graph
                wp.capture_launch(self.feedback_graph)
                self.sim_step += SIM_SUBSTEPS
            else:
                for _ in range(substeps):
                    self._simulate_substep()
        except RuntimeError as error:
            self.failure = f"t={self.sim_step * SIM_DT:.3f}s: {error}"
            raise

        self.sim_time = self.sim_step * SIM_DT
        self._sync_reference_data()
        fault = packing_fault(self.reference_model, self.reference_data, self.sim_time)
        if fault is not None:
            self.failure = f"t={self.sim_time:.3f}s: {fault}"
            raise RuntimeError(self.failure)
        self.hold_times.append(self.sim_time)
        self.hold_qpos.append(self.reference_data.qpos.copy())
        if self.sim_step >= self.total_steps:
            self._validate_completion()

    def _validate_completion(self) -> None:
        """Apply the same released-hold acceptance check in the viewer and tests."""
        if self.completion_metrics is not None:
            return
        passed, metrics = verify_hold(self.reference_model, self.hold_times, self.hold_qpos)
        self.completion_metrics = metrics
        print(f"closure: {metrics}", flush=True)
        if not passed:
            self.failure = f"Released carton did not remain closed: {metrics}"
            raise RuntimeError(self.failure)

    def render(self):
        """Render the current Newton state."""
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        """Check finite state, and validate closure after a complete rollout."""
        if self.failure is not None:
            raise AssertionError(self.failure)
        if self.sim_step <= 0:
            raise AssertionError("The box-packing example did not advance")
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise AssertionError("The box-packing joint state is non-finite")
        if self.sim_step >= self.total_steps:
            self._validate_completion()

    @staticmethod
    def create_parser():
        """Create command-line options for feedback and solver accuracy."""
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--open-loop",
            action="store_true",
            help="Replay controls without carton-frame feedback (diagnostic).",
        )
        parser.add_argument(
            "--solver-tolerance",
            type=float,
            default=1.0e-7,
            help="MuJoCo Newton solver tolerance.",
        )
        parser.set_defaults(
            num_frames=math.ceil(46001 / SIM_SUBSTEPS),
            render_fps=REFERENCE_RENDER_FPS,
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

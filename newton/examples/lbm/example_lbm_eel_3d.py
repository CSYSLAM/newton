# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example LBM Eel 3D
#
# Three-dimensional D3Q27 lattice-Boltzmann simulation of an articulated
# 12-segment eel swimming through a fluid. The articulated body is driven by
# SolverMuJoCo (22 yaw/roll motors), the fluid by SolverLBM, and the two are
# coupled bidirectionally every substep. A physics-parameterized traveling
# wave maps four scalar controls (A, omega, k_wave, head_bias) to the 22
# actuator targets.
#
# Command: python -m newton.examples lbm_eel_3d
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverLBM, SolverMuJoCo

from newton.examples.lbm._lbm_render import vorticity_to_image

# Traveling-wave control (ported from the Open HOME-LBM eel environment).
_N_YAW_JOINTS = 11
_WAVE_S = np.array([i / (_N_YAW_JOINTS - 1) for i in range(_N_YAW_JOINTS)], dtype=np.float32)
_WAVE_ENVELOPE = (0.05 + 0.95 * _WAVE_S).astype(np.float32)
_YAW_ACTUATORS = list(range(0, 22, 2))   # 11 yaw motors at even ctrl indices
_ROLL_ACTUATORS = list(range(1, 22, 2))  # 11 roll motors at odd ctrl indices

_PRESETS = {
    "forward": {"A": 0.8, "omega": -0.5, "k_wave": 0.5, "head_bias": 0.0},
    "turn_l": {"A": 0.7, "omega": -0.5, "k_wave": 0.5, "head_bias": -0.6},
    "turn_r": {"A": 0.7, "omega": -0.5, "k_wave": 0.5, "head_bias": 0.6},
    "fast": {"A": 0.8, "omega": -0.7, "k_wave": 0.6, "head_bias": 0.0},
    "freeze": {"A": 0.0, "omega": 0.0, "k_wave": 0.5, "head_bias": 0.0},
}


def _wave_to_ctrl(params: dict, sim_time: float, ctrl_range: np.ndarray) -> np.ndarray:
    """Map traveling-wave parameters to the 22 MuJoCo actuator targets.

    The wave angle is normalized to [-1, 1] and mapped through each actuator's
    ctrl range (the eel's position actuators use the joint-angle range).
    """
    a = float(params["A"])
    omega = float(params["omega"]) * np.pi * 2.0
    k_wave = float(params["k_wave"]) * 1.5
    head_bias = float(params["head_bias"])
    phase = omega * sim_time + k_wave * np.pi * _WAVE_S
    theta = np.clip(a * _WAVE_ENVELOPE * np.sin(phase) + head_bias * (1.0 - _WAVE_S), -1.0, 1.0)
    lo = ctrl_range[:, 0]
    hi = ctrl_range[:, 1]
    ctrl = np.empty(22, dtype=np.float32)
    for i, act in enumerate(_YAW_ACTUATORS):
        ctrl[act] = lo[act] + (theta[i] + 1.0) * 0.5 * (hi[act] - lo[act])
    for act in _ROLL_ACTUATORS:
        ctrl[act] = 0.5 * (lo[act] + hi[act])  # centered roll target
    return ctrl


class Example:
    def __init__(self, viewer, args):
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.per_frame_steps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.args = args

        self.preset = args.preset
        self.params = dict(_PRESETS[self.preset])
        self._time_accum = 0.0
        self.vmax = None

        nx, ny, nz = args.nx, args.ny, args.nz
        scale = args.lbm_scale * nx

        builder = newton.ModelBuilder()
        builder.add_mjcf(
            newton.examples.get_asset("eel_3d.xml"),
            ctrl_direct=True,  # position actuators read control.mujoco.ctrl
        )
        self.model = builder.finalize()
        # The reference MJCF declares zero gravity but Newton imports -9.81 Z;
        # restore the intended weightless fluid environment.
        self.model.set_gravity((0.0, 0.0, 0.0))
        self.ctrl_range = self.model.mujoco.actuator_ctrlrange.numpy().astype(np.float32)

        # Root (head) lattice position; each segment is placed from its initial
        # COM displacement relative to the head.
        root_com = self._body_com(0)
        root_lbm = np.array([nx * 0.5, ny * 0.6, nz * 0.5], dtype=np.float32)

        config = SolverLBM.Config(
            nx=nx,
            ny=ny,
            nz=nz,
            lbm_scale=args.lbm_scale,
            viscosity=args.viscosity,
        )
        self.lbm = SolverLBM(self.model, config)
        for body in range(self.model.body_count):
            com = self._body_com(body)
            lbm_pos = root_lbm + (com - root_com) * scale
            self.lbm.add_solid(body_index=body, lbm_position=tuple(lbm_pos))
        self.lbm.finalize()

        self.rigid_solver = SolverMuJoCo(self.model)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=(nx * 0.5, ny * 0.7, nz * 2.5), pitch=-0.5, yaw=0.0)

        self.capture()

    def _body_com(self, body_index: int) -> np.ndarray:
        """Initial world-frame COM of a body."""
        model = self.model
        tf7 = model.body_q.numpy()[body_index]
        p = tf7[:3]
        q = tf7[3:7]
        x, y, z, w = q
        rot = np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ]
        )
        com = model.body_com.numpy()[body_index]
        return (rot @ com + p).astype(np.float32)

    # ---- Simulation ------------------------------------------------------

    def simulate(self):
        # Time-varying actuator targets are written on the device before graph
        # replay (see step()); the captured body of this method only replays
        # the fixed coupling kernels.
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.lbm.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.rigid_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self._time_accum += self.sim_dt * self.sim_substeps

    def step(self):
        # Push the traveling-wave actuator targets to the device (outside the
        # captured graph) so the replay reads fresh ctrl values each frame.
        self.control.mujoco.ctrl.assign(_wave_to_ctrl(self.params, self._time_accum, self.ctrl_range))
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    # ---- Rendering -------------------------------------------------------

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)

        img, self.vmax = vorticity_to_image(self.lbm.vorticity_projection("side"), self.vmax)
        self.viewer.log_image("eel/vorticity_side", img)
        self.viewer.end_frame()

    # ---- UI -------------------------------------------------------------

    def gui(self, imgui):
        changed, name = imgui.combo("Preset", list(_PRESETS).index(self.preset), list(_PRESETS.keys()))
        if changed:
            self.set_preset(list(_PRESETS.keys())[name])
        _changed, self.params["A"] = imgui.slider_float("Amplitude A", self.params["A"], 0.0, 1.0)
        _changed, self.params["omega"] = imgui.slider_float("Omega", self.params["omega"], -1.0, 1.0)
        _changed, self.params["k_wave"] = imgui.slider_float("k_wave", self.params["k_wave"], -1.0, 1.0)
        _changed, self.params["head_bias"] = imgui.slider_float("Head bias", self.params["head_bias"], -1.0, 1.0)

    def set_preset(self, name: str) -> None:
        self.preset = name
        self.params = dict(_PRESETS[name])

    def capture(self):
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    # ---- Tests -----------------------------------------------------------

    def test_post_step(self):
        if self.sim_time > 2.0 and self.preset == "forward":
            head = self.state_0.body_q.numpy()[self._head_body()]
            if not np.isfinite(head).all():
                raise AssertionError("eel head pose is non-finite")

    def test_final(self):
        head = self.state_0.body_q.numpy()[self._head_body()]
        # Head must stay inside the lattice domain (in lattice units).
        scale = self.args.lbm_scale * self.args.nx
        head_lbm = head[:3] * scale + np.array([self.args.nx * 0.5, self.args.ny * 0.6, self.args.nz * 0.5])
        inside = (
            0.0 <= head_lbm[0] <= self.args.nx
            and 0.0 <= head_lbm[1] <= self.args.ny
            and 0.0 <= head_lbm[2] <= self.args.nz
        )
        if not inside:
            raise AssertionError(f"eel head left the lattice domain: {head_lbm}")

    def _head_body(self) -> int:
        return 0

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--nx", type=int, default=96)
        parser.add_argument("--ny", type=int, default=160)
        parser.add_argument("--nz", type=int, default=40)
        parser.add_argument("--lbm-scale", type=float, default=0.5)
        parser.add_argument("--viscosity", type=float, default=0.1)
        parser.add_argument("--fps", type=int, default=30)
        parser.add_argument("--per-frame-steps", type=int, default=5)
        parser.add_argument("--preset", type=str, default="forward", choices=list(_PRESETS))
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

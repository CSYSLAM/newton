# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Alternate independent scenes to reduce clock/load bias in A/B timing.

Original finger control is now restored. Its existing final volume assertion
can fail; historical passing results used the withdrawn compression controller.
"""

import json
import math
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import MethodType
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import warp as wp

import newton.examples
import newton.ik as ik
import newton.viewer
from newton._src.solvers.mjvbd_v2 import full_contact_pipeline as fc
from newton._src.solvers.mjvbd_v2.vbd.particle_vbd_kernels import apply_truncation_ts
from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD
from newton.examples.mjvbdv2.example_mjvbd_v2_inflatable_bag_grasp import Example
from newton.examples.mjvbdv2.support import (
    example_vbd_mjvbd_v2_dexforce_recorded_inflatable_bag_pick_release as robot_reference,
)

original_ik = ik.IKSolver
original_truncation = SolverVBD._penetration_free_truncation
reference_small = fc._make_compact_soft_face_kernel(0, 256)
reference_large = fc._make_compact_soft_face_kernel(257, 2147483647)


def reference_runtime_ik(self):
    self.ik_solver.step(self.ik_q, self.ik_q, iterations=robot_reference.RUNTIME_IK_ITERATIONS)


def reference_ik(*args, **kwargs):
    kwargs["compact_dof_mask"] = False
    return original_ik(*args, **kwargs)


def reference_truncation(self, particle_q_out=None, *, empty_contact_set=False):
    if self.particle_enable_self_contact or self._particle_truncation_cache is not None:
        return original_truncation(self, particle_q_out, empty_contact_set=empty_contact_set)
    self.truncation_ts.fill_(1.0)
    wp.launch(
        apply_truncation_ts,
        dim=self.model.particle_count,
        inputs=[self.pos_prev_collision_detection, self.particle_displacements, self.truncation_ts, wp.inf],
        outputs=[
            self.particle_displacements,
            particle_q_out,
            self.particle_chebyshev_collided if self.particle_chebyshev_guarded else None,
            self.particle_chebyshev_cleanup_status,
        ],
        device=self.device,
    )


wp.config.log_level = wp.LOG_WARNING
viewers = [newton.viewer.ViewerNull(), newton.viewer.ViewerNull()]
examples = []
rows = []
try:
    with ExitStack() as stack:
        stack.enter_context(patch.object(ik, "IKSolver", reference_ik))
        stack.enter_context(patch.object(SolverVBD, "_penetration_free_truncation", reference_truncation))
        stack.enter_context(patch.object(fc, "_SMALL_FACE_BATCH_LIMIT", 256))
        stack.enter_context(patch.object(fc, "_create_compact_soft_face_contacts_small", reference_small))
        stack.enter_context(patch.object(fc, "_create_compact_soft_face_contacts_large", reference_large))
        reference = Example(viewers[0], newton.examples.default_args(Example.create_parser()))
        reference._solve_runtime_ik = MethodType(reference_runtime_ik, reference)
        reference.step()  # Capture the reference graph before restoring dispatch.
        reference.test_post_step()
        examples.append(reference)
    optimized = Example(viewers[1], newton.examples.default_args(Example.create_parser()))
    optimized.step()
    optimized.test_post_step()
    examples.append(optimized)
    count = math.ceil(optimized.script_duration / optimized.frame_dt) + 2
    for frame in range(1, count):
        times = [0.0, 0.0]
        for index in (0, 1) if frame % 2 == 0 else (1, 0):
            example = examples[index]
            wp.synchronize()
            start = time.perf_counter()
            example.step()
            wp.synchronize()
            times[index] = time.perf_counter() - start
            example.test_post_step()
        rows.append([frame, *times, optimized.active_phase_name])
        if frame % 100 == 0:
            print("FRAME", frame, "mean ms", np.mean(np.array([r[1:3] for r in rows[-80:]]), axis=0) * 1000, flush=True)
    for example in examples:
        example.test_final()
    measured = np.array([r[1:3] for r in rows if r[0] >= 30])
    result = {
        "mean_step_ms": (measured.mean(axis=0) * 1000).tolist(),
        "step_time_reduction": float(1.0 - measured[:, 1].mean() / measured[:, 0].mean()),
        "minimum_volume_ratio": [e.minimum_volume_ratio for e in examples],
        "maximum_root_position_error_m": [e.maximum_root_position_error for e in examples],
        "maximum_pressure_pa": [e.maximum_pressure for e in examples],
        "lifted_bag_center_z_m": [e.lifted_bag_center_z for e in examples],
        "final_bag_center_z_m": [e._bag_center_z() for e in examples],
        "acceptance": "both complete sequences passed",
    }
    folder = Path.cwd() / "newton/tests/outputs/inflatable_perf"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "interleaved.json").write_text(json.dumps({"summary": result, "frames": rows}, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
finally:
    for viewer in viewers:
        viewer.close()

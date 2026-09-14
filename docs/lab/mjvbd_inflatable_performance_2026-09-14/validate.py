# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare full default inflatable grasp runs; keep failed assertions in the report."""

import argparse
import ctypes
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import numpy as np
import warp as wp

import newton.examples
import newton.viewer
from newton.examples.mjvbdv2.example_mjvbd_v2_inflatable_bag_grasp import Example

p = argparse.ArgumentParser()
p.add_argument("--output", default="optimized")
p.add_argument("--trace", action="store_true")
p.add_argument("--pneumatic-color-coupling", action="store_true")
p.add_argument("--frames", type=int)
p.add_argument("--baseline", action="store_true")
p.add_argument("--reference-compute", action="store_true")
p.add_argument("--no-render", action="store_true")
p.add_argument("--screenshots", action="store_true")
o = p.parse_args()
folder = Path.cwd() / "newton/tests/outputs/inflatable_perf"
folder.mkdir(parents=True, exist_ok=True)
if o.baseline or o.reference_compute:
    import newton.ik as ik
    from newton._src.solvers.mjvbd_v2.vbd.particle_vbd_kernels import apply_truncation_ts
    from newton._src.solvers.mjvbd_v2.vbd.solver_vbd import SolverVBD
    from newton.examples.mjvbdv2.support import (
        example_vbd_mjvbd_v2_dexforce_recorded_inflatable_bag_pick_release as robot_reference,
    )

    def reference_runtime_ik(self):
        self.ik_solver.step(self.ik_q, self.ik_q, iterations=robot_reference.RUNTIME_IK_ITERATIONS)

    robot_reference.Example._solve_runtime_ik = reference_runtime_ik
    original_ik = ik.IKSolver
    original_truncation = SolverVBD._penetration_free_truncation

    def reference_ik(*args, **kwargs):
        kwargs["compact_dof_mask"] = False
        if o.baseline:
            kwargs["joint_dof_mask"] = None
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

    ik.IKSolver = reference_ik
    SolverVBD._penetration_free_truncation = reference_truncation
if o.reference_compute:
    from newton._src.solvers.mjvbd_v2 import full_contact_pipeline as fc

    fc._SMALL_FACE_BATCH_LIMIT = 256
    fc._create_compact_soft_face_contacts_small = fc._make_compact_soft_face_kernel(0, 256)
    fc._create_compact_soft_face_contacts_large = fc._make_compact_soft_face_kernel(257, 2147483647)
if o.baseline:
    baseline_source = subprocess.run(
        ["git", "show", "37797903:newton/_src/solvers/mjvbd_v2/vbd/pneumatic_kernels.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (folder / "pneumatic_before.py").write_text(baseline_source)
    import importlib.util

    from newton._src.solvers.mjvbd_v2 import full_contact_pipeline as fc
    from newton._src.solvers.mjvbd_v2.vbd import pneumatic_kernels as pk

    fc._ENABLE_SMALL_FACE_BATCH = False
    spec = importlib.util.spec_from_file_location("inflatable_pneumatic_before", folder / "pneumatic_before.py")
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    launch = wp.launch

    def baseline_launch(kernel, *args, **kwargs):
        if kernel is pk.update_single_cavity_volume_pressure_and_accumulate_force:
            kernel = old.update_single_cavity_volume_pressure_and_accumulate_force
        return launch(kernel, *args, **kwargs)

    wp.launch = baseline_launch
wp.config.log_level = wp.LOG_WARNING
v = (
    newton.viewer.ViewerNull()
    if o.no_render
    else newton.viewer.ViewerGL(
        width=1280, height=720, headless=True, enable_cuda_interop=newton.viewer.ViewerGL.CudaInterop.NONE
    )
)
example_args = newton.examples.default_args(Example.create_parser())
example_args.pneumatic_color_coupling = o.pneumatic_color_coupling
e = Example(v, example_args)
count = o.frames or int(np.ceil(e.script_duration / e.frame_dt)) + 2
print(
    "CONFIG",
    count,
    e.script_duration,
    e.model.particle_count,
    e.model.tri_count,
    e.sim_substeps,
    e.solver.vbd_solver.iterations,
    [len(x) for x in e.model.particle_color_groups],
    flush=True,
)
print("PHASES", [(x.name, x.duration) for x in e.phases], flush=True)
rows = []
plastic = []
stiffness = []
bodies = []
positions = []
pressures = []
volumes = []
failures = []
cudart = ctypes.CDLL("libcudart.so.12") if o.trace else None
try:
    for i in range(count):
        if o.trace and i in (300, 500):
            cudart.cudaProfilerStart()
        wp.synchronize()
        t = time.perf_counter()
        e.step()
        wp.synchronize()
        step = time.perf_counter() - t
        t = time.perf_counter()
        if not o.no_render:
            e.render()
        wp.synchronize()
        render = time.perf_counter() - t
        try:
            e.test_post_step()
        except AssertionError as exc:
            failures.append((i, str(exc)))
        rows.append((i, step, render, e.active_phase_name))
        positions.append(e.state_0.particle_q.numpy())
        bodies.append(e.state_0.body_q.numpy())
        plastic.append(e.model.edge_rest_angle.numpy())
        stiffness.append(e.model.edge_bending_properties.numpy()[:, 0])
        pressures.append(e.state_0.pneumatic.absolute_pressure.numpy())
        volumes.append(e.state_0.pneumatic.volume.numpy())
        if i % 100 == 99:
            print(
                "FRAME",
                i,
                e.active_phase_name,
                "step",
                np.mean([r[1] for r in rows[-80:]]) * 1000,
                "render",
                np.mean([r[2] for r in rows[-80:]]) * 1000,
                flush=True,
            )
        if o.screenshots and not o.no_render and i in (150, 360, 607):
            import pyglet

            gl = v.renderer.gl
            width, height = v.renderer._screen_width, v.renderer._screen_height
            pixels = ctypes.create_string_buffer(width * height * 3)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, v.renderer._frame_fbo)
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
            gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, pixels)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            pyglet.image.ImageData(width, height, "RGB", pixels.raw, pitch=width * 3).save(
                str(folder / f"{o.output}_{i}.png")
            )
        if o.trace and i in (302, 502):
            cudart.cudaProfilerStop()
    try:
        e.test_final()
    except AssertionError as exc:
        failures.append(("final", str(exc)))
    # test_final stops at the existing volume failure; check plasticity separately.
    offset = e.model.edge_rest_angle.numpy() - e.authored_edge_rest_angle.numpy()
    authored = e.authored_edge_bending_properties.numpy()[:, 0]
    bending = e.model.edge_bending_properties.numpy()[:, 0]
    checks = {
        "finite_positions": bool(np.all(np.isfinite(positions))),
        "finite_pressure": bool(np.all(np.isfinite(pressures))),
        "plastic_angle_bound": bool(
            np.all(np.isfinite(offset)) and np.max(np.abs(offset)) <= e.plastic_max_angle + 1.0e-5
        ),
        "plastic_stiffness_bound": bool(
            np.all(np.isfinite(bending))
            and np.all(bending >= authored)
            and np.all(bending <= authored * (1.0 + e.plastic_hardening) + 1.0e-5)
        ),
    }
    (folder / (o.output + "_checks.json")).write_text(json.dumps(checks, indent=2))
    for name, passed in checks.items():
        if not passed:
            failures.append(("independent", name))
    print(
        "METRICS",
        e.initial_bag_center_z,
        e.lifted_bag_center_z,
        e._bag_center_z(),
        float(np.max(np.abs(e.model.edge_rest_angle.numpy() - e.authored_edge_rest_angle.numpy()))),
        e.plastic_max_angle,
        "root_position_error",
        e.maximum_root_position_error,
        "root_angle_error_deg",
        np.degrees(e.maximum_root_angle_error),
        flush=True,
    )
    metrics = {
        "minimum_volume_ratio": e.minimum_volume_ratio,
        "maximum_pressure_pa": e.maximum_pressure,
        "maximum_root_position_error_m": e.maximum_root_position_error,
        "maximum_root_angle_error_deg": float(np.degrees(e.maximum_root_angle_error)),
        "initial_bag_center_z_m": e.initial_bag_center_z,
        "lifted_bag_center_z_m": e.lifted_bag_center_z,
        "final_bag_center_z_m": e._bag_center_z(),
        "failure_count": len(failures),
    }
    (folder / (o.output + "_metrics.json")).write_text(json.dumps(metrics, indent=2) + "\n")
    print("FINAL", e.minimum_volume_ratio, e.maximum_pressure, "failures", len(failures), flush=True)
finally:
    (folder / (o.output + "_failures.json")).write_text(json.dumps(failures))
    (folder / (o.output + ".json")).write_text(json.dumps(rows))
    np.savez_compressed(
        folder / (o.output + ".npz"),
        positions=positions,
        bodies=bodies,
        plastic=plastic,
        stiffness=stiffness,
        pressure=pressures,
        volume=volumes,
    )
    v.close()

# A diagnostic run continues to collect data but still reports failed acceptance.
if failures:
    raise SystemExit(1)

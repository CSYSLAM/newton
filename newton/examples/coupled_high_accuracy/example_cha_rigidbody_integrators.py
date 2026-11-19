# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare four free-rigid-body integrators on Dzhanibekov T handles.

Command: uv run python -m newton.examples cha_dzhanibekov_integrators
"""

import numpy as np
import warp as wp

import newton
from newton._src.solvers.coupled_high_accuracy.rigid_body_dynamics import integrate_rigid_body
from newton.solvers import coupled_high_accuracy as s3d
from newton.viewer import ViewerGL

FPS = 60
FRAME_DT = 1.0 / FPS

STEM_RADIUS = 0.10
STEM_HALF_HEIGHT = 0.5
STEM_DENSITY = 50.0
BAR_RADIUS = 0.20
BAR_HALF_HEIGHT = 1.0
BAR_DENSITY = 100.0

INITIAL_ANGULAR_VELOCITY = wp.vec3(0.01, 20.0, 0.0)
BODY_SPACING = 2.8

MODE_SPECS = (
    (
        "Explicit Euler",
        s3d.InertiaMode.EXPLICIT_EULER,
        wp.vec3(0.82, 0.20, 0.12),
        wp.vec3(1.00, 0.58, 0.12),
    ),
    (
        "Implicit Euler",
        s3d.InertiaMode.IMPLICIT_EULER,
        wp.vec3(0.10, 0.38, 0.78),
        wp.vec3(0.12, 0.72, 0.92),
    ),
    (
        "LIE-group midpoint",
        s3d.InertiaMode.LIE_GROUP_MIDPOINT,
        wp.vec3(0.12, 0.58, 0.26),
        wp.vec3(0.58, 0.82, 0.18),
    ),
    (
        "Midpoint Cayley",
        s3d.InertiaMode.MIDPOINT_CAYLEY,
        wp.vec3(0.48, 0.22, 0.68),
        wp.vec3(0.88, 0.38, 0.64),
    ),
)


@wp.kernel
def _initialize_angular_momenta(
    body_indices: wp.array[wp.int32],
    body_inertia: wp.array[wp.mat33],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    angular_momenta: wp.array[wp.vec3],
):
    index = wp.tid()
    body = body_indices[index]
    rotation = wp.transform_get_rotation(body_q[body])
    world_angular_velocity = wp.spatial_bottom(body_qd[body])
    body_angular_velocity = wp.quat_rotate_inv(rotation, world_angular_velocity)
    angular_momenta[index] = body_inertia[body] * body_angular_velocity


@wp.kernel
def _advance_free_bodies(
    body_indices: wp.array[wp.int32],
    integrator_codes: wp.array[wp.int32],
    body_inertia: wp.array[wp.mat33],
    body_com: wp.array[wp.vec3],
    dt: float,
    angular_momenta: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    index = wp.tid()
    body = body_indices[index]
    pose = body_q[body]
    rotation = wp.transform_get_rotation(pose)
    linear_velocity = wp.spatial_top(body_qd[body])
    center = wp.transform_get_translation(pose) + wp.quat_rotate(rotation, body_com[body])
    next_momentum, relative_rotation_matrix = integrate_rigid_body(
        body_inertia[body],
        angular_momenta[index],
        dt,
        integrator_codes[index],
    )
    next_rotation = wp.normalize(rotation * wp.quat_from_matrix(relative_rotation_matrix))
    next_body_angular_velocity = wp.inverse(body_inertia[body]) * next_momentum
    next_world_angular_velocity = wp.quat_rotate(next_rotation, next_body_angular_velocity)
    next_center = center + dt * linear_velocity

    angular_momenta[index] = next_momentum
    body_q[body] = wp.transform(
        next_center - wp.quat_rotate(next_rotation, body_com[body]),
        next_rotation,
    )
    body_qd[body] = wp.spatial_vector(linear_velocity, next_world_angular_velocity)


def _t_handle_center_of_mass() -> np.ndarray:
    bar_mass = 2.0 * np.pi * BAR_RADIUS**2 * BAR_HALF_HEIGHT * BAR_DENSITY
    stem_mass = 2.0 * np.pi * STEM_RADIUS**2 * STEM_HALF_HEIGHT * STEM_DENSITY
    bar_center = np.array((0.0, 0.0, 0.0), dtype=np.float32)
    stem_center = np.array((0.0, -(BAR_RADIUS + STEM_HALF_HEIGHT), 0.0), dtype=np.float32)
    return (bar_mass * bar_center + stem_mass * stem_center) / (bar_mass + stem_mass)


def _build_t_handle(
    label: str,
    bar_color: wp.vec3,
    stem_color: wp.vec3,
) -> tuple[newton.ModelBuilder, int]:
    """Build one T handle at the shared physical initial pose."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
    center_of_mass = _t_handle_center_of_mass()
    bar_center = np.array((0.0, 0.0, 0.0), dtype=np.float32)
    stem_center = np.array((0.0, -(BAR_RADIUS + STEM_HALF_HEIGHT), 0.0), dtype=np.float32)
    bar_rotation = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -0.5 * wp.pi)
    stem_rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * wp.pi)

    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()),
        label=label,
    )

    bar_cfg = builder.default_shape_cfg.copy()
    bar_cfg.density = BAR_DENSITY
    bar_cfg.has_shape_collision = False
    builder.add_shape_cylinder(
        body,
        xform=wp.transform(wp.vec3(*(bar_center - center_of_mass)), bar_rotation),
        radius=BAR_RADIUS,
        half_height=BAR_HALF_HEIGHT,
        cfg=bar_cfg,
        color=bar_color,
    )

    stem_cfg = builder.default_shape_cfg.copy()
    stem_cfg.density = STEM_DENSITY
    stem_cfg.has_shape_collision = False
    builder.add_shape_cylinder(
        body,
        xform=wp.transform(wp.vec3(*(stem_center - center_of_mass)), stem_rotation),
        radius=STEM_RADIUS,
        half_height=STEM_HALF_HEIGHT,
        cfg=stem_cfg,
        color=stem_color,
    )
    builder.body_qd[body] = wp.spatial_vector(
        0.0,
        0.0,
        0.0,
        INITIAL_ANGULAR_VELOCITY[0],
        INITIAL_ANGULAR_VELOCITY[1],
        INITIAL_ANGULAR_VELOCITY[2],
    )
    return builder, body


def build_model(device: wp.Device) -> tuple[newton.Model, list[int]]:
    """Build four physically coincident T handles in independent worlds."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=wp.vec3(0.0))
    body_indices = []
    for label, _mode, bar_color, stem_color in MODE_SPECS:
        handle_builder, handle_body = _build_t_handle(
            label,
            bar_color,
            stem_color,
        )
        body_indices.append(builder.body_count + handle_body)
        builder.add_world(handle_builder)

    builder.add_ground_plane()
    return builder.finalize(device=device), body_indices


def _check_identical_initial_conditions(
    model: newton.Model,
    state: newton.State,
    bodies: list[int],
) -> None:
    fields = {
        "pose": state.body_q.numpy()[bodies],
        "velocity": state.body_qd.numpy()[bodies],
        "center of mass": model.body_com.numpy()[bodies],
        "mass": model.body_mass.numpy()[bodies],
        "inertia": model.body_inertia.numpy()[bodies],
    }
    mismatched = [
        name for name, values in fields.items() if not np.array_equal(values, np.broadcast_to(values[0], values.shape))
    ]
    if mismatched:
        raise RuntimeError("T handles have different initial conditions: " + ", ".join(mismatched))


def main() -> None:
    """Integrate the four handles directly and display them in ViewerGL."""
    device = wp.get_device("cuda:0" if wp.is_cuda_available() else "cpu")
    model, bodies = build_model(device)
    initial_state = model.state()
    state = model.state()
    state.assign(initial_state)
    _check_identical_initial_conditions(model, initial_state, bodies)
    body_indices = wp.array(bodies, dtype=wp.int32, device=device)
    integrator_codes = wp.array(
        [int(mode.value) for _label, mode, _bar, _stem in MODE_SPECS],
        dtype=wp.int32,
        device=device,
    )
    angular_momenta = wp.empty(len(bodies), dtype=wp.vec3, device=device)
    wp.launch(
        _initialize_angular_momenta,
        dim=len(bodies),
        inputs=[body_indices, model.body_inertia, state.body_q, state.body_qd],
        outputs=[angular_momenta],
        device=device,
    )

    for index, (label, _mode, _bar, _stem) in enumerate(MODE_SPECS):
        print(f"x={(-1.5 + index) * BODY_SPACING:+.1f}: {label}")

    viewer = ViewerGL(vsync=False)
    viewer.set_model(model)
    viewer.set_world_offsets((0.0, BODY_SPACING, 0.0))
    viewer.set_camera(pos=wp.vec3(15.0, 0.0, 3.0), pitch=0.0, yaw=180.0)
    sim_time = 0.0

    def reset() -> None:
        nonlocal sim_time
        state.assign(initial_state)
        wp.launch(
            _initialize_angular_momenta,
            dim=len(bodies),
            inputs=[body_indices, model.body_inertia, state.body_q, state.body_qd],
            outputs=[angular_momenta],
            device=device,
        )
        sim_time = 0.0

    viewer.set_reset_callback(reset)

    try:
        while viewer.is_running():
            if viewer.should_step():
                wp.launch(
                    _advance_free_bodies,
                    dim=len(bodies),
                    inputs=[
                        body_indices,
                        integrator_codes,
                        model.body_inertia,
                        model.body_com,
                        FRAME_DT,
                    ],
                    outputs=[angular_momenta, state.body_q, state.body_qd],
                    device=device,
                )
                sim_time += FRAME_DT

            viewer.begin_frame(sim_time)
            viewer.log_state(state)
            viewer.end_frame()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()

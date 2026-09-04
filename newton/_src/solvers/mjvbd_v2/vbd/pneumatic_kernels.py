# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels used by VBD pneumatic cavities."""

from __future__ import annotations

import warp as wp

_PNEUMATIC_CAVITY_INITIALIZE_BLOCK_DIM = 256
_PNEUMATIC_CAVITY_UPDATE_BLOCK_DIM = 128
_PNEUMATIC_SINGLE_CAVITY_FUSED_MAX_PARTICLES = 512


@wp.func
def _cavity_face_volume_contribution_from_anchor(
    face: int,
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    face_triangle: wp.array[wp.int32],
    face_sign: wp.array[float],
    anchor: wp.vec3,
):
    triangle = face_triangle[face]
    p0 = particle_q[tri_indices[triangle, 0]] - anchor
    p1 = particle_q[tri_indices[triangle, 1]] - anchor
    p2 = particle_q[tri_indices[triangle, 2]] - anchor
    return face_sign[face] * wp.dot(p0, wp.cross(p1, p2)) / 6.0


@wp.func
def _cavity_face_volume_contribution(
    face: int,
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    face_cavity: wp.array[wp.int32],
    face_triangle: wp.array[wp.int32],
    face_sign: wp.array[float],
    cavity_anchor_positions: wp.array[wp.vec3],
):
    return _cavity_face_volume_contribution_from_anchor(
        face,
        particle_q,
        tri_indices,
        face_triangle,
        face_sign,
        cavity_anchor_positions[face_cavity[face]],
    )


@wp.kernel
def accumulate_cavity_volume(
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    face_cavity: wp.array[wp.int32],
    face_triangle: wp.array[wp.int32],
    face_sign: wp.array[float],
    anchor_particle: wp.array[wp.int32],
    volumes: wp.array[float],
):
    """Accumulate signed tetrahedron volumes for all cavity faces."""
    face = wp.tid()
    cavity = face_cavity[face]
    triangle = face_triangle[face]
    anchor = particle_q[anchor_particle[cavity]]
    p0 = particle_q[tri_indices[triangle, 0]] - anchor
    p1 = particle_q[tri_indices[triangle, 1]] - anchor
    p2 = particle_q[tri_indices[triangle, 2]] - anchor
    wp.atomic_add(volumes, cavity, face_sign[face] * wp.dot(p0, wp.cross(p1, p2)) / 6.0)


@wp.kernel
def evaluate_pressure(
    dt: float,
    mode: wp.array[wp.int32],
    rest_volume: wp.array[float],
    reference_absolute_pressure: wp.array[float],
    ambient_pressure: wp.array[float],
    heat_capacity_ratio: wp.array[float],
    target_volume: wp.array[float],
    volume_stiffness: wp.array[float],
    bulk_damping: wp.array[float],
    min_volume: wp.array[float],
    max_absolute_pressure: wp.array[float],
    current_volume: wp.array[float],
    previous_volume: wp.array[float],
    pressure_scale: wp.array[float],
    prescribed_gauge_pressure: wp.array[float],
    target_volume_scale: wp.array[float],
    absolute_pressure: wp.array[float],
    gauge_pressure: wp.array[float],
    curvature: wp.array[float],
    volume_rate: wp.array[float],
    clamp_flags: wp.array[wp.int32],
):
    """Evaluate the pressure law and its positive scalar curvature."""
    cavity = wp.tid()
    _evaluate_pressure_for_cavity(
        cavity,
        dt,
        mode,
        rest_volume,
        reference_absolute_pressure,
        ambient_pressure,
        heat_capacity_ratio,
        target_volume,
        volume_stiffness,
        bulk_damping,
        min_volume,
        max_absolute_pressure,
        current_volume,
        previous_volume,
        pressure_scale,
        prescribed_gauge_pressure,
        target_volume_scale,
        absolute_pressure,
        gauge_pressure,
        curvature,
        volume_rate,
        clamp_flags,
    )


@wp.func
def _evaluate_pressure_for_cavity(
    cavity: int,
    dt: float,
    mode: wp.array[wp.int32],
    rest_volume: wp.array[float],
    reference_absolute_pressure: wp.array[float],
    ambient_pressure: wp.array[float],
    heat_capacity_ratio: wp.array[float],
    target_volume: wp.array[float],
    volume_stiffness: wp.array[float],
    bulk_damping: wp.array[float],
    min_volume: wp.array[float],
    max_absolute_pressure: wp.array[float],
    current_volume: wp.array[float],
    previous_volume: wp.array[float],
    pressure_scale: wp.array[float],
    prescribed_gauge_pressure: wp.array[float],
    target_volume_scale: wp.array[float],
    absolute_pressure: wp.array[float],
    gauge_pressure: wp.array[float],
    curvature: wp.array[float],
    volume_rate: wp.array[float],
    clamp_flags: wp.array[wp.int32],
):
    """Evaluate one cavity pressure from its current volume."""
    raw_volume = current_volume[cavity]
    minimum = min_volume[cavity]
    volume = wp.max(raw_volume, minimum)
    flags = wp.int32(0)
    if raw_volume < minimum:
        flags = flags | wp.int32(1)

    pressure_scale_value = pressure_scale[cavity]
    rate = (raw_volume - previous_volume[cavity]) / dt
    pressure = reference_absolute_pressure[cavity]
    gauge = 0.0
    pressure_curvature = 0.0
    mode_value = mode[cavity]

    if mode_value == 0:
        raw_pressure = reference_absolute_pressure[cavity] * rest_volume[cavity] / volume
        gauge = pressure_scale_value * (raw_pressure - ambient_pressure[cavity])
        pressure = ambient_pressure[cavity] + gauge
        pressure_curvature = wp.max(pressure_scale_value, 0.0) * raw_pressure / volume
    elif mode_value == 1:
        raw_pressure = reference_absolute_pressure[cavity] * wp.pow(
            rest_volume[cavity] / volume, heat_capacity_ratio[cavity]
        )
        gauge = pressure_scale_value * (raw_pressure - ambient_pressure[cavity])
        pressure = ambient_pressure[cavity] + gauge
        pressure_curvature = wp.max(pressure_scale_value, 0.0) * heat_capacity_ratio[cavity] * raw_pressure / volume
    elif mode_value == 2:
        target = target_volume[cavity] * target_volume_scale[cavity]
        pressure_curvature = wp.max(pressure_scale_value, 0.0) * volume_stiffness[cavity]
        gauge = pressure_scale_value * volume_stiffness[cavity] * (target - raw_volume)
        pressure = ambient_pressure[cavity] + gauge
    else:
        gauge = pressure_scale_value * prescribed_gauge_pressure[cavity]
        pressure = ambient_pressure[cavity] + gauge

    gauge = gauge - bulk_damping[cavity] * rate
    pressure = ambient_pressure[cavity] + gauge
    pressure_curvature = pressure_curvature + bulk_damping[cavity] / dt

    max_pressure = max_absolute_pressure[cavity]
    if pressure < 0.0:
        pressure = 0.0
        gauge = -ambient_pressure[cavity]
        pressure_curvature = 0.0
        flags = flags | wp.int32(2)
    if pressure > max_pressure:
        pressure = max_pressure
        gauge = max_pressure - ambient_pressure[cavity]
        pressure_curvature = 0.0
        flags = flags | wp.int32(4)

    absolute_pressure[cavity] = pressure
    gauge_pressure[cavity] = gauge
    curvature[cavity] = pressure_curvature
    volume_rate[cavity] = rate
    clamp_flags[cavity] = flags


@wp.kernel
def initialize_cavity_volume_and_pressure(
    cavity_count: int,
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    face_triangle: wp.array[wp.int32],
    face_sign: wp.array[float],
    anchor_particle: wp.array[wp.int32],
    cavity_face_offsets: wp.array[wp.int32],
    cavity_faces: wp.array[wp.int32],
    dt: float,
    mode: wp.array[wp.int32],
    rest_volume: wp.array[float],
    reference_absolute_pressure: wp.array[float],
    ambient_pressure: wp.array[float],
    heat_capacity_ratio: wp.array[float],
    target_volume: wp.array[float],
    volume_stiffness: wp.array[float],
    bulk_damping: wp.array[float],
    min_volume: wp.array[float],
    max_absolute_pressure: wp.array[float],
    previous_volume: wp.array[float],
    pressure_scale: wp.array[float],
    prescribed_gauge_pressure: wp.array[float],
    target_volume_scale: wp.array[float],
    cavity_anchor_positions: wp.array[wp.vec3],
    face_volume_contribution: wp.array[float],
    current_volume: wp.array[float],
    absolute_pressure: wp.array[float],
    gauge_pressure: wp.array[float],
    curvature: wp.array[float],
    volume_rate: wp.array[float],
    clamp_flags: wp.array[wp.int32],
):
    """Initialize cached face volumes and pressure in one block per cavity."""
    tid = wp.tid()
    block_dim = wp.block_dim()
    cavity = tid // block_dim
    lane = tid - cavity * block_dim
    anchor = particle_q[anchor_particle[cavity]]
    begin = cavity_face_offsets[cavity]
    end = cavity_face_offsets[cavity + 1]
    volume = float(0.0)
    slot = begin + lane
    while slot < end:
        face = cavity_faces[slot]
        contribution = _cavity_face_volume_contribution_from_anchor(
            face,
            particle_q,
            tri_indices,
            face_triangle,
            face_sign,
            anchor,
        )
        face_volume_contribution[face] = contribution
        volume += contribution
        slot += block_dim

    cavity_volume = wp.tile_sum(wp.tile(volume))[0]
    if lane == 0:
        cavity_anchor_positions[cavity] = anchor
        current_volume[cavity] = cavity_volume
        _evaluate_pressure_for_cavity(
            cavity,
            dt,
            mode,
            rest_volume,
            reference_absolute_pressure,
            ambient_pressure,
            heat_capacity_ratio,
            target_volume,
            volume_stiffness,
            bulk_damping,
            min_volume,
            max_absolute_pressure,
            current_volume,
            previous_volume,
            pressure_scale,
            prescribed_gauge_pressure,
            target_volume_scale,
            absolute_pressure,
            gauge_pressure,
            curvature,
            volume_rate,
            clamp_flags,
        )


@wp.kernel
def update_cavity_volume_and_pressure_by_color(
    color: int,
    cavity_count: int,
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    face_cavity: wp.array[wp.int32],
    face_triangle: wp.array[wp.int32],
    face_sign: wp.array[float],
    cavity_anchor_positions: wp.array[wp.vec3],
    color_cavity_face_offsets: wp.array[wp.int32],
    color_cavity_faces: wp.array[wp.int32],
    dt: float,
    mode: wp.array[wp.int32],
    rest_volume: wp.array[float],
    reference_absolute_pressure: wp.array[float],
    ambient_pressure: wp.array[float],
    heat_capacity_ratio: wp.array[float],
    target_volume: wp.array[float],
    volume_stiffness: wp.array[float],
    bulk_damping: wp.array[float],
    min_volume: wp.array[float],
    max_absolute_pressure: wp.array[float],
    previous_volume: wp.array[float],
    pressure_scale: wp.array[float],
    prescribed_gauge_pressure: wp.array[float],
    target_volume_scale: wp.array[float],
    face_volume_contribution: wp.array[float],
    current_volume: wp.array[float],
    absolute_pressure: wp.array[float],
    gauge_pressure: wp.array[float],
    curvature: wp.array[float],
    volume_rate: wp.array[float],
    clamp_flags: wp.array[wp.int32],
):
    """Update one color's unique faces and pressure in one block per cavity."""
    tid = wp.tid()
    block_dim = wp.block_dim()
    cavity = tid // block_dim
    lane = tid - cavity * block_dim
    row = color * cavity_count + cavity
    begin = color_cavity_face_offsets[row]
    end = color_cavity_face_offsets[row + 1]
    delta = float(0.0)
    slot = begin + lane
    while slot < end:
        face = color_cavity_faces[slot]
        previous = face_volume_contribution[face]
        current = _cavity_face_volume_contribution(
            face,
            particle_q,
            tri_indices,
            face_cavity,
            face_triangle,
            face_sign,
            cavity_anchor_positions,
        )
        face_volume_contribution[face] = current
        delta += current - previous
        slot += block_dim

    volume_delta = wp.tile_sum(wp.tile(delta))[0]
    if lane == 0:
        current_volume[cavity] += volume_delta
        _evaluate_pressure_for_cavity(
            cavity,
            dt,
            mode,
            rest_volume,
            reference_absolute_pressure,
            ambient_pressure,
            heat_capacity_ratio,
            target_volume,
            volume_stiffness,
            bulk_damping,
            min_volume,
            max_absolute_pressure,
            current_volume,
            previous_volume,
            pressure_scale,
            prescribed_gauge_pressure,
            target_volume_scale,
            absolute_pressure,
            gauge_pressure,
            curvature,
            volume_rate,
            clamp_flags,
        )


@wp.func
def _accumulate_pressure_force_for_particle(
    particle: int,
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    face_cavity: wp.array[wp.int32],
    face_triangle: wp.array[wp.int32],
    face_sign: wp.array[float],
    particle_face_offsets: wp.array[wp.int32],
    particle_faces: wp.array[wp.int32],
    gauge_pressure: wp.array[float],
    curvature: wp.array[float],
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
):
    force = wp.vec3(0.0)
    hessian = wp.mat33(0.0)
    for adjacent_index in range(particle_face_offsets[particle], particle_face_offsets[particle + 1]):
        face = particle_faces[adjacent_index]
        cavity = face_cavity[face]
        triangle = face_triangle[face]
        p0 = particle_q[tri_indices[triangle, 0]]
        p1 = particle_q[tri_indices[triangle, 1]]
        p2 = particle_q[tri_indices[triangle, 2]]
        gradient = face_sign[face] * wp.cross(p1 - p0, p2 - p0) / 6.0
        force += gauge_pressure[cavity] * gradient
        hessian += curvature[cavity] * wp.outer(gradient, gradient)

    particle_forces[particle] += force
    particle_hessians[particle] += hessian


@wp.kernel
def update_single_cavity_volume_pressure_and_accumulate_force(
    update_color: int,
    particle_ids: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    face_cavity: wp.array[wp.int32],
    face_triangle: wp.array[wp.int32],
    face_sign: wp.array[float],
    cavity_anchor_positions: wp.array[wp.vec3],
    color_cavity_face_offsets: wp.array[wp.int32],
    color_cavity_faces: wp.array[wp.int32],
    dt: float,
    mode: wp.array[wp.int32],
    rest_volume: wp.array[float],
    reference_absolute_pressure: wp.array[float],
    ambient_pressure: wp.array[float],
    heat_capacity_ratio: wp.array[float],
    target_volume: wp.array[float],
    volume_stiffness: wp.array[float],
    bulk_damping: wp.array[float],
    min_volume: wp.array[float],
    max_absolute_pressure: wp.array[float],
    previous_volume: wp.array[float],
    pressure_scale: wp.array[float],
    prescribed_gauge_pressure: wp.array[float],
    target_volume_scale: wp.array[float],
    particle_face_offsets: wp.array[wp.int32],
    particle_faces: wp.array[wp.int32],
    face_volume_contribution: wp.array[float],
    current_volume: wp.array[float],
    absolute_pressure: wp.array[float],
    gauge_pressure: wp.array[float],
    curvature: wp.array[float],
    volume_rate: wp.array[float],
    clamp_flags: wp.array[wp.int32],
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
):
    """Update one cavity, then accumulate the next color's pressure terms."""
    lane = wp.tid()
    block_dim = wp.block_dim()
    begin = color_cavity_face_offsets[update_color]
    end = color_cavity_face_offsets[update_color + 1]
    delta = float(0.0)
    slot = begin + lane
    while slot < end:
        face = color_cavity_faces[slot]
        previous = face_volume_contribution[face]
        current = _cavity_face_volume_contribution(
            face,
            particle_q,
            tri_indices,
            face_cavity,
            face_triangle,
            face_sign,
            cavity_anchor_positions,
        )
        face_volume_contribution[face] = current
        delta += current - previous
        slot += block_dim

    volume_delta = wp.tile_sum(wp.tile(delta))[0]
    if lane == 0:
        current_volume[0] += volume_delta
        _evaluate_pressure_for_cavity(
            0,
            dt,
            mode,
            rest_volume,
            reference_absolute_pressure,
            ambient_pressure,
            heat_capacity_ratio,
            target_volume,
            volume_stiffness,
            bulk_damping,
            min_volume,
            max_absolute_pressure,
            current_volume,
            previous_volume,
            pressure_scale,
            prescribed_gauge_pressure,
            target_volume_scale,
            absolute_pressure,
            gauge_pressure,
            curvature,
            volume_rate,
            clamp_flags,
        )

    # The reduction is a block barrier after lane zero updates pressure.
    pressure_ready = wp.tile_sum(wp.tile(float(lane == 0)))[0]
    particle_index = lane
    while particle_index < particle_ids.shape[0] and pressure_ready > 0.0:
        _accumulate_pressure_force_for_particle(
            particle_ids[particle_index],
            particle_q,
            tri_indices,
            face_cavity,
            face_triangle,
            face_sign,
            particle_face_offsets,
            particle_faces,
            gauge_pressure,
            curvature,
            particle_forces,
            particle_hessians,
        )
        particle_index += block_dim


@wp.kernel
def accumulate_pressure_force_and_hessian(
    particle_ids: wp.array[wp.int32],
    particle_q: wp.array[wp.vec3],
    tri_indices: wp.array2d[wp.int32],
    face_cavity: wp.array[wp.int32],
    face_triangle: wp.array[wp.int32],
    face_sign: wp.array[float],
    particle_face_offsets: wp.array[wp.int32],
    particle_faces: wp.array[wp.int32],
    gauge_pressure: wp.array[float],
    curvature: wp.array[float],
    particle_forces: wp.array[wp.vec3],
    particle_hessians: wp.array[wp.mat33],
):
    """Accumulate pressure terms once for each particle in a color group."""
    _accumulate_pressure_force_for_particle(
        particle_ids[wp.tid()],
        particle_q,
        tri_indices,
        face_cavity,
        face_triangle,
        face_sign,
        particle_face_offsets,
        particle_faces,
        gauge_pressure,
        curvature,
        particle_forces,
        particle_hessians,
    )


@wp.kernel
def reset_pneumatic_state(
    world_mask: wp.array[wp.bool],
    reset_all: bool,
    world_count: int,
    cavity_world: wp.array[wp.int32],
    rest_volume: wp.array[float],
    reference_absolute_pressure: wp.array[float],
    volume: wp.array[float],
    absolute_pressure: wp.array[float],
    volume_rate: wp.array[float],
    clamp_flags: wp.array[wp.int32],
):
    """Restore pneumatic observables for reset-selected worlds."""
    cavity = wp.tid()
    world = cavity_world[cavity]
    selected = reset_all
    if not reset_all:
        world_index = world
        if world_index < 0:
            if world_mask.shape[0] == world_count:
                selected = False
            else:
                world_index = world_count
                selected = world_mask[world_index]
        else:
            selected = world_mask[world_index]
    if selected:
        volume[cavity] = rest_volume[cavity]
        absolute_pressure[cavity] = reference_absolute_pressure[cavity]
        volume_rate[cavity] = 0.0
        clamp_flags[cavity] = 0

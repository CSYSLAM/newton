# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reuse identical middle sweep blocks in a device-side CUDA graph loop."""

import warp as wp


@wp.kernel
def set_count(value: int, count: wp.array[int]):
    count[0] = value


@wp.kernel
def decrement(count: wp.array[int]):
    count[0] -= 1


def install_iteration_loop(solver, block=4):
    """Keep first/last blocks explicit and repeat only periodic middle work."""
    if solver.iterations % block or solver.iterations < 3 * block:
        raise ValueError("Loop probe needs at least three complete blocks")
    if solver.particle_chebyshev_enabled or solver._pneumatic_enabled:
        raise ValueError("Loop probe has not validated Chebyshev/pneumatic bookkeeping")
    if solver.particle_collision_detection_interval not in (-1, 0):
        raise ValueError("Loop probe requires substep/first-sweep collision refresh")
    if solver.particle_multilevel_fallback_iterations > solver.iterations:
        raise ValueError("Loop probe requires no extra fallback sweeps")
    expected = tuple(range(block, solver.iterations + 1, block))
    if tuple(solver.particle_multilevel_checkpoints) != expected:
        raise ValueError("Loop probe needs periodic coarse checkpoints")
    rigid = solver._solve_rigid_body_iteration
    particle = solver._solve_particle_iteration
    count = wp.zeros(1, dtype=int, device=solver.device)

    def ignored_rigid(*args, **kwargs):
        return None

    def solve_particles(state_in, state_out, control, contacts, dt, iteration):
        if iteration != 0:
            return

        def solve_block(start):
            for offset in range(block):
                rigid(state_in, state_out, control, contacts, dt)
                particle(state_in, state_out, control, contacts, dt, start + offset)

        if not solver.device.is_capturing:
            for start in range(0, solver.iterations, block):
                solve_block(start)
            return
        solve_block(0)
        wp.launch(set_count, dim=1, inputs=[solver.iterations // block - 2, count], device=solver.device)

        def middle():
            solve_block(block)
            wp.launch(decrement, dim=1, inputs=[count], device=solver.device)

        wp.capture_while(count, middle)
        solve_block(solver.iterations - block)

    solver._solve_rigid_body_iteration = ignored_rigid
    solver._solve_particle_iteration = solve_particles

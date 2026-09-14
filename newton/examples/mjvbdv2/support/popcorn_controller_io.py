# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Batch controller transfers without changing targets or snapshot lifetimes."""

import warp as wp


@wp.kernel(enable_backward=False)
def write_wrist_targets(
    p0: wp.vec3,
    p1: wp.vec3,
    r0: wp.vec4,
    r1: wp.vec4,
    positions0: wp.array[wp.vec3],
    positions1: wp.array[wp.vec3],
    rotations0: wp.array[wp.vec4],
    rotations1: wp.array[wp.vec4],
):
    positions0[0] = p0
    positions1[0] = p1
    rotations0[0] = r0
    rotations1[0] = r1


@wp.kernel(enable_backward=False)
def pack_state(particles: wp.array[float], bodies: wp.array[float], joints: wp.array[float], output: wp.array[float]):
    i = wp.tid()
    first, second = particles.shape[0], particles.shape[0] + bodies.shape[0]
    if i < first:
        output[i] = particles[i]
    elif i < second:
        output[i] = bodies[i - first]
    else:
        output[i] = joints[i - second]


class ControllerSnapshot:
    """Copy the same float32 state once per controller phase, never across steps."""

    def __init__(self, state):
        self.counts = (state.particle_q.size * 3, state.body_q.size * 7, state.joint_q.size)
        self.buffer = wp.empty(sum(self.counts), dtype=float, device=state.particle_q.device)

    def read(self, state):
        wp.launch(
            pack_state,
            self.buffer.size,
            [
                state.particle_q.view(wp.float32).flatten(),
                state.body_q.view(wp.float32).flatten(),
                state.joint_q,
                self.buffer,
            ],
            device=self.buffer.device,
        )
        values = self.buffer.numpy()
        if self.buffer.device.is_cpu:
            values = values.copy()
        values.setflags(write=False)
        first, bodies, _ = self.counts
        return {
            "particle_q": values[:first].reshape(-1, 3),
            "body_q": values[first : first + bodies].reshape(-1, 7),
            "joint_q": values[first + bodies :],
        }

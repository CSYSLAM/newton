"""Particle force kernels shared by the W1 package scenes."""

import warp as wp


@wp.kernel
def viscous_drag(velocity: wp.array[wp.vec3], mass: wp.array[float], rate: float, force: wp.array[wp.vec3]):
    i = wp.tid()
    force[i] = -rate * mass[i] * velocity[i]

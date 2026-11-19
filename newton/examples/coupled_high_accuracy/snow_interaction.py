# SPDX-FileCopyrightText: Copyright (c) 2026 The Style3D Sim Developers
# SPDX-License-Identifier: Apache-2.0

"""Viewer picking for a snowball composed entirely of granular particles."""

import math

import numpy as np
import warp as wp

from newton._src.viewer.kernels import PickingState


@wp.kernel
def _accumulate_seed_motion(
    positions: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    masses: wp.array[float],
    inverse_mass: float,
    center: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
):
    i = wp.tid()
    weight = masses[i] * inverse_mass
    wp.atomic_add(center, 0, weight * positions[i])
    wp.atomic_add(velocity, 0, weight * velocities[i])


@wp.kernel
def _pick_seed(
    positions: wp.array[wp.vec3],
    radii: wp.array[float],
    center: wp.array[wp.vec3],
    seed_count: int,
    origin: wp.vec3,
    direction: wp.vec3,
    pick_body: wp.array[int],
    pick_distance: wp.array[float],
    pick_state: wp.array[PickingState],
):
    distance = float(1.0e10)
    # Clicks are infrequent; a serial search avoids locks and extra per-frame raycast storage.
    for i in range(seed_count):
        delta = origin - positions[i]
        along = wp.dot(delta, direction)
        radius = radii[i]
        discriminant = along * along - wp.dot(delta, delta) + radius * radius
        if discriminant >= 0.0:
            candidate = -along - wp.sqrt(discriminant)
            if candidate < 0.0:
                candidate = -along + wp.sqrt(discriminant)
            if candidate >= 0.0 and candidate < distance:
                distance = candidate
    pick_body[0] = -1
    pick_distance[0] = distance
    if distance < 1.0e10:
        point = origin + distance * direction
        state = pick_state[0]
        # Keep the clicked offset so pressing the mouse never snaps the center to the surface.
        state.picked_point_local = point - center[0]
        state.picked_point_world = point
        state.picking_target_world = point
        pick_state[0] = state
        pick_body[0] = 0


@wp.kernel
def _update_target(
    origin: wp.vec3,
    direction: wp.vec3,
    distance: float,
    pick_state: wp.array[PickingState],
):
    state = pick_state[0]
    state.picking_target_world = origin + distance * direction
    pick_state[0] = state


@wp.kernel
def _compute_drag_acceleration(
    center: wp.array[wp.vec3],
    velocity: wp.array[wp.vec3],
    pick_body: wp.array[int],
    pick_state: wp.array[PickingState],
    acceleration: wp.array[wp.vec3],
):
    acceleration[0] = wp.vec3(0.0)
    if pick_body[0] >= 0:
        state = pick_state[0]
        point = center[0] + state.picked_point_local
        state.picked_point_world = point
        value = state.pick_stiffness * (state.picking_target_world - point) - state.pick_damping * velocity[0]
        magnitude = wp.length(value)
        limit = state.pick_max_acceleration * 9.81
        if magnitude > limit:
            value *= limit / magnitude
        acceleration[0] = value
        pick_state[0] = state


@wp.kernel
def _apply_seed_force(masses: wp.array[float], acceleration: wp.array[wp.vec3], forces: wp.array[wp.vec3]):
    i = wp.tid()
    # A common acceleration pulls the seed without imposing a shape or replacing the granular solve.
    forces[i] += masses[i] * acceleration[0]


@wp.kernel
def _copy_picking_line(pick_state: wp.array[PickingState], starts: wp.array[wp.vec3], ends: wp.array[wp.vec3]):
    starts[0] = pick_state[0].picked_point_world
    ends[0] = pick_state[0].picking_target_world


class SnowballPicking:
    """Match the viewer picking protocol while applying forces to the initial snowball particles."""

    def __init__(
        self,
        model,
        seed_count: int,
        pick_stiffness: float = 80.0,
        pick_damping: float = 14.0,
        pick_max_acceleration: float = 4.0,
    ):
        if seed_count < 1 or seed_count > model.particle_count:
            raise ValueError("seed_count must select a nonempty prefix of the particle array")
        parameters = (pick_stiffness, pick_damping, pick_max_acceleration)
        if not all(math.isfinite(value) and value >= 0.0 for value in parameters):
            raise ValueError("Picking stiffness, damping, and maximum acceleration must be finite and nonnegative")
        self.model = model
        self.seed_count = seed_count
        total_mass = float(model.particle_mass.numpy()[:seed_count].sum(dtype=np.float64))
        if total_mass <= 0.0:
            raise ValueError("The snowball must have positive mass")
        self.inverse_mass = 1.0 / total_mass
        self.center = wp.zeros(1, dtype=wp.vec3, device=model.device)
        self.velocity = wp.zeros(1, dtype=wp.vec3, device=model.device)
        self.acceleration = wp.zeros(1, dtype=wp.vec3, device=model.device)
        self.pick_body = wp.array([-1], dtype=int, device=model.device)
        self._pick_distance = wp.zeros(1, dtype=float, device=model.device)
        state = np.zeros(1, dtype=PickingState.numpy_dtype())
        state[0]["pick_stiffness"] = pick_stiffness
        state[0]["pick_damping"] = pick_damping
        state[0]["pick_max_acceleration"] = pick_max_acceleration
        self.pick_state = wp.array(state, dtype=PickingState, device=model.device)
        self.pick_dist = 0.0
        self.picking_active = False
        self.world_offsets = None
        self.visible_worlds_mask = None
        self._viewer = None
        self._line_starts = wp.zeros(1, dtype=wp.vec3, device=model.device)
        self._line_ends = wp.zeros(1, dtype=wp.vec3, device=model.device)
        self._line_colors = wp.array([wp.vec3(0.0, 1.0, 1.0)], dtype=wp.vec3, device=model.device)

    def attach(self, viewer):
        """Retain stock right-button callbacks and supply a line renderer that needs no rigid body."""
        self._viewer = viewer
        viewer.picking = self
        viewer.picking_enabled = True
        if hasattr(viewer, "_render_picking_line"):
            # The stock line renderer indexes body_world, but this example contains no rigid bodies.
            viewer._render_picking_line = self._render_picking_line

    def is_picking(self) -> bool:
        return self.picking_active

    def release(self):
        self.pick_body.fill_(-1)
        self.picking_active = False

    @staticmethod
    def _ray(ray_start, ray_dir):
        direction = np.asarray(ray_dir, dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if not math.isfinite(length) or length <= 0.0:
            raise ValueError("Picking ray must have a finite nonzero direction")
        direction /= length
        return wp.vec3(*ray_start), wp.vec3(*direction)

    def _update_motion(self, state):
        self.center.zero_()
        self.velocity.zero_()
        wp.launch(
            _accumulate_seed_motion,
            dim=self.seed_count,
            inputs=[
                state.particle_q,
                state.particle_qd,
                self.model.particle_mass,
                self.inverse_mass,
            ],
            outputs=[self.center, self.velocity],
            device=self.model.device,
        )

    def pick(self, state, ray_start, ray_dir):
        origin, direction = self._ray(ray_start, ray_dir)
        self._update_motion(state)
        wp.launch(
            _pick_seed,
            dim=1,
            inputs=[
                state.particle_q,
                self.model.particle_radius,
                self.center,
                self.seed_count,
                origin,
                direction,
            ],
            outputs=[self.pick_body, self._pick_distance, self.pick_state],
            device=self.model.device,
        )
        self.pick_dist = float(self._pick_distance.numpy()[0])
        self.picking_active = self.pick_dist < 1.0e10

    def update(self, ray_start, ray_dir):
        if self.picking_active:
            origin, direction = self._ray(ray_start, ray_dir)
            wp.launch(
                _update_target,
                dim=1,
                inputs=[origin, direction, self.pick_dist, self.pick_state],
                device=self.model.device,
            )

    def _apply_picking_force(self, state):
        # The device flag controls activation so picking also works after CUDA graph capture.
        self._update_motion(state)
        wp.launch(
            _compute_drag_acceleration,
            dim=1,
            inputs=[self.center, self.velocity, self.pick_body, self.pick_state],
            outputs=[self.acceleration],
            device=self.model.device,
        )
        wp.launch(
            _apply_seed_force,
            dim=self.seed_count,
            inputs=[self.model.particle_mass, self.acceleration],
            outputs=[state.particle_f],
            device=self.model.device,
        )

    def _render_picking_line(self, state):
        if not self.picking_active:
            self._viewer.log_lines("picking_line", None, None, None)
            return
        wp.launch(
            _copy_picking_line,
            dim=1,
            inputs=[self.pick_state],
            outputs=[self._line_starts, self._line_ends],
            device=self.model.device,
        )
        self._viewer.log_lines("picking_line", self._line_starts, self._line_ends, self._line_colors, hidden=False)

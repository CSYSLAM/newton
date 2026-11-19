"""Geometric ground-contact tracking for SynReal G1 locomotion."""

from __future__ import annotations

from typing import Any

import torch
import warp as wp


@wp.kernel
def clear_raw_body_contacts(raw_body_contact: wp.array[wp.int32]) -> None:
    raw_body_contact[wp.tid()] = 0


@wp.kernel
def aggregate_ground_contacts(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[wp.float32],
    contact_margin1: wp.array[wp.float32],
    shape_body: wp.array[wp.int32],
    body_world: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    maximum_separation: float,
    raw_body_contact: wp.array[wp.int32],
) -> None:
    contact = wp.tid()
    if contact >= contact_count[0]:
        return

    shape0 = contact_shape0[contact]
    shape1 = contact_shape1[contact]
    if shape0 < 0 or shape1 < 0:
        return
    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    if (body0 < 0 and body1 < 0) or (body0 >= 0 and body1 >= 0):
        return

    point0_world = contact_point0[contact]
    point1_world = contact_point1[contact]
    if body0 >= 0:
        point0_world = wp.transform_point(body_q[body0], point0_world)
    if body1 >= 0:
        point1_world = wp.transform_point(body_q[body1], point1_world)
    separation = (
        wp.dot(contact_normal[contact], point1_world - point0_world)
        - contact_margin0[contact]
        - contact_margin1[contact]
    )
    if separation > maximum_separation:
        return

    robot_body = body0
    if robot_body < 0:
        robot_body = body1
    if body_world[robot_body] >= 0:
        wp.atomic_max(raw_body_contact, robot_body, 1)


@wp.kernel
def update_contact_state(
    raw_body_contact: wp.array[wp.int32],
    debounce_counter: wp.array[wp.int32],
    stable_body_contact: wp.array[wp.int32],
    history_mask: wp.array[wp.int32],
    contact_time: wp.array[wp.float32],
    air_time: wp.array[wp.float32],
    debounce_substeps: int,
    history_bits: int,
    dt: float,
) -> None:
    body = wp.tid()
    counter = 0
    if raw_body_contact[body] != 0:
        counter = debounce_counter[body] + 1
        counter = min(counter, debounce_substeps)
    debounce_counter[body] = counter

    stable = 0
    if counter >= debounce_substeps:
        stable = 1
        contact_time[body] = contact_time[body] + dt
        air_time[body] = 0.0
    else:
        contact_time[body] = 0.0
        air_time[body] = air_time[body] + dt
    stable_body_contact[body] = stable
    history_mask[body] = ((history_mask[body] << 1) | stable) & history_bits


class GeometryContactTracker:
    """Track debounced body-ground contact at every physics substep."""

    def __init__(
        self,
        model: Any,
        contacts: Any,
        *,
        num_envs: int,
        bodies_per_env: int,
        foot_body_indices: tuple[int, int],
        undesired_body_indices: tuple[int, ...],
        debounce_substeps: int,
        history_substeps: int,
        maximum_separation: float,
        dt: float,
        device: Any,
        stream: Any,
    ) -> None:
        if debounce_substeps < 1:
            raise ValueError("debounce_substeps must be at least one")
        if not 1 <= history_substeps <= 30:
            raise ValueError("history_substeps must be in [1, 30]")
        self.model = model
        self.contacts = contacts
        self.num_envs = num_envs
        self.bodies_per_env = bodies_per_env
        self.body_count = num_envs * bodies_per_env
        self.maximum_separation = maximum_separation
        self.debounce_substeps = debounce_substeps
        self.history_bits = (1 << history_substeps) - 1
        self.dt = dt
        self.device = device
        self.stream = stream
        self.raw = wp.zeros(self.body_count, dtype=wp.int32, device=device)
        self.debounce = wp.zeros(self.body_count, dtype=wp.int32, device=device)
        self.stable = wp.zeros(self.body_count, dtype=wp.int32, device=device)
        self.history = wp.zeros(self.body_count, dtype=wp.int32, device=device)
        self.contact_time = wp.zeros(self.body_count, dtype=wp.float32, device=device)
        self.air_time = wp.zeros(self.body_count, dtype=wp.float32, device=device)
        self._stable_torch = wp.to_torch(self.stable).view(num_envs, bodies_per_env)
        self._history_torch = wp.to_torch(self.history).view(num_envs, bodies_per_env)
        self._debounce_torch = wp.to_torch(self.debounce).view(num_envs, bodies_per_env)
        self._contact_time_torch = wp.to_torch(self.contact_time).view(num_envs, bodies_per_env)
        self._air_time_torch = wp.to_torch(self.air_time).view(num_envs, bodies_per_env)
        self.foot_body_indices = torch.tensor(foot_body_indices, dtype=torch.long, device=str(device))
        self.undesired_body_indices = torch.tensor(undesired_body_indices, dtype=torch.long, device=str(device))

    def update(self, body_q: Any) -> None:
        wp.launch(
            clear_raw_body_contacts,
            dim=self.body_count,
            inputs=[self.raw],
            device=self.device,
            stream=self.stream,
        )
        wp.launch(
            aggregate_ground_contacts,
            dim=int(self.contacts.rigid_contact_max),
            inputs=[
                self.contacts.rigid_contact_count,
                self.contacts.rigid_contact_shape0,
                self.contacts.rigid_contact_shape1,
                self.contacts.rigid_contact_point0,
                self.contacts.rigid_contact_point1,
                self.contacts.rigid_contact_normal,
                self.contacts.rigid_contact_margin0,
                self.contacts.rigid_contact_margin1,
                self.model.shape_body,
                self.model.body_world,
                body_q,
                self.maximum_separation,
                self.raw,
            ],
            device=self.device,
            stream=self.stream,
        )
        wp.launch(
            update_contact_state,
            dim=self.body_count,
            inputs=[
                self.raw,
                self.debounce,
                self.stable,
                self.history,
                self.contact_time,
                self.air_time,
                self.debounce_substeps,
                self.history_bits,
                self.dt,
            ],
            device=self.device,
            stream=self.stream,
        )

    @property
    def foot_contact(self) -> torch.Tensor:
        return self._stable_torch.index_select(1, self.foot_body_indices).bool()

    @property
    def recent_foot_contact(self) -> torch.Tensor:
        return self._history_torch.index_select(1, self.foot_body_indices) != 0

    @property
    def recent_undesired_contact(self) -> torch.Tensor:
        return self._history_torch.index_select(1, self.undesired_body_indices) != 0

    def reset(self, env_ids: torch.Tensor) -> None:
        self._stable_torch[env_ids] = 0
        self._history_torch[env_ids] = 0
        self._debounce_torch[env_ids] = 0
        self._contact_time_torch[env_ids] = 0.0
        self._air_time_torch[env_ids] = 0.0

    @property
    def overflow(self) -> torch.Tensor:
        count = wp.to_torch(self.contacts.rigid_contact_count)[0]
        return count > int(self.contacts.rigid_contact_max)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental exact-pose contact reuse for immutable rigid geometry."""

import warp as wp

from newton._src.sim.collide import ContactWriterData


@wp.kernel
def _stable_shapes(
    q: wp.array[wp.transform],
    old_q: wp.array[wp.transform],
    data: wp.array[wp.vec4],
    old_data: wp.array[wp.vec4],
    gap: wp.array[float],
    old_gap: wp.array[float],
    velocity: wp.array[wp.vec3],
    angular: wp.array[wp.vec3],
    stable: wp.array[int],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    old_body_q: wp.array[wp.transform],
):
    i = wp.tid()
    equal = True
    for j in range(7):
        equal = equal and q[i][j] == old_q[i][j]
    body = shape_body[i]
    if body >= 0:
        for j in range(7):
            equal = equal and body_q[body][j] == old_body_q[body][j]
    for j in range(4):
        equal = equal and data[i][j] == old_data[i][j]
    equal = equal and gap[i] == old_gap[i]
    if velocity:
        equal = equal and wp.dot(velocity[i], velocity[i]) == 0.0
    if angular:
        equal = equal and wp.dot(angular[i], angular[i]) == 0.0
    stable[i] = int(equal)


@wp.kernel
def _filter_pairs(
    pairs: wp.array[wp.vec2i],
    count: wp.array[int],
    stable: wp.array[int],
    old: wp.array2d[int],
    current: wp.array2d[int],
    reuse: wp.array2d[int],
    active: wp.array[wp.vec2i],
    active_count: wp.array[int],
    old_count: wp.array[int],
    limit: int,
    reused: wp.array[int],
):
    i = wp.tid()
    if i >= wp.min(count[0], pairs.shape[0]):
        return
    if count[0] > pairs.shape[0]:
        active[i] = pairs[i]
        if i == 0:
            active_count[0] = count[0]
        return
    pair = pairs[i]
    a = wp.min(pair[0], pair[1])
    b = wp.max(pair[0], pair[1])
    current[a, b] = 1
    if stable[a] and stable[b] and old[a, b] and old_count[0] <= limit and count[0] <= pairs.shape[0]:
        reuse[a, b] = 1
        wp.atomic_add(reused, 0, 1)
    else:
        slot = wp.atomic_add(active_count, 0, 1)
        active[slot] = pair


@wp.func
def _copy_contact(old: ContactWriterData, out: ContactWriterData, i: int, j: int):
    out.out_shape0[j] = old.out_shape0[i]
    out.out_shape1[j] = old.out_shape1[i]
    out.out_point0[j] = old.out_point0[i]
    out.out_point1[j] = old.out_point1[i]
    out.out_offset0[j] = old.out_offset0[i]
    out.out_offset1[j] = old.out_offset1[i]
    out.out_normal[j] = old.out_normal[i]
    out.out_margin0[j] = old.out_margin0[i]
    out.out_margin1[j] = old.out_margin1[i]
    out.out_tids[j] = old.out_tids[i]
    if out.out_sort_key:
        out.out_sort_key[j] = old.out_sort_key[i]
    if out.out_stiffness:
        out.out_stiffness[j] = old.out_stiffness[i]
        out.out_damping[j] = old.out_damping[i]
        out.out_friction[j] = old.out_friction[i]


@wp.kernel
def _replay(old: ContactWriterData, out: ContactWriterData, reuse: wp.array2d[int]):
    i = wp.tid()
    if i >= wp.min(old.contact_count[0], old.contact_max):
        return
    a = old.out_shape0[i]
    b = old.out_shape1[i]
    if a < 0 or b < 0 or not reuse[wp.min(a, b), wp.max(a, b)]:
        return
    j = wp.atomic_add(out.contact_count, 0, 1)
    if j < out.contact_max:
        _copy_contact(old, out, i, j)


@wp.kernel
def _save_contacts(
    source: ContactWriterData,
    target: ContactWriterData,
    q: wp.array[wp.transform],
    old_q: wp.array[wp.transform],
    data: wp.array[wp.vec4],
    old_data: wp.array[wp.vec4],
    gap: wp.array[float],
    old_gap: wp.array[float],
    old_body_q: wp.array[wp.transform],
):
    i = wp.tid()
    if i == 0:
        target.contact_count[0] = source.contact_count[0]
    if i < wp.min(source.contact_count[0], source.contact_max):
        _copy_contact(source, target, i, i)
    if i < q.shape[0]:
        old_q[i] = q[i]
        old_data[i] = data[i]
        old_gap[i] = gap[i]
    if i < old_body_q.shape[0]:
        old_body_q[i] = source.body_q[i]


_FIELDS = (
    "out_shape0",
    "out_shape1",
    "out_point0",
    "out_point1",
    "out_offset0",
    "out_offset1",
    "out_normal",
    "out_margin0",
    "out_margin1",
    "out_tids",
    "out_sort_key",
    "out_stiffness",
    "out_damping",
    "out_friction",
    "contact_count",
)


class StationaryContactCache:
    """Delegate narrow phase while reusing unchanged fixed-geometry pairs.

    Experimental: invalidate with reset() after editing mesh vertices, SDF
    contents, shape properties or collision settings. A warm-up call before
    graph capture is required. This wrapper never integrates or freezes bodies.
    """

    def __init__(self, pipeline):
        self.owner = pipeline
        self.model = pipeline.model
        self.narrow_phase = pipeline.narrow_phase

    def __getattr__(self, name):
        return getattr(self.narrow_phase, name)

    def reset(self):
        """Discard prior pair membership and contacts without moving any body."""
        if "_buffers" in self.__dict__:
            buffers = self._buffers
            buffers[4].zero_()
            buffers[9].contact_count.zero_()

    def launch_custom_write(self, **kwargs):
        model = self.model
        # The generic pipeline recognizes only its native NarrowPhase type when
        # forwarding optional cooking metadata. Forward that metadata explicitly.
        kwargs.setdefault("mesh_edge_centers", model.mesh_edge_centers)
        kwargs.setdefault("mesh_edge_halves", model.mesh_edge_halves)
        kwargs.setdefault("shape_support_data", model._shape_support_data)
        kwargs.setdefault("support_lut", model._convex_support_lut)
        kwargs.setdefault("support_vertex_offsets", model._convex_support_vertex_offsets)
        kwargs.setdefault("support_neighbors", model._convex_support_neighbors)
        kwargs.setdefault("hydroelastic_shape_sdf_data_prepared", self.owner._hydro_shape_sdf_data_prepared)
        writer = kwargs["writer_data"]
        if not hasattr(writer, "out_point0"):
            return self.narrow_phase.launch_custom_write(**kwargs)
        q = kwargs["shape_transform"]
        data = kwargs["shape_data"]
        gap = kwargs["shape_gap"]
        pairs = kwargs["candidate_pair"]
        count = kwargs["candidate_pair_count"]
        if not hasattr(self, "_buffers"):
            if q.device.is_capturing:
                raise RuntimeError("Warm up stationary contact caching before graph capture")
            if 12 * q.size * q.size > 128 * 1024 * 1024:
                raise ValueError("Stationary pair caching exceeds its 128 MiB budget")
            n = q.size
            old = ContactWriterData()
            old.contact_max = writer.contact_max
            self.old_body_q = wp.clone(writer.body_q)
            for field in _FIELDS:
                value = getattr(writer, field)
                if value is not None:
                    setattr(old, field, wp.zeros_like(value))
            self._buffers = (
                wp.zeros_like(q),
                wp.zeros_like(data),
                wp.zeros_like(gap),
                wp.zeros(n, dtype=int, device=q.device),
                *[wp.zeros((n, n), dtype=int, device=q.device) for _ in range(3)],
                wp.empty_like(pairs),
                wp.zeros(1, dtype=int, device=q.device),
                old,
                wp.zeros(1, dtype=int, device=q.device),
            )
        old_q, old_data, old_gap, stable, previous, current, reuse, active, active_count, old, reused = self._buffers
        if old.contact_max != writer.contact_max or active.size != pairs.size or old_q.size != q.size:
            raise ValueError("Recreate the stationary contact cache after changing buffer capacities")
        current.zero_()
        reuse.zero_()
        active_count.zero_()
        wp.launch(
            _stable_shapes,
            q.size,
            [
                q,
                old_q,
                data,
                old_data,
                gap,
                old_gap,
                kwargs.get("shape_linear_velocity"),
                kwargs.get("shape_angular_velocity"),
                stable,
                writer.shape_body,
                writer.body_q,
                self.old_body_q,
            ],
            device=q.device,
        )
        wp.launch(
            _filter_pairs,
            pairs.size,
            [
                pairs,
                count,
                stable,
                previous,
                current,
                reuse,
                active,
                active_count,
                old.contact_count,
                writer.contact_max,
                reused,
            ],
            device=q.device,
        )
        kwargs["candidate_pair"] = active
        kwargs["candidate_pair_count"] = active_count
        self.narrow_phase.launch_custom_write(**kwargs)
        wp.launch(_replay, writer.contact_max, [old, writer, reuse], device=q.device)
        wp.launch(
            _save_contacts,
            max(writer.contact_max, q.size, self.old_body_q.size, 1),
            [writer, old, q, old_q, data, old_data, gap, old_gap, self.old_body_q],
            device=q.device,
        )
        wp.copy(previous, current)

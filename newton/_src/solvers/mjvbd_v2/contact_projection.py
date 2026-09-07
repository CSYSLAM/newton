# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free contact blocks for the translation-only Galerkin operator."""

import warp as wp


@wp.struct
class ContactProjectionData:
    count: wp.array[wp.int32]
    overflow: wp.array[wp.int32]
    capacity: int
    edge_count: wp.array[wp.int32]
    edge_keys: wp.array[wp.int64]
    edge_clusters: wp.array[wp.vec2i]
    edge_blocks: wp.array[wp.mat33]
    row_heads: wp.array[wp.int32]
    row_next: wp.array[wp.int32]
    diagonal_correction: wp.array[wp.mat33]
    packed_ready: wp.array[wp.int32]
    packed_offsets: wp.array[wp.int32]
    packed_columns: wp.array[wp.int32]
    packed_blocks: wp.array[wp.mat33]


@wp.func
def _insert_block(row: int, column: int, block: wp.mat33, data: ContactProjectionData):
    a, b = wp.min(row, column), wp.max(row, column)
    key = wp.int64(a) * wp.int64(data.row_heads.shape[0]) + wp.int64(b)
    mask = data.edge_keys.shape[0] - 1
    hashed = wp.uint32(a) * wp.uint32(73856093) ^ wp.uint32(b) * wp.uint32(19349663)
    slot = int(hashed & wp.uint32(mask))
    inserted = wp.bool(False)
    for _probe in range(64):
        previous = wp.atomic_cas(data.edge_keys, slot, wp.int64(-1), key)
        if previous == -1 or previous == key:
            # Storage is cleared in a separate launch: there is no race
            # between initializing a new block and another thread adding it.
            wp.atomic_add(data.edge_blocks, slot, block)
            if previous == -1:
                data.edge_clusters[slot] = wp.vec2i(a, b)
                data.row_next[2 * slot] = wp.atomic_exch(data.row_heads, a, 2 * slot)
                data.row_next[2 * slot + 1] = wp.atomic_exch(data.row_heads, b, 2 * slot + 1)
                wp.atomic_add(data.edge_count, 0, 1)
            inserted = True
            break
        slot = (slot + 1) & mask
    if not inserted:
        wp.atomic_or(data.overflow, 0, 2)


@wp.func
def append_contact(
    particles: wp.vec4i,
    weights: wp.vec4,
    hessian: wp.mat33,
    fine_to_coarse: wp.array[wp.int32],
    data: ContactProjectionData,
):
    """Project one fixed contact stencil, merging repeated cluster ids."""
    clusters = wp.vec4i(-1)
    projected = wp.vec4(0.0)
    squares = wp.vec4(0.0)
    count = int(0)
    for i in range(4):
        particle = particles[i]
        if particle >= 0:
            cluster = fine_to_coarse[particle]
            if cluster >= 0:
                slot = int(-1)
                for j in range(4):
                    if clusters[j] == cluster:
                        slot = j
                if slot < 0:
                    slot = count
                    clusters[slot] = cluster
                    count += 1
                projected[slot] += weights[i]
                squares[slot] += weights[i] * weights[i]
    norm_squared = wp.ddot(hessian, hessian)
    if not wp.isfinite(norm_squared):
        wp.atomic_or(data.overflow, 0, 4)
    elif count > 0 and norm_squared > 0.0:
        record = wp.atomic_add(data.count, 0, 1)
        if record >= data.capacity:
            wp.atomic_or(data.overflow, 0, 1)
        else:
            for i in range(4):
                cluster = clusters[i]
                if cluster >= 0:
                    # Replace sum_i b_i^2 K by (sum_i b_i)^2 K in this cluster.
                    wp.atomic_add(
                        data.diagonal_correction, cluster, (projected[i] * projected[i] - squares[i]) * hessian
                    )
                    if projected[i] != 0.0:
                        for j in range(i + 1, 4):
                            if clusters[j] >= 0 and projected[j] != 0.0:
                                _insert_block(cluster, clusters[j], (projected[i] * projected[j]) * hessian, data)


@wp.kernel(enable_backward=False)
def project_records(
    particles: wp.array[wp.vec4i],
    weights: wp.array[wp.vec4],
    hessians: wp.array[wp.mat33],
    fine_to_coarse: wp.array[wp.int32],
    data: ContactProjectionData,
):
    record = wp.tid()
    append_contact(particles[record], weights[record], hessians[record], fine_to_coarse, data)


@wp.func
def off_diagonal_product(row: int, vector: wp.array[wp.vec3], data: ContactProjectionData):
    value = wp.vec3(0.0)
    if data.packed_ready:
        if data.packed_ready[0] != 0:
            for entry in range(data.packed_offsets[row], data.packed_offsets[row + 1]):
                value += data.packed_blocks[entry] * vector[data.packed_columns[entry]]
            return value
    if data.row_heads:
        entry = data.row_heads[row]
        while entry >= 0:
            slot = entry // 2
            column = data.edge_clusters[slot][1 - entry % 2]
            value += data.edge_blocks[slot] * vector[column]
            entry = data.row_next[entry]
    return value


@wp.kernel(enable_backward=False)
def apply_diagonal(
    data: ContactProjectionData,
    diagonal_slots: wp.array[wp.int32],
    blocks: wp.array[wp.mat33],
):
    row = wp.tid()
    blocks[diagonal_slots[row]] += data.diagonal_correction[row]


@wp.kernel(enable_backward=False)
def reject_overflow(data: ContactProjectionData, status: wp.array[wp.int32]):
    if data.overflow[0] != 0:
        status[0] = status[0] | 32


@wp.kernel(enable_backward=False)
def _reset(data: ContactProjectionData):
    index = wp.tid()
    if index == 0:
        data.count[0] = 0
        data.edge_count[0] = 0
        data.overflow[0] = 0
        if data.packed_ready:
            data.packed_ready[0] = 0
    if index < data.edge_keys.shape[0]:
        data.edge_keys[index] = wp.int64(-1)
        data.edge_blocks[index] = wp.mat33(0.0)
    if index < data.row_heads.shape[0]:
        data.row_heads[index] = -1
        data.diagonal_correction[index] = wp.mat33(0.0)


@wp.kernel(enable_backward=False)
def _count_rows(data: ContactProjectionData, counts: wp.array[wp.int32]):
    row = wp.tid()
    count = int(0)
    if row < data.row_heads.shape[0]:
        entry = data.row_heads[row]
        while entry >= 0:
            count += 1
            entry = data.row_next[entry]
    counts[row] = count


@wp.kernel(enable_backward=False)
def _pack_rows(data: ContactProjectionData):
    row = wp.tid()
    output = data.packed_offsets[row]
    entry = data.row_heads[row]
    while entry >= 0:
        slot = entry // 2
        data.packed_columns[output] = data.edge_clusters[slot][1 - entry % 2]
        data.packed_blocks[output] = data.edge_blocks[slot]
        output += 1
        entry = data.row_next[entry]
    # Consumers are separate launches on the same stream. Retain list order
    # so layout conversion introduces no change to each row's summation order.
    if row == 0:
        data.packed_ready[0] = 1


class ContactProjection:
    """Own bounded contact storage; reject rather than truncate overflowing solves."""

    def __init__(self, cluster_count: int, capacity: int, device):
        if cluster_count < 1 or capacity < 1:
            raise ValueError("Contact projection requires positive cluster and record capacities")
        self.data = ContactProjectionData()
        self.data.count = wp.zeros(1, dtype=wp.int32, device=device)
        self.data.overflow = wp.zeros(1, dtype=wp.int32, device=device)
        self.data.capacity = capacity
        hash_capacity = 1 << max(0, (64 * cluster_count - 1).bit_length())
        self.data.edge_count = wp.zeros(1, dtype=wp.int32, device=device)
        self.data.edge_keys = wp.empty(hash_capacity, dtype=wp.int64, device=device)
        self.data.edge_clusters = wp.empty(hash_capacity, dtype=wp.vec2i, device=device)
        self.data.edge_blocks = wp.empty(hash_capacity, dtype=wp.mat33, device=device)
        self.data.row_heads = wp.full(cluster_count, -1, dtype=wp.int32, device=device)
        self.data.row_next = wp.empty(2 * hash_capacity, dtype=wp.int32, device=device)
        self.data.diagonal_correction = wp.zeros(cluster_count, dtype=wp.mat33, device=device)
        self.device = device
        self._packed_counts = None

    def reset(self):
        wp.launch(
            _reset, dim=max(self.data.edge_keys.size, self.data.row_heads.size), inputs=[self.data], device=self.device
        )

    def correct_diagonal(self, blocks, diagonal_slots):
        wp.launch(
            apply_diagonal,
            dim=self.data.row_heads.size,
            inputs=[self.data, diagonal_slots],
            outputs=[blocks],
            device=self.device,
        )

    def compact(self):
        """Pack complete hash rows without host count readback or contact truncation."""
        rows = self.data.row_heads.size
        if self._packed_counts is None:
            self._packed_counts = wp.empty(rows + 1, dtype=wp.int32, device=self.device)
            self.data.packed_ready = wp.zeros(1, dtype=wp.int32, device=self.device)
            self.data.packed_offsets = wp.empty(rows + 1, dtype=wp.int32, device=self.device)
            # Each occupied hash slot contributes exactly two directed entries.
            entries = 2 * self.data.edge_keys.size
            self.data.packed_columns = wp.empty(entries, dtype=wp.int32, device=self.device)
            self.data.packed_blocks = wp.empty(entries, dtype=wp.mat33, device=self.device)
        wp.launch(_count_rows, dim=rows + 1, inputs=[self.data], outputs=[self._packed_counts], device=self.device)
        wp.utils.array_scan(self._packed_counts, self.data.packed_offsets, inclusive=False)
        wp.launch(_pack_rows, dim=rows, inputs=[self.data], device=self.device)

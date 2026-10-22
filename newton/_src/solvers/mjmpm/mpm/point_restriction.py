# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Construct PIC point-to-cell adjacency without sorting padded cell slots."""

import warp as wp
import warp.fem as fem


class PicPointRestriction(fem.SpaceRestriction):
    """Restriction for unique PIC evaluation nodes, including multi-cell particle samples."""

    def __init__(self, space_partition, domain, *, device=None, temporary_store=None):
        self.space_partition = space_partition
        self.space_topology = space_partition.space_topology
        self.domain = domain
        self._node_count_dev = None
        self._dof_partition_indices = None
        self._dof_partition_element_offsets = None
        self._dof_element_indices = None
        self._dof_indices_in_element = None
        self.rebuild(device=device, temporary_store=temporary_store)

    def rebuild(self, device=None, temporary_store=None):
        self.node_arg_value.invalidate(self)
        topology = self.space_topology
        partition = self.space_partition
        domain = self.domain
        pic = topology._quadrature
        point_count = pic.evaluation_point_count()
        node_count = partition.node_count()
        device = wp.get_device(device)

        counts = fem.borrow_temporary(temporary_store, shape=node_count + 1, dtype=int, device=device)
        point_cells = fem.borrow_temporary(temporary_store, shape=node_count, dtype=int, device=device)
        point_local = fem.borrow_temporary(temporary_store, shape=node_count, dtype=int, device=device)
        counts.zero_()
        point_cells.fill_(fem.NULL_ELEMENT_INDEX)
        self._dof_partition_element_offsets = fem.borrow_temporary(
            temporary_store, shape=node_count + 1, dtype=int, device=device
        )
        self._dof_partition_indices = fem.borrow_temporary(temporary_store, shape=node_count, dtype=int, device=device)
        self._dof_element_indices = fem.borrow_temporary(temporary_store, shape=node_count, dtype=int, device=device)
        self._dof_indices_in_element = fem.borrow_temporary(temporary_store, shape=node_count, dtype=int, device=device)

        @fem.cache.dynamic_kernel(suffix=(domain.name, partition.name, pic.name))
        def locate_points(
            domain_arg: domain.ElementIndexArg,
            partition_arg: partition.PartitionArg,
            cell_offsets: wp.array[int],
            max_per_cell: int,
            counts: wp.array[int],
            cells: wp.array[int],
            local: wp.array[int],
        ):
            point = wp.tid()
            if point >= cell_offsets[cell_offsets.shape[0] - 1]:
                return
            cell = wp.lower_bound(cell_offsets, point + 1) - 1
            index_in_cell = point - cell_offsets[cell]
            if index_in_cell >= max_per_cell:
                return
            domain_cell = cell
            if wp.static(not pic._use_domain_element_indices):
                domain_cell = domain.element_partition_index(domain_arg, cell)
            if domain_cell == fem.NULL_ELEMENT_INDEX:
                return
            node = partition.partition_node_index(partition_arg, point)
            if node < 0 or node >= cells.shape[0]:
                return
            counts[node] = 1
            cells[node] = domain_cell
            local[node] = index_in_cell

        wp.launch(
            locate_points,
            dim=point_count,
            inputs=[
                domain.element_index_arg_value(device),
                partition.partition_arg_value(device),
                pic.cell_particle_offsets,
                topology.MAX_NODES_PER_ELEMENT,
                counts,
                point_cells,
                point_local,
            ],
            device=device,
        )
        wp.utils.array_scan(counts, self._dof_partition_element_offsets, inclusive=False)
        wp.launch(
            compact_adjacency,
            dim=node_count,
            inputs=[counts, self._dof_partition_element_offsets, point_cells, point_local],
            outputs=[self._dof_partition_indices, self._dof_element_indices, self._dof_indices_in_element],
            device=device,
        )
        self._node_count = node_count
        self._node_count_dev = self._dof_partition_element_offsets[node_count:]
        counts.release()
        point_cells.release()
        point_local.release()


@wp.kernel(enable_backward=False)
def compact_adjacency(
    counts: wp.array[int],
    offsets: wp.array[int],
    cells: wp.array[int],
    local: wp.array[int],
    nodes: wp.array[int],
    element_indices: wp.array[int],
    indices_in_element: wp.array[int],
):
    node = wp.tid()
    total = offsets[offsets.shape[0] - 1]
    if node >= total:
        nodes[node] = fem.NULL_NODE_INDEX
        element_indices[node] = fem.NULL_ELEMENT_INDEX
        indices_in_element[node] = 0
    if counts[node]:
        offset = offsets[node]
        nodes[offset] = node
        element_indices[offset] = cells[node]
        indices_in_element[offset] = local[node]


def make_point_restriction(space_partition, domain, *, device=None, temporary_store=None):
    """Specialize unique PIC evaluation nodes; retain the general restriction otherwise."""
    topology = space_partition.space_topology
    pic = getattr(topology, "_quadrature", None)
    if (
        wp.get_device(device).is_cuda
        and isinstance(pic, fem.PicQuadrature)
        and getattr(topology, "_use_evaluation_point_index", False)
        and pic.domain is domain
        and topology._geo_partition is domain.geometry_partition
        and isinstance(domain, fem.Cells)
        and domain.dimension == topology.dimension
    ):
        return PicPointRestriction(space_partition, domain, device=device, temporary_store=temporary_store)
    return fem.make_space_restriction(
        space_partition=space_partition, domain=domain, device=device, temporary_store=temporary_store
    )

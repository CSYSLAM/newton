# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""LBVH-based broad phase collision detection.

Builds a Linear BVH over all shape AABBs using Warp's GPU-accelerated BVH,
then queries each shape's AABB against the BVH to find overlapping pairs.
Provides O(N log N) performance for large scenes where many AABB pairs
overlap.

Note:
    BVH broad phase uses ``wp.Bvh.refit()`` and ``wp.copy()`` which are
    Python-to-C bridge calls.  These operations are captured in CUDA graphs
    but have higher per-frame overhead than pure Warp kernel launches,
    resulting in lower CUDA-graph replay throughput compared to NxN for
    small-to-medium scenes.  For scenes where CUDA graph acceleration is
    critical and N is modest (under ~2000 shapes), NxN may outperform BVH.
    BVH becomes competitive at very large N where NxN's O(N²) cost dominates.

See Also:
    :class:`BroadPhaseSAP` in ``broad_phase_sap.py`` for sweep-and-prune.
    :class:`BroadPhaseAllPairs` in ``broad_phase_nxn.py`` for small scenes.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..core.types import Devicelike
from .broad_phase_common import (
    is_pair_excluded,
    precompute_world_map,
    test_world_and_group_pair,
    write_pair,
)

wp.set_module_options({"enable_backward": False})


@wp.kernel(enable_backward=False)
def _bvh_broadphase_kernel(
    # BVH id for query
    bvh_id: wp.uint64,
    # Shape AABB data
    shape_bounding_box_lower: wp.array[wp.vec3],
    shape_bounding_box_upper: wp.array[wp.vec3],
    # Collision filtering
    collision_group: wp.array[int],
    shape_world: wp.array[int],
    # World mapping
    world_index_map: wp.array[int],
    world_slice_ends: wp.array[int],
    num_regular_worlds: int,
    # Exclusion filter
    filter_pairs: wp.array[wp.vec2i],
    num_filter_pairs: int,
    # Output
    candidate_pair: wp.array[wp.vec2i],
    candidate_pair_count: wp.array[int],
    max_candidate_pair: int,
):
    shape_id = wp.tid()

    lower = shape_bounding_box_lower[shape_id]
    upper = shape_bounding_box_upper[shape_id]

    # Query this shape's AABB against the BVH
    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    hit_index = int(0)
    while wp.bvh_query_next(query, hit_index):
        # Enforce canonical ordering (i < j) to avoid duplicate pairs
        if hit_index <= shape_id:
            continue

        # World and collision group filtering
        world1 = shape_world[shape_id]
        world2 = shape_world[hit_index]
        group1 = collision_group[shape_id]
        group2 = collision_group[hit_index]

        # Skip pairs where both are global (world -1) unless in the
        # dedicated -1 segment.  For BVH broadphase we handle this
        # simply: both global shapes are valid pairs since the BVH
        # contains all shapes and we only write canonical (i<j) pairs.
        if not test_world_and_group_pair(world1, world2, group1, group2):
            continue

        # Skip explicitly excluded pairs
        if num_filter_pairs > 0 and is_pair_excluded(
            wp.vec2i(shape_id, hit_index), filter_pairs, num_filter_pairs
        ):
            continue

        write_pair(
            wp.vec2i(shape_id, hit_index),
            candidate_pair,
            candidate_pair_count,
            max_candidate_pair,
        )


class BroadPhaseBvh:
    """LBVH-based broad phase collision detection.

    Builds a Linear BVH over all shape AABBs using Warp's GPU-accelerated
    BVH construction, then queries each shape's AABB against the BVH to find
    overlapping pairs.  Provides O(N log N) performance for large scenes.

    The BVH is built during initialization and refitted on each call to
    :meth:`launch` (O(N) bounds update preserving tree topology).
    """

    def __init__(
        self,
        shape_world: wp.array[wp.int32] | np.ndarray,
        shape_flags: wp.array[wp.int32] | np.ndarray | None = None,
        device: Devicelike | None = None,
    ) -> None:
        """Initialize the LBVH broad phase with world ID information.

        Args:
            shape_world: Array of world IDs (numpy or warp array).
                Positive/zero values represent distinct worlds, negative values
                represent shared entities that belong to all worlds.
            shape_flags: Optional array of shape flags (numpy or warp array).
                If provided, only shapes with the COLLIDE_SHAPES flag will be
                included in collision checks.
            device: Device to store the precomputed arrays on. If None, uses
                CPU for numpy arrays or the device of the input warp array.
        """
        # Convert to numpy if it's a warp array
        if isinstance(shape_world, wp.array):
            shape_world_np = shape_world.numpy()
            if device is None:
                device = shape_world.device
        else:
            shape_world_np = shape_world
            if device is None:
                device = "cpu"

        # Convert shape_flags to numpy if provided
        shape_flags_np = None
        if shape_flags is not None:
            if isinstance(shape_flags, wp.array):
                shape_flags_np = shape_flags.numpy()
            else:
                shape_flags_np = shape_flags

        # Precompute the world map (filters out non-colliding shapes if flags provided)
        index_map_np, slice_ends_np = precompute_world_map(shape_world_np, shape_flags_np)

        # Calculate number of regular worlds (excluding dedicated -1 segment at end)
        num_regular_worlds = max(0, len(slice_ends_np) - 1)

        # Store as warp arrays
        self.world_index_map = wp.array(index_map_np, dtype=wp.int32, device=device)
        self.world_slice_ends = wp.array(slice_ends_np, dtype=wp.int32, device=device)
        self.num_regular_worlds = int(num_regular_worlds)

        # Build BVH upfront with dummy AABBs so that launch() only needs
        # to refit (a pure kernel launch compatible with CUDA graph capture).
        # wp.Bvh() performs host-side allocation that cannot be captured.
        shape_count = len(shape_world_np)
        with wp.ScopedDevice(device):
            self._bvh_lower = wp.zeros(shape_count, dtype=wp.vec3)
            self._bvh_upper = wp.zeros(shape_count, dtype=wp.vec3)
        self._bvh = wp.Bvh(self._bvh_lower, self._bvh_upper)
        self._bvh_built = True
        self._shape_count = shape_count

    def launch(
        self,
        shape_lower: wp.array[wp.vec3],
        shape_upper: wp.array[wp.vec3],
        shape_gap: wp.array[float] | None,
        shape_collision_group: wp.array[int],
        shape_world: wp.array[int],
        shape_count: int,
        candidate_pair: wp.array[wp.vec2i],
        candidate_pair_count: wp.array[int],
        device: Devicelike | None = None,
        filter_pairs: wp.array[wp.vec2i] | None = None,
        num_filter_pairs: int | None = None,
        skip_count_zero: bool = False,
    ) -> None:
        """Launch the LBVH broad phase collision detection.

        Refits the BVH over all shape AABBs, then queries each shape against
        the BVH to find overlapping pairs.

        Args:
            shape_lower: Array of lower bounds for each shape's AABB.
            shape_upper: Array of upper bounds for each shape's AABB.
            shape_gap: Optional array of per-shape effective gaps. Ignored
                by BVH broadphase since AABBs are expected to be pre-expanded.
            shape_collision_group: Array of collision group IDs for each shape.
            shape_world: Array of world indices for each shape.
            shape_count: Number of active bounding boxes to check.
            candidate_pair: Output array to store overlapping shape pairs.
            candidate_pair_count: Output array to store number of overlapping
                pairs found.
            device: Device to launch on. If None, uses the device of the input
                arrays.
            filter_pairs: Sorted array of excluded shape pairs.
            num_filter_pairs: Number of valid entries in filter_pairs.
            skip_count_zero: If True, skip the internal
                ``candidate_pair_count.zero_()``. The caller guarantees
                ``candidate_pair_count[0] == 0`` on entry.
        """
        max_candidate_pair = candidate_pair.shape[0]

        if not skip_count_zero:
            candidate_pair_count.zero_()

        if device is None:
            device = shape_lower.device

        # Ensure persistent AABB arrays exist and match shape_count
        if self._shape_count != shape_count:
            with wp.ScopedDevice(device):
                self._bvh_lower = wp.zeros(shape_count, dtype=wp.vec3)
                self._bvh_upper = wp.zeros(shape_count, dtype=wp.vec3)
            # Rebuild BVH when shape count changes
            self._bvh = wp.Bvh(self._bvh_lower, self._bvh_upper)
            self._shape_count = shape_count

        # Copy current AABB data into persistent arrays (BVH holds references to these)
        wp.copy(self._bvh_lower, shape_lower)
        wp.copy(self._bvh_upper, shape_upper)

        # Refit the BVH with updated bounds
        self._bvh.refit()

        # Exclusion filter: empty array and 0 when not provided or empty
        if filter_pairs is None or filter_pairs.shape[0] == 0:
            filter_pairs_arr = wp.empty(0, dtype=wp.vec2i, device=device)
            n_filter = 0
        else:
            filter_pairs_arr = filter_pairs
            n_filter = num_filter_pairs if num_filter_pairs is not None else filter_pairs.shape[0]

        # Launch the BVH query kernel: one thread per shape
        wp.launch(
            kernel=_bvh_broadphase_kernel,
            dim=shape_count,
            inputs=[
                self._bvh.id,
                shape_lower,
                shape_upper,
                shape_collision_group,
                shape_world,
                self.world_index_map,
                self.world_slice_ends,
                self.num_regular_worlds,
                filter_pairs_arr,
                n_filter,
            ],
            outputs=[
                candidate_pair,
                candidate_pair_count,
                max_candidate_pair,
            ],
            device=device,
            record_tape=False,
        )

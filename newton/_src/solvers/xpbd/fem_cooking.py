# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-FileCopyrightText: Copyright (c) 2008-2025 NVIDIA Corporation
# SPDX-FileCopyrightText: Copyright (c) 2004-2008 AGEIA Technologies, Inc.
# SPDX-FileCopyrightText: Copyright (c) 2001-2004 NovodeX AG
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of NVIDIA CORPORATION nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""PhysX FEM cloth constraint categorization and eight-partition copy chains.

Reference: PxgFEMClothUtil::categorizeClothConstraints and
PxgFEMClothCore::combineTrianglePairPartitions (PhysX 5.6.1).
The implementation keeps explicit validity bits instead of overloading inverse
mass as an uninitialized-copy sentinel, so fixed vertices remain unambiguous.
"""

import numpy as np
import warp as wp


def color_elements(elements):
    """Choose the first free partition in input order, as in PhysX bitmask coloring."""
    used, groups = {}, []
    for index, vertices in enumerate(elements):
        forbidden = set().union(*(used.get(int(v), set()) for v in vertices))
        color = 0
        while color in forbidden:
            color += 1
        if color == len(groups):
            groups.append([])
        groups[color].append(index)
        for v in vertices:
            used.setdefault(int(v), set()).add(color)
    return groups


def categorize(faces, hinges):
    """Visit every triangle once in either a shared pair or the nonshared list."""
    lookup = {tuple(sorted(face)): i for i, face in enumerate(faces)}
    visited = set()
    shared, nonshared = [], []
    # PhysX initialTriangleData sorts undirected edge keys in descending order
    # before greedy shared-pair selection. Preserve original hinge IDs.
    order = sorted(range(len(hinges)), key=lambda i: (max(hinges[i, 2:]), min(hinges[i, 2:])), reverse=True)
    pair_faces = [(-1, -1)] * len(hinges)
    for h in order:
        vertices = hinges[h]
        if np.any(vertices < 0):
            continue
        a, b, c, d = vertices
        t0, t1 = lookup[tuple(sorted((a, c, d)))], lookup[tuple(sorted((b, c, d)))]
        pair_faces[h] = (t0, t1)
        if t0 not in visited and t1 not in visited:
            shared.append(h)
            visited.update((t0, t1))
        else:
            nonshared.append(h)
    singles = [i for i in range(len(faces)) if i not in visited]
    return (
        np.asarray(shared, dtype=np.int32),
        np.asarray(nonshared, dtype=np.int32),
        singles,
        np.asarray(pair_faces, dtype=np.int32),
    )


def cook_copy_chains(elements, particle_count, maximum_partitions=8):
    """Reproduce modulo-combined partitions, forward remaps and terminal-copy CSR."""
    groups = color_elements(elements)
    lanes = max(1, (len(groups) + maximum_partitions - 1) // maximum_partitions)
    order, ends = [], []
    tables = np.full((particle_count, maximum_partitions * lanes), -1, dtype=np.int32)
    count = len(elements)
    for phase in range(min(len(groups), maximum_partitions)):
        for lane in range(lanes):
            color = phase + maximum_partitions * lane
            if color < len(groups):
                for element in groups[color]:
                    slot = len(order)
                    order.append(element)
                    for corner, vertex in enumerate(elements[element]):
                        tables[vertex, phase * lanes + lane] = slot + corner * count
        ends.append(len(order))
    remap = np.full(4 * count, -1, dtype=np.int32)
    offsets = [0]
    for vertex in range(particle_count):
        table = tables[vertex]
        occupied = np.zeros(len(table), dtype=bool)
        terminal = []
        for phase in range(maximum_partitions):
            next_start = (phase + 1) * lanes
            for lane in range(lanes):
                slot = int(table[phase * lanes + lane])
                if slot < 0:
                    continue
                found = False
                for other in range(next_start, len(table)):
                    if table[other] >= 0 and not occupied[other]:
                        remap[slot] = table[other]
                        occupied[other] = True
                        next_start += 1
                        found = True
                        break
                if not found:
                    terminal.append(slot)
        # PhysX assigns terminal slots in table order, not discovery order.
        terminal.sort(key=lambda slot: int(np.flatnonzero(table == slot)[0]))
        for j, slot in enumerate(terminal):
            remap[slot] = 4 * count + offsets[-1] + j
        offsets.append(offsets[-1] + len(terminal))
    return np.asarray(order, dtype=np.int32), ends, remap, np.asarray(offsets, dtype=np.int32)


class PairBatch:
    """Own a cooked shared/nonshared pair stage and its persistent copy storage."""

    def __init__(self, model, ids, pair_faces):
        hinges = model.edge_indices.numpy()
        order, self.ends, remap, offsets = cook_copy_chains(hinges[ids], model.particle_count)
        ordered = ids[order]
        device = model.device
        self.ids = wp.array(ordered, dtype=int, device=device)
        self.vertices = wp.array(hinges[ordered], dtype=wp.vec4i, device=device)
        self.faces = wp.array(pair_faces[ordered], dtype=wp.vec2i, device=device)
        self.remap = wp.array(remap, dtype=int, device=device)
        self.offsets = wp.array(offsets, dtype=int, device=device)
        size = len(remap) + int(offsets[-1])
        self.copies = wp.empty(size, dtype=wp.vec3, device=device)
        self.valid = wp.zeros(size, dtype=int, device=device)
        # Each shared triangle is parameterized from the common edge, exactly
        # as clothSharedEnergySolvePerTrianglePair, not its original vertex order.
        rest = model.particle_q.numpy().astype(np.float64)
        poses = []
        areas = []
        for h in hinges[ordered]:
            a, b, c, d = h
            for tip in (a, b):
                e0, e1 = rest[d] - rest[c], rest[tip] - rest[c]
                axis = e0 / np.linalg.norm(e0)
                transverse = e1 - axis * np.dot(axis, e1)
                dm = np.array([[np.linalg.norm(e0), np.dot(axis, e1)], [0.0, np.linalg.norm(transverse)]])
                poses.append(np.linalg.inv(dm))
                areas.append(np.linalg.det(dm) / 2)
        self.poses = wp.array(np.asarray(poses, dtype=np.float32).reshape(-1, 2, 2), dtype=wp.mat22, device=device)
        self.areas = wp.array(areas, dtype=float, device=device)

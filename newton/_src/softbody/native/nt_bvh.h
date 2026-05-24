// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// Standalone BVH for box-query against triangle AABBs.
// Algorithm matches PhysX ExtBVH + ExtUtilities behavior:
//   - build: median-split along longest axis
//   - query: stack traversal with box-box overlap test

#pragma once

#include "nt_types.h"
#include <vector>

namespace nt {

struct BVHNode {
    Bounds3 bounds;   // AABB
    Pi32 left;        // <0: leaf with primitive index = ~left; >=0: internal left child
    Pi32 right;       // >=0: internal right child; unused for leaf

    bool isLeaf() const { return left < 0; }
    Pi32 getPrimitiveIndex() const { return ~left; }
};

struct BVH {
    std::vector<BVHNode> nodes;

    void build(const Bounds3* items, Pi32 numItems);
    void query(const Bounds3& box, std::vector<Pi32>& results) const;
};

// ---------- implementation ----------

inline void BVH::build(const Bounds3* items, Pi32 numItems) {
    nodes.clear();
    if (numItems == 0) return;

    struct StackEntry { Pi32 start, end, nodeIndex; };
    std::vector<StackEntry> stack;
    std::vector<Pi32> indices(numItems);
    for (Pi32 i = 0; i < numItems; i++) indices[i] = i;

    // pre-allocate nodes
    nodes.reserve(numItems * 2);

    // push root
    nodes.push_back(BVHNode());
    stack.push_back({0, numItems, 0});

    while (!stack.empty()) {
        StackEntry se = stack.back();
        stack.pop_back();

        Pi32 start = se.start, end = se.end;
        Pi32 nodeIdx = se.nodeIndex;

        // compute bounds for this range
        Bounds3 nodeBounds;
        nodeBounds.setEmpty();
        for (Pi32 i = start; i < end; i++) {
            nodeBounds.include(items[indices[i]].minimum);
            nodeBounds.include(items[indices[i]].maximum);
        }

        Pi32 count = end - start;

        if (count == 1) {
            // leaf
            nodes[nodeIdx].bounds = nodeBounds;
            nodes[nodeIdx].left = ~indices[start]; // negative => leaf
            nodes[nodeIdx].right = -1;
            continue;
        }

        // find longest axis
        Vec3 ext = nodeBounds.getDimensions();
        Pi32 axis = 0;
        if (ext[1] > ext[0]) axis = 1;
        if (ext[2] > ext[axis]) axis = 2;

        // sort indices by center along axis
        std::sort(indices.data() + start, indices.data() + end,
            [&](Pi32 a, Pi32 b) {
                return items[a].getCenter()[axis] < items[b].getCenter()[axis];
            });

        Pi32 mid = start + count / 2;

        // allocate children
        Pi32 leftIdx = Pi32(nodes.size());
        nodes.push_back(BVHNode());
        nodes.push_back(BVHNode());
        Pi32 rightIdx = leftIdx + 1;

        nodes[nodeIdx].bounds = nodeBounds;
        nodes[nodeIdx].left = leftIdx;
        nodes[nodeIdx].right = rightIdx;

        stack.push_back({mid, end, rightIdx});
        stack.push_back({start, mid, leftIdx});
    }
}

inline void BVH::query(const Bounds3& box, std::vector<Pi32>& results) const {
    results.clear();
    if (nodes.empty()) return;

    Pi32 stack[64];
    Pi32 sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        Pi32 idx = stack[--sp];
        const BVHNode& node = nodes[idx];

        if (!node.bounds.intersects(box))
            continue;

        if (node.isLeaf()) {
            results.push_back(node.getPrimitiveIndex());
        } else {
            if (sp + 2 <= 64) {
                stack[sp++] = node.right;
                stack[sp++] = node.left;
            }
        }
    }
}

} // namespace nt
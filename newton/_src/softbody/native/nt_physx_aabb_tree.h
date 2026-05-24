// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// PhysX-exact AABB tree port.
// Build produces BvhNode array with PhysX bit-packed mData encoding:
//   bit 0: leaf flag
//   bits 1-4: nbPrimitives (max 15)
//   bits 5-31: primitive index (for single-primitive leaves)
//
// Internal nodes store: mData = posChildIndex << 1
// Leaf nodes store:     mData = (nbPrims << 1) | 1 | (primIndex << 5)
//
// Ported from:
//   - GuAABBTree.h / GuAABBTree.cpp
//   - GuAABBTreeNode.h
//   - GuAABBTreeBounds.h
//   - GuAABBTreeBuildStats.h
//   - GuAABBTreeQuery.h

#pragma once

#include "nt_types.h"
#include <vector>

namespace nt {

// ---------- BvhNode ----------
// Ported from GuAABBTreeNode.h (AABBTreeBuildNode runtime format)

struct BvhNode {
    Bounds3 mBV;
    Pu32 mData;

    NT_FORCE_INLINE bool isLeaf() const { return (mData & 1) != 0; }
    NT_FORCE_INLINE Pi32 getPosIndex() const { return Pi32(mData >> 1); }
    NT_FORCE_INLINE Pi32 getNegIndex() const { return getPosIndex() + 1; }
    NT_FORCE_INLINE Pi32 getNbPrimitives() const { return Pi32((mData >> 1) & 15); }
    NT_FORCE_INLINE Pi32 getPrimitiveIndex() const { return Pi32(mData >> 5); }
};

// ---------- AabbTreeBounds ----------
// Ported from GuAABBTreeBounds.h

class AabbTreeBounds {
public:
    AabbTreeBounds() : mBounds(nullptr), mNbNodes(0) {}
    ~AabbTreeBounds() { release(); }

    void init(Pi32 nbNodes) {
        release();
        // PhysX allocates nbNodes+1 entries
        mBounds = new Bounds3[nbNodes + 1];
        mNbNodes = nbNodes;
    }

    void release() {
        delete[] mBounds;
        mBounds = nullptr;
        mNbNodes = 0;
    }

    Bounds3* getBounds() { return mBounds; }
    const Bounds3* getBounds() const { return mBounds; }
    Pi32 getNbNodes() const { return mNbNodes; }

private:
    Bounds3* mBounds;
    Pi32 mNbNodes;
};

// ---------- BuildStats ----------
// Ported from GuAABBTreeBuildStats.h

struct BuildStats {
    Pi32 mCount;
    Pi32 mNbLeaves;
    Pi32 mNbObjects;

    BuildStats() : mCount(0), mNbLeaves(0), mNbObjects(0) {}

    NT_FORCE_INLINE void setCount(Pi32 nb) { mCount = nb; }
    NT_FORCE_INLINE Pi32 getCount() const { return mCount; }

    NT_FORCE_INLINE void incLeaves(Pi32 nb) { mNbLeaves += nb; }
    NT_FORCE_INLINE Pi32 getNbLeaves() const { return mNbLeaves; }

    NT_FORCE_INLINE void incObjects(Pi32 nb) { mNbObjects += nb; }
    NT_FORCE_INLINE Pi32 getNbObjects() const { return mNbObjects; }
};

// ---------- AabbTreeBuildParams ----------
// Ported from GuAABBTree.h (AABBTreeBuildParams)

struct AabbTreeBuildParams {
    Pi32 mLimit;          // max prims per leaf (1 for single-prim leaves)
    Pi32 mNbPrimitives;
    const AabbTreeBounds* mBounds;
    Vec3* mCache;         // center cache

    AabbTreeBuildParams(Pi32 limit, Pi32 nbPrims, const AabbTreeBounds* bounds)
        : mLimit(limit), mNbPrimitives(nbPrims), mBounds(bounds), mCache(nullptr) {}
};

// ---------- AabbTreeBuildNode ----------
// Ported from GuAABBTreeNode.h (AABBTreeBuildNode)

struct AabbTreeBuildNode {
    Bounds3 mBV;
    AabbTreeBuildNode* mPos;   // positive child (also used as sibling link during build)
    Pi32 mNodeIndex;           // index in remap array (leaf: start index; internal: not used)
    Pi32 mNbPrimitives;        // 0 = internal node, >0 = leaf

    NT_FORCE_INLINE bool isLeaf() const { return mPos == nullptr; }
    NT_FORCE_INLINE AabbTreeBuildNode* getPos() const { return mPos; }
    NT_FORCE_INLINE AabbTreeBuildNode* getNeg() const { return mPos ? mPos + 1 : nullptr; }
};

// ---------- NodeAllocator ----------
// Ported from GuAABBTree.h (NodeAllocator)

struct Slab {
    AabbTreeBuildNode* mPool;
    Pi32 mNbUsedNodes;
    Pi32 mMaxNbNodes;
};

class NodeAllocator {
public:
    NodeAllocator();
    ~NodeAllocator();

    void init(Pi32 nbPrimitives, Pi32 limit);
    void release();

    AabbTreeBuildNode* getBiNode();
    Pi32 getTotalNbNodes() const { return mTotalNbNodes; }
    AabbTreeBuildNode* getPool(Pi32 index) const { return mPool[index]; }
    Pi32 getPoolSize() const { return Pi32(mPool.size()); }
    const std::vector<Slab>& getSlabs() const { return mSlabs; }
    const std::vector<AabbTreeBuildNode*>& getPoolVec() const { return mPool; }

private:
    std::vector<Slab> mSlabs;
    std::vector<AabbTreeBuildNode*> mPool;
    Pi32 mCurrentSlabIndex;
    Pi32 mTotalNbNodes;
};

// ---------- TraversalControl ----------
// Ported from GuAABBTreeQuery.h

enum class TraversalControl {
    eDontGoDeeper,
    eGoDeeper,
    eGoDeeperNegFirst,
    eAbort
};

// ---------- TinyTetBvh ----------

struct TinyTetBvh {
    std::vector<BvhNode> mTree;

    // Build BVH from tetrahedra
    static void constructFromTetrahedra(
        const Pu32* tetrahedra,
        Pi32 numTetrahedra,
        const Vec3* points,
        TinyTetBvh& result,
        Pf32 enlargement = 1e-4f);

    // Traverse BVH with controller
    template <typename TraversalController>
    void traverse(TraversalController& controller) const {
        if (mTree.empty()) return;
        traverseBvh(mTree.data(), Pi32(mTree.size()), controller);
    }
};

// ---------- buildAabbTree ----------
// Ported from GuAABBTree.cpp buildAABBTree

void buildAabbTree(Pi32 nbBounds, const AabbTreeBounds& bounds, std::vector<BvhNode>& tree);

// ---------- traverseBvh ----------
// Ported from GuAABBTreeQuery.h traverseBVH

template <typename TraversalController>
static void traverseBvh(const BvhNode* nodes, Pi32 numNodes, TraversalController& controller) {
    // PhysX-exact: descend-first traversal (GuAABBTreeQuery.h traverseBVH)
    Pi32 stack[64];
    Pi32 stackIndex = 0;
    Pi32 index = 0;  // root

    while (true) {
        const BvhNode& node = nodes[index];

        TraversalControl rc = controller.analyze(node, index);
        if (rc == TraversalControl::eAbort)
            return;

        if (!node.isLeaf() && (rc == TraversalControl::eGoDeeper || rc == TraversalControl::eGoDeeperNegFirst)) {
            if (rc == TraversalControl::eGoDeeperNegFirst) {
                stack[stackIndex++] = node.getPosIndex();
                index = node.getNegIndex();
            } else {
                stack[stackIndex++] = node.getNegIndex();
                index = node.getPosIndex();
            }
            continue;
        }

        if (stackIndex == 0) break;
        index = stack[--stackIndex];
    }
}

} // namespace nt

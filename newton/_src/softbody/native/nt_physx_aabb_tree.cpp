// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// PhysX-exact AABB tree build logic.
// Ported from GuAABBTree.cpp:
//   - buildAABBTree (initAABBTreeBuild + buildHierarchy + flattenTree)
//   - subdivide
//   - reshuffle
//   - NodeAllocator
//
// Build produces BvhNode array with PhysX bit-packed mData encoding:
//   bit 0: leaf flag
//   bits 1-4: nbPrimitives (max 15)
//   bits 5-31: primitive index (for single-primitive leaves)
//
// Internal nodes store: mData = posChildIndex << 1
// Leaf nodes store:     mData = (nbPrims << 1) | 1 | (primIndex << 5)

#include "nt_physx_aabb_tree.h"
#include <cstring>
#include <algorithm>

namespace nt {

// ---------- NodeAllocator ----------

NodeAllocator::NodeAllocator() : mCurrentSlabIndex(0), mTotalNbNodes(0) {}

NodeAllocator::~NodeAllocator() { release(); }

void NodeAllocator::release() {
    for (Pi32 i = 0; i < Pi32(mSlabs.size()); i++) {
        delete[] mSlabs[i].mPool;
    }
    mSlabs.clear();
    mPool.clear();
    mCurrentSlabIndex = 0;
    mTotalNbNodes = 0;
}

void NodeAllocator::init(Pi32 nbPrimitives, Pi32 limit) {
    release();

    // PhysX: estimatedFinalSize = maxSize <= 1024 ? maxSize : maxSize / limit
    Pi32 maxSize = nbPrimitives * 2 - 1;
    Pi32 estimatedFinalSize = maxSize <= 1024 ? maxSize : maxSize / limit;

    Slab s;
    s.mPool = new AabbTreeBuildNode[estimatedFinalSize];
    std::memset(s.mPool, 0, sizeof(AabbTreeBuildNode) * estimatedFinalSize);
    s.mNbUsedNodes = 1;  // PhysX allocates root node at init, not at getBiNode
    s.mMaxNbNodes = estimatedFinalSize;

    // Setup initial node (matches PhysX: mPool->mNodeIndex = 0, mPool->mNbPrimitives = nbPrimitives)
    s.mPool[0].mNodeIndex = 0;
    s.mPool[0].mNbPrimitives = nbPrimitives;

    mSlabs.push_back(s);
    mCurrentSlabIndex = 0;
    mTotalNbNodes = 1;
}

AabbTreeBuildNode* NodeAllocator::getBiNode() {
    mTotalNbNodes += 2;
    Slab& currentSlab = mSlabs[mCurrentSlabIndex];
    if (currentSlab.mNbUsedNodes + 2 <= currentSlab.mMaxNbNodes) {
        AabbTreeBuildNode* biNode = currentSlab.mPool + currentSlab.mNbUsedNodes;
        currentSlab.mNbUsedNodes += 2;
        return biNode;
    } else {
        // Allocate new slab (matches PhysX: size = 1024)
        Pi32 size = 1024;
        AabbTreeBuildNode* pool = new AabbTreeBuildNode[size];
        std::memset(pool, 0, sizeof(AabbTreeBuildNode) * size);

        mSlabs.push_back(Slab{pool, 2, size});
        mCurrentSlabIndex++;
        return pool;
    }
}

// ---------- reshuffle ----------
// Ported from GuAABBTree.cpp reshuffle

Pi32 reshuffle(Pi32 nb, Pi32* prims, const Vec3* centers, float splitValue, Pi32 axis) {
    Pi32 nbPos = 0;
    for (Pi32 i = 0; i < nb; i++) {
        if (centers[prims[i]][axis] > splitValue) {
            Pi32 t = prims[i];
            prims[i] = prims[nbPos];
            prims[nbPos] = t;
            nbPos++;
        }
    }
    return nbPos;
}

// ---------- split ----------
// Ported from GuAABBTree.cpp split
// Uses box center as split value (default strategy BVH_SPLATTER_POINTS)

static Pi32 split(const Bounds3& box, Pi32 nb, Pi32* prims, Pi32 axis, const Vec3* centers) {
    // Default split value = middle of the axis (using only the box)
    float splitValue = box.getCenter()[axis];

    return reshuffle(nb, prims, centers, splitValue, axis);
}

// ---------- subdivide ----------
// Ported from GuAABBTree.cpp AABBTreeBuildNode::subdivide
// Uses variance analysis for axis selection and box-center split

static void subdivideNode(AabbTreeBuildNode* node, const AabbTreeBounds& bounds,
                           const Vec3* centers, Pi32* indices, NodeAllocator& allocator) {
    Pi32* primitives = indices + node->mNodeIndex;
    Pi32 nbPrims = node->mNbPrimitives;

    // Compute global box & means for current node (matches PhysX: mBV, meansV)
    Bounds3 nodeBV;
    nodeBV.setEmpty();
    Vec3 means(0.0f);
    for (Pi32 i = 0; i < nbPrims; i++) {
        Pi32 index = primitives[i];
        nodeBV.include(bounds.getBounds()[index].minimum);
        nodeBV.include(bounds.getBounds()[index].maximum);
        means = means + centers[index];
    }
    node->mBV = nodeBV;

    // Check the user-defined limit. Also ensures we stop subdividing if we reach a leaf node.
    if (nbPrims <= 1)  // PhysX limit = 1 for single-prim leaves
        return;

    // Compute variances (matches PhysX: varsV)
    Vec3 vars(0.0f);
    for (Pi32 i = 0; i < nbPrims; i++) {
        Pi32 index = primitives[i];
        Vec3 diff = centers[index] - means * (1.0f / float(nbPrims));
        vars = vars + Vec3(diff.x * diff.x, diff.y * diff.y, diff.z * diff.z);
    }
    vars = vars * (1.0f / float(nbPrims - 1));

    // Choose axis with greatest variance (matches PhysX: PxLargestAxis)
    Pi32 axis = 0;
    if (vars[1] > vars[0]) axis = 1;
    if (vars[2] > vars[axis]) axis = 2;

    // Split along the axis (matches PhysX: split)
    Pi32 nbPos = split(node->mBV, nbPrims, primitives, axis, centers);

    // Check split validity (matches PhysX: validSplit check)
    if (!nbPos || nbPos == nbPrims) {
        // All boxes lie in the same sub-space. Arbitrary 50-50 split.
        nbPos = nbPrims >> 1;
    }

    // Now create children and assign their pointers (matches PhysX: mPos = allocator.getBiNode())
    node->mPos = allocator.getBiNode();

    // Assign children (matches PhysX: Pos->mNodeIndex, Pos->mNbPrimitives, Neg->mNodeIndex, Neg->mNbPrimitives)
    AabbTreeBuildNode* Pos = node->mPos;
    AabbTreeBuildNode* Neg = Pos + 1;
    Pos->mNodeIndex = node->mNodeIndex;
    Pos->mNbPrimitives = nbPos;
    Neg->mNodeIndex = node->mNodeIndex + nbPos;
    Neg->mNbPrimitives = nbPrims - nbPos;
}

// ---------- buildHierarchy ----------
// Ported from GuAABBTree.cpp buildHierarchy
// Stack-based iterative subdivide

static void buildHierarchy(AabbTreeBuildNode* root, const AabbTreeBounds& bounds,
                            const Vec3* centers, Pi32* indices, NodeAllocator& allocator) {
    std::vector<AabbTreeBuildNode*> stack;
    stack.reserve(256);
    stack.push_back(root);

    while (!stack.empty()) {
        AabbTreeBuildNode* node = stack.back();
        stack.pop_back();

        subdivideNode(node, bounds, centers, indices, allocator);

        // If node became internal, push children (neg first so pos is processed first)
        if (!node->isLeaf()) {
            AabbTreeBuildNode* Pos = node->mPos;
            stack.push_back(Pos + 1);
            stack.push_back(Pos);
        }
    }
}

// ---------- flattenTree ----------
// Ported from GuAABBTree.cpp flattenTree
// Uses slab-based traversal and indices as remap for leaf primitive indices

static Pi32 flattenTree(const NodeAllocator& nodeAllocator, BvhNode* dest, const Pi32* remap) {
    Pi32 offset = 0;
    const Pi32 nbSlabs = Pi32(nodeAllocator.getSlabs().size());
    for (Pi32 s = 0; s < nbSlabs; s++) {
        const Slab& currentSlab = nodeAllocator.getSlabs()[s];

        AabbTreeBuildNode* pool = currentSlab.mPool;
        for (Pi32 i = 0; i < currentSlab.mNbUsedNodes; i++) {
            dest[offset].mBV = pool[i].mBV;
            if (pool[i].isLeaf()) {
                Pi32 index = pool[i].mNodeIndex;
                if (remap)
                    index = remap[index];

                const Pi32 nbPrims = pool[i].mNbPrimitives;
                dest[offset].mData = (Pu32(index) << 5) | ((Pu32(nbPrims) & 15) << 1) | 1;
            } else {
                if (!pool[i].mPos) {
                    // Degenerate — should not happen
                    offset++;
                    continue;
                }
                // Find which slab contains mPos and compute global offset
                Pi32 localNodeIndex = 0;
                Pi32 nodeBase = 0;
                bool found = false;
                for (Pi32 j = 0; j < nbSlabs; j++) {
                    const Slab& slab = nodeAllocator.getSlabs()[j];
                    if (pool[i].mPos >= slab.mPool && pool[i].mPos < slab.mPool + slab.mNbUsedNodes) {
                        localNodeIndex = Pi32(pool[i].mPos - slab.mPool);
                        found = true;
                        break;
                    }
                    nodeBase += slab.mNbUsedNodes;
                }
                if (!found) {
                    offset++;
                    continue;
                }
                const Pi32 nodeIndex = nodeBase + localNodeIndex;
                dest[offset].mData = Pu32(nodeIndex) << 1;
            }
            offset++;
        }
    }
    return offset;
}

// ---------- buildAabbTree ----------
// Ported from GuAABBTree.cpp buildAABBTree

void buildAabbTree(Pi32 nbBounds, const AabbTreeBounds& bounds, std::vector<BvhNode>& tree) {
    if (nbBounds == 0) return;

    // Create cache of primitive centers (matches PhysX: mCache)
    Vec3* centers = new Vec3[nbBounds];
    for (Pi32 i = 0; i < nbBounds; i++) {
        centers[i] = bounds.getBounds()[i].getCenter();
    }

    // Create primitive index array (matches PhysX: indices)
    Pi32* indices = new Pi32[nbBounds];
    for (Pi32 i = 0; i < nbBounds; i++) indices[i] = i;

    // Init node allocator (matches PhysX: nodeAllocator.init)
    NodeAllocator allocator;
    allocator.init(nbBounds, 1);

    // Root node is already allocated in allocator.init at pool[0]
    AabbTreeBuildNode* root = allocator.getSlabs()[0].mPool;

    // Build hierarchy (matches PhysX: buildHierarchy)
    buildHierarchy(root, bounds, centers, indices, allocator);

    // Flatten tree (matches PhysX: flattenTree with indices as remap)
    Pi32 nbNodes = allocator.getTotalNbNodes();
    tree.resize(nbNodes);
    Pi32 actualNodes = flattenTree(allocator, tree.data(), indices);
    tree.resize(actualNodes);

    delete[] indices;
    delete[] centers;
}

// ---------- TinyTetBvh::constructFromTetrahedra ----------

void TinyTetBvh::constructFromTetrahedra(
    const Pu32* tetrahedra,
    Pi32 numTetrahedra,
    const Vec3* points,
    TinyTetBvh& result,
    Pf32 enlargement)
{
    if (numTetrahedra == 0) {
        result.mTree.clear();
        return;
    }

    // Compute per-tet AABBs (matches PhysX: constructFromTetrahedra)
    AabbTreeBounds tetBounds;
    tetBounds.init(numTetrahedra);

    for (Pi32 i = 0; i < numTetrahedra; i++) {
        Pi32 i0 = Pi32(tetrahedra[4 * i]);
        Pi32 i1 = Pi32(tetrahedra[4 * i + 1]);
        Pi32 i2 = Pi32(tetrahedra[4 * i + 2]);
        Pi32 i3 = Pi32(tetrahedra[4 * i + 3]);

        Bounds3& b = tetBounds.getBounds()[i];
        b.setEmpty();
        b.include(points[i0]);
        b.include(points[i1]);
        b.include(points[i2]);
        b.include(points[i3]);

        // PhysX: box.fattenFast(enlargement)
        b.fattenFast(enlargement);
    }

    buildAabbTree(numTetrahedra, tetBounds, result.mTree);
}

} // namespace nt
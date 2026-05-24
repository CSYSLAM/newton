// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// PhysX-exact closest-distance traversal controller for tet mesh queries.
// Ported from GuAABBTree.h ClosestDistanceToTetmeshTraversalController.

#pragma once

#include "nt_types.h"
#include "nt_physx_aabb_tree.h"
#include "nt_physx_distance_tetrahedron.h"

namespace nt {

class ClosestDistanceToTetmeshTraversalController {
public:
    ClosestDistanceToTetmeshTraversalController()
        : mClosestDistanceSquared(NT_MAX_F32), mTetrahedra(nullptr), mPoints(nullptr), mNodes(nullptr),
          mQueryPoint(0.0f), mClosestPoint(0.0f), mClosestTetId(-1) {}

    ClosestDistanceToTetmeshTraversalController(
        const Pu32* tetrahedra,
        const Vec3* points,
        const BvhNode* nodes)
        : mClosestDistanceSquared(NT_MAX_F32), mTetrahedra(tetrahedra), mPoints(points), mNodes(nodes),
          mQueryPoint(0.0f), mClosestPoint(0.0f), mClosestTetId(-1)
    {
        initialize(tetrahedra, points, nodes);
    }

    void initialize(
        const Pu32* tetrahedra,
        const Vec3* points,
        const BvhNode* nodes)
    {
        mTetrahedra = tetrahedra;
        mPoints = points;
        mNodes = nodes;
        mQueryPoint = Vec3(0.0f);
        mClosestPoint = Vec3(0.0f);
        mClosestTetId = -1;
        mClosestDistanceSquared = NT_MAX_F32;
    }

    void setQueryPoint(const Vec3& queryPoint) {
        mQueryPoint = queryPoint;
        mClosestDistanceSquared = NT_MAX_F32;
        mClosestPoint = Vec3(0.0f);
        mClosestTetId = -1;
    }

    const Vec3& getClosestPoint() const { return mClosestPoint; }
    Pi32 getClosestTetId() const { return mClosestTetId; }

    void setClosestStart(Pf32 closestDistanceSquared, Pi32 closestTetId, const Vec3& closestPoint) {
        mClosestDistanceSquared = closestDistanceSquared;
        mClosestTetId = closestTetId;
        mClosestPoint = closestPoint;
    }

    Pf32 distancePointBoxSquared(const Bounds3& box, const Vec3& point) const {
        // PhysX: box.minimum.maximum(box.maximum.minimum(point))
        Vec3 closestPt = box.minimum.maximum(box.maximum.minimum(point));
        return (closestPt - point).magnitudeSquared();
    }

    TraversalControl analyze(const BvhNode& node, Pi32 /*nodeIndex*/) {
        // PhysX: if (distancePointBoxSquared(node.mBV, mQueryPoint) >= mClosestDistanceSquared)
        // Note: PhysX uses >= (prune when equal), not > (strict)
        if (distancePointBoxSquared(node.mBV, mQueryPoint) >= mClosestDistanceSquared)
            return TraversalControl::eDontGoDeeper;

        if (node.isLeaf()) {
            // PhysX: single primitive per leaf (limit=1)
            const Pi32 j = node.getPrimitiveIndex();
            const Pu32* tet = &mTetrahedra[4 * j];

            Vec3 cp;
            Pf32 cpDist2;
            closestPtPointTetrahedronWithInsideCheck(
                mQueryPoint,
                mPoints[Pi32(tet[0])], mPoints[Pi32(tet[1])], mPoints[Pi32(tet[2])], mPoints[Pi32(tet[3])],
                cp, cpDist2);

            if (cpDist2 < mClosestDistanceSquared) {
                mClosestDistanceSquared = cpDist2;
                mClosestTetId = j;
                mClosestPoint = cp;
            }
            if (cpDist2 == 0.0f)
                return TraversalControl::eAbort;

            return TraversalControl::eDontGoDeeper;
        }

        // Internal node — PhysX-exact: per-child pruning + ordering
        const BvhNode& nodePos = mNodes[node.getPosIndex()];
        const Pf32 distSquaredPos = distancePointBoxSquared(nodePos.mBV, mQueryPoint);
        const BvhNode& nodeNeg = mNodes[node.getNegIndex()];
        const Pf32 distSquaredNeg = distancePointBoxSquared(nodeNeg.mBV, mQueryPoint);

        if (distSquaredPos < distSquaredNeg) {
            if (distSquaredPos < mClosestDistanceSquared)
                return TraversalControl::eGoDeeper;
        } else {
            if (distSquaredNeg < mClosestDistanceSquared)
                return TraversalControl::eGoDeeperNegFirst;
        }
        return TraversalControl::eDontGoDeeper;
    }

private:
    Pf32 mClosestDistanceSquared;
    const Pu32* mTetrahedra;
    const Vec3* mPoints;
    const BvhNode* mNodes;
    Vec3 mQueryPoint;
    Vec3 mClosestPoint;
    Pi32 mClosestTetId;
};

} // namespace nt

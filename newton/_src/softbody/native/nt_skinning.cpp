// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// Volume Skinning implementation.
// Uses PhysX-exact TinyTetBvh + ClosestDistanceToTetmeshTraversalController.
// Matches ExtDeformableSkinning.cpp initializeInterpolatedVertices:
//   constructFromTetrahedra → traversalController → setQueryPoint →
//   traverse → getClosestTetId/getClosestPoint → computeBarycentricExact

#include "nt_skinning.h"
#include "nt_types.h"
#include "nt_physx_aabb_tree.h"
#include "nt_physx_tetmesh_traversal.h"
#include "nt_physx_distance_tetrahedron.h"

namespace nt {

void VolumeSkinning::compute(const std::vector<Vec3>& tetVerts,
                             const std::vector<Pu32>& tetIndices,
                             const std::vector<Vec3>& renderVerts,
                             VolumeEmbeddingImpl impl)
{
    compute(tetVerts.data(), (Pi32)tetVerts.size(),
            tetIndices.data(), (Pi32)tetIndices.size(),
            renderVerts.data(), (Pi32)renderVerts.size(),
            impl);
}

void VolumeSkinning::compute(const Vec3* tetVerts, Pi32 nbTetVerts,
                             const Pu32* tetIndices, Pi32 nbTetIndices,
                             const Vec3* renderVerts, Pi32 nbRenderVerts,
                             VolumeEmbeddingImpl impl)
{
    skinWeights.resize(nbRenderVerts);
    storedTetIndices.assign(tetIndices, tetIndices + nbTetIndices);

    Pi32 numTets = nbTetIndices / 4;
    if (numTets == 0) {
        for (Pi32 i = 0; i < nbRenderVerts; i++) {
            skinWeights[i].tetIndex = 0;
            skinWeights[i].weights[0] = 1.0f;
            skinWeights[i].weights[1] = 0.0f;
            skinWeights[i].weights[2] = 0.0f;
            skinWeights[i].weights[3] = 0.0f;
        }
        return;
    }

    switch (impl) {
    case VolumeEmbeddingImpl::PhysXExactCpu:
        computePhysXExactCpu(tetVerts, nbTetVerts, tetIndices, nbTetIndices,
                             renderVerts, nbRenderVerts);
        break;
    case VolumeEmbeddingImpl::LegacyCustom:
        // LegacyCustom falls through to PhysXExactCpu for now
        computePhysXExactCpu(tetVerts, nbTetVerts, tetIndices, nbTetIndices,
                             renderVerts, nbRenderVerts);
        break;
    }
}

void VolumeSkinning::computePhysXExactCpu(const Vec3* tetVerts, Pi32 nbTetVerts,
                                           const Pu32* tetIndices, Pi32 nbTetIndices,
                                           const Vec3* renderVerts, Pi32 nbRenderVerts)
{
    Pi32 numTets = nbTetIndices / 4;

    // Step 1: build TinyTetBvh (matches PhysX constructFromTetrahedra)
    TinyTetBvh bvh;
    TinyTetBvh::constructFromTetrahedra(tetIndices, numTets, tetVerts, bvh);

    // Step 2: create traversal controller (matches PhysX ClosestDistanceToTetmeshTraversalController)
    ClosestDistanceToTetmeshTraversalController controller;
    controller.initialize(tetIndices, tetVerts, bvh.mTree.data());

    // Step 3: for each render vertex, traverse BVH and compute barycentric
    for (Pi32 i = 0; i < nbRenderVerts; i++) {
        controller.setQueryPoint(renderVerts[i]);
        bvh.traverse(controller);

        const Pi32 closestTetId = controller.getClosestTetId();
        const Vec3 closestPoint = controller.getClosestPoint();

        SkinWeight& sw = skinWeights[i];
        if (closestTetId >= 0) {
            sw.tetIndex = closestTetId;
            computeBarycentricExact(
                tetVerts[tetIndices[4 * closestTetId]],
                tetVerts[tetIndices[4 * closestTetId + 1]],
                tetVerts[tetIndices[4 * closestTetId + 2]],
                tetVerts[tetIndices[4 * closestTetId + 3]],
                closestPoint,
                sw.weights
            );
        }
        else {
            sw.tetIndex = 0;
            sw.weights[0] = 1.0f;
            sw.weights[1] = 0.0f;
            sw.weights[2] = 0.0f;
            sw.weights[3] = 0.0f;
        }
    }
}

// ---- Deform ----

void VolumeSkinning::deform(const Vec3* deformedTetVerts, Vec3* deformedRenderVerts) const
{
    for (Pi32 i = 0; i < (Pi32)skinWeights.size(); i++) {
        const SkinWeight& sw = skinWeights[i];
        Pi32 ti = sw.tetIndex;

        Pi32 i0 = (Pi32)storedTetIndices[4 * ti];
        Pi32 i1 = (Pi32)storedTetIndices[4 * ti + 1];
        Pi32 i2 = (Pi32)storedTetIndices[4 * ti + 2];
        Pi32 i3 = (Pi32)storedTetIndices[4 * ti + 3];

        deformedRenderVerts[i] =
            deformedTetVerts[i0] * sw.weights[0] +
            deformedTetVerts[i1] * sw.weights[1] +
            deformedTetVerts[i2] * sw.weights[2] +
            deformedTetVerts[i3] * sw.weights[3];
    }
}

void VolumeSkinning::deform(const std::vector<Vec3>& deformedTetVerts,
                              std::vector<Vec3>& deformedRenderVerts) const
{
    deformedRenderVerts.resize(skinWeights.size());
    deform(deformedTetVerts.data(), deformedRenderVerts.data());
}

} // namespace nt
// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// Volume Skinning / Embedding.
// Uses PhysX-exact TinyTetBvh + ClosestDistanceToTetmeshTraversalController
// to find the closest point on the tet mesh for each render vertex, then
// computes barycentric coordinates in the closest tet.
// Matches ExtDeformableSkinning.cpp initializeInterpolatedVertices.

#pragma once

#include "nt_types.h"
#include <vector>

namespace nt {

enum class VolumeEmbeddingImpl {
    LegacyCustom,
    PhysXExactCpu
};

struct SkinWeight {
    Pi32 tetIndex;       // which tet this vertex is embedded in
    Pf32 weights[4];     // barycentric weights for the 4 tet vertices
};

class VolumeSkinning
{
public:
    VolumeSkinning() = default;
    ~VolumeSkinning() = default;

    void compute(const Vec3* tetVerts, Pi32 nbTetVerts,
                 const Pu32* tetIndices, Pi32 nbTetIndices,
                 const Vec3* renderVerts, Pi32 nbRenderVerts,
                 VolumeEmbeddingImpl impl = VolumeEmbeddingImpl::PhysXExactCpu);

    void compute(const std::vector<Vec3>& tetVerts,
                 const std::vector<Pu32>& tetIndices,
                 const std::vector<Vec3>& renderVerts,
                 VolumeEmbeddingImpl impl = VolumeEmbeddingImpl::PhysXExactCpu);

    const std::vector<SkinWeight>& getSkinWeights() const { return skinWeights; }

    void deform(const Vec3* deformedTetVerts, Vec3* deformedRenderVerts) const;

    void deform(const std::vector<Vec3>& deformedTetVerts,
                std::vector<Vec3>& deformedRenderVerts) const;

    Pi32 getEmbeddingCount() const { return (Pi32)skinWeights.size(); }

private:
    void computePhysXExactCpu(const Vec3* tetVerts, Pi32 nbTetVerts,
                              const Pu32* tetIndices, Pi32 nbTetIndices,
                              const Vec3* renderVerts, Pi32 nbRenderVerts);

    std::vector<SkinWeight> skinWeights;
    std::vector<Pu32> storedTetIndices;
};

} // namespace nt
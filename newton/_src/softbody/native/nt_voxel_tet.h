// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// VoxelTetrahedralizer, ported from PhysX ExtVoxelTetrahedralizer.h.
// Algorithm is identical; PhysX types replaced with standalone equivalents.

#pragma once

#include "nt_types.h"
#include "nt_bvh.h"
#include "nt_multi_list.h"
#include <vector>

namespace nt {

struct VoxelTetDebugStats {
    Pi32 numSurfaceVoxels;
    Pi32 numInnerVoxels;
    Pi32 numBorderVoxels;
    Pi32 numVoxelGridX;
    Pi32 numVoxelGridY;
    Pi32 numVoxelGridZ;
    Pi32 numUniqueVerts;
    Pi32 numSurfaceVerts;
    Pi32 numTets;
    Pi32 numInnerTets;
    Pi32 numBorderTets;
    Pi32 numEdges;
    Pf32 gridSpacing;
};

class VoxelTetrahedralizer
{
public:
    VoxelTetrahedralizer();

    void clear();
    void createTetMesh(const std::vector<Vec3>& verts, const std::vector<Pu32>& triIds,
        Pi32 resolution, Pi32 numRelaxationIters = 5, Pf32 relMinTetVolume = 0.05f,
        Pf32 surfaceDistRatio = 0.2f);

    void readBack(std::vector<Vec3>& tetVertices, std::vector<Pu32>& tetIndices);

    VoxelTetDebugStats getDebugStats() const;

private:
    void voxelize(Pu32 resolution);
    void createTets(bool subdivBorder, Pu32 numTetsPerVoxel);
    void buildBVH();
    void createUniqueTetVertices();
    void findTargetPositions(Pf32 surfaceDist);
    void conserveVolume(Pf32 relMinVolume);
    void relax(Pi32 numIters, Pf32 relMinVolume);

    // input mesh

    std::vector<Vec3> surfaceVerts;
    std::vector<Pi32> surfaceTriIds;
    Bounds3 surfaceBounds;

    // voxel grid

    struct Voxel {
        void init(Pi32 _xi, Pi32 _yi, Pi32 _zi)
        {
            xi = _xi; yi = _yi; zi = _zi;
            for (Pi32 i = 0; i < 6; i++)
                neighbors[i] = -1;
            for (Pi32 i = 0; i < 8; i++)
                ids[i] = -1;
            parent = -1;
            inner = false;
        }
        bool isAt(Pi32 _xi, Pi32 _yi, Pi32 _zi) {
            return xi == _xi && yi == _yi && zi == _zi;
        }
        Pi32 xi, yi, zi;
        Pi32 neighbors[6];
        Pi32 parent;
        Pi32 ids[8];
        bool inner;
    };

    Vec3 gridOrigin;
    Pf32 gridSpacing;
    Pi32 numVoxelGridX;
    Pi32 numVoxelGridY;
    Pi32 numVoxelGridZ;
    std::vector<Voxel> voxels;

    BVH bvh;

    // tet mesh

    std::vector<Vec3> tetVerts;
    std::vector<Vec3> origTetVerts;
    std::vector<Pi32> tetIds;

    // relaxation

    std::vector<bool> isSurfaceVert;
    std::vector<Vec3> targetVertPos;
    std::vector<Pi32> queryTris;
    std::vector<Pi32> edgeIds;
};

} // namespace nt

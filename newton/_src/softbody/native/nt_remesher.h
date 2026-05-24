// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// Surface Remesher, ported from PhysX ExtRemesher.h/cpp.
// Algorithm is identical; PhysX types replaced with standalone equivalents.
// Pipeline: voxelize(resolution) -> marchingCubes -> removeDuplicateVertices
//           -> pruneInternalSurfaces -> project -> optional createVertexMap
//           -> computeNormals

#pragma once

#include "nt_types.h"
#include <vector>

namespace nt {

struct RemeshDebugStats {
    Pi32 inputVertexCount = 0;
    Pi32 inputTriangleCount = 0;
    Pi32 outputVertexCount = 0;
    Pi32 outputTriangleCount = 0;
    Pi32 numCells = 0;
    Pi32 numIslands = 0;
    Pi32 numPrunedIslands = 0;
};

class SurfaceRemesher
{
public:
    SurfaceRemesher();
    ~SurfaceRemesher();

    void clear();

    void remesh(const Vec3* inputVerts, Pi32 nbVertices,
                const Pu32* inputTriIds, Pi32 nbTriangleIndices,
                Pi32 resolution = 100, std::vector<Pu32>* vertexMap = nullptr);

    void remesh(const std::vector<Vec3>& inputVerts,
                const std::vector<Pu32>& inputTriIds,
                Pi32 resolution = 100, std::vector<Pu32>* vertexMap = nullptr);

    void readBack(std::vector<Vec3>& outputVertices, std::vector<Pu32>& outputTriIds);

    RemeshDebugStats getDebugStats() const;

private:
    static const Pi32 HASH_SIZE = 170111;

    struct Cell {
        Pi32 xi, yi, zi;
        Pi32 next;
        void init(Pi32 _xi, Pi32 _yi, Pi32 _zi) {
            xi = _xi; yi = _yi; zi = _zi; next = -1;
        }
    };

    std::vector<Cell> cells;
    std::vector<Pi32> firstCell;

    std::vector<Vec3> vertices;
    std::vector<Pi32> triIds;
    std::vector<Vec3> normals;
    std::vector<Pi32> triNeighbors;
    std::vector<Pi32> cellOfVertex;

    Bounds3 meshBounds;
    Pf32 gridSpacing;

    RemeshDebugStats debugStats;

    void addCell(Pi32 xi, Pi32 yi, Pi32 zi);
    Pi32 getCellNr(Pi32 xi, Pi32 yi, Pi32 zi) const;
    bool cellExists(Pi32 xi, Pi32 yi, Pi32 zi) const;

    void marchingCubes();
    void removeDuplicateVertices();
    void findTriNeighbors();
    void pruneInternalSurfaces();
    void project(const Vec3* inputVerts, const Pu32* inputTriIds,
                 Pi32 nbTriangleIndices, Pf32 searchDist, Pf32 surfaceDist);
    void createVertexMap(const Vec3* inputVerts, Pi32 nbVertices,
                         const Vec3& gridOrigin, Pf32& spacing,
                         std::vector<Pu32>& vertexMap);
    void computeNormals();
};

} // namespace nt

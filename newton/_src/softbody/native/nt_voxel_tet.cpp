// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// VoxelTetrahedralizer implementation, ported from PhysX ExtVoxelTetrahedralizer.cpp.
// Algorithm is line-by-line identical; PxVec3->Vec3, PxArray->std::vector, etc.

#include "nt_voxel_tet.h"
#include "nt_union_find.h"

using namespace nt;

// -------------------------------------------------------------------------------------
static Pi32 cubeNeighbors[6][3] = { { -1,0,0 }, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1} };
static const Pi32 cubeCorners[8][3] = { {0,0,0}, {1,0,0},{1,1,0},{0,1,0}, {0,0,1}, {1,0,1},{1,1,1},{0,1,1} };
static const Pi32 cubeFaces[6][4] = { {0,3,7,4},{1,2,6,5},{0,1,5,4},{3,2,6,7},{0,1,2,3},{4,5,6,7} };
static const Pi32 oppNeighbor[6] = { 1,0,3,2,5,4 };

static const Pi32 tetEdges[12][2] = { {0,1},{1,2},{2,0},{0,3},{1,3},{2,3},  {1,0},{2,1},{0,2},{3,0},{3,1},{3,2} };

static Pi32 cubeSixTets[6][4] = {
    { 0, 4, 5, 7 },{ 1, 5, 6, 7 },{ 1, 0, 5, 7 },{ 1, 2, 3, 6 },{ 3, 1, 6, 7 },{ 0, 1, 3, 7 } };

static Pi32 cubeFiveTets[2][5][4] = {
    { { 0, 1, 2, 5 },{ 0, 2, 3, 7 },{ 0, 5, 2, 7 },{ 0, 5, 7, 4 },{ 2, 7, 5, 6 } },
    { { 1, 2, 3, 6 },{ 1, 3, 0, 4 },{ 1, 6, 3, 4 },{ 1, 6, 4, 5 },{ 3, 4, 6, 7 } },
};

static Pi32 cubeSixSubdivTets[12][4] = {
    {0,4,5,8}, {0,5,1,8}, {3,2,6,8}, {3,6,7,8},
    {0,3,7,8}, {0,7,4,8}, {1,5,6,8}, {1,6,2,8},
    {0,1,3,8}, {1,2,3,8}, {5,4,7,8}, {5,7,6,8}
};

static Pi32 cubeFiveSubdivTets[2][12][4] = {
    {
        {0,1,2,8}, {0,2,3,8}, {4,7,5,8}, {5,7,6,8},
        {0,7,4,8}, {0,3,7,8}, {1,5,2,8}, {2,5,6,8},
        {0,5,1,8}, {0,4,5,8}, {3,2,7,8}, {2,6,7,8}
    },
    {
        {0,1,3,8}, {1,2,3,8}, {4,7,6,8}, {4,6,5,8},
        {0,3,4,8}, {3,7,4,8}, {1,5,6,8}, {1,6,2,8},
        {0,4,1,8}, {1,4,5,8}, {3,2,6,8}, {3,6,7,8}
    }
};

static const Pi32 volIdOrder[4][3] = { {1, 3, 2}, {0, 2, 3}, {0, 3, 1}, {0, 1, 2} };

// -------------------------------------------------------------------------------------
static bool boxTriangleIntersection(
    Vec3 p0, Vec3 p1, Vec3 p2, Vec3 center, Vec3 extents);
static void getClosestPointOnTriangle(
    Vec3 p1, Vec3 p2, Vec3 p3, Vec3 p, Vec3& closest, Vec3& bary);

// -------------------------------------------------------------------------------------
VoxelTetrahedralizer::VoxelTetrahedralizer()
{
    clear();
}

// -------------------------------------------------------------------------------------
void VoxelTetrahedralizer::clear()
{
    surfaceVerts.clear();
    surfaceTriIds.clear();
    surfaceBounds.setEmpty();

    tetVerts.clear();
    origTetVerts.clear();
    isSurfaceVert.clear();
    targetVertPos.clear();

    tetIds.clear();
    voxels.clear();
    gridOrigin = Vec3(0);
    gridSpacing = 0.0f;
    numVoxelGridX = 0;
    numVoxelGridY = 0;
    numVoxelGridZ = 0;
}

// -----------------------------------------------------------------------------------
void VoxelTetrahedralizer::readBack(std::vector<Vec3>& _tetVertices, std::vector<Pu32>& _tetIndices)
{
    _tetVertices = tetVerts;
    _tetIndices.resize(tetIds.size());

    for (Pu32 i = 0; i < tetIds.size(); i++)
        _tetIndices[i] = Pu32(tetIds[i]);
}

// -----------------------------------------------------------------------------------
VoxelTetDebugStats VoxelTetrahedralizer::getDebugStats() const
{
    VoxelTetDebugStats stats{};
    stats.numSurfaceVoxels = 0;
    stats.numInnerVoxels = 0;
    stats.numBorderVoxels = 0;
    stats.numVoxelGridX = numVoxelGridX;
    stats.numVoxelGridY = numVoxelGridY;
    stats.numVoxelGridZ = numVoxelGridZ;
    for (Pu32 i = 0; i < voxels.size(); i++) {
        if (voxels[i].inner)
            stats.numInnerVoxels++;
        else {
            bool hasNeighbor = false;
            for (Pi32 j = 0; j < 6; j++)
                if (voxels[i].neighbors[j] >= 0) { hasNeighbor = true; break; }
            if (hasNeighbor)
                stats.numBorderVoxels++;
            else
                stats.numSurfaceVoxels++;
        }
    }
    stats.numUniqueVerts = Pi32(tetVerts.size());
    stats.numSurfaceVerts = 0;
    for (Pu32 i = 0; i < isSurfaceVert.size(); i++)
        if (isSurfaceVert[i])
            stats.numSurfaceVerts++;
    stats.numTets = Pi32(tetIds.size()) / 4;
    stats.numInnerTets = 0;
    stats.numBorderTets = 0;
    stats.numEdges = Pi32(edgeIds.size()) / 2;
    stats.gridSpacing = gridSpacing;
    return stats;
}

// -----------------------------------------------------------------------------------
void VoxelTetrahedralizer::createTetMesh(const std::vector<Vec3>& verts, const std::vector<Pu32>& triIds,
    Pi32 resolution, Pi32 numRelaxationIters, Pf32 relMinTetVolume, Pf32 surfaceDistRatio)
{
    surfaceVerts = verts;
    surfaceTriIds.resize(triIds.size());
    for (Pu32 i = 0; i < triIds.size(); i++)
        surfaceTriIds[i] = Pi32(triIds[i]);

    surfaceBounds.setEmpty();

    for (Pu32 i = 0; i < surfaceVerts.size(); i++)
        surfaceBounds.include(surfaceVerts[i]);

    buildBVH();

    voxelize(resolution);

    bool subdivBorder = true;
    int numTetsPerVoxel = 5;       // or 6

    createTets(subdivBorder, numTetsPerVoxel);

    findTargetPositions(surfaceDistRatio * gridSpacing);

    relax(numRelaxationIters, relMinTetVolume);
}

// -----------------------------------------------------------------------------------
void VoxelTetrahedralizer::buildBVH()
{
    Pi32 numTris = Pi32(surfaceTriIds.size()) / 3;

    if (numTris == 0)
        return;

    std::vector<Bounds3> bvhBounds(numTris);

    for (Pi32 i = 0; i < numTris; i++) {
        Bounds3& b = bvhBounds[i];
        b.setEmpty();
        b.include(surfaceVerts[surfaceTriIds[3 * i]]);
        b.include(surfaceVerts[surfaceTriIds[3 * i + 1]]);
        b.include(surfaceVerts[surfaceTriIds[3 * i + 2]]);
    }

    bvh.build(&bvhBounds[0], bvhBounds.size());
}

// -----------------------------------------------------------------------------------
void VoxelTetrahedralizer::voxelize(Pu32 resolution)
{
    tetIds.clear();
    tetVerts.clear();

    Bounds3 meshBounds;
    meshBounds.setEmpty();

    for (Pu32 i = 0; i < surfaceVerts.size(); i++)
        meshBounds.include(surfaceVerts[i]);

    gridSpacing = meshBounds.getDimensions().magnitude() / resolution;
    meshBounds.fattenSafe(gridSpacing);
    gridOrigin = meshBounds.minimum;

    voxels.clear();

    Pi32 numX = Pi32((meshBounds.maximum.x - meshBounds.minimum.x) / gridSpacing) + 1;
    Pi32 numY = Pi32((meshBounds.maximum.y - meshBounds.minimum.y) / gridSpacing) + 1;
    Pi32 numZ = Pi32((meshBounds.maximum.z - meshBounds.minimum.z) / gridSpacing) + 1;
    numVoxelGridX = numX;
    numVoxelGridY = numY;
    numVoxelGridZ = numZ;
    Pi32 numCells = numX * numY * numZ;

    std::vector<Pi32> voxelOfCell(numCells, -1);
    Bounds3 voxelBounds, faceBounds;

    // create intersected voxels

    for (Pi32 i = 0; i < numCells; i++) {
        Pi32 zi = i % numZ;
        Pi32 yi = (i / numZ) % numY;
        Pi32 xi = (i / numZ / numY);

        voxelBounds.minimum = meshBounds.minimum + Vec3(Pf32(xi), Pf32(yi), Pf32(zi)) * gridSpacing;
        voxelBounds.maximum = voxelBounds.minimum + Vec3(gridSpacing);

        bvh.query(voxelBounds, queryTris);

        for (Pu32 j = 0; j < queryTris.size(); j++) {
            Pi32 triNr = queryTris[j];

            const Vec3& p0 = surfaceVerts[surfaceTriIds[3 * triNr]];
            const Vec3& p1 = surfaceVerts[surfaceTriIds[3 * triNr + 1]];
            const Vec3& p2 = surfaceVerts[surfaceTriIds[3 * triNr + 2]];

            if (boxTriangleIntersection(p0, p1, p2, voxelBounds.getCenter(), voxelBounds.getExtents())) {
                // volume
                if (voxelOfCell[i] < 0) {
                    voxelOfCell[i] = Pi32(voxels.size());
                    voxels.resize(voxels.size() + 1);
                    voxels.back().init(xi, yi, zi);
                }
            }
        }
    }

    // flood outside

    std::vector<Pi32> stack;
    stack.push_back(0);

    while (!stack.empty()) {
        Pi32 nr = stack.back();
        stack.pop_back();

        if (voxelOfCell[nr] == -1) {
            voxelOfCell[nr] = -2;      // outside

            Pi32 z0 = nr % numZ;
            Pi32 y0 = (nr / numZ) % numY;
            Pi32 x0 = (nr / numZ / numY);

            for (Pi32 i = 0; i < 6; i++) {
                Pi32 xi = x0 + cubeNeighbors[i][0];
                Pi32 yi = y0 + cubeNeighbors[i][1];
                Pi32 zi = z0 + cubeNeighbors[i][2];
                if (xi >= 0 && xi < numX && yi >= 0 && yi < numY && zi >= 0 && zi < numZ) {
                    Pi32 adj = (xi * numY + yi) * numZ + zi;
                    if (voxelOfCell[adj] == -1)
                        stack.push_back(adj);
                }
            }
        }
    }

    // create voxels for the inside

    for (Pi32 i = 0; i < numCells; i++) {
        if (voxelOfCell[i] == -1) {
            voxelOfCell[i] = Pi32(voxels.size());
            voxels.resize(voxels.size() + 1);
            Pi32 zi = i % numZ;
            Pi32 yi = (i / numZ) % numY;
            Pi32 xi = (i / numZ / numY);
            voxels.back().init(xi, yi, zi);
            voxels.back().inner = true;
        }
    }

    // find neighbors

    for (Pu32 i = 0; i < voxels.size(); i++) {
        Voxel& v = voxels[i];

        voxelBounds.minimum = meshBounds.minimum + Vec3(Pf32(v.xi), Pf32(v.yi), Pf32(v.zi)) * gridSpacing;
        voxelBounds.maximum = voxelBounds.minimum + Vec3(gridSpacing);

        for (Pi32 j = 0; j < 6; j++) {

            Pi32 xi = v.xi + cubeNeighbors[j][0];
            Pi32 yi = v.yi + cubeNeighbors[j][1];
            Pi32 zi = v.zi + cubeNeighbors[j][2];

            if (xi < 0 || xi >= numX || yi < 0 || yi >= numY || zi < 0 || zi >= numZ)
                continue;

            Pi32 neighbor = voxelOfCell[(xi * numY + yi) * numZ + zi];
            if (neighbor < 0)
                continue;

            if (v.inner || voxels[neighbor].inner) {
                v.neighbors[j] = neighbor;
                continue;
            }

            faceBounds = voxelBounds;
            Pf32 eps = 1e-4f;
            switch (j) {
                case 0: faceBounds.maximum.x = faceBounds.minimum.x + eps; break;
                case 1: faceBounds.minimum.x = faceBounds.maximum.x - eps; break;
                case 2: faceBounds.maximum.y = faceBounds.minimum.y + eps; break;
                case 3: faceBounds.minimum.y = faceBounds.maximum.y - eps; break;
                case 4: faceBounds.maximum.z = faceBounds.minimum.z + eps; break;
                case 5: faceBounds.minimum.z = faceBounds.maximum.z - eps; break;
            }
            bvh.query(faceBounds, queryTris);

            bool intersected = false;

            for (Pu32 k = 0; k < queryTris.size(); k++) {
                Pi32 triNr = queryTris[k];

                const Vec3& p0 = surfaceVerts[surfaceTriIds[3 * triNr]];
                const Vec3& p1 = surfaceVerts[surfaceTriIds[3 * triNr + 1]];
                const Vec3& p2 = surfaceVerts[surfaceTriIds[3 * triNr + 2]];

                if (boxTriangleIntersection(p0, p1, p2, faceBounds.getCenter(), faceBounds.getExtents())) {
                    intersected = true;
                    break;
                }
            }

            if (intersected)
                v.neighbors[j] = neighbor;
        }
    }
}

// -----------------------------------------------------------------------------------
void VoxelTetrahedralizer::createUniqueTetVertices()
{
    // start with each voxel having its own vertices

    std::vector<Vec3> verts;
    for (Pu32 i = 0; i < voxels.size(); i++) {
        Voxel& v = voxels[i];

        for (Pi32 j = 0; j < 8; j++) {
            v.ids[j] = Pi32(verts.size());
            verts.push_back(gridOrigin + Vec3(
                Pf32(v.xi + cubeCorners[j][0]),
                Pf32(v.yi + cubeCorners[j][1]),
                Pf32(v.zi + cubeCorners[j][2])) * gridSpacing);
        }
    }

    // unify vertices

    UnionFind* u = new UnionFind();
    u->init(Pi32(verts.size()));

    for (Pu32 i = 0; i < voxels.size(); i++) {
        Voxel& v0 = voxels[i];
        for (Pi32 j = 0; j < 6; j++) {
            Pi32 n = v0.neighbors[j];
            if (n < 0)
                continue;
            Voxel& v1 = voxels[n];

            for (Pi32 k = 0; k < 4; k++) {
                Pi32 id0 = v0.ids[cubeFaces[j][k]];
                Pi32 id1 = v1.ids[cubeFaces[oppNeighbor[j]][k]];
                u->makeSet(id0, id1);
            }
        }
    }

    u->computeSetNrs();

    tetVerts.clear();

    for (Pu32 i = 0; i < voxels.size(); i++) {
        Voxel& v = voxels[i];

        for (Pi32 j = 0; j < 8; j++) {
            Pi32 setNr = u->getSetNr(v.ids[j]);
            if (Pi32(tetVerts.size()) <= setNr)
                tetVerts.resize(setNr + 1, Vec3(0));
            tetVerts[setNr] = verts[v.ids[j]];
            v.ids[j] = setNr;
        }
    }

    origTetVerts = tetVerts;

    delete u;
}

// -------------------------------------------------------------------------------------
void VoxelTetrahedralizer::findTargetPositions(Pf32 surfaceDist)
{
    targetVertPos = tetVerts;

    for (Pu32 i = 0; i < voxels.size(); i++) {

        Voxel& v = voxels[i];

        Bounds3 voxelBounds;
        voxelBounds.minimum = gridOrigin + Vec3(Pf32(v.xi), Pf32(v.yi), Pf32(v.zi)) * gridSpacing;
        voxelBounds.maximum = voxelBounds.minimum + Vec3(gridSpacing);
        voxelBounds.fattenFast(0.1f * gridSpacing);
        bvh.query(voxelBounds, queryTris);

        for (Pi32 j = 0; j < 8; j++) {
            Pi32 id = v.ids[j];
            if (!isSurfaceVert[id])
                continue;

            Vec3& p = tetVerts[id];

            Pf32 minDist2 = NT_MAX_F32;
            Vec3 closest(0);

            for (Pu32 k = 0; k < queryTris.size(); k++) {

                Pi32 triNr = queryTris[k];
                const Vec3& p0 = surfaceVerts[surfaceTriIds[3 * triNr]];
                const Vec3& p1 = surfaceVerts[surfaceTriIds[3 * triNr + 1]];
                const Vec3& p2 = surfaceVerts[surfaceTriIds[3 * triNr + 2]];
                Vec3 c, bary;
                getClosestPointOnTriangle(p0, p1, p2, p, c, bary);
                Pf32 dist2 = (c - p).magnitudeSquared();
                if (dist2 < minDist2) {
                    minDist2 = dist2;
                    closest = c;
                }
            }
            if (minDist2 < NT_MAX_F32) {
                Vec3 n = p - closest;
                n.normalize();
                targetVertPos[id] = closest + n * surfaceDist;
            }
        }
    }
}

// -----------------------------------------------------------------------------------
void VoxelTetrahedralizer::createTets(bool subdivBorder, Pu32 numTetsPerVoxel)
{
    if (numTetsPerVoxel < 5 || numTetsPerVoxel > 6)
        return;

    createUniqueTetVertices();

    std::vector<Voxel> prevVoxels;

    std::vector<Pi32> numVertVoxels(tetVerts.size(), 0);
    tetIds.clear();

    for (Pu32 i = 0; i < voxels.size(); i++) {
        Voxel& v = voxels[i];
        for (Pi32 j = 0; j < 8; j++)
            numVertVoxels[v.ids[j]]++;

        Pi32 parity = (v.xi + v.yi + v.zi) % 2;

        if (v.inner || !subdivBorder) {
            if (numTetsPerVoxel == 6) {
                for (Pi32 j = 0; j < 6; j++) {
                    tetIds.push_back(v.ids[cubeSixTets[j][0]]);
                    tetIds.push_back(v.ids[cubeSixTets[j][1]]);
                    tetIds.push_back(v.ids[cubeSixTets[j][2]]);
                    tetIds.push_back(v.ids[cubeSixTets[j][3]]);
                }
            }
            else if (numTetsPerVoxel == 5) {
                for (Pi32 j = 0; j < 5; j++) {
                    tetIds.push_back(v.ids[cubeFiveTets[parity][j][0]]);
                    tetIds.push_back(v.ids[cubeFiveTets[parity][j][1]]);
                    tetIds.push_back(v.ids[cubeFiveTets[parity][j][2]]);
                    tetIds.push_back(v.ids[cubeFiveTets[parity][j][3]]);
                }
            }
        }
        else {
            Vec3 p(0);
            for (Pi32 j = 0; j < 8; j++)
                p += tetVerts[v.ids[j]];
            p = p * (1.0f / 8.0f);
            Pi32 newId = Pi32(tetVerts.size());
            tetVerts.push_back(p);
            origTetVerts.push_back(p);
            numVertVoxels.push_back(8);

            for (Pi32 j = 0; j < 12; j++) {

                const int* localIds;
                if (numTetsPerVoxel == 6)
                    localIds = cubeSixSubdivTets[j];
                else
                    localIds = cubeFiveSubdivTets[parity][j];

                for (Pi32 k = 0; k < 4; k++) {
                    Pi32 id = localIds[k] < 8 ? v.ids[localIds[k]] : newId;
                    tetIds.push_back(id);
                }
            }
        }
    }

    isSurfaceVert.resize(tetVerts.size(), false);
    for (Pu32 i = 0; i < tetVerts.size(); i++)
        isSurfaceVert[i] = numVertVoxels[i] < 8;

    // randomize tets (disabled in PhysX, keep disabled)

    // edges

    MultiList<int> adjVerts;
    edgeIds.clear();

    adjVerts.clear();
    adjVerts.reserve(Pi32(tetVerts.size()));

    Pu32 numTets = tetIds.size() / 4;

    for (Pu32 i = 0; i < numTets; i++) {
        for (Pi32 j = 0; j < 6; j++) {
            Pi32 id0 = tetIds[4 * i + tetEdges[j][0]];
            Pi32 id1 = tetIds[4 * i + tetEdges[j][1]];

            if (!adjVerts.exists(id0, id1)) {
                edgeIds.push_back(id0);
                edgeIds.push_back(id1);

                adjVerts.addUnique(id0, id1);
                adjVerts.addUnique(id1, id0);
            }
        }
    }
}

// -----------------------------------------------------------------------------------
void VoxelTetrahedralizer::conserveVolume(Pf32 relMinVolume)
{
    Vec3 grads[4];
    Pu32 numTets = tetIds.size() / 4;

    for (Pu32 i = 0; i < numTets; i++) {
        Pi32* ids = &tetIds[4 * i];

        Pf32 w = 0.0f;

        for (Pi32 j = 0; j < 4; j++) {
            Pi32 id0 = ids[volIdOrder[j][0]];
            Pi32 id1 = ids[volIdOrder[j][1]];
            Pi32 id2 = ids[volIdOrder[j][2]];

            grads[j] = (tetVerts[id1] - tetVerts[id0]).cross(tetVerts[id2] - tetVerts[id0]);
            w += grads[j].magnitudeSquared();
        }

        if (w == 0.0f)
            continue;

        Vec3& p0 = tetVerts[ids[0]];
        Pf32 V = (tetVerts[ids[1]] - p0).cross(tetVerts[ids[2]] - p0).dot(tetVerts[ids[3]] - p0);

        Vec3& origP0 = origTetVerts[ids[0]];
        Pf32 origV = (origTetVerts[ids[1]] - origP0).cross(origTetVerts[ids[2]] - origP0).dot(origTetVerts[ids[3]] - origP0);

        Pf32 minV = relMinVolume * origV;

        if (V < minV) {

            Pf32 C = V - minV;
            Pf32 lambda = -C / w;

            for (Pi32 j = 0; j < 4; j++) {
                tetVerts[ids[j]] += grads[j] * lambda;
            }
        }
    }
}

// -------------------------------------------------------------------------------------
void VoxelTetrahedralizer::relax(Pi32 numIters, Pf32 relMinVolume)
{
    const Pf32 targetScale = 0.3f;
    const Pf32 edgeScale = 0.3f;

    for (Pi32 iter = 0; iter < numIters; iter++) {
        Pu32 numVerts = tetVerts.size();

        for (Pu32 i = 0; i < numVerts; i++) {
            if (isSurfaceVert[i]) {
                Vec3 offset = (targetVertPos[i] - tetVerts[i]) * targetScale;
                tetVerts[i] += offset;
            }
        }

        for (Pu32 i = 0; i < edgeIds.size(); i += 2) {
            Pi32 id0 = edgeIds[i];
            Pi32 id1 = edgeIds[i + 1];
            Pf32 w0 = isSurfaceVert[id0] ? 0.0f : 1.0f;
            Pf32 w1 = isSurfaceVert[id1] ? 0.0f : 1.0f;
            Pf32 w = w0 + w1;
            if (w == 0.0f)
                continue;
            Vec3& p0 = tetVerts[id0];
            Vec3& p1 = tetVerts[id1];

            Vec3 e = (p1 - p0) * edgeScale;

            if (w == 1.0f)
                e *= 0.5f;

            p0 += w0 / w * e;
            p1 -= w1 / w * e;
        }
        conserveVolume(relMinVolume);
    }

    Pi32 volIters = 2;

    for (Pi32 volIter = 0; volIter < volIters; volIter++)
        conserveVolume(relMinVolume);
}

// -----------------------------------------------------------------------------------
static Pf32 max3(Pf32 f0, Pf32 f1, Pf32 f2) {
    return ntMax(f0, ntMax(f1, f2));
}

static Pf32 min3(Pf32 f0, Pf32 f1, Pf32 f2) {
    return ntMin(f0, ntMin(f1, f2));
}

static Pf32 minMax(Pf32 f0, Pf32 f1, Pf32 f2) {
    return ntMax(-max3(f0, f1, f2), min3(f0, f1, f2));
}

// -----------------------------------------------------------------------------------
// PT: TODO: refactor with other SDK implementation
static bool boxTriangleIntersection(
    Vec3 p0, Vec3 p1, Vec3 p2, Vec3 center, Vec3 extents)
{
    Vec3 v0 = p0 - center, v1 = p1 - center, v2 = p2 - center;
    Vec3 f0 = p1 - p0, f1 = p2 - p1, f2 = p0 - p2;
    Pf32 r;

    Vec3 n = f0.cross(f1);
    Pf32 d = n.dot(v0);
    r = extents.x * fabsf(n.x) + extents.y * fabsf(n.y) + extents.z * fabsf(n.z);
    if (d > r || d < -r)
        return false;

    if (max3(v0.x, v1.x, v2.x) < -extents.x || min3(v0.x, v1.x, v2.x) > extents.x)
        return false;

    if (max3(v0.y, v1.y, v2.y) < -extents.y || min3(v0.y, v1.y, v2.y) > extents.y)
        return false;

    if (max3(v0.z, v1.z, v2.z) < -extents.z || min3(v0.z, v1.z, v2.z) > extents.z)
        return false;

    Vec3 a00(0.0f, -f0.z, f0.y);
    r = extents.y * fabsf(f0.z) + extents.z * fabsf(f0.y);
    if (minMax(v0.dot(a00), v1.dot(a00), v2.dot(a00)) > r)
        return false;

    Vec3 a01(0.0f, -f1.z, f1.y);
    r = extents.y * fabsf(f1.z) + extents.z * fabsf(f1.y);
    if (minMax(v0.dot(a01), v1.dot(a01), v2.dot(a01)) > r)
        return false;

    Vec3 a02(0.0f, -f2.z, f2.y);
    r = extents.y * fabsf(f2.z) + extents.z * fabsf(f2.y);
    if (minMax(v0.dot(a02), v1.dot(a02), v2.dot(a02)) > r)
        return false;

    Vec3 a10(f0.z, 0.0f, -f0.x);
    r = extents.x * fabsf(f0.z) + extents.z * fabsf(f0.x);
    if (minMax(v0.dot(a10), v1.dot(a10), v2.dot(a10)) > r)
        return false;

    Vec3 a11(f1.z, 0.0f, -f1.x);
    r = extents.x * fabsf(f1.z) + extents.z * fabsf(f1.x);
    if (minMax(v0.dot(a11), v1.dot(a11), v2.dot(a11)) > r)
        return false;

    Vec3 a12(f2.z, 0.0f, -f2.x);
    r = extents.x * fabsf(f2.z) + extents.z * fabsf(f2.x);
    if (minMax(v0.dot(a12), v1.dot(a12), v2.dot(a12)) > r)
        return false;

    Vec3 a20(-f0.y, f0.x, 0.0f);
    r = extents.x * fabsf(f0.y) + extents.y * fabsf(f0.x);
    if (minMax(v0.dot(a20), v1.dot(a20), v2.dot(a20)) > r)
        return false;

    Vec3 a21(-f1.y, f1.x, 0.0f);
    r = extents.x * fabsf(f1.y) + extents.y * fabsf(f1.x);
    if (minMax(v0.dot(a21), v1.dot(a21), v2.dot(a21)) > r)
        return false;

    Vec3 a22(-f2.y, f2.x, 0.0f);
    r = extents.x * fabsf(f2.y) + extents.y * fabsf(f2.x);
    if (minMax(v0.dot(a22), v1.dot(a22), v2.dot(a22)) > r)
        return false;

    return true;
}

// -----------------------------------------------------------------------------------
// PT: TODO: refactor with other implementation
static void getClosestPointOnTriangle(
    Vec3 p1, Vec3 p2, Vec3 p3, Vec3 p, Vec3& closest, Vec3& bary)
{
    Vec3 e0 = p2 - p1;
    Vec3 e1 = p3 - p1;
    Vec3 tmp = p1 - p;

    Pf32 a = e0.dot(e0);
    Pf32 b = e0.dot(e1);
    Pf32 c = e1.dot(e1);
    Pf32 d = e0.dot(tmp);
    Pf32 e = e1.dot(tmp);
    Vec3 coords, clampedCoords;
    coords.x = b * e - c * d;    // s * det
    coords.y = b * d - a * e;    // t * det
    coords.z = a * c - b * b;    // det

    clampedCoords = Vec3(0.0f, 0.0f, 0.0f);
    if (coords.x <= 0.0f) {
        if (c != 0.0f)
            clampedCoords.y = -e / c;
    }
    else if (coords.y <= 0.0f) {
        if (a != 0.0f)
            clampedCoords.x = -d / a;
    }
    else if (coords.x + coords.y > coords.z) {
        Pf32 denominator = a + c - b - b;
        Pf32 numerator = c + e - b - d;
        if (denominator != 0.0f) {
            clampedCoords.x = numerator / denominator;
            clampedCoords.y = 1.0f - clampedCoords.x;
        }
    }
    else {    // all inside
        if (coords.z != 0.0f) {
            clampedCoords.x = coords.x / coords.z;
            clampedCoords.y = coords.y / coords.z;
        }
    }
    clampedCoords.x = ntMax(clampedCoords.x, 0.0f);
    clampedCoords.y = ntMax(clampedCoords.y, 0.0f);
    clampedCoords.x = ntMin(clampedCoords.x, 1.0f);
    clampedCoords.y = ntMin(clampedCoords.y, 1.0f);

    closest = p1 + e0 * clampedCoords.x + e1 * clampedCoords.y;

    bary.x = 1.0f - clampedCoords.x - clampedCoords.y;
    bary.y = clampedCoords.x;
    bary.z = clampedCoords.y;
}

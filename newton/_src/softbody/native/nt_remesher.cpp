// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// Surface Remesher implementation.
// Ported from PhysX ExtRemesher.cpp — algorithm is identical;
// PhysX types replaced with standalone equivalents.

#include "nt_remesher.h"
#include "nt_marching_cubes_table.h"
#include "nt_box_triangle.h"
#include "nt_bvh.h"
#include "nt_types.h"

#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <queue>
#include <cassert>
#include <cstring>

namespace nt {

static const Pi32 INVALID_ID = -1;
static const Pi32 REMESH_HASH_SIZE = 170111;

static Pu32 hashCell(Pi32 xi, Pi32 yi, Pi32 zi)
{
    Pu32 h = Pu32((xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481));
    return h % REMESH_HASH_SIZE;
}

SurfaceRemesher::SurfaceRemesher() { clear(); }
SurfaceRemesher::~SurfaceRemesher() = default;

void SurfaceRemesher::clear()
{
    cells.clear();
    firstCell.assign(HASH_SIZE, INVALID_ID);
    vertices.clear();
    triIds.clear();
    normals.clear();
    triNeighbors.clear();
    cellOfVertex.clear();
    memset(&debugStats, 0, sizeof(debugStats));
}

// ---- Cell hash map ----

void SurfaceRemesher::addCell(Pi32 xi, Pi32 yi, Pi32 zi)
{
    Pu32 h = hashCell(xi, yi, zi);

    cells.push_back(Cell());
    Cell& c = cells.back();
    c.init(xi, yi, zi);
    c.next = firstCell[h];
    firstCell[h] = (Pi32)cells.size() - 1;
}

Pi32 SurfaceRemesher::getCellNr(Pi32 xi, Pi32 yi, Pi32 zi) const
{
    Pi32 nr = firstCell[hashCell(xi, yi, zi)];
    while (nr >= 0) {
        const Cell& c = cells[nr];
        if (c.xi == xi && c.yi == yi && c.zi == zi)
            return nr;
        nr = c.next;
    }
    return INVALID_ID;
}

bool SurfaceRemesher::cellExists(Pi32 xi, Pi32 yi, Pi32 zi) const
{
    return getCellNr(xi, yi, zi) >= 0;
}

// ---- Main entry ----

void SurfaceRemesher::remesh(const std::vector<Vec3>& inputVerts,
                              const std::vector<Pu32>& inputTriIds,
                              Pi32 resolution,
                              std::vector<Pu32>* vertexMap)
{
    remesh(inputVerts.data(), (Pi32)inputVerts.size(),
           inputTriIds.data(), (Pi32)inputTriIds.size(), resolution, vertexMap);
}

void SurfaceRemesher::remesh(const Vec3* inputVerts, Pi32 nbVertices,
                              const Pu32* inputTriIds, Pi32 nbTriangleIndices,
                              Pi32 resolution, std::vector<Pu32>* vertexMap)
{
    clear();

    debugStats.inputVertexCount = nbVertices;
    debugStats.inputTriangleCount = nbTriangleIndices / 3;

    // 1. Compute mesh bounds
    meshBounds.setEmpty();
    for (Pi32 i = 0; i < nbVertices; i++)
        meshBounds.include(inputVerts[i]);

    Vec3 dims = meshBounds.getDimensions();
    gridSpacing = ntMax(dims.x, ntMax(dims.y, dims.z)) / (Pf32)resolution;

    // Fatten bounds by 3 * spacing (matching PhysX)
    meshBounds.fattenFast(3.0f * gridSpacing);

    // 2. Voxelize: for each triangle, find overlapping cells
    Pi32 numTris = nbTriangleIndices / 3;
    firstCell.assign(HASH_SIZE, INVALID_ID);

    for (Pi32 i = 0; i < numTris; i++) {
        const Vec3& p0 = inputVerts[inputTriIds[3 * i]];
        const Vec3& p1 = inputVerts[inputTriIds[3 * i + 1]];
        const Vec3& p2 = inputVerts[inputTriIds[3 * i + 2]];

        Bounds3 triBounds;
        triBounds.setEmpty();
        triBounds.include(p0);
        triBounds.include(p1);
        triBounds.include(p2);

        Pi32 x0 = (Pi32)std::floor((triBounds.minimum.x - meshBounds.minimum.x) / gridSpacing);
        Pi32 y0 = (Pi32)std::floor((triBounds.minimum.y - meshBounds.minimum.y) / gridSpacing);
        Pi32 z0 = (Pi32)std::floor((triBounds.minimum.z - meshBounds.minimum.z) / gridSpacing);

        Pi32 x1 = (Pi32)std::floor((triBounds.maximum.x - meshBounds.minimum.x) / gridSpacing) + 1;
        Pi32 y1 = (Pi32)std::floor((triBounds.maximum.y - meshBounds.minimum.y) / gridSpacing) + 1;
        Pi32 z1 = (Pi32)std::floor((triBounds.maximum.z - meshBounds.minimum.z) / gridSpacing) + 1;

        for (Pi32 xi = x0; xi <= x1; xi++) {
            for (Pi32 yi = y0; yi <= y1; yi++) {
                for (Pi32 zi = z0; zi <= z1; zi++) {
                    Vec3 cellCenter;
                    cellCenter.x = meshBounds.minimum.x + (xi + 0.5f) * gridSpacing;
                    cellCenter.y = meshBounds.minimum.y + (yi + 0.5f) * gridSpacing;
                    cellCenter.z = meshBounds.minimum.z + (zi + 0.5f) * gridSpacing;

                    Vec3 halfExt(gridSpacing * 0.5f, gridSpacing * 0.5f, gridSpacing * 0.5f);

                    if (!intersectTriangleBox(cellCenter, halfExt, p0, p1, p2))
                        continue;

                    if (!cellExists(xi, yi, zi))
                        addCell(xi, yi, zi);
                }
            }
        }
    }

    debugStats.numCells = (Pi32)cells.size();

    // 3. Run marching cubes to generate isosurface
    marchingCubes();

    // 4. Remove duplicate vertices
    removeDuplicateVertices();

    // 5. Prune internal surfaces
    pruneInternalSurfaces();

    // 6. Project onto input surface
    project(inputVerts, inputTriIds, nbTriangleIndices,
            2.0f * gridSpacing, 0.1f * gridSpacing);

    if (vertexMap != nullptr)
        createVertexMap(inputVerts, nbVertices, meshBounds.minimum, gridSpacing, *vertexMap);

    // 7. Compute normals
    computeNormals();

    debugStats.outputVertexCount = (Pi32)vertices.size();
    debugStats.outputTriangleCount = (Pi32)triIds.size() / 3;
}

// ---- Marching Cubes ----

void SurfaceRemesher::marchingCubes()
{
    vertices.clear();
    cellOfVertex.clear();
    triIds.clear();

    Pi32 edgeVertId[12];
    Vec3 cornerPos[8];
    Pi32 cornerVoxelNr[8];

    for (Pi32 i = 0; i < (Pi32)cells.size(); i++)
    {
        const Cell& c = cells[i];

        // handle a 2x2x2 block of cells to cover the boundary
        for (Pi32 dx = 0; dx < 2; dx++) {
            for (Pi32 dy = 0; dy < 2; dy++) {
                for (Pi32 dz = 0; dz < 2; dz++) {
                    Pi32 xi = c.xi + dx;
                    Pi32 yi = c.yi + dy;
                    Pi32 zi = c.zi + dz;

                    // check if we are responsible for this cell
                    Pi32 maxCellNr = i;
                    for (Pi32 rx = xi - 1; rx <= xi; rx++)
                        for (Pi32 ry = yi - 1; ry <= yi; ry++)
                            for (Pi32 rz = zi - 1; rz <= zi; rz++)
                                maxCellNr = ntMax(maxCellNr, getCellNr(rx, ry, rz));

                    if (maxCellNr != i)
                        continue;

                    Pi32 code = 0;
                    for (Pi32 j = 0; j < 8; j++) {
                        Pi32 mx = xi - 1 + marchingCubeCorners[j][0];
                        Pi32 my = yi - 1 + marchingCubeCorners[j][1];
                        Pi32 mz = zi - 1 + marchingCubeCorners[j][2];
                        cornerVoxelNr[j] = getCellNr(mx, my, mz);

                        if (cornerVoxelNr[j] >= 0)
                            code |= (1 << j);

                        cornerPos[j].x = meshBounds.minimum.x + (mx + 0.5f) * gridSpacing;
                        cornerPos[j].y = meshBounds.minimum.y + (my + 0.5f) * gridSpacing;
                        cornerPos[j].z = meshBounds.minimum.z + (mz + 0.5f) * gridSpacing;
                    }

                    Pi32 first = firstMarchingCubesId[code];
                    Pi32 num = firstMarchingCubesId[code + 1] - first;

                    // create vertices and tris
                    for (Pi32 j = 0; j < 12; j++)
                        edgeVertId[j] = -1;

                    for (Pi32 j = num - 1; j >= 0; j--) {
                        Pi32 edgeId = marchingCubesIds[first + j];
                        if (edgeVertId[edgeId] < 0) {
                            Pi32 id0 = marchingCubeEdges[edgeId][0];
                            Pi32 id1 = marchingCubeEdges[edgeId][1];
                            Vec3& p0 = cornerPos[id0];
                            Vec3& p1 = cornerPos[id1];
                            edgeVertId[edgeId] = (Pi32)vertices.size();
                            vertices.push_back((p0 + p1) * 0.5f);
                            cellOfVertex.push_back(ntMax(cornerVoxelNr[id0], cornerVoxelNr[id1]));
                        }
                        triIds.push_back(edgeVertId[edgeId]);
                    }
                }
            }
        }
    }
}

// ---- Remove duplicate vertices (sort-based, matching PhysX) ----

void SurfaceRemesher::removeDuplicateVertices()
{
    if (vertices.empty()) return;

    Pf32 eps = 1e-5f;

    struct Ref {
        Pf32 d;
        Pi32 id;
        bool operator < (const Ref& r) const { return d < r.d; }
    };

    Pi32 numVerts = (Pi32)vertices.size();

    std::vector<Ref> refs(numVerts);
    for (Pi32 i = 0; i < numVerts; i++) {
        Vec3& p = vertices[i];
        refs[i].d = p.x + 0.3f * p.y + 0.1f * p.z;
        refs[i].id = i;
    }

    std::sort(refs.begin(), refs.end());

    std::vector<Pi32> idMap(vertices.size(), INVALID_ID);
    std::vector<Vec3> oldVerts = vertices;
    std::vector<Pi32> oldCellOfVertex = cellOfVertex;
    vertices.clear();
    cellOfVertex.clear();

    Pi32 nr = 0;
    while (nr < numVerts) {
        Ref& r = refs[nr];
        nr++;
        if (idMap[r.id] >= 0)
            continue;
        idMap[r.id] = (Pi32)vertices.size();
        vertices.push_back(oldVerts[r.id]);
        cellOfVertex.push_back(oldCellOfVertex[r.id]);

        Pi32 i = nr;
        while (i < numVerts && std::fabs(refs[i].d - r.d) < eps) {
            Pi32 id = refs[i].id;
            if ((oldVerts[r.id] - oldVerts[id]).magnitudeSquared() < eps * eps)
                idMap[id] = idMap[r.id];
            i++;
        }
    }

    for (Pi32 i = 0; i < (Pi32)triIds.size(); i++)
        triIds[i] = idMap[triIds[i]];
}

// ---- Find triangle neighbors (sort-based, matching PhysX) ----

void SurfaceRemesher::findTriNeighbors()
{
    Pi32 numTris = (Pi32)triIds.size() / 3;
    triNeighbors.clear();
    triNeighbors.resize(3 * numTris, INVALID_ID);

    struct Edge {
        Pi32 id0, id1, triNr, edgeNr;
        void init(Pi32 _id0, Pi32 _id1, Pi32 _triNr, Pi32 _edgeNr) {
            this->id0 = std::min(_id0, _id1);
            this->id1 = std::max(_id0, _id1);
            this->triNr = _triNr;
            this->edgeNr = _edgeNr;
        }
        bool operator < (const Edge& e) const {
            if (id0 < e.id0) return true;
            if (id0 > e.id0) return false;
            return id1 < e.id1;
        }
        bool operator == (const Edge& e) const {
            return id0 == e.id0 && id1 == e.id1;
        }
    };

    std::vector<Edge> edges(triIds.size());

    for (Pi32 i = 0; i < numTris; i++) {
        for (Pi32 j = 0; j < 3; j++) {
            Pi32 id0 = triIds[3 * i + j];
            Pi32 id1 = triIds[3 * i + (j + 1) % 3];
            edges[3 * i + j].init(id0, id1, i, j);
        }
    }

    std::sort(edges.begin(), edges.end());

    Pi32 nr = 0;
    while (nr < (Pi32)edges.size()) {
        Edge& e0 = edges[nr];
        nr++;
        while (nr < (Pi32)edges.size() && edges[nr] == e0) {
            Edge& e1 = edges[nr];
            triNeighbors[3 * e0.triNr + e0.edgeNr] = e1.triNr;
            triNeighbors[3 * e1.triNr + e1.edgeNr] = e0.triNr;
            nr++;
        }
    }
}

// ---- Prune internal surfaces (flood fill, remove negative volume islands) ----

void SurfaceRemesher::pruneInternalSurfaces()
{
    Pi32 numTris = (Pi32)triIds.size() / 3;
    if (numTris == 0) return;

    findTriNeighbors();

    std::vector<Pi32> oldTriIds = triIds;
    triIds.clear();

    std::vector<bool> visited(numTris, false);
    std::vector<Pi32> stack;

    Pi32 numIslands = 0;
    Pi32 numPruned = 0;

    for (Pi32 i = 0; i < numTris; i++) {
        if (visited[i])
            continue;
        stack.clear();
        stack.push_back(i);
        Pi32 islandStart = (Pi32)triIds.size();

        Pf32 vol = 0.0f;

        while (!stack.empty()) {
            Pi32 triNr = stack.back();
            stack.pop_back();
            if (visited[triNr])
                continue;
            visited[triNr] = true;
            for (Pi32 j = 0; j < 3; j++)
                triIds.push_back(oldTriIds[3 * triNr + j]);

            const Vec3& p0 = vertices[oldTriIds[3 * triNr]];
            const Vec3& p1 = vertices[oldTriIds[3 * triNr + 1]];
            const Vec3& p2 = vertices[oldTriIds[3 * triNr + 2]];
            vol += p0.cross(p1).dot(p2);

            for (Pi32 j = 0; j < 3; j++) {
                Pi32 n = triNeighbors[3 * triNr + j];
                if (n >= 0 && !visited[n])
                    stack.push_back(n);
            }
        }

        numIslands++;
        if (vol <= 0.0f) {
            numPruned++;
            triIds.resize(islandStart);
        }
    }

    debugStats.numIslands = numIslands;
    debugStats.numPrunedIslands = numPruned;

    // remove unreferenced vertices
    std::vector<Pi32> idMap(vertices.size(), INVALID_ID);
    std::vector<Vec3> oldVerts = vertices;
    std::vector<Pi32> oldCellOfVertex = cellOfVertex;
    vertices.clear();
    cellOfVertex.clear();

    for (Pi32 i = 0; i < (Pi32)triIds.size(); i++) {
        Pi32 id = triIds[i];
        if (idMap[id] < 0) {
            idMap[id] = (Pi32)vertices.size();
            vertices.push_back(oldVerts[id]);
            cellOfVertex.push_back(oldCellOfVertex[id]);
        }
        triIds[i] = idMap[id];
    }
}

// ---- Closest point on triangle (local copy, matches PhysX implementation) ----

static void getClosestPointOnTriangle(const Vec3& pos, const Vec3& p0, const Vec3& p1, const Vec3& p2,
                                        Vec3& closest, Vec3& bary)
{
    Vec3 e0 = p1 - p0;
    Vec3 e1 = p2 - p0;
    Vec3 tmp = p0 - pos;

    Pf32 a = e0.dot(e0);
    Pf32 b = e0.dot(e1);
    Pf32 c = e1.dot(e1);
    Pf32 d = e0.dot(tmp);
    Pf32 e = e1.dot(tmp);
    Vec3 coords, clampedCoords;
    coords.x = b * e - c * d;    // s * det
    coords.y = b * d - a * e;    // t * det
    coords.z = a * c - b * b;    // det

    clampedCoords = Vec3(0.0f);
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
    bary.y = ntMin(ntMax(clampedCoords.x, 0.0f), 1.0f);
    bary.z = ntMin(ntMax(clampedCoords.y, 0.0f), 1.0f);
    bary.x = 1.0f - bary.y - bary.z;
    closest = p0 * bary.x + p1 * bary.y + p2 * bary.z;
}

// ---- Project onto input surface ----

void SurfaceRemesher::project(const Vec3* inputVerts, const Pu32* inputTriIds,
                               Pi32 nbTriangleIndices, Pf32 searchDist, Pf32 surfaceDist)
{
    if (vertices.empty()) return;

    Pi32 numInputTris = nbTriangleIndices / 3;
    if (numInputTris == 0) return;

    // Build BVH for input triangles
    std::vector<Bounds3> triBounds(numInputTris);
    for (Pi32 i = 0; i < numInputTris; i++) {
        triBounds[i].setEmpty();
        triBounds[i].include(inputVerts[inputTriIds[3 * i]]);
        triBounds[i].include(inputVerts[inputTriIds[3 * i + 1]]);
        triBounds[i].include(inputVerts[inputTriIds[3 * i + 2]]);
    }

    BVH bvh;
    bvh.build(triBounds.data(), numInputTris);

    std::vector<Pi32> bvhTris;

    // Project vertices to closest point on surface
    for (Pi32 i = 0; i < (Pi32)vertices.size(); i++) {
        Vec3& p = vertices[i];
        Bounds3 pb;
        pb.setEmpty();
        pb.include(p);
        pb.fattenFast(searchDist);

        bvh.query(pb, bvhTris);

        Pf32 minDist2 = NT_MAX_F32;
        Vec3 closest(0.0f);

        for (Pi32 j = 0; j < (Pi32)bvhTris.size(); j++) {
            Pi32 triNr = bvhTris[j];
            const Vec3& p0 = inputVerts[inputTriIds[3 * triNr]];
            const Vec3& p1 = inputVerts[inputTriIds[3 * triNr + 1]];
            const Vec3& p2 = inputVerts[inputTriIds[3 * triNr + 2]];
            Vec3 c, bary;
            getClosestPointOnTriangle(p, p0, p1, p2, c, bary);
            Pf32 dist2 = (c - p).magnitudeSquared();
            if (dist2 < minDist2) {
                minDist2 = dist2;
                closest = c;
            }
        }

        if (minDist2 < NT_MAX_F32) {
            Vec3 n = p - closest;
            n.normalize();
            p = closest + n * surfaceDist;
        }
    }
}

// ---- Compute normals ----

void SurfaceRemesher::computeNormals()
{
    Pi32 nbVerts = (Pi32)vertices.size();

    normals.resize(nbVerts, Vec3(0.0f));

    for (Pi32 i = 0; i < (Pi32)triIds.size(); i += 3) {
        Pi32 v0 = triIds[i];
        Pi32 v1 = triIds[i + 1];
        Pi32 v2 = triIds[i + 2];

        Vec3 n = (vertices[v1] - vertices[v0]).cross(vertices[v2] - vertices[v0]);
        normals[v0] += n;
        normals[v1] += n;
        normals[v2] += n;
    }

    for (Pi32 i = 0; i < nbVerts; i++)
        normals[i].normalize();
}

static const Pi32 cellNeighbors[6][3] = { {-1, 0, 0}, {1, 0, 0}, {0, -1, 0},
                                          {0, 1, 0}, {0, 0, -1}, {0, 0, 1} };

void SurfaceRemesher::createVertexMap(const Vec3* inputVerts, Pi32 nbVertices,
                                      const Vec3& gridOrigin, Pf32& spacing,
                                      std::vector<Pu32>& vertexMap)
{
    std::vector<Pi32> vertexOfCell(cells.size(), INVALID_ID);
    std::vector<Pi32> front[2];
    Pi32 frontNr = 0;

    for (Pi32 i = 0; i < (Pi32)vertices.size(); i++) {
        Pi32 cellNr = cellOfVertex[i];
        if (cellNr >= 0) {
            if (vertexOfCell[cellNr] < 0) {
                vertexOfCell[cellNr] = i;
                front[frontNr].push_back(cellNr);
            }
        }
    }

    while (!front[frontNr].empty()) {
        front[1 - frontNr].clear();

        for (Pi32 i = 0; i < (Pi32)front[frontNr].size(); i++) {
            Pi32 cellNr = front[frontNr][i];
            Cell& c = cells[cellNr];
            for (Pi32 j = 0; j < 6; j++) {
                Pi32 n = getCellNr(
                    c.xi + cellNeighbors[j][0],
                    c.yi + cellNeighbors[j][1],
                    c.zi + cellNeighbors[j][2]
                );
                if (n >= 0 && vertexOfCell[n] < 0) {
                    vertexOfCell[n] = vertexOfCell[cellNr];
                    front[1 - frontNr].push_back(n);
                }
            }
        }
        frontNr = 1 - frontNr;
    }

    vertexMap.clear();
    vertexMap.resize(nbVertices, 0);

    for (Pi32 i = 0; i < nbVertices; i++) {
        const Vec3& p = inputVerts[i];
        Pi32 xi = Pi32(std::floor((p.x - gridOrigin.x) / spacing));
        Pi32 yi = Pi32(std::floor((p.y - gridOrigin.y) / spacing));
        Pi32 zi = Pi32(std::floor((p.z - gridOrigin.z) / spacing));

        Pi32 cellNr = getCellNr(xi, yi, zi);
        vertexMap[i] = cellNr >= 0 ? Pu32(vertexOfCell[cellNr]) : 0;
    }
}

// ---- Read back ----

void SurfaceRemesher::readBack(std::vector<Vec3>& outputVertices, std::vector<Pu32>& outputTriIds)
{
    outputVertices = vertices;
    outputTriIds.resize(triIds.size());
    for (Pi32 i = 0; i < (Pi32)triIds.size(); i++)
        outputTriIds[i] = (Pu32)triIds[i];
}

RemeshDebugStats SurfaceRemesher::getDebugStats() const
{
    return debugStats;
}

} // namespace nt

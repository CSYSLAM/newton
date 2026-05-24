// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0
//
// pybind11 binding for VoxelTetrahedralizer.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nt_voxel_tet.h"

#include <string>

namespace py = pybind11;

static py::dict voxelize_soft_body(
    py::array_t<float, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<int, py::array::c_style | py::array::forcecast> triangles,
    int resolution,
    int num_relaxation_iters = 5,
    float rel_min_tet_volume = 0.05f,
    float surface_dist_ratio = 0.2f)
{
    auto v_buf = vertices.request();
    auto t_buf = triangles.request();

    if (v_buf.ndim != 2 || v_buf.shape[1] != 3)
        throw std::runtime_error("vertices must have shape (N, 3)");
    if (t_buf.ndim != 2 || t_buf.shape[1] != 3)
        throw std::runtime_error("triangles must have shape (M, 3)");

    int numVerts = int(v_buf.shape[0]);
    int numTris = int(t_buf.shape[0]);

    // Validate triangle indices before casting to uint32
    const int* t_ptr = static_cast<const int*>(t_buf.ptr);
    for (int i = 0; i < numTris * 3; i++) {
        if (t_ptr[i] < 0 || t_ptr[i] >= numVerts)
            throw std::runtime_error("triangle index out of range: index " + std::to_string(i)
                + " value " + std::to_string(t_ptr[i]) + " numVerts " + std::to_string(numVerts));
    }

    // Convert to Vec3 + uint32 arrays
    std::vector<Vec3> verts(numVerts);
    const float* v_ptr = static_cast<const float*>(v_buf.ptr);
    for (int i = 0; i < numVerts; i++) {
        verts[i] = Vec3(v_ptr[3*i], v_ptr[3*i+1], v_ptr[3*i+2]);
    }

    std::vector<uint32_t> triIds(numTris * 3);
    for (int i = 0; i < numTris * 3; i++) {
        triIds[i] = uint32_t(t_ptr[i]);
    }

    // Run voxel tetrahedralization
    nt::VoxelTetrahedralizer vtet;
    vtet.createTetMesh(verts, triIds, resolution, num_relaxation_iters, rel_min_tet_volume, surface_dist_ratio);

    std::vector<Vec3> tetVerts;
    std::vector<uint32_t> tetIndices;
    vtet.readBack(tetVerts, tetIndices);

    // Collect debug stats
    nt::VoxelTetDebugStats stats = vtet.getDebugStats();
    py::dict debug_stats;
    debug_stats["num_surface_voxels"]   = stats.numSurfaceVoxels;
    debug_stats["num_inner_voxels"]     = stats.numInnerVoxels;
    debug_stats["num_border_voxels"]    = stats.numBorderVoxels;
    debug_stats["num_voxel_grid_x"]     = stats.numVoxelGridX;
    debug_stats["num_voxel_grid_y"]     = stats.numVoxelGridY;
    debug_stats["num_voxel_grid_z"]     = stats.numVoxelGridZ;
    debug_stats["num_unique_verts"]     = stats.numUniqueVerts;
    debug_stats["num_surface_verts"]    = stats.numSurfaceVerts;
    debug_stats["num_tets"]             = stats.numTets;
    debug_stats["num_edges"]            = stats.numEdges;
    debug_stats["grid_spacing"]         = stats.gridSpacing;

    // Convert to numpy arrays
    int numTetVerts = int(tetVerts.size());
    int numTetIndices = int(tetIndices.size());

    // Create vertex array
    float* vertData = new float[numTetVerts * 3];
    for (int i = 0; i < numTetVerts; i++) {
        vertData[3*i]   = tetVerts[i].x;
        vertData[3*i+1] = tetVerts[i].y;
        vertData[3*i+2] = tetVerts[i].z;
    }
    py::capsule vertCapsule(vertData, [](void* p) { delete[] static_cast<float*>(p); });
    auto result_verts = py::array_t<float>(
        {numTetVerts, 3},
        {3 * sizeof(float), sizeof(float)},
        vertData,
        vertCapsule
    );

    // Create index array
    int* idxData = new int[numTetIndices];
    for (int i = 0; i < numTetIndices; i++) {
        idxData[i] = int(tetIndices[i]);
    }
    py::capsule idxCapsule(idxData, [](void* p) { delete[] static_cast<int*>(p); });
    auto result_indices = py::array_t<int>(
        {numTetIndices},
        {sizeof(int)},
        idxData,
        idxCapsule
    );

    py::dict result;
    result["tet_vertices"] = result_verts;
    result["tet_indices"] = result_indices;
    result["debug_stats"] = debug_stats;
    return result;
}

PYBIND11_MODULE(_voxel_tet, m) {
    m.doc() = "Native voxel tetrahedralization module";
    m.def("voxelize_soft_body", &voxelize_soft_body,
        py::arg("vertices"),
        py::arg("triangles"),
        py::arg("resolution"),
        py::arg("num_relaxation_iters") = 5,
        py::arg("rel_min_tet_volume") = 0.05f,
        py::arg("surface_dist_ratio") = 0.2f);
}

// SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "nt_voxel_tet.h"
#include "nt_remesher.h"
#include "nt_skinning.h"
#include "nt_physx_aabb_tree.h"

#include <string>
#include <vector>

namespace py = pybind11;

// ---- Voxel Tet (original, unchanged) ----

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
    debug_stats["num_inner_tets"]       = stats.numInnerTets;
    debug_stats["num_border_tets"]      = stats.numBorderTets;
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

// ---- Surface Remesh ----

static py::tuple remesh_surface(
    py::array_t<float, py::array::c_style | py::array::forcecast> vertices,
    py::array_t<int, py::array::c_style | py::array::forcecast> tri_ids,
    int resolution,
    bool return_vertex_map = false)
{
    auto v_buf = vertices.request();
    auto t_buf = tri_ids.request();

    if (v_buf.ndim != 2 || v_buf.shape[1] != 3)
        throw std::runtime_error("vertices must have shape (N, 3)");
    if (t_buf.ndim != 2 || t_buf.shape[1] != 3)
        throw std::runtime_error("tri_ids must have shape (M, 3)");

    Pi32 nbVerts = (Pi32)v_buf.shape[0];
    Pi32 nbTriIndices = (Pi32)t_buf.shape[0] * 3;

    nt::SurfaceRemesher remesher;

    std::vector<Pu32> vertexMapData;
    std::vector<Pu32>* vertexMapPtr = return_vertex_map ? &vertexMapData : nullptr;

    remesher.remesh(static_cast<const Vec3*>(v_buf.ptr), nbVerts,
                    static_cast<const Pu32*>(t_buf.ptr), nbTriIndices, resolution,
                    vertexMapPtr);

    nt::RemeshDebugStats stats = remesher.getDebugStats();

    // Read back into capsule-owned vectors
    auto* outVerts = new std::vector<Vec3>();
    auto* outTriIds = new std::vector<Pu32>();
    remesher.readBack(*outVerts, *outTriIds);

    Pi32 numOutVerts = (Pi32)outVerts->size();
    Pi32 numOutTriIds = (Pi32)outTriIds->size();

    auto vertCapsule = py::capsule(outVerts, [](void* p) {
        delete static_cast<std::vector<Vec3>*>(p);
    });
    auto triCapsule = py::capsule(outTriIds, [](void* p) {
        delete static_cast<std::vector<Pu32>*>(p);
    });

    auto vertArr = py::array_t<float>(
        std::vector<Py_ssize_t>{numOutVerts, 3},
        std::vector<Py_ssize_t>{sizeof(float) * 3, sizeof(float)},
        reinterpret_cast<float*>(outVerts->data()),
        vertCapsule
    );

    auto triArr = py::array_t<unsigned int>(
        std::vector<Py_ssize_t>{numOutTriIds / 3, 3},
        std::vector<Py_ssize_t>{sizeof(unsigned int) * 3, sizeof(unsigned int)},
        reinterpret_cast<unsigned int*>(outTriIds->data()),
        triCapsule
    );

    py::dict statsDict;
    statsDict["input_vertex_count"]     = stats.inputVertexCount;
    statsDict["input_triangle_count"]  = stats.inputTriangleCount;
    statsDict["output_vertex_count"]    = stats.outputVertexCount;
    statsDict["output_triangle_count"]  = stats.outputTriangleCount;
    statsDict["num_cells"]              = stats.numCells;
    statsDict["num_islands"]            = stats.numIslands;
    statsDict["num_pruned_islands"]     = stats.numPrunedIslands;

    if (return_vertex_map) {
        Pi32 numVm = (Pi32)vertexMapData.size();
        auto* outVm = new std::vector<Pu32>(std::move(vertexMapData));
        auto vmCapsule = py::capsule(outVm, [](void* p) {
            delete static_cast<std::vector<Pu32>*>(p);
        });
        auto vmArr = py::array_t<unsigned int>(
            std::vector<Py_ssize_t>{numVm},
            std::vector<Py_ssize_t>{sizeof(unsigned int)},
            reinterpret_cast<unsigned int*>(outVm->data()),
            vmCapsule
        );
        return py::make_tuple(vertArr, triArr, statsDict, vmArr);
    } else {
        return py::make_tuple(vertArr, triArr, statsDict);
    }
}

// ---- Volume Skinning ----

static py::tuple compute_volume_embedding(
    py::array_t<float, py::array::c_style | py::array::forcecast> tet_verts,
    py::array_t<int, py::array::c_style | py::array::forcecast> tet_indices,
    py::array_t<float, py::array::c_style | py::array::forcecast> render_verts,
    const std::string& impl = "physx_exact_cpu")
{
    auto tv_buf = tet_verts.request();
    auto ti_buf = tet_indices.request();
    auto rv_buf = render_verts.request();

    if (tv_buf.ndim != 2 || tv_buf.shape[1] != 3)
        throw std::runtime_error("tet_verts must have shape (N, 3)");
    if (ti_buf.ndim != 2 || ti_buf.shape[1] != 4)
        throw std::runtime_error("tet_indices must have shape (M, 4)");
    if (rv_buf.ndim != 2 || rv_buf.shape[1] != 3)
        throw std::runtime_error("render_verts must have shape (K, 3)");

    Pi32 nbTetVerts = (Pi32)tv_buf.shape[0];
    Pi32 nbTetIndices = (Pi32)ti_buf.shape[0] * 4;
    Pi32 nbRenderVerts = (Pi32)rv_buf.shape[0];

    nt::VolumeEmbeddingImpl implEnum = nt::VolumeEmbeddingImpl::PhysXExactCpu;
    if (impl == "physx_exact_cpu") implEnum = nt::VolumeEmbeddingImpl::PhysXExactCpu;
    else if (impl == "legacy_custom") implEnum = nt::VolumeEmbeddingImpl::LegacyCustom;

    nt::VolumeSkinning skinner;
    skinner.compute(static_cast<const Vec3*>(tv_buf.ptr), nbTetVerts,
                    static_cast<const Pu32*>(ti_buf.ptr), nbTetIndices,
                    static_cast<const Vec3*>(rv_buf.ptr), nbRenderVerts,
                    implEnum);

    const std::vector<nt::SkinWeight>& weights = skinner.getSkinWeights();

    Pi32 nbWeights = (Pi32)weights.size();

    auto* outTetIdx = new std::vector<Pi32>(nbWeights);
    auto* outBary = new std::vector<Pf32>(nbWeights * 4);

    for (Pi32 i = 0; i < nbWeights; i++) {
        (*outTetIdx)[i] = weights[i].tetIndex;
        (*outBary)[i * 4 + 0] = weights[i].weights[0];
        (*outBary)[i * 4 + 1] = weights[i].weights[1];
        (*outBary)[i * 4 + 2] = weights[i].weights[2];
        (*outBary)[i * 4 + 3] = weights[i].weights[3];
    }

    auto tetIdxCapsule = py::capsule(outTetIdx, [](void* p) {
        delete static_cast<std::vector<Pi32>*>(p);
    });
    auto baryCapsule = py::capsule(outBary, [](void* p) {
        delete static_cast<std::vector<Pf32>*>(p);
    });

    auto tetIdxArr = py::array_t<int>(
        std::vector<Py_ssize_t>{nbWeights},
        std::vector<Py_ssize_t>{sizeof(int)},
        reinterpret_cast<int*>(outTetIdx->data()),
        tetIdxCapsule
    );

    auto baryArr = py::array_t<float>(
        std::vector<Py_ssize_t>{nbWeights, 4},
        std::vector<Py_ssize_t>{sizeof(float) * 4, sizeof(float)},
        reinterpret_cast<float*>(outBary->data()),
        baryCapsule
    );

    return py::make_tuple(tetIdxArr, baryArr);
}

static py::array_t<float> deform_with_embedding(
    py::array_t<float, py::array::c_style | py::array::forcecast> deformed_tet_verts,
    py::array_t<int, py::array::c_style | py::array::forcecast> tet_indices,
    py::array_t<int, py::array::c_style | py::array::forcecast> skin_tet_idx,
    py::array_t<float, py::array::c_style | py::array::forcecast> skin_weights)
{
    auto dtv_buf = deformed_tet_verts.request();
    auto ti_buf = tet_indices.request();
    auto sti_buf = skin_tet_idx.request();
    auto sw_buf = skin_weights.request();

    if (dtv_buf.ndim != 2 || dtv_buf.shape[1] != 3)
        throw std::runtime_error("deformed_tet_verts must have shape (N, 3)");
    if (ti_buf.ndim != 2 || ti_buf.shape[1] != 4)
        throw std::runtime_error("tet_indices must have shape (M, 4)");
    if (sti_buf.ndim != 1)
        throw std::runtime_error("skin_tet_idx must be 1D");
    if (sw_buf.ndim != 2 || sw_buf.shape[1] != 4)
        throw std::runtime_error("skin_weights must have shape (K, 4)");

    Pi32 nbRenderVerts = (Pi32)sti_buf.shape[0];
    const Vec3* deformedVerts = static_cast<const Vec3*>(dtv_buf.ptr);
    const Pu32* tetIdx = static_cast<const Pu32*>(ti_buf.ptr);
    const Pi32* sTetIdx = static_cast<const Pi32*>(sti_buf.ptr);
    const Pf32* sWeights = static_cast<const Pf32*>(sw_buf.ptr);

    auto* outVerts = new std::vector<Vec3>(nbRenderVerts);

    for (Pi32 i = 0; i < nbRenderVerts; i++) {
        Pi32 ti = sTetIdx[i];
        Pi32 i0 = (Pi32)tetIdx[4 * ti];
        Pi32 i1 = (Pi32)tetIdx[4 * ti + 1];
        Pi32 i2 = (Pi32)tetIdx[4 * ti + 2];
        Pi32 i3 = (Pi32)tetIdx[4 * ti + 3];

        Pf32 w0 = sWeights[i * 4 + 0];
        Pf32 w1 = sWeights[i * 4 + 1];
        Pf32 w2 = sWeights[i * 4 + 2];
        Pf32 w3 = sWeights[i * 4 + 3];

        (*outVerts)[i] = deformedVerts[i0] * w0 + deformedVerts[i1] * w1 +
                          deformedVerts[i2] * w2 + deformedVerts[i3] * w3;
    }

    auto capsule = py::capsule(outVerts, [](void* p) {
        delete static_cast<std::vector<Vec3>*>(p);
    });

    return py::array_t<float>(
        std::vector<Py_ssize_t>{nbRenderVerts, 3},
        std::vector<Py_ssize_t>{sizeof(float) * 3, sizeof(float)},
        reinterpret_cast<float*>(outVerts->data()),
        capsule
    );
}

// ---- Build Info ----

static py::dict get_skin_build_info() {
    py::dict info;
    // Report build-time capabilities, not a hardcoded "current impl"
    info["available_backends"] = py::make_tuple("physx_exact_cpu", "legacy_custom");
    info["default_backend"] = "physx_exact_cpu";
    info["bvh_impl"] = "physx_aabb_tree";
    info["closest_point_impl"] = "physx_closestPtPointTetrahedronWithInsideCheck";
    info["traversal_impl"] = "physx_ClosestDistanceToTetmeshTraversalController";
    info["barycentric_impl"] = "physx_computeBarycentricExact";
    info["triangle_distance_impl"] = "physx_closestPtPointTriangle2";
    info["cuda_enabled"] = false;
    info["build_source"] = "PhysX 4.1 port (GuAABBTree, GuDistancePointTetrahedron, ExtDeformableSkinning, PxMathUtils)";
    return info;
}

static py::dict debug_dump_bvh(
    py::array_t<float, py::array::c_style | py::array::forcecast> tet_verts,
    py::array_t<int, py::array::c_style | py::array::forcecast> tet_indices)
{
    auto tv_buf = tet_verts.request();
    auto ti_buf = tet_indices.request();
    Pi32 nbTetVerts = Pi32(tv_buf.shape[0]);
    Pi32 numTets = Pi32(ti_buf.shape[0]);  // shape[0] is number of rows (tets), each with 4 indices

    nt::TinyTetBvh bvh;
    nt::TinyTetBvh::constructFromTetrahedra(
        static_cast<const Pu32*>(ti_buf.ptr), numTets,
        static_cast<const Vec3*>(tv_buf.ptr), bvh);

    py::dict result;
    result["num_nodes"] = Pi32(bvh.mTree.size());

    py::list nodes;
    for (Pi32 i = 0; i < Pi32(bvh.mTree.size()); i++) {
        const nt::BvhNode& n = bvh.mTree[i];
        py::dict nd;
        nd["is_leaf"] = n.isLeaf();
        if (n.isLeaf()) {
            nd["nb_prims"] = n.getNbPrimitives();
            nd["prim_index"] = n.getPrimitiveIndex();
        } else {
            nd["pos_index"] = n.getPosIndex();
            nd["neg_index"] = n.getNegIndex();
        }
        py::list mn; mn.append(n.mBV.minimum.x); mn.append(n.mBV.minimum.y); mn.append(n.mBV.minimum.z);
        py::list mx; mx.append(n.mBV.maximum.x); mx.append(n.mBV.maximum.y); mx.append(n.mBV.maximum.z);
        nd["min"] = mn;
        nd["max"] = mx;
        nodes.append(nd);
    }
    result["nodes"] = nodes;
    return result;
}

// ---- Module ----

PYBIND11_MODULE(_voxel_tet, m) {
    m.doc() = "Newton native voxel tetrahedralizer, surface remesher, and volume skinning";

    m.def("voxelize_soft_body", &voxelize_soft_body,
          py::arg("vertices"),
          py::arg("triangles"),
          py::arg("resolution"),
          py::arg("num_relaxation_iters") = 5,
          py::arg("rel_min_tet_volume") = 0.05f,
          py::arg("surface_dist_ratio") = 0.2f);

    m.def("remesh_surface", &remesh_surface,
          py::arg("vertices"),
          py::arg("tri_ids"),
          py::arg("resolution"),
          py::arg("return_vertex_map") = false);

    m.def("compute_volume_embedding", &compute_volume_embedding,
          py::arg("tet_verts"),
          py::arg("tet_indices"),
          py::arg("render_verts"),
          py::arg("impl") = "physx_exact_cpu");

    m.def("deform_with_embedding", &deform_with_embedding,
          py::arg("deformed_tet_verts"),
          py::arg("tet_indices"),
          py::arg("skin_tet_idx"),
          py::arg("skin_weights"));

    m.def("_skin_build_info", &get_skin_build_info);
    m.def("_debug_dump_bvh", &debug_dump_bvh);
}
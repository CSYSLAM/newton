cmake_minimum_required(VERSION 3.15)
project(_voxel_tet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(pybind11 REQUIRED)

pybind11_add_module(_voxel_tet
    nt_pybind.cpp
    nt_voxel_tet.cpp
    nt_union_find.cpp
    nt_remesher.cpp
    nt_skinning.cpp
    nt_physx_aabb_tree.cpp
)

# Compiler flags
if(MSVC)
    target_compile_options(_voxel_tet PRIVATE /O2 /fp:fast /W3)
else()
    target_compile_options(_voxel_tet PRIVATE -O2 -ffast-math -Wall)
endif()

# Install target: copy into newton/_src/softbody/
install(TARGETS _voxel_tet
    LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}/../"
)

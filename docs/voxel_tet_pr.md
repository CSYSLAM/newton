# Voxel Tetrahedralizer — Pybind11 原生模块

## 概述

将 PhysX 的 VoxelTetrahedralizer 通过 pybind11 封装为 Newton 的原生 C++ 扩展。与传统 Delaunay 四面体化不同，体素方案的四面体数量仅取决于 `resolution` 参数，与表面顶点数完全解耦。

## 修改方案

### 新增文件

| 文件 | 作用 | 行数 |
|------|------|------|
| `newton/_src/softbody/__init__.py` | 包初始化 | 2 |
| `newton/_src/softbody/_voxelize_native.py` | Python 封装，输入校验，条件导入 | 125 |
| `newton/_src/softbody/_voxel_tet.cp310-win_amd64.pyd` | 编译产物（Windows/Python 3.10） | — |
| `newton/_src/softbody/native/nt_types.h` | 独立类型：Vec3, Vec3d, Bounds3 等 | 178 |
| `newton/_src/softbody/native/nt_bvh.h` | 独立 BVH：中轴分裂构建 + 栈遍历查询 | 130 |
| `newton/_src/softbody/native/nt_multi_list.h` | MultiList：多链表 + 空闲列表，边去重用 | 111 |
| `newton/_src/softbody/native/nt_union_find.h` | UnionFind 声明 | 35 |
| `newton/_src/softbody/native/nt_union_find.cpp` | UnionFind 实现 | 63 |
| `newton/_src/softbody/native/nt_voxel_tet.h` | VoxelTetrahedralizer 类声明 + DebugStats | 107 |
| `newton/_src/softbody/native/nt_voxel_tet.cpp` | 完整算法实现 | 766 |
| `newton/_src/softbody/native/nt_pybind.cpp` | pybind11 绑定 | 126 |
| `newton/_src/softbody/native/CMakeLists.txt` | CMake 构建配置 | 26 |
| `newton/tests/test_voxel_tet.py` | 18 个单元测试 | 215 |
| `scripts/build_voxel_tet.py` | 构建/安装/检查脚本 | 173 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `newton/_src/geometry/tetgen.py` | 三个函数添加 `voxel` 后端 + `surface_dist_ratio` 参数透传 |
| `newton/examples/multiphysics/example_softbody_cow_custom_tet.py` | 示例使用 `backend="voxel"` |
| `newton/tests/test_tetrahedralization_python.py` | 添加 voxel 后端测试 |

## PhysX 依赖解耦

VoxelTetrahedralizer 原本依赖 PhysX 基础类型，全部替换为独立实现：

| PhysX 类型 | 替换 |
|-----------|------|
| `PxVec3` / `PxVec3d` | `Vec3` / `Vec3d`（nt_types.h） |
| `PxBounds3` | `Bounds3` / `Bounds3d`（nt_types.h） |
| `PxArray<T>` | `std::vector<T>` |
| `PxI32/PxU32/PxF32` | `int32_t/uint32_t/float` |
| `PxMax/PxMin/PxSwap` | `std::max/std::min/std::swap` |
| `PX_ASSERT/PX_FORCE_INLINE` | `assert()/inline` |
| `PxHashMap<K,V>` | `std::unordered_map` |
| `Gu::BVHNode` + `buildAABBTree()` + `traverseBVH()` | 重写为独立 BVH（nt_bvh.h） |

## 算法管线

```
buildBVH(surface triangles)
  → voxelize(resolution)
  → createTets(subdivBorder=True, numTetsPerVoxel=5)
  → findTargetPositions(surfaceDist = surface_dist_ratio * gridSpacing)
  → relax(numIters, relMinVolume)
```

1. **voxelize**: 对每个 cell 做 BVH 查询 + SAT box-triangle intersection → 标记表面 voxel；BFS flood fill → 标记外部；剩余 → 内部
2. **createTets**: UnionFind 焊接相邻 voxel 共享顶点；内部 voxel 5 tet/voxel（parity 交替）；边界 voxel 加中心顶点 → 12 tet/voxel；边去重用 MultiList
3. **findTargetPositions**: 对每个表面顶点，BVH 查找最近三角 → `getClosestPointOnTriangle()` → target = closest + normal × surfaceDist
4. **relax**: 表面顶点拉向 target（scale=0.3）；边 Laplacian 平滑（scale=0.3，`if (w==1.0f) e*=0.5f` 匹配 PhysX）；conserveVolume 修正负体积 tet

## 编译命令

### 前提

- Visual Studio 2019（MSVC，x64）
- CMake >= 3.15
- pybind11（`pip install pybind11`）
- Python 3.10

### 方式一：使用构建脚本

```powershell
# 进入 newton 根目录
cd c:\csy_work\CG\Engine\newton

# 激活虚拟环境
& .venv\Scripts\Activate.ps1

# 仅编译
python scripts/build_voxel_tet.py

# 编译并安装到包目录
python scripts/build_voxel_tet.py --install

# 验证模块加载
python scripts/build_voxel_tet.py --check
```

### 方式二：手动 CMake

```powershell
cd newton\_src\softbody\native
mkdir build
cd build

# 配置（VS2019 + x64）
cmake .. -Dpybind11_DIR=$(python -m pybind11 --cmakedir) -G "Visual Studio 16 2019" -A x64

# 编译 Release
cmake --build . --config Release

# 安装：复制 .pyd 到 softbody 包目录
copy Release\_voxel_tet.cp310-win_amd64.pyd ..\..\
```

### 修改代码后重新编译

```powershell
# 1. 修改 C++ 源码
# 2. 重新编译
cd newton\_src\softbody\native\build
cmake --build . --config Release

# 3. 安装（如果 Python 进程占用 .pyd，先关闭）
copy Release\_voxel_tet.cp310-win_amd64.pyd ..\..\

# 4. 测试
python -m unittest newton.tests.test_voxel_tet -v
```

### 常见问题

**Q: `copy` 报 "Device or resource busy"**
A: 有 Python 进程正在使用旧 .pyd。关闭所有 Python 进程后重试：
```powershell
taskkill /f /im python.exe
```

**Q: CMake 找不到 pybind11**
A: 确认虚拟环境已激活，且 pybind11 已安装：
```powershell
pip install pybind11
python -m pybind11 --cmakedir  # 验证输出路径
```

## pybind11 数组传递

**关键发现**：pybind11 的 `request()` + 原始指针写入方式在某些环境下会静默失败（Python 端看到全零）。唯一可靠的方式是：

```cpp
// 正确：py::capsule + new[] 分配
float* data = new float[N * 3];
// ... 填充 data ...
py::capsule cap(data, [](void* p) { delete[] static_cast<float*>(p); });
auto arr = py::array_t<float>(
    {N, 3},                       // shape
    {3*sizeof(float), sizeof(float)},  // strides
    data,                         // pointer
    cap                            // parent (管理生命周期)
);
```

```cpp
// 错误：request() + 原始指针写入（可能静默失败）
auto arr = py::array_t<float>({N, 3});
auto buf = arr.request();
float* ptr = static_cast<float*>(buf.ptr);
// 写入 ptr → Python 可能看不到数据
```

## Python 接口

```python
from newton._src.softbody._voxelize_native import voxelize_soft_body_native

result = voxelize_soft_body_native(
    vertices,             # np.ndarray, shape (N,3), float32
    triangles,            # np.ndarray, shape (M,3), int32
    resolution=32,        # 体素网格密度
    num_relaxation_iters=5,
    rel_min_tet_volume=0.05,
    surface_dist_ratio=0.2,
)
# result["tet_vertices"]  → float32 (K,3)
# result["tet_indices"]   → int32 flattened (T*4,)
# result["debug_stats"]   → dict
```

通过 tetgen.py 集成：

```python
from newton._src.geometry.tetgen import tetrahedralize_surface_mesh

tv, ti = tetrahedralize_surface_mesh(
    vertices, faces,
    backend="voxel",
    resolution=16,
    surface_dist_ratio=0.2,
)
```

## 测试

```powershell
# 运行 voxel tet 测试
python -m unittest newton.tests.test_voxel_tet -v

# 运行 tetgen 集成测试
python -m unittest newton.tests.test_tetrahedralization_python -v
```

18 个测试覆盖：基础四面体化、索引有效性、NaN 检查、体积为正、debug_stats、surface_dist_ratio、summarize、输入校验、tetgen voxel 后端、球体网格、高分辨率、零松弛迭代。

## 运行示例

```powershell
cd c:\csy_work\CG\Engine\newton
& .venv\Scripts\Activate.ps1
python -m newton.examples softbody_cow_custom_tet
```
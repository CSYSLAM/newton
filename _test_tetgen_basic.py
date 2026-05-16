import sys
import os
import importlib.util
import numpy as np

PKG = "C:/csy_work/CG/Engine/newton/newton/_src/geometry/tetgen_python"

loaded = {}
def load(name, path, deps=None):
    if name in loaded:
        return loaded[name]
    if deps:
        for d in deps:
            if d not in loaded:
                raise RuntimeError(f"Dependency {d} not loaded")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    loaded[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load in dependency order
opts_mod = load("tetgen_python.options", f"{PKG}/options.py")
pred_mod = load("tetgen_python.predicates", f"{PKG}/predicates.py")
mesh_mod = load("tetgen_python.mesh", f"{PKG}/mesh.py")

# delaunay depends on mesh and predicates
# We need to patch relative imports
sys.modules["tetgen_python"] = type(sys)("tetgen_python")
sys.modules["tetgen_python"].mesh = mesh_mod
sys.modules["tetgen_python"].predicates = pred_mod
sys.modules["tetgen_python"].options = opts_mod

# For relative imports to work, we need parent packages
# Let's just monkey-patch the imports instead

# Actually, let's just directly test the key functions by reading and exec'ing
# Let me try a different approach - modify sys.path and import the package properly

# Clean up previous attempts
for k in list(sys.modules.keys()):
    if "tetgen_python" in k:
        del sys.modules[k]

# Set up the package hierarchy properly
pkg_init = f"{PKG}/__init__.py"
spec = importlib.util.spec_from_file_location("tetgen_python", pkg_init, submodule_search_locations=[PKG])
pkg_mod = importlib.util.module_from_spec(spec)
sys.modules["tetgen_python"] = pkg_mod
spec.loader.exec_module(pkg_mod)

# Now import submodules
def load_sub(name):
    path = f"{PKG}/{name}.py"
    full_name = f"tetgen_python.{name}"
    spec = importlib.util.spec_from_file_location(full_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod

options = load_sub("options")
predicates = load_sub("predicates")
mesh_m = load_sub("mesh")
delaunay = load_sub("delaunay")
cleanup = load_sub("cleanup")
diagnostics = load_sub("diagnostics")
quality = load_sub("quality")
api = load_sub("api")

print("=== Predicates ===")
a = np.array([0.0, 0.0, 0.0])
b = np.array([1.0, 0.0, 0.0])
c = np.array([0.0, 1.0, 0.0])
d = np.array([0.0, 0.0, 1.0])
assert predicates.orient3d(a, b, c, d) > 0
assert abs(predicates.tet_volume(a, b, c, d) - 1.0/6.0) < 1e-10
e = np.array([0.2, 0.2, 0.2])
assert predicates.insphere(a, b, c, d, e) > 0
e2 = np.array([10.0, 10.0, 10.0])
assert predicates.insphere(a, b, c, d, e2) < 0
print("  PASS")

print("\n=== Options ===")
opts = options.TetgenOptions.from_newton_args(quality=2.5, max_volume=0.01, verbose=True)
assert opts.minratio == 2.5 and opts.maxvolume == 0.01
opts.validate()
try:
    options.TetgenOptions(minratio=0.5).validate()
    assert False
except ValueError:
    pass
print("  PASS")

print("\n=== Mesh ===")
verts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64)
mesh = mesh_m.TetMeshData(vertices=verts)
mesh.add_tet(0, 1, 2, 3)
assert mesh.num_tets == 1
mesh.delete_tet(0)
assert len(mesh.active_tet_indices()) == 0
print("  PASS")

print("\n=== Cleanup ===")
mesh2 = mesh_m.TetMeshData(vertices=verts.copy())
mesh2.add_tet(0, 1, 2, 3)
nodes, elems = cleanup.cleanup_mesh(mesh2)
assert len(nodes) == 4 and len(elems) == 1
orient = predicates.orient3d(nodes[elems[0,0]], nodes[elems[0,1]], nodes[elems[0,2]], nodes[elems[0,3]])
assert orient > 0
print("  PASS")

print("\n=== Diagnostics ===")
errors = diagnostics.validate_tet_mesh(nodes.astype(np.float32), elems)
assert len(errors) == 0
print("  PASS")

print("\n=== Delaunay ===")
pts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64)
mesh_d = delaunay.delaunay_from_points(pts)
print(f"  4 pts -> {len(mesh_d.active_tet_indices())} tets")
assert len(mesh_d.active_tet_indices()) > 0

pts5 = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0.25,0.25,0.25]], dtype=np.float64)
mesh5 = delaunay.delaunay_from_points(pts5)
print(f"  5 pts -> {len(mesh5.active_tet_indices())} tets")
assert len(mesh5.active_tet_indices()) > 0
print("  PASS")

print("\n=== Quality ===")
ratio = quality.radius_edge_ratio(np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]], dtype=np.float64), 0, 1, 2, 3)
print(f"  ratio: {ratio:.4f}")
assert 0 < ratio < 10
print("  PASS")

print("\n=== API (cube tetrahedralization) ===")
cube_v = np.array([
    [0,0,0],[1,0,0],[1,1,0],[0,1,0],
    [0,0,1],[1,0,1],[1,1,1],[0,1,1],
], dtype=np.float32)
cube_f = np.array([
    [0,2,1],[0,3,2],
    [4,5,6],[4,6,7],
    [0,1,5],[0,5,4],
    [2,3,7],[2,7,6],
    [0,4,7],[0,7,3],
    [1,2,6],[1,6,5],
], dtype=np.int32)
tet_v, tet_i = api.tetrahedralize_surface_mesh_python(cube_v, cube_f, quality=2.5, verbose=True)
assert len(tet_v) > 0 and len(tet_i) > 0
assert len(tet_i) % 4 == 0
n_tets = len(tet_i) // 4
print(f"  {len(tet_v)} vertices, {n_tets} tetrahedra")

elems_cube = tet_i.reshape(-1, 4)
for i in range(len(elems_cube)):
    v0, v1, v2, v3 = elems_cube[i]
    vol = predicates.tet_volume(tet_v[v0], tet_v[v1], tet_v[v2], tet_v[v3])
    assert vol > 0, f"tet {i} volume={vol}"
print("  All positive volumes: PASS")

for i in range(len(elems_cube)):
    for v in elems_cube[i]:
        assert 0 <= v < len(tet_v)
print("  No out-of-range: PASS")

print("\n=== TetGenPython compat ===")
t = api.TetGenPython(cube_v, cube_f)
t.tetrahedralize(quality=True, minratio=2.5, quiet=True)
assert len(t.node) > 0 and len(t.elem) > 0
assert t.node.shape[1] == 3 and t.elem.shape[1] == 4
print("  PASS")

print("\n" + "="*50)
print("ALL TESTS PASSED!")

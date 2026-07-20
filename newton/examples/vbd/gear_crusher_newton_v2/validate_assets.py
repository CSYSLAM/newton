#!/usr/bin/env python3
"""Offline integrity and topology checks for the packaged assets."""
from pathlib import Path
import json
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent
A = ROOT / "assets"

arm = np.load(A / "armadillo_stanford_tet.npz")
v = np.asarray(arm["vertices"], dtype=np.float64)
t = np.asarray(arm["tets"], dtype=np.int32)
s = np.asarray(arm["surface_triangles"], dtype=np.int32)
p = v[t]
vol = np.einsum(
    "ij,ij->i",
    np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
    p[:, 3] - p[:, 0],
) / 6.0
assert np.all(np.isfinite(v))
assert np.all(vol > 0.0), f"Found {np.count_nonzero(vol <= 0.0)} inverted/degenerate tets"
assert 14000 <= len(v) <= 17000
assert 55000 <= len(t) <= 65000
assert np.max(t) < len(v) and np.min(t) >= 0
assert np.max(s) < len(v) and np.min(s) >= 0

gear = trimesh.load(A / "crusher_drum_18t.obj", force="mesh", process=True)
assert gear.is_watertight
assert gear.is_winding_consistent
assert len(gear.split(only_watertight=False)) == 1

report = {
    "armadillo_vertices": int(len(v)),
    "armadillo_tetrahedra": int(len(t)),
    "armadillo_boundary_triangles": int(len(s)),
    "minimum_positive_tet_volume": float(vol.min()),
    "gear_vertices": int(len(gear.vertices)),
    "gear_triangles": int(len(gear.faces)),
    "gear_watertight": bool(gear.is_watertight),
    "gear_winding_consistent": bool(gear.is_winding_consistent),
}
print(json.dumps(report, indent=2))

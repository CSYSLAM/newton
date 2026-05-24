#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Build the native voxel tetrahedralization module.

Usage:
    python scripts/build_voxel_tet.py          # build only
    python scripts/build_voxel_tet.py --install # build + install into package
    python scripts/build_voxel_tet.py --check   # verify installed module loads
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _root_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _native_dir() -> Path:
    return _root_dir() / "newton" / "_src" / "softbody" / "native"


def _install_dir() -> Path:
    return _root_dir() / "newton" / "_src" / "softbody"


def _pyd_name() -> str:
    ext = ".pyd" if platform.system() == "Windows" else ".so"
    return f"_voxel_tet{ext}"


def _find_built_pyd(build_dir: Path) -> Path | None:
    name = _pyd_name()
    # Windows: build/Release/_voxel_tet.cp310-win_amd64.pyd
    for candidate in [
        build_dir / "Release" / name,
        build_dir / name,
    ]:
        if candidate.exists():
            return candidate
    # Search for any matching .pyd/.so
    for p in build_dir.rglob(name):
        return p
    # Also search for the versioned name
    for p in build_dir.rglob("_voxel_tet.*"):
        if p.suffix in (".pyd", ".so"):
            return p
    return None


def build(args: argparse.Namespace) -> int:
    native_dir = _native_dir()
    build_dir = native_dir / "build"
    build_dir.mkdir(exist_ok=True)

    pybind11_dir = subprocess.check_output(
        [sys.executable, "-m", "pybind11", "--cmakedir"],
        text=True,
    ).strip()

    generator = args.generator
    if generator is None and platform.system() == "Windows":
        generator = "Visual Studio 16 2019"

    cmake_cmd = [
        "cmake", str(native_dir),
        f"-Dpybind11_DIR={pybind11_dir}",
    ]
    if generator:
        cmake_cmd += ["-G", generator]
    if platform.system() == "Windows" and generator and "Visual Studio" in generator:
        cmake_cmd += ["-A", "x64"]

    print(f"[build] Configuring in {build_dir}")
    print(f"[build]   {' '.join(cmake_cmd)}")
    r = subprocess.run(cmake_cmd, cwd=build_dir)
    if r.returncode != 0:
        print(f"[build] CMake configure failed (exit {r.returncode})")
        return r.returncode

    print("[build] Compiling Release...")
    r = subprocess.run(["cmake", "--build", ".", "--config", "Release"], cwd=build_dir)
    if r.returncode != 0:
        print(f"[build] Build failed (exit {r.returncode})")
        return r.returncode

    pyd = _find_built_pyd(build_dir)
    if pyd is None:
        print("[build] ERROR: compiled .pyd/.so not found in build directory")
        return 1

    print(f"[build] Compiled: {pyd}")

    if args.install:
        return install(pyd)
    else:
        print(f"[build] To install, copy to: {_install_dir()}")
        print(f"[build]   cp \"{pyd}\" \"{_install_dir() / _pyd_name()}\"")
        print("[build] Or re-run with --install")

    return 0


def install(source: Path | None = None) -> int:
    if source is None:
        build_dir = _native_dir() / "build"
        source = _find_built_pyd(build_dir)
        if source is None:
            print("[install] ERROR: no compiled module found. Run build first.")
            return 1

    dest = _install_dir() / source.name
    print(f"[install] {source} -> {dest}")

    try:
        shutil.copy2(source, dest)
    except PermissionError:
        print("[install] ERROR: target is locked. Close all Python processes and retry.")
        return 1

    print("[install] Done.")
    return 0


def check() -> int:
    print("[check] Importing native voxel tet module...")
    try:
        from newton._src.softbody._voxelize_native import voxelize_soft_body_native  # noqa: F401
    except ImportError as e:
        print(f"[check] FAILED: {e}")
        return 1

    # Quick functional test
    import numpy as np
    from newton._src.softbody._voxelize_native import voxelize_soft_body_native

    verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                       [0,0,1],[1,0,1],[1,1,1],[0,1,1]], dtype=np.float32)
    faces = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],
                       [0,1,5],[0,5,4],[2,3,7],[2,7,6],
                       [0,3,7],[0,7,4],[1,2,6],[1,6,5]], dtype=np.int32)
    result = voxelize_soft_body_native(verts, faces, resolution=8)
    n_tets = len(result["tet_indices"]) // 4
    print(f"[check] Cube res=8: {len(result['tet_vertices'])} verts, {n_tets} tets")
    print(f"[check] Debug stats: {result['debug_stats']}")
    assert n_tets > 0, "No tets generated!"
    assert not np.any(np.isnan(result["tet_vertices"])), "NaN in vertices!"
    print("[check] PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/install/check the native voxel tet module")
    parser.add_argument("--install", action="store_true", help="Install the compiled module into the package")
    parser.add_argument("--check", action="store_true", help="Verify the installed module loads and works")
    parser.add_argument("--generator", default=None, help="CMake generator (e.g. 'Visual Studio 16 2019')")
    args = parser.parse_args()

    if args.check:
        return check()
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record MJMPM source isolation, runtime versions, and optional numerical tests."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import io
import json
import platform
import subprocess
import time
import unittest
from pathlib import Path

import warp as wp

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "newton/_src/solvers/mjmpm"


def main() -> None:
    """Write a machine-readable audit without fetching dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/lab/mjmpm/validation.json")
    args = parser.parse_args()
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    imports = set()
    violations = []
    for path in PRIVATE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            modules = []
            if isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            for module in modules:
                if not module.startswith("newton"):
                    continue
                imports.add(module)
                if module.startswith("newton._src.solvers.") and not module.startswith("newton._src.solvers.mjmpm."):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: {module}")
                if module == "newton.solvers":
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}: public solver import")
    manifest = json.loads((PRIVATE / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))

    def line_ending_hashes(path: Path) -> set[str]:
        # The manifest may come from a CRLF (Windows) or LF checkout of the same sources.
        lf = path.read_bytes().replace(b"\r\n", b"\n")
        return {hashlib.sha256(lf).hexdigest(), hashlib.sha256(lf.replace(b"\n", b"\r\n")).hexdigest()}

    original_sources_unchanged = all(
        item["sha256"] in line_ending_hashes(ROOT / item["source"]) for item in manifest["files"]
    )
    report = {
        "base_revision": manifest["revision"],
        "reference_revision": subprocess.check_output(
            ["git", "rev-parse", "FAST_MJVBDV2"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("warp-lang", "mujoco", "mujoco-warp")},
        "devices": [{"alias": str(device), "name": device.name} for device in wp.get_devices()],
        "migrated_file_count": len(manifest["files"]),
        "private_python_file_count": len(list(PRIVATE.rglob("*.py"))),
        "original_sources_unchanged": original_sources_unchanged,
        "shared_solver_import_violations": violations,
        "newton_imports": sorted(imports),
    }
    if args.run_tests:
        output = io.StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromName("newton.tests.mjmpm.test_solver")
        start = time.perf_counter()
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
        report["tests"] = {
            "count": result.testsRun,
            "passed": result.wasSuccessful(),
            "skipped": len(result.skipped),
            "seconds": time.perf_counter() - start,
            "output": output.getvalue(),
        }
        print(output.getvalue())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)
    if violations or not original_sources_unchanged or report.get("tests", {}).get("passed") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

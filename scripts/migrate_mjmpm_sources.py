# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the initial private MJMPM backend snapshot from pinned local sources."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import subprocess
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "newton/_src/solvers/mjmpm"
MODULES = {
    "newton._src.solvers.implicit_mpm": "newton._src.solvers.mjmpm.mpm",
    "newton._src.solvers.mujoco": "newton._src.solvers.mjmpm.mujoco",
    "newton._src.solvers.coupled": "newton._src.solvers.mjmpm.coupling",
    "newton._src.solvers.solver": "newton._src.solvers.mjmpm.base",
    "newton._src.sim.collide": "newton._src.solvers.mjmpm.collision.pipeline",
}
STRUCT_NAMES = {
    "MaterialParameters": "MJMPMMaterialParameters",
    "Collider": "MJMPMCollider",
    "ContactWriterData": "MJMPMContactWriterData",
}


def isolate_struct_names(content: str) -> str:
    """Give Warp/FEM argument types unique keys when both backends coexist."""
    tokens = tokenize.generate_tokens(io.StringIO(content).readline)
    renamed = tokenize.untokenize(
        token._replace(string=STRUCT_NAMES.get(token.string, token.string)) if token.type == tokenize.NAME else token
        for token in tokens
    )
    return renamed.replace('"Collider"', '"MJMPMCollider"')


def remap(module: str) -> str:
    for source, destination in MODULES.items():
        if module == source or module.startswith(source + "."):
            return destination + module[len(source) :]
    return module


def migrate() -> None:
    """Copy source files once and resolve their imports to the private boundary."""
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources = list((ROOT / "newton/_src/solvers/implicit_mpm").glob("*.py"))
    sources += list((ROOT / "newton/_src/solvers/mujoco").glob("*.py"))
    sources += [
        ROOT / "newton/_src/solvers/coupled" / name
        for name in ("interface.py", "proxy_utils.py", "model_view.py", "solver_coupled.py", "solver_coupled_proxy.py")
    ]
    sources += [ROOT / "newton/_src/solvers/solver.py", ROOT / "newton/_src/sim/collide.py"]
    manifest = {"revision": revision, "files": []}
    for source in sorted(sources):
        relative = source.relative_to(ROOT)
        original_module = ".".join(relative.with_suffix("").parts)
        is_package = source.name == "__init__.py"
        if is_package:
            original_module = original_module.removesuffix(".__init__")
        destination_module = remap(original_module)
        destination = ROOT.joinpath(*destination_module.split("."))
        destination = destination / "__init__.py" if is_package else destination.with_suffix(".py")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite private backend {destination}")
        content = source.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)
        tree = ast.parse(content)
        replacements = []
        package = original_module if is_package else original_module.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                original = "." * node.level + (node.module or "")
                resolved = importlib.util.resolve_name(original, package)
                prefix = f"from {original} import"
                replacements.append((node.lineno - 1, prefix, f"from {remap(resolved)} import"))
        for index, before, after in replacements:
            lines[index] = lines[index].replace(before, after, 1)
        rewritten = isolate_struct_names("".join(lines))
        rewritten = rewritten.replace(
            "newton.solvers.SolverMuJoCo.OverflowBits.SOLVER_NCONMAX",
            "SolverMuJoCo.OverflowBits.SOLVER_NCONMAX",
        )
        # The base reset/default contact helpers must use our private detector too.
        if original_module == "newton._src.solvers.solver":
            rewritten = rewritten.replace("BodyFlags, CollisionPipeline,", "BodyFlags,")
            rewritten = rewritten.replace(
                "import warp as wp\n",
                "import warp as wp\n\nfrom newton._src.solvers.mjmpm.collision.pipeline import CollisionPipeline\n",
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rewritten, encoding="utf-8", newline="\n")
        manifest["files"].append(
            {
                "source": relative.as_posix(),
                "destination": destination.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    for directory in ("coupling", "collision"):
        (TARGET / directory / "__init__.py").write_text(
            "# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers\n"
            "# SPDX-License-Identifier: Apache-2.0\n",
            encoding="utf-8",
        )
    (TARGET / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Migrated {len(sources)} files from {revision}")


if __name__ == "__main__":
    migrate()

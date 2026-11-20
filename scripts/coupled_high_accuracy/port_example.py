# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mechanical first pass of porting an engineRTS example to ``newton/examples/coupled_high_accuracy``.

Rewrites imports from ``style3d_sim`` to the Newton solver module, the ``NewtonSolver`` alias and the engineRTS
``sys.path`` hack, and renames the file to ``example_cha_<name>.py``. Scene-specific API updates are done by hand
afterwards.

    python scripts/coupled_high_accuracy/port_example.py ~/code/agent/engineRTS/examples/example_ball.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[2] / "newton" / "examples" / "coupled_high_accuracy"
MODULE = "newton._src.solvers.coupled_high_accuracy"
PACKAGE = "newton.examples.coupled_high_accuracy"

RULES = (
    (re.compile(r"^\s*sys\.path\.insert\(0, str\(Path\(__file__\)\.resolve\(\)\.parents\[\d\]\)\)\n", re.M), ""),
    (
        re.compile(r"^(\s*)import style3d_sim as s3d$", re.M),
        r"\1from newton.solvers import coupled_high_accuracy as cha",
    ),
    (re.compile(r"^(\s*)from style3d_sim\.(\w+) import", re.M), rf"\1from {MODULE}.\2 import"),
    (re.compile(r"\bs3d\.NewtonSolver\b"), "cha.SolverCoupledHighAccuracy"),
    (re.compile(r"\bs3d\b"), "cha"),
    # Newton main removed the transitional flag; joint position targets always use the joint_q layout.
    (re.compile(r"^\s*newton\.use_coord_layout_targets = True\n", re.M), ""),
    (re.compile(r"(\w+) if newton\.use_coord_layout_targets else \w+"), r"\1"),
    (re.compile(r"python examples/example_(\w+)\.py"), r"python -m newton.examples cha_\1"),
    (re.compile(r"\bstyle3d_sim\b(?!\.)"), "coupled_high_accuracy"),
)


def import_helpers(text: str) -> str:
    """engineRTS tries ``from examples.<dir>.<helper> import ...`` and falls back to the script directory; Newton
    imports the helper from this example package."""
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        if lines[i] == "try:" and i + 1 < len(lines) and re.match(r"    from examples\.\w+\.\w+ import", lines[i + 1]):
            j, body = i + 1, []
            while not lines[j].startswith("except ModuleNotFoundError"):
                body.append(lines[j])
                j += 1
            j += 1
            while j < len(lines) and lines[j].startswith("    "):
                j += 1
            out.append(re.sub(r"^    from examples\.\w+\.(\w+) import", rf"from {PACKAGE}.\1 import", body[0]))
            out.extend(line.removeprefix("    ") for line in body[1:])
            i = j
            continue
        if lines[i] == "if __package__:" and i + 1 < len(lines) and re.match(r"    from \.[\w.]+ import", lines[i + 1]):
            j, body = i + 1, []
            while lines[j] != "else:":
                body.append(lines[j])
                j += 1
            j += 1
            while j < len(lines) and lines[j].startswith("    "):
                j += 1
            helper = re.match(r"    from \.([\w.]+) import", body[0]).group(1)
            if helper.startswith("example_"):
                helper = "example_cha_" + helper.removeprefix("example_")
            out.append(re.sub(r"^    from \.[\w.]+ import", f"from {PACKAGE}.{helper} import", body[0]))
            out.extend(line.removeprefix("    ") for line in body[1:])
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def port(source: Path) -> Path:
    text = import_helpers(source.read_text())
    for pattern, replacement in RULES:
        text = pattern.sub(replacement, text)
    # Helper modules keep their names; examples get a ``cha_`` prefix so they never shadow Newton's own.
    if source.name.startswith("example_"):
        target = TARGET / f"example_cha_{source.name.removeprefix('example_')}"
    else:
        target = TARGET / source.name
    target.write_text(text)
    return target


if __name__ == "__main__":
    TARGET.mkdir(parents=True, exist_ok=True)
    for argument in sys.argv[1:]:
        print(port(Path(argument).expanduser()))

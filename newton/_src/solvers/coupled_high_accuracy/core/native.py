# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Load the original native solver core for side-by-side comparison with the Warp port.

The native extension is a development oracle only: it comes from an engineRTS ``style3d_sim`` build and is never
required by Newton. Set ``NEWTON_CHA_NATIVE_CORE`` to the ``_core*.so`` path, or rely on the default engineRTS
location below.
"""

from __future__ import annotations

import glob
import importlib.machinery
import importlib.util
import os
import sys
from types import ModuleType

_DEFAULT_GLOB = os.path.expanduser("~/code/agent/engineRTS/.venv/lib/python3*/site-packages/style3d_sim/_core*.so")
# The extension's init symbol is PyInit__core, so the module name must end in "_core".
_MODULE_NAME = "_newton_cha_native._core"


def native_core_path() -> str:
    path = os.environ.get("NEWTON_CHA_NATIVE_CORE")
    if path:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"NEWTON_CHA_NATIVE_CORE does not name a file: {path}")
        return path
    matches = sorted(glob.glob(_DEFAULT_GLOB))
    if not matches:
        raise ModuleNotFoundError(
            "The native comparison backend needs a built style3d_sim _core extension. "
            "Set NEWTON_CHA_NATIVE_CORE to its path, or use NEWTON_CHA_BACKEND=warp."
        )
    return matches[-1]


def load() -> ModuleType:
    module = sys.modules.get(_MODULE_NAME)
    if module is not None:
        return module
    path = native_core_path()
    loader = importlib.machinery.ExtensionFileLoader(_MODULE_NAME, path)
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules[_MODULE_NAME] = module
    return module

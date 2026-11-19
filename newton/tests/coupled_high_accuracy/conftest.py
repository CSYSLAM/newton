# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""pytest configuration for the high-accuracy coupled solver tests ported from engineRTS.

This directory is not a package, so Newton's unittest discovery skips it; run it with
``uv run --with pytest pytest newton/tests/coupled_high_accuracy``. The solver core backend is chosen with
``NEWTON_CHA_BACKEND`` (``native`` or ``warp``).
"""

import sys
from pathlib import Path

# Some tests share scene helpers by importing sibling test modules.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dynamic MuJoCo source + particle/soft VBD, single-driven no feedback (``DESIGN.md`` 4.1)."""

from __future__ import annotations

from ._one_way import OneWayBackend

__all__ = ["OneWayDynamicSoftBackend"]


class OneWayDynamicSoftBackend(OneWayBackend):
    _source_is_dynamic = True
    _vbd_core = "soft"

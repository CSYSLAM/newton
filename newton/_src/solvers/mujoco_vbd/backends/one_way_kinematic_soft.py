# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kinematic source + particle/soft VBD, immovable proxy (``DESIGN.md`` 4.1)."""

from __future__ import annotations

from ._one_way import OneWayBackend

__all__ = ["OneWayKinematicSoftBackend"]


class OneWayKinematicSoftBackend(OneWayBackend):
    _source_is_dynamic = False
    _vbd_core = "soft"

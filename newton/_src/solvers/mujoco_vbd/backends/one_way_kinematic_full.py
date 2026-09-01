# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Kinematic source + full contact/VBD rigid, immovable proxy (``DESIGN.md`` 4.1)."""

from __future__ import annotations

from ._one_way import OneWayBackend

__all__ = ["OneWayKinematicFullBackend"]


class OneWayKinematicFullBackend(OneWayBackend):
    _source_is_dynamic = False
    _vbd_core = "full"

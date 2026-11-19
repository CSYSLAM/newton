# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Host-side sparse matrix templates of the solver core.

The native host builds the constant (projective-dynamics) system matrix in a ``std::vector<std::map<int, float>>``
and packs it into a slot-major ELL template. :class:`SparseRows` reproduces that accumulation exactly: float32
additions per entry in contribution order, starting from the entry's previous value (0 for a new entry), and rows
traversed by ascending column.
"""

from __future__ import annotations

import numpy as np


class SparseRows:
    """Off-diagonal entries of ``rows`` rows, keyed by (row, column)."""

    def __init__(self, rows: int) -> None:
        self.rows = rows
        self.keys = np.zeros(0, dtype=np.int64)
        self.values = np.zeros(0, dtype=np.float32)

    def add(self, row: np.ndarray, column: np.ndarray, value: np.ndarray) -> None:
        """Accumulate contributions given in the native loop order."""
        if not len(row):
            return
        new_keys = np.asarray(row, dtype=np.int64) * self.rows + np.asarray(column, dtype=np.int64)
        unique, inverse = np.unique(np.concatenate([self.keys, new_keys]), return_inverse=True)
        accumulated = np.zeros(len(unique), dtype=np.float32)
        existing = len(self.keys)
        accumulated[inverse[:existing]] = self.values
        # np.add.at is unbuffered: each float32 addition happens in array order, like the native += sequence.
        np.add.at(accumulated, inverse[existing:], np.asarray(value, dtype=np.float32))
        self.keys, self.values = unique, accumulated

    def negate(self) -> None:
        self.values = -self.values

    def row_columns(self) -> tuple[np.ndarray, np.ndarray]:
        return self.keys // self.rows, self.keys % self.rows

    def counts(self) -> np.ndarray:
        row, _ = self.row_columns()
        return np.bincount(row, minlength=self.rows).astype(np.int32)

    def ell(self, values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]:
        """Slot-major ELL (columns padded with -1, values with 0) and the maximum row length.

        Keys are sorted by row then column, so each row's slots follow ascending columns as in a std::map.
        """
        row, column = self.row_columns()
        counts = self.counts()
        max_entries = int(counts.max(initial=0))
        starts = np.zeros(self.rows + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        slot = np.arange(len(row)) - starts[row]
        columns = np.full(max_entries * self.rows, -1, dtype=np.int32)
        packed = np.zeros(max_entries * self.rows, dtype=np.float32)
        target = slot * self.rows + row
        columns[target] = column
        packed[target] = self.values if values is None else values
        return columns, packed, max_entries

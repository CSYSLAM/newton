# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels of the solver core, one module per native ``kernels/*.cu`` file.

Conventions shared by all modules:

* An optional native pointer argument is an array that may be empty; ``a.shape[0] == 0`` stands for ``nullptr``.
* Native ``glm::mat3`` values are column-major. Every matrix here is a ``wp.mat33`` in ordinary row/column math, so
  a native ``m[c][r]`` reads as ``m[r, c]``; spans that hold native matrices are transposed when they are read.
* ELL matrices keep the native slot-major layout (``entries[slot * rows + row]``) with separate column and value
  arrays instead of the native ``{int column; T value}`` structs.
"""

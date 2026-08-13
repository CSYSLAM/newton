# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid + tetrahedral soft + cloth coupling through MJVBDV2's pure-VBD fallback.

Use ``--solver vbd`` to run the identical model with native ``SolverVBD``.

Run::

    python -m newton.examples mjvbd_v2_rigid_soft_cloth
    python -m newton.examples mjvbd_v2_rigid_soft_cloth --solver vbd
"""

from newton.examples.mjvbdv2.mjvbd_v2_demo import VBDMixExample, run_vbd_mix_example


class Example(VBDMixExample):
    pass


if __name__ == "__main__":
    run_vbd_mix_example()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MJVBDV2 dynamic/kinematic robot coupled one-way to VBD cloth.

Run::

    python -m newton.examples mjvbd_v2_robot_cloth --joint-mode dynamic
    python -m newton.examples mjvbd_v2_robot_cloth --joint-mode kinematic
"""

from newton.examples.mjvbdv2.mjvbd_v2_demo import RobotCouplingExample, run_robot_example


class Example(RobotCouplingExample):
    OBJECTS = frozenset({"cloth"})


if __name__ == "__main__":
    run_robot_example(Example)

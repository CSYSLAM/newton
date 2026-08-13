# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MJVBDV2 robot coupled to mutually coupled VBD rigid and soft objects.

Run::

    python -m newton.examples mjvbd_v2_robot_rigid_soft --joint-mode dynamic
    python -m newton.examples mjvbd_v2_robot_rigid_soft --joint-mode kinematic
"""

from newton.examples.mjvbdv2.mjvbd_v2_demo import RobotCouplingExample, run_robot_example


class Example(RobotCouplingExample):
    OBJECTS = frozenset({"rigid", "soft"})


if __name__ == "__main__":
    run_robot_example(Example)

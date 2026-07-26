# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""MJVBDV2 dynamic/kinematic robot coupled one-way to a VBD rigid body.

Run::

    python -m newton.examples mjvbd_v2_robot_rigid --joint-mode dynamic
    python -m newton.examples mjvbd_v2_robot_rigid --joint-mode kinematic
"""

from newton.examples.multiphysics.mjvbd_v2_demo import RobotCouplingExample, run_robot_example


class Example(RobotCouplingExample):
    OBJECTS = frozenset({"rigid"})


if __name__ == "__main__":
    run_robot_example(Example)

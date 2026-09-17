# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate streaming visual-state recordings independently of the GPU solver."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from newton.examples.mjvbdv2.support.conveyor_recording import SCENE_OPTIONS, RecordingReader, RecordingWriter


class _Array:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def numpy(self):
        return self.values.copy()

    def assign(self, values):
        self.values = values.copy()


def _scene():
    return SimpleNamespace(
        model=SimpleNamespace(
            body_label=["robot", "parcel"],
            **{name: _Array([0]) for name in ("tri_indices", "shape_body", "shape_transform", "shape_scale")},
        ),
        state_0=SimpleNamespace(body_q=_Array(np.zeros((2, 7))), particle_q=_Array(np.zeros((5, 3)))),
        args=SimpleNamespace(**dict.fromkeys(SCENE_OPTIONS, 1)),
        parcels=[SimpleNamespace(picked=False, released=False, sorted=False) for _ in range(4)],
        sim_time=0.0,
        belt_offset=0.0,
        active=0,
        phase="feed",
    )


class TestConveyorRecording(unittest.TestCase):
    def test_round_trip_and_partial_prefix(self):
        """Restore geometry and visual status and ignore unwritten capacity after interruption."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recording"
            source = _scene()
            writer = RecordingWriter(path, source, 10)
            writer.append(source)
            source.state_0.particle_q.assign(np.ones((5, 3), dtype=np.float32))
            source.state_0.body_q.assign(np.ones((2, 7), dtype=np.float32))
            source.sim_time, source.belt_offset, source.active, source.phase = 1 / 60, 0.2, 2, "release"
            source.parcels[2].picked = True
            source.parcels[2].released = True
            writer.append(source)
            writer.close(complete=False)
            reader = RecordingReader(path)
            self.assertEqual(reader.count, 2)
            self.assertFalse(reader.metadata["complete"])
            target = _scene()
            reader.validate_scene(target)
            reader.restore(target, 1)
            np.testing.assert_array_equal(target.state_0.particle_q.numpy(), source.state_0.particle_q.numpy())
            np.testing.assert_array_equal(target.state_0.body_q.numpy(), source.state_0.body_q.numpy())
            self.assertEqual(
                (target.phase, target.active, target.belt_offset, target.sim_time), ("release", 2, 0.2, 1 / 60)
            )
            self.assertIs(type(target.belt_offset), float)
            self.assertTrue(target.parcels[2].released)
            reader.restore(target, 0)
            self.assertFalse(target.parcels[2].released)
            with self.assertRaises(IndexError):
                reader.restore(target, 2)
            target.model.body_label.reverse()
            with self.assertRaises(ValueError):
                reader.validate_scene(target)
            with self.assertRaises(FileExistsError):
                RecordingWriter(path, source, 10)


if __name__ == "__main__":
    unittest.main()

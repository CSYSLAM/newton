# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check exact W1 bag replay, partial recordings, and incompatible caches."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from newton.examples.mjvbdv2._webxr_teleop import JsonlTrajectoryRecorder
from newton.examples.mjvbdv2.support.w1_bag_recording import (
    Playback,
    RecordingReader,
    RecordingWriter,
    TeleopRecordingReader,
    run_recording,
    run_replay,
    scene_signature,
)


class _Array:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def numpy(self):
        return self.values.copy()

    def assign(self, values):
        self.values = values.copy()

    @property
    def shape(self):
        return self.values.shape


def _scene():
    return SimpleNamespace(
        model=SimpleNamespace(
            body_label=["robot", "snack"],
            shape_source=[],
            **{
                name: _Array([0])
                for name in ("tri_indices", "shape_body", "shape_transform", "shape_scale", "shape_type")
            },
        ),
        state_0=SimpleNamespace(body_q=_Array(np.zeros((2, 7))), particle_q=_Array(np.zeros((5, 3)))),
        args=SimpleNamespace(snacks=1, robot_setback=0.03, substeps=10, iterations=18),
        rest=np.zeros((5, 3), dtype=np.float32),
        frame=0,
        sim_time=0.0,
    )


class TestW1BagRecording(unittest.TestCase):
    def test_teleop_jsonl_replays_full_states_without_physics(self):
        """Restore robot, rigid bodies, cloth deformation and velocities from JSONL."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teleop.jsonl"
            target = _scene()
            fields = {"joint_q": (6,), "joint_qd": (6,), "body_qd": (2, 6), "particle_qd": (5, 3)}
            for name, shape in fields.items():
                setattr(target.state_0, name, _Array(np.zeros(shape)))
            recorder = JsonlTrajectoryRecorder(
                path,
                {
                    "scene": "w1-bag-packing",
                    "recordingKind": "full-state",
                    "frameDtSeconds": 1 / 60,
                    "sceneSignature": scene_signature(target),
                    "sceneOptions": vars(target.args),
                },
            )
            recorder.start()
            samples = []
            for frame in range(2):
                sample = {
                    "frame": frame,
                    "simulationTimeSeconds": frame / 60,
                    "bodyPoses": np.full((2, 7), frame + 1).tolist(),
                    "bagParticleQ": np.full((5, 3), frame + 2).tolist(),
                    "jointQ": np.full(6, frame + 3).tolist(),
                    "jointQd": np.full(6, frame + 4).tolist(),
                    "bodyVelocities": np.full((2, 6), frame + 5).tolist(),
                    "bagParticleQd": np.full((5, 3), frame + 6).tolist(),
                }
                samples.append(sample)
                recorder.append(sample)
                recorder.append_event({"event": "scene-reset"})
            recorder.close()
            shown = []
            target.render = lambda: shown.append(
                {name: getattr(target.state_0, name).numpy() for name in ("body_q", "particle_q", *fields)}
            )
            target.step = Mock(side_effect=AssertionError("Replay stepped physics"))
            factory, viewer = Mock(), Mock()
            factory.create_render_scene.return_value = target
            args = SimpleNamespace(
                replay=path, start_frame=0, loop=False, num_frames=10, unthrottled=True, viewer="null", headless=True
            )
            with patch("builtins.print"):
                run_replay(factory, viewer, args)
            self.assertEqual(len(shown), 2)
            for index, state in enumerate(shown):
                for name, key in {
                    "body_q": "bodyPoses",
                    "particle_q": "bagParticleQ",
                    "joint_q": "jointQ",
                    "joint_qd": "jointQd",
                    "body_qd": "bodyVelocities",
                    "particle_qd": "bagParticleQd",
                }.items():
                    np.testing.assert_array_equal(state[name], samples[index][key])
            target.step.assert_not_called()
            viewer.close.assert_called_once()
            with path.open("ab") as stream:
                stream.write(b'{"type":"frame","bodyPoses":[')
            reader = TeleopRecordingReader(path)
            self.assertFalse(reader.metadata["complete"])
            self.assertEqual(reader.count, 2)
            reader.restore(target, 1)
            with self.assertRaises(IndexError):
                reader.restore(target, 2)
            target.rest[0, 0] = 0.01
            with self.assertRaisesRegex(ValueError, "geometry"):
                reader.validate_scene(target)
            target.rest[0, 0] = 0
            target.state_0.body_q = _Array(np.zeros((1, 7)))
            with self.assertRaisesRegex(ValueError, "shape"):
                reader.restore(target, 0)

    def test_exact_round_trip_and_interrupted_prefix(self):
        """Restore every visual field exactly and exclude unwritten frames after interruption."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip"
            scene = _scene()
            writer = RecordingWriter(path, scene, 10)
            writer.append(scene)
            scene.state_0.body_q.assign(np.arange(14, dtype=np.float32).reshape(2, 7))
            scene.state_0.particle_q.assign(np.arange(15, dtype=np.float32).reshape(5, 3))
            scene.sim_time = 1 / 60
            writer.append(scene)
            writer.close(complete=False)
            reader, target = RecordingReader(path), _scene()
            reader.validate_scene(target)
            self.assertEqual(reader.count, 2)
            self.assertFalse(reader.metadata["complete"])
            self.assertEqual(reader.metadata["scene_options"]["robot_setback"], 0.03)
            reader.restore(target, 1)
            for name in ("body_q", "particle_q"):
                np.testing.assert_array_equal(
                    getattr(target.state_0, name).numpy(), getattr(scene.state_0, name).numpy()
                )
            self.assertEqual((target.frame, target.sim_time), (1, 1 / 60))
            reader.restore(target, 0)
            np.testing.assert_array_equal(target.state_0.body_q.numpy(), np.zeros((2, 7)))
            with self.assertRaises(IndexError):
                reader.restore(target, 2)
            with self.assertRaises(IndexError):
                reader.restore(target, -1)
            with self.assertRaises(FileExistsError):
                RecordingWriter(path, scene, 10)

    def test_reject_incompatible_geometry_and_layout(self):
        """Reject changed meshes and array layouts before replaying the wrong scene."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip"
            scene = _scene()
            writer = RecordingWriter(path, scene, 1)
            writer.append(scene)
            writer.close(complete=True)
            reader = RecordingReader(path)
            scene.model.body_label.reverse()
            with self.assertRaisesRegex(ValueError, "geometry"):
                reader.validate_scene(scene)
            scene = _scene()
            scene.rest[0, 0] = 0.01
            with self.assertRaisesRegex(ValueError, "geometry"):
                reader.validate_scene(scene)
            scene = _scene()
            scene.state_0.particle_q = _Array(np.zeros((6, 3)))
            with self.assertRaisesRegex(ValueError, "shape or dtype"):
                reader.validate_scene(scene)

    def test_reject_truncated_and_nonfinite_recording(self):
        """Reject missing frames and nonfinite timestamps instead of displaying corrupt data."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip"
            scene = _scene()
            writer = RecordingWriter(path, scene, 1)
            writer.append(scene)
            writer.close(complete=False)
            writer.arrays["sim_time"][0] = np.nan
            writer.close(complete=False)
            with self.assertRaisesRegex(ValueError, "Nonfinite sim_time"):
                RecordingReader(path).restore(_scene(), 0)
            metadata = json.loads((path / "metadata.json").read_text())
            metadata["frame_count"] = 2
            (path / "metadata.json").write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "truncated"):
                RecordingReader(path)

    def test_loop_and_last_frame(self):
        """Visit the final frame before stopping or wrapping, including single-frame clips."""
        player = Playback(SimpleNamespace(count=3), start_frame=1)
        self.assertTrue(player.advance())
        self.assertEqual(player.frame, 2)
        self.assertFalse(player.advance())
        self.assertEqual(player.frame, 2)
        player.loop = True
        self.assertTrue(player.advance())
        self.assertEqual(player.frame, 0)
        single = Playback(SimpleNamespace(count=1), loop=True)
        self.assertTrue(single.advance())
        self.assertEqual(single.frame, 0)
        with self.assertRaises(ValueError):
            Playback(SimpleNamespace(count=3), start_frame=3)

    def test_replay_uses_recorded_options_and_includes_endpoints(self):
        """Restore recorded scene options and show both endpoints without stepping physics."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip"
            source = _scene()
            writer = RecordingWriter(path, source, 2)
            writer.append(source)
            source.sim_time = 1 / 60
            writer.append(source)
            writer.close(complete=False)
            target, shown = _scene(), []
            target.render = lambda: shown.append(target.sim_time)
            target.step = Mock(side_effect=AssertionError("Replay stepped physics"))
            factory = Mock()
            factory.create_render_scene.return_value = target
            viewer = Mock()
            args = SimpleNamespace(
                replay=str(path),
                start_frame=0,
                loop=False,
                num_frames=100,
                unthrottled=True,
                viewer="null",
                headless=True,
                snacks=2,
                robot_setback=0.5,
            )
            with patch("builtins.print"):
                run_replay(factory, viewer, args)
            self.assertEqual(shown, [0.0, 1 / 60])
            self.assertEqual((args.snacks, args.robot_setback), (1, 0.03))
            target.step.assert_not_called()
            factory.assert_not_called()
            viewer.close.assert_called_once()

    def test_interrupt_keeps_last_finished_frame(self):
        """Flush completed frames and close the viewer when a recording is interrupted."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip"
            scene = _scene()

            def step():
                if scene.frame == 1:
                    raise KeyboardInterrupt
                scene.frame += 1
                scene.sim_time += 1 / 60

            scene.step = step
            args = SimpleNamespace(record=str(path), num_frames=3, viewer="null", test=False, snacks=1)
            viewer = Mock()
            with patch("builtins.print"), self.assertRaises(KeyboardInterrupt):
                run_recording(lambda viewer, args: scene, viewer, args)
            reader = RecordingReader(path)
            self.assertEqual(reader.count, 2)
            self.assertFalse(reader.metadata["complete"])
            viewer.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

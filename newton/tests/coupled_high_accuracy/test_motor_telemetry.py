"""Command indexing, post-step actuator snapshots, and frame/CSV alignment regressions."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from newton.examples.coupled_high_accuracy.demos.piper_bag_transport.motor_telemetry import (
    MotorTelemetryCSV,
    capture_motor_sample,
)


@pytest.fixture(autouse=True, scope="module")
def local_warp_cache(tmp_path_factory):
    wp.config.kernel_cache_dir = str(Path(tmp_path_factory.mktemp("warp-motor-telemetry-tests")))


@pytest.fixture
def joints():
    return [
        {"name": "joint1", "type": "revolute", "position_unit": "rad", "effort_unit": "Nm"},
        {"name": "joint7", "type": "prismatic", "position_unit": "m", "effort_unit": "N"},
    ]


def test_substeps_snapshot_actual_commands_with_distinct_coordinate_and_dof_indices(tmp_path, joints):
    with wp.ScopedDevice("cpu"):
        targets = wp.array([5, 2], dtype=wp.int32)
        dofs = wp.array([3, 0], dtype=wp.int32)
        positions = wp.array([100, 101, 0.012, 103, 104, 0.3], dtype=wp.float32)
        velocities = wp.array([0.002, 200, 201, 0.2], dtype=wp.float32)
        feedforward = wp.array([-4, 300, 301, 9], dtype=wp.float32)
        effort = wp.zeros(4, dtype=wp.float32)
        clock = wp.zeros(1, dtype=wp.float32)
        samples = wp.empty((4, 2, 5), dtype=wp.float32)
        for substep in range(4):
            clock.assign([(substep + 1) / 240])
            effort.assign([-substep - 1, 0, 0, 2 * substep + 1])
            wp.launch(
                capture_motor_sample,
                dim=2,
                inputs=[substep, 1 / 240, clock, targets, dofs, positions, velocities, feedforward, effort],
                outputs=[samples],
            )
        effort.fill_(99)
        snapshot = samples.numpy()
    np.testing.assert_allclose(snapshot[:, :, 0], np.tile([0.3, 0.012], (4, 1)))
    np.testing.assert_allclose(snapshot[:, :, 1], np.tile([0.2, 0.002], (4, 1)))
    np.testing.assert_array_equal(snapshot[:, :, 2], np.tile([9, -4], (4, 1)))
    np.testing.assert_array_equal(snapshot[:, :, 3], [[1, -1], [3, -2], [5, -3], [7, -4]])
    np.testing.assert_allclose(snapshot[:, 0, 4], np.arange(4) / 240, atol=1e-9)

    writer = MotorTelemetryCSV(tmp_path / "motors.csv", joints, 60, 4)
    writer.append(0, snapshot[:1], initial=True)
    writer.append(1, snapshot)
    writer.finish()
    with writer.path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 4
    assert rows[0]["feedback_valid"] == "0" and rows[0]["feedback_effort"] == ""
    assert rows[1]["feedback_effort_mean"] == ""
    assert float(rows[2]["time_s"]) == 1 / 60
    assert float(rows[2]["target_time_s"]) == pytest.approx(3 / 240)
    assert float(rows[2]["feedback_effort"]) == 7
    assert float(rows[2]["feedback_effort_mean"]) == 4
    assert float(rows[2]["feedback_effort_min"]) == 1
    assert float(rows[3]["feedback_effort_max"]) == -1
    assert rows[2]["effort_unit"] == "Nm" and rows[3]["effort_unit"] == "N"
    metadata = json.loads(writer.path.with_suffix(".json").read_text())
    assert metadata["complete"] and metadata["frame_count"] == 2
    assert metadata["feedback_sampling"] == "post_step"
    assert metadata["feedback_model"] == "limited_pd_plus_feedforward"


def test_recording_rejects_skipped_frames_and_nonfinite_or_incomplete_substeps(tmp_path, joints):
    writer = MotorTelemetryCSV(tmp_path / "motors.csv", joints, 60, 4)
    initial = np.zeros((1, 2, 5))
    writer.append(0, initial, initial=True)
    with pytest.raises(ValueError, match="contiguous"):
        writer.append(2, np.zeros((4, 2, 5)))
    with pytest.raises(ValueError, match="every motor and substep"):
        writer.append(1, np.zeros((3, 2, 5)))
    broken = np.zeros((4, 2, 5))
    broken[2, 1, 3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        writer.append(1, broken)
    writer.finish()
    with pytest.raises(ValueError, match="finalized"):
        writer.append(1, np.zeros((4, 2, 5)))

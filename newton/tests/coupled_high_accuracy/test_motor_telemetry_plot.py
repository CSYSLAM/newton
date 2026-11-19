"""CPU checks for telemetry integrity and cumulative, expanding plot histories."""

import csv
import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from newton.examples.coupled_high_accuracy.demos.piper_bag_transport.plot_motor_telemetry import (
    JOINT_NAMES,
    TelemetryPlot,
    load_telemetry,
    render_video,
)


@pytest.fixture
def telemetry_rows():
    rows = []
    for frame in range(4):
        for index, name in enumerate(JOINT_NAMES):
            rotary = index < 6
            effort = (index + 1) * frame * 0.01
            rows.append(
                {
                    "frame": frame,
                    "time_s": frame / 60,
                    "target_time_s": max(0, frame / 60 - 1 / 600),
                    "joint_name": name,
                    "joint_type": "revolute" if rotary else "prismatic",
                    "position_unit": "rad" if rotary else "m",
                    "effort_unit": "N*m" if rotary else "N",
                    "target_position": index * 0.01 + frame * 0.001,
                    "target_velocity": 0.0,
                    "feedforward_effort": 0.0,
                    "feedback_effort": effort if frame else "",
                    "feedback_effort_mean": effort if frame else "",
                    "feedback_effort_min": effort if frame else "",
                    "feedback_effort_max": effort if frame else "",
                    "feedback_valid": int(frame > 0),
                }
            )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_csv_accepts_unordered_complete_frames_and_keeps_unsolved_feedback_missing(tmp_path, telemetry_rows):
    data = load_telemetry(write_csv(tmp_path / "telemetry.csv", telemetry_rows[::-1]))
    np.testing.assert_allclose(data.time_s, np.arange(4) / 60)
    assert data.target_position.shape == data.feedback_effort.shape == (4, 8)
    assert np.isnan(data.feedback_effort[0]).all()
    assert np.isfinite(data.feedback_effort[1:]).all()


@pytest.mark.parametrize("post_step", (False, True))
def test_feedback_labels_preserve_the_recorded_sampling_definition(tmp_path, telemetry_rows, post_step):
    csv_path = write_csv(tmp_path / "telemetry.csv", telemetry_rows)
    if post_step:
        csv_path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "csv": csv_path.name,
                    "feedback_sampling": "post_step",
                    "feedback_model": "limited_pd_plus_feedforward",
                }
            ),
            encoding="utf-8",
        )
    data = load_telemetry(csv_path)
    chart = TelemetryPlot(data, "feedback")
    try:
        labels = [text.get_text() for text in chart.figure.texts]
        assert any("post-step actuator effort" in text for text in labels) == post_step
        assert data.feedback_sampling == ("post_step" if post_step else "unspecified")
    finally:
        chart.close()


@pytest.mark.parametrize(
    "damage, message",
    (
        ("missing_frame", "contiguous frames"),
        ("missing_joint", "all eight joints"),
        ("duplicate", "Duplicate telemetry sample"),
        ("wrong_units", "inconsistent joint type or units"),
        ("nonfinite_target", "target_position must be finite"),
        ("nonfinite_feedback", "feedback_effort must be finite"),
        ("invented_initial_feedback", "unsolved feedback must be empty"),
        ("unsolved_later_feedback", "only frame zero"),
        ("different_row_time", "inconsistent sample times"),
        ("reversed_time", "strictly increase"),
        ("wrong_bounds", "outside its substep bounds"),
    ),
)
def test_csv_rejects_broken_or_misleading_histories(tmp_path, telemetry_rows, damage, message):
    rows = telemetry_rows
    if damage == "missing_frame":
        rows = [row for row in rows if row["frame"] != 1]
    elif damage == "missing_joint":
        rows.pop(8)
    elif damage == "duplicate":
        rows.append(rows[8].copy())
    elif damage == "wrong_units":
        rows[8]["position_unit"] = "m"
    elif damage == "nonfinite_target":
        rows[8]["target_position"] = "nan"
    elif damage == "nonfinite_feedback":
        rows[8]["feedback_effort"] = "inf"
    elif damage == "invented_initial_feedback":
        rows[0]["feedback_effort"] = 0.0
    elif damage == "unsolved_later_feedback":
        rows[8]["feedback_valid"] = 0
    elif damage == "different_row_time":
        rows[8]["time_s"] = 1.0
    elif damage == "reversed_time":
        for row in rows:
            if row["frame"] == 2:
                row["time_s"] = 0.0
    elif damage == "wrong_bounds":
        rows[8]["feedback_effort"] = 100.0
    with pytest.raises(ValueError, match=message):
        load_telemetry(write_csv(tmp_path / "telemetry.csv", rows))


@pytest.mark.parametrize("kind", ("inputs", "feedback"))
def test_plot_expands_to_current_time_without_discarding_history_or_showing_future(tmp_path, telemetry_rows, kind):
    data = load_telemetry(write_csv(tmp_path / "telemetry.csv", telemetry_rows))
    chart = TelemetryPlot(data, kind)
    try:
        initial_y_limits = [axis.get_ylim() for axis in chart.axes]
        if kind == "feedback":
            assert np.isnan(chart.lines[0].get_ydata()).all()
            assert len(chart.dots[0].get_xdata()) == 0
        for frame in (1, 3):
            chart.update(frame)
            for axis, limits in zip(chart.axes, initial_y_limits, strict=True):
                assert axis.get_xlim() == (0.0, data.time_s[frame])
                assert axis.get_ylim() == limits
                assert all(spine.get_visible() for spine in axis.spines.values())
            for line in chart.lines:
                np.testing.assert_array_equal(line.get_xdata(), data.time_s[: frame + 1])
            assert not any(text.get_visible() for text in chart.waiting)
        assert "rad" in chart.axes[0].get_ylabel() if kind == "inputs" else "N·m" in chart.axes[0].get_ylabel()
        assert "(m)" in chart.axes[1].get_ylabel() if kind == "inputs" else "(N)" in chart.axes[1].get_ylabel()
        chart.figure.canvas.draw()
        assert chart.figure.canvas.get_width_height() == (1920, 1080)
    finally:
        chart.close()


def test_tiny_video_stream_encodes_and_fully_decodes(tmp_path, telemetry_rows):
    pytest.importorskip("imageio_ffmpeg")
    csv_path = write_csv(tmp_path / "telemetry.csv", telemetry_rows)
    csv_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "csv": csv_path.name,
                "feedback_sampling": "post_step",
                "feedback_model": "limited_pd_plus_feedforward",
            }
        ),
        encoding="utf-8",
    )
    data = load_telemetry(csv_path)
    result = render_video(data, csv_path, tmp_path / "motor_feedback.mp4", "feedback", 60, width=640, height=360)
    assert result["verification"]["full_decode"] == "passed"
    assert result["frame_count"] == 4
    assert result["feedback_sampling"] == "post_step"
    assert (tmp_path / "motor_feedback.png").is_file()
    assert (tmp_path / "motor_feedback.video.json").is_file()

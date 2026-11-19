"""Animate complete motor histories on axes that expand from zero to the current sample time.

python demos/piper_bag_transport/plot_motor_telemetry.py --csv outputs/motor_telemetry.csv --output-dir outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "utils"))
from encode_video import find_ffmpeg, verify_video

if __package__:
    from .run_artifacts import file_sha256, validate_run_manifest
else:
    from run_artifacts import file_sha256, validate_run_manifest


JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 9))
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#7955A2", "#434A54", "#0072B2", "#D55E00")
REQUIRED_COLUMNS = {
    "frame",
    "time_s",
    "joint_name",
    "joint_type",
    "position_unit",
    "effort_unit",
    "target_position",
    "target_velocity",
    "feedforward_effort",
    "feedback_effort",
    "feedback_effort_mean",
    "feedback_effort_min",
    "feedback_effort_max",
    "feedback_valid",
}


@dataclass(frozen=True)
class MotorTelemetry:
    time_s: np.ndarray
    target_position: np.ndarray
    feedback_effort: np.ndarray
    feedback_sampling: str = "unspecified"


def _finite(row: dict[str, str], column: str, line_number: int) -> float:
    try:
        value = float(row[column])
    except (ValueError, TypeError) as error:
        raise ValueError(f"CSV line {line_number}: {column} must contain a number") from error
    if not np.isfinite(value):
        raise ValueError(f"CSV line {line_number}: {column} must be finite")
    return value


def load_telemetry(path: Path) -> MotorTelemetry:
    """Validate a standalone long-format CSV, including every joint of every simulation frame."""
    records = {}
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing telemetry columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                frame = int(row["frame"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"CSV line {line_number}: frame must be a nonnegative integer") from error
            name = row["joint_name"]
            if frame < 0 or name not in JOINT_NAMES:
                raise ValueError(f"CSV line {line_number}: invalid frame or joint name")
            key = (frame, name)
            if key in records:
                raise ValueError(f"Duplicate telemetry sample for frame {frame}, {name}")
            rotary = JOINT_NAMES.index(name) < 6
            expected_type, expected_position = ("revolute", "rad") if rotary else ("prismatic", "m")
            effort_units = {"Nm", "N*m", "N m", "N·m"} if rotary else {"N"}
            if (
                row["joint_type"] != expected_type
                or row["position_unit"] != expected_position
                or row["effort_unit"] not in effort_units
            ):
                raise ValueError(f"CSV line {line_number}: inconsistent joint type or units for {name}")
            time_s = _finite(row, "time_s", line_number)
            target = _finite(row, "target_position", line_number)
            for column in ("target_velocity", "feedforward_effort"):
                _finite(row, column, line_number)
            if "target_time_s" in row:
                _finite(row, "target_time_s", line_number)
            if row["feedback_valid"] != str(int(frame > 0)):
                raise ValueError(f"CSV line {line_number}: only frame zero may have unsolved feedback")
            feedback_columns = ("feedback_effort", "feedback_effort_mean", "feedback_effort_min", "feedback_effort_max")
            if frame == 0:
                if any((row[column] or "").strip() for column in feedback_columns):
                    raise ValueError(f"CSV line {line_number}: unsolved feedback must be empty")
                effort = np.nan
            else:
                effort, mean, minimum, maximum = (_finite(row, column, line_number) for column in feedback_columns)
                tolerance = 1.0e-8 * max(1.0, abs(minimum), abs(maximum))
                if minimum > maximum or not minimum - tolerance <= mean <= maximum + tolerance:
                    raise ValueError(f"CSV line {line_number}: inconsistent feedback bounds")
                if not minimum - tolerance <= effort <= maximum + tolerance:
                    raise ValueError(f"CSV line {line_number}: feedback effort is outside its substep bounds")
            records[key] = (time_s, target, effort)
    frames = sorted({frame for frame, _ in records})
    if len(frames) < 2 or frames != list(range(len(frames))):
        raise ValueError("Telemetry requires contiguous frames starting at zero and at least one solved frame")
    times, targets, efforts = [], [], []
    for frame in frames:
        if any((frame, name) not in records for name in JOINT_NAMES):
            raise ValueError(f"Frame {frame} does not contain all eight joints")
        values = np.asarray([records[frame, name] for name in JOINT_NAMES])
        if not np.allclose(values[:, 0], values[0, 0], rtol=0.0, atol=1.0e-9):
            raise ValueError(f"Frame {frame} contains inconsistent sample times")
        times.append(values[0, 0])
        targets.append(values[:, 1])
        efforts.append(values[:, 2])
    time_s = np.asarray(times)
    if abs(time_s[0]) > 1.0e-9 or np.any(np.diff(time_s) <= 0.0):
        raise ValueError("Telemetry time must start at zero and strictly increase")
    feedback_sampling = "unspecified"
    sidecar_path = path.with_suffix(".json")
    if sidecar_path.is_file():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("csv") != path.name:
            raise ValueError("Telemetry metadata references a different CSV")
        # Legacy histories may contain nonlinear solver samples; only an explicit marker permits the new label.
        if sidecar.get("feedback_sampling") == "post_step":
            if sidecar.get("feedback_model") != "limited_pd_plus_feedforward":
                raise ValueError("Post-step telemetry has an unsupported actuator effort model")
            feedback_sampling = "post_step"
    return MotorTelemetry(time_s, np.asarray(targets), np.asarray(efforts), feedback_sampling)


def _limits(values: np.ndarray, minimum_span: float) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    low, high = float(finite.min()), float(finite.max())
    span = max(high - low, minimum_span)
    midpoint = (low + high) * 0.5
    return midpoint - 0.58 * span, midpoint + 0.58 * span


class TelemetryPlot:
    """Keep the complete observed history visible while the right boundary follows the current time."""

    def __init__(self, telemetry: MotorTelemetry, kind: str, width: int = 1920, height: int = 1080):
        if kind not in ("inputs", "feedback"):
            raise ValueError("Plot kind must be inputs or feedback")
        self.telemetry = telemetry
        self.kind = kind
        self.values = telemetry.target_position if kind == "inputs" else telemetry.feedback_effort
        # Scale typography with preview resolution so axis labels retain the full-size layout and margins.
        dpi = 120 * min(width / 1920, height / 1080)
        self.figure, self.axes = plt.subplots(2, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
        self.figure.set_facecolor("#FFFFFF")
        self.figure.subplots_adjust(left=0.10, right=0.955, bottom=0.09, top=0.795, hspace=0.49)
        title = "Motor position commands" if kind == "inputs" else "Motor effort feedback"
        subtitle = (
            "Piper arm  /  commanded joint positions"
            if kind == "inputs"
            else (
                "Piper arm  /  post-step actuator effort at the final physics substep"
                if telemetry.feedback_sampling == "post_step"
                else "Piper arm  /  recorded effort at the final physics substep"
            )
        )
        self.figure.text(0.10, 0.945, title, fontsize=23, weight="semibold", color="#142333", va="center")
        self.figure.text(0.10, 0.895, subtitle, fontsize=12, color="#526273", va="center")
        self.clock = self.figure.text(0.955, 0.945, "", ha="right", va="center", fontsize=18, color="#142333")
        self.lines, self.dots, self.waiting = [], [], []
        labels = (
            ("Commanded angle (rad)", "Commanded position (m)")
            if kind == "inputs"
            else ("Motor torque (N·m)", "Motor force (N)")
        )
        for panel, (axis, indices) in enumerate(zip(self.axes, (range(6), range(6, 8)), strict=True)):
            axis.set_facecolor("#FCFDFE")
            for spine in axis.spines.values():
                spine.set_visible(True)
                spine.set_color("#415263")
                spine.set_linewidth(0.9)
            axis.tick_params(which="both", direction="in", top=True, right=True, labelsize=11, colors="#344454")
            axis.tick_params(which="major", length=6, width=0.9, pad=8)
            axis.tick_params(which="minor", length=3, width=0.6)
            axis.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=3))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))
            axis.ticklabel_format(axis="both", style="plain", useOffset=False)
            axis.grid(which="major", color="#CBD5DE", linewidth=0.65, alpha=0.6)
            axis.set_axisbelow(True)
            axis.set_xlabel("Time (s)", fontsize=12, color="#243647", labelpad=8)
            axis.set_ylabel(labels[panel], fontsize=12, color="#243647", labelpad=12)
            axis.set_title("Revolute joints" if panel == 0 else "Finger sliders", loc="left", fontsize=13, pad=19)
            span = (0.1 if panel == 0 else 0.001) if kind == "inputs" else (0.1 if panel == 0 else 1.0)
            axis.set_ylim(*_limits(self.values[:, indices], span))
            for index in indices:
                (line,) = axis.plot([], [], color=COLORS[index], linewidth=1.8, label=f"J{index + 1}", alpha=0.94)
                (dot,) = axis.plot([], [], color=COLORS[index], marker="o", markersize=4, linestyle="none")
                self.lines.append(line)
                self.dots.append(dot)
            axis.legend(
                loc="lower right",
                bbox_to_anchor=(1.0, 1.028),
                ncol=len(indices),
                frameon=False,
                fontsize=11,
                handlelength=2.3,
                columnspacing=1.45,
                borderaxespad=0,
            )
            self.waiting.append(
                axis.text(
                    0.5,
                    0.5,
                    "Awaiting first solved sample",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="#637386",
                    fontsize=13,
                )
            )
        self.update(0)

    def update(self, frame: int) -> None:
        time_s = self.telemetry.time_s[: frame + 1]
        current = float(time_s[-1])
        # Matplotlib cannot represent [0, 0]; the first unsolved instant gets one sample interval.
        right = current if current > 0.0 else float(self.telemetry.time_s[1])
        for axis in self.axes:
            axis.set_xlim(0.0, right)
        self.clock.set_text(f"t = {current:05.2f} s")
        for index, (line, dot) in enumerate(zip(self.lines, self.dots, strict=True)):
            observed = self.values[: frame + 1, index]
            line.set_data(time_s, observed)
            valid = np.isfinite(observed[-1])
            dot.set_data([current] if valid else [], [observed[-1]] if valid else [])
        for label in self.waiting:
            label.set_visible(self.kind == "feedback" and frame == 0)

    def close(self) -> None:
        plt.close(self.figure)


def render_video(
    telemetry: MotorTelemetry,
    csv_path: Path,
    output: Path,
    kind: str,
    fps: float,
    width: int = 1920,
    height: int = 1080,
    crf: int = 16,
) -> dict:
    """Stream Agg pixels directly to H.264, retaining the final chart and a verified video manifest."""
    if fps <= 0 or not np.isfinite(fps) or not 0 <= crf <= 51:
        raise ValueError("Frame rate must be positive and CRF must be between zero and 51")
    if width < 320 or height < 240 or width % 2 or height % 2:
        raise ValueError("Video dimensions must be even and at least 320 by 240")
    run_path = csv_path.resolve().parent / "run_manifest.json"
    run = validate_run_manifest(run_path) if run_path.is_file() else None
    csv_hash = file_sha256(csv_path)
    if run is not None:
        if run["artifacts"]["motor_telemetry"]["sha256"] != csv_hash:
            raise ValueError("The chart CSV is not the telemetry recorded by this simulation")
        if fps != run["timeline"]["fps"] or len(telemetry.time_s) != run["timeline"]["frame_count"]:
            raise ValueError("Chart frame count and frame rate must match the sealed simulation")
    ffmpeg = find_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".encoding.mp4")
    log_path = output.with_suffix(".encode.log")
    command = [
        ffmpeg,
        "-hide_banner",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-vf",
        "scale=in_range=pc:out_range=tv:out_color_matrix=bt709",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    chart = TelemetryPlot(telemetry, kind, width, height)
    count = len(telemetry.time_s)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT)
            try:
                for frame in range(count):
                    chart.update(frame)
                    chart.figure.canvas.draw()
                    process.stdin.write(chart.figure.canvas.buffer_rgba())
                    if frame % 120 == 0 or frame == count - 1:
                        print(f"{output.name}: {frame + 1}/{count} frames", flush=True)
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError(f"Video encoding failed; see {log_path}")
            except BaseException:
                process.kill()
                process.wait()
                raise
        verification = verify_video(ffmpeg, temporary, (width, height), fps, count)
        temporary.replace(output)
        chart.figure.savefig(output.with_suffix(".png"), dpi=chart.figure.dpi, facecolor="white")
    finally:
        chart.close()
    if isinstance(verification["ffprobe"], dict):
        verification["ffprobe"]["format"]["filename"] = output.name
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "input_csv": str(csv_path.resolve()),
        "input_sha256": csv_hash,
        "output": output.name,
        "output_sha256": file_sha256(output),
        "quantity": "target_position" if kind == "inputs" else "feedback_effort",
        "joint_names": list(JOINT_NAMES),
        "units": ["rad", "m"] if kind == "inputs" else ["N·m", "N"],
        "frame_count": count,
        "width": width,
        "height": height,
        "fps": fps,
        "first_frame": 0,
        "last_frame": count - 1,
        "duration_seconds": count / fps,
        "simulation_duration_seconds": float(telemetry.time_s[-1]),
        "axis_time_range": "[0, current sample time]; frame zero uses one sample interval",
        "missing_feedback": "Unsolved frame-zero feedback omitted",
        "feedback_sampling": telemetry.feedback_sampling,
        "codec": "h264",
        "pixel_format": "yuv420p",
        "crf": crf,
        "preset": "slow",
        "verification": verification,
    }
    if file_sha256(csv_path) != csv_hash:
        raise RuntimeError("Telemetry CSV changed while the chart was rendering")
    if run is not None:
        manifest.update(run_id=run["run_id"], source_metadata_sha256=run["metadata"]["sha256"])
    output.with_suffix(".video.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified: {output}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Standalone long-format motor telemetry CSV")
    parser.add_argument("--output-dir", type=Path, help="Defaults to the CSV directory")
    parser.add_argument("--fps", type=float, default=60, help="Playback frame rate; one video frame per CSV sample")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--kind", choices=("both", "inputs", "feedback"), default="both")
    args = parser.parse_args()
    telemetry = load_telemetry(args.csv)
    output_dir = args.output_dir or args.csv.parent
    for kind in ("inputs", "feedback") if args.kind == "both" else (args.kind,):
        render_video(
            telemetry, args.csv, output_dir / f"motor_{kind}.mp4", kind, args.fps, args.width, args.height, args.crf
        )


if __name__ == "__main__":
    main()

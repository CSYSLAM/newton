"""Seal portable simulation exports and verify that all three videos use the same samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path

ROLES = ("usd", "raw_states", "motor_telemetry")
SCHEMA_VERSION = 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def portable_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(
        not Path(relative).is_absolute() and path.is_relative_to(root),
        f"Nonportable path outside the artifact directory: {relative}",
    )
    return path


def check_hash(path, expected):
    require(Path(path).is_file(), f"Missing artifact: {path}")
    require(file_sha256(path) == expected, f"Artifact hash changed: {path}")


def snapshot_sources(metadata, root, repository):
    """Preserve the exact exported source bytes, so later renames cannot invalidate a historical run."""
    root, repository = Path(root).resolve(), Path(repository).resolve()
    snapshots = dict(metadata.get("source_snapshots", {}))
    for relative, expected in metadata.get("source_sha256", {}).items():
        snapshot = snapshots.get(relative, (Path("sources") / relative).as_posix())
        target = portable_path(root, snapshot)
        if relative not in snapshots and target.is_file() and file_sha256(target) != expected:
            # A new export may share an output folder with an older run; retain both source versions.
            snapshot = (Path("sources") / expected / relative).as_posix()
            target = portable_path(root, snapshot)
        if not target.is_file():
            source = portable_path(repository, relative)
            check_hash(source, expected)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        check_hash(target, expected)
        snapshots[relative] = snapshot
    metadata["source_snapshots"] = snapshots


def validate_sources(metadata, root, repository=None):
    snapshots = metadata.get("source_snapshots", {})
    for relative, expected in metadata.get("source_sha256", {}).items():
        if relative in snapshots:
            path = portable_path(root, snapshots[relative])
        else:
            require(repository is not None, f"Missing portable source snapshot: {relative}")
            path = portable_path(repository, relative)
        check_hash(path, expected)


def _export_identity(metadata, root):
    sim = metadata["simulation"]
    frames, samples, fps = sim["frames"], sim["samples"], float(sim["fps"])
    require(isinstance(frames, int) and frames >= 1 and samples == frames + 1, "Invalid simulation frame range")
    require(math.isfinite(fps) and fps > 0, "Invalid simulation frame rate")
    hashes = metadata["file_sha256"] | metadata.get("texture_sha256", {})
    for relative, expected in hashes.items():
        check_hash(portable_path(root, relative), expected)
    artifacts = {}
    for role in ROLES:
        require(role in metadata["files"], f"Same-run video export requires {role}")
        relative = metadata["files"][role]
        require(relative in metadata["file_sha256"], f"Missing export hash for {role}")
        artifacts[role] = {"path": relative, "sha256": metadata["file_sha256"][relative]}
    timeline = {"first_frame": 0, "last_frame": frames, "frame_count": samples, "fps": fps}
    identity = {"artifacts": {role: entry["sha256"] for role, entry in artifacts.items()}, "timeline": timeline}
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return run_id, artifacts, timeline


def seal_run(metadata_path, repository=None):
    """Bind USD, raw states, and CSV using hashes recorded by the single export loop."""
    metadata_path = Path(metadata_path).resolve()
    root = metadata_path.parent
    metadata = read_json(metadata_path)
    repository = Path(repository) if repository is not None else Path(__file__).resolve().parents[2]
    # Verify the original hashes before copying or adding any provenance; never bless modified exports.
    run_id, artifacts, timeline = _export_identity(metadata, root)
    snapshot_sources(metadata, root, repository)
    validate_sources(metadata, root)
    if metadata != read_json(metadata_path):
        write_json(metadata_path, metadata)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "metadata": {"path": metadata_path.name, "sha256": file_sha256(metadata_path)},
        "artifacts": artifacts,
        "timeline": timeline,
    }
    write_json(root / "run_manifest.json", manifest)
    return manifest


def validate_run_manifest(path):
    path = Path(path).resolve()
    root = path.parent
    manifest = read_json(path)
    require(manifest.get("schema_version") == SCHEMA_VERSION, "Unsupported run manifest schema")
    metadata_record = manifest["metadata"]
    metadata_path = portable_path(root, metadata_record["path"])
    check_hash(metadata_path, metadata_record["sha256"])
    metadata = read_json(metadata_path)
    validate_sources(metadata, root)
    run_id, artifacts, timeline = _export_identity(metadata, root)
    require(manifest.get("run_id") == run_id, "Run identity hash does not match the exported samples")
    require(manifest.get("artifacts") == artifacts, "Run artifact hashes differ from the original export")
    require(manifest.get("timeline") == timeline, "Run timeline differs from the original export")
    return manifest


def _check_timeline(record, timeline, label):
    for key, expected in timeline.items():
        require(record.get(key) == expected, f"{label}: {key} differs from the simulation")


def _decode_video(path, width, height, fps, count):
    from demos.utils.encode_video import find_ffmpeg, verify_video

    result = verify_video(find_ffmpeg(), path, (width, height), fps, count)
    require(isinstance(result.get("ffprobe"), dict), "ffprobe is required to verify synchronized video clocks")
    streams = result["ffprobe"]["streams"]
    require(
        len(streams) == 1 and int(streams[0].get("nb_frames", -1)) == count,
        f"Video frame count could not be verified: {path}",
    )
    start = float(streams[0].get("start_time", "nan"))
    duration = float(streams[0].get("duration", "nan"))
    # ffprobe rounds stream timestamps to microseconds; container duration may have coarser rounding.
    require(math.isclose(start, 0.0, rel_tol=0, abs_tol=1e-6), f"Video does not start at simulation time zero: {path}")
    require(
        math.isclose(duration, count / fps, rel_tol=0, abs_tol=1e-6),
        f"Video duration differs from the complete simulation sample range: {path}",
    )
    return result


def verify_bundle(output_dir, scene_video="piper_bag_transport.mp4", decode_video=None):
    """Verify content hashes, full frame coverage, clock alignment, and a fresh decode of all videos."""
    root = Path(output_dir).resolve()
    run = validate_run_manifest(root / "run_manifest.json")
    timeline = run["timeline"]
    decode_video = _decode_video if decode_video is None else decode_video
    videos = {}
    frame_count = 0
    for kind, relative in (("scene", scene_video), ("inputs", "motor_inputs.mp4"), ("feedback", "motor_feedback.mp4")):
        video_path = portable_path(root, relative)
        record = read_json(video_path.with_suffix(".video.json"))
        require(record.get("output") == video_path.name, f"{kind}: incorrect output filename")
        require(record.get("run_id") == run["run_id"], f"{kind}: video belongs to a different simulation")
        _check_timeline(record, timeline, kind)
        duration = timeline["frame_count"] / timeline["fps"]
        require(
            math.isclose(record.get("duration_seconds", -1), duration, rel_tol=0, abs_tol=1e-9),
            f"{kind}: video duration differs from the simulation",
        )
        check_hash(video_path, record.get("output_sha256"))
        if kind == "scene":
            require(
                record.get("source_usd_sha256") == run["artifacts"]["usd"]["sha256"], "Scene video uses a different USD"
            )
            render_reference = record["render_manifest"]
            render_path = portable_path(root, render_reference["path"])
            check_hash(render_path, render_reference["sha256"])
            rendered = read_json(render_path)
            require(
                rendered.get("run_id") == run["run_id"]
                and rendered.get("usd_sha256") == run["artifacts"]["usd"]["sha256"],
                "Rendered frames belong to a different simulation",
            )
            _check_timeline(rendered, timeline, "Rendered frames")
            expected_frames = {str(frame) for frame in range(timeline["first_frame"], timeline["last_frame"] + 1)}
            require(set(rendered["frames"]) == expected_frames, "Rendered frame provenance is incomplete")
            for frame, entry in rendered["frames"].items():
                require(entry["path"] == f"frame_{int(frame):04d}.png", "Unexpected rendered frame filename")
                check_hash(portable_path(render_path.parent, entry["path"]), entry["sha256"])
            frame_count = len(expected_frames)
        else:
            require(
                record.get("input_sha256") == run["artifacts"]["motor_telemetry"]["sha256"],
                f"{kind}: video uses a different telemetry CSV",
            )
            quantity = "target_position" if kind == "inputs" else "feedback_effort"
            require(record.get("quantity") == quantity, f"{kind}: wrong chart quantity")
        verification = decode_video(
            video_path, record["width"], record["height"], timeline["fps"], timeline["frame_count"]
        )
        require(verification.get("full_decode") == "passed", f"{kind}: full video decode did not pass")
        videos[kind] = {"path": relative, "sha256": record["output_sha256"], "verification": verification}
    return {
        "status": "passed",
        "run_id": run["run_id"],
        "timeline": timeline,
        "metadata_sha256": run["metadata"]["sha256"],
        "rendered_frames_verified": frame_count,
        "videos": videos,
        "verified_utc": datetime.now(UTC).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    manifest = seal_run(args.metadata)
    print(f"Sealed simulation {manifest['run_id']} ({manifest['timeline']['frame_count']} samples)", flush=True)


if __name__ == "__main__":
    main()

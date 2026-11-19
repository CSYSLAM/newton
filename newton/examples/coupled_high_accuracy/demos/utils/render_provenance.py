"""Bind rendered frames and encoded movies to one immutable simulation run."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_run(path: Path) -> dict:
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("schema_version") != 1 or not run.get("run_id"):
        raise ValueError(f"Invalid simulation run manifest: {path}")
    usd = run["artifacts"]["usd"]
    if sha256(path.parent / usd["path"]) != usd["sha256"]:
        raise ValueError("Simulation USD differs from the sealed run manifest")
    return run


def validate_scene_run(run: dict, usd_hashes: list[str], first: int, last: int, fps: float) -> None:
    if not usd_hashes or set(usd_hashes) != {run["artifacts"]["usd"]["sha256"]}:
        raise ValueError("Blender USD sources do not match the sealed simulation run")
    timeline = run["timeline"]
    if not timeline["first_frame"] <= first <= last <= timeline["last_frame"]:
        raise ValueError("Blender frame range is outside the sealed simulation run")
    if not math.isclose(fps, timeline["fps"], rel_tol=0, abs_tol=1.0e-8):
        raise ValueError("Blender frame rate differs from the sealed simulation run")


class RenderManifest:
    """Reject stale PNG reuse and commit the fingerprint of each successfully written frame."""

    def __init__(
        self,
        folder: Path,
        blend: Path,
        run_path: Path,
        first: int,
        last: int,
        fps: float,
        settings: dict,
        *,
        overwrite: bool = False,
    ):
        self.folder = folder
        self.path = folder / "render_manifest.json"
        run = load_run(run_path)
        self.identity = {
            "schema_version": 1,
            "run_id": run["run_id"],
            "usd_sha256": run["artifacts"]["usd"]["sha256"],
            "blend_sha256": sha256(blend),
            "first_frame": first,
            "last_frame": last,
            "frame_count": last - first + 1,
            "fps": fps,
            "settings": settings,
        }
        previous = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else None
        matching = previous is not None and all(previous.get(key) == value for key, value in self.identity.items())
        if not overwrite:
            if previous is not None and not matching:
                raise ValueError("Rendered frame provenance changed; use a new frame directory or --overwrite")
            if previous is None and next(folder.glob("frame_*.png"), None) is not None:
                raise ValueError("Existing PNG frames have no run provenance; use a new directory or --overwrite")
        self.data = dict(previous) if matching else {**self.identity, "frames": {}}
        self.data["run_manifest"] = Path(os.path.relpath(run_path, folder)).as_posix()
        atomic_json(self.path, self.data)

    def validate_existing(self, frame: int, path: Path) -> None:
        recorded = self.data["frames"].get(str(frame))
        if recorded is None or recorded["path"] != path.name or recorded["sha256"] != sha256(path):
            raise ValueError(f"PNG frame {frame} is missing or differs from its provenance; rerender with --overwrite")

    def record(self, frame: int, path: Path) -> None:
        self.data["frames"][str(frame)] = {"path": path.name, "sha256": sha256(path)}
        atomic_json(self.path, self.data)


def verify_frame_provenance(paths: list[Path], output_dir: Path, first: int, fps: float) -> dict:
    """Return video provenance only after every requested frame matches its recorded source run."""
    frame_manifest = paths[0].parent / "render_manifest.json"
    if not frame_manifest.exists():
        if (output_dir / "run_manifest.json").exists():
            raise ValueError("This run has no rendered-frame provenance; rebuild and render before encoding")
        return {}
    rendered = json.loads(frame_manifest.read_text(encoding="utf-8"))
    run_path = (frame_manifest.parent / rendered["run_manifest"]).resolve()
    run = load_run(run_path)
    if rendered["run_id"] != run["run_id"] or rendered["usd_sha256"] != run["artifacts"]["usd"]["sha256"]:
        raise ValueError("Rendered frames belong to a different simulation run")
    if not math.isclose(fps, rendered["fps"], rel_tol=0, abs_tol=1.0e-8):
        raise ValueError("Encoding frame rate differs from the recorded rendering frame rate")
    for index, path in enumerate(paths, start=first):
        recorded = rendered["frames"].get(str(index))
        if recorded is None or recorded["path"] != path.name or recorded["sha256"] != sha256(path):
            raise ValueError(f"Rendered PNG frame {index} differs from the recorded simulation run")
    return {
        "run_id": run["run_id"],
        "source_usd_sha256": rendered["usd_sha256"],
        "render_manifest": {
            "path": Path(os.path.relpath(frame_manifest, output_dir)).as_posix(),
            "sha256": sha256(frame_manifest),
        },
    }

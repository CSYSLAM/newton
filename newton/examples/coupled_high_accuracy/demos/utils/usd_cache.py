"""Record imported USD versions and reject stale cache references before rendering."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import bpy
from pxr import Usd

SOURCE_HASHES = "usd_source_sha256"


def usd_sources():
    for cache in bpy.data.cache_files:
        path = Path(bpy.path.abspath(cache.filepath)).resolve()
        if path.suffix.lower() in {".usd", ".usda", ".usdc"}:
            yield cache, path


def record_usd_sources(scene):
    # Static imported geometry is baked into the .blend and cannot follow a replaced USD.
    scene[SOURCE_HASHES] = json.dumps(
        {cache.filepath: hashlib.sha256(path.read_bytes()).hexdigest() for cache, path in usd_sources()}
    )


def validate_usd_sources(scene):
    recorded = json.loads(scene.get(SOURCE_HASHES, "{}"))
    stages, errors = {}, []
    for cache, path in usd_sources():
        if not path.is_file():
            errors.append(f"Missing USD file: {path}")
            continue
        expected = recorded.get(cache.filepath)
        if recorded and expected is None:
            errors.append(f"USD reference changed since the project was built: {cache.filepath}")
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"USD changed since the project was built: {cache.filepath}")
        stages[cache.filepath] = Usd.Stage.Open(str(path))
    # Legacy projects lack fingerprints, but missing prims still reveal orphaned imported objects.
    for obj in scene.objects:
        readers = [modifier for modifier in obj.modifiers if modifier.type == "MESH_SEQUENCE_CACHE"]
        readers.extend(constraint for constraint in obj.constraints if constraint.type == "TRANSFORM_CACHE")
        for reader in readers:
            if reader.cache_file is None:
                continue
            stage = stages.get(reader.cache_file.filepath)
            if stage is not None and not stage.GetPrimAtPath(reader.object_path):
                errors.append(f"Stale object {obj.name}: USD prim is missing: {reader.object_path}")
    if errors:
        raise RuntimeError(
            "USD/project mismatch. Rebuild the .blend from the current USD before rendering. "
            "Back up any manual scene edits first.\n" + "\n".join(errors)
        )

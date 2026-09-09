# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Download the W1 Pikka model pinned to Dexsim MR 1269 using authenticated glab."""

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from urllib.parse import quote

HOST = "192.168.3.16"
PROJECT = 77
REVISION = "57acb4907f639d2af86cbb159b5fd0cf20e0f396"
SOURCE_ROOT = "resources/robots/W1-hand-obj"
ROBOT_FILE = "DexforceW1V021_pikka_gripper_simple_visual_collision.urdf"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "w1-pikka-gripper"


def download(relative: str, destination: Path) -> bytes:
    """Retrieve one pinned file without exposing GitLab credentials."""
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid asset path: {relative}")
    endpoint = f"projects/{PROJECT}/repository/files/{quote(f'{SOURCE_ROOT}/{relative}', safe='')}/raw?ref={REVISION}"
    result = subprocess.run(["glab", "api", endpoint, "--hostname", HOST], check=True, capture_output=True, timeout=120)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download")
    temporary.write_bytes(result.stdout)
    temporary.replace(destination)
    return result.stdout


def main() -> None:
    """Download the URDF and exactly its referenced meshes, with a hash manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--body-assets", type=Path, default=DEFAULT_OUTPUT.parent / "W1-hand-obj")
    args = parser.parse_args()
    root = args.output.expanduser().resolve()
    urdf = download(ROBOT_FILE, root / ROBOT_FILE)
    paths = sorted({mesh.attrib["filename"] for mesh in ET.fromstring(urdf).iter("mesh")})
    hashes = {ROBOT_FILE: hashlib.sha256(urdf).hexdigest()}
    reused = []
    for index, path in enumerate(paths, 1):
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid mesh path: {path}")
        # The MR references the separately distributed W1 body meshes, which
        # are absent from GitLab. Reuse the same named local W1 asset files.
        body_mesh = args.body_assets / path
        if path.startswith("Visual/") and body_mesh.is_file():
            content = body_mesh.read_bytes()
            destination = root / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            reused.append(path)
        else:
            content = download(path, root / path)
        hashes[path] = hashlib.sha256(content).hexdigest()
        print(f"[{index}/{len(paths)}] {path} ({len(content)} bytes)", flush=True)
    manifest = {
        "source": f"http://{HOST}/Engine/dexsim/-/merge_requests/1269",
        "revision": REVISION,
        "sourceRoot": SOURCE_ROOT,
        "urdf": ROBOT_FILE,
        "sha256": hashes,
        "reusedBodyMeshes": reused,
        "bodyMeshSource": str(args.body_assets.resolve()),
        "license": "Source assets retain their original rights; no relicensing is implied.",
    }
    (root / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloaded {len(hashes)} files to {root}", flush=True)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Locate assets from the Style3D Sim asset repository."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPOSITORY = "https://github.com/Style3D/style3d-sim-assets.git"
_ASSETS_DIR_ENV = "STYLE3D_SIM_ASSETS_DIR"


def _cache_root() -> Path:
    return Path.home() / ".cache" / "style3d_sim" / "assets"


def get_assets() -> str:
    """Return the asset root, cloning the repository on first use.

    Set STYLE3D_SIM_ASSETS_DIR to use an existing checkout instead. The asset
    repository root must contain the asset groups directly.
    """
    override = os.environ.get(_ASSETS_DIR_ENV)
    if override:
        root = Path(override).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"{_ASSETS_DIR_ENV} is not a directory: {root}")
        return str(root)

    root = _cache_root()
    if root.is_dir():
        return str(root)
    if root.exists():
        raise NotADirectoryError(f"Asset cache path is not a directory: {root}")

    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="assets-", dir=root.parent) as staging:
        checkout = Path(staging) / "checkout"
        print(f"Downloading Style3D Sim assets from {_REPOSITORY}...", file=sys.stderr, flush=True)  # noqa: T201 (progress of a one-off asset clone)
        try:
            subprocess.run(
                ["git", "clone", "--progress", "--depth", "1", "--", _REPOSITORY, str(checkout)],
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Git is required to download Style3D Sim assets") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Could not clone Style3D Sim assets from {_REPOSITORY} (git exit code {exc.returncode})"
            ) from exc

        # Publish only a completed checkout; another process may have populated the cache meanwhile.
        if not root.exists():
            try:
                checkout.rename(root)
            except OSError:
                if not root.is_dir():
                    raise
    print(f"Style3D Sim assets ready at {root}", file=sys.stderr, flush=True)  # noqa: T201 (progress of a one-off asset clone)
    return str(root)


def get_asset(filename: str | os.PathLike[str]) -> str:
    """Return an asset path, refreshing a stale default checkout when needed."""
    root = Path(get_assets()).resolve()
    asset = (root / filename).resolve()
    if not asset.is_relative_to(root):
        raise ValueError(f"Asset path escapes the asset root: {filename}")
    if asset.is_file():
        return str(asset)
    if os.environ.get(_ASSETS_DIR_ENV):
        raise FileNotFoundError(f"Asset does not exist in {_ASSETS_DIR_ENV}: {filename}")
    # New demos may reference assets added after the cached repository was cloned.
    print(f"Asset {filename} is missing; updating Style3D Sim assets at {root}...", file=sys.stderr, flush=True)  # noqa: T201 (progress of a one-off asset clone)
    try:
        subprocess.run(["git", "-C", str(root), "pull", "--ff-only", "--progress"], check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Git is required to update Style3D Sim assets") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Could not update Style3D Sim assets at {root} (git exit code {exc.returncode})") from exc
    if not asset.is_file():
        raise FileNotFoundError(f"Asset does not exist in the Style3D Sim repository: {filename}")
    return str(asset)

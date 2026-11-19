import subprocess
from pathlib import Path

import pytest

from newton._src.solvers.coupled_high_accuracy import assets


def test_get_assets_clones_once_and_get_asset_uses_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "cache" / "assets"
    calls = []

    def clone(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        asset = Path(command[-1]) / "joints" / "ball" / "ball.urdf"
        asset.parent.mkdir(parents=True)
        asset.write_text("<robot/>", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.delenv("STYLE3D_SIM_ASSETS_DIR", raising=False)
    monkeypatch.setattr(assets, "_cache_root", lambda: root)
    monkeypatch.setattr(assets.subprocess, "run", clone)

    assert assets.get_assets() == str(root)
    assert Path(assets.get_asset("joints/ball/ball.urdf")).read_text(encoding="utf-8") == "<robot/>"
    assert assets.get_assets() == str(root)
    assert len(calls) == 1
    assert calls[0][:7] == ["git", "clone", "--progress", "--depth", "1", "--", assets._REPOSITORY]
    output = capsys.readouterr().err
    assert "Downloading Style3D Sim assets" in output
    assert f"Style3D Sim assets ready at {root}" in output


def test_get_assets_uses_existing_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STYLE3D_SIM_ASSETS_DIR", str(tmp_path))
    monkeypatch.setattr(assets, "_cache_root", lambda: pytest.fail("Unexpected cache access"))

    assert assets.get_assets() == str(tmp_path)


def test_get_asset_refreshes_stale_default_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "cache" / "assets"
    root.mkdir(parents=True)
    calls = []

    def update(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        asset = root / "models" / "apple" / "apple.usd"
        asset.parent.mkdir(parents=True)
        asset.write_bytes(b"usd")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.delenv("STYLE3D_SIM_ASSETS_DIR", raising=False)
    monkeypatch.setattr(assets, "_cache_root", lambda: root)
    monkeypatch.setattr(assets.subprocess, "run", update)

    assert Path(assets.get_asset("models/apple/apple.usd")).read_bytes() == b"usd"
    assert calls == [["git", "-C", str(root), "pull", "--ff-only", "--progress"]]


def test_get_asset_missing_in_override_does_not_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STYLE3D_SIM_ASSETS_DIR", str(tmp_path))
    monkeypatch.setattr(assets.subprocess, "run", lambda *args, **kwargs: pytest.fail("Unexpected update"))

    with pytest.raises(FileNotFoundError, match="STYLE3D_SIM_ASSETS_DIR"):
        assets.get_asset("models/apple/apple.usd")


def test_get_asset_missing_after_update(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STYLE3D_SIM_ASSETS_DIR", raising=False)
    monkeypatch.setattr(assets, "_cache_root", lambda: tmp_path)
    calls = []

    def update(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(assets.subprocess, "run", update)

    with pytest.raises(FileNotFoundError, match="does not exist in the Style3D Sim repository"):
        assets.get_asset("models/missing.usd")
    assert len(calls) == 1


def test_get_asset_failed_update_preserves_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = tmp_path / "existing.usd"
    existing.write_bytes(b"existing")
    monkeypatch.delenv("STYLE3D_SIM_ASSETS_DIR", raising=False)
    monkeypatch.setattr(assets, "_cache_root", lambda: tmp_path)

    def fail_update(command: list[str], **kwargs: object) -> None:
        raise subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(assets.subprocess, "run", fail_update)

    with pytest.raises(RuntimeError, match="Could not update.*git exit code 128"):
        assets.get_asset("models/apple/apple.usd")
    assert existing.read_bytes() == b"existing"


def test_get_assets_does_not_publish_failed_clone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "cache" / "assets"
    monkeypatch.delenv("STYLE3D_SIM_ASSETS_DIR", raising=False)
    monkeypatch.setattr(assets, "_cache_root", lambda: root)

    def fail_clone(command: list[str], **kwargs: object) -> None:
        raise subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(assets.subprocess, "run", fail_clone)

    with pytest.raises(RuntimeError, match="git exit code 128"):
        assets.get_assets()
    assert not root.exists()


def test_get_asset_rejects_paths_outside_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STYLE3D_SIM_ASSETS_DIR", str(tmp_path))

    with pytest.raises(ValueError, match="escapes the asset root"):
        assets.get_asset("../outside.usd")

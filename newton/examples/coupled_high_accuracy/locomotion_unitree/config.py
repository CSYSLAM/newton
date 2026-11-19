"""Configuration loading for the Unitree-compatible Go2 task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from locomotion.policy_spec import PolicySpec, load_policy_spec

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "go2.yaml"


def load_config() -> tuple[PolicySpec, dict[str, Any]]:
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return load_policy_spec(CONFIG_PATH), data["unitree"]

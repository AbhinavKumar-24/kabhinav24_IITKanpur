"""Configuration loading. All thresholds live in config.yaml and are fixed before evaluation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def deep_update(base: dict, update: dict) -> dict:
    """Return a copy of ``base`` with ``update`` merged in recursively."""
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load config.yaml (or ``path``) and apply optional overrides."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = deep_update(cfg, overrides)
    cfg["_root"] = str(cfg_path.resolve().parent)
    return cfg


def repo_path(cfg: dict, *parts: str) -> Path:
    """Path relative to the folder that holds the config file (the repository root)."""
    return Path(cfg["_root"]).joinpath(*parts)

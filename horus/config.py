"""Shared helpers for loading YAML configs and resolving paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Return the repository root (parent of the ``horus`` package)."""
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``configs/data.yaml`` (or an explicit path) and resolve relative paths."""
    cfg_path = Path(path) if path else repo_root() / "configs" / "data.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = repo_root()
    paths = cfg.setdefault("paths", {})
    raw = Path(paths.get("raw_dataset", "Dataset"))
    processed = Path(paths.get("processed_root", "data/processed"))
    paths["raw_dataset"] = str(raw if raw.is_absolute() else root / raw)
    paths["processed_root"] = str(
        processed if processed.is_absolute() else root / processed
    )
    return cfg


def events_clip_dir(cfg: dict[str, Any], game: str, clip: str) -> Path:
    """Return ``data/processed/events/<game>/<clip>/``."""
    root = Path(cfg["paths"]["processed_root"])
    sub = cfg["paths"].get("events_subdir", "events")
    return root / sub / game / clip

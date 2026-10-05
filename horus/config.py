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


def resolve_checkpoint_path(
    train_cfg: dict[str, Any],
    default_name: str,
    *,
    name_override: str | None = None,
) -> Path:
    """``checkpoint_dir`` / filename from config (or ``default_name``).

    ``train.checkpoint_name`` or ``name_override`` wins when set. A bare stem
    gets ``.pt`` appended. Creates the directory if needed.
    """
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    if not ckpt_dir.is_absolute():
        ckpt_dir = repo_root() / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    raw = name_override if name_override else train_cfg.get("checkpoint_name")
    name = str(raw).strip() if raw else default_name
    if not name:
        name = default_name
    if not name.endswith(".pt"):
        name = f"{name}.pt"
    return ckpt_dir / name

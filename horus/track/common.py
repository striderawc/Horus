"""Shared helpers for stage-1 trackers."""

from __future__ import annotations

from typing import Any

import numpy as np


def dt_from_cfg(cfg: dict[str, Any]) -> float:
    """Resolve tracker ``dt``: ``tracker.dt`` or ``1 / source.fps``."""
    tcfg = cfg.get("tracker") or {}
    dt = tcfg.get("dt")
    if dt is not None:
        return float(dt)
    fps = float((cfg.get("source") or {}).get("fps", 30))
    if fps <= 0:
        raise ValueError(f"source.fps must be positive, got {fps}")
    return 1.0 / fps


def ensure_spd(
    mat: np.ndarray,
    *,
    min_eig: float,
    max_eig: float | None = None,
) -> np.ndarray:
    """Symmetrize and clamp eigenvalues so the matrix stays SPD."""
    mat = 0.5 * (mat + mat.T)
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, min_eig, max_eig)
    return (vecs * vals) @ vecs.T

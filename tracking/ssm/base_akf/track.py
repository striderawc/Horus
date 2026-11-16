"""Measurement loop around a predict / update SSM."""

from __future__ import annotations

from typing import Protocol

import numpy as np


def gate(
    mean: np.ndarray,
    cov: np.ndarray,
    n_sigma: float,
) -> tuple[float, float, float, float]:
    """Axis-aligned uncertainty frame around the predicted position."""
    half = n_sigma * np.sqrt(np.maximum(np.diag(cov)[:2], 0.0))
    x, y = float(mean[0]), float(mean[1])
    return (x - half[0], y - half[1], x + half[0], y + half[1])


def inside(z: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    x0, y0, x1, y1 = box
    return x0 <= float(z[0]) <= x1 and y0 <= float(z[1]) <= y1


def track(
    ssm,
    measurements: list[np.ndarray | None],
    *,
    gate_sigma: float = 3.0,
) -> list[dict | None]:
    """Run one clip: predict → gate → update (only if ``y`` is inside the frame).

    Each item is ``{pred, post, box, y}`` or ``None`` before the first hit.
    ``y`` is the measurement actually used (``None`` if missing or gated out).
    ``pred`` is ``None`` on the frame that starts the track.
    """
    out: list[dict | None] = []
    for raw in measurements:
        z = None if raw is None else np.asarray(raw, dtype=np.float64)

        if ssm.mean is None:
            if z is None:
                out.append(None)
                continue
            ssm.initiate(z)
            box = gate(ssm.mean, ssm.cov, gate_sigma)
            out.append({"pred": None, "post": ssm.mean[:2].copy(), "box": box, "z": z})
            continue

        ssm.predict()
        pred = ssm.mean[:2].copy()
        box = gate(ssm.mean, ssm.cov, gate_sigma)
        used = z if (z is not None and inside(z, box)) else None
        ssm.update(used)
        out.append({"pred": pred, "post": ssm.mean[:2].copy(), "box": box, "z": used})
    return out

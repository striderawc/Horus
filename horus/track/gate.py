"""Axis-aligned uncertainty gates and square crop windows for detectors."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from horus.track.state import TrackState


def gate_window(
    state: TrackState, n_sigma: float = 3.0
) -> tuple[float, float, float, float]:
    """Axis-aligned search box from position mean ± ``n_sigma`` std.

    Uses marginal position variances ``cov[0,0]`` / ``cov[1,1]``.
    Returns ``(x0, y0, x1, y1)`` in image coordinates.
    """
    x, y = float(state.mean[0]), float(state.mean[1])
    sx = n_sigma * float(np.sqrt(max(state.cov[0, 0], 0.0)))
    sy = n_sigma * float(np.sqrt(max(state.cov[1, 1], 0.0)))
    return (x - sx, y - sy, x + sx, y + sy)


def snap_square_window(
    cx: float,
    cy: float,
    side: float,
    buckets: Sequence[int],
    img_w: int,
    img_h: int,
) -> tuple[int, int, int]:
    """Snap a square window to a bucket size, centered on ``(cx, cy)``.

    ``side`` is the desired full side length (e.g. ``max(gate_w, gate_h)``).
    Picks the smallest bucket ≥ ``side``, or the largest bucket if none fit.
    Clamps the top-left so the crop stays inside ``[0, img_w) × [0, img_h)``
    when possible; if ``size`` exceeds an image dim, origin is 0 on that axis
    (caller should pad via ``crop_chw_stack``).

    Returns ``(x0, y0, size)`` as integers.
    """
    if img_w < 1 or img_h < 1:
        raise ValueError(f"img_w/img_h must be >= 1, got {img_w}x{img_h}")
    buckets_sorted = sorted(int(b) for b in buckets)
    if not buckets_sorted or any(b < 1 for b in buckets_sorted):
        raise ValueError(f"buckets must be positive ints, got {list(buckets)}")

    side = float(max(side, 1.0))
    size = buckets_sorted[-1]
    for b in buckets_sorted:
        if b >= side:
            size = b
            break

    # Ideal top-left for a centered crop.
    x0 = int(round(cx - 0.5 * size))
    y0 = int(round(cy - 0.5 * size))

    # Clamp into the image when the crop fits; else pin to 0.
    if size <= img_w:
        x0 = max(0, min(x0, img_w - size))
    else:
        x0 = 0
    if size <= img_h:
        y0 = max(0, min(y0, img_h - size))
    else:
        y0 = 0
    return x0, y0, size


def gate_to_square_crop(
    state: TrackState,
    *,
    buckets: Sequence[int],
    img_w: int,
    img_h: int,
    n_sigma: float = 3.0,
) -> tuple[int, int, int]:
    """``gate_window`` → square bucket crop ``(x0, y0, size)``."""
    x0, y0, x1, y1 = gate_window(state, n_sigma=n_sigma)
    side = max(x1 - x0, y1 - y0)
    cx, cy = float(state.mean[0]), float(state.mean[1])
    return snap_square_window(cx, cy, side, buckets, img_w, img_h)

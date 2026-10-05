"""Spatial crop helpers for track-gated and training jitter crops."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def crop_to_fullframe(
    xy_crop: np.ndarray | torch.Tensor,
    x0: float,
    y0: float,
) -> np.ndarray | torch.Tensor:
    """Map crop-local ``(x, y)`` to full-frame coordinates."""
    if isinstance(xy_crop, torch.Tensor):
        offset = xy_crop.new_tensor([float(x0), float(y0)])
        return xy_crop + offset
    xy = np.asarray(xy_crop, dtype=np.float64)
    return xy + np.array([float(x0), float(y0)], dtype=xy.dtype)


def fullframe_to_crop(
    xy_full: np.ndarray | torch.Tensor,
    x0: float,
    y0: float,
) -> np.ndarray | torch.Tensor:
    """Map full-frame ``(x, y)`` to crop-local coordinates."""
    if isinstance(xy_full, torch.Tensor):
        offset = xy_full.new_tensor([float(x0), float(y0)])
        return xy_full - offset
    xy = np.asarray(xy_full, dtype=np.float64)
    return xy - np.array([float(x0), float(y0)], dtype=xy.dtype)


def crop_chw_stack(
    x: torch.Tensor | np.ndarray,
    x0: int,
    y0: int,
    size: int,
    *,
    pad_value: float = 0.0,
) -> torch.Tensor | np.ndarray:
    """Crop ``(C, H, W)`` (or ``(H, W)``) to ``(..., size, size)`` with edge pad.

    If the requested window extends past the image, missing pixels are filled
    with ``pad_value`` (constant pad).
    """
    is_torch = isinstance(x, torch.Tensor)
    if is_torch:
        tensor = x
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
            squeeze = True
        elif tensor.ndim == 3:
            squeeze = False
        else:
            raise ValueError(f"expected (C,H,W) or (H,W), got {tuple(tensor.shape)}")
        _, h, w = tensor.shape
        x1, y1 = x0 + size, y0 + size
        # Overlap with the valid image region.
        src_x0 = max(0, x0)
        src_y0 = max(0, y0)
        src_x1 = min(w, x1)
        src_y1 = min(h, y1)
        out = tensor.new_full((tensor.shape[0], size, size), float(pad_value))
        if src_x0 < src_x1 and src_y0 < src_y1:
            dst_x0 = src_x0 - x0
            dst_y0 = src_y0 - y0
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            dst_y1 = dst_y0 + (src_y1 - src_y0)
            out[:, dst_y0:dst_y1, dst_x0:dst_x1] = tensor[
                :, src_y0:src_y1, src_x0:src_x1
            ]
        return out.squeeze(0) if squeeze else out

    arr = np.asarray(x)
    if arr.ndim == 2:
        arr = arr[None, ...]
        squeeze = True
    elif arr.ndim == 3:
        squeeze = False
    else:
        raise ValueError(f"expected (C,H,W) or (H,W), got {arr.shape}")
    _, h, w = arr.shape
    x1, y1 = x0 + size, y0 + size
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)
    out = np.full((arr.shape[0], size, size), pad_value, dtype=arr.dtype)
    if src_x0 < src_x1 and src_y0 < src_y1:
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        out[:, dst_y0:dst_y1, dst_x0:dst_x1] = arr[
            :, src_y0:src_y1, src_x0:src_x1
        ]
    return out[0] if squeeze else out


def ball_in_crop(
    bx: float,
    by: float,
    x0: int,
    y0: int,
    size: int,
    *,
    margin: float = 0.0,
) -> bool:
    """True if ball ``(bx, by)`` lies inside the crop with ``margin`` inset."""
    m = float(margin)
    return (
        (x0 + m) <= bx <= (x0 + size - 1 - m)
        and (y0 + m) <= by <= (y0 + size - 1 - m)
    )


def sample_positive_crop(
    bx: float,
    by: float,
    size: int,
    img_w: int,
    img_h: int,
    rng: np.random.Generator,
    *,
    jitter_std_frac: float = 0.15,
    margin: float = 4.0,
    max_tries: int = 32,
) -> tuple[int, int, int, bool]:
    """Jittered crop around the ball. Returns ``(x0, y0, size, in_crop)``."""
    from horus.track.gate import snap_square_window

    std = max(float(jitter_std_frac) * size, 0.0)
    for _ in range(max_tries):
        jx = float(rng.normal(0.0, std)) if std > 0 else 0.0
        jy = float(rng.normal(0.0, std)) if std > 0 else 0.0
        x0, y0, sz = snap_square_window(
            bx + jx, by + jy, float(size), [size], img_w, img_h
        )
        if ball_in_crop(bx, by, x0, y0, sz, margin=margin):
            return x0, y0, sz, True
    # Fall back: center exactly on the ball (may still fail near borders).
    x0, y0, sz = snap_square_window(bx, by, float(size), [size], img_w, img_h)
    return x0, y0, sz, ball_in_crop(bx, by, x0, y0, sz, margin=margin)


def sample_negative_crop(
    bx: float | None,
    by: float | None,
    size: int,
    img_w: int,
    img_h: int,
    rng: np.random.Generator,
    *,
    margin: float = 4.0,
    max_tries: int = 64,
) -> tuple[int, int, int]:
    """Random crop that does not contain the ball (if ball coords given)."""
    from horus.track.gate import snap_square_window

    for _ in range(max_tries):
        cx = float(rng.uniform(0, max(img_w - 1, 1)))
        cy = float(rng.uniform(0, max(img_h - 1, 1)))
        x0, y0, sz = snap_square_window(cx, cy, float(size), [size], img_w, img_h)
        if bx is None or by is None:
            return x0, y0, sz
        if not ball_in_crop(bx, by, x0, y0, sz, margin=margin):
            return x0, y0, sz
    # Last resort: pin to a corner away from the ball.
    corners = [(0, 0), (max(0, img_w - size), 0), (0, max(0, img_h - size)),
               (max(0, img_w - size), max(0, img_h - size))]
    if bx is not None and by is not None:
        best = max(
            corners,
            key=lambda c: (c[0] + size / 2 - bx) ** 2 + (c[1] + size / 2 - by) ** 2,
        )
        return best[0], best[1], size
    return corners[0][0], corners[0][1], size


def pick_bucket(buckets: Sequence[int], rng: np.random.Generator) -> int:
    """Uniformly sample a crop bucket size."""
    buckets_list = [int(b) for b in buckets]
    if not buckets_list:
        raise ValueError("buckets must be non-empty")
    return int(buckets_list[int(rng.integers(0, len(buckets_list)))])

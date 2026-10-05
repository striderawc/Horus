"""Gaussian / circle heatmap and coordinate targets."""

from __future__ import annotations

import numpy as np
import torch


def scale_coords(
    x: float,
    y: float,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> tuple[float, float]:
    """Scale pixel coordinates from source to training resolution."""
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    return x * sx, y * sy


def make_gaussian_heatmap(
    x: float,
    y: float,
    height: int,
    width: int,
    sigma: float = 2.5,
    amplitude: float = 1.0,
    visible: bool = True,
) -> np.ndarray:
    """Return ``(H, W)`` float32 Gaussian heatmap (zeros if not visible)."""
    heat = np.zeros((height, width), dtype=np.float32)
    if not visible:
        return heat
    # Clamp center into image for drawing; caller may still keep raw coords
    cx = float(x)
    cy = float(y)
    if cx < 0 or cy < 0 or cx >= width or cy >= height:
        # Still draw if slightly out of bounds but near edge
        if cx < -3 * sigma or cy < -3 * sigma or cx >= width + 3 * sigma or cy >= height + 3 * sigma:
            return heat

    radius = int(max(1, round(3 * sigma)))
    x0 = max(0, int(np.floor(cx - radius)))
    x1 = min(width, int(np.ceil(cx + radius)) + 1)
    y0 = max(0, int(np.floor(cy - radius)))
    y1 = min(height, int(np.ceil(cy + radius)) + 1)
    if x0 >= x1 or y0 >= y1:
        return heat

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    heat[y0:y1, x0:x1] = amplitude * np.exp(
        -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)
    )
    return heat


def make_circle_heatmap(
    x: float,
    y: float,
    height: int,
    width: int,
    radius: float = 3.0,
    visible: bool = True,
) -> np.ndarray:
    """Return ``(H, W)`` binary disk of 1s around ``(x, y)`` (zeros if not visible)."""
    heat = np.zeros((height, width), dtype=np.float32)
    if not visible:
        return heat
    cx = float(x)
    cy = float(y)
    r = float(max(radius, 0.0))
    if r <= 0.0:
        ix, iy = int(round(cx)), int(round(cy))
        if 0 <= ix < width and 0 <= iy < height:
            heat[iy, ix] = 1.0
        return heat
    if cx < -r or cy < -r or cx >= width + r or cy >= height + r:
        return heat

    x0 = max(0, int(np.floor(cx - r)))
    x1 = min(width, int(np.ceil(cx + r)) + 1)
    y0 = max(0, int(np.floor(cy - r)))
    y1 = min(height, int(np.ceil(cy + r)) + 1)
    if x0 >= x1 or y0 >= y1:
        return heat

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    heat[y0:y1, x0:x1][mask] = 1.0
    return heat


def build_circle_targets(
    coords: torch.Tensor,
    visible: torch.Tensor | None,
    height: int,
    width: int,
    radius: float = 3.0,
) -> torch.Tensor:
    """Batch of circle heatmaps ``(B, 1, H, W)`` from pixel ``coords`` ``(B, 2)``."""
    b = int(coords.shape[0])
    out = torch.zeros(b, 1, height, width, dtype=torch.float32, device=coords.device)
    vis = None if visible is None else visible.reshape(-1)
    for i in range(b):
        is_vis = True if vis is None else bool(vis[i].item() > 0.5)
        disk = make_circle_heatmap(
            float(coords[i, 0].item()),
            float(coords[i, 1].item()),
            height=height,
            width=width,
            radius=radius,
            visible=is_vis,
        )
        out[i, 0] = torch.from_numpy(disk).to(device=coords.device)
    return out


def build_targets(
    x: float,
    y: float,
    visibility: int,
    status: int,
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
    sigma: float,
    amplitude: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Build heatmap + coords tensors at training resolution."""
    sx, sy = scale_coords(x, y, src_w, src_h, dst_w, dst_h)
    visible = visibility != 0
    heat = make_gaussian_heatmap(
        sx,
        sy,
        height=dst_h,
        width=dst_w,
        sigma=sigma,
        amplitude=amplitude,
        visible=visible,
    )
    return {
        "heatmap": torch.from_numpy(heat).unsqueeze(0),  # (1, H, W)
        "coords": torch.tensor([sx, sy], dtype=torch.float32),
        "visibility": torch.tensor(visibility, dtype=torch.int64),
        "status": torch.tensor(status, dtype=torch.int64),
        "visible": torch.tensor(1 if visible else 0, dtype=torch.float32),
    }

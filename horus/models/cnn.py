"""2D encoder–decoder CNN: 3 RGB frames → Gaussian ball heatmap."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class BallHeatmapCNN(nn.Module):
    """TrackNet-lite: 9-channel input (3 RGB frames concat) → 1-channel heatmap."""

    IN_CH = 9  # 3 frames × 3 RGB

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        b, b2, b4 = base, base * 2, base * 4
        self.down1 = _block(self.IN_CH, b)
        self.down2 = _block(b, b2)
        self.down3 = _block(b2, b4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _block(b4, b4)
        self.up3 = _block(b4, b2)
        self.up2 = _block(b2, b)
        self.up1 = _block(b, b)
        self.head = nn.Conv2d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 9, H, W)
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        x = self.bottleneck(self.pool(d3))
        x = self.up3(F.interpolate(x, size=d3.shape[-2:], mode="bilinear", align_corners=False))
        x = self.up2(F.interpolate(x, size=d2.shape[-2:], mode="bilinear", align_corners=False))
        x = self.up1(F.interpolate(x, size=d1.shape[-2:], mode="bilinear", align_corners=False))
        return torch.sigmoid(self.head(x))


def heatmap_to_coords(heat: torch.Tensor) -> torch.Tensor:
    """Argmax peak of ``(B, 1, H, W)`` → ``(B, 2)`` as (x, y)."""
    b, _, h, w = heat.shape
    flat = heat.view(b, -1)
    idx = flat.argmax(dim=1)
    y = (idx // w).float()
    x = (idx % w).float()
    return torch.stack([x, y], dim=1)


def weighted_heatmap_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 100.0,
) -> torch.Tensor:
    """MSE with higher weight on ball pixels (target intensity).

    Background stays weight 1; peak cells scale up to ``pos_weight``. Stops the
    easy all-zeros solution that tanks unweighted MSE while leaving val_px bad.
    """
    w = 1.0 + (pos_weight - 1.0) * target
    return (w * (pred - target).pow(2)).mean()

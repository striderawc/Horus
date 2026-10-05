"""2D encoder–decoder CNN: 3 RGB frames → Gaussian ball heatmap."""

from __future__ import annotations

import math

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
    """TrackNet-lite encoder–decoder → 1-channel Gaussian heatmap.

    ``in_ch``: 9 RGB, 2 frame-diff, 6 event-frame (3×2 polarity).

    With ``predict_uncertainty=True``, ``forward`` returns
    ``(heatmap, log_sigma)`` where ``log_sigma`` is ``(B,)`` from a GAP head
    on the bottleneck (isotropic measurement noise for the tracker).
    """

    def __init__(
        self,
        in_ch: int = 9,
        base: int = 16,
        *,
        predict_uncertainty: bool = False,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.predict_uncertainty = bool(predict_uncertainty)
        b, b2, b4 = base, base * 2, base * 4
        self.down1 = _block(in_ch, b)
        self.down2 = _block(b, b2)
        self.down3 = _block(b2, b4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _block(b4, b4)
        self.up3 = _block(b4, b2)
        self.up2 = _block(b2, b)
        self.up1 = _block(b, b)
        self.head = nn.Conv2d(b, 1, 1)
        if self.predict_uncertainty:
            self.sigma_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(b4, 1),
            )
            # Init near log(5) ≈ typical tracker.r_meas
            nn.init.constant_(self.sigma_head[-1].bias, 1.6)
        else:
            self.sigma_head = None

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (B, in_ch, H, W)
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        bn = self.bottleneck(self.pool(d3))
        x = self.up3(
            F.interpolate(bn, size=d3.shape[-2:], mode="bilinear", align_corners=False)
        )
        x = self.up2(
            F.interpolate(x, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        )
        x = self.up1(
            F.interpolate(x, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        )
        heat = torch.sigmoid(self.head(x))
        if self.sigma_head is None:
            return heat
        log_sigma = self.sigma_head(bn).squeeze(-1)
        return heat, log_sigma


def unpack_model_output(
    out: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize model forward to ``(heatmap, log_sigma|None)``."""
    if isinstance(out, tuple):
        heat, log_sigma = out
        return heat, log_sigma
    return out, None


def gaussian_nll_loss(
    pred: torch.Tensor,
    coords: torch.Tensor,
    log_sigma: torch.Tensor,
    visible: torch.Tensor | None = None,
    *,
    height: int | None = None,
    width: int | None = None,
    temperature: float = 0.1,
    sigma_min: float = 0.5,
    sigma_max: float = 64.0,
) -> torch.Tensor:
    """Gaussian NLL on heatmap peak vs GT coords (positives only).

    ``0.5 * ((‖xŷ−xy‖/σ)² + 2 log σ)`` with σ = clamp(exp(log_σ)).
    """
    if pred.ndim == 4 and height is None and width is None:
        height, width = int(pred.shape[-2]), int(pred.shape[-1])
    if height is None or width is None:
        raise ValueError("height and width are required for gaussian_nll_loss")
    pred_xy = pred_to_pixel_coords(
        pred, height=height, width=width, temperature=temperature
    )
    log_s = log_sigma.reshape(-1)
    if visible is not None:
        mask = visible.reshape(-1) > 0.5
        if not mask.any():
            # Differentiable zero so all-negative batches still backward.
            return pred.sum() * 0.0 + log_sigma.sum() * 0.0
        pred_xy = pred_xy[mask]
        coords = coords[mask]
        log_s = log_s[mask]
    target = coords.to(device=pred_xy.device, dtype=pred_xy.dtype)
    # Clamp log_sigma in the loss for stability (still grads to raw head).
    log_s = log_s.clamp(min=math.log(sigma_min), max=math.log(sigma_max))
    residual = (pred_xy - target).norm(dim=1)
    inv_var = torch.exp(-2.0 * log_s)
    return (0.5 * (residual.pow(2) * inv_var + 2.0 * log_s)).mean()


def heatmap_to_coords(heat: torch.Tensor) -> torch.Tensor:
    """Argmax peak of ``(B, 1, H, W)`` → ``(B, 2)`` as (x, y)."""
    b, _, h, w = heat.shape
    flat = heat.view(b, -1)
    idx = flat.argmax(dim=1)
    y = (idx // w).float()
    x = (idx % w).float()
    return torch.stack([x, y], dim=1)


def soft_heatmap_to_coords(
    heat: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Differentiable soft-argmax of ``(B, 1, H, W)`` → ``(B, 2)`` as (x, y).

    Spatial softmax over the heatmap (scaled by ``1/temperature``), then
    expectation over pixel coordinates. Smaller ``temperature`` → sharper peak.
    """
    b, _, h, w = heat.shape
    flat = heat.reshape(b, -1)
    weights = torch.softmax(flat / max(float(temperature), 1e-6), dim=1)
    device, dtype = heat.device, heat.dtype
    xs = torch.arange(w, device=device, dtype=dtype)
    ys = torch.arange(h, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.reshape(-1)
    grid_y = grid_y.reshape(-1)
    x = (weights * grid_x).sum(dim=1)
    y = (weights * grid_y).sum(dim=1)
    return torch.stack([x, y], dim=1)


def pred_to_pixel_coords(
    pred: torch.Tensor,
    *,
    height: int | None = None,
    width: int | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Map model output to pixel ``(x, y)``.

    - Heatmap ``(B, 1, H, W)``: hard-argmax with soft-argmax STE for grads.
    - Normalized coords ``(B, 2)`` in ``[0, 1]``: scale by ``(W-1, H-1)``.
    """
    if pred.ndim == 2 and pred.shape[-1] == 2:
        if height is None or width is None:
            raise ValueError("height and width are required for normalized (B, 2) preds")
        scale = pred.new_tensor([float(width - 1), float(height - 1)])
        return pred * scale
    if pred.ndim == 4 and pred.shape[1] == 1:
        soft_xy = soft_heatmap_to_coords(pred, temperature=temperature)
        hard_xy = heatmap_to_coords(pred)
        # STE: forward = hard argmax, backward = soft-argmax.
        return hard_xy + (soft_xy - soft_xy.detach())
    raise ValueError(
        f"expected heatmap (B,1,H,W) or normalized coords (B,2), got {tuple(pred.shape)}"
    )


def coord_distance_loss(
    pred: torch.Tensor,
    coords: torch.Tensor,
    visible: torch.Tensor | None = None,
    *,
    height: int | None = None,
    width: int | None = None,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Mean Euclidean error / image diagonal (so a corner-to-corner miss is ~1).

    Pixel distance is divided by ``sqrt((W-1)^2 + (H-1)^2)`` so grads are
    resolution-independent. ``val_px`` stays in raw pixels.

    For heatmaps, peak is hard-argmax (STE soft grads). For normalized
    ``(B, 2)`` outputs, pass input ``height`` / ``width`` to denormalize.

    Invisible samples (``visible <= 0.5``) are skipped; if none remain, returns 0.
    """
    if pred.ndim == 4 and height is None and width is None:
        height, width = int(pred.shape[-2]), int(pred.shape[-1])
    if height is None or width is None:
        raise ValueError("height and width are required for coord_distance_loss")
    pred_xy = pred_to_pixel_coords(
        pred, height=height, width=width, temperature=temperature
    )

    if visible is not None:
        mask = visible.reshape(-1) > 0.5
        if not mask.any():
            # Differentiable zero so all-negative batches still backward.
            return pred.sum() * 0.0
        pred_xy = pred_xy[mask]
        coords = coords[mask]
    target = coords.to(device=pred_xy.device, dtype=pred_xy.dtype)
    denom = float(((width - 1) ** 2 + (height - 1) ** 2) ** 0.5)
    denom = max(denom, 1.0)
    return ((pred_xy - target).norm(dim=1) / denom).mean()


def weighted_heatmap_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = 100.0,
) -> torch.Tensor:
    """MSE with higher weight on ball pixels (target intensity).

    Background stays weight 1; peak cells scale up to ``pos_weight``. Kept for
    ablations; training now uses ``coord_distance_loss`` by default.
    """
    w = 1.0 + (pos_weight - 1.0) * target
    return (w * (pred - target).pow(2)).mean()


def _circle_target(
    pred: torch.Tensor,
    coords: torch.Tensor,
    visible: torch.Tensor | None,
    radius: float,
) -> torch.Tensor:
    """Binary disk heatmap ``(B, 1, H, W)`` matching ``pred`` spatial size."""
    if pred.ndim != 4 or pred.shape[1] != 1:
        raise ValueError(f"expected heatmap (B,1,H,W), got {tuple(pred.shape)}")
    _, _, height, width = pred.shape
    from horus.data.targets import build_circle_targets

    return build_circle_targets(
        coords, visible, height=height, width=width, radius=radius
    ).to(device=pred.device, dtype=pred.dtype)


def circle_mse_loss(
    pred: torch.Tensor,
    coords: torch.Tensor,
    visible: torch.Tensor | None = None,
    *,
    radius: float = 3.0,
) -> torch.Tensor:
    """MSE of sigmoid heatmap ``(B, 1, H, W)`` vs a disk of 1s around each ball.

    Invisible samples get an all-zero target (still supervised). Builds the
    circle from pixel ``coords`` so the dataset can keep Gaussian heatmaps for
    the CNN path.
    """
    target = _circle_target(pred, coords, visible, radius)
    return torch.nn.functional.mse_loss(pred, target)


def circle_bce_loss(
    pred: torch.Tensor,
    coords: torch.Tensor,
    visible: torch.Tensor | None = None,
    *,
    radius: float = 3.0,
) -> torch.Tensor:
    """BCE of sigmoid heatmap ``(B, 1, H, W)`` vs a disk of 1s around each ball.

    Invisible samples get an all-zero target (still supervised). Kept for
    ablations; SNN training defaults to ``circle_mse_loss``.
    """
    target = _circle_target(pred, coords, visible, radius)
    return torch.nn.functional.binary_cross_entropy(pred, target)

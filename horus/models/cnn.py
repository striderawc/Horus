"""2D encoder–decoder CNN: sequence → stride-1 ball heatmap + presence."""

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
    """TrackNet-lite encoder–decoder → heatmap logits + presence logit.

    ``in_ch``: 9 RGB, 2 frame-diff, 6 event-frame (3×2 polarity).

    Forward returns ``(heatmap_logits, presence_logit)`` with shapes
    ``(B, 1, H, W)`` and ``(B,)``. Apply sigmoid outside for viz / focal.
    """

    def __init__(
        self,
        in_ch: int = 9,
        base: int = 16,
        *,
        predict_presence: bool = True,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.predict_presence = bool(predict_presence)
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
        if self.predict_presence:
            self.presence_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(b4, 1),
            )
        else:
            self.presence_head = None

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
        heat_logits = self.head(x)
        if self.presence_head is None:
            return heat_logits
        presence = self.presence_head(bn).squeeze(-1)
        return heat_logits, presence


def unpack_model_output(
    out: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize model forward to ``(heatmap_logits, presence_logit|None)``."""
    if isinstance(out, tuple):
        heat, presence = out
        return heat, presence
    return out, None


def penalty_reduced_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """CornerNet / CenterNet penalty-reduced focal loss on a Gaussian heatmap.

    ``logits`` / ``target``: ``(B, 1, H, W)`` with ``target ∈ [0, 1]``.
    Normalized by the number of positive peaks (``target == 1``), or by batch
    size when there are no positives (all-negative / invisible crops).
    """
    if logits.shape != target.shape:
        raise ValueError(
            f"logits/target shape mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}"
        )
    pred = torch.sigmoid(logits).clamp(eps, 1.0 - eps)
    pos_mask = target.eq(1.0)
    neg_mask = ~pos_mask

    pos_loss = -((1.0 - pred) ** alpha) * torch.log(pred)
    neg_weights = (1.0 - target) ** beta
    neg_loss = -(neg_weights) * (pred**alpha) * torch.log(1.0 - pred)

    n_pos = int(pos_mask.sum().item())
    loss = pos_loss * pos_mask.float() + neg_loss * neg_mask.float()
    if n_pos > 0:
        return loss.sum() / float(n_pos)
    return loss.sum() / float(max(logits.shape[0], 1))


def presence_bce_loss(
    presence_logit: torch.Tensor,
    present: torch.Tensor,
) -> torch.Tensor:
    """BCE-with-logits for ball-in-crop / visible presence."""
    target = present.reshape(-1).to(
        device=presence_logit.device, dtype=presence_logit.dtype
    )
    logit = presence_logit.reshape(-1)
    return F.binary_cross_entropy_with_logits(logit, target)


def focal_presence_loss(
    heat_logits: torch.Tensor,
    presence_logit: torch.Tensor,
    heatmap_target: torch.Tensor,
    present: torch.Tensor,
    *,
    alpha: float = 2.0,
    beta: float = 4.0,
    presence_weight: float = 1.0,
) -> torch.Tensor:
    """``L_hm + λ * L_presence``."""
    hm = penalty_reduced_focal_loss(
        heat_logits, heatmap_target, alpha=alpha, beta=beta
    )
    pr = presence_bce_loss(presence_logit, present)
    return hm + float(presence_weight) * pr


def heatmap_to_coords(heat: torch.Tensor) -> torch.Tensor:
    """Argmax peak of ``(B, 1, H, W)`` → ``(B, 2)`` as (x, y).

    Works on logits or probabilities (argmax is unchanged by sigmoid).
    """
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


def soft_heatmap_spatial_std(
    heat: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Isotropic soft spatial std ``(B,)`` from softmax mass (peak sharpness).

    Sharper peaks → smaller std. ``heat`` may be logits or probabilities.
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
    mean_x = (weights * grid_x).sum(dim=1)
    mean_y = (weights * grid_y).sum(dim=1)
    var_x = (weights * (grid_x - mean_x[:, None]).pow(2)).sum(dim=1)
    var_y = (weights * (grid_y - mean_y[:, None]).pow(2)).sum(dim=1)
    return torch.sqrt(0.5 * (var_x + var_y) + 1e-8)


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
    """Mean Euclidean error / image diagonal (ablation).

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
    """MSE with higher weight on ball pixels (ablation)."""
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
    """MSE of sigmoid heatmap vs a disk (ablation). Accepts logits or probs."""
    probs = torch.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred
    target = _circle_target(probs, coords, visible, radius)
    return torch.nn.functional.mse_loss(probs, target)


def circle_bce_loss(
    pred: torch.Tensor,
    coords: torch.Tensor,
    visible: torch.Tensor | None = None,
    *,
    radius: float = 3.0,
) -> torch.Tensor:
    """BCE of sigmoid heatmap vs a disk (ablation). Accepts logits or probs."""
    probs = torch.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred
    target = _circle_target(probs, coords, visible, radius)
    return torch.nn.functional.binary_cross_entropy(probs, target)

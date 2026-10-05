"""Shared heatmap / coord training loop for CNN and SNN models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from horus.data.targets import build_circle_targets
from horus.eval.metrics import visible_pixel_distances
from horus.models import (
    circle_bce_loss,
    circle_mse_loss,
    coord_distance_loss,
    gaussian_nll_loss,
    unpack_model_output,
)
from horus.train.tb import log_heatmap_images, log_scalars

LossKind = Literal["coord", "circle_mse", "circle_bce"]


def _batch_loss(
    pred: torch.Tensor,
    coords: torch.Tensor,
    visible: torch.Tensor,
    *,
    height: int,
    width: int,
    loss: LossKind = "coord",
    circle_radius: float = 3.0,
    log_sigma: torch.Tensor | None = None,
    nll_weight: float = 0.0,
) -> torch.Tensor:
    if loss == "circle_mse":
        base = circle_mse_loss(pred, coords, visible, radius=circle_radius)
    elif loss == "circle_bce":
        base = circle_bce_loss(pred, coords, visible, radius=circle_radius)
    else:
        # Positives: coord peak loss. Negatives: push heatmap toward empty
        # (coord loss alone skips them and can yield a non-differentiable 0).
        base = coord_distance_loss(pred, coords, visible, height=height, width=width)
        neg = visible.reshape(-1) <= 0.5
        if neg.any():
            base = base + pred[neg].pow(2).mean()
    if log_sigma is not None and nll_weight > 0:
        nll = gaussian_nll_loss(
            pred, coords, log_sigma, visible, height=height, width=width
        )
        return base + float(nll_weight) * nll
    return base


@dataclass(frozen=True)
class TrainEpochStats:
    """Aggregates from one training epoch."""

    loss: float
    grad_norm: float
    steps: int
    # Mean L2 grad norm per top-level module (e.g. down1, head).
    layer_grad_norms: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalEpochStats:
    """Aggregates from one eval pass (loss + localization accuracy)."""

    loss: float
    mean_px: float
    acc_at_5: float
    acc_at_10: float
    n_visible: int


def layer_grad_norms(model: nn.Module) -> dict[str, float]:
    """L2 grad norm per top-level module, from existing ``.grad`` only.

    Uses sum-of-squares → Python floats (no concatenated grad tensor), so this
    stays cheap even when the autograd graph for long SNN unrolls is large.
    """
    sq: dict[str, float] = {}
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        layer = name.split(".", 1)[0]
        # .item() immediately — do not retain GPU reductions across params.
        sq[layer] = sq.get(layer, 0.0) + float(grad.detach().pow(2).sum().item())
    return {layer: total**0.5 for layer, total in sq.items()}


def global_grad_norm(model: nn.Module) -> float:
    """L2 norm of all parameter gradients (0 if none)."""
    layers = layer_grad_norms(model)
    if not layers:
        return 0.0
    return float(sum(v * v for v in layers.values()) ** 0.5)


def _optimizer_lr(opt: torch.optim.Optimizer) -> float:
    return float(opt.param_groups[0]["lr"])


def _visible_mask(batch: dict) -> torch.Tensor:
    """Prefer ``in_crop`` when present (crop-mode positives)."""
    if "in_crop" in batch:
        return batch["in_crop"]
    return batch["visible"]


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    loss: LossKind = "coord",
    circle_radius: float = 3.0,
    nll_weight: float = 0.0,
) -> EvalEpochStats:
    """Mean training loss plus visible-ball localization metrics."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    dist_sum, dist_n = 0.0, 0
    hit5, hit10 = 0, 0
    for batch in loader:
        x = batch["input"].to(device)
        coords = batch["coords"].to(device)
        visible = _visible_mask(batch).to(device)
        _, _, h, w = x.shape
        heat, log_sigma = unpack_model_output(model(x))
        total_loss += _batch_loss(
            heat,
            coords,
            visible,
            height=h,
            width=w,
            loss=loss,
            circle_radius=circle_radius,
            log_sigma=log_sigma,
            nll_weight=nll_weight,
        ).item()
        n_batches += 1
        dists = visible_pixel_distances(
            heat, coords, visible, height=h, width=w
        )
        if dists.numel():
            dist_sum += float(dists.sum().item())
            n = int(dists.numel())
            dist_n += n
            hit5 += int((dists <= 5.0).sum().item())
            hit10 += int((dists <= 10.0).sum().item())
    loss_val = total_loss / max(n_batches, 1)
    if dist_n == 0:
        return EvalEpochStats(
            loss=loss_val, mean_px=float("nan"), acc_at_5=0.0, acc_at_10=0.0, n_visible=0
        )
    return EvalEpochStats(
        loss=loss_val,
        mean_px=dist_sum / dist_n,
        acc_at_5=hit5 / dist_n,
        acc_at_10=hit10 / dist_n,
        n_visible=dist_n,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    *,
    writer: SummaryWriter | None = None,
    global_step: int = 0,
    log_every: int = 50,
    loss: LossKind = "coord",
    circle_radius: float = 3.0,
    nll_weight: float = 0.0,
) -> tuple[TrainEpochStats, int]:
    """One pass over ``loader``. Returns epoch stats and updated ``global_step``."""
    model.train()
    total_loss, total_gn, n = 0.0, 0.0, 0
    layer_totals: dict[str, float] = {}
    log_every = max(int(log_every), 1)
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["input"].to(device)
        coords = batch["coords"].to(device)
        visible = _visible_mask(batch).to(device)
        _, _, h, w = x.shape
        opt.zero_grad(set_to_none=True)
        heat, log_sigma = unpack_model_output(model(x))
        batch_loss = _batch_loss(
            heat,
            coords,
            visible,
            height=h,
            width=w,
            loss=loss,
            circle_radius=circle_radius,
            log_sigma=log_sigma,
            nll_weight=nll_weight,
        )
        batch_loss.backward()
        layers = layer_grad_norms(model)
        gn = float(sum(v * v for v in layers.values()) ** 0.5) if layers else 0.0
        opt.step()

        total_loss += batch_loss.item()
        total_gn += gn
        for layer, value in layers.items():
            layer_totals[layer] = layer_totals.get(layer, 0.0) + value
        n += 1
        global_step += 1

        if writer is not None and (global_step % log_every == 0 or n == 1):
            step_scalars: dict[str, float] = {
                "train/loss_step": batch_loss.item(),
                "train/grad_norm_step": gn,
                "train/lr": _optimizer_lr(opt),
            }
            for layer, value in layers.items():
                step_scalars[f"train/grad_norm_layer/{layer}"] = value
            log_scalars(writer, step_scalars, global_step)

    denom = max(n, 1)
    stats = TrainEpochStats(
        loss=total_loss / denom,
        grad_norm=total_gn / denom,
        steps=n,
        layer_grad_norms={k: v / denom for k, v in layer_totals.items()},
    )
    return stats, global_step


def log_epoch_metrics(
    writer: SummaryWriter,
    epoch: int,
    train: TrainEpochStats,
    val: EvalEpochStats,
    *,
    lr: float,
    extra: Mapping[str, float] | None = None,
) -> None:
    """Epoch-level TensorBoard scalars (train + val + optional extras)."""
    scalars: dict[str, float] = {
        "train/loss": train.loss,
        "train/grad_norm": train.grad_norm,
        "train/lr_epoch": lr,
        "val/loss": val.loss,
        "val/mean_px": val.mean_px,
        "val/acc_at_5": val.acc_at_5,
        "val/acc_at_10": val.acc_at_10,
    }
    for layer, value in train.layer_grad_norms.items():
        scalars[f"train/grad_norm_layer/{layer}"] = float(value)
    if extra:
        scalars.update(extra)
    log_scalars(writer, scalars, epoch)


@torch.inference_mode()
def log_loader_heatmaps(
    writer: SummaryWriter,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    step: int,
    *,
    max_samples: int = 1,
    tag: str = "debug/heatmap_gt_pred_abserr",
    circle_radius: float | None = None,
) -> None:
    """Log GT | pred heatmap strips, or pred/GT coord scalars for coord heads.

    When ``circle_radius`` is set, GT is a binary disk built from ``coords``
    (SNN circle-BCE path) instead of the dataset Gaussian heatmap.
    """
    model.eval()
    batch = next(iter(loader))
    x = batch["input"].to(device)
    pred, _ = unpack_model_output(model(x))
    pred = pred.cpu()
    if pred.ndim == 2 and pred.shape[-1] == 2:
        # Direct (x, y) head — log normalized coords instead of heatmaps.
        coords = batch["coords"]
        _, _, h, w = batch["input"].shape
        scale = torch.tensor([float(w - 1), float(h - 1)])
        gt_norm = coords / scale
        n = min(int(max_samples), pred.shape[0])
        for i in range(n):
            prefix = "debug/coords" if n == 1 else f"debug/coords/sample{i}"
            writer.add_scalar(f"{prefix}/pred_x", float(pred[i, 0]), step)
            writer.add_scalar(f"{prefix}/pred_y", float(pred[i, 1]), step)
            writer.add_scalar(f"{prefix}/gt_x", float(gt_norm[i, 0]), step)
            writer.add_scalar(f"{prefix}/gt_y", float(gt_norm[i, 1]), step)
        return
    if circle_radius is not None:
        _, _, h, w = pred.shape
        y = build_circle_targets(
            batch["coords"],
            batch["visible"],
            height=h,
            width=w,
            radius=float(circle_radius),
        )
    else:
        y = batch["heatmap"]
    log_heatmap_images(
        writer, pred, y, step, tag=tag, max_samples=max_samples
    )

"""Shared heatmap training loop for CNN and SNN models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from horus.eval.metrics import visible_pixel_distances
from horus.models import weighted_heatmap_mse
from horus.train.tb import log_scalars


@dataclass(frozen=True)
class TrainEpochStats:
    """Aggregates from one training epoch."""

    loss: float
    grad_norm: float
    steps: int


@dataclass(frozen=True)
class EvalEpochStats:
    """Aggregates from one eval pass (loss + localization accuracy)."""

    loss: float
    mean_px: float
    acc_at_5: float
    acc_at_10: float
    n_visible: int


def global_grad_norm(model: nn.Module) -> float:
    """L2 norm of all parameter gradients (0 if none)."""
    grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.cat(grads).norm(2).item())


def _optimizer_lr(opt: torch.optim.Optimizer) -> float:
    return float(opt.param_groups[0]["lr"])


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: float,
) -> EvalEpochStats:
    """Mean weighted heatmap loss and visible-ball localization metrics."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    dist_sum, dist_n = 0.0, 0
    hit5, hit10 = 0, 0
    for batch in loader:
        x = batch["input"].to(device)
        y = batch["heatmap"].to(device)
        pred = model(x)
        total_loss += weighted_heatmap_mse(pred, y, pos_weight).item()
        n_batches += 1
        dists = visible_pixel_distances(
            pred, batch["coords"].to(device), batch["visible"].to(device)
        )
        if dists.numel():
            dist_sum += float(dists.sum().item())
            n = int(dists.numel())
            dist_n += n
            hit5 += int((dists <= 5.0).sum().item())
            hit10 += int((dists <= 10.0).sum().item())
    loss = total_loss / max(n_batches, 1)
    if dist_n == 0:
        return EvalEpochStats(
            loss=loss, mean_px=float("nan"), acc_at_5=0.0, acc_at_10=0.0, n_visible=0
        )
    return EvalEpochStats(
        loss=loss,
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
    pos_weight: float,
    *,
    writer: SummaryWriter | None = None,
    global_step: int = 0,
    log_every: int = 50,
) -> tuple[TrainEpochStats, int]:
    """One pass over ``loader``. Returns epoch stats and updated ``global_step``."""
    model.train()
    total_loss, total_gn, n = 0.0, 0.0, 0
    log_every = max(int(log_every), 1)
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["input"].to(device)
        y = batch["heatmap"].to(device)
        opt.zero_grad(set_to_none=True)
        loss = weighted_heatmap_mse(model(x), y, pos_weight)
        loss.backward()
        gn = global_grad_norm(model)
        opt.step()

        total_loss += loss.item()
        total_gn += gn
        n += 1
        global_step += 1

        if writer is not None and (global_step % log_every == 0 or n == 1):
            log_scalars(
                writer,
                {
                    "train/loss_step": loss.item(),
                    "train/grad_norm_step": gn,
                    "train/lr": _optimizer_lr(opt),
                },
                global_step,
            )

    stats = TrainEpochStats(
        loss=total_loss / max(n, 1),
        grad_norm=total_gn / max(n, 1),
        steps=n,
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
    if extra:
        scalars.update(extra)
    log_scalars(writer, scalars, epoch)

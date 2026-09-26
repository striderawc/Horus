"""Shared heatmap training loop for CNN and SNN models."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from horus.eval.metrics import visible_pixel_distances
from horus.models import weighted_heatmap_mse


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: float,
) -> tuple[float, float]:
    """Mean weighted heatmap loss and mean visible pixel error."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    dist_sum, dist_n = 0.0, 0
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
            dist_n += int(dists.numel())
    loss = total_loss / max(n_batches, 1)
    px = dist_sum / max(dist_n, 1)
    return loss, px


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: float,
) -> float:
    """One pass over ``loader``. Returns mean weighted heatmap loss."""
    model.train()
    total, n = 0.0, 0
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["input"].to(device)
        y = batch["heatmap"].to(device)
        opt.zero_grad(set_to_none=True)
        loss = weighted_heatmap_mse(model(x), y, pos_weight)
        loss.backward()
        opt.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)

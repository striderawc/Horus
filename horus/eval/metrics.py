"""Accuracy, FLOPs, and latency measurements for heatmap models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from horus.models import heatmap_to_coords


@dataclass(frozen=True)
class AccuracyResult:
    mean_px: float
    acc_at_5: float
    acc_at_10: float
    n_visible: int


def visible_pixel_distances(
    pred_heat: torch.Tensor,
    gt_coords: torch.Tensor,
    visible: torch.Tensor,
) -> torch.Tensor:
    """Euclidean pixel error for visible samples. Returns ``(N_vis,)``."""
    mask = visible > 0.5
    if not mask.any():
        return pred_heat.new_empty(0)
    pred_xy = heatmap_to_coords(pred_heat[mask])
    gt_xy = gt_coords[mask]
    return (pred_xy - gt_xy).norm(dim=1)


@torch.inference_mode()
def measure_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    thresholds: Sequence[float] = (5.0, 10.0),
) -> AccuracyResult:
    """Mean pixel error and Acc@k on visible frames over ``loader``."""
    model.eval()
    dist_sum = 0.0
    n_vis = 0
    # Support exactly the planned Acc@5 / Acc@10; extra thresholds ignored for result fields.
    thr = list(thresholds)
    t5 = float(thr[0]) if thr else 5.0
    t10 = float(thr[1]) if len(thr) > 1 else 10.0
    hit5 = 0
    hit10 = 0

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        pred = model(x)
        dists = visible_pixel_distances(
            pred,
            batch["coords"].to(device, non_blocking=True),
            batch["visible"].to(device, non_blocking=True),
        )
        if dists.numel() == 0:
            continue
        dist_sum += float(dists.sum().item())
        n = int(dists.numel())
        n_vis += n
        hit5 += int((dists <= t5).sum().item())
        hit10 += int((dists <= t10).sum().item())

    if n_vis == 0:
        return AccuracyResult(mean_px=float("nan"), acc_at_5=0.0, acc_at_10=0.0, n_visible=0)
    return AccuracyResult(
        mean_px=dist_sum / n_vis,
        acc_at_5=hit5 / n_vis,
        acc_at_10=hit10 / n_vis,
        n_visible=n_vis,
    )


def input_channels(modality: str, cfg: dict[str, Any]) -> int:
    """Channel count of a stacked sequence for ``modality``."""
    seq = int(cfg["sequence"]["length"])
    if modality == "rgb":
        return seq * 3
    if modality == "diff":
        return max(seq - 1, 1)
    if modality == "event_frame":
        return seq * 2
    if modality == "voxel":
        bins = int(cfg.get("events", {}).get("voxel", {}).get("num_bins", 5))
        return seq * bins
    raise ValueError(f"Unknown modality: {modality}")


def measure_flops(
    model: nn.Module,
    *,
    in_ch: int,
    height: int,
    width: int,
    device: torch.device,
) -> float | None:
    """Return FLOPs for one forward, or ``None`` if fvcore is unavailable."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return None

    model.eval()
    dummy = torch.zeros(1, in_ch, height, width, device=device)
    with torch.inference_mode():
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return float(flops.total())


@torch.inference_mode()
def measure_latency(
    model: nn.Module,
    *,
    in_ch: int,
    height: int,
    width: int,
    device: torch.device,
    warmup: int = 10,
    runs: int = 50,
) -> float:
    """Mean forward latency in milliseconds (batch size 1)."""
    model.eval()
    x = torch.zeros(1, in_ch, height, width, device=device)

    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    return (elapsed / max(runs, 1)) * 1000.0

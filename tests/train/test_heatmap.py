"""Unit tests for heatmap train helpers (no dataset required)."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from horus.train.heatmap import (
    EvalEpochStats,
    evaluate,
    global_grad_norm,
    train_one_epoch,
)
from horus.train.tb import resolve_log_dir


class _ToyHeatmapDS(Dataset):
    def __init__(self, n: int = 4, h: int = 8, w: int = 8) -> None:
        self.n = n
        self.h = h
        self.w = w

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        heat = torch.zeros(1, self.h, self.w)
        heat[0, 2, 3] = 1.0
        return {
            "input": torch.randn(2, self.h, self.w),
            "heatmap": heat,
            "coords": torch.tensor([3.0, 2.0]),
            "visible": torch.tensor(1.0),
        }


class _TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(x))


def test_global_grad_norm_zero_without_grads():
    model = _TinyCNN()
    assert global_grad_norm(model) == 0.0


def test_global_grad_norm_positive_after_backward():
    model = _TinyCNN()
    x = torch.randn(1, 2, 4, 4)
    y = torch.zeros(1, 1, 4, 4)
    y[0, 0, 1, 1] = 1.0
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    assert global_grad_norm(model) > 0.0


def test_train_one_epoch_and_evaluate_shapes():
    ds = _ToyHeatmapDS()
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    model = _TinyCNN()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    device = torch.device("cpu")

    train_stats, step = train_one_epoch(
        model, loader, opt, device, pos_weight=10.0, log_every=1
    )
    assert train_stats.steps == 2
    assert step == 2
    assert train_stats.loss >= 0.0
    assert train_stats.grad_norm >= 0.0

    val = evaluate(model, loader, device, pos_weight=10.0)
    assert isinstance(val, EvalEpochStats)
    assert val.n_visible == 4
    assert 0.0 <= val.acc_at_5 <= 1.0
    assert 0.0 <= val.acc_at_10 <= 1.0
    assert val.mean_px >= 0.0


def test_resolve_log_dir(tmp_path: Path):
    out = resolve_log_dir(tmp_path / "runs", "rgb_cnn", repo=tmp_path)
    assert out.is_dir()
    assert out.parent.name == "rgb_cnn"
    assert out.parent.parent.name == "runs"

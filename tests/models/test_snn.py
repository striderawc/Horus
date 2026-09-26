"""Shape and gradient checks for BallHeatmapSNN (no dataset required)."""

from __future__ import annotations

import torch

from horus.models import BallHeatmapSNN


def test_snn_forward_shape_and_backward():
    model = BallHeatmapSNN()
    model.train()
    x = torch.rand(2, 6, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)
    assert torch.isfinite(y).all()
    y.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads
    assert all(g is not None and torch.isfinite(g).all() for g in grads)

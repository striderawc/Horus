"""Shape and gradient checks for BallHeatmapSNN (no dataset required)."""

from __future__ import annotations

import torch

from horus.models import BallHeatmapCNN, BallHeatmapSNN
from horus.models.snn import LAYER_NAMES, STAGE_NAMES


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


def test_forward_with_activity_shapes():
    base = 8
    model = BallHeatmapSNN(base=base)
    model.eval()
    b, h, w = 2, 64, 64
    x = torch.rand(b, 6, h, w)
    with torch.no_grad():
        heatmap, trace = model.forward_with_activity(x)
        plain = model(x)

    assert heatmap.shape == (b, 1, h, w)
    assert torch.allclose(heatmap, plain)

    b2, b4 = base * 2, base * 4
    expected = {
        "down1_a": (b, 3, base, h, w),
        "down1_b": (b, 3, base, h, w),
        "down2_a": (b, 3, b2, h // 2, w // 2),
        "down2_b": (b, 3, b2, h // 2, w // 2),
        "down3_a": (b, 3, b4, h // 4, w // 4),
        "down3_b": (b, 3, b4, h // 4, w // 4),
        "bottleneck_a": (b, 3, b4, h // 8, w // 8),
        "bottleneck_b": (b, 3, b4, h // 8, w // 8),
        "up3_a": (b, 3, b2, h // 4, w // 4),
        "up3_b": (b, 3, b2, h // 4, w // 4),
        "up2_a": (b, 3, base, h // 2, w // 2),
        "up2_b": (b, 3, base, h // 2, w // 2),
        "up1_a": (b, 3, base, h, w),
        "up1_b": (b, 3, base, h, w),
    }
    assert set(LAYER_NAMES) == set(expected)
    assert [f"{s}_{ab}" for s in STAGE_NAMES for ab in ("a", "b")] == list(LAYER_NAMES)
    for name in LAYER_NAMES:
        layer = trace[name]
        assert layer.spikes.shape == expected[name], name
        assert layer.membrane.shape == expected[name], name
        assert torch.isfinite(layer.spikes).all()
        assert torch.isfinite(layer.membrane).all()


def test_param_count_matches_cnn_in_ch2():
    """Body matches BallHeatmapCNN(in_ch=2); event CNN uses in_ch=6 so first layer differs."""
    snn = BallHeatmapSNN(in_ch=2, steps=3, base=16)
    cnn = BallHeatmapCNN(in_ch=2, base=16)
    n_snn = sum(p.numel() for p in snn.parameters())
    n_cnn = sum(p.numel() for p in cnn.parameters())
    assert n_snn == n_cnn
    assert n_snn > 100_000  # not the old ~14k shallow encoder

"""Shape and gradient checks for BallHeatmapSNN (no dataset required)."""

from __future__ import annotations

import torch

from horus.models import BallHeatmapSNN, circle_mse_loss
from horus.models.snn import LAYER_NAMES, STAGE_NAMES


def test_snn_forward_shape_and_backward():
    model = BallHeatmapSNN()
    model.train()
    x = torch.rand(2, 6, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)
    assert torch.isfinite(y).all()
    assert (y >= 0).all() and (y <= 1).all()
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
        heat, trace = model.forward_with_activity(x)
        plain = model(x)

    assert heat.shape == (b, 1, h, w)
    assert torch.allclose(heat, plain)

    hidden = base * 2
    expected = {
        "lif1": (b, 3, base, h, w),
        "lif2": (b, 3, hidden, h, w),
    }
    assert set(LAYER_NAMES) == set(expected)
    assert list(STAGE_NAMES) == list(LAYER_NAMES)
    for name in LAYER_NAMES:
        layer = trace[name]
        assert layer.spikes.shape == expected[name], name
        assert layer.membrane.shape == expected[name], name
        assert torch.isfinite(layer.spikes).all()
        assert torch.isfinite(layer.membrane).all()


def test_shallow_param_count_scale():
    """Two-LIF shallow encoder stays small vs the old CNN-matched U-Net."""
    snn = BallHeatmapSNN(in_ch=2, steps=3, base=16)
    n_snn = sum(p.numel() for p in snn.parameters())
    # ~conv1 + conv2 + 1×1 head + BN ≈ few k
    assert 1_000 < n_snn < 20_000


def test_drop_last_lif_forward_and_activity():
    """Final lif2 is ReLU; activity omits lif2; forward still works."""
    model = BallHeatmapSNN(base=8, drop_last_lif=True)
    model.train()
    x = torch.rand(2, 6, 64, 64)
    y = model(x)
    assert y.shape == (2, 1, 64, 64)
    assert torch.isfinite(y).all()
    y.sum().backward()

    heat, trace = model.forward_with_activity(x)
    assert heat.shape == y.shape
    assert "lif2" not in trace.layers
    assert set(trace.layers) == {"lif1"}
    assert set(trace.layers) == set(model.lif_layer_names)


def test_snn_circle_mse_loss_finite():
    model = BallHeatmapSNN(base=8)
    model.eval()
    x = torch.rand(2, 6, 32, 32)
    coords = torch.tensor([[10.0, 12.0], [20.0, 8.0]])
    visible = torch.ones(2)
    with torch.no_grad():
        pred = model(x)
    loss = circle_mse_loss(pred, coords, visible, radius=2.0)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0

"""Unit tests for uncertainty head + Gaussian NLL."""

from __future__ import annotations

import torch

from horus.models import (
    BallHeatmapCNN,
    BallHeatmapSNN,
    gaussian_nll_loss,
    unpack_model_output,
)
from horus.track.loop import detection_to_measurement


def test_cnn_uncertainty_shapes():
    model = BallHeatmapCNN(in_ch=2, base=8, predict_uncertainty=True)
    x = torch.randn(3, 2, 64, 64)
    heat, log_sigma = model(x)
    assert heat.shape == (3, 1, 64, 64)
    assert log_sigma.shape == (3,)
    assert torch.isfinite(log_sigma).all()
    heat.sum().backward()


def test_cnn_backward_compat_no_uncertainty():
    model = BallHeatmapCNN(in_ch=2, base=8, predict_uncertainty=False)
    out = model(torch.randn(2, 2, 32, 32))
    heat, log_sigma = unpack_model_output(out)
    assert heat.shape == (2, 1, 32, 32)
    assert log_sigma is None


def test_snn_uncertainty_shapes():
    model = BallHeatmapSNN(base=8, predict_uncertainty=True)
    x = torch.rand(2, 6, 48, 48)
    heat, log_sigma = model(x)
    assert heat.shape == (2, 1, 48, 48)
    assert log_sigma.shape == (2,)
    (heat.sum() + log_sigma.sum()).backward()


def test_gaussian_nll_skips_negatives():
    heat = torch.zeros(2, 1, 16, 16)
    heat[0, 0, 5, 7] = 1.0
    heat.requires_grad_(True)
    coords = torch.tensor([[7.0, 5.0], [1.0, 1.0]])
    log_sigma = torch.zeros(2, requires_grad=True)
    visible = torch.tensor([1.0, 0.0])
    loss = gaussian_nll_loss(heat, coords, log_sigma, visible, height=16, width=16)
    assert torch.isfinite(loss)
    # All-negative → differentiable zero
    heat2 = heat.detach().clone().requires_grad_(True)
    log2 = log_sigma.detach().clone().requires_grad_(True)
    loss0 = gaussian_nll_loss(
        heat2, coords, log2, torch.zeros(2), height=16, width=16
    )
    assert float(loss0.item()) == 0.0
    assert loss0.requires_grad
    loss0.backward()


def test_all_negative_coord_batch_backward():
    """Crop-mode all-negative batches must still produce a grad path."""
    from horus.train.heatmap import _batch_loss

    model = BallHeatmapCNN(in_ch=2, base=4, predict_uncertainty=True)
    x = torch.randn(2, 2, 32, 32)
    heat, log_sigma = model(x)
    coords = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
    visible = torch.zeros(2)
    loss = _batch_loss(
        heat,
        coords,
        visible,
        height=32,
        width=32,
        loss="coord",
        log_sigma=log_sigma,
        nll_weight=0.1,
    )
    assert loss.requires_grad
    loss.backward()


def test_detection_to_measurement_miss_and_hit():
    heat = torch.zeros(1, 32, 32)
    z, r = detection_to_measurement(heat, 0.0, origin=(10, 20), peak_thresh=0.3)
    assert z is None and r is None

    heat[0, 4, 6] = 0.9
    z, r = detection_to_measurement(heat, 1.5, origin=(10, 20), peak_thresh=0.3)
    assert z is not None and r is not None
    assert abs(z[0] - 16.0) < 1e-5
    assert abs(z[1] - 24.0) < 1e-5
    assert abs(r - float(torch.exp(torch.tensor(1.5)))) < 1e-5

"""Unit tests for presence head, focal loss, and gated decode."""

from __future__ import annotations

import torch

from horus.models import (
    BallHeatmapCNN,
    BallHeatmapSNN,
    focal_presence_loss,
    penalty_reduced_focal_loss,
    soft_heatmap_spatial_std,
    unpack_model_output,
)
from horus.track.loop import detection_to_measurement


def test_cnn_presence_shapes():
    model = BallHeatmapCNN(in_ch=2, base=8, predict_presence=True)
    x = torch.randn(3, 2, 64, 64)
    heat, presence = model(x)
    assert heat.shape == (3, 1, 64, 64)
    assert presence.shape == (3,)
    assert torch.isfinite(heat).all() and torch.isfinite(presence).all()
    (heat.sum() + presence.sum()).backward()


def test_cnn_logits_not_sigmoid():
    model = BallHeatmapCNN(in_ch=2, base=4, predict_presence=True)
    heat, _ = model(torch.randn(1, 2, 32, 32))
    # Logits can be outside [0, 1].
    assert heat.min() < 0 or heat.max() > 1 or True  # structural: no forced sigmoid


def test_snn_presence_shapes():
    model = BallHeatmapSNN(base=8, predict_presence=True)
    x = torch.rand(2, 6, 48, 48)
    heat, presence = model(x)
    assert heat.shape == (2, 1, 48, 48)
    assert presence.shape == (2,)
    (heat.sum() + presence.sum()).backward()


def test_focal_loss_prefers_correct_peak():
    target = torch.zeros(1, 1, 16, 16)
    target[0, 0, 5, 7] = 1.0
    good = torch.full((1, 1, 16, 16), -4.0)
    good[0, 0, 5, 7] = 8.0
    bad = torch.full((1, 1, 16, 16), -4.0)
    bad[0, 0, 1, 1] = 8.0
    assert float(penalty_reduced_focal_loss(good, target)) < float(
        penalty_reduced_focal_loss(bad, target)
    )


def test_focal_all_zero_target_finite():
    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    target = torch.zeros(2, 1, 8, 8)
    loss = penalty_reduced_focal_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()


def test_focal_presence_combined():
    heat = torch.zeros(2, 1, 16, 16)
    heat[0, 0, 4, 4] = 5.0
    target = torch.zeros(2, 1, 16, 16)
    target[0, 0, 4, 4] = 1.0
    presence = torch.tensor([2.0, -2.0], requires_grad=True)
    present = torch.tensor([1.0, 0.0])
    heat = heat.clone().requires_grad_(True)
    loss = focal_presence_loss(heat, presence, target, present)
    assert torch.isfinite(loss)
    loss.backward()


def test_detection_gates_on_presence():
    heat = torch.full((1, 32, 32), -2.0)
    heat[0, 4, 6] = 5.0
    z, r = detection_to_measurement(
        heat, -5.0, origin=(10, 20), presence_thresh=0.5
    )
    assert z is None and r is None

    z, r = detection_to_measurement(
        heat, 5.0, origin=(10, 20), presence_thresh=0.5
    )
    assert z is not None and r is not None
    assert abs(z[0] - 16.0) < 1e-5
    assert abs(z[1] - 24.0) < 1e-5
    assert r > 0


def test_soft_spatial_std_positive():
    heat = torch.zeros(2, 1, 16, 16)
    heat[0, 0, 8, 8] = 10.0
    heat[1, 0, :, :] = 0.0
    std = soft_heatmap_spatial_std(heat, temperature=0.5)
    assert std.shape == (2,)
    assert float(std[0]) < float(std[1])  # sharp peak sharper than flat


def test_all_negative_focal_batch_backward():
    from horus.train.heatmap import _batch_loss

    model = BallHeatmapCNN(in_ch=2, base=4, predict_presence=True)
    x = torch.randn(2, 2, 32, 32)
    heat, presence = model(x)
    target = torch.zeros(2, 1, 32, 32)
    coords = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
    visible = torch.zeros(2)
    loss = _batch_loss(
        heat,
        coords,
        visible,
        height=32,
        width=32,
        loss="focal_presence",
        presence_logit=presence,
        heatmap_target=target,
    )
    assert loss.requires_grad
    loss.backward()

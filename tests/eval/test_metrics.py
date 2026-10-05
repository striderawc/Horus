"""Unit tests for eval metrics (no dataset required)."""

from __future__ import annotations

import torch
import torch.nn as nn

from horus.eval.bench import ModelResult, format_table
from horus.eval.metrics import (
    input_channels,
    measure_latency,
    visible_pixel_distances,
)
from horus.models import heatmap_to_coords


def test_visible_pixel_distances_perfect():
    # Peak at (3, 1) → heatmap argmax should match gt
    heat = torch.zeros(2, 1, 4, 5)
    heat[0, 0, 1, 3] = 1.0
    heat[1, 0, 2, 4] = 1.0
    gt = torch.tensor([[3.0, 1.0], [4.0, 2.0]])
    visible = torch.tensor([1.0, 1.0])
    dists = visible_pixel_distances(heat, gt, visible)
    assert dists.shape == (2,)
    assert torch.allclose(dists, torch.zeros(2), atol=1e-5)


def test_visible_pixel_distances_skips_invisible():
    heat = torch.zeros(2, 1, 4, 5)
    heat[0, 0, 0, 0] = 1.0
    heat[1, 0, 0, 1] = 1.0
    gt = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    visible = torch.tensor([1.0, 0.0])
    dists = visible_pixel_distances(heat, gt, visible)
    assert dists.shape == (1,)
    assert float(dists[0]) < 1e-5


def test_heatmap_to_coords_roundtrip():
    heat = torch.zeros(1, 1, 8, 8)
    heat[0, 0, 5, 2] = 0.9
    xy = heatmap_to_coords(heat)
    assert torch.equal(xy, torch.tensor([[2.0, 5.0]]))


def test_soft_heatmap_to_coords_near_peak():
    from horus.models import soft_heatmap_to_coords

    heat = torch.zeros(1, 1, 8, 8)
    heat[0, 0, 5, 2] = 1.0
    xy = soft_heatmap_to_coords(heat, temperature=0.01)
    assert abs(float(xy[0, 0]) - 2.0) < 0.05
    assert abs(float(xy[0, 1]) - 5.0) < 0.05


def test_coord_distance_loss_zero_at_peak():
    from horus.models import coord_distance_loss

    heat = torch.zeros(2, 1, 8, 8)
    heat[0, 0, 5, 2] = 1.0
    heat[1, 0, 1, 4] = 1.0
    coords = torch.tensor([[2.0, 5.0], [4.0, 1.0]])
    visible = torch.tensor([1.0, 1.0])
    loss = coord_distance_loss(heat, coords, visible, temperature=0.01)
    assert float(loss) < 1e-5
    # Invisible sample ignored
    loss_mask = coord_distance_loss(
        heat, coords, torch.tensor([1.0, 0.0]), temperature=0.01
    )
    assert float(loss_mask) < 1e-5


def test_coord_distance_loss_matches_val_px_scaled():
    from horus.models import coord_distance_loss

    heat = torch.zeros(2, 1, 8, 8)
    heat[0, 0, 0, 0] = 1.0
    heat[1, 0, 7, 7] = 1.0
    coords = torch.tensor([[3.0, 4.0], [1.0, 1.0]])
    visible = torch.tensor([1.0, 1.0])
    loss = coord_distance_loss(heat, coords, visible)
    dists = visible_pixel_distances(heat, coords, visible)
    diag = (7**2 + 7**2) ** 0.5
    assert abs(float(loss) - float(dists.mean()) / diag) < 1e-5


def test_coord_distance_loss_normalized_xy():
    from horus.models import coord_distance_loss

    # pred center of 8x8 → pixel (3.5, 3.5); target (0,0); / diagonal
    pred = torch.tensor([[0.5, 0.5]])
    coords = torch.tensor([[0.0, 0.0]])
    loss = coord_distance_loss(pred, coords, height=8, width=8)
    pixel = (2 * 3.5**2) ** 0.5
    diag = (7**2 + 7**2) ** 0.5
    assert abs(float(loss) - pixel / diag) < 1e-5
    assert 0.0 < float(loss) <= 1.0 + 1e-5


def test_input_channels():
    cfg = {
        "sequence": {"length": 3},
        "events": {"voxel": {"num_bins": 5}},
    }
    assert input_channels("rgb", cfg) == 9
    assert input_channels("diff", cfg) == 2
    assert input_channels("event_frame", cfg) == 6
    assert input_channels("voxel", cfg) == 15


def test_measure_latency_runs():
    model = nn.Sequential(nn.Conv2d(3, 1, 1), nn.Sigmoid())
    ms = measure_latency(
        model, in_ch=3, height=16, width=16, device=torch.device("cpu"), warmup=1, runs=3
    )
    assert ms >= 0.0


def test_format_table_winners():
    results = [
        ModelResult(
            name="a",
            arch="rgb_cnn",
            modality="rgb",
            mean_px=10.0,
            acc_at_5=0.4,
            acc_at_10=0.6,
            n_visible=100,
            gflops=2.0,
            latency_ms=5.0,
            device="cpu",
            acc_per_gflop=0.2,
            acc_per_ms=0.08,
        ),
        ModelResult(
            name="b",
            arch="rgb_cnn",
            modality="rgb",
            mean_px=5.0,
            acc_at_5=0.5,
            acc_at_10=0.7,
            n_visible=100,
            gflops=10.0,
            latency_ms=2.0,
            device="cpu",
            acc_per_gflop=0.05,
            acc_per_ms=0.25,
        ),
    ]
    text = format_table(results)
    assert "Best accuracy" in text and "b" in text
    assert "Best acc/GFLOP" in text and "a" in text
    assert "Best acc/ms" in text and "b" in text

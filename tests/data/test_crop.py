"""Unit tests for spatial crop helpers (no dataset required)."""

from __future__ import annotations

import numpy as np
import torch

from horus.data.crop import (
    ball_in_crop,
    crop_chw_stack,
    crop_to_fullframe,
    fullframe_to_crop,
    sample_negative_crop,
    sample_positive_crop,
)


def test_crop_chw_stack_interior():
    x = torch.arange(3 * 20 * 30, dtype=torch.float32).reshape(3, 20, 30)
    out = crop_chw_stack(x, 5, 4, 8)
    assert out.shape == (3, 8, 8)
    assert torch.equal(out, x[:, 4:12, 5:13])


def test_crop_chw_stack_pads_out_of_bounds():
    x = torch.ones(1, 10, 10)
    out = crop_chw_stack(x, -2, -3, 8, pad_value=0.0)
    assert out.shape == (1, 8, 8)
    # Top-left padded region should be zeros.
    assert float(out[0, 0, 0]) == 0.0
    # Overlap with source should be ones.
    assert float(out[0, 3, 2]) == 1.0


def test_coord_roundtrip():
    xy = np.array([40.0, 50.0])
    crop = fullframe_to_crop(xy, 10, 20)
    assert np.allclose(crop, [30.0, 30.0])
    back = crop_to_fullframe(crop, 10, 20)
    assert np.allclose(back, xy)


def test_sample_positive_keeps_ball_in_crop():
    rng = np.random.default_rng(0)
    bx, by = 200.0, 150.0
    for size in (64, 128):
        x0, y0, sz, in_crop = sample_positive_crop(
            bx, by, size, 640, 360, rng, jitter_std_frac=0.1, margin=4.0
        )
        assert sz == size
        assert in_crop
        assert ball_in_crop(bx, by, x0, y0, sz, margin=4.0)


def test_sample_negative_excludes_ball():
    rng = np.random.default_rng(1)
    bx, by = 320.0, 180.0
    for _ in range(20):
        x0, y0, sz = sample_negative_crop(
            bx, by, 64, 640, 360, rng, margin=4.0
        )
        assert not ball_in_crop(bx, by, x0, y0, sz, margin=4.0)

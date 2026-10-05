"""Unit tests for shared gate / snap crop helpers."""

from __future__ import annotations

import numpy as np

from horus.track import (
    ConstantVelocityKF,
    TrackState,
    gate_to_square_crop,
    gate_window,
    snap_square_window,
)


def test_gate_window_matches_filter_method():
    kf = ConstantVelocityKF(dt=1.0 / 30.0, init_pos_std=5.0)
    state = kf.initiate(np.array([100.0, 200.0]))
    assert gate_window(state, n_sigma=3.0) == kf.gate_window(state, n_sigma=3.0)


def test_snap_square_window_picks_smallest_covering_bucket():
    x0, y0, size = snap_square_window(
        cx=100.0, cy=100.0, side=70.0, buckets=[64, 128, 256], img_w=640, img_h=360
    )
    assert size == 128
    assert 0 <= x0 <= 640 - size
    assert 0 <= y0 <= 360 - size
    # Centered near (100, 100)
    assert abs((x0 + size / 2) - 100.0) < 2.0
    assert abs((y0 + size / 2) - 100.0) < 2.0


def test_snap_clamps_near_border():
    x0, y0, size = snap_square_window(
        cx=10.0, cy=10.0, side=128.0, buckets=[128], img_w=640, img_h=360
    )
    assert size == 128
    assert x0 == 0
    assert y0 == 0


def test_gate_to_square_crop_grows_with_uncertainty():
    mean = np.array([200.0, 180.0, 0.0, 0.0])
    cov_small = np.diag([4.0, 4.0, 1.0, 1.0])
    cov_large = np.diag([400.0, 400.0, 1.0, 1.0])
    s_small = TrackState(mean=mean, cov=cov_small)
    s_large = TrackState(mean=mean.copy(), cov=cov_large)
    _, _, sz_s = gate_to_square_crop(
        s_small, buckets=[64, 128, 256], img_w=640, img_h=360, n_sigma=3.0
    )
    _, _, sz_l = gate_to_square_crop(
        s_large, buckets=[64, 128, 256], img_w=640, img_h=360, n_sigma=3.0
    )
    assert sz_l >= sz_s

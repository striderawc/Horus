"""Unit tests for the constant-velocity Kalman filter (no dataset required)."""

from __future__ import annotations

import numpy as np

from horus.config import load_config
from horus.track import (
    ConstantVelocityKF,
    add_gaussian_noise,
    build_kf,
    synthetic_cv_trajectory,
)


def test_predict_moves_by_v_dt():
    kf = ConstantVelocityKF(dt=0.1, q_pos=1e-6, q_vel=1e-6, r_meas=1.0)
    state = kf.initiate(np.array([10.0, 20.0]))
    state.mean[2:] = np.array([3.0, -4.0])
    pred = kf.predict(state)
    assert np.allclose(pred.mean[:2], [10.0 + 0.3, 20.0 - 0.4])
    assert np.allclose(pred.mean[2:], [3.0, -4.0])


def test_update_pulls_toward_measurement():
    kf = ConstantVelocityKF(dt=1.0 / 30.0, q_pos=1.0, q_vel=10.0, r_meas=5.0)
    state = kf.initiate(np.array([0.0, 0.0]))
    # Large prior uncertainty + measurement far from mean → estimate moves toward z
    z = np.array([50.0, -30.0])
    updated = kf.update(state, z)
    assert float(np.linalg.norm(updated.position - z)) < float(
        np.linalg.norm(state.position - z)
    )


def test_update_none_is_identity():
    kf = ConstantVelocityKF(dt=1.0 / 30.0)
    state = kf.initiate(np.array([1.0, 2.0]))
    out = kf.update(state, None)
    assert np.allclose(out.mean, state.mean)
    assert np.allclose(out.cov, state.cov)
    # Defensive copy
    out.mean[0] = 999.0
    assert state.mean[0] == 1.0


def test_gate_window_grows_with_uncertainty():
    kf = ConstantVelocityKF(dt=1.0 / 30.0, init_pos_std=5.0)
    state = kf.initiate(np.array([100.0, 200.0]))
    x0, y0, x1, y1 = kf.gate_window(state, n_sigma=3.0)
    assert x0 < 100.0 < x1
    assert y0 < 200.0 < y1
    # Wider after a few predicts without updates
    for _ in range(5):
        state = kf.predict(state)
    x0b, y0b, x1b, y1b = kf.gate_window(state, n_sigma=3.0)
    assert (x1b - x0b) > (x1 - x0)
    assert (y1b - y0b) > (y1 - y0)


def test_noisy_cv_track_beats_raw_measurements():
    dt = 1.0 / 30.0
    sigma = 5.0
    n_steps = 80
    rng = np.random.default_rng(0)
    true_xy = synthetic_cv_trajectory(
        n_steps, x0=(100.0, 150.0), v0=(40.0, -25.0), dt=dt
    )
    meas = add_gaussian_noise(true_xy, sigma=sigma, rng=rng)

    kf = ConstantVelocityKF(
        dt,
        q_pos=0.5,
        q_vel=5.0,
        r_meas=sigma,
        init_pos_std=sigma,
        init_vel_std=80.0,
    )
    state = kf.initiate(meas[0])
    est = np.zeros_like(true_xy)
    est[0] = state.position
    for t in range(1, n_steps):
        state = kf.predict(state)
        state = kf.update(state, meas[t])
        est[t] = state.position

    # Skip warm-up so velocity can lock on
    warm = 15
    meas_rmse = float(np.sqrt(np.mean(np.sum((meas[warm:] - true_xy[warm:]) ** 2, axis=1))))
    est_rmse = float(np.sqrt(np.mean(np.sum((est[warm:] - true_xy[warm:]) ** 2, axis=1))))
    assert est_rmse < meas_rmse
    assert est_rmse < sigma  # should beat the measurement noise floor on average


def test_build_kf_resolves_dt_from_fps():
    cfg = load_config()
    cfg = {**cfg, "tracker": {**(cfg.get("tracker") or {}), "dt": None}}
    kf = build_kf(cfg)
    assert np.isclose(kf.dt, 1.0 / float(cfg["source"]["fps"]))


def test_synthetic_trajectory_length_and_motion():
    xy = synthetic_cv_trajectory(5, x0=(0.0, 0.0), v0=(1.0, 2.0), dt=1.0)
    assert xy.shape == (5, 2)
    assert np.allclose(xy[0], [0.0, 0.0])
    assert np.allclose(xy[-1], [4.0, 8.0])

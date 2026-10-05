"""Unit tests for the adaptive constant-acceleration Kalman filter."""

from __future__ import annotations

import numpy as np

from horus.config import load_config
from horus.track import (
    AdaptiveConstantAccelerationKF,
    add_gaussian_noise,
    build_ca_akf,
    synthetic_cv_trajectory,
)


def test_state_is_6d():
    akf = AdaptiveConstantAccelerationKF(dt=1.0 / 30.0)
    state = akf.initiate(np.array([10.0, 20.0]))
    assert state.dim == 6
    assert state.mean.shape == (6,)
    assert state.cov.shape == (6, 6)
    assert state.acceleration is not None
    assert np.allclose(state.acceleration, [0.0, 0.0])


def test_predict_uses_acceleration():
    dt = 0.1
    akf = AdaptiveConstantAccelerationKF(
        dt, q_pos=1e-9, q_vel=1e-9, q_acc=1e-9, r_meas=1.0
    )
    state = akf.initiate(np.array([0.0, 0.0]))
    state.mean[:] = np.array([0.0, 0.0, 3.0, -4.0, 2.0, 1.0])
    pred = akf.predict(state)
    # x' = x + v dt + 0.5 a dt^2
    assert np.allclose(
        pred.mean[:2],
        [3.0 * dt + 0.5 * 2.0 * dt**2, -4.0 * dt + 0.5 * 1.0 * dt**2],
        atol=1e-6,
    )
    assert np.allclose(pred.mean[2:4], [3.0 + 2.0 * dt, -4.0 + 1.0 * dt], atol=1e-6)
    assert np.allclose(pred.mean[4:6], [2.0, 1.0], atol=1e-6)


def test_update_none_skips_adaptation():
    akf = AdaptiveConstantAccelerationKF(dt=1.0 / 30.0, r_meas=5.0)
    state = akf.initiate(np.array([1.0, 2.0]))
    q0, r0 = akf.Q.copy(), akf.R.copy()
    pred = akf.predict(state)
    out = akf.update(pred, None)
    assert akf._update_count == 0
    assert np.allclose(akf.Q, q0)
    assert np.allclose(akf.R, r0)
    assert np.allclose(out.mean, pred.mean)


def test_noisy_ca_track_beats_raw_measurements():
    dt = 1.0 / 30.0
    sigma = 5.0
    n_steps = 100
    rng = np.random.default_rng(2)
    t = np.arange(n_steps) * dt
    # Constant acceleration ground truth
    x0, y0 = 80.0, 200.0
    v0 = np.array([20.0, -10.0])
    a0 = np.array([15.0, -8.0])
    true_xy = np.stack(
        [
            x0 + v0[0] * t + 0.5 * a0[0] * t**2,
            y0 + v0[1] * t + 0.5 * a0[1] * t**2,
        ],
        axis=1,
    )
    meas = add_gaussian_noise(true_xy, sigma=sigma, rng=rng)

    akf = AdaptiveConstantAccelerationKF(
        dt,
        q_pos=0.5,
        q_vel=5.0,
        q_acc=20.0,
        r_meas=sigma,
        init_pos_std=sigma,
        init_vel_std=80.0,
        init_acc_std=40.0,
    )
    state = akf.initiate(meas[0])
    est = np.zeros_like(true_xy)
    est[0] = state.position
    for i in range(1, n_steps):
        state = akf.predict(state)
        state = akf.update(state, meas[i])
        est[i] = state.position

    warm = 20
    meas_rmse = float(np.sqrt(np.mean(np.sum((meas[warm:] - true_xy[warm:]) ** 2, axis=1))))
    est_rmse = float(np.sqrt(np.mean(np.sum((est[warm:] - true_xy[warm:]) ** 2, axis=1))))
    assert est_rmse < meas_rmse
    # After warm-up, should be in the ballpark of the measurement noise
    assert est_rmse < 1.5 * sigma


def test_cv_trajectory_still_tracks():
    """CA filter should still beat noisy measurements on CV motion."""
    dt = 1.0 / 30.0
    sigma = 5.0
    n_steps = 80
    rng = np.random.default_rng(3)
    true_xy = synthetic_cv_trajectory(
        n_steps, x0=(100.0, 150.0), v0=(40.0, -25.0), dt=dt
    )
    meas = add_gaussian_noise(true_xy, sigma=sigma, rng=rng)
    akf = AdaptiveConstantAccelerationKF(
        dt, q_pos=0.5, q_vel=5.0, q_acc=10.0, r_meas=sigma,
        init_pos_std=sigma, init_vel_std=80.0, init_acc_std=30.0,
    )
    state = akf.initiate(meas[0])
    est = np.zeros_like(true_xy)
    est[0] = state.position
    for i in range(1, n_steps):
        state = akf.predict(state)
        state = akf.update(state, meas[i])
        est[i] = state.position
    warm = 15
    meas_rmse = float(np.sqrt(np.mean(np.sum((meas[warm:] - true_xy[warm:]) ** 2, axis=1))))
    est_rmse = float(np.sqrt(np.mean(np.sum((est[warm:] - true_xy[warm:]) ** 2, axis=1))))
    assert est_rmse < meas_rmse


def test_build_ca_akf_reads_config():
    cfg = load_config()
    cfg = {
        **cfg,
        "tracker": {
            **(cfg.get("tracker") or {}),
            "q_acc": 77.0,
            "init_acc_std": 33.0,
            "akf_forgetting": 0.91,
        },
    }
    akf = build_ca_akf(cfg)
    assert isinstance(akf, AdaptiveConstantAccelerationKF)
    assert np.isclose(akf.q_acc, 77.0)
    assert np.isclose(akf.init_acc_std, 33.0)
    assert np.isclose(akf.forgetting, 0.91)
    assert np.isclose(akf.dt, 1.0 / float(cfg["source"]["fps"]))

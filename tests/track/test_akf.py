"""Unit tests for the adaptive constant-velocity Kalman filter."""

from __future__ import annotations

import numpy as np

from horus.config import load_config
from horus.track import (
    AdaptiveConstantVelocityKF,
    add_gaussian_noise,
    build_akf,
    synthetic_cv_trajectory,
)


def test_predict_matches_base_kf_motion():
    akf = AdaptiveConstantVelocityKF(dt=0.1, q_pos=1e-6, q_vel=1e-6, r_meas=1.0)
    state = akf.initiate(np.array([10.0, 20.0]))
    state.mean[2:] = np.array([3.0, -4.0])
    pred = akf.predict(state)
    assert np.allclose(pred.mean[:2], [10.0 + 0.3, 20.0 - 0.4])


def test_update_none_skips_adaptation():
    akf = AdaptiveConstantVelocityKF(dt=1.0 / 30.0, r_meas=5.0)
    state = akf.initiate(np.array([1.0, 2.0]))
    q0, r0 = akf.Q.copy(), akf.R.copy()
    pred = akf.predict(state)
    out = akf.update(pred, None)
    assert akf._update_count == 0
    assert np.allclose(akf.Q, q0)
    assert np.allclose(akf.R, r0)
    assert np.allclose(out.mean, pred.mean)


def test_r_adapts_toward_true_measurement_noise():
    """Under-init R, then feed noisy CV measurements → learned R grows."""
    dt = 1.0 / 30.0
    true_sigma = 12.0
    n_steps = 150
    rng = np.random.default_rng(0)
    true_xy = synthetic_cv_trajectory(
        n_steps, x0=(100.0, 150.0), v0=(30.0, -15.0), dt=dt
    )
    meas = add_gaussian_noise(true_xy, sigma=true_sigma, rng=rng)

    akf = AdaptiveConstantVelocityKF(
        dt,
        q_pos=0.5,
        q_vel=5.0,
        r_meas=2.0,  # deliberately too small
        init_pos_std=true_sigma,
        init_vel_std=80.0,
        forgetting=0.97,
        r_min=0.5,
        r_max_scale=200.0,
    )
    r0 = float(np.sqrt(akf.R[0, 0]))
    state = akf.initiate(meas[0])
    for t in range(1, n_steps):
        state = akf.predict(state)
        state = akf.update(state, meas[t])

    r_learned = float(np.sqrt(0.5 * (akf.R[0, 0] + akf.R[1, 1])))
    assert r_learned > r0
    # Should move meaningfully toward the true noise (not perfect match)
    assert r_learned > 0.4 * true_sigma


def test_q_changes_with_model_mismatch():
    """Accelerating truth vs CV model → adapted Q grows from its prior."""
    dt = 1.0 / 30.0
    n_steps = 100
    t = np.arange(n_steps) * dt
    # Constant acceleration → CV model is wrong
    true_xy = np.stack([50.0 + 20.0 * t + 40.0 * t**2, 100.0 + 10.0 * t], axis=1)
    meas = true_xy.copy()

    akf = AdaptiveConstantVelocityKF(
        dt,
        q_pos=0.1,
        q_vel=0.1,
        r_meas=1.0,
        init_pos_std=2.0,
        init_vel_std=20.0,
        forgetting=0.9,
        q_min=1e-6,
        q_max_scale=1e4,
    )
    q_trace0 = float(np.trace(akf.Q))
    state = akf.initiate(meas[0])
    for i in range(1, n_steps):
        state = akf.predict(state)
        state = akf.update(state, meas[i])
    assert float(np.trace(akf.Q)) > q_trace0


def test_noisy_cv_track_beats_raw_measurements():
    dt = 1.0 / 30.0
    sigma = 5.0
    n_steps = 80
    rng = np.random.default_rng(1)
    true_xy = synthetic_cv_trajectory(
        n_steps, x0=(100.0, 150.0), v0=(40.0, -25.0), dt=dt
    )
    meas = add_gaussian_noise(true_xy, sigma=sigma, rng=rng)

    akf = AdaptiveConstantVelocityKF(
        dt,
        q_pos=0.5,
        q_vel=5.0,
        r_meas=sigma,
        init_pos_std=sigma,
        init_vel_std=80.0,
    )
    state = akf.initiate(meas[0])
    est = np.zeros_like(true_xy)
    est[0] = state.position
    for t in range(1, n_steps):
        state = akf.predict(state)
        state = akf.update(state, meas[t])
        est[t] = state.position

    warm = 15
    meas_rmse = float(np.sqrt(np.mean(np.sum((meas[warm:] - true_xy[warm:]) ** 2, axis=1))))
    est_rmse = float(np.sqrt(np.mean(np.sum((est[warm:] - true_xy[warm:]) ** 2, axis=1))))
    assert est_rmse < meas_rmse


def test_initiate_resets_learned_noise():
    akf = AdaptiveConstantVelocityKF(dt=1.0 / 30.0, r_meas=5.0, forgetting=0.9)
    state = akf.initiate(np.array([0.0, 0.0]))
    for _ in range(20):
        state = akf.predict(state)
        state = akf.update(state, state.position + 3.0)
    assert akf._update_count == 20
    akf.initiate(np.array([10.0, 10.0]))
    assert akf._update_count == 0
    assert np.allclose(akf.Q, akf._Q0)
    assert np.allclose(akf.R, akf._R0)


def test_build_akf_reads_config():
    cfg = load_config()
    cfg = {
        **cfg,
        "tracker": {
            **(cfg.get("tracker") or {}),
            "akf_forgetting": 0.9,
            "akf_r_min": 1.0,
        },
    }
    akf = build_akf(cfg)
    assert isinstance(akf, AdaptiveConstantVelocityKF)
    assert np.isclose(akf.forgetting, 0.9)
    assert np.isclose(akf.r_min, 1.0)
    assert np.isclose(akf.dt, 1.0 / float(cfg["source"]["fps"]))

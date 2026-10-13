"""Unit tests for the constant-velocity EKF (drag / gravity)."""

from __future__ import annotations

import numpy as np

from horus.config import load_config
from horus.track import (
    ConstantVelocityEKF,
    ConstantVelocityKF,
    add_gaussian_noise,
    build_ekf,
    synthetic_cv_trajectory,
)
from horus.track.ekf import transition_f, transition_jacobian


def test_zero_drag_matches_linear_kf_predict():
    dt = 0.1
    ekf = ConstantVelocityEKF(dt, q_pos=1.0, q_vel=10.0, drag=0.0, gravity=0.0)
    kf = ConstantVelocityKF(dt, q_pos=1.0, q_vel=10.0)
    z0 = np.array([10.0, 20.0])
    s_ekf = ekf.initiate(z0)
    s_kf = kf.initiate(z0)
    s_ekf.mean[2:] = np.array([3.0, -4.0])
    s_kf.mean[2:] = np.array([3.0, -4.0])
    p_ekf = ekf.predict(s_ekf)
    p_kf = kf.predict(s_kf)
    assert np.allclose(p_ekf.mean, p_kf.mean)
    assert np.allclose(p_ekf.cov, p_kf.cov)


def test_drag_slows_velocity():
    dt = 0.1
    ekf = ConstantVelocityEKF(
        dt, q_pos=1e-9, q_vel=1e-9, drag=0.05, gravity=0.0
    )
    state = ekf.initiate(np.array([0.0, 0.0]))
    state.mean[2:] = np.array([40.0, 0.0])
    pred = ekf.predict(state)
    assert pred.mean[2] < 40.0
    assert np.isclose(pred.mean[3], 0.0, atol=1e-9)
    assert pred.mean[0] > state.mean[0]


def test_jacobian_matches_finite_difference():
    dt = 1.0 / 30.0
    drag = 0.02
    gravity = 200.0
    mean = np.array([100.0, 80.0, 25.0, -15.0], dtype=np.float64)
    jac = transition_jacobian(mean, dt, drag=drag, gravity=gravity)
    eps = 1e-5
    num = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        d = np.zeros(4)
        d[i] = eps
        fp = transition_f(mean + d, dt, drag=drag, gravity=gravity)
        fm = transition_f(mean - d, dt, drag=drag, gravity=gravity)
        num[:, i] = (fp - fm) / (2.0 * eps)
    assert np.allclose(jac, num, rtol=1e-4, atol=1e-4)


def test_update_none_is_identity():
    ekf = ConstantVelocityEKF(dt=1.0 / 30.0, drag=0.01)
    state = ekf.initiate(np.array([1.0, 2.0]))
    out = ekf.update(state, None)
    assert np.allclose(out.mean, state.mean)
    assert np.allclose(out.cov, state.cov)


def test_noisy_cv_track_beats_raw_measurements():
    dt = 1.0 / 30.0
    sigma = 5.0
    n_steps = 80
    rng = np.random.default_rng(0)
    true_xy = synthetic_cv_trajectory(
        n_steps, x0=(100.0, 150.0), v0=(40.0, -25.0), dt=dt
    )
    meas = add_gaussian_noise(true_xy, sigma=sigma, rng=rng)
    ekf = ConstantVelocityEKF(
        dt,
        q_pos=0.5,
        q_vel=5.0,
        r_meas=sigma,
        init_pos_std=sigma,
        init_vel_std=80.0,
        drag=0.0,
        gravity=0.0,
    )
    state = ekf.initiate(meas[0])
    est = np.zeros_like(true_xy)
    est[0] = state.position
    for t in range(1, n_steps):
        state = ekf.predict(state)
        state = ekf.update(state, meas[t])
        est[t] = state.position
    warm = 15
    meas_rmse = float(
        np.sqrt(np.mean(np.sum((meas[warm:] - true_xy[warm:]) ** 2, axis=1)))
    )
    est_rmse = float(
        np.sqrt(np.mean(np.sum((est[warm:] - true_xy[warm:]) ** 2, axis=1)))
    )
    assert est_rmse < meas_rmse
    assert est_rmse < sigma


def test_build_ekf_reads_drag_gravity():
    cfg = load_config()
    cfg = {
        **cfg,
        "tracker": {
            **(cfg.get("tracker") or {}),
            "dt": None,
            "ekf_drag": 0.03,
            "ekf_gravity": 400.0,
        },
    }
    ekf = build_ekf(cfg)
    assert np.isclose(ekf.dt, 1.0 / float(cfg["source"]["fps"]))
    assert np.isclose(ekf.drag, 0.03)
    assert np.isclose(ekf.gravity, 400.0)

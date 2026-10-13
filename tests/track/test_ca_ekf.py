"""Unit tests for the constant-acceleration EKF (drag / gravity)."""

from __future__ import annotations

import numpy as np

from horus.config import load_config
from horus.track import (
    AdaptiveConstantAccelerationKF,
    ConstantAccelerationEKF,
    add_gaussian_noise,
    build_ca_ekf,
    synthetic_cv_trajectory,
)
from horus.track.ca_ekf import ca_transition_f, ca_transition_jacobian


def test_state_is_6d():
    ekf = ConstantAccelerationEKF(dt=1.0 / 30.0)
    state = ekf.initiate(np.array([10.0, 20.0]))
    assert state.dim == 6
    assert state.acceleration is not None
    assert np.allclose(state.acceleration, [0.0, 0.0])


def test_zero_drag_matches_linear_ca_predict():
    dt = 0.1
    ekf = ConstantAccelerationEKF(
        dt, q_pos=1.0, q_vel=10.0, q_acc=50.0, drag=0.0, gravity=0.0
    )
    akf = AdaptiveConstantAccelerationKF(
        dt, q_pos=1.0, q_vel=10.0, q_acc=50.0
    )
    z0 = np.array([0.0, 0.0])
    s_ekf = ekf.initiate(z0)
    s_akf = akf.initiate(z0)
    s_ekf.mean[:] = np.array([0.0, 0.0, 3.0, -4.0, 2.0, 1.0])
    s_akf.mean[:] = np.array([0.0, 0.0, 3.0, -4.0, 2.0, 1.0])
    p_ekf = ekf.predict(s_ekf)
    p_akf = akf.predict(s_akf)
    assert np.allclose(p_ekf.mean, p_akf.mean)
    assert np.allclose(p_ekf.cov, p_akf.cov)


def test_jacobian_matches_finite_difference():
    dt = 1.0 / 30.0
    drag = 0.02
    gravity = 200.0
    mean = np.array([100.0, 80.0, 25.0, -15.0, 3.0, -2.0], dtype=np.float64)
    jac = ca_transition_jacobian(mean, dt, drag=drag, gravity=gravity)
    eps = 1e-5
    num = np.zeros((6, 6), dtype=np.float64)
    for i in range(6):
        d = np.zeros(6)
        d[i] = eps
        fp = ca_transition_f(mean + d, dt, drag=drag, gravity=gravity)
        fm = ca_transition_f(mean - d, dt, drag=drag, gravity=gravity)
        num[:, i] = (fp - fm) / (2.0 * eps)
    assert np.allclose(jac, num, rtol=1e-4, atol=1e-4)


def test_update_none_is_identity():
    ekf = ConstantAccelerationEKF(dt=1.0 / 30.0, drag=0.01)
    state = ekf.initiate(np.array([1.0, 2.0]))
    out = ekf.update(state, None)
    assert np.allclose(out.mean, state.mean)
    assert np.allclose(out.cov, state.cov)


def test_noisy_track_beats_raw_measurements():
    dt = 1.0 / 30.0
    sigma = 5.0
    n_steps = 80
    rng = np.random.default_rng(1)
    true_xy = synthetic_cv_trajectory(
        n_steps, x0=(100.0, 150.0), v0=(40.0, -25.0), dt=dt
    )
    meas = add_gaussian_noise(true_xy, sigma=sigma, rng=rng)
    ekf = ConstantAccelerationEKF(
        dt,
        q_pos=0.5,
        q_vel=5.0,
        q_acc=20.0,
        r_meas=sigma,
        init_pos_std=sigma,
        init_vel_std=80.0,
        init_acc_std=50.0,
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


def test_build_ca_ekf_reads_drag_gravity():
    cfg = load_config()
    cfg = {
        **cfg,
        "tracker": {
            **(cfg.get("tracker") or {}),
            "ekf_drag": 0.04,
            "ekf_gravity": 300.0,
            "q_acc": 40.0,
        },
    }
    ekf = build_ca_ekf(cfg)
    assert np.isclose(ekf.drag, 0.04)
    assert np.isclose(ekf.gravity, 300.0)
    assert np.isclose(ekf.q_acc, 40.0)

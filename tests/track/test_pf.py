"""Unit tests for the constant-velocity particle filter (no dataset required)."""

from __future__ import annotations

import numpy as np

from horus.config import load_config
from horus.track import (
    ConstantVelocityPF,
    add_gaussian_noise,
    build_pf,
    synthetic_cv_trajectory,
)


def test_predict_moves_cloud_by_v_dt():
    from horus.track import TrackState

    pf = ConstantVelocityPF(
        dt=0.1, n_particles=200, q_pos=1e-8, q_vel=1e-8, r_meas=1.0, seed=0
    )
    state = pf.initiate(np.array([10.0, 20.0]))
    particles = state.particles.copy()
    assert particles is not None
    particles[:] = np.array([10.0, 20.0, 3.0, -4.0])
    weights = np.full(pf.n_particles, 1.0 / pf.n_particles)
    state = TrackState(
        mean=particles.mean(axis=0),
        cov=np.cov(particles.T) + 1e-9 * np.eye(4),
        particles=particles,
        weights=weights,
    )
    pred = pf.predict(state)
    assert np.allclose(pred.mean[:2], [10.0 + 0.3, 20.0 - 0.4], atol=0.05)
    assert np.allclose(pred.mean[2:], [3.0, -4.0], atol=0.05)


def test_update_pulls_toward_measurement():
    pf = ConstantVelocityPF(
        dt=1.0 / 30.0,
        n_particles=800,
        q_pos=1.0,
        q_vel=10.0,
        r_meas=5.0,
        init_pos_std=30.0,
        init_vel_std=50.0,
        seed=1,
    )
    state = pf.initiate(np.array([0.0, 0.0]))
    z = np.array([50.0, -30.0])
    updated = pf.update(state, z)
    assert float(np.linalg.norm(updated.position - z)) < float(
        np.linalg.norm(state.position - z)
    )


def test_update_none_preserves_moments():
    pf = ConstantVelocityPF(dt=1.0 / 30.0, n_particles=100, seed=2)
    state = pf.initiate(np.array([1.0, 2.0]))
    mean0, cov0 = state.mean.copy(), state.cov.copy()
    out = pf.update(state, None)
    assert np.allclose(out.mean, mean0)
    assert np.allclose(out.cov, cov0)
    assert out.particles is not None
    out.mean[0] = 999.0
    assert state.mean[0] == mean0[0]


def test_gate_window_uses_moments():
    pf = ConstantVelocityPF(
        dt=1.0 / 30.0, n_particles=300, init_pos_std=5.0, seed=3
    )
    state = pf.initiate(np.array([100.0, 200.0]))
    x0, y0, x1, y1 = pf.gate_window(state, n_sigma=3.0)
    assert x0 < 100.0 < x1
    assert y0 < 200.0 < y1


def test_noisy_cv_track_beats_raw_measurements():
    dt = 1.0 / 30.0
    sigma = 5.0
    n_steps = 80
    rng = np.random.default_rng(0)
    true_xy = synthetic_cv_trajectory(
        n_steps, x0=(100.0, 150.0), v0=(40.0, -25.0), dt=dt
    )
    meas = add_gaussian_noise(true_xy, sigma=sigma, rng=rng)

    pf = ConstantVelocityPF(
        dt,
        n_particles=600,
        q_pos=0.5,
        q_vel=5.0,
        r_meas=sigma,
        init_pos_std=sigma,
        init_vel_std=80.0,
        seed=0,
    )
    state = pf.initiate(meas[0])
    est = np.zeros_like(true_xy)
    est[0] = state.position
    for t in range(1, n_steps):
        state = pf.predict(state)
        state = pf.update(state, meas[t])
        est[t] = state.position

    warm = 15
    meas_rmse = float(np.sqrt(np.mean(np.sum((meas[warm:] - true_xy[warm:]) ** 2, axis=1))))
    est_rmse = float(np.sqrt(np.mean(np.sum((est[warm:] - true_xy[warm:]) ** 2, axis=1))))
    assert est_rmse < meas_rmse
    assert est_rmse < sigma


def test_build_pf_resolves_dt_and_n_particles():
    cfg = load_config()
    cfg = {
        **cfg,
        "tracker": {
            **(cfg.get("tracker") or {}),
            "dt": None,
            "n_particles": 123,
            "pf_seed": 7,
        },
    }
    pf = build_pf(cfg)
    assert np.isclose(pf.dt, 1.0 / float(cfg["source"]["fps"]))
    assert pf.n_particles == 123


def test_predict_requires_particles():
    pf = ConstantVelocityPF(dt=1.0 / 30.0, n_particles=50, seed=0)
    from horus.track import TrackState

    bare = TrackState(mean=np.zeros(4), cov=np.eye(4))
    try:
        pf.predict(bare)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "particles" in str(exc)

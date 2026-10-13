"""Tests for per-step r_meas and closed-loop track↔detect."""

from __future__ import annotations

import numpy as np
import torch

from horus.track import (
    ConstantVelocityKF,
    ConstantVelocityPF,
    run_track_detect_loop,
    synthetic_cv_trajectory,
)
from horus.track.loop import stub_detect_fn


def test_kf_update_r_meas_overrides():
    kf = ConstantVelocityKF(dt=0.1, q_pos=1.0, q_vel=10.0, r_meas=5.0)
    state = kf.initiate(np.array([0.0, 0.0]))
    # Inflate prior so R dominates the update weight.
    state.cov[:2, :2] = np.eye(2) * 1e6
    z = np.array([100.0, 50.0])

    tight = kf.update(state, z, r_meas=0.1)
    loose = kf.update(state, z, r_meas=50.0)
    # Smaller R → estimate closer to measurement.
    assert float(np.linalg.norm(tight.position - z)) < float(
        np.linalg.norm(loose.position - z)
    )
    # Filter-owned R unchanged by override.
    assert np.isclose(kf.R[0, 0], 25.0)


def test_pf_update_r_meas():
    pf = ConstantVelocityPF(
        dt=0.1, n_particles=200, r_meas=5.0, q_pos=1e-6, q_vel=1e-6, seed=0
    )
    state = pf.initiate(np.array([10.0, 20.0]))
    state = pf.predict(state)
    out = pf.update(state, np.array([12.0, 21.0]), r_meas=1.0)
    assert out.particles is not None
    assert np.isfinite(out.mean).all()


def test_closed_loop_stub_runs():
    dt = 1.0 / 30.0
    n = 20
    true_xy = synthetic_cv_trajectory(n, x0=(100.0, 100.0), v0=(20.0, 5.0), dt=dt)
    kf = ConstantVelocityKF(dt=dt, r_meas=3.0)
    stacks = [torch.zeros(1, 360, 640) for _ in range(n)]
    detect = stub_detect_fn(list(true_xy), noise_sigma=2.0, rng=np.random.default_rng(0))
    results = run_track_detect_loop(
        kf,
        z0=true_xy[0],
        frame_stacks=stacks,
        detect_fn=detect,
        img_w=640,
        img_h=360,
        buckets=[64, 128, 256],
        presence_thresh=0.5,
    )
    assert len(results) == n
    assert results[0].detected
    # Most frames should detect with the stub.
    assert sum(r.detected for r in results) >= n // 2
    # Crop sizes are valid buckets when used.
    for r in results[1:]:
        if r.crop_size:
            assert r.crop_size in (64, 128, 256)

#!/usr/bin/env python3
"""Smoke-test the constant-velocity KF on a synthetic noisy trajectory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running without install when developing
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from horus.config import load_config
from horus.track import (
    add_gaussian_noise,
    build_kf,
    synthetic_cv_trajectory,
)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to data.yaml (default: configs/data.yaml)",
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=None,
        help="Observation noise std in px (default: tracker.r_meas)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show a matplotlib plot if matplotlib is installed",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    kf = build_kf(cfg)
    sigma = float(args.noise_sigma) if args.noise_sigma is not None else kf.r_meas
    rng = np.random.default_rng(args.seed)

    true_xy = synthetic_cv_trajectory(
        args.steps,
        x0=(80.0, 200.0),
        v0=(45.0, -20.0),
        dt=kf.dt,
    )
    meas = add_gaussian_noise(true_xy, sigma=sigma, rng=rng)

    state = kf.initiate(meas[0])
    est = np.zeros_like(true_xy)
    est[0] = state.position
    for t in range(1, args.steps):
        state = kf.predict(state)
        state = kf.update(state, meas[t])
        est[t] = state.position

    warm = min(15, max(0, args.steps // 4))
    print(f"dt={kf.dt:.6f} steps={args.steps} noise_sigma={sigma:.2f}")
    print(f"meas_rmse (all)  = {_rmse(meas, true_xy):.3f} px")
    print(f"kf_rmse   (all)  = {_rmse(est, true_xy):.3f} px")
    print(f"meas_rmse (warm) = {_rmse(meas[warm:], true_xy[warm:]):.3f} px")
    print(f"kf_rmse   (warm) = {_rmse(est[warm:], true_xy[warm:]):.3f} px")
    x0, y0, x1, y1 = kf.gate_window(state, n_sigma=3.0)
    print(f"final_gate_window = ({x0:.1f}, {y0:.1f})–({x1:.1f}, {y1:.1f})")
    print("OK")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping plot")
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(true_xy[:, 0], true_xy[:, 1], "k-", label="true", linewidth=2)
        ax.scatter(meas[:, 0], meas[:, 1], s=12, c="C1", alpha=0.5, label="meas")
        ax.plot(est[:, 0], est[:, 1], "C0-", label="KF", linewidth=1.5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.legend()
        ax.set_title("Constant-velocity KF (oracle noisy measurements)")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

"""Oracle / synthetic measurement helpers for tracker development."""

from __future__ import annotations

import numpy as np


def synthetic_cv_trajectory(
    n_steps: int,
    x0: np.ndarray | tuple[float, float],
    v0: np.ndarray | tuple[float, float],
    dt: float,
) -> np.ndarray:
    """Generate a noise-free constant-velocity position sequence ``(T, 2)``."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    x0 = np.asarray(x0, dtype=np.float64).reshape(2)
    v0 = np.asarray(v0, dtype=np.float64).reshape(2)
    t = np.arange(n_steps, dtype=np.float64) * dt
    return x0[None, :] + t[:, None] * v0[None, :]


def add_gaussian_noise(
    true_xy: np.ndarray,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add i.i.d. Gaussian noise to a ``(T, 2)`` position sequence."""
    true_xy = np.asarray(true_xy, dtype=np.float64)
    if true_xy.ndim != 2 or true_xy.shape[1] != 2:
        raise ValueError(f"true_xy must have shape (T, 2), got {true_xy.shape}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=true_xy.shape)
    return true_xy + noise

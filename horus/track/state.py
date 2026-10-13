"""Track state for SSM / Kalman / particle filters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackState:
    """Track estimate: mean and covariance.

    Convention:
    - CV / CV-EKF filters: ``[x, y, vx, vy]`` (dim 4)
    - CA / CA-EKF filters: ``[x, y, vx, vy, ax, ay]`` (dim 6)

    Particle filters may also attach ``particles`` ``(N, D)`` and ``weights``
    ``(N,)``. Kalman filters leave those as ``None``.
    """

    mean: np.ndarray  # (D,)
    cov: np.ndarray  # (D, D)
    particles: np.ndarray | None = None  # (N, D)
    weights: np.ndarray | None = None  # (N,)

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float64).reshape(-1)
        dim = int(self.mean.shape[0])
        if dim < 4:
            raise ValueError(f"TrackState mean dim must be >= 4, got {dim}")
        self.cov = np.asarray(self.cov, dtype=np.float64).reshape(dim, dim)
        if self.particles is not None:
            self.particles = np.asarray(self.particles, dtype=np.float64)
            if self.particles.ndim != 2 or self.particles.shape[1] != dim:
                raise ValueError(
                    f"particles must have shape (N, {dim}), got {self.particles.shape}"
                )
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
            if self.particles is not None and self.weights.shape[0] != self.particles.shape[0]:
                raise ValueError(
                    f"weights length {self.weights.shape[0]} != "
                    f"n_particles {self.particles.shape[0]}"
                )

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    @property
    def position(self) -> np.ndarray:
        """``(x, y)`` position mean."""
        return self.mean[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        """``(vx, vy)`` velocity mean."""
        return self.mean[2:4].copy()

    @property
    def acceleration(self) -> np.ndarray | None:
        """``(ax, ay)`` if this is a CA state, else ``None``."""
        if self.dim < 6:
            return None
        return self.mean[4:6].copy()

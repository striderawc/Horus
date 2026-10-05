"""Constant-velocity bootstrap particle filter for 2D ball tracking."""

from __future__ import annotations

from typing import Any

import numpy as np

from horus.track.common import dt_from_cfg
from horus.track.gate import gate_window as _gate_window
from horus.track.state import TrackState


def _weighted_moments(
    particles: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = weights @ particles
    diff = particles - mean
    cov = (diff * weights[:, None]).T @ diff
    cov = 0.5 * (cov + cov.T)
    return mean, cov


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = weights.shape[0]
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0  # guard against FP drift
    return np.searchsorted(cumulative, positions, side="left")


class ConstantVelocityPF:
    """Bootstrap PF with CV dynamics and Gaussian position likelihood.

    Same public surface as ``ConstantVelocityKF``: ``initiate`` / ``predict`` /
    ``update`` / ``gate_window``. State carries particles + weights; ``mean`` /
    ``cov`` are always the weighted moments for viz / gating.
    """

    def __init__(
        self,
        dt: float,
        *,
        n_particles: int = 500,
        q_pos: float = 1.0,
        q_vel: float = 10.0,
        r_meas: float = 5.0,
        init_pos_std: float = 10.0,
        init_vel_std: float = 50.0,
        ess_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if n_particles < 2:
            raise ValueError(f"n_particles must be >= 2, got {n_particles}")
        if not (0.0 < ess_ratio <= 1.0):
            raise ValueError(f"ess_ratio must be in (0, 1], got {ess_ratio}")

        self.dt = float(dt)
        self.n_particles = int(n_particles)
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.r_meas = float(r_meas)
        self.init_pos_std = float(init_pos_std)
        self.init_vel_std = float(init_vel_std)
        self.ess_ratio = float(ess_ratio)
        self.rng = np.random.default_rng(seed)

        self.F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._q_std = np.array(
            [
                np.sqrt(self.q_pos),
                np.sqrt(self.q_pos),
                np.sqrt(self.q_vel),
                np.sqrt(self.q_vel),
            ],
            dtype=np.float64,
        )
        self._init_std = np.array(
            [
                self.init_pos_std,
                self.init_pos_std,
                self.init_vel_std,
                self.init_vel_std,
            ],
            dtype=np.float64,
        )
        self._log_lik_norm = -np.log(2.0 * np.pi * self.r_meas**2)

    def _pack(self, particles: np.ndarray, weights: np.ndarray) -> TrackState:
        weights = weights / weights.sum()
        mean, cov = _weighted_moments(particles, weights)
        return TrackState(
            mean=mean,
            cov=cov,
            particles=particles.copy(),
            weights=weights.copy(),
        )

    def _require_particles(self, state: TrackState) -> tuple[np.ndarray, np.ndarray]:
        if state.particles is None or state.weights is None:
            raise ValueError(
                "ConstantVelocityPF requires TrackState.particles/weights; "
                "initiate with this filter or pass a PF state through."
            )
        return state.particles, state.weights

    def initiate(self, z: np.ndarray) -> TrackState:
        """Sample particles around a position measurement ``z = (x, y)``."""
        z = np.asarray(z, dtype=np.float64).reshape(2)
        center = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float64)
        particles = self.rng.normal(
            loc=center,
            scale=self._init_std,
            size=(self.n_particles, 4),
        )
        weights = np.full(self.n_particles, 1.0 / self.n_particles, dtype=np.float64)
        return self._pack(particles, weights)

    def predict(self, state: TrackState) -> TrackState:
        """Propagate particles with CV dynamics + process noise."""
        particles, weights = self._require_particles(state)
        noise = self.rng.normal(0.0, self._q_std, size=particles.shape)
        particles = particles @ self.F.T + noise
        return self._pack(particles, weights)

    def update(
        self,
        state: TrackState,
        z: np.ndarray | None,
        *,
        r_meas: float | None = None,
    ) -> TrackState:
        """Importance weight + optional resample. ``z=None`` leaves weights as-is.

        Optional ``r_meas`` overrides the likelihood σ for this step only.
        """
        particles, weights = self._require_particles(state)
        if z is None:
            return self._pack(particles, weights)

        z = np.asarray(z, dtype=np.float64).reshape(2)
        sigma = float(self.r_meas if r_meas is None else r_meas)
        if sigma <= 0:
            raise ValueError(f"r_meas must be positive, got {sigma}")
        innov = particles[:, :2] - z[None, :]
        # log N(z; Hx, R) with isotropic R = r^2 I
        log_lik_norm = -np.log(2.0 * np.pi * sigma**2)
        log_w = log_lik_norm - 0.5 * np.sum((innov / sigma) ** 2, axis=1)
        log_w = log_w - log_w.max()
        weights = weights * np.exp(log_w)
        w_sum = weights.sum()
        if w_sum <= 0 or not np.isfinite(w_sum):
            weights = np.full(self.n_particles, 1.0 / self.n_particles)
        else:
            weights = weights / w_sum

        ess = 1.0 / float(np.sum(weights**2))
        if ess < self.ess_ratio * self.n_particles:
            idx = _systematic_resample(weights, self.rng)
            particles = particles[idx]
            weights = np.full(self.n_particles, 1.0 / self.n_particles)

        return self._pack(particles, weights)

    def gate_window(
        self, state: TrackState, n_sigma: float = 3.0
    ) -> tuple[float, float, float, float]:
        """Axis-aligned search box from position mean ± ``n_sigma`` std."""
        return _gate_window(state, n_sigma=n_sigma)


def build_pf(cfg: dict[str, Any]) -> ConstantVelocityPF:
    """Build a PF from a loaded Horus config (``tracker:`` + ``source.fps``)."""
    tcfg = cfg.get("tracker") or {}
    seed = tcfg.get("pf_seed", None)
    return ConstantVelocityPF(
        dt_from_cfg(cfg),
        n_particles=int(tcfg.get("n_particles", 500)),
        q_pos=float(tcfg.get("q_pos", 1.0)),
        q_vel=float(tcfg.get("q_vel", 10.0)),
        r_meas=float(tcfg.get("r_meas", 5.0)),
        init_pos_std=float(tcfg.get("init_pos_std", 10.0)),
        init_vel_std=float(tcfg.get("init_vel_std", 50.0)),
        ess_ratio=float(tcfg.get("ess_ratio", 0.5)),
        seed=None if seed is None else int(seed),
    )

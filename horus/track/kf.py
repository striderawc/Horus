"""Constant-velocity Kalman filter for 2D ball tracking."""

from __future__ import annotations

from typing import Any

import numpy as np

from horus.track.common import dt_from_cfg
from horus.track.gate import gate_window as _gate_window
from horus.track.state import TrackState


class ConstantVelocityKF:
    """Discrete constant-velocity KF with state ``[x, y, vx, vy]``.

    Position measurements ``z = (x, y)``. Missed detections: pass ``z=None``
    to ``update`` (predict-only; state unchanged by the update step).
    """

    def __init__(
        self,
        dt: float,
        *,
        q_pos: float = 1.0,
        q_vel: float = 10.0,
        r_meas: float = 5.0,
        init_pos_std: float = 10.0,
        init_vel_std: float = 50.0,
    ) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.dt = float(dt)
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.r_meas = float(r_meas)
        self.init_pos_std = float(init_pos_std)
        self.init_vel_std = float(init_vel_std)

        self.F = np.array(
            [
                [1.0, 0.0, self.dt, 0.0],
                [0.0, 1.0, 0.0, self.dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.Q = np.diag(
            [self.q_pos, self.q_pos, self.q_vel, self.q_vel]
        ).astype(np.float64)
        self.R = (self.r_meas**2) * np.eye(2, dtype=np.float64)

    def initiate(self, z: np.ndarray) -> TrackState:
        """Start a track from a position measurement ``z = (x, y)``."""
        z = np.asarray(z, dtype=np.float64).reshape(2)
        mean = np.array([z[0], z[1], 0.0, 0.0], dtype=np.float64)
        cov = np.diag(
            [
                self.init_pos_std**2,
                self.init_pos_std**2,
                self.init_vel_std**2,
                self.init_vel_std**2,
            ]
        ).astype(np.float64)
        return TrackState(mean=mean, cov=cov)

    def predict(self, state: TrackState) -> TrackState:
        """Propagate state one step with the CV motion model."""
        mean = self.F @ state.mean
        cov = self.F @ state.cov @ self.F.T + self.Q
        # Symmetrize for numerical stability
        cov = 0.5 * (cov + cov.T)
        return TrackState(mean=mean, cov=cov)

    def update(
        self,
        state: TrackState,
        z: np.ndarray | None,
        *,
        r_meas: float | None = None,
    ) -> TrackState:
        """Measurement update. ``z=None`` leaves the state unchanged.

        Optional ``r_meas`` overrides filter ``R`` for this step only
        (``R = r_meas² I``), e.g. detector-reported uncertainty.
        """
        if z is None:
            return TrackState(mean=state.mean.copy(), cov=state.cov.copy())

        z = np.asarray(z, dtype=np.float64).reshape(2)
        R = self.R if r_meas is None else (float(r_meas) ** 2) * np.eye(
            2, dtype=np.float64
        )
        innov = z - self.H @ state.mean
        S = self.H @ state.cov @ self.H.T + R
        K = state.cov @ self.H.T @ np.linalg.inv(S)
        mean = state.mean + K @ innov
        I = np.eye(4, dtype=np.float64)
        cov = (I - K @ self.H) @ state.cov
        cov = 0.5 * (cov + cov.T)
        return TrackState(mean=mean, cov=cov)

    def gate_window(
        self, state: TrackState, n_sigma: float = 3.0
    ) -> tuple[float, float, float, float]:
        """Axis-aligned search box from position mean ± ``n_sigma`` std."""
        return _gate_window(state, n_sigma=n_sigma)


def build_kf(cfg: dict[str, Any]) -> ConstantVelocityKF:
    """Build a KF from a loaded Horus config (``tracker:`` + ``source.fps``)."""
    tcfg = cfg.get("tracker") or {}
    return ConstantVelocityKF(
        dt_from_cfg(cfg),
        q_pos=float(tcfg.get("q_pos", 1.0)),
        q_vel=float(tcfg.get("q_vel", 10.0)),
        r_meas=float(tcfg.get("r_meas", 5.0)),
        init_pos_std=float(tcfg.get("init_pos_std", 10.0)),
        init_vel_std=float(tcfg.get("init_vel_std", 50.0)),
    )

"""Constant-velocity EKF with quadratic drag (and optional gravity)."""

from __future__ import annotations

from typing import Any

import numpy as np

from horus.track.common import dt_from_cfg
from horus.track.gate import gate_window as _gate_window
from horus.track.state import TrackState


def _speed(vx: float, vy: float, eps: float = 1e-12) -> float:
    return float(np.hypot(vx, vy) + eps)


def transition_f(
    mean: np.ndarray,
    dt: float,
    *,
    drag: float,
    gravity: float,
) -> np.ndarray:
    """Nonlinear discrete transition ``x_{k+1} = f(x_k)``.

    State ``[x, y, vx, vy]``. Drag acceleration is ``-drag * ‖v‖ v``;
    gravity adds a constant ``+gravity`` to ``vy`` (image ``y`` down).
    """
    x, y, vx, vy = (float(v) for v in np.asarray(mean, dtype=np.float64).reshape(4))
    speed = _speed(vx, vy)
    ax = -drag * speed * vx
    ay = gravity - drag * speed * vy
    return np.array(
        [
            x + vx * dt,
            y + vy * dt,
            vx + ax * dt,
            vy + ay * dt,
        ],
        dtype=np.float64,
    )


def transition_jacobian(
    mean: np.ndarray,
    dt: float,
    *,
    drag: float,
    gravity: float,
) -> np.ndarray:
    """Analytic Jacobian ``F = ∂f/∂x`` at ``mean`` (``gravity`` unused)."""
    del gravity  # constant bias; does not enter F
    _, _, vx, vy = (float(v) for v in np.asarray(mean, dtype=np.float64).reshape(4))
    f = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if drag == 0.0:
        return f

    speed = _speed(vx, vy)
    d_svx_dvx = (vx * vx + speed * speed) / speed
    d_svx_dvy = (vx * vy) / speed
    d_svy_dvx = (vy * vx) / speed
    d_svy_dvy = (vy * vy + speed * speed) / speed
    bdt = drag * dt
    f[2, 2] = 1.0 - bdt * d_svx_dvx
    f[2, 3] = -bdt * d_svx_dvy
    f[3, 2] = -bdt * d_svy_dvx
    f[3, 3] = 1.0 - bdt * d_svy_dvy
    return f


class ConstantVelocityEKF:
    """EKF with state ``[x, y, vx, vy]`` and quadratic-drag dynamics.

    Position measurements ``z = (x, y)`` stay linear. With ``drag=0`` and
    ``gravity=0`` the predict step matches the linear CV Kalman filter.
    Same public API as ``ConstantVelocityKF`` (``initiate`` / ``predict`` /
    ``update`` / ``gate_window``), including optional per-step ``r_meas``.
    """

    dim = 4

    def __init__(
        self,
        dt: float,
        *,
        q_pos: float = 1.0,
        q_vel: float = 10.0,
        r_meas: float = 5.0,
        init_pos_std: float = 10.0,
        init_vel_std: float = 50.0,
        drag: float = 0.0,
        gravity: float = 0.0,
    ) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        if drag < 0:
            raise ValueError(f"drag must be >= 0, got {drag}")

        self.dt = float(dt)
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self.r_meas = float(r_meas)
        self.init_pos_std = float(init_pos_std)
        self.init_vel_std = float(init_vel_std)
        self.drag = float(drag)
        self.gravity = float(gravity)

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
        """Propagate with the nonlinear drag (+ gravity) motion model."""
        if state.dim != self.dim:
            raise ValueError(
                f"EKF expects dim={self.dim}, got TrackState dim={state.dim}"
            )
        mean = transition_f(
            state.mean, self.dt, drag=self.drag, gravity=self.gravity
        )
        f_jac = transition_jacobian(
            state.mean, self.dt, drag=self.drag, gravity=self.gravity
        )
        cov = f_jac @ state.cov @ f_jac.T + self.Q
        cov = 0.5 * (cov + cov.T)
        return TrackState(mean=mean, cov=cov)

    def update(
        self,
        state: TrackState,
        z: np.ndarray | None,
        *,
        r_meas: float | None = None,
    ) -> TrackState:
        """Linear position update. ``z=None`` leaves the state unchanged.

        Optional ``r_meas`` overrides filter ``R`` for this step only
        (``R = r_meas² I``), e.g. detector-reported uncertainty.
        """
        if state.dim != self.dim:
            raise ValueError(
                f"EKF expects dim={self.dim}, got TrackState dim={state.dim}"
            )
        if z is None:
            return TrackState(mean=state.mean.copy(), cov=state.cov.copy())

        z = np.asarray(z, dtype=np.float64).reshape(2)
        r = self.R if r_meas is None else (float(r_meas) ** 2) * np.eye(
            2, dtype=np.float64
        )
        innov = z - self.H @ state.mean
        s = self.H @ state.cov @ self.H.T + r
        k_gain = state.cov @ self.H.T @ np.linalg.inv(s)
        mean = state.mean + k_gain @ innov
        eye = np.eye(self.dim, dtype=np.float64)
        cov = (eye - k_gain @ self.H) @ state.cov
        cov = 0.5 * (cov + cov.T)
        return TrackState(mean=mean, cov=cov)

    def gate_window(
        self, state: TrackState, n_sigma: float = 3.0
    ) -> tuple[float, float, float, float]:
        """Axis-aligned search box from position mean ± ``n_sigma`` std."""
        return _gate_window(state, n_sigma=n_sigma)


def build_ekf(cfg: dict[str, Any]) -> ConstantVelocityEKF:
    """Build an EKF from a loaded Horus config (``tracker:`` + ``source.fps``)."""
    tcfg = cfg.get("tracker") or {}
    return ConstantVelocityEKF(
        dt_from_cfg(cfg),
        q_pos=float(tcfg.get("q_pos", 1.0)),
        q_vel=float(tcfg.get("q_vel", 10.0)),
        r_meas=float(tcfg.get("r_meas", 5.0)),
        init_pos_std=float(tcfg.get("init_pos_std", 10.0)),
        init_vel_std=float(tcfg.get("init_vel_std", 50.0)),
        drag=float(tcfg.get("ekf_drag", 0.0)),
        gravity=float(tcfg.get("ekf_gravity", 0.0)),
    )

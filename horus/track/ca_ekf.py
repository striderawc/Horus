"""Constant-acceleration EKF with quadratic drag (and optional gravity)."""

from __future__ import annotations

from typing import Any

import numpy as np

from horus.track.common import dt_from_cfg
from horus.track.gate import gate_window as _gate_window
from horus.track.state import TrackState


def _speed(vx: float, vy: float, eps: float = 1e-12) -> float:
    return float(np.hypot(vx, vy) + eps)


def ca_transition_f(
    mean: np.ndarray,
    dt: float,
    *,
    drag: float,
    gravity: float,
) -> np.ndarray:
    """Nonlinear discrete CA transition ``x_{k+1} = f(x_k)``.

    State ``[x, y, vx, vy, ax, ay]``. Effective acceleration is
    ``a - drag * ‖v‖ v`` plus ``+gravity`` on ``ay`` (image ``y`` down).
    Accel states themselves are random-walk (unchanged by ``f``).
    """
    x, y, vx, vy, ax, ay = (
        float(v) for v in np.asarray(mean, dtype=np.float64).reshape(6)
    )
    speed = _speed(vx, vy)
    ax_eff = ax - drag * speed * vx
    ay_eff = ay + gravity - drag * speed * vy
    dt2 = 0.5 * dt * dt
    return np.array(
        [
            x + vx * dt + dt2 * ax_eff,
            y + vy * dt + dt2 * ay_eff,
            vx + ax_eff * dt,
            vy + ay_eff * dt,
            ax,
            ay,
        ],
        dtype=np.float64,
    )


def ca_transition_jacobian(
    mean: np.ndarray,
    dt: float,
    *,
    drag: float,
    gravity: float,
) -> np.ndarray:
    """Analytic Jacobian ``F = ∂f/∂x`` at ``mean`` (``gravity`` unused)."""
    del gravity
    _, _, vx, vy, _, _ = (
        float(v) for v in np.asarray(mean, dtype=np.float64).reshape(6)
    )
    dt2 = 0.5 * dt * dt
    f = np.array(
        [
            [1.0, 0.0, dt, 0.0, dt2, 0.0],
            [0.0, 1.0, 0.0, dt, 0.0, dt2],
            [0.0, 0.0, 1.0, 0.0, dt, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
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

    dax_dvx = -drag * d_svx_dvx
    dax_dvy = -drag * d_svx_dvy
    day_dvx = -drag * d_svy_dvx
    day_dvy = -drag * d_svy_dvy

    f[0, 2] = dt + dt2 * dax_dvx
    f[0, 3] = dt2 * dax_dvy
    f[1, 2] = dt2 * day_dvx
    f[1, 3] = dt + dt2 * day_dvy
    f[2, 2] = 1.0 + dt * dax_dvx
    f[2, 3] = dt * dax_dvy
    f[3, 2] = dt * day_dvx
    f[3, 3] = 1.0 + dt * day_dvy
    return f


class ConstantAccelerationEKF:
    """EKF with state ``[x, y, vx, vy, ax, ay]`` and quadratic-drag dynamics.

    Position measurements ``z = (x, y)`` stay linear. With ``drag=0`` and
    ``gravity=0`` the predict step matches the linear CA Kalman filter.
    Same public API as the other stage-1 trackers.
    """

    dim = 6

    def __init__(
        self,
        dt: float,
        *,
        q_pos: float = 1.0,
        q_vel: float = 10.0,
        q_acc: float = 50.0,
        r_meas: float = 5.0,
        init_pos_std: float = 10.0,
        init_vel_std: float = 50.0,
        init_acc_std: float = 100.0,
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
        self.q_acc = float(q_acc)
        self.r_meas = float(r_meas)
        self.init_pos_std = float(init_pos_std)
        self.init_vel_std = float(init_vel_std)
        self.init_acc_std = float(init_acc_std)
        self.drag = float(drag)
        self.gravity = float(gravity)

        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.Q = np.diag(
            [
                self.q_pos,
                self.q_pos,
                self.q_vel,
                self.q_vel,
                self.q_acc,
                self.q_acc,
            ]
        ).astype(np.float64)
        self.R = (self.r_meas**2) * np.eye(2, dtype=np.float64)

    def initiate(self, z: np.ndarray) -> TrackState:
        """Start a CA track from ``z = (x, y)``."""
        z = np.asarray(z, dtype=np.float64).reshape(2)
        mean = np.array(
            [z[0], z[1], 0.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        cov = np.diag(
            [
                self.init_pos_std**2,
                self.init_pos_std**2,
                self.init_vel_std**2,
                self.init_vel_std**2,
                self.init_acc_std**2,
                self.init_acc_std**2,
            ]
        ).astype(np.float64)
        return TrackState(mean=mean, cov=cov)

    def predict(self, state: TrackState) -> TrackState:
        """Propagate with nonlinear CA + drag (+ gravity) dynamics."""
        if state.dim != self.dim:
            raise ValueError(
                f"CA EKF expects dim={self.dim}, got TrackState dim={state.dim}"
            )
        mean = ca_transition_f(
            state.mean, self.dt, drag=self.drag, gravity=self.gravity
        )
        f_jac = ca_transition_jacobian(
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
        """Linear position update. ``z=None`` leaves the state unchanged."""
        if state.dim != self.dim:
            raise ValueError(
                f"CA EKF expects dim={self.dim}, got TrackState dim={state.dim}"
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


def build_ca_ekf(cfg: dict[str, Any]) -> ConstantAccelerationEKF:
    """Build a CA EKF from a loaded Horus config (``tracker:``)."""
    tcfg = cfg.get("tracker") or {}
    return ConstantAccelerationEKF(
        dt_from_cfg(cfg),
        q_pos=float(tcfg.get("q_pos", 1.0)),
        q_vel=float(tcfg.get("q_vel", 10.0)),
        q_acc=float(tcfg.get("q_acc", 50.0)),
        r_meas=float(tcfg.get("r_meas", 5.0)),
        init_pos_std=float(tcfg.get("init_pos_std", 10.0)),
        init_vel_std=float(tcfg.get("init_vel_std", 50.0)),
        init_acc_std=float(tcfg.get("init_acc_std", 100.0)),
        drag=float(tcfg.get("ekf_drag", 0.0)),
        gravity=float(tcfg.get("ekf_gravity", 0.0)),
    )
